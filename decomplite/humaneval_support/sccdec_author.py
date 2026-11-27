"""sc²dec run as the AUTHORS wrote it, on a cross-architecture corpus."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess

from decompbench.config import DATA_DIR, EXEBENCH_REPOS
from decompbench.dataset.builder import repo_dir
from decompbench.llms.registry import call
from decompbench.pipelines.base import (
    Cell, DecompResult, sanitise_function_for_path, stripped_variant,
)
from decompbench.pipelines.llm4decompile_asm import (
    clean_objdump_for_llm4decompile,
)
from decompbench.pipelines.sccdec import (
    SccdecPipeline, _is_context_overflow, _prompt_chars, _render_sent,
)

# The ONE importable path to the vendor.
_VENDOR = "decompbench.pipelines.sccdec_vendor.src.sccdec.eval"


class AuthorExampleUnusable(Exception):
    """The CANDIDATE is why pass 2 could not run -> fall back to C₁, still score
    the cell. Mirrors `sccdec_selfctx.CandidateUnusable`."""

    def __init__(self, reason: str, detail: str = ""):
        assert reason in SKIP_REASONS, f"unknown skip reason {reason!r}"
        self.reason = reason
        super().__init__(detail or reason)


class AuthorHarnessUnavailable(RuntimeError):
    """WE are why this row cannot run -> propagate, so the dispatcher writes
    an error row instead of laundering our breakage into a score.
    """


# Closed set. Anything not listed raises, per `sccdec_selfctx`'s
# allow-list-with-raise-by-default.
SKIP_REASONS = frozenset({
    "gate_no_compile",         # upstream's own `else` (eval.py:385-386). The
                               # most important bucket: the author's gate said
                               # C₁ does not compile, so sc²dec never fires.
    "example_none",            # compile() returned None: gcc produced no .so
    "example_symbol_missing",  # compile() raised ValueError("compile fails")
    "prompt_budget",           # vLLM 400'd the doubled pass-2 prompt
})


def _last_code_block(text: str) -> str:
    """Upstream's extraction (eval.py:280-284), reproduced."""
    codes = re.findall(r"```\w*\n(?P<code>.*?)```", text, re.DOTALL)
    return codes[-1] if codes else text


def _c_onlyfunc(c_func: str, candidate: str) -> str:
    """`build_icl_example`'s include-hoist (eval.py:292-297), reproduced."""
    # --- begin upstream (verbatim) ---
    c_include = ""
    for line in c_func.split("\n"):
        if "#include" in line:
            c_include += line + "\n"
            c_func = c_func.replace(line, "")
    c_onlyfunc = c_include + "\n" + candidate
    # --- end upstream ---
    return c_onlyfunc


def _example_path(cell: Cell, model_key: str, variant: str):
    """Where this row's HOST-X86 example listing is written."""
    fn = sanitise_function_for_path(cell.function, cell.language)
    return (DATA_DIR / "sccdec_author_example" / cell.language / cell.repo
            / cell.arch / cell.opt / cell.compiler / cell.obfuscation
            / cell.module / model_key / variant / f"{fn}.asm")


def _host_gcc() -> tuple[str, str]:
    """(version, target triple) of the `gcc` `evaluate_func` will invoke."""
    def _q(flag):
        try:
            return subprocess.run(["gcc", flag], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:                                   # noqa: BLE001
            return ""
    return _q("-dumpversion"), _q("-dumpmachine")


class SccdecAuthorPipeline(SccdecPipeline):
    """assembly → LLM4Decompile → the AUTHOR's host-x86 gate + example → re-
    prompt.
    """

    name = "sccdec_author"


    # ---- the oracle seams, and the one switch ----------------------------
    #
    # Extracted from `run()` verbatim, so `sccdec_author` behaves exactly as it
    # did — its tests, including the pass-1 byte-identity check against
    # `sccdec` and the `func_name=cell.function` pin, are the guard for that.
    # They exist for `sccdec_author_xarch`, which runs the SAME two-pass
    # protocol with the same prompts and overrides only these: the oracle is
    # the difference between the two columns, and duplicating 170 lines of
    # `run()` to change it would guarantee the rest drifts apart.
    #
    # `_author_gate` is the odd one out — a flag, not a method. It does not
    # change HOW a step is done, it decides whether two whole steps happen at
    # all (`_c_test` and `_gate`), and one of them drops rows. See its comment.
    #
    # `_asm_message` and `_pass2_messages` are deliberately NOT seams: both
    # columns must ask the model the same thing, or a difference in scores stops
    # being attributable to the oracle.

    #: The sidecar file the trace is written to.
    _trace_sidecar = ".sccdec_author.json"

    #: Does `run()` wrap the two passes in the author's recompile+re-execute gate
    #: (`evaluate_func`, eval.py:375)?
    _author_gate = True

    def _preflight_oracle(self, cell: Cell) -> None:
        """Prove the oracle's tools exist before a token is spent."""
        for tool in ("gcc", "objdump"):
            if shutil.which(tool) is None:
                raise AuthorHarnessUnavailable(
                    f"{self.name}: {tool} not on PATH; the author's oracle "
                    f"builds and disassembles on the HOST.")

    def _c_test(self, cell: Cell) -> str:
        """The generated harness `evaluate_func` concatenates into `combine.c`."""
        from decompbench.eval.exebench_ctest import c_test_for_cell
        return c_test_for_cell(cell)

    def _oracle_fields(self, cell: Cell) -> dict:
        """Per-row provenance for WHICH COMPILER judged this candidate."""
        gcc_version, gcc_triple = _host_gcc()
        return {
            "dialect_mismatch": cell.arch != "x86_64",
            "oracle_arch": gcc_triple,
            "host_gcc": gcc_version,
        }

    def _gate(self, ev, c_func: str, c_test: str, candidate: str,
              cell: Cell) -> tuple:
        """The author's gate AND metric (`eval.py:375`): `(flag_compile,
        flag_run)`.
        """
        flag_compile, flag_run, _ = _await(ev.evaluate_func(c_func, c_test,
                                                            candidate))
        return flag_compile, flag_run

    def _build_example(self, ev, c_func: str, candidate: str,
                       cell: Cell) -> str:
        """Upstream's `compile()`: the self-constructed one-shot example, as
        cleaned disassembly text.
        """
        try:
            asm = _await(ev.compile(_c_onlyfunc(c_func, candidate), cell.opt,
                                    func_name=cell.function))
        except ValueError as exc:                       # "compile fails"
            raise AuthorExampleUnusable(
                "example_symbol_missing",
                f"{cell.function!r} not in the host objdump: {exc}") from exc
        if asm is None:
            raise AuthorExampleUnusable(
                "example_none",
                "host gcc produced no shared object for C₁")
        return asm

    def run(self, cell: Cell, model_key: str, variant: str) -> DecompResult:
        # ---- everything that can fail, before a single token is spent -------
        # (`sccdec`'s ordering rule; the provider guard is inherited and runs
        # first, inside super()'s preamble which we repeat here deliberately.)
        from decompbench.config import MODELS
        provider = MODELS.get(model_key, {}).get("provider")
        if provider != "vllm":
            raise ValueError(
                f"pipeline {self.name!r} requires a vLLM-served model; "
                f"{model_key!r} is provider={provider!r}. The pass-2 context "
                f"guard classifies vLLM's HTTP 400; a provider that truncates "
                f"the prompt instead (Ollama) would silently measure a "
                f"half-length one-shot example.")

        if cell.repo not in EXEBENCH_REPOS and not (
                cell.repo == 'multipl_e_humaneval' and not self._author_gate):
            raise AuthorHarnessUnavailable(
                f"{self.name}: repo {cell.repo!r} is not supported — the "
                f"author's `evaluate_func` needs per-function IO pairs, which "
                f"only {sorted(EXEBENCH_REPOS)} ship.")

        self._preflight_oracle(cell)

        # Guarded by the gate flag, not merely used by it: upstream's `eval.py`
        # constructs a module-scope `httpx.AsyncClient` (eval.py:14) that nobody
        # closes.
        ev = None
        if self._author_gate:
            try:
                ev = __import__(_VENDOR, fromlist=["evaluate_func"])
            except ImportError as exc:
                raise AuthorHarnessUnavailable(
                    f"{self.name}: cannot import the vendored upstream at "
                    f"{_VENDOR} ({exc}). That dotted path is the only supported "
                    f"one — `sccdec_vendor/` and `src/` are namespace packages."
                ) from exc

        src = repo_dir(cell.repo, cell.language) / f"{cell.module}.c"
        if not src.exists():
            raise AuthorHarnessUnavailable(
                f"{self.name}: no reference TU at {src}")
        c_func = src.read_text(errors="replace")

        # UnsupportedModule propagates on purpose: it is architecture-
        # INDEPENDENT, so raising drops the module from all nine arch columns
        # equally and the gradient stays balanced.
        c_test = self._c_test(cell) if self._author_gate else None

        from decompbench.pipelines.linked_asm import linked_asm_for_variant
        raw_asm, asm_path = linked_asm_for_variant(cell, variant)
        query_asm = clean_objdump_for_llm4decompile(raw_asm, cell.function)

        # ---- pass 1 (identical request to `sccdec`'s) -----------------------
        stripped = stripped_variant(variant)
        query_msg = self._asm_message(query_asm)
        p1_msgs = [query_msg]
        r1 = call(model_key, "", "", messages=p1_msgs)
        c1 = _last_code_block(r1["text"])

        trace = {
            "scc_applied": False,
            "author_scored": False,
            "flag_compile": None, "flag_run": None,
            "flag_re_compile": None, "flag_re_run": None,
            "skip_reason": None, "skip_detail": None,
            "stripped": stripped,
            **self._oracle_fields(cell),
            "cfunc_has_deps": _has_deps(c_func, cell.module),
            # Keys kept with explicit nulls when there is no harness, so the two
            # columns' traces stay one schema and a reader diffing them is not
            # told a field was dropped (same rule as `host_gcc` below).
            "ctest_sha256": _sha256(c_test) if c_test is not None else None,
            "ctest_pairs": (c_test.count("static int dbx_pair_")
                            if c_test is not None else None),
            "query_asm_path": str(asm_path),
            "query_asm_chars": len(query_asm),
            "example_path": None, "example_chars": None,
            "extraction_differs_pass1": _extraction_differs(r1["text"]),
            "extraction_differs_pass2": None,
            "passes": [{
                "pass": 1,
                "prompt_chars": _prompt_chars(p1_msgs),
                "latency_s": r1["latency_s"],
                "input_tokens": r1["input_tokens"],
                "output_tokens": r1["output_tokens"],
            }],
        }
        chosen, raw_text, sent_msgs = c1, r1["text"], p1_msgs
        intermediate = str(asm_path)

        # ---- the AUTHOR's gate (eval.py:375) --------------------------------
        flag_compile = flag_run = flag_re_compile = flag_re_run = None
        if self._author_gate:
            flag_compile, flag_run = self._gate(ev, c_func, c_test, c1, cell)
            # Upstream's own default when pass 2 does not fire (eval.py:399-403):
            # the re_* pair ALIASES the first pair, so `Δ_author` is 0 for that
            # row rather than undefined.
            flag_re_compile, flag_re_run = flag_compile, flag_run
            trace["flag_compile"], trace["flag_run"] = flag_compile, flag_run

        try:
            # The `_author_gate` half is load-bearing, not decorative: `None !=
            # 1` is True, so without it an ungated column would reject every row.
            if self._author_gate and flag_compile != 1:  # eval.py:385-386
                raise AuthorExampleUnusable(
                    "gate_no_compile",
                    "the author's gate rejected C₁: it does not compile against "
                    "the generated c_test on this host")

            asm1 = self._build_example(ev, c_func, c1, cell)

            example = _example_path(cell, model_key, variant)
            example.parent.mkdir(parents=True, exist_ok=True)
            example.write_text(asm1 + "\n")
            intermediate = str(example)
            trace["example_path"] = str(example)
            trace["example_chars"] = len(asm1)

            p2_msgs = self._pass2_messages(query_msg, asm1, c1)
            try:
                r2 = call(model_key, "", "", messages=p2_msgs)
            except RuntimeError as exc:
                if not _is_context_overflow(exc):
                    raise
                raise AuthorExampleUnusable("prompt_budget", str(exc)) from exc

            c2 = _last_code_block(r2["text"])
            trace["extraction_differs_pass2"] = _extraction_differs(r2["text"])
            trace["passes"].append({
                "pass": 2,
                "prompt_chars": _prompt_chars(p2_msgs),
                "latency_s": r2["latency_s"],
                "input_tokens": r2["input_tokens"],
                "output_tokens": r2["output_tokens"],
            })
            if self._author_gate:
                flag_re_compile, flag_re_run = self._gate(ev, c_func, c_test,
                                                          c2, cell)
            trace["scc_applied"] = True
            trace["author_scored"] = self._author_gate
            chosen, raw_text, sent_msgs = c2, r2["text"], p2_msgs
        except AuthorExampleUnusable as exc:
            trace["skip_reason"] = exc.reason
            trace["skip_detail"] = str(exc)[:2000]

        trace["flag_re_compile"], trace["flag_re_run"] = flag_re_compile, flag_re_run

        return DecompResult(
            text=chosen + "\n",
            raw_text=raw_text,
            latency_s=sum(p["latency_s"] for p in trace["passes"]),
            input_tokens=sum(p["input_tokens"] for p in trace["passes"]),
            output_tokens=sum(p["output_tokens"] for p in trace["passes"]),
            pipeline=self.name,
            system_prompt="",
            user_prompt=_render_sent(sent_msgs),
            intermediate_path=intermediate,
            reasoning=r1.get("reasoning", ""),
            sidecars={
                ".pass1.c": c1 + "\n",
                self._trace_sidecar: json.dumps(trace, indent=2) + "\n",
            },
            extra=dict(trace),
        )


def _await(coro):
    """Run one vendored coroutine to completion."""
    return asyncio.run(coro)


def _sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _extraction_differs(raw: str) -> bool:
    """Would decompbench's `strip_fences` and upstream's `codes[-1]` disagree
    about WHICH CODE they picked?
    """
    from decompbench.pipelines.direct import strip_fences
    return strip_fences(raw).strip() != _last_code_block(raw).strip()


def _has_deps(c_func: str, module: str) -> bool:
    """Does the reference TU carry anything besides the target definition?"""
    body = "\n".join(ln for ln in c_func.split("\n") if "#include" not in ln)
    before = body.split(module)[0]
    return bool(re.search(r"\b(struct|union|enum|typedef)\b|\w+\s*\([^;]*\)\s*[;{]",
                          before))
