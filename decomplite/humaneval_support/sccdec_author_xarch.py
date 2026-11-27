"""sc²dec's own oracle, run on the CELL'S architecture instead of the host's."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from decompbench.pipelines.base import Cell
from decompbench.pipelines.llm4decompile_asm import (
    _COMMENT_RE,
    clean_objdump_for_llm4decompile,
)
from decomplite.humaneval_support.sccdec_author import (
    AuthorExampleUnusable,
    AuthorHarnessUnavailable,
    SccdecAuthorPipeline,
)
from decompbench.pipelines import xarch_oracle as XO


def _cleaner_delta_lines(raw_objdump: str) -> int:
    """How many disassembly lines upstream's `split("#")[0]` would truncate
    but this repo's ARM-safe regex leaves alone.
    """
    n = 0
    for line in raw_objdump.split("\n"):
        if "#" in line and not _COMMENT_RE.search(line):
            n += 1
    return n


class SccdecAuthorXarchPipeline(SccdecAuthorPipeline):
    """sc²dec's two-pass protocol with the author's oracle on the target ISA."""

    name = "sccdec_author_xarch"
    _trace_sidecar = ".sccdec_xarch.json"

    #: No gate. See the module docstring: the recompile+re-execute half of
    #: upstream's `evaluate_func` is the one part of this column that could not
    #: be transposed onto the target ISA without inventing a harness, a driver
    #: and a per-pair budget that are nobody's method — and it suppressed four
    #: fifths of the sc²dec population to do it.
    _author_gate = False

    # ---- seam 1: preflight ------------------------------------------------
    def _preflight_oracle(self, cell: Cell) -> None:
        """Prove the cross toolchain exists before a token is spent."""
        if not XO.supported_arch(cell.arch):
            raise AuthorHarnessUnavailable(
                f"{self.name}: no cross oracle for arch {cell.arch!r} "
                f"(supported: {sorted(XO._SUPPORTED_ARCHES)}).")
        from decompbench import archs

        prof = archs.get(cell.arch)
        for tool in (prof.cc, prof.objdump):
            if shutil.which(tool) is None:
                raise AuthorHarnessUnavailable(
                    f"{self.name}: {tool} not on PATH; this column's oracle "
                    f"builds and disassembles for {cell.arch}, not for the host.")
    # ---- seam 2: provenance ------------------------------------------------
    def _oracle_fields(self, cell: Cell) -> dict:
        cc = XO.oracle_cc(cell.arch)
        return {
            **({'oracle_suite': 'humaneval', 'example_context': 'original_types_and_helpers'}
               if cell.repo == 'multipl_e_humaneval' else {}),
            # False BY CONSTRUCTION, and stated rather than implied: it is the
            # entire claim this column makes, so a reader must not have to infer
            # it from the pipeline's name.
            "dialect_mismatch": False,
            "oracle_arch": cell.arch,
            "oracle_cc": cc,
            "oracle_cc_id": XO.oracle_cc_id(cc),
            # "local" on every arch, unconditionally.
            "oracle_transport": "local",
            # Filled in by `_build_example`; present here so the key exists even
            # on a row whose example was never built, where its absence would
            # otherwise be indistinguishable from zero.
            "cleaner_delta_lines": None,
            "oracle_cflags": None,
            # `1` a rename was applied (the stripped case, whose RATE is the
            # measurement), `0` normalisation is on and none was needed, `None`
            # either the flag is off or no example was built.
            "example_name_normalized": None,
            # The parent records `host_gcc` because its oracle IS the host's.
            # Kept, with an explicit null, so the two traces have one schema and
            # a reader diffing them is not told a field was dropped.
            "host_gcc": None,
        }

    # ---- seam 3: the example, and the only gate ---------------------------
    def _build_example(self, ev, c_func: str, candidate: str,
                       cell: Cell) -> str:
        """Upstream's `compile()` for the cell's architecture: `-shared -fPIC
        -O{n} -lm`, then `objdump -d`, then upstream's `<func>:` slice.
        """
        from decompbench import archs
        from decompbench.archs.c import C_BASE_CFLAGS
        from decomplite.humaneval_support.sccdec_author import _c_onlyfunc
        # DEFERRED, like `sccdec_selfctx`'s imports: `eval/functional` pulls the
        # eval harness, and `pipelines/__init__` imports this module eagerly
        # just to LIST pipeline names. Listing must stay cheap.
        from decompbench.eval.functional import _maybe_rename as maybe_rename

        prof = archs.get(cell.arch)
        opt = cell.opt.strip("-")
        with tempfile.TemporaryDirectory(
                dir=str(_scratch_root()), prefix=f"scx_ex_{cell.arch}_") as td:
            work = Path(td)
            # The objdump slice below finds the example function BY NAME, so the
            # symbol in the example image has to be `cell.function`.
            renamed, rename_frag = maybe_rename(candidate, "c", cell.function)
            self._last_name_normalized = rename_frag.get("name_normalized")
            if cell.repo == 'multipl_e_humaneval':
                from decomplite.humaneval_support.candidate import assemble
                try:
                    assembled = assemble(c_func, cell.function, renamed)
                except ValueError as exc:
                    raise AuthorExampleUnusable('example_none', str(exc)) from exc
            else:
                assembled = _c_onlyfunc(c_func, renamed)
            (work / "code.c").write_text(assembled)
            # FULL pinned cflags, unlike a4d's oracle — see the module
            # docstring.
            ok, err, cflags = XO.xarch_compile(
                work / "code.c", work / "code.out", cell.arch,
                mode=XO.MODE_SHARED, opt=opt, extra_flags=C_BASE_CFLAGS,
                cwd=work)
            self._last_cflags = list(cflags)
            if not ok:
                raise AuthorExampleUnusable(
                    "example_none",
                    f"{prof.cc} produced no shared object for C₁: {err[:300]}")
            try:
                r = subprocess.run([prof.objdump, "-d", str(work / "code.out")],
                                   capture_output=True, text=True,
                                   errors="replace", timeout=10)
            except subprocess.TimeoutExpired as exc:
                # Upstream returns None on an objdump timeout, which the parent
                # reads as `example_none`. Same outcome, named.
                raise AuthorExampleUnusable(
                    "example_none",
                    f"{prof.objdump} timed out on C₁'s shared object") from exc
            raw = r.stdout or ""
            self._last_cleaner_delta = _cleaner_delta_lines(raw)
            # Upstream's own check (eval.py:153-155), restored explicitly.
            if f"<{cell.function}>:" not in raw:
                norm = ("applied" if self._last_name_normalized
                        else "off or ineffective")
                raise AuthorExampleUnusable(
                    "example_symbol_missing",
                    f"{cell.function!r} is not in the {cell.arch} objdump of "
                    f"C₁'s shared object (normalisation {norm})")
            try:
                return clean_objdump_for_llm4decompile(raw, cell.function)
            except ValueError as exc:                   # "compile fails"
                raise AuthorExampleUnusable(
                    "example_symbol_missing",
                    f"{cell.function!r} not in the {cell.arch} objdump: {exc}"
                ) from exc

    # ---- the fields the example step measures ------------------------------
    def run(self, cell: Cell, model_key: str, variant: str):
        """The parent's `run()`, with what the example step measured folded into
        the trace afterwards.
        """
        self._last_cleaner_delta = None
        self._last_cflags = None
        self._last_name_normalized = None
        res = self.parent_run(cell, model_key, variant)
        key = self._trace_sidecar
        raw = res.sidecars.get(key)
        if raw is None:
            return res
        measured = {
            "cleaner_delta_lines": self._last_cleaner_delta,
            "oracle_cflags": self._last_cflags,
            "example_name_normalized": self._last_name_normalized,
        }
        trace = json.loads(raw)
        trace.update(measured)
        res.sidecars[key] = json.dumps(trace, indent=2) + "\n"
        if isinstance(res.extra, dict):
            res.extra.update(measured)
        return res

    #: Named alias for the parent's `run`, so the override above reads as
    #: "the parent's run plus a post-step" rather than as a `super()` call whose
    #: target a reader has to resolve.
    parent_run = SccdecAuthorPipeline.run


def _scratch_root() -> Path:
    """Shared with `agent4decompile_author_xarch` — one scratch root for both
    columns, so "the sidecar refuses a `/tmp` work dir" is solved in one
    place.
    """
    return XO.scratch_root()
