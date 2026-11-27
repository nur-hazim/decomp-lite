"""Execute original method implementations against decomp-lite input channels."""
import hashlib
import functools
import json
import os
from pathlib import Path
import tempfile
import subprocess
import time

PROTOCOL = "decomplite-methods-v1"
BINDINGS = {"llm4decompile_asm": "llm4decompile-6.7b-v1.5-vllm",
            "llm4decompile_ghidra": "llm4decompile-22b-v2-vllm",
            "degpt": "gemma4-31b-it-vllm", "fidelitygpt": "gemma4-31b-it-vllm",
            "agent4decompile_author_xarch": "gemma4-31b-it-vllm",
            "sk2decompile": "sk2decompile-struct-vllm", "nova": "nova-6.7b-bcr",
            "sccdec_author_xarch": "sccdec-lora-paper-vllm"}
_STATE = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_suite(root):
    from decomplite.humaneval import verify
    return verify(root)


def revision(root, suite):
    from decompbench.config import ROOT
    files = [p for p in Path(__file__).parent.rglob("*") if p.is_file() and p.suffix in (".py", ".txt") and "__pycache__" not in p.parts]
    for folder in ("decompbench/pipelines", "decompbench/eval", "decompbench/dataset", "decompbench/archs", "decompbench/llms", "prompts"):
        files.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".json", ".c", ".ini") and "sk2decompile_vendor" not in p.parts and "__pycache__" not in p.parts)
    h=hashlib.sha256();h.update(suite["selection_sha256"].encode());h.update(digest(suite["files"]).encode())
    for path in sorted(set(files)):
        relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else "lite/"+str(path.relative_to(Path(__file__).parent))
        h.update(relative.replace("\\","/").encode());h.update(path.read_bytes().replace(b"\r\n",b"\n"))
    settings={k:v for k,v in os.environ.items() if (k=="THINKING" or k.startswith(("A4D_","AGENT4DECOMPILE_","DEGPT_","FIDELITYGPT_","SK2DECOMPILE_","DECOMPBENCH_NORMALIZE_"))) and not any(s in k for s in ("KEY","TOKEN","BASE","URL","PYTHON"))}
    h.update(digest(settings).encode())
    return h.hexdigest()


def initialize(root):
    global _STATE
    import decompbench.config as config
    import decompbench.db as db
    root = Path(root).resolve()
    suite = json.loads((root / "suite.json").read_text())
    os.environ["RQ_SELECTION"] = "2"
    config.DATA_DIR = root / "data"
    config.RESULTS_DIR = root / "results"
    config.REPORTS_DIR = root / "reports"

    def no_database(*args, **kwargs):
        raise RuntimeError("the decomp-lite method worker cannot access PostgreSQL")

    # Set BEFORE importing method/evaluation modules, including import-bound shims.
    db.connect = no_database
    db._import_psycopg = no_database
    from decompbench import pipelines
    from decomplite.humaneval_support.author import run_humaneval
    from decomplite.humaneval_support.sccdec_author_xarch import SccdecAuthorXarchPipeline
    from decompbench.pipelines.base import Pipeline
    class HumanEvalAgent(Pipeline):
        name = "agent4decompile_author_xarch"
        def run(self, cell, model_key, variant):
            return run_humaneval(self, cell, model_key, variant)
    pipelines._REGISTRY["agent4decompile_author_xarch"] = HumanEvalAgent()
    pipelines._REGISTRY["sccdec_author_xarch"] = SccdecAuthorXarchPipeline()
    from decompbench.pipelines import agent4decompile_author as author, xarch_oracle
    author._FIELDS_ROOT = root / "data"
    scratch = root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    xarch_oracle.scratch_root = lambda: scratch
    _STATE = {"root": root, "suite": suite, "pipelines": pipelines,
              "identities": {r["source_key"]: r for r in suite["selection"]["functions"]},
              "models": json.loads(json.dumps(config.MODELS))}


def validate_request(payload, health):
    if payload.get("protocol") != PROTOCOL or payload.get("revision") != health["revision"]:
        raise ValueError("method worker revision/protocol changed; restart the decomp-lite command")
    method = payload.get("method")
    if method not in BINDINGS or payload.get("model") != BINDINGS[method]:
        raise ValueError("method/checkpoint binding does not match the eight-method configuration")
    rows = {r["source_key"]: r for r in health["selection"]["functions"]}
    row = rows.get(payload.get("source_key"))
    if row is None or row["context_sha256"] != payload.get("context_sha256"):
        raise ValueError("source identity/hash is not in the staged HumanEval corpus")
    if payload.get("opt") not in ("O0", "O1", "O2", "O3", "Os") or payload.get("input_kind") != "decomplite-stripped-shared-object":
        raise ValueError("requires decomp-lite stripped x86-64 input at one of the five opts")
    channel = "asm" if method in ("llm4decompile_asm", "nova", "sccdec_author_xarch") else "pseudoc"
    channels = payload.get("channels", {})
    if set(channels) != {channel} or not isinstance(channels[channel], str) or not channels[channel].strip():
        raise ValueError("method requires its complete nonempty " + channel + " input")
    return row


def configure_endpoints():
    import re
    from dotenv import dotenv_values
    from decompbench import config
    config.MODELS.clear()
    config.MODELS.update(json.loads(json.dumps(_STATE["models"])))
    exports = Path(os.environ.get("DECOMPLITE_METHOD_ENDPOINTS_FILE",
                                  str(config.ROOT / "docker_setup/models/open_source_vllm_setup/vllm.env")))
    values = dotenv_values(exports) if exports.exists() else {}
    for key, model in config.MODELS.items():
        env = "VLLM_BASE_URL_" + re.sub("[^A-Za-z0-9]", "_", key).upper()
        url = values.get(env) or os.environ.get(env)
        if url:
            model["base_url"] = url.rstrip("/")


def preflight(methods):
    import requests
    from decompbench.config import MODELS
    from decompbench.llms.registry import local_base_url
    configure_endpoints()
    required = {BINDINGS[m] for m in methods}
    if "sk2decompile" in methods:
        required.add("sk2decompile-ident-vllm")
    for key in required:
        if key == "nova-6.7b-bcr":
            url = os.environ.get("NOVA_SERVER_URL", "http://127.0.0.1:11436").rstrip("/")
            requests.get(url + "/health", timeout=10).raise_for_status()
        else:
            headers = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]} if os.environ.get("VLLM_API_KEY") else {}
            response = requests.get(local_base_url(MODELS[key]) + "/v1/models", headers=headers, timeout=10)
            response.raise_for_status()
            if MODELS[key]["model"] not in {r["id"] for r in response.json().get("data", [])}:
                raise RuntimeError(f"endpoint does not advertise {MODELS[key]['model']!r}")
    if "degpt" in methods:
        from decompbench.pipelines.degpt_vendor import degpt_python
        _check_interpreter("degpt", degpt_python())
    if "fidelitygpt" in methods:
        from decompbench.pipelines.fidelitygpt_vendor import fidelitygpt_python
        _check_interpreter("fidelitygpt", fidelitygpt_python())
        base = os.environ.get("FIDELITY_EMBED_BASE", "").rstrip("/")
        if not base:
            raise RuntimeError("set FIDELITY_EMBED_BASE to the intended embedding server")
        key = os.environ.get("FIDELITY_EMBED_KEY") or os.environ.get("VLLM_API_KEY")
        headers = {"Authorization": "Bearer " + key} if key else {}
        response = requests.get(base + "/models", headers=headers, timeout=10)
        response.raise_for_status()
        wanted = os.environ.get("FIDELITY_EMBED_MODEL", "qwen3-embedding:8b")
        if wanted not in {r["id"] for r in response.json().get("data", [])}:
            raise RuntimeError("embedding server does not advertise " + wanted)


@functools.lru_cache(maxsize=4)
def _check_interpreter(method, interpreter):
    code = ("from cinspector.interfaces import CCode; import tiktoken, Levenshtein, openai; CCode('int f(void){return 0;}')"
            if method == "degpt" else
            "import langchain, chromadb, networkx; from langchain_openai import ChatOpenAI")
    result = subprocess.run([interpreter, "-c", code], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"{method} isolated environment is incomplete: {result.stderr[-2000:]}")


def execute(payload, job_id):
    from decompbench.pipelines.base import Cell
    configure_endpoints()
    row = _STATE["identities"][payload["source_key"]]
    with tempfile.TemporaryDirectory(prefix="cell-", dir=_STATE["root"] / "scratch") as directory:
        work = Path(directory)
        asm = work / "input.asm"
        asm.write_text(payload["channels"].get("asm", ""))
        pseudo = work / "pseudoc.c"
        pseudo.write_text(payload["channels"].get("pseudoc", ""))
        cell = Cell(binary_id=0, repo=row["repo"], module=row["name"],
                    function=row["fname"], arch="x86_64", opt=payload["opt"], asm_path=str(asm),
                    asm_path_stripped=str(asm), asm_is_authoritative=True,
                    pseudoc_path=str(pseudo) if "pseudoc" in payload["channels"] else None,
                    pseudoc_name=payload.get("pseudoc_name"))
        started = time.monotonic()
        result = _STATE["pipelines"].get(payload["method"]).run(cell, payload["model"], payload["method"] + "_stripped")
        extra = result.extra
        if "llm_calls" in extra:
            calls = extra["llm_calls"]
        elif payload["method"] == "sk2decompile":
            calls = 2
        elif "passes" in extra:
            calls = len(extra["passes"])
        else:
            calls = next((v for k, v in extra.items() if k.endswith("llm_calls")), 1)
        return {"text": result.text, "raw_text": result.raw_text, "system_prompt": result.system_prompt,
                "user_prompt": result.user_prompt, "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens, "latency_s": result.latency_s,
                "wall_seconds": time.monotonic() - started, "llm_calls": calls,
                "sidecars": result.sidecars, "extra": extra, "source_key": row["source_key"],
                "job_id": job_id, "input_kind": payload["input_kind"], "oracle_suite": "humaneval"}
