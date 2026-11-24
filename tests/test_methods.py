"""Named method routing, interruption-safe jobs, and strict rescoring."""
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest
import yaml

from decomplite import benchmark as bench, build as B, config, evaluate as E, methods
from decomplite.datasets import Sample

ROOT = Path(__file__).resolve().parents[1]


def test_all_eight_methods_have_bound_models_and_exactly_20000_cells(monkeypatch, tmp_path):
    cfg = config.load(ROOT / "benchmark_methods_500.yaml")
    manifest = json.loads(cfg.identity_manifest.read_text())

    def dataset(selection):
        split = selection.name.removeprefix("exebench_")
        return {r["name"]: Sample(selection.name, r["name"], tmp_path / (r["name"] + ".c"))
                for r in manifest["functions"] if r["split"] == split}

    monkeypatch.setattr(bench, "load_dataset", dataset)
    samples, opts, cells = bench.resolve(cfg, {})
    assert len(samples) == 500 and len(opts) == 5 and len(cells) == 20000
    assert {p.method for p in cfg.pipelines} == set(config.METHOD_INPUTS)
    assert all(c.model == c.pipeline.model for c in cells)
    assert len(bench.resolve(cfg, {"pipeline": "degpt"})[2]) == 2500
    assert len(bench.resolve(cfg, {"model": "nova-6.7b-bcr"})[2]) == 2500
    assert cfg.evaluation_protocol == "strict-v1"


@pytest.mark.parametrize("recipe,match", [
    ({"method": "invented", "model": "m"}, "unknown method"),
    ({"method": "degpt", "model": "m", "iterations": 1}, "recipe keys"),
    ({"method": "degpt", "model": "absent"}, "known `model`"),
])
def test_named_methods_are_not_generic_recipe_aliases(recipe, match):
    with pytest.raises(ValueError, match=match):
        config.parse_pipeline(Path("config.yaml"), "named", recipe, {"m": {}})


@pytest.mark.parametrize("observed,expected", [
    ({}, {"returnv": 1}), ({"x": 1}, {"x": 1, "y": 2}),
    ({"x": 1, "extra": 1}, {"x": 1}), ({"x": []}, {"x": [1]}),
    ({"x": True}, {"x": 1}), ({"x": float("nan")}, {"x": 1}),
])
def test_strict_scoring_rejects_incomplete_or_invalid_outputs(observed, expected):
    assert not E._strict_diff_io(observed, expected)


def test_strict_scoring_retains_existing_float_tolerance():
    assert E._strict_diff_io({"x": 1.0005}, {"x": 1.0})
    assert not E._strict_diff_io({"x": 1.1}, {"x": 1.0})
    assert E._strict_diff_io({"x": 9007199254740993}, {"x": 9007199254740993})
    assert not E._strict_diff_io({"x": 9007199254740992}, {"x": 9007199254740993})


def test_nonzero_exit_cannot_pass_strict_json_tests(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps([{"input": {}, "output": {"x": 1}}]))

    def fail_after_writing(args, work, timeout):
        Path(args[2]).write_text('{"x":1}')
        return 7

    monkeypatch.setattr(E, "_run", fail_after_writing)
    assert E._run_tests(tmp_path / "bin", tmp_path, pairs)[0] == 1
    count, total, note = E._run_tests(tmp_path / "bin", tmp_path, pairs, protocol="strict-v1")
    assert count == 0 and total == 1 and "exit 7" in note


def test_line_markers_do_not_hide_original_function_or_change_offsets():
    source = b'#include <stdio.h>\n# 1 "original.c"\nint actual(int x) { return x + 1; }\n'
    assert E._function_defs(source) == [("actual", source.index(b"int actual"), len(source) - 1)]
    recursive = '# 1 "original.c"\nint old(int x){return x ? old(x-1) : 0;}\n'
    assert E.rename_function(recursive, "actual") == recursive.replace("old", "actual")


@pytest.fixture
def method_case(tmp_path, monkeypatch):
    source = tmp_path / "f.c"
    source.write_text("int f(void) { return 1; }\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    selection = {"functions": [dict(source_key=f"test_real:ex{i:06d}", split="test_real", name=f"f{i}",
                                    fname="f", function_id=f"ex{i:06d}", context_sha256=source_sha)
                                for i in range(500)]}
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps(selection))
    sample = Sample("exebench_test_real", "f0", source, function="f")
    cfg = config.BenchmarkConfig(("x86_64",), ("O0",), ("gcc",),
        (config.Pipeline("degpt", ("pseudoc",), ("m",), (), "none", 1,
                         method="degpt", model="m"),), ("m",), (), tmp_path / "reports",
        model_registry={"m": {}}, identity_manifest=manifest, evaluation_protocol="strict-v1")
    monkeypatch.setattr(methods, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path / "results")
    binary = tmp_path / "stripped.so"
    binary.write_bytes(b"stripped fixture")
    built = B.Build(sample.repo, sample.name, "O0", True, stripped=str(binary), function="f")
    monkeypatch.setattr(B, "build", lambda *a, **kw: built)
    monkeypatch.setattr(bench.R, "materialize", lambda *a, **kw: {"pseudoc": "int FUN_1234(void) { return 1; }"})
    return cfg, sample, selection


@pytest.fixture
def method_server(method_case, monkeypatch):
    cfg, sample, selection = method_case
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply({"protocol": methods.PROTOCOL, "revision": "tested-revision",
                            "selection_sha256": methods.digest(selection)})
            else:
                self.reply({"status": "complete", "result": {"text": "int anonymous(void){return 1;}",
                    "llm_calls": 7, "latency_s": 3, "input_tokens": 40, "output_tokens": 12,
                    "sidecars": {"workflow": "original workflow"}}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            self.reply({"ready": body.get("methods")} if self.path == "/preflight"
                       else {"job_id": "a" * 64})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("DECOMPLITE_METHOD_WORKER_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield seen
    finally:
        server.shutdown()
        server.server_close()


def test_method_generation_records_real_counts_and_resumes(method_case, method_server):
    cfg, sample, selection = method_case
    cfg = methods.prepare(cfg, online=True, selected={"degpt"})
    cell = bench.Cell(sample, "O0", "m", cfg.pipelines[0])
    assert bench.decompile_cell(cfg, cell, force=False) == "ok"
    assert cell.meta()["llm_calls"] == 7
    assert cell.meta()["source_key"] == "test_real:ex000000"
    assert cell.meta()["method_trace"]["sidecars"]["workflow"] == "original workflow"
    assert bench.decompile_cell(cfg, cell, force=False) == "cached"
    jobs = [body for route, body in method_server if route == "/jobs"]
    assert len(jobs) == 1 and jobs[0]["channels"]["pseudoc"].startswith("int FUN_")
    assert jobs[0]["pseudoc_name"] == "FUN_1234"
    assert method_server[0] == ("/preflight", {"methods": ["degpt"]})
    offline = methods.prepare(cfg)
    assert bench.Cell(sample, "O0", "m", offline.pipelines[0]).is_current()


def test_changed_original_source_is_rejected_before_job_submission(method_case, method_server):
    cfg, sample, selection = method_case
    cfg = methods.prepare(cfg, online=True)
    sample.source_path.write_text("int f(void){return 2;}")
    cell = bench.Cell(sample, "O0", "m", cfg.pipelines[0])
    assert "source hash changed" in bench.decompile_cell(cfg, cell, force=False)
    assert not any(route == "/jobs" for route, body in method_server)


def test_offline_eval_does_not_contact_method_or_model_servers(method_case, monkeypatch):
    cfg, sample, selection = method_case
    monkeypatch.setattr(methods.requests, "request", lambda *a, **kw: pytest.fail("unexpected network call"))
    methods.prepare(cfg, online=False)


@pytest.mark.skipif(not __import__("shutil").which("gcc"), reason="requires GCC")
def test_strict_native_harness_rejects_crash_after_valid_output(tmp_path):
    source = tmp_path / "f.c"
    source.write_text("# 1\nint f(void){return 0;}\n")
    driver = tmp_path / "f_test.c"
    driver.write_text('#include <stdio.h>\n#include "f.c"\n'
                      'int main(int n,char **v){FILE *o=fopen(v[2],"w");'
                      'fputs("{\\\"value\\\":1}",o);fclose(o);return f()+7;}\n')
    pairs = tmp_path / "f_io.json"
    pairs.write_text(json.dumps([{"input": {}, "output": {"value": 1}}]))
    sample = Sample("fixture", "f", source, function="f", test_path=driver, io_path=pairs)
    answer = "int f(void){return 0;}"
    assert E.harness(sample, answer, "f")["test_passed"] == 1
    strict = E.harness(sample, answer, "f", protocol="strict-v1")
    assert strict["compile_ok"] == 1 and strict["test_passed"] == 0


@pytest.mark.skipif(not __import__("shutil").which("gcc"), reason="requires GCC")
def test_cli_rescores_existing_answer_without_llm_calls_or_overwriting_report(tmp_path, monkeypatch):
    import csv
    from decomplite import cli

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "f.c").write_text("# 1\nint f(void){return 0;}\n")
    (dataset / "f_test.c").write_text(
        '#include <stdio.h>\n#include "f.c"\n'
        'int main(int n,char **v){FILE *o=fopen(v[2],"w");fputs("{}",o);fclose(o);return f();}\n')
    (dataset / "f_io.json").write_text(json.dumps([{"input": {}, "output": {"value": 1}}]))
    path = tmp_path / "benchmark.yaml"
    legacy_report = tmp_path / "historical"
    legacy_report.mkdir()
    (legacy_report / "summary.csv").write_text("historical report retained")
    path.write_text(yaml.safe_dump({"opts": ["O0"], "model": "vllm-gemma4-31b-it",
        "models_file": str(ROOT / "models.yaml"), "pipelines": {"single_pass": {"input": "asm"}},
        "repos": [{"name": "fixture", "path": str(dataset)}], "report": {"out_dir": str(legacy_report)}}))
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(B, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(bench.llm, "call", lambda *a, **kw: pytest.fail("rescoring invoked a model"))
    cfg = config.load(path)
    sample = Sample("fixture", "f", dataset / "f.c", test_path=dataset / "f_test.c", io_path=dataset / "f_io.json")
    cell = bench.Cell(sample, "O0", cfg.models[0], cfg.pipelines[0])
    cell.out_path.parent.mkdir(parents=True)
    answer = "int anonymous(void){return 0;}\n"
    cell.out_path.write_text(answer)
    cell.meta_path.write_text(json.dumps({"spec": cell.pipeline.spec(cell.model), "status": "ok"}))
    strict = tmp_path / "strict"
    assert cli.main(["benchmark", "--config", str(path), "--phase", "eval",
                     "--evaluation-protocol", "strict-v1", "--report-dir", str(strict)]) == 0
    with (strict / "summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["evaluation_protocol"] == "strict-v1"
    assert (row["generated"], row["standalone_compile"], row["compile_ok"], row["test_passed"]) == ("1", "1", "1", "0")
    assert cell.out_path.read_text() == answer
    assert (legacy_report / "summary.csv").read_text() == "historical report retained"


@pytest.mark.skipif(not __import__("shutil").which("gcc"), reason="requires GCC")
def test_missing_header_failure_is_retried_without_regenerating_good_builds(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "DATA_DIR", tmp_path / "data")
    source = tmp_path / "f.c"
    source.write_text('#include "needed.h"\nint f(void){return VALUE;}\n')
    sample = Sample("fixture", "f", source)
    assert not B.build(sample, "O0").ok
    (tmp_path / "needed.h").write_text("#define VALUE 1\n")
    built = B.build(sample, "O0")
    assert built.ok, built.error
    monkeypatch.setattr(B, "compile_object", lambda *a, **kw: pytest.fail("successful cached build recompiled"))
    assert B.build(sample, "O0").ok
