"""Thin client for the persistent, PostgreSQL-free decompbench method worker."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

import requests

from decomplite import build as B
from decomplite.config import DATA_DIR

PROTOCOL = "decomplite-methods-v1"
_STOP = threading.Event()
_IDENTITIES = {}


class MethodWorkerUnavailable(B.ToolchainError):
    """Infrastructure failure: stop the phase instead of scoring model failures."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cancel_waits():
    _STOP.set()


def _url():
    return os.environ.get("DECOMPLITE_METHOD_WORKER_URL", "http://127.0.0.1:11900").rstrip("/")


def _request(verb, route, *, timeout=(10, 30), **kwargs):
    try:
        response = requests.request(verb, _url() + route, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise MethodWorkerUnavailable(f"method worker unavailable at {_url()}: {exc}") from exc
    if response.status_code in (409, 503):
        raise MethodWorkerUnavailable(response.text[:1500])
    if response.status_code >= 400:
        raise RuntimeError(f"method worker [{response.status_code}]: {response.text[:1500]}")
    try:
        return response.json()
    except ValueError as exc:
        raise MethodWorkerUnavailable("method worker returned invalid JSON") from exc


def prepare(cfg, *, online=False, selected=None):
    """Pin method cache keys to the worker implementation; eval works offline."""
    if not any(p.method for p in cfg.pipelines):
        return cfg
    snapshot = DATA_DIR / "method_worker/backend.json"
    health = {}
    if online:
        _STOP.clear()
        health = _request("GET", "/health")
        if health.get("protocol") != PROTOCOL:
            raise MethodWorkerUnavailable("method worker protocol mismatch")
        manifest = json.loads(cfg.identity_manifest.read_text())
        if health.get("selection_sha256") != digest(manifest):
            raise MethodWorkerUnavailable("worker selection differs from the existing 500-function manifest")
        required = sorted({p.method for p in cfg.pipelines if p.method and (selected is None or p.name in selected)})
        # Cold imports of both isolated method environments can exceed one
        # ordinary polling timeout. Individual prerequisite checks are bounded.
        _request("POST", "/preflight", timeout=(10, 120), json={"methods": required})
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        B.write_atomic(snapshot, json.dumps(health, indent=2))
    elif snapshot.is_file():
        health = json.loads(snapshot.read_text())
    revision = health.get("revision", "")
    return replace(cfg, pipelines=tuple(replace(p, revision=revision) if p.method else p for p in cfg.pipelines))


def identity(cfg, sample):
    key = str(cfg.identity_manifest)
    if key not in _IDENTITIES:
        manifest = json.loads(cfg.identity_manifest.read_text())
        rows = manifest.get("functions", [])
        if len(rows) != 500 or len({r["source_key"] for r in rows}) != 500:
            raise ValueError("expected the complete, unique original 500-function selection")
        _IDENTITIES[key] = {(r["split"], r["name"]): r for r in rows}
    splits = {"exebench_test_real": "test_real", "exebench_valid_real": "valid_real"}
    if sample.repo not in splits:
        raise ValueError(f"named methods need the staged ExeBench 500 dataset, got {sample.repo!r}")
    row = _IDENTITIES[key].get((splits[sample.repo], sample.name))
    if row is None:
        raise ValueError(f"sample {sample.repo}/{sample.name} is absent from the selected 500")
    source_sha = hashlib.sha256(sample.source_path.read_bytes()).hexdigest()
    if source_sha != row["context_sha256"]:
        raise ValueError(f"original source hash changed for {row['source_key']}; no model call made")
    return row


def generate(cfg, cell, built, contents, *, force=False):
    if _STOP.is_set():
        raise MethodWorkerUnavailable("method job submission cancelled")
    if not cell.pipeline.revision:
        raise MethodWorkerUnavailable("prepare the method worker before generating named methods")
    from decomplite.evaluate import _function_defs

    row = identity(cfg, cell.sample)
    payload = {"protocol": PROTOCOL, "revision": cell.pipeline.revision,
               "method": cell.pipeline.method, "model": cell.model, "opt": cell.opt,
               "source_key": row["source_key"], "context_sha256": row["context_sha256"],
               "input_kind": "decomplite-stripped-shared-object",
               "binary_sha256": hashlib.sha256(Path(built.stripped).read_bytes()).hexdigest(),
               "channels": {name: contents[name] for name in cell.pipeline.input}}
    if cell.pipeline.ghidra_version:
        payload["ghidra_version"] = cell.pipeline.ghidra_version
    if "pseudoc" in contents:
        defs = _function_defs(contents["pseudoc"].encode())
        if len(defs) != 1:
            raise ValueError(f"expected one Ghidra function, found {len(defs)}")
        payload["pseudoc_name"] = defs[0][0]
    if force:
        payload["attempt"] = uuid.uuid4().hex
    submitted = _request("POST", "/jobs", json=payload)
    job_id = submitted["job_id"]
    start = time.monotonic()
    limit = float(os.environ.get("DECOMPLITE_METHOD_JOB_TIMEOUT", "0"))
    while not _STOP.is_set():
        result = _request("GET", "/jobs/" + job_id)
        if result["status"] == "error":
            if result.get("infrastructure"):
                raise MethodWorkerUnavailable(result["error"])
            raise RuntimeError(result["error"])
        if result["status"] == "complete":
            answer = result["result"]
            if not answer.get("text", "").strip():
                raise RuntimeError("method returned empty C")
            # Sidecars retain the real workflow and prompts, not a fabricated
            # single-pass transcript or a one-call count for the RPC itself.
            return answer["text"], {"turns": [], "stopped": "method completed",
                "selection": list(cell.pipeline.input), "available": list(payload["channels"]),
                "llm_calls": answer["llm_calls"], "latency_s": answer["latency_s"],
                "input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"],
                "worker_job_id": job_id, "worker_revision": cell.pipeline.revision,
                "source_key": row["source_key"], "input_binary_sha256": payload["binary_sha256"],
                "input_kind": payload["input_kind"], "method_trace": answer,
                "input_ghidra_version": cell.pipeline.ghidra_version,
                "sample_cflags": list(cell.sample.cflags)}
        if limit > 0 and time.monotonic() - start >= limit:
            raise MethodWorkerUnavailable(f"job {job_id} exceeded the client wait limit; worker progress is retained")
        _STOP.wait(0.5)
    raise MethodWorkerUnavailable(f"stopped waiting for {job_id}; rerun to recover the persisted job")
