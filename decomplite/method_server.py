#!/usr/bin/env python3
"""Persistent method jobs for decomp-lite. Never invokes the benchmark or DB."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import traceback

ROOT = Path(os.environ["DECOMPLITE_BACKEND_ROOT"]).resolve()
sys.path.insert(0, str(ROOT))
from decomplite import method_worker as worker


def atomic(path, value):
    temporary = path.with_suffix(".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def perform(root, job_id, payload):
    import requests

    root = Path(root)
    output = root / "jobs" / (job_id + ".json")
    with (root / "jobs" / (job_id + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            previous = json.loads(output.read_text())
            if previous.get("status") == "complete":
                return
        try:
            result = worker.execute(payload, job_id)
            value = {"status": "complete", "result": result}
        except Exception as exc:
            value = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc(),
                     "infrastructure": isinstance(exc, (OSError, requests.RequestException))}
        atomic(output, value)


def ready():
    return True


class Service:
    def __init__(self, root, workers=1):
        self.root = Path(root).resolve()
        suite = worker.verify_suite(self.root)
        worker.initialize(self.root)
        self.info = {"protocol": worker.PROTOCOL, "revision": worker.revision(self.root, suite),
                     "selection_sha256": suite["selection_sha256"], "methods": sorted(worker.BINDINGS),
                     "functions": len(suite["selection"]["functions"]), "dataset": "multipl_e_humaneval", "postgresql": False}
        self.validation = dict(self.info, selection=suite["selection"])
        (self.root / "jobs/history").mkdir(parents=True, exist_ok=True)
        self.pool = ProcessPoolExecutor(max_workers=workers, initializer=worker.initialize,
                                        initargs=(str(self.root),))
        # Start processes before ThreadingHTTPServer begins handling requests;
        # on Python versions using fork, avoid forking from an HTTP thread.
        self.pool.submit(ready).result()
        self.pending = {}
        self.lock = threading.Lock()

    def submit(self, payload):
        worker.validate_request(payload, self.validation)
        job_id = worker.digest(payload)
        output = self.root / "jobs" / (job_id + ".json")
        with self.lock:
            if output.exists():
                old = json.loads(output.read_text())
                if old.get("status") == "complete":
                    return job_id
                # An explicit new POST retries an error and retains its audit.
                os.replace(output, self.root / "jobs/history" / (job_id + "." + str(time.time_ns()) + ".json"))
            if job_id not in self.pending or self.pending[job_id].done():
                atomic(self.root / "jobs" / (job_id + ".request.json"), payload)
                self.pending[job_id] = self.pool.submit(perform, str(self.root), job_id, payload)
        return job_id

    def result(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{64}", job_id):
            raise ValueError("invalid job ID")
        output = self.root / "jobs" / (job_id + ".json")
        if output.is_file():
            return json.loads(output.read_text())
        future = self.pending.get(job_id)
        if future is None:
            raise ValueError("unknown job; POST its original request to resume")
        if future.done() and future.exception():
            return {"status": "error", "infrastructure": True, "error": str(future.exception())}
        return {"status": "running" if future.running() else "queued"}


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        try:
            if self.path == "/health":
                self.reply(200, self.server.service.info)
            elif self.path.startswith("/jobs/"):
                self.reply(200, self.server.service.result(self.path.split("/")[-1]))
            else:
                self.reply(404, {"error": "unknown route"})
        except ValueError as exc:
            self.reply(400, {"error": str(exc)})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8 * 1024 * 1024:
                raise ValueError("request must be between 1 byte and 8 MiB")
            payload = json.loads(self.rfile.read(length))
            if self.path == "/jobs":
                if payload.get("revision") != self.server.service.info["revision"]:
                    self.reply(409, {"error": "worker revision changed; restart the decomp-lite command"})
                    return
                self.reply(202, {"job_id": self.server.service.submit(payload)})
            elif self.path == "/preflight":
                methods = payload.get("methods", [])
                if not isinstance(methods, list) or not set(methods).issubset(worker.BINDINGS):
                    raise ValueError("unknown methods")
                try:
                    worker.preflight(methods)
                except Exception as exc:
                    self.reply(503, {"error": "method prerequisites unavailable: " + str(exc)})
                    return
                self.reply(200, {"ready": methods})
            else:
                self.reply(404, {"error": "unknown route"})
        except (ValueError, KeyError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass  # Persistent jobs continue; clients can reconnect.
        except Exception:
            logging.exception("worker request failed")
            self.reply(503, {"error": "worker infrastructure error; inspect the worker log"})

    def log_message(self, fmt, *args):
        logging.info(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", default=11900, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--no-dotenv", action="store_true", help="use only the supplied process environment")
    args = parser.parse_args()
    from dotenv import load_dotenv
    if not args.no_dotenv:
        load_dotenv(ROOT / ".env", override=False)
    args.suite_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.suite_root / "worker.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.service = Service(args.suite_root, max(1, args.workers))
    except Exception:
        logging.exception("method worker initialization failed")
        server.server_close()
        raise
    (args.suite_root / "worker.pid").write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    logging.info("ready on %s:%s: %s", args.host, args.port, server.service.info)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.service.pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
