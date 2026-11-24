"""Configure GPU placement without changing the decompbench launcher code."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def configure_vllm(bench_root, gpu):
    path = Path(bench_root) / "docker_setup/models/open_source_vllm_setup/.env"
    text = path.read_text() if path.exists() else ""
    # The launcher sources this file AFTER reading the caller's environment.
    text = re.sub(r"(?m)^[ \t]*(?:export[ \t]+)?VLLM_GPU[ \t]*=.*$", "", text)
    updated = text.rstrip() + f"\nVLLM_GPU={gpu}\n"
    if not path.exists() or path.read_text() != updated:
        path.write_text(updated)
    print(f"Configured vLLM GPU {gpu} in {path}")


def docker(*args, check=True):
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def inspect(name):
    result = docker("container", "inspect", name, check=False)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def embedding(prefix, gpu):
    original = prefix + "-ollama-embed"
    name = original + "-gpu" + str(gpu)
    existing = inspect(name)
    if existing:
        requests = existing["HostConfig"].get("DeviceRequests") or []
        if not any(r.get("DeviceIDs") == [str(gpu)] for r in requests):
            raise RuntimeError(f"{name} exists but is not pinned to GPU {gpu}; inspect its configuration")
    else:
        old = inspect(original)
        storage = (["--volumes-from", original + ":rw"] if old else
                   ["-v", prefix + "_decomplite_ollama:/root/.ollama"])
        docker("create", "--name", name, "--gpus", "device=" + str(gpu),
               "-p", "127.0.0.1:11435:11434", *storage, "ollama/ollama")
    # Keep the established worker endpoint while preserving the old container.
    old = inspect(original)
    if old and old.get("State", {}).get("Running"):
        docker("stop", original)
    docker("start", name)
    print(f"Embedding server {name}: GPU {gpu}, localhost:11435")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["vllm", "embedding"])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--bench-root", type=Path)
    parser.add_argument("--prefix", default="hazim")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if args.action == "vllm":
        if args.bench_root is None:
            parser.error("vllm requires --bench-root")
        configure_vllm(args.bench_root, args.gpu)
    else:
        embedding(args.prefix, args.gpu)
