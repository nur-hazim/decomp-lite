"""GPU placement must survive launcher env overrides and old containers."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("method_gpu", Path(__file__).resolve().parents[1] / "scripts/method_gpu.py")
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def test_gpu_setting_overrides_old_value_and_preserves_other_settings(tmp_path):
    env = tmp_path / "docker_setup/models/open_source_vllm_setup/.env"
    env.parent.mkdir(parents=True)
    env.write_text('# settings\nHF_TOKEN="fixture-token"\nexport VLLM_GPU=0\nVLLM_TP=1\n')
    gpu.configure_vllm(tmp_path, 1)
    first = env.read_text()
    assert 'HF_TOKEN="fixture-token"' in first and 'VLLM_TP=1' in first
    assert first.count('VLLM_GPU=') == 1 and first.endswith('VLLM_GPU=1\n')
    gpu.configure_vllm(tmp_path, 1)
    assert env.read_text() == first


def test_embedding_uses_gpu_one_and_preserves_existing_model_storage(monkeypatch):
    monkeypatch.setattr(gpu, "inspect", lambda name: {"State": {"Running": True}} if name == "hazim-ollama-embed" else None)
    calls = []
    monkeypatch.setattr(gpu, "docker", lambda *args, **kwargs: calls.append(args))
    gpu.embedding("hazim", 1)
    create = calls[0]
    assert create[create.index("--gpus") + 1] == "device=1"
    assert create[create.index("--volumes-from") + 1] == "hazim-ollama-embed:rw"
    assert calls[1:] == [("stop", "hazim-ollama-embed"), ("start", "hazim-ollama-embed-gpu1")]


def test_embedding_rejects_existing_container_on_wrong_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "inspect", lambda name: {"HostConfig": {"DeviceRequests": [{"DeviceIDs": ["0"]}]}})
    monkeypatch.setattr(gpu, "docker", lambda *args, **kwargs: pytest.fail("changed a wrongly configured container"))
    with pytest.raises(RuntimeError, match="not pinned to GPU 1"):
        gpu.embedding("hazim", 1)
