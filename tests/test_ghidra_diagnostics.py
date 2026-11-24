"""Do not lose script failures behind Ghidra's final analysis timing table."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from decomplite import build as B, representations as R


def test_failed_extraction_preserves_complete_logs_and_meaningful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "DATA_DIR", tmp_path)
    monkeypatch.setattr(R, "_ghidra_root", lambda: tmp_path)
    (tmp_path / 'Ghidra').mkdir()
    (tmp_path / 'Ghidra/application.properties').write_text('application.version=12.1.2\n')
    stdout = ('[dump_function_by_addr] FAIL: no function at 00101100\n'
              + 'Low-level Error: Overlapping input varnodes\n'
              + 'analysis timing table\n' * 1000)
    stderr = 'diagnostic from native decompiler\n'

    class Process:
        returncode = 0

        def communicate(self, timeout=None):
            return stdout, stderr

    monkeypatch.setattr(R.subprocess, "Popen", lambda *a, **kw: Process())
    image = tmp_path / 'sample.so'
    image.write_bytes(b'fixture')
    built = B.Build('fixture', 'f', 'O2', True, stripped=str(image), section='.text')
    assert R.materialize(built, ['pseudoc']) == {'pseudoc': ''}
    message = R.channel_error(built, 'pseudoc')
    assert 'no function at 00101100' in message and 'full logs:' in message
    assert 'Overlapping input varnodes' in message
    directory = R.cache_dir(built)
    assert (directory / 'ghidra.stdout.log').read_text() == stdout
    assert (directory / 'ghidra.stderr.log').read_text() == stderr
    assert json.loads((directory / 'ghidra.run.json').read_text())['returncode'] == 0


def test_manifest_retains_error_start_instead_of_only_its_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(R, 'pseudoc_provenance', lambda: {'ghidra_version': '12.1.2'})
    message = 'REAL FAILURE: ' + 'x' * 4000

    def fail(built):
        raise RuntimeError(message)

    monkeypatch.setitem(R._EXTRACTORS, 'pseudoc', fail)
    image = tmp_path / 'sample.so'
    image.write_bytes(b'fixture')
    built = B.Build('fixture', 'f', 'O3', True, stripped=str(image))
    R.materialize(built, ['pseudoc'])
    assert R.channel_error(built, 'pseudoc') == message


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-group timeout')
def test_timeout_keeps_output_before_terminating_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(R, '_GHIDRA_TIMEOUT_S', 0.3)
    with pytest.raises(RuntimeError, match='timed out'):
        R._run_headless([sys.executable, '-u', '-c',
                         'import time; print("before timeout", flush=True); time.sleep(30)'],
                        log_dir=tmp_path)
    assert 'before timeout' in (tmp_path / 'ghidra.stdout.log').read_text()


def test_parallel_runtime_startups_cannot_share_or_overwrite_settings(tmp_path, monkeypatch):
    # Model the runtime's copy-then-import behavior with real subprocesses.
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'parent-settings'))
    code = '''import os,sys,time
from pathlib import Path
root=Path(os.environ['XDG_CONFIG_HOME'])
p=root/'ghidra/dev/jython_cachedir/jython-src/introspect.py'
p.parent.mkdir(parents=True)
p.write_text(sys.argv[1])
time.sleep(0.2)
assert p.read_text()==sys.argv[1], 'another runtime overwrote introspect.py'
print(root)
'''

    def run(index):
        directory = tmp_path / str(index)
        result = R._run_headless([sys.executable, '-c', code, str(index)], log_dir=directory)
        assert result.returncode == 0, result.stderr
        metadata = json.loads((directory / 'ghidra.run.json').read_text())
        assert metadata['settings_isolation'] == 'per-process'
        assert result.stdout.strip() == metadata['xdg_config_home']
        return metadata['xdg_config_home']

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(run, range(4)))
    assert len(set(paths)) == 4
    assert all(not Path(path).exists() for path in paths)
    assert not (tmp_path / 'parent-settings').exists()
