"""An upgrade must refresh every old pseudo-C without rebuilding ASM inputs."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from decomplite import build as B, config, representations as R

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('old_provenance', [None, {'ghidra_version': '11.3.1'}])
def test_upgrade_refreshes_old_pseudoc_only_then_resumes(tmp_path, monkeypatch, old_provenance):
    monkeypatch.setattr(R, 'DATA_DIR', tmp_path)
    provenance = {'ghidra_version': '12.1.2', 'post_script_sha256': 'new-script'}
    monkeypatch.setattr(R, 'pseudoc_provenance', lambda: provenance)
    image = tmp_path / 'input.so'
    image.write_bytes(b'unchanged ELF fixture')
    built = B.Build('fixture', 'f', 'O2', True, stripped=str(image))
    directory = R.cache_dir(built)
    directory.mkdir(parents=True)
    (directory / 'asm.txt').write_text('cached ASM')
    (directory / 'pseudoc.c').write_text('old pseudo-C')
    pseudo_meta = {'available': True}
    if old_provenance is not None:
        pseudo_meta['provenance'] = old_provenance
    (directory / 'manifest.json').write_text(json.dumps({
        'image_mtime': image.stat().st_mtime,
        'channels': {'asm': {'available': True}, 'pseudoc': pseudo_meta}}))
    calls = []

    def extract(b):
        calls.append(b)
        return 'new pseudo-C'

    monkeypatch.setitem(R._EXTRACTORS, 'asm', lambda b: pytest.fail('re-extracted cached assembly'))
    monkeypatch.setitem(R._EXTRACTORS, 'pseudoc', extract)
    for attempt in range(2):
        assert R.materialize(built, ['asm', 'pseudoc']) == {'asm': 'cached ASM', 'pseudoc': 'new pseudo-C'}
    assert len(calls) == 1
    assert json.loads((directory / 'manifest.json').read_text())['channels']['pseudoc']['provenance'] == provenance
    # A later script change also invalidates only the pseudo-C.
    provenance['post_script_sha256'] = 'changed-script'
    R.materialize(built, ['asm', 'pseudoc'])
    assert len(calls) == 2


def test_version_pin_uses_installed_properties_not_environment_claim(tmp_path, monkeypatch):
    (tmp_path / 'Ghidra').mkdir()
    (tmp_path / 'Ghidra/application.properties').write_text('application.version=11.3.1\n')
    monkeypatch.setenv('GHIDRA_VERSION', '12.1.2')
    monkeypatch.setattr(R, '_ghidra_root', lambda: tmp_path)
    with pytest.raises(B.ToolchainError, match='requires Ghidra 12.1.2, found 11.3.1'):
        R.require_ghidra_version('12.1.2')


def test_five_pseudoc_pipelines_are_pinned_and_three_asm_pipelines_are_independent(tmp_path):
    import yaml
    raw = yaml.safe_load((ROOT / 'benchmark_humaneval.yaml').read_text())
    manifest = tmp_path / 'selection.json'
    manifest.write_text('{}')
    raw['identity_manifest'] = str(manifest)
    raw['models_file'] = str(ROOT / 'models_methods.yaml')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw))
    cfg = config.load(path)
    pseudo = [p for p in cfg.pipelines if 'pseudoc' in p.input]
    asm = [p for p in cfg.pipelines if p.input == ('asm',)]
    assert len(pseudo) == 5 and len(asm) == 3
    assert all(p.ghidra_version == '12.1.2' and p.spec(p.model)['ghidra_version'] == '12.1.2' for p in pseudo)
    assert all(not p.ghidra_version and 'ghidra_version' not in p.spec(p.model) for p in asm)


@pytest.mark.skipif(not shutil.which('bash') or not shutil.which('unzip'), reason='requires Bash/unzip')
def test_installer_activates_bundled_jython_and_refuses_wrong_existing_version(tmp_path):
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, 'w') as archive:
        archive.writestr('Jython/Module.manifest', '')
    fixture = tmp_path / 'release.zip'
    with zipfile.ZipFile(fixture, 'w') as archive:
        prefix = 'ghidra_12.1.2_PUBLIC/'
        entry = zipfile.ZipInfo(prefix + 'support/analyzeHeadless')
        entry.create_system = 3
        entry.external_attr = 0o100755 << 16
        archive.writestr(entry, '#!/bin/sh\nexit 0\n')
        archive.writestr(prefix + 'Ghidra/application.properties', 'application.version=12.1.2\n')
        archive.writestr(prefix + 'Extensions/Ghidra/ghidra_12.1.2_PUBLIC_Jython.zip', nested.getvalue())
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    java = binaries / 'java'
    java.write_text('#!/bin/sh\necho \'openjdk version "21.0.0"\' >&2\n')
    curl = binaries / 'curl'
    curl.write_text('''#!/usr/bin/env python3
import json,os,shutil,sys
if '-o' in sys.argv:
    shutil.copyfile(os.environ['GHIDRA_TEST_ZIP'], sys.argv[sys.argv.index('-o')+1])
else:
    print(json.dumps({'assets': [{'name': 'ghidra_12.1.2_PUBLIC_fixture.zip', 'browser_download_url': 'https://example.invalid/fixture.zip'}]}))
''')
    java.chmod(0o755)
    curl.chmod(0o755)
    destination = tmp_path / 'installed'
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'],
               GHIDRA_TEST_ZIP=str(fixture), GHIDRA_INSTALL_DIR=str(destination))
    command = ['bash', str(ROOT / 'scripts/install_ghidra.sh'), '12.1.2']
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    assert (destination / 'Ghidra/Extensions/Jython/Module.manifest').exists()
    second = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second.returncode == 0 and 'already installed' in second.stdout
    props = destination / 'Ghidra/application.properties'
    props.write_text('application.version=11.3.1\n')
    wrong = subprocess.run(command, env=env, capture_output=True, text=True)
    assert wrong.returncode != 0 and 'new version-specific directory' in wrong.stderr
    assert props.read_text() == 'application.version=11.3.1\n'
