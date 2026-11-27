"""The host worker launcher must stage the Lite code and select HumanEval only."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]

def test_embedded_process_stopper_is_valid_python():
    tree=ast.parse((ROOT/'scripts/start_humaneval_worker.py').read_text(encoding='utf-8-sig'))
    stopper=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='stopper' for t in n.targets))
    compile(stopper,'worker-stopper','exec')
    assert 'SIGTERM' in stopper and 'Refusing unrelated recorded PID' in stopper

def test_launcher_copies_bundled_code_and_starts_humaneval_module(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('human_starter',ROOT/'scripts/start_humaneval_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite=tmp_path/'data/humaneval';suite.mkdir(parents=True)
    (suite/'suite.json').write_text(json.dumps({'selection_sha256':'fixture-hash'}))
    calls=[]
    def docker(*args,capture=False):
        calls.append(args)
        if args[0]=='inspect':return json.dumps([{'HostConfig':{'NetworkMode':'host'}}])
        if args[:3]==('exec','hazim-ubuntu','printenv'):
            return '/project' if args[3]=='PROJECT_DIR' else '/venv'
        return ''
    checks=iter([None,{'dataset':'multipl_e_humaneval','selection_sha256':'fixture-hash'}])
    monkeypatch.setattr(module,'docker',docker);monkeypatch.setattr(module,'health',lambda:next(checks))
    monkeypatch.setattr(sys,'argv',['starter','--lite-root',str(tmp_path)])
    monkeypatch.delenv('METHOD_CONTAINER',raising=False)
    module.main()
    start=next(c for c in calls if 'decomplite.method_server' in c)
    assert '--suite-root' in start and '/project/data/decomplite_humaneval/suite' in start
    assert 'DECOMPLITE_BACKEND_ROOT=/project' in start
    assert any(c[0]=='cp' and str(tmp_path/'decomplite') in c for c in calls)
