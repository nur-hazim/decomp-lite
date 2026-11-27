"""Only temporary test fixtures are synthesized; never benchmark implementations."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace
import pytest
import yaml
from decomplite import humaneval as H, evaluate as E, build as B, config, benchmark as bench, methods
from decomplite.humaneval_support.candidate import assemble

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def corpus(tmp_path,monkeypatch):
    root=tmp_path/'corpus';(root/'_harness').mkdir(parents=True)
    entries={};frozen=[]
    for i in range(161):
        name=f'he_HumanEval_{i}_fixture'
        source=f'#include <stdint.h>\nint64_t {name}(void){{return {i};}}\n'
        count=8 if i<3 else 7
        tests='#include <assert.h>\nint main(void){\n'+(f'assert({name}()=={i});\n'*count)+'return 0;}\n'
        header=f'#include <stdint.h>\nint64_t {name}(void);\n'
        for rel,text in [(name+'.c',source),('_harness/'+name+'_tests.c.txt',tests),('_harness/'+name+'_decl.h',header)]:
            (root/rel).write_bytes(text.encode())
        frozen.append(dict(name=f'HumanEval_{i}_fixture',language='c',prompt='#include <stdint.h>\n'+f'int64_t he_fixture(void) {{\n',tests=tests.replace(name,'he_fixture')))
        entries[name]=dict(task=f'HumanEval_{i}_fixture',assertions=count,source_sha256=H.sha(source.encode()),tests_sha256=H.sha(tests.encode()),header_sha256=H.sha(header.encode()))
    expected=tmp_path/'frozen.jsonl';expected.write_text('\n'.join(map(json.dumps,frozen))+'\n')
    monkeypatch.setattr(H,'FROZEN_TASKS',expected)
    (root/H.MANIFEST).write_text(json.dumps(dict(repo=H.REPO_NAME,functions=entries,prompts_sha256=H.sha(expected.read_bytes()))))
    return root

@pytest.fixture
def staged(corpus,tmp_path):return H.stage(corpus,tmp_path/'suite')

def human_config(staged,tmp_path):
    raw=yaml.safe_load((ROOT/'benchmark_humaneval.yaml').read_text())
    raw['identity_manifest']=str(staged/'selection.json')
    raw['models_file']=str(ROOT/'models_methods.yaml')
    raw['repos'][0]['path']=str(staged/'data/repos/c'/H.REPO_NAME)
    raw['repos'][0]['functions']=[x['name'] for x in H.verify(staged)['selection']['functions']]
    path=tmp_path/'run.yaml';path.write_text(yaml.safe_dump(raw))
    return config.load(path)

def test_staging_is_complete_hashed_and_idempotent(corpus,staged):
    suite=H.verify(staged)
    assert suite['selection']['count']==161
    assert suite['selection']['assertions']==1130
    assert H.stage(corpus,staged)==staged
    assert all(r['repo']==H.REPO_NAME for r in suite['selection']['functions'])

def test_tampering_or_partial_corpus_is_rejected(corpus,staged):
    source=next((staged/'data/repos/c'/H.REPO_NAME).glob('he_*.c'))
    source.write_text('changed')
    with pytest.raises(ValueError,match='changed'):H.verify(staged)
    (corpus/'he_HumanEval_0_fixture.c').unlink()
    with pytest.raises(ValueError,match='Missing'):H.stage(corpus,staged.parent/'another')

def test_humaneval_only_plan_has_6440_bound_cells(staged,tmp_path):
    cfg=human_config(staged,tmp_path)
    samples,opts,cells=bench.resolve(cfg,{})
    assert len(samples)==161 and len(opts)==5 and len(cells)==6440
    assert {c.sample.repo for c in cells}=={H.REPO_NAME}
    assert {c.pipeline.name for c in cells}==set(config.METHOD_INPUTS)
    assert all(c.model==c.pipeline.model for c in cells)
    assert cfg.evaluation_protocol=='strict-v1'
    assert all('-UNDEBUG' in s.cflags for s in samples)
    for sample in samples:assert methods.identity(cfg,sample)['name']==sample.name
    with pytest.raises(ValueError,match='absent'):
        methods.identity(cfg,replace(samples[0],repo='exebench_test_real'))

def test_all_default_configs_are_humaneval_only():
    for name in ['benchmark.yaml','benchmark_humaneval.yaml','benchmark_methods_500.yaml']:
        raw=yaml.safe_load((ROOT/name).read_text())
        assert [r['name'] for r in raw['repos']]==[H.REPO_NAME]
        assert 'functions_from' not in raw
        assert len(raw['pipelines'])==8

def test_candidate_retains_types_and_original_helpers():
    original='typedef long Item;\nItem helper(Item x){return x+1;}\nItem target(Item x){return helper(x);}\n'
    result=assemble(original,'target','Item target(Item x){return extra(x);}\nItem extra(Item x){return helper(x)+1;}')
    assert result.count('typedef long Item;')==1
    assert result.count('Item helper(Item x){return x+1;}')==1
    assert 'Item extra(Item x);' in result
    assert 'return extra(x);' in result

NATIVE=platform.system()=='Linux' and platform.machine() in ('x86_64','AMD64') and shutil.which('gcc')
@pytest.mark.skipif(not NATIVE,reason='requires native x86-64 Linux GCC')
def test_native_humaneval_harness_accepts_correct_and_rejects_wrong(staged,tmp_path,monkeypatch):
    monkeypatch.setattr(E,'DATA_DIR',tmp_path/'eval')
    cfg=human_config(staged,tmp_path);samples,_,_=bench.resolve(cfg,{})
    sample=next(s for s in samples if s.name=='he_HumanEval_0_fixture')
    good=E.harness(sample,sample.source_path.read_text(),sample.name,protocol='strict-v1')
    bad=E.harness(sample,f'long {sample.name}(void){{return 999;}}',sample.name,protocol='strict-v1')
    assert good['compile_ok']==1 and good['test_passed']==1
    assert bad['compile_ok']==1 and bad['test_passed']==0

@pytest.mark.skipif(not NATIVE,reason='requires native x86-64 Linux GCC')
def test_all_805_reference_fixtures_and_checkpoint_resume(staged,tmp_path,monkeypatch):
    monkeypatch.setattr(H,'SUITE',staged);monkeypatch.setattr(H,'ROOT',tmp_path)
    monkeypatch.setattr(E,'DATA_DIR',tmp_path/'eval')
    H.references(workers=4)
    import csv
    with (tmp_path/'reports/humaneval_reference/references.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==805 and all(r['test_passed']=='1' for r in rows)
    monkeypatch.setattr(E,'harness',lambda *a,**kw:pytest.fail('passing reference was reexecuted'))
    H.references(workers=4)

@pytest.mark.skipif(not NATIVE,reason='backend is Linux-only')
def test_worker_initializes_all_eight_methods_without_postgres(staged):
    pytest.importorskip('decompbench')
    from decomplite import method_worker as W
    W.initialize(staged)
    assert set(W.BINDINGS)==set(config.METHOD_INPUTS)
    assert W._STATE['pipelines'].get('agent4decompile_author_xarch').__class__.__name__=='HumanEvalAgent'
    assert W._STATE['pipelines'].get('sccdec_author_xarch').__class__.__module__.startswith('decomplite.humaneval_support')
    import decompbench.db as db
    with pytest.raises(RuntimeError,match='cannot access PostgreSQL'):db.connect()
@pytest.mark.skipif(not NATIVE,reason='backend is Linux-only')
def test_each_worker_method_receives_humaneval_identity_and_keeps_counts(staged,monkeypatch):
    pytest.importorskip('decompbench')
    from decomplite import method_worker as W
    from decompbench.pipelines.base import DecompResult
    suite=H.verify(staged);row=suite['selection']['functions'][0];seen=[]
    class Fake:
        def run(self,cell,model,variant):
            seen.append(cell)
            assert cell.repo==H.REPO_NAME and cell.module==row['name'] and cell.function==row['name']
            assert cell.asm_is_authoritative
            return DecompResult('int recovered(void){return 1;}','raw',.5,17,4,variant,extra={'llm_calls':3})
    monkeypatch.setattr(W,'_STATE',{'root':staged,'identities':{row['source_key']:row},'pipelines':SimpleNamespace(get=lambda name:Fake())})
    monkeypatch.setattr(W,'configure_endpoints',lambda:None)
    (staged/'scratch').mkdir(exist_ok=True)
    health={'revision':'fixture','selection':suite['selection']}
    for method,model in W.BINDINGS.items():
        channel='asm' if method in ['nova','llm4decompile_asm','sccdec_author_xarch'] else 'pseudoc'
        payload=dict(protocol=W.PROTOCOL,revision='fixture',method=method,model=model,
            source_key=row['source_key'],context_sha256=row['context_sha256'],opt='O2',
            input_kind='decomplite-stripped-shared-object',channels={channel:'int FUN_1(void){return 1;}'})
        assert W.validate_request(payload,health)==row
        result=W.execute(payload,'fixture-job')
        assert result['input_tokens']==17 and result['output_tokens']==4 and result['llm_calls']==3
        assert result['oracle_suite']=='humaneval'
    assert len(seen)==8
@pytest.mark.skipif(not NATIVE,reason='requires native author oracle')
def test_agent4_uses_frozen_humaneval_tests_without_exebench_or_llm(staged,tmp_path,monkeypatch):
    pytest.importorskip('decompbench')
    from decomplite.humaneval_support.author import run_humaneval
    from decompbench.pipelines.base import Cell
    from decompbench import config as C
    from decompbench.dataset import builder
    from decompbench.pipelines import ghidra
    from decompbench.llms import registry
    corpus=staged/'data/repos/c'/H.REPO_NAME;name='he_HumanEval_0_fixture'
    monkeypatch.setattr(builder,'repo_dir',lambda *a,**kw:corpus)
    monkeypatch.setattr(ghidra,'ghidra_decompile',lambda *a,**kw:((corpus/(name+'.c')).read_text(),'fixture'))
    monkeypatch.setattr(ghidra,'pseudoc_cache_path',lambda *a,**kw:tmp_path/'pseudo.c')
    monkeypatch.setattr(registry,'call',lambda *a,**kw:pytest.fail('already passing initial code called a model'))
    monkeypatch.setitem(C.MODELS,'fixture-model',{'provider':'vllm'})
    cell=Cell(0,H.REPO_NAME,name,name,'x86_64','O0',str(tmp_path/'asm'))
    result=run_humaneval(SimpleNamespace(name='agent4decompile_author_xarch'),cell,'fixture-model','fixture_stripped')
    assert result.extra['oracle_suite']=='humaneval'
    assert result.extra['l3_pass_count']==1 and result.extra['l3_total']==1
    assert result.extra['llm_calls']==0

@pytest.mark.skipif(not NATIVE,reason='requires native SCCDec self-example toolchain')
def test_sccdec_example_compiles_with_original_types_and_helpers(tmp_path,monkeypatch):
    pytest.importorskip('decompbench')
    from decomplite.humaneval_support import sccdec_author_xarch as X
    from decompbench.pipelines.base import Cell
    monkeypatch.setattr(X,'_scratch_root',lambda:tmp_path)
    name='he_HumanEval_0_fixture'
    source='typedef long Item;\nItem helper(Item x){return x+1;}\n'+f'Item {name}(Item x){{return helper(x);}}\n'
    candidate=f'Item {name}(Item x){{return helper(x)+2;}}'
    cell=Cell(0,H.REPO_NAME,name,name,'x86_64','O0',str(tmp_path/'asm'))
    result=X.SccdecAuthorXarchPipeline()._build_example(None,source,candidate,cell)
    assert '<'+name+'>:' in result and 'ret' in result
def test_complete_accepted_export_imports_before_staging(corpus,tmp_path):
    from decomplite.humaneval_support.workload import install,rename_symbol
    frozen=[json.loads(x) for x in H.FROZEN_TASKS.read_text().splitlines() if x.strip()]
    accepted=tmp_path/'accepted';accepted.mkdir();entries=[]
    for row in frozen:
        task=row['name'];name='he_'+task
        source=rename_symbol((corpus/(name+'.c')).read_text(),name,'he_fixture').encode()
        (accepted/(task+'.c')).write_bytes(source)
        (accepted/(task+'_tests.c.txt')).write_bytes(row['tests'].encode())
        entries.append(dict(name=task,candidate_index=0,source_sha256=H.sha(source)))
    (accepted/'corpus-manifest.json').write_text(json.dumps(dict(complete=True,tasks=161,tasks_with_passing_candidate=161,prompts_sha256=H.sha(H.FROZEN_TASKS.read_bytes()),accepted=entries)))
    destination=tmp_path/'imported'
    result=install(accepted,H.FROZEN_TASKS,destination)
    assert len(result['functions'])==161
    assert install(accepted,H.FROZEN_TASKS,destination)==result
    suite=H.stage(destination,tmp_path/'accepted_suite')
    assert H.verify(suite)['selection']['count']==161

def _reformatted_prompts(corpus,tmp_path):
    original=H.FROZEN_TASKS.read_bytes()
    rows=[json.loads(x) for x in original.decode().splitlines() if x.strip()]
    path=tmp_path/'accepted-original.jsonl'
    # Different row/key order, compact JSON and CRLF: all decoded content identical.
    data=('\r\n'.join(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=True) for row in reversed(rows))+'\r\n').encode()
    assert H.sha(data)!=H.sha(original)
    path.write_bytes(data)
    manifest_path=corpus/H.MANIFEST;manifest=json.loads(manifest_path.read_text())
    manifest['prompts_sha256']=H.sha(data);manifest_path.write_text(json.dumps(manifest))
    return path

def test_serialization_only_difference_is_proven_and_retained(corpus,tmp_path):
    expected=_reformatted_prompts(corpus,tmp_path)
    staged=H.stage(corpus,tmp_path/'reformatted_suite',expected=expected)
    assert (staged/'accepted-prompts.jsonl').read_bytes()==expected.read_bytes()
    validation=json.loads((staged/'prompt_validation.json').read_text())
    assert validation['raw_hash_matches'] is False
    assert H.verify(staged)['selection']['count']==161
    # Offline verification does not depend on the original external prompt path.
    expected.unlink()
    H.verify(staged)

def test_mismatch_without_bound_prompt_file_is_not_bypassed(corpus,tmp_path):
    expected=_reformatted_prompts(corpus,tmp_path)
    with pytest.raises(ValueError,match='Supply --expected'):
        H.stage(corpus,tmp_path/'no_proof')
    wrong=tmp_path/'wrong.jsonl';wrong.write_bytes(H.FROZEN_TASKS.read_bytes())
    with pytest.raises(ValueError,match='does not match corpus prompts_sha256'):
        H.stage(corpus,tmp_path/'wrong_proof',expected=wrong)
    assert not (tmp_path/'no_proof').exists()

@pytest.mark.parametrize('field',['prompt','tests','language'])
def test_real_record_changes_are_rejected_even_with_matching_manifest_hash(corpus,tmp_path,field):
    expected=_reformatted_prompts(corpus,tmp_path)
    rows=[json.loads(x) for x in expected.read_text().splitlines()]
    rows[0][field]+=' CHANGED'
    expected.write_text('\n'.join(map(json.dumps,rows))+'\n')
    p=corpus/H.MANIFEST;manifest=json.loads(p.read_text());manifest['prompts_sha256']=H.sha(expected.read_bytes());p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='content change'):
        H.stage(corpus,tmp_path/'changed',expected=expected)
    assert not (tmp_path/'changed').exists()

def test_duplicate_task_records_rejected(corpus,tmp_path):
    expected=_reformatted_prompts(corpus,tmp_path)
    rows=expected.read_text().splitlines();rows[1]=rows[0]
    expected.write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError,match='Duplicate task ID'):
        H.equivalent_tasks(expected,H.FROZEN_TASKS)
