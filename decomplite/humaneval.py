"""Stage a complete accepted HumanEval C corpus and validate its native tests."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

from decomplite.config import ROOT, DATA_DIR
from decomplite.humaneval_support.workload import load_manifest, install, rename_symbol, MANIFEST, REPO_NAME

SUITE = DATA_DIR / 'humaneval'
FROZEN_TASKS = Path(__file__).parent/'humaneval_support/humaneval-c.jsonl'
PROTOCOL = 'decomplite-methods-v1'
OPTS = ('O0','O1','O2','O3','Os')

def sha(data): return hashlib.sha256(data).hexdigest()
def digest(value): return sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())

def check_frozen(corpus, manifest):
    rows=[json.loads(line) for line in FROZEN_TASKS.read_text(encoding='utf-8').splitlines() if line.strip()]
    if manifest.get('prompts_sha256') != sha(FROZEN_TASKS.read_bytes()):
        raise ValueError('HumanEval corpus was not accepted against the bundled frozen prompts/tests')
    if set(manifest['functions']) != {'he_'+r['name'] for r in rows}:
        raise ValueError('HumanEval task IDs differ from the complete frozen corpus')
    import re
    for row in rows:
        name='he_'+row['name'];original='he_'+re.sub(r'^HumanEval_\d+_', '', row['name'])
        tests=rename_symbol(row['tests'],original,name)
        header=rename_symbol(row['prompt'].rstrip()[:-1].rstrip()+';\n',original,name)
        if (corpus/'_harness'/(name+'_tests.c.txt')).read_text(encoding='utf-8')!=tests:
            raise ValueError('Frozen tests changed for '+name)
        if (corpus/'_harness'/(name+'_decl.h')).read_text(encoding='utf-8')!=header:
            raise ValueError('Frozen declaration changed for '+name)

def stage(corpus, destination=SUITE):
    corpus, destination = Path(corpus).resolve(), Path(destination).resolve()
    manifest = load_manifest(corpus, verify_sources=True)
    check_frozen(corpus,manifest)
    if sum(x['assertions'] for x in manifest['functions'].values()) != 1130:
        raise ValueError('Expected the complete frozen 161-task/1130-assertion C port')
    if destination.exists():
        old = json.loads((destination/'suite.json').read_text())
        if old.get('corpus_sha256') != sha((corpus/MANIFEST).read_bytes()):
            raise ValueError('A different HumanEval corpus is already staged; do not mix cached experiments')
        verify(destination)
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.humaneval-',dir=destination.parent) as td:
        staged=Path(td)/'suite'; target=staged/'data/repos/c'/REPO_NAME
        (target/'_harness').mkdir(parents=True)
        shutil.copyfile(corpus/MANIFEST,target/MANIFEST)
        rows=[]
        for name,entry in sorted(manifest['functions'].items()):
            for relative in [name+'.c','_harness/'+name+'_tests.c.txt','_harness/'+name+'_decl.h']:
                shutil.copyfile(corpus/relative,target/relative)
            tests=(target/'_harness'/(name+'_tests.c.txt')).read_text(encoding='utf-8')
            # Tests remain unchanged after the original workload's recorded symbol namespace change.
            (target/(name+'_test.c')).write_bytes(('#include "'+name+'.c"\n'+tests).encode())
            rows.append(dict(repo=REPO_NAME,split='humaneval',name=name,fname=name,
                function_id=entry['task'],source_key='humaneval:'+entry['task'],
                context_sha256=entry['source_sha256'],assertions=entry['assertions']))
        selection=dict(schema=2,dataset=REPO_NAME,count=161,assertions=1130,functions=rows)
        (staged/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        files={str(p.relative_to(staged)).replace('\\','/'):sha(p.read_bytes()) for p in staged.rglob('*') if p.is_file()}
        suite=dict(protocol=PROTOCOL,dataset=REPO_NAME,selection=selection,selection_sha256=digest(selection),
                   corpus_sha256=sha((corpus/MANIFEST).read_bytes()),files=files)
        (staged/'suite.json').write_text(json.dumps(suite,indent=2),encoding='utf-8')
        staged.rename(destination)
    verify(destination)
    return destination

def verify(root=None):
    root=Path(SUITE if root is None else root).resolve(); suite=json.loads((root/'suite.json').read_text())
    if suite.get('dataset')!=REPO_NAME or suite.get('protocol')!=PROTOCOL:
        raise ValueError('Expected a HumanEval-only worker suite')
    if suite['selection_sha256']!=digest(suite['selection']) or len(suite['selection']['functions'])!=161:
        raise ValueError('HumanEval selection is incomplete or changed')
    for relative,expected in suite['files'].items():
        p=(root/relative).resolve()
        if not p.is_relative_to(root) or sha(p.read_bytes())!=expected:
            raise ValueError('Staged HumanEval input changed: '+relative)
    manifest=load_manifest(root/'data/repos/c'/REPO_NAME,verify_sources=True)
    check_frozen(root/'data/repos/c'/REPO_NAME,manifest)
    return suite

def references(workers=4):
    from decomplite import build as B, evaluate as E
    from decomplite.datasets import Sample
    from decomplite.config import Pipeline
    suite=verify(); corpus=SUITE/'data/repos/c'/REPO_NAME
    tc=B.toolchain()
    if not tc.can_execute(): raise ValueError('Reference validation requires native x86-64 Linux; run the provided Docker command')
    compiler=subprocess.check_output([tc.cc,'--version'],text=True).splitlines()[0]
    engine=sha(b"".join(p.read_bytes() for p in [Path(E.__file__),Path(B.__file__),Path(__file__),*sorted((Path(__file__).parent/"humaneval_support").glob("*.py"))]))
    root=ROOT/'reports/humaneval_reference';root.mkdir(parents=True,exist_ok=True)
    key=digest(dict(suite=suite['selection_sha256'],compiler=compiler,engine=engine))
    cache=root/'checkpoints'/key;cache.mkdir(parents=True,exist_ok=True)
    def one(item):
        row,opt=item;p=cache/(row['name']+'_'+opt+'.json')
        if p.is_file():
            result=json.loads(p.read_text())
            if result.get('test_passed')==1:return result
        sample=Sample(REPO_NAME,row['name'],corpus/(row['name']+'.c'),function=row['name'],
            cflags=('-std=c11','-pedantic-errors','-UNDEBUG','-'+opt),test_path=corpus/(row['name']+'_test.c'))
        verdict=E.harness(sample,sample.source_path.read_text(),sample.function,protocol='strict-v1')
        result=dict(function=row['name'],task=row['function_id'],opt=opt,assertions=row['assertions'],**verdict)
        B.write_atomic(p,json.dumps(result,indent=2));return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(one,[(r,o) for r in suite['selection']['functions'] for o in OPTS]))
    with (root/'references.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    failed=[r for r in rows if r['compile_ok']!=1 or r['test_passed']!=1]
    if failed:raise ValueError(f'{len(failed)}/805 original HumanEval references failed; inspect {root}/references.csv. No model calls should be started.')
    print('Validated all 805 original reference suites (161 tasks x five opts)')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare');g=s.add_mutually_exclusive_group(required=True);g.add_argument('--corpus',type=Path);g.add_argument('--accepted',type=Path);s.add_argument('--expected',type=Path)
    v=sub.add_parser('verify');v.add_argument('--root',type=Path,default=SUITE)
    v=sub.add_parser('references');v.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.command=='prepare':
        source=a.corpus
        if a.accepted:
            if not a.expected:a.expected=FROZEN_TASKS
            if sha(a.expected.read_bytes())!=sha(FROZEN_TASKS.read_bytes()):p.error('--expected differs from the bundled frozen prompts/tests')
            source=DATA_DIR/'humaneval_imported'
            install(a.accepted,a.expected,source)
        print('Staged HumanEval only:',stage(source),'(161 tasks; 6440 method cells)')
    elif a.command=='verify':print(json.dumps({'selection_sha256':verify(a.root)['selection_sha256'],'functions':161}))
    else:references(a.workers)
if __name__=='__main__':main()
