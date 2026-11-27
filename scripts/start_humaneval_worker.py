"""Launch only the bundled Lite HumanEval worker inside the existing method container."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


def docker(*args, capture=False):
    return subprocess.run(['docker',*map(str,args)],check=True,text=True,
        stdout=subprocess.PIPE if capture else None).stdout

def health():
    try:
        with urlopen('http://127.0.0.1:11900/health',timeout=2) as r:return json.load(r)
    except OSError:return None

def main():
    import os
    p=argparse.ArgumentParser();p.add_argument('--lite-root',type=Path,required=True);a=p.parse_args()
    lite=a.lite_root.resolve();suite=lite/'data/humaneval'
    expected=json.loads((suite/'suite.json').read_text())['selection_sha256']
    container=os.environ.get('METHOD_CONTAINER','hazim-ubuntu')
    docker('start',container,capture=True)
    info=json.loads(docker('inspect',container,capture=True))[0]
    if info['HostConfig']['NetworkMode']!='host':raise SystemExit('Method container must use host networking')
    project=docker('exec',container,'printenv','PROJECT_DIR',capture=True).strip()
    venv=docker('exec',container,'printenv','VIRTUAL_ENV',capture=True).strip()
    if not project or not venv:raise SystemExit('Worker needs PROJECT_DIR and VIRTUAL_ENV')
    root=project+'/data/decomplite_humaneval';runtime=root+'/runtime';remote_suite=root+'/suite'
    # Stop only a worker whose recorded PID and command line match the known server.
    # Never kill a process merely because it occupies the port.
    stopper="""import os,signal,time
from pathlib import Path
project=Path(os.environ['PROJECT_DIR'])
for base in [project/'data/decomplite_humaneval/suite',project/'data/decomplite_methods_500']:
 p=base/'worker.pid'
 if not p.is_file():continue
 pid=int(p.read_text());proc=Path('/proc')/str(pid)
 try:
  if not proc.exists() or (proc/'stat').read_text().split(') ',1)[1].startswith('Z'):
   p.unlink(missing_ok=True);continue
  cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
 except FileNotFoundError:
  p.unlink(missing_ok=True);continue
 if ('decomplite.method_server' not in cmd and '/tools/decomplite_methods/serve.py' not in cmd) or '--port 11900' not in cmd:
  raise SystemExit('Refusing unrelated recorded PID '+str(pid))
 os.kill(pid,signal.SIGTERM)
 for attempt in range(240):
  try:
   if not proc.exists() or (proc/'stat').read_text().split(') ',1)[1].startswith('Z'):break
  except FileNotFoundError:break
  time.sleep(.25)
 else:raise SystemExit('Worker is still finishing jobs; rerun after it exits')
 p.unlink(missing_ok=True)
"""
    docker('exec','-u','ubuntu',container,venv+'/bin/python','-c',stopper)
    if health() is not None:raise SystemExit('Port 11900 is occupied by an untracked service; inspect it before retrying')
    docker('exec','-u','ubuntu',container,'mkdir','-p',runtime,remote_suite)
    docker('cp',str(lite/'decomplite'),container+':'+runtime+'/')
    docker('cp',str(suite)+'/.',container+':'+remote_suite+'/')
    docker('exec','-u','root',container,'chown','-R','ubuntu:ubuntu',root)
    embed=os.environ.get('FIDELITY_EMBED_BASE','http://127.0.0.1:11435/v1')
    model=os.environ.get('FIDELITY_EMBED_MODEL','qwen3-embedding:8b')
    docker('exec','-d','-u','ubuntu','-e','RQ_SELECTION=2',
        '-e','DECOMPLITE_BACKEND_ROOT='+project,'-e','PYTHONPATH='+runtime+':'+project,
        '-e','FIDELITY_EMBED_BASE='+embed,'-e','FIDELITY_EMBED_MODEL='+model,
        '-w',project,container,venv+'/bin/python','-m','decomplite.method_server',
        '--suite-root',remote_suite,'--workers',os.environ.get('METHOD_WORKERS','4'),'--port','11900')
    for _ in range(180):
        current=health()
        if current:
            if current.get('dataset')!='multipl_e_humaneval' or current.get('selection_sha256')!=expected:
                raise SystemExit('Worker dataset/selection differs from staged HumanEval')
            print(json.dumps(current));return
        time.sleep(1)
    docker('exec',container,'tail','-n','60',remote_suite+'/worker.log')
    raise SystemExit('HumanEval worker did not become ready; inspect log above')
if __name__=='__main__':main()
