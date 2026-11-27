"""Generated HumanEval C functions with their unchanged frozen C tests."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

from decomplite.humaneval_support.workload import REPO_NAME, load_manifest
from decompbench.repos.base import RepoProfile, TestResult, run_capture


def _list_sources(root: Path) -> list[Path]:
    manifest = load_manifest(root, verify_sources=True)
    return [Path(name + '.c') for name in sorted(manifest['functions'])]


def _sources_for_functions(root: Path, functions: frozenset[str]) -> list[Path]:
    available = {p.stem: p for p in _list_sources(root)}
    return [available[name] for name in sorted(functions) if name in available]


def _run_tests(work: Path, function: str, module: str) -> TestResult:
    # Work-tree sources have deliberately been spliced with the candidate;
    # verify harnesses, not the pristine source hash, at evaluation time.
    manifest = load_manifest(work, function=module)
    if function != module or module not in manifest['functions']:
        raise ValueError(f'Unexpected HumanEval function/module {function}/{module}')
    tests = (work / '_harness' / f'{module}_tests.c.txt').read_text(encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='.humaneval-test-', dir=work) as temporary:
        directory = Path(temporary)
        harness = directory / 'test.c'
        executable = directory / ('test.exe' if os.name == 'nt' else 'test.x')
        harness.write_text(f'#include "{(work / (module + ".c")).resolve().as_posix()}"\n' + tests, encoding='utf-8')
        try:
            compiled = run_capture(['gcc', '-std=c11', '-pedantic-errors', '-UNDEBUG', '-O0',
                                    str(harness), '-o', str(executable), '-lm'], cwd=work, timeout=60)
            if compiled.returncode:
                return TestResult('humaneval', False, 0, 1,
                                  (compiled.stderr or compiled.stdout)[-2000:], build_ok=False)
            result = run_capture([str(executable.resolve())], cwd=work, timeout=15)
            passed = result.returncode == 0
            return TestResult('humaneval', passed, int(passed), 1,
                              (result.stderr or result.stdout or
                               f'Frozen suite ({manifest["functions"][module]["assertions"]} assertions), exit {result.returncode}')[-2000:])
        except subprocess.TimeoutExpired as exc:
            return TestResult('humaneval', False, 0, 1, f'HumanEval timeout: {exc}',
                              build_ok=executable.exists())


PROFILE = RepoProfile(
    name=REPO_NAME, repo_url='', language='c', supported_archs=('x86_64',),
    supported_compilers=('gcc',), list_sources=_list_sources,
    sources_for_functions=_sources_for_functions, run_tests=_run_tests,
    reuse_work_tree=True,
)
