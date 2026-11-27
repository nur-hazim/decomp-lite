"""HumanEval constraint adapter for the existing Agent4Decompile author loop.

The refiner/prompts/iteration policy stay upstream. Its L3 oracle executes the
frozen HumanEval C suite, not ExeBench IO pairs or recovered author records.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

from decompbench.pipelines.agent4decompile_author import _AuthorEvaluator, _AuthorRefiner, _stage
from decompbench.pipelines.base import DecompResult, stripped_variant
from decomplite.humaneval_support.candidate import assemble
from decomplite.humaneval_support.workload import MANIFEST, REPO_NAME, load_manifest
from decompbench.repos.base import run_capture


class HumanEvalEvaluator(_AuthorEvaluator):
    def __init__(self, temp_dir, cell, root):
        super().__init__(temp_dir=temp_dir)
        if cell.arch != 'x86_64' or platform.machine().lower() not in ('x86_64', 'amd64'):
            raise ValueError('The HumanEval author oracle requires native x86-64 execution')
        from decompbench.archs import get
        self.arch = get(cell.arch)
        self.cell = cell
        self.root = Path(temp_dir) / 'corpus'
        (self.root / '_harness').mkdir(parents=True)
        load_manifest(root, verify_sources=True, function=cell.module)
        shutil.copy2(root / MANIFEST, self.root / MANIFEST)
        for suffix in ('_tests.c.txt', '_decl.h'):
            shutil.copy2(root / '_harness' / (cell.module + suffix), self.root / '_harness' / (cell.module + suffix))
        self.reference = (root / (cell.module + '.c')).read_text()
        self.source = self.root / (cell.module + '.c')
        self.source.write_text(self.reference)

    def _assemble(self, code):
        from decompbench.eval.functional import _maybe_rename as maybe_rename
        from decompbench.pipelines.agent4decompile_vendor.src.refinement.refiner import strip_main_from_code
        code, _ = maybe_rename(strip_main_from_code(code), 'c', self.cell.function)
        self.source.write_text(assemble(self.reference, self.cell.function, code))

    def _compile(self, code, syntax):
        try:
            self._assemble(code)
            cmd = [self.arch.cc, *self.arch.cflags, '-std=c11', '-w']
            cmd += ['-fsyntax-only', str(self.source)] if syntax else ['-c', str(self.source), '-o', str(self.root / 'candidate.o')]
            result = run_capture(cmd, cwd=self.root, timeout=60)
            return result.returncode == 0, (result.stderr or result.stdout or 'HumanEval C compile passed')[-2000:]
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)

    def evaluate_syntax(self, code):
        return self._compile(code, True)

    def evaluate_compilation(self, code):
        return self._compile(code, False)

    def evaluate_all(self, code, original_binary=None, test_cases=None,
                     constraint_level=3, exebench_data=None):
        syntax, message = self.evaluate_syntax(code)
        report = {'syntax': {'pass': syntax, 'message': message},
                  'compilation': {'pass': False, 'message': 'Fix syntax first'},
                  'execution': {'pass': None, 'message': 'Skipped', 'details': {}}}
        if syntax and constraint_level > 1:
            compiled, message = self.evaluate_compilation(code)
            report['compilation'] = {'pass': compiled, 'message': message}
            if compiled and constraint_level >= 3:
                from decomplite.humaneval_support.profile import _run_tests
                result = _run_tests(self.root, self.cell.function, self.cell.module)
                report['execution'] = {'pass': result.passed, 'message': result.output_tail,
                    'details': dict(passed=result.pass_count, total=result.total, suite='humaneval')}
        self.last_l3 = report['execution']['details']
        return report


def run_humaneval(pipeline, cell, model_key, variant):
    from decompbench.config import MODELS
    from decompbench.dataset.builder import repo_dir
    from decompbench.eval.functional import _maybe_rename as maybe_rename
    from decompbench.pipelines.ghidra import ghidra_decompile, pseudoc_cache_path
    from decompbench.pipelines.direct import strip_fences
    if cell.repo != REPO_NAME or cell.arch != 'x86_64' or MODELS[model_key]['provider'] != 'vllm':
        raise ValueError('HumanEval author refinement requires the x86-64 HumanEval profile and a configured vLLM model')
    root = repo_dir(cell.repo, 'c')
    load_manifest(root, verify_sources=True, function=cell.module)
    iterations = int(os.environ.get('A4D_AUTHOR_MAX_ITERS', '5'))
    level = int(os.environ.get('A4D_AUTHOR_CONSTRAINT_LEVEL', '3'))
    started = time.monotonic()
    ghidra, _ = ghidra_decompile(cell, stripped=stripped_variant(variant))
    initial, rename = maybe_rename(ghidra, 'c', cell.function)
    with tempfile.TemporaryDirectory(prefix='a4d_humaneval_') as temporary:
        evaluator = HumanEvalEvaluator(temporary, cell, root)
        refiner = _AuthorRefiner(model_key, evaluator, iterations, level)
        result = refiner.refine(initial_code=initial, binary_name=cell.function, decompiler='ghidra')
        details = dict(evaluator.last_l3)
    if result.error_message and result.error_message.startswith('LLM failed'):
        raise RuntimeError(result.error_message)
    extra = dict(oracle='author', oracle_suite='humaneval', oracle_arch='x86_64',
                 oracle_context='original_types_and_helpers', l3_suite_timeout_s=15,
                 oracle_transport='native', l3_qemu=False, l2_form='object',
                 author_stage=_stage(result), max_iterations=iterations, constraint_level=level,
                 iterations=result.iterations, llm_calls=refiner.llm_calls,
                 syntax_valid=result.syntax_valid, compiles=result.compiles,
                 re_executable=result.re_executable, refine_success=result.success,
                 l3_pass_count=details.get('passed'), l3_total=details.get('total'),
                 iteration_history=result.iteration_history, **rename)
    return DecompResult(text=strip_fences(result.refined_code) + '\n', raw_text=result.refined_code,
                        latency_s=time.monotonic() - started, input_tokens=refiner.input_tokens,
                        output_tokens=refiner.output_tokens, pipeline=pipeline.name,
                        system_prompt=refiner.SYSTEM_PROMPT, user_prompt=initial,
                        intermediate_path=str(pseudoc_cache_path(cell, stripped=stripped_variant(variant))),
                        sidecars={'.a4d_humaneval.json': json.dumps(extra, indent=2)}, extra=extra)
