"""Install a complete, evaluated HumanEval export as a local benchmark repo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

REPO_NAME = 'multipl_e_humaneval'
MANIFEST = 'humaneval-workload.json'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rename_symbol(text: str, old: str, new: str) -> str:
    """Rename C identifier tokens, preserving comments and string literals."""
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/|\b[A-Za-z_]\w*\b')
    return tokens.sub(lambda match: new if match.group() == old else match.group(), text)


def load_manifest(root: Path, *, verify_sources=False, function: str | None = None) -> dict:
    path = root / MANIFEST
    if not path.is_file():
        raise ValueError(f'HumanEval workload not installed at {root}; use python -m decomplite.humaneval prepare --help')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    functions = manifest.get('functions', {})
    if manifest.get('repo') != REPO_NAME or len(functions) != 161:
        raise ValueError('HumanEval workload must contain exactly 161 functions')
    if function is not None and function not in functions:
        raise ValueError(f'Unknown HumanEval target {function}')
    selected = functions if function is None else {function: functions[function]}
    for name, entry in selected.items():
        if not re.fullmatch(r'he_[A-Za-z_][A-Za-z_0-9]*', name):
            raise ValueError(f'Invalid HumanEval symbol: {name!r}')
        source = root / f'{name}.c'
        test = root / '_harness' / f'{name}_tests.c.txt'
        header = root / '_harness' / f'{name}_decl.h'
        if not source.is_file() or not test.is_file() or not header.is_file():
            raise ValueError(f'Missing source or harness for {name}')
        if verify_sources and digest(source.read_bytes()) != entry['source_sha256']:
            raise ValueError(f'Reference source changed: {source}')
        if digest(test.read_bytes()) != entry['tests_sha256'] or digest(header.read_bytes()) != entry['header_sha256']:
            raise ValueError(f'Frozen harness changed for {name}')
    return manifest


def install(accepted: Path, expected: Path, destination: Path) -> dict:
    accepted, expected, destination = map(lambda p: Path(p).resolve(), (accepted, expected, destination))
    export = json.loads((accepted / 'corpus-manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in expected.read_text(encoding='utf-8').splitlines() if line.strip()]
    frozen = {r['name']: r for r in rows}
    if len(rows) != 161 or len(frozen) != 161:
        raise ValueError('Expected exactly 161 distinct frozen HumanEval tasks')
    if (export.get('complete') is not True or export.get('tasks') != 161
            or export.get('tasks_with_passing_candidate') != 161):
        raise ValueError('Refusing partial corpus: all 161 tasks must have a passing C candidate')
    if export.get('prompts_sha256') != digest(expected.read_bytes()):
        raise ValueError('Accepted export was evaluated against a different prompt file')
    selected = {e['name']: e for e in export['accepted']}
    if len(export['accepted']) != 161 or set(selected) != set(frozen):
        raise ValueError('Accepted task IDs do not match the complete frozen corpus')
    prepared = {}
    for task, row in frozen.items():
        if row.get('language') != 'c' or not re.fullmatch(r'HumanEval_\d+_[A-Za-z_][A-Za-z_0-9]*', task):
            raise ValueError(f'Unexpected task/language: {task}')
        original = 'he_' + re.sub(r'^HumanEval_\d+_', '', task)
        function = 'he_' + task
        if function in prepared:
            raise ValueError(f'Duplicate target symbol {function}')
        prefix = row['prompt'].rstrip()
        if not prefix.endswith('{') or not re.search(r'\b' + re.escape(original) + r'\s*\(', prefix.splitlines()[-1]):
            raise ValueError(f'Frozen interface does not declare {function}')
        source = (accepted / f'{task}.c').read_bytes()
        tests = (accepted / f'{task}_tests.c.txt').read_bytes()
        if digest(source) != selected[task]['source_sha256']:
            raise ValueError(f'Accepted source hash mismatch: {task}')
        if tests.decode('utf-8') != row['tests']:
            raise ValueError(f'Accepted tests differ from frozen tests: {task}')
        source_original_sha256, tests_original_sha256 = digest(source), digest(tests)
        source = rename_symbol(source.decode('utf-8'), original, function).encode('utf-8')
        tests = rename_symbol(tests.decode('utf-8'), original, function).encode('utf-8')
        header = rename_symbol(prefix[:-1].rstrip() + ';\n', original, function).encode('utf-8')
        from decomplite.humaneval_support.syntax import top_level_function_defs, function_definition_name
        _, definitions = top_level_function_defs(source)
        targets = [node for node in definitions if function_definition_name(node, source) == function]
        if len(targets) != 1:
            raise ValueError(f'{task}: expected one complete definition of {original}')
        storage = {source[node.start_byte:node.end_byte] for node in targets[0].children
                   if node.type == 'storage_class_specifier'}
        if storage & {b'static', b'inline'}:
            raise ValueError(f'{task}: benchmark target must be a non-static, non-inline exported function; repair and reevaluate it first')
        prepared[function] = (source, tests, header, dict(task=task, original_symbol=original,
            source_original_sha256=source_original_sha256, tests_original_sha256=tests_original_sha256,
            candidate_index=selected[task]['candidate_index'], source_sha256=digest(source),
            tests_sha256=digest(tests), header_sha256=digest(header),
            assertions=tests.count(b'assert(')))
    manifest = dict(repo=REPO_NAME, prompts_sha256=digest(expected.read_bytes()),
                    export_manifest_sha256=digest((accepted / 'corpus-manifest.json').read_bytes()),
                    functions={fn: values[3] for fn, values in sorted(prepared.items())})
    if destination.exists():
        existing = load_manifest(destination, verify_sources=True)
        if existing != manifest:
            raise ValueError('A different HumanEval corpus is already installed. Use a separate checkout; cached builds must not be silently reused.')
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.humaneval-import-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'corpus'
        (staged / '_harness').mkdir(parents=True)
        for function, (source, tests, header, _) in prepared.items():
            (staged / f'{function}.c').write_bytes(source)
            (staged / '_harness' / f'{function}_tests.c.txt').write_bytes(tests)
            (staged / '_harness' / f'{function}_decl.h').write_bytes(header)
        (staged / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        load_manifest(staged, verify_sources=True)
        for attempt in range(5):
            try:
                staged.rename(destination)
                break
            except PermissionError:
                # Windows scanners can briefly hold freshly written C files;
                # never remove/overwrite a destination to work around a lock.
                if os.name != 'nt' or attempt == 4:
                    raise
                time.sleep(0.1)
    return manifest


