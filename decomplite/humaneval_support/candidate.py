"""Splice one recovered HumanEval function into its original C type context."""
from __future__ import annotations


def assemble(reference: str, function: str, candidate: str) -> str:
    from decomplite.humaneval_support.syntax import top_level_function_defs, function_definition_name

    ref = reference.encode('utf-8')
    text = candidate.encode('utf-8')
    _, originals = top_level_function_defs(ref)
    _, recovered = top_level_function_defs(text)
    targets = [node for node in originals if function_definition_name(node, ref) == function]
    definitions = [node for node in recovered if function_definition_name(node, text) == function]
    if len(targets) != 1 or len(definitions) != 1:
        raise ValueError(f'Expected one reference and one recovered definition of {function}')
    existing = {function_definition_name(node, ref) for node in originals}
    additions = [node for node in recovered
                 if function_definition_name(node, text) not in existing]
    prototypes, helpers = [], []
    for node in additions:
        body = next((child for child in node.children if child.type == 'compound_statement'), None)
        if body is None:
            raise ValueError('Recovered helper has no function body')
        prototypes.append(text[node.start_byte:body.start_byte].rstrip() + b';\n')
        helpers.append(text[node.start_byte:node.end_byte] + b'\n')
    target = definitions[0]
    body = next(child for child in target.children if child.type == 'compound_statement')
    prototypes.insert(0, text[target.start_byte:body.start_byte].rstrip() + b';\n')
    replacement = b''.join(prototypes + helpers) + text[target.start_byte:target.end_byte]
    original = targets[0]
    return (ref[:original.start_byte] + replacement + ref[original.end_byte:]).decode('utf-8')
