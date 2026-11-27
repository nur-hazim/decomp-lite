from decomplite.evaluate import _parser, _mask_line_markers, _declared_name

def top_level_function_defs(data):
    tree = _parser().parse(_mask_line_markers(data))
    return tree, [n for n in tree.root_node.children if n.type == "function_definition"]

def function_definition_name(node, data):
    return _declared_name(node.child_by_field_name("declarator"), data)
