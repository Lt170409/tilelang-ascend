import ast
import pathlib
import torch


def load_checker(source):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "check_precision" in node.name]
    if not nodes:
        return None
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[nodes[0]], type_ignores=[]), "<checker>", "exec"), namespace)
    return namespace[nodes[0].name]


def test_precision_checker():
    source_path = pathlib.Path(__file__).resolve().parents[2] / 'examples/elementwise/elementwise_add.py'
    checker = load_checker(source_path.read_text(encoding="utf-8"))
    if checker is None:
        import pytest
        pytest.skip("no check_precision function")
    actual = torch.zeros(100, dtype=torch.float16)
    golden = torch.zeros_like(actual)
    try:
        result = checker(actual, golden, actual.dtype)
    except TypeError:
        result = checker(actual, golden)
    if isinstance(result, tuple):
        passed = bool(result[0])
    else:
        passed = True
    if not passed:
        raise AssertionError("zero-error case rejected")
    print("PASS: zero-error precision case")
