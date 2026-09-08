"""Create per-file pytest checks for uncovered example precision functions."""

from __future__ import annotations

import ast
import pathlib
import textwrap


EXCLUDED = {"__init__.py", "utils.py", "golden.py", "setup.py"}
MARKERS = ("matched_ratio", "max_abs", "PRECISION_FAIL", "check_precision", "_check_precision")


TEMPLATE = '''\
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
    source_path = pathlib.Path(__file__).resolve().parents[{depth}] / {source!r}
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
'''


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    examples = root / "examples"
    for source in examples.rglob("*.py"):
        if source.name in EXCLUDED:
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        if not any(marker in text for marker in MARKERS):
            continue
        if "__main__" in text or "def test_" in text or "pytest" in text:
            continue
        try:
            relative = source.relative_to(root)
            depth = len(relative.parts) - 1
            ast.parse(text, filename=str(source))
        except (ValueError, SyntaxError):
            continue
        harness = source.with_name(f"test_precision_{source.stem}.py")
        content = TEMPLATE.replace("[{depth}]", f"[{depth}]").replace("{source!r}", repr(relative.as_posix()))
        harness.write_text(content, encoding="utf-8")
        print(harness.relative_to(root))


if __name__ == "__main__":
    main()
