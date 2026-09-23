from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORM_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_transform.py"
SOURCE = TRANSFORM_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _class(name: str) -> ast.ClassDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    cls = _class(class_name)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"missing {class_name}.{method_name}")


def _source(node: ast.AST) -> str:
    segment = ast.get_source_segment(SOURCE, node)
    assert segment is not None
    return segment


def test_rc1_direct_move_auto_commit_reprojects_contact_overlay_after_depsgraph_update() -> None:
    modal = _source(_method("BAW_OT_rigped_direct_move_axis", "modal"))
    commit = modal.index("commit_rigped_auto_writer_results(tuple(deferred_auto_results))")
    tail = modal[commit:]
    depsgraph = tail.index("context.view_layer.update()")
    refresh = tail.index("_refresh_current_sliding_public_overlays(")
    rollback = tail.index("except (RuntimeError, ValueError, ReferenceError) as exc:")

    assert 0 < depsgraph < refresh < rollback
    post_refresh = tail[refresh:rollback]
    assert "capabilities=self._sliding_guard_capabilities" in post_refresh
    assert "allow_seed=False" in post_refresh
