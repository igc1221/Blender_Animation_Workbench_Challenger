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
    for node in _class(class_name).body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"missing {class_name}.{method_name}")


def _source(node: ast.AST) -> str:
    segment = ast.get_source_segment(SOURCE, node)
    assert segment is not None
    return segment


def test_direct_rotate_owns_exact_gesture_undo_boundary() -> None:
    cls = _source(_class("BAW_OT_rigped_direct_rotate_axis"))
    header = cls[: cls.index("axis: EnumProperty")]
    assert 'bl_options: ClassVar[set[str]] = {"REGISTER", "BLOCKING"}' in header

    modal = _source(_method("BAW_OT_rigped_direct_rotate_axis", "modal"))
    start_push = modal.index(
        'bpy.ops.ed.undo_push(message="AWB Rigped Direct Rotate Start")'
    )
    restore = modal.rfind("self._restore_preview(context)", 0, start_push)
    replay = modal.index("self._apply_preview(context)", start_push)
    capture = modal.index("_capture_direct_rotate_recorded_result(", replay)
    final_push = modal.index(
        'bpy.ops.ed.undo_push(message="AWB Rigped Direct Rotate")',
        capture,
    )
    assert restore < start_push < replay < capture < final_push
