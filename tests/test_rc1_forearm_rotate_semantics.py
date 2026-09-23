from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension" / "blender_animation_workbench"
TRANSFORM_PATH = EXT / "rigped_transform.py"
SOURCE = TRANSFORM_PATH.read_text(encoding="utf-8")
BUILDER_SOURCE = (EXT / "rigped_humanoid_builder.py").read_text(encoding="utf-8")
SNAP_SOURCE = (EXT / "phase4_representation_snap.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


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


def test_rc1_forearm_axis_contract_is_native_local_x_hinge_y_roll_z_swivel() -> None:
    axes = _source(_function("direct_transform_axes"))
    assert "ForeArm" not in axes
    assert "return _orientation_axes_for_control(context, selected.active_control)" in axes

    assert '"ForeArm.L": ("X",' in SOURCE
    assert '"ForeArm.R": ("X",' in SOURCE
    assert '"Calf.L": ("X",' in SOURCE
    assert '"Calf.R": ("X",' in SOURCE

    session = _source(_function("_forearm_special_z_session"))
    assert "local_basis.col[2]" in session
    assert "ContactKeyType.FREE" in session

    assert "owner.lock_ik_z = True" in BUILDER_SOURCE
    assert "owner.use_ik_limit_x = True" in BUILDER_SOURCE
    assert 'return -1 if str(owner_role) in {"MCH_FOREARM.L", "MCH_FOREARM.R"} else 1' in BUILDER_SOURCE
    assert "axis_index = 0" in SNAP_SOURCE
    assert "fallback = -1" in SNAP_SOURCE


def test_rc1_forearm_swivel_keeps_wrist_position_and_hand_follows_forearm() -> None:
    solved = _source(_function("_apply_solved_two_bone_fk_pose"))
    assert "terminal_follows_second: bool = False" in solved
    assert "session.second_start_matrix.inverted_safe()" in solved
    assert "@ session.terminal_start_matrix" in solved
    assert "terminal_matrix = second_matrix @ terminal_relative" in solved
    assert "terminal_matrix.translation = desired_end" in solved

    swivel = _source(_function("_apply_forearm_special_z_rotation"))
    assert "terminal_follows_second=True" in swivel


def test_rc1_forbidden_hinge_axis_bounds_to_zero_instead_of_preserving_requested_angle() -> None:
    source = _source(_function("_bounded_hinge_direct_rotate_angle"))
    assert "effective = False" in source
    assert "effective = True" in source
    assert "if not effective:" in source
    assert "return 0.0" in source


def test_rc1_forbidden_hinge_axis_skips_sliding_fk_to_ik_sync_preview() -> None:
    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    guard = preview.index("hinge_axis_effective = any(")
    early_return = preview.index(
        "if hinge_states and not generic_states and not hinge_axis_effective:"
    )
    sliding_sync = preview.index(
        "_apply_direct_rotate_sliding_sync(context, session)",
        early_return,
    )
    assert guard < early_return < sliding_sync
    assert "self._current_angle = 0.0" in preview[early_return:sliding_sync]
