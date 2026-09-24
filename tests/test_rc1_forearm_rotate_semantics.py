from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension" / "blender_animation_workbench"
TRANSFORM_PATH = EXT / "rigped_transform.py"
SOURCE = TRANSFORM_PATH.read_text(encoding="utf-8")
BUILDER_SOURCE = (EXT / "rigped_humanoid_builder.py").read_text(encoding="utf-8")
FIT_COMMIT_SOURCE = (EXT / "rigped_fit_commit.py").read_text(encoding="utf-8")
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


def test_rc1_lower_limb_axis_contract_is_x_hinge_y_roll_z_swivel() -> None:
    axes = _source(_function("direct_transform_axes"))
    assert "ForeArm" not in axes
    assert "Calf" not in axes
    assert "return _orientation_axes_for_control(context, selected.active_control)" in axes

    assert '"ForeArm.L": ("X",' in SOURCE
    assert '"ForeArm.R": ("X",' in SOURCE
    assert '"Calf.L": ("X",' in SOURCE
    assert '"Calf.R": ("X",' in SOURCE

    session = _source(_function("_lower_limb_special_z_session"))
    assert "local_basis.col[2]" in session
    assert "ContactKeyType.FREE" in session
    assert "ContactKeyType.SLIDING" in session
    assert '"ForeArm.L"' in session
    assert '"Calf.L"' in session

    assert "owner.lock_ik_z = True" in BUILDER_SOURCE
    assert "owner.use_ik_limit_x = True" in BUILDER_SOURCE
    assert 'return -1 if str(owner_role) in {"MCH_FOREARM.L", "MCH_FOREARM.R"} else 1' in BUILDER_SOURCE
    assert "axis_index = 0" in SNAP_SOURCE
    assert "fallback = -1" in SNAP_SOURCE


def test_rc1_lower_limb_swivel_free_follows_terminal_sliding_pins_terminal() -> None:
    solved = _source(_function("_apply_solved_two_bone_fk_pose"))
    assert "terminal_follows_second: bool = False" in solved
    assert "session.second_start_matrix.inverted_safe()" in solved
    assert "@ session.terminal_start_matrix" in solved
    assert "terminal_matrix = second_matrix @ terminal_relative" in solved
    assert "terminal_matrix.translation = desired_end" in solved

    swivel = _source(_function("_apply_lower_limb_special_z_rotation"))
    assert "terminal_follows_second: bool" in swivel
    assert "terminal_follows_second=terminal_follows_second" in swivel

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert "terminal_follows_second=not bool(self._sliding_syncs)" in preview


def test_rc1_sliding_lower_long_roll_is_shared_by_forearm_and_calf() -> None:
    replay = _source(_function("_sync_generated_sliding_lower_roll_from_public_pose"))
    assert '"MCH_ForeArm.L"' in replay
    assert '"MCH_Calf.L"' in replay
    assert '"ForeArm.L"' in replay
    assert '"Calf.L"' in replay

    live = _source(_function("_apply_sliding_lower_long_roll"))
    assert "terminal world transform fixed" in live
    assert '_RIGPED_LOCAL_AXES["Y"]' in live

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert '{"ForeArm.L", "ForeArm.R", "Calf.L", "Calf.R"}' in preview
    assert "_apply_sliding_lower_long_roll(" in preview


def test_rc1_sliding_rotate_terminal_policy_is_axis_specific() -> None:
    source = _source(_function("_apply_direct_rotate_sliding_syncs"))
    assert "pin_terminal: bool = True" in source
    assert "if session.terminal_selected or not pin_terminal" in source
    assert "else session.start_ik_state" in source

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert 'self._orientation == "LOCAL"' in preview
    assert 'self.axis == "X"' in preview
    assert "sliding_lower_global = (" in preview
    assert 'self._orientation == "GLOBAL"' in preview
    assert 'self.axis in {"X", "Y", "Z"}' in preview
    assert "sliding_lower_transports_terminal = (" in preview
    assert "sliding_lower_local_x or sliding_lower_global" in preview
    assert "pin_terminal=not sliding_lower_transports_terminal" in preview


def test_rc1_forbidden_hinge_axis_bounds_to_zero_instead_of_preserving_requested_angle() -> None:
    source = _source(_function("_bounded_hinge_direct_rotate_angle"))
    assert "effective = False" in source
    assert "effective = True" in source
    assert "if not effective:" in source
    assert "return 0.0" in source


def test_rc1_forbidden_hinge_axis_skips_sliding_fk_to_ik_sync_preview() -> None:
    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    guard = preview.index("hinge_axis_effective = sliding_lower_local_x or any(")
    early_return = preview.index(
        "if hinge_states and not generic_states and not hinge_axis_effective:"
    )
    sliding_sync = preview.index(
        "_apply_direct_rotate_sliding_syncs(",
        early_return,
    )
    assert guard < early_return < sliding_sync
    assert "self._current_angle = 0.0" in preview[early_return:sliding_sync]


def test_rc1_sliding_lower_local_x_is_effective_without_removing_forbidden_axis_guard() -> None:
    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    supported = preview.index("sliding_lower_local_x = (")
    guard = preview.index("if hinge_states and not generic_states and not hinge_axis_effective:")
    early_return = preview.index("self._current_angle = 0.0", guard)
    sliding_sync = preview.index("_apply_direct_rotate_sliding_syncs(", early_return)
    assert supported < guard < early_return < sliding_sync
    assert 'self._orientation == "LOCAL"' in preview[supported:guard]
    assert 'self.axis == "X"' in preview[supported:guard]
    assert "sliding_lower_local_x or sliding_lower_global" in preview[supported:sliding_sync]
    assert "pin_terminal=not sliding_lower_transports_terminal" in preview[sliding_sync:]


def test_rc1_local_z_sign_is_captured_once_from_frozen_joint_tangent() -> None:
    sign = _source(_function("_lower_limb_special_z_axis_sign"))
    assert "joint_offset = Vector(session.joint_world) - root" in sign
    assert "input_tangent = input_axis.cross(joint_offset)" in sign
    assert "semantic_tangent = semantic_axis.cross(joint_offset)" in sign
    assert "abs(alignment) <= 1e-4" in sign
    assert "return 1.0 if alignment > 0.0 else -1.0" in sign

    invoke = _source(_method("BAW_OT_rigped_direct_rotate_axis", "invoke"))
    assert "self._forearm_special_axis_sign = _lower_limb_special_z_axis_sign(" in invoke
    assert "_mapped_direct_rotate_axis(" in invoke
    assert 'axis_name="Z"' in invoke
    assert 'if lower_name in {"ForeArm.L", "ForeArm.R"}' not in invoke
    assert "axis_sign *= -1.0" not in invoke

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert "self._current_angle * self._forearm_special_axis_sign" in preview
    assert "terminal_follows_second=not bool(self._sliding_syncs)" in preview


def test_rc1_sliding_global_rotate_uses_shared_hinge_roll_and_transports_terminal() -> None:
    invoke = _source(_method("BAW_OT_rigped_direct_rotate_axis", "invoke"))
    assert "Sliding GLOBAL lower-link Rotate uses the same bounded hinge + axial-roll" in invoke
    assert "_sliding_global_lower_sessions" not in SOURCE
    assert "_project_global_rotation_to_sliding_lower_dofs" not in SOURCE
    assert "sliding_global_projected_dofs" not in SOURCE

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert "sliding_lower_global = (" in preview
    assert 'self._orientation == "GLOBAL"' in preview
    assert 'self.axis in {"X", "Y", "Z"}' in preview
    assert "if hinge_states:" in preview
    assert "_bounded_hinge_direct_rotate_angle(" in preview
    assert "_hinge_direct_rotate_desired(" in preview
    assert "sliding_lower_transports_terminal" in preview
    assert "pin_terminal=not sliding_lower_transports_terminal" in preview


def test_rc1_builder_forearm_roll_comes_from_chain_geometry_with_safe_fallback() -> None:
    assert 'str(entry.semantic_key) == "awb.forearm"' in BUILDER_SOURCE
    assert "bend_normal = upper.cross(lower)" in BUILDER_SOURCE
    assert "bend_normal.length > 1e-5" in BUILDER_SOURCE
    assert "local_x = -bend_normal.normalized()" in BUILDER_SOURCE
    assert "local_x = local_x - direction * float(local_x.dot(direction))" in BUILDER_SOURCE
    assert "local_z = local_x.cross(direction)" in BUILDER_SOURCE
    assert "bone.align_roll(local_z)" in BUILDER_SOURCE
    assert "bone.align_roll(reference)" in BUILDER_SOURCE


def test_rc1_fit_commit_canonicalizes_forearm_targets_as_column_basis() -> None:
    assert "def _canonical_forearm_target_orientation(" in FIT_COMMIT_SOURCE
    assert "if bend_normal.length <= 1e-5:" in FIT_COMMIT_SOURCE
    assert "return forearm_target.orientation" in FIT_COMMIT_SOURCE
    assert "local_x = -bend_normal.normalized()" in FIT_COMMIT_SOURCE
    assert "local_x = local_x - local_y * float(local_x.dot(local_y))" in FIT_COMMIT_SOURCE
    assert "local_z = local_x.cross(local_y)" in FIT_COMMIT_SOURCE
    assert "(local_x.x, local_y.x, local_z.x)" in FIT_COMMIT_SOURCE
    assert "(local_x.y, local_y.y, local_z.y)" in FIT_COMMIT_SOURCE
    assert "(local_x.z, local_y.z, local_z.z)" in FIT_COMMIT_SOURCE
    assert 'if semantic_key == "awb.forearm":' in FIT_COMMIT_SOURCE
    assert "forearm_target_names.add(str(bone.name))" in FIT_COMMIT_SOURCE
    assert "FIT_F4_FOREARM_FRAME_PARENT_MISSING" in FIT_COMMIT_SOURCE
