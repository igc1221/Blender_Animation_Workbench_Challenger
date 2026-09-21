from __future__ import annotations

import ast
from pathlib import Path

SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "rigped_transform.py"
)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
DEBUG_REPLAY_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "debug_replay.py"
).read_text(encoding="utf-8")
DEBUG_REPLAY_TREE = ast.parse(DEBUG_REPLAY_SOURCE)


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _class(name: str) -> ast.ClassDef:
    for node in TREE.body:
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


def test_e7_direct_move_consumes_one_frozen_operation_domain() -> None:
    begin = _source(_function("_begin_direct_move_states"))
    assert "operation_domain: OperationDomainSnapshot" in begin
    assert "operation_domain.selected_control_runtime_keys" in begin
    assert "operation_domain.active_control_runtime_key" in begin
    assert "operation_domain.contact_mapping_ids" in begin
    assert "_selected_direct_move_controls(" not in begin
    assert "resolve_operation_domain(" not in begin

    invoke = _source(_method("BAW_OT_rigped_direct_move_axis", "invoke"))
    assert invoke.index("resolve_operation_domain(") < invoke.index("_begin_direct_move_states(")
    assert "_sliding_capabilities_for_character(" in invoke
    assert "operation_domain.character_id" in invoke
    assert "_current_sliding_capabilities(" not in invoke


def test_e7_move_guard_is_com_only_and_root_remains_e8() -> None:
    invoke = _source(_method("BAW_OT_rigped_direct_move_axis", "invoke"))
    assert '== "COM"' in invoke
    assert "_capture_sliding_dependency_guards(" in invoke

    apply = _source(_function("_apply_direct_move_delta"))
    assert "dependency_guards:" in apply
    assert "_refresh_frozen_sliding_dependency_overlays(" in apply
    assert "_refresh_current_sliding_public_overlays(" in apply


def test_e7_rotate_guard_is_passive_only_and_scoped_to_body_roles() -> None:
    invoke = _source(_method("BAW_OT_rigped_direct_rotate_axis", "invoke"))
    assert "_passive_sliding_capabilities(" in invoke
    assert "_capture_sliding_dependency_guards(" in invoke
    assert "not self._sliding_syncs" in invoke
    for role in ("COM", "Pelvis", "Spine", "Head", "Clavicle.L", "Clavicle.R"):
        assert f'"{role}"' in invoke
    assert '"Root"' not in invoke
    assert "active_ids & passive_ids" in invoke
    assert "active_ids | passive_ids != affected_ids" in invoke

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert "_apply_direct_rotate_sliding_sync(context, session)" in preview
    assert "_refresh_frozen_sliding_dependency_overlays(" in preview
    assert "capabilities=self._sliding_guard_capabilities" in preview


def test_e7_authority_guard_hard_fails_target_pole_drift_only() -> None:
    capture = _source(_function("_capture_sliding_dependency_guards"))
    assert "capture_sliding_authored_reference(capability)" in capture
    assert "start_ik_influence" in capture
    assert "start_ik_mute" in capture
    assert "start_terminal_ik_influence" in capture
    assert "start_terminal_ik_mute" in capture
    assert "start_hinge_settings" in capture
    assert "start_result_terminal" in capture
    assert "start_public_terminal" in capture

    diagnostics = _source(_function("_sliding_dependency_guard_diagnostics"))
    assert "target_position > matrix_position_tolerance" in diagnostics
    assert "target_rotation > matrix_rotation_tolerance" in diagnostics
    assert "pole_position > matrix_position_tolerance" in diagnostics
    assert "pole_rotation > matrix_rotation_tolerance" in diagnostics
    assert "pole_angle_delta > scalar_tolerance" in diagnostics
    assert "result_target" not in diagnostics
    assert "public_result" not in diagnostics

    assertion = _source(_function("_assert_sliding_dependency_authority"))
    assert '"SLIDING_BODY_AUTHORITY_DRIFT"' in assertion
    assert "raise RigpedSemanticMoveError" in assertion


def test_e7_preview_uses_frozen_set_and_measurement_only_residuals() -> None:
    refresh = _source(_function("_refresh_frozen_sliding_dependency_overlays"))
    assert "_seed_stalled_sliding_native_ik(" in refresh
    assert "_refresh_current_sliding_public_overlays(" in refresh
    assert "capabilities=capabilities" in refresh
    assert "allow_seed=False" in refresh
    assert '"result_target_position"' in refresh
    assert '"result_target_rotation"' in refresh
    assert '"public_result_position"' in refresh
    assert '"public_result_rotation"' in refresh
    assert '"SLIDING_BODY_DEPENDENCY_PREVIEW"' in refresh
    assert "raise RigpedSemanticMoveError" not in refresh


def test_e7_restore_owns_authority_and_seed_without_reseeding() -> None:
    move_restore = _source(_function("_restore_direct_move_states"))
    assert "_restore_sliding_dependency_guards(" in move_restore
    assert "_restore_sliding_hidden_seed_snapshots(" in move_restore
    assert "allow_seed=False" in move_restore
    assert "_assert_sliding_dependency_authority(" in move_restore

    rotate_restore = _source(
        _method("BAW_OT_rigped_direct_rotate_axis", "_restore_preview")
    )
    assert "_restore_sliding_dependency_guards(" in rotate_restore
    assert "_restore_sliding_hidden_seed_snapshots(" in rotate_restore
    assert "allow_seed=False" in rotate_restore
    assert "_assert_sliding_dependency_authority(" in rotate_restore



def test_e7_semantic_replay_direct_move_uses_frozen_domain_and_guard() -> None:
    replay = next(
        node
        for node in ast.walk(DEBUG_REPLAY_TREE)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_execute_direct_move_action"
    )
    source = ast.get_source_segment(DEBUG_REPLAY_SOURCE, replay)
    assert source is not None
    assert "resolve_operation_domain(scene, control_context)" in source
    assert "_begin_direct_move_states(control_context, operation_domain)" in source
    assert "_sliding_capabilities_for_character(" in source
    assert "operation_domain.character_id" in source
    assert "_capture_sliding_dependency_guards(frozen_sliding)" in source
    assert "dependency_guards=dependency_guards" in source
    assert "capabilities=frozen_sliding" in source
    assert "_begin_direct_move_states(context)" not in source
