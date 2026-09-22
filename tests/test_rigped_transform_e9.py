from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORM_PATH = (
    ROOT
    / "extension"
    / "blender_animation_workbench"
    / "rigped_transform.py"
)
OVERLAY_PATH = (
    ROOT
    / "extension"
    / "blender_animation_workbench"
    / "rigped_ik_pivot_overlay.py"
)
TRANSFORM_SOURCE = TRANSFORM_PATH.read_text(encoding="utf-8")
OVERLAY_SOURCE = OVERLAY_PATH.read_text(encoding="utf-8")
TRANSFORM_TREE = ast.parse(TRANSFORM_SOURCE)
OVERLAY_TREE = ast.parse(OVERLAY_SOURCE)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    cls = _class(TRANSFORM_TREE, class_name)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"missing {class_name}.{method_name}")


def _source(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def test_e9_character_sliding_membership_is_collection_not_single_limb() -> None:
    source = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_sliding_capabilities_for_character"),
    )
    assert "for mapping in view.definition.kinematics:" in source
    assert "capabilities.append(capability)" in source
    assert "return tuple(capabilities)" in source
    assert "ContactKeyType.SLIDING" in source


def test_e9_dependency_guard_capture_and_restore_cover_every_limb() -> None:
    capture = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_capture_sliding_dependency_guards"),
    )
    restore = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_restore_sliding_dependency_guards"),
    )
    hidden_restore = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_restore_sliding_hidden_seed_snapshots"),
    )

    assert "for capability in capabilities:" in capture
    assert "guards.append(" in capture
    assert "return tuple(guards)" in capture
    assert "for guard in guards:" in restore
    assert "_apply_control_state(capability.native_ik.ik_target" in restore
    assert "_restore_hinge_settings(" in restore
    assert "for snapshot in snapshots:" in hidden_restore


def test_e9_body_preview_batches_all_limb_overlays_before_stability_gate() -> None:
    refresh = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_refresh_current_sliding_public_overlays"),
    )
    sync_loop = refresh.index("for capability in capabilities:")
    update_after_sync = refresh.index(
        "context.view_layer.update()",
        refresh.index("for _pass_index in range"),
    )
    residual_loop = refresh.index(
        "for capability in capabilities:",
        refresh.index("converged = True"),
    )

    assert sync_loop < update_after_sync < residual_loop
    assert "last_position = max(last_position, position)" in refresh
    assert "last_rotation = max(last_rotation, rotation)" in refresh
    assert "if rotation > rotation_tolerance:" in refresh
    assert "stable = True" in refresh
    assert "stable = False" in refresh
    assert "if stable:" in refresh
    assert "return len(capabilities)" in refresh


def test_e9_dependency_preview_emits_per_limb_diagnostics_without_scalar_collision() -> None:
    refresh = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_refresh_frozen_sliding_dependency_overlays"),
    )
    assert "per_limb = []" in refresh
    assert "for guard, before, after, did_seed, authority in zip(" in refresh
    assert "strict=True" in refresh
    assert '"mapping_id": guard.reference.mapping_id' in refresh
    assert '"result_target_position": float(target_position)' in refresh
    assert '"public_result_rotation": float(public_rotation)' in refresh
    assert "mapping_ids=tuple(guard.reference.mapping_id for guard in guards)" in refresh
    assert "limbs=tuple(per_limb)" in refresh


def test_e9_direct_move_is_one_body_operation_with_character_wide_passive_guards() -> None:
    operator_source = _source(
        TRANSFORM_SOURCE,
        _class(TRANSFORM_TREE, "BAW_OT_rigped_direct_move_axis"),
    )
    invoke = _source(
        TRANSFORM_SOURCE,
        _method("BAW_OT_rigped_direct_move_axis", "invoke"),
    )
    modal = _source(
        TRANSFORM_SOURCE,
        _method("BAW_OT_rigped_direct_move_axis", "modal"),
    )

    assert 'bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}' in operator_source
    assert "_sliding_capabilities_for_character(" in invoke
    assert "_capture_sliding_hidden_seed_snapshots(" in invoke
    assert "_capture_sliding_dependency_guards(self._sliding_guard_capabilities)" in invoke
    assert '== "COM"' in invoke

    assert "dependency_guards=self._sliding_dependency_guards" in modal
    assert "seed_snapshots=self._sliding_seed_snapshots" in modal
    assert 'if event.type in {\"ESC\", \"RIGHTMOUSE\"}' in modal
    assert modal.count("_restore_direct_move_states(") >= 4
    assert "bpy.ops.ed.undo_push(" not in modal


def test_e9_rotate_partitions_active_and_passive_sliding_domains_without_overlap() -> None:
    invoke = _source(
        TRANSFORM_SOURCE,
        _method("BAW_OT_rigped_direct_rotate_axis", "invoke"),
    )
    restore = _source(
        TRANSFORM_SOURCE,
        _method("BAW_OT_rigped_direct_rotate_axis", "_restore_preview"),
    )

    assert "_passive_sliding_capabilities(" in invoke
    assert "active_ids & passive_ids" in invoke
    assert "active_ids | passive_ids != affected_ids" in invoke
    assert "_capture_sliding_dependency_guards(self._sliding_guard_capabilities)" in invoke
    assert "_restore_sliding_dependency_guards(" in restore
    assert "_restore_sliding_hidden_seed_snapshots(" in restore
    assert "allow_seed=False" in restore
    assert "_assert_sliding_dependency_authority(" in restore


def test_e9_passive_dependency_path_never_writes_body_compensation_or_full_body_solve() -> None:
    refresh = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_refresh_frozen_sliding_dependency_overlays"),
    )
    batch = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_refresh_current_sliding_public_overlays"),
    )
    lowered = (refresh + "\n" + batch).lower()

    for forbidden in (
        "full_body",
        "fullbody",
        "balance_solver",
        "body_compensation",
        "com.location =",
        "pelvis.location =",
        "com.matrix_basis =",
        "pelvis.matrix_basis =",
    ):
        assert forbidden not in lowered


def test_e9_red_cue_overlay_enumerates_all_four_independent_hidden_targets() -> None:
    source = _source(
        OVERLAY_SOURCE,
        _function(OVERLAY_TREE, "_ik_pivot_world_positions"),
    )
    for target in ("IK_Hand.L", "IK_Hand.R", "IK_Foot.L", "IK_Foot.R"):
        assert target in OVERLAY_SOURCE
    assert "for state_name, target_name in _LIMB_TARGETS:" in source
    assert "positions.append(" in source
