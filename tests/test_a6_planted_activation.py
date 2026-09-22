from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTACT_UI_PATH = ROOT / "extension" / "blender_animation_workbench" / "phase4_contact_ui.py"
CONTACT_PATH = ROOT / "extension" / "blender_animation_workbench" / "phase4_contact_authoring.py"
PIVOT_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_ik_pivot_overlay.py"
TRANSFORM_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_transform.py"
AUTO_KEY_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_auto_key.py"
DRAWING_PATH = ROOT / "extension" / "blender_animation_workbench" / "trackbar_drawing.py"


def _source(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _function(path: Path, name: str) -> str:
    source, tree = _source(path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"missing function {name} in {path}")


def _class(path: Path, name: str) -> str:
    source, tree = _source(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"missing class {name} in {path}")


def test_a6_public_contact_cycle_defaults_to_free_sliding_planted() -> None:
    enabled = _function(CONTACT_UI_PATH, "contact_enabled_types")
    assert "ContactKeyType.FREE" in enabled
    assert "ContactKeyType.SLIDING" in enabled
    assert "ContactKeyType.PLANTED" in enabled
    assert "enabled.append(ContactKeyType.PLANTED)" in enabled
    assert "sliding_enabled and bool(preferences.contact_planted_enabled)" in enabled
    assert "return default_cycle" in enabled
    assert "return tuple(enabled) or default_cycle" in enabled

    command = _function(CONTACT_PATH, "execute_contact_command")
    signature = command.split(") -> ContactAuthoringResult:", 1)[0]
    assert "ContactKeyType.FREE" in signature
    assert "ContactKeyType.SLIDING" in signature
    assert "ContactKeyType.PLANTED" in signature


def test_a6_direct_only_c_executes_normal_free_direct_plan() -> None:
    command = _function(CONTACT_PATH, "execute_contact_command")
    assert "Direct-only C is intentionally not part of this stabilization." not in command
    assert "if not mapping_ids:" in command
    assert "execute_direct_key_plan(" in command
    assert "trigger=WriterTrigger.CONTACT_AUTHORING" in command
    assert "ContactKeyType.FREE if direct_result.applied else None" in command
    assert "mapping_contact_types=()" in command


def test_a6_contact_ui_uses_unique_writer_operation_id_per_press() -> None:
    operator = _class(CONTACT_UI_PATH, "BAW_OT_contact")
    assert 'writer_operation_id = f"baw-contact:{trace_operation_id}"' in operator
    assert "operation_id=writer_operation_id" in operator
    assert 'operation_id="baw-contact"' not in operator


def test_a6_same_frame_body_auto_refreshes_only_existing_planted_activation_rotation_rows() -> None:
    helper = _function(
        AUTO_KEY_PATH,
        "_augment_direct_plan_with_planted_activation_fk_dependencies",
    )
    assert "previous_type is not ContactKeyType.FREE" in helper
    assert "exact_type is not ContactKeyType.PLANTED" in helper
    assert "capability.fk_binding_ids" in helper
    assert "capability.authored_terminal_binding_id" in helper
    assert "AllocationIntent.EXISTING_FCURVE" in helper
    assert "_key_exists_at_time(curve, time)" in helper
    assert "dependency_footprint=replace(" in helper
    assert "write_footprint=replace(" in helper
    assert "AWB_CONTACT_STATE_PROPERTY" in helper
    assert "hold" not in helper.lower()
    assert "ik_target" not in helper.lower()

    planner = _function(AUTO_KEY_PATH, "plan_rigped_auto_direct_move")
    committer = _function(AUTO_KEY_PATH, "commit_rigped_auto_direct_move")
    for source in (planner, committer):
        assert "_augment_direct_plan_with_planted_activation_fk_dependencies(" in source

    direct_move = _class(TRANSFORM_PATH, "BAW_OT_rigped_direct_move_axis")
    assert "contact_activation_mapping_ids=(" in direct_move
    assert "_sliding_capability_mapping_ids(self._sliding_guard_capabilities)" in direct_move


def test_a6_planted_preference_is_exposed_and_default_on() -> None:
    preferences = _class(CONTACT_UI_PATH, "BAW_AP_preferences")
    assert 'contact_planted_enabled: BoolProperty(' in preferences
    planted = preferences.split('contact_planted_enabled: BoolProperty(', 1)[1].split(")", 1)[0]
    assert 'description="Include Planted in the Contact cycle"' in planted
    assert "default=True" in planted

    draw = _function(CONTACT_UI_PATH, "draw")
    assert 'row.prop(self, "contact_planted_enabled", toggle=True)' in draw




def test_a6_max_style_planted_requires_and_inherits_active_ik_anchor() -> None:
    planner = _function(CONTACT_PATH, "build_contact_intent_plan")
    assert "target_type is ContactKeyType.PLANTED and physical_type is ContactKeyType.FREE" in planner
    assert "Planted requires an active Sliding/Planted IK contact anchor; author Sliding first." in planner
    assert "physical_type is ContactKeyType.SLIDING" in planner
    assert "inherit_active_anchor=inherit_active_anchor" in planner

    single = _function(CONTACT_PATH, "execute_contact_intent_plan")
    batch = _function(CONTACT_PATH, "_prepare_contact_intent_for_batch")
    for source in (single, batch):
        assert "if intent.inherit_active_anchor" in source
        assert "capability.native_ik.ik_target.target.matrix.copy()" in source
        assert "_hold_state_for_contact_point(hold, contact_matrix, local_point)" in source
        assert "_point_state_for_local_contact(hold, contact_matrix, local_point)" in source


def test_a6_authoritative_red_pivot_includes_sliding_and_planted() -> None:
    source = PIVOT_PATH.read_text(encoding="utf-8")
    assert "int(ContactStateValue.SLIDING)" in source
    assert "int(ContactStateValue.PLANTED)" in source
    assert "_authoritative_contact_state_at_frame(" in source


def test_a6_planted_track_bar_color_remains_light_blue() -> None:
    source = DRAWING_PATH.read_text(encoding="utf-8")
    assert "ContactKeyType.PLANTED: (0.36, 0.72, 0.98, 1.0)" in source


def test_a6_planted_replay_skips_public_fk_joint_reclamp() -> None:
    clamp = _function(TRANSFORM_PATH, "apply_rigped_joint_limits_to_rig")
    assert "ContactKeyType.SLIDING" in clamp
    assert "ContactKeyType.PLANTED" in clamp
    assert "contact_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}" in clamp


def test_a6_planted_replay_reuses_straight_limb_stall_seed_without_new_authority() -> None:
    replay = _function(TRANSFORM_PATH, "_sync_rigped_sliding_replay_display")
    assert "contact_type is ContactKeyType.PLANTED" in replay
    assert "_seed_stalled_sliding_native_ik(" in replay
    assert "contact_type not in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}" in replay


def test_a6_live_body_overlay_membership_includes_planted() -> None:
    collector = _function(TRANSFORM_PATH, "_sliding_capabilities_for_character")
    assert "ContactKeyType.SLIDING" in collector
    assert "ContactKeyType.PLANTED" in collector
    assert "contact_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}" in collector


def test_a6_planted_move_remains_fail_closed() -> None:
    source = TRANSFORM_PATH.read_text(encoding="utf-8")
    assert "Planted semantic Move is fail-closed; release or slide the Contact first." in source
