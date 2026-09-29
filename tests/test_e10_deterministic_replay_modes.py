from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = ROOT / "extension" / "blender_animation_workbench" / "debug_replay.py"
TRACE_PATH = ROOT / "extension" / "blender_animation_workbench" / "debug_trace.py"
TRACKBAR_GIZMO_PATH = ROOT / "extension" / "blender_animation_workbench" / "trackbar_gizmo.py"
RIGPED_TRANSFORM_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_transform.py"
CONTACT_PATH = ROOT / "extension" / "blender_animation_workbench" / "phase4_contact_authoring.py"
CONTACT_UI_PATH = ROOT / "extension" / "blender_animation_workbench" / "phase4_contact_ui.py"
RUNNER_PATH = ROOT / "scripts" / "replay_user_final_test_via_mcp.py"


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


def test_e10_replay_mode_is_explicit_and_reported() -> None:
    replay_mode = _class(REPLAY_PATH, "ReplayMode")
    runner = _function(REPLAY_PATH, "run_semantic_replay")
    assert 'RECORDED_RESULT = "RECORDED_RESULT"' in replay_mode
    assert 'COMMAND = "COMMAND"' in replay_mode
    assert "replay_mode: ReplayMode | str," in runner
    assert "replay_mode: ReplayMode | str =" not in runner
    assert "resolved_replay_mode = ReplayMode(replay_mode)" in runner
    assert '"replay_mode": resolved_replay_mode.value' in runner


def test_e10_command_mode_uses_current_command_without_historical_equality_gate() -> None:
    command = _function(REPLAY_PATH, "_execute_contact_command_action")
    assert "execute_contact_command(" in command
    assert "contact_enabled_types(context)" in command
    assert 'payload["recorded_contact_type"]' in command
    assert 'payload["recorded_mapping_contact_types"]' in command
    assert "RECORDED_RESULT_MISMATCH" not in command


def test_e10_recorded_result_bypasses_command_semantics_and_current_enabled_types() -> None:
    recorded = _function(REPLAY_PATH, "_execute_contact_recorded_result_action")
    assert "resolve_operation_domain(" in recorded
    assert "execute_contact_command(" not in recorded
    assert "contact_enabled_types(context)" not in recorded
    assert "enabled_types=_RECORDED_CONTACT_TYPES" in recorded
    assert "forced_cycle_type=target_type" in recorded
    assert "forced_mapping_types=dict(ordered_targets)" in recorded


def test_e10_recorded_result_fails_closed_on_incomplete_or_mismatched_coverage() -> None:
    recorded = _function(REPLAY_PATH, "_execute_contact_recorded_result_action")
    assert "legacy broad-C history has direct controls but no recorded direct-key coverage" in recorded
    assert "UNSUPPORTED_DIRECT_REPLAY" in recorded
    assert "multi-mapping recorded replay requires explicit mapping_contact_types coverage" in recorded
    assert "MAPPING_COVERAGE_MISMATCH" in recorded
    assert "single-mapping recorded replay is missing contact_type" in recorded
    assert "recorded Planted replay lacks a frozen plant-space/contact-point payload" in recorded


def test_e10_single_recorded_result_validates_actual_mapping_coverage() -> None:
    recorded = _function(REPLAY_PATH, "_execute_contact_recorded_result_action")
    single_execute = _function(CONTACT_PATH, "execute_contact_intent_plan")
    assert "expected_mapping_result = ((mapping_id, target_type),)" in recorded
    assert "actual_mappings != expected_mapping_result" in recorded
    assert "mapping_contact_types=((intent.mapping_id, intent.target_type),)" in single_execute


def test_e10_batch_planner_accepts_explicit_per_mapping_targets_before_intent_build() -> None:
    batch = _function(CONTACT_PATH, "build_contact_batch_intent_plan")
    assert "forced_mapping_types: dict[str, ContactKeyType] | None = None" in batch
    coverage = batch.index("I20_FORCED_MAPPING_COVERAGE_MISMATCH")
    intent_build = batch.index("intents: list[ContactIntentPlan]")
    assert coverage < intent_build
    assert "if mode is ContactAuthoringMode.CYCLE and forced_targets is None:" in batch
    assert "forced_targets[str(mapping.mapping_id)]" in batch


def test_e10_new_contact_capture_records_direct_binding_coverage() -> None:
    operator = _class(CONTACT_UI_PATH, "BAW_OT_contact")
    assert "resolve_operation_domain(context.scene, control_context)" in operator
    assert "replay_domain.snapshot.supported_direct_binding_ids" in operator
    assert 'replay_action["direct_binding_ids"] = replay_direct_binding_ids' in operator


def test_e10_user_final_runner_requires_explicit_mode_without_hidden_fallback() -> None:
    source, _tree = _source(RUNNER_PATH)
    semantic = _function(RUNNER_PATH, "_semantic_code")
    user_final = _function(RUNNER_PATH, "_resolve_user_final")
    main = _function(RUNNER_PATH, "main")
    assert 'replay_mode={replay_mode!r}' in semantic
    assert '"replay_mode": {replay_mode!r}' in semantic
    assert 'bpy.ops.ed.undo_push(message=f"AWB Replay Resume {test_id}")' in semantic
    assert '"FINISHED" not in undo_checkpoint' in semantic
    assert 'manifest.get("replay_mode")' in user_final
    assert "Semantic replay requires COMMAND or RECORDED_RESULT mode." in user_final
    assert 'or "COMMAND"' not in user_final
    assert '"--replay-mode"' in main
    assert '"RECORDED_RESULT"' in source


def test_e10_latest_replay_retains_three_distinct_sessions() -> None:
    source, _tree = _source(TRACE_PATH)
    rotate = _function(TRACE_PATH, "_rotate_replay_history_for_new_session")
    persist = _function(TRACE_PATH, "persist_latest_replay_script")
    assert '_REPLAY_PREVIOUS_FILENAME = "awb_replay_previous.json"' in source
    assert '_REPLAY_PREVIOUS2_FILENAME = "awb_replay_previous2.json"' in source
    assert 'incoming_session == current_session' in rotate
    assert '_write_replay_script_file(previous2_path, previous)' in rotate
    assert '_write_replay_script_file(previous_path, current)' in rotate
    assert "_rotate_replay_history_for_new_session(script)" in persist


def test_e10_key_edit_commit_is_frozen_into_latest_replay() -> None:
    trace_source, _tree = _source(TRACE_PATH)
    multi_drag = _function(TRACKBAR_GIZMO_PATH, "_finish_multi_key_drag")
    single_drag = _function(TRACKBAR_GIZMO_PATH, "_finish_key_drag")
    for commit in (multi_drag, single_drag):
        assert '"KEY_EDIT_COMMIT"' in commit
        assert '"kind": "KEY_EDIT"' in commit
        assert '"source_frames"' in commit
        assert '"delta_frames"' in commit
        assert '"controls"' in commit
    assert '"KEY_EDIT_COMMIT"' in trace_source


def test_e10_key_edit_replay_uses_production_trackbar_writers() -> None:
    execute = _function(REPLAY_PATH, "_execute_key_edit_action")
    route = _function(REPLAY_PATH, "_semantic_replay_action_route")
    runner = _function(REPLAY_PATH, "run_semantic_replay")
    assert "move_selected_key_frames_for_context(" in execute
    assert "clone_selected_key_frames_for_context(" in execute
    assert 'kind == "KEY_EDIT"' in route
    assert 'elif kind == "KEY_EDIT"' in runner


def test_e10_archived_semantic_replays_declare_mode_explicitly() -> None:
    expected = {
        "E2": "RECORDED_RESULT",
        "E3": "COMMAND",
        "E4": "COMMAND",
        "E5": "COMMAND",
        "E6": "COMMAND",
        "E7": "RECORDED_RESULT",
        "E8": "RECORDED_RESULT",
        "E9": "RECORDED_RESULT",
    }
    for test_id, replay_mode in expected.items():
        manifest_path = ROOT / "debug" / "user_final_tests" / "core" / "a5" / test_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["runner"] == "semantic_replay"
        assert manifest["replay_mode"] == replay_mode


def test_e10_rotate_recorded_result_captures_operation_owned_state() -> None:
    capture = _function(RIGPED_TRANSFORM_PATH, "_capture_direct_rotate_recorded_result")
    rigped_source, _tree = _source(RIGPED_TRANSFORM_PATH)
    assert '"schema": "awb-direct-rotate-recorded-result/v1"' in capture
    assert '"pose_states"' in capture
    assert '"sliding"' in capture
    assert "capability.native_ik.solver_owner" in capture
    assert "capability.native_ik.ik_target" in capture
    assert "capability.native_ik.pole_target" in capture
    assert "_capture_hinge_settings(solver_owner)" in capture
    assert "recorded_result = _capture_direct_rotate_recorded_result(" in rigped_source
    assert '"recorded_result": recorded_result' in rigped_source


def test_e10_rotate_recorded_result_is_wired_and_legacy_semantic_replay_fails_closed() -> None:
    execute = _function(REPLAY_PATH, "_execute_free_direct_rotate_action")
    validate = _function(REPLAY_PATH, "_validate_direct_rotate_recorded_result")
    restore = _function(REPLAY_PATH, "_apply_direct_rotate_recorded_result")
    runner = _function(REPLAY_PATH, "run_semantic_replay")
    assert 'recorded_result = action.get("recorded_result")' in execute
    assert "_apply_direct_rotate_recorded_result(" in execute
    assert "UNSUPPORTED_LEGACY_ROTATE_RESULT" in execute
    assert "UNSUPPORTED_COMMAND_ROTATE_SEMANTICS" in execute
    assert '"awb-direct-rotate-recorded-result/v1"' in validate
    assert "_restore_hinge_settings(solver_owner, row[\"hinge_settings\"])" in restore
    assert "_refresh_current_sliding_public_overlays(" in restore
    assert "replay_mode=resolved_replay_mode" in runner


def test_e10_rotate_recorded_result_prevalidates_full_coverage_before_mutation() -> None:
    validate = _function(REPLAY_PATH, "_validate_direct_rotate_recorded_result")
    apply_result = _function(REPLAY_PATH, "_apply_direct_rotate_recorded_result")
    execute = _function(REPLAY_PATH, "_execute_free_direct_rotate_action")

    assert "duplicate pose identities" in validate
    assert "pose coverage mismatch" in validate
    assert "duplicate Sliding mappings" in validate
    assert "solver identity mismatch" in validate
    assert "Sliding state is incomplete" in validate
    assert "feedback coverage mismatch" in validate
    assert "hinge settings are incomplete" in validate
    assert apply_result.index("_validate_direct_rotate_recorded_result(") < apply_result.index("constraint.influence = 0.0")
    assert "del solver_owner[AWB_CONTACT_STATE_PROPERTY]" in apply_result
    assert "recorded_result,\n            resolved,\n            sliding_syncs," in execute
    assert "if resolved_for_delta:" in execute
    assert execute.index("if resolved_for_delta:") < execute.index("delta_world_quaternion")
    assert "semantic_rotate_requires_result = bool(sliding_syncs) or len(resolved) > 1" in execute
    assert "quaternion_values: tuple[float, ...] = ()" in execute
    assert "if bool(getattr(scene, \"baw_auto_key_enabled\", False)):" in execute
    assert "replay_mode is ReplayMode.COMMAND and bool(" not in execute
    assert "multi-control/Sliding" in execute