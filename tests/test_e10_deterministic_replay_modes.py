from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = ROOT / "extension" / "blender_animation_workbench" / "debug_replay.py"
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
    assert "replay_mode: ReplayMode | str = ReplayMode.COMMAND" in runner
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


def test_e10_user_final_runner_routes_explicit_mode_with_legacy_command_fallback() -> None:
    source, _tree = _source(RUNNER_PATH)
    semantic = _function(RUNNER_PATH, "_semantic_code")
    main = _function(RUNNER_PATH, "main")
    assert 'replay_mode={replay_mode!r}' in semantic
    assert '"replay_mode": {replay_mode!r}' in semantic
    assert 'manifest.get("replay_mode") or "COMMAND"' in main
    assert '"--replay-mode"' in main
    assert '"RECORDED_RESULT"' in source
