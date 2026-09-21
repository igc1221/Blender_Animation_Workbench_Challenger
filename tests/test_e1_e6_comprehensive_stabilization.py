from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORM_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_transform.py"
CONTACT_PATH = ROOT / "extension" / "blender_animation_workbench" / "phase4_contact_authoring.py"
REPLAY_PATH = ROOT / "extension" / "blender_animation_workbench" / "debug_replay.py"
CONTACT_UI_PATH = ROOT / "extension" / "blender_animation_workbench" / "phase4_contact_ui.py"


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


def test_b1_active_sliding_rotate_exposes_public_input_then_restores_feedback_before_native_solve() -> None:
    sync = _function(TRANSFORM_PATH, "_apply_direct_rotate_sliding_sync")
    disable_ik = sync.index("native_ik.influence = 0.0")
    expose_feedback = sync.index("constraint.mute = False", disable_ik)
    update_public = sync.index("context.view_layer.update()", expose_feedback)
    sample_hidden = sync.index("desired_terminal = capability.result_terminal.target.matrix.copy()", update_public)
    restore_feedback = sync.index("constraint.mute = bool(previous_mute)", sample_hidden)
    restore_native = sync.index("native_ik.influence = previous_ik_influence", restore_feedback)
    assert disable_ik < expose_feedback < update_public < sample_hidden < restore_feedback < restore_native


def test_b1_sliding_rotate_cancel_restores_frozen_feedback_authority() -> None:
    session = _class(TRANSFORM_PATH, "SlidingRotateSyncSession")
    assert "start_feedback_mutes" in session
    restore = _function(TRANSFORM_PATH, "_restore_direct_rotate_sliding_sync_session")
    assert "session.start_feedback_mutes" in restore
    assert "constraint.mute = bool(start_mute)" in restore


def test_b3_direct_body_move_and_rotate_own_hidden_seed_cancel_state() -> None:
    move = _class(TRANSFORM_PATH, "BAW_OT_rigped_direct_move_axis")
    rotate = _class(TRANSFORM_PATH, "BAW_OT_rigped_direct_rotate_axis")
    assert "_sliding_seed_snapshots" in move
    assert "_capture_sliding_hidden_seed_snapshots(" in move
    assert "seed_snapshots=self._sliding_seed_snapshots" in move
    assert "_sliding_seed_snapshots" in rotate
    assert "_capture_sliding_hidden_seed_snapshots(" in rotate
    assert "_restore_sliding_hidden_seed_snapshots(" in rotate
    assert "allow_seed=False" in rotate


def test_b3_seed_failure_restores_native_ik_mute_and_body_cancel_avoids_reseeding() -> None:
    seed = _function(TRANSFORM_PATH, "_seed_stalled_sliding_native_ik")
    assert "finally:" in seed
    assert "native_ik.mute = was_muted" in seed
    restore_move = _function(TRANSFORM_PATH, "_restore_direct_move_states")
    assert "_restore_sliding_hidden_seed_snapshots(" in restore_move
    assert "allow_seed=False" in restore_move


def test_b2_single_contact_feedback_authority_is_verified_before_commit() -> None:
    execute = _function(CONTACT_PATH, "execute_contact_intent_plan")
    configure = execute.index("_journal_limb_fk_feedback_muted(")
    update = execute.index("bpy.context.view_layer.update()", configure)
    pose_verify = execute.index("Contact feedback-authority transition changed", update)
    commit = execute.index("journal.commit()", pose_verify)
    assert configure < update < pose_verify < commit


def test_b2_batch_contact_feedback_authority_is_verified_before_commit() -> None:
    execute = _function(CONTACT_PATH, "execute_contact_batch_intent_plan")
    configure = execute.index("_journal_limb_fk_feedback_muted(")
    update = execute.index("bpy.context.view_layer.update()", configure)
    pose_verify = execute.index("feedback-authority transition changed", update)
    commit = execute.index("journal.commit()", pose_verify)
    assert configure < update < pose_verify < commit


def test_n1_replay_rejects_missing_scalar_result_and_validates_mixed_mapping_results() -> None:
    replay = _function(REPLAY_PATH, "_execute_contact_action")
    assert "if actual is None:" in replay
    assert 'expected_mappings = action.get("mapping_contact_types")' in replay
    assert "if not actual_mappings:" in replay
    assert "actual_mappings != normalized_expected" in replay


def test_n1_contact_capture_includes_per_mapping_results_for_mixed_batch_replay() -> None:
    operator = _class(CONTACT_UI_PATH, "BAW_OT_contact")
    assert 'replay_action["mapping_contact_types"]' in operator
    assert "for mapping_id, contact_type in result.mapping_contact_types" in operator


def test_uninitialized_contact_replay_feedback_uses_free_fk_baseline() -> None:
    sync = _function(TRANSFORM_PATH, "_sync_rigped_sliding_replay_display")
    assert "contact_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}" in sync
