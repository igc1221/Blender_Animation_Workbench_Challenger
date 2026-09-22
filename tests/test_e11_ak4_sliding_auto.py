from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "extension" / "blender_animation_workbench" / "phase4_writer.py"
CONTACT = ROOT / "extension" / "blender_animation_workbench" / "phase4_contact_authoring.py"
AUTO = ROOT / "extension" / "blender_animation_workbench" / "rigped_auto_key.py"
TRANSFORM = ROOT / "extension" / "blender_animation_workbench" / "rigped_transform.py"
PREFLIGHT = ROOT / "extension" / "blender_animation_workbench" / "phase4_preflight.py"


def _source(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _function(path: Path, name: str) -> str:
    source, tree = _source(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"missing function {name} in {path}")


def _method(path: Path, class_name: str, method_name: str) -> str:
    source, tree = _source(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    segment = ast.get_source_segment(source, item)
                    assert segment is not None
                    return segment
    raise AssertionError(f"missing {class_name}.{method_name} in {path}")


def test_e11_direct_writer_can_prepare_without_committing() -> None:
    execute = _function(WRITER, "execute_direct_key_plan")
    assert "defer_commit: bool = False" in execute
    assert '"DIRECT_WRITE_PREPARED"' in execute
    assert "pending_journal=journal" in execute
    assert "operation_id=plan.operation_id" in execute

    finalize = _function(WRITER, "finalize_deferred_direct_writer_result")
    assert "journal.commit()" in finalize
    assert "_COMMITTED_OPERATION_IDS.add(result.operation_id)" in finalize


def test_e11_contact_single_and_batch_can_prepare_without_committing() -> None:
    single = _function(CONTACT, "execute_contact_intent_plan")
    batch = _function(CONTACT, "execute_contact_batch_intent_plan")
    for source in (single, batch):
        assert "defer_commit: bool = False" in source
        assert "pending_journal=journal" in source
        assert "operation_id=contact_plan.operation_id" in source

    finalize = _function(CONTACT, "finalize_deferred_contact_authoring_result")
    assert "journal.commit()" in finalize


def test_e11_composite_auto_transaction_prevalidates_then_finalizes_or_rolls_back_reverse() -> None:
    commit = _function(AUTO, "commit_rigped_auto_writer_results")
    rollback = _function(AUTO, "rollback_rigped_auto_writer_results")

    assert "result.operation_id is None" in commit
    assert 'result.pending_journal.state.value != "OPEN"' in commit
    assert "finalize_deferred_direct_writer_result(result)" in commit
    assert "finalize_deferred_contact_authoring_result(result)" in commit
    assert 'trace_event(' in commit and '"AUTO_TRANSACTION_COMMIT"' in commit
    assert "for result in reversed(results):" in rollback
    assert 'journal.state.value == "OPEN"' in rollback
    assert "journal.rollback_or_raise()" in rollback


def test_e11_semantic_move_uses_one_deferred_auto_transaction_and_release_gate() -> None:
    modal = _method(TRANSFORM, "BAW_OT_rigped_semantic_move_axis", "modal")
    assert "auto_enabled_at_release = bool(" in modal
    assert "self._auto_plan if auto_enabled_at_release else None" in modal
    assert "self._auto_direct_plan if auto_enabled_at_release else None" in modal
    assert modal.count("defer_commit=True") >= 2
    assert "commit_rigped_auto_writer_results(tuple(deferred_auto_results))" in modal
    assert "rollback_rigped_auto_writer_results(tuple(deferred_auto_results))" in modal


def test_e11_fk_move_defers_contact_and_direct_until_whole_gesture_succeeds() -> None:
    modal = _method(TRANSFORM, "BAW_OT_rigped_fk_joint_move_axis", "modal")
    assert "auto_enabled_at_release = bool(" in modal
    assert "deferred_auto_contact_mapping_ids = (" in modal
    assert modal.count("defer_commit=True") >= 2
    assert modal.index("commit_fk_joint_moves(context, sessions)") < modal.index(
        "commit_rigped_auto_writer_results(tuple(deferred_auto_results))"
    )
    assert "rollback_rigped_auto_writer_results(tuple(deferred_auto_results))" in modal


def test_e11_direct_rotate_defers_mixed_contact_direct_and_rechecks_auto() -> None:
    modal = _method(TRANSFORM, "BAW_OT_rigped_direct_rotate_axis", "modal")
    assert "auto_enabled_at_release = bool(" in modal
    assert "self._deferred_auto_contact_mapping_ids" in modal
    assert modal.count("defer_commit=True") >= 2
    assert "allow_storage_rebind=True" in modal
    assert "commit_rigped_auto_writer_results(tuple(deferred_auto_results))" in modal
    assert "rollback_rigped_auto_writer_results(tuple(deferred_auto_results))" in modal


def test_e11_direct_move_rechecks_auto_at_release_before_keying() -> None:
    modal = _method(TRANSFORM, "BAW_OT_rigped_direct_move_axis", "modal")
    release_gate = modal.index("auto_enabled_at_release = bool(")
    commit = modal.index("commit_rigped_auto_direct_move(", release_gate)
    assert release_gate < commit
    assert "self._auto_plan if auto_enabled_at_release else None" in modal


def test_e11_same_frame_direct_writer_replaces_existing_key_in_place() -> None:
    write = _function(WRITER, "_write_channel_key")
    assert "existing = _key_at_frame(fcurve, time)" in write
    assert "if existing is None:" in write
    assert "existing.co.y = channel.target_value" in write
    assert "existing.handle_left.y" in write
    assert "existing.handle_right.y" in write


def test_e11_sliding_auto_does_not_invent_zero_frame_contact_baseline() -> None:
    single = _function(AUTO, "plan_rigped_auto_anchor")
    batch = _function(AUTO, "plan_rigped_auto_contact_batch")

    assert "if built.plan.target_type is ContactKeyType.FREE and current_time > 0.0:" in single
    assert "if intent.target_type is ContactKeyType.SLIDING:" in batch
    assert "continue" in batch
    assert "0F" in batch


def test_e11_body_only_direct_auto_uses_selected_direct_authority_not_contact_writer() -> None:
    move = _function(AUTO, "plan_rigped_auto_direct_move")
    direct_plan = _function(PREFLIGHT, "build_direct_key_plan")

    assert "requested_families=(ChannelFamily.POSITION,)" in move
    assert "build_contact_intent_plan" not in move
    assert "build_contact_batch_intent_plan" not in move
    assert "contracts = _selected_contracts(target, active_only=active_only)" in direct_plan
    assert "AWB_CONTACT_STATE_PROPERTY" not in direct_plan


def test_e11_sliding_anchor_reuses_existing_transform_closure_not_a_second_auto_solver() -> None:
    rows = _function(CONTACT, "_build_transform_rows")
    anchor = _function(AUTO, "plan_rigped_auto_anchor")

    assert "build_contact_intent_plan(" in anchor
    assert "ContactAuthoringMode.ANCHOR" in anchor
    assert "_channels_for_state(ik_contract" in rows
    assert "_channels_for_state(pole_contract" in rows
    assert "auto_solver" not in (anchor + rows).lower()
