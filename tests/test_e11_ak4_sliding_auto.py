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


def test_e11_composite_auto_transaction_prevalidates_then_group_commits_or_rolls_back_reverse() -> None:
    commit = _function(AUTO, "commit_rigped_auto_writer_results")
    rollback = _function(AUTO, "rollback_rigped_auto_writer_results")

    assert "result.operation_id is None" in commit
    assert 'result.pending_journal.state.value != "OPEN"' in commit
    assert "MutationJournal.commit_group(" in commit
    assert commit.index("MutationJournal.commit_group(") < commit.index(
        "finalize_deferred_direct_writer_result(result)"
    )
    assert commit.index("MutationJournal.commit_group(") < commit.index(
        "finalize_deferred_contact_authoring_result(result)"
    )
    assert 'trace_event(' in commit and '"AUTO_TRANSACTION_COMMIT"' in commit
    assert "for result in reversed(results):" in rollback
    assert 'journal.state.value == "OPEN"' in rollback
    assert "journal.rollback()" in rollback
    assert '"AUTO_WRITER_ROLLBACK_BEGIN"' in rollback
    assert '"AUTO_WRITER_ROLLBACK_END"' in rollback
    assert "ExceptionGroup" in rollback


def test_e11_rollback_attempts_earlier_journal_after_later_failure() -> None:
    rollback_source = _function(AUTO, "rollback_rigped_auto_writer_results")
    calls: list[str] = []
    traces: list[str] = []

    class State:
        value = "OPEN"

    class ReportStatus:
        value = "VERIFIED"

    class Report:
        status = ReportStatus()
        residue_receipts = ()
        quarantine_keys = ()

    class Journal:
        state = State()

        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def rollback(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} rollback failed")
            return Report()

    class Result:
        applied = True
        diagnostics = ()

        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.operation_id = name
            self.pending_journal = Journal(name, fail=fail)

    namespace = {
        "AutoWriterResult": object,
        "trace_event": lambda _channel, event, **_data: traces.append(event),
    }
    exec(rollback_source, namespace)  # noqa: S102 - executes extracted local production source only
    rollback = namespace["rollback_rigped_auto_writer_results"]

    try:
        rollback((Result("A"), Result("B", fail=True)))
    except ExceptionGroup as exc:
        assert len(exc.exceptions) == 1
    else:
        raise AssertionError("aggregate rollback failure was not surfaced")

    assert calls == ["B", "A"]
    assert traces.count("AUTO_WRITER_ROLLBACK_BEGIN") == 2
    assert traces.count("AUTO_WRITER_ROLLBACK_END") == 2


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


def test_e11_direct_move_keeps_reconciliation_inside_deferred_commit_boundary() -> None:
    modal = _method(TRANSFORM, "BAW_OT_rigped_direct_move_axis", "modal")
    release = modal[modal.index('if event.type == "LEFTMOUSE" and event.value == "RELEASE":'):]
    writer = release.index("commit_rigped_auto_direct_move(")
    deferred = release.index("defer_commit=True", writer)
    refresh = release.index("_refresh_current_sliding_public_overlays(", deferred)
    group_commit = release.index(
        "commit_rigped_auto_writer_results(tuple(deferred_auto_results))",
        refresh,
    )
    selection_clear = release.index("clear_key_selection_for_context(context)", group_commit)
    assert writer < deferred < refresh < group_commit < selection_clear


def test_e11_direct_rotate_reconciles_frozen_set_before_group_commit_and_selection_cleanup() -> None:
    modal = _method(TRANSFORM, "BAW_OT_rigped_direct_rotate_axis", "modal")
    release = modal[modal.index('if event.type == "LEFTMOUSE" and event.value == "RELEASE":'):]
    writer = release.index("commit_rigped_auto_direct_rotate(")
    deferred = release.index("defer_commit=True", writer)
    refresh = release.index("_refresh_current_sliding_public_overlays(", deferred)
    frozen = release.index("capabilities=self._sliding_affected_capabilities", refresh)
    group_commit = release.index(
        "commit_rigped_auto_writer_results(tuple(deferred_auto_results))",
        frozen,
    )
    selection_clear = release.index("clear_key_selection_for_context(context)", group_commit)
    assert writer < deferred < refresh < frozen < group_commit < selection_clear


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


def test_e11_body_auto_covers_root_and_com_without_collapsing_dependency_semantics() -> None:
    invoke = _method(TRANSFORM, "BAW_OT_rigped_direct_move_axis", "invoke")
    move = _function(AUTO, "plan_rigped_auto_direct_move")

    # E7/E8 dependency semantics remain intentionally distinct: COM guards keep
    # Sliding target world authority fixed, while Root-relative targets follow Root.
    assert '_resolved_control_role_name(state.control) == "COM"' in invoke
    assert "_capture_sliding_dependency_guards(self._sliding_guard_capabilities)" in invoke

    # AUTO itself is not COM-only: every supported direct Move goes through the
    # selected direct-control planner, so Root and COM both key only their own
    # authored direct authority.
    assert "plan_rigped_auto_direct_move(" in invoke
    assert "requested_families=(ChannelFamily.POSITION,)" in move
    assert "active_only=active_only" in move


def test_e11_multi_limb_auto_preserves_mapping_authority_and_defers_planted() -> None:
    plan = _function(AUTO, "plan_rigped_auto_contact_batch")
    commit = _function(AUTO, "commit_rigped_auto_contact_batch")

    assert "mapping_ids=mapping_ids" in plan
    assert "ContactKeyType.FREE" in plan
    assert "ContactKeyType.SLIDING" in plan
    assert "AK_AUTO_MULTI_PLANTED_DEFERRED" in plan
    assert "begin_types" in commit and "final_types" in commit
    assert "AK_AUTO_CONTACT_AUTHORITY_CHANGED" in commit


def test_e11_escape_paths_restore_preview_without_auto_writer_commit() -> None:
    for class_name in (
        "BAW_OT_rigped_semantic_move_axis",
        "BAW_OT_rigped_fk_joint_move_axis",
        "BAW_OT_rigped_direct_move_axis",
        "BAW_OT_rigped_direct_rotate_axis",
    ):
        modal = _method(TRANSFORM, class_name, "modal")
        marker = 'if event.type in {"ESC", "RIGHTMOUSE"}:'
        assert marker in modal
        escape = modal[modal.index(marker):]
        assert 'return {"CANCELLED"}' in escape
        escape = escape[: escape.index('return {"CANCELLED"}') + len('return {"CANCELLED"}')]
        assert "commit_rigped_auto_anchor(" not in escape
        assert "commit_rigped_auto_contact_batch(" not in escape
        assert "commit_rigped_auto_direct_move(" not in escape
        assert "commit_rigped_auto_direct_rotate(" not in escape
        assert "commit_rigped_auto_writer_results(" not in escape


def test_e11_undo_ownership_remains_one_gesture_boundary() -> None:
    source, tree = _source(TRANSFORM)

    classes = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        in {
            "BAW_OT_rigped_semantic_move_axis",
            "BAW_OT_rigped_fk_joint_move_axis",
            "BAW_OT_rigped_direct_move_axis",
            "BAW_OT_rigped_direct_rotate_axis",
        }
    }
    assert '{"REGISTER", "UNDO", "BLOCKING"}' in classes["BAW_OT_rigped_direct_move_axis"]
    assert '{"REGISTER", "UNDO", "BLOCKING"}' in classes["BAW_OT_rigped_direct_rotate_axis"]
    assert "bpy.ops.ed.undo_push(" in classes["BAW_OT_rigped_semantic_move_axis"]
    assert 'bpy.ops.ed.undo_push(message="AWB Rigped FK Move")' in classes[
        "BAW_OT_rigped_fk_joint_move_axis"
    ]


def test_e11_passive_scrub_replay_display_never_authors_animation() -> None:
    sync = _function(TRANSFORM, "_sync_rigped_sliding_replay_display")
    for forbidden in (
        "commit_rigped_auto_",
        "execute_contact_intent_plan(",
        "execute_contact_batch_intent_plan(",
        "execute_direct_key_plan(",
        "keyframe_insert(",
        "keyframe_points.insert(",
    ):
        assert forbidden not in sync
    assert "_apply_pose_bone_rotation_from_matrix(" in sync
    assert "view_layer.update()" in sync


def test_e11_direct_first_key_baseline_stays_existing_direct_policy() -> None:
    move = _function(AUTO, "plan_rigped_auto_direct_move")
    rotate = _function(AUTO, "plan_rigped_auto_direct_rotate")
    helper = _function(AUTO, "_direct_plan_baseline_rows_by_control")

    assert "current_time > 0.0" in move
    assert "current_time > 0.0" in rotate
    assert "if channels and keyed_count == 0:" in helper
    assert "AK_AUTO_PARTIAL_DIRECT_POSITION" in helper
    assert "AK_AUTO_PARTIAL_DIRECT_ROTATION" in helper


def test_e11_sliding_move_and_rotate_auto_share_existing_anchor_authority() -> None:
    move_invoke = _method(TRANSFORM, "BAW_OT_rigped_semantic_move_axis", "invoke")
    rotate_invoke = _method(TRANSFORM, "BAW_OT_rigped_direct_rotate_axis", "invoke")

    assert "plan_rigped_auto_anchor(" in move_invoke
    assert "Sliding Move Auto must preserve Sliding authority." in move_invoke
    assert "plan_rigped_auto_anchor(" in rotate_invoke
    assert "ContactKeyType.SLIDING" in rotate_invoke
