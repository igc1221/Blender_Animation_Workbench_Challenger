from __future__ import annotations

import ast
from pathlib import Path

SOURCE_PATH = (
    Path(__file__).parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "phase4_contact_authoring.py"
)


def _tree() -> ast.Module:
    return ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    for node in _tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function: {name}")


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls(function: ast.FunctionDef, name: str) -> tuple[ast.Call, ...]:
    return tuple(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) == name
    )


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def test_e3_contact_command_freezes_one_operation_domain() -> None:
    function = _function("execute_contact_command")

    domain_calls = _calls(function, "resolve_operation_domain")

    assert len(domain_calls) == 1


def test_e3_mixed_direct_subset_is_planned_as_free_closure() -> None:
    function = _function("execute_contact_command")
    direct_calls = _calls(function, "build_direct_key_plan")

    assert len(direct_calls) == 1
    direct = direct_calls[0]
    assert isinstance(_keyword(direct, "binding_ids"), ast.Call)
    assert isinstance(_keyword(direct, "operation_domain"), ast.Name)
    assert _keyword(direct, "operation_domain").id == "operation_domain"
    free_marker = _keyword(direct, "include_rigped_free_marker")
    assert isinstance(free_marker, ast.Constant)
    assert free_marker.value is True


def test_e3_coordinator_executes_direct_only_transaction_but_keeps_mixed_closure_owned_by_contact() -> None:
    function = _function("execute_contact_command")

    direct_writes = _calls(function, "execute_direct_key_plan")
    assert len(direct_writes) == 1
    trigger = _keyword(direct_writes[0], "trigger")
    assert isinstance(trigger, ast.Attribute)
    assert trigger.attr == "CONTACT_AUTHORING"

    for name in ("execute_contact_intent_plan", "execute_contact_batch_intent_plan"):
        calls = _calls(function, name)
        assert len(calls) == 1
        closure = _keyword(calls[0], "closure_plan")
        assert isinstance(closure, ast.Name)
        assert closure.id == "direct_plan"


def test_e3_batch_uses_frozen_operation_domain_mapping_ids() -> None:
    function = _function("execute_contact_command")
    calls = _calls(function, "build_contact_batch_intent_plan")

    assert len(calls) == 1
    mapping_ids = _keyword(calls[0], "mapping_ids")
    assert isinstance(mapping_ids, ast.Name)
    assert mapping_ids.id == "mapping_ids"

    source = ast.get_source_segment(SOURCE_PATH.read_text(encoding="utf-8"), function)
    assert source is not None
    assert "if not mapping_ids:" in source
    assert "execute_direct_key_plan(" in source
    assert "mapping_contact_types=()" in source


def test_e3_contact_transactions_prepare_direct_semantic_backing_property() -> None:
    single = _function("execute_contact_intent_plan")
    batch = _function("execute_contact_batch_intent_plan")

    # Single Contact already used the semantic-property helper for AUTO baseline
    # rows; E3 requires an additional call for the merged closure rows.
    assert len(_calls(single, "_prepare_semantic_state_property")) >= 2
    # Batch Contact had no semantic backing-property preparation before E3.
    assert len(_calls(batch, "_prepare_semantic_state_property")) >= 1


def test_e3_contact_latch_is_journaled_before_persistent_commit() -> None:
    helper = _function("_journal_contact_authoring_latch")
    assert len(_calls(helper, "IDPropertyMutationReceipt")) == 1
    assert len(_calls(helper, "_set_contact_authoring_latch")) == 1

    for name in ("execute_contact_intent_plan", "execute_contact_batch_intent_plan"):
        function = _function(name)
        latch_calls = _calls(function, "_journal_contact_authoring_latch")
        commit_calls = tuple(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "journal"
            and node.func.attr == "commit"
        )
        assert latch_calls
        assert len(commit_calls) == 1
        assert max(call.lineno for call in latch_calls) < commit_calls[0].lineno
        for call in latch_calls:
            assert len(call.args) >= 4
            assert isinstance(call.args[3], ast.Name)
            assert call.args[3].id == "journal"


def test_e4_mixed_contact_failure_traces_rollback_result_and_residue() -> None:
    helper = _function("_trace_contact_rollback_report")
    helper_source = ast.get_source_segment(
        SOURCE_PATH.read_text(encoding="utf-8"),
        helper,
    )
    assert helper_source is not None
    for field in (
        "journal_state",
        "receipt_count",
        "rollback_status",
        "residue_count",
        "residue_receipts",
        "quarantine",
        "rollback_diagnostics",
    ):
        assert field in helper_source

    expectations = (
        (
            "execute_contact_intent_plan",
            "CONTACT_ROLLBACK_BEGIN",
            "CONTACT_ROLLBACK_END",
        ),
        (
            "execute_contact_batch_intent_plan",
            "CONTACT_BATCH_ROLLBACK_BEGIN",
            "CONTACT_BATCH_ROLLBACK_END",
        ),
    )
    for function_name, begin_event, end_event in expectations:
        function = _function(function_name)
        source = ast.get_source_segment(
            SOURCE_PATH.read_text(encoding="utf-8"),
            function,
        )
        assert source is not None
        assert source.count(f'\"{begin_event}\"') == 1
        assert source.count(f'\"{end_event}\"') == 1
        rollback_index = source.index("rollback = journal.rollback()")
        end_index = source.index(f'\"{end_event}\"', rollback_index)
        reevaluate_index = source.index(
            "_reevaluate_contact_frame_preserving_public_pose(",
            rollback_index,
        )
        assert source.index(f'\"{begin_event}\"') < rollback_index
        assert rollback_index < end_index
        assert end_index < reevaluate_index
