from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts.awb_debug_replay_split import (
    COVERAGE_FULL_AUTOMATIC,
    COVERAGE_PRE_FINAL_GESTURE,
    COVERAGE_STATE_ONLY,
    ORACLE_BOUND,
    ORACLE_DEFERRED_TO_MANUAL,
    PATH_GUI_INPUT,
    PATH_SEMANTIC,
    PATH_STATE_ONLY,
    build_failure_oracle,
    build_gui_replay_payload,
    build_replay_handoff,
    classify_replay,
    enrich_incident_replay,
    evaluate_failure_oracle,
)

ROOT = Path(__file__).resolve().parents[1]
REPLAY_SOURCE = ROOT / "extension" / "blender_animation_workbench" / "debug_replay.py"
RUNNER_SOURCE = ROOT / "scripts" / "replay_awb_incident.py"


def _record(
    seq: int,
    event: str,
    *,
    session: str = "session-current",
    trace: str | None = "trace-1",
    requested_mode: str | None = None,
    terminal_status: str | None = None,
    route_outcome: str | None = "CLAIMED",
    operation_id: str | None = None,
) -> dict:
    data = {}
    if requested_mode is not None:
        data["requested_mode"] = requested_mode
    raw = {
        "session_id": session,
        "seq": seq,
        "event": event,
        "trace_id": trace,
        "operation_id": operation_id,
        "terminal_status": terminal_status,
        "route_outcome": route_outcome,
        "state": {
            "blend_file": r"E:\AIProjects\Tools\Blender_Animation_Workbench\build\rigped_animate_manual_baseline.blend",
            "mode": "POSE",
            "active_object": "AWB_Rigped",
            "selected_objects": ["AWB_Rigped"],
            "selected_pose_bones": ["Foot.L"],
        },
        "data": data,
    }
    return {
        "session_id": session,
        "seq": seq,
        "event": event,
        "trace_id": trace,
        "operation_id": operation_id,
        "terminal_status": terminal_status,
        "route_outcome": route_outcome,
        "record": raw,
    }


def _wer_records() -> list[dict]:
    return [
        _record(2, "TRANSFORM_TOOL_INGRESS", trace="trace-w", requested_mode="MOVE"),
        _record(
            5,
            "TRANSFORM_TOOL_TERMINAL",
            trace="trace-w",
            requested_mode="MOVE",
            terminal_status="FINISHED",
        ),
        _record(6, "TRANSFORM_TOOL_INGRESS", trace="trace-e", requested_mode="ROTATE"),
        _record(
            10,
            "TRANSFORM_TOOL_TERMINAL",
            trace="trace-e",
            requested_mode="ROTATE",
            terminal_status="FINISHED",
        ),
        _record(11, "TRANSFORM_TOOL_INGRESS", trace="trace-r", requested_mode="SCALE"),
        _record(
            14,
            "TRANSFORM_TOOL_TERMINAL",
            trace="trace-r",
            requested_mode="SCALE",
            terminal_status="FINISHED",
        ),
    ]


def _semantic(*actions: dict) -> dict:
    return {
        "schema": "awb-semantic-replay/v1",
        "source_session_id": "session-current",
        "blend_file": r"E:\AIProjects\Tools\Blender_Animation_Workbench\build\rigped_animate_manual_baseline.blend",
        "action_count": len(actions),
        "actions": list(actions),
    }


def test_bb5_wer_gui_payload_is_exact_and_has_frozen_context():
    semantic = _semantic({"kind": "AUTO_KEY", "source_seq": 1, "enabled": False})
    payload = build_gui_replay_payload(
        _wer_records(),
        current_session_id="session-current",
        semantic_replay=semantic,
        freshness="FRESH",
    )

    assert payload is not None
    assert payload["schema"] == "awb-gui-replay/v1"
    assert [item["key"] for item in payload["actions"]] == ["W", "E", "R"]
    assert [item["requested_mode"] for item in payload["actions"]] == [
        "MOVE",
        "ROTATE",
        "SCALE",
    ]
    assert [(item["type"], item["value"]) for item in payload["events"]] == [
        ("W", "PRESS"),
        ("W", "RELEASE"),
        ("E", "PRESS"),
        ("E", "RELEASE"),
        ("R", "PRESS"),
        ("R", "RELEASE"),
    ]
    assert payload["expected_mode"] == "POSE"
    assert payload["expected_context"]["active_object"] == "AWB_Rigped"
    assert payload["expected_context"]["selected_pose_bones"] == ["Foot.L"]
    assert payload["semantic_prefix"]["action_count"] == 1


def test_bb5_gui_payload_fails_closed_on_missing_terminal_or_budget():
    incomplete = [_record(2, "TRANSFORM_TOOL_INGRESS", requested_mode="MOVE")]
    assert (
        build_gui_replay_payload(
            incomplete,
            current_session_id="session-current",
            semantic_replay=_semantic(),
            freshness="FRESH",
        )
        is None
    )
    assert (
        build_gui_replay_payload(
            _wer_records(),
            current_session_id="session-current",
            semantic_replay=_semantic(),
            freshness="FRESH",
            max_actions=2,
        )
        is None
    )


def test_bb5_stale_semantic_prefix_blocks_gui_full_automatic():
    payload = build_gui_replay_payload(
        _wer_records(),
        current_session_id="session-current",
        semantic_replay=_semantic(
            {"kind": "AUTO_KEY", "source_seq": 1, "enabled": True}
        ),
        freshness="PARTIAL",
    )
    assert payload is None


def test_bb5_classification_separates_semantic_gui_and_state_only():
    semantic = _semantic({"kind": "AUTO_KEY", "source_seq": 1})
    assert classify_replay(
        semantic_replay=semantic,
        freshness="FRESH",
        gui_replay=None,
        handoff=None,
        state_available=True,
    ) == (COVERAGE_FULL_AUTOMATIC, PATH_SEMANTIC)

    gui = build_gui_replay_payload(
        _wer_records(),
        current_session_id="session-current",
        semantic_replay=_semantic(),
        freshness="FRESH",
    )
    assert classify_replay(
        semantic_replay=_semantic(),
        freshness="FRESH",
        gui_replay=gui,
        handoff=None,
        state_available=True,
    ) == (COVERAGE_FULL_AUTOMATIC, PATH_GUI_INPUT)

    assert classify_replay(
        semantic_replay=semantic,
        freshness="PARTIAL",
        gui_replay=None,
        handoff=None,
        state_available=True,
    ) == (COVERAGE_STATE_ONLY, PATH_STATE_ONLY)


def test_bb5_pre_final_handoff_uses_only_evidence_before_unmatched_transform():
    semantic = _semantic(
        {"kind": "AUTO_KEY", "source_seq": 2},
        {"kind": "CONTACT", "source_seq": 5},
        {"kind": "MOVE", "source_seq": 20},
    )
    records = [
        _record(
            12,
            "TRANSFORM_BEGIN",
            trace="trace-modal",
            operation_id="op-modal",
            route_outcome=None,
        )
    ]
    handoff = build_replay_handoff(
        records,
        current_session_id="session-current",
        semantic_replay=semantic,
    )

    assert handoff is not None
    assert handoff["reason"] == "UNSUPPORTED_MODAL_INPUT"
    assert handoff["pending_source_seq"] == 12
    assert handoff["semantic_prefix"]["action_count"] == 2
    assert [a["source_seq"] for a in handoff["semantic_prefix"]["actions"]] == [2, 5]

    assert classify_replay(
        semantic_replay=semantic,
        freshness="FRESH",
        gui_replay=None,
        handoff=handoff,
        state_available=True,
    ) == (COVERAGE_PRE_FINAL_GESTURE, PATH_SEMANTIC)


def test_bb5_failure_oracle_uses_stable_identity_and_null_states():
    divergence = {
        "result": "VIOLATION",
        "invariant_id": "generic.hash_diff_consistency.v1",
        "reason": "DIFF_AFTER_HASH_MISMATCH",
        "checkpoint_seq": 2,
        "paths": ["/semantic_state_hash"],
        "trace_id": "ephemeral-trace",
        "span_id": "ephemeral-span",
    }
    oracle = build_failure_oracle(
        {"first_divergence": divergence},
        coverage=COVERAGE_FULL_AUTOMATIC,
    )
    assert oracle["status"] == ORACLE_BOUND
    assert oracle["invariant_id"] == divergence["invariant_id"]
    assert oracle["finding_code"] == divergence["reason"]
    assert "trace_id" not in oracle
    assert "span_id" not in oracle

    deferred = build_failure_oracle(
        {"first_divergence": None},
        coverage=COVERAGE_PRE_FINAL_GESTURE,
    )
    assert deferred["status"] == ORACLE_DEFERRED_TO_MANUAL


def test_bb5_oracle_matches_streamed_first_divergence_even_if_findings_omitted():
    oracle = {
        "schema": "awb-failure-oracle/v1",
        "status": "BOUND",
        "oracle_kind": "DIVERGENCE",
        "invariant_id": "generic.hash_diff_consistency.v1",
        "finding_code": "DIFF_AFTER_HASH_MISMATCH",
        "checkpoint_seq": 2,
        "boundary": None,
        "paths": ["/semantic_state_hash"],
        "expected_verdict": "VIOLATION",
    }
    result = evaluate_failure_oracle(
        oracle,
        {
            "first_divergence": {
                "result": "VIOLATION",
                "invariant_id": oracle["invariant_id"],
                "reason": oracle["finding_code"],
                "checkpoint_seq": 2,
                "paths": ["/semantic_state_hash"],
            },
            "findings": [],
        },
    )
    assert result == {"status": "BOUND", "verdict": "REPRODUCED"}


def test_bb5_incident_enrichment_recommends_gui_for_verified_wer_only():
    base = {
        "schema": "awb-debug-incident-replay/v1",
        "incident_id": "INC-test",
        "status": "AVAILABLE",
        "freshness": "FRESH",
        "source": "live_in_memory_builder",
        "source_session_id": "session-current",
        "captured_trace_session_id": "session-current",
        "captured_last_seq": 14,
        "semantic_replay": _semantic(),
        "unavailable_reason": None,
    }
    replay = enrich_incident_replay(
        base,
        timeline_records=_wer_records(),
        divergences={"first_divergence": None},
        state_available=True,
    )
    assert replay["coverage"] == "FULL_AUTOMATIC"
    assert replay["recommended_path"] == "GUI_INPUT"
    assert replay["semantic_replay_mode"] == "RECORDED_RESULT"
    assert replay["failure_oracle"]["status"] == "NO_EVIDENCE"


def test_bb5_gui_executor_source_contract_is_async_and_area_targeted():
    source = REPLAY_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "run_gui_input_replay" in names
    assert "confirm_gui_input_replay" in names
    assert "area.x + area.width // 2" in source
    assert "area.y + area.height // 2" in source
    assert 'set_replay_execution_active(True)' in source
    assert 'set_replay_execution_active(False)' in source
    assert 'bool(item.get("replay_execution", False))' in source
    assert '"EXECUTION_UNAVAILABLE_HEADLESS"' in source
    assert "requires PRESS followed by RELEASE" in source


def test_bb5_incident_runner_dispatches_incident_schema_not_user_final_schema():
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert '"awb-debug-incident-replay/v1"' in source
    assert '"GUI_INPUT"' in source
    assert '"SEMANTIC"' in source
    assert "confirm_gui_input_replay" in source
    assert "evaluate_failure_oracle" in source
    assert "_PersistentBlenderMcpClient" in source


def test_bb5_replay_schemas_are_versioned_and_generic():
    for name in (
        "AWB_DEBUG_INCIDENT_REPLAY_V1.schema.json",
        "AWB_DEBUG_GUI_REPLAY_V1.schema.json",
        "AWB_DEBUG_FAILURE_ORACLE_V1.schema.json",
        "AWB_DEBUG_REPLAY_HANDOFF_V1.schema.json",
    ):
        payload = json.loads(
            (ROOT / "docs" / "DEBUG" / "schemas" / name).read_text(encoding="utf-8")
        )
        assert payload["$id"].endswith("/v1")
        assert "rigped" not in json.dumps(payload, ensure_ascii=False).lower()

def test_bb5_gui_replay_trace_uses_execution_identity_not_seq_only():
    trace_source = (
        ROOT / "extension" / "blender_animation_workbench" / "debug_trace.py"
    ).read_text(encoding="utf-8")
    replay_source = REPLAY_SOURCE.read_text(encoding="utf-8")

    assert "_REPLAY_EXECUTION_ID" in trace_source
    assert '"replay_execution_id": _REPLAY_EXECUTION_ID' in trace_source
    assert "execution_id: str | None = None" in trace_source
    assert "set_replay_execution_active(True, execution_id=execution_id)" in replay_source
    assert 'item.get("replay_execution_id") == execution_id' in replay_source
    causal_block = replay_source[
        replay_source.index("def _gui_replay_causal_matches(") :
        replay_source.index("def confirm_gui_input_replay(")
    ]
    assert "start_seq" not in causal_block


def test_bb5_incident_runner_passes_full_ticket_to_stateless_confirmation():
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert "def _gui_confirm_code(ticket: dict[str, Any])" in source
    assert "result = confirm_gui_input_replay(ticket)" in source
    assert "_gui_confirm_code(ticket)" in source
    assert "replay_execution_id=" in source

