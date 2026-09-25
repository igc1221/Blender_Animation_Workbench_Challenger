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

def test_bb5_gui_payload_rejects_trailing_unterminated_ingress_after_success():
    records = [
        _record(2, "TRANSFORM_TOOL_INGRESS", trace="trace-w", requested_mode="MOVE"),
        _record(
            5,
            "TRANSFORM_TOOL_TERMINAL",
            trace="trace-w",
            requested_mode="MOVE",
            terminal_status="FINISHED",
        ),
        _record(6, "TRANSFORM_TOOL_INGRESS", trace="trace-e", requested_mode="ROTATE"),
    ]
    assert (
        build_gui_replay_payload(
            records,
            current_session_id="session-current",
            semantic_replay=_semantic(),
            freshness="FRESH",
        )
        is None
    )


def test_bb5_replay_execution_records_are_excluded_from_replay_extraction():
    records = _wer_records()
    for item in records:
        item["replay_execution"] = True
        item["record"]["replay_execution"] = True

    assert (
        build_gui_replay_payload(
            records,
            current_session_id="session-current",
            semantic_replay=_semantic(),
            freshness="FRESH",
        )
        is None
    )

    modal = _record(
        20,
        "TRANSFORM_BEGIN",
        trace="trace-modal",
        operation_id="op-modal",
        route_outcome=None,
    )
    modal["replay_execution"] = True
    modal["record"]["replay_execution"] = True
    assert (
        build_replay_handoff(
            [modal],
            current_session_id="session-current",
            semantic_replay=_semantic(),
        )
        is None
    )


def test_bb5_pure_gui_runner_accepts_null_semantic_replay_mode(monkeypatch):
    import scripts.replay_awb_incident as incident_runner

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def execute_code(self, _code):
            self.calls += 1
            if self.calls == 1:
                return {
                    "status": "QUEUED",
                    "start_seq": 0,
                    "ticket": {
                        "schema": "awb-gui-replay-ticket/v1",
                        "execution_id": "gui-replay:test",
                        "status": "QUEUED",
                        "expected_action_count": 1,
                        "actions": [
                            {
                                "requested_mode": "MOVE",
                                "expected_terminal_status": "FINISHED",
                                "expected_route_outcome": "CLAIMED",
                            }
                        ],
                    },
                }
            return {
                "schema": "awb-gui-replay-result/v1",
                "execution_id": "gui-replay:test",
                "status": "CONFIRMED",
                "matched_action_count": 1,
                "expected_action_count": 1,
                "matched_routes": [],
            }

    monkeypatch.setattr(incident_runner.time, "sleep", lambda _seconds: None)
    replay = {
        "semantic_replay_mode": None,
        "gui_replay": {
            "semantic_prefix": None,
        },
    }
    result = incident_runner._run_gui(FakeClient(), replay)
    assert result["status"] == "CONFIRMED"


def test_bb5_gui_prefix_and_events_share_one_execution_identity():
    import scripts.replay_awb_incident as incident_runner

    gui_replay = {
        "semantic_prefix": _semantic(
            {"kind": "AUTO_KEY", "source_seq": 1, "enabled": False}
        )
    }
    code = incident_runner._gui_queue_code(
        gui_replay,
        "RECORDED_RESULT",
        "gui-replay:shared",
    )
    assert code.count("replay_execution_id='gui-replay:shared'") == 2

    evidence = incident_runner._evidence_code(
        0,
        replay_execution_id="gui-replay:shared",
    )
    assert "item.get('replay_execution_id') == 'gui-replay:shared'" in evidence


def test_bb5_failed_gui_execution_cannot_report_clean_oracle(monkeypatch, tmp_path):
    import scripts.replay_awb_incident as incident_runner

    replay = {
        "schema": "awb-debug-incident-replay/v1",
        "incident_id": "INC-test-failed",
        "coverage": "FULL_AUTOMATIC",
        "recommended_path": "GUI_INPUT",
        "semantic_replay_mode": None,
        "gui_replay": {"semantic_prefix": None},
        "handoff": None,
        "failure_oracle": {
            "schema": "awb-failure-oracle/v1",
            "status": "BOUND",
            "oracle_kind": "DIVERGENCE",
            "invariant_id": "generic.hash_diff_consistency.v1",
            "finding_code": "DIFF_AFTER_HASH_MISMATCH",
            "checkpoint_seq": 2,
            "boundary": "AFTER_OPERATION",
            "paths": ["/semantic_state_hash"],
            "expected_verdict": "VIOLATION",
        },
    }
    (tmp_path / "replay.json").write_text(
        json.dumps(replay),
        encoding="utf-8",
    )

    class FakePersistentClient:
        def __init__(self, timeout=60.0):
            self.timeout = timeout
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

        def execute_code(self, _code):
            self.calls += 1
            if self.calls == 1:
                return {
                    "status": "FAILED",
                    "start_seq": 0,
                    "ticket": {
                        "status": "FAILED",
                        "reason": "BASELINE_MODE_MISMATCH",
                    },
                }
            return {"checkpoints": [], "timeline": []}

    monkeypatch.setattr(
        incident_runner,
        "_persistent_blender_client",
        lambda timeout: FakePersistentClient(timeout=timeout),
    )
    result = incident_runner.run_incident(tmp_path)
    assert result["execution_status"] == "FAILED"
    assert result["oracle_evaluation"]["status"] == "EVALUATION_ERROR"
    assert result["oracle_evaluation"]["verdict"] == "UNAVAILABLE"


def test_bb5_empty_blend_guard_does_not_normalize_empty_to_dot():
    import scripts.replay_awb_incident as incident_runner

    code = incident_runner._baseline_guard_code("")
    assert "expected_blend_raw = ''" in code
    assert "if expected_blend_raw" in code
    assert 'else ""' in code


def test_bb5_interleaved_semantic_action_blocks_gui_and_semantic_false_proof():
    base = {
        "schema": "awb-debug-incident-replay/v1",
        "incident_id": "INC-interleaved",
        "status": "AVAILABLE",
        "freshness": "FRESH",
        "source": "live_in_memory_builder",
        "source_session_id": "session-current",
        "captured_trace_session_id": "session-current",
        "captured_last_seq": 14,
        "semantic_replay": _semantic(
            {"kind": "AUTO_KEY", "source_seq": 7, "enabled": False}
        ),
        "unavailable_reason": None,
    }
    replay = enrich_incident_replay(
        base,
        timeline_records=_wer_records(),
        divergences={"first_divergence": None},
        state_available=True,
    )
    assert replay["gui_replay"] is None
    assert replay["coverage"] == "STATE_ONLY"
    assert replay["recommended_path"] == "STATE_ONLY"


def test_bb5_supported_gui_prefix_plus_unmatched_modal_requires_handoff():
    records = [
        _record(2, "TRANSFORM_TOOL_INGRESS", trace="trace-w", requested_mode="MOVE"),
        _record(
            5,
            "TRANSFORM_TOOL_TERMINAL",
            trace="trace-w",
            requested_mode="MOVE",
            terminal_status="FINISHED",
        ),
        _record(
            6,
            "TRANSFORM_BEGIN",
            trace="trace-modal",
            operation_id="op-modal",
            route_outcome=None,
        ),
    ]
    base = {
        "schema": "awb-debug-incident-replay/v1",
        "incident_id": "INC-handoff",
        "status": "AVAILABLE",
        "freshness": "FRESH",
        "source": "live_in_memory_builder",
        "source_session_id": "session-current",
        "captured_trace_session_id": "session-current",
        "captured_last_seq": 6,
        "semantic_replay": _semantic(),
        "unavailable_reason": None,
    }
    replay = enrich_incident_replay(
        base,
        timeline_records=records,
        divergences={"first_divergence": None},
        state_available=True,
    )
    assert replay["gui_replay"] is not None
    assert replay["handoff"] is not None
    assert replay["coverage"] == "PRE_FINAL_GESTURE"
    assert replay["recommended_path"] == "GUI_INPUT"

