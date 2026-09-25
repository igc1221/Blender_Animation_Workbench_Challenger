from __future__ import annotations

import json
from typing import get_type_hints

from scripts import awb_debug_rigped_invariants as rigped_invariants
from scripts.awb_debug_divergence import (
    RESULT_ERROR,
    RESULT_SKIP,
    RESULT_VIOLATION,
    InvariantOutcome,
    InvariantRegistry,
    InvariantSpec,
    default_generic_registry,
    evaluate_incident_divergences,
)
from scripts.awb_debug_rigped_invariants import register_rigped_invariants


def _checkpoint(
    seq: int,
    *,
    checkpoint_id: str | None = None,
    boundary: str = "BEFORE_OPERATION",
    previous_id: str | None = None,
    previous_status: str = "NONE",
    state_hash: str | None = None,
    diff=None,
    session_id: str | None = "session-1",
    trace_id: str | None = "trace-1",
    span_id: str | None = "span-1",
    operation_id: str | None = "op-1",
    domain_probes=None,
):
    return {
        "schema": "awb-debug-checkpoint/v1",
        "incident_id": "INC-test",
        "checkpoint_id": checkpoint_id or f"cp-{seq}",
        "checkpoint_seq": seq,
        "session_id": session_id,
        "boundary": boundary,
        "status": "AVAILABLE",
        "confidence": "OBSERVED_LIVE",
        "source": {"event": "TEST"},
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": None,
        "operation_id": operation_id,
        "parent_operation_id": None,
        "subsystem": "operator",
        "lifecycle_phase": "commit" if boundary != "BEFORE_OPERATION" else "invoke",
        "evaluation_phase": "live_context",
        "context_identity": {},
        "blender_state": {},
        "native_state": {},
        "action_fcurves": {},
        "depsgraph_state": {},
        "domain_probes": list(domain_probes or []),
        "semantic_state_hash": state_hash,
        "hash_schema": "awb-debug-state-hash/sha256-v1",
        "normalization_schema": "awb-debug-state-normalized/v1",
        "previous_checkpoint_id": previous_id,
        "previous_runtime_checkpoint_id": None,
        "previous_checkpoint_status": previous_status,
        "diff_from_previous": diff,
        "unavailable_reason": None,
    }


def _bundle(*checkpoints):
    return {
        "schema": "awb-debug-state-checkpoints/v1",
        "incident_id": "INC-test",
        "status": "AVAILABLE",
        "reason": None,
        "checkpoint_count": len(checkpoints),
        "invalid_checkpoint_count": 0,
        "first_changed_checkpoint_id": None,
        "checkpoints": list(checkpoints),
    }


def _terminal(
    *,
    session_id: str = "session-1",
    trace_id: str = "trace-1",
    span_id: str = "span-1",
    operation_id: str = "op-1",
    terminal_status: str = "FINISHED",
):
    return {
        "session_id": session_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "operation_id": operation_id,
        "terminal_status": terminal_status,
        "event": "TEST_TERMINAL",
    }


def test_normal_before_after_pair_has_no_generic_divergence():
    before = _checkpoint(1, state_hash="hash-before")
    after = _checkpoint(
        2,
        boundary="AFTER_OPERATION",
        previous_id="cp-1",
        previous_status="RESOLVED",
        state_hash="hash-after",
        diff={
            "changed": True,
            "before_hash": "hash-before",
            "after_hash": "hash-after",
            "changes": [{"path": "/native_state/location/0"}],
        },
    )

    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(before, after),
        timeline_records=[_terminal()],
    )

    assert result["status"] == "AVAILABLE"
    assert result["violation_count"] == 0
    assert result["error_count"] == 0
    assert result["first_divergence"] is None


def test_capture_gap_is_skip_not_divergence():
    after = _checkpoint(
        2,
        boundary="AFTER_OPERATION",
        previous_id=None,
        previous_status="UNRESOLVED_CAPTURE_GAP",
        state_hash="hash-after",
        diff={
            "changed": True,
            "before_hash": "hash-before",
            "after_hash": "hash-after",
            "changes": [{"path": "/x"}],
        },
    )

    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(after),
        timeline_records=[_terminal()],
    )

    assert result["violation_count"] == 0
    reasons = {item["reason"] for item in result["findings"]}
    assert "CAPTURE_GAP" in reasons
    assert all(item["result"] != RESULT_VIOLATION for item in result["findings"])


def test_resolved_missing_previous_is_first_divergence():
    checkpoint = _checkpoint(
        2,
        boundary="AFTER_OPERATION",
        previous_id="cp-missing",
        previous_status="RESOLVED",
        state_hash="hash-after",
        diff={"changed": True, "changes": [{"path": "/x"}]},
    )

    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(checkpoint),
        timeline_records=[_terminal()],
    )

    first = result["first_divergence"]
    assert first is not None
    assert first["checkpoint_seq"] == 2
    assert first["invariant_id"] == "generic.checkpoint_chain.v1"
    assert first["reason"] == "PREVIOUS_CHECKPOINT_MISSING"


def test_hash_diff_mismatch_reasons_are_specific():
    before = _checkpoint(1, state_hash="hash-a")
    unchanged_with_different_hash = _checkpoint(
        2,
        boundary="AFTER_OPERATION",
        previous_id="cp-1",
        previous_status="RESOLVED",
        state_hash="hash-b",
        diff={
            "changed": False,
            "before_hash": "hash-a",
            "after_hash": "hash-b",
            "changes": [],
        },
    )
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(before, unchanged_with_different_hash),
        timeline_records=[_terminal()],
    )
    reasons = {item["reason"] for item in result["findings"] if item["result"] == RESULT_VIOLATION}
    assert "HASH_NORMALIZATION_INSTABILITY" in reasons

    changed_with_equal_hash = _checkpoint(
        2,
        boundary="AFTER_OPERATION",
        previous_id="cp-1",
        previous_status="RESOLVED",
        state_hash="hash-a",
        diff={
            "changed": True,
            "before_hash": "hash-a",
            "after_hash": "hash-a",
            "changes": [{"path": "/x"}],
        },
    )
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(before, changed_with_equal_hash),
        timeline_records=[_terminal()],
    )
    reasons = {item["reason"] for item in result["findings"] if item["result"] == RESULT_VIOLATION}
    assert "DIFF_HASH_COLLISION" in reasons


def test_terminal_pairing_is_session_scoped():
    checkpoint = _checkpoint(
        1,
        boundary="AFTER_OPERATION",
        previous_status="NONE",
        state_hash="hash-a",
    )
    old_session_terminal = _terminal(session_id="old-session")
    current_nonterminal = {
        "session_id": "session-1",
        "trace_id": "trace-1",
        "span_id": "span-1",
        "operation_id": "op-1",
        "terminal_status": None,
        "event": "MID_OPERATION",
    }

    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(checkpoint),
        timeline_records=[old_session_terminal, current_nonterminal],
    )

    violations = [item for item in result["findings"] if item["result"] == RESULT_VIOLATION]
    assert any(item["reason"] == "TERMINAL_TRACE_EVENT_MISSING" for item in violations)


def test_missing_same_session_timeline_is_skip():
    checkpoint = _checkpoint(
        1,
        boundary="AFTER_OPERATION",
        state_hash="hash-a",
    )
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(checkpoint),
        timeline_records=[_terminal(session_id="old-session")],
    )

    assert result["violation_count"] == 0
    assert any(
        item["result"] == RESULT_SKIP and item["reason"] == "TIMELINE_TRACE_UNAVAILABLE"
        for item in result["findings"]
    )


def test_invariant_exception_is_isolated_and_not_first_divergence():
    registry = InvariantRegistry()

    def explode(_checkpoint, _context, _timeline):
        raise ValueError("synthetic evaluator failure")

    def violate(_checkpoint, _context, _timeline):
        return InvariantOutcome(
            result=RESULT_VIOLATION,
            reason="SYNTHETIC_VIOLATION",
            message="synthetic",
        )

    registry.register(
        InvariantSpec("test.explode.v1", "1", "ERROR", "domain", explode)
    )
    registry.register(
        InvariantSpec("test.violate.v1", "1", "ERROR", "domain", violate)
    )

    checkpoints = [_checkpoint(2), _checkpoint(1)]
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(*checkpoints),
        timeline_records=[],
        registry=registry,
    )

    assert result["error_count"] == 2
    assert result["violation_count"] == 2
    first = result["first_divergence"]
    assert first["checkpoint_seq"] == 1
    assert first["invariant_id"] == "test.violate.v1"
    assert first["result"] == RESULT_VIOLATION
    assert result["findings"][0]["result"] == RESULT_ERROR
    assert result["findings"][1]["result"] == RESULT_VIOLATION


def test_evidence_is_immutable_to_invariant():
    source = _checkpoint(1)
    registry = InvariantRegistry()

    def mutate(checkpoint, _context, _timeline):
        checkpoint["boundary"] = "CORRUPTED"
        return InvariantOutcome(result="PASS")

    registry.register(
        InvariantSpec("test.mutate.v1", "1", "ERROR", "domain", mutate)
    )
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(source),
        timeline_records=[],
        registry=registry,
    )

    assert source["boundary"] == "BEFORE_OPERATION"
    assert result["error_count"] == 1
    assert result["findings"][0]["reason"] == "INVARIANT_EVALUATOR_EXCEPTION"


def test_findings_are_deterministic_and_bounded():
    registry = InvariantRegistry()

    def skip(_checkpoint, _context, _timeline):
        return InvariantOutcome(result=RESULT_SKIP, reason="SYNTHETIC_SKIP")

    for index in range(5):
        registry.register(
            InvariantSpec(
                f"test.skip.{index}",
                "1",
                "INFO",
                "generic",
                skip,
            )
        )

    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(_checkpoint(2), _checkpoint(1)),
        timeline_records=[],
        registry=registry,
        max_findings_per_checkpoint=2,
        max_findings=3,
    )

    assert result["truncated"] is True
    assert result["finding_count"] == 3
    assert result["dropped_finding_count"] == 7
    assert [item["checkpoint_seq"] for item in result["findings"]] == [1, 1, 2]
    assert [item["registration_order"] for item in result["findings"]] == [1, 2, 1]


def test_rigped_pack_skips_without_explicit_invariant_claims():
    registry = default_generic_registry()
    register_rigped_invariants(registry)
    checkpoint = _checkpoint(
        1,
        domain_probes=[
            {
                "name": "rigped",
                "status": "AVAILABLE",
                "state": {
                    "rig_object": "AWB_Rigped",
                    "semantic_transform_mode": "FK_MOVE",
                },
            }
        ],
    )

    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(checkpoint),
        timeline_records=[],
        registry=registry,
    )

    assert result["violation_count"] == 0
    assert any(
        item["scope"] == "domain"
        and item["reason"] == "RIGPED_EXPLICIT_CLAIMS_UNAVAILABLE"
        for item in result["findings"]
    )


def test_rigped_pack_uses_only_explicit_frozen_claims():
    registry = default_generic_registry()
    register_rigped_invariants(registry)
    checkpoint = _checkpoint(
        1,
        domain_probes=[
            {
                "name": "rigped",
                "status": "AVAILABLE",
                "state": {
                    "invariant_schema": "awb-rigped-invariant-claims/v1",
                    "invariant_claims": [
                        {
                            "status": "VIOLATION",
                            "reason": "RIGPED_TEST_CLAIM",
                            "message": "Explicit frozen claim failed.",
                            "paths": ["/domain_probes/rigped/test"],
                            "expected": "locked",
                            "observed": "moved",
                        }
                    ]
                },
            }
        ],
    )

    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(checkpoint),
        timeline_records=[],
        registry=registry,
    )

    first = result["first_divergence"]
    assert first is not None
    assert first["scope"] == "domain"
    assert first["invariant_id"] == "rigped.explicit_claims.v1"
    assert first["reason"] == "RIGPED_TEST_CLAIM"


def test_first_divergence_survives_exhausted_findings_budget():
    registry = InvariantRegistry()

    def late_violation(checkpoint, _context, _timeline):
        if checkpoint.get("checkpoint_seq") == 40:
            return InvariantOutcome(
                result=RESULT_VIOLATION,
                reason="LATE_VIOLATION",
                message="Late violation must survive bounded finding storage.",
            )
        return InvariantOutcome(result=RESULT_SKIP, reason="BENIGN_SKIP")

    registry.register(
        InvariantSpec("test.late.v1", "1", "ERROR", "generic", late_violation)
    )
    checkpoints = [_checkpoint(index) for index in range(1, 41)]

    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(*checkpoints),
        timeline_records=[],
        registry=registry,
        max_findings=3,
        max_findings_per_checkpoint=1,
    )

    assert result["finding_count"] == 3
    assert result["truncated"] is True
    assert result["skip_count"] == 39
    assert result["violation_count"] == 1
    assert result["first_divergence"]["checkpoint_seq"] == 40
    assert result["first_divergence"]["reason"] == "LATE_VIOLATION"


def test_nested_list_outcome_is_json_serializable():
    registry = InvariantRegistry()

    def nested(checkpoint, _context, _timeline):
        return InvariantOutcome(
            result=RESULT_VIOLATION,
            reason="NESTED",
            message="nested",
            expected=[checkpoint],
            observed=[{"nested": checkpoint}],
        )

    registry.register(
        InvariantSpec("test.nested.v1", "1", "ERROR", "generic", nested)
    )
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(_checkpoint(1)),
        timeline_records=[],
        registry=registry,
    )

    json.dumps(result)
    assert isinstance(result["first_divergence"]["expected"][0], dict)
    assert isinstance(result["first_divergence"]["observed"][0]["nested"], dict)


def test_rigped_type_hints_resolve():
    hints = get_type_hints(rigped_invariants._rigped_explicit_claims)
    assert "checkpoint" in hints
    assert "return" in hints


def test_rigped_error_claim_is_error_not_pass():
    registry = default_generic_registry()
    register_rigped_invariants(registry)
    checkpoint = _checkpoint(
        1,
        domain_probes=[
            {
                "name": "rigped",
                "status": "AVAILABLE",
                "state": {
                    "invariant_schema": "awb-rigped-invariant-claims/v1",
                    "invariant_claims": [
                        {
                            "status": "ERROR",
                            "reason": "RIGPED_TEST_ERROR",
                            "message": "Could not evaluate frozen claim.",
                        }
                    ],
                },
            }
        ],
    )

    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(checkpoint),
        timeline_records=[],
        registry=registry,
    )

    assert result["error_count"] == 1
    assert result["violation_count"] == 0
    assert result["first_divergence"] is None
    assert any(
        item["result"] == RESULT_ERROR and item["reason"] == "RIGPED_TEST_ERROR"
        for item in result["findings"]
    )


def test_terminal_boundary_status_mismatch_is_violation():
    checkpoint = _checkpoint(
        1,
        boundary="AFTER_OPERATION",
        state_hash="hash-a",
    )
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(checkpoint),
        timeline_records=[_terminal(terminal_status="FAILED")],
    )

    assert result["first_divergence"] is not None
    assert result["first_divergence"]["reason"] == "TERMINAL_STATUS_BOUNDARY_MISMATCH"


def test_terminal_pairing_uses_operation_before_child_span():
    checkpoint = _checkpoint(
        1,
        boundary="AFTER_OPERATION",
        state_hash="hash-a",
        span_id="child-span",
        operation_id="op-1",
    )
    terminal = _terminal(
        span_id="root-span",
        operation_id="op-1",
        terminal_status="FINISHED",
    )
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(checkpoint),
        timeline_records=[terminal],
    )

    assert result["violation_count"] == 0
    assert not any(
        item["reason"] == "TERMINAL_TRACE_EVENT_MISSING"
        for item in result["findings"]
    )


def test_diff_flag_and_paths_must_agree():
    before = _checkpoint(1, state_hash="hash-a")
    changed_without_paths = _checkpoint(
        2,
        boundary="AFTER_OPERATION",
        previous_id="cp-1",
        previous_status="RESOLVED",
        state_hash="hash-b",
        diff={
            "changed": True,
            "before_hash": "hash-a",
            "after_hash": "hash-b",
            "changes": [],
        },
    )
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(before, changed_without_paths),
        timeline_records=[_terminal()],
    )
    assert any(
        item["reason"] == "DIFF_CHANGED_WITHOUT_PATHS"
        for item in result["findings"]
    )

    unchanged_with_paths = _checkpoint(
        2,
        boundary="AFTER_OPERATION",
        previous_id="cp-1",
        previous_status="RESOLVED",
        state_hash="hash-a",
        diff={
            "changed": False,
            "before_hash": "hash-a",
            "after_hash": "hash-a",
            "changes": [{"path": "/x"}],
        },
    )
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(before, unchanged_with_paths),
        timeline_records=[_terminal()],
    )
    assert any(
        item["reason"] == "DIFF_UNCHANGED_WITH_PATHS"
        for item in result["findings"]
    )


def test_first_divergence_tie_breaks_by_registration_order():
    registry = InvariantRegistry()

    def violate_a(_checkpoint, _context, _timeline):
        return InvariantOutcome(
            result=RESULT_VIOLATION,
            reason="A",
            message="a",
        )

    def violate_b(_checkpoint, _context, _timeline):
        return InvariantOutcome(
            result=RESULT_VIOLATION,
            reason="B",
            message="b",
        )

    registry.register(InvariantSpec("test.a.v1", "1", "ERROR", "generic", violate_a))
    registry.register(InvariantSpec("test.b.v1", "1", "ERROR", "generic", violate_b))
    result = evaluate_incident_divergences(
        incident_id="INC-test",
        checkpoint_bundle=_bundle(_checkpoint(1)),
        timeline_records=[],
        registry=registry,
    )

    assert result["violation_count"] == 2
    assert result["first_divergence"]["invariant_id"] == "test.a.v1"
    assert result["first_divergence"]["registration_order"] == 1
