from __future__ import annotations

from copy import deepcopy

from scripts.awb_debug_differential_replay import (
    build_replay_run_receipt,
    compare_replay_receipts,
)


def _semantic_result() -> dict:
    return {
        "schema": "awb-semantic-replay-result/v1",
        "replay_mode": "RECORDED_RESULT",
        "replay_execution_id": "runtime-execution-good",
        "action_count": 1,
        "step_manifest": [
            {
                "step_index": 1,
                "step_id": "semantic:0001",
                "source_seq": 7,
                "action_kind": "MOVE",
                "action_route": "FREE_FK_MOVE",
                "checkpoint_boundaries": [
                    "BEFORE_OPERATION",
                    "AFTER_OPERATION",
                ],
            }
        ],
        "results": [{"kind": "MOVE"}],
    }


def _checkpoint(
    boundary: str,
    *,
    state_hash: str,
    location: float,
    runtime_checkpoint_id: str,
    runtime_trace_id: str,
) -> dict:
    return {
        "schema": "awb-debug-checkpoint/v1",
        "checkpoint_id": runtime_checkpoint_id,
        "session_id": "runtime-session",
        "boundary": boundary,
        "status": "AVAILABLE",
        "trace_id": runtime_trace_id,
        "span_id": f"{runtime_trace_id}:span",
        "parent_span_id": None,
        "operation_id": f"{runtime_trace_id}:op",
        "parent_operation_id": None,
        "replay_execution_id": "runtime-execution-good",
        "replay_step_id": "semantic:0001",
        "replay_step_index": 1,
        "replay_action_kind": "MOVE",
        "replay_action_route": "FREE_FK_MOVE",
        "replay_boundary_ordinal": 1,
        "semantic_state_hash": state_hash,
        "hash_schema": "awb-semantic-state-sha256/v1",
        "normalization_schema": "awb-debug-normalized-state/v1",
        "context_identity": {"mode": "POSE"},
        "blender_state": {"frame": 1},
        "native_state": {"objects": [{"name": "Cube", "location": [location, 0.0, 0.0]}]},
        "action_fcurves": {},
        "depsgraph_state": {},
        "domain_probes": [],
    }


def _receipt(
    *,
    run_label: str,
    extension_digest: str,
    after_hash: str = "hash-after",
    after_location: float = 0.0,
    checkpoints: list[dict] | None = None,
) -> dict:
    semantic = _semantic_result()
    rows = checkpoints or [
        _checkpoint(
            "BEFORE_OPERATION",
            state_hash="hash-before",
            location=0.0,
            runtime_checkpoint_id=f"{run_label}:before",
            runtime_trace_id=f"{run_label}:trace-before",
        ),
        _checkpoint(
            "AFTER_OPERATION",
            state_hash=after_hash,
            location=after_location,
            runtime_checkpoint_id=f"{run_label}:after",
            runtime_trace_id=f"{run_label}:trace-after",
        ),
    ]
    return build_replay_run_receipt(
        incident_id="INC-bb6-test",
        run_label=run_label,
        incident_replay_sha256="incident-sha",
        baseline_blend_sha256="blend-sha",
        blender_version="5.2.1 LTS",
        installed_extension_digest=extension_digest,
        installed_extension_root=f"/installed/{run_label}",
        source_identity={"commit": f"commit-{run_label}", "dirty": False},
        runner_identity={"runner_sha256": "runner-sha"},
        semantic_result=semantic,
        checkpoints=rows,
    )


def test_bb6_receipt_contains_stable_manifest_and_normalized_checkpoint_sequence():
    receipt = _receipt(run_label="GOOD", extension_digest="digest-good")

    assert receipt["schema"] == "awb-replay-run-receipt/v1"
    assert receipt["status"] == "VALID"
    assert receipt["step_manifest"][0]["step_id"] == "semantic:0001"
    assert [
        (
            item["alignment"]["step_id"],
            item["alignment"]["boundary"],
            item["alignment"]["within_step_ordinal"],
        )
        for item in receipt["checkpoints"]
    ] == [
        ("semantic:0001", "BEFORE_OPERATION", 1),
        ("semantic:0001", "AFTER_OPERATION", 1),
    ]
    assert receipt["installed_extension_digest"] == "digest-good"
    assert receipt["hash_schema"] == "awb-semantic-state-sha256/v1"
    assert receipt["normalization_schema"] == "awb-debug-normalized-state/v1"


def test_bb6_self_check_allows_same_installed_identity_but_normal_mode_rejects_it():
    receipt = _receipt(run_label="SELF", extension_digest="digest-same")

    self_check = compare_replay_receipts(receipt, deepcopy(receipt), mode="SELF_CHECK")
    assert self_check["status"] == "MATCH"
    assert self_check["aligned_checkpoint_count"] == 2

    normal = compare_replay_receipts(receipt, deepcopy(receipt), mode="NORMAL")
    assert normal["status"] == "INVALID"
    assert normal["reason"] == "SAME_INSTALLED_EXTENSION_IDENTITY"


def test_bb6_normal_mode_ignores_cross_run_uuid_provenance_for_equality():
    good = _receipt(run_label="GOOD", extension_digest="digest-good")
    current = _receipt(run_label="CURRENT", extension_digest="digest-current")

    result = compare_replay_receipts(good, current, mode="NORMAL")

    assert result["status"] == "MATCH"
    assert result["aligned_checkpoint_count"] == 2


def test_bb6_first_difference_is_reported_by_stable_step_boundary_and_path():
    good = _receipt(run_label="GOOD", extension_digest="digest-good")
    current = _receipt(
        run_label="CURRENT",
        extension_digest="digest-current",
        after_hash="hash-after-current",
        after_location=1.25,
    )

    result = compare_replay_receipts(good, current, mode="NORMAL")

    assert result["status"] == "DIFFERENT"
    assert result["first_difference"]["alignment"] == {
        "step_id": "semantic:0001",
        "boundary": "AFTER_OPERATION",
        "within_step_ordinal": 1,
    }
    assert "/native_state/objects/0/location/0" in result["first_difference"]["paths"]


def test_bb6_missing_or_duplicate_checkpoint_alignment_is_unaligned_not_skipped():
    good = _receipt(run_label="GOOD", extension_digest="digest-good")

    missing_rows = [
        _checkpoint(
            "BEFORE_OPERATION",
            state_hash="hash-before",
            location=0.0,
            runtime_checkpoint_id="current:before",
            runtime_trace_id="current:trace-before",
        )
    ]
    current_missing = _receipt(
        run_label="CURRENT",
        extension_digest="digest-current",
        checkpoints=missing_rows,
    )
    assert "MISSING_CHECKPOINT_ALIGNMENT" in current_missing["issues"]
    result = compare_replay_receipts(good, current_missing, mode="NORMAL")
    assert result["status"] == "UNALIGNED"
    assert result["reason"] == "ALIGNMENT_CONTRACT_MISMATCH"

    duplicate_rows = [
        _checkpoint(
            "BEFORE_OPERATION",
            state_hash="hash-before",
            location=0.0,
            runtime_checkpoint_id="current:before:1",
            runtime_trace_id="current:trace-before:1",
        ),
        _checkpoint(
            "BEFORE_OPERATION",
            state_hash="hash-before",
            location=0.0,
            runtime_checkpoint_id="current:before:2",
            runtime_trace_id="current:trace-before:2",
        ),
        _checkpoint(
            "AFTER_OPERATION",
            state_hash="hash-after",
            location=0.0,
            runtime_checkpoint_id="current:after",
            runtime_trace_id="current:trace-after",
        ),
    ]
    current_duplicate = _receipt(
        run_label="CURRENT",
        extension_digest="digest-current",
        checkpoints=duplicate_rows,
    )
    assert "DUPLICATE_CHECKPOINT_ALIGNMENT" in current_duplicate["issues"]
    result = compare_replay_receipts(good, current_duplicate, mode="NORMAL")
    assert result["status"] == "UNALIGNED"
    assert result["reason"] == "AMBIGUOUS_ALIGNMENT"


def test_bb6_receipt_fails_closed_when_execution_identity_is_incomplete():
    semantic = _semantic_result()
    receipt = build_replay_run_receipt(
        incident_id="INC-bb6-test",
        run_label="CURRENT",
        incident_replay_sha256="incident-sha",
        baseline_blend_sha256="blend-sha",
        blender_version="5.2.1 LTS",
        installed_extension_digest="",
        installed_extension_root="/installed/current",
        source_identity={"commit": "current", "dirty": False},
        runner_identity={"runner_sha256": "runner-sha"},
        semantic_result=semantic,
        checkpoints=[
            _checkpoint(
                "BEFORE_OPERATION",
                state_hash="hash-before",
                location=0.0,
                runtime_checkpoint_id="current:before",
                runtime_trace_id="current:trace-before",
            ),
            _checkpoint(
                "AFTER_OPERATION",
                state_hash="hash-after",
                location=0.0,
                runtime_checkpoint_id="current:after",
                runtime_trace_id="current:trace-after",
            ),
        ],
    )
    assert receipt["status"] == "INVALID"
    assert "EXECUTION_IDENTITY_INCOMPLETE" in receipt["issues"]


def test_bb6_live_replay_evidence_assigns_frozen_checkpoint_sequence_before_bb4():
    from scripts.replay_awb_incident import _evaluate_replay_evidence

    before = _checkpoint(
        "BEFORE_OPERATION",
        state_hash="hash-before",
        location=0.0,
        runtime_checkpoint_id="runtime:before",
        runtime_trace_id="trace-runtime",
    )
    after = _checkpoint(
        "AFTER_OPERATION",
        state_hash="hash-after",
        location=1.0,
        runtime_checkpoint_id="runtime:after",
        runtime_trace_id="trace-runtime",
    )
    after["previous_checkpoint_id"] = "runtime:before"
    after["diff_from_previous"] = {"changed": True, "changes": []}

    divergences, _oracle = _evaluate_replay_evidence(
        incident_id="INC-bb6-live-freeze",
        evidence={"checkpoints": [before, after], "timeline": []},
        oracle=None,
    )

    chain_violations = [
        finding
        for finding in divergences["findings"]
        if finding["invariant_id"] == "generic.checkpoint_chain.v1"
        and finding["result"] == "VIOLATION"
    ]
    assert chain_violations == []


def test_bb6_symmetric_missing_checkpoint_is_unaligned_from_manifest_not_silently_matched():
    missing_rows = [
        _checkpoint(
            "BEFORE_OPERATION",
            state_hash="hash-before",
            location=0.0,
            runtime_checkpoint_id="shared:before",
            runtime_trace_id="shared:trace-before",
        )
    ]
    good = _receipt(
        run_label="GOOD",
        extension_digest="digest-good",
        checkpoints=missing_rows,
    )
    current = _receipt(
        run_label="CURRENT",
        extension_digest="digest-current",
        checkpoints=deepcopy(missing_rows),
    )

    assert good["status"] == "VALID"
    assert current["status"] == "VALID"
    result = compare_replay_receipts(good, current, mode="NORMAL")

    assert result["status"] == "UNALIGNED"
    assert result["reason"] == "ALIGNMENT_CONTRACT_MISMATCH"
    assert {
        (item["side"], tuple(item["key"]), item["reason"])
        for item in result["unaligned"]
    } == {
        ("GOOD", ("semantic:0001", "AFTER_OPERATION", 1), "MISSING"),
        ("CURRENT", ("semantic:0001", "AFTER_OPERATION", 1), "MISSING"),
    }


def test_bb6_checkpoint_route_must_match_step_manifest_ownership():
    good = _receipt(run_label="GOOD", extension_digest="digest-good")
    current_rows = [
        _checkpoint(
            "BEFORE_OPERATION",
            state_hash="hash-before",
            location=0.0,
            runtime_checkpoint_id="current:before",
            runtime_trace_id="current:trace-before",
        ),
        _checkpoint(
            "AFTER_OPERATION",
            state_hash="hash-after",
            location=0.0,
            runtime_checkpoint_id="current:after",
            runtime_trace_id="current:trace-after",
        ),
    ]
    current_rows[1]["replay_action_route"] = "WRONG_ROUTE"
    current = _receipt(
        run_label="CURRENT",
        extension_digest="digest-current",
        checkpoints=current_rows,
    )

    result = compare_replay_receipts(good, current, mode="NORMAL")

    assert result["status"] == "UNALIGNED"
    assert result["reason"] == "ALIGNMENT_CONTRACT_MISMATCH"
    assert result["unaligned"][0]["reason"] == "CHECKPOINT_OWNERSHIP_MISMATCH"
    assert result["unaligned"][0]["observed"]["action_route"] == "WRONG_ROUTE"


def test_bb6_self_check_rejects_distinct_receipts_even_with_same_installed_identity():
    good = _receipt(run_label="GOOD", extension_digest="digest-same")
    current = _receipt(run_label="CURRENT", extension_digest="digest-same")

    result = compare_replay_receipts(good, current, mode="SELF_CHECK")

    assert result["status"] == "INVALID"
    assert result["reason"] == "SELF_CHECK_REQUIRES_IDENTICAL_RECEIPT"
