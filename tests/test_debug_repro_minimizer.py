from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.awb_debug_repro_minimizer import (
    _promoted_replay,
    build_reduction_plan,
    compile_candidate_replay,
    evaluate_reduction_failure,
    minimize_incident,
    minimize_units,
    normalize_reduction_path,
    qualify_replay,
    reduction_failure_signature,
    validate_candidate,
)


def _oracle(*, checkpoint_seq: int = 6, paths: list[str] | None = None) -> dict:
    return {
        "schema": "awb-failure-oracle/v1",
        "status": "BOUND",
        "oracle_kind": "DIVERGENCE",
        "invariant_id": "generic.hash_diff_consistency.v1",
        "finding_code": "DIFF_AFTER_HASH_MISMATCH",
        "checkpoint_seq": checkpoint_seq,
        "boundary": "AFTER_OPERATION",
        "paths": paths if paths is not None else ["/diff_from_previous/after_hash"],
        "expected_verdict": "VIOLATION",
    }


def _semantic_replay(actions: list[dict], *, baseline: str = "baseline.blend") -> dict:
    return {
        "schema": "awb-debug-incident-replay/v1",
        "incident_id": "INC-BB7-TEST",
        "status": "AVAILABLE",
        "freshness": "FRESH",
        "source": "test",
        "source_session_id": "session",
        "captured_trace_session_id": "session",
        "captured_last_seq": 50,
        "semantic_replay": {
            "schema": "awb-semantic-replay/v1",
            "source_session_id": "session",
            "blend_file": baseline,
            "action_count": len(actions),
            "actions": deepcopy(actions),
        },
        "semantic_replay_mode": "RECORDED_RESULT",
        "coverage": "FULL_AUTOMATIC",
        "recommended_path": "SEMANTIC",
        "gui_replay": None,
        "handoff": None,
        "failure_oracle": _oracle(),
        "unavailable_reason": None,
    }


def _first_divergence(
    *,
    invariant_id: str = "generic.hash_diff_consistency.v1",
    reason: str = "DIFF_AFTER_HASH_MISMATCH",
    checkpoint_seq: int = 2,
    paths: list[str] | None = None,
) -> dict:
    return {
        "result": "VIOLATION",
        "invariant_id": invariant_id,
        "reason": reason,
        "checkpoint_seq": checkpoint_seq,
        "boundary": "AFTER_OPERATION",
        "paths": paths if paths is not None else ["/diff_from_previous/after_hash"],
    }


def test_bb7_reduction_signature_drops_checkpoint_seq_but_keeps_semantic_failure_identity():
    one = reduction_failure_signature(_oracle(checkpoint_seq=2))
    two = reduction_failure_signature(_oracle(checkpoint_seq=99))

    assert one == two
    assert one["boundary"] == "AFTER_OPERATION"
    assert one["paths"] == ["/diff_from_previous/after_hash"]


def test_bb7_reduction_paths_fail_closed_on_volatile_or_array_ordinal_components():
    with pytest.raises(ValueError):
        normalize_reduction_path("/checkpoints/2/native_state")
    with pytest.raises(ValueError):
        normalize_reduction_path("/trace_id")

    replay = _semantic_replay(
        [{"kind": "SCRUB", "source_seq": 1, "frames": [{"frame": 1}]}]
    )
    replay["failure_oracle"] = _oracle(paths=["/objects/0/location"])
    qualified = qualify_replay(replay)
    assert qualified["eligible"] is False
    assert "FAILURE_ORACLE_NOT_REDUCTION_SAFE" in qualified["problems"]


def test_bb7_reduction_oracle_requires_matching_first_divergence_not_later_finding():
    signature = reduction_failure_signature(_oracle())
    divergences = {
        "status": "AVAILABLE",
        "first_divergence": _first_divergence(
            invariant_id="generic.checkpoint_chain.v1",
            reason="PREVIOUS_CHECKPOINT_MISSING",
            checkpoint_seq=1,
            paths=["/native_state/precondition_failure"],
        ),
        "findings": [
            _first_divergence(checkpoint_seq=1),
        ],
    }

    result = evaluate_reduction_failure(
        signature,
        divergences,
        execution_status="EXECUTED",
    )

    assert result["verdict"] == "NEW_VIOLATION"
    assert result["reason"] == "FIRST_DIVERGENCE_CHANGED"


def test_bb7_reduction_oracle_reproduces_same_signature_after_checkpoint_renumbering():
    signature = reduction_failure_signature(_oracle(checkpoint_seq=8))
    divergences = {
        "status": "AVAILABLE",
        "first_divergence": _first_divergence(checkpoint_seq=2),
        "findings": [],
    }

    result = evaluate_reduction_failure(
        signature,
        divergences,
        execution_status="EXECUTED",
    )

    assert result["verdict"] == "REPRODUCED"
    assert result["checkpoint_seq"] == 2


def test_bb7_semantic_dependency_graph_rejects_retained_dependent_without_prerequisite():
    replay = _semantic_replay(
        [
            {"kind": "AUTO_KEY", "source_seq": 1, "enabled": True},
            {
                "kind": "MOVE",
                "source_seq": 2,
                "controls": ["Foot.L"],
                "delta_world": [1.0, 0.0, 0.0],
            },
            {"kind": "SCRUB", "source_seq": 3, "frames": [{"frame": 2}]},
        ]
    )
    plan = build_reduction_plan(replay)

    valid, reason = validate_candidate(
        plan,
        ["semantic:0002", "semantic:0003"],
    )

    assert valid is False
    assert reason == "DEPENDENCY_PREREQUISITE_MISSING"

    candidate = compile_candidate_replay(
        replay,
        plan,
        ["semantic:0001", "semantic:0002"],
    )
    assert [item["source_seq"] for item in candidate["semantic_replay"]["actions"]] == [1, 2]


def test_bb7_gui_units_keep_press_release_atomic_and_require_one_gui_action():
    replay = _semantic_replay([])
    replay["recommended_path"] = "GUI_INPUT"
    replay["semantic_replay"] = None
    replay["semantic_replay_mode"] = None
    replay["gui_replay"] = {
        "schema": "awb-gui-replay/v1",
        "source_session_id": "session",
        "target_area": "VIEW_3D",
        "expected_blend": "baseline.blend",
        "expected_mode": "POSE",
        "expected_context": {},
        "source_seq_range": [10, 20],
        "semantic_prefix": None,
        "action_count": 2,
        "event_count": 4,
        "actions": [
            {
                "stroke_id": 1,
                "source_seq": 10,
                "terminal_source_seq": 11,
                "requested_mode": "MOVE",
                "key": "W",
            },
            {
                "stroke_id": 2,
                "source_seq": 20,
                "terminal_source_seq": 21,
                "requested_mode": "ROTATE",
                "key": "E",
            },
        ],
        "events": [
            {"stroke_id": 1, "source_seq": 10, "type": "W", "value": "PRESS", "requested_mode": "MOVE"},
            {"stroke_id": 1, "source_seq": 10, "type": "W", "value": "RELEASE", "requested_mode": "MOVE"},
            {"stroke_id": 2, "source_seq": 20, "type": "E", "value": "PRESS", "requested_mode": "ROTATE"},
            {"stroke_id": 2, "source_seq": 20, "type": "E", "value": "RELEASE", "requested_mode": "ROTATE"},
        ],
    }

    plan = build_reduction_plan(replay)
    candidate = compile_candidate_replay(replay, plan, ["gui:0002"])

    assert candidate["gui_replay"]["action_count"] == 1
    assert candidate["gui_replay"]["event_count"] == 2
    assert [item["value"] for item in candidate["gui_replay"]["events"]] == ["PRESS", "RELEASE"]
    assert candidate["gui_replay"]["source_seq_range"] == [20, 21]

    valid, reason = validate_candidate(plan, [])
    assert valid is False
    assert reason == "EMPTY_OR_UNKNOWN_UNIT"


def test_bb7_ddmin_is_deterministic_one_minimal_and_memoized():
    plan = {
        "units": [
            {"unit_id": f"semantic:{index:04d}", "group": "SEMANTIC"}
            for index in range(1, 7)
        ],
        "dependencies": {
            f"semantic:{index:04d}": set()
            for index in range(1, 7)
        },
        "minimum_gui_units": 0,
    }
    calls: list[tuple[str, ...]] = []

    def evaluate(retained: tuple[str, ...]) -> dict:
        calls.append(retained)
        return {
            "verdict": (
                "REPRODUCED"
                if "semantic:0005" in retained
                else "CLEAN"
            )
        }

    result = minimize_units(plan, evaluate, max_trials=64)

    assert result["status"] == "MINIMIZED"
    assert result["one_minimal"] is True
    assert result["retained_units"] == ["semantic:0005"]
    assert len(calls) == len(set(calls))


def test_bb7_ddmin_fails_closed_when_trial_budget_exhausts_before_minimality():
    plan = {
        "units": [
            {"unit_id": f"semantic:{index:04d}", "group": "SEMANTIC"}
            for index in range(1, 5)
        ],
        "dependencies": {
            f"semantic:{index:04d}": set()
            for index in range(1, 5)
        },
        "minimum_gui_units": 0,
    }

    result = minimize_units(
        plan,
        lambda _retained: {"verdict": "REPRODUCED"},
        max_trials=1,
    )

    assert result["status"] == "BUDGET_EXCEEDED"
    assert result["one_minimal"] is False


def test_bb7_promoted_replay_rebinds_standard_oracle_checkpoint_sequence():
    replay = _semantic_replay(
        [{"kind": "SCRUB", "source_seq": 5, "frames": [{"frame": 2}]}]
    )
    promoted = _promoted_replay(replay, rebound_checkpoint_seq=2)

    assert replay["failure_oracle"]["checkpoint_seq"] == 6
    assert promoted["failure_oracle"]["checkpoint_seq"] == 2


def test_bb7_full_minimizer_runs_independent_stability_trials_and_keeps_identity_locked(
    monkeypatch,
    tmp_path,
):
    import scripts.awb_debug_repro_minimizer as minimizer

    baseline = tmp_path / "baseline.blend"
    baseline.write_bytes(b"bb7-baseline")
    incident = tmp_path / "incident"
    incident.mkdir()
    replay = _semantic_replay(
        [
            {"kind": "SCRUB", "source_seq": 1, "frames": [{"frame": 1}]},
            {"kind": "SCRUB", "source_seq": 2, "frames": [{"frame": 2}]},
            {"kind": "SCRUB", "source_seq": 3, "frames": [{"frame": 3}]},
            {"kind": "SCRUB", "source_seq": 4, "frames": [{"frame": 4}]},
        ],
        baseline=str(baseline),
    )
    (incident / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    run_count = 0
    reset_ids: list[str] = []

    def fake_run_incident(
        candidate_dir,
        *,
        run_label,
        reset_baseline,
        expected_baseline_sha256,
        timeout_seconds,
    ):
        nonlocal run_count
        del run_label, timeout_seconds
        assert reset_baseline is True
        assert expected_baseline_sha256
        run_count += 1
        reset_id = f"reset:{run_count}"
        reset_ids.append(reset_id)
        candidate = json.loads((candidate_dir / "replay.json").read_text(encoding="utf-8"))
        source_seqs = [
            item["source_seq"]
            for item in candidate["semantic_replay"]["actions"]
        ]
        has_target = 3 in source_seqs
        return {
            "execution_status": "EXECUTED",
            "replay_divergences": {
                "status": "AVAILABLE",
                "first_divergence": (
                    _first_divergence(checkpoint_seq=2)
                    if has_target
                    else None
                ),
            },
            "runtime_identity": {
                "blender_version": "5.2.1 LTS",
                "installed_extension_digest": "digest-fixed",
                "installed_extension_root": "installed-root",
            },
            "source_identity": {
                "commit": "commit-fixed",
                "dirty": False,
            },
            "runner_identity": {"runner_sha256": "runner-fixed"},
            "baseline_blend_sha256": expected_baseline_sha256,
            "reset_identity": {
                "reset_id": reset_id,
                "baseline_path": str(baseline.resolve()),
                "baseline_sha256": expected_baseline_sha256,
            },
        }

    monkeypatch.setattr(minimizer, "run_incident", fake_run_incident)

    result = minimize_incident(
        incident,
        max_trials=32,
        stability_runs=3,
        timeout_seconds=10.0,
        promote=False,
    )

    assert result["status"] == "STABLE"
    assert result["original_unit_count"] == 4
    assert result["minimized_unit_count"] == 1
    assert result["retained_units"] == ["semantic:0003"]
    assert len(result["stability_runs"]) == 3
    assert len(set(reset_ids[-3:])) == 3
    assert result["locked_execution_identity"]["installed_extension_digest"] == "digest-fixed"


def test_bb7_incident_runner_resets_verified_baseline_before_candidate_replay(
    monkeypatch,
    tmp_path,
):
    import re

    import scripts.replay_awb_incident as runner

    baseline = tmp_path / "baseline.blend"
    baseline.write_bytes(b"baseline")
    incident = tmp_path / "incident"
    incident.mkdir()
    replay = _semantic_replay(
        [{"kind": "SCRUB", "source_seq": 1, "frames": [{"frame": 1}]}],
        baseline=str(baseline),
    )
    (incident / "replay.json").write_text(json.dumps(replay), encoding="utf-8")

    calls: list[str] = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute_code(self, code):
            calls.append(code)
            if "bpy.ops.wm.open_mainfile" in code:
                match = re.search(r'"reset_id": \'([^\']+)\'', code)
                assert match is not None
                return {
                    "status": "RESET",
                    "reset_id": match.group(1),
                    "blend_file": str(baseline),
                }
            return {
                "checkpoints": [],
                "timeline": [],
                "runtime_identity": {
                    "blender_version": "5.2.1 LTS",
                    "installed_extension_digest": "digest",
                    "installed_extension_root": "installed",
                },
            }

    monkeypatch.setattr(
        runner,
        "_persistent_blender_client",
        lambda *, timeout: FakeClient(),
    )
    monkeypatch.setattr(
        runner,
        "_run_semantic",
        lambda client, replay, *, prefix_only: {
            "status": "EXECUTED",
            "start_seq": 0,
            "semantic_result": None,
        },
    )

    result = runner.run_incident(
        incident,
        reset_baseline=True,
        expected_baseline_sha256=runner.sha256_file(baseline),
        timeout_seconds=5.0,
    )

    assert "bpy.ops.wm.open_mainfile" in calls[0]
    assert result["reset_identity"]["baseline_sha256"] == runner.sha256_file(baseline)
    assert result["execution_status"] == "EXECUTED"


def test_bb7_source_identity_ignores_runtime_artifacts_but_not_source_changes(
    monkeypatch,
    tmp_path,
):
    import scripts.replay_awb_incident as runner

    status_text = {
        "value": (
            "?? debug/awb_interaction_trace.jsonl\n"
            "?? debug/runtime_error_log/runtime.jsonl\n"
            "?? build/candidate.json\n"
            "?? external_workers/request.md\n"
        )
    }

    def fake_run(args, **_kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="commit-fixed\n")
        assert args[:3] == ["git", "status", "--porcelain"]
        assert "--untracked-files=all" in args
        return SimpleNamespace(returncode=0, stdout=status_text["value"])

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    identity = runner._git_identity(tmp_path)
    assert identity["commit"] == "commit-fixed"
    assert identity["dirty"] is False

    status_text["value"] += "?? scripts/new_source_helper.py\n"
    identity = runner._git_identity(tmp_path)
    assert identity["dirty"] is True
