from __future__ import annotations

import json
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from scripts import awb_debug_git_bisect_trial as trial

COMMIT = "a" * 40
BASELINE = b"frozen baseline bytes"


@pytest.fixture
def workspace_tmp():
    root = Path.cwd() / "build" / f"bb9-trial-test-{uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _oracle() -> dict[str, Any]:
    return {
        "schema": "awb-failure-oracle/v1",
        "status": "BOUND",
        "oracle_kind": "DIVERGENCE",
        "invariant_id": "generic.hash_diff_consistency.v1",
        "finding_code": "DIFF_AFTER_HASH_MISMATCH",
        "checkpoint_seq": 8,
        "boundary": "AFTER_OPERATION",
        "paths": ["/diff_from_previous/after_hash"],
        "expected_verdict": "VIOLATION",
    }


def _replay(*, path: str = "SEMANTIC") -> dict[str, Any]:
    replay = {
        "schema": "awb-debug-incident-replay/v1",
        "incident_id": "INC-BB9-TEST",
        "status": "AVAILABLE",
        "freshness": "FRESH",
        "coverage": "FULL_AUTOMATIC",
        "recommended_path": path,
        "semantic_replay_mode": "RECORDED_RESULT",
        "semantic_replay": {
            "schema": "awb-semantic-replay/v1",
            "blend_file": "source.blend",
            "action_count": 1,
            "actions": [{"kind": "SCRUB", "source_seq": 1, "frames": [{"frame": 33}]}],
        },
        "failure_oracle": _oracle(),
    }
    if path == "GUI_INPUT":
        replay["gui_replay"] = {
            "schema": "awb-gui-replay/v1",
            "action_count": 1,
            "actions": [{"requested_mode": "ROTATE"}],
        }
    return replay


def _clean_result(digest: str = "expected-digest") -> dict[str, Any]:
    return {
        "schema": "awb-debug-git-bisect-bootstrap-result/v1",
        "path": "SEMANTIC",
        "execution_status": "EXECUTED",
        "semantic_result": {"schema": "awb-semantic-replay-result/v1", "action_count": 1},
        "checkpoints": [
            {
                "checkpoint_seq": 1,
                "checkpoint_id": "cp-1",
                "boundary": "BEFORE_OPERATION",
                "semantic_state_hash": "abc",
            }
        ],
        "timeline": [],
        "runtime_identity": {
            "blender_version": "5.2.1 LTS",
            "installed_extension_digest": digest,
        },
    }


def _args(tmp_path: Path, *, path: str = "SEMANTIC") -> Namespace:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    trial_dir = tmp_path / "trial"
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(_replay(path=path)), encoding="utf-8")
    baseline = tmp_path / "baseline.blend"
    baseline.write_bytes(BASELINE)
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"blender stub")
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_text("# frozen bootstrap stub\n", encoding="utf-8")
    return Namespace(
        candidate_root=candidate,
        expected_commit=COMMIT,
        trial_dir=trial_dir,
        replay=replay_path,
        baseline_master=baseline,
        expected_baseline_sha256=trial.sha256_file(baseline),
        blender=blender,
        bootstrap=bootstrap,
        foreground=False,
        event_simulation=False,
        timeout=2.0,
    )


def _install_successful_trial(monkeypatch: pytest.MonkeyPatch, args: Namespace) -> None:
    monkeypatch.setattr(trial, "_runtime_version", lambda *_args: "Blender 5.2.1 LTS")
    monkeypatch.setattr(trial, "_git_identity", lambda *_args: (COMMIT, ""))

    def install(_blender, _candidate_root, trial_dir, _timeout):
        extension_id = "blender_animation_workbench"
        user_extensions = trial_dir / "blender-user-extensions"
        installed_root = user_extensions / "user_default" / extension_id
        installed_root.mkdir(parents=True)
        (installed_root / "blender_manifest.toml").write_text("id='test'\n", encoding="utf-8")
        digest = trial.sha256_tree(installed_root)
        return digest, digest, {"BLENDER_USER_EXTENSIONS": str(user_extensions)}, extension_id

    def execute(_blender, _env, trial_dir, *_args, **_kwargs):
        installed_root = (
            trial_dir
            / "blender-user-extensions"
            / "user_default"
            / "blender_animation_workbench"
        )
        result = _clean_result(trial.sha256_tree(installed_root))
        result["runtime_identity"].update(
            {
                "installed_extension_root": str(installed_root),
                "addon_module_file": str(installed_root / "__init__.py"),
            }
        )
        return result, "EXECUTED", ""

    monkeypatch.setattr(
        trial,
        "build_and_install_candidate",
        install,
    )
    monkeypatch.setattr(
        trial,
        "_run_blender_trial",
        execute,
    )
    monkeypatch.setattr(trial, "_recorded_sys_path", lambda: [str(args.trial_dir.parent)])


def test_bb9_exit_mapping_is_clean_good_exact_bad_and_uncertain_skip():
    assert trial.map_verdict("CLEAN") == (0, "GOOD_CLEAN")
    assert trial.map_verdict("REPRODUCED") == (1, "BAD_EXACT_TARGET_REPRODUCED")
    assert trial.map_verdict("NEW_VIOLATION") == (125, "SKIP_UNPROVEN_OR_DIFFERENT")
    assert trial.map_verdict("UNAVAILABLE")[0] == 125


def test_bb9_clean_complete_candidate_returns_good_zero(workspace_tmp, monkeypatch):
    args = _args(workspace_tmp)
    _install_successful_trial(monkeypatch, args)

    assert trial.run_trial(args) == 0
    receipt = json.loads((args.trial_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["oracle_verdict"] == "CLEAN"
    assert receipt["exit_mapping"] == "GOOD_CLEAN"


def test_bb9_exact_target_failure_returns_bad_one(workspace_tmp, monkeypatch):
    args = _args(workspace_tmp)
    _install_successful_trial(monkeypatch, args)
    expected_signature = trial.reduction_failure_signature(_oracle())
    exact = _clean_result()
    exact["checkpoints"] = [
        {
            "checkpoint_seq": 1,
            "checkpoint_id": "before",
            "boundary": "BEFORE_OPERATION",
            "semantic_state_hash": "previous",
        },
        {
            "checkpoint_seq": 2,
            "checkpoint_id": "after",
            "previous_checkpoint_id": "before",
            "previous_checkpoint_status": "AVAILABLE",
            "boundary": "AFTER_OPERATION",
            "semantic_state_hash": "current",
            "diff_from_previous": {
                "before_hash": "previous",
                "after_hash": "wrong",
                "changed": True,
                "changes": [{"path": "/stable/value"}],
            },
        },
    ]
    def execute(_blender, env, trial_dir, *_args, **_kwargs):
        actual_digest = trial.sha256_tree(
            trial_dir
            / "blender-user-extensions"
            / "user_default"
            / "blender_animation_workbench"
        )
        exact["runtime_identity"]["installed_extension_digest"] = actual_digest
        installed_root = (
            trial_dir
            / "blender-user-extensions"
            / "user_default"
            / "blender_animation_workbench"
        )
        exact["runtime_identity"].update(
            {
                "installed_extension_root": str(installed_root),
                "addon_module_file": str(installed_root / "__init__.py"),
            }
        )
        return exact, "EXECUTED", ""

    monkeypatch.setattr(trial, "_run_blender_trial", execute)

    assert args.expected_commit == COMMIT
    assert trial.run_trial(args) == 1
    receipt = json.loads((args.trial_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["oracle_verdict"] == "REPRODUCED"
    assert receipt["observed_oracle_signature"] == expected_signature


def test_bb9_catch_all_candidate_exception_maps_to_skip(workspace_tmp, monkeypatch):
    args = _args(workspace_tmp)
    monkeypatch.setattr(trial, "_recorded_sys_path", lambda: [str(workspace_tmp)])
    monkeypatch.setattr(trial, "_runtime_version", lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))

    assert trial.run_trial(args) == trial.EXIT_SKIP
    receipt = json.loads((args.trial_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["exit_code"] == 125
    assert receipt["oracle_verdict"] == "UNAVAILABLE"


def test_bb9_candidate_sys_path_contamination_aborts_instead_of_bad(workspace_tmp, monkeypatch):
    args = _args(workspace_tmp)
    monkeypatch.setattr(trial, "_recorded_sys_path", lambda: [str(args.candidate_root)])

    assert trial.run_trial(args) >= 128
    receipt = json.loads((args.trial_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["execution_status"] == "IDENTITY_INVALID"
    assert receipt["exit_code"] >= 128
    assert receipt["oracle_verdict"] != "REPRODUCED"


def test_bb9_built_installed_digest_mismatch_is_skip(workspace_tmp, monkeypatch):
    args = _args(workspace_tmp)
    _install_successful_trial(monkeypatch, args)
    monkeypatch.setattr(
        trial,
        "build_and_install_candidate",
        lambda *_args: ("built", "installed-mismatch", {}, "blender_animation_workbench"),
    )

    assert trial.run_trial(args) == trial.EXIT_SKIP
    receipt = json.loads((args.trial_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["reason"] == "BUILT_INSTALLED_EXTENSION_DIGEST_MISMATCH"
    assert receipt["oracle_verdict"] == "UNAVAILABLE"


def test_bb9_dirty_tracked_candidate_is_skip(workspace_tmp, monkeypatch):
    args = _args(workspace_tmp)
    _install_successful_trial(monkeypatch, args)
    monkeypatch.setattr(trial, "_git_identity", lambda *_args: (COMMIT, " M tracked.py"))

    assert trial.run_trial(args) == trial.EXIT_SKIP
    receipt = json.loads((args.trial_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["candidate_dirty_before"] is True
    assert receipt["reason"] == "CANDIDATE_TRACKED_WORKTREE_DIRTY_BEFORE"


def test_bb9_candidate_dirty_after_preparation_is_skip(workspace_tmp, monkeypatch):
    args = _args(workspace_tmp)
    _install_successful_trial(monkeypatch, args)
    identities = iter([(COMMIT, ""), (COMMIT, " M tracked.py")])
    monkeypatch.setattr(trial, "_git_identity", lambda *_args: next(identities))

    assert trial.run_trial(args) == trial.EXIT_SKIP
    receipt = json.loads((args.trial_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["candidate_dirty_before"] is False
    assert receipt["candidate_dirty_after"] is True
    assert receipt["reason"] == "CANDIDATE_CHANGED_DURING_PREPARATION"


def test_bb9_baseline_drift_after_blender_run_is_skip(workspace_tmp, monkeypatch):
    args = _args(workspace_tmp)
    _install_successful_trial(monkeypatch, args)

    def mutate_trial_baseline(*call_args, **_kwargs):
        baseline_copy = call_args[3]
        baseline_copy.write_bytes(b"changed by blender or external process")
        digest = trial.sha256_tree(
            args.trial_dir
            / "blender-user-extensions"
            / "user_default"
            / "blender_animation_workbench"
        )
        result = _clean_result(digest)
        installed_root = (
            args.trial_dir
            / "blender-user-extensions"
            / "user_default"
            / "blender_animation_workbench"
        )
        result["runtime_identity"].update(
            {
                "installed_extension_root": str(installed_root),
                "addon_module_file": str(installed_root / "__init__.py"),
            }
        )
        return result, "EXECUTED", ""

    monkeypatch.setattr(trial, "_run_blender_trial", mutate_trial_baseline)

    assert trial.run_trial(args) == trial.EXIT_SKIP
    receipt = json.loads((args.trial_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["reason"] == "BASELINE_DRIFT_DETECTED"
    assert receipt["oracle_verdict"] == "UNAVAILABLE"


def test_bb9_different_first_failure_signature_is_skip(workspace_tmp):
    target = trial.reduction_failure_signature(_oracle())
    changed = _clean_result()
    changed["checkpoints"] = [
        {
            "checkpoint_seq": 1,
            "checkpoint_id": "before",
            "boundary": "BEFORE_OPERATION",
            "semantic_state_hash": "one",
        },
        {
            "checkpoint_seq": 2,
            "checkpoint_id": "after",
            "previous_checkpoint_id": "before",
            "previous_checkpoint_status": "AVAILABLE",
            "boundary": "AFTER_OPERATION",
            "semantic_state_hash": "two",
            "diff_from_previous": {
                "before_hash": "wrong",
                "after_hash": "two",
                "changed": True,
                "changes": [{"path": "/stable/value"}],
            },
        },
    ]
    changed["timeline"] = []

    verdict, observed, reason = trial.evaluate_trial_oracle(target, changed)
    assert verdict == "NEW_VIOLATION"
    assert observed != target
    assert reason == "DIFFERENT_FIRST_DIVERGENCE"
    assert trial.map_verdict(verdict)[0] == 125


def test_bb9_gui_input_without_foreground_event_simulation_is_skip(workspace_tmp, monkeypatch):
    args = _args(workspace_tmp, path="GUI_INPUT")
    _install_successful_trial(monkeypatch, args)

    assert trial.run_trial(args) == trial.EXIT_SKIP
    receipt = json.loads((args.trial_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["reason"] == "GUI_INPUT_REQUIRES_FOREGROUND_EVENT_SIMULATION"
    assert receipt["exit_code"] == 125


def test_bb9_matching_first_divergence_ignores_checkpoint_sequence():
    target = trial.reduction_failure_signature(_oracle())
    result = _clean_result()
    result["checkpoints"] = [
        {
            "checkpoint_seq": 99,
            "checkpoint_id": "previous",
            "boundary": "BEFORE_OPERATION",
            "semantic_state_hash": "state-previous",
        },
        {
            "checkpoint_seq": 2,
            "checkpoint_id": "runtime-volatile-id",
            "boundary": "AFTER_OPERATION",
            "previous_checkpoint_id": "previous",
            "previous_checkpoint_status": "AVAILABLE",
            "semantic_state_hash": "state-current",
            "diff_from_previous": {
                "before_hash": "state-previous",
                "after_hash": "unexpected",
                "changed": True,
                "changes": [{"path": "/stable/value"}],
            },
        },
    ]

    verdict, observed, _reason = trial.evaluate_trial_oracle(target, result)
    assert verdict == "REPRODUCED"
    assert observed == target
