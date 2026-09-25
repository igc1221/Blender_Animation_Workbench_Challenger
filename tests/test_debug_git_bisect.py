from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from scripts import awb_debug_git_bisect as bisect

GOOD = "a" * 40
PARENT = "b" * 40
FIRST_BAD = "c" * 40
BAD = "d" * 40


@pytest.fixture
def scratch_dir():
    path = Path.cwd() / "tests" / f".git-bisect-test-{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _completed(args=(), *, code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, code, stdout, stderr)


def test_refs_resolve_to_full_shas_and_good_must_be_ancestor(monkeypatch, scratch_dir):
    calls = []

    def fake_git(repo, *args):
        calls.append((repo, args))
        if args[0] == "rev-parse":
            ref = args[-1].split("^{commit}", 1)[0]
            sha = {"good-ref": GOOD, "bad-ref": BAD}[ref]
            return _completed(args, stdout=sha + "\n")
        if args[0] == "merge-base":
            return _completed(args)
        raise AssertionError(args)

    monkeypatch.setattr(bisect, "_git", fake_git)
    assert bisect.validate_commit_range(scratch_dir, "good-ref", "bad-ref") == (GOOD, BAD)
    assert all(call[0] == scratch_dir for call in calls)
    assert calls[-1][1] == ("merge-base", "--is-ancestor", GOOD, BAD)

    def not_ancestor(repo, *args):
        if args[0] == "rev-parse":
            ref = args[-1].split("^{commit}", 1)[0]
            return _completed(args, stdout={"good-ref": GOOD, "bad-ref": BAD}[ref] + "\n")
        return _completed(args, code=1)

    monkeypatch.setattr(bisect, "_git", not_ancestor)
    with pytest.raises(ValueError, match="ancestor"):
        bisect.validate_commit_range(scratch_dir, "good-ref", "bad-ref")


def test_existing_git_operation_and_dirty_candidate_are_refused(monkeypatch, scratch_dir):
    git_dir = scratch_dir / "gitdir"
    common_dir = scratch_dir / "common"
    git_dir.mkdir()
    common_dir.mkdir()
    (git_dir / "BISECT_START").write_text(BAD, encoding="utf-8")
    monkeypatch.setattr(bisect, "_worktree_paths", lambda repo: [scratch_dir])

    def fake_checked_git(repo, *args):
        if args == ("rev-parse", "--absolute-git-dir"):
            return _completed(args, stdout=str(git_dir))
        if args == ("rev-parse", "--git-common-dir"):
            return _completed(args, stdout=str(common_dir))
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return _completed(args, stdout=" M tracked.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(bisect, "_checked_git", fake_checked_git)
    with pytest.raises(RuntimeError, match="operation metadata"):
        bisect.refuse_in_progress_operations(scratch_dir)
    with pytest.raises(RuntimeError, match="dirty"):
        bisect.ensure_clean_worktree(scratch_dir)


def test_endpoint_gate_requires_repeatable_verdict_and_identical_bad_signature():
    observations = [
        {"endpoint": "GOOD", "verdict": "GOOD"},
        {"endpoint": "GOOD", "verdict": "GOOD"},
        {"endpoint": "BAD", "verdict": "BAD", "failure_signature": {"code": "X", "path": "/pose"}},
        {"endpoint": "BAD", "verdict": "BAD", "failure_signature": {"path": "/pose", "code": "X"}},
    ]
    assert bisect.endpoint_gate(observations) == (True, None)

    observations[-1]["failure_signature"] = {"code": "Y", "path": "/pose"}
    assert bisect.endpoint_gate(observations) == (False, "BAD_ENDPOINT_FAILURE_SIGNATURE_CHANGED")
    observations[-1]["failure_signature"] = observations[-2]["failure_signature"]
    observations[-1]["verdict"] = "SKIP"
    assert bisect.endpoint_gate(observations) == (False, "BAD_ENDPOINT_NOT_REPRODUCIBLE")


def test_direct_parent_must_have_direct_good_trial_and_skip_is_ambiguous(monkeypatch, scratch_dir):
    relation = {
        (PARENT, BAD): True,
        (PARENT, GOOD): False,
        (FIRST_BAD, BAD): True,
        (FIRST_BAD, GOOD): False,
    }
    def ancestor(left, right):
        return relation.get((left, right), False)

    ok, unresolved = bisect.validate_direct_parents(
        [PARENT],
        good=GOOD,
        bad=BAD,
        trial_by_commit={PARENT: {"verdict": "GOOD"}},
        is_ancestor=ancestor,
    )
    assert (ok, unresolved) == (True, [])

    monkeypatch.setattr(bisect, "_parents", lambda repo, commit: [PARENT])
    monkeypatch.setattr(
        bisect,
        "_is_ancestor",
        lambda repo, left, right: ancestor(left, right),
    )
    result = bisect._first_bad_result(
        repo=scratch_dir,
        good=GOOD,
        bad=BAD,
        stdout=f"{FIRST_BAD} is the first bad commit\n",
        stderr="",
        bisect_log="",
        records=[
            {"commit_sha": FIRST_BAD, "verdict": "BAD"},
            {"commit_sha": PARENT, "verdict": "SKIP", "reason": "REPLAY_UNAVAILABLE"},
        ],
    )
    assert result["status"] == "AMBIGUOUS"
    assert result["reason"] == "DIRECT_PARENT_NOT_DIRECTLY_TRIALED_GOOD"
    assert result["unresolved_parents"] == [{"commit": PARENT, "verdict": "SKIP"}]


def test_first_bad_parser_rejects_conflicting_git_output():
    assert bisect.parse_first_bad_commit(f"{FIRST_BAD} is the first bad commit") == FIRST_BAD
    assert bisect.parse_first_bad_commit(
        f"{FIRST_BAD} is the first bad commit",
        f"# first bad commit: [{BAD}] message",
    ) is None


def test_runner_exit_mapping_only_allows_frozen_contract_codes():
    assert [bisect.classify_runner_exit(code) for code in (0, 1, 125, 128, 2)] == [
        "GOOD",
        "BAD",
        "SKIP",
        "ABORT",
        "INVALID",
    ]
    assert bisect.runner_exit_code(2) == 128
    assert bisect.runner_exit_code(130) == 130


def test_frozen_trial_driver_is_valid_python_and_runner_path_is_absolute(scratch_dir):
    compile(bisect._driver_source(), "trial_driver.py", "exec")
    command = bisect._trial_driver_command(
        driver=scratch_dir / "driver.py",
        candidate=scratch_dir / "candidate",
        session_dir=scratch_dir / "session",
        runner=scratch_dir / "frozen" / "runner.py",
        bootstrap=None,
        replay=None,
    )
    assert command[1] == "-I"
    assert Path(command[2]).is_absolute()
    assert command[command.index("--runner") + 1] == str((scratch_dir / "frozen" / "runner.py").resolve())


def _orchestrator_fixture(monkeypatch, scratch_dir, *, run_raises=False, endpoints=None):
    main_repo = scratch_dir / "main"
    main_repo.mkdir()
    temp_root = scratch_dir / "isolated"
    temp_root.mkdir()
    runner = scratch_dir / "trial_runner.py"
    runner.write_text("# frozen runner input\n", encoding="utf-8")
    result_path = scratch_dir / "final-result.json"
    git_calls = []
    bisect_worktree = {"path": None}

    def fake_mkdtemp(*, prefix, dir):
        path = Path(dir) / f"{prefix}{uuid4().hex}"
        path.mkdir()
        return str(path)

    def fake_checked_git(repo, *args):
        if args[:2] == ("worktree", "add"):
            bisect_worktree["path"] = Path(args[3])
            bisect_worktree["path"].mkdir()
        if args[0] == "show":
            return _completed(args, stdout=PARENT + "\n")
        return _completed(args)

    def fake_git(repo, *args):
        git_calls.append((Path(repo), args))
        if args[:2] == ("bisect", "start"):
            return _completed(args)
        if args[:2] == ("bisect", "run"):
            if run_raises:
                raise RuntimeError("simulated git bisect run boundary failure")
            return _completed(args, stdout=f"{FIRST_BAD} is the first bad commit\n")
        if args[:2] == ("bisect", "log"):
            return _completed(args, stdout=f"# first bad commit: [{FIRST_BAD}] message\n")
        if args[:2] == ("bisect", "reset"):
            return _completed(args)
        if args[:2] == ("worktree", "remove"):
            return _completed(args)
        if args[0] == "show":
            return _completed(args, stdout=PARENT + "\n")
        if args[0] == "merge-base":
            ancestor, descendant = args[-2:]
            return _completed(args, code=0 if (ancestor, descendant) in {(PARENT, BAD), (FIRST_BAD, BAD)} else 1)
        raise AssertionError((repo, args))

    monkeypatch.setattr(bisect, "refuse_in_progress_operations", lambda repo: None)
    monkeypatch.setattr(bisect.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(bisect, "validate_commit_range", lambda repo, good, bad: (GOOD, BAD))
    monkeypatch.setattr(bisect, "_checked_git", fake_checked_git)
    monkeypatch.setattr(bisect, "_git", fake_git)
    monkeypatch.setattr(bisect, "ensure_clean_worktree", lambda repo: None)

    endpoint_values = endpoints or [
        {"verdict": "GOOD", "mapped_verdict": "GOOD"},
        {"verdict": "GOOD", "mapped_verdict": "GOOD"},
        {"verdict": "BAD", "mapped_verdict": "BAD", "failure_signature": {"code": "X"}},
        {"verdict": "BAD", "mapped_verdict": "BAD", "failure_signature": {"code": "X"}},
    ]
    endpoint_index = {"value": 0}

    def fake_endpoint_trial(*, endpoint, sha, attempt, **kwargs):
        item = dict(endpoint_values[endpoint_index["value"]])
        endpoint_index["value"] += 1
        return {**item, "endpoint": endpoint, "commit_sha": sha, "trial_id": f"endpoint-{endpoint}-{attempt}"}

    monkeypatch.setattr(bisect, "_invoke_endpoint_trial", fake_endpoint_trial)
    monkeypatch.setattr(
        bisect,
        "_read_trial_records",
        lambda session_dir: [
            {"schema": bisect.TRIAL_RECORD_SCHEMA, "stage": "bisect", "commit_sha": PARENT, "verdict": "GOOD"},
            {"schema": bisect.TRIAL_RECORD_SCHEMA, "stage": "bisect", "commit_sha": FIRST_BAD, "verdict": "BAD"},
        ],
    )
    output = bisect.orchestrate(
        repo=main_repo,
        good="good-ref",
        bad="bad-ref",
        runner=runner,
        result_path=result_path,
        session_parent=temp_root,
        keep_session=True,
    )
    return output, result_path, git_calls, main_repo, bisect_worktree["path"]


def test_real_bisect_commands_target_only_isolated_worktree_and_reset_in_finally(monkeypatch, scratch_dir):
    result, result_path, git_calls, main_repo, candidate = _orchestrator_fixture(monkeypatch, scratch_dir)

    bisect_calls = [(repo, args) for repo, args in git_calls if args and args[0] == "bisect"]
    assert [args[1] for _, args in bisect_calls] == ["start", "run", "log", "reset"]
    assert all(repo == candidate for repo, _ in bisect_calls)
    assert all(repo != main_repo for repo, _ in bisect_calls)
    assert result["status"] == "SUCCESS"
    assert result["reset"]["returncode"] == 0
    assert json.loads(result_path.read_text(encoding="utf-8"))["reset"]["returncode"] == 0
    assert result["frozen_inputs"]["runner"]["sha256"] == bisect.sha256_file(
        Path(result["frozen_inputs"]["runner"]["path"])
    )


def test_finally_resets_after_bisect_run_raises(monkeypatch, scratch_dir):
    result, _, git_calls, _, candidate = _orchestrator_fixture(monkeypatch, scratch_dir, run_raises=True)
    assert result["status"] == "ABORTED"
    assert result["reset"]["returncode"] == 0
    assert any(repo == candidate and args[:2] == ("bisect", "reset") for repo, args in git_calls)


def test_endpoint_runner_abort_aborts_without_starting_bisect(monkeypatch, scratch_dir):
    endpoint_values = [
        {"verdict": "UNAVAILABLE", "mapped_verdict": "ABORT"},
        {"verdict": "UNAVAILABLE", "mapped_verdict": "ABORT"},
        {"verdict": "BAD", "mapped_verdict": "BAD", "failure_signature": {"code": "X"}},
        {"verdict": "BAD", "mapped_verdict": "BAD", "failure_signature": {"code": "X"}},
    ]
    result, _, git_calls, _, _ = _orchestrator_fixture(
        monkeypatch,
        scratch_dir,
        endpoints=endpoint_values,
    )
    assert result["status"] == "ABORTED"
    assert result["reason"] == "TRIAL_RUNNER_ABORTED"
    assert not any(args[:2] == ("bisect", "start") for _, args in git_calls)


def test_unreproducible_endpoint_stays_unavailable_and_never_starts_bisect(monkeypatch, scratch_dir):
    endpoint_values = [
        {"verdict": "GOOD", "mapped_verdict": "GOOD"},
        {"verdict": "GOOD", "mapped_verdict": "GOOD"},
        {"verdict": "BAD", "mapped_verdict": "BAD", "failure_signature": {"code": "X"}},
        {"verdict": "SKIP", "mapped_verdict": "SKIP", "failure_signature": None},
    ]
    result, _, git_calls, _, _ = _orchestrator_fixture(
        monkeypatch,
        scratch_dir,
        endpoints=endpoint_values,
    )
    assert result["status"] == "UNAVAILABLE"
    assert result["reason"] == "BAD_ENDPOINT_NOT_REPRODUCIBLE"
    assert not any(args[:2] == ("bisect", "start") for _, args in git_calls)
