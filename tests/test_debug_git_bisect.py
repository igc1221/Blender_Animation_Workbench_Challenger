from __future__ import annotations

import json
import shutil
import subprocess
import sys
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
    path = Path.cwd() / "build" / f"bb9-git-bisect-test-{uuid4().hex}"
    path.mkdir(parents=True)
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
            {"stage": "confirmation", "commit_sha": FIRST_BAD, "verdict": "BAD"},
            {
                "stage": "confirmation",
                "commit_sha": PARENT,
                "verdict": "SKIP",
                "reason": "REPLAY_UNAVAILABLE",
            },
        ],
    )
    assert result["status"] == "AMBIGUOUS"
    assert result["reason"] == "DIRECT_PARENT_NOT_DIRECTLY_TRIALED_GOOD"
    assert result["unresolved_parents"] == [{"commit": PARENT, "verdict": "SKIP"}]


def test_first_bad_parser_rejects_conflicting_git_output():
    assert bisect.parse_first_bad_commit(f"{FIRST_BAD} is the first bad commit") == FIRST_BAD
    assert bisect.parse_first_bad_commit(f"{FIRST_BAD} is the first 'bad' commit") == FIRST_BAD
    assert bisect.parse_first_bad_commit(
        f"{FIRST_BAD} is the first bad commit",
        f"# first bad commit: [{BAD}] message",
    ) is None
    assert bisect.parse_first_bad_commit(
        f"{FIRST_BAD} is the first 'bad' commit",
        f"# first 'bad' commit: [{BAD}] message",
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
        bootstrap=scratch_dir / "frozen" / "bootstrap.py",
        replay=scratch_dir / "frozen" / "replay.json",
        baseline_master=scratch_dir / "frozen" / "baseline.blend",
        expected_baseline_sha256="0" * 64,
        blender=scratch_dir / "Blender" / "blender.exe",
    )
    assert command[1] == "-I"
    assert Path(command[2]).is_absolute()
    assert command[command.index("--runner") + 1] == str((scratch_dir / "frozen" / "runner.py").resolve())


def test_frozen_driver_and_runner_protocol_round_trip_without_mocks(scratch_dir):
    candidate = scratch_dir / "candidate"
    candidate.mkdir()
    subprocess.run(["git", "-C", str(candidate), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(candidate), "config", "user.email", "bb9-test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(candidate), "config", "user.name", "BB9 Test"], check=True)
    (candidate / "state.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(candidate), "add", "state.txt"], check=True)
    subprocess.run(["git", "-C", str(candidate), "commit", "-q", "-m", "clean"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    session = scratch_dir / "session"
    session.mkdir()
    driver = session / "trial_driver.py"
    driver.write_text(bisect._driver_source(), encoding="utf-8")
    runner = session / "runner.py"
    runner.write_text(
        """from __future__ import annotations
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--candidate-root", required=True)
p.add_argument("--expected-commit", required=True)
p.add_argument("--trial-dir", required=True)
p.add_argument("--replay", required=True)
p.add_argument("--baseline-master", required=True)
p.add_argument("--expected-baseline-sha256", required=True)
p.add_argument("--blender", required=True)
p.add_argument("--bootstrap", required=True)
p.add_argument("--foreground", action="store_true")
p.add_argument("--event-simulation", action="store_true")
args = p.parse_args()
trial_dir = Path(args.trial_dir)
trial_dir.mkdir(parents=True, exist_ok=False)
(trial_dir / "receipt.json").write_text(
    json.dumps({
        "schema": "awb-debug-git-bisect-trial/v1",
        "commit_sha": args.expected_commit,
        "verdict": "GOOD",
        "reason": "COMPLETE_NO_VIOLATION",
    }),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    bootstrap = session / "bootstrap.py"
    bootstrap.write_text("# bootstrap\n", encoding="utf-8")
    replay = session / "replay.json"
    replay.write_text("{}", encoding="utf-8")
    baseline = session / "baseline.blend"
    baseline.write_bytes(b"baseline")
    blender = session / "blender.exe"
    blender.write_bytes(b"stub")
    expected_sha = bisect.sha256_file(baseline)

    command = bisect._trial_driver_command(
        driver=driver,
        candidate=candidate,
        session_dir=session,
        runner=runner,
        bootstrap=bootstrap,
        replay=replay,
        baseline_master=baseline,
        expected_baseline_sha256=expected_sha,
        blender=blender,
        trial_id="endpoint-good-1",
    )
    completed = subprocess.run(
        command,
        cwd=candidate,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metadata = json.loads((session / "trials" / "endpoint-good-1.meta.json").read_text(encoding="utf-8"))
    assert metadata["commit_sha"] == commit
    assert metadata["verdict"] == "GOOD"
    assert metadata["stage"] == "endpoint"


def _orchestrator_fixture(monkeypatch, scratch_dir, *, run_raises=False, endpoints=None):
    main_repo = scratch_dir / "main"
    main_repo.mkdir()
    temp_root = scratch_dir / "isolated"
    temp_root.mkdir()
    runner = scratch_dir / "trial_runner.py"
    runner.write_text("# frozen runner input\n", encoding="utf-8")
    bootstrap = scratch_dir / "bootstrap.py"
    bootstrap.write_text("# frozen bootstrap\n", encoding="utf-8")
    replay = scratch_dir / "replay.json"
    replay.write_text(json.dumps({"recommended_path": "SEMANTIC"}), encoding="utf-8")
    baseline = scratch_dir / "baseline.blend"
    baseline.write_bytes(b"golden baseline")
    blender = scratch_dir / "blender.exe"
    blender.write_bytes(b"blender stub")
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
    records = [
        {
            "schema": bisect.TRIAL_RECORD_SCHEMA,
            "stage": "bisect",
            "commit_sha": PARENT,
            "verdict": "GOOD",
        },
        {
            "schema": bisect.TRIAL_RECORD_SCHEMA,
            "stage": "bisect",
            "commit_sha": FIRST_BAD,
            "verdict": "BAD",
        },
    ]

    def fake_endpoint_trial(*, endpoint, sha, attempt, trial_id_override=None, **kwargs):
        if trial_id_override:
            verdict = "BAD" if endpoint == "CONFIRM_BAD" else "GOOD"
            item = {
                "schema": bisect.TRIAL_RECORD_SCHEMA,
                "stage": "confirmation",
                "commit_sha": sha,
                "verdict": verdict,
                "mapped_verdict": verdict,
                "trial_id": trial_id_override,
                "failure_signature": {"code": "X"} if verdict == "BAD" else None,
            }
            records.append(item)
            return {**item, "endpoint": endpoint}
        item = dict(endpoint_values[endpoint_index["value"]])
        endpoint_index["value"] += 1
        return {**item, "endpoint": endpoint, "commit_sha": sha, "trial_id": f"endpoint-{endpoint}-{attempt}"}

    monkeypatch.setattr(bisect, "_invoke_endpoint_trial", fake_endpoint_trial)
    monkeypatch.setattr(bisect, "_read_trial_records", lambda session_dir: list(records))
    output = bisect.orchestrate(
        repo=main_repo,
        good="good-ref",
        bad="bad-ref",
        runner=runner,
        bootstrap=bootstrap,
        replay=replay,
        baseline_master=baseline,
        expected_baseline_sha256=bisect.sha256_file(baseline),
        blender=blender,
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
    run_args = next(args for _, args in bisect_calls if args[:2] == ("bisect", "run"))
    assert run_args[2] == sys.executable
    assert run_args[3] == "-I"
    assert len(run_args) > 8
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
