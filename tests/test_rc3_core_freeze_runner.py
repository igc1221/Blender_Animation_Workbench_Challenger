"""Contract checks for the RC3 Core Freeze evidence runner."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_rc3_core_freeze_candidate.py"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verification_targets.json"
NAMES = (
    "rc2-p1-characterization",
    "phase4-i13-runtime",
    "phase4-i14-runtime",
    "rc1-sliding-forearm-rotate",
    "phase4-i12-runtime",
    "phase4-e9-multilimb-sliding",
    "phase4-i20-multilimb-contact",
    "phase4-e11-sliding-auto",
    "phase4-i16-runtime",
    "rc2-replay-characterization",
    "phase4-i15-runtime",
)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("rc3_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    targets = {
        name: {
            "timeout_seconds": 31 + index,
            "mutates_state": index == 3,
            "steps": [
                {"command": f"configured-{index}-{step}", "args": ["--exact arg", name]}
                for step in range(2 if index == 3 else 1)
            ],
        }
        for index, name in enumerate(NAMES)
    }
    path = tmp_path / "verification_targets.json"
    path.write_text(
        json.dumps({"schema": "awb-verification-targets/v1", "targets": targets}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "TARGETS_PATH", path)
    monkeypatch.setattr(runner, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(runner, "_git_head", lambda: "fixed-head")
    return runner, targets, path


def _kind(argv: list[str]) -> str:
    joined = " ".join(argv)
    if " -m pytest " in f" {joined} ":
        return "pytest"
    if " -m ruff " in f" {joined} ":
        return "ruff"
    return argv[0]


def _fake_run(calls: list, *, fail_kind: str | None = None, code: int = 0):
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        kind = _kind(argv)
        failed = kind == fail_kind
        kwargs["stdout"].write(("FAIL\n" if failed else "GATE_OK\n").encode())
        return subprocess.CompletedProcess(argv, code if failed else 0)

    return run


def test_exact_static_and_runtime_order(harness, monkeypatch, capsys) -> None:
    runner, targets, path = harness
    original = path.read_bytes()
    calls: list = []
    monkeypatch.setattr(runner.subprocess, "run", _fake_run(calls=calls))
    assert runner.TARGET_NAMES == NAMES
    assert runner.main() == 0

    project_python = str(runner.ROOT / ".venv" / "Scripts" / "python.exe")
    assert calls[0][0] == [project_python, "-m", "pytest", "-q"]
    assert calls[1][0] == [project_python, "-m", "ruff", "check", "extension", "tests"]

    expected = [(name, step) for name in NAMES for step in targets[name]["steps"]]
    assert [argv for argv, _ in calls[2:]] == [
        [step["command"], *step["args"]] for _, step in expected
    ]
    assert [kw["timeout"] for _, kw in calls[2:]] == [
        targets[name]["timeout_seconds"] for name, _ in expected
    ]
    assert all(kw["cwd"] == runner.ROOT for _, kw in calls)

    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["schema"] == "awb-rc3-core-freeze-candidate-result/v1"
    assert receipt["status"] == "PASS"
    assert receipt["git_head"] == "fixed-head"
    assert receipt["static_gate"]["status"] == "PASS"
    assert receipt["static_gate"]["pytest"]["status"] == "PASS"
    assert receipt["static_gate"]["ruff"]["status"] == "PASS"
    assert [item["name"] for item in receipt["runtime_targets"]] == list(NAMES)
    assert receipt["runtime_targets"][3]["executed_step_labels"] == ["1/2", "2/2"]
    assert receipt["i19_excluded"] is True
    assert receipt["user_acceptance_required"] is True
    assert "USER" in receipt["frozen_contract_note"]
    assert "RC3_CORE_FREEZE_AUTOMATION_PASS" in capsys.readouterr().out
    assert path.read_bytes() == original


def test_pytest_failure_skips_ruff_and_runtime(harness, monkeypatch) -> None:
    runner, _, _ = harness
    calls: list = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_run(calls=calls, fail_kind="pytest", code=7),
    )
    assert runner.main() == 7
    assert len(calls) == 1
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["failed_stage"] == "static_pytest"
    assert receipt["static_gate"]["status"] == "FAIL"
    assert receipt["static_gate"]["ruff"]["status"] == "NOT_RUN"
    assert all(item["status"] == "NOT_RUN" for item in receipt["runtime_targets"])


def test_ruff_failure_skips_runtime(harness, monkeypatch) -> None:
    runner, _, _ = harness
    calls: list = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_run(calls=calls, fail_kind="ruff", code=8),
    )
    assert runner.main() == 8
    assert len(calls) == 2
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["failed_stage"] == "static_ruff"
    assert receipt["static_gate"]["pytest"]["status"] == "PASS"
    assert receipt["static_gate"]["ruff"]["returncode"] == 8
    assert all(item["status"] == "NOT_RUN" for item in receipt["runtime_targets"])


@pytest.mark.parametrize(
    "failed_name,step_index",
    [(NAMES[0], 0), (NAMES[3], 0), (NAMES[3], 1), (NAMES[-1], 0)],
)
def test_first_runtime_failure_stops_later_work(
    harness,
    monkeypatch,
    capsys,
    failed_name,
    step_index,
) -> None:
    runner, targets, _ = harness
    calls: list = []
    failure = targets[failed_name]["steps"][step_index]["command"]
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_run(calls=calls, fail_kind=failure, code=13),
    )
    assert runner.main() == 13
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["failed_stage"] == "runtime_target"
    assert receipt["failed_target"] == failed_name
    failed = receipt["runtime_targets"][NAMES.index(failed_name)]
    assert failed["status"] == "FAIL"
    assert failed["returncode"] == 13
    assert failed["failed_step"] == f"{step_index + 1}/{len(targets[failed_name]['steps'])}"
    assert all(
        item["status"] == "NOT_RUN"
        for item in receipt["runtime_targets"][NAMES.index(failed_name) + 1 :]
    )
    assert f"RC3_RUNTIME_FAIL {failed_name} 13" in capsys.readouterr().out


def test_invalid_config_and_i19_fail_before_static(harness, monkeypatch) -> None:
    runner, targets, path = harness
    calls: list = []
    monkeypatch.setattr(runner.subprocess, "run", _fake_run(calls=calls))
    bad = [
        {"schema": "other", "targets": targets},
        {
            "schema": "awb-verification-targets/v1",
            "targets": {**targets, NAMES[0]: {**targets[NAMES[0]], "steps": []}},
        },
        {
            "schema": "awb-verification-targets/v1",
            "targets": {
                **targets,
                NAMES[0]: {
                    **targets[NAMES[0]],
                    "steps": [{"command": "x", "args": ["verify_i19"]}],
                },
            },
        },
    ]
    for payload in bad:
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert runner.main() == 2
        receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
        assert receipt["failed_stage"] == "configuration"
        assert receipt["static_gate"]["status"] == "NOT_RUN"
    assert not calls


def test_real_config_has_exact_targets_and_excludes_i19() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "awb-verification-targets/v1"
    for name in NAMES:
        spec = payload["targets"][name]
        assert spec["steps"]
        for step in spec["steps"]:
            joined = " ".join([name, step["command"], *step["args"]]).lower()
            assert "i19" not in joined


def test_bounded_logs(harness, monkeypatch) -> None:
    runner, _, _ = harness

    def noisy(argv, **kwargs):
        kwargs["stdout"].write(b"x" * 250_000 + b"\nTAIL_OK\n")
        kwargs["stderr"].write(b"y" * 250_000 + b"\nTAIL_ERR\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(runner.subprocess, "run", noisy)
    assert runner.main() == 0
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["static_gate"]["pytest"]["stdout_summary"] == ["TAIL_OK"]
    assert receipt["runtime_targets"][0]["stderr_summary"] == ["TAIL_ERR"]
    assert len(runner.RESULT_PATH.read_bytes()) < 40_000


def test_runner_imports_only_standard_library() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert imports <= {
        "__future__",
        "json",
        "subprocess",
        "tempfile",
        "time",
        "pathlib",
    }
