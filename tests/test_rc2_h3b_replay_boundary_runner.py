"""Pure contract tests for the IC5 replay evidence runner."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_rc2_h3_replay_boundary.py"
)
STATIC_TESTS = (
    "tests/test_e10_deterministic_replay_modes.py",
    "tests/test_rc2_w3_replay_payload.py",
    "tests/test_rc2_h3_replay_orchestration_contract.py",
)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_spec = importlib.util.spec_from_file_location("rc2_h3b_runner", RUNNER_PATH)
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)

    steps = [
        {"command": "configured-first", "args": ["--literal $env:value", "one"]},
        {"command": "configured-second", "args": ["-NoProfile", "two"]},
    ]
    targets_path = tmp_path / "verification_targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "schema": "awb-verification-targets/v1",
                "targets": {
                    "rc2-replay-characterization": {
                        "timeout_seconds": 41,
                        "mutates_state": False,
                        "steps": steps,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "TARGETS_PATH", targets_path)
    monkeypatch.setattr(runner, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(runner, "_git_head", lambda: "fixed-test-head")
    return runner, steps


def test_h3b_static_failure_never_launches_configured_blender(harness, monkeypatch) -> None:
    runner, _steps = harness
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 7, "static failure detail\n", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.main() == 7
    assert len(calls) == 1
    assert calls[0][1:] == ["-m", "pytest", "-q", *STATIC_TESTS]

    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["schema"] == "awb-rc2-h3-replay-boundary-result/v1"
    assert receipt["status"] == "FAIL"
    assert receipt["failed_stage"] == "static_gate"
    assert receipt["static_gate"]["returncode"] == 7
    assert receipt["static_gate"]["tests"] == list(STATIC_TESTS)
    assert receipt["runtime_target"]["status"] == "NOT_RUN"
    assert receipt["runtime_target"]["executed_step_labels"] == []


def test_h3b_pass_uses_exact_configured_steps_in_order(harness, monkeypatch, capsys) -> None:
    runner, steps = harness
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "EXISTING_GATE_OK\n", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.main() == 0
    assert calls[0][0][1:] == ["-m", "pytest", "-q", *STATIC_TESTS]
    assert [argv for argv, _kwargs in calls[1:]] == [
        [step["command"], *step["args"]] for step in steps
    ]
    assert all(kwargs["cwd"] == runner.ROOT for _argv, kwargs in calls)
    assert [kwargs["timeout"] for _argv, kwargs in calls[1:]] == [41.0, 41.0]

    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["git_head"] == "fixed-test-head"
    assert receipt["failed_stage"] is None
    assert receipt["static_gate"]["status"] == "PASS"
    runtime = receipt["runtime_target"]
    assert runtime["name"] == "rc2-replay-characterization"
    assert runtime["status"] == "PASS"
    assert runtime["mutates_state"] is False
    assert runtime["executed_step_labels"] == ["1/2", "2/2"]
    assert runtime["stdout_summary"] == ["EXISTING_GATE_OK"]
    assert receipt["total_ms"] >= 0
    assert "RC2_H3_IC5_PASS" in capsys.readouterr().out
    assert not (runner.ROOT / "build" / "visual_development" / "verification_target.json").exists()


def test_h3b_runtime_failure_records_first_step_and_stops(harness, monkeypatch) -> None:
    runner, steps = harness
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        code = 13 if argv[0] == steps[0]["command"] else 0
        return subprocess.CompletedProcess(argv, code, "first runtime failure\n", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.main() == 13
    assert len(calls) == 2
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["failed_stage"] == "runtime_target"
    assert receipt["runtime_target"]["returncode"] == 13
    assert receipt["runtime_target"]["failed_step"] == "1/2"
    assert receipt["runtime_target"]["executed_step_labels"] == ["1/2"]
    assert receipt["runtime_target"]["stdout_summary"] == ["first runtime failure"]


def test_h3b_runner_imports_no_product_or_shared_request_owner() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
    assert imports <= {"__future__", "json", "subprocess", "sys", "time", "pathlib"}
    assert "run_configured_verification" not in source
    assert "visual_development" not in source
