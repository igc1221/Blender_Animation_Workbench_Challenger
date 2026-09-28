"""Pure contracts for the configured IC4 evidence orchestrator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_rc2_h2_representation_boundary.py"
)
EXPECTED_NAMES = (
    "rc1-sliding-forearm-rotate",
    "phase4-e9-multilimb-sliding",
    "phase4-i12-runtime",
    "phase4-i15-runtime",
)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_spec = importlib.util.spec_from_file_location("rc2_h2c_runner", RUNNER_PATH)
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)

    targets = {
        name: {
            "timeout_seconds": 37 + index,
            "mutates_state": index == 0,
            "steps": [
                {"command": f"verifier-{index}-{step}", "args": ["--exact arg", name]}
                for step in range(2 if index == 0 else 1)
            ],
        }
        for index, name in enumerate(EXPECTED_NAMES)
    }
    targets_path = tmp_path / "verification_targets.json"
    targets_path.write_text(
        json.dumps({"schema": "awb-verification-targets/v1", "targets": targets}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "TARGETS_PATH", targets_path)
    monkeypatch.setattr(runner, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(runner, "_git_head", lambda: "fixed-test-head")
    return runner, targets


def test_ic4_runs_only_the_four_configured_targets_serially(harness, monkeypatch) -> None:
    runner, targets = harness
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "EXISTING_VERIFIER_OK\n", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.TARGET_NAMES == EXPECTED_NAMES
    assert runner.main() == 0

    expected_steps = [
        (name, step)
        for name in EXPECTED_NAMES
        for step in targets[name]["steps"]
    ]
    assert len(calls) == len(expected_steps)
    for (argv, kwargs), (name, step) in zip(calls, expected_steps, strict=True):
        assert argv == [step["command"], *step["args"]]
        assert kwargs["cwd"] == runner.ROOT
        assert kwargs["timeout"] == targets[name]["timeout_seconds"]

    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["schema"] == "awb-rc2-h2-representation-boundary-result/v1"
    assert receipt["status"] == "PASS"
    assert receipt["git_head"] == "fixed-test-head"
    assert receipt["failed_target"] is None
    assert receipt["total_ms"] >= 0
    assert [target["name"] for target in receipt["targets"]] == list(EXPECTED_NAMES)
    assert receipt["targets"][0]["executed_step_labels"] == ["1/2", "2/2"]
    assert not (runner.ROOT / "build" / "visual_development" / "verification_target.json").exists()


def test_ic4_stops_on_exact_failed_target_and_returncode(harness, monkeypatch) -> None:
    runner, targets = harness
    calls: list[list[str]] = []
    failed_step = targets[EXPECTED_NAMES[1]]["steps"][0]

    def fake_run(argv, **kwargs):
        calls.append(argv)
        code = 9 if argv[0] == failed_step["command"] else 0
        return subprocess.CompletedProcess(argv, code, "first failure detail\n", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.main() == 9
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["failed_target"] == EXPECTED_NAMES[1]
    assert [target["name"] for target in receipt["targets"]] == list(EXPECTED_NAMES[:2])
    failed = receipt["targets"][1]
    assert failed["returncode"] == 9
    assert failed["steps"][0]["returncode"] == 9
    assert failed["steps"][0]["stdout_summary"] == ["first failure detail"]
    assert len(calls) == 3
