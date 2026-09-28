"""Contract checks for the IC6 evidence-only integrated runner."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_rc2_ic6_integrated_candidate.py"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verification_targets.json"
STATIC_TESTS = (
    "tests/test_phase4_operation_plan.py",
    "tests/test_phase4_mutation_journal.py",
    "tests/test_e11_ak4_sliding_auto.py",
    "tests/test_rc2_h2_representation_ownership.py",
    "tests/test_rc2_h2b_projection_overlay_contract.py",
    "tests/test_rc2_w3_replay_payload.py",
    "tests/test_rc2_h3_replay_orchestration_contract.py",
    "tests/test_e10_deterministic_replay_modes.py",
)
NAMES = (
    "rc2-p1-characterization",
    "rc1-sliding-forearm-rotate",
    "phase4-e9-multilimb-sliding",
    "phase4-i12-runtime",
    "phase4-i15-runtime",
    "rc2-replay-characterization",
)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("ic6_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    targets = {
        name: {
            "timeout_seconds": 31 + index,
            "mutates_state": index == 1,
            "steps": [
                {"command": f"configured-{index}-{step}", "args": ["--exact arg", name]}
                for step in range(2 if index == 1 else 1)
            ],
        }
        for index, name in enumerate(NAMES)
    }
    path = tmp_path / "verification_targets.json"
    path.write_text(json.dumps({"schema": "awb-verification-targets/v1", "targets": targets}), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "TARGETS_PATH", path)
    monkeypatch.setattr(runner, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(runner, "_git_head", lambda: "fixed-head")
    return runner, targets, path


def _fake_run(calls: list, failure: str | None = None, code: int = 0):
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        kwargs["stdout"].write(("FAIL\n" if argv[0] == failure else "GATE_OK\n").encode())
        return subprocess.CompletedProcess(argv, code if argv[0] == failure else 0)
    return run


def test_exact_static_order_and_serial_configured_commands(harness, monkeypatch, capsys) -> None:
    runner, targets, path = harness
    original = path.read_bytes()
    calls: list = []
    monkeypatch.setattr(runner.subprocess, "run", _fake_run(calls))
    assert runner.STATIC_TESTS == STATIC_TESTS
    assert runner.TARGET_NAMES == NAMES
    assert runner.main() == 0
    expected = [(name, step) for name in NAMES for step in targets[name]["steps"]]
    project_python = str(runner.ROOT / ".venv" / "Scripts" / "python.exe")
    assert calls[0][0] == [project_python, "-m", "pytest", "-q", *STATIC_TESTS]
    assert [argv for argv, _ in calls[1:]] == [[step["command"], *step["args"]] for _, step in expected]
    assert [kw["timeout"] for _, kw in calls[1:]] == [targets[name]["timeout_seconds"] for name, _ in expected]
    assert all(kw["cwd"] == runner.ROOT for _, kw in calls)
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["schema"] == "awb-rc2-ic6-integrated-candidate-result/v1"
    assert receipt["status"] == "PASS" and receipt["git_head"] == "fixed-head"
    assert receipt["static_gate"]["tests"] == list(STATIC_TESTS)
    assert receipt["static_gate"]["status"] == "PASS"
    assert [target["name"] for target in receipt["runtime_targets"]] == list(NAMES)
    assert [target["mutates_state"] for target in receipt["runtime_targets"]] == [False, True, False, False, False, False]
    assert receipt["runtime_targets"][1]["executed_step_labels"] == ["1/2", "2/2"]
    assert receipt["runtime_targets"][1]["executed_step_count"] == 2
    assert receipt["failed_stage"] is None and receipt["failed_target"] is None
    assert receipt["i19_excluded"] is True
    assert "does not decide" in receipt["frozen_contract_note"]
    assert receipt["total_ms"] >= 0
    assert "RC2_IC6_PASS" in capsys.readouterr().out
    assert path.read_bytes() == original
    assert not (runner.ROOT / "build" / "visual_development" / "verification_target.json").exists()


def test_static_failure_skips_all_runtime(harness, monkeypatch, capsys) -> None:
    runner, _, _ = harness
    calls: list = []
    project_python = str(runner.ROOT / ".venv" / "Scripts" / "python.exe")
    monkeypatch.setattr(runner.subprocess, "run", _fake_run(calls, project_python, 7))
    assert runner.main() == 7
    assert len(calls) == 1
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL" and receipt["failed_stage"] == "static_gate"
    assert receipt["failed_target"] is None
    assert receipt["static_gate"]["returncode"] == 7
    assert all(item["status"] == "NOT_RUN" for item in receipt["runtime_targets"])
    assert "RC2_IC6_STATIC_FAIL 7" in capsys.readouterr().out


@pytest.mark.parametrize("failed_name,step_index", [(NAMES[0], 0), (NAMES[1], 0), (NAMES[1], 1), (NAMES[-1], 0)])
def test_first_runtime_failure_skips_later_steps_and_targets(harness, monkeypatch, capsys, failed_name, step_index) -> None:
    runner, targets, _ = harness
    calls: list = []
    failure = targets[failed_name]["steps"][step_index]["command"]
    monkeypatch.setattr(runner.subprocess, "run", _fake_run(calls, failure, 13))
    assert runner.main() == 13
    expected = [step for name in NAMES[:NAMES.index(failed_name) + 1] for step in targets[name]["steps"]]
    expected = expected[:next(i for i, step in enumerate(expected) if step["command"] == failure) + 1]
    assert [argv for argv, _ in calls[1:]] == [[step["command"], *step["args"]] for step in expected]
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["failed_stage"] == "runtime_target" and receipt["failed_target"] == failed_name
    failed = receipt["runtime_targets"][NAMES.index(failed_name)]
    assert failed["status"] == "FAIL" and failed["returncode"] == 13
    assert failed["failed_step"] == f"{step_index + 1}/{len(targets[failed_name]['steps'])}"
    assert all(item["status"] == "NOT_RUN" for item in receipt["runtime_targets"][NAMES.index(failed_name) + 1:])
    assert f"RC2_IC6_RUNTIME_FAIL {failed_name} 13" in capsys.readouterr().out


def test_invalid_schema_and_invalid_target_fail_before_static(harness, monkeypatch) -> None:
    runner, targets, path = harness
    calls: list = []
    monkeypatch.setattr(runner.subprocess, "run", _fake_run(calls))
    for payload in (
        {"schema": "other", "targets": targets},
        {"schema": "awb-verification-targets/v1", "targets": {**targets, NAMES[0]: {**targets[NAMES[0]], "steps": []}}},
        {"schema": "awb-verification-targets/v1", "targets": {**targets, NAMES[0]: {**targets[NAMES[0]], "steps": [{"command": "x", "args": ["verify_i19"]}]}}},
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert runner.main() == 2
        receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
        assert receipt["failed_stage"] == "configuration" and receipt["static_gate"]["status"] == "NOT_RUN"
    assert not calls


def test_real_config_is_exact_and_i19_excluded() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "awb-verification-targets/v1"
    for name in NAMES:
        spec = payload["targets"][name]
        assert spec["steps"]
        assert all("i19" not in " ".join([step["command"], *step["args"]]).lower() for step in spec["steps"])


def test_bounded_stdout_and_stderr_even_for_large_logs(harness, monkeypatch) -> None:
    runner, _, _ = harness
    def noisy(argv, **kwargs):
        kwargs["stdout"].write(b"x" * 250_000 + b"\nTAIL_OK\n")
        kwargs["stderr"].write(b"y" * 250_000 + b"\nTAIL_ERR\n")
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(runner.subprocess, "run", noisy)
    assert runner.main() == 0
    receipt = json.loads(runner.RESULT_PATH.read_text(encoding="utf-8"))
    assert receipt["static_gate"]["stdout_summary"] == ["TAIL_OK"]
    assert receipt["runtime_targets"][0]["stderr_summary"] == ["TAIL_ERR"]
    assert len(runner.RESULT_PATH.read_bytes()) < 30_000


def test_imports_only_standard_library_and_no_product_modules() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    assert imports <= {"__future__", "json", "subprocess", "tempfile", "time", "pathlib"}
