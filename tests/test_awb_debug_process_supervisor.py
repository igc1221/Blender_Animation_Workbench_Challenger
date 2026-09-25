from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from scripts import awb_debug_process_supervisor as supervisor
from scripts.awb_debug_supervisor_child_fixture import child_argv
from scripts.awb_native_incident_evidence import ProcessCpuTimeSample
from scripts.awb_supervisor_core import IncidentClassification as Classification

ROOT = Path(__file__).resolve().parents[1]
INCIDENT_ID = "INC-20260926-101500-0a1b2c3d"
TEST_FILE_PREFIX = f"awb-process-supervisor-test-{uuid4().hex}-"


@pytest.fixture
def tmp_path():
    work = ROOT / "build"
    try:
        yield work
    finally:
        for artifact in work.glob(f"{TEST_FILE_PREFIX}*"):
            if artifact.is_file():
                artifact.unlink()


def _artifact(root: Path, name: str) -> Path:
    return root / f"{TEST_FILE_PREFIX}{name}"


def _run(tmp_path: Path, argv: list[str], **kwargs):
    return supervisor.run_supervised_process(
        argv,
        stdout_path=_artifact(tmp_path, "child.stdout.bin"),
        stderr_path=_artifact(tmp_path, "child.stderr.bin"),
        cwd=ROOT,
        sample_cpu_time=False,
        **kwargs,
    )


def _canonical_heartbeat_child(path: Path, *, incident_id: str, sequence: int = 1) -> list[str]:
    source = (
        "import os, sys, time\n"
        "from datetime import UTC, datetime\n"
        "from scripts.awb_supervisor_heartbeat import make_heartbeat, write_heartbeat\n"
        f"write_heartbeat({str(path)!r}, make_heartbeat(incident_id={incident_id!r}, "
        f"pid=os.getpid(), sequence={sequence}, now_utc=datetime(2099, 1, 1, tzinfo=UTC)))\n"
        "time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0)\n"
    )
    executable = getattr(sys, "_base_executable", sys.executable)
    return [executable, "-c", source]


def test_clean_and_high_volume_children_use_regular_file_redirection(tmp_path):
    line_count = 25000
    line_size = 96
    outcome = _run(
        tmp_path,
        child_argv(
            "high_volume",
            line_count=line_count,
            line_size=line_size,
            summary_path=str(_artifact(tmp_path, "high-volume-summary.json")),
        ),
        timeout_seconds=30,
    )

    assert outcome.classification is Classification.CLEAN_EXIT
    assert outcome.stdout_path.stat().st_size == line_count * line_size
    assert outcome.stderr_path.read_bytes() == b""
    assert outcome.document["lifecycle"]["exit_code"] == 0
    assert outcome.document["schema"] == "awb-debug-supervisor-result/v1"


def test_python_failure_requires_explicit_fixture_receipt(tmp_path):
    receipt = _artifact(tmp_path, "fixture-summary.json")
    argv = child_argv("exception", summary_path=str(receipt))
    implicit = _run(tmp_path / "implicit", argv)
    assert implicit.classification is Classification.UNKNOWN

    def fixture_evidence():
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if payload.get("mode") == "exception" and payload.get("exit_kind") == "UNRAISED":
            return supervisor.PythonFailureEvidence(
                source="FIXTURE_RECEIPT", detail="fixture summary records an unraised exception"
            )
        return None

    explicit = _run(
        tmp_path / "explicit",
        argv,
        python_failure_evidence_provider=fixture_evidence,
    )
    assert explicit.classification is Classification.PYTHON_FAILURE
    assert "Traceback" in explicit.stderr_path.read_text(encoding="utf-8", errors="replace")


def test_native_classification_uses_only_allowlisted_windows_statuses(tmp_path):
    assert supervisor._known_native_ntstatus(0xC0000005, windows=True) == 0xC0000005
    assert supervisor._known_native_ntstatus(-1073741819, windows=True) == 0xC0000005
    assert supervisor._known_native_ntstatus(0xC0000006, windows=True) is None
    assert supervisor._known_native_ntstatus(0xC0000005, windows=False) is None
    assert supervisor.WINDOWS_NATIVE_NTSTATUS_ALLOWLIST == {0xC0000005, 0xC0000409}

    if os.name != "nt":
        pytest.skip("native fixture preserves exact NTSTATUS exit codes only on Windows")
    outcome = _run(tmp_path, child_argv("native_exit"))
    assert outcome.classification is Classification.NATIVE_CRASH
    assert outcome.document["lifecycle"]["exception_code"] == "0xC0000005"


def test_canonical_heartbeat_is_monitored_using_supervisor_local_age(tmp_path):
    heartbeat_path = _artifact(tmp_path, "child.heartbeat.json")
    outcome = _run(
        tmp_path,
        _canonical_heartbeat_child(heartbeat_path, incident_id=INCIDENT_ID),
        heartbeat_path=heartbeat_path,
        incident_id=INCIDENT_ID,
        heartbeat_stall_timeout_seconds=2,
    )

    heartbeat = outcome.document["heartbeat"]
    assert outcome.classification is Classification.CLEAN_EXIT
    assert heartbeat["status"] == "FRESH"
    assert heartbeat["last_heartbeat_utc"].startswith("2099-")
    assert heartbeat["beats_observed"] == 1
    assert heartbeat["age_seconds"] is not None


@pytest.mark.parametrize("invalid_kind", ["malformed", "mismatched"])
def test_malformed_or_mismatched_heartbeat_is_incomplete_and_not_clean(tmp_path, invalid_kind):
    heartbeat_path = _artifact(tmp_path, f"{invalid_kind}.heartbeat.json")
    if invalid_kind == "malformed":
        source = f"from pathlib import Path; Path({str(heartbeat_path)!r}).write_text('{{bad')"
    else:
        argv = _canonical_heartbeat_child(heartbeat_path, incident_id="another-incident")
        outcome = _run(
            tmp_path,
            argv,
            heartbeat_path=heartbeat_path,
            incident_id=INCIDENT_ID,
        )
        assert outcome.classification is not Classification.CLEAN_EXIT
        assert not outcome.evidence_complete
        assert outcome.document["evidence"]["status"] == "PARTIAL"
        assert outcome.document["heartbeat"]["status"] == "UNAVAILABLE"
        return
    outcome = _run(
        tmp_path,
        [sys.executable, "-c", source],
        heartbeat_path=heartbeat_path,
        incident_id=INCIDENT_ID,
    )
    assert outcome.classification is not Classification.CLEAN_EXIT
    assert not outcome.evidence_complete
    assert outcome.document["evidence"]["status"] == "PARTIAL"
    assert outcome.document["heartbeat"]["status"] == "UNAVAILABLE"


def test_stale_canonical_heartbeat_causes_explicit_supervisor_termination(tmp_path):
    heartbeat_path = _artifact(tmp_path, "stall.heartbeat.json")
    argv = _canonical_heartbeat_child(heartbeat_path, incident_id=INCIDENT_ID)
    argv.append("10")
    outcome = _run(
        tmp_path,
        argv,
        heartbeat_path=heartbeat_path,
        incident_id=INCIDENT_ID,
        heartbeat_stall_timeout_seconds=0.2,
        timeout_seconds=5,
        poll_interval_seconds=0.02,
        termination_grace_seconds=0.1,
    )

    termination = outcome.document["termination"]["supervisor_initiated_termination"]
    assert outcome.classification is Classification.HEARTBEAT_STALL
    assert termination["initiated"] is True
    assert termination["initiated_by"] == "SUPERVISOR"
    assert termination["reason_code"] == "HEARTBEAT_TIMEOUT"
    assert termination["requested_utc"]


def test_cpu_delta_requires_same_pid_and_process_creation_token(tmp_path, monkeypatch):
    calls = 0

    def samples(pid: int) -> ProcessCpuTimeSample:
        nonlocal calls
        calls += 1
        creation_token = 101 if calls == 1 else 202
        return ProcessCpuTimeSample(
            status="AVAILABLE",
            pid=pid,
            user_seconds=float(calls),
            kernel_seconds=0.0,
            total_seconds=float(calls),
            creation_time_100ns=creation_token,
        )

    monkeypatch.setattr(supervisor, "sample_process_cpu_time", samples)
    outcome = supervisor.run_supervised_process(
        [sys.executable, "-c", "import time; time.sleep(.15)"],
        stdout_path=_artifact(tmp_path, "cpu.stdout"),
        stderr_path=_artifact(tmp_path, "cpu.stderr"),
        cwd=ROOT,
        sample_cpu_time=True,
        cpu_sample_interval_seconds=0.02,
        poll_interval_seconds=0.01,
    )

    assert len(outcome.cpu_time.samples) >= 2
    assert outcome.cpu_time.status == "INCOMPLETE"
    assert outcome.cpu_time.delta_seconds is None


def test_job_object_is_optional_and_result_writer_is_atomic(tmp_path):
    outcome = _run(
        tmp_path,
        child_argv("clean"),
        use_windows_job_object=True,
    )
    result_path = _artifact(tmp_path, "supervisor_result.json")
    result_path.write_text("old result\n", encoding="utf-8")
    supervisor.write_supervisor_result(result_path, outcome)

    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written == outcome.document
    assert written["schema"] == supervisor.SUPERVISOR_RESULT_SCHEMA
    assert not list(tmp_path.glob(f".{TEST_FILE_PREFIX}supervisor_result.json.*.tmp"))
    assert outcome.job_object_status in {"UNSUPPORTED", "ASSIGNED"} or (
        outcome.job_object_status.startswith(("ERROR", "ASSIGNMENT_ERROR", "CLOSE_ERROR"))
    )


def test_result_writer_rejects_bad_document_without_replacing_previous_file(tmp_path):
    outcome = _run(tmp_path, child_argv("clean"))
    result_path = _artifact(tmp_path, "supervisor_result.json")
    result_path.write_text("preserve me\n", encoding="utf-8")
    invalid = dict(outcome.document)
    invalid["unexpected"] = True

    with pytest.raises(ValueError, match="top-level"):
        supervisor.write_supervisor_result(result_path, invalid)
    assert result_path.read_text(encoding="utf-8") == "preserve me\n"
