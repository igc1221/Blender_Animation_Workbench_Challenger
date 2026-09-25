from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import awb_debug_process_supervisor as process_supervisor
from scripts import awb_supervisor_heartbeat as raw_heartbeat
from scripts.awb_debug_supervisor_child_fixture import child_argv
from tests.test_debug_native_escalation_schemas import SchemaValidator

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_SCHEMA = "AWB_DEBUG_SUPERVISOR_RESULT_V1.schema.json"
RAW_SCHEMA = "AWB_SUPERVISOR_HEARTBEAT_V1.schema.json"
INCIDENT_ID = "INC-20260926-101500-0a1b2c3d"
PREFIX = "bb10-schema-parity-"


@pytest.fixture
def parity_dir():
    build = ROOT / "build"
    build.mkdir(parents=True, exist_ok=True)
    try:
        yield build
    finally:
        for artifact in build.glob(f"{PREFIX}*"):
            if artifact.is_file():
                artifact.unlink()


def _run(root: Path, name: str, argv: list[str], **kwargs: Any):
    return process_supervisor.run_supervised_process(
        argv,
        stdout_path=root / f"{PREFIX}{name}.stdout.bin",
        stderr_path=root / f"{PREFIX}{name}.stderr.bin",
        cwd=ROOT,
        sample_cpu_time=False,
        **kwargs,
    )


def test_supervisor_producer_outputs_validate_against_result_schema(
    parity_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    validator = SchemaValidator.from_file(SUPERVISOR_SCHEMA)
    outcomes = [
        _run(parity_dir, "clean", child_argv("clean")),
        _run(
            parity_dir,
            "launch-failure",
            [str(parity_dir / f"{PREFIX}missing-child")],
        ),
    ]

    heartbeat_path = parity_dir / f"{PREFIX}corrupt-heartbeat.json"
    malformed = (
        "from pathlib import Path; import time; "
        f"Path({str(heartbeat_path)!r}).write_text('{{broken'); time.sleep(0.05)"
    )
    outcomes.append(
        _run(
            parity_dir,
            "heartbeat-corruption",
            [sys.executable, "-c", malformed],
            heartbeat_path=heartbeat_path,
            incident_id=INCIDENT_ID,
        )
    )

    outcomes.append(
        _run(
            parity_dir,
            "supervisor-termination",
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=0.02,
            poll_interval_seconds=0.005,
            termination_grace_seconds=0.05,
        )
    )

    class TerminationFailureProcess:
        pid = 54321

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.returncode: int | None = None
            self.terminate_attempted = False
            self.kill_attempted = False

        def poll(self) -> int | None:
            if self.terminate_attempted and self.kill_attempted:
                self.returncode = 0
            return self.returncode

        def terminate(self) -> None:
            self.terminate_attempted = True
            raise OSError("injected terminate failure")

        def kill(self) -> None:
            self.kill_attempted = True
            raise OSError("injected kill failure")

    monkeypatch.setattr(process_supervisor.subprocess, "Popen", TerminationFailureProcess)
    failed = _run(
        parity_dir,
        "termination-failure",
        ["synthetic-child"],
        timeout_seconds=0.001,
        termination_grace_seconds=0,
    )
    decision = failed.document["termination"]["supervisor_initiated_termination"]
    assert decision["initiated"] is False
    assert decision["initiated_by"] == "NONE"
    assert decision["reason_code"] == "UNCLASSIFIED"
    assert decision["reason"] is None
    outcomes.append(failed)
    monkeypatch.undo()

    def oversized_reason() -> None:
        raise RuntimeError("bad\x00detail\n" + "x" * 3000)

    oversized = _run(
        parity_dir,
        "oversized-reason",
        child_argv("clean"),
        python_failure_evidence_provider=oversized_reason,
    )
    reason = oversized.document["evidence"]["reason"]
    assert len(reason) == 2048
    assert "\x00" not in reason
    assert "\n" not in reason
    outcomes.append(oversized)

    for outcome in outcomes:
        errors = validator.errors(outcome.document)
        assert not errors, errors


@pytest.mark.parametrize(
    "incident_id",
    [
        "INC-٢٠٢٦٠٩٢٦-101500-0a1b2c3d",
        "INC-20260926-١٠١٥٠٠-0a1b2c3d",
    ],
)
def test_supervisor_incident_id_requires_ascii_digits(parity_dir: Path, incident_id: str):
    with pytest.raises(ValueError, match="incident_id must match"):
        _run(parity_dir, "unicode-incident", child_argv("clean"), incident_id=incident_id)


@pytest.mark.parametrize("field", ["incident_id", "awb_session_id", "trace_id"])
def test_raw_heartbeat_runtime_and_schema_accept_identifier_at_max_length(field: str):
    validator = SchemaValidator.from_file(RAW_SCHEMA)
    payload = {
        "schema": raw_heartbeat.HEARTBEAT_SCHEMA,
        "incident_id": "i",
        "pid": 1234,
        "sequence": 1,
        "utc": "2026-09-26T10:15:00Z",
    }
    payload[field] = "x" * raw_heartbeat.MAX_IDENTIFIER_LENGTH
    record = raw_heartbeat.Heartbeat.from_dict(payload)
    assert validator.is_valid(payload), validator.errors(payload)
    assert getattr(record, field) == payload[field]


@pytest.mark.parametrize(
    "utc",
    [
        "2026-09-26T10:15:00Z",
        "2026-09-26T10:15:00+00:00",
        "2026-09-26T10:15:00-00:00",
    ],
)
def test_raw_heartbeat_runtime_and_schema_agree_on_accepted_utc(utc: str):
    validator = SchemaValidator.from_file(RAW_SCHEMA)
    payload = {
        "schema": raw_heartbeat.HEARTBEAT_SCHEMA,
        "incident_id": "incident-7",
        "pid": 1234,
        "sequence": 1,
        "utc": utc,
    }
    assert validator.is_valid(payload), validator.errors(payload)
    assert raw_heartbeat.Heartbeat.from_dict(payload).utc == utc


@pytest.mark.parametrize(
    "utc",
    [
        "2026-09-26T10:15:00+0000",
        "2026-09-26T10:15:00-0000",
        "2026-09-26T10:15:00+00",
        "2026-09-26T10:15:00-00",
        "2026-09-26T10:15:00+09:00",
    ],
)
def test_raw_heartbeat_runtime_and_schema_agree_on_rejected_utc(utc: str):
    validator = SchemaValidator.from_file(RAW_SCHEMA)
    payload = {
        "schema": raw_heartbeat.HEARTBEAT_SCHEMA,
        "incident_id": "incident-7",
        "pid": 1234,
        "sequence": 1,
        "utc": utc,
    }
    assert not validator.is_valid(payload)
    with pytest.raises((raw_heartbeat.HeartbeatError, raw_heartbeat.HeartbeatCorruptError)):
        raw_heartbeat.Heartbeat.from_dict(payload)
