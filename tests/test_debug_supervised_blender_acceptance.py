from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts import awb_debug_bb10_child_bootstrap as child_bootstrap
from scripts import awb_debug_process_supervisor as process_supervisor
from scripts import awb_debug_supervised_blender_acceptance as acceptance
from scripts import awb_debug_supervised_blender_child as supervised_child
from scripts import awb_supervisor_core, awb_supervisor_heartbeat

ROOT = Path(__file__).resolve().parents[1]


def _scratch_dir() -> Path:
    path = ROOT / "build" / f"supervised-acceptance-test-{uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _remove_known_paths(*paths: Path) -> None:
    for path in paths:
        if path.is_file():
            path.unlink()
    for path in paths:
        if path.is_dir():
            path.rmdir()


def test_incident_id_is_preallocated_in_canonical_shape():
    incident_id = acceptance.make_incident_id(
        now=datetime(2026, 9, 26, 1, 2, 3, tzinfo=UTC),
        token="0123abcd",
    )
    assert incident_id == "INC-20260926-010203-0123abcd"


def test_child_command_is_background_factory_only_and_uses_canonical_child():
    scratch = _scratch_dir()
    try:
        blender = scratch / "blender.exe"
        heartbeat = scratch / "canonical-heartbeat.json"
        status = scratch / "child-status.json"
        fault = scratch / "faulthandler.log"
        command = acceptance.build_child_command(
            blender,
            incident_id="INC-20260926-010203-0123abcd",
            heartbeat_path=heartbeat,
            status_path=status,
            faulthandler_path=fault,
        )

        assert command[:5] == [
            str(blender.resolve()),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
        ]
        assert command[command.index("--python") + 1] == str(acceptance.CHILD_SCRIPT.resolve())
        assert "--" in command
        assert "--awb-bb10-incident-id" in command
        assert (
            command[command.index("--awb-bb10-incident-id") + 1]
            == "INC-20260926-010203-0123abcd"
        )
        assert command[command.index("--awb-bb10-heartbeat-path") + 1] == str(heartbeat.resolve())
        assert command[command.index("--awb-bb10-status-path") + 1] == str(status.resolve())
        assert command[command.index("--awb-bb10-faulthandler-path") + 1] == str(fault.resolve())
        assert not any(item.casefold().endswith((".blend", ".blend1")) for item in command)
        child_options = supervised_child.load_child_options(argv=command, env={})
        bootstrap_config = child_bootstrap.load_config(argv=command, env={})
        assert child_options.lifetime_seconds == acceptance.CHILD_LIFETIME_SECONDS
        assert bootstrap_config.incident_id == "INC-20260926-010203-0123abcd"
        assert bootstrap_config.heartbeat_path == heartbeat.resolve()
        assert bootstrap_config.status_path == status.resolve()
        assert bootstrap_config.faulthandler_path == fault.resolve()

        unsafe = [*command, "--crash"]
        with pytest.raises(acceptance.AcceptanceError, match="unsafe Blender command flag"):
            acceptance.assert_isolated_command(unsafe, blender=blender, trial_root=scratch)
    finally:
        _remove_known_paths(scratch)


def test_isolated_environment_replaces_every_inherited_blender_user_path():
    scratch = _scratch_dir()
    trial = scratch / "trial"
    env = acceptance.build_isolated_env(
        trial,
        base_env={
            "PATH": "C:/Windows",
            "BLENDER_USER_CONFIG": "C:/workspace/portable_user/config",
            "BLENDER_USER_CUSTOM": "C:/workspace/portable_user/custom",
        },
    )

    assert {key for key in env if key.startswith("BLENDER_USER_")} == {
        "BLENDER_USER_RESOURCES",
        "BLENDER_USER_EXTENSIONS",
        "BLENDER_USER_CONFIG",
        "BLENDER_USER_SCRIPTS",
        "BLENDER_USER_DATAFILES",
    }
    user_paths = [value for key, value in env.items() if key.startswith("BLENDER_USER_")]
    assert all(
        Path(value).resolve() == trial.resolve()
        or trial.resolve() in Path(value).resolve().parents
        for value in user_paths
    )
    assert not any("portable_user" in value for value in user_paths)
    _remove_known_paths(
        trial / "blender_user" / "resources",
        trial / "blender_user" / "extensions",
        trial / "blender_user",
        trial,
        scratch,
    )


def _passing_receipt() -> dict[str, object]:
    incident_id = "INC-20260926-010203-0123abcd"
    pid = 9876
    return {
        "incident_id": incident_id,
        "blender": {"version": "5.2.1", "sha256": "a" * 64, "size_bytes": 1},
        "classification": "CLEAN_EXIT",
        "job_object_status": "UNSUPPORTED",
        "supervisor": {
            "schema": process_supervisor.SUPERVISOR_RESULT_SCHEMA,
            "incident_id": incident_id,
            "lifecycle": {"status": "COMPLETED", "exit_code": 0},
            "termination": {"supervisor_initiated_termination": {"initiated": False}},
            "process": {"pid": pid},
            "heartbeat": {
                "status": "FRESH",
                "heartbeat_id": incident_id,
                "age_seconds": 0.1,
                "beats_observed": 3,
            },
        },
        "canonical_heartbeat": {
            "schema": "awb-supervisor-heartbeat/v1",
            "incident_id": incident_id,
            "pid": pid,
            "sequence": 4,
            "utc": "2026-09-26T01:02:03.123Z",
        },
        "child_status": {
            "schema": child_bootstrap.STATUS_SCHEMA,
            "execution_status": "CHILD_LIFETIME_COMPLETE",
            "incident_id": incident_id,
            "child_pid": pid,
            "heartbeat": {"delivered_count": 4},
        },
        "isolation": {"enforced": True},
        "stdio_redirection": {"direct_file_redirection": True},
    }


def test_receipt_summary_accepts_clean_exit_fresh_heartbeat_and_optional_job_object():
    assert acceptance.summarize_receipt(_passing_receipt(), windows=False) == []
    assert acceptance.summarize_receipt(_passing_receipt(), windows=True) == [
        "WINDOWS_JOB_OBJECT_NOT_ASSIGNED"
    ]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda r: r.update(classification="HEARTBEAT_STALL/HANG_SUSPECTED"),
            "SUPERVISED_PROCESS_NOT_CLEAN_EXIT",
        ),
        (
            lambda r: r["supervisor"]["heartbeat"].update(status="STALE"),
            "SUPERVISOR_HEARTBEAT_NOT_FRESH",
        ),
        (
            lambda r: r["canonical_heartbeat"].update(
                incident_id="INC-20260926-010203-deadbeef"
            ),
            "CANONICAL_HEARTBEAT_INCIDENT_MISMATCH",
        ),
        (
            lambda r: r["supervisor"]["lifecycle"].update(exit_code=1),
            "SUPERVISED_LIFECYCLE_NOT_SUCCESSFUL",
        ),
        (
            lambda r: r["stdio_redirection"].update(direct_file_redirection=False),
            "STDIO_NOT_REDIRECTED_DIRECTLY_TO_FILES",
        ),
    ],
)
def test_receipt_summary_fails_closed_on_acceptance_defects(mutate, reason):
    receipt = _passing_receipt()
    mutate(receipt)
    assert reason in acceptance.summarize_receipt(receipt, windows=False)


def test_runner_writes_receipt_from_supervisor_and_child_evidence(monkeypatch):
    run_id = uuid4().hex[:8]
    output_root = ROOT / "build" / f"supervised-acceptance-test-{run_id}"
    output_root.mkdir(parents=True)
    result_path = output_root / "receipt.json"
    trial_parent = output_root / "trials"
    blender = output_root / "blender.exe"
    blender.write_bytes(b"test stub")
    monkeypatch.setattr(acceptance, "DEFAULT_BLENDER", blender)
    monkeypatch.setattr(acceptance, "TRIAL_PARENT", trial_parent)
    monkeypatch.setattr(
        acceptance,
        "probe_binary_identity",
        lambda *_args, **_kwargs: {
            "path": str(blender),
            "version": "5.2.1",
            "sha256": "a" * 64,
            "size_bytes": 9,
            "returncode": 0,
        },
    )

    incident_id = acceptance.make_incident_id()
    monkeypatch.setattr(acceptance, "make_incident_id", lambda: incident_id)

    def fake_run(argv, **kwargs):
        assert argv[0] == str(blender.resolve())
        acceptance.assert_isolated_env(kwargs["env"], fixture_root=trial_parent / incident_id)
        heartbeat_path = Path(kwargs["heartbeat_path"])
        heartbeat = awb_supervisor_heartbeat.make_heartbeat(
            incident_id=incident_id,
            pid=24680,
            sequence=3,
        )
        awb_supervisor_heartbeat.write_heartbeat(heartbeat_path, heartbeat)
        status_path = Path(
            next(
                argv[index + 1]
                for index, value in enumerate(argv[:-1])
                if value == "--awb-bb10-status-path"
            )
        )
        status_path.write_text(
            json.dumps(
                {
                    "schema": child_bootstrap.STATUS_SCHEMA,
                    "execution_status": "CHILD_LIFETIME_COMPLETE",
                    "incident_id": incident_id,
                    "child_pid": 24680,
                    "heartbeat": {"delivered_count": 3},
                }
            ),
            encoding="utf-8",
        )
        Path(kwargs["stdout_path"]).write_bytes(b"stdout\n")
        Path(kwargs["stderr_path"]).write_bytes(b"")
        termination = {
            "initiated": False,
            "initiated_by": "NONE",
            "reason_code": "UNCLASSIFIED",
            "reason": None,
            "requested_utc": None,
            "grace_period_seconds": None,
            "forced": False,
            "capture_attempted_before_termination": False,
            "capture_completed": False,
        }
        doc = {
            "schema": process_supervisor.SUPERVISOR_RESULT_SCHEMA,
            "supervision_id": "supervision-test",
            "supervisor_session_id": "session-test",
            "incident_id": incident_id,
            "incident_id_status": "BOUND",
            "started_utc": "2026-09-26T01:02:03.123Z",
            "completed_utc": "2026-09-26T01:02:05.123Z",
            "process": {
                "pid": 24680,
                "process_start_token": None,
                "image_path": str(blender),
                "command_line": None,
                "architecture": "X64",
            },
            "lifecycle": {
                "status": "COMPLETED",
                "exit_code": 0,
                "exit_kind": "EXIT_CODE",
                "exception_code": None,
                "duration_seconds": 2.0,
                "watchdog_recovered": False,
            },
            "termination": {"supervisor_initiated_termination": termination},
            "heartbeat": {
                "status": "FRESH",
                "heartbeat_id": incident_id,
                "last_heartbeat_utc": heartbeat.utc,
                "age_seconds": 0.1,
                "beats_observed": 3,
            },
            "artifacts": [],
            "evidence": {
                "status": "UNAVAILABLE",
                "reason": "acceptance-only",
                "preserved_preceding_causal_evidence": False,
                "incident_manifest_path": None,
                "incident_manifest_sha256": None,
                "runtime_log_path": None,
                "runtime_log_sha256": None,
            },
            "hypothesis": "UNAVAILABLE",
            "escalation_required": False,
        }
        return SimpleNamespace(
            document=doc,
            classification=awb_supervisor_core.IncidentClassification.CLEAN_EXIT,
            job_object_status="ASSIGNED",
        )

    monkeypatch.setattr(process_supervisor, "run_supervised_process", fake_run)
    monkeypatch.setattr(
        process_supervisor,
        "write_supervisor_result",
        lambda path, outcome: Path(path).write_text(json.dumps(outcome.document), encoding="utf-8"),
    )
    try:
        receipt = acceptance.run_acceptance(result_path=result_path, faulthandler=False)
        persisted = json.loads(result_path.read_text(encoding="utf-8"))
        assert receipt["status"] == "PASS"
        assert persisted["incident_id"] == incident_id
        assert persisted["stdio_redirection"]["direct_file_redirection"] is True
        assert persisted["canonical_heartbeat"]["sequence"] == 3
        assert persisted["job_object_status"] == "ASSIGNED"
        assert persisted["isolation"]["portable_user_touched"] is False
    finally:
        trial = trial_parent / incident_id
        _remove_known_paths(
            result_path,
            blender,
            trial / "canonical-heartbeat.json",
            trial / "child-status.json",
            trial / "blender-stdout.log",
            trial / "blender-stderr.log",
            trial / "supervisor-result.json",
            trial / "blender_user" / "resources",
            trial / "blender_user" / "extensions",
            trial / "blender_user",
            trial,
            trial_parent,
            output_root,
        )


def test_generic_supervisor_forwards_disposable_child_environment():
    scratch = _scratch_dir()
    stdout = scratch / "child.stdout"
    stderr = scratch / "child.stderr"
    env = {**os.environ, "AWB_SUPERVISOR_ENV_PROBE": "disposable-only"}
    outcome = process_supervisor.run_supervised_process(
        [sys.executable, "-c", "import os; print(os.environ['AWB_SUPERVISOR_ENV_PROBE'])"],
        stdout_path=stdout,
        stderr_path=stderr,
        timeout_seconds=5.0,
        sample_cpu_time=False,
        cwd=scratch,
        env=env,
    )
    assert outcome.classification is awb_supervisor_core.IncidentClassification.CLEAN_EXIT
    assert stdout.read_text(encoding="utf-8").strip() == "disposable-only"
    _remove_known_paths(stdout, stderr, scratch)
