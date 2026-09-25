from __future__ import annotations

import stat
import subprocess
from types import SimpleNamespace

import pytest

from scripts import awb_supervisor_core as supervisor_core
from scripts.awb_supervisor_core import (
    CLASSIFICATION_PRECEDENCE,
    IncidentEvidence,
    classify_incident,
    file_redirection_kwargs,
)
from scripts.awb_supervisor_core import (
    IncidentClassification as Classification,
)


def test_classification_precedence_is_explicit_and_failure_evidence_wins():
    assert CLASSIFICATION_PRECEDENCE == (
        Classification.LAUNCH_FAILURE,
        Classification.PYTHON_FAILURE,
        Classification.NATIVE_CRASH,
        Classification.HEARTBEAT_STALL,
        Classification.CLEAN_EXIT,
        Classification.UNKNOWN,
    )
    assert classify_incident(
        IncidentEvidence(
            process_started=False,
            launch_error="CreateProcess failed",
            python_failure_evidence=True,
            native_crash_evidence=True,
        )
    ) is Classification.LAUNCH_FAILURE
    assert classify_incident(
        IncidentEvidence(
            process_started=True,
            exit_code=0,
            python_failure_evidence=True,
        )
    ) is Classification.PYTHON_FAILURE
    assert classify_incident(
        IncidentEvidence(
            process_started=True,
            process_alive=False,
            exit_code=-11,
            native_crash_evidence=True,
            heartbeat_stale=True,
        )
    ) is Classification.NATIVE_CRASH


def test_supervisor_termination_is_never_classified_as_native_crash():
    result = classify_incident(
        IncidentEvidence(
            process_started=True,
            process_alive=False,
            exit_code=-9,
            native_crash_evidence=True,
            supervisor_initiated_termination=True,
            heartbeat_stale=True,
        )
    )
    assert result is Classification.HEARTBEAT_STALL
    assert classify_incident(
        IncidentEvidence(
            process_started=True,
            process_alive=False,
            exit_code=-9,
            native_crash_evidence=True,
            supervisor_initiated_termination=True,
        )
    ) is Classification.UNKNOWN


def test_stall_clean_and_insufficient_evidence_cases():
    assert classify_incident(
        IncidentEvidence(process_started=True, process_alive=True, heartbeat_stale=True)
    ) is Classification.HEARTBEAT_STALL
    assert classify_incident(
        IncidentEvidence(process_started=True, process_alive=False, exit_code=0)
    ) is Classification.CLEAN_EXIT
    assert classify_incident(
        IncidentEvidence(process_started=True, exit_code=7)
    ) is Classification.UNKNOWN
    assert classify_incident(IncidentEvidence()) is Classification.UNKNOWN


def test_stdio_contract_accepts_regular_files_and_rejects_pipes(monkeypatch):
    class FileHandle:
        def __init__(self, descriptor):
            self.descriptor = descriptor

        def fileno(self):
            return self.descriptor

    monkeypatch.setattr(
        supervisor_core.os,
        "fstat",
        lambda descriptor: SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600 if descriptor == 42 else stat.S_IFIFO | 0o600
        ),
    )
    stdout_file = FileHandle(42)
    stderr_file = FileHandle(42)
    assert file_redirection_kwargs(stdout_file, stderr_file) == {
        "stdout": stdout_file,
        "stderr": stderr_file,
    }
    with pytest.raises(TypeError, match="stderr_file"):
        file_redirection_kwargs(stdout_file, subprocess.PIPE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="regular log file"):
        file_redirection_kwargs(stdout_file, FileHandle(43))
