from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "awb_native_incident_evidence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "awb_native_incident_evidence_tested", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence_tmp():
    """Keep temporary evidence inside the writable project workspace."""

    parent = ROOT / "tests" / ".awb_native_evidence_tmp"
    parent.mkdir(exist_ok=True)
    temp = parent / uuid4().hex
    temp.mkdir()
    try:
        yield temp
    finally:
        resolved_parent = parent.resolve()
        resolved_temp = temp.resolve()
        if resolved_temp.parent != resolved_parent:
            raise RuntimeError("test evidence directory escaped its workspace parent")
        shutil.rmtree(resolved_temp)
        if not any(resolved_parent.iterdir()):
            resolved_parent.rmdir()


def _published_bundle(tmp_path: Path) -> tuple[Path, bytes]:
    module = _load_module()
    bundle = tmp_path / "INC-20260926-120000-a1b2c3d4"
    bundle.mkdir()
    manifest = {
        "schema": module.MANIFEST_SCHEMA,
        "incident_id": bundle.name,
        "capture_status": "COMPLETE",
        "artifacts": {"interaction_trace": {"sha256": "unchanged"}},
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    (bundle / "manifest.json").write_bytes(manifest_bytes)
    return bundle, manifest_bytes


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ({"forced_termination": True}, True),
        ({"termination_kind": "KILL_ON_JOB_CLOSE"}, True),
        ({"termination_reason": "watchdog forced termination"}, True),
        ({"termination_kind": "SUPERVISOR_FORCED_TERMINATION"}, True),
        ({"termination_kind": "WATCHDOG_TIMEOUT"}, False),
        ({"termination_reason": "user requested normal shutdown"}, False),
        ({}, False),
    ],
)
def test_forced_termination_inputs_are_excluded(inputs, expected):
    module = _load_module()
    assert module.is_forced_termination(**inputs) is expected


def test_non_windows_job_object_is_explicitly_unsupported(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_IS_WINDOWS", False)

    result = module.create_kill_on_close_job()

    assert result.status == "UNSUPPORTED"
    assert result.owner is None
    assert result.reason == "WINDOWS_ONLY"


def test_non_windows_process_cpu_sampling_is_explicitly_unsupported(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_IS_WINDOWS", False)

    sample = module.sample_process_cpu_time(4321)

    assert sample.status == "UNSUPPORTED"
    assert sample.pid == 4321
    assert sample.total_seconds is None


class _FakeKernel32:
    def __init__(self, *, set_information_ok=True, assign_ok=True):
        self.set_information_ok = set_information_ok
        self.assign_ok = assign_ok
        self.closed_handles = []
        self.info_class = None
        self.info_size = None
        self.limit_flags = None
        self.assigned = None

    def CreateJobObjectW(self, _attributes, _name):
        return 100

    def SetInformationJobObject(self, _job, info_class, info_ptr, size):
        module = _load_module()
        self.info_class = info_class
        self.info_size = size
        info = module.ctypes.cast(
            info_ptr,
            module.ctypes.POINTER(module._JOBOBJECT_EXTENDED_LIMIT_INFORMATION),
        ).contents
        self.limit_flags = info.BasicLimitInformation.LimitFlags
        return self.set_information_ok

    def OpenProcess(self, _access, _inherit, pid):
        self.opened_pid = pid
        return 200

    def AssignProcessToJobObject(self, job, process):
        self.assigned = (job, process)
        return self.assign_ok

    def CloseHandle(self, handle):
        self.closed_handles.append(handle)
        return True

    def GetLastError(self):
        return 123


def test_job_object_enables_kill_on_close_and_closes_owned_handles(monkeypatch):
    module = _load_module()
    fake_api = _FakeKernel32()
    monkeypatch.setattr(module, "_IS_WINDOWS", True)
    monkeypatch.setattr(module, "_load_windows_api", lambda: fake_api)

    result = module.create_kill_on_close_job()

    assert result.status == "AVAILABLE"
    assert result.owner is not None
    assert fake_api.info_class == module.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION
    assert fake_api.info_size > 0
    assert fake_api.limit_flags == module.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    result.owner.assign_pid(4321)
    assert fake_api.opened_pid == 4321
    assert fake_api.assigned == (100, 200)
    assert 200 in fake_api.closed_handles
    result.owner.close()
    result.owner.close()
    assert fake_api.closed_handles.count(100) == 1
    assert result.owner.closed


def test_job_object_creation_failure_closes_unconfigured_job(monkeypatch):
    module = _load_module()
    fake_api = _FakeKernel32(set_information_ok=False)
    monkeypatch.setattr(module, "_IS_WINDOWS", True)
    monkeypatch.setattr(module, "_load_windows_api", lambda: fake_api)

    result = module.create_kill_on_close_job()

    assert result.status == "ERROR"
    assert "SetInformationJobObject failed" in result.reason
    assert fake_api.closed_handles == [100]


def test_process_cpu_time_sampler_is_mockable_and_uses_creation_identity(monkeypatch):
    module = _load_module()

    class CpuApi(_FakeKernel32):
        def GetProcessTimes(self, _process, created, exited, kernel, user):
            values = (
                (created, 50_000_000),
                (exited, 0),
                (kernel, 20_000_000),
                (user, 30_000_000),
            )
            for pointer, ticks in values:
                value = module.ctypes.cast(
                    pointer, module.ctypes.POINTER(module._FILETIME)
                ).contents
                value.dwLowDateTime = ticks & 0xFFFFFFFF
                value.dwHighDateTime = ticks >> 32
            return True

    api = CpuApi()
    monkeypatch.setattr(module, "_IS_WINDOWS", True)
    monkeypatch.setattr(module, "_load_windows_api", lambda: api)

    sample = module.sample_process_cpu_time(4321)

    assert sample.status == "AVAILABLE"
    assert sample.creation_time_100ns == 50_000_000
    assert sample.user_seconds == pytest.approx(3)
    assert sample.kernel_seconds == pytest.approx(2)
    assert sample.total_seconds == pytest.approx(5)
    assert api.closed_handles == [200]


def test_artifact_copy_hashes_size_and_uses_only_sanitized_bundle_reference(evidence_tmp):
    module = _load_module()
    bundle, _manifest_bytes = _published_bundle(evidence_tmp)
    source = evidence_tmp / "private-source-location" / "dump.dmp"
    source.parent.mkdir()
    payload = b"native dump bytes\x00\x01"
    source.write_bytes(payload)

    copied = module.copy_native_artifact(
        bundle,
        source,
        name=r"C:\Users\private\..\private dump.dmp",
    )

    assert copied == {
        "path": "native_artifacts/private_dump.dmp",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert (bundle / copied["path"]).read_bytes() == payload
    assert str(source) not in json.dumps(copied)
    assert module.inventory_native_artifacts(bundle) == [copied]
    with pytest.raises(FileExistsError):
        module.copy_native_artifact(bundle, source, name="private dump.dmp")
    assert (bundle / copied["path"]).read_bytes() == payload


def test_native_escalation_writer_is_additive_create_once_and_records_forced_termination(
    evidence_tmp,
):
    module = _load_module()
    bundle, manifest_bytes = _published_bundle(evidence_tmp)
    source = evidence_tmp / "supervisor" / "process.dmp"
    source.parent.mkdir()
    source.write_bytes(b"minidump")
    copied = module.copy_native_artifact(bundle, source)

    document = module.write_native_escalation(
        bundle,
        incident_id=bundle.name,
        termination_kind="KILL_ON_JOB_CLOSE",
    )

    assert document["schema"] == module.NATIVE_ESCALATION_SCHEMA
    assert document["incident_id"] == bundle.name
    assert document["incident_id_status"] == "BOUND"
    assert document["preconditions"]["incident_manifest_path"] == "manifest.json"
    assert document["preconditions"]["incident_manifest_sha256"] == hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    assert document["analysis"]["status"] == "NOT_ATTEMPTED"
    assert document["verdict"]["responsibility"] == "UNDETERMINED"
    assert document["supervisor_initiated_termination"]["initiated_by"] == "TOOL_CHAIN"
    assert document["supervisor_initiated_termination"]["reason_code"] == "UNCLASSIFIED"
    assert document["supervisor_initiated_termination"]["forced"] is True
    assert document["artifacts"] == [
        {
            "name": "process.dmp",
            "kind": "DUMP",
            "status": "PRESENT",
            "source_path": copied["path"],
            "size_bytes": copied["size_bytes"],
            "sha256": copied["sha256"],
            "hash_algorithm": "sha256",
            "captured_bytes": copied["size_bytes"],
            "bounded": False,
            "truncated": False,
            "collected_utc": document["created_utc"],
            "missing_reason": None,
        }
    ]
    written = json.loads((bundle / "native_escalation.json").read_text(encoding="utf-8"))
    assert written == document
    written_bytes = (bundle / "native_escalation.json").read_bytes()
    assert (bundle / "manifest.json").read_bytes() == manifest_bytes
    assert str(source) not in json.dumps(written)
    with pytest.raises(FileExistsError):
        module.write_native_escalation(bundle, incident_id=bundle.name)
    assert (bundle / "native_escalation.json").read_bytes() == written_bytes
    assert (bundle / "native_artifacts").is_dir()
    assert (bundle / "manifest.json").read_bytes() == manifest_bytes


def test_writer_fails_closed_when_artifact_inventory_exceeds_schema_limit(evidence_tmp):
    module = _load_module()
    bundle, manifest_bytes = _published_bundle(evidence_tmp)
    native_dir = bundle / "native_artifacts"
    native_dir.mkdir()
    for index in range(65):
        (native_dir / f"artifact-{index:02d}.dat").write_bytes(b"artifact")

    with pytest.raises(ValueError, match="limit of 64"):
        module.write_native_escalation(bundle)

    assert not (bundle / "native_escalation.json").exists()
    assert (bundle / "manifest.json").read_bytes() == manifest_bytes


def test_writer_rejects_manifest_identity_mismatch_without_writing(evidence_tmp):
    module = _load_module()
    bundle, manifest_bytes = _published_bundle(evidence_tmp)

    with pytest.raises(ValueError, match="does not match"):
        module.write_native_escalation(
            bundle,
            incident_id="INC-20260926-120001-deadbeef",
        )

    assert not (bundle / "native_escalation.json").exists()
    assert (bundle / "manifest.json").read_bytes() == manifest_bytes


def test_writer_records_unassessed_non_forced_evidence_and_empty_artifact_folder(
    evidence_tmp,
):
    module = _load_module()
    bundle, _manifest_bytes = _published_bundle(evidence_tmp)

    document = module.write_native_escalation(bundle)

    assert document["supervisor_initiated_termination"] is None
    assert document["tooling"]["status"] == "NOT_ATTEMPTED"
    assert document["dump"]["status"] == "NOT_CAPTURED"
    assert document["analysis"]["status"] == "NOT_ATTEMPTED"
    assert document["verdict"] == {
        "classification": "UNDETERMINED",
        "responsibility": "UNDETERMINED",
        "summary": "Native root cause has not been established.",
    }
    assert document["artifacts"] == []
    assert (bundle / "native_artifacts").is_dir()


@pytest.mark.parametrize(
    ("unsafe", "expected"),
    [
        (r"..\..\Users\alice\private dump.dmp", "private_dump.dmp"),
        ("C:/secret/native.dmp", "native.dmp"),
        ("CON", "_CON"),
        ("../../", "artifact"),
    ],
)
def test_artifact_name_sanitization_removes_paths_and_windows_reserved_names(
    unsafe, expected
):
    module = _load_module()
    assert module.sanitize_artifact_name(unsafe) == expected
