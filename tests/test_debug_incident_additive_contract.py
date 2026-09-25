"""Additive incident evidence contract (BB-10).

Focused tests for the incident manifest additive contract only:

* ``manifest.json`` predeclares every BB-10 additive artifact name.
* Declaration never fabricates the additive files themselves.
* Published manifest bytes are byte-identical after native additive evidence.
* Create-once and incident-id binding stay fail-closed.

These tests never touch supervisor, heartbeat, or product code.
"""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAPTURE_MODULE_PATH = ROOT / "scripts" / "capture_awb_incident.py"
EVIDENCE_MODULE_PATH = ROOT / "scripts" / "awb_native_incident_evidence.py"
MANIFEST_SCHEMA_PATH = (
    ROOT / "docs" / "DEBUG" / "schemas" / "AWB_INCIDENT_MANIFEST_V1.schema.json"
)

# Names the published incident manifest must predeclare for BB-10 additive evidence.
BB10_ADDITIVE_ARTIFACTS = (
    "supervisor_result.json",
    "native_escalation.json",
    "native_artifacts/",
    "performance.json",
)
# Entries the native additive evidence module actually creates inside a bundle.
NATIVE_ADDITIVE_OUTPUTS = ("native_artifacts", "native_escalation.json")
# Names capture itself writes into the frozen bundle.
CAPTURE_ONLY_ARTIFACTS = ("analysis.json", "README.md")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _capture_module():
    return _load_module("capture_awb_incident_additive_tested", CAPTURE_MODULE_PATH)


def _evidence_module():
    return _load_module("awb_native_incident_evidence_additive_tested", EVIDENCE_MODULE_PATH)


@pytest.fixture
def evidence_tmp():
    """Keep temporary incident evidence inside the writable project workspace."""

    parent = ROOT / "tests" / ".awb_incident_additive_tmp"
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        )
    )


def _dump_source(parent: Path, *, name: str = "process.dmp", payload: bytes = b"dump") -> Path:
    source = parent / "source" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    return source


def _prepare_capture_root(tmp_path: Path) -> tuple[Path, dict]:
    """Build the smallest project root that publishes a COMPLETE incident bundle."""

    root = tmp_path / "project"
    debug = root / "debug"
    runtime = debug / "runtime_error_log"
    runtime.mkdir(parents=True)
    blend = root / "sample.blend"

    interaction = debug / "awb_interaction_trace.jsonl"
    _write_jsonl(
        interaction,
        [
            {
                "schema": "awb-interaction-trace/v1",
                "session_id": "session-current",
                "seq": 1,
                "utc": "2026-09-26T00:00:00.000+00:00",
                "monotonic_ns": 100,
                "channel": "LIFECYCLE",
                "event": "SESSION_START",
                "operation_id": None,
                "state": {"mode": "OBJECT", "active_object": "Cube"},
                "data": {},
            }
        ],
    )
    runtime_context = runtime / "awb_runtime_context.json"
    runtime_context.write_text(
        json.dumps({"schema": "awb-runtime-context/v1", "session_id": "session-current"}),
        encoding="utf-8",
    )
    (runtime / "blender_runtime.log").write_text(
        "\n=== AWB BLENDER RUNTIME SESSION "
        "2026-09-26T00:00:00.000+00:00 "
        f"blend={blend} ===\n"
        "AWB additive contract runtime log\n",
        encoding="utf-8",
    )

    live = {
        "probe_schema": "awb-debug-live-probe/v1",
        "pid": 4242,
        "binary_path": str(root / "vendor" / "blender-5.2.1-windows-x64" / "blender.exe"),
        "blender_version": "5.2.1",
        "blender_build_hash": "buildhash",
        "blender_build_branch": "blender-v5.2-release",
        "addon_module_file": str(root / "extension" / "awb" / "__init__.py"),
        "blend_file": str(blend),
        "source_paths": {
            "debug_root": str(debug),
            "interaction_trace": str(interaction),
            "precision_trace": str(debug / "awb_precision_trace.jsonl"),
            "runtime_context": str(runtime_context),
            "runtime_errors": str(runtime / "awb_runtime_errors.jsonl"),
            "replay": str(debug / "awb_replay_latest.json"),
            "raw_input_trace": str(debug / "awb_raw_input_trace.jsonl"),
        },
        "trace_session_id": "session-current",
        "last_trace_seq": 1,
        "last_trace_event": "SESSION_START",
        "last_causal": {
            "trace_id": "trace:synthetic",
            "span_id": "span:session",
            "parent_span_id": None,
            "operation_id": None,
            "parent_operation_id": None,
            "subsystem": "runtime",
            "lifecycle_phase": "session",
            "evaluation_phase": "live_context",
            "terminal_status": None,
            "route_outcome": None,
        },
        "semantic_replay": {
            "schema": "awb-semantic-replay/v1",
            "source_session_id": "session-current",
            "action_count": 1,
            "actions": [{"kind": "GENERIC_SYNTHETIC"}],
        },
        "checkpoint": {
            "mode": "OBJECT",
            "frame": 1,
            "subframe": 0.0,
            "context_identity": {
                "window": "ACTIVE_WINDOW",
                "screen": "Layout",
                "workspace": "Layout",
                "area_type": "VIEW_3D",
                "region_type": "WINDOW",
            },
        },
        "state_checkpoints": [
            {
                "schema": "awb-debug-checkpoint/v1",
                "incident_id": None,
                "checkpoint_id": "runtime:before",
                "boundary": "BEFORE_OPERATION",
                "status": "AVAILABLE",
                "confidence": "OBSERVED_LIVE",
                "source": {"event": "SESSION_START"},
                "trace_id": "trace:synthetic",
                "span_id": "span:session",
                "parent_span_id": None,
                "operation_id": None,
                "parent_operation_id": None,
                "subsystem": "runtime",
                "lifecycle_phase": "session",
                "evaluation_phase": "live_context",
                "context_identity": {"area_type": "VIEW_3D"},
                "blender_state": {"frame": 1},
                "native_state": {},
                "action_fcurves": {},
                "depsgraph_state": {},
                "domain_probes": [],
                "semantic_state_hash": "hash-before",
                "hash_schema": "awb-debug-state-hash/sha256-v1",
                "normalization_schema": "awb-debug-state-normalized/v1",
                "previous_checkpoint_id": None,
                "diff_from_previous": None,
                "unavailable_reason": None,
            }
        ],
    }
    return root, live


def _publish_incident(
    root: Path,
    live: dict,
    *,
    incident_id: str = "INC-20260926-120000-1234abcd",
) -> Path:
    module = _capture_module()
    result = module.capture_incident(
        root=root,
        live_probe=lambda _root: live,
        incident_id=incident_id,
    )
    return Path(result["incident_path"])


def _file_digests(bundle: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle.iterdir())
        if path.is_file()
    }


def _synthetic_bundle(parent: Path, *, manifest_overrides: dict | None = None) -> Path:
    module = _evidence_module()
    bundle = parent / "INC-20260926-130000-feedbeef"
    bundle.mkdir(parents=True)
    manifest = {
        "schema": module.MANIFEST_SCHEMA,
        "incident_id": bundle.name,
        "capture_status": "COMPLETE",
        "additive_artifacts": list(BB10_ADDITIVE_ARTIFACTS),
        "artifacts": {},
    }
    manifest.update(manifest_overrides or {})
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def test_manifest_predeclares_every_bb10_additive_artifact(evidence_tmp):
    root, live = _prepare_capture_root(evidence_tmp)
    incident = _publish_incident(root, live)

    manifest = json.loads((incident / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest["additive_artifacts"]

    assert manifest["capture_status"] == "COMPLETE"
    for name in BB10_ADDITIVE_ARTIFACTS:
        assert name in declared, name
    for name in CAPTURE_ONLY_ARTIFACTS:
        assert name in declared, name
    assert len(declared) == len(set(declared))
    assert set(declared) == (set(BB10_ADDITIVE_ARTIFACTS) | set(CAPTURE_ONLY_ARTIFACTS))


def test_manifest_additive_declaration_is_required_by_the_manifest_schema(evidence_tmp):
    schema = json.loads(MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    root, live = _prepare_capture_root(evidence_tmp)
    manifest = json.loads(
        (_publish_incident(root, live) / "manifest.json").read_text(encoding="utf-8")
    )

    assert "additive_artifacts" in schema["required"]
    assert schema["properties"]["additive_artifacts"]["items"] == {"type": "string"}
    assert set(manifest) >= set(schema["required"])
    assert isinstance(manifest["additive_artifacts"], list)


def test_predeclared_additive_names_are_relative_and_carry_no_local_paths(evidence_tmp):
    root, live = _prepare_capture_root(evidence_tmp)
    manifest = json.loads(
        (_publish_incident(root, live) / "manifest.json").read_text(encoding="utf-8")
    )

    for name in manifest["additive_artifacts"]:
        assert name == name.strip()
        assert not name.startswith(("/", "\\", "~"))
        assert "\\" not in name
        assert ":" not in name
        assert ".." not in name
        assert not Path(name).is_absolute()


def test_capture_declaration_never_fabricates_additive_files(evidence_tmp):
    root, live = _prepare_capture_root(evidence_tmp)
    incident = _publish_incident(root, live)

    manifest = json.loads((incident / "manifest.json").read_text(encoding="utf-8"))
    for name in BB10_ADDITIVE_ARTIFACTS:
        assert not (incident / name.rstrip("/")).exists(), name
    assert not (incident / "native_artifacts").exists()
    assert not (set(manifest["artifacts"]) & set(BB10_ADDITIVE_ARTIFACTS))
    assert not (set(manifest["artifacts"]) & set(NATIVE_ADDITIVE_OUTPUTS))


def test_native_additive_evidence_leaves_published_manifest_bytes_unchanged(evidence_tmp):
    root, live = _prepare_capture_root(evidence_tmp)
    incident = _publish_incident(root, live)
    manifest_path = incident / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    frozen_before = _file_digests(incident)

    evidence = _evidence_module()
    source = _dump_source(evidence_tmp, payload=b"synthetic native dump")
    copied = evidence.copy_native_artifact(bundle_dir=incident, source_path=source)
    document = evidence.write_native_escalation(
        incident,
        incident_id=incident.name,
        termination_kind="KILL_ON_JOB_CLOSE",
    )

    assert manifest_path.read_bytes() == manifest_bytes
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_sha256
    assert document["incident_id"] == incident.name
    assert len(document["artifacts"]) == 1
    artifact = document["artifacts"][0]
    assert artifact["source_path"] == copied["path"]
    assert artifact["size_bytes"] == copied["size_bytes"]
    assert artifact["sha256"] == copied["sha256"]
    assert artifact["kind"] == "DUMP"
    assert artifact["status"] == "PRESENT"
    assert {path.name for path in incident.iterdir()} - set(frozen_before) == set(
        NATIVE_ADDITIVE_OUTPUTS
    )
    for name, digest in frozen_before.items():
        assert hashlib.sha256((incident / name).read_bytes()).hexdigest() == digest, name


def test_additive_native_evidence_publishes_only_sanitized_bundle_relative_refs(evidence_tmp):
    root, live = _prepare_capture_root(evidence_tmp)
    incident = _publish_incident(root, live)
    evidence = _evidence_module()
    source = _dump_source(evidence_tmp, name="private dump.dmp", payload=b"private bytes")

    copied = evidence.copy_native_artifact(bundle_dir=incident, source_path=source)
    document = evidence.write_native_escalation(incident, incident_id=incident.name)

    assert copied["path"] == "native_artifacts/private_dump.dmp"
    assert copied["size_bytes"] == len(b"private bytes")
    assert copied["sha256"] == hashlib.sha256(b"private bytes").hexdigest()
    portable = json.dumps(document, ensure_ascii=False)
    assert "private_dump.dmp" in portable
    for leak in (str(source), str(evidence_tmp), "private dump"):
        assert leak not in portable, leak
    assert (incident / copied["path"]).is_file()


def test_native_evidence_is_create_once_and_never_rewrites_published_evidence(evidence_tmp):
    root, live = _prepare_capture_root(evidence_tmp)
    incident = _publish_incident(root, live)
    manifest_path = incident / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    evidence = _evidence_module()
    source = _dump_source(evidence_tmp, payload=b"first dump")
    copied = evidence.copy_native_artifact(incident, source)
    document = evidence.write_native_escalation(incident, incident_id=incident.name)
    escalation_path = incident / "native_escalation.json"
    escalation_bytes = escalation_path.read_bytes()

    with pytest.raises(FileExistsError):
        evidence.write_native_escalation(incident, incident_id=incident.name)
    with pytest.raises(FileExistsError):
        evidence.copy_native_artifact(incident, source)

    assert escalation_path.read_bytes() == escalation_bytes
    assert json.loads(escalation_bytes) == document
    assert (incident / copied["path"]).read_bytes() == b"first dump"
    assert evidence.inventory_native_artifacts(incident) == [copied]
    assert manifest_path.read_bytes() == manifest_bytes


def test_incident_id_mismatch_fails_closed_without_creating_additive_evidence(evidence_tmp):
    root, live = _prepare_capture_root(evidence_tmp)
    incident = _publish_incident(root, live)
    manifest_bytes = (incident / "manifest.json").read_bytes()
    entries_before = sorted(path.name for path in incident.iterdir())
    evidence = _evidence_module()

    with pytest.raises(ValueError, match="does not match"):
        evidence.write_native_escalation(
            incident,
            incident_id="INC-20260926-120001-deadbeef",
        )

    assert not (incident / "native_escalation.json").exists()
    assert not (incident / "native_artifacts").exists()
    assert sorted(path.name for path in incident.iterdir()) == entries_before
    assert (incident / "manifest.json").read_bytes() == manifest_bytes


def test_malformed_or_unpublished_manifest_fails_closed(evidence_tmp):
    evidence = _evidence_module()
    source = _dump_source(evidence_tmp, payload=b"dump")

    malformed = _synthetic_bundle(evidence_tmp / "malformed")
    (malformed / "manifest.json").write_text(
        json.dumps(
            {
                "schema": evidence.MANIFEST_SCHEMA,
                "incident_id": "not-an-incident-id",
                "capture_status": "COMPLETE",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid incident_id"):
        evidence.write_native_escalation(malformed)

    unpublished = _synthetic_bundle(
        evidence_tmp / "unpublished",
        manifest_overrides={"capture_status": "PARTIAL"},
    )
    with pytest.raises(ValueError, match="published complete incident"):
        evidence.write_native_escalation(unpublished)
    with pytest.raises(ValueError, match="published complete incident"):
        evidence.copy_native_artifact(unpublished, source)

    for bundle in (malformed, unpublished):
        assert not (bundle / "native_escalation.json").exists()
        assert not (bundle / "native_artifacts").exists()


def test_unsanitized_artifact_name_fails_closed_without_writing_escalation(evidence_tmp):
    bundle = _synthetic_bundle(evidence_tmp)
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    evidence = _evidence_module()
    native_dir = bundle / "native_artifacts"
    native_dir.mkdir()
    offender = native_dir / "not sanitized.dmp"
    offender.write_bytes(b"foreign artifact")
    source = _dump_source(evidence_tmp, payload=b"dump")

    copied = evidence.copy_native_artifact(bundle, source)
    with pytest.raises(ValueError, match="is not sanitized"):
        evidence.write_native_escalation(bundle)

    assert copied["path"] == "native_artifacts/process.dmp"
    assert not (bundle / "native_escalation.json").exists()
    assert offender.read_bytes() == b"foreign artifact"
    assert sorted(path.name for path in native_dir.iterdir()) == [
        "not sanitized.dmp",
        "process.dmp",
    ]
    assert (bundle / "manifest.json").read_bytes() == manifest_bytes


def test_capture_readme_states_the_immutable_manifest_additive_contract(evidence_tmp):
    root, live = _prepare_capture_root(evidence_tmp)
    incident = _publish_incident(root, live)

    readme = (incident / "README.md").read_text(encoding="utf-8").lower()

    assert "additive" in readme
    assert "manifest.json" in readme
    assert "immutable" in readme
    assert "frozen capture evidence" in readme
