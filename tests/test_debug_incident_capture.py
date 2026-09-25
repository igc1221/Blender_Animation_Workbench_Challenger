from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "capture_awb_incident.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("capture_awb_incident_tested", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    path.write_bytes(data)
    return data


def _prepare_project_root(tmp_path: Path, *, external_debug: bool = False):
    root = tmp_path / "project"
    root.mkdir()
    project_debug = root / "debug"
    runtime = project_debug / "runtime_error_log"
    runtime.mkdir(parents=True)

    source_debug = (tmp_path / "external_debug") if external_debug else project_debug
    source_runtime = source_debug / "runtime_error_log"
    source_runtime.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "schema": "awb-interaction-trace/v1",
            "session_id": "session-current",
            "seq": 1,
            "utc": "2026-09-25T00:00:00.000+00:00",
            "monotonic_ns": 100,
            "channel": "LIFECYCLE",
            "event": "SESSION_START",
            "operation_id": None,
            "parent_operation_id": None,
            "state": {"mode": "OBJECT", "active_object": "Cube"},
            "data": {},
        },
        {
            "schema": "awb-interaction-trace/v1",
            "session_id": "session-current",
            "seq": 2,
            "utc": "2026-09-25T00:00:01.000+00:00",
            "monotonic_ns": 200,
            "channel": "OPERATOR",
            "event": "TRANSFORM_BEGIN",
            "operation_id": "op-1",
            "parent_operation_id": None,
            "state": {
                "mode": "OBJECT",
                "frame": 4,
                "subframe": 0.0,
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "selected_pose_bones": [],
            },
            "data": {"axis": "X"},
        },
        {
            "schema": "awb-interaction-trace/v1",
            "session_id": "session-current",
            "seq": 3,
            "utc": "2026-09-25T00:00:02.000+00:00",
            "monotonic_ns": 300,
            "channel": "OPERATOR",
            "event": "TRANSFORM_FAIL",
            "operation_id": "op-1",
            "parent_operation_id": None,
            "trace_id": "trace:synthetic",
            "span_id": "span:operator",
            "parent_span_id": "span:ingress",
            "subsystem": "operator",
            "lifecycle_phase": "fail",
            "evaluation_phase": "live_context",
            "terminal_status": "FAILED",
            "route_outcome": "CLAIMED",
            "dropped": 2,
            "truncated": False,
            "state": {
                "mode": "OBJECT",
                "frame": 4,
                "subframe": 0.0,
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "selected_pose_bones": [],
            },
            "data": {"error": "synthetic"},
        },
    ]
    interaction = source_debug / "awb_interaction_trace.jsonl"
    interaction_bytes = _write_jsonl(interaction, records)
    precision = source_debug / "awb_precision_trace.jsonl"
    precision_bytes = _write_jsonl(
        precision,
        [
            {
                "schema": "awb-precision-trace/v1",
                "session_id": "session-current",
                "seq": 1,
                "frame": 4,
                "subframe": 0.0,
                "bones": [],
            }
        ],
    )
    runtime_context = source_runtime / "awb_runtime_context.json"
    runtime_context.write_text(
        json.dumps({"schema": "awb-runtime-context/v1", "session_id": "session-current"}),
        encoding="utf-8",
    )
    runtime_errors = source_runtime / "awb_runtime_errors.jsonl"
    runtime_errors_bytes = _write_jsonl(
        runtime_errors,
        [{"schema": "awb-runtime-error/v1", "message": "synthetic"}],
    )
    replay = source_debug / "awb_replay_latest.json"
    replay.write_text(
        json.dumps(
            {
                "schema": "awb-semantic-replay/v1",
                "source_session_id": "older-session",
                "action_count": 0,
                "actions": [],
            }
        ),
        encoding="utf-8",
    )

    raw_runtime = runtime / "blender_runtime.log"
    raw_runtime_bytes = (
        "\n=== AWB BLENDER RUNTIME SESSION "
        "2026-09-25T00:00:00.000+00:00 "
        f"blend={root / 'sample.blend'} ===\n"
        "AWB synthetic runtime log\n"
    ).encode()
    raw_runtime.write_bytes(raw_runtime_bytes)

    live = {
        "probe_schema": "awb-debug-live-probe/v1",
        "pid": 4242,
        "binary_path": str(root / "vendor" / "blender-5.2.1-windows-x64" / "blender.exe"),
        "blender_version": "5.2.1",
        "blender_build_hash": "buildhash",
        "blender_build_branch": "blender-v5.2-release",
        "addon_module_file": str(root / "extension" / "blender_animation_workbench" / "__init__.py"),
        "blend_file": str(root / "sample.blend"),
        "source_paths": {
            "debug_root": str(source_debug),
            "interaction_trace": str(interaction),
            "precision_trace": str(precision),
            "runtime_context": str(runtime_context),
            "runtime_errors": str(runtime_errors),
            "replay": str(replay),
        },
        "trace_session_id": "session-current",
        "last_trace_seq": 3,
        "last_trace_event": "TRANSFORM_FAIL",
        "last_causal": {
            "trace_id": "trace:synthetic",
            "span_id": "span:operator",
            "parent_span_id": "span:ingress",
            "operation_id": "op-1",
            "parent_operation_id": None,
            "subsystem": "operator",
            "lifecycle_phase": "fail",
            "evaluation_phase": "live_context",
            "terminal_status": "FAILED",
            "route_outcome": "CLAIMED",
        },
        "semantic_replay": {
            "schema": "awb-semantic-replay/v1",
            "source_session_id": "session-current",
            "action_count": 1,
            "actions": [{"kind": "GENERIC_SYNTHETIC"}],
        },
        "checkpoint": {
            "mode": "OBJECT",
            "frame": 4,
            "subframe": 0.0,
            "active_object": {"name": "Cube", "type": "MESH"},
            "selected_objects": [{"name": "Cube", "type": "MESH"}],
            "selected_pose_bones": [],
            "context_identity": {
                "window": "ACTIVE_WINDOW",
                "screen": "Layout",
                "workspace": "Layout",
                "area_type": "VIEW_3D",
                "region_type": "WINDOW",
            },
            "native_lifecycle": {
                "status": "UNAVAILABLE_BB2",
                "operator": None,
                "modal_owner": None,
            },
        },
    }

    source_bytes = {
        interaction: interaction_bytes,
        precision: precision_bytes,
        runtime_errors: runtime_errors_bytes,
        raw_runtime: raw_runtime_bytes,
    }
    return root, live, source_bytes


def test_capture_generic_live_incident_is_immutable_and_fresh(tmp_path: Path):
    module = _load_module()
    root, live, source_bytes = _prepare_project_root(tmp_path, external_debug=True)

    result = module.capture_incident(
        root=root,
        live_probe=lambda _root: live,
        incident_id="INC-20260925-120000-1234abcd",
    )

    incident = Path(result["incident_path"])
    assert incident.is_dir()
    manifest = json.loads((incident / "manifest.json").read_text(encoding="utf-8"))
    analysis = json.loads((incident / "analysis.json").read_text(encoding="utf-8"))
    state_before = json.loads((incident / "state_before.json").read_text(encoding="utf-8"))
    state_failure = json.loads((incident / "state_failure.json").read_text(encoding="utf-8"))
    replay = json.loads((incident / "replay.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "awb-debug-incident-manifest/v1"
    assert manifest["evidence_status"] == "COMPLETE"
    assert result["evidence_status"] == "COMPLETE"
    assert manifest["source_identity"]["debug_root_source"] == "live_query"
    assert manifest["source_identity"]["debug_root"] == live["source_paths"]["debug_root"]
    assert manifest["source_identity"]["raw_runtime_root_source"] == "project_launcher"
    assert Path(manifest["source_identity"]["raw_runtime_root"]) == (
        root / "debug" / "runtime_error_log"
    ).resolve()
    assert manifest["source_identity"]["blender"]["blend_file"] == live["blend_file"]
    assert manifest["artifacts"]["blender_runtime_current"]["source_authority"] == "project_launcher"
    assert manifest["artifacts"]["blender_runtime_current"]["live_blend_match"] is True
    assert analysis["live_probe_status"] == "AVAILABLE"
    assert analysis["viewport"]["reason"] == "NO_SAFE_READ_ONLY_CAPTURE_PATH_BB1"
    assert not (incident / "viewport.png").exists()

    assert state_before["status"] == "AVAILABLE"
    assert state_before["confidence"] == "TRACE_PAIRED_BEGIN"
    assert state_before["operation_id"] == "op-1"
    assert state_before["source"]["seq"] == 2

    assert state_failure["status"] == "AVAILABLE"
    assert state_failure["trace_id"] == "trace:synthetic"
    assert state_failure["span_id"] == "span:operator"
    assert state_failure["parent_span_id"] == "span:ingress"
    assert state_failure["operation_id"] == "op-1"
    assert state_failure["subsystem"] == "operator"
    assert state_failure["lifecycle_phase"] == "fail"
    assert state_failure["evaluation_phase"] == "live_context"
    assert state_failure["domain_probes"] == []
    assert "rigped" not in json.dumps(state_failure, ensure_ascii=False).lower()

    timeline_rows = [
        json.loads(line)
        for line in (incident / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    causal_terminal = next(row for row in timeline_rows if row["event"] == "TRANSFORM_FAIL")
    assert causal_terminal["trace_id"] == "trace:synthetic"
    assert causal_terminal["span_id"] == "span:operator"
    assert causal_terminal["subsystem"] == "operator"
    assert causal_terminal["lifecycle_phase"] == "fail"
    assert causal_terminal["terminal_status"] == "FAILED"
    assert causal_terminal["route_outcome"] == "CLAIMED"
    assert causal_terminal["dropped"] == 2
    assert causal_terminal["truncated"] is False

    assert replay["status"] == "AVAILABLE"
    assert replay["freshness"] == "FRESH"
    assert replay["source"] == "live_in_memory_builder"
    assert replay["source_session_id"] == "session-current"

    frozen_interaction = incident / "interaction_trace.current.jsonl"
    expected_hash = hashlib.sha256(frozen_interaction.read_bytes()).hexdigest()
    assert manifest["artifacts"]["interaction_current"]["sha256"] == expected_hash

    for source, expected in source_bytes.items():
        assert source.read_bytes() == expected


def test_capture_falls_back_when_blender_mcp_is_unavailable(tmp_path: Path):
    module = _load_module()
    root, _live, _source_bytes = _prepare_project_root(tmp_path)

    def unavailable(_root):
        raise TimeoutError("synthetic MCP unavailable")

    result = module.capture_incident(
        root=root,
        live_probe=unavailable,
        incident_id="INC-20260925-120001-1234abcd",
    )
    incident = Path(result["incident_path"])
    manifest = json.loads((incident / "manifest.json").read_text(encoding="utf-8"))
    analysis = json.loads((incident / "analysis.json").read_text(encoding="utf-8"))
    state_failure = json.loads((incident / "state_failure.json").read_text(encoding="utf-8"))
    replay = json.loads((incident / "replay.json").read_text(encoding="utf-8"))

    assert manifest["source_identity"]["debug_root_source"] == "fallback_project"
    assert analysis["live_probe_status"] == "UNAVAILABLE"
    assert "TimeoutError" in analysis["live_probe_error"]
    assert state_failure["status"] == "UNAVAILABLE"
    assert state_failure["unavailable_reason"] == "BLENDER_MCP_UNAVAILABLE"
    assert replay["status"] == "AVAILABLE"
    assert replay["freshness"] == "UNVERIFIED_ON_DISK"


def test_capture_never_overwrites_published_incident(tmp_path: Path):
    module = _load_module()
    root, live, _source_bytes = _prepare_project_root(tmp_path)
    incident_id = "INC-20260925-120002-1234abcd"

    first = module.capture_incident(
        root=root,
        live_probe=lambda _root: live,
        incident_id=incident_id,
    )
    manifest_path = Path(first["incident_path"]) / "manifest.json"
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(FileExistsError):
        module.capture_incident(
            root=root,
            live_probe=lambda _root: live,
            incident_id=incident_id,
        )

    assert manifest_path.read_bytes() == manifest_before


def test_failed_capture_is_not_published_as_complete(tmp_path: Path):
    module = _load_module()
    root, live, _source_bytes = _prepare_project_root(tmp_path)
    incident_id = "INC-20260925-120003-1234abcd"

    with pytest.raises(module.CaptureFatalError):
        module.capture_incident(
            root=root,
            live_probe=lambda _root: live,
            incident_id=incident_id,
            fault_after_stage="source_artifacts",
        )

    incidents = root / "debug" / "incidents"
    assert not (incidents / incident_id).exists()
    staging = list(incidents.glob(f".staging-{incident_id}-*"))
    assert len(staging) == 1
    partial = json.loads((staging[0] / "capture_partial.json").read_text(encoding="utf-8"))
    assert partial["capture_status"] == "PARTIAL"
    assert (incidents / f".reserve-{incident_id}").is_file()
    with pytest.raises(FileExistsError, match="already reserved"):
        module.capture_incident(
            root=root,
            live_probe=lambda _root: live,
            incident_id=incident_id,
        )


def test_bounded_jsonl_tail_is_line_aligned_and_records_truncation(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "trace.jsonl"
    _write_jsonl(
        source,
        [
            {"schema": "test/v1", "seq": index, "payload": "x" * 80}
            for index in range(20)
        ],
    )
    data, meta = module._read_source_once(
        source,
        max_bytes=350,
        line_aligned_tail=True,
    )
    assert data is not None
    assert meta["status"] == "TRUNCATED"
    assert meta["truncated"] is True
    assert meta["captured_offset"] > 0
    rows, errors = module._parse_jsonl(data)
    assert errors == 0
    assert rows
    assert rows[-1]["seq"] == 19


def test_state_before_requires_same_session_operation_begin_pair():
    module = _load_module()
    records = [
        {
            "session_id": "s1",
            "seq": 1,
            "event": "TRANSFORM_BEGIN",
            "operation_id": "other-op",
            "state": {"mode": "OBJECT"},
        },
        {
            "session_id": "s2",
            "seq": 2,
            "event": "TRANSFORM_FAIL",
            "operation_id": "op-2",
            "state": {"mode": "OBJECT"},
        },
    ]
    checkpoint = module._state_before_checkpoint(
        "INC-test",
        records,
        current_session_id="s2",
    )
    assert checkpoint["status"] == "UNAVAILABLE"
    assert checkpoint["unavailable_reason"] == "NO_MATCHING_BEGIN"


def test_optional_precision_and_error_logs_do_not_downgrade_live_capture(tmp_path: Path):
    module = _load_module()
    root, live, _source_bytes = _prepare_project_root(tmp_path)
    Path(live["source_paths"]["precision_trace"]).unlink()
    Path(live["source_paths"]["runtime_errors"]).unlink()
    Path(live["source_paths"]["replay"]).unlink()

    result = module.capture_incident(
        root=root,
        live_probe=lambda _root: live,
        incident_id="INC-20260925-120004-1234abcd",
    )

    incident = Path(result["incident_path"])
    manifest = json.loads((incident / "manifest.json").read_text(encoding="utf-8"))
    assert result["evidence_status"] == "COMPLETE"
    assert manifest["evidence_status"] == "COMPLETE"
    assert manifest["artifacts"]["precision_current"]["status"] == "MISSING"
    assert manifest["artifacts"]["runtime_errors_current"]["status"] == "MISSING"
    assert manifest["artifacts"]["replay_on_disk_current"]["status"] == "MISSING"


def test_state_before_never_falls_back_to_previous_session_terminal():
    module = _load_module()
    records = [
        {
            "session_id": "previous",
            "seq": 1,
            "event": "TRANSFORM_BEGIN",
            "operation_id": "old-op",
            "state": {"mode": "OBJECT"},
        },
        {
            "session_id": "previous",
            "seq": 2,
            "event": "TRANSFORM_FAIL",
            "operation_id": "old-op",
            "state": {"mode": "OBJECT"},
        },
        {
            "session_id": "current",
            "seq": 1,
            "event": "TRANSFORM_BEGIN",
            "operation_id": "new-op",
            "state": {"mode": "OBJECT"},
        },
    ]

    checkpoint = module._state_before_checkpoint(
        "INC-test",
        records,
        current_session_id="current",
    )
    assert checkpoint["status"] == "UNAVAILABLE"
    assert (
        checkpoint["unavailable_reason"]
        == "NO_TERMINAL_OPERATION_IN_CURRENT_SESSION"
    )


def test_invalid_incident_id_cannot_escape_incident_root(tmp_path: Path):
    module = _load_module()
    root, live, _source_bytes = _prepare_project_root(tmp_path)

    with pytest.raises(ValueError, match="Invalid incident id"):
        module.capture_incident(
            root=root,
            live_probe=lambda _root: live,
            incident_id="../outside",
        )
    assert not (root / "outside").exists()


def test_post_publish_result_write_failure_does_not_recreate_staging(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    root, live, _source_bytes = _prepare_project_root(tmp_path)
    incident_id = "INC-20260925-120005-1234abcd"

    monkeypatch.setattr(module, "ROOT", root)
    result_path = root / "build" / "awb_debug_capture_result.json"
    monkeypatch.setattr(module, "RESULT_PATH", result_path)
    original_atomic_write = module._atomic_write_json

    def fail_only_result(path, payload):
        if Path(path) == result_path:
            raise OSError("synthetic result write failure")
        return original_atomic_write(path, payload)

    monkeypatch.setattr(module, "_atomic_write_json", fail_only_result)

    result = module.capture_incident(
        root=root,
        live_probe=lambda _root: live,
        incident_id=incident_id,
    )

    incidents = root / "debug" / "incidents"
    assert result["ok"] is True
    assert "synthetic result write failure" in result["result_write_error"]
    assert (incidents / incident_id / "manifest.json").is_file()
    assert list(incidents.glob(f".staging-{incident_id}-*")) == []
    assert not (incidents / f".reserve-{incident_id}").exists()


def test_malformed_live_probe_degrades_to_file_only_capture(tmp_path: Path):
    module = _load_module()
    root, _live, _source_bytes = _prepare_project_root(tmp_path)

    result = module.capture_incident(
        root=root,
        live_probe=lambda _root: {},
        incident_id="INC-20260925-120006-1234abcd",
    )

    incident = Path(result["incident_path"])
    analysis = json.loads((incident / "analysis.json").read_text(encoding="utf-8"))
    assert result["evidence_status"] == "PARTIAL"
    assert analysis["live_probe_status"] == "UNAVAILABLE"
    assert "live probe schema" in analysis["live_probe_error"]


def test_stale_launcher_runtime_log_prevents_complete_evidence(tmp_path: Path):
    module = _load_module()
    root, live, _source_bytes = _prepare_project_root(tmp_path)
    raw_runtime = root / "debug" / "runtime_error_log" / "blender_runtime.log"
    raw_runtime.write_text(
        "\n=== AWB BLENDER RUNTIME SESSION "
        "2026-09-25T00:00:00.000+00:00 "
        f"blend={root / 'different.blend'} ===\n",
        encoding="utf-8",
    )

    result = module.capture_incident(
        root=root,
        live_probe=lambda _root: live,
        incident_id="INC-20260925-120007-1234abcd",
    )

    incident = Path(result["incident_path"])
    manifest = json.loads((incident / "manifest.json").read_text(encoding="utf-8"))
    assert result["evidence_status"] == "PARTIAL"
    assert manifest["artifacts"]["blender_runtime_current"]["live_blend_match"] is False


def test_state_before_keeps_domain_specific_legacy_state_out_of_core_blender_state():
    module = _load_module()
    records = [
        {
            "session_id": "current",
            "seq": 1,
            "event": "TRANSFORM_BEGIN",
            "operation_id": "op",
            "state": {
                "mode": "POSE",
                "frame": 7,
                "active_object": "Rig",
                "selected_objects": ["Rig"],
                "selected_pose_bones": ["Bone"],
                "rigped_transform_drag": True,
                "semantic_transform_mode": "ROTATE",
            },
        },
        {
            "session_id": "current",
            "seq": 2,
            "event": "TRANSFORM_FAIL",
            "operation_id": "op",
            "state": {"mode": "POSE"},
        },
    ]

    checkpoint = module._state_before_checkpoint(
        "INC-test",
        records,
        current_session_id="current",
    )
    assert checkpoint["status"] == "AVAILABLE"
    assert "rigped_transform_drag" not in checkpoint["blender_state"]
    assert "semantic_transform_mode" not in checkpoint["blender_state"]
    assert checkpoint["source"]["legacy_state"]["rigped_transform_drag"] is True


def test_live_probe_payload_is_read_only_and_bb2_fields_are_not_fabricated():
    module = _load_module()
    code = module.LIVE_PROBE_CODE
    forbidden = (
        "bpy.ops",
        "persist_latest_replay_script",
        "trace_event(",
        "view_layer.update",
        "evaluated_depsgraph_get",
        "frame_set(",
    )
    for token in forbidden:
        assert token not in code
    assert '"status": "UNAVAILABLE_BB2"' in code


def test_schemas_are_generic_and_do_not_require_rigped():
    for name in (
        "AWB_INCIDENT_MANIFEST_V1.schema.json",
        "AWB_INCIDENT_TIMELINE_V1.schema.json",
        "AWB_DEBUG_CHECKPOINT_V1.schema.json",
    ):
        payload = json.loads((ROOT / "docs" / "DEBUG" / "schemas" / name).read_text(encoding="utf-8"))
        lowered = json.dumps(payload, ensure_ascii=False).lower()
        assert "rigped" not in lowered
