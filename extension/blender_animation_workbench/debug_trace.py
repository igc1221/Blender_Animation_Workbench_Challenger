from __future__ import annotations

import json
import math
import sys
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns
from typing import Any
from uuid import uuid4

import bpy
from bpy.app.handlers import persistent

from .debug_causal import (
    LifecycleHandlerRecursionGuard,
    OperationCausalRegistry,
    TraceCausalContext,
    TraceRouteOutcome,
    TraceTerminalStatus,
    new_trace_child,
    new_trace_root,
    validate_trace_evaluation_phase,
    validate_trace_lifecycle_phase,
    validate_trace_subsystem,
)

_TRACE_FILENAME = "awb_interaction_trace.jsonl"
_PRECISION_TRACE_FILENAME = "awb_precision_trace.jsonl"
_REPLAY_FILENAME = "awb_replay_latest.json"
_REPLAY_PREVIOUS_FILENAME = "awb_replay_previous.json"
_REPLAY_PREVIOUS2_FILENAME = "awb_replay_previous2.json"
_RUNTIME_CONTEXT_FILENAME = "awb_runtime_context.json"
_RUNTIME_ERRORS_FILENAME = "awb_runtime_errors.jsonl"
_TRACE_MAX_BYTES = 4 * 1024 * 1024
_PRECISION_TRACE_MAX_BYTES = 2 * 1024 * 1024
_TAIL_READ_CHUNK_BYTES = 64 * 1024
_EXPLICIT_TERMINAL_EVENTS: dict[
    str,
    tuple[TraceTerminalStatus, str, TraceRouteOutcome],
] = {
    "TRANSFORM_COMMIT": (
        TraceTerminalStatus.FINISHED,
        "commit",
        TraceRouteOutcome.CLAIMED,
    ),
    "TRANSFORM_CANCEL": (
        TraceTerminalStatus.CANCELLED,
        "cancel",
        TraceRouteOutcome.CLAIMED,
    ),
    "TRANSFORM_FAIL": (
        TraceTerminalStatus.FAILED,
        "fail",
        TraceRouteOutcome.CLAIMED,
    ),
    "CONTACT_COMMIT": (
        TraceTerminalStatus.FINISHED,
        "commit",
        TraceRouteOutcome.CLAIMED,
    ),
    "CONTACT_FAIL": (
        TraceTerminalStatus.FAILED,
        "fail",
        TraceRouteOutcome.CLAIMED,
    ),
}

_SESSION_ID = uuid4().hex
_SEQUENCE = 0
_PRECISION_SEQUENCE = 0
_LAST_FRAME: tuple[int, float] | None = None
_WRITE_GUARD = False
_PRECISION_WRITE_GUARD = False
_REPLAY_EXECUTION_ACTIVE = False
_OPERATION_PARENTS: dict[str, str] = {}
_OPERATION_CAUSAL = OperationCausalRegistry(max_bindings=2048)
_LIFECYCLE_HANDLER_GUARD = LifecycleHandlerRecursionGuard()
_DEPSGRAPH_TRACE_ENABLED = False
_DEPSGRAPH_TRACE_SAMPLE_NS = 100_000_000
_DEPSGRAPH_LAST_TRACE_NS = 0
_DEPSGRAPH_DROPPED_SAMPLES = 0
_DEPSGRAPH_SAMPLE_ACTIVE = False
_SCRUB_STATE: dict[str, Any] | None = None
_PRECISION_PREVIOUS_BONES: dict[tuple[str, str], dict[str, Any]] = {}
_LAST_TRACE_RECORD: dict[str, Any] | None = None
_ORIGINAL_SYS_EXCEPTHOOK = sys.excepthook
_ORIGINAL_THREADING_EXCEPTHOOK = getattr(threading, "excepthook", None)


def set_replay_execution_active(active: bool) -> None:
    global _REPLAY_EXECUTION_ACTIVE
    _REPLAY_EXECUTION_ACTIVE = bool(active)


def replay_execution_active() -> bool:
    return bool(_REPLAY_EXECUTION_ACTIVE)


def new_trace_operation_id(prefix: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(prefix)).strip("-")
    return f"{safe or 'awb'}:{uuid4().hex}"


def new_trace_causal_root(prefix: str = "awb") -> TraceCausalContext:
    return new_trace_root(prefix)


def new_trace_causal_child(
    parent: TraceCausalContext,
    prefix: str = "span",
) -> TraceCausalContext:
    return new_trace_child(parent, prefix)


def bind_operation_causal_context(
    operation_id: str | None,
    causal: TraceCausalContext,
) -> TraceCausalContext | None:
    return _OPERATION_CAUSAL.bind(operation_id, causal)


def operation_causal_context(operation_id: str | None) -> TraceCausalContext | None:
    return _OPERATION_CAUSAL.get(operation_id)


def release_operation_causal_context(operation_id: str | None) -> None:
    _OPERATION_CAUSAL.forget(operation_id)
    if operation_id is not None:
        _OPERATION_PARENTS.pop(str(operation_id), None)


def set_depsgraph_trace_enabled(enabled: bool) -> None:
    global _DEPSGRAPH_TRACE_ENABLED
    _DEPSGRAPH_TRACE_ENABLED = bool(enabled)
    pre_handlers = getattr(bpy.app.handlers, "depsgraph_update_pre", None)
    post_handlers = getattr(bpy.app.handlers, "depsgraph_update_post", None)
    if _DEPSGRAPH_TRACE_ENABLED:
        if pre_handlers is not None and _trace_depsgraph_update_pre not in pre_handlers:
            pre_handlers.append(_trace_depsgraph_update_pre)
        if post_handlers is not None and _trace_depsgraph_update_post not in post_handlers:
            post_handlers.append(_trace_depsgraph_update_post)
    else:
        if pre_handlers is not None and _trace_depsgraph_update_pre in pre_handlers:
            pre_handlers.remove(_trace_depsgraph_update_pre)
        if post_handlers is not None and _trace_depsgraph_update_post in post_handlers:
            post_handlers.remove(_trace_depsgraph_update_post)


def depsgraph_trace_enabled() -> bool:
    return bool(_DEPSGRAPH_TRACE_ENABLED)


def link_trace_operation(
    child_operation_id: str | None,
    parent_operation_id: str | None,
) -> TraceCausalContext | None:
    if not child_operation_id or not parent_operation_id or child_operation_id == parent_operation_id:
        return None
    child_id = str(child_operation_id)
    parent_id = str(parent_operation_id)
    _OPERATION_PARENTS[child_id] = parent_id
    return _OPERATION_CAUSAL.link(
        child_id,
        parent_id,
        child_prefix="operation",
    )


def trace_scrub_begin(context, *, source: str) -> None:
    global _SCRUB_STATE
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    frame = int(getattr(scene, "frame_current", 0))
    subframe = float(getattr(scene, "frame_subframe", 0.0))
    if _SCRUB_STATE is not None:
        trace_scrub_end(context, cancelled=False)
    _PRECISION_PREVIOUS_BONES.clear()
    _SCRUB_STATE = {
        "source": str(source),
        "start_frame": frame,
        "start_subframe": subframe,
        "min_frame": frame,
        "max_frame": frame,
        "frame_change_count": 0,
    }
    trace_event(
        "INPUT",
        "SCRUB_BEGIN",
        context=context,
        source=str(source),
        start_frame=frame,
        start_subframe=subframe,
    )
    _precision_frame_sample(scene)


def trace_scrub_end(context, *, cancelled: bool) -> None:
    global _SCRUB_STATE
    state = _SCRUB_STATE
    if state is None:
        return
    scene = getattr(context, "scene", None)
    end_frame = int(getattr(scene, "frame_current", state["start_frame"])) if scene is not None else state["start_frame"]
    end_subframe = float(getattr(scene, "frame_subframe", state["start_subframe"])) if scene is not None else state["start_subframe"]
    trace_event(
        "INPUT",
        "SCRUB_END",
        context=context,
        source=state["source"],
        start_frame=state["start_frame"],
        start_subframe=state["start_subframe"],
        end_frame=end_frame,
        end_subframe=end_subframe,
        min_frame=state["min_frame"],
        max_frame=state["max_frame"],
        frame_change_count=state["frame_change_count"],
        cancelled=bool(cancelled),
    )
    _SCRUB_STATE = None


def _debug_root_path() -> Path:
    filepath = str(getattr(bpy.data, "filepath", "") or "")
    if filepath:
        blend_dir = Path(filepath).resolve().parent
        if blend_dir.name.casefold() == "build":
            return blend_dir.parent / "debug"
        if (
            blend_dir.name.casefold() == "golden"
            and blend_dir.parent.name.casefold() == "baselines"
        ):
            return blend_dir.parent.parent / "debug"
        return blend_dir / "debug"
    tempdir = str(getattr(bpy.app, "tempdir", "") or "")
    if tempdir:
        return Path(tempdir).resolve() / "awb_debug"
    return Path.cwd() / "debug"


def debug_root_path() -> str:
    return str(_debug_root_path())


def _trace_path() -> Path:
    return _debug_root_path() / _TRACE_FILENAME


def trace_path() -> str:
    return str(_trace_path())


def _precision_trace_path() -> Path:
    return _trace_path().with_name(_PRECISION_TRACE_FILENAME)


def precision_trace_path() -> str:
    return str(_precision_trace_path())


def _replay_path() -> Path:
    return _trace_path().with_name(_REPLAY_FILENAME)


def replay_path() -> str:
    return str(_replay_path())


def _replay_previous_path() -> Path:
    return _replay_path().with_name(_REPLAY_PREVIOUS_FILENAME)


def _replay_previous2_path() -> Path:
    return _replay_path().with_name(_REPLAY_PREVIOUS2_FILENAME)


def _runtime_error_root_path() -> Path:
    return _debug_root_path() / "runtime_error_log"


def _runtime_context_path() -> Path:
    return _runtime_error_root_path() / _RUNTIME_CONTEXT_FILENAME


def _runtime_errors_path() -> Path:
    return _runtime_error_root_path() / _RUNTIME_ERRORS_FILENAME


def runtime_context_path() -> str:
    return str(_runtime_context_path())


def runtime_errors_path() -> str:
    return str(_runtime_errors_path())


def _write_runtime_context_snapshot(record: dict[str, Any]) -> None:
    try:
        path = _runtime_context_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError:
        pass


def _append_runtime_error_record(
    source: str,
    exc_type: str,
    message: str,
    traceback_text: str,
    *,
    context=None,
) -> None:
    try:
        last_trace = _LAST_TRACE_RECORD or {}
        record = {
            "schema": "awb-runtime-error/v1",
            "utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "session_id": _SESSION_ID,
            "source": str(source),
            "error_type": str(exc_type),
            "error": str(message),
            "traceback": str(traceback_text),
            "state": _context_snapshot(context),
            "last_trace": {
                "seq": last_trace.get("seq"),
                "channel": last_trace.get("channel"),
                "event": last_trace.get("event"),
                "operation_id": last_trace.get("operation_id"),
                "parent_operation_id": last_trace.get("parent_operation_id"),
                "trace_id": last_trace.get("trace_id"),
                "span_id": last_trace.get("span_id"),
                "parent_span_id": last_trace.get("parent_span_id"),
                "subsystem": last_trace.get("subsystem"),
                "lifecycle_phase": last_trace.get("lifecycle_phase"),
                "evaluation_phase": last_trace.get("evaluation_phase"),
                "terminal_status": last_trace.get("terminal_status"),
                "route_outcome": last_trace.get("route_outcome"),
                "state": last_trace.get("state"),
                "data": last_trace.get("data"),
            },
            "last_observed_causal": {
                "session_id": last_trace.get("session_id"),
                "seq": last_trace.get("seq"),
                "trace_id": last_trace.get("trace_id"),
                "span_id": last_trace.get("span_id"),
                "parent_span_id": last_trace.get("parent_span_id"),
                "subsystem": last_trace.get("subsystem"),
                "lifecycle_phase": last_trace.get("lifecycle_phase"),
                "evaluation_phase": last_trace.get("evaluation_phase"),
            },
        }
        path = _runtime_errors_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    except Exception:  # noqa: BLE001, S110 -- diagnostics must never block Blender
        pass


def _runtime_excepthook(exc_type, exc_value, exc_tb) -> None:
    traceback_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _append_runtime_error_record(
        "sys.excepthook",
        getattr(exc_type, "__name__", str(exc_type)),
        str(exc_value),
        traceback_text,
        context=getattr(bpy, "context", None),
    )
    _ORIGINAL_SYS_EXCEPTHOOK(exc_type, exc_value, exc_tb)


def _runtime_threading_excepthook(args) -> None:
    traceback_text = "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    )
    _append_runtime_error_record(
        "threading.excepthook",
        getattr(args.exc_type, "__name__", str(args.exc_type)),
        str(args.exc_value),
        traceback_text,
        context=getattr(bpy, "context", None),
    )
    if _ORIGINAL_THREADING_EXCEPTHOOK is not None:
        _ORIGINAL_THREADING_EXCEPTHOOK(args)


def _read_replay_script_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_replay_script_file(path: Path, script: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(script, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _rotate_replay_history_for_new_session(script: dict[str, Any]) -> None:
    current_path = _replay_path()
    current = _read_replay_script_file(current_path)
    if current is None:
        return

    current_session = str(current.get("source_session_id") or "")
    incoming_session = str(script.get("source_session_id") or "")
    if not incoming_session or incoming_session == current_session:
        return

    previous_path = _replay_previous_path()
    previous2_path = _replay_previous2_path()
    previous = _read_replay_script_file(previous_path)
    if previous is not None:
        _write_replay_script_file(previous2_path, previous)
    _write_replay_script_file(previous_path, current)


def _archive_existing_session_file(path: Path) -> None:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return
        archive = path.with_name(f"{path.stem}.previous{path.suffix}")
        if archive.exists():
            archive.unlink()
        path.replace(archive)
    except OSError:
        # Diagnostics must never block Blender startup.
        return


def _archive_runtime_errors_for_new_session() -> None:
    path = _runtime_errors_path()
    records = _read_jsonl_tail(path, 1)
    if not records:
        return
    recorded_session = str(records[-1].get("session_id") or "")
    if recorded_session == _SESSION_ID:
        return
    _archive_existing_session_file(path)


def _read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    limit = max(1, int(limit))
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            chunks: list[bytes] = []
            newline_count = 0
            while position > 0 and newline_count <= limit:
                read_size = min(_TAIL_READ_CHUNK_BYTES, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
        raw = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
        records: list[dict[str, Any]] = []
        for line in raw.splitlines()[-limit:]:
            try:
                item = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(item, dict):
                records.append(item)
        return records
    except OSError:
        return []


def read_recent_trace_events(limit: int = 80, *, latest_session_only: bool = True) -> list[dict[str, Any]]:
    records = _read_jsonl_tail(_trace_path(), limit)
    if not latest_session_only or not records:
        return records
    latest_session = records[-1].get("session_id")
    if not latest_session:
        return records
    return [item for item in records if item.get("session_id") == latest_session]


def read_recent_precision_events(limit: int = 120, *, latest_session_only: bool = True) -> list[dict[str, Any]]:
    records = _read_jsonl_tail(_precision_trace_path(), limit)
    if not latest_session_only or not records:
        return records
    latest_session = records[-1].get("session_id")
    if not latest_session:
        return records
    return [item for item in records if item.get("session_id") == latest_session]


def summarize_recent_precision_events(limit: int = 600) -> dict[str, Any]:
    records = read_recent_precision_events(limit, latest_session_only=True)
    if not records:
        return {
            "sample_count": 0,
            "frame_range": None,
            "sampled_bones": [],
            "largest_one_frame_rotation_steps": [],
            "latest_limbs": [],
        }

    largest_by_bone: dict[str, dict[str, Any]] = {}
    sampled_bones: set[str] = set()
    skipped_step_count = 0
    one_frame_step_count = 0

    for record in records:
        frame = int(record.get("frame", 0))
        for bone_name, bone in (record.get("bones") or {}).items():
            sampled_bones.add(str(bone_name))
            frame_delta = bone.get("frame_delta")
            one_frame_rotation = bone.get("one_frame_rotation_step_deg")
            if frame_delta is not None and abs(abs(float(frame_delta)) - 1.0) > 1e-6:
                skipped_step_count += 1
            if one_frame_rotation is None:
                continue
            one_frame_step_count += 1
            candidate = {
                "bone": str(bone_name),
                "frame": frame,
                "frame_delta": float(frame_delta) if frame_delta is not None else None,
                "rotation_step_deg": float(one_frame_rotation),
                "translation_step": bone.get("one_frame_translation_step"),
                "roles": bone.get("roles") or (),
            }
            existing = largest_by_bone.get(str(bone_name))
            if existing is None or candidate["rotation_step_deg"] > existing["rotation_step_deg"]:
                largest_by_bone[str(bone_name)] = candidate

    largest = sorted(
        largest_by_bone.values(),
        key=lambda item: float(item["rotation_step_deg"]),
        reverse=True,
    )
    return {
        "sample_count": len(records),
        "frame_range": [
            min(int(item.get("frame", 0)) for item in records),
            max(int(item.get("frame", 0)) for item in records),
        ],
        "sampled_bones": sorted(sampled_bones),
        "one_frame_step_count": one_frame_step_count,
        "skipped_step_count": skipped_step_count,
        "largest_one_frame_rotation_steps": largest[:16],
        "latest_selected_pose_bones": records[-1].get("selected_pose_bones") or (),
        "latest_limbs": records[-1].get("limbs") or (),
    }


def _build_replay_script_from_records(
    events: list[dict[str, Any]],
    precision: list[dict[str, Any]],
) -> dict[str, Any]:
    session_id = events[-1].get("session_id") if events else None
    actions: list[dict[str, Any]] = []
    scrub_begin: dict[str, Any] | None = None

    for event in events:
        if bool(event.get("replay_execution", False)):
            continue
        data = event.get("data") or {}
        replay_action = data.get("replay_action")
        if isinstance(replay_action, dict):
            action = dict(replay_action)
            action["source_event"] = str(event.get("event") or "")
            action["source_seq"] = event.get("seq")
            action["operation_id"] = event.get("operation_id")
            actions.append(action)
            continue

        event_name = str(event.get("event") or "")
        if event_name == "SCRUB_BEGIN":
            scrub_begin = event
            continue
        if event_name != "SCRUB_END":
            continue

        begin_ns = int(scrub_begin.get("monotonic_ns", 0)) if scrub_begin is not None else 0
        end_ns = int(event.get("monotonic_ns", 0))
        frames: list[tuple[int, float]] = []
        for sample in precision:
            sample_ns = int(sample.get("monotonic_ns", 0))
            if sample_ns < begin_ns or sample_ns > end_ns:
                continue
            frame_item = (
                int(sample.get("frame", 0)),
                float(sample.get("subframe", 0.0)),
            )
            if not frames or frames[-1] != frame_item:
                frames.append(frame_item)

        if not frames:
            start_frame = int(data.get("start_frame", 0))
            start_subframe = float(data.get("start_subframe", 0.0))
            end_frame = int(data.get("end_frame", start_frame))
            end_subframe = float(data.get("end_subframe", start_subframe))
            frames = [(start_frame, start_subframe)]
            if frames[-1] != (end_frame, end_subframe):
                frames.append((end_frame, end_subframe))

        end_state = event.get("state") or {}
        actions.append(
            {
                "kind": "SCRUB",
                "source_event": "SCRUB_END",
                "source_seq": event.get("seq"),
                "source": str(data.get("source", "SCRUB")),
                "controls": tuple(end_state.get("selected_pose_bones") or ()),
                "frames": [
                    {"frame": frame, "subframe": subframe}
                    for frame, subframe in frames
                ],
                "cancelled": bool(data.get("cancelled", False)),
            }
        )
        scrub_begin = None

    return {
        "schema": "awb-semantic-replay/v1",
        "source_session_id": session_id,
        "blend_file": (
            str((events[-1].get("state") or {}).get("blend_file") or "")
            if events
            else ""
        ),
        "action_count": len(actions),
        "actions": actions,
    }


def build_latest_replay_script(
    *,
    event_limit: int = 4000,
    precision_limit: int = 12000,
) -> dict[str, Any]:
    events = read_recent_trace_events(event_limit, latest_session_only=True)
    precision = read_recent_precision_events(precision_limit, latest_session_only=True)
    return _build_replay_script_from_records(events, precision)


def build_previous_replay_script(
    *,
    event_limit: int = 4000,
    precision_limit: int = 12000,
) -> dict[str, Any]:
    event_path = _trace_path().with_name(f"{_trace_path().stem}.previous{_trace_path().suffix}")
    precision_path = _precision_trace_path().with_name(
        f"{_precision_trace_path().stem}.previous{_precision_trace_path().suffix}"
    )
    events = _read_jsonl_tail(event_path, event_limit)
    precision = _read_jsonl_tail(precision_path, precision_limit)
    if events:
        latest_session = events[-1].get("session_id")
        if latest_session:
            events = [item for item in events if item.get("session_id") == latest_session]
            precision = [item for item in precision if item.get("session_id") == latest_session]
    return _build_replay_script_from_records(events, precision)


def recover_previous_replay_script() -> str:
    script = build_previous_replay_script()
    path = _replay_path()
    if int(script.get("action_count", 0)) <= 0:
        return str(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(
            json.dumps(script, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError:
        return str(path)
    return str(path)


def persist_latest_replay_script() -> str:
    script = build_latest_replay_script()
    path = _replay_path()
    if int(script.get("action_count", 0)) <= 0 and path.exists():
        return str(path)
    try:
        _rotate_replay_history_for_new_session(script)
        _write_replay_script_file(path, script)
    except OSError:
        return str(path)
    return str(path)


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return str(value)


def _context_snapshot(context) -> dict[str, Any]:
    context = context or getattr(bpy, "context", None)
    scene = getattr(context, "scene", None) if context is not None else None
    active_object = getattr(context, "active_object", None) if context is not None else None
    selected_objects = []
    selected_pose_bones = []
    try:
        selected_objects = [str(obj.name) for obj in tuple(getattr(context, "selected_objects", ()) or ())]
    except (ReferenceError, RuntimeError):
        pass
    try:
        selected_pose_bones = [
            str(bone.name) for bone in tuple(getattr(context, "selected_pose_bones", ()) or ())
        ]
    except (ReferenceError, RuntimeError):
        pass

    frame = int(getattr(scene, "frame_current", 0)) if scene is not None else None
    subframe = float(getattr(scene, "frame_subframe", 0.0)) if scene is not None else None
    auto_key = bool(getattr(scene, "baw_auto_key_enabled", False)) if scene is not None else False
    semantic_mode = (
        str(getattr(scene, "baw_rigped_semantic_transform_mode", "NONE"))
        if scene is not None
        else "NONE"
    )
    tool_settings = getattr(scene, "tool_settings", None) if scene is not None else None
    window_manager = getattr(context, "window_manager", None) if context is not None else None
    return {
        "blend_file": str(getattr(bpy.data, "filepath", "") or ""),
        "mode": str(getattr(context, "mode", "")) if context is not None else "",
        "frame": frame,
        "subframe": subframe,
        "active_object": str(getattr(active_object, "name", "")) if active_object is not None else None,
        "selected_objects": selected_objects,
        "selected_pose_bones": selected_pose_bones,
        "awb_auto_key": auto_key,
        "native_auto_key": bool(getattr(tool_settings, "use_keyframe_insert_auto", False)),
        "semantic_transform_mode": semantic_mode,
        "trackbar_frame_drag": bool(
            getattr(window_manager, "baw_trackbar_frame_drag_active", False)
        ) if window_manager is not None else False,
        "rigped_transform_drag": bool(
            getattr(window_manager, "baw_rigped_semantic_move_drag_active", False)
        ) if window_manager is not None else False,
    }


def _rotate_if_needed(path: Path, *, max_bytes: int = _TRACE_MAX_BYTES) -> None:
    try:
        if not path.exists() or path.stat().st_size < int(max_bytes):
            return
        archive = path.with_name(f"{path.stem}.previous{path.suffix}")
        if archive.exists():
            archive.unlink()
        path.replace(archive)
    except OSError:
        # Tracing is diagnostic-only and must never interfere with animation.
        return


def _precision_fallback_pose_bones(context) -> tuple[tuple[Any, Any, tuple[str, ...]], ...]:
    try:
        selected = tuple(getattr(context, "selected_pose_bones", ()) or ())
    except (ReferenceError, RuntimeError):
        return ()
    active_object = getattr(context, "active_object", None)
    if active_object is None:
        return ()
    ordered: list[tuple[Any, Any, tuple[str, ...]]] = []
    seen: set[tuple[str, str]] = set()
    for pose_bone in selected:
        current = pose_bone
        depth = 0
        while current is not None and depth < 3:
            name = str(getattr(current, "name", ""))
            key = (str(getattr(active_object, "name", "")), name)
            if name and key not in seen:
                role = ("SELECTED",) if depth == 0 else (f"PARENT_{depth}",)
                ordered.append((active_object, current, role))
                seen.add(key)
            current = getattr(current, "parent", None)
            depth += 1
    return tuple(ordered)


def _precision_semantic_limb_targets(
    context,
    scene,
) -> tuple[
    tuple[tuple[Any, Any, tuple[str, ...]], ...],
    tuple[dict[str, Any], ...],
]:
    """Resolve full generated Rigped limb(s) touched by the native selection.

    Precision tracing must follow the whole semantic dependency domain rather
    than only the currently selected public control. This lets a Calf selection
    still capture the terminal Foot plus hidden IK target/pole/solver authority.
    """

    try:
        from .character_metadata import resolve_character
        from .phase4_contact_authoring import _selected_mappings
        from .phase4_contact_model import AWB_CONTACT_STATE_PROPERTY, type_for_state_value
        from .rigped_contract import resolve_rigped_target
        from .semantic_adapter import control_context_for_context
    except Exception:  # noqa: BLE001 -- tracing imports must degrade to fallback only
        return _precision_fallback_pose_bones(context), ()

    try:
        resolution = resolve_rigped_target(scene, control_context_for_context(context))
        target = resolution.target
        if target is None or not target.selected_binding_ids:
            return _precision_fallback_pose_bones(context), ()
        view = resolve_character(scene, target.character_id)
        mappings = _selected_mappings(view, target.selected_binding_ids)
    except Exception:  # noqa: BLE001 -- precision tracing is best-effort diagnostics
        return _precision_fallback_pose_bones(context), ()

    if not mappings:
        return _precision_fallback_pose_bones(context), ()

    ordered: list[tuple[Any, Any, tuple[str, ...]]] = []
    roles_by_key: dict[tuple[str, str], set[str]] = {}
    target_by_key: dict[tuple[str, str], tuple[Any, Any]] = {}
    limb_rows: list[dict[str, Any]] = []

    def add_resolved(resolved, role: str) -> None:
        if resolved is None:
            return
        owner = getattr(resolved, "owner_object", None)
        pose_bone = getattr(resolved, "target", None)
        if owner is None or not isinstance(pose_bone, bpy.types.PoseBone):
            return
        key = (str(getattr(owner, "name", "")), str(getattr(pose_bone, "name", "")))
        if not key[1]:
            return
        roles_by_key.setdefault(key, set()).add(role)
        target_by_key[key] = (owner, pose_bone)

    for mapping, capability in mappings:
        for resolved in tuple(getattr(capability, "fk_controls", ()) or ()):
            add_resolved(resolved, "FK")
        add_resolved(getattr(capability, "authored_terminal", None), "TERMINAL")
        for resolved in tuple(getattr(capability, "result_controls", ()) or ()):
            add_resolved(resolved, "RESULT")
        add_resolved(getattr(capability, "result_terminal", None), "RESULT_TERMINAL")

        native_ik = getattr(capability, "native_ik", None)
        if native_ik is not None:
            add_resolved(getattr(native_ik, "solver_owner", None), "IK_SOLVER")
            add_resolved(getattr(native_ik, "ik_target", None), "IK_TARGET")
            add_resolved(getattr(native_ik, "pole_target", None), "IK_POLE")

        solver_resolved = getattr(native_ik, "solver_owner", None) if native_ik is not None else None
        solver_bone = getattr(solver_resolved, "target", None)
        raw_contact_state = None
        contact_type = None
        if isinstance(solver_bone, bpy.types.PoseBone):
            try:
                raw_contact_state = float(solver_bone.get(AWB_CONTACT_STATE_PROPERTY, -1.0))
                resolved_type = type_for_state_value(raw_contact_state)
                contact_type = resolved_type.value if resolved_type is not None else None
            except Exception:  # noqa: BLE001 -- malformed trace data must not affect animation
                raw_contact_state = None
                contact_type = None

        constraint = getattr(native_ik, "constraint", None) if native_ik is not None else None
        terminal_constraint = getattr(capability, "terminal_ik_constraint", None)
        limb_rows.append(
            {
                "mapping_id": str(getattr(mapping, "mapping_id", "")),
                "contact_type": contact_type,
                "contact_state_raw": raw_contact_state,
                "ik_influence": (
                    float(getattr(constraint, "influence", 0.0))
                    if constraint is not None
                    else None
                ),
                "terminal_ik_influence": (
                    float(getattr(terminal_constraint, "influence", 0.0))
                    if terminal_constraint is not None
                    else None
                ),
                "pole_angle": (
                    float(getattr(constraint, "pole_angle", 0.0))
                    if constraint is not None
                    else None
                ),
                "solver_bone": (
                    str(getattr(solver_bone, "name", ""))
                    if solver_bone is not None
                    else None
                ),
                "ik_target": (
                    str(getattr(getattr(getattr(native_ik, "ik_target", None), "target", None), "name", ""))
                    if native_ik is not None
                    else None
                ),
                "pole_target": (
                    str(getattr(getattr(getattr(native_ik, "pole_target", None), "target", None), "name", ""))
                    if native_ik is not None
                    else None
                ),
            }
        )

    for key, pair in target_by_key.items():
        owner, pose_bone = pair
        ordered.append((owner, pose_bone, tuple(sorted(roles_by_key.get(key, ())))))
    return tuple(ordered), tuple(limb_rows)


def _quaternion_step_deg(
    previous: tuple[float, ...] | None,
    current: tuple[float, float, float, float],
) -> float | None:
    if previous is None or len(previous) != 4:
        return None
    dot = abs(sum(float(a) * float(b) for a, b in zip(previous, current)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _precision_frame_sample(scene) -> None:
    global _PRECISION_SEQUENCE, _PRECISION_WRITE_GUARD
    if _PRECISION_WRITE_GUARD or _SCRUB_STATE is None:
        return
    context = getattr(bpy, "context", None)
    active_object = getattr(context, "active_object", None) if context is not None else None
    if active_object is None or str(getattr(active_object, "type", "")) != "ARMATURE":
        return
    targets, limbs = _precision_semantic_limb_targets(context, scene)
    if not targets:
        return
    _PRECISION_WRITE_GUARD = True
    try:
        path = _precision_trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(path, max_bytes=_PRECISION_TRACE_MAX_BYTES)
        payload: dict[str, Any] = {}
        object_name = str(getattr(active_object, "name", ""))
        current_frame = int(getattr(scene, "frame_current", 0))
        current_subframe = float(getattr(scene, "frame_subframe", 0.0))
        current_time = float(current_frame) + current_subframe
        for owner, pose_bone, roles in targets:
            owner_name = str(getattr(owner, "name", ""))
            bone_name = str(getattr(pose_bone, "name", ""))
            world_matrix = owner.matrix_world @ pose_bone.matrix
            world_location = tuple(float(value) for value in world_matrix.to_translation())
            world_quaternion = tuple(float(value) for value in world_matrix.to_quaternion())
            basis_quaternion = tuple(float(value) for value in pose_bone.matrix_basis.to_quaternion())
            key = (owner_name, bone_name)
            previous = _PRECISION_PREVIOUS_BONES.get(key)
            rotation_step = _quaternion_step_deg(
                previous.get("world_quaternion") if previous is not None else None,
                world_quaternion,
            )
            translation_step = None
            frame_delta = None
            if previous is not None:
                prev_loc = previous.get("world_location")
                if prev_loc is not None and len(prev_loc) == 3:
                    translation_step = math.sqrt(
                        sum((float(a) - float(b)) ** 2 for a, b in zip(prev_loc, world_location))
                    )
                previous_time = previous.get("frame_time")
                if previous_time is not None:
                    frame_delta = current_time - float(previous_time)
            is_one_frame_step = (
                frame_delta is not None
                and abs(abs(float(frame_delta)) - 1.0) <= 1e-6
            )
            _PRECISION_PREVIOUS_BONES[key] = {
                "world_quaternion": world_quaternion,
                "world_location": world_location,
                "frame_time": current_time,
            }
            payload[bone_name] = {
                "owner_object": owner_name,
                "roles": roles,
                "world_location": world_location,
                "world_quaternion": world_quaternion,
                "basis_quaternion": basis_quaternion,
                "frame_delta": frame_delta,
                "rotation_step_deg": rotation_step,
                "translation_step": translation_step,
                "one_frame_rotation_step_deg": rotation_step if is_one_frame_step else None,
                "one_frame_translation_step": translation_step if is_one_frame_step else None,
            }
        _PRECISION_SEQUENCE += 1
        record = {
            "schema": "awb-precision-trace/v1",
            "session_id": _SESSION_ID,
            "seq": _PRECISION_SEQUENCE,
            "replay_execution": bool(_REPLAY_EXECUTION_ACTIVE),
            "utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "monotonic_ns": monotonic_ns(),
            "frame": current_frame,
            "subframe": current_subframe,
            "source": str(_SCRUB_STATE.get("source", "SCRUB")),
            "active_object": object_name,
            "selected_pose_bones": [
                str(getattr(item, "name", "")) for item in tuple(getattr(context, "selected_pose_bones", ()) or ())
            ],
            "limbs": limbs,
            "bones": payload,
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    except Exception:  # noqa: BLE001, S110 -- diagnostics must never alter user operations
        pass
    finally:
        _PRECISION_WRITE_GUARD = False


def trace_event(
    channel: str,
    event: str,
    *,
    operation_id: str | None = None,
    parent_operation_id: str | None = None,
    causal: TraceCausalContext | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    subsystem: str | None = None,
    lifecycle_phase: str | None = None,
    evaluation_phase: str | None = None,
    terminal_status: TraceTerminalStatus | None = None,
    route_outcome: TraceRouteOutcome | None = None,
    dropped: int | None = None,
    truncated: bool | None = None,
    context=None,
    **data,
) -> None:
    global _LAST_TRACE_RECORD, _SEQUENCE, _WRITE_GUARD
    if _WRITE_GUARD:
        return
    _WRITE_GUARD = True
    try:
        path = _trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(path)
        _SEQUENCE += 1
        resolved_parent = parent_operation_id
        if resolved_parent is None and operation_id is not None:
            resolved_parent = _OPERATION_PARENTS.get(str(operation_id))
        bound_causal = causal or operation_causal_context(operation_id)
        resolved_trace_id = trace_id if trace_id is not None else (
            bound_causal.trace_id if bound_causal is not None else None
        )
        resolved_span_id = span_id if span_id is not None else (
            bound_causal.span_id if bound_causal is not None else None
        )
        resolved_parent_span_id = (
            parent_span_id
            if parent_span_id is not None
            else (bound_causal.parent_span_id if bound_causal is not None else None)
        )
        terminal_contract = _EXPLICIT_TERMINAL_EVENTS.get(str(event))
        if terminal_contract is not None:
            mapped_terminal, mapped_phase, mapped_route = terminal_contract
            if terminal_status is None:
                terminal_status = mapped_terminal
            if lifecycle_phase is None:
                lifecycle_phase = mapped_phase
            if route_outcome is None:
                route_outcome = mapped_route
            if subsystem is None:
                subsystem = "operator"
        resolved_terminal = (
            terminal_status.value
            if isinstance(terminal_status, TraceTerminalStatus)
            else (str(terminal_status) if terminal_status is not None else None)
        )
        resolved_route = (
            route_outcome.value
            if isinstance(route_outcome, TraceRouteOutcome)
            else (str(route_outcome) if route_outcome is not None else None)
        )
        record = {
            "schema": "awb-interaction-trace/v1",
            "session_id": _SESSION_ID,
            "seq": _SEQUENCE,
            "replay_execution": bool(_REPLAY_EXECUTION_ACTIVE),
            "utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "monotonic_ns": monotonic_ns(),
            "channel": str(channel),
            "event": str(event),
            "operation_id": operation_id,
            "parent_operation_id": resolved_parent,
            "trace_id": resolved_trace_id,
            "span_id": resolved_span_id,
            "parent_span_id": resolved_parent_span_id,
            "span_origin": (
                bound_causal.span_origin if bound_causal is not None else None
            ),
            "subsystem": str(subsystem) if subsystem is not None else None,
            "lifecycle_phase": str(lifecycle_phase) if lifecycle_phase is not None else None,
            "evaluation_phase": str(evaluation_phase) if evaluation_phase is not None else None,
            "terminal_status": resolved_terminal,
            "route_outcome": resolved_route,
            "dropped": int(dropped) if dropped is not None else None,
            "truncated": bool(truncated) if truncated is not None else None,
            "state": _context_snapshot(context),
            "data": _json_safe(data),
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        _LAST_TRACE_RECORD = record
        _write_runtime_context_snapshot(record)
        if (
            not _REPLAY_EXECUTION_ACTIVE
            and str(event) in {
                "AUTO_KEY_TOGGLE",
                "CONTACT_COMMIT",
                "TRANSFORM_COMMIT",
                "SCRUB_END",
            }
        ):
            persist_latest_replay_script()
        if terminal_status is not None and operation_id is not None:
            _OPERATION_CAUSAL.forget(operation_id)
            _OPERATION_PARENTS.pop(str(operation_id), None)
    except Exception:  # noqa: BLE001, S110 -- trace persistence is strictly non-blocking
        # Never let diagnostics alter the user's Blender operation.
        pass
    finally:
        _WRITE_GUARD = False


def trace_lifecycle_event(
    event: str,
    *,
    subsystem: str,
    phase: str,
    operation_id: str | None = None,
    parent_operation_id: str | None = None,
    causal: TraceCausalContext | None = None,
    evaluation_phase: str | None = None,
    terminal_status: TraceTerminalStatus | None = None,
    route_outcome: TraceRouteOutcome | None = None,
    dropped: int | None = None,
    truncated: bool | None = None,
    context=None,
    **data,
) -> None:
    resolved_subsystem = validate_trace_subsystem(subsystem)
    resolved_phase = validate_trace_lifecycle_phase(phase)
    resolved_evaluation = validate_trace_evaluation_phase(evaluation_phase)
    if terminal_status is not None and not isinstance(terminal_status, TraceTerminalStatus):
        raise TypeError("terminal_status must be TraceTerminalStatus or None")
    if route_outcome is not None and not isinstance(route_outcome, TraceRouteOutcome):
        raise TypeError("route_outcome must be TraceRouteOutcome or None")
    trace_event(
        "LIFECYCLE",
        event,
        operation_id=operation_id,
        parent_operation_id=parent_operation_id,
        causal=causal,
        subsystem=resolved_subsystem,
        lifecycle_phase=resolved_phase,
        evaluation_phase=resolved_evaluation,
        terminal_status=terminal_status,
        route_outcome=route_outcome,
        dropped=dropped,
        truncated=truncated,
        context=context,
        **data,
    )
    if terminal_status is not None and operation_id is not None:
        _OPERATION_CAUSAL.forget(operation_id)
        _OPERATION_PARENTS.pop(str(operation_id), None)


def trace_exception(
    channel: str,
    event: str,
    exc: BaseException,
    *,
    operation_id: str | None = None,
    context=None,
    **data,
) -> None:
    traceback_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    trace_event(
        channel,
        event,
        operation_id=operation_id,
        context=context,
        error_type=type(exc).__name__,
        error=str(exc),
        traceback=traceback_text,
        **data,
    )
    _append_runtime_error_record(
        "trace_exception",
        type(exc).__name__,
        str(exc),
        traceback_text,
        context=context,
    )


def _trace_standalone_handler_event(
    event: str,
    *,
    phase: str,
    evaluation_phase: str,
    **data,
) -> None:
    if not _LIFECYCLE_HANDLER_GUARD.enter():
        return
    try:
        trace_lifecycle_event(
            event,
            subsystem="handler",
            phase=phase,
            evaluation_phase=evaluation_phase,
            context=getattr(bpy, "context", None),
            **data,
        )
    finally:
        _LIFECYCLE_HANDLER_GUARD.exit()


@persistent
def _trace_load_pre(*_args) -> None:
    _trace_standalone_handler_event(
        "FILE_LOAD_PRE",
        phase="handler_pre",
        evaluation_phase="pre_handler",
    )
    _OPERATION_PARENTS.clear()
    _OPERATION_CAUSAL.reset()


@persistent
def _trace_load_post_fail(*_args) -> None:
    _OPERATION_PARENTS.clear()
    _OPERATION_CAUSAL.reset()
    _trace_standalone_handler_event(
        "FILE_LOAD_FAIL",
        phase="fail",
        evaluation_phase="post_handler",
        terminal_status=TraceTerminalStatus.FAILED,
    )


@persistent
def _trace_save_pre(*_args) -> None:
    _trace_standalone_handler_event(
        "FILE_SAVE_PRE",
        phase="handler_pre",
        evaluation_phase="pre_handler",
    )


@persistent
def _trace_save_post(*_args) -> None:
    _trace_standalone_handler_event(
        "FILE_SAVE_POST",
        phase="handler_post",
        evaluation_phase="post_handler",
        terminal_status=TraceTerminalStatus.FINISHED,
    )


@persistent
def _trace_save_post_fail(*_args) -> None:
    _trace_standalone_handler_event(
        "FILE_SAVE_FAIL",
        phase="fail",
        evaluation_phase="post_handler",
        terminal_status=TraceTerminalStatus.FAILED,
    )


@persistent
def _trace_undo_pre(*_args) -> None:
    _trace_standalone_handler_event(
        "UNDO_PRE",
        phase="handler_pre",
        evaluation_phase="pre_handler",
    )


@persistent
def _trace_undo_post(*_args) -> None:
    _trace_standalone_handler_event(
        "UNDO_POST",
        phase="handler_post",
        evaluation_phase="post_handler",
        terminal_status=TraceTerminalStatus.FINISHED,
    )


@persistent
def _trace_redo_pre(*_args) -> None:
    _trace_standalone_handler_event(
        "REDO_PRE",
        phase="handler_pre",
        evaluation_phase="pre_handler",
    )


@persistent
def _trace_redo_post(*_args) -> None:
    _trace_standalone_handler_event(
        "REDO_POST",
        phase="handler_post",
        evaluation_phase="post_handler",
        terminal_status=TraceTerminalStatus.FINISHED,
    )


@persistent
def _trace_depsgraph_update_pre(*_args) -> None:
    global _DEPSGRAPH_LAST_TRACE_NS, _DEPSGRAPH_DROPPED_SAMPLES, _DEPSGRAPH_SAMPLE_ACTIVE
    _DEPSGRAPH_SAMPLE_ACTIVE = False
    if not _DEPSGRAPH_TRACE_ENABLED:
        return
    now = monotonic_ns()
    if now - _DEPSGRAPH_LAST_TRACE_NS < _DEPSGRAPH_TRACE_SAMPLE_NS:
        _DEPSGRAPH_DROPPED_SAMPLES += 1
        return
    _DEPSGRAPH_LAST_TRACE_NS = now
    dropped = _DEPSGRAPH_DROPPED_SAMPLES
    _DEPSGRAPH_DROPPED_SAMPLES = 0
    if not _LIFECYCLE_HANDLER_GUARD.enter():
        return
    _DEPSGRAPH_SAMPLE_ACTIVE = True
    try:
        trace_lifecycle_event(
            "DEPSGRAPH_UPDATE_PRE",
            subsystem="depsgraph",
            phase="depsgraph_pre",
            evaluation_phase="pre_handler",
            dropped=dropped,
            context=getattr(bpy, "context", None),
        )
    finally:
        _LIFECYCLE_HANDLER_GUARD.exit()


@persistent
def _trace_depsgraph_update_post(*_args) -> None:
    global _DEPSGRAPH_SAMPLE_ACTIVE
    if not _DEPSGRAPH_TRACE_ENABLED or not _DEPSGRAPH_SAMPLE_ACTIVE:
        return
    _DEPSGRAPH_SAMPLE_ACTIVE = False
    if not _LIFECYCLE_HANDLER_GUARD.enter():
        return
    try:
        trace_lifecycle_event(
            "DEPSGRAPH_UPDATE_POST",
            subsystem="depsgraph",
            phase="depsgraph_post",
            evaluation_phase="post_handler",
            context=getattr(bpy, "context", None),
        )
    finally:
        _LIFECYCLE_HANDLER_GUARD.exit()


@persistent
def _trace_frame_change_post(scene, *_args) -> None:
    global _LAST_FRAME
    if not _LIFECYCLE_HANDLER_GUARD.enter():
        return
    try:
        current = (
            int(getattr(scene, "frame_current", 0)),
            float(getattr(scene, "frame_subframe", 0.0)),
        )
        previous = _LAST_FRAME
        _LAST_FRAME = current
        if previous is None or previous == current:
            return
        if _SCRUB_STATE is not None:
            _SCRUB_STATE["frame_change_count"] = int(_SCRUB_STATE["frame_change_count"]) + 1
            _SCRUB_STATE["min_frame"] = min(int(_SCRUB_STATE["min_frame"]), current[0])
            _SCRUB_STATE["max_frame"] = max(int(_SCRUB_STATE["max_frame"]), current[0])
            _precision_frame_sample(scene)
            return
        trace_event(
            "INPUT",
            "FRAME_CHANGE",
            subsystem="animation",
            lifecycle_phase="handler_post",
            evaluation_phase="post_handler",
            context=bpy.context,
            from_frame=previous[0],
            from_subframe=previous[1],
            to_frame=current[0],
            to_subframe=current[1],
        )
    finally:
        _LIFECYCLE_HANDLER_GUARD.exit()


@persistent
def _trace_load_post(*_args) -> None:
    global _LAST_FRAME, _SCRUB_STATE
    _SCRUB_STATE = None
    _OPERATION_PARENTS.clear()
    _OPERATION_CAUSAL.reset()
    # At addon registration time Blender may not yet expose the target .blend
    # filepath, so the initial trace path can live under the temp directory.
    # Once load_post fires, preserve and rotate any existing trace beside the
    # actual loaded file before appending this session's FILE_LOAD event.
    persist_latest_replay_script()
    _archive_existing_session_file(_trace_path())
    _archive_existing_session_file(_precision_trace_path())
    _archive_runtime_errors_for_new_session()
    _PRECISION_PREVIOUS_BONES.clear()
    scene = getattr(bpy.context, "scene", None)
    _LAST_FRAME = (
        int(getattr(scene, "frame_current", 0)),
        float(getattr(scene, "frame_subframe", 0.0)),
    ) if scene is not None else None
    trace_lifecycle_event(
        "FILE_LOAD",
        subsystem="handler",
        phase="handler_post",
        evaluation_phase="post_handler",
        terminal_status=TraceTerminalStatus.FINISHED,
        context=bpy.context,
    )


def _register_handler_list(name: str, callback) -> None:
    handlers = getattr(bpy.app.handlers, name, None)
    if handlers is not None and callback not in handlers:
        handlers.append(callback)


def _unregister_handler_list(name: str, callback) -> None:
    handlers = getattr(bpy.app.handlers, name, None)
    if handlers is not None and callback in handlers:
        handlers.remove(callback)


def register_debug_trace_handlers() -> None:
    global _LAST_FRAME
    # Preserve the outgoing session's semantic replay before rotating its
    # bounded trace files. Replay-only sessions produce zero user actions, and
    # persist_latest_replay_script() deliberately leaves an existing compact
    # replay untouched in that case.
    persist_latest_replay_script()
    _archive_existing_session_file(_trace_path())
    _archive_existing_session_file(_precision_trace_path())
    _archive_runtime_errors_for_new_session()
    _OPERATION_PARENTS.clear()
    _OPERATION_CAUSAL.reset()
    _PRECISION_PREVIOUS_BONES.clear()
    scene = getattr(bpy.context, "scene", None)
    _LAST_FRAME = (
        int(getattr(scene, "frame_current", 0)),
        float(getattr(scene, "frame_subframe", 0.0)),
    ) if scene is not None else None
    _register_handler_list("frame_change_post", _trace_frame_change_post)
    _register_handler_list("load_pre", _trace_load_pre)
    _register_handler_list("load_post", _trace_load_post)
    _register_handler_list("load_post_fail", _trace_load_post_fail)
    _register_handler_list("save_pre", _trace_save_pre)
    _register_handler_list("save_post", _trace_save_post)
    _register_handler_list("save_post_fail", _trace_save_post_fail)
    _register_handler_list("undo_pre", _trace_undo_pre)
    _register_handler_list("undo_post", _trace_undo_post)
    _register_handler_list("redo_pre", _trace_redo_pre)
    _register_handler_list("redo_post", _trace_redo_post)
    # Depsgraph tracing is intentionally opt-in because it is high-rate.
    if _DEPSGRAPH_TRACE_ENABLED:
        set_depsgraph_trace_enabled(True)
    sys.excepthook = _runtime_excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _runtime_threading_excepthook
    trace_lifecycle_event(
        "SESSION_START",
        subsystem="runtime",
        phase="invoke",
        evaluation_phase="live_context",
        context=bpy.context,
        trace_path=trace_path(),
    )


def unregister_debug_trace_handlers() -> None:
    trace_lifecycle_event(
        "SESSION_END",
        subsystem="runtime",
        phase="commit",
        evaluation_phase="live_context",
        terminal_status=TraceTerminalStatus.FINISHED,
        context=bpy.context,
    )
    _unregister_handler_list("frame_change_post", _trace_frame_change_post)
    _unregister_handler_list("load_pre", _trace_load_pre)
    _unregister_handler_list("load_post", _trace_load_post)
    _unregister_handler_list("load_post_fail", _trace_load_post_fail)
    _unregister_handler_list("save_pre", _trace_save_pre)
    _unregister_handler_list("save_post", _trace_save_post)
    _unregister_handler_list("save_post_fail", _trace_save_post_fail)
    _unregister_handler_list("undo_pre", _trace_undo_pre)
    _unregister_handler_list("undo_post", _trace_undo_post)
    _unregister_handler_list("redo_pre", _trace_redo_pre)
    _unregister_handler_list("redo_post", _trace_redo_post)
    set_depsgraph_trace_enabled(False)
    _OPERATION_CAUSAL.reset()
    if sys.excepthook is _runtime_excepthook:
        sys.excepthook = _ORIGINAL_SYS_EXCEPTHOOK
    if (
        hasattr(threading, "excepthook")
        and threading.excepthook is _runtime_threading_excepthook
        and _ORIGINAL_THREADING_EXCEPTHOOK is not None
    ):
        threading.excepthook = _ORIGINAL_THREADING_EXCEPTHOOK
