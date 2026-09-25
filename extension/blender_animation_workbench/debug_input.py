# Diagnostic instrumentation must never block Blender product input on observation failure.
# ruff: noqa: BLE001, S110, S112
from __future__ import annotations

import copy
import json
import math
from collections import OrderedDict, defaultdict, deque
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import bpy

try:
    from bpy.app.handlers import persistent
except ImportError:
    def persistent(function):
        return function

SCHEMA = "awb-raw-input-trace/v1"
_TRACE_FILENAME = "awb_raw_input_trace.jsonl"
_TRACE_MAX_BYTES = 2 * 1024 * 1024
_RECORD_MAX_BYTES = 8 * 1024
_RING_MAX_RECORDS = 2048
_PENDING_MAX_RECORDS = 1024
_ACTIVE_OWNER_MAX = 256
_MOUSEMOVE_SAMPLE_EVERY = 8
_MOUSEMOVE_MAX_PER_OWNER = 256
_KEYMAP_MATCH_MAX = 24
_KEYMAP_PROBE_ID = "awb.debug_raw_input_probe"
_OBSERVER_ID = "wm.awb_raw_input_observer"
_WRAPPED_SENTINEL = "__awb_debug_input_wrapped__"
_OWNER_ATTR = "_awb_raw_input_owner_id"
_COMMON_MODAL_DISCRETE_TYPES = {
    "LEFTMOUSE",
    "RIGHTMOUSE",
    "MIDDLEMOUSE",
    "ESC",
    "RET",
    "NUMPAD_ENTER",
    "SPACE",
}
_RECORD_KIND_BY_EVENT = {
    "KEYMAP_RAW_EVENT": "RAW_INPUT",
    "OBSERVER_RAW_EVENT": "RAW_INPUT",
    "MODAL_RAW_EVENT": "RAW_INPUT",
    "OPERATOR_INGRESS": "OPERATOR_INGRESS",
    "MODAL_OWNER_BEGIN": "MODAL_OWNER_BEGIN",
    "MODAL_ROUTE": "MODAL_ROUTE",
    "MODAL_OWNER_TERMINAL": "MODAL_OWNER_TERMINAL",
    "OPERATOR_INVOKE_FAILURE": "OPERATOR_INVOKE_FAILURE",
    "MODAL_OWNER_FAILURE": "MODAL_OWNER_FAILURE",
}
_AWB_OPERATOR_PREFIXES = ("baw.", "awb.")
_AWB_NATIVE_OPERATORS = {
    "screen.screen_full_area",
    "view3d.move",
    "view3d.rotate",
    "view3d.toggle_xray",
    "view3d.view_axis",
    "view3d.view_camera",
    "view3d.view_selected",
}

_SESSION_ID = uuid4().hex
_RAW_INPUT_SEQ = 0
_OWNER_SEQ = 0
_RING: deque[dict[str, Any]] = deque(maxlen=_RING_MAX_RECORDS)
_PENDING_BY_FINGERPRINT: dict[tuple[Any, ...], deque[int]] = defaultdict(deque)
_PENDING_ORDER: deque[int] = deque()
_PENDING_FINGERPRINT_BY_SEQ: dict[int, tuple[Any, ...]] = {}
_ACTIVE_OWNERS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_RING_DROPPED = 0
_PENDING_DROPPED = 0
_FILE_DROPPED = 0
_MOUSEMOVE_SEEN = 0
_MOUSEMOVE_SAMPLED = 0
_MOUSEMOVE_DROPPED = 0
_MOUSEMOVE_OWNER_SEEN: dict[str, int] = {}
_MOUSEMOVE_OWNER_SAMPLED: dict[str, int] = {}
_RAW_INPUT_TRACE_PATH_OVERRIDE: Path | None = None
_OBSERVER_ENABLED = False
_OBSERVER_GENERATION = 0
_OBSERVER_WINDOW_TOKENS: set[int] = set()
_OBSERVED_EVENT_TYPES: set[str] = set(_COMMON_MODAL_DISCRETE_TYPES)


class BAW_OT_raw_input_probe(bpy.types.Operator):
    """Transparent diagnostic probe inserted ahead of AWB-owned keymap entries."""

    bl_idname = _KEYMAP_PROBE_ID
    bl_label = "AWB Raw Input Probe"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    def invoke(self, context, event):
        try:
            record_keymap_probe_event(event, context)
        except Exception:
            pass
        return {"PASS_THROUGH"}

    def execute(self, _context):
        return {"PASS_THROUGH"}


class BAW_OT_raw_input_observer(bpy.types.Operator):
    """Passive per-window raw-input observer that never claims product input."""

    bl_idname = _OBSERVER_ID
    bl_label = "AWB Raw Input Observer"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    _generation: int = 0
    _window_token: int = 0

    def invoke(self, context, _event):
        window = getattr(context, "window", None)
        if window is None:
            return {"CANCELLED"}
        try:
            token = int(window.as_pointer())
        except Exception:
            token = id(window)
        self._generation = _OBSERVER_GENERATION
        self._window_token = token
        _OBSERVER_WINDOW_TOKENS.add(token)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not _OBSERVER_ENABLED or self._generation != _OBSERVER_GENERATION:
            _OBSERVER_WINDOW_TOKENS.discard(self._window_token)
            return {"CANCELLED"}
        try:
            record_observer_event(event, context)
        except Exception:
            pass
        return {"PASS_THROUGH"}


def raw_input_trace_path() -> str:
    if _RAW_INPUT_TRACE_PATH_OVERRIDE is not None:
        return str(_RAW_INPUT_TRACE_PATH_OVERRIDE)
    try:
        from . import debug_trace

        return str(Path(debug_trace.debug_root_path()) / _TRACE_FILENAME)
    except Exception:
        return str(Path.cwd() / "debug" / _TRACE_FILENAME)


def read_recent_raw_input_events(limit: int = 256) -> list[dict[str, Any]]:
    bounded_limit = max(0, min(int(limit), _RING_MAX_RECORDS))
    if not bounded_limit:
        return []
    return copy.deepcopy(list(_RING)[-bounded_limit:])


def raw_input_stats() -> dict[str, int]:
    return {
        "raw_input_seq": _RAW_INPUT_SEQ,
        "ring_dropped": _RING_DROPPED,
        "pending_dropped": _PENDING_DROPPED,
        "file_dropped": _FILE_DROPPED,
        "mousemove_seen": _MOUSEMOVE_SEEN,
        "mousemove_sampled": _MOUSEMOVE_SAMPLED,
        "mousemove_dropped": _MOUSEMOVE_DROPPED,
        "pending_keymap_events": len(_PENDING_FINGERPRINT_BY_SEQ),
    }


def reset_raw_input_trace() -> None:
    """Reset in-memory state for focused tests and isolated diagnostic sessions."""
    global _RAW_INPUT_SEQ, _OWNER_SEQ, _RING_DROPPED, _PENDING_DROPPED
    global _FILE_DROPPED, _MOUSEMOVE_SEEN, _MOUSEMOVE_SAMPLED, _MOUSEMOVE_DROPPED

    _RAW_INPUT_SEQ = 0
    _OWNER_SEQ = 0
    _RING.clear()
    _PENDING_BY_FINGERPRINT.clear()
    _PENDING_ORDER.clear()
    _PENDING_FINGERPRINT_BY_SEQ.clear()
    _ACTIVE_OWNERS.clear()
    _MOUSEMOVE_OWNER_SEEN.clear()
    _MOUSEMOVE_OWNER_SAMPLED.clear()
    _RING_DROPPED = 0
    _PENDING_DROPPED = 0
    _FILE_DROPPED = 0
    _MOUSEMOVE_SEEN = 0
    _MOUSEMOVE_SAMPLED = 0
    _MOUSEMOVE_DROPPED = 0


def _replay_context() -> tuple[bool, str | None]:
    try:
        from . import debug_trace

        active = bool(debug_trace.replay_execution_active())
        execution_id = debug_trace.replay_execution_id() if active else None
        return active, str(execution_id) if execution_id else None
    except Exception:
        return False, None


def _safe_event_value(event: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(event, name, default)
    except Exception:
        return default
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else default
    if isinstance(value, str):
        return value[:128]
    return default


_EVENT_FIELDS = (
    "type",
    "value",
    "unicode",
    "ascii",
    "shift",
    "ctrl",
    "alt",
    "oskey",
    "key_modifier",
    "direction",
    "is_repeat",
    "mouse_x",
    "mouse_y",
    "mouse_region_x",
    "mouse_region_y",
    "x",
    "y",
    "pressure",
    "tilt",
)
_FINGERPRINT_FIELDS = (
    "type",
    "value",
    "shift",
    "ctrl",
    "alt",
    "oskey",
    "key_modifier",
    "direction",
    "is_repeat",
    "mouse_x",
    "mouse_y",
)


def _event_values(event: Any) -> dict[str, Any]:
    return {
        name: value
        for name in _EVENT_FIELDS
        if (value := _safe_event_value(event, name)) is not None
    }


def _minimal_context(context: Any) -> dict[str, str]:
    if context is None:
        return {}
    result: dict[str, str] = {}
    for field in ("mode",):
        value = _safe_event_value(context, field)
        if value is not None:
            result[field] = str(value)[:64]
    for owner_name, field in (("area", "type"), ("region", "type"), ("space_data", "type")):
        try:
            owner = getattr(context, owner_name, None)
        except Exception:
            owner = None
        value = _safe_event_value(owner, field)
        if value is not None:
            result[f"{owner_name}_{field}"] = str(value)[:64]
    return result


def _fingerprint(event: Any, replay: tuple[bool, str | None] | None = None) -> tuple[Any, ...]:
    replay_active, replay_id = replay or _replay_context()
    values = _event_values(event)
    return (
        tuple((field, values.get(field)) for field in _FINGERPRINT_FIELDS),
        bool(replay_active),
        str(replay_id) if replay_id else None,
    )


def _fingerprint_json(fingerprint: tuple[Any, ...]) -> list[Any]:
    event_fields, replay_active, replay_id = fingerprint
    return [list(pair) for pair in event_fields] + [replay_active, replay_id]


def _trace_path() -> Path:
    return Path(raw_input_trace_path())


def _rotate_if_needed(path: Path, incoming_bytes: int) -> bool:
    try:
        if incoming_bytes > _TRACE_MAX_BYTES:
            return False
        if not path.exists() or path.stat().st_size + incoming_bytes <= _TRACE_MAX_BYTES:
            return True
        archive = path.with_name(f"{path.stem}.previous{path.suffix}")
        if archive.exists():
            archive.unlink()
        path.replace(archive)
    except OSError:
        return False
    return True


def _append_record_to_file(record: dict[str, Any]) -> None:
    global _FILE_DROPPED

    path = _trace_path()
    try:
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        encoded = payload.encode("utf-8")
        if len(encoded) > _RECORD_MAX_BYTES:
            _FILE_DROPPED += 1
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if not _rotate_if_needed(path, len(encoded)):
            _FILE_DROPPED += 1
            return
        with path.open("ab") as handle:
            handle.write(encoded)
    except (OSError, TypeError, ValueError):
        _FILE_DROPPED += 1


def _emit(
    name: str,
    *,
    source: str,
    context: Any = None,
    event: Any = None,
    owner_id: str | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global _RAW_INPUT_SEQ, _RING_DROPPED

    replay_active, replay_id = _replay_context()
    event_values = _event_values(event) if event is not None else {}
    fingerprint = _fingerprint(event, (replay_active, replay_id)) if event is not None else None
    if len(_RING) == _RING_MAX_RECORDS:
        _RING_DROPPED += 1
    _RAW_INPUT_SEQ += 1
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "record_kind": _RECORD_KIND_BY_EVENT.get(str(name), str(name)),
        "session_id": _SESSION_ID,
        "raw_input_seq": _RAW_INPUT_SEQ,
        "utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "event_name": str(name),
        "event": event_values if event is not None else None,
        "source": str(source),
        "origin": "REPLAY" if replay_active else "LIVE_USER",
        "replay_execution": replay_active,
        "replay_execution_id": replay_id,
        "owner_id": str(owner_id) if owner_id else None,
        "context": _minimal_context(context),
        "event_data": event_values,
        "event_fingerprint": _fingerprint_json(fingerprint) if fingerprint is not None else None,
        "counters": {
            "ring_dropped": _RING_DROPPED,
            "pending_dropped": _PENDING_DROPPED,
            "file_dropped": _FILE_DROPPED,
            "mousemove_seen": _MOUSEMOVE_SEEN,
            "mousemove_sampled": _MOUSEMOVE_SAMPLED,
            "mousemove_dropped": _MOUSEMOVE_DROPPED,
        },
    }
    if fields:
        record.update(fields)
    _RING.append(record)
    _append_record_to_file(record)
    return record


def _prune_pending_order() -> None:
    while _PENDING_ORDER and _PENDING_ORDER[0] not in _PENDING_FINGERPRINT_BY_SEQ:
        _PENDING_ORDER.popleft()
    if len(_PENDING_ORDER) > _PENDING_MAX_RECORDS * 2:
        _PENDING_ORDER_copy = [
            seq for seq in _PENDING_ORDER if seq in _PENDING_FINGERPRINT_BY_SEQ
        ]
        _PENDING_ORDER.clear()
        _PENDING_ORDER.extend(_PENDING_ORDER_copy)


def _enqueue_keymap_event(fingerprint: tuple[Any, ...], raw_input_seq: int) -> None:
    global _PENDING_DROPPED

    _PENDING_BY_FINGERPRINT[fingerprint].append(raw_input_seq)
    _PENDING_FINGERPRINT_BY_SEQ[raw_input_seq] = fingerprint
    _PENDING_ORDER.append(raw_input_seq)
    while len(_PENDING_FINGERPRINT_BY_SEQ) > _PENDING_MAX_RECORDS:
        _prune_pending_order()
        if not _PENDING_ORDER:
            break
        dropped_seq = _PENDING_ORDER.popleft()
        dropped_fingerprint = _PENDING_FINGERPRINT_BY_SEQ.pop(dropped_seq, None)
        if dropped_fingerprint is None:
            continue
        queue = _PENDING_BY_FINGERPRINT.get(dropped_fingerprint)
        if queue:
            try:
                queue.remove(dropped_seq)
            except ValueError:
                pass
            if not queue:
                _PENDING_BY_FINGERPRINT.pop(dropped_fingerprint, None)
        _PENDING_DROPPED += 1
    _prune_pending_order()


def _consume_keymap_event(event: Any) -> int | None:
    fingerprint = _fingerprint(event)
    queue = _PENDING_BY_FINGERPRINT.get(fingerprint)
    if not queue:
        return None
    while queue:
        raw_input_seq = queue.popleft()
        if _PENDING_FINGERPRINT_BY_SEQ.pop(raw_input_seq, None) is not None:
            if not queue:
                _PENDING_BY_FINGERPRINT.pop(fingerprint, None)
            _prune_pending_order()
            return raw_input_seq
    _PENDING_BY_FINGERPRINT.pop(fingerprint, None)
    return None


def _keymap_item_matches_event(item: Any, event: Any) -> bool:
    values = _event_values(event)
    if str(getattr(item, "type", "")) != str(values.get("type", "")):
        return False
    if str(getattr(item, "value", "")) != str(values.get("value", "")):
        return False
    if not bool(getattr(item, "any", False)):
        for modifier in ("shift", "ctrl", "alt", "oskey"):
            if bool(getattr(item, modifier, False)) != bool(values.get(modifier, False)):
                return False
    key_modifier = str(getattr(item, "key_modifier", "NONE") or "NONE")
    if key_modifier != "NONE" and key_modifier != str(values.get("key_modifier", "NONE") or "NONE"):
        return False
    direction = str(getattr(item, "direction", "ANY") or "ANY")
    return direction == "ANY" or direction == str(values.get("direction", "ANY") or "ANY")


def _operator_poll(operator_id: str) -> bool | None:
    try:
        category, name = str(operator_id).split(".", 1)
        operator = getattr(getattr(bpy.ops, category), name)
        return bool(operator.poll())
    except Exception:
        return None


def _binding_id(keymap_name: str, item: Any) -> str:
    signature = _keymap_item_signature(item)
    signature_text = ",".join(str(value) for value in signature)
    return f"{keymap_name!s}|{getattr(item, 'idname', '')!s}|{signature_text}"


def _keymap_probe_evidence(context: Any, event: Any) -> dict[str, Any]:
    try:
        wm = getattr(context, "window_manager", None) or getattr(bpy.context, "window_manager", None)
        keyconfigs = getattr(wm, "keyconfigs", None)
        keyconfig = getattr(keyconfigs, "user", None)
    except Exception:
        keyconfig = None
    if keyconfig is None:
        return {
            "seen": True,
            "bounded": True,
            "authority": "keyconfigs.user",
            "complete": False,
            "truncated": False,
            "cross_keymap_ordering": "UNAVAILABLE",
            "bindings": [],
            "competing_matches": [],
        }

    bindings: list[dict[str, Any]] = []
    truncated = False
    try:
        keymaps = tuple(getattr(keyconfig, "keymaps", ()) or ())
    except Exception:
        keymaps = ()
    for keymap in keymaps:
        try:
            items = tuple(getattr(keymap, "keymap_items", ()) or ())
        except Exception:
            continue
        for item in items:
            try:
                operator_id = str(getattr(item, "idname", ""))
                if operator_id == _KEYMAP_PROBE_ID or not _keymap_item_matches_event(item, event):
                    continue
                enabled = bool(getattr(item, "active", True))
                owner = (
                    "AWB" if operator_id.startswith(_AWB_OPERATOR_PREFIXES) else "OTHER"
                )
                eligible = _operator_poll(operator_id) if enabled and owner == "AWB" else None
                bindings.append(
                    {
                        "binding_id": _binding_id(str(getattr(keymap, "name", "")), item),
                        "keymap": str(getattr(keymap, "name", "")),
                        "operator_id": operator_id,
                        "owner": owner,
                        "status": "ENABLED" if enabled else "DISABLED",
                        "enabled": enabled,
                        "eligible": eligible,
                    }
                )
                if len(bindings) >= _KEYMAP_MATCH_MAX:
                    truncated = True
                    break
            except Exception:
                continue
        if truncated:
            break

    active_matches = [binding for binding in bindings if binding.get("enabled") is True]
    awb_bindings = [binding for binding in bindings if binding.get("owner") == "AWB"]
    result: dict[str, Any] = {
        "seen": True,
        "bounded": True,
        "authority": "keyconfigs.user",
        "complete": not truncated,
        "truncated": truncated,
        "cross_keymap_ordering": (
            "UNRESOLVED" if len(active_matches) > 1 else "NOT_REQUIRED"
        ),
        "bindings": bindings,
        "competing_matches": active_matches,
    }
    if len(awb_bindings) == 1:
        result["binding"] = awb_bindings[0]
    return result


def _context_window_token(context: Any) -> int | None:
    window = getattr(context, "window", None) if context is not None else None
    if window is None:
        return None
    try:
        return int(window.as_pointer())
    except Exception:
        return id(window)


def _active_modal_owner_for_context(context: Any) -> str | None:
    window_token = _context_window_token(context)
    matches = [
        owner_id
        for owner_id, state in _ACTIVE_OWNERS.items()
        if state.get("window_token") in (None, window_token)
    ]
    return matches[0] if len(matches) == 1 else None


def _observer_active_for_context(context: Any) -> bool:
    token = _context_window_token(context)
    return bool(_OBSERVER_ENABLED and token is not None and token in _OBSERVER_WINDOW_TOKENS)


def _refresh_observed_event_types() -> set[str]:
    observed = set(_COMMON_MODAL_DISCRETE_TYPES)
    try:
        wm = getattr(getattr(bpy, "context", None), "window_manager", None)
        keyconfigs = getattr(wm, "keyconfigs", None)
        keyconfig = getattr(keyconfigs, "addon", None)
        keymaps = tuple(getattr(keyconfig, "keymaps", ()) or ()) if keyconfig else ()
    except Exception:
        keymaps = ()
    for keymap in keymaps:
        try:
            items = tuple(getattr(keymap, "keymap_items", ()) or ())
        except Exception:
            continue
        for item in items:
            try:
                operator_id = str(getattr(item, "idname", ""))
                event_type = str(getattr(item, "type", ""))
            except Exception:
                continue
            if (
                operator_id != _KEYMAP_PROBE_ID
                and operator_id.startswith(_AWB_OPERATOR_PREFIXES)
                and event_type not in {"", "NONE", "MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}
            ):
                observed.add(event_type)
    _OBSERVED_EVENT_TYPES.clear()
    _OBSERVED_EVENT_TYPES.update(observed)
    return set(_OBSERVED_EVENT_TYPES)


def record_observer_event(event: Any, context: Any = None) -> dict[str, Any] | None:
    event_type = str(_safe_event_value(event, "type", "") or "")
    owner_id = _active_modal_owner_for_context(context)
    sample_fields: dict[str, int] | None = None
    if event_type == "MOUSEMOVE":
        if owner_id is None:
            return None
        should_record, sample_fields = _should_record_modal_event(event, owner_id)
        if not should_record:
            return None
    elif event_type not in _OBSERVED_EVENT_TYPES:
        return None

    fields: dict[str, Any] = {}
    if event_type != "MOUSEMOVE":
        fields["keymap_probe"] = _keymap_probe_evidence(context, event)
    if owner_id is not None:
        fields["modal_entry"] = True
        fields["modal_owner_id"] = owner_id
    if sample_fields:
        fields.update(sample_fields)

    record = _emit(
        "OBSERVER_RAW_EVENT",
        source="PASS_THROUGH_OBSERVER",
        context=context,
        event=event,
        owner_id=owner_id,
        fields=fields,
    )
    event_value = str(_safe_event_value(event, "value", "") or "")
    if owner_id is not None or event_value != "RELEASE":
        fingerprint = _fingerprint(
            event,
            (record["replay_execution"], record["replay_execution_id"]),
        )
        _enqueue_keymap_event(fingerprint, int(record["raw_input_seq"]))
    return record


def _invoke_observer_for_window(window: Any) -> bool:
    if window is None:
        return False
    try:
        token = int(window.as_pointer())
    except Exception:
        token = id(window)
    if token in _OBSERVER_WINDOW_TOKENS:
        return True
    try:
        with bpy.context.temp_override(window=window):
            result = bpy.ops.wm.awb_raw_input_observer("INVOKE_DEFAULT")
    except Exception:
        return False
    return "RUNNING_MODAL" in set(result)


def _observer_reconcile_timer() -> float | None:
    if not _OBSERVER_ENABLED:
        return None
    try:
        wm = getattr(bpy.context, "window_manager", None)
        windows = tuple(getattr(wm, "windows", ()) or ())
    except Exception:
        windows = ()
    live_tokens: set[int] = set()
    for window in windows:
        try:
            live_tokens.add(int(window.as_pointer()))
        except Exception:
            live_tokens.add(id(window))
    _OBSERVER_WINDOW_TOKENS.intersection_update(live_tokens)
    for window in windows:
        _invoke_observer_for_window(window)
    return 1.0


@persistent
def _observer_load_post(_dummy: Any) -> None:
    _OBSERVER_WINDOW_TOKENS.clear()


def register_raw_input_observers() -> None:
    global _OBSERVER_ENABLED, _OBSERVER_GENERATION

    _OBSERVER_GENERATION += 1
    _OBSERVER_ENABLED = True
    _OBSERVER_WINDOW_TOKENS.clear()
    _refresh_observed_event_types()
    try:
        wm = getattr(bpy.context, "window_manager", None)
        windows = tuple(getattr(wm, "windows", ()) or ())
    except Exception:
        windows = ()
    for window in windows:
        _invoke_observer_for_window(window)
    try:
        if not bpy.app.timers.is_registered(_observer_reconcile_timer):
            bpy.app.timers.register(_observer_reconcile_timer, first_interval=1.0, persistent=True)
    except Exception:
        pass
    try:
        if _observer_load_post not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(_observer_load_post)
    except Exception:
        pass


def unregister_raw_input_observers() -> None:
    global _OBSERVER_ENABLED, _OBSERVER_GENERATION

    _OBSERVER_ENABLED = False
    _OBSERVER_GENERATION += 1
    try:
        if bpy.app.timers.is_registered(_observer_reconcile_timer):
            bpy.app.timers.unregister(_observer_reconcile_timer)
    except Exception:
        pass
    try:
        if _observer_load_post in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(_observer_load_post)
    except Exception:
        pass
    _OBSERVER_WINDOW_TOKENS.clear()


def record_keymap_probe_event(event: Any, context: Any = None) -> dict[str, Any]:
    keymap_probe = _keymap_probe_evidence(context, event)
    record = _emit(
        "KEYMAP_RAW_EVENT",
        source="KEYMAP_PROBE",
        context=context,
        event=event,
        fields={"keymap_probe": keymap_probe},
    )
    if str(_safe_event_value(event, "value", "")) != "RELEASE":
        fingerprint = _fingerprint(
            event,
            (record["replay_execution"], record["replay_execution_id"]),
        )
        _enqueue_keymap_event(fingerprint, int(record["raw_input_seq"]))
    return record


def record_operator_ingress(operator_id: str, context: Any, event: Any) -> dict[str, Any]:
    correlated_raw_input_seq = _consume_keymap_event(event) if event is not None else None
    correlation_status = (
        "CORRELATED" if correlated_raw_input_seq is not None else "UNCORRELATED"
    )
    return _emit(
        "OPERATOR_INGRESS",
        source="OPERATOR_INVOKE",
        context=context,
        event=event,
        fields={
            "operator_id": str(operator_id),
            "correlation_status": correlation_status,
            "correlated_raw_input_seq": correlated_raw_input_seq,
            "keymap_raw_input_seq": correlated_raw_input_seq,
        },
    )


def _normalize_operator_result(result: Any) -> list[str]:
    if result is None:
        return []
    if isinstance(result, str):
        return [result]
    try:
        return sorted(str(value) for value in result)
    except TypeError:
        return [str(result)]


def _operator_id(operator_class: type) -> str:
    return str(getattr(operator_class, "bl_idname", "") or operator_class.__name__)


def _new_modal_owner(operator_id: str, context: Any = None) -> str:
    global _OWNER_SEQ

    _OWNER_SEQ += 1
    owner_id = f"{_SESSION_ID}:modal:{_OWNER_SEQ}"
    _ACTIVE_OWNERS[owner_id] = {
        "operator_id": str(operator_id),
        "window_token": _context_window_token(context),
        "mousemove_seen": 0,
        "mousemove_sampled": 0,
    }
    while len(_ACTIVE_OWNERS) > _ACTIVE_OWNER_MAX:
        retired_owner, _ = _ACTIVE_OWNERS.popitem(last=False)
        _MOUSEMOVE_OWNER_SEEN.pop(retired_owner, None)
        _MOUSEMOVE_OWNER_SAMPLED.pop(retired_owner, None)
    return owner_id


def _safe_set_owner(instance: Any, owner_id: str) -> None:
    try:
        setattr(instance, _OWNER_ATTR, owner_id)
    except Exception:
        try:
            object.__setattr__(instance, _OWNER_ATTR, owner_id)
        except Exception:
            return


def _get_owner(instance: Any) -> str | None:
    try:
        value = getattr(instance, _OWNER_ATTR, None)
    except Exception:
        return None
    return str(value) if value else None


def _should_record_modal_event(
    event: Any,
    owner_id: str | None,
) -> tuple[bool, dict[str, int] | None]:
    global _MOUSEMOVE_SEEN, _MOUSEMOVE_SAMPLED, _MOUSEMOVE_DROPPED

    if str(_safe_event_value(event, "type", "")) != "MOUSEMOVE":
        return True, None
    _MOUSEMOVE_SEEN += 1
    owner_state = _ACTIVE_OWNERS.get(owner_id) if owner_id else None
    sample_key = owner_id if owner_state is not None else "UNCORRELATED_MODAL_OWNER"
    seen = _MOUSEMOVE_OWNER_SEEN.get(sample_key, 0) + 1
    _MOUSEMOVE_OWNER_SEEN[sample_key] = seen
    sampled = _MOUSEMOVE_OWNER_SAMPLED.get(sample_key, 0)
    if owner_state is not None:
        sampled = int(owner_state.get("mousemove_sampled", 0))
    capture = (
        (seen - 1) % _MOUSEMOVE_SAMPLE_EVERY == 0
        and sampled < _MOUSEMOVE_MAX_PER_OWNER
    )
    if capture:
        _MOUSEMOVE_SAMPLED += 1
        sampled += 1
        _MOUSEMOVE_OWNER_SAMPLED[sample_key] = sampled
        if owner_state is not None:
            owner_state["mousemove_seen"] = seen
            owner_state["mousemove_sampled"] = sampled
        return True, {"mousemove_sampled_index": sampled, "mousemove_seen_index": seen}
    _MOUSEMOVE_DROPPED += 1
    if owner_state is not None:
        owner_state["mousemove_seen"] = seen
    return False, None


def _record_modal_raw_event(
    operator_id: str,
    owner_id: str | None,
    context: Any,
    event: Any,
) -> dict[str, Any] | None:
    should_record, sample_fields = _should_record_modal_event(event, owner_id)
    if not should_record:
        return None
    fields: dict[str, Any] = {
        "operator_id": operator_id,
        "modal_entry": True,
        "modal_owner_id": owner_id,
    }
    if sample_fields:
        fields.update(sample_fields)
    return _emit(
        "MODAL_RAW_EVENT",
        source="MODAL_OWNER",
        context=context,
        event=event,
        owner_id=owner_id,
        fields=fields,
    )


def _terminal_modal_owner(owner_id: str | None) -> None:
    if owner_id:
        _ACTIVE_OWNERS.pop(owner_id, None)
        _MOUSEMOVE_OWNER_SEEN.pop(owner_id, None)
        _MOUSEMOVE_OWNER_SAMPLED.pop(owner_id, None)


def _owner_mousemove_summary(owner_id: str | None) -> dict[str, int]:
    owner_state = _ACTIVE_OWNERS.get(owner_id) if owner_id else None
    seen = int(owner_state.get("mousemove_seen", 0)) if owner_state else 0
    sampled = int(owner_state.get("mousemove_sampled", 0)) if owner_state else 0
    return {
        "mousemove_owner_seen": seen,
        "mousemove_owner_sampled": sampled,
        "mousemove_owner_dropped": max(0, seen - sampled),
    }


def wrap_operator_class(operator_class: type) -> None:
    if not isinstance(operator_class, type):
        return
    try:
        if not issubclass(operator_class, bpy.types.Operator):
            return
    except TypeError:
        return
    if operator_class in {BAW_OT_raw_input_probe, BAW_OT_raw_input_observer} or _operator_id(
        operator_class
    ) in {_KEYMAP_PROBE_ID, _OBSERVER_ID}:
        return

    for method_name in ("invoke", "modal"):
        original = operator_class.__dict__.get(method_name)
        if original is None or getattr(original, _WRAPPED_SENTINEL, False):
            continue
        if method_name == "invoke":
            wrapped = _wrap_invoke(operator_class, original)
        else:
            wrapped = _wrap_modal(operator_class, original)
        setattr(wrapped, _WRAPPED_SENTINEL, True)
        wrapped.__awb_debug_input_original__ = original
        setattr(operator_class, method_name, wrapped)


def _wrap_invoke(operator_class: type, original):
    operator_id = _operator_id(operator_class)

    @wraps(original)
    def invoke(self, context, event, *args, **kwargs):
        try:
            ingress = record_operator_ingress(operator_id, context, event)
        except Exception:
            ingress = {"raw_input_seq": None, "correlated_raw_input_seq": None}
        correlated_raw_input_seq = ingress.get("correlated_raw_input_seq")
        try:
            result = original(self, context, event, *args, **kwargs)
        except Exception as exc:
            try:
                _emit(
                    "OPERATOR_INVOKE_FAILURE",
                    source="OPERATOR_INVOKE",
                    context=context,
                    event=event,
                    fields={
                        "operator_id": operator_id,
                        "correlated_raw_input_seq": correlated_raw_input_seq,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:512],
                    },
                )
            except Exception:
                pass
            raise
        try:
            result_flags = _normalize_operator_result(result)
        except Exception:
            result_flags = []
        if "RUNNING_MODAL" in result_flags:
            try:
                owner_id = _new_modal_owner(operator_id, context)
                _safe_set_owner(self, owner_id)
                _emit(
                    "MODAL_OWNER_BEGIN",
                    source="OPERATOR_INVOKE",
                    context=context,
                    event=event,
                    owner_id=owner_id,
                    fields={
                        "operator_id": operator_id,
                        "active": True,
                        "correlated_raw_input_seq": correlated_raw_input_seq,
                    },
                )
            except Exception:
                pass
        return result

    return invoke


def _wrap_modal(operator_class: type, original):
    operator_id = _operator_id(operator_class)

    @wraps(original)
    def modal(self, context, event, *args, **kwargs):
        owner_id = _get_owner(self)
        if _observer_active_for_context(context):
            correlated_raw_input_seq = _consume_keymap_event(event)
        else:
            try:
                modal_raw = _record_modal_raw_event(operator_id, owner_id, context, event)
            except Exception:
                modal_raw = None
            correlated_raw_input_seq = (
                modal_raw.get("raw_input_seq") if isinstance(modal_raw, dict) else None
            )
        try:
            result = original(self, context, event, *args, **kwargs)
        except Exception as exc:
            try:
                owner_mousemove = _owner_mousemove_summary(owner_id)
                _emit(
                    "MODAL_OWNER_FAILURE",
                    source="MODAL_OWNER",
                    context=context,
                    event=event,
                    owner_id=owner_id,
                    fields={
                        "operator_id": operator_id,
                        "correlated_raw_input_seq": correlated_raw_input_seq,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:512],
                        **owner_mousemove,
                    },
                )
            except Exception:
                pass
            finally:
                _terminal_modal_owner(owner_id)
            raise
        try:
            result_flags = _normalize_operator_result(result)
        except Exception:
            result_flags = []
        try:
            _emit(
                "MODAL_ROUTE",
                source="MODAL_OWNER",
                context=context,
                event=event,
                owner_id=owner_id,
                fields={
                    "operator_id": operator_id,
                    "correlated_raw_input_seq": correlated_raw_input_seq,
                    "valid": True,
                    "result": result_flags,
                },
            )
        except Exception:
            pass
        if "FINISHED" in result_flags or "CANCELLED" in result_flags:
            terminal_status = "FINISHED" if "FINISHED" in result_flags else "CANCELLED"
            try:
                owner_mousemove = _owner_mousemove_summary(owner_id)
                _emit(
                    "MODAL_OWNER_TERMINAL",
                    source="MODAL_OWNER",
                    context=context,
                    event=event,
                    owner_id=owner_id,
                    fields={
                        "operator_id": operator_id,
                        "status": terminal_status,
                        **owner_mousemove,
                    },
                )
            except Exception:
                pass
            finally:
                _terminal_modal_owner(owner_id)
        return result

    return modal


def instrument_operator_classes(operator_classes) -> None:
    for operator_class in tuple(operator_classes):
        wrap_operator_class(operator_class)


def restore_operator_instrumentation(operator_classes) -> None:
    for operator_class in tuple(operator_classes):
        if not isinstance(operator_class, type):
            continue
        for method_name in ("invoke", "modal"):
            wrapped = operator_class.__dict__.get(method_name)
            if wrapped is None or not getattr(wrapped, _WRAPPED_SENTINEL, False):
                continue
            original = getattr(wrapped, "__awb_debug_input_original__", None)
            if original is not None:
                setattr(operator_class, method_name, original)


def _is_awb_relevant_operator(operator_id: str) -> bool:
    return operator_id.startswith(_AWB_OPERATOR_PREFIXES) or operator_id in _AWB_NATIVE_OPERATORS


def _keymap_item_signature(item: Any) -> tuple[Any, ...]:
    return (
        str(getattr(item, "type", "")),
        str(getattr(item, "value", "")),
        bool(getattr(item, "any", False)),
        bool(getattr(item, "shift", False)),
        bool(getattr(item, "ctrl", False)),
        bool(getattr(item, "alt", False)),
        bool(getattr(item, "oskey", False)),
        str(getattr(item, "key_modifier", "NONE") or "NONE"),
        str(getattr(item, "direction", "ANY") or "ANY"),
    )


def _remove_existing_probe_items(keyconfig: Any) -> None:
    try:
        keymaps = tuple(getattr(keyconfig, "keymaps", ()) or ())
    except Exception:
        return
    for keymap in keymaps:
        try:
            keymap_items = getattr(keymap, "keymap_items", None)
            if keymap_items is None:
                continue
            items = tuple(keymap_items)
        except Exception:
            continue
        for item in items:
            try:
                is_probe = str(getattr(item, "idname", "")) == _KEYMAP_PROBE_ID
            except Exception:
                continue
            if is_probe:
                try:
                    keymap_items.remove(item)
                except Exception:
                    pass


def _new_probe_item(keymap: Any, signature: tuple[Any, ...]) -> Any:
    event_type, value, any_modifier, shift, ctrl, alt, oskey, key_modifier, direction = signature
    options: dict[str, Any] = {"any": any_modifier, "head": True}
    if not any_modifier:
        options.update(shift=shift, ctrl=ctrl, alt=alt, oskey=oskey)
    if key_modifier != "NONE":
        options["key_modifier"] = key_modifier
    if direction != "ANY":
        options["direction"] = direction
    return keymap.keymap_items.new(_KEYMAP_PROBE_ID, event_type, value, **options)


def register_keymap_probes() -> None:
    try:
        wm = getattr(getattr(bpy, "context", None), "window_manager", None)
        keyconfigs = getattr(wm, "keyconfigs", None)
        keyconfig = getattr(keyconfigs, "addon", None)
    except Exception:
        return
    if keyconfig is None:
        return
    try:
        _remove_existing_probe_items(keyconfig)
        keymaps = tuple(getattr(keyconfig, "keymaps", ()) or ())
    except Exception:
        return
    for keymap in keymaps:
        signatures: set[tuple[Any, ...]] = set()
        try:
            items = tuple(getattr(keymap, "keymap_items", ()) or ())
        except Exception:
            continue
        for item in items:
            try:
                operator_id = str(getattr(item, "idname", ""))
                if operator_id == _KEYMAP_PROBE_ID or not _is_awb_relevant_operator(operator_id):
                    continue
                signature = _keymap_item_signature(item)
                if signature[0] in {"", "NONE", "MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
                    continue
                signatures.add(signature)
            except Exception:
                continue
        probe_signatures = set(signatures)
        for signature in tuple(signatures):
            if signature[1] == "PRESS":
                probe_signatures.add(
                    (
                        signature[0],
                        "RELEASE",
                        signature[2],
                        signature[3],
                        signature[4],
                        signature[5],
                        signature[6],
                        signature[7],
                        signature[8],
                    )
                )
        # Wildcard modifier entries cover exact variants of the same event in
        # this map; prefer one broad probe to prevent duplicate raw records.
        wildcard_groups = {
            (sig[0], sig[1], sig[7], sig[8]) for sig in probe_signatures if sig[2]
        }
        for signature in sorted(probe_signatures):
            event_group = (signature[0], signature[1], signature[7], signature[8])
            if not signature[2] and event_group in wildcard_groups:
                continue
            try:
                _new_probe_item(keymap, signature)
            except Exception:
                continue


def unregister_keymap_probes() -> None:
    try:
        wm = getattr(getattr(bpy, "context", None), "window_manager", None)
        keyconfigs = getattr(wm, "keyconfigs", None)
        keyconfig = getattr(keyconfigs, "addon", None)
    except Exception:
        keyconfig = None
    if keyconfig is not None:
        try:
            _remove_existing_probe_items(keyconfig)
        except Exception:
            pass
    restore_workspace_tool_keymap_probes()


def _remove_tool_probe_entries(tool_classes) -> None:
    for tool_class in tuple(tool_classes):
        try:
            keymap = tuple(getattr(tool_class, "bl_keymap", ()) or ())
        except Exception:
            continue
        cleaned = tuple(
            entry
            for entry in keymap
            if not (isinstance(entry, (tuple, list)) and entry and entry[0] == _KEYMAP_PROBE_ID)
        )
        try:
            tool_class.bl_keymap = cleaned
        except Exception:
            continue


def install_workspace_tool_keymap_probes(tool_classes) -> None:
    """Insert HEAD probes in AWB WorkSpaceTool keymap declarations."""
    tools = tuple(tool_classes)
    _remove_tool_probe_entries(tools)
    for tool_class in tools:
        try:
            keymap = tuple(getattr(tool_class, "bl_keymap", ()) or ())
        except Exception:
            continue
        instrumented: list[Any] = []
        seen: set[tuple[Any, ...]] = set()
        for entry in keymap:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                instrumented.append(entry)
                continue
            operator_id = str(entry[0])
            event_args = entry[1]
            if not _is_awb_relevant_operator(operator_id) or not isinstance(event_args, dict):
                instrumented.append(entry)
                continue
            event_type = str(event_args.get("type", ""))
            if event_type in {"", "NONE", "MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
                instrumented.append(entry)
                continue
            signature = (
                event_type,
                str(event_args.get("value", "PRESS")),
                bool(event_args.get("any", False)),
                bool(event_args.get("shift", False)),
                bool(event_args.get("ctrl", False)),
                bool(event_args.get("alt", False)),
                bool(event_args.get("oskey", False)),
                str(event_args.get("key_modifier", "NONE") or "NONE"),
                str(event_args.get("direction", "ANY") or "ANY"),
            )
            if signature not in seen:
                probe_args = dict(event_args)
                probe_args["head"] = True
                instrumented.append((_KEYMAP_PROBE_ID, probe_args, {}))
                seen.add(signature)
            instrumented.append(tuple(entry))
        try:
            tool_class.bl_keymap = tuple(instrumented)
        except Exception:
            continue


def restore_workspace_tool_keymap_probes(tool_classes=None) -> None:
    if tool_classes is None:
        try:
            from . import viewport_keymap

            tool_classes = getattr(viewport_keymap, "_AWB_SELECTION_TOOLS", ())
        except Exception:
            return
    _remove_tool_probe_entries(tool_classes)


def prepare_workspace_tool_keymap_probes() -> None:
    try:
        from . import viewport_keymap

        tool_classes = getattr(viewport_keymap, "_AWB_SELECTION_TOOLS", ())
    except Exception:
        return
    install_workspace_tool_keymap_probes(tool_classes)


def _reset_raw_input_path_override_for_tests(path: str | Path | None) -> None:
    global _RAW_INPUT_TRACE_PATH_OVERRIDE

    _RAW_INPUT_TRACE_PATH_OVERRIDE = Path(path) if path is not None else None
