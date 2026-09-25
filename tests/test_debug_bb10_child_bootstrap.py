from __future__ import annotations

import ast
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import awb_debug_bb10_child_bootstrap as bootstrap
from scripts import awb_supervisor_heartbeat as supervisor_heartbeat

SOURCE = Path(bootstrap.__file__).read_text(encoding="utf-8-sig")


class RecordingTransport:
    """Minimal object satisfying the public heartbeat transport boundary."""

    name = "RECORDING"

    def __init__(self, *, result: bool = True) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.result = result
        self.lock = threading.Lock()

    def emit(self, payload: Any) -> bool:
        with self.lock:
            self.payloads.append(json.loads(json.dumps(payload, default=str)))
        return self.result


class ExplodingTransport:
    name = "EXPLODING"

    def __init__(self) -> None:
        self.calls = 0

    def emit(self, payload: Any) -> bool:
        self.calls += 1
        raise RuntimeError("sink is broken")


class FakeDebugTrace:
    def __init__(self, session: Any = "sess-1", events: Any = None) -> None:
        self._session = session
        self._events = events if events is not None else []
        self.calls: list[tuple[Any, Any]] = []

    def session_id(self) -> Any:
        if isinstance(self._session, Exception):
            raise self._session
        return self._session

    def read_recent_trace_events(self, limit: int, latest_session_only: bool = True) -> Any:
        self.calls.append((limit, latest_session_only))
        if isinstance(self._events, Exception):
            raise self._events
        return self._events


@pytest.fixture(autouse=True)
def _clean_bootstrap_globals():
    bootstrap.clear_heartbeat_transport()
    yield
    bootstrap.stop_active_heartbeat("TEST_TEARDOWN")
    bootstrap.clear_heartbeat_transport()
    _release_faulthandler()


def _release_faulthandler() -> None:
    """Undo a real faulthandler activation so the pytest process stays clean."""
    import faulthandler

    faulthandler.disable()
    handle = bootstrap._FAULTHANDLER_HANDLE
    bootstrap._FAULTHANDLER_HANDLE = None
    if handle is not None:
        try:
            handle.close()
        except OSError:
            pass


def _config(**overrides: str) -> bootstrap.ChildBootstrapConfig:
    env = {"AWB_BB10_INCIDENT_ID": "INC-BB10-TEST-0001"}
    env.update(overrides)
    return bootstrap.load_config(argv=[], env=env)


# ---------------------------------------------------------------------------
# Contract parsing
# ---------------------------------------------------------------------------


def test_config_reads_incident_id_and_trial_owned_paths_from_environment(tmp_path):
    config = _config(
        AWB_BB10_HEARTBEAT_PATH=str(tmp_path / "heartbeat.json"),
        AWB_BB10_HEARTBEAT_LOG_PATH=str(tmp_path / "heartbeat.jsonl"),
        AWB_BB10_FAULTHANDLER_PATH=str(tmp_path / "fault.txt"),
        AWB_BB10_STATUS_PATH=str(tmp_path / "status.json"),
        AWB_BB10_AWB_MODULE="bl_ext.user_default.blender_animation_workbench",
    )

    assert config.incident_id == "INC-BB10-TEST-0001"
    assert config.heartbeat_path == tmp_path / "heartbeat.json"
    assert config.heartbeat_log_path == tmp_path / "heartbeat.jsonl"
    assert config.faulthandler_path == tmp_path / "fault.txt"
    assert config.status_path == tmp_path / "status.json"
    assert config.awb_module == "bl_ext.user_default.blender_animation_workbench"
    assert config.heartbeat_mode == "TIMER"
    assert config.thread_fallback is True


def test_cli_overrides_environment_and_only_reads_tokens_after_double_dash(tmp_path):
    env = {
        "AWB_BB10_INCIDENT_ID": "INC-FROM-ENV",
        "AWB_BB10_HEARTBEAT_PATH": str(tmp_path / "env.json"),
    }
    config = bootstrap.load_config(
        argv=[
            "--factory-startup",
            str(tmp_path / "baseline.blend"),
            "--awb-bb10-incident-id",
            "IGNORED-BEFORE-DOUBLE-DASH",
            "--",
            "--awb-bb10-incident-id=INC-FROM-CLI",
            "--awb-bb10-heartbeat-mode",
            "thread",
        ],
        env=env,
    )

    assert config.incident_id == "INC-FROM-CLI"
    assert config.heartbeat_mode == "THREAD"
    assert config.heartbeat_path == tmp_path / "env.json"


def test_missing_incident_id_is_rejected_fail_closed():
    with pytest.raises(bootstrap.BootstrapConfigError):
        bootstrap.load_config(argv=[], env={})
    with pytest.raises(bootstrap.BootstrapConfigError):
        bootstrap.load_config(argv=[], env={"AWB_BB10_INCIDENT_ID": "   "})


@pytest.mark.parametrize(
    "incident_id",
    ["../escape", "INC 1", "C:\\temp\\x", "id;drop", "a" * 200, "/leading-slash", "id\n"],
)
def test_unsafe_incident_id_is_rejected(incident_id):
    with pytest.raises(bootstrap.BootstrapConfigError):
        _config(AWB_BB10_INCIDENT_ID=incident_id)


def test_blend_output_paths_are_rejected(tmp_path):
    with pytest.raises(bootstrap.BootstrapConfigError):
        _config(AWB_BB10_HEARTBEAT_PATH=str(tmp_path / "rigped.blend"))
    with pytest.raises(bootstrap.BootstrapConfigError):
        _config(AWB_BB10_FAULTHANDLER_PATH=str(tmp_path / "state.blend1"))


def test_directory_output_path_is_rejected(tmp_path):
    with pytest.raises(bootstrap.BootstrapConfigError):
        _config(AWB_BB10_HEARTBEAT_PATH=str(tmp_path))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.001", bootstrap.MIN_HEARTBEAT_INTERVAL), ("2.5", 2.5), ("999", 60.0)],
)
def test_heartbeat_interval_is_clamped(raw, expected):
    assert _config(AWB_BB10_HEARTBEAT_INTERVAL=raw).heartbeat_interval == expected


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "soon"])
def test_invalid_heartbeat_interval_is_rejected(raw):
    with pytest.raises(bootstrap.BootstrapConfigError):
        _config(AWB_BB10_HEARTBEAT_INTERVAL=raw)


def test_heartbeat_mode_vocabulary_is_explicit():
    assert _config(AWB_BB10_HEARTBEAT_MODE="off").heartbeat_mode == "OFF"
    with pytest.raises(bootstrap.BootstrapConfigError):
        _config(AWB_BB10_HEARTBEAT_MODE="SLEEP")


def test_thread_fallback_flag_is_explicit():
    assert _config(AWB_BB10_THREAD_FALLBACK="0").thread_fallback is False
    assert _config(AWB_BB10_THREAD_FALLBACK="YES").thread_fallback is True


def test_invalid_awb_module_name_is_rejected():
    with pytest.raises(bootstrap.BootstrapConfigError):
        _config(AWB_BB10_AWB_MODULE="blender animation workbench")


# ---------------------------------------------------------------------------
# faulthandler
# ---------------------------------------------------------------------------


def test_faulthandler_is_opt_in(tmp_path):
    assert bootstrap.enable_faulthandler(None) == {
        "status": "DISABLED_NOT_REQUESTED",
        "path": None,
    }
    assert not (tmp_path / "fault.txt").exists()


def test_faulthandler_enables_on_trial_owned_file(tmp_path):
    captured: dict[str, Any] = {}

    class Sink:
        @staticmethod
        def is_enabled() -> bool:
            return False

        @staticmethod
        def enable(file: Any, all_threads: bool = True) -> None:
            captured["file"] = file
            captured["all_threads"] = all_threads

    target = tmp_path / "nested" / "fault.txt"
    status = bootstrap.enable_faulthandler(target, module=Sink)

    assert status == {"status": "ENABLED", "path": str(target)}
    assert captured["all_threads"] is True
    assert Path(captured["file"].name).name == "fault.txt"
    assert target.parent.is_dir()
    captured["file"].close()


def test_faulthandler_reports_existing_sink_without_rebinding(tmp_path):
    class Sink:
        @staticmethod
        def is_enabled() -> bool:
            return True

        @staticmethod
        def enable(file: Any, all_threads: bool = True) -> None:
            raise AssertionError("must not rebind an already enabled sink")

    status = bootstrap.enable_faulthandler(tmp_path / "fault.txt", module=Sink)
    assert status["status"] == "ALREADY_ENABLED"


def test_faulthandler_failure_is_contained(tmp_path):
    class Sink:
        @staticmethod
        def is_enabled() -> bool:
            return False

        @staticmethod
        def enable(file: Any, all_threads: bool = True) -> None:
            raise OSError("cannot open fault sink")

    status = bootstrap.enable_faulthandler(tmp_path / "fault.txt", module=Sink)
    assert status["status"] == "FAILED"
    assert "OSError" in status["reason"]


# ---------------------------------------------------------------------------
# Read-only AWB hints
# ---------------------------------------------------------------------------


def test_hints_report_not_loaded_without_awb(monkeypatch):
    monkeypatch.setattr(
        bootstrap, "_locate_awb_debug_trace", lambda name: (None, f"{name}.debug_trace")
    )
    hints = bootstrap.read_awb_hints()
    assert hints["status"] == "AWB_DEBUG_TRACE_NOT_LOADED"
    assert hints["session_id"] is None
    assert hints["last_trace"] is None


def test_hints_expose_session_and_bounded_last_trace(monkeypatch):
    module = FakeDebugTrace(
        events=[
            {
                "step": "semantic:0004",
                "boundary": "AFTER_OPERATION",
                "action_kind": "SCRUB",
                "session_id": "sess-1",
                "trace_id": "t-1",
                "step_label": "x" * 500,
                "nla_tracks": [{"name": "secret-product-state"}],
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "blender_animation_workbench.debug_trace", module)

    hints = bootstrap.read_awb_hints("blender_animation_workbench")

    assert hints["status"] == "READABLE"
    assert hints["session_id"] == "sess-1"
    assert module.calls == [(1, True)]
    last = hints["last_trace"]
    assert last["step"] == "semantic:0004"
    assert last["boundary"] == "AFTER_OPERATION"
    assert "step_label" not in last
    assert "nla_tracks" not in last


def test_hints_are_unreadable_when_session_probe_raises(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "blender_animation_workbench.debug_trace",
        FakeDebugTrace(session=RuntimeError("no session")),
    )
    hints = bootstrap.read_awb_hints("blender_animation_workbench")
    assert hints["status"] == "UNREADABLE"
    assert hints["last_trace"] is None


def test_hints_are_partial_when_trace_tail_read_fails(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "blender_animation_workbench.debug_trace",
        FakeDebugTrace(events=OSError("trace unreadable")),
    )
    hints = bootstrap.read_awb_hints("blender_animation_workbench")
    assert hints["status"] == "PARTIAL"
    assert hints["session_id"] == "sess-1"


def test_hint_reader_never_imports_awb():
    assert "importlib" not in SOURCE
    assert "importlib.import_module" not in SOURCE


def test_hint_locating_falls_back_to_any_loaded_debug_trace(monkeypatch):
    module = FakeDebugTrace(session="bl_ext-session")
    monkeypatch.setitem(sys.modules, "bl_ext.user_default.awb.debug_trace", module)
    monkeypatch.delitem(sys.modules, "blender_animation_workbench.debug_trace", raising=False)
    resolved, name = bootstrap._locate_awb_debug_trace("blender_animation_workbench")
    assert resolved is module
    assert name == "bl_ext.user_default.awb.debug_trace"


# ---------------------------------------------------------------------------
# Heartbeat transport boundary
# ---------------------------------------------------------------------------


def test_file_transport_replaces_slot_atomically_and_appends_history(tmp_path):
    slot = tmp_path / "hb" / "heartbeat.json"
    log = tmp_path / "hb" / "heartbeat.jsonl"
    transport = bootstrap.FileHeartbeatTransport(slot, log_path=log)

    assert transport.emit({"schema": bootstrap.PROGRESS_SCHEMA, "sequence": 1}) is True
    assert transport.emit({"schema": bootstrap.PROGRESS_SCHEMA, "sequence": 2}) is True

    latest = json.loads(slot.read_text(encoding="utf-8"))
    assert latest["sequence"] == 2
    history = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [item["sequence"] for item in history] == [1, 2]
    assert sorted(p.name for p in slot.parent.iterdir()) == [
        "heartbeat.json",
        "heartbeat.jsonl",
    ]
    assert transport.delivered == 2


def test_file_transport_survives_unwritable_history(tmp_path):
    slot = tmp_path / "heartbeat.json"
    log = tmp_path / "history.jsonl"
    log.mkdir()
    transport = bootstrap.FileHeartbeatTransport(slot, log_path=log)
    assert transport.emit({"sequence": 1}) is True
    assert transport.log_failures == 1
    assert json.loads(slot.read_text(encoding="utf-8"))["sequence"] == 1


def test_null_transport_reports_undelivered():
    assert bootstrap.NullHeartbeatTransport().emit({"sequence": 1}) is False


def test_registered_transport_wins_over_trial_owned_default(tmp_path):
    config = _config(AWB_BB10_HEARTBEAT_PATH=str(tmp_path / "heartbeat.json"))
    assert isinstance(
        bootstrap.resolve_heartbeat_transport(config), bootstrap.FileHeartbeatTransport
    )

    registered = bootstrap.register_heartbeat_transport(RecordingTransport())
    assert bootstrap.resolve_heartbeat_transport(config) is registered
    assert bootstrap.explicit_heartbeat_transport() is registered

    bootstrap.clear_heartbeat_transport()
    assert bootstrap.explicit_heartbeat_transport() is None
    assert isinstance(
        bootstrap.resolve_heartbeat_transport(config), bootstrap.FileHeartbeatTransport
    )


def test_missing_heartbeat_path_resolves_to_inert_transport():
    transport = bootstrap.resolve_heartbeat_transport(_config())
    assert isinstance(transport, bootstrap.NullHeartbeatTransport)
    assert transport.name == "NULL"


# ---------------------------------------------------------------------------
# Heartbeat emission
# ---------------------------------------------------------------------------


def _heartbeat(
    transport: Any,
    *,
    interval: float = 0.05,
    mode: str = "TIMER",
    thread_fallback: bool = True,
    hint_reader: Any = None,
    static_fields: Any = None,
) -> bootstrap.ChildHeartbeat:
    return bootstrap.ChildHeartbeat(
        config=bootstrap.ChildBootstrapConfig(
            incident_id="INC-BB10-TEST-0001",
            heartbeat_interval=interval,
            heartbeat_mode=mode,
            thread_fallback=thread_fallback,
        ),
        transport=transport,
        static_fields=(
            static_fields if static_fields is not None else {"faulthandler": {"status": "OFF"}}
        ),
        monotonic=lambda: 12.5,
        wall=lambda: 1750000000.0,
        hint_reader=hint_reader,
    )


def test_heartbeat_payload_is_bounded_identity_evidence():
    transport = RecordingTransport()
    heartbeat = _heartbeat(
        transport,
        hint_reader=lambda: {"status": "READABLE", "session_id": "sess-9", "last_trace": None},
    )

    assert heartbeat.emit_once() is True
    assert heartbeat.emit_once() is True

    first, second = transport.payloads
    assert first["schema"] == bootstrap.PROGRESS_SCHEMA
    assert first["incident_id"] == "INC-BB10-TEST-0001"
    assert (first["sequence"], second["sequence"]) == (1, 2)
    assert first["monotonic_seconds"] == 12.5
    assert first["wall_epoch_seconds"] == 1750000000.0
    assert first["wall_utc"].endswith("Z")
    assert first["transport"] == "RECORDING"
    assert first["faulthandler"] == {"status": "OFF"}
    assert first["awb_hints"]["session_id"] == "sess-9"
    for forbidden in ("fcurves", "selection", "nla_tracks", "pose_bones", "action"):
        assert forbidden not in first


def test_heartbeat_contains_no_product_state_even_when_hints_are_rich():
    transport = RecordingTransport()
    heartbeat = _heartbeat(
        transport,
        hint_reader=lambda: {
            "status": "READABLE",
            "session_id": "sess-9",
            "last_trace": {"step": "semantic:0004", "fcurve_values": [1.0, 2.0]},
        },
    )
    heartbeat.emit_once()
    payload = transport.payloads[0]
    assert payload["awb_hints"]["last_trace"] == {"step": "semantic:0004"}


def test_sanitize_hints_bounds_and_allowlists_any_hint_source():
    clean = bootstrap.sanitize_hints(
        {
            "status": "READABLE",
            "session_id": "s" * 500,
            "last_trace": {
                "boundary": "AFTER_OPERATION",
                "fcurve_values": [1.0],
                "pose_bone_names": ["secret"],
            },
            "extra": {"nested": ["a", "b"]},
        }
    )
    assert clean["session_id"].endswith("...")
    assert clean["last_trace"] == {"boundary": "AFTER_OPERATION"}
    assert clean["extra"] == {"nested": ["a", "b"]}
    assert len(clean) == 4


def test_static_context_cannot_shadow_identity_fields():
    transport = RecordingTransport()
    heartbeat = _heartbeat(
        transport,
        static_fields={"schema": "spoofed", "incident_id": "spoofed", "trial_label": "t-1"},
    )
    heartbeat.emit_once()
    payload = transport.payloads[0]
    assert payload["schema"] == bootstrap.PROGRESS_SCHEMA
    assert payload["incident_id"] == "INC-BB10-TEST-0001"
    assert payload["trial_label"] == "t-1"


def test_broken_transport_never_escapes_the_heartbeat():
    transport = ExplodingTransport()
    heartbeat = _heartbeat(transport)
    assert heartbeat.emit_once() is False
    assert heartbeat.emit_once() is False
    snapshot = heartbeat.snapshot()
    assert snapshot["transport_failures"] == 2
    assert "RuntimeError" in snapshot["last_error"]
    assert transport.calls == 2


def test_failing_hint_reader_is_contained():
    def broken_reader() -> dict[str, Any]:
        raise ValueError("hint source is gone")

    transport = RecordingTransport()
    heartbeat = _heartbeat(transport, hint_reader=broken_reader)
    assert heartbeat.emit_once() is True
    hints = transport.payloads[0]["awb_hints"]
    assert hints["status"] == "UNREADABLE"
    assert "ValueError" in hints["reason"]


def test_undelivered_transport_is_counted_separately():
    transport = RecordingTransport(result=False)
    heartbeat = _heartbeat(transport)
    assert heartbeat.emit_once() is False
    snapshot = heartbeat.snapshot()
    assert snapshot["emit_count"] == 1
    assert snapshot["delivered_count"] == 0


def test_timer_mode_registers_once_and_reschedules_until_stopped():
    registered: list[tuple[Any, float]] = []

    def registrar(function: Any, interval: float) -> str:
        registered.append((function, interval))
        return "timer-handle"

    transport = RecordingTransport()
    heartbeat = _heartbeat(transport, interval=0.25)
    status = heartbeat.start(register_timer=registrar)

    assert status == {"status": "STARTED", "mode": "TIMER"}
    assert len(registered) == 1
    assert registered[0][1] == 0.25
    assert len(transport.payloads) == 1

    function = registered[0][0]
    assert function() == 0.25
    assert len(transport.payloads) == 2

    heartbeat.stop()
    assert function() is None
    assert len(transport.payloads) == 2
    assert heartbeat.snapshot()["start"] == {"status": "STOPPED"}


def test_timer_mode_without_registrar_falls_back_to_worker_thread():
    transport = RecordingTransport()
    heartbeat = _heartbeat(transport, interval=0.05)
    status = heartbeat.start(register_timer=None)
    try:
        assert status == {"status": "THREAD_FALLBACK", "mode": "THREAD"}
        deadline = time.monotonic() + 3.0
        while len(transport.payloads) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(transport.payloads) >= 2
    finally:
        heartbeat.stop()


def test_thread_fallback_can_be_refused():
    transport = RecordingTransport()
    heartbeat = _heartbeat(transport, thread_fallback=False)
    status = heartbeat.start(register_timer=None)
    assert status == {"status": "UNAVAILABLE_NO_TIMER", "mode": "TIMER"}
    assert len(transport.payloads) == 1


def test_thread_mode_emits_periodically_and_stops():
    transport = RecordingTransport()
    heartbeat = _heartbeat(transport, interval=0.05, mode="THREAD")
    status = heartbeat.start(register_timer=None)
    try:
        assert status == {"status": "STARTED", "mode": "THREAD"}
        deadline = time.monotonic() + 3.0
        while len(transport.payloads) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(transport.payloads) >= 2
    finally:
        heartbeat.stop()
    count = len(transport.payloads)
    time.sleep(0.15)
    assert len(transport.payloads) == count


def test_off_mode_emits_nothing():
    transport = RecordingTransport()
    heartbeat = _heartbeat(transport, mode="OFF")
    assert heartbeat.start(register_timer=None) == {"status": "OFF", "mode": "OFF"}
    assert transport.payloads == []


# ---------------------------------------------------------------------------
# Blender timer wiring
# ---------------------------------------------------------------------------


def _fake_bpy(*, background: bool = False, register: Any = None) -> SimpleNamespace:
    calls: list[dict[str, Any]] = []

    def _register(function: Any, **kwargs: Any) -> str:
        calls.append({"function": function, **kwargs})
        return "handle"

    return SimpleNamespace(
        app=SimpleNamespace(
            background=background,
            timers=SimpleNamespace(register=register if register is not None else _register),
        ),
        _calls=calls,
    )


def test_timer_registrar_pins_persistent_timer():
    fake = _fake_bpy()
    registrar = bootstrap.blender_timer_registrar(fake)
    assert registrar is not None

    def tick() -> float:
        return 0.5

    registrar(tick, 0.5)
    assert fake._calls == [{"function": tick, "first_interval": 0.5, "persistent": True}]


def test_timer_registrar_is_absent_in_background_or_without_timers():
    assert bootstrap.blender_timer_registrar(_fake_bpy(background=True)) is None
    assert bootstrap.blender_timer_registrar(_fake_bpy(register="not-callable")) is None
    assert bootstrap.blender_timer_registrar(SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# Whole-bootstrap behavior
# ---------------------------------------------------------------------------


def test_run_bootstrap_writes_trial_owned_artifacts_without_blender(tmp_path):
    env = {
        "AWB_BB10_INCIDENT_ID": "INC-BB10-ENDTOEND",
        "AWB_BB10_HEARTBEAT_PATH": str(tmp_path / "heartbeat.json"),
        "AWB_BB10_HEARTBEAT_LOG_PATH": str(tmp_path / "heartbeat.jsonl"),
        "AWB_BB10_STATUS_PATH": str(tmp_path / "status.json"),
        "AWB_BB10_HEARTBEAT_MODE": "THREAD",
        "AWB_BB10_HEARTBEAT_INTERVAL": "0.05",
        "AWB_BB10_FAULTHANDLER_PATH": str(tmp_path / "fault.txt"),
    }
    status = bootstrap.run_bootstrap(argv=[], env=env)
    try:
        assert status["execution_status"] == "BOOTSTRAP_ACTIVE"
        assert status["incident_id"] == "INC-BB10-ENDTOEND"
        assert status["blender_timers_available"] is False
        assert status["heartbeat_start"] == {"status": "STARTED", "mode": "THREAD"}
        assert status["faulthandler"]["status"] in {"ENABLED", "ALREADY_ENABLED"}
        assert status["awb_hints"]["status"] in {
            "AWB_DEBUG_TRACE_NOT_LOADED",
            "READABLE",
            "PARTIAL",
            "UNREADABLE",
        }

        written = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        assert written["execution_status"] == "BOOTSTRAP_ACTIVE"
        beat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
        assert beat["incident_id"] == "INC-BB10-ENDTOEND"
        assert beat["schema"] == bootstrap.PROGRESS_SCHEMA
        assert bootstrap.active_heartbeat() is not None
    finally:
        final = bootstrap.stop_active_heartbeat("BOOTSTRAP_EXITED")

    assert final["execution_status"] == "BOOTSTRAP_EXITED"
    assert bootstrap.active_heartbeat() is None
    exited = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert exited["execution_status"] == "BOOTSTRAP_EXITED"
    assert exited["heartbeat"]["delivered_count"] >= 1


def test_run_bootstrap_rejects_a_missing_incident_id_without_writing(tmp_path):
    status_path = tmp_path / "status.json"
    status = bootstrap.run_bootstrap(
        argv=[],
        env={"AWB_BB10_STATUS_PATH": str(status_path)},
    )
    assert status["execution_status"] == "CONFIG_REJECTED"
    assert "AWB_BB10_INCIDENT_ID" in status["reason"]
    assert not status_path.exists()
    assert bootstrap.active_heartbeat() is None


def test_run_bootstrap_uses_injected_transport_and_hint_reader():
    transport = RecordingTransport()
    seen: list[int] = []

    def hint_reader() -> dict[str, Any]:
        seen.append(1)
        return {"status": "READABLE", "session_id": "sess-x", "last_trace": None}

    status = bootstrap.run_bootstrap(
        argv=[],
        env={"AWB_BB10_INCIDENT_ID": "INC-INJECTED"},
        transport=transport,
        hint_reader=hint_reader,
    )
    try:
        assert status["execution_status"] == "BOOTSTRAP_ACTIVE"
        assert transport.payloads
        assert transport.payloads[0]["transport"] == "RECORDING"
        assert transport.payloads[0]["awb_hints"]["session_id"] == "sess-x"
        assert seen
        assert bootstrap.explicit_heartbeat_transport() is transport
    finally:
        bootstrap.stop_active_heartbeat()


def test_run_bootstrap_survives_a_broken_transport(tmp_path):
    status = bootstrap.run_bootstrap(
        argv=[],
        env={
            "AWB_BB10_INCIDENT_ID": "INC-BROKEN-SINK",
            "AWB_BB10_HEARTBEAT_PATH": str(tmp_path / "heartbeat.json"),
        },
        transport=ExplodingTransport(),
    )
    try:
        assert status["execution_status"] == "BOOTSTRAP_ACTIVE"
        assert status["heartbeat"]["transport_failures"] == 1
    finally:
        bootstrap.stop_active_heartbeat()


# ---------------------------------------------------------------------------
# Source contract
# ---------------------------------------------------------------------------


def test_module_is_stdlib_only_and_never_imports_bpy_at_module_scope():
    tree = ast.parse(SOURCE)
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(name.split(".")[0] == "bpy" for name in imported)
    assert not any(name.startswith("blender_animation_workbench") for name in imported)


def test_module_has_no_product_state_mutation_surface():
    for forbidden in (
        "bpy.ops",
        "bpy.data",
        "bpy.context",
        "keyframe_insert",
        "save_mainfile",
        "evaluated_depsgraph_get",
        "bpy.app.driver_namespace",
    ):
        assert forbidden not in SOURCE


def test_module_does_not_own_supervisor_or_publication_surfaces():
    for out_of_scope in ("JobObject", "win32job", "CreateJobObject", "WerSubmit", "LocalDumps"):
        assert out_of_scope not in SOURCE


# ---------------------------------------------------------------------------
# Progress payload naming contract
# ---------------------------------------------------------------------------

SCHEMA_DIR = Path(bootstrap.__file__).resolve().parents[1] / "docs" / "DEBUG" / "schemas"
PROGRESS_SCHEMA_FILE = SCHEMA_DIR / "AWB_BB10_CHILD_PROGRESS_V1.schema.json"
CANONICAL_SCHEMA_FILE = SCHEMA_DIR / "AWB_SUPERVISOR_HEARTBEAT_V1.schema.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_errors(schema: dict[str, Any], payload: Any) -> list[str]:
    """Minimal check of the JSON Schema subset used by the progress schema."""
    if not isinstance(payload, dict):
        return ["root:not-an-object"]
    properties = schema["properties"]
    errors = [f"missing:{name}" for name in schema["required"] if name not in payload]
    if schema.get("additionalProperties") is False:
        errors.extend(f"unknown:{name}" for name in payload if name not in properties)
    for name, value in payload.items():
        spec = properties.get(name)
        if spec is None:
            continue
        if "const" in spec and value != spec["const"]:
            errors.append(f"const:{name}")
        expected = spec.get("type")
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        invalid_type = (
            (expected == "object" and not isinstance(value, dict))
            or (expected == "string" and not isinstance(value, str))
            or (expected == "number" and not is_number)
            or (
                expected == "integer"
                and (not isinstance(value, int) or isinstance(value, bool))
            )
        )
        if invalid_type:
            errors.append(f"type:{name}")
        if is_number:
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"minimum:{name}")
            if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
                errors.append(f"exclusiveMinimum:{name}")
        if isinstance(value, str) and "pattern" in spec and not re.search(spec["pattern"], value):
            errors.append(f"pattern:{name}")
    return errors


def test_progress_schema_id_is_not_a_heartbeat_id():
    schema = _read_json(PROGRESS_SCHEMA_FILE)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == bootstrap.PROGRESS_SCHEMA
    assert schema["properties"]["schema"]["const"] == bootstrap.PROGRESS_SCHEMA
    assert schema["title"].endswith("v1")
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    # Distinct from the canonical supervisor record, in both directions.
    canonical = _read_json(CANONICAL_SCHEMA_FILE)
    assert bootstrap.PROGRESS_SCHEMA != supervisor_heartbeat.HEARTBEAT_SCHEMA
    assert schema["$id"] != canonical["$id"]
    assert set(schema["properties"]) & set(canonical["properties"]) == {
        "schema",
        "incident_id",
        "sequence",
    }


def test_progress_schema_documents_the_canonical_boundary():
    schema = _read_json(PROGRESS_SCHEMA_FILE)
    description = schema["description"]
    assert supervisor_heartbeat.HEARTBEAT_SCHEMA in description
    assert "read_heartbeat" in description
    assert "awb-bb10-child-heartbeat/v1" in description
    # The child-local monotonic value is documented as diagnostic self-data that
    # no supervisor may combine with its own clock.
    monotonic = schema["properties"]["monotonic_seconds"]["description"]
    assert "CHILD-LOCAL" in monotonic
    assert "diagnostic self-data" in monotonic
    assert "read_heartbeat" in monotonic
    # The canonical schema points back at the progress payload, so the two
    # documents cannot be read as two equivalent heartbeats.
    assert bootstrap.PROGRESS_SCHEMA in _read_json(CANONICAL_SCHEMA_FILE)["description"]


def test_progress_schema_bounds_match_the_emitted_payload():
    transport = RecordingTransport()
    assert _heartbeat(transport).emit_once() is True
    emitted = transport.payloads[0]

    schema = _read_json(PROGRESS_SCHEMA_FILE)
    assert _schema_errors(schema, emitted) == []
    assert emitted["schema"] == bootstrap.PROGRESS_SCHEMA
    assert set(schema["required"]) <= set(emitted)
    # The closed schema knows every key the emitter can actually produce.
    assert set(emitted) == set(schema["properties"])


def test_emitted_progress_payload_is_rejected_by_the_canonical_reader(tmp_path):
    """A supervisor must never read a progress slot as a canonical heartbeat."""
    transport = RecordingTransport()
    assert _heartbeat(transport).emit_once() is True
    progress = transport.payloads[0]
    slot = tmp_path / "progress.json"
    slot.write_text(json.dumps(progress, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(supervisor_heartbeat.HeartbeatCorruptError):
        supervisor_heartbeat.read_heartbeat(slot)
    with pytest.raises(supervisor_heartbeat.HeartbeatCorruptError):
        supervisor_heartbeat.Heartbeat.from_dict(progress)


def test_module_documents_the_supervisor_must_never_read_a_progress_payload():
    for required_note in (
        bootstrap.PROGRESS_SCHEMA,
        supervisor_heartbeat.HEARTBEAT_SCHEMA,
        "read_heartbeat",
    ):
        assert required_note in SOURCE
    # The retired id may only appear in the docstring sentence that retires it.
    occurrences = SOURCE.count("awb-bb10-child-heartbeat/v1")
    assert occurrences == 1
    assert "retired" in SOURCE
