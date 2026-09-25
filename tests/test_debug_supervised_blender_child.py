from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts import awb_debug_bb10_child_bootstrap as bootstrap
from scripts import awb_debug_supervised_blender_child as child
from scripts import awb_supervisor_heartbeat as supervisor_heartbeat

BLENDER_ARGV = ["blender", "--background", "--factory-startup", "--python", "child.py"]


@pytest.fixture
def heartbeat_file():
    path = Path(__file__).with_name(f".heartbeat-{uuid4().hex}.json")
    yield path
    path.unlink(missing_ok=True)


def _bpy(*, filepath="", background=True, objects=(), collections=(), addons=()):
    return SimpleNamespace(
        app=SimpleNamespace(background=background),
        data=SimpleNamespace(
            filepath=filepath,
            objects=objects,
            collections=collections,
        ),
        context=SimpleNamespace(
            preferences=SimpleNamespace(addons={name: object() for name in addons})
        ),
    )


def _child_payload(**overrides):
    payload = {
        "schema": bootstrap.PROGRESS_SCHEMA,
        "incident_id": "INC-BB10-0007",
        "child_pid": 4321,
        "sequence": 9,
        "wall_utc": "2026-09-26T02:03:04.123Z",
        "awb_hints": {
            "status": "READABLE",
            "session_id": "awb-session-2",
            "last_trace": {"trace_id": "trace-5", "fcurve_value": 12.0},
        },
        "monotonic_seconds": 123.0,
        "wall_epoch_seconds": 999.0,
    }
    payload.update(overrides)
    return payload


def test_adapter_writes_only_authoritative_supervisor_heartbeat(heartbeat_file):
    path = heartbeat_file
    transport = child.SupervisorHeartbeatTransport(path)

    assert transport.emit(_child_payload())
    result = supervisor_heartbeat.read_heartbeat(
        path,
        expected_incident_id="INC-BB10-0007",
        expected_pid=4321,
    )

    assert result.to_dict() == {
        "schema": supervisor_heartbeat.HEARTBEAT_SCHEMA,
        "incident_id": "INC-BB10-0007",
        "pid": 4321,
        "sequence": 9,
        "utc": "2026-09-26T02:03:04.123Z",
        "awb_session_id": "awb-session-2",
        "trace_id": "trace-5",
    }
    assert set(result.to_dict()) == {
        "schema",
        "incident_id",
        "pid",
        "sequence",
        "utc",
        "awb_session_id",
        "trace_id",
    }


def test_optional_awb_hints_are_omitted_when_unavailable():
    heartbeat = child.child_payload_to_heartbeat(
        _child_payload(awb_hints={"status": "AWB_DEBUG_TRACE_NOT_LOADED"})
    )

    assert heartbeat.awb_session_id is None
    assert heartbeat.trace_id is None
    assert "awb_session_id" not in heartbeat.to_dict()
    assert "trace_id" not in heartbeat.to_dict()


@pytest.mark.parametrize(
    "override",
    [
        {"schema": "wrong/v1"},
        {"schema": "awb-bb10-child-heartbeat/v1"},
        {"wall_utc": "not-a-timestamp"},
        {"incident_id": ""},
        {"child_pid": 0},
        {"sequence": True},
    ],
)
def test_adapter_rejects_malformed_bootstrap_payloads(override):
    with pytest.raises(supervisor_heartbeat.HeartbeatError):
        child.child_payload_to_heartbeat(_child_payload(**override))


def test_adapter_refuses_to_re_adapt_a_canonical_supervisor_record():
    canonical = supervisor_heartbeat.make_heartbeat(
        incident_id="INC-BB10-0007",
        pid=4321,
        sequence=9,
        now_utc=datetime(2026, 9, 26, 2, 3, 4, 123000, tzinfo=UTC),
    ).to_dict()

    with pytest.raises(supervisor_heartbeat.HeartbeatError, match="already a canonical"):
        child.child_payload_to_heartbeat(canonical)


# ---------------------------------------------------------------------------
# Bootstrap progress payload vs canonical supervisor heartbeat
# ---------------------------------------------------------------------------


class _CapturingTransport:
    name = "CAPTURING"

    def __init__(self):
        self.payloads: list[dict] = []

    def emit(self, payload):
        self.payloads.append(dict(payload))
        return True


def _real_progress_payload(**overrides):
    """Build an actual bootstrap progress payload, not a hand-written one."""
    config = bootstrap.ChildBootstrapConfig(
        incident_id="INC-BB10-0007",
        heartbeat_interval=1.0,
        heartbeat_mode=bootstrap.HEARTBEAT_MODE_TIMER,
    )
    transport = _CapturingTransport()
    heartbeat = bootstrap.ChildHeartbeat(
        config=config,
        transport=transport,
        static_fields={"faulthandler": {"status": "DISABLED_NOT_REQUESTED"}},
        monotonic=lambda: 4321.5,
        wall=lambda: 1790388184.123,
        hint_reader=lambda: {
            "status": "READABLE",
            "session_id": "awb-session-2",
            "last_trace": {"trace_id": "trace-5", "fcurve_value": 12.0},
        },
    )
    assert heartbeat.emit_once() is True
    payload = dict(transport.payloads[0])
    payload.update(overrides)
    return payload


def test_supervised_adapter_overrides_the_bootstrap_progress_payload(heartbeat_file):
    """The bytes in the heartbeat slot are canonical, never the progress payload."""
    progress = _real_progress_payload()
    assert progress["schema"] == bootstrap.PROGRESS_SCHEMA
    assert progress["monotonic_seconds"] == 4321.5

    transport = child.SupervisorHeartbeatTransport(heartbeat_file)
    assert transport.emit(progress) is True

    text = heartbeat_file.read_text(encoding="utf-8")
    on_disk = json.loads(text)
    assert on_disk["schema"] == supervisor_heartbeat.HEARTBEAT_SCHEMA
    assert bootstrap.PROGRESS_SCHEMA not in text
    # The canonical record keeps none of the child-local diagnostic self-data.
    for progress_only in (
        "monotonic_seconds",
        "wall_epoch_seconds",
        "interval_seconds",
        "transport",
        "awb_hints",
        "faulthandler",
        "child_pid",
    ):
        assert progress_only not in on_disk

    observed = supervisor_heartbeat.read_heartbeat(
        heartbeat_file,
        expected_incident_id="INC-BB10-0007",
        expected_pid=progress["child_pid"],
    )
    assert observed.sequence == 1
    assert observed.awb_session_id == "awb-session-2"
    assert observed.trace_id == "trace-5"
    assert observed.utc == "2026-09-26T02:03:04.123Z"


def test_progress_payload_is_never_accepted_by_the_canonical_reader(heartbeat_file):
    """A supervisor must not be able to read a progress slot as a heartbeat."""
    progress = _real_progress_payload()
    heartbeat_file.write_text(json.dumps(progress, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(supervisor_heartbeat.HeartbeatCorruptError):
        supervisor_heartbeat.read_heartbeat(heartbeat_file)
    with pytest.raises(supervisor_heartbeat.HeartbeatCorruptError):
        supervisor_heartbeat.Heartbeat.from_dict(progress)
    # Same payload, routed through the supervised adapter: the canonical reader
    # then succeeds, because the adapter owns the conversion.
    transport = child.SupervisorHeartbeatTransport(heartbeat_file)
    assert transport.emit(progress) is True
    assert supervisor_heartbeat.read_heartbeat(heartbeat_file).sequence == 1


def test_supervisor_staleness_uses_its_own_clock_not_the_child_monotonic_value(heartbeat_file):
    """Child-local monotonic self-data must not influence the supervisor verdict."""
    now = [1_000.0]
    monitor = supervisor_heartbeat.HeartbeatMonitor(2.0, clock=lambda: now[0])
    transport = child.SupervisorHeartbeatTransport(heartbeat_file)

    for sequence, child_monotonic in ((1, 90_000.0), (2, 90_001.5)):
        assert transport.emit(
            _real_progress_payload(sequence=sequence, monotonic_seconds=child_monotonic)
        )
        now[0] += 1.0
        status = monitor.observe(supervisor_heartbeat.read_heartbeat(heartbeat_file))

    assert status.sequence == 2
    assert status.elapsed_since_advance == 0.0
    assert status.stale is False

    # A child monotonic value ~90,000s away must not move the verdict: only
    # supervisor-local elapsed time counts.
    now[0] += 1.9
    assert monitor.status().stale is False
    now[0] += 0.2
    assert monitor.status().stale is True


def test_retired_child_heartbeat_schema_id_is_not_a_live_contract():
    adapter_source = Path(child.__file__).read_text(encoding="utf-8")
    assert "awb-bb10-child-heartbeat/v1" not in adapter_source
    assert "PROGRESS_SCHEMA" in adapter_source
    assert bootstrap.PROGRESS_SCHEMA != supervisor_heartbeat.HEARTBEAT_SCHEMA
    assert not hasattr(bootstrap, "HEARTBEAT_SCHEMA")


def test_stalled_transport_suppresses_file_advancement(heartbeat_file):
    path = heartbeat_file
    transport = child.SupervisorHeartbeatTransport(path)
    assert transport.emit(_child_payload(sequence=1))
    first = supervisor_heartbeat.read_heartbeat(path)

    transport.set_stalled(True)
    assert not transport.emit(_child_payload(sequence=2))
    assert transport.suppressed_count == 1
    assert supervisor_heartbeat.read_heartbeat(path) == first

    transport.set_stalled(False)
    assert transport.emit(_child_payload(sequence=3))
    assert supervisor_heartbeat.read_heartbeat(path).sequence == 3


def test_stall_mode_writes_startup_heartbeat_then_stalls(heartbeat_file):
    transport = child.SupervisorHeartbeatTransport(heartbeat_file, stall_after_first=True)

    assert transport.emit(_child_payload(sequence=1))
    assert not transport.emit(_child_payload(sequence=2))
    assert transport.suppressed_count == 1
    assert supervisor_heartbeat.read_heartbeat(heartbeat_file).sequence == 1


@pytest.mark.parametrize(
    ("argv", "bpy", "message"),
    [
        (["blender", "--factory-startup"], _bpy(), "--background"),
        (["blender", "--background"], _bpy(), "--factory-startup"),
        (BLENDER_ARGV, _bpy(background=False), "background mode"),
        (BLENDER_ARGV, _bpy(filepath="authored.blend"), ".blend file"),
        (BLENDER_ARGV, _bpy(objects=[SimpleNamespace(name="AWB_Rigped")]), "AWB authored"),
        (
            BLENDER_ARGV,
            _bpy(addons=["blender_animation_workbench"]),
            "AWB authored",
        ),
    ],
)
def test_runtime_guard_rejects_non_diagnostic_blender(argv, bpy, message):
    with pytest.raises(child.ChildSafetyError, match=message):
        child.validate_diagnostic_blender(bpy, argv)


def test_runtime_guard_allows_factory_background_without_awb_data():
    child.validate_diagnostic_blender(_bpy(), BLENDER_ARGV)


@pytest.mark.parametrize("raw", [None, "0", "-1", "nan", "inf", "3601"])
def test_child_lifetime_must_be_requested_finite_positive_and_bounded(raw):
    env = {} if raw is None else {"AWB_BB10_CHILD_LIFETIME_SECONDS": raw}
    with pytest.raises(ValueError):
        child.load_child_options(argv=[], env=env)


def test_child_options_parse_cli_and_environment_with_cli_precedence():
    options = child.load_child_options(
        argv=[
            "blender",
            "--background",
            "--",
            "--awb-bb10-child-lifetime-seconds",
            "4.5",
            "--awb-bb10-heartbeat-stall",
        ],
        env={
            "AWB_BB10_CHILD_LIFETIME_SECONDS": "2",
            "AWB_BB10_HEARTBEAT_STALL": "false",
        },
    )
    assert options.lifetime_seconds == 4.5
    assert options.heartbeat_stall is True


@pytest.mark.parametrize("stall", [False, True])
def test_stay_alive_honors_bounded_lifetime_and_restores_transport(stall):
    now = [10.0]
    sleeps = []
    transport = child.SupervisorHeartbeatTransport("unused.json")

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    child.stay_alive(
        child.ChildOptions(lifetime_seconds=0.6, heartbeat_stall=stall),
        transport,
        monotonic=lambda: now[0],
        sleep=fake_sleep,
    )

    assert sum(sleeps) == pytest.approx(0.6)
    assert not transport._stalled


@pytest.mark.parametrize(
    ("delivered", "expected_result", "expected_stop"),
    [(1, 0, "CHILD_LIFETIME_COMPLETE"), (0, 1, "CHILD_FAILED")],
)
def test_main_validates_initial_delivery_and_stops(
    monkeypatch,
    delivered,
    expected_result,
    expected_stop,
):
    stopped = []
    bootstrap_calls = []
    transport = child.SupervisorHeartbeatTransport("unused.json")
    options = child.ChildOptions(lifetime_seconds=1.0)
    config = SimpleNamespace(
        heartbeat_path=transport.path,
        heartbeat_mode="THREAD",
    )
    monkeypatch.setattr(child, "load_child_options", lambda **_kwargs: options)
    monkeypatch.setattr(child.bootstrap, "load_config", lambda **_kwargs: config)
    def fake_run_bootstrap(**kwargs):
        bootstrap_calls.append(kwargs)
        kwargs["transport"].delivered_count = delivered
        return {
            "execution_status": "BOOTSTRAP_ACTIVE",
            "heartbeat_start": {"status": "STARTED"},
        }

    monkeypatch.setattr(child.bootstrap, "run_bootstrap", fake_run_bootstrap)
    monkeypatch.setattr(
        child.bootstrap,
        "stop_active_heartbeat",
        lambda execution_status: stopped.append(execution_status),
    )
    monkeypatch.setattr(child, "stay_alive", lambda *_args, **_kwargs: None)

    result = child.main(
        argv=BLENDER_ARGV,
        env={},
        bpy_module=_bpy(),
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert result == expected_result
    assert stopped == [expected_stop]
    assert bootstrap_calls[0]["transport"].name == "SUPERVISOR_HEARTBEAT_FILE"
