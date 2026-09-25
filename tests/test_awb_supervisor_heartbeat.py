from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import awb_supervisor_heartbeat as heartbeat_module
from scripts.awb_supervisor_heartbeat import (
    HEARTBEAT_SCHEMA,
    Heartbeat,
    HeartbeatCorruptError,
    HeartbeatError,
    HeartbeatMonitor,
    make_heartbeat,
    read_heartbeat,
    write_heartbeat,
)


def _heartbeat(sequence: int, *, utc: str = "2000-01-01T00:00:00Z") -> Heartbeat:
    return Heartbeat("incident-7", 1234, sequence, utc, "session-3", "trace-9")


_MAX_IDENTIFIER = heartbeat_module.MAX_IDENTIFIER_LENGTH
_MAX_UTC = heartbeat_module.MAX_UTC_LENGTH
_IDENTIFIER_FIELDS = ("incident_id", "awb_session_id", "trace_id")


def _identifier(limit: int) -> str:
    return "x" * limit


def _dataclass_arguments(field: str, value: str) -> dict[str, object]:
    arguments: dict[str, object] = {
        "incident_id": "incident-7",
        "pid": 1234,
        "sequence": 0,
        "utc": "2000-01-01T00:00:00Z",
        "awb_session_id": None,
        "trace_id": None,
    }
    arguments[field] = value
    return arguments


def _factory_arguments(field: str, value: str) -> dict[str, object]:
    arguments: dict[str, object] = {
        "incident_id": "incident-7",
        "pid": 1234,
        "sequence": 0,
        "awb_session_id": None,
        "trace_id": None,
        "now_utc": datetime(2026, 9, 26, tzinfo=UTC),
    }
    arguments[field] = value
    return arguments


@pytest.mark.parametrize("field", _IDENTIFIER_FIELDS)
def test_identifier_length_boundary_is_inclusive_at_the_schema_limit(field: str):
    at_limit = _identifier(_MAX_IDENTIFIER)
    record = Heartbeat(**_dataclass_arguments(field, at_limit))
    payload = record.to_dict()

    assert len(payload[field]) == _MAX_IDENTIFIER
    assert Heartbeat.from_dict(payload) == record

    over_limit = _identifier(_MAX_IDENTIFIER + 1)
    with pytest.raises(HeartbeatError, match=f"{field} must be at most {_MAX_IDENTIFIER}"):
        Heartbeat(**_dataclass_arguments(field, over_limit))


@pytest.mark.parametrize("field", _IDENTIFIER_FIELDS)
def test_overlength_identifier_is_rejected_on_every_runtime_entry_point(field: str, tmp_path):
    over_limit = _identifier(_MAX_IDENTIFIER + 1)

    with pytest.raises(HeartbeatError, match=f"{field} must be at most {_MAX_IDENTIFIER}"):
        make_heartbeat(**_factory_arguments(field, over_limit))

    payload = _heartbeat(3).to_dict()
    payload[field] = over_limit
    with pytest.raises(HeartbeatCorruptError, match=f"{field} must be at most {_MAX_IDENTIFIER}"):
        Heartbeat.from_dict(payload)

    path = tmp_path / "heartbeat.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HeartbeatCorruptError, match=f"{field} must be at most {_MAX_IDENTIFIER}"):
        read_heartbeat(path)


@pytest.mark.parametrize("field", _IDENTIFIER_FIELDS)
def test_boundary_length_identifiers_survive_write_and_read_round_trip(field: str, tmp_path):
    at_limit = _identifier(_MAX_IDENTIFIER)
    record = make_heartbeat(**_factory_arguments(field, at_limit))
    path = tmp_path / "heartbeat.json"

    write_heartbeat(path, record)

    assert read_heartbeat(path) == record
    assert read_heartbeat(path, expected_incident_id=record.incident_id) == record
    written = json.loads(path.read_text(encoding="utf-8"))
    assert len(written[field]) == _MAX_IDENTIFIER


def test_utc_length_boundary_matches_the_schema_maximum():
    # No real ISO-8601 instant approaches the 64 character envelope, so the
    # boundary is pinned on the length rule itself: exactly at the limit the
    # length rule must not be what rejects the value, one past it must be.
    at_limit = "Z" * _MAX_UTC
    over_limit = "Z" * (_MAX_UTC + 1)

    with pytest.raises(HeartbeatError, match="must be valid ISO-8601"):
        Heartbeat("incident-7", 1234, 0, at_limit)
    with pytest.raises(HeartbeatError, match=f"must be at most {_MAX_UTC} characters"):
        Heartbeat("incident-7", 1234, 0, over_limit)


@pytest.mark.parametrize(
    "utc",
    [
        "2000-01-01T00:00:00Z",
        "2000-01-01T00:00:00+00:00",
        "2000-01-01T00:00:00-00:00",
        "2000-01-01T00:00:00.123Z",
        "2000-01-01T00:00:00.123456Z",
        "2000-01-01T00:00:00.123+00:00",
    ],
)
def test_explicit_utc_forms_are_accepted(utc: str):
    record = Heartbeat("incident-7", 1234, 0, utc)

    assert record.utc == utc
    assert Heartbeat.from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    "utc",
    [
        "2000-01-01T00:00:00",
        "2000-01-01T00:00:00+09:00",
        "2000-01-01T00:00:00-05:00",
        "2000-01-01T00:00:00z",
        "",
        "not-a-timestampZ",
    ],
)
def test_non_utc_or_unparseable_utc_is_rejected(utc: str):
    with pytest.raises(HeartbeatError):
        Heartbeat("incident-7", 1234, 0, utc)


@pytest.mark.parametrize(
    "utc",
    [
        "2000-01-01T00:00:00+0000",
        "2000-01-01T00:00:00-0000",
        "2000-01-01T00:00:00+00",
        "2000-01-01T00:00:00-00",
        "2000-01-01T00:00:00+01:00",
        "2000-01-01T00:00:00 UTC",
    ],
)
def test_utc_offset_forms_outside_the_schema_pattern_fail_closed(utc: str):
    # The schema only admits a serialized "Z" or "+00:00"/"-00:00" suffix, so any
    # other CPython-parseable zero offset spelling must be rejected here too.
    with pytest.raises(HeartbeatError, match="UTC"):
        Heartbeat("incident-7", 1234, 0, utc)


def test_overlength_utc_on_disk_is_treated_as_corrupt(tmp_path):
    path = tmp_path / "heartbeat.json"
    payload = _heartbeat(3).to_dict()
    payload["utc"] = "2000-01-01T00:00:00." + "0" * 44 + "Z"
    assert len(payload["utc"]) == _MAX_UTC + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HeartbeatCorruptError, match=f"must be at most {_MAX_UTC} characters"):
        read_heartbeat(path)


def test_multibyte_identifier_limit_counts_characters_like_the_schema():
    # maxLength is a code-point count in JSON Schema, so len() is the match and a
    # non-ASCII identifier is bounded by the same 128 character envelope.
    at_limit = "\u00e9" * _MAX_IDENTIFIER
    over_limit = "\u00e9" * (_MAX_IDENTIFIER + 1)

    record = Heartbeat(at_limit, 1234, 0, "2000-01-01T00:00:00Z")
    assert len(record.incident_id) == _MAX_IDENTIFIER
    assert Heartbeat.from_dict(record.to_dict()) == record

    with pytest.raises(HeartbeatError, match=f"at most {_MAX_IDENTIFIER} characters"):
        Heartbeat(over_limit, 1234, 0, "2000-01-01T00:00:00Z")


def test_to_dict_cannot_emit_an_out_of_contract_identifier():
    # to_dict is a projection of already-validated state, so every emitted
    # identifier must stay inside the schema envelope.
    record = make_heartbeat(
        incident_id="INC-20260926-101500-0a1b2c3d",
        pid=1234,
        sequence=1,
        awb_session_id="session:" + "a" * 32,
        trace_id="trace:" + "b" * 32,
        now_utc=datetime(2026, 9, 26, tzinfo=UTC),
    )
    payload = record.to_dict()

    assert payload["utc"].endswith("Z")
    for name in ("incident_id", "awb_session_id", "trace_id"):
        assert 1 <= len(payload[name]) <= _MAX_IDENTIFIER
    assert 1 <= len(payload["utc"]) <= _MAX_UTC


def test_schema_bounds_are_the_source_of_the_runtime_limits():
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "DEBUG"
        / "schemas"
        / "AWB_SUPERVISOR_HEARTBEAT_V1.schema.json"
    )
    properties = json.loads(schema_path.read_text(encoding="utf-8"))["properties"]

    for name in ("incident_id", "awb_session_id", "trace_id"):
        assert properties[name]["maxLength"] == _MAX_IDENTIFIER
        assert properties[name]["pattern"] == "\\S"
    assert properties["utc"]["maxLength"] == _MAX_UTC
    assert properties["utc"]["pattern"] == heartbeat_module.UTC_SUFFIX_PATTERN


def test_atomic_writer_flushes_and_replaces_complete_schema(monkeypatch):
    events = []
    bytes_written = bytearray()

    class MemoryStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append("close")

        def write(self, data):
            events.append("write")
            bytes_written.extend(data)

        def flush(self):
            events.append("flush")

        def fileno(self):
            return 77

    stream = MemoryStream()
    monkeypatch.setattr(
        heartbeat_module.Path,
        "mkdir",
        lambda *_args, **_kwargs: events.append("mkdir"),
    )
    monkeypatch.setattr(
        heartbeat_module.tempfile,
        "mkstemp",
        lambda **kwargs: (events.append(("mkstemp", kwargs["dir"])) or (77, "heartbeat.tmp")),
    )
    monkeypatch.setattr(
        heartbeat_module.os,
        "fdopen",
        lambda fd, mode: (events.append(("fdopen", fd, mode)) or stream),
    )
    monkeypatch.setattr(
        heartbeat_module.os,
        "fsync",
        lambda fd: events.append(("fsync", fd)),
    )
    monkeypatch.setattr(
        heartbeat_module.os,
        "replace",
        lambda source, target: events.append(("replace", source, target)),
    )
    heartbeat = make_heartbeat(
        incident_id="incident-7",
        pid=1234,
        sequence=8,
        awb_session_id="session-3",
        trace_id="trace-9",
        now_utc=datetime(2026, 9, 26, tzinfo=UTC),
    )

    path = Path("heartbeat.json")
    write_heartbeat(path, heartbeat)

    payload = json.loads(bytes_written.decode("utf-8"))
    assert payload["schema"] == HEARTBEAT_SCHEMA
    assert payload["sequence"] == 8
    assert payload["utc"].endswith("Z")
    assert payload["awb_session_id"] == "session-3"
    assert payload["trace_id"] == "trace-9"
    assert events[-1] == ("replace", Path("heartbeat.tmp"), path)
    assert events.index(("fsync", 77)) < events.index(("replace", Path("heartbeat.tmp"), path))


@pytest.mark.parametrize(
    "contents",
    [
        "{truncated",
        '{"schema":"awb-supervisor-heartbeat/v1","schema":"awb-supervisor-heartbeat/v1"}',
        json.dumps(
            {
                "schema": "unsupported/v9",
                "incident_id": "incident-7",
                "pid": 1234,
                "sequence": 1,
                "utc": "2026-09-26T00:00:00Z",
            }
        ),
        json.dumps(
            {
                "schema": "awb-supervisor-heartbeat/v1",
                "incident_id": "incident-7",
                "pid": 1234,
                "sequence": True,
                "utc": "2026-09-26T00:00:00Z",
            }
        ),
    ],
)
def test_reader_rejects_corrupt_or_invalid_heartbeat(monkeypatch, contents):
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: contents)

    with pytest.raises(HeartbeatCorruptError):
        read_heartbeat("heartbeat.json")


def test_reader_rejects_wrong_incident_or_process_identity(monkeypatch):
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: json.dumps(_heartbeat(1).to_dict()),
    )

    with pytest.raises(HeartbeatCorruptError, match="incident_id"):
        read_heartbeat("heartbeat.json", expected_incident_id="another-incident")
    with pytest.raises(HeartbeatCorruptError, match="pid"):
        read_heartbeat("heartbeat.json", expected_pid=5678)


def test_monitor_uses_local_elapsed_time_and_sequence_progress_for_staleness():
    local_clock = [10.0]
    monitor = HeartbeatMonitor(2.5, clock=lambda: local_clock[0])

    first = monitor.observe(_heartbeat(5, utc="2099-12-31T23:59:59Z"))
    assert first.sequence_advanced
    assert first.elapsed_since_advance == 0.0
    assert not first.stale

    local_clock[0] = 12.5
    repeated = monitor.observe(_heartbeat(5, utc="1900-01-01T00:00:00Z"))
    assert not repeated.sequence_advanced
    assert repeated.stale
    assert repeated.elapsed_since_advance == 2.5

    local_clock[0] = 13.0
    advanced = monitor.observe(_heartbeat(6))
    assert advanced.sequence_advanced
    assert not advanced.stale
    assert advanced.elapsed_since_advance == 0.0

    local_clock[0] = 15.5
    assert monitor.status().stale


def test_monitor_detects_missing_first_heartbeat_from_launch_elapsed_time():
    local_clock = [20.0]
    monitor = HeartbeatMonitor(3.0, clock=lambda: local_clock[0])

    local_clock[0] = 23.0
    missing = monitor.status()
    assert missing.sequence is None
    assert missing.elapsed_since_advance == 3.0
    assert missing.stale

    first = monitor.observe(_heartbeat(0))
    assert first.sequence_advanced
    assert not first.stale


def test_sequence_regression_does_not_refresh_stale_timer():
    local_clock = [0.0]
    monitor = HeartbeatMonitor(1.0, clock=lambda: local_clock[0])
    monitor.observe(_heartbeat(4))
    local_clock[0] = 1.0

    regressed = monitor.observe(_heartbeat(3))

    assert regressed.sequence_regressed
    assert regressed.stale
    assert regressed.sequence == 4


def test_invalid_heartbeat_values_and_timeout_fail_closed():
    with pytest.raises(HeartbeatError, match="positive integer"):
        Heartbeat("incident-7", 0, 0, "2026-09-26T00:00:00Z")
    with pytest.raises(HeartbeatError, match="UTC"):
        Heartbeat("incident-7", 1234, 0, "2026-09-26T00:00:00+02:00")
    with pytest.raises(ValueError, match="positive"):
        HeartbeatMonitor(0)
