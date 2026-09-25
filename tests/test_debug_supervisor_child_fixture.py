"""Acceptance tests for the BB-10 deterministic synthetic supervisor child fixture.

These tests only cover the fixture itself. No supervisor, watchdog, classifier or
verdict logic is implemented or asserted here; the tests prove that the fixture
is deterministic, invokable and honest about its own output volume so that a
separate supervisor implementation can rely on it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import pytest

from scripts import awb_debug_supervisor_child_fixture as fixture

ROOT = Path(__file__).resolve().parents[1]
ONE_MEBIBYTE = 1024 * 1024

MODE_CASES: dict[str, dict[str, Any]] = {
    "clean": {},
    "exception": {},
    "high_volume": {"line_count": 64, "line_size": 32},
    "heartbeat_stall": {"heartbeat_count": 2, "stall_seconds": 0.0},
    "malformed_heartbeat": {"heartbeat_count": 2},
    "native_exit": {},
}

EXPECTED_EXIT: dict[str, tuple[str, int]] = {
    "clean": (fixture.EXIT_KIND_RETURN, 0),
    "exception": (fixture.EXIT_KIND_UNRAISED, fixture.EXCEPTION_MODE_EXIT_CODE),
    "high_volume": (fixture.EXIT_KIND_RETURN, 0),
    "heartbeat_stall": (fixture.EXIT_KIND_RETURN, 0),
    "malformed_heartbeat": (fixture.EXIT_KIND_RETURN, 0),
    "native_exit": (fixture.EXIT_KIND_OS_EXIT, fixture.WINDOWS_NATIVE_ACCESS_VIOLATION),
}


class ChildRun(NamedTuple):
    returncode: int
    elapsed: float
    stdout_path: Path
    stderr_path: Path
    summary_path: Path

    @property
    def exit_code(self) -> int:
        return int(self.returncode) & fixture.MAX_WINDOWS_EXIT_CODE

    def summary(self) -> dict[str, Any]:
        return json.loads(self.summary_path.read_text(encoding="utf-8"))


@pytest.fixture
def work_dir():
    root = ROOT / "build" / f"bb10-child-fixture-{uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_child(mode: str, work: Path, *, label: str = "0", **options: Any) -> ChildRun:
    """Run the fixture with stdout/stderr redirected to files, never to a pipe."""
    summary_path = work / f"{mode}.{label}.summary.json"
    argv = fixture.child_argv(mode, summary_path=summary_path, **options)
    stdout_path = work / f"{mode}.{label}.stdout.bin"
    stderr_path = work / f"{mode}.{label}.stderr.txt"
    started = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        completed = subprocess.run(
            argv,
            stdout=out,
            stderr=err,
            cwd=str(ROOT),
            timeout=300,
            check=False,
        )
    return ChildRun(
        returncode=completed.returncode,
        elapsed=time.monotonic() - started,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        summary_path=summary_path,
    )


def test_fixture_is_pure_stdlib_and_blender_free():
    assert fixture.SCRIPT_PATH.is_file()
    source = fixture.SCRIPT_PATH.read_text(encoding="utf-8-sig")
    for forbidden in (
        "import bpy",
        "blender_animation_workbench",
        "import subprocess",
        "import socket",
        "import urllib",
        "import requests",
    ):
        assert forbidden not in source


def test_supported_modes_cover_required_supervisor_scenarios():
    assert set(fixture.MODES) == set(MODE_CASES)
    assert set(fixture.HEARTBEAT_MODES) == {"heartbeat_stall", "malformed_heartbeat"}


def test_child_argv_rejects_unknown_mode_and_option():
    with pytest.raises(ValueError):
        fixture.child_argv("not-a-mode")
    with pytest.raises(ValueError):
        fixture.child_argv("clean", not_a_real_option=1)
    argv = fixture.child_argv("clean", exit_code=0, stall_announce=False)
    assert argv[0] == sys.executable
    assert argv[1] == str(fixture.SCRIPT_PATH)
    assert argv[2:4] == ["--mode", "clean"]
    assert "--exit-code" in argv
    assert "--stall-announce" not in argv


def test_option_names_matches_declared_cli_surface():
    names = fixture.option_names()
    assert {
        "mode",
        "summary_path",
        "exit_code",
        "line_count",
        "line_size",
        "heartbeat_path",
        "heartbeat_count",
        "heartbeat_interval",
        "heartbeat_fsync",
        "stall_seconds",
        "stall_announce",
        "startup_stall_seconds",
    } <= names
    assert "help" not in names


@pytest.mark.parametrize("mode", sorted(MODE_CASES))
def test_every_mode_emits_well_formed_summary_receipt(mode, work_dir):
    options = dict(MODE_CASES[mode])
    if mode in fixture.HEARTBEAT_MODES:
        options["heartbeat_path"] = str(work_dir / f"{mode}.heartbeat.jsonl")
    run = run_child(mode, work_dir, **options)
    summary = run.summary()
    expected_kind, expected_code = EXPECTED_EXIT[mode]

    assert summary["schema"] == fixture.SUMMARY_SCHEMA
    assert summary["mode"] == mode
    assert set(summary) == set(fixture.SUMMARY_KEYS)
    assert summary["exit_kind"] == expected_kind
    assert summary["exit_code_requested"] == expected_code
    assert run.exit_code == expected_code
    assert summary["stall_seconds_applied"] == 0.0
    if mode in fixture.HEARTBEAT_MODES:
        assert summary["heartbeat_path"] == options["heartbeat_path"]
    else:
        assert summary["heartbeat_path"] is None


@pytest.mark.parametrize("mode", sorted(MODE_CASES))
def test_summary_receipt_never_carries_wall_clock_or_pid(mode, work_dir):
    options = dict(MODE_CASES[mode])
    if mode in fixture.HEARTBEAT_MODES:
        options["heartbeat_path"] = str(work_dir / f"{mode}.heartbeat.jsonl")
    summary = run_child(mode, work_dir, **options).summary()
    serialized = json.dumps(summary, sort_keys=True)
    assert "wall_clock" not in serialized
    assert "monotonic" not in serialized
    assert "pid" not in serialized
    assert "timestamp" not in serialized


def test_clean_mode_is_deterministic_across_runs(work_dir):
    first = run_child("clean", work_dir, label="a")
    second = run_child("clean", work_dir, label="b")

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.stdout_path.read_bytes() == b""
    assert second.stdout_path.read_bytes() == b""
    assert first.summary_path.read_bytes() == second.summary_path.read_bytes()


def test_clean_mode_can_simulate_clean_nonzero_exit(work_dir):
    run = run_child("clean", work_dir, exit_code=3)
    assert run.exit_code == 3
    assert run.stdout_path.read_bytes() == b""
    assert run.stderr_path.read_bytes() == b""
    assert run.summary()["exit_code_requested"] == 3
    assert run.summary()["exit_kind"] == fixture.EXIT_KIND_RETURN


def test_exception_mode_exits_nonzero_with_real_traceback(work_dir):
    run = run_child("exception", work_dir)
    assert run.exit_code == fixture.EXCEPTION_MODE_EXIT_CODE
    assert run.exit_code != 0
    stderr = run.stderr_path.read_text(encoding="utf-8", errors="replace")
    assert "Traceback (most recent call last)" in stderr
    assert "RuntimeError" in stderr
    assert "AWB_SUPERVISOR_FIXTURE_EXCEPTION" in stderr
    summary = run.summary()
    assert summary["exit_kind"] == fixture.EXIT_KIND_UNRAISED
    assert summary["stdout_lines"] == 0
    assert summary["stdout_bytes"] == 0


def test_high_volume_mode_completes_full_volume_without_pipe(work_dir):
    line_count = fixture.DEFAULT_HIGH_VOLUME_LINE_COUNT
    line_size = fixture.DEFAULT_HIGH_VOLUME_LINE_SIZE
    first = run_child(
        "high_volume", work_dir, label="a", line_count=line_count, line_size=line_size
    )
    second = run_child(
        "high_volume", work_dir, label="b", line_count=line_count, line_size=line_size
    )
    expected_bytes = line_count * line_size

    for run in (first, second):
        assert run.exit_code == 0
        assert run.stderr_path.read_bytes() == b""
        assert run.stdout_path.stat().st_size == expected_bytes
        summary = run.summary()
        assert summary["stdout_lines"] == line_count
        assert summary["stdout_bytes"] == expected_bytes
        assert summary["high_volume_line_count"] == line_count
        assert summary["high_volume_line_size"] == line_size

    assert first.stdout_path.read_bytes() == second.stdout_path.read_bytes()
    lines = first.stdout_path.read_bytes().split(b"\n")
    assert lines[-1] == b""
    assert len(lines) - 1 == line_count
    assert lines[0] + b"\n" == fixture.high_volume_line(1, line_size)
    assert lines[line_count - 1] + b"\n" == fixture.high_volume_line(line_count, line_size)


def test_high_volume_default_volume_exceeds_any_pipe_buffer(work_dir):
    default_bytes = fixture.DEFAULT_HIGH_VOLUME_LINE_COUNT * fixture.DEFAULT_HIGH_VOLUME_LINE_SIZE
    assert default_bytes >= ONE_MEBIBYTE

    run = run_child(
        "high_volume",
        work_dir,
        line_count=1,
        line_size=fixture.DEFAULT_HIGH_VOLUME_LINE_SIZE,
    )
    assert run.exit_code == 0
    assert run.stdout_path.stat().st_size == fixture.DEFAULT_HIGH_VOLUME_LINE_SIZE
    assert run.summary()["high_volume_line_count"] == 1


def test_high_volume_mode_cannot_be_drained_by_a_naive_pipe_reader(work_dir):
    line_count = 80000
    line_size = 96
    argv = fixture.child_argv("high_volume", line_count=line_count, line_size=line_size)
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
    )
    completed_without_reading = False
    try:
        try:
            process.wait(timeout=10)
            completed_without_reading = True
        except subprocess.TimeoutExpired:
            completed_without_reading = False
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=60)
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass

    assert completed_without_reading is False
    assert line_count * line_size > ONE_MEBIBYTE


def test_heartbeat_stall_mode_stops_heartbeats_while_still_alive(work_dir):
    heartbeat_path = work_dir / "stall.heartbeat.jsonl"
    stall_seconds = 0.25
    run = run_child(
        "heartbeat_stall",
        work_dir,
        heartbeat_path=str(heartbeat_path),
        heartbeat_count=3,
        heartbeat_interval=0.01,
        stall_seconds=stall_seconds,
    )

    assert run.exit_code == 0
    summary = run.summary()
    assert summary["heartbeat_valid_records"] == 3
    assert summary["heartbeat_malformed_records"] == 0
    assert summary["stall_seconds_applied"] == round(stall_seconds, 6)
    assert run.elapsed >= stall_seconds

    raw = heartbeat_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    records = [json.loads(line) for line in raw.splitlines()]
    assert [record["heartbeat_seq"] for record in records] == [1, 2, 3]
    assert all(record["state"] == fixture.HEARTBEAT_STATE_RUNNING for record in records)


def test_heartbeat_stall_announce_emits_one_extra_marker_record(work_dir):
    heartbeat_path = work_dir / "announced.heartbeat.jsonl"
    run = run_child(
        "heartbeat_stall",
        work_dir,
        heartbeat_path=str(heartbeat_path),
        heartbeat_count=2,
        stall_announce=True,
    )
    assert run.exit_code == 0
    records = [
        json.loads(line) for line in heartbeat_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["state"] for record in records] == [
        fixture.HEARTBEAT_STATE_RUNNING,
        fixture.HEARTBEAT_STATE_RUNNING,
        fixture.HEARTBEAT_STATE_STALL_ANNOUNCED,
    ]
    assert run.summary()["heartbeat_valid_records"] == 3


def test_malformed_heartbeat_mode_emits_one_valid_then_one_malformed_record(work_dir):
    heartbeat_path = work_dir / "malformed.heartbeat.jsonl"
    run = run_child(
        "malformed_heartbeat",
        work_dir,
        heartbeat_path=str(heartbeat_path),
        heartbeat_count=3,
    )

    assert run.exit_code == 0
    summary = run.summary()
    assert summary["heartbeat_valid_records"] == 2
    assert summary["heartbeat_malformed_records"] == 1

    raw = heartbeat_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    lines = raw.splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["heartbeat_seq"] == 1
    assert json.loads(lines[1])["heartbeat_seq"] == 2
    assert lines[2].startswith(f'{{"schema": "{fixture.HEARTBEAT_SCHEMA}", "heartbeat_seq": ')
    assert lines[2].strip() != ""
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[2])


def test_heartbeat_records_publish_wall_clock_only(work_dir):
    heartbeat_path = work_dir / "clock.heartbeat.jsonl"
    run_child("heartbeat_stall", work_dir, heartbeat_path=str(heartbeat_path), heartbeat_count=1)
    record = json.loads(heartbeat_path.read_text(encoding="utf-8").splitlines()[0])

    assert set(record) == set(fixture.HEARTBEAT_RECORD_KEYS)
    assert record["schema"] == fixture.HEARTBEAT_SCHEMA
    assert isinstance(record["wall_clock_unix"], float)
    assert record["wall_clock_unix"] > 0.0
    assert isinstance(record["pid"], int)

    source = fixture.SCRIPT_PATH.read_text(encoding="utf-8-sig")
    assert "time.monotonic" not in source
    assert "perf_counter" not in source


def test_native_exit_mode_reports_windows_crash_exit_code(work_dir):
    run = run_child("native_exit", work_dir)
    assert run.exit_code == fixture.WINDOWS_NATIVE_ACCESS_VIOLATION
    summary = run.summary()
    assert summary["exit_kind"] == fixture.EXIT_KIND_OS_EXIT
    assert summary["exit_code_requested"] == fixture.WINDOWS_NATIVE_ACCESS_VIOLATION
    stderr = run.stderr_path.read_text(encoding="utf-8", errors="replace")
    assert "Traceback" not in stderr
    assert "RuntimeError" not in stderr


def test_native_exit_mode_honours_custom_exit_code(work_dir):
    run = run_child(
        "native_exit",
        work_dir,
        exit_code=fixture.WINDOWS_NATIVE_STACK_BUFFER_OVERRUN,
    )
    assert run.exit_code == fixture.WINDOWS_NATIVE_STACK_BUFFER_OVERRUN
    assert run.summary()["exit_code_requested"] == fixture.WINDOWS_NATIVE_STACK_BUFFER_OVERRUN


@pytest.mark.parametrize("mode", sorted(fixture.HEARTBEAT_MODES))
def test_heartbeat_modes_fail_closed_without_heartbeat_path(mode, work_dir):
    argv = fixture.child_argv(mode)
    stdout_path = work_dir / f"{mode}.novalid.stdout.bin"
    stderr_path = work_dir / f"{mode}.novalid.stderr.txt"
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        completed = subprocess.run(
            argv, stdout=out, stderr=err, cwd=str(ROOT), timeout=120, check=False
        )
    assert completed.returncode == fixture.USAGE_ERROR_EXIT_CODE
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    assert "--heartbeat-path is required" in stderr


def test_malformed_heartbeat_mode_requires_two_records(work_dir):
    argv = fixture.child_argv(
        "malformed_heartbeat",
        heartbeat_path=str(work_dir / "unused.heartbeat.jsonl"),
        heartbeat_count=1,
    )
    stderr_path = work_dir / "malformed.invalid.stderr.txt"
    stdout_path = work_dir / "malformed.invalid.stdout.bin"
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        completed = subprocess.run(
            argv, stdout=out, stderr=err, cwd=str(ROOT), timeout=120, check=False
        )
    assert completed.returncode == fixture.USAGE_ERROR_EXIT_CODE
    assert "--heartbeat-count must be >= 2" in stderr_path.read_text(
        encoding="utf-8", errors="replace"
    )


def test_high_volume_line_is_exact_width_for_any_index():
    for index in (1, 9, 10, 999999, 10**10, 10**12):
        line = fixture.high_volume_line(index, 96)
        assert len(line) == 96
        assert line.endswith(b"\n")
        assert line.startswith(f"{index:010d}:".encode("ascii"))
        assert len(line[:-1]) == 95


def test_high_volume_line_rejects_undersized_line_size():
    with pytest.raises(ValueError):
        fixture.high_volume_line(1, fixture.MIN_HIGH_VOLUME_LINE_SIZE // 2)


def test_heartbeat_writer_appends_and_counts_records(work_dir):
    path = work_dir / "unit.heartbeat.jsonl"
    state = fixture.HEARTBEAT_STATE_RUNNING
    with fixture.HeartbeatWriter(path) as writer:
        writer.write_valid(mode="clean", seq=1, state=state, note="n")
        writer.write_malformed(seq=2)
        assert writer.valid_records == 1
        assert writer.malformed_records == 1
        with fixture.HeartbeatWriter(path) as appended:
            appended.write_valid(mode="clean", seq=3, state=state, note="n")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["heartbeat_seq"] == 1
    assert json.loads(lines[2])["heartbeat_seq"] == 3


def test_heartbeat_writer_rejects_record_larger_than_atomicity_bound(work_dir):
    path = work_dir / "bounded.heartbeat.jsonl"
    with fixture.HeartbeatWriter(path) as writer:
        with pytest.raises(ValueError):
            writer.write_record(b"x" * fixture.MAX_HEARTBEAT_RECORD_BYTES)
        writer.write_record(b"x" * (fixture.MAX_HEARTBEAT_RECORD_BYTES - 1))
    assert path.stat().st_size == fixture.MAX_HEARTBEAT_RECORD_BYTES
