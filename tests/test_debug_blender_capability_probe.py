from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from scripts import awb_debug_blender_capability_probe as probe

ROOT = Path(__file__).resolve().parents[1]
TARGETS_FILE = ROOT / "scripts" / "verification_targets.json"
TARGET_NAME = "bb10-blender-capability-probe"

HELP_TEXT = """
Usage: blender [args] [file] [args]

General options:
  --background           Run in background mode
  --factory-startup      Start with factory startup file
  --enable-event-simulate
  --debug                Enable all debug options
  --debug-handlers       Install exception handler
  --debug-jobs           Enable jobs profiling
  --debug-gdb            Set the gdb breakpoint to attach in the debug handler
  --debug-win32          Debug Win32
  --crash                Crash Blender, for testing
"""

VERSION_TEXT = """
Blender 5.2.1
Path: C:\\blender\\blender.exe
Date: 2026-01-02 03:04:05
Hash: 5f7b0c11a2d3e4f5061728394a5b6c7d8e9f00112
"""


@pytest.fixture
def fixture_root():
    path = Path.cwd() / "build" / f"bb10-probe-test-{uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _passing_receipt() -> dict:
    return {
        "schema": probe.RECEIPT_SCHEMA,
        "blender": {
            "status": "OK",
            "version": "5.2.1",
            "sha256": "a" * 64,
        },
        "capabilities": {
            "status": "OK",
            "isolation": ["--background", "--factory-startup"],
            "intentionally_crash_invoked": False,
        },
        "faulthandler": {
            "status": "SUPPORTED",
            "requested": True,
            "isolation_enforced": True,
            "child_blender_version": "5.2.1",
        },
        "self_check": {
            "isolation_enforced": True,
            "attached_session_used": False,
            "golden_baseline_touched": False,
        },
    }


def test_help_flags_are_discovered_and_classified_without_execution():
    flags = probe.extract_cli_flags(HELP_TEXT)
    assert "--crash" in flags
    assert "--debug-jobs" in flags
    assert flags == tuple(dict.fromkeys(flags))

    classified = probe.classify_cli_flags(flags)
    assert classified["crash_invoking"] == ["--crash"]
    assert "--debug" in classified["debug"]
    assert "--debug-win32" in classified["debug"]
    assert classified["isolation"] == ["--background", "--factory-startup"]
    assert classified["event_simulation"] == ["--enable-event-simulate"]


def test_version_output_is_parsed_into_identity_fields():
    parsed = probe.parse_version_output(VERSION_TEXT)
    assert parsed["version"] == "5.2.1"
    assert parsed["version_string"] == "Blender 5.2.1"
    assert parsed["build_hash"].startswith("5f7b0c11")
    assert parsed["build_date"] == "2026-01-02 03:04:05"
    assert probe.parse_version_output("")["version"] == ""


def test_child_command_is_accepted_and_unsafe_variants_fail_closed(fixture_root):
    blender = fixture_root / "blender.exe"
    blender.write_bytes(b"MZ")
    command = probe.build_child_command(blender)
    assert command[:1] == [str(blender)]
    assert probe.assert_isolated_child_command(
        command, fixture_root=fixture_root, blender=blender
    ) == command

    for unsafe in (
        [str(blender), "--factory-startup", "--crash", "--python", "x.py"],
        [str(blender), "--background", "--factory-startup", "--debug-jobs"],
        [str(blender), "--background", "--factory-startup", "--enable-event-simulate"],
        [str(blender), "--background", "--factory-startup", "session.blend"],
        [str(blender), "--background", "--factory-startup", "Baselines/golden/rigped.blend"],
        [str(blender), "--background", "--factory-startup", "debug/incidents/INC-1"],
    ):
        with pytest.raises(probe.IsolationError):
            probe.assert_isolated_child_command(
                unsafe, fixture_root=fixture_root, blender=blender
            )

    with pytest.raises(probe.IsolationError):
        probe.assert_isolated_child_command(
            command, fixture_root=fixture_root / "missing", blender=blender
        )


def test_isolated_env_stays_inside_fixture_and_never_reuses_user_session(fixture_root):
    env = probe.build_isolated_env(fixture_root)
    probe.assert_isolated_child_env(env, fixture_root=fixture_root)
    assert env["BLENDER_USER_RESOURCES"].startswith(str(fixture_root))

    leaked = dict(env)
    leaked["BLENDER_USER_CONFIG"] = str(ROOT / "portable_user" / "config")
    with pytest.raises(probe.IsolationError):
        probe.assert_isolated_child_env(leaked, fixture_root=fixture_root)

    escaped = dict(env)
    escaped["BLENDER_USER_SCRIPTS"] = str(ROOT / "scripts")
    with pytest.raises(probe.IsolationError):
        probe.assert_isolated_child_env(escaped, fixture_root=fixture_root)


def test_receipt_parsing_maps_defects_to_fail_closed_reasons():
    assert probe.summarize_receipt(_passing_receipt()) == (probe.STATUS_PASS, [])

    assert probe.summarize_receipt(None) == (probe.STATUS_FAIL, ["RECEIPT_SCHEMA_INVALID"])
    assert probe.summarize_receipt({"schema": "other"}) == (
        probe.STATUS_FAIL,
        ["RECEIPT_SCHEMA_INVALID"],
    )

    unavailable = _passing_receipt()
    unavailable["blender"] = {"status": "UNAVAILABLE"}
    assert probe.summarize_receipt(unavailable) == (
        probe.STATUS_SKIP,
        ["BLENDER_BINARY_UNAVAILABLE"],
    )

    wrong_version = _passing_receipt()
    wrong_version["blender"]["version"] = "4.2.0"
    assert "BLENDER_VERSION_MISMATCH:4.2.0" in probe.summarize_receipt(wrong_version)[1]

    no_digest = _passing_receipt()
    no_digest["blender"]["sha256"] = ""
    assert "BLENDER_BINARY_DIGEST_MISSING" in probe.summarize_receipt(no_digest)[1]

    crashed = _passing_receipt()
    crashed["capabilities"]["intentionally_crash_invoked"] = True
    assert "UNSAFE_CRASH_INVOCATION_RECORDED" in probe.summarize_receipt(crashed)[1]

    no_isolation = _passing_receipt()
    no_isolation["capabilities"]["isolation"] = ["--background"]
    assert "BLENDER_ISOLATION_FLAGS_NOT_REPORTED" in probe.summarize_receipt(no_isolation)[1]

    attached = _passing_receipt()
    attached["self_check"]["attached_session_used"] = True
    assert "ATTACHED_SESSION_USE_RECORDED" in probe.summarize_receipt(attached)[1]

    golden = _passing_receipt()
    golden["self_check"]["golden_baseline_touched"] = True
    assert "GOLDEN_BASELINE_USE_RECORDED" in probe.summarize_receipt(golden)[1]

    bad_faulthandler = _passing_receipt()
    bad_faulthandler["faulthandler"]["status"] = "UNSUPPORTED"
    assert probe.summarize_receipt(bad_faulthandler) == (
        probe.STATUS_FAIL,
        ["FAULTHANDLER_UNSUPPORTED"],
    )

    wrong_child = _passing_receipt()
    wrong_child["faulthandler"]["child_blender_version"] = "4.1.0"
    assert "CHILD_BLENDER_VERSION_MISMATCH:4.1.0" in probe.summarize_receipt(wrong_child)[1]

    no_child_identity = _passing_receipt()
    no_child_identity["faulthandler"]["child_blender_version"] = ""
    assert "CHILD_BLENDER_IDENTITY_MISSING" in probe.summarize_receipt(no_child_identity)[1]

    optional = _passing_receipt()
    optional["faulthandler"] = {"status": "SKIPPED", "requested": False}
    assert probe.summarize_receipt(
        optional, require_faulthandler=False
    ) == (probe.STATUS_PASS, [])
    assert "FAULTHANDLER_PROBE_NOT_REQUESTED" in probe.summarize_receipt(optional)[1]

    assert probe.EXIT_CODES == {probe.STATUS_PASS: 0, probe.STATUS_FAIL: 1, probe.STATUS_SKIP: 125}


def test_missing_blender_binary_reports_skip_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "DEFAULT_BLENDER", tmp_path / "absent-blender.exe")
    receipt = probe.run_probe(
        blender=None,
        expected_version=probe.DEFAULT_EXPECTED_VERSION,
        faulthandler=True,
        timeout=1.0,
        fixture_parent=tmp_path,
        keep_fixture=False,
    )
    assert receipt["status"] == probe.STATUS_SKIP
    assert receipt["blender"]["status"] == "UNAVAILABLE"
    assert receipt["reasons"] == ["BLENDER_BINARY_UNAVAILABLE"]


def test_probe_passes_against_a_fake_disposable_binary(tmp_path, monkeypatch):
    fake_blender = tmp_path / "blender.exe"
    fake_blender.write_bytes(b"MZ fake blender payload")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, cwd, env, timeout):
        argv = [str(item) for item in command]
        calls.append((argv, dict(env)))
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, VERSION_TEXT, "")
        if argv[1:] == ["--help"]:
            return subprocess.CompletedProcess(argv, 0, HELP_TEXT, "")
        child_config = json.loads(Path(env["AWB_BB10_CHILD_CONFIG"]).read_text(encoding="utf-8"))
        Path(child_config["result_path"]).write_text(
            json.dumps(
                {
                    "schema": "awb-debug-blender-capability-child/v1",
                    "status": "OK",
                    "reasons": [],
                    "blender": {
                        "version": "5.2.1",
                        "version_string": "5.2.1",
                        "binary_path": str(fake_blender),
                        "background": True,
                    },
                    "faulthandler": {
                        "status": "SUPPORTED",
                        "module_available": True,
                        "dump_bytes": 512,
                    },
                }
            ),
            encoding="utf-8",
        )
        Path(child_config["faulthandler_dump_path"]).write_text(
            "thread trace", encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(probe, "_run", fake_run)
    result_path = tmp_path / "receipt.json"
    exit_code = probe.main(
        [
            "--blender",
            str(fake_blender),
            "--result",
            str(result_path),
            "--fixture-parent",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    receipt = json.loads(result_path.read_text(encoding="utf-8"))
    assert receipt["status"] == probe.STATUS_PASS
    assert receipt["blender"]["version"] == "5.2.1"
    assert receipt["blender"]["sha256"]
    assert receipt["faulthandler"]["status"] == "SUPPORTED"
    assert receipt["faulthandler"]["child_blender_version"] == "5.2.1"
    assert receipt["faulthandler"]["isolation_enforced"] is True
    assert receipt["fixture_removed"] is True
    assert not Path(receipt["fixture_root"]).exists()

    child_calls = [argv for argv, _env in calls if "--python" in argv]
    assert len(child_calls) == 1
    assert "--crash" not in child_calls[0]
    assert not any(arg.lower().endswith(".blend") for arg in child_calls[0])
    for _argv, env in calls:
        for key, value in env.items():
            if key.startswith("BLENDER_USER_"):
                assert "portable_user" not in value.replace("\\", "/").lower()


def test_child_script_refuses_to_run_outside_a_frozen_config(monkeypatch):
    monkeypatch.delenv("AWB_BB10_CHILD_CONFIG", raising=False)
    from scripts import awb_debug_blender_capability_child as child

    with pytest.raises(RuntimeError, match="AWB_BB10_CHILD_CONFIG"):
        child._load_config()


def test_catalog_target_is_registered_and_safe():
    payload = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    assert payload["schema"] == "awb-verification-targets/v1"
    spec = payload["targets"][TARGET_NAME]
    assert spec["mutates_state"] is False
    assert int(spec["timeout_seconds"]) > 0
    steps = spec["steps"]
    assert len(steps) == 1
    command = steps[0]["command"]
    args = steps[0]["args"]
    assert command == ".\\.venv\\Scripts\\python.exe"
    assert args == [".\\scripts\\awb_debug_blender_capability_probe.py"]
    assert (ROOT / "scripts" / "awb_debug_blender_capability_probe.py").is_file()
    assert (ROOT / "scripts" / "awb_debug_blender_capability_child.py").is_file()
    joined = " ".join(args).replace("\\", "/").lower()
    for forbidden in ("baselines/golden", "rigped_animate", "debug/incidents", "--crash"):
        assert forbidden not in joined


def test_target_selection_matches_configured_verifier_rules(tmp_path):
    from scripts import run_configured_verification as runner

    (tmp_path / "verification_target.json").write_text(
        json.dumps({"target": f"  {TARGET_NAME.upper()}  "}), encoding="utf-8"
    )
    request = json.loads((tmp_path / "verification_target.json").read_text(encoding="utf-8"))
    selected = str(request.get("target", "")).strip().lower()
    assert selected == TARGET_NAME

    targets = runner._load_targets()
    spec = targets[selected]
    assert isinstance(spec, dict)
    steps = spec["steps"]
    assert isinstance(steps, list) and steps
    for raw_step in steps:
        assert isinstance(raw_step, dict)
        assert raw_step["command"]
        args = raw_step["args"]
        assert isinstance(args, list) and all(isinstance(item, str) for item in args)
    assert "bb10-not-registered" not in targets
    assert callable(runner._load_targets) and callable(runner._write_result)


def test_child_script_never_introduces_crash_or_debug_cli_flags():
    child_source = (ROOT / "scripts" / "awb_debug_blender_capability_child.py").read_text(
        encoding="utf-8"
    )
    harness_source = (
        ROOT / "scripts" / "awb_debug_blender_capability_probe.py"
    ).read_text(encoding="utf-8")
    assert "faulthandler.enable" in child_source
    assert "faulthandler.dump_traceback" in child_source
    for token in ("--crash", "--debug-jobs", "_sigsegv", "os.kill", "TerminateProcess"):
        assert token not in child_source
    assert "bpy.ops" not in child_source
    assert "kill" not in harness_source
    assert "EnumWindows" not in harness_source
