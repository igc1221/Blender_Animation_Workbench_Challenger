import importlib.util
from pathlib import Path


def test_extension_source_exists():
    root = Path(__file__).parents[1]
    init_py = root / "extension" / "blender_animation_workbench" / "__init__.py"
    manifest = root / "extension" / "blender_animation_workbench" / "blender_manifest.toml"
    assert init_py.is_file()
    assert manifest.is_file()
    assert importlib.util.spec_from_file_location("blender_animation_workbench", init_py) is not None


def test_phase2_modules_do_not_import_phase3_character_layer():
    root = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
    for filename in ("semantic_model.py", "semantic_adapter.py", "semantic_query.py"):
        source = (root / filename).read_text(encoding="utf-8-sig")
        assert ".character_" not in source
        assert "blender_animation_workbench.character_" not in source


def test_character_operator_surface_has_one_bind_path_and_explicit_repair():
    root = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
    source = (root / "character_ops.py").read_text(encoding="utf-8-sig")
    assert 'bl_idname = "baw.bind_character_active"' in source
    assert 'bl_idname = "baw.repair_character_bone_binding"' in source
    assert 'bl_idname = "baw.bind_character_bone"' not in source
    assert 'bl_idname = "baw.rebind_character_bone"' not in source
    assert "\nCLASSES = (" not in source


def test_user_final_recovery_blender_code_is_attached_session_fail_closed():
    root = Path(__file__).parents[1]
    dispatcher = (root / "scripts" / "replay_user_final_test_via_mcp.py").read_text(
        encoding="utf-8-sig"
    )
    e11_replay = (
        root / "debug" / "user_final_tests" / "E11" / "user_final_replay.json"
    ).read_text(encoding="utf-8-sig")

    assert '"schema": "awb-user-final-recovery/v1"' in e11_replay
    assert '"recovery_kind":' in e11_replay
    assert 'startswith("awb-user-final-recovery/")' in dispatcher
    assert "Recovery blender_code runners are isolated/background-only" in dispatcher


def test_rigped_golden_baseline_is_single_immutable_v1_pair():
    import hashlib
    import json

    root = Path(__file__).parents[1]
    golden = root / "baselines" / "golden"
    blend = golden / "rigped_animate_manual_baseline_v1.blend"
    manifest_path = golden / "rigped_animate_manual_baseline_v1.json"

    assert sorted(path.name for path in golden.glob("rigped_animate_manual_baseline_v*.blend")) == [
        "rigped_animate_manual_baseline_v1.blend"
    ]
    assert sorted(path.name for path in golden.glob("rigped_animate_manual_baseline_v*.json")) == [
        "rigped_animate_manual_baseline_v1.json"
    ]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(blend.read_bytes()).hexdigest()
    assert manifest["golden_path"] == "baselines/golden/rigped_animate_manual_baseline_v1.blend"
    assert manifest["sha256"] == digest == "8c5ee34f94ee5445d73c058545ebd2f89e0e783ef787b0e7d1258694edd68577"

    prepare_source = (root / "scripts" / "prepare_rigped_animate_baseline_via_mcp.py").read_text(
        encoding="utf-8"
    )
    assert "rigped_animate_manual_baseline_v1.blend" in prepare_source
    assert "rigped_animate_manual_baseline_v1.json" in prepare_source
    assert "rigped_animate_manual_baseline_v3" not in prepare_source
    assert "rigped_animate_manual_baseline_v4" not in prepare_source


def test_debug_trace_routes_golden_baseline_to_project_debug():
    root = Path(__file__).parents[1]
    source = (root / "extension" / "blender_animation_workbench" / "debug_trace.py").read_text(
        encoding="utf-8-sig"
    )

    assert 'blend_dir.name.casefold() == "build"' in source
    assert 'blend_dir.name.casefold() == "golden"' in source
    assert 'blend_dir.parent.name.casefold() == "baselines"' in source
    assert 'return blend_dir.parent.parent / "debug"' in source


def test_blender_runtime_errors_are_persisted_with_semantic_context():
    root = Path(__file__).parents[1]
    launcher = (root / "scripts" / "launch_blender_hidden.py").read_text(encoding="utf-8-sig")
    trace = (
        root / "extension" / "blender_animation_workbench" / "debug_trace.py"
    ).read_text(encoding="utf-8-sig")

    assert 'RUNTIME_ERROR_ROOT = DEBUG_ROOT / "runtime_error_log"' in launcher
    assert 'RUNTIME_LOG = RUNTIME_ERROR_ROOT / "blender_runtime.log"' in launcher
    assert 'RUNTIME_LOG_PREVIOUS = RUNTIME_ERROR_ROOT / "blender_runtime.previous.log"' in launcher
    assert "stdout=runtime_log" in launcher
    assert "stderr=subprocess.STDOUT" in launcher

    assert 'def _runtime_error_root_path() -> Path:' in trace
    assert 'return _debug_root_path() / "runtime_error_log"' in trace
    assert '_RUNTIME_CONTEXT_FILENAME = "awb_runtime_context.json"' in trace
    assert '_RUNTIME_ERRORS_FILENAME = "awb_runtime_errors.jsonl"' in trace
    assert "_write_runtime_context_snapshot(record)" in trace
    assert "_append_runtime_error_record(" in trace
    assert "traceback.format_exception" in trace
    assert "sys.excepthook = _runtime_excepthook" in trace
    assert "threading.excepthook = _runtime_threading_excepthook" in trace
    assert "def _archive_runtime_errors_for_new_session()" in trace
    assert 'recorded_session == _SESSION_ID' in trace


def test_verification_targets_are_consolidated_in_one_catalog():
    import json

    root = Path(__file__).parents[1]
    targets_path = root / "scripts" / "verification_targets.json"
    runner_path = root / "scripts" / "run_configured_verification.py"

    payload = json.loads(targets_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "awb-verification-targets/v1"
    targets = payload["targets"]

    required = {
        "phase1l-gui",
        "phase3-s2-runtime",
        "phase3-s8-library",
        "phase4-i12-runtime",
        "phase4-i16-gui",
        "phase4-i16-runtime",
        "gui-pipeline",
        "phase3-character-metadata",
        "phase4-e9-multilimb-sliding",
        "phase4-e11-sliding-auto",
        "phase4-fit-fk-backend",
        "phase4-i19-continuous-contact",
        "phase4-i20-multilimb-contact",
        "phase4-i20-multilimb-gui",
        "phase4-viewport-performance",
        "rc1-sliding-forearm-rotate",
    }
    assert required.issubset(targets)
    assert len(targets) >= 61
    for name, spec in targets.items():
        assert spec["steps"], name
        for step in spec["steps"]:
            assert step["command"], name
            assert isinstance(step.get("args", []), list), name

    runner = runner_path.read_text(encoding="utf-8-sig")
    assert 'TARGETS_FILE = ROOT / "scripts" / "verification_targets.json"' in runner
    assert "subprocess.run(" in runner
