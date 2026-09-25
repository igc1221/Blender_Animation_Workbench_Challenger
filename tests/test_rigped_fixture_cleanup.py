from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP_PATH = ROOT / "scripts" / "prepare_rigped_animate_baseline_via_mcp.py"
E9_PATH = ROOT / "scripts" / "verify_phase4_e9_multilimb_sliding.py"
GOLDEN_PATH = ROOT / "baselines" / "golden" / "rigped_animate_manual_baseline_v1.blend"
MANIFEST_PATH = ROOT / "baselines" / "golden" / "rigped_animate_manual_baseline_v1.json"


def _source(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text)


def _assigned_string(tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError(f"missing string assignment {name}")


def _function_source(text: str, tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(text, node)
            assert segment is not None
            return segment
    raise AssertionError(f"missing function {name}")


def test_golden_manual_baseline_hash_matches_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = str(manifest["sha256"]).lower()
    actual = hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest()

    assert manifest["status"] == "GOLDEN_IMMUTABLE"
    assert expected == "8c5ee34f94ee5445d73c058545ebd2f89e0e783ef787b0e7d1258694edd68577"
    assert actual == expected
    assert GOLDEN_PATH.stat().st_size == manifest["size_bytes"] == 974052


def test_manual_baseline_prepare_copies_golden_then_opens_working_copy() -> None:
    text, tree = _source(PREP_PATH)
    code = _assigned_string(tree, "CODE_TEMPLATE")
    prepare = _function_source(text, tree, "_prepare_working_copy")
    main = _function_source(text, tree, "main")

    assert "bpy.ops.wm.open_mainfile(filepath=working_path, load_ui=True)" in code
    assert "golden.read_bytes()" in prepare
    assert "working.write_bytes(golden_bytes)" in prepare
    assert "actual_golden != expected" in prepare
    assert "actual_working != expected" in prepare
    assert "root / 'baselines' / 'golden' / 'rigped_animate_manual_baseline_v1.blend'" in prepare
    assert "root / 'build' / 'rigped_animate_manual_baseline.blend'" in prepare
    assert "CODE_TEMPLATE.replace('__WORKING_PATH__', repr(str(working)))" in main
    assert "read_setup_descriptor(view)" in code
    assert "setup descriptor is stale/invalid" in code
    assert "character_ids(bpy.context.scene)" in code
    assert "resolve_character(bpy.context.scene, ids[0])" in code
    assert "rigped_animate_manual_baseline.blend1" not in text

    for forbidden in (
        "build_generated_rigped_humanoid",
        "fitted_humanoid_spec",
        "remove_character",
        "bpy.data.objects.remove",
        "bpy.ops.object.delete",
        "bpy.ops.baw.rigped_animate_mode",
    ):
        assert forbidden not in text


def test_e9_fixture_reset_does_not_depend_on_visibility_or_selection() -> None:
    text, tree = _source(E9_PATH)
    reset = _function_source(text, tree, "reset_scene")

    assert "bpy.ops.object.delete" not in reset
    assert "for character_id in tuple(character_ids(scene)):" in reset
    assert "remove_character(scene, character_id)" in reset
    assert "for obj in tuple(bpy.data.objects):" in reset
    assert "bpy.data.objects.remove(obj, do_unlink=True)" in reset
    assert 'obj.name.startswith("AWB_Rigped")' in reset
    assert 'collection.name.startswith("AWB Rigped")' in reset
