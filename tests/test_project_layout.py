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
