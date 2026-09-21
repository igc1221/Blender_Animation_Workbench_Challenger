from pathlib import Path

ROOT = Path(__file__).parents[1]
EXT = ROOT / "extension" / "blender_animation_workbench"


def _source(name: str) -> str:
    return (EXT / name).read_text(encoding="utf-8")


def test_f5_legacy_live_editbone_fit_module_is_retired():
    assert not (EXT / "rigped_fit_transform.py").exists()

    production_sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in EXT.glob("*.py")
    }
    for source in production_sources.values():
        assert "rigped_fit_transform" not in source

    for legacy_name in (
        "BAW_OT_rigped_fit_commit_transform",
        "BAW_OT_rigped_fit_history_undo",
        "BAW_OT_rigped_fit_history_redo",
        "BAW_OT_rigped_fit_scale_axis",
        "BAW_OT_rigped_fit_transform_axis",
    ):
        assert all(legacy_name not in source for source in production_sources.values())


def test_f5_fit_ui_has_no_native_rest_history_authority():
    ui = _source("rigped_create_fit_ui.py")

    for legacy_name in (
        "_BoneRestSnapshot",
        "_FitHistoryEntry",
        "rest_snapshot",
        "history_cursor",
        "capture_fit_structural_state",
        "restore_fit_structural_state",
        "record_fit_structural_change",
        "undo_fit_structural_change",
        "redo_fit_structural_change",
        "_enter_fit_box_wire_display",
        "show_native_bone_overlays_for_fit",
    ):
        assert legacy_name not in ui

    assert "begin_fit_semantic_session(" in ui
    assert "commit_fit_semantic_session_atomic(context, semantic_session)" in ui
    assert "end_fit_semantic_session(context)" in ui


def test_f5_transform_routes_keep_figure_and_remove_editbone_fit_route():
    gizmo = _source("global_transform_gizmo.py")
    keymap = _source("viewport_keymap.py")

    assert 'return "FIGURE", "MOVE"' in gizmo
    assert 'return "FIGURE", "ROTATE"' in gizmo
    assert 'return "FIGURE", "SCALE"' in gizmo
    assert 'return "FIT",' not in gizmo
    assert '("FIT", BAW_OT_rigped_fit' not in gizmo

    assert "baw.rigped_fit_history_undo" not in keymap
    assert "baw.rigped_fit_history_redo" not in keymap
    assert "FIT_F5_NON_OBJECT_TRANSFORM_BLOCKED" in keymap
    assert "class BAW_OT_guard_figure_native_edit" in keymap
    assert 'if name in {"Object Mode", "Pose", "Armature"}:' in keymap
    assert 'for event_type in ("G", "S"):' in keymap

    guard_start = keymap.index("class BAW_OT_guard_figure_native_edit")
    guard_end = keymap.index("class BAW_OT_block_native_select_click", guard_start)
    guard_block = keymap[guard_start:guard_end]
    assert '"FIT_F5_OBJECT_HOST_RESTORE_FAILED"' in guard_block
    assert 'return {"CANCELLED"}' not in guard_block


def test_f5_shared_projection_math_is_owned_by_neutral_viewport_module():
    gizmo = _source("global_transform_gizmo.py")
    transform = _source("rigped_transform.py")
    math_source = _source("viewport_transform_math.py")

    assert "def _active_edit_bone(context):" in gizmo
    assert "from .viewport_transform_math import " in gizmo
    assert "from .viewport_transform_math import " in transform
    assert "def axis_point(" in math_source
    assert "def plane_point(" in math_source
    assert "def rotation_vector(" in math_source
    assert "region_2d_to_origin_3d" in math_source
    assert "region_2d_to_vector_3d" in math_source
    assert "intersect_line_plane" in math_source


def test_f5_f4_commit_remains_the_bounded_native_editbone_adapter():
    commit = _source("rigped_fit_commit.py")

    assert "class FitEditBoneBefore" in commit
    assert "class FitNativeBeforeImage" in commit
    assert "def commit_fit_semantic_session_atomic" in commit
    assert "edit_bones" in commit
    assert "rigped_fit_transform" not in commit
