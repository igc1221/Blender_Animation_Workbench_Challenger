from pathlib import Path

ROOT = Path(__file__).parents[1]
EXT = ROOT / "extension" / "blender_animation_workbench"


def _source(name: str) -> str:
    return (EXT / name).read_text(encoding="utf-8")


def _function_block(source: str, name: str, next_name: str) -> str:
    start = source.index(f"def {name}(")
    end = source.index(f"def {next_name}(", start)
    return source[start:end]


def test_f2_figure_entry_is_object_hosted_and_never_enters_edit_mode():
    source = _source("rigped_create_fit_ui.py")
    block = _function_block(source, "_begin_fit", "_restore_fit_transform_settings")

    assert "begin_fit_semantic_session" in block
    assert 'tool_set_by_id(name="baw.select_object")' in block
    assert 'mode_set(mode="EDIT")' not in block
    assert "_capture_edit_rest" not in block
    assert "edit_bones" not in block


def test_f2_cancel_discards_semantic_session_without_editbone_restore():
    source = _source("rigped_create_fit_ui.py")
    start = source.index("class BAW_OT_rigped_fit_cancel")
    block = source[start : source.index("_PICKER_ROLE_TEXT", start)]

    assert "end_fit_semantic_session(context)" in block
    assert "_restore_edit_rest" not in block
    assert 'mode_set(mode="EDIT")' not in block


def test_f2_runtime_session_is_read_only_against_native_rest_and_object_transform():
    source = _source("rigped_fit_session.py")

    assert "edit_bones" not in source
    assert "mode_set(" not in source
    assert "bpy.ops.transform" not in source
    assert ".head =" not in source
    assert ".tail =" not in source
    assert "FitRestPartSnapshot" in source
    assert "FitBodyGeometrySnapshot" in source
    assert "apply_fit_part_selection" in source


def test_f2_draw_and_pick_share_fitdraft_geometry_snapshot():
    overlay = _source("rigped_box_wire_overlay.py")
    keymap = _source("viewport_keymap.py")

    assert "fit_geometry_snapshot_for_rig" in overlay
    assert "fit_geometry_part_by_name" in overlay
    assert "fit_geometry_snapshot(context)" in keymap
    assert "_fit_part_pick_id" in keymap
    assert "_fit_part_box_crossing_ids" in keymap
    assert "FIT_SELECTION_CLICK_RESULT" in keymap
    assert "FIT_SELECTION_BOX_RESULT" in keymap


def test_f2_selection_tool_owns_press_to_release_modal_gesture():
    keymap = _source("viewport_keymap.py")

    assert '{"type": "LEFTMOUSE", "value": "PRESS"}' in keymap
    assert '{"type": "LEFTMOUSE", "value": "CLICK_DRAG"}' not in keymap
    assert 'if event.value != "PRESS":' in keymap
    assert 'if event.type == "LEFTMOUSE" and event.value == "RELEASE":' in keymap


def test_figure_object_route_never_falls_through_to_native_transform_tools():
    keymap = _source("viewport_keymap.py")
    gizmo = _source("global_transform_gizmo.py")

    assert "FIT_F3_TRANSFORM_TOOL_ACTIVE" in keymap
    assert "FIT_F3_TRANSFORM_NOT_IMPLEMENTED" in keymap
    assert 'fit_state is not None and context.mode == "OBJECT"' in keymap
    assert 'fit_ui_state(context) is not None and getattr(context, "mode", "") == "OBJECT"' in gizmo
    assert 'return "FIGURE", "MOVE"' in gizmo
    assert "BAW_OT_figure_fit_move_axis" in gizmo
    assert "apply_fit_move_preview" in gizmo
