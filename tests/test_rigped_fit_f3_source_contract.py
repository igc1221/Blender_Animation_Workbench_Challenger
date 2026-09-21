from pathlib import Path

ROOT = Path(__file__).parents[1]
EXT = ROOT / "extension" / "blender_animation_workbench"


def _source(name: str) -> str:
    return (EXT / name).read_text(encoding="utf-8")


def test_f3_move_uses_session_owned_gesture_and_frozen_matrix():
    session = _source("rigped_fit_session.py")

    assert "class FitMoveGestureBaseline" in session
    assert "active_move_gesture: FitMoveGestureBaseline | None" in session
    assert "def begin_fit_move_gesture" in session
    assert "def commit_fit_move_gesture" in session
    assert "def cancel_fit_move_gesture" in session
    assert "Matrix(session.matrix_world_frozen)" in session
    assert "session.rig_object.matrix_world.to_3x3()" not in session


def test_f3_preview_serial_is_separate_from_semantic_revision():
    session = _source("rigped_fit_session.py")

    assert "preview_serial: int = 0" in session
    assert "session.preview_serial += 1" in session
    assert "session.revision = gesture.revision + 1" in session
    assert "session.revision += 1" not in session


def test_f3_cancel_restores_exact_gesture_baseline():
    session = _source("rigped_fit_session.py")

    start = session.index("def cancel_fit_move_gesture")
    block = session[start:]
    assert "session.draft = gesture.draft" in block
    assert "session.geometry = gesture.geometry" in block
    assert "session.revision = gesture.revision" in block
    assert "session.preview_serial = gesture.preview_serial" in block
    assert "session.active_move_gesture = None" in block


def test_f3_figure_route_suppresses_native_workspace_tool_gizmo():
    gizmo = _source("global_transform_gizmo.py")

    assert "fit_ui_state(context) is not None" in gizmo
    assert "context.space_data.show_gizmo_tool = False" in gizmo
    assert "return route == \"FIGURE\" and mode == \"MOVE\"" in gizmo
    assert "begin_fit_move_gesture" in gizmo
    assert "commit_fit_move_gesture" in gizmo
    assert "cancel_fit_move_gesture" in gizmo
