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
    assert "session.selected_part_ids = set(gesture.selected_part_ids)" in block
    assert "session.active_part_id = gesture.active_part_id" in block
    assert "session.active_move_gesture = None" in block


def test_f3_figure_route_suppresses_native_workspace_tool_gizmo():
    gizmo = _source("global_transform_gizmo.py")
    keymap = _source("viewport_keymap.py")
    ui = _source("rigped_create_fit_ui.py")

    assert "def fit_ui_state_present" in ui
    assert "fit_ui_state_present(context)" in gizmo
    assert "context.space_data.show_gizmo_tool = False" in gizmo
    assert "return route == \"FIGURE\" and mode == \"MOVE\"" in gizmo
    assert "def _force_figure_safe_workspace_tool" in gizmo
    assert '{"builtin.move", "builtin.rotate", "builtin.scale"}' in gizmo
    assert 'bpy.ops.wm.tool_set_by_id(name=safe_tool)' in gizmo
    assert '_activate_awb_transform_workspace_tool(context, "MOVE")' in keymap
    assert 'bpy.ops.wm.tool_set_by_id(name="baw.select_object")' in keymap
    assert 'if state is None:\n            return "", ""' in gizmo
    assert "fit_host_present = fit_ui_state_present(context)" in keymap
    assert 'issues=("FIT_F3_SESSION_MISSING",)' in keymap
    assert "begin_fit_move_gesture" in gizmo
    assert "commit_fit_move_gesture" in gizmo
    assert "cancel_fit_move_gesture" in gizmo


def test_f3_session_freshness_is_full_and_fail_closed():
    session = _source("rigped_fit_session.py")
    runtime = _source("rigped_fit_runtime.py")

    assert "validate_fit_runtime_session(context.scene, session.token)" in session
    assert "FIT_F3_ACTIVE_OBJECT_CHANGED" in session
    assert "FIT_F3_NATIVE_SELECTION_CHANGED" in session
    assert "FIT_F3_OWNER_DATA_CHANGED" in session
    assert "FIT_F3_TOPOLOGY_CHANGED" in session
    assert "FIT_F3_NATIVE_REST_CHANGED" in session
    assert "FIT_F3_APPEARANCE_CHANGED" in session
    assert "FIT_ANIMATION_CHANGED_DURING_SESSION" in runtime
    assert "read_setup_descriptor(view)" in runtime
    assert "FIT_SESSION_SETUP_SIGNATURE_INVALID" in runtime


def test_f3_rejects_shear_and_apply_cannot_drop_semantic_draft():
    session = _source("rigped_fit_session.py")
    ui = _source("rigped_create_fit_ui.py")

    assert "normalized[left].dot(normalized[right])" in session
    assert "FIT_F3_OBJECT_MATRIX_UNSUPPORTED" in session
    assert "def fit_figure_move_available" in session
    assert "_frozen_world3(session)" in session
    assert "validate_fit_semantic_session(context, semantic_session)" in ui
    assert "semantic_session.draft.values != semantic_session.document.baseline_values" in ui
    assert "Fit Apply blocked until semantic draft commit is implemented (F4)" in ui


def test_f3_draw_snapshot_path_stays_lightweight_but_input_boundaries_are_full():
    session = _source("rigped_fit_session.py")
    keymap = _source("viewport_keymap.py")

    assert "def validate_fit_semantic_snapshot_access" in session
    snapshot_start = session.index("def fit_geometry_snapshot(context)")
    snapshot_end = session.index("def fit_geometry_part_by_name", snapshot_start)
    snapshot_block = session[snapshot_start:snapshot_end]
    assert "validate_fit_semantic_snapshot_access" in snapshot_block
    assert "validate_fit_semantic_session(context, session)" not in snapshot_block

    pivot_start = session.index("def fit_active_part_world_pivot_axes")
    pivot_end = session.index("def _frozen_world3", pivot_start)
    pivot_block = session[pivot_start:pivot_end]
    assert "validate_fit_semantic_snapshot_access" in pivot_block
    assert "validate_fit_semantic_session" not in pivot_block

    assert "FIT_SELECTION_CLICK_REFUSED" in keymap
    assert "FIT_SELECTION_BOX_REFUSED" in keymap
    assert "FIT_F3_TRANSFORM_REFUSED" in keymap
    assert "fit_state = fit_ui_state(context)" in keymap
    assert "(fit_state is None or session is None)" in keymap
    assert "validate_fit_semantic_session(context, session)" in keymap


def test_f3_commit_primitive_revalidates_canonical_setup_signature():
    runtime = _source("rigped_fit_runtime.py")

    start = runtime.index("def commit_fit_session")
    block = runtime[start:]
    assert "descriptor, descriptor_issues = read_setup_descriptor(view)" in block
    assert "descriptor.signature != session.setup_signature" in block
    assert "descriptor.revision != session.setup_revision" in block
    assert "FIT_SESSION_DESCRIPTOR_CHANGED" in block


def test_f3b1_com_rotate_has_explicit_capability_and_session_lifecycle():
    session = _source("rigped_fit_session.py")

    assert "fit_operations_for_role(" in session
    assert '"awb.com",' in _source("rigped_fit_commands.py")
    assert "class FitRotateGestureBaseline" in session
    assert "active_rotate_gesture: FitRotateGestureBaseline | None" in session
    assert "world_axis: tuple[float, float, float]" in session
    assert "def fit_figure_rotate_available" in session
    assert "def begin_fit_rotate_gesture" in session
    assert "def apply_fit_rotate_preview" in session
    assert "def commit_fit_rotate_gesture" in session
    assert "def cancel_fit_rotate_gesture" in session
    assert "world_axis=tuple(float(value) for value in frozen_axis)" in session
    assert "world_vector = Vector(gesture.world_axis)" in session
    assert "world3 = _frozen_world3(session)" in session
    assert "Quaternion(rig_axis, angle).normalized()" in session


def test_f3b1_com_rotate_keeps_center_compensation_in_pure_command():
    commands = _source("rigped_fit_commands.py")

    start = commands.index("def rotate_fit_part_rig_local")
    block = commands[start:]
    assert 'if role == "com":' in block
    assert "pivot = _vec_scale(_vec_add(rest.head, rest.tail), 0.5)" in block
    assert "rotate_vector(delta, _vec_sub(rest.head, pivot))" in block
    assert "attachment_offset=new_attachment" in block
    assert "local_orientation=new_local_orientation" in block


def test_f3_figure_rotate_matches_full_awb_gizmo_and_never_calls_native_rotate():
    session = _source("rigped_fit_session.py")
    gizmo = _source("global_transform_gizmo.py")
    keymap = _source("viewport_keymap.py")
    init = _source("__init__.py")

    assert 'return "FIGURE", "ROTATE"' in gizmo
    assert 'if self.mode in {"MOVE", "ROTATE", "SCALE"}' in keymap
    assert "BAW_OT_figure_fit_rotate_axis" in init
    assert '("FIGURE", BAW_OT_figure_fit_rotate_axis.bl_idname)' in gizmo
    assert 'handles = ("X", "Y", "Z", "VIEW", "FREE")' in gizmo

    start = gizmo.index("class BAW_OT_figure_fit_rotate_axis")
    end = gizmo.index("class BAW_GT_free_rotate_disk", start)
    block = gizmo[start:end]
    assert '("X", "X", "Rotate around X")' in block
    assert '("Y", "Y", "Rotate around Y")' in block
    assert '("Z", "Z", "Rotate around Z")' in block
    assert '("VIEW", "View", "Rotate around the current view axis")' in block
    assert '("FREE", "Free", "Free virtual-trackball rotation")' in block
    assert "bpy.ops.transform.rotate" not in block
    assert "apply_fit_rotate_preview" in block
    assert "apply_fit_rotate_quaternion_preview" in block
    assert "def apply_fit_rotate_quaternion_preview" in session


def test_f3_completion_scale_route_is_semantic_and_registered():
    session = _source("rigped_fit_session.py")
    gizmo = _source("global_transform_gizmo.py")
    keymap = _source("viewport_keymap.py")
    init = _source("__init__.py")
    commands = _source("rigped_fit_commands.py")

    assert "def fit_figure_scale_available" in session
    assert "class FitScaleGestureBaseline" in session
    assert "def begin_fit_scale_gesture" in session
    assert "def apply_fit_scale_preview" in session
    assert "def commit_fit_scale_gesture" in session
    assert "def cancel_fit_scale_gesture" in session
    assert 'return "FIGURE", "SCALE"' in gizmo
    assert "class BAW_OT_figure_fit_scale_axis" in gizmo
    assert '("FIGURE", BAW_OT_figure_fit_scale_axis.bl_idname)' in gizmo
    assert "BAW_OT_figure_fit_scale_axis" in init
    assert 'if self.mode in {"MOVE", "ROTATE", "SCALE"}' in keymap
    assert "if fit_mode_before == self.mode:" in keymap
    assert 'orientation_mode=fit_orientation_mode(context)' in gizmo
    assert 'if self.mode == "SCALE":\n                    set_fit_orientation_mode' not in keymap
    assert "def scale_fit_part_local" in commands
    assert "bpy.ops.transform.resize" not in gizmo[
        gizmo.index("class BAW_OT_figure_fit_scale_axis") :
        gizmo.index("class BAW_GT_free_rotate_disk")
    ]
