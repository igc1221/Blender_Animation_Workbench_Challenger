from __future__ import annotations

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from .phase4_contact_model import AWB_CONTACT_STATE_PROPERTY, ContactStateValue
from .semantic_adapter import assigned_channelbag

_DRAW_HANDLE = None
_POINT_SHADER = None
_SLIDING_STATE_CACHE_KEY = None
_SLIDING_STATE_CACHE_NAMES: tuple[str, ...] = ()

# Generated Rigped naming is stable by contract. The red dot is an animator-facing
# cue for the evaluated Hand/Foot pivot, not a second visible animation object.
# Draw it from the solved result terminal so it stays welded to the limb even
# though the hidden IK target owns the underlying Sliding solve.
_LIMB_TARGETS = (
    ("MCH_ForeArm.L", "MCH_Hand.L"),
    ("MCH_ForeArm.R", "MCH_Hand.R"),
    ("MCH_Calf.L", "MCH_Foot.L"),
    ("MCH_Calf.R", "MCH_Foot.R"),
)


def _point_shader():
    global _POINT_SHADER
    if _POINT_SHADER is None:
        _POINT_SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _POINT_SHADER


def _authoritative_contact_state_at_frame(
    state_bone,
    frame: float,
    curves_by_path: dict[str, object],
) -> int | None:
    """Resolve Contact state from one pre-indexed authored FCurve table."""

    if AWB_CONTACT_STATE_PROPERTY not in state_bone:
        return None
    try:
        state_path = state_bone.path_from_id(f'["{AWB_CONTACT_STATE_PROPERTY}"]')
    except (ReferenceError, TypeError, ValueError):
        return None

    state_curve = curves_by_path.get(state_path)
    if state_curve is None or not state_curve.keyframe_points:
        return None

    effective = None
    for point in state_curve.keyframe_points:
        time = float(point.co.x)
        if time <= frame + 1e-4 and (effective is None or time > effective[0]):
            effective = (time, float(point.co.y))
    if effective is None:
        return None
    value = effective[1]
    rounded = round(value)
    if abs(value - float(rounded)) > 1e-4:
        return None
    return int(rounded)


def _sliding_state_names_at_frame(rig, pose, frame: float) -> tuple[str, ...]:
    """Cache expensive Contact FCurve lookup across repeated viewport redraws."""

    global _SLIDING_STATE_CACHE_KEY, _SLIDING_STATE_CACHE_NAMES

    animation_data = getattr(rig, "animation_data", None)
    action = getattr(animation_data, "action", None)
    slot = getattr(animation_data, "action_slot", None)
    action_pointer = int(action.as_pointer()) if action is not None else 0
    slot_handle = int(getattr(slot, "handle", 0) or 0)
    raw_states = []
    for state_name, _target_name in _LIMB_TARGETS:
        state_bone = pose.get(state_name)
        if state_bone is None or AWB_CONTACT_STATE_PROPERTY not in state_bone:
            raw_states.append(None)
            continue
        try:
            raw_states.append(float(state_bone[AWB_CONTACT_STATE_PROPERTY]))
        except (TypeError, ValueError, ReferenceError):
            raw_states.append(None)

    cache_key = (
        int(rig.as_pointer()),
        action_pointer,
        slot_handle,
        round(float(frame), 6),
        tuple(raw_states),
    )
    if cache_key == _SLIDING_STATE_CACHE_KEY:
        return _SLIDING_STATE_CACHE_NAMES

    bag = assigned_channelbag(rig)
    if bag is None:
        _SLIDING_STATE_CACHE_KEY = cache_key
        _SLIDING_STATE_CACHE_NAMES = ()
        return ()

    # One FCurve scan per cache miss, not one full scan per limb per draw call.
    curves_by_path = {
        str(curve.data_path): curve
        for curve in bag.fcurves
        if int(getattr(curve, "array_index", 0)) == 0
    }
    names: list[str] = []
    for state_name, _target_name in _LIMB_TARGETS:
        state_bone = pose.get(state_name)
        if state_bone is None:
            continue
        state = _authoritative_contact_state_at_frame(
            state_bone,
            frame,
            curves_by_path,
        )
        if state == int(ContactStateValue.SLIDING):
            names.append(state_name)

    _SLIDING_STATE_CACHE_KEY = cache_key
    _SLIDING_STATE_CACHE_NAMES = tuple(names)
    return _SLIDING_STATE_CACHE_NAMES


def _ik_pivot_world_positions(context) -> tuple[tuple[float, float, float], ...]:
    if context.mode != "POSE":
        return ()
    rig = context.active_object
    if rig is None or getattr(rig, "type", None) != "ARMATURE":
        return ()

    pose = rig.pose.bones
    frame = float(context.scene.frame_current) + float(getattr(context.scene, "frame_subframe", 0.0))
    sliding_names = set(_sliding_state_names_at_frame(rig, pose, frame))
    if not sliding_names:
        return ()

    positions: list[tuple[float, float, float]] = []
    for state_name, target_name in _LIMB_TARGETS:
        if state_name not in sliding_names:
            continue
        target_bone = pose.get(target_name)
        if target_bone is None:
            continue
        world = rig.matrix_world @ target_bone.matrix.translation
        positions.append((float(world.x), float(world.y), float(world.z)))
    return tuple(positions)


def _draw_rigped_ik_pivots() -> None:
    context = bpy.context
    if context.area is None or context.area.type != "VIEW_3D":
        return
    positions = _ik_pivot_world_positions(context)
    if not positions:
        return

    shader = _point_shader()
    batch = batch_for_shader(shader, "POINTS", {"pos": positions})
    previous_blend = gpu.state.blend_get()
    previous_depth = gpu.state.depth_test_get()
    try:
        gpu.state.blend_set("ALPHA")
        # Biped-style contact pivot is an animator cue, so keep it readable even
        # when the target sits directly on/in front of the terminal control.
        gpu.state.depth_test_set("NONE")
        gpu.state.point_size_set(11.0)
        shader.bind()
        shader.uniform_float("color", (1.0, 0.035, 0.035, 1.0))
        batch.draw(shader)
    finally:
        gpu.state.point_size_set(1.0)
        gpu.state.depth_test_set(previous_depth)
        gpu.state.blend_set(previous_blend)


def _tag_all_view3d_redraw() -> None:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def register_rigped_ik_pivot_overlay() -> None:
    global _DRAW_HANDLE
    if _DRAW_HANDLE is not None:
        return
    _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        _draw_rigped_ik_pivots,
        (),
        "WINDOW",
        "POST_VIEW",
    )
    _tag_all_view3d_redraw()


def unregister_rigped_ik_pivot_overlay() -> None:
    global _DRAW_HANDLE, _POINT_SHADER, _SLIDING_STATE_CACHE_KEY, _SLIDING_STATE_CACHE_NAMES
    if _DRAW_HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
        _DRAW_HANDLE = None
    _POINT_SHADER = None
    _SLIDING_STATE_CACHE_KEY = None
    _SLIDING_STATE_CACHE_NAMES = ()
    _tag_all_view3d_redraw()
