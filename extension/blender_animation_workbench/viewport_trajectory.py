from __future__ import annotations

from math import ceil, floor
from time import perf_counter
from typing import ClassVar

import bpy
import gpu
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .rigped_contract import RIGPED_SETUP_PROPERTY
from .semantic_adapter import (
    active_control_for_context,
    control_context_for_context,
    runtime_control_key,
    trackbar_controls_for_context,
)
from .semantic_query import query_keys

_SELECTED_PATH_POINTS: dict[int, set[int]] = {}
_SELECTED_OVERLAY_BATCHES: dict[int, object | None] = {}
_LAST_LIVE_REFRESH: dict[tuple[int, ...], float] = {}
_LIVE_REFRESH_COST: dict[tuple[int, ...], float] = {}
_PATH_KEY_SIGNATURES: dict[int, tuple[float, ...]] = {}
_OWNER_TRANSFORM_SIGNATURES: dict[tuple[int, int], tuple[int, tuple]] = {}
_NATIVE_TRANSFORM_PREVIEW_BASE: dict[
    tuple[int, int],
    tuple[
        int,
        int,
        tuple[tuple[float, float, float], ...],
        tuple[float, float, float],
        int | None,
        int | None,
    ],
] = {}
_AWB_OBJECT_TRAJECTORY_NAMES: set[str] = set()
_AWB_POSE_TRAJECTORIES: set[tuple[str, str]] = set()
_AUTO_REFRESH_TIMER_PENDING = False
_AUTO_REFRESH_LAST_CHANGE = 0.0
_AUTO_REFRESH_SETTLE_DELAY = 0.08
_AUTO_REFRESH_LIVE_INTERVAL = 1.0 / 30.0
_AUTO_REFRESH_LIVE_CONTROL_LIMIT = 2
_HISTORY_REFRESH_TIMER_PENDING = False
_CONTEXT_REFRESH_TIMER_PENDING = False
_REFRESH_HANDLER_GUARD = False
_TRAJECTORY_SESSION_ACTIVE = False
_LAST_CONTEXT_CONTROL_IDS: tuple[str, ...] | None = None
_POINT_SHADER = None
_DRAW_HANDLE = None


def trajectory_visibility_enabled(context) -> bool:
    scene = getattr(context, "scene", None)
    return scene is None or bool(getattr(scene, "baw_trajectory_visible", True))


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


def _configure_motion_path_settings(settings, *, pose: bool) -> None:
    settings.type = "RANGE"
    settings.range = "KEYS_ALL"
    settings.frame_step = 1
    settings.show_frame_numbers = False
    settings.show_keyframe_highlight = True
    settings.show_keyframe_numbers = False
    if pose and hasattr(settings, "show_keyframe_action_all"):
        # Blender documents this whole-Action search as slower. AWB controls
        # already have an explicit PoseBone identity, so keep it disabled.
        settings.show_keyframe_action_all = False
    if pose and hasattr(settings, "bake_location"):
        settings.bake_location = "HEADS"
    if hasattr(settings, "use_camera_space_bake"):
        settings.use_camera_space_bake = False


def _configure_path_display(path) -> None:
    path.lines = True
    path.line_thickness = 2


def _restore_scene_frame(scene, frame: int) -> None:
    if int(scene.frame_current) != int(frame):
        scene.frame_set(int(frame))


def _point_shader():
    global _POINT_SHADER
    if _POINT_SHADER is None:
        _POINT_SHADER = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
    return _POINT_SHADER


def _rebuild_selected_overlay_batch(path, indices: set[int]) -> None:
    path_id = int(path.as_pointer())
    if bpy.app.background or not indices:
        _SELECTED_OVERLAY_BATCHES[path_id] = None
        return
    positions = tuple(
        tuple(float(value) for value in path.points[index].co)
        for index in sorted(indices)
        if 0 <= index < len(path.points)
    )
    if not positions:
        _SELECTED_OVERLAY_BATCHES[path_id] = None
        return
    _SELECTED_OVERLAY_BATCHES[path_id] = batch_for_shader(
        _point_shader(),
        "POINTS",
        {"pos": positions},
    )


def _note_trajectory_context(context, controls) -> None:
    global _LAST_CONTEXT_CONTROL_IDS, _TRAJECTORY_SESSION_ACTIVE
    if _REFRESH_HANDLER_GUARD:
        return

    control_ids = tuple(resolved.control.control_id for resolved in controls)
    for resolved in controls:
        if getattr(resolved.target, "motion_path", None) is None:
            continue
        _TRAJECTORY_SESSION_ACTIVE = True
        if context.mode == "OBJECT":
            _AWB_OBJECT_TRAJECTORY_NAMES.add(resolved.owner_object.name)
        elif context.mode == "POSE":
            _AWB_POSE_TRAJECTORIES.add((resolved.owner_object.name, resolved.target.name))

    previous = _LAST_CONTEXT_CONTROL_IDS
    _LAST_CONTEXT_CONTROL_IDS = control_ids
    if previous is not None and control_ids != previous and _TRAJECTORY_SESSION_ACTIVE:
        _queue_context_refresh()


def _draw_selected_trajectory_keys() -> None:
    context = bpy.context
    if (
        context.area is None
        or context.area.type != "VIEW_3D"
        or not trajectory_visibility_enabled(context)
    ):
        return

    controls = trackbar_controls_for_context(context)
    _note_trajectory_context(context, controls)
    if not controls:
        return
    active = active_control_for_context(context)
    active_control_id = active.control.control_id if active is not None else None

    drawable = []
    for resolved in controls:
        path = getattr(resolved.target, "motion_path", None)
        if path is None:
            continue
        batch = _SELECTED_OVERLAY_BATCHES.get(int(path.as_pointer()))
        if batch is not None:
            drawable.append((resolved.control.control_id, batch))
    if not drawable:
        return

    shader = _point_shader()
    previous_blend = gpu.state.blend_get()
    previous_depth = gpu.state.depth_test_get()
    try:
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        for control_id, batch in drawable:
            # The active control's selected trajectory keys stay unmistakable,
            # while selected keys on the rest of the multi-control set remain
            # visible without adding a second path renderer.
            gpu.state.point_size_set(11.0 if control_id == active_control_id else 8.0)
            batch.draw(shader)
    finally:
        gpu.state.point_size_set(1.0)
        gpu.state.depth_test_set(previous_depth)
        gpu.state.blend_set(previous_blend)


def register_trajectory_draw_handler() -> None:
    global _DRAW_HANDLE
    if _DRAW_HANDLE is not None:
        return
    _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        _draw_selected_trajectory_keys,
        (),
        "WINDOW",
        "POST_VIEW",
    )
    _tag_all_view3d_redraw()


def unregister_trajectory_draw_handler() -> None:
    global _DRAW_HANDLE, _POINT_SHADER
    if _DRAW_HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
        _DRAW_HANDLE = None
    clear_trajectory_runtime_cache()
    _SELECTED_OVERLAY_BATCHES.clear()
    _POINT_SHADER = None
    _tag_all_view3d_redraw()


def sync_active_trajectory_key_selection(context, *, semantic=None) -> bool:
    """Mirror Track Bar selection onto every trajectory in the current context.

    The historical function name is kept because Track Bar callers already use
    it, but Phase 1J multi-control Pose context means the sync target is now all
    selected AWB controls. Per-path work remains incremental: only point indices
    whose selection state changed are touched. A caller may pass the current
    read-cycle semantic query to avoid a second FCurve/key scan.
    """
    semantic = semantic if semantic is not None else query_keys(control_context_for_context(context))
    controls = semantic.context.controls
    if not controls:
        return False

    selected_by_control: dict[str, set[int]] = {
        resolved.control.control_id: set() for resolved in controls
    }
    for key in semantic.keys:
        if context.mode == "OBJECT" and "POSITION" not in key.channel_names:
            continue
        if key.selected and key.control.control_id in selected_by_control:
            selected_by_control[key.control.control_id].add(round(float(key.frame)))

    synced = False
    needs_redraw = False
    for resolved in controls:
        path = getattr(resolved.target, "motion_path", None)
        if path is None or not path.points or not hasattr(path.points[0], "select"):
            continue

        selected_frames = selected_by_control[resolved.control.control_id]
        first_frame = int(path.frame_start)
        desired_indices = {
            frame - first_frame
            for frame in selected_frames
            if 0 <= frame - first_frame < len(path.points)
        }

        path_id = int(path.as_pointer())
        previous_indices = _SELECTED_PATH_POINTS.get(path_id, set())
        changed_indices = previous_indices.symmetric_difference(desired_indices)
        # Native Undo/Redo can restore MotionPathPoint.select independently of
        # AWB's runtime cache. Reconcile the actual native point state too, or
        # a cache hit can incorrectly skip re-selecting a multi-control path.
        for index in previous_indices.union(desired_indices):
            if 0 <= index < len(path.points):
                should_select = index in desired_indices
                if bool(path.points[index].select) != should_select:
                    changed_indices.add(index)
        for index in changed_indices:
            if 0 <= index < len(path.points):
                path.points[index].select = index in desired_indices
        _SELECTED_PATH_POINTS[path_id] = desired_indices
        # Rebuild only each path's tiny selected-point batch. This also updates
        # cached world-space marker positions after native MotionPath refreshes.
        _rebuild_selected_overlay_batch(path, desired_indices)
        synced = True
        needs_redraw = needs_redraw or bool(changed_indices or desired_indices)

    if needs_redraw:
        _tag_all_view3d_redraw()
    return synced


def clear_trajectory_runtime_cache() -> None:
    _SELECTED_PATH_POINTS.clear()
    _LAST_LIVE_REFRESH.clear()
    _LIVE_REFRESH_COST.clear()
    _PATH_KEY_SIGNATURES.clear()
    _OWNER_TRANSFORM_SIGNATURES.clear()
    _NATIVE_TRANSFORM_PREVIEW_BASE.clear()
    _AWB_OBJECT_TRAJECTORY_NAMES.clear()
    _AWB_POSE_TRAJECTORIES.clear()


def reset_trajectory_runtime_state_for_verification() -> None:
    """Reset AWB trajectory runtime/session state between deterministic GUI phases.

    Verification fixtures rebuild unsaved Scene data through MCP/Python while the
    extension module itself stays loaded. A previous Phase 1L run can therefore
    leave the trajectory session active even after its objects are deleted. The
    product never calls this helper; the shared GUI baseline uses it only to make
    phase boundaries equivalent to a fresh AWB session without relaunching Blender.
    """
    global _AUTO_REFRESH_TIMER_PENDING, _HISTORY_REFRESH_TIMER_PENDING
    global _CONTEXT_REFRESH_TIMER_PENDING, _REFRESH_HANDLER_GUARD
    global _TRAJECTORY_SESSION_ACTIVE, _LAST_CONTEXT_CONTROL_IDS

    for callback in (
        _run_auto_refresh_timer,
        _run_context_refresh_timer,
        _run_history_refresh_timer,
    ):
        if bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)

    _AUTO_REFRESH_TIMER_PENDING = False
    _HISTORY_REFRESH_TIMER_PENDING = False
    _CONTEXT_REFRESH_TIMER_PENDING = False
    _REFRESH_HANDLER_GUARD = False
    _TRAJECTORY_SESSION_ACTIVE = False
    _LAST_CONTEXT_CONTROL_IDS = None
    clear_trajectory_runtime_cache()
    _SELECTED_OVERLAY_BATCHES.clear()


def _run_for_single_selected_object(context, obj, operator) -> set[str]:
    selected_before = list(context.selected_objects or [])
    active_before = context.view_layer.objects.active
    try:
        for selected in selected_before:
            selected.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return operator()
    finally:
        obj.select_set(False)
        for selected in selected_before:
            if selected.name in context.view_layer.objects:
                selected.select_set(True)
        if active_before is not None and active_before.name in context.view_layer.objects:
            context.view_layer.objects.active = active_before


def _run_for_single_selected_pose_bone(context, obj, pose_bone, operator) -> set[str]:
    armature = obj.data
    selected_before = [bone.name for bone in obj.pose.bones if bone.select]
    active_before = armature.bones.active.name if armature.bones.active is not None else None
    try:
        for bone in obj.pose.bones:
            bone.select = False
        pose_bone.select = True
        armature.bones.active = pose_bone.bone
        return operator()
    finally:
        for bone in obj.pose.bones:
            bone.select = bone.name in selected_before
        armature.bones.active = armature.bones.get(active_before) if active_before else None


def _control_key_metadata_for_context(
    context,
    *,
    semantic=None,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, ...]]]:
    semantic = semantic if semantic is not None else query_keys(control_context_for_context(context))
    ranges: dict[str, tuple[float, float]] = {}
    frames_by_control: dict[str, set[float]] = {}
    for key in semantic.keys:
        control_id = key.control.control_id
        frame = float(key.frame)
        current = ranges.get(control_id)
        if current is None:
            ranges[control_id] = (frame, frame)
        else:
            ranges[control_id] = (min(current[0], frame), max(current[1], frame))
        frames_by_control.setdefault(control_id, set()).add(round(frame, 6))
    signatures = {
        control_id: tuple(sorted(frames))
        for control_id, frames in frames_by_control.items()
    }
    return ranges, signatures


def _control_key_ranges_for_context(context) -> dict[str, tuple[float, float]]:
    return _control_key_metadata_for_context(context)[0]


def _active_control_key_range(
    context,
    key_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float] | None:
    resolved = active_control_for_context(context)
    if resolved is None:
        return None
    ranges = key_ranges if key_ranges is not None else _control_key_ranges_for_context(context)
    return ranges.get(resolved.control.control_id)


def _expected_native_path_bounds(key_range: tuple[float, float]) -> tuple[int, int]:
    first, last = key_range
    return floor(first), ceil(last) + 1


def _path_needs_recalculate(
    path,
    key_range: tuple[float, float],
    *,
    exact_range: bool,
    key_signature: tuple[float, ...] | None = None,
) -> bool:
    first, last = key_range
    if path is None:
        return True
    path_id = int(path.as_pointer())
    if key_signature is not None and _PATH_KEY_SIGNATURES.get(path_id) != key_signature:
        # paths_update() refreshes sampled coordinates but Blender does not
        # reliably rebuild native keyframe-highlight metadata. Any structural
        # key-time change therefore needs a bounded recalculate for that path.
        return True
    # Native paths_update() cannot grow the cached frame range. Expand only
    # when the edit actually leaves the current cache during a modal preview.
    if first < float(path.frame_start) or last >= float(path.frame_end):
        return True
    if not exact_range:
        return False
    expected_start, expected_end = _expected_native_path_bounds(key_range)
    return int(path.frame_start) != expected_start or int(path.frame_end) != expected_end


def _forget_path_runtime_state(path_id: int) -> None:
    _SELECTED_PATH_POINTS.pop(path_id, None)
    _SELECTED_OVERLAY_BATCHES.pop(path_id, None)
    _PATH_KEY_SIGNATURES.pop(path_id, None)


def _clear_object_motion_path(context, obj) -> bool:
    path = obj.motion_path
    if path is None:
        return True
    path_id = int(path.as_pointer())
    result = _run_for_single_selected_object(
        context,
        obj,
        lambda: bpy.ops.object.paths_clear(only_selected=True),
    )
    cleared = "FINISHED" in result and obj.motion_path is None
    if cleared:
        _forget_path_runtime_state(path_id)
        _AWB_OBJECT_TRAJECTORY_NAMES.discard(obj.name)
    return cleared


def _clear_pose_motion_path(context, obj, pose_bone) -> bool:
    path = pose_bone.motion_path
    if path is None:
        return True
    path_id = int(path.as_pointer())
    result = _run_for_single_selected_pose_bone(
        context,
        obj,
        pose_bone,
        lambda: bpy.ops.pose.paths_clear(only_selected=True),
    )
    cleared = "FINISHED" in result and pose_bone.motion_path is None
    if cleared:
        _forget_path_runtime_state(path_id)
        _AWB_POSE_TRAJECTORIES.discard((obj.name, pose_bone.name))
    return cleared


def refresh_active_object_trajectory(
    context,
    *,
    exact_range: bool = False,
    resolved=None,
    key_ranges: dict[str, tuple[float, float]] | None = None,
    key_signatures: dict[str, tuple[float, ...]] | None = None,
    sync_selection: bool = True,
    allow_recalculate: bool = True,
) -> bool:
    """Create/update one selected Object's Blender-native Motion Path cache."""
    if context.mode != "OBJECT":
        return False

    resolved = resolved if resolved is not None else active_control_for_context(context)
    if resolved is None:
        return False
    obj = resolved.owner_object
    initial_frame = int(context.scene.frame_current)
    if key_ranges is None or key_signatures is None:
        local_ranges, local_signatures = _control_key_metadata_for_context(context)
        if key_ranges is None:
            key_ranges = local_ranges
        if key_signatures is None:
            key_signatures = local_signatures
    control_id = resolved.control.control_id
    key_range = key_ranges.get(control_id)
    key_signature = key_signatures.get(control_id)
    if key_range is None:
        if exact_range and obj.motion_path is not None:
            cleared = _clear_object_motion_path(context, obj)
            _restore_scene_frame(context.scene, initial_frame)
            if cleared:
                _tag_all_view3d_redraw()
            return cleared
        return False
    if key_range[1] <= key_range[0] or (
        key_signature is not None and len(key_signature) < 2
    ):
        # Blender cannot calculate a native Motion Path whose keyed extent is a
        # single frame (for example 0..0 after deleting the last non-zero key).
        # A trajectory with fewer than two distinct keyed frames has no segment
        # to display, so clear any stale path instead of calling paths_calculate.
        if obj.motion_path is not None:
            cleared = _clear_object_motion_path(context, obj)
            _restore_scene_frame(context.scene, initial_frame)
            if cleared:
                _tag_all_view3d_redraw()
            return cleared
        return False

    anim_data = getattr(obj, "animation_data", None)
    if anim_data is None or anim_data.action is None:
        return False

    settings = obj.animation_visualization.motion_path
    _configure_motion_path_settings(settings, pose=False)
    recalculate = _path_needs_recalculate(
        obj.motion_path,
        key_range,
        exact_range=exact_range,
        key_signature=key_signature,
    )
    structural_change_pending = recalculate
    if recalculate and not allow_recalculate:
        path = obj.motion_path
        if (
            path is None
            or key_range[0] < float(path.frame_start)
            or key_range[1] >= float(path.frame_end)
        ):
            return False
        # During a native gizmo drag, keep the existing cache allocation and
        # only refresh coordinates. Final settle will rebuild native key-marker
        # metadata if the Auto Key operation inserted a new key time.
        recalculate = False

    if recalculate and obj.motion_path is not None and not _clear_object_motion_path(context, obj):
        _restore_scene_frame(context.scene, initial_frame)
        return False

    if recalculate:
        result = _run_for_single_selected_object(
            context,
            obj,
            lambda: bpy.ops.object.paths_calculate(
                display_type="RANGE",
                range="KEYS_ALL",
            ),
        )
    else:
        result = _run_for_single_selected_object(
            context,
            obj,
            bpy.ops.object.paths_update,
        )

    _restore_scene_frame(context.scene, initial_frame)
    if "FINISHED" not in result or obj.motion_path is None:
        return False

    _configure_path_display(obj.motion_path)
    _AWB_OBJECT_TRAJECTORY_NAMES.add(obj.name)
    if key_signature is not None and (allow_recalculate or not structural_change_pending):
        _PATH_KEY_SIGNATURES[int(obj.motion_path.as_pointer())] = key_signature
    if sync_selection:
        sync_active_trajectory_key_selection(context)
    _tag_all_view3d_redraw()
    return True


def refresh_selected_object_trajectories(
    context,
    *,
    exact_range: bool = False,
    allow_recalculate: bool = True,
) -> bool:
    """Refresh all selected Object-mode trajectories with one semantic-key scan."""
    if context.mode != "OBJECT":
        return False
    semantic = query_keys(control_context_for_context(context))
    controls = semantic.context.controls
    if not controls:
        return False

    key_ranges, key_signatures = _control_key_metadata_for_context(
        context,
        semantic=semantic,
    )
    active_before = context.view_layer.objects.active
    refreshed_any = False
    try:
        for resolved in controls:
            context.view_layer.objects.active = resolved.owner_object
            refreshed_any = (
                refresh_active_object_trajectory(
                    context,
                    exact_range=exact_range,
                    resolved=resolved,
                    key_ranges=key_ranges,
                    key_signatures=key_signatures,
                    sync_selection=False,
                    allow_recalculate=allow_recalculate,
                )
                or refreshed_any
            )
    finally:
        if active_before is not None and active_before.name in context.view_layer.objects:
            context.view_layer.objects.active = active_before

    if refreshed_any:
        sync_active_trajectory_key_selection(context, semantic=semantic)
        _tag_all_view3d_redraw()
    return refreshed_any


def refresh_active_pose_bone_trajectory(
    context,
    *,
    exact_range: bool = False,
    key_ranges: dict[str, tuple[float, float]] | None = None,
    key_signatures: dict[str, tuple[float, ...]] | None = None,
    sync_selection: bool = True,
    allow_recalculate: bool = True,
) -> bool:
    """Create/update only the active PoseBone's Blender-native Motion Path cache."""
    obj = context.active_object
    pose_bone = context.active_pose_bone
    if context.mode != "POSE" or obj is None or obj.type != "ARMATURE" or pose_bone is None:
        return False
    initial_frame = int(context.scene.frame_current)
    resolved = active_control_for_context(context)
    if resolved is None:
        return False
    if key_ranges is None or key_signatures is None:
        local_ranges, local_signatures = _control_key_metadata_for_context(context)
        if key_ranges is None:
            key_ranges = local_ranges
        if key_signatures is None:
            key_signatures = local_signatures
    control_id = resolved.control.control_id
    key_range = key_ranges.get(control_id)
    key_signature = key_signatures.get(control_id)
    if key_range is None:
        if exact_range and pose_bone.motion_path is not None:
            cleared = _clear_pose_motion_path(context, obj, pose_bone)
            _restore_scene_frame(context.scene, initial_frame)
            if cleared:
                _tag_all_view3d_redraw()
            return cleared
        return False
    if key_range[1] <= key_range[0] or (
        key_signature is not None and len(key_signature) < 2
    ):
        # Match Object mode: a one-frame keyed extent cannot produce a native
        # Motion Path segment. Clear the stale bone path instead of asking
        # Blender to calculate an invalid start/end range.
        if pose_bone.motion_path is not None:
            cleared = _clear_pose_motion_path(context, obj, pose_bone)
            _restore_scene_frame(context.scene, initial_frame)
            if cleared:
                _tag_all_view3d_redraw()
            return cleared
        return False

    anim_data = getattr(obj, "animation_data", None)
    if anim_data is None or anim_data.action is None:
        return False

    settings = obj.pose.animation_visualization.motion_path
    _configure_motion_path_settings(settings, pose=True)
    recalculate = _path_needs_recalculate(
        pose_bone.motion_path,
        key_range,
        exact_range=exact_range,
        key_signature=key_signature,
    )
    structural_change_pending = recalculate
    if recalculate and not allow_recalculate:
        path = pose_bone.motion_path
        if (
            path is None
            or key_range[0] < float(path.frame_start)
            or key_range[1] >= float(path.frame_end)
        ):
            return False
        recalculate = False

    if (
        recalculate
        and pose_bone.motion_path is not None
        and not _clear_pose_motion_path(context, obj, pose_bone)
    ):
        _restore_scene_frame(context.scene, initial_frame)
        return False

    if recalculate:
        result = _run_for_single_selected_pose_bone(
            context,
            obj,
            pose_bone,
            lambda: bpy.ops.pose.paths_calculate(
                display_type="RANGE",
                range="KEYS_ALL",
                bake_location="HEADS",
            ),
        )
    else:
        result = _run_for_single_selected_pose_bone(
            context,
            obj,
            pose_bone,
            bpy.ops.pose.paths_update,
        )

    _restore_scene_frame(context.scene, initial_frame)
    if "FINISHED" not in result or pose_bone.motion_path is None:
        return False

    _configure_path_display(pose_bone.motion_path)
    _AWB_POSE_TRAJECTORIES.add((obj.name, pose_bone.name))
    if key_signature is not None and (allow_recalculate or not structural_change_pending):
        _PATH_KEY_SIGNATURES[int(pose_bone.motion_path.as_pointer())] = key_signature
    if sync_selection:
        sync_active_trajectory_key_selection(context)
    _tag_all_view3d_redraw()
    return True


def refresh_selected_pose_trajectories(
    context,
    *,
    exact_range: bool = False,
    allow_recalculate: bool = True,
) -> bool:
    """Refresh the current Phase 1J selected PoseBone trajectory set.

    Semantic-key range collection and selection synchronization are each done
    once for the whole set so multi-control modal edits scale linearly with the
    number of affected controls rather than repeating Track Bar enumeration for
    every bone.
    """
    obj = context.active_object
    if context.mode != "POSE" or obj is None or obj.type != "ARMATURE":
        return False

    semantic = query_keys(control_context_for_context(context))
    controls = semantic.context.controls
    if not controls:
        return False
    key_ranges, key_signatures = _control_key_metadata_for_context(
        context,
        semantic=semantic,
    )
    armature = obj.data
    active_before = armature.bones.active.name if armature.bones.active is not None else None
    refreshed_any = False
    try:
        for resolved in controls:
            pose_bone = resolved.target
            armature.bones.active = pose_bone.bone
            refreshed_any = (
                refresh_active_pose_bone_trajectory(
                    context,
                    exact_range=exact_range,
                    key_ranges=key_ranges,
                    key_signatures=key_signatures,
                    sync_selection=False,
                    allow_recalculate=allow_recalculate,
                )
                or refreshed_any
            )
    finally:
        armature.bones.active = armature.bones.get(active_before) if active_before else None

    if refreshed_any:
        sync_active_trajectory_key_selection(context, semantic=semantic)
        _tag_all_view3d_redraw()
    return refreshed_any


def refresh_active_trajectory(
    context,
    *,
    exact_range: bool = False,
    allow_recalculate: bool = True,
) -> bool:
    global _LAST_CONTEXT_CONTROL_IDS, _TRAJECTORY_SESSION_ACTIVE
    if not trajectory_visibility_enabled(context):
        return False
    if context.mode == "OBJECT":
        refreshed = refresh_selected_object_trajectories(
            context,
            exact_range=exact_range,
            allow_recalculate=allow_recalculate,
        )
    elif context.mode == "POSE":
        refreshed = refresh_selected_pose_trajectories(
            context,
            exact_range=exact_range,
            allow_recalculate=allow_recalculate,
        )
    else:
        return False

    if refreshed:
        controls = trackbar_controls_for_context(context)
        _TRAJECTORY_SESSION_ACTIVE = True
        _LAST_CONTEXT_CONTROL_IDS = tuple(
            resolved.control.control_id for resolved in controls
        )
    return refreshed


def _adaptive_live_refresh_interval(refresh_key: tuple[int, ...]) -> float:
    """Return a cost-aware cadence that keeps modal path work bounded.

    Cheap paths can update at 60 Hz. Expensive paths reserve roughly 60% of
    the interval for the rest of Blender/UI work, with a 12 Hz floor so live
    feedback remains visibly continuous on long paths.
    """
    previous_cost = _LIVE_REFRESH_COST.get(refresh_key)
    if previous_cost is None:
        return 1.0 / 30.0
    return min(1.0 / 12.0, max(1.0 / 60.0, previous_cost * 2.5))


def refresh_active_trajectory_live(
    context,
    *,
    force: bool = False,
    exact_range: bool = False,
    min_interval: float | None = None,
    allow_recalculate: bool = True,
) -> bool:
    """Refresh the current trajectory set at an adaptive bounded modal cadence."""
    controls = trackbar_controls_for_context(context)
    path_controls = [
        resolved
        for resolved in controls
        if getattr(resolved.target, "motion_path", None) is not None
    ]
    if not path_controls:
        if force and _TRAJECTORY_SESSION_ACTIVE:
            return refresh_active_trajectory(
                context,
                exact_range=exact_range,
                allow_recalculate=allow_recalculate,
            )
        return False

    owner_pointer = int(path_controls[0].owner_object.as_pointer())
    target_pointers = sorted(int(resolved.target.as_pointer()) for resolved in path_controls)
    refresh_key = (owner_pointer, *target_pointers)
    interval = (
        max(0.0, float(min_interval))
        if min_interval is not None
        else _adaptive_live_refresh_interval(refresh_key)
    )
    now = perf_counter()
    previous = _LAST_LIVE_REFRESH.get(refresh_key, 0.0)
    if not force and now - previous < interval:
        return False

    started = perf_counter()
    refreshed = refresh_active_trajectory(
        context,
        exact_range=exact_range,
        allow_recalculate=allow_recalculate,
    )
    finished = perf_counter()
    if refreshed:
        cost = finished - started
        previous_cost = _LIVE_REFRESH_COST.get(refresh_key)
        _LIVE_REFRESH_COST[refresh_key] = (
            cost if previous_cost is None else previous_cost * 0.7 + cost * 0.3
        )
        _LAST_LIVE_REFRESH[refresh_key] = finished
    return refreshed


def _semantic_move_drag_active(context) -> bool:
    window_manager = getattr(context, "window_manager", None)
    return bool(
        window_manager is not None
        and getattr(window_manager, "baw_rigped_semantic_move_drag_active", False)
    )


def _active_generated_rigped(context) -> bool:
    active = getattr(context, "active_object", None)
    if active is None or getattr(active, "type", None) != "ARMATURE":
        return False
    getter = getattr(active, "get", None)
    return bool(callable(getter) and getter(RIGPED_SETUP_PROPERTY))


def _context_has_trajectory(context) -> bool:
    if context.mode not in {"OBJECT", "POSE"}:
        return False
    return any(
        getattr(resolved.target, "motion_path", None) is not None
        for resolved in trackbar_controls_for_context(context)
    )


def _live_native_trajectory_refresh_allowed(context) -> bool:
    """Use bounded live native-path preview for the common 1-2 control case."""
    scene = getattr(context, "scene", None)
    if (
        scene is None
        or not bool(getattr(scene, "baw_trajectory_live_preview", True))
        or bool(getattr(scene, "baw_trajectory_edit_mode", False))
    ):
        return False
    controls = tuple(trackbar_controls_for_context(context))
    if context.mode == "POSE" and any(
        bool(getattr(resolved.owner_object, "get", lambda *_args: None)(RIGPED_SETUP_PROPERTY))
        for resolved in controls
    ):
        # Generated Rigped pose solves can touch several constrained bones for
        # one visible control movement. Re-evaluating native Motion Paths at
        # 30 Hz during that solve makes the viewport feel dramatically slower.
        # Preserve the path, but refresh once the transform settles instead.
        return False
    path_control_count = sum(
        getattr(resolved.target, "motion_path", None) is not None
        for resolved in controls
    )
    return 0 < path_control_count <= _AUTO_REFRESH_LIVE_CONTROL_LIMIT


def _live_control_world_position(resolved) -> Vector:
    target = resolved.target
    if isinstance(target, bpy.types.PoseBone):
        return Vector(resolved.owner_object.matrix_world @ target.head)
    return Vector(target.matrix_world.translation)


def _position_key_frames_for_control(
    context,
    control_id: str,
    *,
    semantic=None,
) -> tuple[int, ...]:
    semantic = semantic if semantic is not None else query_keys(control_context_for_context(context))
    return tuple(
        sorted(
            {
                round(float(key.frame))
                for key in semantic.keys
                if key.control.control_id == control_id and "POSITION" in key.channel_names
            }
        )
    )


def _preview_segment_weight(
    sample_frame: int,
    current_frame: int,
    previous_frame: int | None,
    next_frame: int | None,
) -> float:
    """Smoothly spread the live key delta across its adjacent trajectory arcs."""
    if sample_frame == current_frame:
        return 1.0
    if sample_frame < current_frame:
        if previous_frame is None or sample_frame < previous_frame:
            return 0.0
        span = current_frame - previous_frame
        if span <= 0:
            return 0.0
        t = (sample_frame - previous_frame) / span
    else:
        if next_frame is None or sample_frame > next_frame:
            return 0.0
        span = next_frame - current_frame
        if span <= 0:
            return 0.0
        t = (next_frame - sample_frame) / span
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _preview_current_native_transform_path_points(context) -> int:
    """Deform the adjacent cached trajectory arcs from the live transform.

    Blender updates Object/PoseBone transforms during a native gizmo drag before
    Auto Key commits the FCurve value. A normal Motion Path update therefore
    still evaluates the old animation until release. Keep an immutable snapshot
    of the native path at drag start and spread the current world-space delta
    over the previous-key -> current -> next-key arc. The final native refresh
    after release/cancel remains authoritative.
    """
    scene = getattr(context, "scene", None)
    if scene is None:
        return 0
    frame = int(scene.frame_current)
    semantic = query_keys(control_context_for_context(context))
    changed = 0
    live_control_keys: set[tuple[int, int]] = set()
    for resolved in semantic.context.controls:
        path = getattr(resolved.target, "motion_path", None)
        if path is None:
            continue
        control_id = resolved.control.control_id
        control_key = runtime_control_key(resolved)
        live_control_keys.add(control_key)
        index = frame - int(path.frame_start)
        if not (0 <= index < len(path.points)):
            continue
        path_id = int(path.as_pointer())
        preview = _NATIVE_TRANSFORM_PREVIEW_BASE.get(control_key)
        if (
            preview is None
            or int(preview[0]) != frame
            or int(preview[1]) != path_id
            or len(preview[2]) != len(path.points)
        ):
            key_frames = _position_key_frames_for_control(
                context,
                control_id,
                semantic=semantic,
            )
            previous_frame = max((key for key in key_frames if key < frame), default=None)
            next_frame = min((key for key in key_frames if key > frame), default=None)
            base_points = tuple(
                tuple(float(value) for value in point.co)
                for point in path.points
            )
            base_current = base_points[index]
            preview = (
                frame,
                path_id,
                base_points,
                base_current,
                previous_frame,
                next_frame,
            )
            _NATIVE_TRANSFORM_PREVIEW_BASE[control_key] = preview

        _, _, base_points, base_current, previous_frame, next_frame = preview
        world = _live_control_world_position(resolved)
        delta = world - Vector(base_current)
        if delta.length_squared <= 1e-12:
            continue
        for point_index, point in enumerate(path.points):
            sample_frame = int(path.frame_start) + point_index
            weight = _preview_segment_weight(
                sample_frame,
                frame,
                previous_frame,
                next_frame,
            )
            if weight <= 0.0:
                continue
            point.co = Vector(base_points[point_index]) + delta * weight
        selected_indices = _SELECTED_PATH_POINTS.get(path_id)
        if selected_indices:
            _rebuild_selected_overlay_batch(path, selected_indices)
        changed += 1

    for control_key in tuple(_NATIVE_TRANSFORM_PREVIEW_BASE):
        if control_key not in live_control_keys:
            _NATIVE_TRANSFORM_PREVIEW_BASE.pop(control_key, None)
    if changed:
        _tag_all_view3d_redraw()
    return changed


def _native_transform_modal_is_active(context) -> bool:
    """Keep native Motion Path operators out of an active gizmo transform."""
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return False
    for window in wm.windows:
        for operator in window.modal_operators:
            identifiers = {
                str(getattr(operator, "bl_idname", "")),
                str(getattr(operator, "idname", "")),
                str(getattr(getattr(operator, "bl_rna", None), "identifier", "")),
            }
            lowered = {identifier.lower() for identifier in identifiers if identifier}
            if any(
                identifier in {
                    "transform_ot_translate",
                    "transform_ot_rotate",
                    "transform_ot_resize",
                }
                or "gizmo_tweak" in identifier
                for identifier in lowered
            ):
                return True
    return False


def _control_transform_signature(resolved) -> tuple:
    target = resolved.target
    rotation_mode = str(getattr(target, "rotation_mode", "XYZ"))
    if rotation_mode == "QUATERNION":
        rotation = target.rotation_quaternion
    elif rotation_mode == "AXIS_ANGLE":
        rotation = target.rotation_axis_angle
    else:
        rotation = target.rotation_euler
    return (
        tuple(round(float(value), 9) for value in target.location),
        rotation_mode,
        tuple(round(float(value), 9) for value in rotation),
        tuple(round(float(value), 9) for value in target.scale),
    )


def _owner_transform_value_changed(context) -> bool:
    """Return True only for same-frame control transform value changes.

    Native Motion Path evaluation can dirty the animated owner after a refresh.
    Treating every owner depsgraph update as a new animation edit creates a
    self-refresh loop that can compete with the next native gizmo transform.
    """
    scene = context.scene
    frame = int(scene.frame_current)
    live_control_keys: set[tuple[int, int]] = set()
    changed = False
    for resolved in trackbar_controls_for_context(context):
        if getattr(resolved.target, "motion_path", None) is None:
            continue
        control_key = runtime_control_key(resolved)
        live_control_keys.add(control_key)
        signature = _control_transform_signature(resolved)
        previous = _OWNER_TRANSFORM_SIGNATURES.get(control_key)
        _OWNER_TRANSFORM_SIGNATURES[control_key] = (frame, signature)
        if previous is None:
            continue
        previous_frame, previous_signature = previous
        if int(previous_frame) == frame and previous_signature != signature:
            changed = True

    for control_key in tuple(_OWNER_TRANSFORM_SIGNATURES):
        if control_key not in live_control_keys:
            _OWNER_TRANSFORM_SIGNATURES.pop(control_key, None)
    return changed


def _depsgraph_trajectory_change_kind(context, depsgraph) -> str | None:
    controls = trackbar_controls_for_context(context)
    owner_ids: set[int] = set()
    action_ids: set[int] = set()
    for resolved in controls:
        if getattr(resolved.target, "motion_path", None) is None:
            continue
        owner_ids.add(int(resolved.owner_object.as_pointer()))
        anim_data = getattr(resolved.owner_object, "animation_data", None)
        action = getattr(anim_data, "action", None) if anim_data is not None else None
        if action is not None:
            action_ids.add(int(action.as_pointer()))
    if not owner_ids and not action_ids:
        return None

    owner_touched = False
    for update in depsgraph.updates:
        updated_id = update.id
        original = getattr(updated_id, "original", updated_id)
        if not hasattr(original, "as_pointer"):
            continue
        pointer = int(original.as_pointer())
        if pointer in action_ids:
            return "ACTION"
        if pointer in owner_ids:
            owner_touched = True
    return "OWNER" if owner_touched else None


def _run_auto_refresh_timer():
    global _AUTO_REFRESH_TIMER_PENDING, _REFRESH_HANDLER_GUARD
    if bpy.app.background or _REFRESH_HANDLER_GUARD:
        _AUTO_REFRESH_TIMER_PENDING = False
        return None

    context = bpy.context
    if _semantic_move_drag_active(context):
        return 0.05
    scene = getattr(context, "scene", None)
    if scene is None or not _context_has_trajectory(context):
        _AUTO_REFRESH_TIMER_PENDING = False
        return None

    # The depsgraph stream is the authoritative native-transform signal on
    # Blender 5.2. For the normal 1-2 control case, refresh existing Motion
    # Path coordinates at a bounded cadence while the gizmo is still moving.
    # Never clear/recalculate native path allocation mid-transform; structural
    # key-time changes are rebuilt once the input stream settles.
    elapsed = perf_counter() - _AUTO_REFRESH_LAST_CHANGE
    if elapsed < _AUTO_REFRESH_SETTLE_DELAY:
        if _live_native_trajectory_refresh_allowed(context):
            _REFRESH_HANDLER_GUARD = True
            try:
                _preview_current_native_transform_path_points(context)
            finally:
                _REFRESH_HANDLER_GUARD = False
            return _AUTO_REFRESH_LIVE_INTERVAL
        return max(0.01, _AUTO_REFRESH_SETTLE_DELAY - elapsed)

    # Once input settles, avoid the final structural rebuild until Blender has
    # actually left any transform operator it exposes through modal_operators.
    if _native_transform_modal_is_active(context):
        return 0.03

    _AUTO_REFRESH_TIMER_PENDING = False
    _NATIVE_TRANSFORM_PREVIEW_BASE.clear()
    _REFRESH_HANDLER_GUARD = True
    try:
        # Coalesce depsgraph bursts to ~30 Hz. Key-time signature changes force
        # a bounded native recalculate; value-only edits stay on paths_update().
        refresh_active_trajectory_live(context, force=True)
    finally:
        _REFRESH_HANDLER_GUARD = False
    return


def _queue_auto_refresh() -> None:
    global _AUTO_REFRESH_TIMER_PENDING
    if bpy.app.background or _AUTO_REFRESH_TIMER_PENDING:
        return
    _AUTO_REFRESH_TIMER_PENDING = True
    bpy.app.timers.register(_run_auto_refresh_timer, first_interval=1.0 / 30.0)


def _clear_stale_trajectories(context) -> bool:
    """Keep AWB native paths scoped to the currently selected control set."""
    controls = trackbar_controls_for_context(context)
    current_object_names = {
        resolved.owner_object.name
        for resolved in controls
        if context.mode == "OBJECT"
    }
    current_pose_ids = {
        (resolved.owner_object.name, resolved.target.name)
        for resolved in controls
        if context.mode == "POSE"
    }

    cleared_any = False
    for name in tuple(_AWB_OBJECT_TRAJECTORY_NAMES.difference(current_object_names)):
        obj = bpy.data.objects.get(name)
        if obj is None:
            _AWB_OBJECT_TRAJECTORY_NAMES.discard(name)
            continue
        if getattr(obj, "motion_path", None) is None:
            _AWB_OBJECT_TRAJECTORY_NAMES.discard(name)
            continue
        cleared_any = _clear_object_motion_path(context, obj) or cleared_any

    for obj_name, bone_name in tuple(_AWB_POSE_TRAJECTORIES.difference(current_pose_ids)):
        obj = bpy.data.objects.get(obj_name)
        pose_bone = (
            obj.pose.bones.get(bone_name)
            if obj is not None and obj.type == "ARMATURE" and obj.pose is not None
            else None
        )
        if pose_bone is None:
            _AWB_POSE_TRAJECTORIES.discard((obj_name, bone_name))
            continue
        if getattr(pose_bone, "motion_path", None) is None:
            _AWB_POSE_TRAJECTORIES.discard((obj_name, bone_name))
            continue
        cleared_any = _clear_pose_motion_path(context, obj, pose_bone) or cleared_any
    return cleared_any


def _run_context_refresh_timer():
    global _CONTEXT_REFRESH_TIMER_PENDING, _REFRESH_HANDLER_GUARD
    _CONTEXT_REFRESH_TIMER_PENDING = False
    if bpy.app.background or _REFRESH_HANDLER_GUARD or not _TRAJECTORY_SESSION_ACTIVE:
        return
    context = bpy.context
    if context.mode not in {"OBJECT", "POSE"}:
        return

    _REFRESH_HANDLER_GUARD = True
    try:
        cleared = _clear_stale_trajectories(context)
        refreshed = refresh_active_trajectory(context, exact_range=True)
        if cleared and not refreshed:
            _tag_all_view3d_redraw()
    finally:
        _REFRESH_HANDLER_GUARD = False
    return


def _queue_context_refresh() -> None:
    global _CONTEXT_REFRESH_TIMER_PENDING
    if bpy.app.background or _CONTEXT_REFRESH_TIMER_PENDING:
        return
    _CONTEXT_REFRESH_TIMER_PENDING = True
    bpy.app.timers.register(_run_context_refresh_timer, first_interval=0.0)


def _run_history_refresh_timer():
    global _HISTORY_REFRESH_TIMER_PENDING, _REFRESH_HANDLER_GUARD
    _HISTORY_REFRESH_TIMER_PENDING = False
    if bpy.app.background or _REFRESH_HANDLER_GUARD:
        return
    context = bpy.context
    if not _context_has_trajectory(context):
        return

    _REFRESH_HANDLER_GUARD = True
    try:
        refresh_active_trajectory(context, exact_range=True)
    finally:
        _REFRESH_HANDLER_GUARD = False
    return


def _queue_history_refresh() -> None:
    global _HISTORY_REFRESH_TIMER_PENDING
    if bpy.app.background or _HISTORY_REFRESH_TIMER_PENDING:
        return
    _HISTORY_REFRESH_TIMER_PENDING = True
    bpy.app.timers.register(_run_history_refresh_timer, first_interval=0.0)


@persistent
def _trajectory_depsgraph_update_post(scene, depsgraph) -> None:
    global _AUTO_REFRESH_LAST_CHANGE

    if bpy.app.background or _REFRESH_HANDLER_GUARD:
        return
    context = bpy.context
    if _semantic_move_drag_active(context):
        return
    if getattr(context, "scene", None) is not scene:
        return
    if _active_generated_rigped(context) and _native_transform_modal_is_active(context):
        # During a native Rigped W/E/R drag, do not repeatedly rebuild semantic
        # controls or Motion Path preview data. Keep one cheap debounce alive and
        # perform the authoritative path refresh after the transform releases.
        _AUTO_REFRESH_LAST_CHANGE = perf_counter()
        _queue_auto_refresh()
        return
    if not _context_has_trajectory(context):
        return

    change_kind = _depsgraph_trajectory_change_kind(context, depsgraph)
    if change_kind == "ACTION":
        _AUTO_REFRESH_LAST_CHANGE = perf_counter()
        # Key insertion/deletion/time edits must update native key markers even
        # when AWB Auto Key is off and the user edits through Blender itself.
        _queue_auto_refresh()
    elif change_kind == "OWNER":
        # Motion Path evaluation itself can dirty the owner. Only a same-frame
        # P/R/S value change is evidence of a real control transform; otherwise
        # queuing here would let a refresh schedule itself forever. This also
        # keeps visible paths responsive when Blender Auto Key is off.
        if _owner_transform_value_changed(context):
            _AUTO_REFRESH_LAST_CHANGE = perf_counter()
            _queue_auto_refresh()


@persistent
def _trajectory_history_post(*_args) -> None:
    _queue_history_refresh()


def register_trajectory_change_handlers() -> None:
    if _trajectory_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_trajectory_depsgraph_update_post)
    if _trajectory_history_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_trajectory_history_post)
    if _trajectory_history_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_trajectory_history_post)


def unregister_trajectory_change_handlers() -> None:
    global _AUTO_REFRESH_TIMER_PENDING, _AUTO_REFRESH_LAST_CHANGE
    global _CONTEXT_REFRESH_TIMER_PENDING
    global _HISTORY_REFRESH_TIMER_PENDING, _LAST_CONTEXT_CONTROL_IDS
    global _TRAJECTORY_SESSION_ACTIVE
    for handlers, callback in (
        (bpy.app.handlers.depsgraph_update_post, _trajectory_depsgraph_update_post),
        (bpy.app.handlers.undo_post, _trajectory_history_post),
        (bpy.app.handlers.redo_post, _trajectory_history_post),
    ):
        if callback in handlers:
            handlers.remove(callback)
    for callback in (
        _run_auto_refresh_timer,
        _run_context_refresh_timer,
        _run_history_refresh_timer,
    ):
        if bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    _AUTO_REFRESH_TIMER_PENDING = False
    _AUTO_REFRESH_LAST_CHANGE = 0.0
    _CONTEXT_REFRESH_TIMER_PENDING = False
    _HISTORY_REFRESH_TIMER_PENDING = False
    _LAST_CONTEXT_CONTROL_IDS = None
    _TRAJECTORY_SESSION_ACTIVE = False


def clear_active_trajectories(context) -> bool:
    """Clear the native Motion Paths owned by the current AWB control context."""
    global _TRAJECTORY_SESSION_ACTIVE, _LAST_CONTEXT_CONTROL_IDS

    controls = trackbar_controls_for_context(context)
    cleared_any = False
    if context.mode == "OBJECT":
        for resolved in controls:
            obj = resolved.owner_object
            if getattr(obj, "motion_path", None) is not None:
                cleared_any = _clear_object_motion_path(context, obj) or cleared_any
    elif context.mode == "POSE":
        for resolved in controls:
            pose_bone = resolved.target
            if getattr(pose_bone, "motion_path", None) is not None:
                cleared_any = (
                    _clear_pose_motion_path(context, resolved.owner_object, pose_bone)
                    or cleared_any
                )

    _TRAJECTORY_SESSION_ACTIVE = False
    _LAST_CONTEXT_CONTROL_IDS = None
    clear_trajectory_runtime_cache()
    _SELECTED_OVERLAY_BATCHES.clear()
    _tag_all_view3d_redraw()
    return cleared_any


class BAW_OT_toggle_trajectory_visibility(bpy.types.Operator):
    bl_idname = "baw.toggle_trajectory_visibility"
    bl_label = "Toggle Trajectory"
    bl_description = "Show or hide AWB Motion Paths for the current control selection"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return getattr(context, "scene", None) is not None

    def execute(self, context):
        scene = context.scene
        enabled = not bool(getattr(scene, "baw_trajectory_visible", True))
        scene.baw_trajectory_visible = enabled
        if enabled:
            refreshed = refresh_active_trajectory(context, exact_range=True)
            if bool(getattr(scene, "baw_auto_key_enabled", False)):
                # Motion Path calculation evaluates the animation state. When
                # AUTO was already enabled before Trajectory, that evaluation
                # can invalidate AWB's transform snapshot/frame baseline until
                # the user manually cycles AUTO. Re-arm the existing AUTO
                # session here without changing its visible state.
                from .trackbar_keying import resync_awb_auto_key_after_external_evaluation

                resync_awb_auto_key_after_external_evaluation(context)
            if not refreshed:
                self.report(
                    {"INFO"},
                    "Trajectory is ON; select an animated AWB control to show a path",
                )
        else:
            scene.baw_trajectory_edit_mode = False
            scene.baw_trajectory_add_key_mode = False
            scene.baw_trajectory_transform_mode = "MOVE"
            from .trajectory_edit import (
                _set_native_tool_gizmo_hidden,
                _set_path_edit_flag,
                restore_trajectory_edit_view_state,
            )

            _set_path_edit_flag(context, False)
            _set_native_tool_gizmo_hidden(context, False)
            restore_trajectory_edit_view_state()
            clear_active_trajectories(context)
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_refresh_trajectory(bpy.types.Operator):
    bl_idname = "baw.refresh_trajectory"
    bl_label = "Refresh Trajectory"
    bl_description = "Build or update native Blender motion paths for the current AWB control context"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        controls = trackbar_controls_for_context(context)
        if not controls:
            return False
        obj = controls[0].owner_object
        anim_data = getattr(obj, "animation_data", None)
        return anim_data is not None and anim_data.action is not None

    def execute(self, context):
        if hasattr(context.scene, "baw_trajectory_visible"):
            context.scene.baw_trajectory_visible = True
        if not refresh_active_trajectory(context, exact_range=True):
            self.report({"WARNING"}, "Active AWB control has no refreshable animation trajectory")
            return {"CANCELLED"}
        return {"FINISHED"}
