from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import atan2, hypot
from typing import ClassVar

import bpy
import gpu
from bpy.props import EnumProperty
from bpy_extras.view3d_utils import (
    location_3d_to_region_2d,
    region_2d_to_location_3d,
    region_2d_to_origin_3d,
    region_2d_to_vector_3d,
)
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.geometry import intersect_line_plane

from .semantic_adapter import (
    active_control_for_context,
    channel_binding_token,
    channels_for_property,
    trackbar_controls_for_context,
)
from .semantic_query import semantic_keys_for_context
from .trackbar_drawing import TRACKBAR_INTERACTION_HEIGHT
from .trackbar_key_edit import (
    apply_multi_key_preview_transaction,
    apply_selection_range_scale_transaction,
    restore_multi_key_preview_transaction,
    restore_selection_range_scale_transaction,
    snapshot_multi_key_preview_transaction,
    snapshot_selection_range_scale_transaction,
)
from .trackbar_model import clear_key_selection_for_context
from .viewport_trajectory import (
    refresh_active_trajectory,
    refresh_active_trajectory_live,
    sync_active_trajectory_key_selection,
)

_HIT_RADIUS_PX = 14.0
_PATH_HIT_RADIUS_PX = 10.0
_DRAG_THRESHOLD_PX = 2.0
_TRAJECTORY_EDIT_KEYMAP_ITEMS: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []
_PREVIOUS_SHOW_GIZMO_TOOL: dict[int, bool] = {}
_SLIDE_GIZMO_MODAL_PIVOT: Vector | None = None
_TANGENT_DRAW_HANDLE = None
_TANGENT_LINE_SHADER = None
_TANGENT_POINT_SHADER = None
_TANGENT_HIT_RADIUS_PX = 11.0


@dataclass(frozen=True)
class TrajectoryAxisSnapshot:
    axis: int
    value: float
    handle_left_y: float
    handle_right_y: float


@dataclass(frozen=True)
class TrajectoryPositionSnapshot:
    control_id: str
    owner_name: str
    resolved: object
    binding_token: tuple[int | None, int | None, int | None, int | None]
    frame: float
    world_position: tuple[float, float, float]
    world_to_local: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    axes: tuple[TrajectoryAxisSnapshot, ...]


@dataclass(frozen=True)
class TrajectoryTangentAxisSnapshot:
    axis: int
    key_value: float
    handle_left: tuple[float, float]
    handle_right: tuple[float, float]
    handle_left_type: str
    handle_right_type: str


@dataclass(frozen=True)
class TrajectoryTangentSnapshot:
    control_id: str
    owner_name: str
    resolved: object
    binding_token: tuple[int | None, int | None, int | None, int | None]
    frame: float
    world_position: tuple[float, float, float]
    world_to_local: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    axes: tuple[TrajectoryTangentAxisSnapshot, ...]
    left_available: bool
    right_available: bool


@dataclass
class TrajectorySlideOwnerTransaction:
    control_id: str
    owner_name: str
    resolved: object
    binding_token: tuple[int | None, int | None, int | None, int | None]
    source_frames: tuple[int, ...]
    min_delta: int
    max_delta: int
    sampled_values: dict[int, tuple[float, float, float]]
    transaction: object


@dataclass
class TrajectoryRetimeOwnerTransaction:
    control_id: str
    owner_name: str
    resolved: object
    binding_token: tuple[int | None, int | None, int | None, int | None]
    source_frames: tuple[int, ...]
    source_handle_frame: int
    transaction: object


def _location_fcurves_for_control(resolved) -> dict[int, object]:
    """Resolve only the current control's concrete XYZ Position channels."""
    return {
        channel.array_index: channel.fcurve
        for channel in channels_for_property(resolved, "location")
        if 0 <= channel.array_index <= 2
    }


def _bound_location_fcurves(
    resolved,
    binding_token: tuple[int | None, int | None, int | None, int | None],
) -> dict[int, object]:
    """Resolve XYZ channels only while the captured Action/Slot/Bag binding matches."""
    try:
        if channel_binding_token(resolved.owner_object) != binding_token:
            return {}
        return _location_fcurves_for_control(resolved)
    except ReferenceError:
        return {}


def _trajectory_edit_mode_supported(context) -> bool:
    return context.mode in {"OBJECT", "POSE"}


def _active_trajectory_control_id(context) -> str | None:
    resolved = active_control_for_context(context)
    return resolved.control.control_id if resolved is not None else None


def _key_at_frame(fcurve, frame: float, epsilon: float = 1e-4):
    return next(
        (
            key
            for key in fcurve.keyframe_points
            if abs(float(key.co.x) - float(frame)) <= epsilon
        ),
        None,
    )


def _selected_position_identities(context) -> set[tuple[str, int]]:
    return {
        (key.control.control_id, round(float(key.frame)))
        for key in semantic_keys_for_context(context)
        if key.selected and "POSITION" in key.channel_names
    }


def _control_for_id(context, control_id: str):
    return next(
        (
            resolved
            for resolved in trackbar_controls_for_context(context)
            if resolved.control.control_id == control_id
        ),
        None,
    )


def _delete_selected_trajectory_position_keys(context) -> bool:
    identities = sorted(_selected_position_identities(context))
    if not _trajectory_edit_mode_supported(context) or not identities:
        return False

    controls = {
        resolved.control.control_id: resolved
        for resolved in trackbar_controls_for_context(context)
    }
    changed = False
    touched_objects = set()
    for control_id, frame in identities:
        resolved = controls.get(control_id)
        if resolved is None:
            continue
        obj = resolved.owner_object
        fcurves = _location_fcurves_for_control(resolved)
        for axis in range(3):
            fcurve = fcurves.get(axis)
            if fcurve is None:
                continue
            key = _key_at_frame(fcurve, frame)
            if key is None:
                continue
            fcurve.keyframe_points.remove(key)
            fcurve.update()
            changed = True
            touched_objects.add(obj.name)

    if not changed:
        return False

    for name in touched_objects:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})

    context.scene.frame_set(context.scene.frame_current)
    selected_frames = sorted(
        round(float(key.frame))
        for key in semantic_keys_for_context(context)
        if key.selected
    )
    context.scene.baw_has_selected_key = bool(selected_frames)
    if selected_frames:
        context.scene.baw_selected_key_frame = int(selected_frames[-1])
    sync_active_trajectory_key_selection(context)
    refresh_active_trajectory_live(context, force=True, exact_range=True)
    if context.area is not None:
        context.area.tag_redraw()
    return True


def _set_control_position_key_selection(
    context,
    control_id: str,
    frame: int,
    *,
    mode: str,
    epsilon: float = 1e-4,
) -> bool:
    if mode not in {"SET", "ADD", "TOGGLE", "SUB"}:
        raise ValueError(f"Unsupported trajectory selection mode: {mode}")
    resolved = _control_for_id(context, control_id)
    if resolved is None or not _trajectory_edit_mode_supported(context):
        return False

    fcurves = _location_fcurves_for_control(resolved)
    target_keys = [
        key
        for fcurve in fcurves.values()
        if (key := _key_at_frame(fcurve, frame, epsilon)) is not None
    ]
    if not target_keys:
        return False

    if mode == "SET":
        clear_key_selection_for_context(context)
        select_target = True
    elif mode == "ADD":
        select_target = True
    elif mode == "SUB":
        select_target = False
    else:
        select_target = not any(bool(key.select_control_point) for key in target_keys)

    for key in target_keys:
        key.select_control_point = select_target
        key.select_left_handle = select_target
        key.select_right_handle = select_target

    scene = context.scene
    selected_frames = sorted(frame for _control, frame in _selected_position_identities(context))
    scene.baw_has_selected_key = bool(selected_frames)
    if selected_frames:
        scene.baw_selected_key_frame = int(frame if select_target else selected_frames[-1])
    sync_active_trajectory_key_selection(context)
    if context.area is not None:
        context.area.tag_redraw()
    return True


def _trajectory_key_hit(context, mouse_x: float, mouse_y: float):
    region = context.region
    region_data = context.region_data
    if (
        not _trajectory_edit_mode_supported(context)
        or region is None
        or region_data is None
        or mouse_y <= TRACKBAR_INTERACTION_HEIGHT
    ):
        return None

    controls = {
        resolved.control.control_id: resolved
        for resolved in trackbar_controls_for_context(context)
    }
    if not controls:
        return None

    active_id = _active_trajectory_control_id(context)
    candidates = []
    for key in semantic_keys_for_context(context):
        if "POSITION" not in key.channel_names:
            continue
        resolved = controls.get(key.control.control_id)
        if resolved is None:
            continue
        path = getattr(resolved.target, "motion_path", None)
        if path is None:
            continue
        frame = round(float(key.frame))
        index = frame - int(path.frame_start)
        if not (0 <= index < len(path.points)):
            continue
        world = Vector(path.points[index].co)
        screen = location_3d_to_region_2d(region, region_data, world)
        if screen is None:
            continue
        distance = hypot(float(screen.x) - mouse_x, float(screen.y) - mouse_y)
        if distance > _HIT_RADIUS_PX:
            continue
        # Distance wins. Active object only breaks effectively-overlapping ties.
        active_bias = 0 if key.control.control_id == active_id else 1
        candidates.append((distance, active_bias, key.control.control_id, frame, world, key.selected))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0]


def _trajectory_path_hit(context, mouse_x: float, mouse_y: float):
    region = context.region
    region_data = context.region_data
    if (
        not _trajectory_edit_mode_supported(context)
        or region is None
        or region_data is None
        or mouse_y <= TRACKBAR_INTERACTION_HEIGHT
    ):
        return None

    controls = trackbar_controls_for_context(context)
    if not controls:
        return None

    active_id = _active_trajectory_control_id(context)
    mouse = Vector((float(mouse_x), float(mouse_y)))
    candidates = []
    for resolved in controls:
        path = getattr(resolved.target, "motion_path", None)
        if path is None or len(path.points) < 2:
            continue
        projected = []
        for point in path.points:
            world = Vector(point.co)
            screen = location_3d_to_region_2d(region, region_data, world)
            projected.append(
                None if screen is None else Vector((float(screen.x), float(screen.y)))
            )

        for index in range(len(path.points) - 1):
            start = projected[index]
            end = projected[index + 1]
            if start is None or end is None:
                continue
            segment = end - start
            length_squared = float(segment.length_squared)
            if length_squared <= 1e-9:
                parameter = 0.0
                nearest = start
            else:
                parameter = max(
                    0.0,
                    min(1.0, float((mouse - start).dot(segment)) / length_squared),
                )
                nearest = start + segment * parameter
            distance = float((mouse - nearest).length)
            if distance > _PATH_HIT_RADIUS_PX:
                continue

            frame_float = float(path.frame_start) + float(index) + parameter
            frame = round(frame_float)
            first_frame = int(path.frame_start)
            last_frame = int(path.frame_start) + len(path.points) - 1
            frame = max(first_frame, min(last_frame, frame))
            target_index = frame - first_frame
            world = Vector(path.points[target_index].co)
            active_bias = 0 if resolved.control.control_id == active_id else 1
            candidates.append(
                (
                    distance,
                    active_bias,
                    resolved.control.control_id,
                    frame,
                    world,
                )
            )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0]


def _ensure_trajectory_position_key(context, control_id: str, frame: int) -> tuple[bool, bool]:
    resolved = _control_for_id(context, control_id)
    if resolved is None or not _trajectory_edit_mode_supported(context):
        return False, False
    obj = resolved.owner_object
    target = resolved.target
    if _has_active_constraints(target):
        return False, False
    if isinstance(target, bpy.types.PoseBone) and bool(target.bone.use_connect):
        return False, False

    existing = _location_fcurves_for_control(resolved)
    if set(existing) == {0, 1, 2} and all(
        _key_at_frame(existing[axis], frame) is not None for axis in range(3)
    ):
        return False, True

    scene = context.scene
    initial_frame = int(scene.frame_current)
    initial_subframe = float(getattr(scene, "frame_subframe", 0.0))
    inserted = False
    try:
        scene.frame_set(int(frame))
        # Explicitly insert X/Y/Z one by one. Blender's broad location insert
        # can respect existing/available-channel state and leave a partial
        # Position triplet when the Action began as an X-only trajectory.
        for axis in range(3):
            current_curves = _location_fcurves_for_control(resolved)
            current_curve = current_curves.get(axis)
            if current_curve is not None and _key_at_frame(current_curve, frame) is not None:
                continue
            insert_kwargs = {
                "data_path": "location",
                "index": axis,
                "frame": float(frame),
            }
            if context.mode == "OBJECT":
                insert_kwargs["group"] = "AWB Trajectory"
            inserted = bool(target.keyframe_insert(**insert_kwargs)) or inserted
    finally:
        scene.frame_set(initial_frame, subframe=initial_subframe)

    fcurves = _location_fcurves_for_control(resolved)
    complete = set(fcurves) == {0, 1, 2} and all(
        _key_at_frame(fcurves[axis], frame) is not None for axis in range(3)
    )
    if complete:
        obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})
        context.view_layer.update()
    return inserted, complete


def _add_trajectory_position_keys(
    context,
    identities: tuple[tuple[str, int], ...],
) -> tuple[bool, bool]:
    changed = False
    selectable: list[tuple[str, int]] = []
    for control_id, frame in identities:
        inserted, complete = _ensure_trajectory_position_key(context, control_id, frame)
        changed = changed or inserted
        if complete:
            selectable.append((control_id, int(frame)))

    if not selectable:
        return changed, False

    clear_key_selection_for_context(context)
    selected = False
    for control_id, frame in selectable:
        selected = (
            _set_control_position_key_selection(
                context,
                control_id,
                frame,
                mode="ADD",
            )
            or selected
        )

    scene = context.scene
    scene.baw_has_selected_key = selected
    if selected:
        scene.baw_selected_key_frame = int(selectable[-1][1])
    if changed:
        refresh_active_trajectory_live(context, force=True, exact_range=True)
    else:
        sync_active_trajectory_key_selection(context)
    if context.area is not None:
        context.area.tag_redraw()
    return changed, selected


def _has_active_constraints(obj) -> bool:
    return any(
        not bool(getattr(constraint, "mute", False))
        and float(getattr(constraint, "influence", 1.0)) > 0.0
        for constraint in obj.constraints
    )


def _world_to_local_delta_matrix(obj) -> Matrix:
    if obj.parent is None:
        return Matrix.Identity(3)
    parent_space = obj.parent.matrix_world @ obj.matrix_parent_inverse
    return parent_space.to_3x3().inverted_safe()


def _world_to_control_location_delta_matrix(resolved) -> Matrix | None:
    target = resolved.target
    obj = resolved.owner_object
    if not isinstance(target, bpy.types.PoseBone):
        return _world_to_local_delta_matrix(obj)
    if bool(target.bone.use_connect):
        return None

    # PoseBone.location is expressed in the bone's LOCAL transform space, while
    # MotionPath points are world-space. Blender's convert_space() already owns
    # the parent/rest/inherit-scale/local-location rules, so derive the affine
    # location basis from it instead of duplicating armature matrix math here.
    base = obj.convert_space(
        pose_bone=target,
        matrix=Matrix.Identity(4),
        from_space="LOCAL",
        to_space="WORLD",
    ).translation
    columns: list[Vector] = []
    for axis in range(3):
        unit = Vector((0.0, 0.0, 0.0))
        unit[axis] = 1.0
        world = obj.convert_space(
            pose_bone=target,
            matrix=Matrix.Translation(unit),
            from_space="LOCAL",
            to_space="WORLD",
        ).translation
        columns.append(Vector(world) - Vector(base))
    local_to_world = Matrix(
        tuple(tuple(float(value) for value in column) for column in columns)
    ).transposed()
    if abs(float(local_to_world.determinant())) <= 1e-10:
        return None
    return local_to_world.inverted_safe()


def _trajectory_position_snapshots_for_context(context) -> tuple[TrajectoryPositionSnapshot, ...]:
    if not _trajectory_edit_mode_supported(context):
        return ()
    identities = sorted(_selected_position_identities(context))
    if not identities:
        return ()

    controls = {
        resolved.control.control_id: resolved
        for resolved in trackbar_controls_for_context(context)
    }
    scene = context.scene
    initial_frame = int(scene.frame_current)
    snapshots: list[TrajectoryPositionSnapshot] = []
    try:
        for control_id, frame in identities:
            resolved = controls.get(control_id)
            if resolved is None:
                continue
            obj = resolved.owner_object
            target = resolved.target
            if _has_active_constraints(target):
                continue
            fcurves = _location_fcurves_for_control(resolved)
            if set(fcurves) != {0, 1, 2}:
                continue
            axis_snapshots = []
            for axis in range(3):
                key = _key_at_frame(fcurves[axis], frame)
                if key is None:
                    axis_snapshots = []
                    break
                axis_snapshots.append(
                    TrajectoryAxisSnapshot(
                        axis=axis,
                        value=float(key.co.y),
                        handle_left_y=float(key.handle_left.y),
                        handle_right_y=float(key.handle_right.y),
                    )
                )
            if len(axis_snapshots) != 3:
                continue

            path = getattr(target, "motion_path", None)
            if path is None:
                continue
            path_index = frame - int(path.frame_start)
            if not (0 <= path_index < len(path.points)):
                continue

            scene.frame_set(frame)
            world_to_local = _world_to_control_location_delta_matrix(resolved)
            if world_to_local is None:
                continue
            snapshots.append(
                TrajectoryPositionSnapshot(
                    control_id=control_id,
                    owner_name=obj.name,
                    resolved=resolved,
                    binding_token=channel_binding_token(obj),
                    frame=float(frame),
                    world_position=tuple(float(value) for value in path.points[path_index].co),
                    world_to_local=tuple(
                        tuple(float(world_to_local[row][column]) for column in range(3))
                        for row in range(3)
                    ),
                    axes=tuple(axis_snapshots),
                )
            )
    finally:
        if int(scene.frame_current) != initial_frame:
            scene.frame_set(initial_frame)
    return tuple(snapshots)


def _tangent_segment_is_bezier(fcurve, key, side: str) -> bool:
    points = list(fcurve.keyframe_points)
    try:
        index = points.index(key)
    except ValueError:
        return False
    side_name = str(side).upper()
    if side_name == "LEFT":
        return index > 0 and str(points[index - 1].interpolation) == "BEZIER"
    if side_name == "RIGHT":
        return index + 1 < len(points) and str(key.interpolation) == "BEZIER"
    raise ValueError(f"Unsupported tangent side: {side!r}")


def _trajectory_tangent_snapshots_for_context(context) -> tuple[TrajectoryTangentSnapshot, ...]:
    if not _trajectory_edit_mode_supported(context):
        return ()
    identities = sorted(_selected_position_identities(context))
    if not identities:
        return ()
    controls = {
        resolved.control.control_id: resolved
        for resolved in trackbar_controls_for_context(context)
    }
    snapshots: list[TrajectoryTangentSnapshot] = []
    for control_id, frame in identities:
        resolved = controls.get(control_id)
        if resolved is None:
            continue
        obj = resolved.owner_object
        target = resolved.target
        if _has_active_constraints(target):
            continue
        fcurves = _location_fcurves_for_control(resolved)
        if set(fcurves) != {0, 1, 2}:
            continue
        axis_snapshots: list[TrajectoryTangentAxisSnapshot] = []
        keys = {}
        for axis in range(3):
            key = _key_at_frame(fcurves[axis], frame)
            if key is None:
                axis_snapshots = []
                break
            keys[axis] = key
            axis_snapshots.append(
                TrajectoryTangentAxisSnapshot(
                    axis=axis,
                    key_value=float(key.co.y),
                    handle_left=(float(key.handle_left.x), float(key.handle_left.y)),
                    handle_right=(float(key.handle_right.x), float(key.handle_right.y)),
                    handle_left_type=str(key.handle_left_type),
                    handle_right_type=str(key.handle_right_type),
                )
            )
        if len(axis_snapshots) != 3:
            continue
        path = getattr(target, "motion_path", None)
        if path is None:
            continue
        path_index = int(frame) - int(path.frame_start)
        if not (0 <= path_index < len(path.points)):
            continue
        world_to_local = _world_to_control_location_delta_matrix(resolved)
        if world_to_local is None:
            continue
        snapshots.append(
            TrajectoryTangentSnapshot(
                control_id=control_id,
                owner_name=obj.name,
                resolved=resolved,
                binding_token=channel_binding_token(obj),
                frame=float(frame),
                world_position=tuple(float(value) for value in path.points[path_index].co),
                world_to_local=tuple(
                    tuple(float(world_to_local[row][column]) for column in range(3))
                    for row in range(3)
                ),
                axes=tuple(axis_snapshots),
                left_available=all(
                    _tangent_segment_is_bezier(fcurves[axis], keys[axis], "LEFT")
                    for axis in range(3)
                ),
                right_available=all(
                    _tangent_segment_is_bezier(fcurves[axis], keys[axis], "RIGHT")
                    for axis in range(3)
                ),
            )
        )
    return tuple(snapshots)


def _trajectory_tangent_world_endpoint(
    snapshot: TrajectoryTangentSnapshot,
    side: str,
) -> Vector:
    side_name = str(side).upper()
    local_delta = Vector((0.0, 0.0, 0.0))
    for axis_snapshot in snapshot.axes:
        handle = (
            axis_snapshot.handle_left
            if side_name == "LEFT"
            else axis_snapshot.handle_right
        )
        local_delta[axis_snapshot.axis] = float(handle[1]) - float(axis_snapshot.key_value)
    local_to_world = Matrix(snapshot.world_to_local).inverted_safe()
    return Vector(snapshot.world_position) + local_to_world @ local_delta


def _restore_tangent_axis_key(key, axis_snapshot: TrajectoryTangentAxisSnapshot) -> None:
    key.handle_left_type = axis_snapshot.handle_left_type
    key.handle_right_type = axis_snapshot.handle_right_type
    key.handle_left = axis_snapshot.handle_left
    key.handle_right = axis_snapshot.handle_right


def _align_opposite_tangent_handle(
    key,
    axis_snapshot: TrajectoryTangentAxisSnapshot,
    side: str,
    target_y: float,
) -> None:
    frame = float(key.co.x)
    value = float(axis_snapshot.key_value)
    side_name = str(side).upper()
    clicked = axis_snapshot.handle_left if side_name == "LEFT" else axis_snapshot.handle_right
    opposite = axis_snapshot.handle_right if side_name == "LEFT" else axis_snapshot.handle_left
    clicked_vector = Vector((float(clicked[0]) - frame, float(target_y) - value))
    opposite_vector = Vector((float(opposite[0]) - frame, float(opposite[1]) - value))
    if clicked_vector.length <= 1e-9 or opposite_vector.length <= 1e-9:
        return
    target_opposite = -clicked_vector.normalized() * opposite_vector.length
    target = (frame + float(target_opposite.x), value + float(target_opposite.y))
    if side_name == "LEFT":
        key.handle_right = target
    else:
        key.handle_left = target


def _apply_trajectory_tangent_world_target(
    context,
    snapshot: TrajectoryTangentSnapshot,
    side: str,
    target_world: Vector,
    *,
    refresh: bool = True,
) -> bool:
    resolved = snapshot.resolved
    obj = resolved.owner_object
    fcurves = _bound_location_fcurves(resolved, snapshot.binding_token)
    if set(fcurves) != {0, 1, 2}:
        return False
    side_name = str(side).upper()
    local_delta = Matrix(snapshot.world_to_local) @ (
        Vector(target_world) - Vector(snapshot.world_position)
    )
    changed = False
    for axis_snapshot in snapshot.axes:
        fcurve = fcurves.get(axis_snapshot.axis)
        if fcurve is None:
            continue
        key = _key_at_frame(fcurve, snapshot.frame)
        if key is None:
            continue
        _restore_tangent_axis_key(key, axis_snapshot)
        target_y = float(axis_snapshot.key_value) + float(local_delta[axis_snapshot.axis])
        clicked_type = (
            axis_snapshot.handle_left_type
            if side_name == "LEFT"
            else axis_snapshot.handle_right_type
        )
        opposite_type = (
            axis_snapshot.handle_right_type
            if side_name == "LEFT"
            else axis_snapshot.handle_left_type
        )
        if clicked_type in {"AUTO", "AUTO_CLAMPED"}:
            if side_name == "LEFT":
                key.handle_left_type = "ALIGNED"
            else:
                key.handle_right_type = "ALIGNED"
            if opposite_type in {"AUTO", "AUTO_CLAMPED"}:
                if side_name == "LEFT":
                    key.handle_right_type = "ALIGNED"
                else:
                    key.handle_left_type = "ALIGNED"
                _align_opposite_tangent_handle(key, axis_snapshot, side_name, target_y)
        elif clicked_type == "VECTOR":
            if side_name == "LEFT":
                key.handle_left_type = "FREE"
            else:
                key.handle_right_type = "FREE"
        elif clicked_type == "ALIGNED" and opposite_type == "ALIGNED":
            _align_opposite_tangent_handle(key, axis_snapshot, side_name, target_y)

        if side_name == "LEFT":
            key.handle_left.y = target_y
        else:
            key.handle_right.y = target_y
        fcurve.update()
        changed = True
    if changed:
        obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})
        context.scene.frame_set(context.scene.frame_current)
        if refresh:
            refresh_active_trajectory_live(context)
        if context.area is not None:
            context.area.tag_redraw()
    return changed


def _restore_trajectory_tangent_snapshot(
    context,
    snapshot: TrajectoryTangentSnapshot,
    *,
    refresh: bool = True,
) -> bool:
    resolved = snapshot.resolved
    obj = resolved.owner_object
    fcurves = _bound_location_fcurves(resolved, snapshot.binding_token)
    restored = False
    for axis_snapshot in snapshot.axes:
        fcurve = fcurves.get(axis_snapshot.axis)
        if fcurve is None:
            continue
        key = _key_at_frame(fcurve, snapshot.frame)
        if key is None:
            continue
        _restore_tangent_axis_key(key, axis_snapshot)
        fcurve.update()
        restored = True
    if restored:
        obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})
        context.scene.frame_set(context.scene.frame_current)
        if refresh:
            refresh_active_trajectory_live(context, force=True)
        sync_active_trajectory_key_selection(context)
        if context.area is not None:
            context.area.tag_redraw()
    return restored


def _trajectory_tangent_handle_hit(context, mouse_x: float, mouse_y: float):
    if context.region is None or context.region_data is None:
        return None
    candidates = []
    for snapshot in _trajectory_tangent_snapshots_for_context(context):
        for side_name, available in (
            ("LEFT", snapshot.left_available),
            ("RIGHT", snapshot.right_available),
        ):
            if not available:
                continue
            world = _trajectory_tangent_world_endpoint(snapshot, side_name)
            screen = location_3d_to_region_2d(context.region, context.region_data, world)
            if screen is None:
                continue
            distance = hypot(float(screen.x) - mouse_x, float(screen.y) - mouse_y)
            if distance <= _TANGENT_HIT_RADIUS_PX:
                candidates.append((distance, snapshot.control_id, snapshot.frame, side_name, world, snapshot))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0]


def _tangent_handle_color(snapshot: TrajectoryTangentSnapshot, side: str):
    side_name = str(side).upper()
    types = {
        axis.handle_left_type if side_name == "LEFT" else axis.handle_right_type
        for axis in snapshot.axes
    }
    if types and types.issubset({"AUTO", "AUTO_CLAMPED"}):
        return (0.45, 0.78, 1.0, 1.0)
    if types == {"ALIGNED"}:
        return (1.0, 0.78, 0.18, 1.0)
    if types == {"VECTOR"}:
        return (0.35, 0.88, 0.35, 1.0)
    return (0.28, 0.9, 0.38, 1.0) if side_name == "LEFT" else (0.22, 0.48, 1.0, 1.0)


def _tangent_line_shader():
    global _TANGENT_LINE_SHADER
    if _TANGENT_LINE_SHADER is None:
        _TANGENT_LINE_SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _TANGENT_LINE_SHADER


def _tangent_point_shader():
    global _TANGENT_POINT_SHADER
    if _TANGENT_POINT_SHADER is None:
        _TANGENT_POINT_SHADER = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
    return _TANGENT_POINT_SHADER


def _draw_trajectory_tangent_handles() -> None:
    context = bpy.context
    scene = getattr(context, "scene", None)
    if (
        scene is None
        or context.area is None
        or context.area.type != "VIEW_3D"
        or not bool(getattr(scene, "baw_trajectory_edit_mode", False))
        or str(getattr(scene, "baw_trajectory_transform_mode", "MOVE")) != "TANGENT"
        or not _trajectory_edit_mode_supported(context)
    ):
        return
    snapshots = _trajectory_tangent_snapshots_for_context(context)
    if not snapshots:
        return

    line_shader = _tangent_line_shader()
    point_shader = _tangent_point_shader()
    previous_blend = gpu.state.blend_get()
    previous_depth = gpu.state.depth_test_get()
    try:
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        gpu.state.line_width_set(2.0)
        for snapshot in snapshots:
            key_world = Vector(snapshot.world_position)
            for side_name, available in (
                ("LEFT", snapshot.left_available),
                ("RIGHT", snapshot.right_available),
            ):
                if not available:
                    continue
                endpoint = _trajectory_tangent_world_endpoint(snapshot, side_name)
                color = _tangent_handle_color(snapshot, side_name)
                line_batch = batch_for_shader(
                    line_shader,
                    "LINES",
                    {"pos": (tuple(key_world), tuple(endpoint))},
                )
                line_shader.bind()
                line_shader.uniform_float("color", color)
                line_batch.draw(line_shader)
                point_batch = batch_for_shader(
                    point_shader,
                    "POINTS",
                    {"pos": (tuple(endpoint),)},
                )
                point_shader.bind()
                point_shader.uniform_float("color", color)
                gpu.state.point_size_set(10.0)
                point_batch.draw(point_shader)
    finally:
        gpu.state.point_size_set(1.0)
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set(previous_depth)
        gpu.state.blend_set(previous_blend)


def register_trajectory_tangent_draw_handler() -> None:
    global _TANGENT_DRAW_HANDLE
    if _TANGENT_DRAW_HANDLE is not None:
        return
    _TANGENT_DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        _draw_trajectory_tangent_handles,
        (),
        "WINDOW",
        "POST_VIEW",
    )


def unregister_trajectory_tangent_draw_handler() -> None:
    global _TANGENT_DRAW_HANDLE, _TANGENT_LINE_SHADER, _TANGENT_POINT_SHADER
    if _TANGENT_DRAW_HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_TANGENT_DRAW_HANDLE, "WINDOW")
        _TANGENT_DRAW_HANDLE = None
    _TANGENT_LINE_SHADER = None
    _TANGENT_POINT_SHADER = None


def _trajectory_path_samples_for_snapshots(
    context,
    snapshots: tuple[TrajectoryPositionSnapshot, ...],
) -> dict[str, tuple[Vector, ...]]:
    controls = {
        resolved.control.control_id: resolved
        for resolved in trackbar_controls_for_context(context)
    }
    samples: dict[str, tuple[Vector, ...]] = {}
    for control_id in {snapshot.control_id for snapshot in snapshots}:
        resolved = controls.get(control_id)
        if resolved is None:
            continue
        path = getattr(resolved.target, "motion_path", None)
        if path is None or len(path.points) < 2:
            continue
        samples[control_id] = tuple(Vector(point.co) for point in path.points)
    return samples


def _nearest_point_on_polyline(points: tuple[Vector, ...], target: Vector) -> Vector | None:
    if len(points) < 2:
        return None
    best_point: Vector | None = None
    best_distance_squared: float | None = None
    for start, end in pairwise(points):
        segment = end - start
        length_squared = float(segment.length_squared)
        if length_squared <= 1e-12:
            nearest = Vector(start)
        else:
            factor = max(
                0.0,
                min(1.0, float((target - start).dot(segment)) / length_squared),
            )
            nearest = start + segment * factor
        distance_squared = float((target - nearest).length_squared)
        if best_distance_squared is None or distance_squared < best_distance_squared:
            best_distance_squared = distance_squared
            best_point = Vector(nearest)
    return best_point


def _slide_trajectory_snapshots_world_delta(
    context,
    snapshots: tuple[TrajectoryPositionSnapshot, ...],
    path_samples: dict[str, tuple[Vector, ...]],
    world_delta,
    *,
    refresh: bool = True,
) -> bool:
    delta_world = Vector(world_delta)
    targets: dict[tuple[str, int], Vector] = {}
    for snapshot in snapshots:
        points = path_samples.get(snapshot.control_id)
        if not points:
            continue
        desired = Vector(snapshot.world_position) + delta_world
        nearest = _nearest_point_on_polyline(points, desired)
        if nearest is None:
            continue
        targets[(snapshot.control_id, round(float(snapshot.frame)))] = nearest
    return _apply_trajectory_world_targets(
        context,
        snapshots,
        targets,
        refresh=refresh,
    )


def _set_snapshot_values(
    context,
    snapshot: TrajectoryPositionSnapshot,
    local_delta: Vector,
) -> bool:
    resolved = snapshot.resolved
    obj = resolved.owner_object
    fcurves = _bound_location_fcurves(resolved, snapshot.binding_token)
    changed = False
    for axis_snapshot in snapshot.axes:
        fcurve = fcurves.get(axis_snapshot.axis)
        if fcurve is None:
            continue
        key = _key_at_frame(fcurve, snapshot.frame)
        if key is None:
            continue
        delta = float(local_delta[axis_snapshot.axis])
        key.co.y = axis_snapshot.value + delta
        key.handle_left.y = axis_snapshot.handle_left_y + delta
        key.handle_right.y = axis_snapshot.handle_right_y + delta
        fcurve.update()
        changed = True
    if changed:
        obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})
    return changed


def _apply_trajectory_world_delta(
    context,
    snapshots: tuple[TrajectoryPositionSnapshot, ...],
    world_delta,
    *,
    refresh: bool = True,
) -> bool:
    delta_world = Vector(world_delta)
    changed = False
    for snapshot in snapshots:
        world_to_local = Matrix(snapshot.world_to_local)
        local_delta = world_to_local @ delta_world
        changed = _set_snapshot_values(context, snapshot, local_delta) or changed
    if changed:
        context.scene.frame_set(context.scene.frame_current)
        if refresh:
            refresh_active_trajectory_live(context)
        if context.area is not None:
            context.area.tag_redraw()
    return changed


def _apply_trajectory_world_targets(
    context,
    snapshots: tuple[TrajectoryPositionSnapshot, ...],
    targets: dict[tuple[str, int], Vector],
    *,
    refresh: bool = True,
) -> bool:
    changed = False
    for snapshot in snapshots:
        key = (snapshot.control_id, round(float(snapshot.frame)))
        target = targets.get(key)
        if target is None:
            continue
        delta_world = Vector(target) - Vector(snapshot.world_position)
        world_to_local = Matrix(snapshot.world_to_local)
        local_delta = world_to_local @ delta_world
        changed = _set_snapshot_values(context, snapshot, local_delta) or changed
    if changed:
        context.scene.frame_set(context.scene.frame_current)
        if refresh:
            refresh_active_trajectory_live(context)
        if context.area is not None:
            context.area.tag_redraw()
    return changed


def _rotate_trajectory_snapshots(
    context,
    snapshots: tuple[TrajectoryPositionSnapshot, ...],
    pivot: Vector,
    axis: Vector,
    angle: float,
    *,
    refresh: bool = True,
) -> bool:
    rotation = Matrix.Rotation(float(angle), 3, Vector(axis))
    targets = {}
    for snapshot in snapshots:
        relative = Vector(snapshot.world_position) - pivot
        targets[(snapshot.control_id, round(float(snapshot.frame)))] = (
            pivot + rotation @ relative
        )
    return _apply_trajectory_world_targets(
        context,
        snapshots,
        targets,
        refresh=refresh,
    )


def _scale_trajectory_snapshots_axis(
    context,
    snapshots: tuple[TrajectoryPositionSnapshot, ...],
    pivot: Vector,
    axis: Vector,
    factor: float,
    *,
    refresh: bool = True,
) -> bool:
    direction = Vector(axis).normalized()
    targets = {}
    for snapshot in snapshots:
        relative = Vector(snapshot.world_position) - pivot
        parallel = direction * relative.dot(direction)
        perpendicular = relative - parallel
        targets[(snapshot.control_id, round(float(snapshot.frame)))] = (
            pivot + perpendicular + parallel * float(factor)
        )
    return _apply_trajectory_world_targets(
        context,
        snapshots,
        targets,
        refresh=refresh,
    )


def _scale_trajectory_snapshots_uniform(
    context,
    snapshots: tuple[TrajectoryPositionSnapshot, ...],
    pivot: Vector,
    factor: float,
    *,
    refresh: bool = True,
) -> bool:
    targets = {
        (snapshot.control_id, round(float(snapshot.frame))): (
            pivot
            + (Vector(snapshot.world_position) - pivot) * float(factor)
        )
        for snapshot in snapshots
    }
    return _apply_trajectory_world_targets(
        context,
        snapshots,
        targets,
        refresh=refresh,
    )


def _restore_trajectory_position_snapshots(
    context,
    snapshots: tuple[TrajectoryPositionSnapshot, ...],
    *,
    refresh: bool = True,
) -> bool:
    changed = False
    zero = Vector((0.0, 0.0, 0.0))
    for snapshot in snapshots:
        changed = _set_snapshot_values(context, snapshot, zero) or changed
    if changed:
        context.scene.frame_set(context.scene.frame_current)
        if refresh:
            refresh_active_trajectory_live(context, force=True)
        sync_active_trajectory_key_selection(context)
        if context.area is not None:
            context.area.tag_redraw()
    return changed


def _trajectory_selection_signature(context) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(_selected_position_identities(context)))


def _trajectory_selection_pivot(context) -> Vector | None:
    identities = _trajectory_selection_signature(context)
    if not identities:
        return None
    controls = {
        resolved.control.control_id: resolved
        for resolved in trackbar_controls_for_context(context)
    }
    positions: list[Vector] = []
    for control_id, frame in identities:
        resolved = controls.get(control_id)
        if resolved is None:
            continue
        path = getattr(resolved.target, "motion_path", None)
        if path is None:
            continue
        index = int(frame) - int(path.frame_start)
        if 0 <= index < len(path.points):
            positions.append(Vector(path.points[index].co))
    if not positions:
        return None
    pivot = Vector((0.0, 0.0, 0.0))
    for position in positions:
        pivot += position
    return pivot / len(positions)


def _trajectory_single_selection_tangent(context) -> Vector | None:
    identities = _trajectory_selection_signature(context)
    if len(identities) != 1:
        return None
    control_id, frame = identities[0]
    resolved = _control_for_id(context, control_id)
    if resolved is None:
        return None
    path = getattr(resolved.target, "motion_path", None)
    if path is None or len(path.points) < 2:
        return None
    index = int(frame) - int(path.frame_start)
    if not (0 <= index < len(path.points)):
        return None
    before_index = max(0, index - 1)
    after_index = min(len(path.points) - 1, index + 1)
    if before_index == after_index:
        return None
    tangent = Vector(path.points[after_index].co) - Vector(path.points[before_index].co)
    if tangent.length <= 1e-9:
        return None
    tangent.normalize()
    return tangent


def _trajectory_slide_screen_direction(context, pivot: Vector) -> Vector:
    direction = Vector((1.0, 0.0))
    if len(_trajectory_selection_signature(context)) != 1:
        return direction
    tangent = _trajectory_single_selection_tangent(context)
    if tangent is None or context.region is None or context.region_data is None:
        return direction
    pivot_screen = location_3d_to_region_2d(context.region, context.region_data, pivot)
    tangent_screen = location_3d_to_region_2d(
        context.region,
        context.region_data,
        Vector(pivot) + tangent,
    )
    if pivot_screen is None or tangent_screen is None:
        return direction
    projected = Vector(
        (
            float(tangent_screen.x - pivot_screen.x),
            float(tangent_screen.y - pivot_screen.y),
        )
    )
    if projected.length <= 1e-6:
        return direction
    projected.normalize()
    return projected


def _trajectory_slide_gizmo_matrix(context) -> Matrix | None:
    live_pivot = _trajectory_selection_pivot(context)
    pivot = (
        Vector(_SLIDE_GIZMO_MODAL_PIVOT)
        if _SLIDE_GIZMO_MODAL_PIVOT is not None
        else live_pivot
    )
    if pivot is None or context.region is None or context.region_data is None:
        return None
    pivot_screen = location_3d_to_region_2d(context.region, context.region_data, pivot)
    if pivot_screen is None:
        return None
    screen_direction = _trajectory_slide_screen_direction(context, Vector(pivot))
    screen_point = Vector(
        (
            float(pivot_screen.x) + float(screen_direction.x) * 32.0,
            float(pivot_screen.y) + float(screen_direction.y) * 32.0,
        )
    )
    world_point = region_2d_to_location_3d(
        context.region,
        context.region_data,
        screen_point,
        pivot,
    )
    direction = Vector(world_point) - Vector(pivot)
    if direction.length <= 1e-9:
        return None
    direction.normalize()
    rotation = Vector((1.0, 0.0, 0.0)).rotation_difference(direction)
    matrix = rotation.to_matrix().to_4x4()
    matrix.translation = pivot
    return matrix


def _trajectory_axis_vector(axis: str) -> Vector:
    return {
        "X": Vector((1.0, 0.0, 0.0)),
        "Y": Vector((0.0, 1.0, 0.0)),
        "Z": Vector((0.0, 0.0, 1.0)),
    }[axis]


def _trajectory_plane_axes(plane: str) -> tuple[Vector, Vector, Vector]:
    return {
        "XY": (
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
        ),
        "YZ": (
            Vector((0.0, 1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
            Vector((1.0, 0.0, 0.0)),
        ),
        "ZX": (
            Vector((0.0, 0.0, 1.0)),
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 1.0, 0.0)),
        ),
    }[plane]


def _complete_location_key_frames(fcurves: dict[int, object]) -> set[int]:
    if set(fcurves) != {0, 1, 2}:
        return set()
    frame_sets = [
        {round(float(key.co.x)) for key in fcurves[axis].keyframe_points}
        for axis in range(3)
    ]
    return set.intersection(*frame_sets) if frame_sets else set()


def _trajectory_slide_transactions_for_context(
    context,
) -> tuple[tuple[TrajectorySlideOwnerTransaction, ...], int, int] | None:
    identities = sorted(_selected_position_identities(context))
    if not identities or not _trajectory_edit_mode_supported(context):
        return None

    frames_by_control: dict[str, list[int]] = {}
    for control_id, frame in identities:
        frames_by_control.setdefault(control_id, []).append(int(frame))
    controls = {
        resolved.control.control_id: resolved
        for resolved in trackbar_controls_for_context(context)
    }

    scene = context.scene
    owners: list[TrajectorySlideOwnerTransaction] = []
    global_min_delta = -10**9
    global_max_delta = 10**9
    for control_id, raw_frames in frames_by_control.items():
        resolved = controls.get(control_id)
        if resolved is None:
            continue
        obj = resolved.owner_object
        if _has_active_constraints(resolved.target):
            continue
        fcurves = _location_fcurves_for_control(resolved)
        if set(fcurves) != {0, 1, 2}:
            continue
        source_frames = tuple(sorted({int(frame) for frame in raw_frames}))
        transaction = snapshot_multi_key_preview_transaction(
            tuple(fcurves[axis] for axis in range(3)),
            source_frames,
            mode="MOVE",
        )
        if transaction is None:
            continue

        all_frames = _complete_location_key_frames(fcurves)
        unselected = sorted(all_frames.difference(source_frames))
        min_delta = int(scene.frame_start) - min(source_frames)
        max_delta = int(scene.frame_end) - max(source_frames)
        for source_frame in source_frames:
            previous = [frame for frame in unselected if frame < source_frame]
            following = [frame for frame in unselected if frame > source_frame]
            if previous:
                min_delta = max(min_delta, max(previous) + 1 - source_frame)
            if following:
                max_delta = min(max_delta, min(following) - 1 - source_frame)
        min_delta = min(0, min_delta)
        max_delta = max(0, max_delta)

        sample_start = min(source_frames) + min_delta
        sample_end = max(source_frames) + max_delta
        sampled_values = {
            frame: tuple(float(fcurves[axis].evaluate(frame)) for axis in range(3))
            for frame in range(sample_start, sample_end + 1)
        }
        owners.append(
            TrajectorySlideOwnerTransaction(
                control_id=control_id,
                owner_name=obj.name,
                resolved=resolved,
                binding_token=channel_binding_token(obj),
                source_frames=source_frames,
                min_delta=min_delta,
                max_delta=max_delta,
                sampled_values=sampled_values,
                transaction=transaction,
            )
        )
        global_min_delta = max(global_min_delta, min_delta)
        global_max_delta = min(global_max_delta, max_delta)

    if not owners:
        return None
    if global_min_delta > global_max_delta:
        global_min_delta = 0
        global_max_delta = 0
    return tuple(owners), int(global_min_delta), int(global_max_delta)


def _apply_trajectory_slide_delta(
    context,
    owners: tuple[TrajectorySlideOwnerTransaction, ...],
    delta_frames: int,
    *,
    min_delta: int,
    max_delta: int,
) -> int:
    delta = max(int(min_delta), min(int(max_delta), int(delta_frames)))
    destinations: list[tuple[str, tuple[int, ...]]] = []
    for owner in owners:
        resolved = owner.resolved
        obj = resolved.owner_object
        fcurves = _bound_location_fcurves(resolved, owner.binding_token)
        if set(fcurves) != {0, 1, 2}:
            continue
        apply_multi_key_preview_transaction(
            tuple(fcurves[axis] for axis in range(3)),
            owner.transaction,
            delta,
        )
        destination_frames = tuple(frame + delta for frame in owner.source_frames)
        destinations.append((owner.control_id, destination_frames))
        for source_frame, destination_frame in zip(
            owner.source_frames,
            destination_frames,
            strict=True,
        ):
            sampled = owner.sampled_values.get(destination_frame)
            if sampled is None:
                continue
            for axis in range(3):
                fcurve = fcurves[axis]
                key = _key_at_frame(fcurve, destination_frame)
                if key is None:
                    continue
                value_delta = float(sampled[axis]) - float(key.co.y)
                key.co.y = float(sampled[axis])
                key.handle_left.y = float(key.handle_left.y) + value_delta
                key.handle_right.y = float(key.handle_right.y) + value_delta
                fcurve.update()
        obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})

    clear_key_selection_for_context(context)
    for control_id, frames in destinations:
        for frame in frames:
            _set_control_position_key_selection(context, control_id, frame, mode="ADD")
    context.scene.frame_set(context.scene.frame_current)
    refresh_active_trajectory_live(context, force=True, exact_range=True)
    sync_active_trajectory_key_selection(context)
    if context.area is not None:
        context.area.tag_redraw()
    return delta


def _restore_trajectory_slide_transactions(
    context,
    owners: tuple[TrajectorySlideOwnerTransaction, ...],
) -> None:
    for owner in owners:
        resolved = owner.resolved
        obj = resolved.owner_object
        fcurves = _bound_location_fcurves(resolved, owner.binding_token)
        if set(fcurves) != {0, 1, 2}:
            continue
        restore_multi_key_preview_transaction(
            tuple(fcurves[axis] for axis in range(3)),
            owner.transaction,
        )
        obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})
    clear_key_selection_for_context(context)
    for owner in owners:
        for frame in owner.source_frames:
            _set_control_position_key_selection(context, owner.control_id, frame, mode="ADD")
    context.scene.frame_set(context.scene.frame_current)
    refresh_active_trajectory_live(context, force=True, exact_range=True)
    sync_active_trajectory_key_selection(context)
    if context.area is not None:
        context.area.tag_redraw()


class BAW_OT_move_trajectory_axis(bpy.types.Operator):
    bl_idname = "baw.move_trajectory_axis"
    bl_label = "Slide Trajectory Keys"
    bl_description = "Slide selected Position keys forward or backward along the existing path"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}

    axis: EnumProperty(
        items=(("X", "X", "Horizontal trajectory time slide"),),
        default="X",
    )

    _owners: tuple[TrajectorySlideOwnerTransaction, ...] = ()
    _min_delta: int = 0
    _max_delta: int = 0
    _start_mouse = Vector((0.0, 0.0))
    _drag_screen_direction = Vector((1.0, 0.0))
    _applied_delta: int = 0
    _pixels_per_frame: float = 12.0

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and _trajectory_edit_mode_supported(context)
            and bool(_selected_position_identities(context))
        )

    def invoke(self, context, event):
        global _SLIDE_GIZMO_MODAL_PIVOT

        slide = _trajectory_slide_transactions_for_context(context)
        if slide is None:
            self.report(
                {"WARNING"},
                "Trajectory Slide requires unconstrained XYZ Position keys",
            )
            return {"CANCELLED"}
        owners, min_delta, max_delta = slide
        self._owners = owners
        self._min_delta = int(min_delta)
        self._max_delta = int(max_delta)
        self._start_mouse = Vector(
            (float(event.mouse_region_x), float(event.mouse_region_y))
        )
        pivot = _trajectory_selection_pivot(context)
        if pivot is None:
            return {"CANCELLED"}
        self._drag_screen_direction = _trajectory_slide_screen_direction(
            context,
            Vector(pivot),
        )
        self._applied_delta = 0
        if len(_selected_position_identities(context)) > 1:
            _SLIDE_GIZMO_MODAL_PIVOT = Vector(pivot)
        else:
            _SLIDE_GIZMO_MODAL_PIVOT = None
        if context.window is not None:
            context.window.cursor_modal_set("SCROLL_X")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
            projected_pixels = float((mouse - self._start_mouse).dot(self._drag_screen_direction))
            raw_delta = round(projected_pixels / self._pixels_per_frame)
            self._applied_delta = _apply_trajectory_slide_delta(
                context,
                self._owners,
                raw_delta,
                min_delta=self._min_delta,
                max_delta=self._max_delta,
            )
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            global _SLIDE_GIZMO_MODAL_PIVOT

            _SLIDE_GIZMO_MODAL_PIVOT = None
            if context.window is not None:
                context.window.cursor_modal_restore()
            refresh_active_trajectory_live(context, force=True, exact_range=True)
            sync_active_trajectory_key_selection(context)
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            _restore_trajectory_slide_transactions(context, self._owners)
            _SLIDE_GIZMO_MODAL_PIVOT = None
            if context.window is not None:
                context.window.cursor_modal_restore()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_OT_move_trajectory_plane(bpy.types.Operator):
    bl_idname = "baw.move_trajectory_plane"
    bl_label = "Slide Trajectory Key Plane"
    bl_description = "Slide selected trajectory Position keys along their existing path"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}

    plane: EnumProperty(
        items=(
            ("XY", "XY", "Move on the world XY plane"),
            ("YZ", "YZ", "Move on the world YZ plane"),
            ("ZX", "ZX", "Move on the world ZX plane"),
        ),
        default="XY",
    )

    _snapshots: tuple[TrajectoryPositionSnapshot, ...] = ()
    _path_samples: ClassVar[dict[str, tuple[Vector, ...]]] = {}
    _pivot = Vector((0.0, 0.0, 0.0))
    _plane_normal = Vector((0.0, 0.0, 1.0))
    _start_point = Vector((0.0, 0.0, 0.0))

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and context.mode == "OBJECT"
            and bool(_selected_position_identities(context))
        )

    def _plane_point(self, context, mouse_region_x: float, mouse_region_y: float):
        if context.region is None or context.region_data is None:
            return None
        mouse = Vector((float(mouse_region_x), float(mouse_region_y)))
        ray_origin = region_2d_to_origin_3d(context.region, context.region_data, mouse)
        ray_direction = region_2d_to_vector_3d(context.region, context.region_data, mouse)
        if ray_direction.length <= 1e-9:
            return None
        ray_direction.normalize()
        ray_span = 100000.0
        return intersect_line_plane(
            ray_origin,
            ray_origin + ray_direction * ray_span,
            self._pivot,
            self._plane_normal,
            False,
        )

    def invoke(self, context, event):
        snapshots = _trajectory_position_snapshots_for_context(context)
        pivot = _trajectory_selection_pivot(context)
        if not snapshots or pivot is None:
            self.report(
                {"WARNING"},
                "Trajectory Move requires unconstrained XYZ location keys",
            )
            return {"CANCELLED"}

        self._snapshots = snapshots
        self._path_samples = _trajectory_path_samples_for_snapshots(context, snapshots)
        self._pivot = Vector(pivot)
        _axis_a, _axis_b, normal = _trajectory_plane_axes(self.plane)
        self._plane_normal = normal
        start = self._plane_point(
            context,
            event.mouse_region_x,
            event.mouse_region_y,
        )
        if start is None:
            self.report({"WARNING"}, f"Trajectory {self.plane} plane is edge-on to this view")
            return {"CANCELLED"}
        self._start_point = Vector(start)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            current = self._plane_point(
                context,
                event.mouse_region_x,
                event.mouse_region_y,
            )
            if current is None:
                return {"RUNNING_MODAL"}
            delta = Vector(current) - self._start_point
            _slide_trajectory_snapshots_world_delta(
                context,
                self._snapshots,
                self._path_samples,
                delta,
                refresh=True,
            )
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            refresh_active_trajectory_live(context, force=True)
            sync_active_trajectory_key_selection(context)
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            _restore_trajectory_position_snapshots(context, self._snapshots)
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_OT_rotate_trajectory_axis(bpy.types.Operator):
    bl_idname = "baw.rotate_trajectory_axis"
    bl_label = "Rotate Trajectory Keys"
    bl_description = "Rotate selected Position keys around their selection center"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}

    axis: EnumProperty(
        items=(
            ("X", "X", "Rotate around world X"),
            ("Y", "Y", "Rotate around world Y"),
            ("Z", "Z", "Rotate around world Z"),
        ),
        default="Z",
    )

    _snapshots: tuple[TrajectoryPositionSnapshot, ...] = ()
    _pivot = Vector((0.0, 0.0, 0.0))
    _axis = Vector((0.0, 0.0, 1.0))
    _start_vector = Vector((1.0, 0.0, 0.0))

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and context.mode == "OBJECT"
            and len(_selected_position_identities(context)) >= 2
        )

    def _plane_vector(self, context, mouse_region_x: float, mouse_region_y: float):
        if context.region is None or context.region_data is None:
            return None
        mouse = Vector((float(mouse_region_x), float(mouse_region_y)))
        ray_origin = region_2d_to_origin_3d(context.region, context.region_data, mouse)
        ray_direction = region_2d_to_vector_3d(context.region, context.region_data, mouse)
        if ray_direction.length <= 1e-9:
            return None
        ray_direction.normalize()
        point = intersect_line_plane(
            ray_origin,
            ray_origin + ray_direction * 100000.0,
            self._pivot,
            self._axis,
            False,
        )
        if point is None:
            return None
        vector = Vector(point) - self._pivot
        if vector.length <= 1e-7:
            return None
        return vector.normalized()

    def invoke(self, context, event):
        snapshots = _trajectory_position_snapshots_for_context(context)
        pivot = _trajectory_selection_pivot(context)
        if len(snapshots) < 2 or pivot is None:
            self.report({"WARNING"}, "Trajectory Rotate requires at least two Position keys")
            return {"CANCELLED"}
        self._snapshots = snapshots
        self._pivot = Vector(pivot)
        self._axis = _trajectory_axis_vector(self.axis)
        start = self._plane_vector(context, event.mouse_region_x, event.mouse_region_y)
        if start is None:
            self.report({"WARNING"}, f"Trajectory {self.axis} rotate ring is edge-on to this view")
            return {"CANCELLED"}
        self._start_vector = Vector(start)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            current = self._plane_vector(context, event.mouse_region_x, event.mouse_region_y)
            if current is None:
                return {"RUNNING_MODAL"}
            sine = float(self._axis.dot(self._start_vector.cross(current)))
            cosine = float(self._start_vector.dot(current))
            angle = atan2(sine, cosine)
            _rotate_trajectory_snapshots(
                context,
                self._snapshots,
                self._pivot,
                self._axis,
                angle,
                refresh=True,
            )
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            refresh_active_trajectory_live(context, force=True)
            sync_active_trajectory_key_selection(context)
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            _restore_trajectory_position_snapshots(context, self._snapshots)
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


def _trajectory_retime_transactions_for_context(
    context,
    *,
    side: str,
) -> tuple[tuple[TrajectoryRetimeOwnerTransaction, ...], int, int] | None:
    identities = sorted(_selected_position_identities(context))
    source_frames = sorted({frame for _control_id, frame in identities})
    if not _trajectory_edit_mode_supported(context) or len(source_frames) < 2:
        return None
    side_name = str(side).upper()
    if side_name == "RIGHT":
        pivot_frame = int(source_frames[0])
        source_handle_frame = int(source_frames[-1])
    elif side_name == "LEFT":
        pivot_frame = int(source_frames[-1])
        source_handle_frame = int(source_frames[0])
    else:
        raise ValueError(f"Unsupported trajectory Retime side: {side!r}")

    frames_by_control: dict[str, list[int]] = {}
    for control_id, frame in identities:
        frames_by_control.setdefault(control_id, []).append(int(frame))
    controls = {
        resolved.control.control_id: resolved
        for resolved in trackbar_controls_for_context(context)
    }
    owners: list[TrajectoryRetimeOwnerTransaction] = []
    for control_id, frames in frames_by_control.items():
        resolved = controls.get(control_id)
        if resolved is None:
            continue
        obj = resolved.owner_object
        fcurves = _location_fcurves_for_control(resolved)
        if set(fcurves) != {0, 1, 2}:
            continue
        transaction = snapshot_selection_range_scale_transaction(
            tuple(fcurves[axis] for axis in range(3)),
            frames,
            pivot_frame=pivot_frame,
            source_handle_frame=source_handle_frame,
        )
        if transaction is None:
            continue
        owners.append(
            TrajectoryRetimeOwnerTransaction(
                control_id=control_id,
                owner_name=obj.name,
                resolved=resolved,
                binding_token=channel_binding_token(obj),
                source_frames=tuple(sorted(set(frames))),
                source_handle_frame=source_handle_frame,
                transaction=transaction,
            )
        )
    if not owners:
        return None
    return tuple(owners), pivot_frame, source_handle_frame


def _apply_trajectory_retime_factor(
    context,
    owners: tuple[TrajectoryRetimeOwnerTransaction, ...],
    pivot_frame: int,
    source_handle_frame: int,
    factor: float,
) -> bool:
    target_handle_frame = round(
        float(pivot_frame)
        + (float(source_handle_frame) - float(pivot_frame)) * float(factor)
    )
    if target_handle_frame == pivot_frame:
        target_handle_frame += 1 if source_handle_frame > pivot_frame else -1

    changed = False
    destinations: list[tuple[str, tuple[int, ...]]] = []
    for owner in owners:
        resolved = owner.resolved
        obj = resolved.owner_object
        fcurves = _bound_location_fcurves(resolved, owner.binding_token)
        if set(fcurves) != {0, 1, 2}:
            continue
        owner_changed = apply_selection_range_scale_transaction(
            tuple(fcurves[axis] for axis in range(3)),
            owner.transaction,
            target_handle_frame,
        )
        changed = changed or owner_changed
        destinations.append(
            (
                owner.control_id,
                tuple(int(frame) for frame in owner.transaction.selected_destination_frames),
            )
        )
        if owner_changed:
            obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})

    if not destinations:
        return False
    clear_key_selection_for_context(context)
    for control_id, frames in destinations:
        for frame in frames:
            _set_control_position_key_selection(context, control_id, frame, mode="ADD")
    context.scene.frame_set(context.scene.frame_current)
    refresh_active_trajectory_live(context, force=True, exact_range=True)
    sync_active_trajectory_key_selection(context)
    if context.area is not None:
        context.area.tag_redraw()
    return changed


def _restore_trajectory_retime_transactions(
    context,
    owners: tuple[TrajectoryRetimeOwnerTransaction, ...],
) -> bool:
    restored = False
    for owner in owners:
        resolved = owner.resolved
        obj = resolved.owner_object
        fcurves = _bound_location_fcurves(resolved, owner.binding_token)
        if set(fcurves) != {0, 1, 2}:
            continue
        owner_restored = restore_selection_range_scale_transaction(
            tuple(fcurves[axis] for axis in range(3)),
            owner.transaction,
        )
        restored = restored or owner_restored
        if owner_restored:
            obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})
    clear_key_selection_for_context(context)
    for owner in owners:
        for frame in owner.source_frames:
            _set_control_position_key_selection(context, owner.control_id, frame, mode="ADD")
    context.scene.frame_set(context.scene.frame_current)
    refresh_active_trajectory_live(context, force=True, exact_range=True)
    sync_active_trajectory_key_selection(context)
    if context.area is not None:
        context.area.tag_redraw()
    return restored


class BAW_OT_scale_trajectory_axis(bpy.types.Operator):
    bl_idname = "baw.scale_trajectory_axis"
    bl_label = "Retime Trajectory Keys"
    bl_description = "Retime selected Position keys from one endpoint while the opposite endpoint stays fixed"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}

    side: EnumProperty(
        items=(
            ("LEFT", "Left", "Drag the left endpoint while the right endpoint stays fixed"),
            ("RIGHT", "Right", "Drag the right endpoint while the left endpoint stays fixed"),
        ),
        default="RIGHT",
    )

    _retime_owners: tuple[TrajectoryRetimeOwnerTransaction, ...] = ()
    _pivot_frame: int = 0
    _source_handle_frame: int = 0
    _start_mouse = Vector((0.0, 0.0))
    _screen_direction = Vector((1.0, 0.0))
    _pixels_per_frame: float = 12.0

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and _trajectory_edit_mode_supported(context)
            and len(_selected_position_identities(context)) >= 2
        )

    def invoke(self, context, event):
        side_name = str(self.side).upper()
        retime = _trajectory_retime_transactions_for_context(context, side=side_name)
        if retime is None:
            self.report({"WARNING"}, "Trajectory Retime requires at least two Position-key times")
            return {"CANCELLED"}
        owners, pivot_frame, source_handle_frame = retime
        self._retime_owners = owners
        self._pivot_frame = int(pivot_frame)
        self._source_handle_frame = int(source_handle_frame)
        self._start_mouse = Vector(
            (float(event.mouse_region_x), float(event.mouse_region_y))
        )
        self._screen_direction = (
            Vector((-1.0, 0.0)) if side_name == "LEFT" else Vector((1.0, 0.0))
        )
        if context.window is not None:
            context.window.cursor_modal_set("SCROLL_X")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
            projected_pixels = float((mouse - self._start_mouse).dot(self._screen_direction))
            outward_frames = round(projected_pixels / self._pixels_per_frame)
            frame_direction = -1 if str(self.side).upper() == "LEFT" else 1
            target_handle_frame = self._source_handle_frame + frame_direction * outward_frames
            denominator = self._source_handle_frame - self._pivot_frame
            if denominator == 0:
                return {"RUNNING_MODAL"}
            factor = (
                float(target_handle_frame - self._pivot_frame)
                / float(denominator)
            )
            _apply_trajectory_retime_factor(
                context,
                self._retime_owners,
                self._pivot_frame,
                self._source_handle_frame,
                factor,
            )
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if context.window is not None:
                context.window.cursor_modal_restore()
            refresh_active_trajectory_live(context, force=True)
            sync_active_trajectory_key_selection(context)
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            _restore_trajectory_retime_transactions(context, self._retime_owners)
            if context.window is not None:
                context.window.cursor_modal_restore()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_OT_scale_trajectory_uniform(bpy.types.Operator):
    bl_idname = "baw.scale_trajectory_uniform"
    bl_label = "Uniform Scale Trajectory Keys"
    bl_description = "Uniformly scale selected Position keys around their selection center"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}

    _snapshots: tuple[TrajectoryPositionSnapshot, ...] = ()
    _pivot = Vector((0.0, 0.0, 0.0))
    _pivot_screen = Vector((0.0, 0.0))
    _start_radius: float = 1.0

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and context.mode == "OBJECT"
            and len(_selected_position_identities(context)) >= 2
        )

    def invoke(self, context, event):
        snapshots = _trajectory_position_snapshots_for_context(context)
        pivot = _trajectory_selection_pivot(context)
        if len(snapshots) < 2 or pivot is None or context.region_data is None:
            self.report({"WARNING"}, "Trajectory Scale requires at least two Position keys")
            return {"CANCELLED"}
        pivot_screen = location_3d_to_region_2d(context.region, context.region_data, pivot)
        if pivot_screen is None:
            return {"CANCELLED"}
        self._snapshots = snapshots
        self._pivot = Vector(pivot)
        self._pivot_screen = Vector((float(pivot_screen.x), float(pivot_screen.y)))
        mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
        radius = float((mouse - self._pivot_screen).length)
        if radius <= 1e-4:
            return {"CANCELLED"}
        self._start_radius = radius
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
            radius = float((mouse - self._pivot_screen).length)
            factor = max(0.001, radius / self._start_radius)
            _scale_trajectory_snapshots_uniform(
                context,
                self._snapshots,
                self._pivot,
                factor,
                refresh=True,
            )
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            refresh_active_trajectory_live(context, force=True)
            sync_active_trajectory_key_selection(context)
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            _restore_trajectory_position_snapshots(context, self._snapshots)
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_GT_trajectory_slide(bpy.types.Gizmo):
    bl_idname = "BAW_GT_trajectory_slide"

    def setup(self):
        self.use_draw_modal = True
        self.use_draw_hover = False
        self._shape = self.new_custom_shape(
            "LINES",
            (
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (-0.72, 0.18, 0.0),
                (-1.0, 0.0, 0.0),
                (-0.72, -0.18, 0.0),
                (1.0, 0.0, 0.0),
                (0.72, 0.18, 0.0),
                (1.0, 0.0, 0.0),
                (0.72, -0.18, 0.0),
                (-0.08, 0.0, 0.0),
                (0.0, 0.08, 0.0),
                (0.0, 0.08, 0.0),
                (0.08, 0.0, 0.0),
                (0.08, 0.0, 0.0),
                (0.0, -0.08, 0.0),
                (0.0, -0.08, 0.0),
                (-0.08, 0.0, 0.0),
            ),
        )

    def _update_matrix(self, context) -> bool:
        matrix = _trajectory_slide_gizmo_matrix(context)
        if matrix is None:
            return False
        self.matrix_basis = matrix
        return True

    def draw(self, context):
        if self._update_matrix(context):
            self.draw_custom_shape(self._shape)

    def draw_select(self, context, select_id):
        if self._update_matrix(context):
            self.draw_custom_shape(self._shape, select_id=select_id)


class BAW_GGT_trajectory_move(bpy.types.GizmoGroup):
    bl_idname = "BAW_GGT_trajectory_move"
    bl_label = "AWB Trajectory Move Gizmo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options: ClassVar[set[str]] = {"3D", "PERSISTENT", "SHOW_MODAL_ALL"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and not bool(getattr(scene, "baw_trajectory_add_key_mode", False))
            and _trajectory_edit_mode_supported(context)
            and bool(_selected_position_identities(context))
            and _trajectory_selection_pivot(context) is not None
        )

    def setup(self, context):
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        self.slide_gizmo = self.gizmos.new(BAW_GT_trajectory_slide.bl_idname)
        slide_props = self.slide_gizmo.target_set_operator(
            BAW_OT_move_trajectory_axis.bl_idname
        )
        slide_props.axis = "X"
        self.slide_gizmo.color = (0.92, 0.92, 0.92)
        self.slide_gizmo.alpha = 0.9
        self.slide_gizmo.color_highlight = (1.0, 1.0, 1.0)
        self.slide_gizmo.alpha_highlight = 1.0
        self.slide_gizmo.line_width = 3.0
        self.slide_gizmo.scale_basis = 0.9

        self.rotate_gizmos = {}
        for axis_name in ("X", "Y", "Z"):
            gizmo = self.gizmos.new("GIZMO_GT_dial_3d")
            props = gizmo.target_set_operator(BAW_OT_rotate_trajectory_axis.bl_idname)
            props.axis = axis_name
            gizmo.color = colors[axis_name]
            gizmo.alpha = 0.78
            gizmo.color_highlight = (1.0, 1.0, 1.0)
            gizmo.alpha_highlight = 1.0
            gizmo.line_width = 2.0
            gizmo.scale_basis = 0.82
            self.rotate_gizmos[axis_name] = gizmo

        self.retime_gizmos = {}
        for side_name in ("LEFT", "RIGHT"):
            gizmo = self.gizmos.new("GIZMO_GT_arrow_3d")
            props = gizmo.target_set_operator(BAW_OT_scale_trajectory_axis.bl_idname)
            props.side = side_name
            gizmo.draw_style = "BOX"
            gizmo.color = (0.92, 0.92, 0.92)
            gizmo.alpha = 0.88
            gizmo.color_highlight = (1.0, 1.0, 1.0)
            gizmo.alpha_highlight = 1.0
            gizmo.line_width = 2.0
            gizmo.scale_basis = 0.8
            self.retime_gizmos[side_name] = gizmo

    def draw_prepare(self, context):
        selected_identities = _selected_position_identities(context)
        live_pivot = _trajectory_selection_pivot(context)
        pivot = (
            Vector(_SLIDE_GIZMO_MODAL_PIVOT)
            if _SLIDE_GIZMO_MODAL_PIVOT is not None
            else live_pivot
        )
        mode = str(getattr(context.scene, "baw_trajectory_transform_mode", "MOVE"))
        can_rotate = len(selected_identities) >= 2
        can_retime = len({frame for _control_id, frame in selected_identities}) >= 2

        slide_direction: Vector | None = None
        if pivot is not None and context.region is not None and context.region_data is not None:
            pivot_screen = location_3d_to_region_2d(
                context.region,
                context.region_data,
                pivot,
            )
            screen_direction = _trajectory_slide_screen_direction(context, Vector(pivot))
            if pivot_screen is not None:
                screen_point = Vector(
                    (
                        float(pivot_screen.x) + float(screen_direction.x) * 32.0,
                        float(pivot_screen.y) + float(screen_direction.y) * 32.0,
                    )
                )
                world_point = region_2d_to_location_3d(
                    context.region,
                    context.region_data,
                    screen_point,
                    pivot,
                )
                direction = Vector(world_point) - Vector(pivot)
                if direction.length > 1e-9:
                    slide_direction = direction.normalized()

        self.slide_gizmo.hide = pivot is None or mode != "MOVE" or slide_direction is None
        if pivot is not None and slide_direction is not None:
            rotation = Vector((1.0, 0.0, 0.0)).rotation_difference(slide_direction)
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            self.slide_gizmo.matrix_basis = matrix

        for axis_name, gizmo in self.rotate_gizmos.items():
            gizmo.hide = pivot is None or mode != "ROTATE" or not can_rotate
            if pivot is None:
                continue
            axis = _trajectory_axis_vector(axis_name)
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(axis)
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            gizmo.matrix_basis = matrix

        for side_name, gizmo in self.retime_gizmos.items():
            gizmo.hide = (
                pivot is None
                or mode != "SCALE"
                or not can_retime
                or slide_direction is None
            )
            if pivot is None or slide_direction is None:
                continue
            direction = (
                Vector(slide_direction)
                if side_name == "RIGHT"
                else -Vector(slide_direction)
            )
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(direction)
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            gizmo.matrix_basis = matrix


def _set_path_edit_flag(context, enabled: bool) -> None:
    for resolved in trackbar_controls_for_context(context):
        path = getattr(resolved.target, "motion_path", None)
        if path is not None and hasattr(path, "is_modified"):
            path.is_modified = bool(enabled)


def _set_native_tool_gizmo_hidden(context, hidden: bool) -> None:
    spaces = []
    direct_space = getattr(context, "space_data", None)
    if direct_space is not None and getattr(direct_space, "type", None) == "VIEW_3D":
        spaces.append(direct_space)

    window = getattr(context, "window", None)
    screen = getattr(window, "screen", None) if window is not None else None
    if screen is not None:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                if space is not None and space not in spaces:
                    spaces.append(space)

    for space in spaces:
        if not hasattr(space, "show_gizmo_tool"):
            continue
        pointer = int(space.as_pointer())
        if hidden:
            _PREVIOUS_SHOW_GIZMO_TOOL.setdefault(pointer, bool(space.show_gizmo_tool))
            space.show_gizmo_tool = False
        else:
            previous = _PREVIOUS_SHOW_GIZMO_TOOL.pop(pointer, None)
            if previous is not None:
                space.show_gizmo_tool = previous


def restore_trajectory_edit_view_state() -> None:
    if not _PREVIOUS_SHOW_GIZMO_TOOL:
        return
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        _PREVIOUS_SHOW_GIZMO_TOOL.clear()
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            pointer = int(space.as_pointer())
            previous = _PREVIOUS_SHOW_GIZMO_TOOL.pop(pointer, None)
            if previous is not None and hasattr(space, "show_gizmo_tool"):
                space.show_gizmo_tool = previous
    _PREVIOUS_SHOW_GIZMO_TOOL.clear()


class BAW_OT_delete_trajectory_position_keys(bpy.types.Operator):
    bl_idname = "baw.delete_trajectory_position_keys"
    bl_label = "Delete Trajectory Position Keys"
    bl_description = "Delete selected trajectory Position keys without removing other keyed channels"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and _trajectory_edit_mode_supported(context)
        )

    def invoke(self, context, event):
        if event.mouse_region_y <= TRACKBAR_INTERACTION_HEIGHT:
            return {"PASS_THROUGH"}
        if bool(getattr(context.scene, "baw_trajectory_add_key_mode", False)):
            # Add Key owns the trajectory interaction; consume Delete/X so no
            # viewport Object Delete menu/action can leak through.
            return {"FINISHED"}
        if not _selected_position_identities(context):
            # Edit Path Keys is an editing context. Delete/X with no selected
            # path key is a no-op here, never an instruction to delete the rig/object.
            return {"FINISHED"}
        return self.execute(context)

    def execute(self, context):
        return (
            {"FINISHED"}
            if _delete_selected_trajectory_position_keys(context)
            else {"CANCELLED"}
        )


class BAW_OT_toggle_trajectory_tangent(bpy.types.Operator):
    bl_idname = "baw.toggle_trajectory_tangent"
    bl_label = "Toggle Trajectory Tangents"
    bl_description = "Show and edit Bezier tangent handles for selected trajectory Position keys"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and _trajectory_edit_mode_supported(context)
        )

    def execute(self, context):
        scene = context.scene
        scene.baw_trajectory_transform_mode = (
            "MOVE"
            if str(scene.baw_trajectory_transform_mode) == "TANGENT"
            else "TANGENT"
        )
        # Edit Path owns the viewport transform affordance. Reassert the hidden
        # native tool gizmo when Tangent mode toggles so the original object/bone
        # gizmo never competes with trajectory/tangent handles.
        _set_native_tool_gizmo_hidden(context, hidden=True)
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_toggle_trajectory_add_key(bpy.types.Operator):
    bl_idname = "baw.toggle_trajectory_add_key"
    bl_label = "Toggle Trajectory Add Key"
    bl_description = "Keep Add Key active so repeated clicks on a Motion Path insert Position keys"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and _trajectory_edit_mode_supported(context)
            and bool(trackbar_controls_for_context(context))
        )

    def execute(self, context):
        scene = context.scene
        scene.baw_trajectory_add_key_mode = not bool(scene.baw_trajectory_add_key_mode)
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_add_trajectory_key_on_path(bpy.types.Operator):
    bl_idname = "baw.add_trajectory_key_on_path"
    bl_label = "Add Trajectory Position Key"
    bl_description = "Add a Position key to the Motion Path at the clicked frame"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and bool(getattr(scene, "baw_trajectory_add_key_mode", False))
            and _trajectory_edit_mode_supported(context)
        )

    def invoke(self, context, event):
        if event.mouse_region_y <= TRACKBAR_INTERACTION_HEIGHT:
            return {"PASS_THROUGH"}
        hit = _trajectory_path_hit(context, event.mouse_region_x, event.mouse_region_y)
        if hit is None:
            return {"PASS_THROUGH"}

        _distance, _active_bias, control_id, frame, _world = hit
        _changed, selected = _add_trajectory_position_keys(
            context,
            ((control_id, int(frame)),),
        )
        if not selected:
            self.report(
                {"WARNING"},
                "Trajectory Add Key currently requires an unconstrained Object position track",
            )
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_add_trajectory_key_current_frame(bpy.types.Operator):
    bl_idname = "baw.add_trajectory_key_current_frame"
    bl_label = "Add Trajectory Position Key at Current Frame"
    bl_description = "Add Position keys at the current frame for selected trajectory controls"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and _trajectory_edit_mode_supported(context)
            and bool(trackbar_controls_for_context(context))
        )

    def execute(self, context):
        frame = int(context.scene.frame_current)
        identities = tuple(
            (resolved.control.control_id, frame)
            for resolved in trackbar_controls_for_context(context)
        )
        _changed, selected = _add_trajectory_position_keys(context, identities)
        if not selected:
            self.report(
                {"WARNING"},
                "Trajectory Add Key currently requires unconstrained Object position tracks",
            )
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_toggle_trajectory_edit(bpy.types.Operator):
    bl_idname = "baw.toggle_trajectory_edit"
    bl_label = "Toggle Trajectory Key Edit"
    bl_description = "Edit selected Object or PoseBone position keys directly on their Motion Paths"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _trajectory_edit_mode_supported(context) and bool(
            trackbar_controls_for_context(context)
        )

    def execute(self, context):
        scene = context.scene
        enabled = not bool(scene.baw_trajectory_edit_mode)
        if enabled:
            if hasattr(scene, "baw_trajectory_visible"):
                scene.baw_trajectory_visible = True
            if not refresh_active_trajectory(context):
                self.report({"WARNING"}, "No editable Object/PoseBone trajectory is available")
                return {"CANCELLED"}
        scene.baw_trajectory_edit_mode = enabled
        if enabled:
            scene.baw_trajectory_transform_mode = "MOVE"
        else:
            scene.baw_trajectory_add_key_mode = False
        _set_path_edit_flag(context, enabled)
        _set_native_tool_gizmo_hidden(context, hidden=enabled)
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_edit_trajectory_key(bpy.types.Operator):
    bl_idname = "baw.edit_trajectory_key"
    bl_label = "Edit Trajectory Key"
    bl_description = "Select and move Object/PoseBone position keys directly in the 3D trajectory"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}

    _snapshots: tuple[TrajectoryPositionSnapshot, ...] = ()
    _hit_control_id: str = ""
    _hit_frame: int = 0
    _hit_world = Vector((0.0, 0.0, 0.0))
    _start_world = Vector((0.0, 0.0, 0.0))
    _start_mouse_x: float = 0.0
    _start_mouse_y: float = 0.0
    _has_moved: bool = False
    _plain_preserved_group: bool = False
    _tangent_snapshot: TrajectoryTangentSnapshot | None = None
    _tangent_side: str = ""
    _tangent_endpoint_world = Vector((0.0, 0.0, 0.0))

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            scene is not None
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
            and _trajectory_edit_mode_supported(context)
        )

    def invoke(self, context, event):
        if event.mouse_region_y <= TRACKBAR_INTERACTION_HEIGHT:
            return {"PASS_THROUGH"}
        if bool(getattr(context.scene, "baw_trajectory_add_key_mode", False)):
            return {"PASS_THROUGH"}

        transform_mode = str(
            getattr(context.scene, "baw_trajectory_transform_mode", "MOVE")
        )
        self._tangent_snapshot = None
        self._tangent_side = ""
        self._has_moved = False
        if transform_mode == "TANGENT":
            tangent_hit = _trajectory_tangent_handle_hit(
                context,
                event.mouse_region_x,
                event.mouse_region_y,
            )
            if tangent_hit is not None:
                (
                    _distance,
                    _control_id,
                    _frame,
                    side_name,
                    endpoint_world,
                    tangent_snapshot,
                ) = tangent_hit
                self._tangent_snapshot = tangent_snapshot
                self._tangent_side = side_name
                self._tangent_endpoint_world = Vector(endpoint_world)
                self._hit_world = Vector(endpoint_world)
                self._start_mouse_x = float(event.mouse_region_x)
                self._start_mouse_y = float(event.mouse_region_y)
                self._start_world = region_2d_to_location_3d(
                    context.region,
                    context.region_data,
                    Vector((event.mouse_region_x, event.mouse_region_y)),
                    self._hit_world,
                )
                if context.window is not None:
                    context.window.cursor_modal_set("SCROLL_XY")
                context.window_manager.modal_handler_add(self)
                return {"RUNNING_MODAL"}

        hit = _trajectory_key_hit(context, event.mouse_region_x, event.mouse_region_y)
        if hit is None:
            return {"PASS_THROUGH"}

        _distance, _active_bias, control_id, frame, _world, hit_selected = hit
        selected_before = _selected_position_identities(context)
        plain_preserved = (
            not event.shift
            and not event.ctrl
            and bool(hit_selected)
            and len(selected_before) > 1
        )
        if event.ctrl:
            _set_control_position_key_selection(context, control_id, frame, mode="TOGGLE")
            return {"FINISHED"}
        if event.shift:
            _set_control_position_key_selection(context, control_id, frame, mode="ADD")
        elif not plain_preserved:
            _set_control_position_key_selection(context, control_id, frame, mode="SET")

        # A path-key click owns selection only. The previous free-drag prototype
        # stayed modal after LEFTMOUSE press, so a following viewport pan/orbit
        # MOUSEMOVE could be interpreted as a Position edit. Spatial/timing edits
        # must start from an explicit trajectory gizmo instead.
        return {"FINISHED"}

    def modal(self, context, event):
        if self._tangent_snapshot is not None:
            if event.type == "MOUSEMOVE":
                distance = hypot(
                    float(event.mouse_region_x) - self._start_mouse_x,
                    float(event.mouse_region_y) - self._start_mouse_y,
                )
                if distance < _DRAG_THRESHOLD_PX and not self._has_moved:
                    return {"RUNNING_MODAL"}
                current_world = region_2d_to_location_3d(
                    context.region,
                    context.region_data,
                    Vector((event.mouse_region_x, event.mouse_region_y)),
                    self._hit_world,
                )
                target_world = self._tangent_endpoint_world + (
                    Vector(current_world) - self._start_world
                )
                if _apply_trajectory_tangent_world_target(
                    context,
                    self._tangent_snapshot,
                    self._tangent_side,
                    target_world,
                    refresh=True,
                ):
                    self._has_moved = True
                return {"RUNNING_MODAL"}

            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                if context.window is not None:
                    context.window.cursor_modal_restore()
                if self._has_moved:
                    refresh_active_trajectory_live(context, force=True)
                    sync_active_trajectory_key_selection(context)
                return {"FINISHED"}

            if event.type in {"ESC", "RIGHTMOUSE"}:
                _restore_trajectory_tangent_snapshot(
                    context,
                    self._tangent_snapshot,
                    refresh=True,
                )
                if context.window is not None:
                    context.window.cursor_modal_restore()
                return {"CANCELLED"}

            return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE":
            distance = hypot(
                float(event.mouse_region_x) - self._start_mouse_x,
                float(event.mouse_region_y) - self._start_mouse_y,
            )
            if distance < _DRAG_THRESHOLD_PX and not self._has_moved:
                return {"RUNNING_MODAL"}
            current_world = region_2d_to_location_3d(
                context.region,
                context.region_data,
                Vector((event.mouse_region_x, event.mouse_region_y)),
                self._hit_world,
            )
            delta_world = current_world - self._start_world
            if _apply_trajectory_world_delta(context, self._snapshots, delta_world):
                self._has_moved = True
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if context.window is not None:
                context.window.cursor_modal_restore()
            if not self._has_moved and self._plain_preserved_group:
                _set_control_position_key_selection(
                    context,
                    self._hit_control_id,
                    self._hit_frame,
                    mode="SET",
                )
            if self._has_moved:
                refresh_active_trajectory_live(context, force=True)
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            _restore_trajectory_position_snapshots(context, self._snapshots)
            if context.window is not None:
                context.window.cursor_modal_restore()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


def register_trajectory_edit_keymaps() -> None:
    wm = bpy.context.window_manager
    if wm is None or wm.keyconfigs.addon is None:
        return
    unregister_trajectory_edit_keymaps()
    kc = wm.keyconfigs.addon
    km = kc.keymaps.get("3D View")
    if km is None:
        km = kc.keymaps.new(name="3D View", space_type="VIEW_3D", region_type="WINDOW")
    object_km = kc.keymaps.get("Object Mode")
    if object_km is None:
        object_km = kc.keymaps.new(name="Object Mode", space_type="EMPTY", region_type="WINDOW")
    pose_km = kc.keymaps.get("Pose")
    if pose_km is None:
        pose_km = kc.keymaps.new(name="Pose", space_type="EMPTY", region_type="WINDOW")

    for target_km in (km, object_km, pose_km):
        for delete_event in ("DEL", "X"):
            delete_kmi = target_km.keymap_items.new(
                BAW_OT_delete_trajectory_position_keys.bl_idname,
                delete_event,
                "PRESS",
                head=True,
            )
            _TRAJECTORY_EDIT_KEYMAP_ITEMS.append((target_km, delete_kmi))
        current_frame_kmi = target_km.keymap_items.new(
            BAW_OT_add_trajectory_key_current_frame.bl_idname,
            "K",
            "PRESS",
            head=True,
        )
        _TRAJECTORY_EDIT_KEYMAP_ITEMS.append((target_km, current_frame_kmi))

    edit_kmi = km.keymap_items.new(
        BAW_OT_edit_trajectory_key.bl_idname,
        "LEFTMOUSE",
        "PRESS",
        any=True,
        head=True,
    )
    _TRAJECTORY_EDIT_KEYMAP_ITEMS.append((km, edit_kmi))

    # Registered last with head=True so Add Key mode owns path clicks before
    # the normal trajectory key-select/Move operator.
    add_key_kmi = km.keymap_items.new(
        BAW_OT_add_trajectory_key_on_path.bl_idname,
        "LEFTMOUSE",
        "PRESS",
        any=True,
        head=True,
    )
    _TRAJECTORY_EDIT_KEYMAP_ITEMS.append((km, add_key_kmi))


def unregister_trajectory_edit_keymaps() -> None:
    for km, kmi in reversed(_TRAJECTORY_EDIT_KEYMAP_ITEMS):
        if kmi.id != -1:
            km.keymap_items.remove(kmi)
    _TRAJECTORY_EDIT_KEYMAP_ITEMS.clear()
