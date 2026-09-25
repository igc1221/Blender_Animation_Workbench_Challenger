from __future__ import annotations

from math import hypot
from typing import ClassVar

import bpy
import gpu
from bpy.props import EnumProperty
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .character_query import characters_for_context
from .debug_causal import TraceRouteOutcome, TraceTerminalStatus
from .debug_trace import (
    new_trace_causal_child,
    new_trace_causal_root,
    trace_event,
    trace_lifecycle_event,
)
from .phase4_contact_ui import transform_orientation_cycle
from .rigped_box_wire_overlay import (
    box_display_names,
    box_world_vertices,
    rigped_box_display_enabled,
)
from .rigped_contract import (
    RigpedTransformGesture,
    RigpedTransformRoute,
    resolve_rigped_target,
    resolve_rigped_transform,
)
from .rigped_create_fit_ui import (
    fit_orientation_mode,
    fit_transform_mode,
    fit_ui_state,
    fit_ui_state_present,
    set_fit_orientation_mode,
    set_fit_transform_mode,
)
from .rigped_fit_session import (
    apply_fit_part_selection,
    fit_geometry_snapshot,
    fit_semantic_session,
    validate_fit_semantic_session,
)
from .rigped_transform import (
    activate_rigped_fk_joint_move_tool,
    activate_rigped_semantic_move_tool,
    deactivate_rigped_semantic_tool,
    direct_rotate_available,
)
from .semantic_adapter import control_context_for_context
from .trackbar_drawing import TRACKBAR_INTERACTION_HEIGHT
from .ui_language import text

_VIEWPORT_KEYMAP_ITEMS: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []
_PREVIOUS_ROTATE_AROUND_ACTIVE: bool | None = None
_PREVIOUS_FACE_SELECT_COLOR: tuple[float, float, float, float] | None = None

# 3ds Max-style click cycling: a fresh click starts at the nearest candidate;
# repeated clicks within five pixels walk front-to-back through the same stack.
_MAX_PICK_POINT: tuple[int, int] | None = None
_MAX_PICK_MODE: str | None = None
_MAX_PICK_CANDIDATES: tuple[str, ...] = ()
_MAX_PICK_INDEX = -1
_MAX_PICK_RESET_DISTANCE = 5.0

_INTERNAL_RIGPED_COLLECTIONS = frozenset(
    {"Mechanism", "Deform", "Export", "AWB Internal IK"}
)

_VIEWPORT_BOX_DRAW_HANDLE = None
_VIEWPORT_BOX_DRAW_STATE: tuple[int, float, float, float, float] | None = None


def _draw_viewport_box_selection() -> None:
    state = _VIEWPORT_BOX_DRAW_STATE
    context = bpy.context
    area = getattr(context, "area", None)
    if state is None or area is None or int(area.as_pointer()) != state[0]:
        return

    _, x1, y1, x2, y2 = state
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))

    # Match Blender's native box-gesture styling:
    # 1 px line, alternating 4 px gray / 4 px white (dash width 8, factor 0.5),
    # plus the native 5% white interior wash.
    base_segments = []
    white_segments = []
    for index in range(4):
        sx, sy = corners[index]
        ex, ey = corners[(index + 1) % 4]
        base_segments.extend(((sx, sy), (ex, ey)))

        dx = ex - sx
        dy = ey - sy
        length = max(1.0, hypot(dx, dy))
        ux = dx / length
        uy = dy / length
        cursor = 4.0
        while cursor < length:
            stop = min(length, cursor + 4.0)
            white_segments.extend(
                (
                    (sx + ux * cursor, sy + uy * cursor),
                    (sx + ux * stop, sy + uy * stop),
                )
            )
            cursor += 8.0

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    fill = batch_for_shader(shader, "TRI_FAN", {"pos": corners})
    base = batch_for_shader(shader, "LINES", {"pos": base_segments})
    white = (
        batch_for_shader(shader, "LINES", {"pos": white_segments})
        if white_segments
        else None
    )

    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(1.0)
    try:
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.05))
        fill.draw(shader)

        shader.uniform_float("color", (0.4, 0.4, 0.4, 1.0))
        base.draw(shader)

        if white is not None:
            shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
            white.draw(shader)
    finally:
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")


def _set_viewport_box_overlay(context, start: tuple[int, int], end: tuple[int, int]) -> None:
    global _VIEWPORT_BOX_DRAW_STATE
    area = getattr(context, "area", None)
    if area is None:
        return
    _VIEWPORT_BOX_DRAW_STATE = (
        int(area.as_pointer()),
        float(start[0]),
        float(start[1]),
        float(end[0]),
        float(end[1]),
    )
    area.tag_redraw()


def _clear_viewport_box_overlay(context=None) -> None:
    global _VIEWPORT_BOX_DRAW_STATE
    _VIEWPORT_BOX_DRAW_STATE = None
    area = getattr(context, "area", None) if context is not None else None
    if area is not None:
        area.tag_redraw()


def _register_viewport_box_overlay() -> None:
    global _VIEWPORT_BOX_DRAW_HANDLE
    if _VIEWPORT_BOX_DRAW_HANDLE is None:
        _VIEWPORT_BOX_DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_viewport_box_selection,
            (),
            "WINDOW",
            "POST_PIXEL",
        )


def _unregister_viewport_box_overlay() -> None:
    global _VIEWPORT_BOX_DRAW_HANDLE
    if _VIEWPORT_BOX_DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_VIEWPORT_BOX_DRAW_HANDLE, "WINDOW")
        except (RuntimeError, ValueError):
            pass
        _VIEWPORT_BOX_DRAW_HANDLE = None
    _clear_viewport_box_overlay()


def _is_view3d(context) -> bool:
    return context.area is not None and context.area.type == "VIEW_3D"


def _rigped_transform_decision(context, mode: str):
    """Return an I16 Rigped decision, or None for ordinary native contexts."""

    if context.mode not in {"OBJECT", "POSE"}:
        return None
    control_context = control_context_for_context(context)
    if not control_context.controls:
        return None
    summary = characters_for_context(context.scene, control_context)
    if not summary.character_ids:
        return None

    resolution = resolve_rigped_target(context.scene, control_context)
    if resolution.target is None:
        detail = "; ".join(issue.detail for issue in resolution.issues) or "Rigped transform target is ambiguous."
        return detail
    gesture = RigpedTransformGesture(mode)
    return resolve_rigped_transform(resolution.target, gesture)


def _hide_native_tool_gizmo(context) -> None:
    if hasattr(context.space_data, "show_gizmo_tool"):
        context.space_data.show_gizmo_tool = False
    context.area.tag_redraw()


def _active_native_transform_tool_id(context) -> str | None:
    workspace = getattr(context, "workspace", None)
    tools = getattr(workspace, "tools", None) if workspace is not None else None
    if tools is None:
        return None
    try:
        tool = tools.from_space_view3d_mode(context.mode, create=False)
    except (RuntimeError, TypeError):
        return None
    return str(getattr(tool, "idname", "") or "") or None


def _awb_transform_workspace_tool_id(context, mode: str) -> str:
    suffix = {
        "OBJECT": "object",
        "POSE": "pose",
    }.get(str(getattr(context, "mode", "")))
    if suffix is None:
        return {
            "MOVE": "builtin.move",
            "ROTATE": "builtin.rotate",
            "SCALE": "builtin.scale",
        }[str(mode)]
    return f"baw.{str(mode).lower()}_{suffix}"


def _activate_awb_transform_workspace_tool(context, mode: str) -> bool:
    tool_id = _awb_transform_workspace_tool_id(context, mode)
    result = bpy.ops.wm.tool_set_by_id(name=tool_id)
    if "FINISHED" not in result:
        return False
    if hasattr(context.space_data, "show_gizmo_tool"):
        context.space_data.show_gizmo_tool = False
    if context.area is not None:
        context.area.tag_redraw()
    return True


def _cycle_transform_orientation(context) -> str:
    enabled = transform_orientation_cycle(context)
    if context.mode == "POSE":
        # Animate mode cycles only World/Local. VIEW remains available elsewhere
        # but never enters repeated W/E/R transform-orientation cycling here.
        enabled = ("GLOBAL", "LOCAL")
    elif context.mode == "EDIT_ARMATURE":
        # In armature Edit Mode Blender LOCAL is the Armature object's basis.
        # Treat the user's Local preference as NORMAL so the gizmo follows the
        # active edit bone instead of looking identical to Global.
        enabled = tuple(dict.fromkeys("NORMAL" if item == "LOCAL" else item for item in enabled))
    slot = context.scene.transform_orientation_slots[0]
    current = str(slot.type)
    if current in enabled:
        next_orientation = enabled[(enabled.index(current) + 1) % len(enabled)]
    else:
        next_orientation = enabled[0]
    slot.type = next_orientation
    if context.area is not None:
        context.area.tag_redraw()
    return next_orientation


def _cycle_transform_orientation_preserving_semantic_mode(
    context,
    semantic_mode: str,
) -> str:
    """Cycle only orientation while keeping the active AWB transform route alive."""

    next_orientation = _cycle_transform_orientation(context)
    scene = getattr(context, "scene", None)
    if scene is not None and hasattr(scene, "baw_rigped_semantic_transform_mode"):
        scene.baw_rigped_semantic_transform_mode = str(semantic_mode)
    return next_orientation


def _reset_max_pick_cycle() -> None:
    global _MAX_PICK_POINT, _MAX_PICK_MODE, _MAX_PICK_CANDIDATES, _MAX_PICK_INDEX
    _MAX_PICK_POINT = None
    _MAX_PICK_MODE = None
    _MAX_PICK_CANDIDATES = ()
    _MAX_PICK_INDEX = -1


def _pose_bone_is_user_selectable(pose_bone) -> bool:
    bone = pose_bone.bone
    if bool(getattr(bone, "hide", False)) or bool(getattr(bone, "hide_select", False)):
        return False

    collections = tuple(getattr(bone, "collections", ()) or ())
    if collections and not any(bool(getattr(collection, "is_visible", True)) for collection in collections):
        return False

    names = {str(collection.name) for collection in collections}
    return not (
        names
        and "Authored" not in names
        and names.issubset(_INTERNAL_RIGPED_COLLECTIONS)
    )


def _mask_pose_bones_for_pick(rig, excluded: frozenset[str] = frozenset()):
    changed: list[tuple[object, bool]] = []
    for pose_bone in rig.pose.bones:
        bone = pose_bone.bone
        if _pose_bone_is_user_selectable(pose_bone) and str(pose_bone.name) not in excluded:
            continue
        changed.append((bone, bool(bone.hide_select)))
        bone.hide_select = True
    return changed


def _restore_pose_pick_mask(changed) -> None:
    for bone, hidden in changed:
        bone.hide_select = hidden


def _edit_bone_is_user_selectable(edit_bone) -> bool:
    if bool(getattr(edit_bone, "hide", False)) or bool(
        getattr(edit_bone, "hide_select", False)
    ):
        return False

    collections = tuple(getattr(edit_bone, "collections", ()) or ())
    if collections and not any(
        bool(getattr(collection, "is_visible", True)) for collection in collections
    ):
        return False

    names = {str(collection.name) for collection in collections}
    return not (
        names
        and "Authored" not in names
        and names.issubset(_INTERNAL_RIGPED_COLLECTIONS)
    )


def _native_pick_name(
    context,
    location: tuple[int, int],
    mode: str,
    *,
    excluded: tuple[str, ...] = (),
    diagnostics: dict[str, object] | None = None,
) -> str | None:
    excluded_names = frozenset(str(name) for name in excluded)

    if mode == "POSE":
        rig = getattr(context, "active_object", None)
        if rig is None or getattr(rig, "type", None) != "ARMATURE":
            return None
        if rigped_box_display_enabled(rig):
            picked = _box_bone_pick_name(context, rig, location)
            return picked if picked not in excluded_names else None
        selected_before = tuple(
            str(bone.name) for bone in (getattr(context, "selected_pose_bones", ()) or ())
        )
        active_before = getattr(getattr(rig.data, "bones", None), "active", None)
        active_before_name = str(active_before.name) if active_before is not None else None
        masked = _mask_pose_bones_for_pick(rig, excluded_names)
        try:
            for pose_bone in rig.pose.bones:
                pose_bone.select = False
            rig.data.bones.active = None
            bpy.ops.view3d.select(
                extend=False,
                deselect=False,
                toggle=False,
                deselect_all=True,
                enumerate=False,
                location=location,
            )
            clicked = getattr(context, "active_pose_bone", None)
            picked = (
                str(clicked.name)
                if clicked is not None and bool(clicked.select)
                else None
            )
        finally:
            _restore_pose_pick_mask(masked)
            selected_names = set(selected_before)
            for pose_bone in rig.pose.bones:
                pose_bone.select = str(pose_bone.name) in selected_names
            rig.data.bones.active = (
                rig.data.bones.get(active_before_name)
                if active_before_name is not None
                else None
            )
        return picked

    if mode == "OBJECT":
        box_diagnostics: dict[str, object] = {}
        box_pick = _box_object_pick_name(
            context,
            location,
            diagnostics=box_diagnostics if diagnostics is not None else None,
        )
        if diagnostics is not None:
            diagnostics["box"] = box_diagnostics
            diagnostics["excluded_names"] = tuple(sorted(excluded_names))
        if box_pick is not None and box_pick not in excluded_names:
            if diagnostics is not None:
                diagnostics["route"] = "BOX"
                diagnostics["picked"] = box_pick
            return box_pick
        if diagnostics is not None and box_pick is not None:
            diagnostics["box_pick_excluded"] = True
        selected_before = tuple(
            str(obj.name) for obj in (getattr(context, "selected_objects", ()) or ())
        )
        active_before = getattr(context, "active_object", None)
        active_before_name = str(active_before.name) if active_before is not None else None
        masked: list[tuple[object, bool]] = []
        for obj in context.scene.objects:
            names = {str(collection.name) for collection in obj.users_collection}
            internal = bool(
                names
                and "Authored" not in names
                and names.issubset(_INTERNAL_RIGPED_COLLECTIONS)
            )
            if str(obj.name) not in excluded_names and not internal:
                continue
            masked.append((obj, bool(obj.hide_select)))
            obj.hide_select = True
        try:
            for obj in tuple(getattr(context, "selected_objects", ()) or ()):
                obj.select_set(False)
            context.view_layer.objects.active = None
            bpy.ops.view3d.select(
                extend=False,
                deselect=False,
                toggle=False,
                deselect_all=True,
                enumerate=False,
                location=location,
            )
            clicked = getattr(context, "active_object", None)
            picked = (
                str(clicked.name)
                if clicked is not None and bool(clicked.select_get())
                else None
            )
        finally:
            for obj, hidden in masked:
                obj.hide_select = hidden
            for obj in tuple(getattr(context, "selected_objects", ()) or ()):
                obj.select_set(False)
            for name in selected_before:
                obj = context.scene.objects.get(name)
                if obj is not None:
                    obj.select_set(True)
            context.view_layer.objects.active = (
                context.scene.objects.get(active_before_name)
                if active_before_name is not None
                else None
            )
        if diagnostics is not None:
            diagnostics["native_pick"] = picked
            diagnostics["route"] = "NATIVE" if picked is not None else "MISS"
            diagnostics["picked"] = picked
        return picked

    if mode == "EDIT_ARMATURE":
        rig = getattr(context, "active_object", None)
        if rig is None or getattr(rig, "type", None) != "ARMATURE":
            return None
        if rigped_box_display_enabled(rig):
            picked = _box_bone_pick_name(context, rig, location)
            return picked if picked not in excluded_names else None

        edit_bones = rig.data.edit_bones
        active_before = getattr(edit_bones, "active", None)
        active_before_name = (
            str(active_before.name) if active_before is not None else None
        )
        state_before = tuple(
            (
                str(bone.name),
                bool(bone.select),
                bool(bone.select_head),
                bool(bone.select_tail),
                bool(bone.hide_select),
            )
            for bone in edit_bones
        )

        try:
            for bone in edit_bones:
                bone.select = False
                bone.select_head = False
                bone.select_tail = False
                if (
                    not _edit_bone_is_user_selectable(bone)
                    or str(bone.name) in excluded_names
                ):
                    bone.hide_select = True
            edit_bones.active = None

            bpy.ops.view3d.select(
                extend=False,
                deselect=False,
                toggle=False,
                deselect_all=True,
                enumerate=False,
                location=location,
            )
            clicked = getattr(edit_bones, "active", None)
            return (
                str(clicked.name)
                if clicked is not None
                and (
                    bool(clicked.select)
                    or bool(clicked.select_head)
                    or bool(clicked.select_tail)
                )
                else None
            )
        finally:
            state_by_name = {name: values for name, *values in state_before}
            for bone in edit_bones:
                values = state_by_name.get(str(bone.name))
                if values is None:
                    continue
                selected, selected_head, selected_tail, hidden = values
                bone.select = selected
                bone.select_head = selected_head
                bone.select_tail = selected_tail
                bone.hide_select = hidden
            edit_bones.active = (
                edit_bones.get(active_before_name)
                if active_before_name is not None
                else None
            )

    return None


def _native_max_style_pick(
    context,
    location: tuple[int, int],
    cycle_location: tuple[int, int],
    mode: str,
    *,
    diagnostics: dict[str, object] | None = None,
) -> str | None:
    del cycle_location
    # Visible/front-most only: repeated clicks at the same screen position must
    # never cycle through geometry hidden behind the first native hit.
    _reset_max_pick_cycle()
    return _native_pick_name(
        context,
        location,
        mode,
        diagnostics=diagnostics,
    )


def _selection_rect(
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[float, float, float, float]:
    return (
        float(min(start[0], end[0])),
        float(max(start[0], end[0])),
        float(min(start[1], end[1])),
        float(max(start[1], end[1])),
    )


def _rects_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    ax1, ax2, ay1, ay2 = first
    bx1, bx2, by1, by2 = second
    return ax2 >= bx1 and bx2 >= ax1 and ay2 >= by1 and by2 >= ay1


def _point_in_rect(
    point,
    rect: tuple[float, float, float, float],
) -> bool:
    x1, x2, y1, y2 = rect
    return x1 <= float(point[0]) <= x2 and y1 <= float(point[1]) <= y2


def _segment_intersects_rect(
    start,
    end,
    rect: tuple[float, float, float, float],
) -> bool:
    if _point_in_rect(start, rect) or _point_in_rect(end, rect):
        return True

    x1, x2, y1, y2 = rect
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    p = (-dx, dx, -dy, dy)
    q = (
        float(start[0]) - x1,
        x2 - float(start[0]),
        float(start[1]) - y1,
        y2 - float(start[1]),
    )
    low = 0.0
    high = 1.0
    for edge_p, edge_q in zip(p, q, strict=True):
        if abs(edge_p) <= 1e-9:
            if edge_q < 0.0:
                return False
            continue
        ratio = edge_q / edge_p
        if edge_p < 0.0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return False
    return True


def _point_in_polygon_2d(point, polygon) -> bool:
    if len(polygon) < 3:
        return False
    x = float(point[0])
    y = float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= intersection_x:
                inside = not inside
        previous = current
    return inside


def _point_segment_distance_2d(point, start, end) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx = x2 - x1
    dy = y2 - y1
    length_sq = (dx * dx) + (dy * dy)
    if length_sq <= 1e-9:
        return hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest_x = x1 + (dx * t)
    closest_y = y1 + (dy * t)
    return hypot(px - closest_x, py - closest_y)


def _point_hits_polygon_2d(point, polygon, tolerance: float = 6.0) -> bool:
    if _point_in_polygon_2d(point, polygon):
        return True
    if len(polygon) < 2:
        return False
    previous = polygon[-1]
    for current in polygon:
        if _point_segment_distance_2d(point, previous, current) <= tolerance:
            return True
        previous = current
    return False


def _polygon_intersects_rect(polygon, rect) -> bool:
    if not polygon:
        return False
    if any(_point_in_rect(point, rect) for point in polygon):
        return True
    x1, x2, y1, y2 = rect
    rect_corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    if any(_point_in_polygon_2d(corner, polygon) for corner in rect_corners):
        return True
    previous = polygon[-1]
    for current in polygon:
        if _segment_intersects_rect(previous, current, rect):
            return True
        previous = current
    return False


def _convex_hull_2d(points):
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 2:
        return unique

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _project_box_polygon(context, rv3d, vertices):
    projected = []
    for vertex in vertices:
        point = location_3d_to_region_2d(
            context.region,
            rv3d,
            Vector(vertex),
        )
        if point is not None:
            projected.append(point)
    return _convex_hull_2d(projected) if projected else ()


def _box_view_depth(rv3d, vertices) -> float:
    center = Vector((0.0, 0.0, 0.0))
    for vertex in vertices:
        center += Vector(vertex)
    center /= max(len(vertices), 1)
    return float((rv3d.view_matrix @ center).z)


def _box_bone_pick_name(context, rig, location: tuple[int, int]) -> str | None:
    rv3d = getattr(context, "region_data", None) or getattr(
        getattr(context, "space_data", None), "region_3d", None
    )
    if rv3d is None or not rigped_box_display_enabled(rig):
        return None

    hits = []
    for name in box_display_names(rig):
        vertices = box_world_vertices(rig, name)
        if len(vertices) != 8:
            continue
        polygon = _project_box_polygon(context, rv3d, vertices)
        if polygon and _point_in_polygon_2d(location, polygon):
            hits.append((_box_view_depth(rv3d, vertices), name))
    if not hits:
        return None
    return max(hits, key=lambda item: item[0])[1]


def _fit_part_pick_id(
    context,
    location: tuple[int, int],
    *,
    diagnostics: dict[str, object] | None = None,
) -> str | None:
    snapshot = fit_geometry_snapshot(context)
    rv3d = getattr(context, "region_data", None) or getattr(
        getattr(context, "space_data", None), "region_3d", None
    )
    if snapshot is None or rv3d is None:
        if diagnostics is not None:
            diagnostics["rejection"] = "NO_FIT_SNAPSHOT"
        return None

    hits = []
    hit_parts = []
    for part in snapshot.parts:
        if len(part.vertices) != 8:
            continue
        polygon = _project_box_polygon(context, rv3d, part.vertices)
        if polygon and _point_hits_polygon_2d(location, polygon):
            depth = _box_view_depth(rv3d, part.vertices)
            hits.append((depth, part.part_id, part.name_hint))
            hit_parts.append({"part_id": part.part_id, "name": part.name_hint, "depth": depth})
    picked = max(hits, key=lambda item: item[0]) if hits else None
    if diagnostics is not None:
        diagnostics.update(
            {
                "fit_revision": snapshot.revision,
                "fit_parts": len(snapshot.parts),
                "hit_parts": hit_parts[:32],
                "picked_part_id": picked[1] if picked else None,
                "picked_name": picked[2] if picked else None,
                "rejection": None if picked else "NO_FIT_PART_HIT",
            }
        )
    return picked[1] if picked else None


def _box_display_rigs(context):
    scene = getattr(context, "scene", None)
    if scene is None:
        return ()
    return tuple(
        obj
        for obj in scene.objects
        if rigped_box_display_enabled(obj)
        and not obj.hide_get()
        and not bool(getattr(obj, "hide_select", False))
    )


def _box_object_pick_name(
    context,
    location: tuple[int, int],
    *,
    diagnostics: dict[str, object] | None = None,
) -> str | None:
    rv3d = getattr(context, "region_data", None) or getattr(
        getattr(context, "space_data", None), "region_3d", None
    )
    scene = getattr(context, "scene", None)
    if diagnostics is not None:
        region = getattr(context, "region", None)
        diagnostics.update(
            {
                "location": location,
                "region_size": (
                    (int(region.width), int(region.height))
                    if region is not None
                    else None
                ),
                "has_region_3d": rv3d is not None,
                "display_rigs": (
                    [
                        str(obj.name)
                        for obj in scene.objects
                        if rigped_box_display_enabled(obj)
                        and not obj.hide_get()
                        and str(getattr(obj, "mode", "")) != "EDIT"
                    ]
                    if scene is not None
                    else []
                ),
            }
        )
    if rv3d is None:
        if diagnostics is not None:
            diagnostics["rejection"] = "NO_REGION_3D"
        return None

    rigs = _box_display_rigs(context)
    hits = []
    nearest_distance: float | None = None
    nearest_part: dict[str, object] | None = None
    projected_parts = 0
    valid_parts = 0
    hit_parts: list[dict[str, object]] = []
    total_parts = 0
    for rig in rigs:
        names = tuple(box_display_names(rig))
        total_parts += len(names)
        for name in names:
            vertices = box_world_vertices(rig, name)
            if len(vertices) != 8:
                continue
            valid_parts += 1
            polygon = _project_box_polygon(context, rv3d, vertices)
            if polygon:
                projected_parts += 1
            if diagnostics is not None and polygon:
                previous = polygon[-1]
                edge_distance = float("inf")
                for current in polygon:
                    edge_distance = min(
                        edge_distance,
                        _point_segment_distance_2d(location, previous, current),
                    )
                    previous = current
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                if nearest_distance is None or edge_distance < nearest_distance:
                    nearest_distance = edge_distance
                    nearest_part = {
                        "rig": str(rig.name),
                        "part": str(name),
                        "edge_distance": edge_distance,
                        "inside": bool(_point_in_polygon_2d(location, polygon)),
                        "bounds": [min(xs), max(xs), min(ys), max(ys)],
                    }
            if polygon and _point_hits_polygon_2d(location, polygon):
                depth = _box_view_depth(rv3d, vertices)
                hits.append((depth, str(rig.name)))
                if len(hit_parts) < 32:
                    hit_parts.append(
                        {
                            "rig": str(rig.name),
                            "part": str(name),
                            "depth": depth,
                        }
                    )

    picked = max(hits, key=lambda item: item[0])[1] if hits else None
    if diagnostics is not None:
        diagnostics.update(
            {
                "pick_rigs": [str(rig.name) for rig in rigs],
                "total_parts": total_parts,
                "valid_box_parts": valid_parts,
                "projected_parts": projected_parts,
                "hit_parts": hit_parts,
                "nearest_part": nearest_part,
                "picked_rig": picked,
                "rejection": None if picked is not None else "NO_BOX_HIT",
            }
        )
    return picked

def _mesh_object_intersects_rect(context, rv3d, obj, rect, depsgraph) -> bool:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = None
    try:
        mesh = evaluated.to_mesh()
        if mesh is None:
            return False
        matrix_world = evaluated.matrix_world
        projected = [
            location_3d_to_region_2d(
                context.region,
                rv3d,
                matrix_world @ vertex.co,
            )
            for vertex in mesh.vertices
        ]

        if any(point is not None and _point_in_rect(point, rect) for point in projected):
            return True

        for edge in mesh.edges:
            first = projected[edge.vertices[0]]
            second = projected[edge.vertices[1]]
            if (
                first is not None
                and second is not None
                and _segment_intersects_rect(first, second, rect)
            ):
                return True

        x1, x2, y1, y2 = rect
        rect_corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        for polygon in mesh.polygons:
            polygon_points = [
                projected[index]
                for index in polygon.vertices
                if projected[index] is not None
            ]
            if len(polygon_points) < 3:
                continue
            if any(
                _point_in_polygon_2d(corner, polygon_points)
                for corner in rect_corners
            ):
                return True
        return False
    finally:
        if mesh is not None:
            evaluated.to_mesh_clear()


def _fit_part_box_crossing_ids(context, rect) -> tuple[str, ...]:
    snapshot = fit_geometry_snapshot(context)
    rv3d = getattr(context, "region_data", None) or getattr(
        getattr(context, "space_data", None), "region_3d", None
    )
    if snapshot is None or rv3d is None:
        return ()
    hits = []
    for part in snapshot.parts:
        polygon = _project_box_polygon(context, rv3d, part.vertices)
        if polygon and _polygon_intersects_rect(polygon, rect):
            hits.append(part.part_id)
    return tuple(hits)


def _object_box_crossing_hits(context, rect) -> tuple[object, ...]:
    rv3d = getattr(context, "region_data", None) or getattr(
        getattr(context, "space_data", None), "region_3d", None
    )
    if rv3d is None:
        return ()

    depsgraph = context.evaluated_depsgraph_get()
    hits = []
    box_rigs = _box_display_rigs(context)
    box_rig_names = {str(rig.name) for rig in box_rigs}

    for rig in box_rigs:
        matched = False
        for name in box_display_names(rig):
            vertices = box_world_vertices(rig, name)
            if len(vertices) != 8:
                continue
            polygon = _project_box_polygon(context, rv3d, vertices)
            if polygon and _polygon_intersects_rect(polygon, rect):
                matched = True
                break
        if matched:
            hits.append(rig)

    for obj in tuple(getattr(context, "visible_objects", ()) or ()):
        if bool(getattr(obj, "hide_select", False)) or str(obj.name) in box_rig_names:
            continue

        if getattr(obj, "type", "") == "MESH":
            if _mesh_object_intersects_rect(context, rv3d, obj, rect, depsgraph):
                hits.append(obj)
            continue

        projected = []
        for corner in tuple(getattr(obj, "bound_box", ()) or ()):
            point = location_3d_to_region_2d(
                context.region,
                rv3d,
                obj.matrix_world @ Vector(corner),
            )
            if point is not None:
                projected.append(point)
        if projected and _polygon_intersects_rect(_convex_hull_2d(projected), rect):
            hits.append(obj)
    return tuple(hits)


def _edit_bone_box_crossing_hits(context, rect) -> tuple[object, ...]:
    rig = getattr(context, "active_object", None)
    rv3d = getattr(context, "region_data", None) or getattr(
        getattr(context, "space_data", None), "region_3d", None
    )
    if (
        rig is None
        or getattr(rig, "type", None) != "ARMATURE"
        or rv3d is None
        or getattr(rig.data, "edit_bones", None) is None
    ):
        return ()

    hits = []
    if rigped_box_display_enabled(rig):
        for name in box_display_names(rig):
            edit_bone = rig.data.edit_bones.get(name)
            if edit_bone is None or bool(getattr(edit_bone, "hide", False)) or bool(
                getattr(edit_bone, "hide_select", False)
            ):
                continue
            vertices = box_world_vertices(rig, name)
            polygon = _project_box_polygon(context, rv3d, vertices)
            if polygon and _polygon_intersects_rect(polygon, rect):
                hits.append(edit_bone)
        return tuple(hits)

    for edit_bone in rig.data.edit_bones:
        if bool(getattr(edit_bone, "hide", False)) or bool(
            getattr(edit_bone, "hide_select", False)
        ):
            continue

        length = max(1e-6, float(edit_bone.length))
        midpoint = (edit_bone.head + edit_bone.tail) * 0.5
        display_radius = length * 0.10
        ring_points = (
            edit_bone.head,
            edit_bone.tail,
            midpoint + edit_bone.x_axis * display_radius,
            midpoint - edit_bone.x_axis * display_radius,
            midpoint + edit_bone.z_axis * display_radius,
            midpoint - edit_bone.z_axis * display_radius,
        )
        projected = []
        for local_point in ring_points:
            point = location_3d_to_region_2d(
                context.region,
                rv3d,
                rig.matrix_world @ local_point,
            )
            if point is not None:
                projected.append(point)

        if projected and _polygon_intersects_rect(_convex_hull_2d(projected), rect):
            hits.append(edit_bone)
    return tuple(hits)


def _pose_bone_box_crossing_hits(context, rect) -> tuple[object, ...]:
    rig = getattr(context, "active_object", None)
    rv3d = getattr(context, "region_data", None) or getattr(
        getattr(context, "space_data", None), "region_3d", None
    )
    if (
        rig is None
        or getattr(rig, "type", None) != "ARMATURE"
        or rv3d is None
        or not rigped_box_display_enabled(rig)
    ):
        return ()

    hits = []
    for name in box_display_names(rig):
        pose_bone = rig.pose.bones.get(name)
        if pose_bone is None or bool(getattr(pose_bone.bone, "hide", False)) or bool(
            getattr(pose_bone.bone, "hide_select", False)
        ):
            continue
        vertices = box_world_vertices(rig, name)
        polygon = _project_box_polygon(context, rv3d, vertices)
        if polygon and _polygon_intersects_rect(polygon, rect):
            hits.append(pose_bone)
    return tuple(hits)


def _apply_crossing_hits(context, hits, action: str, mode: str):
    if mode == "OBJECT":
        if action == "SET":
            for obj in tuple(getattr(context, "selected_objects", ()) or ()):
                obj.select_set(False)
        for obj in hits:
            obj.select_set(action != "REMOVE")
        if action != "REMOVE" and hits:
            context.view_layer.objects.active = hits[-1]
        elif action == "SET" and not hits:
            context.view_layer.objects.active = None
        return {"FINISHED"}

    if mode == "EDIT_ARMATURE":
        rig = getattr(context, "active_object", None)
        if rig is None or getattr(rig, "type", None) != "ARMATURE":
            return {"CANCELLED"}
        if action == "SET":
            for edit_bone in rig.data.edit_bones:
                _set_edit_bone_selected(edit_bone, False)
        for edit_bone in hits:
            _set_edit_bone_selected(edit_bone, action != "REMOVE")
        if action != "REMOVE" and hits:
            rig.data.edit_bones.active = hits[-1]
        elif action == "SET" and not hits:
            rig.data.edit_bones.active = None
        return {"FINISHED"}

    if mode == "POSE":
        rig = getattr(context, "active_object", None)
        if rig is None or getattr(rig, "type", None) != "ARMATURE":
            return {"CANCELLED"}
        if action == "SET":
            for pose_bone in rig.pose.bones:
                pose_bone.select = False
        for pose_bone in hits:
            pose_bone.select = action != "REMOVE"
        if action != "REMOVE" and hits:
            rig.data.bones.active = rig.data.bones.get(str(hits[-1].name))
        elif action == "SET" and not hits:
            rig.data.bones.active = None
        return {"FINISHED"}

    return {"CANCELLED"}


def _apply_box_pick(
    context,
    start: tuple[int, int],
    end: tuple[int, int],
    action: str,
):
    mode = str(getattr(context, "mode", ""))
    rect = _selection_rect(start, end)

    # Object and Fit both use crossing selection: touching the selection
    # rectangle is enough. Blender's native box selection is origin/containment
    # biased in these modes, which made Animate behave differently.
    if mode == "OBJECT":
        fit_state = fit_ui_state(context)
        session = fit_semantic_session(context)
        if fit_ui_state_present(context) and (fit_state is None or session is None):
            trace_event(
                "INPUT",
                "FIT_SELECTION_BOX_REFUSED",
                context=context,
                action=action,
                rect=rect,
                issues=("FIT_F3_SESSION_MISSING",),
            )
            return {"CANCELLED"}
        if session is not None:
            issues = validate_fit_semantic_session(context, session)
            if issues:
                trace_event(
                    "INPUT",
                    "FIT_SELECTION_BOX_REFUSED",
                    context=context,
                    action=action,
                    rect=rect,
                    issues=issues,
                )
                return {"CANCELLED"}
            hit_part_ids = _fit_part_box_crossing_ids(context, rect)
            apply_fit_part_selection(
                context,
                hit_part_ids,
                action,
            )
            session = fit_semantic_session(context)
            selected_part_ids = ()
            active_part_id = None
            fit_revision = None
            fit_parts = 0
            if session is not None:
                selected_part_ids = tuple(
                    part.part_id
                    for part in session.geometry.parts
                    if part.part_id in session.selected_part_ids
                )
                active_part_id = session.active_part_id
                fit_revision = session.revision
                fit_parts = len(session.geometry.parts)
            trace_event(
                "INPUT",
                "FIT_SELECTION_BOX_RESULT",
                context=context,
                source="AWB_SELECT_BOX",
                action=action,
                rect=rect,
                hit_part_ids=hit_part_ids,
                selected_part_ids=selected_part_ids,
                active_part_id=active_part_id,
                fit_revision=fit_revision,
                fit_parts=fit_parts,
            )
            return {"FINISHED"}
        return _apply_crossing_hits(
            context,
            _object_box_crossing_hits(context, rect),
            action,
            mode,
        )
    if mode == "EDIT_ARMATURE":
        return _apply_crossing_hits(
            context,
            _edit_bone_box_crossing_hits(context, rect),
            action,
            mode,
        )
    if mode == "POSE":
        rig = getattr(context, "active_object", None)
        if rig is not None and rigped_box_display_enabled(rig):
            return _apply_crossing_hits(
                context,
                _pose_bone_box_crossing_hits(context, rect),
                action,
                mode,
            )

    mode_map = {"SET": "SET", "ADD": "ADD", "REMOVE": "SUB"}
    select_mode = mode_map[action]
    x1, x2, y1, y2 = rect
    masked = ()
    rig = None
    if mode == "POSE":
        rig = getattr(context, "active_object", None)
        if rig is not None and getattr(rig, "type", None) == "ARMATURE":
            masked = _mask_pose_bones_for_pick(rig)
    try:
        return bpy.ops.view3d.select_box(
            xmin=int(x1),
            xmax=int(x2),
            ymin=int(y1),
            ymax=int(y2),
            wait_for_input=False,
            mode=select_mode,
        )
    finally:
        if rig is not None:
            _restore_pose_pick_mask(masked)


def _apply_pose_pick(context, bone_name: str, action: str) -> None:
    rig = context.active_object
    target = rig.pose.bones.get(bone_name)
    if target is None:
        return

    if action == "SET":
        for pose_bone in rig.pose.bones:
            pose_bone.select = False
        target.select = True
        rig.data.bones.active = rig.data.bones.get(bone_name)
        return

    if action == "ADD":
        target.select = True
        rig.data.bones.active = rig.data.bones.get(bone_name)
        return

    target.select = False
    selected = tuple(
        pose_bone for pose_bone in rig.pose.bones if bool(pose_bone.select)
    )
    active = getattr(rig.data.bones, "active", None)
    if active is not None and str(active.name) == bone_name:
        rig.data.bones.active = (
            rig.data.bones.get(str(selected[-1].name)) if selected else None
        )


def _apply_object_pick(context, object_name: str, action: str) -> None:
    target = context.scene.objects.get(object_name)
    if target is None:
        return

    if action == "SET":
        for obj in tuple(getattr(context, "selected_objects", ()) or ()):
            obj.select_set(False)
        target.select_set(True)
        context.view_layer.objects.active = target
        return

    if action == "ADD":
        target.select_set(True)
        context.view_layer.objects.active = target
        return

    target.select_set(False)
    selected = tuple(getattr(context, "selected_objects", ()) or ())
    if context.view_layer.objects.active == target:
        context.view_layer.objects.active = selected[-1] if selected else None


def _set_edit_bone_selected(bone, selected: bool) -> None:
    bone.select = selected
    if hasattr(bone, "select_head"):
        bone.select_head = selected
    if hasattr(bone, "select_tail"):
        bone.select_tail = selected


def _apply_edit_bone_pick(context, bone_name: str, action: str) -> None:
    rig = getattr(context, "active_object", None)
    if rig is None or getattr(rig, "type", None) != "ARMATURE":
        return
    edit_bones = rig.data.edit_bones
    target = edit_bones.get(bone_name)
    if target is None:
        return

    if action == "SET":
        for bone in edit_bones:
            _set_edit_bone_selected(bone, False)
        _set_edit_bone_selected(target, True)
        edit_bones.active = target
        return

    if action == "ADD":
        _set_edit_bone_selected(target, True)
        edit_bones.active = target
        return

    _set_edit_bone_selected(target, False)
    selected = tuple(
        bone for bone in edit_bones if bool(getattr(bone, "select", False))
    )
    if getattr(edit_bones, "active", None) == target:
        edit_bones.active = selected[-1] if selected else None


def _clear_click_selection_on_miss(context, mode: str, action: str) -> None:
    if action != "SET":
        return
    if mode == "POSE":
        rig = getattr(context, "active_object", None)
        if rig is not None and getattr(rig, "type", None) == "ARMATURE":
            for pose_bone in rig.pose.bones:
                pose_bone.select = False
            rig.data.bones.active = None
    elif mode == "OBJECT":
        for obj in tuple(getattr(context, "selected_objects", ()) or ()):
            obj.select_set(False)
        context.view_layer.objects.active = None
    elif mode == "EDIT_ARMATURE":
        rig = getattr(context, "active_object", None)
        if rig is not None and getattr(rig, "type", None) == "ARMATURE":
            for bone in rig.data.edit_bones:
                _set_edit_bone_selected(bone, False)
            rig.data.edit_bones.active = None


def apply_awb_click_selection(
    context,
    location: tuple[int, int],
    cycle_location: tuple[int, int],
    action: str = "SET",
    *,
    source: str = "AWB_SELECTION",
):
    """Run the same global AWB click-selection path from non-selection modals."""

    mode = str(getattr(context, "mode", ""))
    fit_state = fit_ui_state(context)
    session = fit_semantic_session(context)
    if (
        mode == "OBJECT"
        and fit_ui_state_present(context)
        and (fit_state is None or session is None)
    ):
        trace_event(
            "INPUT",
            "FIT_SELECTION_CLICK_REFUSED",
            context=context,
            source=source,
            action=action,
            location=location,
            issues=("FIT_F3_SESSION_MISSING",),
        )
        return {"CANCELLED"}
    if mode == "OBJECT" and session is not None:
        issues = validate_fit_semantic_session(context, session)
        if issues:
            trace_event(
                "INPUT",
                "FIT_SELECTION_CLICK_REFUSED",
                context=context,
                source=source,
                action=action,
                location=location,
                issues=issues,
            )
            return {"CANCELLED"}
        fit_diagnostics: dict[str, object] = {}
        picked_part_id = _fit_part_pick_id(
            context,
            location,
            diagnostics=fit_diagnostics,
        )
        apply_fit_part_selection(
            context,
            (() if picked_part_id is None else (picked_part_id,)),
            action,
        )
        trace_event(
            "INPUT",
            "FIT_SELECTION_CLICK_RESULT",
            context=context,
            source=source,
            action=action,
            location=location,
            result=("MISS" if picked_part_id is None else "PICK"),
            picked_part_id=picked_part_id,
            pick_diagnostics=fit_diagnostics,
        )
        return {"FINISHED"}
    active_tool_before = _active_native_transform_tool_id(context)
    if mode in {"POSE", "OBJECT", "EDIT_ARMATURE"}:
        pick_diagnostics: dict[str, object] = {}
        picked = _native_max_style_pick(
            context,
            location,
            cycle_location,
            mode,
            diagnostics=pick_diagnostics if mode == "OBJECT" else None,
        )
        if picked is None:
            _clear_click_selection_on_miss(context, mode, action)
            if mode == "OBJECT":
                trace_event(
                    "INPUT",
                    "SELECTION_CLICK_RESULT",
                    context=context,
                    source=source,
                    action=action,
                    location=location,
                    cycle_location=cycle_location,
                    active_tool_before=active_tool_before,
                    result="MISS",
                    picked=None,
                    pick_diagnostics=pick_diagnostics,
                )
            return {"FINISHED"}
        if mode == "POSE":
            _apply_pose_pick(context, picked, action)
        elif mode == "OBJECT":
            _apply_object_pick(context, picked, action)
        else:
            _apply_edit_bone_pick(context, picked, action)
        if mode == "OBJECT":
            trace_event(
                "INPUT",
                "SELECTION_CLICK_RESULT",
                context=context,
                source=source,
                action=action,
                location=location,
                cycle_location=cycle_location,
                active_tool_before=active_tool_before,
                result="PICK",
                picked=picked,
                pick_diagnostics=pick_diagnostics,
            )
        return {"FINISHED"}

    return {"CANCELLED"}


def prioritize_awb_selection_over_free_rotate(context, event) -> bool:
    """Let a different selectable target beat the invisible Free-Rotate disk."""

    location = (int(event.mouse_region_x), int(event.mouse_region_y))
    action = "REMOVE" if event.alt else ("ADD" if event.ctrl else "SET")
    mode = str(getattr(context, "mode", ""))

    if mode in {"OBJECT", "POSE"}:
        picked = _native_pick_name(context, location, mode)
        if picked is None:
            return False

        if mode == "OBJECT":
            active = getattr(context.view_layer.objects, "active", None)
            target = context.scene.objects.get(picked)
            if (
                action == "SET"
                and active is not None
                and str(active.name) == picked
                and target is not None
                and bool(target.select_get())
            ):
                return False
            _reset_max_pick_cycle()
            _apply_object_pick(context, picked, action)
            return True

        rig = getattr(context, "active_object", None)
        active_bone = getattr(getattr(rig, "data", None), "bones", None)
        active_bone = getattr(active_bone, "active", None)
        target = rig.pose.bones.get(picked) if rig is not None else None
        if (
            action == "SET"
            and active_bone is not None
            and str(active_bone.name) == picked
            and target is not None
            and bool(target.select)
        ):
            return False
        _reset_max_pick_cycle()
        _apply_pose_pick(context, picked, action)
        return True

    if mode == "EDIT_ARMATURE":
        rig = getattr(context, "active_object", None)
        if rig is None or getattr(rig, "type", None) != "ARMATURE":
            return False
        edit_bones = rig.data.edit_bones
        active_before = getattr(edit_bones, "active", None)
        active_before_name = (
            str(active_before.name) if active_before is not None else None
        )
        picked = _native_pick_name(context, location, mode)
        if picked is None:
            return False
        target = edit_bones.get(picked)
        if (
            action == "SET"
            and picked == active_before_name
            and target is not None
            and bool(getattr(target, "select", False))
        ):
            return False
        _reset_max_pick_cycle()
        _apply_edit_bone_pick(context, picked, action)
        return True

    return False


class BAW_OT_set_view_projection(bpy.types.Operator):
    bl_idname = "baw.set_view_projection"
    bl_label = "Set View Projection"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    mode: EnumProperty(
        items=(
            ("PERSP", "Perspective", "Use perspective projection"),
            ("ORTHO", "Orthographic", "Use orthographic projection"),
        )
    )

    @classmethod
    def poll(cls, context):
        return _is_view3d(context)

    def execute(self, context):
        rv3d = context.region_data or context.space_data.region_3d
        if rv3d is None:
            return {"CANCELLED"}

        if rv3d.view_perspective == "CAMERA":
            bpy.ops.view3d.view_camera()

        needs_toggle = (
            (self.mode == "PERSP" and rv3d.view_perspective != "PERSP")
            or (self.mode == "ORTHO" and rv3d.view_perspective != "ORTHO")
        )
        if needs_toggle:
            bpy.ops.view3d.view_persportho()

        return {"FINISHED"}


class BAW_OT_set_transform_tool(bpy.types.Operator):
    bl_idname = "baw.set_transform_tool"
    bl_label = "Set Transform Tool"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    mode: EnumProperty(
        items=(
            ("MOVE", "Move", "Activate the Move gizmo"),
            ("ROTATE", "Rotate", "Activate the Rotate gizmo"),
            ("SCALE", "Scale", "Activate the Scale gizmo"),
        )
    )

    @classmethod
    def poll(cls, context):
        return _is_view3d(context)

    def invoke(self, context, _event):
        return self.execute(context)

    def execute(self, context):
        scene = context.scene
        causal_root = new_trace_causal_root("transform-tool")
        trace_lifecycle_event(
            "TRANSFORM_TOOL_INGRESS",
            subsystem="keymap",
            phase="ingress",
            causal=causal_root,
            route_outcome=TraceRouteOutcome.CLAIMED,
            context=context,
            requested_mode=self.mode,
        )
        def _finish(result: set[str]) -> set[str]:
            cancelled = "CANCELLED" in result
            trace_lifecycle_event(
                "TRANSFORM_TOOL_TERMINAL",
                subsystem="operator",
                phase="cancel" if cancelled else "commit",
                causal=causal_root,
                terminal_status=(
                    TraceTerminalStatus.CANCELLED
                    if cancelled
                    else TraceTerminalStatus.FINISHED
                ),
                route_outcome=(
                    TraceRouteOutcome.REJECTED
                    if cancelled
                    else TraceRouteOutcome.CLAIMED
                ),
                context=context,
                requested_mode=self.mode,
            )
            return result

        semantic_mode_before = str(getattr(scene, "baw_rigped_semantic_transform_mode", "NONE"))
        trace_event(
            "INPUT",
            "TRANSFORM_TOOL_REQUEST",
            causal=causal_root,
            subsystem="input",
            lifecycle_phase="ingress",
            context=context,
            requested_mode=self.mode,
            semantic_mode_before=semantic_mode_before,
            active_tool_before=_active_native_transform_tool_id(context),
        )
        if (
            context.mode in {"OBJECT", "POSE"}
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
        ):
            # Edit Path owns W/E/R while trajectory editing. Switch only the
            # AWB trajectory transform mode; never reactivate Blender's native
            # object/bone tool gizmo, even when Auto Key is enabled.
            deactivate_rigped_semantic_tool(context)
            scene.baw_trajectory_transform_mode = self.mode
            context.space_data.show_gizmo = True
            if hasattr(context.space_data, "show_gizmo_tool"):
                context.space_data.show_gizmo_tool = False
            context.area.tag_redraw()
            return _finish({"FINISHED"})

        fit_host_present = fit_ui_state_present(context)
        fit_state = fit_ui_state(context)
        if fit_host_present and context.mode == "OBJECT":
            session = fit_semantic_session(context)
            if fit_state is None or session is None:
                deactivate_rigped_semantic_tool(context)
                context.space_data.show_gizmo = True
                _hide_native_tool_gizmo(context)
                try:
                    bpy.ops.wm.tool_set_by_id(name="baw.select_object")
                except RuntimeError:
                    pass
                trace_event(
                    "INPUT",
                    "FIT_F3_TRANSFORM_REFUSED",
                    context=context,
                    requested_mode=self.mode,
                    issues=("FIT_F3_SESSION_MISSING",),
                )
                return _finish({"CANCELLED"})
            issues = validate_fit_semantic_session(context, session)
            if issues:
                set_fit_transform_mode(context, "NONE")
                trace_event(
                    "INPUT",
                    "FIT_F3_TRANSFORM_REFUSED",
                    context=context,
                    requested_mode=self.mode,
                    issues=issues,
                )
                return _finish({"CANCELLED"})
            fit_mode_before = fit_transform_mode(context)
            deactivate_rigped_semantic_tool(context)
            context.space_data.show_gizmo = True
            _hide_native_tool_gizmo(context)
            if self.mode in {"MOVE", "ROTATE", "SCALE"}:
                if not _activate_awb_transform_workspace_tool(context, self.mode):
                    set_fit_transform_mode(context, "NONE")
                    return _finish({"CANCELLED"})
                set_fit_transform_mode(context, self.mode)
                if fit_mode_before == self.mode:
                    current = fit_orientation_mode(context)
                    set_fit_orientation_mode(
                        context,
                        "WORLD" if current == "LOCAL" else "LOCAL",
                    )
                trace_event(
                    "INPUT",
                    "FIT_F3_TRANSFORM_TOOL_ACTIVE",
                    context=context,
                    requested_mode=self.mode,
                    orientation=fit_orientation_mode(context),
                )
            else:
                set_fit_transform_mode(context, "NONE")
                try:
                    bpy.ops.wm.tool_set_by_id(name="baw.select_object")
                except RuntimeError:
                    pass
                trace_event(
                    "INPUT",
                    "FIT_F3_TRANSFORM_NOT_IMPLEMENTED",
                    context=context,
                    requested_mode=self.mode,
                )
            return _finish({"FINISHED"})

        if fit_host_present:
            # Figure is Object-hosted. Any externally forced non-Object mode is
            # stale/unsupported while the semantic session is alive. Consume
            # W/E/R here so native EditBone/Pose/Object transforms cannot become
            # a fallback authoring path.
            set_fit_transform_mode(context, "NONE")
            deactivate_rigped_semantic_tool(context)
            context.space_data.show_gizmo = True
            _hide_native_tool_gizmo(context)
            trace_event(
                "INPUT",
                "FIT_F5_NON_OBJECT_TRANSFORM_BLOCKED",
                context=context,
                requested_mode=self.mode,
                host_mode=str(context.mode),
            )
            return _finish({"FINISHED"})

        rigped_decision = _rigped_transform_decision(context, self.mode)
        trace_event(
            "INPUT",
            "TRANSFORM_TOOL_ROUTE",
            context=context,
            requested_mode=self.mode,
            decision_type=type(rigped_decision).__name__,
            decision_route=str(getattr(rigped_decision, "route", rigped_decision)),
        )
        # Rotate owns mixed Rigped selections directly. The older I16 contract
        # can refuse selections that combine Contact-limb controls with direct
        # FK controls even though the Direct Rotate modal now partitions those
        # domains and authors both writers in one gesture.
        if self.mode == "ROTATE" and direct_rotate_available(context):
            direct_mode = "DIRECT_ROTATE"
            if semantic_mode_before == direct_mode:
                next_orientation = _cycle_transform_orientation_preserving_semantic_mode(
                    context,
                    direct_mode,
                )
                trace_event(
                    "INPUT",
                    "TRANSFORM_TOOL_ACTIVATED",
                    context=context,
                    requested_mode=self.mode,
                    semantic_mode=direct_mode,
                    cycled_orientation=True,
                    orientation_after=next_orientation,
                    route_override="MIXED_ROTATE",
                )
                return _finish({"FINISHED"})
            scene.baw_rigped_semantic_transform_mode = direct_mode
            context.space_data.show_gizmo = True
            if not _activate_awb_transform_workspace_tool(context, self.mode):
                trace_event(
                    "ERROR",
                    "TRANSFORM_TOOL_ACTIVATE_FAIL",
                    context=context,
                    requested_mode=self.mode,
                    semantic_mode=direct_mode,
                    route_override="MIXED_ROTATE",
                )
                scene.baw_rigped_semantic_transform_mode = "NONE"
                return _finish({"CANCELLED"})
            _hide_native_tool_gizmo(context)
            trace_event(
                "INPUT",
                "TRANSFORM_TOOL_ACTIVATED",
                context=context,
                requested_mode=self.mode,
                semantic_mode=direct_mode,
                cycled_orientation=False,
                route_override="MIXED_ROTATE",
            )
            return _finish({"FINISHED"})
        if isinstance(rigped_decision, str):
            deactivate_rigped_semantic_tool(context)
            _hide_native_tool_gizmo(context)
            self.report({"WARNING"}, text("rigped.transform.refused", context))
            return _finish({"FINISHED"})
        if rigped_decision is not None:
            if self.mode == "MOVE":
                if semantic_mode_before == "FK_MOVE" and activate_rigped_fk_joint_move_tool(context):
                    if not _activate_awb_transform_workspace_tool(context, "MOVE"):
                        deactivate_rigped_semantic_tool(context)
                        return _finish({"CANCELLED"})
                    _cycle_transform_orientation_preserving_semantic_mode(
                        context,
                        "FK_MOVE",
                    )
                    return _finish({"FINISHED"})
                if activate_rigped_fk_joint_move_tool(context):
                    if not _activate_awb_transform_workspace_tool(context, "MOVE"):
                        deactivate_rigped_semantic_tool(context)
                        return _finish({"CANCELLED"})
                    return _finish({"FINISHED"})
                # Multi-selection Sliding is resolved by the semantic Move
                # operator itself. The older I16 transform contract refuses
                # semantic multi-selection before that operator gets a chance
                # to prove the selected independent Sliding limb domains.
                if semantic_mode_before == "MOVE" and activate_rigped_semantic_move_tool(context):
                    if not _activate_awb_transform_workspace_tool(context, "MOVE"):
                        deactivate_rigped_semantic_tool(context)
                        return _finish({"CANCELLED"})
                    _cycle_transform_orientation_preserving_semantic_mode(
                        context,
                        "MOVE",
                    )
                    return _finish({"FINISHED"})
                if activate_rigped_semantic_move_tool(context):
                    if not _activate_awb_transform_workspace_tool(context, "MOVE"):
                        deactivate_rigped_semantic_tool(context)
                        return _finish({"CANCELLED"})
                    return _finish({"FINISHED"})
            if rigped_decision.route == RigpedTransformRoute.REFUSE:
                deactivate_rigped_semantic_tool(context)
                _hide_native_tool_gizmo(context)
                self.report({"WARNING"}, text("rigped.transform.refused", context))
                return _finish({"FINISHED"})
            if rigped_decision.route in {
                RigpedTransformRoute.SEMANTIC_KINEMATIC,
                RigpedTransformRoute.SEMANTIC_CONTACT,
            }:
                if self.mode == "MOVE":
                    if semantic_mode_before == "MOVE" and activate_rigped_semantic_move_tool(context):
                        if not _activate_awb_transform_workspace_tool(context, "MOVE"):
                            deactivate_rigped_semantic_tool(context)
                            return _finish({"CANCELLED"})
                        _cycle_transform_orientation_preserving_semantic_mode(
                            context,
                            "MOVE",
                        )
                        return _finish({"FINISHED"})
                    if activate_rigped_semantic_move_tool(context):
                        if not _activate_awb_transform_workspace_tool(context, "MOVE"):
                            deactivate_rigped_semantic_tool(context)
                            return _finish({"CANCELLED"})
                        return _finish({"FINISHED"})
                deactivate_rigped_semantic_tool(context)
                _hide_native_tool_gizmo(context)
                self.report({"WARNING"}, text("rigped.transform.semantic_move", context))
                return _finish({"FINISHED"})
            if rigped_decision.route == RigpedTransformRoute.NATIVE and self.mode in {"MOVE", "ROTATE"}:
                direct_mode = "DIRECT_MOVE" if self.mode == "MOVE" else "DIRECT_ROTATE"
                if semantic_mode_before == direct_mode:
                    next_orientation = _cycle_transform_orientation_preserving_semantic_mode(
                        context,
                        direct_mode,
                    )
                    trace_event(
                        "INPUT",
                        "TRANSFORM_TOOL_ACTIVATED",
                        context=context,
                        requested_mode=self.mode,
                        semantic_mode=direct_mode,
                        cycled_orientation=True,
                        orientation_after=next_orientation,
                    )
                    return _finish({"FINISHED"})
                scene.baw_rigped_semantic_transform_mode = direct_mode
                context.space_data.show_gizmo = True
                if not _activate_awb_transform_workspace_tool(context, self.mode):
                    trace_event(
                        "ERROR",
                        "TRANSFORM_TOOL_ACTIVATE_FAIL",
                        context=context,
                        requested_mode=self.mode,
                        semantic_mode=direct_mode,
                    )
                    scene.baw_rigped_semantic_transform_mode = "NONE"
                    return _finish({"CANCELLED"})
                _hide_native_tool_gizmo(context)
                trace_event(
                    "INPUT",
                    "TRANSFORM_TOOL_ACTIVATED",
                    context=context,
                    requested_mode=self.mode,
                    semantic_mode=direct_mode,
                    cycled_orientation=False,
                )
                return _finish({"FINISHED"})

        route_causal = new_trace_causal_child(
            causal_root,
            "transform-tool-route",
        )
        trace_lifecycle_event(
            "TRANSFORM_TOOL_GENERIC_ROUTE",
            subsystem="operator",
            phase="routing",
            causal=route_causal,
            route_outcome=TraceRouteOutcome.CLAIMED,
            context=context,
            requested_mode=self.mode,
            route="GENERIC_WORKSPACE_TOOL",
        )
        tool_id = _awb_transform_workspace_tool_id(context, self.mode)
        active_tool_before = _active_native_transform_tool_id(context)
        deactivate_rigped_semantic_tool(context)
        context.space_data.show_gizmo = True
        if hasattr(context.space_data, "show_gizmo_tool"):
            context.space_data.show_gizmo_tool = False
        if semantic_mode_before == "NONE" and active_tool_before == tool_id:
            _cycle_transform_orientation(context)
            trace_lifecycle_event(
                "TRANSFORM_TOOL_GENERIC_FINISHED",
                subsystem="operator",
                phase="commit",
                causal=route_causal,
                terminal_status=TraceTerminalStatus.FINISHED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
                requested_mode=self.mode,
                route="GENERIC_WORKSPACE_TOOL",
                cycled_orientation=True,
            )
            return {"FINISHED"}
        if not _activate_awb_transform_workspace_tool(context, self.mode):
            trace_lifecycle_event(
                "TRANSFORM_TOOL_GENERIC_CANCELLED",
                subsystem="operator",
                phase="cancel",
                causal=route_causal,
                terminal_status=TraceTerminalStatus.CANCELLED,
                route_outcome=TraceRouteOutcome.REJECTED,
                context=context,
                requested_mode=self.mode,
                route="GENERIC_WORKSPACE_TOOL",
            )
            return {"CANCELLED"}
        context.area.tag_redraw()
        trace_lifecycle_event(
            "TRANSFORM_TOOL_GENERIC_FINISHED",
            subsystem="operator",
            phase="commit",
            causal=route_causal,
            terminal_status=TraceTerminalStatus.FINISHED,
            route_outcome=TraceRouteOutcome.CLAIMED,
            context=context,
            requested_mode=self.mode,
            route="GENERIC_WORKSPACE_TOOL",
            cycled_orientation=False,
        )
        return {"FINISHED"}


_AWB_SELECTION_TOOL_KEYMAP = (
    (
        "baw.awb_select_click",
        {"type": "LEFTMOUSE", "value": "PRESS"},
        {"properties": [("action", "SET")]},
    ),
    (
        "baw.awb_select_click",
        {"type": "LEFTMOUSE", "value": "PRESS", "ctrl": True},
        {"properties": [("action", "ADD")]},
    ),
    (
        "baw.awb_select_click",
        {"type": "LEFTMOUSE", "value": "PRESS", "alt": True},
        {"properties": [("action", "REMOVE")]},
    ),
    (
        "baw.awb_select_click",
        {"type": "LEFTMOUSE", "value": "PRESS", "shift": True},
        {"properties": [("action", "SET")]},
    ),
)


class BAW_WST_select_object(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "OBJECT"
    bl_idname = "baw.select_object"
    bl_label = "AWB Select"
    bl_description = "AWB front-most viewport selection"
    bl_icon = "ops.generic.select"
    bl_widget = None
    bl_cursor = "DEFAULT"
    bl_keymap = _AWB_SELECTION_TOOL_KEYMAP


class BAW_WST_select_pose(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "POSE"
    bl_idname = "baw.select_pose"
    bl_label = "AWB Select"
    bl_description = "AWB front-most viewport selection"
    bl_icon = "ops.generic.select"
    bl_widget = None
    bl_cursor = "DEFAULT"
    bl_keymap = _AWB_SELECTION_TOOL_KEYMAP


class BAW_WST_select_edit_armature(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "EDIT_ARMATURE"
    bl_idname = "baw.select_edit_armature"
    bl_label = "AWB Select"
    bl_description = "AWB front-most viewport selection"
    bl_icon = "ops.generic.select"
    bl_widget = None
    bl_cursor = "CROSSHAIR"
    bl_keymap = _AWB_SELECTION_TOOL_KEYMAP


class BAW_WST_move_object(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "OBJECT"
    bl_idname = "baw.move_object"
    bl_label = "AWB Move"
    bl_description = "AWB Move transform shell"
    bl_icon = "ops.transform.translate"
    bl_widget = None
    bl_cursor = "DEFAULT"
    bl_keymap = _AWB_SELECTION_TOOL_KEYMAP


class BAW_WST_rotate_object(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "OBJECT"
    bl_idname = "baw.rotate_object"
    bl_label = "AWB Rotate"
    bl_description = "AWB Rotate transform shell"
    bl_icon = "ops.transform.rotate"
    bl_widget = None
    bl_cursor = "DEFAULT"
    bl_keymap = _AWB_SELECTION_TOOL_KEYMAP


class BAW_WST_scale_object(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "OBJECT"
    bl_idname = "baw.scale_object"
    bl_label = "AWB Scale"
    bl_description = "AWB Scale transform shell"
    bl_icon = "ops.transform.resize"
    bl_widget = None
    bl_cursor = "DEFAULT"
    bl_keymap = _AWB_SELECTION_TOOL_KEYMAP


class BAW_WST_move_pose(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "POSE"
    bl_idname = "baw.move_pose"
    bl_label = "AWB Move"
    bl_description = "AWB Move transform shell"
    bl_icon = "ops.transform.translate"
    bl_widget = None
    bl_cursor = "DEFAULT"
    bl_keymap = _AWB_SELECTION_TOOL_KEYMAP


class BAW_WST_rotate_pose(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "POSE"
    bl_idname = "baw.rotate_pose"
    bl_label = "AWB Rotate"
    bl_description = "AWB Rotate transform shell"
    bl_icon = "ops.transform.rotate"
    bl_widget = None
    bl_cursor = "DEFAULT"
    bl_keymap = _AWB_SELECTION_TOOL_KEYMAP


class BAW_WST_scale_pose(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "POSE"
    bl_idname = "baw.scale_pose"
    bl_label = "AWB Scale"
    bl_description = "AWB Scale transform shell"
    bl_icon = "ops.transform.resize"
    bl_widget = None
    bl_cursor = "DEFAULT"
    bl_keymap = _AWB_SELECTION_TOOL_KEYMAP


_AWB_SELECTION_TOOLS = (
    BAW_WST_select_object,
    BAW_WST_select_pose,
    BAW_WST_select_edit_armature,
    BAW_WST_move_object,
    BAW_WST_rotate_object,
    BAW_WST_scale_object,
    BAW_WST_move_pose,
    BAW_WST_rotate_pose,
    BAW_WST_scale_pose,
)
_AWB_SELECTION_TOOLS_REGISTERED = False


def _register_awb_selection_tools() -> None:
    global _AWB_SELECTION_TOOLS_REGISTERED
    if _AWB_SELECTION_TOOLS_REGISTERED:
        return
    for tool_cls in _AWB_SELECTION_TOOLS:
        bpy.utils.register_tool(tool_cls, after={"builtin.select"})
    _AWB_SELECTION_TOOLS_REGISTERED = True


def _unregister_awb_selection_tools() -> None:
    global _AWB_SELECTION_TOOLS_REGISTERED
    if not _AWB_SELECTION_TOOLS_REGISTERED:
        return
    for tool_cls in reversed(_AWB_SELECTION_TOOLS):
        bpy.utils.unregister_tool(tool_cls)
    _AWB_SELECTION_TOOLS_REGISTERED = False


class BAW_OT_guard_figure_native_edit(bpy.types.Operator):
    """Keep an active Figure session from falling through to native edit transforms."""

    bl_idname = "baw.guard_figure_native_edit"
    bl_label = "Guard Figure Native Edit"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    action: EnumProperty(
        items=(
            ("BLOCK", "Block", "Consume a native transform while Figure owns authoring"),
            ("RESTORE_OBJECT", "Restore Object", "Return a forced non-Object Figure host to Object Mode"),
        ),
        default="BLOCK",
    )

    @classmethod
    def poll(cls, context):
        return fit_ui_state_present(context)

    def execute(self, context):
        host_mode = str(getattr(context, "mode", ""))
        if self.action == "RESTORE_OBJECT" and host_mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                trace_event(
                    "ERROR",
                    "FIT_F5_OBJECT_HOST_RESTORE_FAILED",
                    context=context,
                    host_mode=host_mode,
                )
                return {"FINISHED"}
            try:
                bpy.ops.wm.tool_set_by_id(name="baw.select_object")
            except RuntimeError:
                pass
            set_fit_transform_mode(context, "NONE")
        trace_event(
            "INPUT",
            "FIT_F5_NATIVE_EDIT_BLOCKED",
            context=context,
            action=self.action,
            host_mode=host_mode,
        )
        return {"FINISHED"}


class BAW_OT_block_native_select_click(bpy.types.Operator):
    """Consume Blender's trailing CLICK select after AWB handled LEFTMOUSE PRESS."""

    bl_idname = "baw.block_native_select_click"
    bl_label = "Block Native Select Click"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return bool(
            _is_view3d(context)
            and getattr(context, "region", None) is not None
            and context.region.type == "WINDOW"
            and getattr(context, "mode", "") in {"OBJECT", "POSE", "EDIT_ARMATURE"}
        )

    def execute(self, _context):
        return {"FINISHED"}


class BAW_OT_awb_select_click(bpy.types.Operator):
    """Apply the frozen AWB scene-selection modifier grammar in the 3D View."""

    bl_idname = "baw.awb_select_click"
    bl_label = "AWB Select"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    action: EnumProperty(
        items=(
            ("SET", "Set", "Replace the current selection"),
            ("ADD", "Add", "Add the clicked item to the current selection"),
            ("REMOVE", "Remove", "Remove the clicked item from the current selection"),
        )
    )

    @classmethod
    def poll(cls, context):
        return bool(
            _is_view3d(context)
            and getattr(context, "region", None) is not None
            and context.region.type == "WINDOW"
            and getattr(context, "mode", "") in {"OBJECT", "POSE", "EDIT_ARMATURE"}
        )

    def invoke(self, context, event):
        self._causal_context = None
        if event.value == "PRESS":
            self._causal_context = new_trace_causal_root("selection")
            trace_lifecycle_event(
                "SELECTION_CLICK_INGRESS",
                subsystem="input",
                phase="ingress",
                causal=self._causal_context,
                context=context,
                operator=self.bl_idname,
                action=self.action,
                event_type=str(event.type),
                event_value=str(event.value),
            )
        active_tool_before = _active_native_transform_tool_id(context)
        if str(getattr(context, "mode", "")) == "OBJECT" and event.value == "PRESS":
            trace_event(
                "INPUT",
                "SELECTION_CLICK_BEGIN",
                causal=self._causal_context,
                subsystem="input",
                lifecycle_phase="ingress",
                context=context,
                operator=self.bl_idname,
                action=self.action,
                event_type=str(event.type),
                event_value=str(event.value),
                location=(int(event.mouse_region_x), int(event.mouse_region_y)),
                window_location=(int(event.mouse_x), int(event.mouse_y)),
                active_tool_before=active_tool_before,
            )

        if active_tool_before in {
            "baw.select_object",
            "baw.select_pose",
            "baw.select_edit_armature",
        }:
            deactivate_rigped_semantic_tool(context)

        if event.value != "PRESS":
            return {"CANCELLED"}

        self._start_window = (
            int(event.mouse_x),
            int(event.mouse_y),
        )
        self._start_region = (
            int(event.mouse_region_x),
            int(event.mouse_region_y),
        )
        if (
            bool(getattr(context.scene, "baw_trackbar_enabled", True))
            and float(self._start_region[1]) <= float(TRACKBAR_INTERACTION_HEIGHT)
        ):
            trace_lifecycle_event(
                "SELECTION_CLICK_ROUTE",
                subsystem="keymap",
                phase="routing",
                causal=self._causal_context,
                route_outcome=TraceRouteOutcome.NATIVE_FALLTHROUGH,
                context=context,
                reason="TRACKBAR_REGION",
            )
            return {"PASS_THROUGH"}

        self._last_region = self._start_region
        self._dragging = False
        self._cursor_changed = False
        context.window_manager.modal_handler_add(self)
        trace_lifecycle_event(
            "SELECTION_MODAL_STARTED",
            subsystem="operator",
            phase="invoke",
            causal=self._causal_context,
            route_outcome=TraceRouteOutcome.CLAIMED,
            context=context,
            operator=self.bl_idname,
            action=self.action,
        )
        return {"RUNNING_MODAL"}

    def _finish_cursor(self, context) -> None:
        if self._cursor_changed and context.window is not None:
            context.window.cursor_modal_restore()
        self._cursor_changed = False

    def _trace_modal_result(self, context, result) -> None:
        causal = getattr(self, "_causal_context", None)
        if causal is None:
            return
        if "FINISHED" in result:
            trace_lifecycle_event(
                "SELECTION_MODAL_FINISHED",
                subsystem="modal",
                phase="commit",
                causal=causal,
                terminal_status=TraceTerminalStatus.FINISHED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
            )
            self._causal_context = None
        elif "CANCELLED" in result:
            trace_lifecycle_event(
                "SELECTION_MODAL_CANCELLED",
                subsystem="modal",
                phase="cancel",
                causal=causal,
                terminal_status=TraceTerminalStatus.CANCELLED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
            )
            self._causal_context = None
        elif "PASS_THROUGH" in result:
            trace_lifecycle_event(
                "SELECTION_MODAL_ROUTE",
                subsystem="modal",
                phase="routing",
                causal=causal,
                route_outcome=TraceRouteOutcome.NATIVE_FALLTHROUGH,
                context=context,
            )

    def _clear_on_miss(self, context, mode: str) -> None:
        if self.action != "SET":
            return
        if mode == "POSE":
            rig = getattr(context, "active_object", None)
            if rig is not None and getattr(rig, "type", None) == "ARMATURE":
                for pose_bone in rig.pose.bones:
                    pose_bone.select = False
                rig.data.bones.active = None
        elif mode == "OBJECT":
            for obj in tuple(getattr(context, "selected_objects", ()) or ()):
                obj.select_set(False)
            context.view_layer.objects.active = None

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            self._last_region = (int(event.mouse_region_x), int(event.mouse_region_y))
            if not self._dragging:
                distance = hypot(
                    self._last_region[0] - self._start_region[0],
                    self._last_region[1] - self._start_region[1],
                )
                if distance >= 6.0:
                    self._dragging = True
                    trace_lifecycle_event(
                        "SELECTION_DRAG_BEGIN",
                        subsystem="modal",
                        phase="modal_tick",
                        causal=getattr(self, "_causal_context", None),
                        route_outcome=TraceRouteOutcome.CLAIMED,
                        context=context,
                        transition="CLICK_TO_BOX",
                    )
                    if context.window is not None and str(getattr(context, "mode", "")) != "EDIT_ARMATURE":
                        context.window.cursor_modal_set("CROSSHAIR")
                        self._cursor_changed = True
            if self._dragging:
                _set_viewport_box_overlay(context, self._start_region, self._last_region)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            try:
                if self._dragging:
                    _reset_max_pick_cycle()
                    result = _apply_box_pick(
                        context,
                        self._start_region,
                        self._last_region,
                        self.action,
                    )
                else:
                    location = (int(event.mouse_region_x), int(event.mouse_region_y))
                    cycle_location = (int(event.mouse_x), int(event.mouse_y))
                    result = apply_awb_click_selection(
                        context,
                        location,
                        cycle_location,
                        self.action,
                        source="AWB_SELECT_MODAL_RELEASE",
                    )
            finally:
                _clear_viewport_box_overlay(context)
                self._finish_cursor(context)
            if context.area is not None:
                context.area.tag_redraw()
            self._trace_modal_result(context, result)
            return result

        if event.type in {"ESC", "RIGHTMOUSE"}:
            _clear_viewport_box_overlay(context)
            self._finish_cursor(context)
            result = {"CANCELLED"}
            self._trace_modal_result(context, result)
            return result

        return {"RUNNING_MODAL"}


class BAW_OT_toggle_wireframe(bpy.types.Operator):
    bl_idname = "baw.toggle_wireframe"
    bl_label = "Toggle Wireframe"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _is_view3d(context)

    def execute(self, context):
        shading = context.space_data.shading
        shading.type = "SOLID" if shading.type == "WIREFRAME" else "WIREFRAME"
        return {"FINISHED"}


class BAW_OT_toggle_edged_faces(bpy.types.Operator):
    bl_idname = "baw.toggle_edged_faces"
    bl_label = "Toggle Edged Faces"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _is_view3d(context)

    def execute(self, context):
        overlay = context.space_data.overlay
        overlay.show_wireframes = not overlay.show_wireframes
        return {"FINISHED"}


class BAW_OT_toggle_selected_faces(bpy.types.Operator):
    bl_idname = "baw.toggle_selected_faces"
    bl_label = "Toggle Selected Face Shading"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _is_view3d(context)

    def execute(self, context):
        overlay = context.space_data.overlay
        overlay.show_faces = not overlay.show_faces
        return {"FINISHED"}


def _add_keymap_item(
    km: bpy.types.KeyMap,
    idname: str,
    event_type: str,
    *,
    alt: bool = False,
    shift: bool = False,
    ctrl: bool = False,
) -> bpy.types.KeyMapItem:
    kmi = km.keymap_items.new(
        idname,
        event_type,
        "PRESS",
        alt=alt,
        shift=shift,
        ctrl=ctrl,
        head=True,
    )
    _VIEWPORT_KEYMAP_ITEMS.append((km, kmi))
    return kmi


def _register_transform_hotkeys(km: bpy.types.KeyMap) -> None:
    move = _add_keymap_item(km, BAW_OT_set_transform_tool.bl_idname, "W")
    move.properties.mode = "MOVE"
    rotate = _add_keymap_item(km, BAW_OT_set_transform_tool.bl_idname, "E")
    rotate.properties.mode = "ROTATE"
    scale = _add_keymap_item(km, BAW_OT_set_transform_tool.bl_idname, "R")
    scale.properties.mode = "SCALE"


def _register_native_select_click_blockers(km: bpy.types.KeyMap) -> None:
    # Blender's global 3D View keymap emits a separate LEFTMOUSE CLICK after the
    # AWB PRESS/modal gesture finishes. In Object/Pose/Edit Armature that second
    # native select can cycle to hidden geometry. Consume every modifier variant
    # only while the AWB selection operator's poll domain is active.
    for ctrl, alt, shift in (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ):
        blocker = km.keymap_items.new(
            BAW_OT_block_native_select_click.bl_idname,
            "LEFTMOUSE",
            "CLICK",
            ctrl=ctrl,
            alt=alt,
            shift=shift,
            head=True,
        )
        _VIEWPORT_KEYMAP_ITEMS.append((km, blocker))


def _register_selection_hotkeys(km: bpy.types.KeyMap) -> None:
    replace = _add_keymap_item(km, BAW_OT_awb_select_click.bl_idname, "LEFTMOUSE")
    replace.properties.action = "SET"
    add = _add_keymap_item(km, BAW_OT_awb_select_click.bl_idname, "LEFTMOUSE", ctrl=True)
    add.properties.action = "ADD"
    remove = _add_keymap_item(km, BAW_OT_awb_select_click.bl_idname, "LEFTMOUSE", alt=True)
    remove.properties.action = "REMOVE"
    # Blender normally treats Shift+Click as selection extension. AWB explicitly
    # reserves multi-select for Ctrl, so Shift alone must not silently retain the
    # native extend-selection behavior. Treat it as ordinary replacement instead.
    shift_replace = _add_keymap_item(km, BAW_OT_awb_select_click.bl_idname, "LEFTMOUSE", shift=True)
    shift_replace.properties.action = "SET"
    _register_native_select_click_blockers(km)


def register_viewport_keymaps() -> None:
    global _PREVIOUS_ROTATE_AROUND_ACTIVE, _PREVIOUS_FACE_SELECT_COLOR

    unregister_viewport_keymaps()
    _register_awb_selection_tools()
    _register_viewport_box_overlay()

    wm = bpy.context.window_manager
    if wm is None or wm.keyconfigs.addon is None:
        return

    inputs = bpy.context.preferences.inputs
    if hasattr(inputs, "use_rotate_around_active"):
        _PREVIOUS_ROTATE_AROUND_ACTIVE = bool(inputs.use_rotate_around_active)
        inputs.use_rotate_around_active = True

    themes = bpy.context.preferences.themes
    if themes:
        face_select = themes[0].view_3d.face_select
        _PREVIOUS_FACE_SELECT_COLOR = tuple(float(value) for value in face_select)
        alpha = max(0.35, _PREVIOUS_FACE_SELECT_COLOR[3])
        themes[0].view_3d.face_select = (0.92, 0.08, 0.08, alpha)

    # Never mutate keyconfigs.active directly. Blender rebuilds that merged
    # keyconfig when workspace tools change, and custom items there can be freed
    # during WM_keyconfig_update_ex. Register only through the add-on keyconfig.
    kc = wm.keyconfigs.addon
    km = kc.keymaps.get("3D View")
    if km is None:
        km = kc.keymaps.new(name="3D View", space_type="VIEW_3D", region_type="WINDOW")

    # Point/drag selection is owned by the active AWB Workspace Tool in
    # Selection/Animate/Fit. Keep this global map for navigation/display only.
    # Max-style viewport navigation.
    _add_keymap_item(km, "view3d.move", "MIDDLEMOUSE")
    _add_keymap_item(km, "view3d.rotate", "MIDDLEMOUSE", alt=True)

    top = _add_keymap_item(km, "view3d.view_axis", "T")
    top.properties.type = "TOP"
    front = _add_keymap_item(km, "view3d.view_axis", "F")
    front.properties.type = "FRONT"
    left = _add_keymap_item(km, "view3d.view_axis", "L")
    left.properties.type = "LEFT"

    persp = _add_keymap_item(km, BAW_OT_set_view_projection.bl_idname, "P")
    persp.properties.mode = "PERSP"
    ortho = _add_keymap_item(km, BAW_OT_set_view_projection.bl_idname, "U")
    ortho.properties.mode = "ORTHO"
    # Camera remains the fallback C action. Contact registers itself at the head
    # of this same 3D View keymap from phase4_contact_ui so there is only one
    # Contact binding and no Pose/3D-View duplicate dispatch.
    _add_keymap_item(km, "view3d.view_camera", "C")

    # Max-style viewport display toggles.
    _add_keymap_item(km, "view3d.toggle_xray", "X", alt=True)
    _add_keymap_item(km, BAW_OT_toggle_selected_faces.bl_idname, "F2")
    _add_keymap_item(km, BAW_OT_toggle_wireframe.bl_idname, "F3")
    _add_keymap_item(km, BAW_OT_toggle_edged_faces.bl_idname, "F4")

    # View management and focus.
    _add_keymap_item(km, "screen.screen_full_area", "W", alt=True)
    _add_keymap_item(km, "view3d.view_selected", "Z")

    # Max-style transform hotkeys: activate persistent gizmo tools.
    _register_transform_hotkeys(km)

    # R conflicts strongly with Blender's default transform bindings, so put
    # W/E/R at the head of the mode-specific maps as well.
    for name in ("Object Mode", "Pose", "Armature"):
        mode_km = kc.keymaps.get(name)
        if mode_km is None:
            mode_km = kc.keymaps.new(name=name, space_type="EMPTY")
        _register_transform_hotkeys(mode_km)
        guard_tab = _add_keymap_item(
            mode_km,
            BAW_OT_guard_figure_native_edit.bl_idname,
            "TAB",
        )
        guard_tab.properties.action = "RESTORE_OBJECT"
        if name in {"Object Mode", "Pose", "Armature"}:
            for event_type in ("G", "S"):
                guard_transform = _add_keymap_item(
                    mode_km,
                    BAW_OT_guard_figure_native_edit.bl_idname,
                    event_type,
                )
                guard_transform.properties.action = "BLOCK"


def unregister_viewport_keymaps() -> None:
    global _PREVIOUS_ROTATE_AROUND_ACTIVE, _PREVIOUS_FACE_SELECT_COLOR

    _unregister_viewport_box_overlay()
    for km, kmi in reversed(_VIEWPORT_KEYMAP_ITEMS):
        if kmi.id != -1:
            km.keymap_items.remove(kmi)
    _VIEWPORT_KEYMAP_ITEMS.clear()
    _unregister_awb_selection_tools()

    if _PREVIOUS_ROTATE_AROUND_ACTIVE is not None:
        inputs = bpy.context.preferences.inputs
        if hasattr(inputs, "use_rotate_around_active"):
            inputs.use_rotate_around_active = _PREVIOUS_ROTATE_AROUND_ACTIVE
        _PREVIOUS_ROTATE_AROUND_ACTIVE = None

    if _PREVIOUS_FACE_SELECT_COLOR is not None:
        themes = bpy.context.preferences.themes
        if themes:
            themes[0].view_3d.face_select = _PREVIOUS_FACE_SELECT_COLOR
        _PREVIOUS_FACE_SELECT_COLOR = None
