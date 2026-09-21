from __future__ import annotations

import bpy
import gpu
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

_BOX_WIRE_PROPERTY = "awb_biped_box_wire_v1"
_DRAW_HANDLE_VIEW = None
_DRAW_HANDLE_PIXEL = None
_SHADER = None
_PIXEL_SHADER = None
_NATIVE_BONE_OVERLAY_STATES: dict[int, bool] = {}
_EDGE_DEPTH_SLICES = 2

_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _shader():
    global _SHADER
    if _SHADER is None:
        _SHADER = gpu.shader.from_builtin("POLYLINE_FLAT_COLOR")
    return _SHADER


def show_native_bone_overlays_for_fit() -> None:
    for space in _iter_view3d_spaces():
        overlay = getattr(space, "overlay", None)
        if overlay is not None and hasattr(overlay, "show_bones"):
            overlay.show_bones = True
    _tag_all_view3d_redraw()


def rigped_box_display_enabled(rig) -> bool:
    return bool(
        rig is not None
        and getattr(rig, "type", None) == "ARMATURE"
        and hasattr(rig, "get")
        and rig.get(_BOX_WIRE_PROPERTY, False)
    )


def _space_pointer(space) -> int | None:
    try:
        return int(space.as_pointer())
    except (AttributeError, ReferenceError):
        return None


def _iter_view3d_spaces():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return ()
    spaces = []
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            if getattr(space, "type", None) == "VIEW_3D":
                spaces.append(space)
    return tuple(spaces)


def _suppress_native_bones_in_space(space) -> None:
    overlay = getattr(space, "overlay", None)
    pointer = _space_pointer(space)
    if overlay is None or pointer is None or not hasattr(overlay, "show_bones"):
        return
    if pointer not in _NATIVE_BONE_OVERLAY_STATES:
        _NATIVE_BONE_OVERLAY_STATES[pointer] = bool(overlay.show_bones)
    overlay.show_bones = False


def _restore_native_bone_overlays(*, ensure_visible: bool = False) -> None:
    for space in _iter_view3d_spaces():
        pointer = _space_pointer(space)
        overlay = getattr(space, "overlay", None)
        if pointer is None or overlay is None or not hasattr(overlay, "show_bones"):
            continue
        if ensure_visible:
            overlay.show_bones = True
        elif pointer in _NATIVE_BONE_OVERLAY_STATES:
            overlay.show_bones = _NATIVE_BONE_OVERLAY_STATES[pointer]
    _NATIVE_BONE_OVERLAY_STATES.clear()


def _any_box_display_enabled() -> bool:
    objects = getattr(bpy.data, "objects", None)
    if objects is None:
        return False
    return any(rigped_box_display_enabled(obj) for obj in objects)


def sync_native_bone_overlay_visibility() -> None:
    if _any_box_display_enabled():
        for space in _iter_view3d_spaces():
            _suppress_native_bones_in_space(space)
    else:
        # Box Display OFF means the native armature is the visible Rigped
        # representation. A saved .blend may already have show_bones disabled
        # from an earlier Box session, so restoring that stale False value would
        # make the Rigped disappear entirely.
        _restore_native_bone_overlays(ensure_visible=True)
    _tag_all_view3d_redraw()


def box_display_names(rig) -> tuple[str, ...]:
    if rig is None or getattr(rig, "type", None) != "ARMATURE":
        return ()
    collection = rig.data.collections.get("Authored")
    if collection is None:
        return ()
    return tuple(bone.name for bone in collection.bones if not bone.hide)


def _edit_box(rig, name: str):
    edit_bone = rig.data.edit_bones.get(name)
    if edit_bone is None or edit_bone.hide:
        return None
    return (
        edit_bone.head.copy(),
        edit_bone.tail.copy(),
        edit_bone.x_axis.normalized(),
        edit_bone.z_axis.normalized(),
        max(float(edit_bone.bbone_x), 1e-5),
        max(float(edit_bone.bbone_z), 1e-5),
    )


def _pose_box(rig, name: str):
    pose_bone = rig.pose.bones.get(name)
    if pose_bone is None or pose_bone.bone.hide:
        return None
    matrix3 = pose_bone.matrix.to_3x3()
    x_column = Vector(matrix3.col[0])
    z_column = Vector(matrix3.col[2])
    x_scale = max(x_column.length, 1e-8)
    z_scale = max(z_column.length, 1e-8)
    return (
        pose_bone.head.copy(),
        pose_bone.tail.copy(),
        x_column.normalized(),
        z_column.normalized(),
        max(float(pose_bone.bone.bbone_x) * x_scale, 1e-5),
        max(float(pose_bone.bone.bbone_z) * z_scale, 1e-5),
    )


def box_world_vertices(rig, name: str):
    """Return the eight world-space box corners for one authored bone."""

    if rig is None or getattr(rig, "type", None) != "ARMATURE":
        return ()
    box = _edit_box(rig, name) if str(getattr(rig, "mode", "")) == "EDIT" else _pose_box(rig, name)
    if box is None:
        return ()
    head, tail, x_axis, z_axis, half_x, half_z = box
    corners = []
    for origin in (head, tail):
        corners.extend(
            (
                origin - (x_axis * half_x) - (z_axis * half_z),
                origin + (x_axis * half_x) - (z_axis * half_z),
                origin + (x_axis * half_x) + (z_axis * half_z),
                origin - (x_axis * half_x) + (z_axis * half_z),
            )
        )
    world = rig.matrix_world
    return tuple(tuple(float(value) for value in (world @ point)) for point in corners)


def _bone_color(rig, name: str) -> tuple[float, float, float, float]:
    bone = rig.data.bones.get(name)
    if bone is None:
        return (1.0, 1.0, 1.0, 1.0)
    custom = bone.color.custom
    mode = str(getattr(rig, "mode", ""))
    if mode == "OBJECT":
        value = custom.normal
    elif mode == "EDIT":
        edit_bone = rig.data.edit_bones.get(name)
        active_bone = getattr(bpy.context, "active_bone", None)
        if active_bone is not None and active_bone.name == name:
            value = custom.active
        elif edit_bone is not None and bool(edit_bone.select):
            value = custom.select
        else:
            value = custom.normal
    else:
        selected_names = {
            pose_bone.name
            for pose_bone in (getattr(bpy.context, "selected_pose_bones", None) or ())
        }
        active_pose_bone = getattr(bpy.context, "active_pose_bone", None)
        if active_pose_bone is not None and active_pose_bone.name == name:
            value = custom.active
        elif name in selected_names:
            value = custom.select
        else:
            value = custom.normal
    return (float(value[0]), float(value[1]), float(value[2]), 1.0)


def _view_depth(region_3d, point) -> float:
    return float((region_3d.view_matrix @ Vector(point)).z)


def _edge_fragments(rig, region_3d):
    fragments = []
    slices = max(1, int(_EDGE_DEPTH_SLICES))
    for name in box_display_names(rig):
        vertices = box_world_vertices(rig, name)
        if len(vertices) != 8:
            continue
        color = _bone_color(rig, name)
        for start_index, end_index in _EDGES:
            start = Vector(vertices[start_index])
            end = Vector(vertices[end_index])
            for slice_index in range(slices):
                first = start.lerp(end, slice_index / slices)
                second = start.lerp(end, (slice_index + 1) / slices)
                midpoint = (first + second) * 0.5
                fragments.append(
                    (
                        _view_depth(region_3d, midpoint),
                        tuple(first),
                        tuple(second),
                        color,
                    )
                )
    fragments.sort(key=lambda item: item[0])
    return fragments


def _draw_polyline_fragments(fragments, line_width: float) -> None:
    if not fragments:
        return
    coords = []
    colors = []
    for _depth, first, second, color in fragments:
        coords.extend((first, second))
        colors.extend((color, color))

    shader = _shader()
    batch = batch_for_shader(shader, "LINES", {"pos": coords, "color": colors})
    shader.bind()
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", max(1.0, float(line_width)))
    batch.draw(shader)


def _draw_box_wire_overlay_view() -> None:
    context = bpy.context
    if context.area is None or context.area.type != "VIEW_3D":
        return
    region_3d = getattr(context, "region_data", None)
    scene = getattr(context, "scene", None)
    if region_3d is None or scene is None:
        return

    rigs = tuple(
        obj
        for obj in scene.objects
        if rigped_box_display_enabled(obj)
        and not obj.hide_get()
        and str(getattr(obj, "mode", "")) != "EDIT"
    )
    if not rigs:
        return

    _suppress_native_bones_in_space(getattr(context, "space_data", None))
    width = float(getattr(scene, "baw_rigped_box_wire_width", 1.5))
    previous_blend = gpu.state.blend_get()
    previous_depth = gpu.state.depth_test_get()
    try:
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        fragments = []
        for rig in rigs:
            fragments.extend(_edge_fragments(rig, region_3d))
        fragments.sort(key=lambda item: item[0])
        _draw_polyline_fragments(fragments, width)
    finally:
        gpu.state.depth_test_set(previous_depth)
        gpu.state.blend_set(previous_blend)


def _convex_hull_2d(points):
    unique = sorted({(round(float(point.x), 5), round(float(point.y), 5)) for point in points})
    if len(unique) <= 2:
        return tuple(unique)

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

    return tuple(lower[:-1] + upper[:-1])


def _projected_box_hulls(rig, region, region_3d):
    polygons = []
    for name in box_display_names(rig):
        vertices = box_world_vertices(rig, name)
        if len(vertices) != 8:
            continue
        projected = []
        for vertex in vertices:
            point = location_3d_to_region_2d(region, region_3d, Vector(vertex))
            if point is not None:
                projected.append(point)
        hull = _convex_hull_2d(projected)
        if len(hull) >= 3:
            polygons.append(hull)
    return tuple(polygons)


def _rig_outline_segments(rig, region, region_3d):
    coords = []
    for polygon in _projected_box_hulls(rig, region, region_3d):
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            coords.extend((first, second))
    return tuple(coords)


def _pixel_shader():
    global _PIXEL_SHADER
    if _PIXEL_SHADER is None:
        _PIXEL_SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _PIXEL_SHADER


def _draw_selection_outline() -> None:
    context = bpy.context
    if context.area is None or context.area.type != "VIEW_3D":
        return
    if str(getattr(context, "mode", "")) != "OBJECT":
        return
    region = getattr(context, "region", None)
    region_3d = getattr(context, "region_data", None)
    if region is None or region_3d is None:
        return

    selected = tuple(
        obj
        for obj in (getattr(context, "selected_objects", None) or ())
        if rigped_box_display_enabled(obj)
    )
    if not selected:
        return

    active = getattr(getattr(context, "view_layer", None), "objects", None)
    active_object = getattr(active, "active", None) if active is not None else None
    shader = _pixel_shader()
    previous_blend = gpu.state.blend_get()
    previous_width = gpu.state.line_width_get()
    try:
        gpu.state.blend_set("ALPHA")
        gpu.state.line_width_set(2.5)
        for rig in selected:
            coords = _rig_outline_segments(rig, region, region_3d)
            if not coords:
                continue
            color = (1.0, 0.55, 0.08, 1.0) if active_object is rig else (1.0, 0.30, 0.03, 1.0)
            batch = batch_for_shader(shader, "LINES", {"pos": coords})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)
    finally:
        gpu.state.line_width_set(previous_width)
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


def register_rigped_box_wire_overlay() -> None:
    global _DRAW_HANDLE_VIEW, _DRAW_HANDLE_PIXEL
    if _DRAW_HANDLE_VIEW is None:
        _DRAW_HANDLE_VIEW = bpy.types.SpaceView3D.draw_handler_add(
            _draw_box_wire_overlay_view,
            (),
            "WINDOW",
            "POST_VIEW",
        )
    if _DRAW_HANDLE_PIXEL is None:
        _DRAW_HANDLE_PIXEL = bpy.types.SpaceView3D.draw_handler_add(
            _draw_selection_outline,
            (),
            "WINDOW",
            "POST_PIXEL",
        )
    _tag_all_view3d_redraw()


def unregister_rigped_box_wire_overlay() -> None:
    global _DRAW_HANDLE_VIEW, _DRAW_HANDLE_PIXEL, _SHADER, _PIXEL_SHADER
    if _DRAW_HANDLE_VIEW is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE_VIEW, "WINDOW")
        _DRAW_HANDLE_VIEW = None
    if _DRAW_HANDLE_PIXEL is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE_PIXEL, "WINDOW")
        _DRAW_HANDLE_PIXEL = None
    _SHADER = None
    _PIXEL_SHADER = None
    _restore_native_bone_overlays()
    _tag_all_view3d_redraw()
