from __future__ import annotations

from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d
from mathutils import Vector
from mathutils.geometry import intersect_line_line, intersect_line_plane


def axis_point(
    context,
    pivot: Vector,
    axis: Vector,
    mouse_x: float,
    mouse_y: float,
) -> Vector | None:
    if context.region is None or context.region_data is None:
        return None
    mouse = Vector((float(mouse_x), float(mouse_y)))
    ray_origin = region_2d_to_origin_3d(context.region, context.region_data, mouse)
    ray_direction = region_2d_to_vector_3d(context.region, context.region_data, mouse)
    if ray_direction.length <= 1e-9:
        return None
    ray_direction.normalize()
    intersections = intersect_line_line(
        pivot - axis * 100000.0,
        pivot + axis * 100000.0,
        ray_origin,
        ray_origin + ray_direction * 100000.0,
    )
    if intersections is None:
        return None
    return Vector(intersections[0])


def plane_point(
    context,
    pivot: Vector,
    normal: Vector,
    mouse_x: float,
    mouse_y: float,
) -> Vector | None:
    if context.region is None or context.region_data is None:
        return None
    mouse = Vector((float(mouse_x), float(mouse_y)))
    ray_origin = region_2d_to_origin_3d(context.region, context.region_data, mouse)
    ray_direction = region_2d_to_vector_3d(context.region, context.region_data, mouse)
    if ray_direction.length <= 1e-9:
        return None
    ray_direction.normalize()
    point = intersect_line_plane(
        ray_origin,
        ray_origin + ray_direction * 100000.0,
        pivot,
        normal,
        False,
    )
    return Vector(point) if point is not None else None


def rotation_vector(
    context,
    pivot: Vector,
    axis: Vector,
    mouse_x: float,
    mouse_y: float,
) -> Vector | None:
    point = plane_point(context, pivot, axis, mouse_x, mouse_y)
    if point is None:
        return None
    radial = Vector(point) - pivot
    radial -= axis * radial.dot(axis)
    if radial.length <= 1e-7:
        return None
    return radial.normalized()
