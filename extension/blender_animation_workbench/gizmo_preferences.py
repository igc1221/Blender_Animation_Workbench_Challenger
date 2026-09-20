from __future__ import annotations

import json
from math import degrees, radians, sqrt
from pathlib import Path
from typing import ClassVar

import blf
import bpy
from bpy.app.handlers import persistent
from bpy.props import StringProperty
from bpy_extras.view3d_utils import (
    location_3d_to_region_2d,
    region_2d_to_location_3d,
)
from mathutils import Quaternion, Vector

from .ui_language import text

GIZMO_HIGHLIGHT_COLOR = (1.0, 0.82, 0.05)
GIZMO_AXIS_COLORS = {
    "X": (0.86, 0.16, 0.12),
    "Y": (0.20, 0.72, 0.18),
    "Z": (0.16, 0.38, 0.92),
}
_PREVIOUS_GIZMO_HIGHLIGHT: tuple[float, float, float] | None = None
_PREFERENCES_SAVE_PENDING = False
_GIZMO_PREFERENCES_CACHE_FILENAME = "awb_gizmo_preferences.json"
_ROTATION_DRAW_HANDLE = None
_ROTATION_DRAW_STATE: tuple[int, tuple[float, float, float], float, float] | None = None

# 3ds Max documents its Linear Roll rotation sensitivity as degrees generated
# per mouse pixel, with 0.5 degree/pixel as the default reference value.
MAX_LINEAR_ROTATION_RADIANS_PER_PIXEL = radians(0.5)


def projected_axis_screen_direction(context, pivot: Vector, axis: Vector) -> Vector | None:
    """Return a stable 2D drag direction for one world-space axis."""
    region = getattr(context, "region", None)
    region_data = getattr(context, "region_data", None)
    if region is None or region_data is None:
        return None
    pivot_2d = location_3d_to_region_2d(region, region_data, Vector(pivot))
    sample_2d = location_3d_to_region_2d(
        region,
        region_data,
        Vector(pivot) + Vector(axis).normalized(),
    )
    if pivot_2d is None or sample_2d is None:
        return None
    delta = Vector(sample_2d) - Vector(pivot_2d)
    if delta.length <= 1e-6:
        return None
    return delta.normalized()


def projection_world_per_pixel(context, pivot: Vector) -> float | None:
    """World distance represented by one screen pixel at the pivot depth.

    This is the projection-transform equivalent of Max's perspective-sensitive
    screen-space drag: depth changes the world scale, axis foreshortening does
    not amplify it.
    """
    region = getattr(context, "region", None)
    region_data = getattr(context, "region_data", None)
    if region is None or region_data is None:
        return None
    pivot_2d = location_3d_to_region_2d(region, region_data, Vector(pivot))
    if pivot_2d is None:
        return None
    base = region_2d_to_location_3d(region, region_data, Vector(pivot_2d), Vector(pivot))
    px = region_2d_to_location_3d(
        region,
        region_data,
        Vector(pivot_2d) + Vector((1.0, 0.0)),
        Vector(pivot),
    )
    py = region_2d_to_location_3d(
        region,
        region_data,
        Vector(pivot_2d) + Vector((0.0, 1.0)),
        Vector(pivot),
    )
    dx = float((Vector(px) - Vector(base)).length)
    dy = float((Vector(py) - Vector(base)).length)
    values = tuple(value for value in (dx, dy) if value > 1e-12)
    return (sum(values) / len(values)) if values else None


def linear_roll_screen_tangent(
    context,
    pivot: Vector,
    axis: Vector,
    mouse: Vector,
    start_radial: Vector | None = None,
) -> Vector | None:
    """Return the fixed screen tangent used by Max-style Linear Roll."""
    region = getattr(context, "region", None)
    region_data = getattr(context, "region_data", None)
    if region is None or region_data is None:
        return None
    pivot = Vector(pivot)
    axis = Vector(axis).normalized()
    pivot_2d = location_3d_to_region_2d(region, region_data, pivot)
    if pivot_2d is None:
        return None

    if start_radial is not None and Vector(start_radial).length > 1e-9:
        radial = Vector(start_radial).normalized()
        tangent_world = axis.cross(radial)
        if tangent_world.length > 1e-9:
            tangent_world.normalize()
            probe_start = location_3d_to_region_2d(
                region, region_data, pivot + radial
            )
            probe_end = location_3d_to_region_2d(
                region, region_data, pivot + radial + tangent_world * 0.05
            )
            if probe_start is not None and probe_end is not None:
                tangent = Vector(probe_end) - Vector(probe_start)
                if tangent.length > 1e-6:
                    return tangent.normalized()

    # Degenerate edge-on fallback: use the visible radial from the gizmo center.
    radial_2d = Vector(mouse) - Vector(pivot_2d)
    if radial_2d.length <= 1e-6:
        return None
    tangent = Vector((-radial_2d.y, radial_2d.x))
    if tangent.length <= 1e-6:
        return None

    # Preserve right-hand rotation direction relative to the viewport.
    basis = region_data.view_matrix.inverted().to_3x3()
    view_normal = Vector(basis.col[2]).normalized()
    if float(axis.dot(view_normal)) < 0.0:
        tangent.negate()
    return tangent.normalized()


def _arcball_vector(center: Vector, mouse: Vector, radius_px: float) -> Vector:
    """Project a screen point onto a smooth virtual trackball."""
    radius = max(float(radius_px), 8.0)
    x = float(mouse.x - center.x) / radius
    y = float(mouse.y - center.y) / radius
    d2 = x * x + y * y
    if d2 <= 0.5:
        z = sqrt(max(0.0, 1.0 - d2))
    else:
        z = 0.5 / sqrt(max(d2, 1e-12))
    vector = Vector((x, y, z))
    if vector.length <= 1e-9:
        return Vector((0.0, 0.0, 1.0))
    return vector.normalized()


def arcball_world_step(
    context,
    pivot: Vector,
    previous_mouse: Vector,
    current_mouse: Vector,
    radius_px: float,
) -> Quaternion:
    """Return one view-independent world-space Arcball rotation step.

    The drag is normalized by the gizmo's screen radius, so camera angle and
    axis foreshortening do not change free-rotate sensitivity.
    """
    region = getattr(context, "region", None)
    region_data = getattr(context, "region_data", None)
    if region is None or region_data is None:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    center = location_3d_to_region_2d(region, region_data, Vector(pivot))
    if center is None:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    previous = _arcball_vector(Vector(center), Vector(previous_mouse), radius_px)
    current = _arcball_vector(Vector(center), Vector(current_mouse), radius_px)
    q_view = previous.rotation_difference(current)
    addon_preferences = addon_gizmo_preferences(context)
    sensitivity_percent = float(
        getattr(
            addon_preferences,
            "gizmo_free_rotate_sensitivity_percent",
            100.0,
        )
    )
    sensitivity = max(0.25, min(3.0, sensitivity_percent / 100.0))
    angle = float(q_view.angle) * sensitivity
    if angle <= 1e-9:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis_view = Vector(q_view.axis)
    basis = region_data.view_matrix.inverted().to_3x3()
    axis_world = Vector(basis @ axis_view)
    if axis_world.length <= 1e-9:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis_world.normalize()
    return Quaternion(axis_world, angle)


def addon_gizmo_preferences(context):
    preferences = getattr(context, "preferences", None)
    addons = getattr(preferences, "addons", None) if preferences is not None else None
    addon = addons.get(__package__) if addons is not None else None
    return getattr(addon, "preferences", None) if addon is not None else None


def register_global_gizmo_theme(context) -> None:
    """Apply AWB's shared gizmo highlight policy."""

    global _PREVIOUS_GIZMO_HIGHLIGHT
    preferences = getattr(context, "preferences", None)
    themes = getattr(preferences, "themes", None) if preferences is not None else None
    if not themes:
        return
    user_interface = getattr(themes[0], "user_interface", None)
    if user_interface is None or not hasattr(user_interface, "gizmo_hi"):
        return
    if _PREVIOUS_GIZMO_HIGHLIGHT is None:
        _PREVIOUS_GIZMO_HIGHLIGHT = tuple(float(value) for value in user_interface.gizmo_hi)
    user_interface.gizmo_hi = GIZMO_HIGHLIGHT_COLOR


def unregister_global_gizmo_theme(context) -> None:
    global _PREVIOUS_GIZMO_HIGHLIGHT
    preferences = getattr(context, "preferences", None)
    themes = getattr(preferences, "themes", None) if preferences is not None else None
    if themes and _PREVIOUS_GIZMO_HIGHLIGHT is not None:
        user_interface = getattr(themes[0], "user_interface", None)
        if user_interface is not None and hasattr(user_interface, "gizmo_hi"):
            user_interface.gizmo_hi = _PREVIOUS_GIZMO_HIGHLIGHT
    _PREVIOUS_GIZMO_HIGHLIGHT = None


@persistent
def _global_gizmo_load_post(_dummy) -> None:
    """Reapply saved global gizmo settings after a .blend restores ToolSettings."""

    initialize_global_gizmo_preferences(bpy.context)


def register_global_gizmo_load_handler() -> None:
    handlers = bpy.app.handlers.load_post
    if _global_gizmo_load_post not in handlers:
        handlers.append(_global_gizmo_load_post)


def unregister_global_gizmo_load_handler() -> None:
    handlers = bpy.app.handlers.load_post
    if _global_gizmo_load_post in handlers:
        handlers.remove(_global_gizmo_load_post)


def gizmo_display_color(base_color, interaction_gizmo):
    if bool(getattr(interaction_gizmo, "is_highlight", False)) or bool(
        getattr(interaction_gizmo, "is_modal", False)
    ):
        return GIZMO_HIGHLIGHT_COLOR
    return base_color


def _tracked_gizmo_preference_names(addon_preferences) -> tuple[str, ...]:
    properties = getattr(getattr(addon_preferences, "bl_rna", None), "properties", ())
    names = []
    for prop in properties:
        identifier = str(getattr(prop, "identifier", ""))
        if identifier.startswith(("gizmo_", "orientation_")):
            names.append(identifier)
    return tuple(names)


def _gizmo_preferences_cache_path() -> Path | None:
    config_root = bpy.utils.user_resource("CONFIG")
    if not config_root:
        return None
    return Path(config_root) / _GIZMO_PREFERENCES_CACHE_FILENAME


def _serialize_gizmo_preference_value(value):
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return list(value)
    except TypeError:
        return None


def _write_gizmo_preferences_cache(addon_preferences) -> None:
    if addon_preferences is None:
        return
    path = _gizmo_preferences_cache_path()
    if path is None:
        return
    payload = {}
    for name in _tracked_gizmo_preference_names(addon_preferences):
        value = _serialize_gizmo_preference_value(getattr(addon_preferences, name))
        if value is not None:
            payload[name] = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def _restore_gizmo_preferences_cache(addon_preferences) -> bool:
    if addon_preferences is None:
        return False
    path = _gizmo_preferences_cache_path()
    if path is None or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    restored = False
    for name in _tracked_gizmo_preference_names(addon_preferences):
        if name not in payload or not hasattr(addon_preferences, name):
            continue
        try:
            value = payload[name]
            current = getattr(addon_preferences, name)
            if not isinstance(current, (bool, int, float, str)) and isinstance(value, list):
                value = tuple(value)
            setattr(addon_preferences, name, value)
            restored = True
        except (AttributeError, TypeError, ValueError):
            continue
    return restored


def _save_user_preferences() -> None:
    global _PREFERENCES_SAVE_PENDING
    _PREFERENCES_SAVE_PENDING = False
    if bpy.app.background:
        return
    _write_gizmo_preferences_cache(addon_gizmo_preferences(bpy.context))
    try:
        bpy.ops.wm.save_userpref()
    except RuntimeError:
        pass
    return


def schedule_user_preferences_save() -> None:
    """Debounce preference writes so sidebar sliders persist across relaunches."""

    global _PREFERENCES_SAVE_PENDING
    if bpy.app.background or _PREFERENCES_SAVE_PENDING:
        return
    _PREFERENCES_SAVE_PENDING = True
    bpy.app.timers.register(_save_user_preferences, first_interval=0.35)


def sync_global_gizmo_size(context, addon_preferences=None) -> None:
    preferences = getattr(context, "preferences", None)
    view = getattr(preferences, "view", None) if preferences is not None else None
    if view is None or not hasattr(view, "gizmo_size"):
        return
    if addon_preferences is None:
        addon_preferences = addon_gizmo_preferences(context)
    if addon_preferences is None or not hasattr(addon_preferences, "gizmo_size_px"):
        return
    view.gizmo_size = round(float(addon_preferences.gizmo_size_px))


def sync_global_rotation_step(context, addon_preferences=None) -> None:
    tool_settings = getattr(getattr(context, "scene", None), "tool_settings", None)
    if tool_settings is None or not hasattr(tool_settings, "snap_angle_increment_3d"):
        return
    if addon_preferences is None:
        addon_preferences = addon_gizmo_preferences(context)
    if addon_preferences is None or not hasattr(addon_preferences, "gizmo_rotation_step_degrees"):
        return
    tool_settings.snap_angle_increment_3d = radians(
        max(0.1, min(180.0, float(addon_preferences.gizmo_rotation_step_degrees)))
    )
    if hasattr(tool_settings, "use_snap_rotate"):
        tool_settings.use_snap_rotate = True


def snapped_rotation_angle(context, event, angle: float) -> float:
    """Always apply Blender's own 3D angle increment to AWB rotation."""

    tool_settings = getattr(getattr(context, "scene", None), "tool_settings", None)
    if tool_settings is None:
        return angle
    step = float(getattr(tool_settings, "snap_angle_increment_3d", 0.0) or 0.0)
    if step <= 1e-9:
        return angle
    return round(angle / step) * step


def _draw_rotation_angle_overlay() -> None:
    state = _ROTATION_DRAW_STATE
    if state is None:
        return
    area = getattr(bpy.context, "area", None)
    region = getattr(bpy.context, "region", None)
    region_data = getattr(bpy.context, "region_data", None)
    if area is None or region is None or region_data is None:
        return
    area_pointer, pivot_values, angle, gizmo_size = state
    if int(area.as_pointer()) != area_pointer:
        return
    screen = location_3d_to_region_2d(region, region_data, Vector(pivot_values))
    if screen is None:
        return

    offset = max(34.0, gizmo_size * 0.46)
    x = float(screen.x) + offset
    y = float(screen.y) - max(18.0, offset * 0.34)
    font_id = 0
    blf.size(font_id, 15)
    blf.color(font_id, 0.92, 0.92, 0.92, 1.0)
    blf.position(font_id, x, y, 0.0)
    blf.draw(font_id, f"{degrees(angle):.1f}°")


def show_rotation_angle(context, angle: float, pivot=None) -> None:
    """Show live rotation data beside the transform gizmo, Max-style."""

    global _ROTATION_DRAW_HANDLE, _ROTATION_DRAW_STATE
    area = getattr(context, "area", None)
    if area is None or pivot is None:
        return
    addon_preferences = addon_gizmo_preferences(context)
    if addon_preferences is not None and not bool(
        getattr(addon_preferences, "gizmo_show_rotation_angle", True)
    ):
        clear_rotation_angle(context)
        return
    gizmo_size = float(getattr(addon_preferences, "gizmo_size_px", 75.0))
    vector = Vector(pivot)
    _ROTATION_DRAW_STATE = (
        int(area.as_pointer()),
        (float(vector.x), float(vector.y), float(vector.z)),
        float(angle),
        gizmo_size,
    )
    if _ROTATION_DRAW_HANDLE is None:
        _ROTATION_DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_rotation_angle_overlay,
            (),
            "WINDOW",
            "POST_PIXEL",
        )
    area.tag_redraw()


def clear_rotation_angle(context) -> None:
    global _ROTATION_DRAW_HANDLE, _ROTATION_DRAW_STATE
    area = getattr(context, "area", None)
    _ROTATION_DRAW_STATE = None
    if _ROTATION_DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_ROTATION_DRAW_HANDLE, "WINDOW")
        except (RuntimeError, ValueError):
            pass
        _ROTATION_DRAW_HANDLE = None
    if area is not None:
        area.header_text_set(None)
        area.tag_redraw()


def initialize_global_gizmo_preferences(context) -> None:
    """Keep the global sidebar values stable across install/relaunch cycles."""

    preferences = getattr(context, "preferences", None)
    view = getattr(preferences, "view", None) if preferences is not None else None
    addon_preferences = addon_gizmo_preferences(context)
    if view is None or addon_preferences is None:
        return
    is_set = getattr(addon_preferences, "is_property_set", None)
    tracked_names = _tracked_gizmo_preference_names(addon_preferences)
    if callable(is_set) and tracked_names and not any(is_set(name) for name in tracked_names):
        _restore_gizmo_preferences_cache(addon_preferences)
    size_is_set = bool(is_set("gizmo_size_px")) if callable(is_set) else True
    if not size_is_set:
        addon_preferences.gizmo_size_px = float(view.gizmo_size)
    else:
        sync_global_gizmo_size(context, addon_preferences)

    tool_settings = getattr(getattr(context, "scene", None), "tool_settings", None)
    rotation_is_set = bool(is_set("gizmo_rotation_step_degrees")) if callable(is_set) else True
    if tool_settings is not None and hasattr(tool_settings, "snap_angle_increment_3d"):
        if not rotation_is_set:
            addon_preferences.gizmo_rotation_step_degrees = degrees(
                float(tool_settings.snap_angle_increment_3d)
            )
        else:
            sync_global_rotation_step(context, addon_preferences)


class BAW_OT_gizmo_label_help(bpy.types.Operator):
    """Tooltip target used only by gizmo setting title text."""

    bl_idname = "baw.gizmo_label_help"
    bl_label = "Gizmo Setting Help"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    tip_key: StringProperty(options={"HIDDEN"})

    @classmethod
    def description(cls, context, properties):
        return text(str(getattr(properties, "tip_key", "")), context)

    def execute(self, _context):
        return {"FINISHED"}


def draw_global_gizmo_preferences(layout, context, addon_preferences=None) -> None:
    """Draw one Blender-wide transform-gizmo UI shared by every AWB mode."""

    if addon_preferences is None:
        addon_preferences = addon_gizmo_preferences(context)

    root = layout.box()
    root.label(text=text("gizmo.title", context))
    if addon_preferences is None:
        return

    def section(property_name: str, label_key: str):
        panel = root.box()
        opened = bool(getattr(addon_preferences, property_name, True))
        header = panel.row(align=True)
        header.alignment = "LEFT"
        header.prop(
            addon_preferences,
            property_name,
            text=text(label_key, context),
            icon="TRIA_DOWN" if opened else "TRIA_RIGHT",
            emboss=False,
        )
        return panel if opened else None

    CONTROL_WIDTH_UNITS = 7.5

    def prop_row(panel, prop_name: str, label_key: str, tip_key: str, *, slider=False):
        # Labels take the remaining left side. Numeric controls keep one fixed
        # width at the right edge, so every slider has the same start/end line
        # without stretching when the panel gets wider.
        row = panel.row(align=False)

        label_cell = row.row(align=True)
        label_cell.alignment = "LEFT"
        label = label_cell.operator(
            BAW_OT_gizmo_label_help.bl_idname,
            text=text(label_key, context),
            emboss=False,
        )
        label.tip_key = tip_key

        control_cell = row.row(align=False)
        control_cell.alignment = "RIGHT"
        control_cell.ui_units_x = CONTROL_WIDTH_UNITS
        control_cell.prop(
            addon_preferences,
            prop_name,
            text="",
            slider=slider,
        )

    def toggle_row(panel, prop_name: str, label_key: str, tip_key: str):
        row = panel.row(align=True)
        row.alignment = "LEFT"
        label = row.operator(
            BAW_OT_gizmo_label_help.bl_idname,
            text=text(label_key, context),
            emboss=False,
        )
        label.tip_key = tip_key
        toggle = row.row(align=True)
        toggle.alignment = "LEFT"
        toggle.scale_x = 1.1
        toggle.scale_y = 1.1
        toggle.prop(addon_preferences, prop_name, text="")

    def color_row(panel, colors, tip_key: str):
        row = panel.row(align=True)
        row.alignment = "LEFT"
        for label_text, prop_name in colors:
            pair = row.row(align=True)
            pair.alignment = "LEFT"
            label = pair.operator(
                BAW_OT_gizmo_label_help.bl_idname,
                text=label_text,
                emboss=False,
            )
            label.tip_key = tip_key
            swatch = pair.row(align=True)
            swatch.scale_x = 0.58
            swatch.prop(addon_preferences, prop_name, text="")

    common = section("gizmo_section_common_open", "gizmo.section.general")
    if common is not None:
        prop_row(
            common,
            "gizmo_size_px",
            "gizmo.overall_size",
            "gizmo.tip.overall_size",
            slider=True,
        )
        prop_row(
            common,
            "gizmo_zoom_compensation_percent",
            "gizmo.zoom_keep_size",
            "gizmo.tip.zoom_keep_size",
            slider=True,
        )
        prop_row(
            common,
            "gizmo_zoom_out_max_percent",
            "gizmo.zoom_out_max",
            "gizmo.tip.zoom_out_max",
            slider=True,
        )

    rotate = section("gizmo_section_rotate_open", "gizmo.section.rotate")
    if rotate is not None:
        prop_row(
            rotate,
            "gizmo_rotate_axis_width",
            "gizmo.rotate_axis_thickness",
            "gizmo.tip.rotate_axis_thickness",
            slider=True,
        )
        prop_row(
            rotate,
            "gizmo_rotate_view_width",
            "gizmo.rotate_outer_thickness",
            "gizmo.tip.rotate_outer_thickness",
            slider=True,
        )
        prop_row(
            rotate,
            "gizmo_rotate_view_scale_percent",
            "gizmo.rotate_outer_distance",
            "gizmo.tip.rotate_outer_distance",
            slider=True,
        )
        prop_row(
            rotate,
            "gizmo_free_rotate_sensitivity_percent",
            "gizmo.rotate_free_sensitivity",
            "gizmo.tip.rotate_free_sensitivity",
            slider=True,
        )
        color_row(
            rotate,
            (
                ("X:", "gizmo_rotate_x_color"),
                ("Y:", "gizmo_rotate_y_color"),
                ("Z:", "gizmo_rotate_z_color"),
                ("V:", "gizmo_rotate_view_color"),
            ),
            "gizmo.tip.rotate_colors",
        )
        prop_row(
            rotate,
            "gizmo_rotation_step_degrees",
            "gizmo.rotate_snap_angle",
            "gizmo.tip.rotate_snap_angle",
            slider=True,
        )
        toggle_row(
            rotate,
            "gizmo_show_rotation_angle",
            "gizmo.rotate_show_angle",
            "gizmo.tip.rotate_show_angle",
        )

    move = section("gizmo_section_move_open", "gizmo.section.move")
    if move is not None:
        prop_row(
            move,
            "gizmo_move_axis_width",
            "gizmo.move_axis_thickness",
            "gizmo.tip.move_axis_thickness",
            slider=True,
        )
        prop_row(
            move,
            "gizmo_move_arrow_length_percent",
            "gizmo.move_arrow_length",
            "gizmo.tip.move_arrow_length",
            slider=True,
        )
        prop_row(
            move,
            "gizmo_move_arrow_head_size_percent",
            "gizmo.move_arrow_tip",
            "gizmo.tip.move_arrow_tip",
            slider=True,
        )
        prop_row(
            move,
            "gizmo_move_plane_size_percent",
            "gizmo.move_plane_box",
            "gizmo.tip.move_plane_box",
            slider=True,
        )
        color_row(
            move,
            (
                ("X:", "gizmo_move_x_color"),
                ("Y:", "gizmo_move_y_color"),
                ("Z:", "gizmo_move_z_color"),
            ),
            "gizmo.tip.move_colors",
        )

    scale = section("gizmo_section_scale_open", "gizmo.section.scale")
    if scale is not None:
        prop_row(
            scale,
            "gizmo_scale_axis_width",
            "gizmo.scale_axis_thickness",
            "gizmo.tip.scale_axis_thickness",
            slider=True,
        )
        prop_row(
            scale,
            "gizmo_scale_axis_length_percent",
            "gizmo.scale_axis_length",
            "gizmo.tip.scale_axis_length",
            slider=True,
        )
        prop_row(
            scale,
            "gizmo_scale_handle_size_percent",
            "gizmo.scale_axis_box",
            "gizmo.tip.scale_axis_box",
            slider=True,
        )
        prop_row(
            scale,
            "gizmo_scale_uniform_size_percent",
            "gizmo.scale_center_box",
            "gizmo.tip.scale_center_box",
            slider=True,
        )
        color_row(
            scale,
            (
                ("X:", "gizmo_scale_x_color"),
                ("Y:", "gizmo_scale_y_color"),
                ("Z:", "gizmo_scale_z_color"),
            ),
            "gizmo.tip.scale_colors",
        )


def gizmo_zoom_scale(
    context,
    reference_view_distance: float | None,
) -> tuple[float, float | None]:
    """Keep the global gizmo readable when the view moves far from its reference zoom.

    Compensation is deliberately one-way: it may enlarge the shared gizmo shell,
    but never shrink it below the Blender-native/global Gizmo Size. This avoids
    the previous close-zoom failure where positive compensation made the gizmo
    even smaller as the camera approached the selection.
    """

    region_data = getattr(context, "region_data", None)
    if region_data is None:
        region_data = getattr(getattr(context, "space_data", None), "region_3d", None)
    if region_data is None:
        return 1.0, reference_view_distance

    current = max(1e-6, float(getattr(region_data, "view_distance", 0.0) or 0.0))
    if current <= 1e-6:
        return 1.0, reference_view_distance
    if reference_view_distance is None or reference_view_distance <= 1e-6:
        return 1.0, current

    addon_preferences = addon_gizmo_preferences(context)
    compensation_percent = float(
        getattr(addon_preferences, "gizmo_zoom_compensation_percent", 50.0)
    )
    strength = max(0.0, min(2.0, compensation_percent / 100.0))
    if strength <= 1e-6:
        return 1.0, reference_view_distance

    ratio = max(1e-6, current / reference_view_distance)
    zoom_delta = max(ratio, 1.0 / ratio)
    exponent = 0.5 * strength

    if ratio >= 1.0:
        # Zoom-out has its own user-tunable hard ceiling so extreme viewport
        # distances never make the shared transform shell dominate the screen.
        zoom_out_max_percent = float(
            getattr(addon_preferences, "gizmo_zoom_out_max_percent", 400.0)
        )
        normalized = max(0.0, min(1.0, (zoom_out_max_percent - 100.0) / 300.0))
        zoom_out_limit = 1.0 + normalized
        scale = min(zoom_out_limit, zoom_delta**exponent)
        return max(1.0, scale), reference_view_distance
    else:
        # Close zoom is the opposite usability problem: Blender's apparent
        # gizmo size can become too small near the selection, so preserve the
        # stronger enlargement range the user accepted for zoom-in.
        maximum = 1.0 + (1.5 * strength)

    scale = min(maximum, zoom_delta**exponent)
    return max(1.0, scale), reference_view_distance
