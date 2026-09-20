from __future__ import annotations

import math

import blf
import gpu
from gpu_extras.batch import batch_for_shader

from .phase4_contact_authoring import contact_key_types_for_context
from .phase4_contact_model import ContactKeyType
from .rigped_animation_baseline import RigpedKeyState, rigped_key_types_for_context
from .trackbar_math import (
    fitted_frame_cell_width,
    frame_cell_bounds,
    nice_tick_step,
    selection_range_frame_span,
)
from .trackbar_model import semantic_keys_for_context, trackbar_extension_cells_for_context

# Max-style vertical stack, bottom to top:
# fixed transport/KEY/AUTO -> frame numbers -> selection range -> key row
# -> moving current-frame control.
# The moving < frame > control is back on its original upper rail, while the
# fixed |</>|, KEY and AUTO controls remain on the bottom row. This keeps the
# key track full-width and lets the current-frame slider follow every frame.
TRACKBAR_HEIGHT = 142
TRACKBAR_INTERACTION_HEIGHT = 142
TRACKBAR_PAD_X = 0
TRACK_LINE_Y = 67
KEY_MARKER_BOTTOM = 68.0
KEY_MARKER_HEIGHT = 46.0
KEY_BORDER = 1
CURRENT_FRAME_CONTROL_Y_MIN = 115.0
CURRENT_FRAME_CONTROL_Y_MAX = 142.0
CURRENT_CELL_TOP = float(TRACKBAR_HEIGHT)
FRAME_CONTROL_Y_MIN = 0.0
FRAME_CONTROL_Y_MAX = 27.0
FRAME_CONTROL_ARROW_WIDTH = 27.0
FRAME_CONTROL_VALUE_WIDTH = 72.0
FRAME_CONTROL_GAP = 0.0
AUTO_KEY_BUTTON_WIDTH = 58.0
AUTO_KEY_BUTTON_GAP = 8.0
KEY_MODE_BUTTON_WIDTH = 48.0
FIXED_CONTROL_GAP = 6.0
TRANSPORT_ENDPOINT_BUTTON_WIDTH = 28.0
TRANSPORT_ENDPOINT_GAP = 2.0
SELECTION_RANGE_Y_MIN = 49.0
SELECTION_RANGE_Y_MAX = 63.0
SELECTION_RANGE_HANDLE_HALF_WIDTH = 10.0

_TRACKBAR_BG = (0.08561311, 0.08561311, 0.08561311, 1.0)  # calibrated screen target: #1C1C1C
_WORK_AREA_BG = _TRACKBAR_BG
_WORK_AREA_END = (0.62, 0.62, 0.62, 1.0)
_MAX_KEY_COLORS = {
    "POSITION": (0.92, 0.236, 0.236, 1.0),
    "ROTATION": (0.24, 0.78, 0.294, 1.0),
    "SCALE": (0.257, 0.455, 0.95, 1.0),
    "OTHER": (0.55, 0.55, 0.55, 1.0),
}
_MAX_KEY_ORDER = ("POSITION", "ROTATION", "SCALE", "OTHER")
_CONTACT_KEY_COLORS = {
    ContactKeyType.FREE: (0.55, 0.55, 0.55, 1.0),
    ContactKeyType.SLIDING: (0.95, 0.76, 0.10, 1.0),
    ContactKeyType.PLANTED: (0.36, 0.72, 0.98, 1.0),
}
_RIGPED_KEY_COLORS = {
    RigpedKeyState.FREE: (0.55, 0.55, 0.55, 1.0),
    RigpedKeyState.SLIDING: (0.95, 0.76, 0.10, 1.0),
    RigpedKeyState.PLANTED: (0.36, 0.72, 0.98, 1.0),
}
_CURRENT_CELL_FILL = (0.24, 0.24, 0.24, 1.0)
_CURRENT_KEY_TINT = (1.0, 0.78, 0.08)
_CURRENT_KEY_TINT_AMOUNT = 0.50
_CURRENT_CELL_EDGE = (1.0, 0.84, 0.18, 1.0)
_FRAME_CONTROL_EDGE = (0.45, 0.45, 0.45, 1.0)
_FRAME_CONTROL_FILL = (0.12, 0.12, 0.12, 1.0)
_FRAME_CONTROL_HOVER = (0.20, 0.20, 0.20, 1.0)
_FRAME_CONTROL_TEXT = (0.92, 0.92, 0.92, 1.0)
_AUTO_KEY_FILL = (0.12, 0.12, 0.12, 1.0)
_AUTO_KEY_ACTIVE_FILL = (0.62, 0.08, 0.08, 1.0)
_AUTO_KEY_ACTIVE_EDGE = (0.95, 0.16, 0.12, 1.0)
_AUTO_KEY_VIEWPORT_EDGE = (0.95, 0.08, 0.06, 1.0)
_KEY_MODE_ACTIVE_FILL = (0.10, 0.28, 0.52, 1.0)
_KEY_MODE_ACTIVE_EDGE = (0.28, 0.58, 0.95, 1.0)
_KEY_MODE_VIEWPORT_EDGE = (0.08, 0.38, 0.95, 1.0)
_BOX_SELECT_FILL = (0.52, 0.52, 0.52, 0.20)
_BOX_SELECT_EDGE = (0.72, 0.72, 0.72, 1.0)
_SELECTION_RANGE_FILL = (0.38, 0.38, 0.38, 1.0)
_SELECTION_RANGE_EDGE = (0.72, 0.72, 0.72, 1.0)
_SELECTION_RANGE_HANDLE = (0.72, 0.72, 0.72, 1.0)

_SHADER = None
_BOX_SELECT_RANGE: tuple[float, float] | None = None


def _shader():
    global _SHADER
    if _SHADER is None:
        _SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _SHADER


def trackbar_bounds(region):
    x_min = float(TRACKBAR_PAD_X)
    x_max = float(max(TRACKBAR_PAD_X + 1, region.width - TRACKBAR_PAD_X))
    return x_min, x_max, 0.0, float(TRACKBAR_INTERACTION_HEIGHT)


def set_box_selection_overlay(start_x: float, end_x: float) -> None:
    global _BOX_SELECT_RANGE
    _BOX_SELECT_RANGE = (float(start_x), float(end_x))


def clear_box_selection_overlay() -> None:
    global _BOX_SELECT_RANGE
    _BOX_SELECT_RANGE = None


def visible_frame_window(scene, region) -> tuple[int, int, int, float, float]:
    x_min, x_max, _y_min, _y_max = trackbar_bounds(region)
    frame_start = int(scene.frame_start)
    frame_end = int(max(scene.frame_start, scene.frame_end))
    default_span = max(1, frame_end - frame_start + 1)

    view_span = int(getattr(scene, "baw_trackbar_view_span", 0))
    if view_span <= 0:
        view_span = default_span
        view_center = (frame_start + frame_end) * 0.5
    else:
        view_center = float(
            getattr(
                scene,
                "baw_trackbar_view_center",
                (frame_start + frame_end) * 0.5,
            )
        )

    start = round(view_center - (view_span - 1) * 0.5)
    end = start + view_span - 1
    return start, end, view_span, x_min, x_max


def frame_cell_width(scene, region) -> float:
    start, end, _frame_count, x_min, x_max = visible_frame_window(scene, region)
    return fitted_frame_cell_width(start, end, x_max - x_min)


def snapped_frame_cell_bounds(
    frame: int,
    visible_start: int,
    x_min: float,
    cell_width: float,
) -> tuple[float, float]:
    """Return one pixel-snapped cell box shared by keys and guide lines."""
    raw_left, raw_right = frame_cell_bounds(frame, visible_start, x_min, cell_width)
    left = float(round(raw_left))
    right = float(round(raw_right))
    if right <= left:
        right = left + 1.0
    return left, right


def selection_range_bounds(scene, region, selected_frames) -> dict[str, object] | None:
    """Return Selection Range Bar geometry from actual selected semantic frames."""
    span = selection_range_frame_span(selected_frames)
    if span is None:
        return None

    left_frame, right_frame = span
    visible_start, _visible_end, _frame_count, x_min, x_max = visible_frame_window(
        scene,
        region,
    )
    cell_width = frame_cell_width(scene, region)
    left_cell = snapped_frame_cell_bounds(
        left_frame,
        visible_start,
        x_min,
        cell_width,
    )
    right_cell = snapped_frame_cell_bounds(
        right_frame,
        visible_start,
        x_min,
        cell_width,
    )
    left_x = (left_cell[0] + left_cell[1]) * 0.5
    right_x = (right_cell[0] + right_cell[1]) * 0.5

    # Fully off-screen ranges have no visible/hittable bar. Partly visible ranges
    # keep true endpoint centers while clipping only the center segment.
    if right_x < x_min or left_x > x_max:
        return None
    center_left = max(x_min, left_x)
    center_right = min(x_max, right_x)
    if center_right < center_left:
        return None

    half_width = SELECTION_RANGE_HANDLE_HALF_WIDTH
    y_min = SELECTION_RANGE_Y_MIN
    y_max = SELECTION_RANGE_Y_MAX
    return {
        "left_frame": left_frame,
        "right_frame": right_frame,
        "left_x": left_x,
        "right_x": right_x,
        "left_handle": (
            left_x - half_width,
            y_min,
            left_x + half_width,
            y_max,
        ),
        "center": (center_left, y_min, center_right, y_max),
        "right_handle": (
            right_x - half_width,
            y_min,
            right_x + half_width,
            y_max,
        ),
    }


def auto_key_button_bounds(region) -> tuple[float, float, float, float]:
    x_min, x_max, _y_min, _y_max = trackbar_bounds(region)
    x2 = x_max
    x1 = max(x_min, x2 - AUTO_KEY_BUTTON_WIDTH)
    return x1, FRAME_CONTROL_Y_MIN, x2, FRAME_CONTROL_Y_MAX


def key_mode_button_bounds(region) -> tuple[float, float, float, float] | None:
    x_min, x_max, _y_min, _y_max = trackbar_bounds(region)
    auto_left = auto_key_button_bounds(region)[0]
    available = x_max - x_min
    required = (
        AUTO_KEY_BUTTON_WIDTH
        + FIXED_CONTROL_GAP
        + KEY_MODE_BUTTON_WIDTH
    )
    if available < required:
        return None
    x2 = auto_left - FIXED_CONTROL_GAP
    x1 = x2 - KEY_MODE_BUTTON_WIDTH
    return x1, FRAME_CONTROL_Y_MIN, x2, FRAME_CONTROL_Y_MAX


def transport_endpoint_button_bounds(region) -> dict[str, tuple[float, float, float, float]]:
    key_bounds = key_mode_button_bounds(region)
    if key_bounds is None:
        return {}
    x_min, x_max, _y_min, _y_max = trackbar_bounds(region)
    available = x_max - x_min
    required = (
        AUTO_KEY_BUTTON_WIDTH
        + FIXED_CONTROL_GAP
        + KEY_MODE_BUTTON_WIDTH
        + FIXED_CONTROL_GAP
        + TRANSPORT_ENDPOINT_BUTTON_WIDTH * 2.0
        + TRANSPORT_ENDPOINT_GAP
    )
    if available < required:
        return {}

    end_x2 = key_bounds[0] - FIXED_CONTROL_GAP
    end_x1 = end_x2 - TRANSPORT_ENDPOINT_BUTTON_WIDTH
    start_x2 = end_x1 - TRANSPORT_ENDPOINT_GAP
    start_x1 = start_x2 - TRANSPORT_ENDPOINT_BUTTON_WIDTH
    return {
        "start": (start_x1, FRAME_CONTROL_Y_MIN, start_x2, FRAME_CONTROL_Y_MAX),
        "end": (end_x1, FRAME_CONTROL_Y_MIN, end_x2, FRAME_CONTROL_Y_MAX),
    }


def _fixed_control_left_edge(region) -> float:
    endpoints = transport_endpoint_button_bounds(region)
    if endpoints:
        return endpoints["start"][0]
    key_bounds = key_mode_button_bounds(region)
    if key_bounds is not None:
        return key_bounds[0]
    return auto_key_button_bounds(region)[0]


def frame_control_bounds(scene, region):
    visible_start, visible_end, _frame_count, x_min, x_max = visible_frame_window(scene, region)
    frame = int(scene.frame_current)
    cell_width = frame_cell_width(scene, region)

    if visible_start <= frame <= visible_end:
        cell_left, cell_right = snapped_frame_cell_bounds(
            frame,
            visible_start,
            x_min,
            cell_width,
        )
        cell_bounds = (
            cell_left,
            KEY_MARKER_BOTTOM,
            cell_right,
            KEY_MARKER_BOTTOM + KEY_MARKER_HEIGHT,
        )
    else:
        cell_bounds = None

    total_width = (
        FRAME_CONTROL_ARROW_WIDTH * 2.0
        + FRAME_CONTROL_VALUE_WIDTH
        + FRAME_CONTROL_GAP * 2.0
    )
    # The moving current-frame control lives on its own upper rail, so it no
    # longer needs to reserve horizontal room for the fixed bottom controls.
    control_x_max = x_max

    if cell_bounds is not None:
        cell_left, _cell_bottom, cell_right, _cell_top = cell_bounds
        current_x = (cell_left + cell_right) * 0.5
    elif frame < visible_start:
        current_x = x_min
    elif frame > visible_end:
        current_x = control_x_max
    else:
        current_x = x_min + (control_x_max - x_min) * 0.5
    group_left = max(
        x_min,
        min(control_x_max - total_width, current_x - total_width * 0.5),
    )
    y1 = CURRENT_FRAME_CONTROL_Y_MIN
    y2 = CURRENT_FRAME_CONTROL_Y_MAX

    left_arrow = (
        group_left,
        y1,
        group_left + FRAME_CONTROL_ARROW_WIDTH,
        y2,
    )
    value_left = left_arrow[2] + FRAME_CONTROL_GAP
    value_box = (
        value_left,
        y1,
        value_left + FRAME_CONTROL_VALUE_WIDTH,
        y2,
    )
    right_left = value_box[2] + FRAME_CONTROL_GAP
    right_arrow = (
        right_left,
        y1,
        right_left + FRAME_CONTROL_ARROW_WIDTH,
        y2,
    )
    return {
        "frame": frame,
        "cell": cell_bounds,
        "left": left_arrow,
        "value": value_box,
        "right": right_arrow,
    }


def _draw_rect(x1, y1, x2, y2, color):
    shader = _shader()
    batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": ((x1, y1), (x2, y1), (x2, y2), (x1, y2))},
        indices=((0, 1, 2), (0, 2, 3)),
    )
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_line(x1, y1, x2, y2, color, width=1.0):
    shader = _shader()
    batch = batch_for_shader(shader, "LINES", {"pos": ((x1, y1), (x2, y2))})
    gpu.state.line_width_set(width)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)


def _draw_selection_range(bounds) -> None:
    center = bounds["center"]
    left_handle = bounds["left_handle"]
    right_handle = bounds["right_handle"]

    center_x0, center_y0, center_x1, center_y1 = center
    if center_x1 > center_x0:
        _draw_rect(center_x0, center_y0, center_x1, center_y1, _SELECTION_RANGE_FILL)
        _draw_line(
            center_x0,
            center_y0,
            center_x1,
            center_y0,
            _SELECTION_RANGE_EDGE,
        )
        _draw_line(
            center_x0,
            center_y1,
            center_x1,
            center_y1,
            _SELECTION_RANGE_EDGE,
        )

    for handle in (left_handle, right_handle):
        x0, y0, x1, y1 = handle
        if x1 > x0:
            _draw_rect(x0, y0, x1, y1, _SELECTION_RANGE_HANDLE)


def _draw_vertical_lines_batch(xs, y1, y2, color, width=1.0):
    pixel_xs = sorted({round(float(x)) for x in xs})
    if not pixel_xs:
        return
    verts = []
    for x in pixel_xs:
        verts.extend(((float(x), y1), (float(x), y2)))
    shader = _shader()
    batch = batch_for_shader(shader, "LINES", {"pos": tuple(verts)})
    gpu.state.line_width_set(width)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)


def _draw_viewport_mode_outline(region, color, width: float = 3.0) -> None:
    """Draw the mode outline around the 3D viewport, excluding the Track Bar."""
    inset = max(1.0, width * 0.5)
    x0 = inset
    # The Track Bar is an independent input surface. Keep AUTO/KEY mode
    # feedback around the actual 3D viewport only, with its lower edge aligned
    # to the Track Bar's upper boundary instead of wrapping the Track Bar too.
    y0 = min(
        float(region.height) - inset,
        float(TRACKBAR_HEIGHT) + inset,
    )
    x1 = max(x0, float(region.width) - inset)
    y1 = max(y0, float(region.height) - inset)
    _draw_line(x0, y0, x1, y0, color, width)
    _draw_line(x1, y0, x1, y1, color, width)
    _draw_line(x1, y1, x0, y1, color, width)
    _draw_line(x0, y1, x0, y0, color, width)


def _frame_draw_step(cell_width: float, min_spacing_px: float = 1.0) -> int:
    if cell_width <= 0.0:
        return 1
    return max(1, math.ceil(min_spacing_px / cell_width))


def _draw_diamond(x, y, radius, color):
    shader = _shader()
    verts = ((x, y + radius), (x + radius, y), (x, y - radius), (x - radius, y))
    batch = batch_for_shader(shader, "TRIS", {"pos": verts}, indices=((0, 1, 2), (0, 2, 3)))
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_triangle(a, b, c, color):
    shader = _shader()
    batch = batch_for_shader(shader, "TRIS", {"pos": (a, b, c)})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _mix_rgb(base, tint, amount: float):
    amount = max(0.0, min(1.0, amount))
    return (
        base[0] * (1.0 - amount) + tint[0] * amount,
        base[1] * (1.0 - amount) + tint[1] * amount,
        base[2] * (1.0 - amount) + tint[2] * amount,
        base[3],
    )


def _draw_channel_key(left, right, channels, *, selected: bool, current: bool = False):
    ordered = [channel for channel in _MAX_KEY_ORDER if channel in channels]
    if not ordered:
        ordered = ["OTHER"]

    left = float(round(left))
    right = float(round(right))
    bottom = KEY_MARKER_BOTTOM
    top = bottom + KEY_MARKER_HEIGHT

    if selected:
        _draw_rect(left, bottom, right, top, (1.0, 1.0, 1.0, 1.0))
        return

    _draw_rect(left, bottom, right, top, (0.08, 0.08, 0.08, 1.0))

    # Fill must touch the complete snapped key cell on every side.
    # Any top/bottom inset leaves a visible 1px gap against the Track Bar rails.
    inner_left = left
    inner_right = right
    inner_bottom = bottom
    inner_top = top

    band_height = (inner_top - inner_bottom) / len(ordered)
    for index, channel in enumerate(ordered):
        y1 = inner_bottom + band_height * index
        y2 = inner_top if index == len(ordered) - 1 else inner_bottom + band_height * (index + 1)
        color = _MAX_KEY_COLORS[channel]
        if current:
            color = _mix_rgb(color, _CURRENT_KEY_TINT, _CURRENT_KEY_TINT_AMOUNT)
        _draw_rect(inner_left, y1, inner_right, y2, color)


def _draw_contact_key(left, right, contact_type: ContactKeyType, *, selected: bool) -> None:
    left = float(round(left))
    right = float(round(right))
    bottom = KEY_MARKER_BOTTOM
    top = bottom + KEY_MARKER_HEIGHT
    color = _CONTACT_KEY_COLORS[contact_type]
    _draw_rect(left, bottom, right, top, color)
    if selected:
        # Preserve the semantic Contact color even when Blender marks the newly
        # authored bundle selected. Selection is indicated by a white outline,
        # never by replacing the state fill with white.
        _draw_line(left, bottom, right, bottom, (1.0, 1.0, 1.0, 1.0), 2.0)
        _draw_line(right, bottom, right, top, (1.0, 1.0, 1.0, 1.0), 2.0)
        _draw_line(right, top, left, top, (1.0, 1.0, 1.0, 1.0), 2.0)
        _draw_line(left, top, left, bottom, (1.0, 1.0, 1.0, 1.0), 2.0)


def _draw_rigped_key(left, right, key_type: RigpedKeyState, *, selected: bool) -> None:
    left = float(round(left))
    right = float(round(right))
    bottom = KEY_MARKER_BOTTOM
    top = bottom + KEY_MARKER_HEIGHT
    color = _RIGPED_KEY_COLORS[key_type]
    _draw_rect(left, bottom, right, top, color)
    if selected:
        _draw_line(left, bottom, right, bottom, (1.0, 1.0, 1.0, 1.0), 2.0)
        _draw_line(right, bottom, right, top, (1.0, 1.0, 1.0, 1.0), 2.0)
        _draw_line(right, top, left, top, (1.0, 1.0, 1.0, 1.0), 2.0)
        _draw_line(left, top, left, bottom, (1.0, 1.0, 1.0, 1.0), 2.0)


def _draw_text(text, x, y, size, color):
    font_id = 0
    blf.position(font_id, x, y, 0)
    blf.size(font_id, size)
    blf.color(font_id, *color)
    blf.draw(font_id, text)


def _draw_frame_control_box(bounds, fill):
    x1, y1, x2, y2 = bounds
    _draw_rect(x1, y1, x2, y2, _FRAME_CONTROL_EDGE)
    _draw_rect(x1 + 1, y1 + 1, x2 - 1, y2 - 1, fill)


def _draw_current_frame_control(bounds, *, draw_cell_fill: bool) -> None:
    frame = int(bounds["frame"])
    cell_bounds = bounds["cell"]
    if cell_bounds is not None:
        cell_left, cell_bottom, cell_right, cell_top = cell_bounds
        if draw_cell_fill:
            # Empty current frame: only the soft gray fill, no yellow outline.
            _draw_rect(cell_left, cell_bottom, cell_right, cell_top, _CURRENT_CELL_FILL)
        else:
            # Current frame with keys: keep the yellow emphasis around the key cell.
            _draw_line(cell_left, cell_bottom, cell_right, cell_bottom, _CURRENT_CELL_EDGE, 1.0)
            _draw_line(cell_left, cell_top, cell_right, cell_top, _CURRENT_CELL_EDGE, 1.0)
            _draw_line(cell_left, cell_bottom, cell_left, cell_top, _CURRENT_CELL_EDGE, 1.0)
            _draw_line(cell_right, cell_bottom, cell_right, cell_top, _CURRENT_CELL_EDGE, 1.0)

    _draw_frame_control_box(bounds["left"], _FRAME_CONTROL_FILL)
    _draw_frame_control_box(bounds["value"], _FRAME_CONTROL_FILL)
    _draw_frame_control_box(bounds["right"], _FRAME_CONTROL_FILL)

    left_x1, left_y1, left_x2, left_y2 = bounds["left"]
    left_cx = (left_x1 + left_x2) * 0.5
    left_cy = (left_y1 + left_y2) * 0.5
    _draw_triangle(
        (left_cx - 5.0, left_cy),
        (left_cx + 4.0, left_cy + 7.0),
        (left_cx + 4.0, left_cy - 7.0),
        _FRAME_CONTROL_TEXT,
    )

    right_x1, right_y1, right_x2, right_y2 = bounds["right"]
    right_cx = (right_x1 + right_x2) * 0.5
    right_cy = (right_y1 + right_y2) * 0.5
    _draw_triangle(
        (right_cx + 5.0, right_cy),
        (right_cx - 4.0, right_cy + 7.0),
        (right_cx - 4.0, right_cy - 7.0),
        _FRAME_CONTROL_TEXT,
    )

    value_x1, value_y1, value_x2, value_y2 = bounds["value"]
    label = str(frame)
    font_id = 0
    font_size = 16
    blf.size(font_id, font_size)
    text_width, text_height = blf.dimensions(font_id, label)
    text_x = value_x1 + ((value_x2 - value_x1) - text_width) * 0.5
    text_y = value_y1 + ((value_y2 - value_y1) - text_height) * 0.5 + 1.0
    _draw_text(label, text_x, text_y, font_size, _FRAME_CONTROL_TEXT)


def _draw_auto_key_button(bounds, *, active: bool) -> None:
    fill = _AUTO_KEY_ACTIVE_FILL if active else _AUTO_KEY_FILL
    edge = _AUTO_KEY_ACTIVE_EDGE if active else _FRAME_CONTROL_EDGE
    x1, y1, x2, y2 = bounds
    _draw_rect(x1, y1, x2, y2, edge)
    _draw_rect(x1 + 1, y1 + 1, x2 - 1, y2 - 1, fill)

    label = "AUTO"
    font_id = 0
    font_size = 12
    blf.size(font_id, font_size)
    text_width, text_height = blf.dimensions(font_id, label)
    text_x = x1 + ((x2 - x1) - text_width) * 0.5
    text_y = y1 + ((y2 - y1) - text_height) * 0.5 + 1.0
    _draw_text(label, text_x, text_y, font_size, _FRAME_CONTROL_TEXT)


def _draw_fixed_button(bounds, label: str, *, active: bool = False) -> None:
    if bounds is None:
        return
    fill = _KEY_MODE_ACTIVE_FILL if active else _FRAME_CONTROL_FILL
    edge = _KEY_MODE_ACTIVE_EDGE if active else _FRAME_CONTROL_EDGE
    x1, y1, x2, y2 = bounds
    if x2 <= x1:
        return
    _draw_rect(x1, y1, x2, y2, edge)
    _draw_rect(x1 + 1, y1 + 1, x2 - 1, y2 - 1, fill)

    font_id = 0
    font_size = 11 if len(label) > 2 else 13
    blf.size(font_id, font_size)
    text_width, text_height = blf.dimensions(font_id, label)
    text_x = x1 + ((x2 - x1) - text_width) * 0.5
    text_y = y1 + ((y2 - y1) - text_height) * 0.5 + 1.0
    _draw_text(label, text_x, text_y, font_size, _FRAME_CONTROL_TEXT)


def draw_trackbar(
    context,
    *,
    semantic_keys_override=None,
    contact_key_types_override=None,
):
    scene = context.scene
    region = context.region
    if scene is None or region is None or region.width < 180:
        return

    visible_start, visible_end, _frame_count, x_min, track_x_max = visible_frame_window(scene, region)
    cell_width = frame_cell_width(scene, region)

    gpu.state.blend_set("ALPHA")
    try:
        _draw_rect(x_min, 0.0, track_x_max, CURRENT_CELL_TOP, _WORK_AREA_BG)
        # Keep the fixed |</>| / KEY / AUTO row on its own darker rail.
        # Draw it after the work-area fill so the rail background is not
        # overwritten by the full-width Track Bar background.
        _draw_rect(0.0, 0.0, float(region.width), float(FRAME_CONTROL_Y_MAX), _TRACKBAR_BG)
        _draw_line(track_x_max, 0.0, track_x_max, CURRENT_CELL_TOP, _WORK_AREA_END, 1.0)
        _draw_line(x_min, TRACK_LINE_Y, track_x_max, TRACK_LINE_Y, (0.46, 0.46, 0.46, 1.0), 1.0)

        visible_count = max(1, visible_end - visible_start + 1)
        tick_step = nice_tick_step(float(visible_count), track_x_max - x_min)
        first_major = int(math.ceil(visible_start / tick_step) * tick_step)

        minor_step = _frame_draw_step(cell_width, 2.0)
        minor_xs = []
        for frame in range(visible_start, visible_end + 1, minor_step):
            if frame >= first_major and frame % tick_step == 0:
                continue
            left, right = frame_cell_bounds(frame, visible_start, x_min, cell_width)
            minor_xs.append((left + right) * 0.5)
        _draw_vertical_lines_batch(
            minor_xs,
            TRACK_LINE_Y - 2,
            TRACK_LINE_Y + 2,
            (0.52, 0.52, 0.52, 1.0),
            1.0,
        )

        major_xs = []
        for frame in range(first_major, visible_end + 1, tick_step):
            if frame < visible_start:
                continue
            left, right = frame_cell_bounds(frame, visible_start, x_min, cell_width)
            center = (left + right) * 0.5
            major_xs.append(center)
            _draw_text(str(frame), center - 7, 35, 10, (0.68, 0.68, 0.68, 1.0))
        _draw_vertical_lines_batch(
            major_xs,
            TRACK_LINE_Y - 2,
            TRACK_LINE_Y + 5,
            (0.52, 0.52, 0.52, 1.0),
            1.0,
        )

        control_bounds = frame_control_bounds(scene, region)

        key_line_bottom = KEY_MARKER_BOTTOM
        # GPU line endpoints rasterize slightly beyond the visual key cap;
        # stop 1px short so the gray guide matches the marker top exactly.
        key_line_top = KEY_MARKER_BOTTOM + KEY_MARKER_HEIGHT - 1.0
        separator_step = _frame_draw_step(cell_width, 1.0)
        separator_xs = [
            snapped_frame_cell_bounds(frame, visible_start, x_min, cell_width)[0]
            for frame in range(visible_start, visible_end + 1, separator_step)
        ]
        _last_left, last_right = snapped_frame_cell_bounds(
            visible_end,
            visible_start,
            x_min,
            cell_width,
        )
        separator_xs.append(last_right)

        channels_by_cell: dict[int, set[str]] = {}
        contact_types_by_cell: dict[int, ContactKeyType] = {}
        rigped_types_by_cell: dict[int, RigpedKeyState] = {}
        selected_cells: set[int] = set()
        selected_frames: set[int] = set()
        # Fit is structural armature Edit Mode, not an animation-authoring
        # context. Avoid rebuilding the semantic key view on every edit-bone
        # mouse move/redraw; the static Track Bar rail remains visible.
        if context.mode == "EDIT_ARMATURE":
            semantic_keys = ()
        elif semantic_keys_override is None:
            semantic_keys = semantic_keys_for_context(context)
        else:
            semantic_keys = semantic_keys_override
        for key in semantic_keys:
            cell_frame = round(key.frame)
            if key.selected:
                selected_frames.add(cell_frame)
            if visible_start <= cell_frame <= visible_end:
                channels_by_cell.setdefault(cell_frame, set()).update(key.channel_names)
                if key.selected:
                    selected_cells.add(cell_frame)

        # Higher-layer semantic cells (notably Rigped Contact Sliding keys) may
        # live only on hidden authority curves. Ordinary selected frames were
        # already collected from ``semantic_keys`` above; query only extension
        # cells here so whole-Rigped redraw does not perform a second full
        # semantic FCurve scan on every mousemove.
        for frame, is_selected in trackbar_extension_cells_for_context(context).items():
            if not is_selected:
                continue
            cell_frame = round(frame)
            selected_frames.add(cell_frame)
            if visible_start <= cell_frame <= visible_end:
                selected_cells.add(cell_frame)

        if context.mode == "POSE":
            try:
                contact_key_types = (
                    contact_key_types_for_context(context)
                    if contact_key_types_override is None
                    else contact_key_types_override
                )
                for frame, contact_type in contact_key_types.items():
                    cell_frame = round(frame)
                    if visible_start <= cell_frame <= visible_end:
                        contact_types_by_cell[cell_frame] = contact_type
            except (AttributeError, KeyError, ReferenceError, RuntimeError, ValueError):
                # Drawing must stay read-only/fail-soft if selection or animation
                # ownership changes during a viewport redraw.
                contact_types_by_cell.clear()
            try:
                for frame, key_type in rigped_key_types_for_context(context).items():
                    cell_frame = round(frame)
                    if visible_start <= cell_frame <= visible_end:
                        rigped_types_by_cell[cell_frame] = key_type
            except (AttributeError, KeyError, ReferenceError, RuntimeError, ValueError):
                rigped_types_by_cell.clear()

        current_frame = int(scene.frame_current)
        for frame, channels in channels_by_cell.items():
            marker_left, marker_right = snapped_frame_cell_bounds(
                frame,
                visible_start,
                x_min,
                cell_width,
            )
            _draw_channel_key(
                marker_left,
                marker_right,
                channels,
                selected=frame in selected_cells,
                current=frame == current_frame,
            )

        # Contact keys are semantic state keys, so their state color owns the
        # whole Track Bar cell even when the transition bundle also contains
        # ordinary transform channels on that frame.
        for frame, contact_type in contact_types_by_cell.items():
            marker_left, marker_right = snapped_frame_cell_bounds(
                frame,
                visible_start,
                x_min,
                cell_width,
            )
            _draw_contact_key(
                marker_left,
                marker_right,
                contact_type,
                selected=frame in selected_cells,
            )

        # Ordinary Rigped semantic state is a fallback only. If a generated
        # limb owns an authoritative Contact semantic state on the same frame,
        # the Contact color above must remain visible (for example Sliding yellow
        # after an earlier broad-selection Free key existed on that limb).
        for frame, key_type in rigped_types_by_cell.items():
            if frame in contact_types_by_cell:
                continue
            marker_left, marker_right = snapped_frame_cell_bounds(
                frame,
                visible_start,
                x_min,
                cell_width,
            )
            _draw_rigped_key(
                marker_left,
                marker_right,
                key_type,
                selected=frame in selected_cells,
            )

        # Draw separators over the key fills so the 1px guide lands on the
        # exact shared snapped boundary instead of being partially hidden by
        # triangle rasterization at the marker's right edge.
        _draw_vertical_lines_batch(
            separator_xs,
            key_line_bottom,
            key_line_top,
            (0.34, 0.34, 0.34, 1.0),
            1.0,
        )

        range_bounds = selection_range_bounds(scene, region, selected_frames)
        if range_bounds is not None:
            _draw_selection_range(range_bounds)

        if _BOX_SELECT_RANGE is not None:
            # BLF/text drawing earlier in this callback may alter GPU blend
            # state. Reassert alpha here so the selection box never becomes an
            # opaque white/black slab on some Blender/GPU backends.
            gpu.state.blend_set("ALPHA")
            start_x, end_x = _BOX_SELECT_RANGE
            left = max(x_min, min(track_x_max, min(start_x, end_x)))
            right = max(x_min, min(track_x_max, max(start_x, end_x)))
            if right > left:
                top = KEY_MARKER_BOTTOM + KEY_MARKER_HEIGHT
                _draw_rect(left, KEY_MARKER_BOTTOM, right, top, _BOX_SELECT_FILL)
                _draw_line(left, KEY_MARKER_BOTTOM, right, KEY_MARKER_BOTTOM, _BOX_SELECT_EDGE, 2.0)
                _draw_line(left, top, right, top, _BOX_SELECT_EDGE, 2.0)
                _draw_line(left, KEY_MARKER_BOTTOM, left, top, _BOX_SELECT_EDGE, 2.0)
                _draw_line(right, KEY_MARKER_BOTTOM, right, top, _BOX_SELECT_EDGE, 2.0)

        # Current-frame key colors are explicitly composited above, so the
        # cell fill is only used on empty frames. The guide edge and controls
        # still draw last as the top visual layer.
        if control_bounds is not None:
            _draw_current_frame_control(
                control_bounds,
                draw_cell_fill=(
                    current_frame not in channels_by_cell
                    and current_frame not in contact_types_by_cell
                    and current_frame not in rigped_types_by_cell
                ),
            )

        key_mode_active = bool(getattr(scene, "baw_key_mode", False))
        key_bounds = key_mode_button_bounds(region)
        if key_bounds is not None:
            _draw_fixed_button(key_bounds, "KEY", active=key_mode_active)

        endpoint_bounds = transport_endpoint_button_bounds(region)
        if endpoint_bounds:
            _draw_fixed_button(endpoint_bounds["start"], "|<")
            _draw_fixed_button(endpoint_bounds["end"], ">|")

        auto_key_active = bool(getattr(scene, "baw_auto_key_enabled", False))
        _draw_auto_key_button(
            auto_key_button_bounds(region),
            active=auto_key_active,
        )

        # Full viewport mode feedback. AUTO is the stronger warning state, so
        # it takes visual priority when AUTO and KEY are enabled together.
        if auto_key_active:
            _draw_viewport_mode_outline(region, _AUTO_KEY_VIEWPORT_EDGE, 3.0)
        elif key_mode_active:
            _draw_viewport_mode_outline(region, _KEY_MODE_VIEWPORT_EDGE, 3.0)

    finally:
        gpu.state.blend_set("NONE")
