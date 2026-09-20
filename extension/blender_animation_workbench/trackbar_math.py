from __future__ import annotations

import math


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def frame_to_x(frame: float, frame_start: float, frame_end: float, x_min: float, x_max: float) -> float:
    if frame_end <= frame_start or x_max <= x_min:
        return x_min
    t = (frame - frame_start) / (frame_end - frame_start)
    return x_min + clamp(t, 0.0, 1.0) * (x_max - x_min)


def x_to_frame(x: float, frame_start: float, frame_end: float, x_min: float, x_max: float) -> float:
    if frame_end <= frame_start or x_max <= x_min:
        return frame_start
    t = clamp((x - x_min) / (x_max - x_min), 0.0, 1.0)
    return frame_start + t * (frame_end - frame_start)


def visible_frame_capacity(pixel_width: float, cell_width: float) -> int:
    if pixel_width <= 0.0 or cell_width <= 0.0:
        return 1
    return max(1, int(pixel_width // cell_width))


def fitted_frame_cell_width(frame_start: int, frame_end: int, pixel_width: float) -> float:
    frame_count = max(1, frame_end - frame_start + 1)
    if pixel_width <= 0.0:
        return 1.0
    return pixel_width / frame_count


def frame_cell_bounds(
    frame: int,
    visible_start: int,
    x_min: float,
    cell_width: float,
) -> tuple[float, float]:
    left = x_min + (frame - visible_start) * cell_width
    return left, left + cell_width


def x_to_cell_frame(
    x: float,
    visible_start: int,
    x_min: float,
    cell_width: float,
    capacity: int,
) -> int:
    if cell_width <= 0.0 or capacity <= 0:
        return visible_start
    cell_index = math.floor((x - x_min) / cell_width)
    cell_index = max(0, min(capacity - 1, cell_index))
    return visible_start + cell_index


def frame_delta_from_pixel_drag(delta_x: float, cell_width: float) -> int:
    if cell_width <= 0.0:
        return 0
    return round(delta_x / cell_width)


def selection_range_frame_span(selected_frames) -> tuple[int, int] | None:
    """Return snapped min/max semantic frames for a multi-selection range."""
    frames = sorted({round(float(frame)) for frame in selected_frames})
    if len(frames) < 2:
        return None
    return frames[0], frames[-1]


def nice_tick_step(frame_span: float, pixel_width: float, target_spacing_px: float = 80.0) -> int:
    if frame_span <= 0.0 or pixel_width <= 0.0:
        return 1

    target_ticks = max(1.0, pixel_width / target_spacing_px)
    raw_step = max(1.0, frame_span / target_ticks)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude

    if normalized <= 1.0:
        nice = 1.0
    elif normalized <= 2.0:
        nice = 2.0
    elif normalized <= 5.0:
        nice = 5.0
    else:
        nice = 10.0

    return max(1, round(nice * magnitude))
