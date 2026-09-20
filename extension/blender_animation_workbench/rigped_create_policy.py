from __future__ import annotations

from dataclasses import replace

from .rigped_humanoid_spec import HumanoidBoneSpec, RigpedHumanoidSpec

_MIN_CREATE_HEIGHT = 0.25
_DRAG_PIXELS_FOR_FULL_SCALE = 400.0
_MOUSE_PIXELS_PER_DOUBLING = 360.0


def humanoid_spec_height(spec: RigpedHumanoidSpec) -> float:
    bones = spec.resolved_bones()
    minimum = min(min(bone.head[2], bone.tail[2]) for bone in bones)
    maximum = max(max(bone.head[2], bone.tail[2]) for bone in bones)
    return float(maximum - minimum)


def height_from_vertical_drag(
    initial_height: float,
    delta_pixels: float,
    *,
    pixels_for_full_scale: float = _DRAG_PIXELS_FOR_FULL_SCALE,
) -> float:
    if initial_height <= 0.0:
        raise ValueError("Initial Create height must be positive.")
    if pixels_for_full_scale <= 0.0:
        raise ValueError("Create drag sensitivity must be positive.")
    scale = 1.0 + (float(delta_pixels) / float(pixels_for_full_scale))
    return max(_MIN_CREATE_HEIGHT, float(initial_height) * scale)


def height_from_free_mouse(
    initial_height: float,
    delta_pixels: float,
    *,
    pixels_per_doubling: float = _MOUSE_PIXELS_PER_DOUBLING,
) -> float:
    """Smooth creation sizing that continues after the placement click is released."""

    if initial_height <= 0.0:
        raise ValueError("Initial Create height must be positive.")
    if pixels_per_doubling <= 0.0:
        raise ValueError("Create mouse sensitivity must be positive.")
    scale = 2.0 ** (float(delta_pixels) / float(pixels_per_doubling))
    return max(_MIN_CREATE_HEIGHT, float(initial_height) * scale)


def _scale_point(point: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(float(value) * scale for value in point)


def fitted_humanoid_spec(
    height: float,
    origin: tuple[float, float, float],
    *,
    base: RigpedHumanoidSpec | None = None,
) -> RigpedHumanoidSpec:
    """Return one immutable generated-humanoid spec fitted to initial height/origin.

    Create preview remains non-authoritative. This helper is called only on
    confirmation so the production builder receives the final topology before
    descriptor publication.
    """

    source = base or RigpedHumanoidSpec()
    source.validate()
    source_height = humanoid_spec_height(source)
    if source_height <= 0.0:
        raise ValueError("Generated humanoid source topology has zero height.")
    if height < _MIN_CREATE_HEIGHT:
        raise ValueError(f"Generated humanoid height must be at least {_MIN_CREATE_HEIGHT}.")
    if len(origin) != 3:
        raise ValueError("Generated humanoid origin requires exactly three coordinates.")

    scale = float(height) / source_height
    scaled: list[HumanoidBoneSpec] = []
    for bone in source.resolved_bones():
        scaled.append(
            replace(
                bone,
                head=_scale_point(bone.head, scale),
                tail=_scale_point(bone.tail, scale),
            )
        )
    return replace(
        source,
        bones=tuple(scaled),
        world_location=tuple(float(value) for value in origin),
    )
