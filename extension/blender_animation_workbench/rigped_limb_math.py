from __future__ import annotations

from mathutils import Vector


def preferred_two_bone_bend(
    baseline_perp: Vector,
    baseline_dir: Vector,
    first_length: float,
    second_length: float,
    *,
    is_arm: bool,
    straight_seed: Vector | None = None,
    fallback: Vector | None = None,
) -> Vector | None:
    """Resolve one stable gesture-start bend direction shared by Fit and Animate.

    The generated arm starts almost perfectly straight. Preserve any meaningful
    animator-authored bend, but for that tiny default arm offset choose the
    opposite branch once so an upward hand move folds in the intended direction.
    No time-varying swivel is introduced here; this is only a mouse-down seed.
    """

    direction = Vector(baseline_dir)
    if direction.length <= 1e-9:
        return None
    direction.normalize()

    bend = Vector(baseline_perp)
    chain_length = max(0.0, float(first_length) + float(second_length))
    bend_epsilon = max(1e-7, chain_length * 1e-4)
    preferred_epsilon = max(bend_epsilon, chain_length * 0.02)

    if is_arm and bend_epsilon < bend.length <= preferred_epsilon:
        bend.negate()

    if bend.length <= bend_epsilon and straight_seed is not None:
        bend = Vector(straight_seed)
        bend -= direction * float(bend.dot(direction))

    if bend.length <= bend_epsilon and fallback is not None:
        bend = Vector(fallback)
        bend -= direction * float(bend.dot(direction))

    if bend.length <= bend_epsilon:
        return None
    bend.normalize()
    return bend
