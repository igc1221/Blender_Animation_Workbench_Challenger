"""Shared generated Rigped solver geometry.

This module is a read-only geometry service. It owns no gesture lifecycle, no
transient snap orchestration, no public projection timing, no Contact/AUTO
policy, no mutation journal, and no depsgraph/view-layer updates. Callers retain
all write authority and timing. This module must not import
phase4_representation_snap or rigped_transform.
"""

from __future__ import annotations

import math
from typing import Any

from .kinematic_runtime import (
    TwoBonePoleSolution,
    solve_capability_pole_position,
    two_bone_world_points,
)


class RigpedSolverGeometryError(RuntimeError):
    pass


def generated_lower_hinge_x_angle(quaternion: Any) -> float:
    """Extract lower-link local-X hinge twist after removing local-Y roll."""

    q = quaternion.normalized()
    w, x, y, z = (
        float(q.w),
        float(q.x),
        float(q.y),
        float(q.z),
    )
    if w < 0.0:
        w, x, y, z = (-w, -x, -y, -z)

    # Match Rigped's authored decomposition: remove the long-axis Y twist
    # first, then read the signed X twist from the remaining swing/hinge
    # rotation. Reading raw quaternion.x/w is not invariant under compound
    # LOCAL-Z swivel + Y roll and can select the opposite knee branch.
    long_norm = math.hypot(w, y)
    if long_norm > 1e-12:
        twist_w = w / long_norm
        twist_y = y / long_norm
        without_long_w = w * twist_w + y * twist_y
        without_long_x = x * twist_w + z * twist_y
    else:
        without_long_w = w
        without_long_x = x

    if without_long_w < 0.0:
        without_long_w = -without_long_w
        without_long_x = -without_long_x
    angle = 2.0 * math.atan2(without_long_x, without_long_w)
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def generated_rigped_hinge_branch(capability: Any) -> int | None:
    """Resolve the authored FK lower-joint branch for the generated hidden solver."""

    solver_name = str(getattr(capability.native_ik.solver_owner.target, "name", ""))
    if solver_name in {"MCH_ForeArm.L", "MCH_ForeArm.R"}:
        axis_index = 0
        fallback = -1
    elif solver_name in {"MCH_Calf.L", "MCH_Calf.R"}:
        axis_index = 0
        fallback = 1
    else:
        return None

    quaternion = capability.fk_controls[1].target.matrix_basis.to_quaternion()
    if axis_index != 0:
        return None
    angle = generated_lower_hinge_x_angle(quaternion)
    # Only an effectively straight authored hinge is branch-ambiguous.
    # A wider dead-zone can misclassify a real near-straight bend onto the
    # opposite solver branch (for example Calf.L at -0.2349 degrees), which
    # produces millimeter-scale FK->IK residuals even though the pose is valid.
    if abs(angle) <= math.radians(0.01):
        return fallback
    return 1 if angle > 0.0 else -1


def initial_pole_solution(capability: Any) -> TwoBonePoleSolution | None:
    points = two_bone_world_points(capability.native_ik)
    if points is None:
        return None
    root, joint, end = points
    upper_length = math.dist(root, joint)
    lower_length = math.dist(joint, end)
    pole_distance = max(upper_length, lower_length, 1e-6)
    epsilon = max(1e-9, pole_distance * 1e-8)
    solution = solve_capability_pole_position(
        capability.native_ik,
        distance=pole_distance,
        epsilon=epsilon,
    )
    if solution.ok or solution.issue_code != "SINGULAR_TWO_BONE_POLE":
        return solution

    # A perfectly straight generated limb has no geometric bend plane, but the
    # Rigped generator already authors a stable pole target for that limb. Use
    # that authored direction as the singular fallback instead of inventing an
    # arbitrary axis.
    pole_target = capability.native_ik.pole_target
    if pole_target is None:
        return solution
    from mathutils import Vector
    root_v = Vector(root)
    joint_v = Vector(joint)
    end_v = Vector(end)
    axis = end_v - root_v
    if axis.length <= epsilon:
        return solution
    axis.normalize()
    pole_world = pole_target.owner_object.matrix_world @ pole_target.target.matrix.translation
    reference = pole_world - joint_v
    perpendicular = reference - axis * reference.dot(axis)
    if perpendicular.length <= epsilon:
        return solution
    perpendicular.normalize()
    position = joint_v + perpendicular * pole_distance
    return TwoBonePoleSolution(tuple(float(value) for value in position), None)


def derived_pole_angle(
    capability: Any,
    pole_world_position: tuple[float, float, float],
) -> float:
    from mathutils import Vector

    base_bone = capability.result_controls[0].target
    tip_bone = capability.result_controls[1].target
    owner_inverse = capability.native_ik.solver_owner.owner_object.matrix_world.inverted()
    pole_position = owner_inverse @ Vector(pole_world_position)
    chain_axis = tip_bone.tail - base_bone.head
    pole_normal = chain_axis.cross(pole_position - base_bone.head)
    projected_pole_axis = pole_normal.cross(base_bone.tail - base_bone.head)
    if (
        chain_axis.length <= 1e-9
        or pole_normal.length <= 1e-9
        or projected_pole_axis.length <= 1e-9
    ):
        raise RigpedSolverGeometryError(
            "I12_SINGULAR_POLE_ANGLE: cannot derive a stable pole angle."
        )

    u = base_bone.x_axis.normalized()
    v = projected_pole_axis.normalized()
    axis = (base_bone.tail - base_bone.head).normalized()
    signed_angle = math.atan2(
        float(axis.dot(u.cross(v))),
        float(u.dot(v)),
    )
    return -signed_angle


def canonical_generated_rigped_pole(
    capability: Any,
    pole_world_position: tuple[float, float, float],
    pole_angle: float,
    branch_sign: int | None,
) -> tuple[tuple[float, float, float], float]:
    """Preserve the pole gauge derived from the evaluated generated limb."""

    del capability, branch_sign
    return tuple(float(value) for value in pole_world_position), float(pole_angle)
