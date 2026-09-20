from __future__ import annotations

from dataclasses import dataclass
from math import cos, sqrt
from typing import Any

from mathutils import Matrix, Vector


@dataclass(frozen=True, slots=True)
class SlidingAuthoredReference:
    """Persistent/native inputs that define one Sliding evaluation."""

    mapping_id: str
    target_matrix: Matrix
    pole_matrix: Matrix | None
    pole_angle: float


@dataclass(frozen=True, slots=True)
class SlidingSolverSeed:
    """Transient two-bone initialization only; never a final solved result."""

    root_world: tuple[float, float, float]
    joint_world: tuple[float, float, float]
    end_world: tuple[float, float, float]
    target_world: tuple[float, float, float]
    bend_world: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SlidingSolvedResult:
    """Matrices read from Blender's native evaluated result chain."""

    first_pose: Matrix
    second_pose: Matrix
    terminal_pose: Matrix


@dataclass(frozen=True, slots=True)
class SlidingPublicDisplay:
    """Derived public channel bases reconstructed only from a native result."""

    first_basis: Matrix
    second_basis: Matrix
    terminal_basis: Matrix


def native_pose_basis_from_matrix(
    pose_bone: Any,
    desired_matrix: Matrix,
    *,
    parent_pose_matrix: Matrix | None = None,
) -> Matrix:
    """Convert a pose-space result to channel basis using Blender bone semantics."""

    rest = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        return pose_bone.bone.convert_local_to_pose(
            desired_matrix,
            rest,
            invert=True,
        )

    parent_pose = (
        parent_pose_matrix.copy()
        if parent_pose_matrix is not None
        else pose_bone.parent.matrix.copy()
    )
    return pose_bone.bone.convert_local_to_pose(
        desired_matrix,
        rest,
        parent_matrix=parent_pose,
        parent_matrix_local=pose_bone.parent.bone.matrix_local,
        invert=True,
    )


def native_pose_matrix_from_basis(
    pose_bone: Any,
    basis: Matrix,
    *,
    parent_pose_matrix: Matrix | None = None,
) -> Matrix:
    """Convert channel basis to pose space using Blender bone inheritance rules."""

    rest = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        return pose_bone.bone.convert_local_to_pose(
            basis,
            rest,
        )

    parent_pose = (
        parent_pose_matrix.copy()
        if parent_pose_matrix is not None
        else pose_bone.parent.matrix.copy()
    )
    return pose_bone.bone.convert_local_to_pose(
        basis,
        rest,
        parent_matrix=parent_pose,
        parent_matrix_local=pose_bone.parent.bone.matrix_local,
    )


def capture_sliding_authored_reference(capability: Any) -> SlidingAuthoredReference:
    """Read authored target/pole authority without mutating Blender state."""

    native_ik = capability.native_ik
    pole_target = native_ik.pole_target
    return SlidingAuthoredReference(
        mapping_id=str(native_ik.mapping_id),
        target_matrix=native_ik.ik_target.target.matrix.copy(),
        pole_matrix=(
            pole_target.target.matrix.copy()
            if pole_target is not None
            else None
        ),
        pole_angle=float(native_ik.constraint.pole_angle),
    )


def capture_native_solved_result(capability: Any) -> SlidingSolvedResult:
    """Read exactly one Blender-native result snapshot."""

    return SlidingSolvedResult(
        first_pose=capability.result_controls[0].target.matrix.copy(),
        second_pose=capability.result_controls[1].target.matrix.copy(),
        terminal_pose=capability.result_terminal.target.matrix.copy(),
    )


def derive_public_display(
    capability: Any,
    solved_result: SlidingSolvedResult,
) -> SlidingPublicDisplay:
    """Derive public channel bases from native solved matrices only."""

    first = capability.fk_controls[0].target
    second = capability.fk_controls[1].target
    terminal = capability.authored_terminal.target
    first_basis = native_pose_basis_from_matrix(
        first,
        solved_result.first_pose,
    )
    second_basis = native_pose_basis_from_matrix(
        second,
        solved_result.second_pose,
        parent_pose_matrix=solved_result.first_pose,
    )
    terminal_basis = native_pose_basis_from_matrix(
        terminal,
        solved_result.terminal_pose,
        parent_pose_matrix=solved_result.second_pose,
    )
    return SlidingPublicDisplay(
        first_basis=first_basis,
        second_basis=second_basis,
        terminal_basis=terminal_basis,
    )


def build_transient_solver_seed(
    *,
    root_world: Vector,
    target_world: Vector,
    preferred_bend_world: Vector,
    first_length: float,
    second_length: float,
    minimum_bend_radians: float,
) -> SlidingSolverSeed | None:
    """Build a deterministic non-singular two-bone seed without moving the target.

    The returned end point may sit slightly inside the authored target when the
    target is at full extension. That mismatch is intentional: Blender native IK
    must consume this seed and solve to the unchanged target. This function never
    mutates a pose, target, pole, constraint, key, or animation datum.
    """

    first = float(first_length)
    second = float(second_length)
    if first <= 1e-9 or second <= 1e-9:
        return None

    root = Vector(root_world)
    target = Vector(target_world)
    direction = target - root
    target_distance = float(direction.length)
    if target_distance <= 1e-9:
        return None
    direction.normalize()

    bend = Vector(preferred_bend_world)
    bend -= direction * float(bend.dot(direction))
    if bend.length <= 1e-9:
        return None
    bend.normalize()

    reach_epsilon = max(1e-6, (first + second) * 1e-6)
    minimum_distance = abs(first - second) + reach_epsilon
    bend_angle = max(1e-6, min(float(minimum_bend_radians), 3.141592653589793 - 1e-6))
    preferred_maximum = sqrt(
        max(
            0.0,
            first * first
            + second * second
            + 2.0 * first * second * cos(bend_angle),
        )
    )
    maximum_distance = max(minimum_distance, min(first + second, preferred_maximum))
    seed_distance = max(minimum_distance, min(maximum_distance, target_distance))
    seed_end = root + direction * seed_distance

    along = (
        first * first
        - second * second
        + seed_distance * seed_distance
    ) / (2.0 * seed_distance)
    height_sq = max(0.0, first * first - along * along)
    height = sqrt(height_sq)
    if height <= 1e-9:
        return None

    seed_joint = root + direction * along + bend * height
    return SlidingSolverSeed(
        root_world=tuple(float(value) for value in root),
        joint_world=tuple(float(value) for value in seed_joint),
        end_world=tuple(float(value) for value in seed_end),
        target_world=tuple(float(value) for value in target),
        bend_world=tuple(float(value) for value in bend),
    )
