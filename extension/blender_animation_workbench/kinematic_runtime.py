from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from typing import TYPE_CHECKING, Any

from .character_model import KinematicMapping
from .semantic_adapter import ResolvedControl, runtime_control_key
from .semantic_model import AWBControlKind

if TYPE_CHECKING:
    from .character_metadata import ResolvedCharacter


class KinematicRuntimeSeverity(StrEnum):
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class KinematicRuntimeIssue:
    code: str
    severity: KinematicRuntimeSeverity
    character_id: str | None = None
    mapping_id: str | None = None
    record_id: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class NativeIKCapability:
    """Operation-scoped native Blender IK capability for one authored mapping.

    Blender RNA references in this result are valid only for the current
    operation. They must never be persisted across Undo/load/delete/reload.
    """

    character_id: str
    mapping_id: str
    driven_chain_id: str
    driven_binding_ids: tuple[str, ...]
    driven_controls: tuple[ResolvedControl, ...]
    ik_target: ResolvedControl
    pole_target: ResolvedControl | None
    solver_owner: ResolvedControl
    constraint: Any
    source_stamp: tuple


@dataclass(frozen=True, slots=True)
class NativeIKResolution:
    capability: NativeIKCapability | None
    issues: tuple[KinematicRuntimeIssue, ...]


@dataclass(frozen=True, slots=True)
class EvaluatedPoseSample:
    binding_id: str
    matrix: Any
    world_matrix: Any


@dataclass(frozen=True, slots=True)
class TwoBonePoleSolution:
    position: tuple[float, float, float] | None
    issue_code: str | None = None

    @property
    def ok(self) -> bool:
        return self.position is not None and self.issue_code is None


def _issue(
    view: ResolvedCharacter,
    mapping_id: str | None,
    code: str,
    detail: str,
    *,
    record_id: str | None = None,
) -> KinematicRuntimeIssue:
    return KinematicRuntimeIssue(
        code=code,
        severity=KinematicRuntimeSeverity.ERROR,
        character_id=view.definition.character_id or None,
        mapping_id=mapping_id,
        record_id=record_id,
        detail=detail,
    )


def _mapping_by_id(view: ResolvedCharacter, mapping_id: str) -> KinematicMapping | None:
    return next(
        (
            mapping
            for mapping in view.definition.kinematics
            if mapping.mapping_id == mapping_id
        ),
        None,
    )


def _chain_by_id(view: ResolvedCharacter, chain_id: str):
    return next(
        (chain for chain in view.definition.chains if chain.chain_id == chain_id),
        None,
    )


def _resolved_by_binding(view: ResolvedCharacter) -> dict[str, ResolvedControl]:
    return {
        str(binding_id): resolved
        for binding_id, resolved in view.resolved_bindings
    }


def _object_pointer(value) -> int | None:
    if value is None:
        return None
    pointer = getattr(value, "as_pointer", None)
    if not callable(pointer):
        return None
    result = int(pointer())
    return result if result else None


def _same_object(left, right) -> bool:
    if left is right:
        return True
    left_pointer = _object_pointer(left)
    right_pointer = _object_pointer(right)
    return (
        left_pointer is not None
        and right_pointer is not None
        and left_pointer == right_pointer
    )


def _constraint_target_matches(
    resolved: ResolvedControl,
    target_object,
    subtarget: str,
) -> bool:
    if resolved.control.kind == AWBControlKind.OBJECT:
        return _same_object(target_object, resolved.target) and not str(subtarget or "")
    if resolved.control.kind == AWBControlKind.BONE:
        return (
            _same_object(target_object, resolved.owner_object)
            and str(subtarget or "") == str(getattr(resolved.target, "name", ""))
        )
    return False


def _constraint_pole_is_empty(constraint) -> bool:
    return getattr(constraint, "pole_target", None) is None and not str(
        getattr(constraint, "pole_subtarget", "") or ""
    )


def _constraint_pole_matches(constraint, resolved: ResolvedControl) -> bool:
    return _constraint_target_matches(
        resolved,
        getattr(constraint, "pole_target", None),
        str(getattr(constraint, "pole_subtarget", "") or ""),
    )


def _iter_native_ik_constraints(owner_object) -> Iterable[tuple[Any, Any]]:
    pose = getattr(owner_object, "pose", None)
    bones = getattr(pose, "bones", ()) if pose is not None else ()
    for pose_bone in bones:
        for constraint in getattr(pose_bone, "constraints", ()):
            if str(getattr(constraint, "type", "")) == "IK":
                yield pose_bone, constraint


def resolve_native_ik_capability(
    view: ResolvedCharacter,
    mapping_id: str,
) -> NativeIKResolution:
    """Resolve one authored mapping to one exact native Blender IK constraint.

    The function is read-only. It performs no key insertion, selection mutation,
    Character mutation, constraint edit, or name-pattern fallback.
    """

    mapping_id = str(mapping_id).strip()
    mapping = _mapping_by_id(view, mapping_id)
    if mapping is None:
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id or None,
                    "MISSING_KINEMATIC_MAPPING",
                    f"Kinematic mapping {mapping_id!r} does not exist.",
                    record_id=mapping_id or None,
                ),
            ),
        )

    driven_chain_id = mapping.reference_chain_id or mapping.fk_chain_id
    if not driven_chain_id:
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping.mapping_id,
                    "MISSING_KINEMATIC_DRIVEN_CHAIN",
                    "Native IK capability requires a reference chain or FK chain.",
                    record_id=mapping.mapping_id,
                ),
            ),
        )

    chain = _chain_by_id(view, driven_chain_id)
    if chain is None:
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping.mapping_id,
                    "MISSING_KINEMATIC_CHAIN_TARGET",
                    f"Driven chain {driven_chain_id!r} does not exist.",
                    record_id=driven_chain_id,
                ),
            ),
        )

    resolved = _resolved_by_binding(view)
    issues: list[KinematicRuntimeIssue] = []
    driven_controls: list[ResolvedControl] = []
    for binding_id in chain.members:
        target = resolved.get(binding_id)
        if target is None:
            issues.append(
                _issue(
                    view,
                    mapping.mapping_id,
                    "UNRESOLVED_KINEMATIC_BINDING",
                    f"Driven-chain binding {binding_id!r} has no resolved native target.",
                    record_id=binding_id,
                )
            )
            continue
        if target.control.kind != AWBControlKind.BONE:
            issues.append(
                _issue(
                    view,
                    mapping.mapping_id,
                    "NATIVE_IK_REQUIRES_POSE_CHAIN",
                    f"Driven-chain binding {binding_id!r} is not a PoseBone control.",
                    record_id=binding_id,
                )
            )
            continue
        driven_controls.append(target)

    if issues:
        return NativeIKResolution(None, tuple(issues))
    if not driven_controls:
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping.mapping_id,
                    "MISSING_KINEMATIC_CHAIN_TARGET",
                    "Driven chain contains no resolved PoseBone controls.",
                    record_id=driven_chain_id,
                ),
            ),
        )

    owner_by_pointer: dict[int, Any] = {}
    for target in driven_controls:
        pointer = _object_pointer(target.owner_object)
        if pointer is None:
            pointer = id(target.owner_object)
        owner_by_pointer[pointer] = target.owner_object
    if len(owner_by_pointer) != 1:
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping.mapping_id,
                    "NATIVE_IK_MULTIPLE_ARMATURE_OWNERS",
                    "Native IK driven-chain controls must belong to one Armature Object.",
                    record_id=driven_chain_id,
                ),
            ),
        )
    owner_object = next(iter(owner_by_pointer.values()))

    if not mapping.ik_target_binding_id:
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping.mapping_id,
                    "MISSING_IK_TARGET",
                    "Native IK capability requires an authored IK target binding.",
                    record_id=mapping.mapping_id,
                ),
            ),
        )
    ik_target = resolved.get(mapping.ik_target_binding_id)
    if ik_target is None:
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping.mapping_id,
                    "MISSING_IK_TARGET",
                    f"IK target binding {mapping.ik_target_binding_id!r} is unresolved.",
                    record_id=mapping.ik_target_binding_id,
                ),
            ),
        )

    pole_target = None
    if mapping.pole_binding_id:
        pole_target = resolved.get(mapping.pole_binding_id)
        if pole_target is None:
            return NativeIKResolution(
                None,
                (
                    _issue(
                        view,
                        mapping.mapping_id,
                        "MISSING_POLE_TARGET",
                        f"Pole binding {mapping.pole_binding_id!r} is unresolved.",
                        record_id=mapping.pole_binding_id,
                    ),
                ),
            )

    driven_keys = {runtime_control_key(target) for target in driven_controls}
    exact: list[tuple[Any, Any, ResolvedControl]] = []
    outside_chain: list[tuple[Any, Any]] = []
    un_authored_pole: list[tuple[Any, Any]] = []
    pole_mismatch: list[tuple[Any, Any]] = []

    driven_by_key = {runtime_control_key(target): target for target in driven_controls}
    for pose_bone, constraint in _iter_native_ik_constraints(owner_object):
        if not _constraint_target_matches(
            ik_target,
            getattr(constraint, "target", None),
            str(getattr(constraint, "subtarget", "") or ""),
        ):
            continue

        if pole_target is None:
            if not _constraint_pole_is_empty(constraint):
                un_authored_pole.append((pose_bone, constraint))
                continue
        elif not _constraint_pole_matches(constraint, pole_target):
            pole_mismatch.append((pose_bone, constraint))
            continue

        pose_key = (_object_pointer(owner_object), _object_pointer(pose_bone))
        if None in pose_key:
            pose_key = (id(owner_object), id(pose_bone))
        if pose_key not in driven_keys:
            outside_chain.append((pose_bone, constraint))
            continue

        solver_owner = driven_by_key[pose_key]
        exact.append((pose_bone, constraint, solver_owner))

    if len(exact) > 1:
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping.mapping_id,
                    "AMBIGUOUS_NATIVE_IK",
                    "Multiple native IK constraints match the authored target/pole mapping.",
                    record_id=mapping.mapping_id,
                ),
            ),
        )
    if not exact:
        if outside_chain:
            code = "NATIVE_IK_OWNER_OUTSIDE_CHAIN"
            detail = "A matching native IK constraint exists, but its owner is outside the authored driven chain."
        elif un_authored_pole:
            code = "NATIVE_IK_POLE_NOT_AUTHORED"
            detail = "The matching native IK constraint uses a pole target that is not authored in the KinematicMapping."
        elif pole_mismatch:
            code = "NATIVE_IK_POLE_MISMATCH"
            detail = "The native IK constraint pole target does not match the authored pole binding."
        else:
            code = "NATIVE_IK_NOT_FOUND"
            detail = "No native IK constraint exactly matches the authored target/pole mapping."
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping.mapping_id,
                    code,
                    detail,
                    record_id=mapping.mapping_id,
                ),
            ),
        )

    _pose_bone, constraint, solver_owner = exact[0]
    if bool(getattr(constraint, "use_rotation", False)):
        return NativeIKResolution(
            None,
            (
                _issue(
                    view,
                    mapping.mapping_id,
                    "NATIVE_IK_ROTATION_TARGET_UNSUPPORTED",
                    "Initial native IK snap support handles position targets only; target-rotation IK requires an explicit orientation-offset adapter.",
                    record_id=mapping.mapping_id,
                ),
            ),
        )

    capability = NativeIKCapability(
        character_id=view.definition.character_id,
        mapping_id=mapping.mapping_id,
        driven_chain_id=driven_chain_id,
        driven_binding_ids=tuple(chain.members),
        driven_controls=tuple(driven_controls),
        ik_target=ik_target,
        pole_target=pole_target,
        solver_owner=solver_owner,
        constraint=constraint,
        source_stamp=view.source_stamp,
    )
    return NativeIKResolution(capability, ())


def capture_evaluated_chain_pose(
    capability: NativeIKCapability,
) -> tuple[EvaluatedPoseSample, ...]:
    """Copy current evaluated PoseBone matrices without retaining live matrix views."""

    samples: list[EvaluatedPoseSample] = []
    for binding_id, resolved in zip(
        capability.driven_binding_ids,
        capability.driven_controls,
        strict=True,
    ):
        matrix = resolved.target.matrix.copy()
        world_matrix = (resolved.owner_object.matrix_world @ matrix).copy()
        samples.append(EvaluatedPoseSample(binding_id, matrix, world_matrix))
    return tuple(samples)


def evaluated_chain_tip_world_position(
    capability: NativeIKCapability,
) -> tuple[float, float, float]:
    """Return the native IK chain-tip point in world space for later target snap."""

    pose_bone = capability.solver_owner.target
    local_point = pose_bone.tail if bool(getattr(capability.constraint, "use_tail", False)) else pose_bone.head
    world_point = capability.solver_owner.owner_object.matrix_world @ local_point
    return tuple(float(value) for value in world_point[:3])


def _vec3(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) < 3:
        raise ValueError("Expected at least three coordinates.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(value, factor: float):
    return (value[0] * factor, value[1] * factor, value[2] * factor)


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(value) -> float:
    return sqrt(_dot(value, value))


def solve_two_bone_pole_position(
    root: Sequence[float],
    joint: Sequence[float],
    end: Sequence[float],
    distance: float,
    *,
    epsilon: float = 1e-6,
) -> TwoBonePoleSolution:
    """Solve a stable pole point from the current bend plane.

    Straight/near-straight chains are intentionally reported as singular. AWB
    must not invent an arbitrary pole direction because that can flip a limb.
    """

    root_v = _vec3(root)
    joint_v = _vec3(joint)
    end_v = _vec3(end)
    axis = _sub(end_v, root_v)
    axis_len_sq = _dot(axis, axis)
    tolerance = abs(float(epsilon))
    if axis_len_sq <= tolerance * tolerance:
        return TwoBonePoleSolution(None, "SINGULAR_TWO_BONE_POLE")

    relative_joint = _sub(joint_v, root_v)
    projection = _add(root_v, _scale(axis, _dot(relative_joint, axis) / axis_len_sq))
    bend = _sub(joint_v, projection)
    bend_length = _length(bend)
    if bend_length <= tolerance:
        return TwoBonePoleSolution(None, "SINGULAR_TWO_BONE_POLE")

    pole_distance = abs(float(distance))
    if pole_distance <= tolerance:
        return TwoBonePoleSolution(None, "SINGULAR_TWO_BONE_POLE")
    direction = _scale(bend, 1.0 / bend_length)
    pole = _add(joint_v, _scale(direction, pole_distance))
    return TwoBonePoleSolution(tuple(float(value) for value in pole), None)


def two_bone_world_points(
    capability: NativeIKCapability,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
] | None:
    """Return root/joint/end world points for an authored two-bone chain."""

    if len(capability.driven_controls) != 2:
        return None
    first, second = capability.driven_controls
    owner = first.owner_object
    root = owner.matrix_world @ first.target.head
    joint = owner.matrix_world @ first.target.tail
    end = owner.matrix_world @ second.target.tail
    return (
        tuple(float(value) for value in root[:3]),
        tuple(float(value) for value in joint[:3]),
        tuple(float(value) for value in end[:3]),
    )


def solve_capability_pole_position(
    capability: NativeIKCapability,
    distance: float,
    *,
    epsilon: float = 1e-6,
) -> TwoBonePoleSolution:
    points = two_bone_world_points(capability)
    if points is None:
        return TwoBonePoleSolution(None, "UNSUPPORTED_TWO_BONE_POLE_CHAIN")
    return solve_two_bone_pole_position(*points, distance, epsilon=epsilon)
