from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .character_model import ControlUsage
from .kinematic_runtime import (
    KinematicRuntimeIssue,
    KinematicRuntimeSeverity,
    NativeIKCapability,
    TwoBonePoleSolution,
    resolve_native_ik_capability,
    solve_capability_pole_position,
    two_bone_world_points,
)
from .semantic_adapter import ResolvedControl, rotation_property, runtime_control_key
from .semantic_model import AWBControlKind


class SnapDirection(StrEnum):
    FK_TO_IK = "FK_TO_IK"
    IK_TO_FK = "IK_TO_FK"


@dataclass(frozen=True, slots=True)
class LimbRepresentationCapability:
    """Operation-local generated-limb correspondence for I12 transient snap proof.

    All RNA references are operation-local only. Persistent relationship authority
    remains the Phase 3 Character graph; this result is a read-only resolution.
    """

    native_ik: NativeIKCapability
    fk_binding_ids: tuple[str, ...]
    fk_controls: tuple[ResolvedControl, ...]
    result_binding_ids: tuple[str, ...]
    result_controls: tuple[ResolvedControl, ...]
    authored_terminal_binding_id: str
    authored_terminal: ResolvedControl
    result_terminal_binding_id: str
    result_terminal: ResolvedControl
    fk_copy_constraints: tuple[Any, ...]
    terminal_fk_constraint: Any
    terminal_ik_constraint: Any


@dataclass(frozen=True, slots=True)
class LimbRepresentationResolution:
    capability: LimbRepresentationCapability | None
    issues: tuple[KinematicRuntimeIssue, ...]

    @property
    def ok(self) -> bool:
        return self.capability is not None and not self.issues


@dataclass(frozen=True, slots=True)
class RepresentationSnapPayload:
    """I12 snap-specific dependency identity layered on existing OperationPlan.

    Character/setup/selection/animation freshness remains owned by OperationPlan.
    This payload freezes only the additional FK/result/terminal native identity
    and constraint correspondence needed by the transient representation proof.
    """

    mapping_id: str
    direction: SnapDirection
    fk_control_runtime_keys: tuple[tuple[int, int], ...]
    result_control_runtime_keys: tuple[tuple[int, int], ...]
    authored_terminal_binding_id: str
    authored_terminal_runtime_key: tuple[int, int]
    result_terminal_binding_id: str
    result_terminal_runtime_key: tuple[int, int]
    fk_copy_constraint_tokens: tuple[int, ...]
    terminal_fk_constraint_token: int
    terminal_ik_constraint_token: int


@dataclass(frozen=True, slots=True)
class SnapResiduals:
    max_position_error: float
    max_rotation_error: float
    terminal_position_error: float
    terminal_rotation_error: float


def _pointer(value: Any) -> int | None:
    if value is None:
        return None
    as_pointer = getattr(value, "as_pointer", None)
    if not callable(as_pointer):
        return None
    try:
        pointer = int(as_pointer())
    except ReferenceError:
        return None
    return pointer or None


def _same_native(left: Any, right: Any) -> bool:
    if left is right:
        return True
    left_pointer = _pointer(left)
    right_pointer = _pointer(right)
    return (
        left_pointer is not None
        and right_pointer is not None
        and left_pointer == right_pointer
    )


def _issue(view, mapping_id: str | None, code: str, detail: str, *, record_id: str | None = None):
    return KinematicRuntimeIssue(
        code=code,
        severity=KinematicRuntimeSeverity.ERROR,
        character_id=view.definition.character_id or None,
        mapping_id=mapping_id,
        record_id=record_id,
        detail=detail,
    )


def _mapping(view, mapping_id: str):
    return next(
        (mapping for mapping in view.definition.kinematics if mapping.mapping_id == mapping_id),
        None,
    )


def _chain(view, chain_id: str | None):
    if not chain_id:
        return None
    return next(
        (chain for chain in view.definition.chains if chain.chain_id == chain_id),
        None,
    )


def _constraint_targets_control(constraint: Any, resolved: ResolvedControl) -> bool:
    target = getattr(constraint, "target", None)
    subtarget = str(getattr(constraint, "subtarget", "") or "")
    if resolved.control.kind == AWBControlKind.OBJECT:
        return _same_native(target, resolved.target) and not subtarget
    if resolved.control.kind == AWBControlKind.BONE:
        return (
            _same_native(target, resolved.owner_object)
            and subtarget == str(getattr(resolved.target, "name", ""))
        )
    return False


def _exact_copy_rotation(owner: ResolvedControl, target: ResolvedControl) -> tuple[Any | None, str | None]:
    matches = tuple(
        constraint
        for constraint in getattr(owner.target, "constraints", ())
        if str(getattr(constraint, "type", "")) == "COPY_ROTATION"
        and _constraint_targets_control(constraint, target)
    )
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, "MISSING_REPRESENTATION_COPY_ROTATION"
    return None, "AMBIGUOUS_REPRESENTATION_COPY_ROTATION"


def _resolved_by_binding(view) -> dict[str, ResolvedControl]:
    return {str(binding_id): resolved for binding_id, resolved in view.resolved_bindings}


def resolve_limb_representation_capability(
    view,
    mapping_id: str,
) -> LimbRepresentationResolution:
    """Resolve exact generated FK/result/terminal correspondence without names.

    This I12 resolver deliberately supports only the generated two-bone limb
    contract. Custom/partial rigs remain legal Character data but fail closed
    until they provide an equally explicit correspondence.
    """

    mapping_id = str(mapping_id).strip()
    mapping = _mapping(view, mapping_id)
    if mapping is None:
        return LimbRepresentationResolution(
            None,
            (_issue(view, mapping_id or None, "MISSING_KINEMATIC_MAPPING", f"Kinematic mapping {mapping_id!r} does not exist."),),
        )

    native_resolution = resolve_native_ik_capability(view, mapping_id)
    if native_resolution.capability is None:
        return LimbRepresentationResolution(None, native_resolution.issues)
    native_ik = native_resolution.capability

    fk_chain = _chain(view, mapping.fk_chain_id)
    result_chain = _chain(view, mapping.reference_chain_id)
    if fk_chain is None or result_chain is None:
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "REPRESENTATION_CHAIN_INCOMPLETE",
                    "Transient FK/IK snap requires both authored FK and result/reference chains.",
                    record_id=mapping_id,
                ),
            ),
        )
    if len(fk_chain.members) != 2 or len(result_chain.members) != 2:
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "UNSUPPORTED_REPRESENTATION_CHAIN_TOPOLOGY",
                    "Initial transient snap supports matched two-bone FK/result chains only.",
                    record_id=mapping_id,
                ),
            ),
        )

    resolved = _resolved_by_binding(view)
    fk_controls = tuple(resolved.get(binding_id) for binding_id in fk_chain.members)
    result_controls = tuple(resolved.get(binding_id) for binding_id in result_chain.members)
    if any(control is None for control in (*fk_controls, *result_controls)):
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "UNRESOLVED_REPRESENTATION_CHAIN",
                    "FK/result correspondence contains an unresolved native control.",
                    record_id=mapping_id,
                ),
            ),
        )

    fk_controls = tuple(control for control in fk_controls if control is not None)
    result_controls = tuple(control for control in result_controls if control is not None)
    if any(control.control.kind != AWBControlKind.BONE for control in (*fk_controls, *result_controls)):
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "REPRESENTATION_REQUIRES_POSE_BONES",
                    "Initial FK/result correspondence requires PoseBone controls.",
                    record_id=mapping_id,
                ),
            ),
        )

    owner = result_controls[0].owner_object
    if any(not _same_native(control.owner_object, owner) for control in (*fk_controls, *result_controls)):
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "REPRESENTATION_MULTIPLE_ARMATURE_OWNERS",
                    "Initial generated limb snap requires FK/result controls on one Armature Object.",
                    record_id=mapping_id,
                ),
            ),
        )

    binding_by_id = {binding.binding_id: binding for binding in view.definition.bindings}
    ik_binding = binding_by_id.get(mapping.ik_target_binding_id or "")
    if ik_binding is None:
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "REPRESENTATION_IK_TARGET_BINDING_MISSING",
                    "The authored IK target binding is missing from the Character graph.",
                    record_id=mapping.ik_target_binding_id or mapping_id,
                ),
            ),
        )

    result_terminal_candidates = tuple(
        binding
        for binding_id in mapping.extras
        if (binding := binding_by_id.get(binding_id)) is not None
        and binding.semantic_key == ik_binding.semantic_key
        and binding.side == ik_binding.side
        and binding.usage == ControlUsage.MECHANISM
    )
    if len(result_terminal_candidates) != 1:
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "AMBIGUOUS_RESULT_TERMINAL_CORRESPONDENCE",
                    "Kinematic extras must resolve exactly one mechanism terminal matching the IK target semantic role and side.",
                    record_id=mapping_id,
                ),
            ),
        )
    result_terminal_binding = result_terminal_candidates[0]

    authored_terminal_candidates = tuple(
        binding
        for binding in view.definition.bindings
        if binding.semantic_key == result_terminal_binding.semantic_key
        and binding.side == result_terminal_binding.side
        and binding.usage == ControlUsage.PRIMARY
        and binding.kind == AWBControlKind.BONE
    )
    if len(authored_terminal_candidates) != 1:
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "AMBIGUOUS_AUTHORED_TERMINAL_CORRESPONDENCE",
                    "The Character graph must resolve exactly one authored primary terminal matching the result terminal semantic role and side.",
                    record_id=mapping_id,
                ),
            ),
        )
    authored_terminal_binding = authored_terminal_candidates[0]

    result_terminal = resolved.get(result_terminal_binding.binding_id)
    authored_terminal = resolved.get(authored_terminal_binding.binding_id)
    if result_terminal is None or authored_terminal is None:
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "UNRESOLVED_TERMINAL_CORRESPONDENCE",
                    "Resolved native controls are missing for the authored/result terminal pair.",
                    record_id=mapping_id,
                ),
            ),
        )
    if result_terminal.control.kind != AWBControlKind.BONE or authored_terminal.control.kind != AWBControlKind.BONE:
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "REPRESENTATION_TERMINAL_REQUIRES_POSE_BONES",
                    "Initial terminal correspondence requires authored/result PoseBones.",
                    record_id=mapping_id,
                ),
            ),
        )
    if any(
        not _same_native(control.owner_object, owner)
        for control in (result_terminal, authored_terminal, native_ik.ik_target)
    ):
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    "REPRESENTATION_TERMINAL_OWNER_MISMATCH",
                    "Initial generated terminal/result/IK target must belong to the same Armature Object.",
                    record_id=mapping_id,
                ),
            ),
        )

    fk_copy_constraints: list[Any] = []
    for result_control, fk_control in zip(result_controls, fk_controls, strict=True):
        constraint, code = _exact_copy_rotation(result_control, fk_control)
        if constraint is None:
            return LimbRepresentationResolution(
                None,
                (
                    _issue(
                        view,
                        mapping_id,
                        code or "REPRESENTATION_COPY_ROTATION_UNRESOLVED",
                        "Each generated result-chain member must have exactly one Copy Rotation constraint from its paired authored FK control.",
                        record_id=mapping_id,
                    ),
                ),
            )
        fk_copy_constraints.append(constraint)

    terminal_fk_constraint, terminal_fk_code = _exact_copy_rotation(result_terminal, authored_terminal)
    if terminal_fk_constraint is None:
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    terminal_fk_code or "TERMINAL_FK_COPY_ROTATION_UNRESOLVED",
                    "Result terminal must have exactly one Copy Rotation constraint from the authored terminal.",
                    record_id=result_terminal_binding.binding_id,
                ),
            ),
        )
    terminal_ik_constraint, terminal_ik_code = _exact_copy_rotation(result_terminal, native_ik.ik_target)
    if terminal_ik_constraint is None:
        return LimbRepresentationResolution(
            None,
            (
                _issue(
                    view,
                    mapping_id,
                    terminal_ik_code or "TERMINAL_IK_COPY_ROTATION_UNRESOLVED",
                    "Result terminal must have exactly one Copy Rotation constraint from the authored IK target.",
                    record_id=result_terminal_binding.binding_id,
                ),
            ),
        )

    return LimbRepresentationResolution(
        LimbRepresentationCapability(
            native_ik=native_ik,
            fk_binding_ids=tuple(fk_chain.members),
            fk_controls=fk_controls,
            result_binding_ids=tuple(result_chain.members),
            result_controls=result_controls,
            authored_terminal_binding_id=authored_terminal_binding.binding_id,
            authored_terminal=authored_terminal,
            result_terminal_binding_id=result_terminal_binding.binding_id,
            result_terminal=result_terminal,
            fk_copy_constraints=tuple(fk_copy_constraints),
            terminal_fk_constraint=terminal_fk_constraint,
            terminal_ik_constraint=terminal_ik_constraint,
        ),
        (),
    )


def build_representation_snap_payload(
    capability: LimbRepresentationCapability,
    direction: SnapDirection,
) -> RepresentationSnapPayload:
    fk_tokens = tuple(_pointer(constraint) for constraint in capability.fk_copy_constraints)
    terminal_fk = _pointer(capability.terminal_fk_constraint)
    terminal_ik = _pointer(capability.terminal_ik_constraint)
    if any(token is None for token in fk_tokens) or terminal_fk is None or terminal_ik is None:
        raise ValueError("Representation snap constraints require live runtime identities.")
    return RepresentationSnapPayload(
        mapping_id=capability.native_ik.mapping_id,
        direction=SnapDirection(direction),
        fk_control_runtime_keys=tuple(
            runtime_control_key(control) for control in capability.fk_controls
        ),
        result_control_runtime_keys=tuple(
            runtime_control_key(control) for control in capability.result_controls
        ),
        authored_terminal_binding_id=capability.authored_terminal_binding_id,
        authored_terminal_runtime_key=runtime_control_key(capability.authored_terminal),
        result_terminal_binding_id=capability.result_terminal_binding_id,
        result_terminal_runtime_key=runtime_control_key(capability.result_terminal),
        fk_copy_constraint_tokens=tuple(int(token) for token in fk_tokens if token is not None),
        terminal_fk_constraint_token=int(terminal_fk),
        terminal_ik_constraint_token=int(terminal_ik),
    )


def representation_payload_matches(
    payload: RepresentationSnapPayload,
    capability: LimbRepresentationCapability,
) -> bool:
    try:
        fresh = build_representation_snap_payload(capability, payload.direction)
    except ValueError:
        return False
    return fresh == payload


class RepresentationSnapError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SnapControlState:
    runtime_key: tuple[int, int]
    location: tuple[float, float, float]
    rotation_property: str
    rotation: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RepresentationSnapResult:
    success: bool
    residuals: SnapResiduals | None = None
    transient_writes: int = 0
    restored_exactly: bool = True
    diagnostics: tuple[Any, ...] = ()
    fk_control_states: tuple[SnapControlState, ...] = ()
    authored_terminal_state: SnapControlState | None = None
    ik_target_state: SnapControlState | None = None
    pole_target_state: SnapControlState | None = None
    pole_angle: float | None = None


def _snap_diagnostic(plan, code: str, detail: str):
    from .phase4_verification import Diagnostic, DiagnosticSeverity, OperationStage

    return Diagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        stage=OperationStage.PREFLIGHT,
        operation=plan.operation_id,
        detail=detail,
        character_id=plan.character_id,
        frame=float(plan.frame) + float(plan.subframe),
    )


def _matrix_error(actual, expected) -> tuple[float, float]:
    position = (actual.to_translation() - expected.to_translation()).length
    actual_q = actual.to_quaternion().normalized()
    expected_q = expected.to_quaternion().normalized()
    rotation = float(actual_q.rotation_difference(expected_q).angle)
    if rotation > math.pi:
        rotation = abs((2.0 * math.pi) - rotation)
    return float(position), float(abs(rotation))


def _character_scale(capability: LimbRepresentationCapability) -> float:
    owner = capability.result_controls[0].owner_object
    dimensions = getattr(owner, "dimensions", None)
    length = float(getattr(dimensions, "length", 0.0) or 0.0)
    return max(1e-6, abs(length))


def _position_tolerance(scale: float) -> float:
    # Blender's evaluated two-bone IK can settle a few microunits away from the
    # FK sample even when the terminal position/orientation is preserved exactly.
    # Keep the gate scale-aware and strict, but allow that solver convergence
    # noise instead of rejecting valid Hand FK->IK snaps. Rotation tolerance stays
    # unchanged and remains the stronger orientation-continuity guard.
    return max(1e-7, abs(float(scale)) * 3.2e-6)


def _rotation_tolerance() -> float:
    return 1e-6


def _residuals_for_expected(
    capability: LimbRepresentationCapability,
    expected_result: tuple[Any, ...],
    expected_terminal: Any,
) -> SnapResiduals:
    position_errors: list[float] = []
    rotation_errors: list[float] = []
    for control, expected in zip(capability.result_controls, expected_result, strict=True):
        position, rotation = _matrix_error(control.target.matrix, expected)
        position_errors.append(position)
        rotation_errors.append(rotation)
    terminal_position, terminal_rotation = _matrix_error(
        capability.result_terminal.target.matrix,
        expected_terminal,
    )
    return SnapResiduals(
        max(position_errors, default=0.0),
        max(rotation_errors, default=0.0),
        terminal_position,
        terminal_rotation,
    )


def _residuals_within_tolerance(
    residuals: SnapResiduals,
    *,
    scale: float,
) -> bool:
    position_tolerance = _position_tolerance(scale)
    rotation_tolerance = _rotation_tolerance()
    return (
        residuals.max_position_error <= position_tolerance
        and residuals.terminal_position_error <= position_tolerance
        and residuals.max_rotation_error <= rotation_tolerance
        and residuals.terminal_rotation_error <= rotation_tolerance
    )


def _signed_angle_on_axis(vector_u, vector_v, axis) -> float:
    u = vector_u.normalized()
    v = vector_v.normalized()
    n = axis.normalized()
    return math.atan2(float(n.dot(u.cross(v))), float(u.dot(v)))


def _derived_pole_angle(
    capability: LimbRepresentationCapability,
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
        raise RepresentationSnapError(
            "I12_SINGULAR_POLE_ANGLE: cannot derive a stable pole angle."
        )
    return -_signed_angle_on_axis(
        base_bone.x_axis,
        projected_pole_axis,
        base_bone.tail - base_bone.head,
    )


def _nearest_angle_about(angle: float, center: float) -> float:
    return float(center + ((float(angle) - center + math.pi) % (2.0 * math.pi) - math.pi))


def _generated_rigped_hinge_branch(capability: LimbRepresentationCapability) -> int | None:
    """Resolve the authored FK lower-joint branch for the generated hidden solver.

    The lower public FK control stores the local joint rotation.  Extract only
    the twist component around the generated hinge axis so unrelated swing/twist
    on the limb does not force a left/right hard-coded branch.
    """

    solver_name = str(getattr(capability.native_ik.solver_owner.target, "name", ""))
    if solver_name in {"MCH_ForeArm.L", "MCH_ForeArm.R"}:
        axis_index = 2
        fallback = -1 if solver_name.endswith(".R") else 1
    elif solver_name in {"MCH_Calf.L", "MCH_Calf.R"}:
        return 1
    else:
        return None

    quaternion = capability.fk_controls[1].target.matrix_basis.to_quaternion().normalized()
    components = (float(quaternion.x), float(quaternion.y), float(quaternion.z))
    component = components[axis_index]
    angle = 2.0 * math.atan2(component, float(quaternion.w))
    angle = ((angle + math.pi) % (2.0 * math.pi)) - math.pi
    if abs(angle) <= math.radians(0.25):
        return fallback
    return 1 if angle > 0.0 else -1


def _configure_generated_rigped_hinge_branch(owner, branch_sign: int) -> bool:
    solver_name = str(getattr(owner, "name", ""))
    if solver_name not in {
        "MCH_ForeArm.L",
        "MCH_ForeArm.R",
        "MCH_Calf.L",
        "MCH_Calf.R",
    }:
        return False
    sign = 1 if int(branch_sign) >= 0 else -1
    before = (
        bool(owner.lock_ik_x),
        bool(owner.lock_ik_y),
        bool(owner.lock_ik_z),
        bool(owner.use_ik_limit_x),
        bool(owner.use_ik_limit_y),
        bool(owner.use_ik_limit_z),
        float(owner.ik_min_x),
        float(owner.ik_max_x),
        float(owner.ik_min_z),
        float(owner.ik_max_z),
    )
    owner.lock_ik_x = False
    owner.lock_ik_y = False
    owner.lock_ik_z = False
    owner.use_ik_limit_x = False
    owner.use_ik_limit_y = False
    owner.use_ik_limit_z = False
    limit = math.radians(179.0)
    if solver_name.startswith("MCH_ForeArm"):
        owner.use_ik_limit_z = True
        owner.ik_min_z = 0.0 if sign > 0 else -limit
        owner.ik_max_z = limit if sign > 0 else 0.0
    else:
        owner.use_ik_limit_x = True
        owner.ik_min_x = 0.0 if sign > 0 else -limit
        owner.ik_max_x = limit if sign > 0 else 0.0
    after = (
        bool(owner.lock_ik_x),
        bool(owner.lock_ik_y),
        bool(owner.lock_ik_z),
        bool(owner.use_ik_limit_x),
        bool(owner.use_ik_limit_y),
        bool(owner.use_ik_limit_z),
        float(owner.ik_min_x),
        float(owner.ik_max_x),
        float(owner.ik_min_z),
        float(owner.ik_max_z),
    )
    return before != after


def _capture_generated_rigped_hinge_settings(owner) -> tuple[Any, ...]:
    return (
        bool(owner.lock_ik_x),
        bool(owner.lock_ik_y),
        bool(owner.lock_ik_z),
        bool(owner.use_ik_limit_x),
        bool(owner.use_ik_limit_y),
        bool(owner.use_ik_limit_z),
        float(owner.ik_min_x),
        float(owner.ik_max_x),
        float(owner.ik_min_y),
        float(owner.ik_max_y),
        float(owner.ik_min_z),
        float(owner.ik_max_z),
    )


def _restore_generated_rigped_hinge_settings(owner, snapshot: tuple[Any, ...]) -> None:
    (
        owner.lock_ik_x,
        owner.lock_ik_y,
        owner.lock_ik_z,
        owner.use_ik_limit_x,
        owner.use_ik_limit_y,
        owner.use_ik_limit_z,
        owner.ik_min_x,
        owner.ik_max_x,
        owner.ik_min_y,
        owner.ik_max_y,
        owner.ik_min_z,
        owner.ik_max_z,
    ) = snapshot


def _canonical_generated_rigped_pole(
    capability: LimbRepresentationCapability,
    pole_world_position: tuple[float, float, float],
    pole_angle: float,
    branch_sign: int | None,
) -> tuple[tuple[float, float, float], float]:
    """Choose the stable equivalent pole gauge for generated Rigped arms.

    Blender IK admits an equivalent representation obtained by mirroring the
    pole target across the root-terminal axis and shifting ``pole_angle`` by pi.
    Keep the gauge tied to the authored elbow branch rather than to left/right;
    this preserves valid FK poses that cross local hinge zero while keeping each
    Sliding branch in one stable half-turn gauge during playback.
    """

    from mathutils import Vector

    solver_name = str(getattr(capability.native_ik.solver_owner.target, "name", ""))
    if solver_name not in {"MCH_ForeArm.L", "MCH_ForeArm.R"} or branch_sign is None:
        return tuple(float(value) for value in pole_world_position), float(pole_angle)
    preferred_center = math.pi if int(branch_sign) > 0 else 0.0

    same_angle = _nearest_angle_about(float(pole_angle), preferred_center)
    mirrored_angle = _nearest_angle_about(float(pole_angle) + math.pi, preferred_center)
    if abs(same_angle - preferred_center) <= abs(mirrored_angle - preferred_center) + 1e-9:
        return tuple(float(value) for value in pole_world_position), same_angle

    points = two_bone_world_points(capability.native_ik)
    if points is None:
        return tuple(float(value) for value in pole_world_position), float(pole_angle)
    root, _joint, end = (Vector(point) for point in points)
    axis = end - root
    if axis.length <= 1e-9:
        return tuple(float(value) for value in pole_world_position), float(pole_angle)
    axis.normalize()
    offset = Vector(pole_world_position) - root
    axial = axis * float(offset.dot(axis))
    mirrored = root + axial - (offset - axial)
    return tuple(float(value) for value in mirrored), mirrored_angle


def _initial_pole_solution(capability: LimbRepresentationCapability):
    from mathutils import Vector

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
    # arbitrary axis. This keeps the default straight leg eligible for the first
    # Free -> Sliding Contact transition while preserving deterministic knee side.
    pole_target = capability.native_ik.pole_target
    if pole_target is None:
        return solution
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


def _constraint_value_is(value: Any, expected: float, *, epsilon: float = 1e-8) -> bool:
    return abs(float(value) - float(expected)) <= epsilon


def _capability_preflight_issue(
    capability: LimbRepresentationCapability,
    direction: SnapDirection,
) -> tuple[str, str] | None:
    constraint = capability.native_ik.constraint
    if int(getattr(constraint, "chain_count", 0)) != 2:
        return (
            "I12_UNSUPPORTED_CHAIN_COUNT",
            "Transient representation snap supports exactly a two-bone native IK chain.",
        )
    if bool(getattr(constraint, "use_rotation", False)):
        return (
            "I12_UNSUPPORTED_IK_ROTATION_TARGET",
            "Native IK target rotation solving is outside the initial I12 contract.",
        )
    if bool(getattr(constraint, "use_stretch", False)):
        return (
            "I12_UNSUPPORTED_IK_STRETCH",
            "Transient representation snap refuses native IK stretch in the initial exact domain.",
        )
    if capability.native_ik.pole_target is None:
        return (
            "I12_POLE_TARGET_REQUIRED",
            "Initial generated FK/IK representation snap requires an authored pole target.",
        )
    if any(not _constraint_value_is(item.influence, 1.0) for item in capability.fk_copy_constraints):
        return (
            "I12_FK_COPY_STATE_UNSUPPORTED",
            "Generated FK Copy Rotation constraints must remain fully active for twist/roll preservation.",
        )
    if not _constraint_value_is(capability.terminal_fk_constraint.influence, 1.0):
        return (
            "I12_TERMINAL_FK_STATE_UNSUPPORTED",
            "Generated terminal FK Copy Rotation must be fully active.",
        )
    if direction is SnapDirection.FK_TO_IK:
        if not _constraint_value_is(constraint.influence, 0.0):
            return (
                "I12_EXPECTED_FK_REPRESENTATION",
                "FK→IK snap requires the generated native IK constraint to start disabled.",
            )
        if not _constraint_value_is(capability.terminal_ik_constraint.influence, 0.0):
            return (
                "I12_EXPECTED_FK_TERMINAL_STATE",
                "FK→IK snap requires terminal IK orientation influence to start disabled.",
            )
    else:
        if not _constraint_value_is(constraint.influence, 1.0):
            return (
                "I12_EXPECTED_IK_REPRESENTATION",
                "IK→FK snap requires the generated native IK constraint to start enabled.",
            )
        if not _constraint_value_is(capability.terminal_ik_constraint.influence, 1.0):
            return (
                "I12_EXPECTED_IK_TERMINAL_STATE",
                "IK→FK snap requires terminal IK orientation influence to start enabled.",
            )
    return None


def _capture_snap_control_state(resolved: ResolvedControl) -> SnapControlState:
    target = resolved.target
    property_name = rotation_property(target)
    return SnapControlState(
        runtime_control_key(resolved),
        tuple(float(component) for component in target.location),
        property_name,
        tuple(float(component) for component in getattr(target, property_name)),
    )


def _raw_pose_fields():
    from .phase4_mutation_journal import RawFieldKind

    return (
        RawFieldKind.POSE_BONE_LOCATION,
        RawFieldKind.POSE_BONE_ROTATION_EULER,
        RawFieldKind.POSE_BONE_ROTATION_QUATERNION,
        RawFieldKind.POSE_BONE_ROTATION_AXIS_ANGLE,
        RawFieldKind.POSE_BONE_SCALE,
    )


def _raw_field_value(target: Any, field) -> tuple[float, ...] | float | bool:
    from .phase4_mutation_journal import RawFieldKind

    if field is RawFieldKind.POSE_BONE_LOCATION:
        value = target.location
    elif field is RawFieldKind.POSE_BONE_ROTATION_EULER:
        value = target.rotation_euler
    elif field is RawFieldKind.POSE_BONE_ROTATION_QUATERNION:
        value = target.rotation_quaternion
    elif field is RawFieldKind.POSE_BONE_ROTATION_AXIS_ANGLE:
        value = target.rotation_axis_angle
    elif field is RawFieldKind.POSE_BONE_SCALE:
        value = target.scale
    else:
        raise ValueError(f"Unsupported I12 pose field: {field!r}")
    return tuple(float(component) for component in value)


def _journal_scope(plan, owner_ptr: int):
    from .phase4_mutation_journal import QuarantineKey

    return QuarantineKey(
        plan.character_id,
        plan.setup_revision,
        plan.setup_signature,
        owner_ptr,
        None,
    )


def _record_pose_matrix_before(journal, plan, resolved: ResolvedControl) -> None:
    from .phase4_mutation_journal import (
        RawFieldMutationReceipt,
        RawFieldSnapshot,
    )

    owner_ptr = _pointer(resolved.owner_object)
    target_ptr = _pointer(resolved.target)
    if owner_ptr is None or target_ptr is None:
        raise RepresentationSnapError("I12 pose target lost its live runtime identity.")
    scope = _journal_scope(plan, int(owner_ptr))
    for field in _raw_pose_fields():
        journal.record(
            RawFieldMutationReceipt(
                journal.next_ordinal(),
                scope,
                int(owner_ptr),
                int(target_ptr),
                RawFieldSnapshot(field, _raw_field_value(resolved.target, field)),
            )
        )


def _constraint_raw_value(constraint: Any, field) -> float | bool:
    from .phase4_mutation_journal import RawFieldKind

    if field is RawFieldKind.CONSTRAINT_INFLUENCE:
        return float(constraint.influence)
    if field is RawFieldKind.CONSTRAINT_MUTE:
        return bool(constraint.mute)
    if field is RawFieldKind.CONSTRAINT_POLE_ANGLE:
        return float(constraint.pole_angle)
    raise ValueError(f"Unsupported I12 constraint field: {field!r}")


def _record_constraint_before(journal, plan, owner: Any, constraint: Any, field) -> None:
    from .phase4_mutation_journal import (
        ConstraintMutationReceipt,
        RawFieldSnapshot,
    )

    owner_ptr = _pointer(owner)
    constraint_ptr = _pointer(constraint)
    if owner_ptr is None or constraint_ptr is None:
        raise RepresentationSnapError("I12 constraint lost its live runtime identity.")
    journal.record(
        ConstraintMutationReceipt(
            journal.next_ordinal(),
            _journal_scope(plan, int(owner_ptr)),
            int(owner_ptr),
            int(constraint_ptr),
            RawFieldSnapshot(field, _constraint_raw_value(constraint, field)),
        )
    )


def execute_representation_snap(
    scene,
    control_context,
    plan,
    payload: RepresentationSnapPayload,
    *,
    selector_character_id: str | None = None,
    hook=None,
) -> RepresentationSnapResult:
    """Execute the proven I12 snap transiently, then restore exact raw state.

    This primitive intentionally leaves no FK/IK mode state, keys, FCurves, or
    raw pose/constraint writes behind. Later Contact authoring may consume the
    evaluated snap result inside its own bounded transaction, but persistent
    replay authority remains native animation data rather than this adapter.
    """

    import bpy
    from mathutils import Vector

    from .character_metadata import resolve_character
    from .phase4_mutation_journal import (
        MutationJournal,
        RawFieldKind,
        is_quarantined,
    )
    from .phase4_mutation_journal_blender import BlenderRollbackExecutor
    from .phase4_operation_plan import OperationType
    from .phase4_preflight import validate_plan_fresh
    from .phase4_verification import NoopStageHook, OperationStage

    if plan.operation_type is not OperationType.KINEMATIC_MOVE:
        raise RepresentationSnapError("I12 requires the existing KINEMATIC_MOVE OperationPlan.")
    dependency = plan.kinematic_dependency
    if dependency is None or dependency.mapping_id != payload.mapping_id:
        raise RepresentationSnapError("I12 payload mapping does not match the OperationPlan dependency.")

    hook = hook or NoopStageHook()
    hook.enter(OperationStage.PREFLIGHT, operation=plan.operation_id)
    freshness = validate_plan_fresh(
        scene,
        control_context,
        plan,
        selector_character_id=selector_character_id,
    )
    if not freshness.ok:
        return RepresentationSnapResult(False, diagnostics=freshness.diagnostics)

    view = resolve_character(scene, plan.character_id)
    resolution = resolve_limb_representation_capability(view, payload.mapping_id)
    capability = resolution.capability
    if capability is None:
        detail = "; ".join(issue.detail for issue in resolution.issues) or payload.mapping_id
        return RepresentationSnapResult(
            False,
            diagnostics=(
                _snap_diagnostic(plan, "I12_REPRESENTATION_CAPABILITY_STALE", detail),
            ),
        )
    if not representation_payload_matches(payload, capability):
        return RepresentationSnapResult(
            False,
            diagnostics=(
                _snap_diagnostic(
                    plan,
                    "I12_STALE_REPRESENTATION_PAYLOAD",
                    "FK/result/terminal native identity or representation constraint correspondence changed after planning.",
                ),
            ),
        )

    issue = _capability_preflight_issue(capability, payload.direction)
    if issue is not None:
        return RepresentationSnapResult(
            False,
            diagnostics=(_snap_diagnostic(plan, issue[0], issue[1]),),
        )

    owner = capability.result_controls[0].owner_object
    owner_ptr = _pointer(owner)
    if owner_ptr is None:
        return RepresentationSnapResult(
            False,
            diagnostics=(
                _snap_diagnostic(plan, "I12_OWNER_IDENTITY_MISSING", "Rigped Armature owner has no live runtime identity."),
            ),
        )
    if is_quarantined(_journal_scope(plan, int(owner_ptr))):
        raise RepresentationSnapError(
            "I12 mutation scope is quarantined after an incomplete rollback; explicit recovery is required."
        )

    expected_result = tuple(control.target.matrix.copy() for control in capability.result_controls)
    expected_terminal = capability.result_terminal.target.matrix.copy()
    pole_solution = None
    pole_position = None
    pole_angle = None
    hinge_branch = None
    hinge_owner = capability.native_ik.solver_owner.target
    hinge_snapshot = None
    if payload.direction is SnapDirection.FK_TO_IK:
        hinge_branch = _generated_rigped_hinge_branch(capability)
        pole_solution = _initial_pole_solution(capability)
        if pole_solution is None or not pole_solution.ok or pole_solution.position is None:
            code = getattr(pole_solution, "issue_code", None) or "I12_UNSUPPORTED_POLE_SOLVE"
            return RepresentationSnapResult(
                False,
                diagnostics=(
                    _snap_diagnostic(
                        plan,
                        str(code),
                        "FK→IK snap cannot derive a stable two-bone pole from the current evaluated limb.",
                    ),
                ),
            )
        try:
            pole_angle = _derived_pole_angle(capability, pole_solution.position)
        except RepresentationSnapError as exc:
            return RepresentationSnapResult(
                False,
                diagnostics=(
                    _snap_diagnostic(plan, "I12_SINGULAR_POLE_ANGLE", str(exc)),
                ),
            )
        pole_position, pole_angle = _canonical_generated_rigped_pole(
            capability,
            pole_solution.position,
            float(pole_angle),
            hinge_branch,
        )

    hook.enter(OperationStage.SNAPSHOT, operation=plan.operation_id)
    journal = MutationJournal(
        plan.operation_id,
        plan.character_id,
        plan.setup_revision,
        plan.setup_signature,
        BlenderRollbackExecutor(),
    )
    mutation_ordinal = 0

    def stage_write(detail: str) -> None:
        nonlocal mutation_ordinal
        mutation_ordinal += 1
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=mutation_ordinal,
            operation=plan.operation_id,
            detail=detail,
        )

    def write_pose_matrix(resolved: ResolvedControl, matrix: Any, detail: str) -> None:
        _record_pose_matrix_before(journal, plan, resolved)
        stage_write(detail)
        resolved.target.matrix = matrix.copy()

    def write_constraint(constraint: Any, field, value: float | bool, detail: str) -> None:
        before = _constraint_raw_value(constraint, field)
        if before == value:
            return
        _record_constraint_before(journal, plan, owner, constraint, field)
        stage_write(detail)
        if field is RawFieldKind.CONSTRAINT_INFLUENCE:
            constraint.influence = float(value)
        elif field is RawFieldKind.CONSTRAINT_MUTE:
            constraint.mute = bool(value)
        elif field is RawFieldKind.CONSTRAINT_POLE_ANGLE:
            constraint.pole_angle = float(value)
        else:
            raise RepresentationSnapError(f"Unsupported I12 constraint field: {field!r}")

    try:
        hook.enter(OperationStage.PLAN, operation=plan.operation_id)
        if payload.direction is SnapDirection.FK_TO_IK:
            assert pole_solution is not None and pole_position is not None
            assert pole_angle is not None
            if hinge_branch is not None:
                hinge_snapshot = _capture_generated_rigped_hinge_settings(hinge_owner)
                _configure_generated_rigped_hinge_branch(hinge_owner, hinge_branch)
                bpy.context.view_layer.update()

            write_pose_matrix(
                capability.native_ik.ik_target,
                expected_terminal,
                "snap IK target to evaluated terminal",
            )
            pole_target = capability.native_ik.pole_target
            assert pole_target is not None
            pole_matrix = pole_target.target.matrix.copy()
            pole_matrix.translation = pole_target.owner_object.matrix_world.inverted() @ Vector(
                pole_position
            )
            write_pose_matrix(pole_target, pole_matrix, "snap pole target to evaluated bend plane")
            write_constraint(
                capability.native_ik.constraint,
                RawFieldKind.CONSTRAINT_POLE_ANGLE,
                float(pole_angle),
                "write derived native IK pole angle",
            )
            bpy.context.view_layer.update()
            write_constraint(
                capability.native_ik.constraint,
                RawFieldKind.CONSTRAINT_INFLUENCE,
                1.0,
                "enable native IK positional solve",
            )
            write_constraint(
                capability.terminal_ik_constraint,
                RawFieldKind.CONSTRAINT_INFLUENCE,
                1.0,
                "enable terminal IK orientation correspondence",
            )
        else:
            for index, (control, desired) in enumerate(
                zip(capability.fk_controls, expected_result, strict=True)
            ):
                write_pose_matrix(
                    control,
                    desired,
                    f"reconstruct authored FK chain member {index}",
                )
                bpy.context.view_layer.update()
            write_pose_matrix(
                capability.authored_terminal,
                expected_terminal,
                "reconstruct authored FK terminal",
            )
            bpy.context.view_layer.update()
            write_constraint(
                capability.native_ik.constraint,
                RawFieldKind.CONSTRAINT_INFLUENCE,
                0.0,
                "release native IK positional solve",
            )
            write_constraint(
                capability.terminal_ik_constraint,
                RawFieldKind.CONSTRAINT_INFLUENCE,
                0.0,
                "release terminal IK orientation correspondence",
            )

        hook.enter(OperationStage.REEVALUATE, operation=plan.operation_id)
        bpy.context.view_layer.update()
        hook.enter(OperationStage.VERIFY, operation=plan.operation_id)
        residuals = _residuals_for_expected(capability, expected_result, expected_terminal)
        if not _residuals_within_tolerance(residuals, scale=_character_scale(capability)):
            raise RepresentationSnapError(
                "I12_RESIDUAL_GATE_FAILED: "
                f"mapping={payload.mapping_id} "
                f"chain_pos={residuals.max_position_error:.9g} "
                f"chain_rot={residuals.max_rotation_error:.9g} "
                f"terminal_pos={residuals.terminal_position_error:.9g} "
                f"terminal_rot={residuals.terminal_rotation_error:.9g}; "
                "evaluated chain/terminal pose exceeded the frozen scale-aware tolerance."
            )

        fk_states = (
            tuple(_capture_snap_control_state(control) for control in capability.fk_controls)
            if payload.direction is SnapDirection.IK_TO_FK
            else ()
        )
        authored_terminal_state = (
            _capture_snap_control_state(capability.authored_terminal)
            if payload.direction is SnapDirection.IK_TO_FK
            else None
        )
        ik_target_state = (
            _capture_snap_control_state(capability.native_ik.ik_target)
            if payload.direction is SnapDirection.FK_TO_IK
            else None
        )
        pole_target_state = (
            _capture_snap_control_state(capability.native_ik.pole_target)
            if payload.direction is SnapDirection.FK_TO_IK
            and capability.native_ik.pole_target is not None
            else None
        )
        captured_pole_angle = (
            float(capability.native_ik.constraint.pole_angle)
            if payload.direction is SnapDirection.FK_TO_IK
            else None
        )
        if hinge_snapshot is not None:
            _restore_generated_rigped_hinge_settings(hinge_owner, hinge_snapshot)
            hinge_snapshot = None
            bpy.context.view_layer.update()

        hook.enter(
            OperationStage.ROLLBACK_WRITE,
            operation=plan.operation_id,
            detail="restore successful transient snap",
        )
        rollback = journal.rollback_or_raise()
        bpy.context.view_layer.update()
        hook.enter(OperationStage.ROLLBACK_VERIFY, operation=plan.operation_id)
        hook.enter(OperationStage.COMMIT, operation=plan.operation_id)
        return RepresentationSnapResult(
            True,
            residuals=residuals,
            transient_writes=mutation_ordinal,
            restored_exactly=not rollback.residue_receipts,
            fk_control_states=fk_states,
            authored_terminal_state=authored_terminal_state,
            ik_target_state=ik_target_state,
            pole_target_state=pole_target_state,
            pole_angle=captured_pole_angle,
        )
    except Exception:
        if hinge_snapshot is not None:
            _restore_generated_rigped_hinge_settings(hinge_owner, hinge_snapshot)
            hinge_snapshot = None
            bpy.context.view_layer.update()
        if str(journal.state) == "OPEN" or getattr(journal.state, "value", None) == "OPEN":
            hook.enter(
                OperationStage.ROLLBACK_WRITE,
                operation=plan.operation_id,
                detail="rollback failed transient snap",
            )
            journal.rollback_or_raise()
            bpy.context.view_layer.update()
            hook.enter(OperationStage.ROLLBACK_VERIFY, operation=plan.operation_id)
        raise
