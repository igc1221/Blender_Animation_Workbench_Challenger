from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import bpy
from mathutils import Matrix, Quaternion, Vector

from .character_metadata import resolve_character
from .character_model import ControlUsage
from .debug_trace import trace_event, trace_exception
from .phase4_contact_model import (
    AWB_CONTACT_AUTHORING_STATE_PROPERTY,
    AWB_CONTACT_EXTERNAL_CONSTRAINT,
    AWB_CONTACT_EXTERNAL_PLACEHOLDER_PROPERTY,
    AWB_CONTACT_PIVOT_CONSTRAINT,
    AWB_CONTACT_STATE_PROPERTY,
    ContactKeyType,
    ContactPlantSpace,
    ContactStateValue,
    state_value_for_type,
    type_for_state_value,
)
from .phase4_mutation_journal import (
    ConstraintMutationReceipt,
    FCurveMutationReceipt,
    FCurveSnapshot,
    IDPropertyMutationReceipt,
    MutationJournal,
    RawFieldKind,
    RawFieldMutationReceipt,
    RawFieldSnapshot,
)
from .phase4_mutation_journal_blender import BlenderRollbackExecutor
from .phase4_operation_plan import (
    AllocationIntent,
    ChannelFamily,
    DependencyFootprint,
    OperationPlan,
    OperationType,
    PlannedChannel,
    PlannedOwnerGroup,
    PlanWriteFootprint,
    ReadFootprint,
    channel_sort_key,
    validate_plan_structure,
)
from .phase4_preflight import (
    _base_allocation_intent,
    _build_channel,
    _find_fcurve,
    _rotation_descriptor,
    _runtime_pointer,
    build_direct_key_plan,
    build_kinematic_dependency_plan,
    validate_plan_fresh,
)
from .phase4_representation_snap import (
    RepresentationSnapPayload,
    SnapControlState,
    SnapDirection,
    build_representation_snap_payload,
    execute_representation_snap,
    representation_payload_matches,
    resolve_limb_representation_capability,
)
from .phase4_verification import (
    Diagnostic,
    DiagnosticSeverity,
    NoopStageHook,
    OperationStage,
    StageHook,
)
from .phase4_writer import (
    DirectWriterError,
    WriterTrigger,
    _capture_fcurve_snapshot,
    _check_scope_not_quarantined,
    _ensure_fcurve,
    _ensure_owner_storage,
    _key_at_frame,
    _prepare_semantic_state_property,
    _scope,
    _write_channel_key,
)
from .rigped_contract import RigpedCapability, resolve_rigped_target
from .rigped_operation_domain import resolve_operation_domain
from .semantic_adapter import (
    assigned_channelbag,
    channel_binding_token,
    control_context_for_context,
    control_property_path,
    runtime_control_key,
)


class ContactAuthoringError(RuntimeError):
    pass


def _authored_public_pose_snapshot(scene) -> tuple[tuple[Any, Matrix], ...]:
    snapshots: list[tuple[Any, Matrix]] = []
    for owner in tuple(getattr(scene, "objects", ()) or ()):
        if getattr(owner, "type", None) != "ARMATURE" or getattr(owner, "pose", None) is None:
            continue
        for pose_bone in owner.pose.bones:
            collections = tuple(getattr(pose_bone.bone, "collections", ()) or ())
            if not any(str(collection.name) == "Authored" for collection in collections):
                continue
            snapshots.append((pose_bone, pose_bone.matrix_basis.copy()))
    return tuple(snapshots)


def _reevaluate_contact_frame_preserving_public_pose(
    scene,
    frame: int,
    subframe: float,
    *,
    feedback_constraints: tuple[Any, ...] = (),
) -> None:
    # Contact C authors semantic/IK authority. It must not visibly re-pose an
    # unrelated Sliding limb merely because Blender re-evaluates the same frame.
    # Snapshot animator-facing Authored channels immediately before the required
    # FCurve evaluation, then restore those pose channels before the next redraw.
    #
    # frame_change_post derives FK-feedback mute from the evaluated Contact
    # state. During an authoring transaction that derived representation change
    # must not escape ahead of the journal-owned final authority step, so freeze
    # and restore only the affected feedback constraints across frame_set().
    snapshots = _authored_public_pose_snapshot(scene)
    feedback_mutes = tuple(bool(constraint.mute) for constraint in feedback_constraints)
    scene.frame_set(frame, subframe=subframe)
    view_layer = getattr(bpy.context, "view_layer", None)
    for constraint, muted in zip(feedback_constraints, feedback_mutes, strict=True):
        constraint.mute = bool(muted)
    if view_layer is not None:
        view_layer.update()
    for pose_bone, matrix_basis in snapshots:
        pose_bone.matrix_basis = matrix_basis
    if view_layer is not None:
        view_layer.update()


class ContactAuthoringMode(StrEnum):
    CYCLE = "CYCLE"
    ANCHOR = "ANCHOR"
    REPLANT = "REPLANT"


@dataclass(frozen=True, slots=True, order=True)
class ContactScalarRowKey:
    label: str
    data_path: str
    array_index: int = 0


@dataclass(frozen=True, slots=True)
class ContactScalarChannel:
    row_key: ContactScalarRowKey
    existing_fcurve_token: int | None
    allocation_intent: AllocationIntent
    target_value: float


@dataclass(frozen=True, slots=True)
class ContactHoldCapability:
    binding_id: str
    control: Any
    constraint: Any
    external_constraint: Any
    point_binding_id: str
    point_control: Any
    pivot_constraint: Any
    control_runtime_key: tuple[int, int]
    constraint_runtime_key: int
    external_constraint_runtime_key: int
    point_control_runtime_key: tuple[int, int]
    pivot_constraint_runtime_key: int


@dataclass(frozen=True, slots=True)
class ContactCurveCleanup:
    fcurve_token: int
    snapshot: FCurveSnapshot


@dataclass(frozen=True, slots=True)
class ContactIntentPlan:
    operation_id: str
    snap_plan: OperationPlan
    mapping_id: str
    target_type: ContactKeyType
    target_plant_space: ContactPlantSpace
    contact_point_local: tuple[float, float, float] | None
    apply_contact_point: bool
    external_target_runtime_key: int | None
    effective_type: ContactKeyType | None
    exact_current_type: ContactKeyType | None
    owner_runtime_key: int
    owner_binding_token: tuple[int | None, int | None, int | None, int | None]
    representation_identity: RepresentationSnapPayload
    transition_payload: RepresentationSnapPayload | None
    hold_capability: ContactHoldCapability
    scalar_channels: tuple[ContactScalarChannel, ...]
    pole_cleanup_fcurve_token: int | None
    pole_cleanup_snapshot: FCurveSnapshot | None
    hold_cleanup_curves: tuple[ContactCurveCleanup, ...]
    baseline_required: bool
    enabled_types: tuple[ContactKeyType, ...]


@dataclass(frozen=True, slots=True)
class ContactTransformRowSnapshot:
    rows: tuple[PlannedChannel, ...]
    keyed_row_count: int


@dataclass(frozen=True, slots=True)
class ContactPlanBuildResult:
    plan: ContactIntentPlan | None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None and not any(
            item.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.BLOCKER}
            for item in self.diagnostics
        )


@dataclass(frozen=True, slots=True)
class ContactBatchIntentPlan:
    operation_id: str
    intents: tuple[ContactIntentPlan, ...]
    mapping_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContactBatchPlanBuildResult:
    plan: ContactBatchIntentPlan | None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None and not any(
            item.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.BLOCKER}
            for item in self.diagnostics
        )


@dataclass(frozen=True, slots=True)
class ContactPreparedIntent:
    intent: ContactIntentPlan
    capability: Any
    hold: ContactHoldCapability
    transform_rows: tuple[PlannedChannel, ...]
    expected_result: tuple[Any, ...]
    expected_terminal: Any
    hinge_branch: int | None
    hold_state: SnapControlState | None
    point_state: SnapControlState | None


@dataclass(frozen=True, slots=True)
class ContactAuthoringResult:
    applied: bool
    contact_type: ContactKeyType | None = None
    rows_written: int = 0
    created_fcurves: int = 0
    diagnostics: tuple[Diagnostic, ...] = ()
    mapping_contact_types: tuple[tuple[str, ContactKeyType], ...] = ()


def _diagnostic(operation_id: str, code: str, detail: str, *, character_id: str | None = None) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        stage=OperationStage.PREFLIGHT,
        operation=operation_id,
        detail=detail,
        character_id=character_id,
    )


def _same_float(left: float, right: float, *, epsilon: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= epsilon


def _curve_key_at(curve, time: float):
    if curve is None:
        return None
    matches = tuple(point for point in curve.keyframe_points if float(point.co.x) == float(time))
    if len(matches) > 1:
        raise ContactAuthoringError(f"Multiple Contact keys exist at exact time {time!r}.")
    return matches[0] if matches else None


def _sorted_points(curve):
    return tuple(sorted(curve.keyframe_points, key=lambda point: float(point.co.x))) if curve is not None else ()


def _exact_scalar_value(curve, time: float) -> float | None:
    point = _curve_key_at(curve, time)
    return float(point.co.y) if point is not None else None


def _evaluated_scalar(curve, time: float, fallback: float) -> float:
    if curve is None or not curve.keyframe_points:
        return float(fallback)
    return float(curve.evaluate(float(time)))


def _build_scalar_channel(owner, *, label: str, data_path: str, value: float) -> ContactScalarChannel:
    bag = assigned_channelbag(owner)
    curve = _find_fcurve(bag, data_path, 0)
    if curve is not None:
        token = _runtime_pointer(curve)
        if token is None:
            raise ContactAuthoringError(f"Existing Contact FCurve {label!r} has no live runtime identity.")
        allocation = AllocationIntent.EXISTING_FCURVE
    else:
        token = None
        allocation = _base_allocation_intent(owner, bag)
    return ContactScalarChannel(
        ContactScalarRowKey(label, data_path, 0),
        token,
        allocation,
        float(value),
    )


def _resolve_contact_hold(view, target, mapping, capability) -> ContactHoldCapability:
    binding_by_id = {binding.binding_id: binding for binding in view.definition.bindings}
    candidates = tuple(
        binding_by_id[binding_id]
        for binding_id in mapping.extras
        if binding_id in binding_by_id
        and binding_by_id[binding_id].semantic_key == "awb.contact"
        and binding_by_id[binding_id].side == mapping.side
        and binding_by_id[binding_id].usage == ControlUsage.MECHANISM
    )
    if len(candidates) != 1:
        raise ContactAuthoringError(
            "Generated Contact mapping must resolve exactly one mechanism point-hold carrier."
        )
    point_candidates = tuple(
        binding_by_id[binding_id]
        for binding_id in mapping.extras
        if binding_id in binding_by_id
        and binding_by_id[binding_id].semantic_key == "awb.contact_point"
        and binding_by_id[binding_id].side == mapping.side
        and binding_by_id[binding_id].usage == ControlUsage.MECHANISM
    )
    if len(point_candidates) != 1:
        raise ContactAuthoringError(
            "Generated Contact mapping must resolve exactly one mechanism continuous-point carrier."
        )
    binding = candidates[0]
    point_binding = point_candidates[0]
    by_id = {contract.binding_id: contract for contract in target.controls}
    contract = by_id.get(binding.binding_id)
    point_contract = by_id.get(point_binding.binding_id)
    if contract is None:
        raise ContactAuthoringError("Resolved Rigped target is missing the Contact point-hold carrier.")
    if point_contract is None:
        raise ContactAuthoringError("Resolved Rigped target is missing the Contact continuous-point carrier.")
    resolved = contract.target
    point_resolved = point_contract.target
    hold_target = resolved.target
    point_target = point_resolved.target
    ik_pose_bone = capability.native_ik.ik_target.target
    constraints = tuple(
        constraint
        for constraint in ik_pose_bone.constraints
        if str(constraint.type) == "COPY_LOCATION"
        and constraint.target == resolved.owner_object
        and str(constraint.subtarget) == str(hold_target.name)
    )
    if len(constraints) != 1:
        raise ContactAuthoringError(
            "Authored IK target must resolve exactly one generated Contact point-hold constraint."
        )
    constraint = constraints[0]
    if str(constraint.target_space) != "WORLD" or str(constraint.owner_space) != "WORLD":
        raise ContactAuthoringError("Contact point-hold constraint must use WORLD target/owner spaces.")
    if not all(bool(getattr(constraint, name, False)) for name in ("use_x", "use_y", "use_z")):
        raise ContactAuthoringError("Contact point-hold constraint must own all position axes.")
    if any(bool(getattr(constraint, name, False)) for name in ("invert_x", "invert_y", "invert_z")):
        raise ContactAuthoringError("Contact point-hold constraint cannot invert position axes.")
    if bool(getattr(constraint, "use_offset", False)):
        raise ContactAuthoringError("Contact point-hold constraint cannot use offset mode.")
    constraint_key = _runtime_pointer(constraint)
    if constraint_key is None:
        raise ContactAuthoringError("Contact point-hold constraint has no live runtime identity.")

    pivot_candidates = tuple(
        item
        for item in ik_pose_bone.constraints
        if str(item.type) == "PIVOT"
        and str(item.name) == AWB_CONTACT_PIVOT_CONSTRAINT
        and item.target == point_resolved.owner_object
        and str(item.subtarget) == str(point_target.name)
    )
    if len(pivot_candidates) != 1:
        raise ContactAuthoringError(
            "Authored IK target must resolve exactly one generated continuous Contact-point pivot constraint."
        )
    pivot_constraint = pivot_candidates[0]
    if str(pivot_constraint.rotation_range) != "ALWAYS_ACTIVE":
        raise ContactAuthoringError("Contact-point pivot constraint must be always active when influenced.")
    if any(abs(float(component)) > 1e-8 for component in pivot_constraint.offset):
        raise ContactAuthoringError("Contact-point pivot constraint must use a zero target offset.")
    pivot_key = _runtime_pointer(pivot_constraint)
    if pivot_key is None:
        raise ContactAuthoringError("Contact-point pivot constraint has no live runtime identity.")

    external_candidates = tuple(
        item
        for item in hold_target.constraints
        if str(item.type) == "CHILD_OF" and str(item.name) == AWB_CONTACT_EXTERNAL_CONSTRAINT
    )
    if len(external_candidates) != 1:
        raise ContactAuthoringError(
            "Generated Contact hold carrier must resolve exactly one dormant external Object-space constraint."
        )
    external_constraint = external_candidates[0]
    if str(external_constraint.target_space) != "WORLD" or str(external_constraint.owner_space) != "WORLD":
        raise ContactAuthoringError("External Contact space constraint must use WORLD target/owner spaces.")
    external_key = _runtime_pointer(external_constraint)
    if external_key is None:
        raise ContactAuthoringError("External Contact space constraint has no live runtime identity.")
    return ContactHoldCapability(
        binding.binding_id,
        contract,
        constraint,
        external_constraint,
        point_binding.binding_id,
        point_contract,
        pivot_constraint,
        runtime_control_key(resolved),
        int(constraint_key),
        int(external_key),
        runtime_control_key(point_resolved),
        int(pivot_key),
    )


def _contact_paths(capability, hold: ContactHoldCapability) -> tuple[str, str, str, str, str, str, str]:
    solver_bone = capability.native_ik.solver_owner.target
    if AWB_CONTACT_STATE_PROPERTY not in solver_bone:
        raise ContactAuthoringError("Generated Contact state carrier is missing from the native solver PoseBone.")
    return (
        solver_bone.path_from_id(f'["{AWB_CONTACT_STATE_PROPERTY}"]'),
        capability.native_ik.constraint.path_from_id("influence"),
        capability.terminal_ik_constraint.path_from_id("influence"),
        capability.native_ik.constraint.path_from_id("pole_angle"),
        hold.constraint.path_from_id("influence"),
        hold.pivot_constraint.path_from_id("influence"),
        hold.external_constraint.path_from_id("influence"),
    )


def _validate_protected_curve(curve, *, label: str, time_epsilon: float = 1e-4) -> None:
    if curve is None:
        return
    if str(curve.extrapolation) != "CONSTANT":
        raise ContactAuthoringError(f"Contact {label} curve contains non-CONSTANT extrapolation.")
    if len(curve.modifiers) > 0:
        raise ContactAuthoringError(f"Contact {label} curve contains unsupported FCurve modifiers.")
    previous_time: float | None = None
    for point in _sorted_points(curve):
        time = float(point.co.x)
        value = float(point.co.y)
        if not (math.isfinite(time) and math.isfinite(value)):
            raise ContactAuthoringError(f"Contact {label} curve contains non-finite key data.")
        if previous_time is not None and abs(time - previous_time) <= abs(float(time_epsilon)):
            raise ContactAuthoringError(f"Contact {label} curve contains duplicate/ambiguous key times.")
        previous_time = time
        if str(point.interpolation) != "CONSTANT":
            raise ContactAuthoringError(f"Contact {label} curve contains non-CONSTANT interpolation.")


def _transition_dependency_row_keys(
    target,
    capability,
    contact_type: ContactKeyType,
    hold: ContactHoldCapability,
) -> tuple[Any, ...]:
    """Return the writer's exact authored transform rows for one Contact state.

    Values are irrelevant here; `_build_transform_rows` is reused so I14 validation cannot
    silently drift from the I13 writer schema. Extra transform keys remain valid ordinary
    animation and are therefore not rejected.
    """

    rows = _build_transform_rows(
        target,
        capability,
        contact_type,
        None,
        hold=hold,
        hold_state=_current_state(hold.control) if contact_type is ContactKeyType.PLANTED else None,
        point_state=_current_state(hold.point_control) if contact_type is ContactKeyType.PLANTED else None,
    )
    return tuple(dict.fromkeys(row.row_key for row in rows))


def _validate_existing_track(
    scene,
    target,
    capability,
    hold: ContactHoldCapability,
    state_curve,
    ik_curve,
    terminal_curve,
    pole_curve,
    hold_curve,
    pivot_curve,
    external_curve,
) -> None:
    solver_bone = capability.native_ik.solver_owner.target
    raw_state = float(solver_bone.get(AWB_CONTACT_STATE_PROPERTY, float(ContactStateValue.UNINITIALIZED)))
    raw_ik = float(capability.native_ik.constraint.influence)
    raw_terminal = float(capability.terminal_ik_constraint.influence)
    raw_hold = float(hold.constraint.influence)
    raw_pivot = float(hold.pivot_constraint.influence)
    raw_external = float(hold.external_constraint.influence)
    bag = assigned_channelbag(capability.native_ik.solver_owner.owner_object)
    hold_location_path = control_property_path(hold.control.target, "location")
    point_location_path = control_property_path(hold.point_control.target, "location")
    hold_location_curves = tuple(
        _find_fcurve(bag, hold_location_path, index)
        for index in range(3)
    )
    point_location_curves = tuple(
        _find_fcurve(bag, point_location_path, index)
        for index in range(3)
    )
    hold_location_has_keys = any(
        curve is not None and len(curve.keyframe_points) > 0
        for curve in hold_location_curves
    )
    point_location_has_keys = any(
        curve is not None and len(curve.keyframe_points) > 0
        for curve in point_location_curves
    )

    if state_curve is None:
        if (
            ik_curve is not None
            or terminal_curve is not None
            or hold_curve is not None
            or pivot_curve is not None
            or external_curve is not None
            or hold_location_has_keys
            or point_location_has_keys
            or (pole_curve is not None and len(pole_curve.keyframe_points) > 0)
        ):
            raise ContactAuthoringError(
                "Contact state authority is absent while Contact-owned companion FCurves already exist."
            )
        if not _same_float(raw_state, float(ContactStateValue.UNINITIALIZED)):
            raise ContactAuthoringError("Unkeyed Contact carrier is not in generated UNINITIALIZED state.")
        if not (
            _same_float(raw_ik, 0.0)
            and _same_float(raw_terminal, 0.0)
            and _same_float(raw_hold, 0.0)
            and _same_float(raw_pivot, 0.0)
            and _same_float(raw_external, 0.0)
        ):
            raise ContactAuthoringError(
                "Unkeyed generated Contact must begin in physical Free state with all Contact influences zero."
            )
        return

    if (
        ik_curve is None
        or terminal_curve is None
        or hold_curve is None
        or pivot_curve is None
        or external_curve is None
    ):
        raise ContactAuthoringError(
            "Contact state FCurve exists without the complete Contact-owned influence companion curves."
        )
    _validate_protected_curve(state_curve, label="state")
    _validate_protected_curve(ik_curve, label="IK influence")
    _validate_protected_curve(terminal_curve, label="terminal IK influence")
    _validate_protected_curve(pole_curve, label="pole angle")
    _validate_protected_curve(hold_curve, label="point-hold influence")
    _validate_protected_curve(pivot_curve, label="continuous-point pivot influence")
    _validate_protected_curve(external_curve, label="external Object-space influence")
    if hold_location_has_keys and any(curve is None for curve in hold_location_curves):
        raise ContactAuthoringError("Contact point-hold carrier has an incomplete XYZ FCurve set.")
    if point_location_has_keys and any(curve is None for curve in point_location_curves):
        raise ContactAuthoringError("Contact continuous-point carrier has an incomplete XYZ FCurve set.")
    for index, curve in enumerate(hold_location_curves):
        _validate_protected_curve(curve, label=f"point-hold location[{index}]")
    for index, curve in enumerate(point_location_curves):
        _validate_protected_curve(curve, label=f"continuous-point location[{index}]")

    state_points = _sorted_points(state_curve)
    if not state_points:
        raise ContactAuthoringError("Contact state FCurve exists but contains no authoritative keys.")
    bag = assigned_channelbag(capability.native_ik.solver_owner.owner_object)
    if bag is None:
        raise ContactAuthoringError("Contact animation owner lost its assigned ChannelBag.")
    frame_start = float(scene.frame_start)
    first_authorable_time: float | None = None
    baseline_at_start = False
    for point in state_points:
        time = float(point.co.x)
        value = float(point.co.y)
        rounded = round(value)
        if not _same_float(value, float(rounded)) or rounded not in {
            int(ContactStateValue.UNINITIALIZED),
            int(ContactStateValue.FREE),
            int(ContactStateValue.SLIDING),
            int(ContactStateValue.PLANTED),
        }:
            raise ContactAuthoringError("Contact state curve contains an unsupported/non-discrete state value.")
        if rounded == int(ContactStateValue.UNINITIALIZED):
            if not _same_float(time, frame_start):
                raise ContactAuthoringError("UNINITIALIZED Contact baseline is only valid at scene.frame_start.")
            baseline_at_start = True
            expected_ik = 0.0
            expected_hold = 0.0
        else:
            if first_authorable_time is None:
                first_authorable_time = time
            expected_ik = 0.0 if rounded == int(ContactStateValue.FREE) else 1.0
            expected_hold = 1.0 if rounded == int(ContactStateValue.PLANTED) else 0.0
        ik_value = _exact_scalar_value(ik_curve, time)
        terminal_value = _exact_scalar_value(terminal_curve, time)
        hold_value = _exact_scalar_value(hold_curve, time)
        pivot_value = _exact_scalar_value(pivot_curve, time)
        external_value = _exact_scalar_value(external_curve, time)
        if (
            ik_value is None
            or terminal_value is None
            or hold_value is None
            or pivot_value is None
            or external_value is None
        ):
            raise ContactAuthoringError("Contact state key is missing a same-time complete influence bundle.")
        if rounded == int(ContactStateValue.PLANTED):
            if not (_same_float(external_value, 0.0) or _same_float(external_value, 1.0)):
                raise ContactAuthoringError("Planted Contact has a non-discrete external Object-space influence.")
            expected_pivot = 0.0 if _same_float(external_value, 1.0) else 1.0
            if _same_float(external_value, 1.0):
                external_target = hold.external_constraint.target
                target_key = _runtime_pointer(external_target)
                if target_key is None or not any(_runtime_pointer(obj) == target_key for obj in scene.objects):
                    raise ContactAuthoringError(
                        "Object-space Planted Contact lost its bound external Object identity."
                    )
        else:
            expected_pivot = 0.0
            if not _same_float(external_value, 0.0):
                raise ContactAuthoringError(
                    "Non-Planted Contact state contains an active external Object-space influence."
                )
        if not (
            _same_float(ik_value, expected_ik)
            and _same_float(terminal_value, expected_ik)
            and _same_float(hold_value, expected_hold)
            and _same_float(pivot_value, expected_pivot)
        ):
            raise ContactAuthoringError("Contact state key disagrees with its same-time influence bundle.")
        pole_value = _exact_scalar_value(pole_curve, time)
        if rounded in {int(ContactStateValue.SLIDING), int(ContactStateValue.PLANTED)}:
            if pole_value is None:
                raise ContactAuthoringError("IK-authoritative Contact state key is missing its same-time pole-angle key.")
        elif pole_value is not None:
            raise ContactAuthoringError("Free Contact state key contains a stray same-time pole-angle key.")

        hold_values = tuple(_exact_scalar_value(curve, time) for curve in hold_location_curves)
        point_values = tuple(_exact_scalar_value(curve, time) for curve in point_location_curves)
        if rounded == int(ContactStateValue.PLANTED):
            if any(component is None for component in hold_values):
                raise ContactAuthoringError(
                    "Planted Contact state key is missing its same-time point-hold XYZ anchor bundle."
                )
            if any(component is None for component in point_values):
                raise ContactAuthoringError(
                    "Planted Contact state key is missing its same-time continuous-point XYZ anchor bundle."
                )
        else:
            if any(component is not None for component in hold_values):
                raise ContactAuthoringError(
                    "Non-Planted Contact state key contains a stray same-time point-hold anchor key."
                )
            if any(component is not None for component in point_values):
                raise ContactAuthoringError(
                    "Non-Planted Contact state key contains a stray same-time continuous-point anchor key."
                )

        contact_type = type_for_state_value(value)
        if contact_type is not None:
            for row_key in _transition_dependency_row_keys(target, capability, contact_type, hold):
                dependency_curve = _find_fcurve(
                    bag,
                    row_key.data_path,
                    row_key.array_index,
                )
                if dependency_curve is None or _curve_key_at(dependency_curve, time) is None:
                    raise ContactAuthoringError(
                        f"{contact_type.value.title()} Contact state key is missing required "
                        f"transition dependency key {row_key.data_path}[{row_key.array_index}] "
                        f"at time {time}."
                    )

    for label, companion_curve in (
        ("IK influence", ik_curve),
        ("terminal IK influence", terminal_curve),
        ("point-hold influence", hold_curve),
        ("continuous-point pivot influence", pivot_curve),
        ("external Object-space influence", external_curve),
    ):
        for point in companion_curve.keyframe_points:
            time = float(point.co.x)
            if _curve_key_at(state_curve, time) is None:
                raise ContactAuthoringError(f"Contact {label} curve contains an orphan key without state authority.")
    if pole_curve is not None:
        for point in pole_curve.keyframe_points:
            time = float(point.co.x)
            state_point = _curve_key_at(state_curve, time)
            state_type = (
                type_for_state_value(float(state_point.co.y))
                if state_point is not None
                else None
            )
            if state_type not in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}:
                raise ContactAuthoringError("Contact pole-angle curve contains an orphan/non-IK-authoritative key.")
    for curve, label in (
        *((curve, "point-hold") for curve in hold_location_curves),
        *((curve, "continuous-point") for curve in point_location_curves),
    ):
        if curve is None:
            continue
        for point in curve.keyframe_points:
            time = float(point.co.x)
            state_point = _curve_key_at(state_curve, time)
            if state_point is None or type_for_state_value(float(state_point.co.y)) is not ContactKeyType.PLANTED:
                raise ContactAuthoringError(
                    f"Contact {label} curve contains an orphan/non-Planted anchor key."
                )

    if first_authorable_time is not None and first_authorable_time > frame_start and not baseline_at_start:
        raise ContactAuthoringError(
            "First mid-clip Contact key lacks an explicit UNINITIALIZED/Free baseline at scene.frame_start."
        )


def _resolve_state_types(state_curve, time: float) -> tuple[ContactKeyType | None, ContactKeyType | None]:
    exact_type: ContactKeyType | None = None
    effective_type: ContactKeyType | None = None
    for point in _sorted_points(state_curve):
        point_time = float(point.co.x)
        if point_time > float(time):
            break
        point_type = type_for_state_value(float(point.co.y))
        if point_type is not None:
            effective_type = point_type
        if point_time == float(time):
            exact_type = point_type
    return exact_type, effective_type


def _contact_authoring_latch_type(capability) -> ContactKeyType | None:
    """Return the last Contact semantic explicitly confirmed by C for this limb."""

    solver_bone = capability.native_ik.solver_owner.target
    raw_value = float(
        solver_bone.get(
            AWB_CONTACT_AUTHORING_STATE_PROPERTY,
            float(ContactStateValue.UNINITIALIZED),
        )
    )
    return type_for_state_value(raw_value)


def _set_contact_authoring_latch(capability, contact_type: ContactKeyType) -> None:
    solver_bone = capability.native_ik.solver_owner.target
    solver_bone[AWB_CONTACT_AUTHORING_STATE_PROPERTY] = float(state_value_for_type(contact_type))


def _journal_contact_authoring_latch(
    capability,
    contact_type: ContactKeyType,
    plan: OperationPlan,
    journal: MutationJournal,
    stage_write,
) -> None:
    owner = capability.native_ik.solver_owner.owner_object
    solver_bone = capability.native_ik.solver_owner.target
    owner_ptr = _runtime_pointer(owner)
    target_ptr = _runtime_pointer(solver_bone)
    if owner_ptr is None or target_ptr is None:
        raise ContactAuthoringError("Contact authoring latch lost runtime identity before commit.")
    before_exists = AWB_CONTACT_AUTHORING_STATE_PROPERTY in solver_bone
    before_value = (
        solver_bone.get(AWB_CONTACT_AUTHORING_STATE_PROPERTY)
        if before_exists
        else None
    )
    journal.record(
        IDPropertyMutationReceipt(
            journal.next_ordinal(),
            _scope(plan, owner_ptr=int(owner_ptr), bag_ptr=None),
            int(owner_ptr),
            int(target_ptr),
            AWB_CONTACT_AUTHORING_STATE_PROPERTY,
            bool(before_exists),
            before_value,
        )
    )
    stage_write("set Contact authoring latch")
    _set_contact_authoring_latch(capability, contact_type)


def _raw_pose_field_value(target, field: RawFieldKind) -> tuple[float, ...]:
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
        raise ValueError(f"Unsupported Contact pose raw field: {field!r}")
    return tuple(float(component) for component in value)


def _journal_pose_matrix_seed(
    resolved,
    desired_matrix,
    plan: OperationPlan,
    journal: MutationJournal,
    stage_write,
    *,
    label: str,
    parent_pose_matrix=None,
) -> None:
    owner_ptr = _runtime_pointer(resolved.owner_object)
    target_ptr = _runtime_pointer(resolved.target)
    if owner_ptr is None or target_ptr is None:
        raise ContactAuthoringError(
            "Contact feedback authority lost hidden result runtime identity before commit."
        )
    scope = _scope(plan, owner_ptr=int(owner_ptr), bag_ptr=None)
    for field in (
        RawFieldKind.POSE_BONE_LOCATION,
        RawFieldKind.POSE_BONE_ROTATION_EULER,
        RawFieldKind.POSE_BONE_ROTATION_QUATERNION,
        RawFieldKind.POSE_BONE_ROTATION_AXIS_ANGLE,
        RawFieldKind.POSE_BONE_SCALE,
    ):
        journal.record(
            RawFieldMutationReceipt(
                journal.next_ordinal(),
                scope,
                int(owner_ptr),
                int(target_ptr),
                RawFieldSnapshot(field, _raw_pose_field_value(resolved.target, field)),
            )
        )

    pose_bone = resolved.target
    rest = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        basis = rest.inverted() @ desired_matrix
    else:
        parent_rest = pose_bone.parent.bone.matrix_local.copy()
        parent_pose = (
            parent_pose_matrix.copy()
            if parent_pose_matrix is not None
            else pose_bone.parent.matrix.copy()
        )
        basis = rest.inverted() @ parent_rest @ parent_pose.inverted() @ desired_matrix
    property_name, _representation, _indices, _values, mode = _rotation_descriptor(pose_bone)
    quaternion = basis.to_quaternion().normalized()
    if property_name == "rotation_quaternion":
        rotation = (quaternion.w, quaternion.x, quaternion.y, quaternion.z)
    elif property_name == "rotation_axis_angle":
        axis = quaternion.axis
        rotation = (quaternion.angle, axis.x, axis.y, axis.z)
    else:
        rotation = tuple(float(component) for component in quaternion.to_euler(mode))

    stage_write(label)
    pose_bone.location = basis.to_translation()
    setattr(pose_bone, property_name, rotation)


def _journal_generated_hinge_branch(
    capability,
    branch_sign: int | None,
    plan: OperationPlan,
    journal: MutationJournal,
    stage_write,
) -> None:
    if branch_sign is None:
        return
    solver_owner = capability.native_ik.solver_owner.target
    owner = capability.native_ik.solver_owner.owner_object
    owner_ptr = _runtime_pointer(owner)
    target_ptr = _runtime_pointer(solver_owner)
    if owner_ptr is None or target_ptr is None:
        raise ContactAuthoringError(
            "Sliding hinge authority lost hidden solver runtime identity before commit."
        )
    scope = _scope(plan, owner_ptr=int(owner_ptr), bag_ptr=None)
    raw_fields = (
        (RawFieldKind.POSE_BONE_LOCK_IK_X, "lock_ik_x"),
        (RawFieldKind.POSE_BONE_LOCK_IK_Y, "lock_ik_y"),
        (RawFieldKind.POSE_BONE_LOCK_IK_Z, "lock_ik_z"),
        (RawFieldKind.POSE_BONE_USE_IK_LIMIT_X, "use_ik_limit_x"),
        (RawFieldKind.POSE_BONE_USE_IK_LIMIT_Y, "use_ik_limit_y"),
        (RawFieldKind.POSE_BONE_USE_IK_LIMIT_Z, "use_ik_limit_z"),
        (RawFieldKind.POSE_BONE_IK_MIN_X, "ik_min_x"),
        (RawFieldKind.POSE_BONE_IK_MAX_X, "ik_max_x"),
        (RawFieldKind.POSE_BONE_IK_MIN_Y, "ik_min_y"),
        (RawFieldKind.POSE_BONE_IK_MAX_Y, "ik_max_y"),
        (RawFieldKind.POSE_BONE_IK_MIN_Z, "ik_min_z"),
        (RawFieldKind.POSE_BONE_IK_MAX_Z, "ik_max_z"),
    )
    for field, attribute in raw_fields:
        value = getattr(solver_owner, attribute)
        before = bool(value) if isinstance(value, bool) else float(value)
        journal.record(
            RawFieldMutationReceipt(
                journal.next_ordinal(),
                scope,
                int(owner_ptr),
                int(target_ptr),
                RawFieldSnapshot(field, before),
            )
        )
    stage_write(f"configure hidden Sliding hinge branch={int(branch_sign)}")
    from .rigped_humanoid_builder import configure_generated_rigped_ik_hinge_branch

    configure_generated_rigped_ik_hinge_branch(solver_owner, int(branch_sign))


def _journal_limb_fk_feedback_muted(
    capability,
    muted: bool,
    plan: OperationPlan,
    journal: MutationJournal,
    stage_write,
    *,
    expected_result: tuple[Any, ...] | None = None,
    expected_terminal=None,
    hinge_branch: int | None = None,
) -> None:
    """Apply derived FK-feedback authority while the Contact journal is still OPEN."""

    owner = capability.native_ik.solver_owner.owner_object
    owner_ptr = _runtime_pointer(owner)
    if owner_ptr is None:
        raise ContactAuthoringError(
            "Contact feedback authority lost owner runtime identity before commit."
        )
    constraints = (
        *capability.fk_copy_constraints,
        capability.terminal_fk_constraint,
    )
    desired = bool(muted)

    # Entering IK authority disconnects public FK Copy Rotation from the hidden
    # result chain. Seed the current evaluated result into hidden raw channels
    # inside the same journal first so native IK starts from the exact visible
    # pose instead of losing the axial/bend input supplied by FK feedback.
    if desired and any(not bool(constraint.mute) for constraint in constraints):
        if expected_result is None or expected_terminal is None:
            raise ContactAuthoringError(
                "Sliding feedback transition is missing its frozen result pose seed."
            )
        if len(expected_result) != len(capability.result_controls):
            raise ContactAuthoringError(
                "Sliding feedback transition result seed cardinality changed."
            )
        for index, (control, expected) in enumerate(
            zip(capability.result_controls, expected_result, strict=True)
        ):
            _journal_pose_matrix_seed(
                control,
                expected,
                plan,
                journal,
                stage_write,
                label=f"seed hidden Sliding result control {index}",
                parent_pose_matrix=(expected_result[index - 1] if index else None),
            )
        _journal_pose_matrix_seed(
            capability.result_terminal,
            expected_terminal,
            plan,
            journal,
            stage_write,
            label="seed hidden Sliding result terminal",
            parent_pose_matrix=expected_result[-1],
        )

    if desired:
        _journal_generated_hinge_branch(
            capability,
            hinge_branch,
            plan,
            journal,
            stage_write,
        )

    for constraint in constraints:
        before = bool(constraint.mute)
        if before == desired:
            continue
        constraint_ptr = _runtime_pointer(constraint)
        if constraint_ptr is None:
            raise ContactAuthoringError(
                "Contact feedback authority lost constraint runtime identity before commit."
            )
        journal.record(
            ConstraintMutationReceipt(
                journal.next_ordinal(),
                _scope(plan, owner_ptr=int(owner_ptr), bag_ptr=None),
                int(owner_ptr),
                int(constraint_ptr),
                RawFieldSnapshot(RawFieldKind.CONSTRAINT_MUTE, before),
            )
        )
        stage_write(
            "mute FK feedback for Sliding authority"
            if desired
            else "restore FK feedback for Free authority"
        )
        constraint.mute = desired


def _trace_contact_rollback_report(
    event: str,
    *,
    operation_id: str,
    journal: MutationJournal,
    context,
    rollback=None,
    **data,
) -> None:
    payload = {
        "journal_state": getattr(journal.state, "value", str(journal.state)),
        "receipt_count": len(journal.receipts),
        **data,
    }
    if rollback is not None:
        payload.update(
            rollback_status=getattr(rollback.status, "value", str(rollback.status)),
            residue_count=len(rollback.residue_receipts),
            residue_receipts=tuple(
                {
                    "ordinal": int(receipt.ordinal),
                    "type": type(receipt).__name__,
                }
                for receipt in rollback.residue_receipts
            ),
            quarantine=tuple(
                {
                    "character_id": item.character_id,
                    "setup_revision": int(item.setup_revision),
                    "setup_signature": item.setup_signature,
                    "owner_ptr": item.owner_ptr,
                    "bag_ptr": item.bag_ptr,
                }
                for item in rollback.quarantine_keys
            ),
            rollback_diagnostics=tuple(
                {
                    "code": item.code,
                    "detail": item.detail,
                }
                for item in rollback.diagnostics
            ),
        )
    trace_event(
        "WRITER",
        event,
        operation_id=operation_id,
        context=context,
        **payload,
    )


def _reset_contact_authoring_latch(capability) -> None:
    solver_bone = capability.native_ik.solver_owner.target
    solver_bone[AWB_CONTACT_AUTHORING_STATE_PROPERTY] = float(ContactStateValue.UNINITIALIZED)


def _next_enabled(current: ContactKeyType, enabled: tuple[ContactKeyType, ...]) -> ContactKeyType:
    index = enabled.index(current)
    return enabled[(index + 1) % len(enabled)]


def _mapping_selection_binding_ids(mapping, capability) -> frozenset[str]:
    allowed = {
        *capability.fk_binding_ids,
        capability.authored_terminal_binding_id,
        mapping.ik_target_binding_id,
    }
    if mapping.pole_binding_id:
        allowed.add(mapping.pole_binding_id)
    return frozenset(str(binding_id) for binding_id in allowed if binding_id)


def _selected_mapping(view, selected_binding_ids: tuple[str, ...]):
    selected = set(selected_binding_ids)
    candidates: list[tuple[Any, Any]] = []
    for mapping in view.definition.kinematics:
        resolution = resolve_limb_representation_capability(view, mapping.mapping_id)
        capability = resolution.capability
        if capability is None:
            continue
        allowed = _mapping_selection_binding_ids(mapping, capability)
        if selected and selected.issubset(allowed):
            candidates.append((mapping, capability))
    if len(candidates) != 1:
        return None, None
    return candidates[0]


def _selected_mappings(view, selected_binding_ids: tuple[str, ...]) -> tuple[tuple[Any, Any], ...]:
    """Partition native selection into dependency-independent generated limb domains.

    Every selected binding must belong to exactly one supported limb domain. Multiple
    selected controls from one limb deduplicate to one mapping. Cross-domain ambiguity
    and non-limb controls fail closed rather than silently dropping part of selection.
    """

    selected = set(selected_binding_ids)
    if not selected:
        return ()
    candidates: list[tuple[Any, Any, frozenset[str]]] = []
    for mapping in view.definition.kinematics:
        resolution = resolve_limb_representation_capability(view, mapping.mapping_id)
        capability = resolution.capability
        if capability is None:
            continue
        allowed = _mapping_selection_binding_ids(mapping, capability)
        if selected.intersection(allowed):
            candidates.append((mapping, capability, allowed))

    resolved: list[tuple[Any, Any]] = []
    covered: set[str] = set()
    for binding_id in selected:
        owners = tuple(item for item in candidates if binding_id in item[2])
        if len(owners) != 1:
            return ()
        mapping, capability, _allowed = owners[0]
        covered.add(binding_id)
        if all(existing[0].mapping_id != mapping.mapping_id for existing in resolved):
            resolved.append((mapping, capability))
    if covered != selected:
        return ()
    return tuple(sorted(resolved, key=lambda item: str(item[0].mapping_id)))


def selected_contact_owner_binding_ids(target) -> tuple[str, ...]:
    """Return selected semantic bindings that explicitly own Contact authoring."""

    by_id = {contract.binding_id: contract for contract in target.controls}
    return tuple(
        binding_id
        for binding_id in target.selected_binding_ids
        if binding_id in by_id and RigpedCapability.CONTACT_OWNER in by_id[binding_id].capabilities
    )


def _selected_contact_owner_mappings(view, target) -> tuple[tuple[Any, Any], ...]:
    """Resolve C-key domains from selected Hand/Foot Contact owners first.

    Toe or neighboring limb selections must not make an otherwise explicit
    Hand/Foot Contact command ambiguous. Cross-limb Hand/Foot multi-selection
    still resolves one mapping per selected Contact owner.
    """

    owner_ids = set(selected_contact_owner_binding_ids(target))
    if not owner_ids:
        return ()
    resolved: list[tuple[Any, Any]] = []
    covered: set[str] = set()
    for mapping in view.definition.kinematics:
        capability = resolve_limb_representation_capability(view, mapping.mapping_id).capability
        if capability is None:
            continue
        terminal_id = str(capability.authored_terminal_binding_id)
        if terminal_id not in owner_ids:
            continue
        resolved.append((mapping, capability))
        covered.add(terminal_id)
    if covered != owner_ids:
        return ()
    return tuple(sorted(resolved, key=lambda item: str(item[0].mapping_id)))


def _selected_contact_command_mappings(view, target) -> tuple[tuple[Any, Any], ...]:
    """Resolve all generated limb domains represented by the current C selection.

    Explicit Hand/Foot Contact owners remain authoritative for their own limb,
    but they must not hide other selected limb controls. Every selected control
    that resolves uniquely to a generated limb contributes that domain, while
    unrelated direct controls such as Root/Spine/Head are ignored.
    """

    resolved: list[tuple[Any, Any]] = list(
        _selected_contact_owner_mappings(view, target)
    )
    for binding_id in target.selected_binding_ids:
        owned = _selected_mappings(view, (binding_id,))
        if len(owned) != 1:
            continue
        mapping, capability = owned[0]
        if all(
            str(existing[0].mapping_id) != str(mapping.mapping_id)
            for existing in resolved
        ):
            resolved.append((mapping, capability))
    return tuple(sorted(resolved, key=lambda item: str(item[0].mapping_id)))


def selected_contact_command_mapping_ids(
    scene,
    control_context,
    *,
    selector_character_id: str | None = None,
) -> tuple[str, ...]:
    """Return generated limb domains that the current Rigped C command owns."""

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return ()
    view = resolve_character(scene, resolution.target.character_id)
    return tuple(
        str(mapping.mapping_id)
        for mapping, _capability in _selected_contact_command_mappings(view, resolution.target)
    )


def selected_contact_command_control_keys(
    scene,
    control_context,
    *,
    selector_character_id: str | None = None,
) -> frozenset[tuple[int, int]]:
    """Return selected native controls consumed by the current Contact C command."""

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    target = resolution.target
    if target is None:
        return frozenset()
    view = resolve_character(scene, target.character_id)
    mappings = _selected_contact_command_mappings(view, target)
    if not mappings:
        return frozenset()

    domain_binding_ids: set[str] = set()
    for mapping, capability in mappings:
        domain_binding_ids.update(_mapping_selection_binding_ids(mapping, capability))
    selected_ids = set(target.selected_binding_ids).intersection(domain_binding_ids)
    resolved_by_binding = dict(view.resolved_bindings)
    return frozenset(
        runtime_control_key(resolved_by_binding[binding_id])
        for binding_id in selected_ids
        if binding_id in resolved_by_binding
    )


def selected_contact_mapping_ids(
    scene,
    control_context,
    *,
    selector_character_id: str | None = None,
) -> tuple[str, ...]:
    """Return deduplicated generated limb Contact domains for the exact native selection."""

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return ()
    view = resolve_character(scene, resolution.target.character_id)
    return tuple(
        str(mapping.mapping_id)
        for mapping, _capability in _selected_mappings(view, resolution.target.selected_binding_ids)
    )


def selected_contact_limb_control_keys(
    scene,
    control_context,
    *,
    selector_character_id: str | None = None,
) -> frozenset[tuple[int, int]]:
    """Return selected native controls that belong to any generated limb domain.

    Unlike the C-command selector, this helper treats UpperArm/ForeArm/Thigh/Calf
    controls as first-class limb members. Auto transform routing uses it to
    distinguish an all-limb multi-selection from a true Contact/direct mix.
    """

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    target = resolution.target
    if target is None:
        return frozenset()
    view = resolve_character(scene, target.character_id)
    resolved_by_binding = dict(view.resolved_bindings)
    limb_binding_ids: set[str] = set()
    for binding_id in target.selected_binding_ids:
        if len(_selected_mappings(view, (binding_id,))) == 1:
            limb_binding_ids.add(binding_id)
    return frozenset(
        runtime_control_key(resolved_by_binding[binding_id])
        for binding_id in limb_binding_ids
        if binding_id in resolved_by_binding
    )


_CONTACT_DISPLAY_CACHE_KEY: tuple[Any, ...] | None = None
_CONTACT_DISPLAY_CACHE_ROWS: tuple[tuple[Any, str], ...] = ()


def _contact_display_selection_key(context) -> tuple[Any, ...] | None:
    if getattr(context, "mode", "") != "POSE":
        return None
    rig = getattr(context, "active_object", None)
    if rig is None or getattr(rig, "type", None) != "ARMATURE":
        return None
    selected = tuple(sorted(str(bone.name) for bone in (getattr(context, "selected_pose_bones", None) or ())))
    if not selected:
        return None
    try:
        # Undo may recreate Blender RNA wrappers while preserving the same
        # semantic rig/selection. Pointer-only cache keys can then hit rows that
        # still hold dead Object references. ID.session_uid changes with the
        # recreated datablock, so include it in the display-cache identity.
        rig_key = int(rig.as_pointer())
        data_key = int(rig.data.as_pointer())
        rig_session = int(getattr(rig, "session_uid", 0) or 0)
        data_session = int(getattr(rig.data, "session_uid", 0) or 0)
    except (ReferenceError, TypeError):
        return None
    return (rig_key, data_key, rig_session, data_session, selected)


def _contact_display_mappings(view, target) -> tuple[tuple[Any, Any], ...]:
    """Resolve every generated Contact limb touched by the native selection.

    Display aggregation is intentionally broader than Contact authoring. A mixed
    selection such as Hand.L + Thigh.L must color both limbs' semantic cells,
    while Root/Spine/other non-limb controls selected by a box-select are simply
    ignored. Authoring remains fail-closed through its stricter resolvers.
    """

    selected = set(target.selected_binding_ids)
    if not selected:
        return ()
    resolved: list[tuple[Any, Any]] = []
    for mapping in view.definition.kinematics:
        capability = resolve_limb_representation_capability(view, mapping.mapping_id).capability
        if capability is None:
            continue
        if not selected.intersection(_mapping_selection_binding_ids(mapping, capability)):
            continue
        resolved.append((mapping, capability))
    return tuple(sorted(resolved, key=lambda item: str(item[0].mapping_id)))


def _contact_display_rows(context) -> tuple[tuple[Any, str], ...]:
    """Resolve stable Contact state carriers only when the native selection changes."""

    global _CONTACT_DISPLAY_CACHE_KEY, _CONTACT_DISPLAY_CACHE_ROWS

    cache_key = _contact_display_selection_key(context)
    if cache_key is None:
        _CONTACT_DISPLAY_CACHE_KEY = None
        _CONTACT_DISPLAY_CACHE_ROWS = ()
        return ()
    if cache_key == _CONTACT_DISPLAY_CACHE_KEY:
        try:
            if all(int(owner.as_pointer()) != 0 for owner, _path in _CONTACT_DISPLAY_CACHE_ROWS):
                return _CONTACT_DISPLAY_CACHE_ROWS
        except (ReferenceError, TypeError):
            pass
        _CONTACT_DISPLAY_CACHE_KEY = None
        _CONTACT_DISPLAY_CACHE_ROWS = ()

    scene = getattr(context, "scene", None)
    control_context = control_context_for_context(context)
    resolution = resolve_rigped_target(scene, control_context)
    target = resolution.target
    rows: list[tuple[Any, str]] = []
    if target is not None and target.selected_binding_ids:
        view = resolve_character(scene, target.character_id)
        # Display is allowed to resolve Contact limbs from broad/mixed native
        # selections. Unlike authoring, it must keep every touched generated limb
        # instead of preferring Hand/Foot Contact owners and dropping another limb
        # such as Thigh/Calf from the same selection.
        mappings = _contact_display_mappings(view, target)
        for _mapping, capability in mappings:
            solver_owner = capability.native_ik.solver_owner
            solver_bone = solver_owner.target
            if AWB_CONTACT_STATE_PROPERTY not in solver_bone:
                continue
            rows.append(
                (
                    solver_owner.owner_object,
                    solver_bone.path_from_id(f'["{AWB_CONTACT_STATE_PROPERTY}"]'),
                )
            )

    _CONTACT_DISPLAY_CACHE_KEY = cache_key
    _CONTACT_DISPLAY_CACHE_ROWS = tuple(rows)
    return _CONTACT_DISPLAY_CACHE_ROWS


def contact_key_types_for_context(context) -> dict[float, ContactKeyType]:
    """Read Contact key colors without re-resolving Rigped semantics every redraw."""

    rows = _contact_display_rows(context)
    if not rows:
        return {}

    types_by_frame: dict[float, set[ContactKeyType]] = {}
    for owner, state_path in rows:
        try:
            bag = assigned_channelbag(owner)
            state_curve = _find_fcurve(bag, state_path, 0)
        except (AttributeError, ReferenceError, RuntimeError):
            continue
        if state_curve is None:
            continue
        for point in state_curve.keyframe_points:
            contact_type = type_for_state_value(float(point.co.y))
            if contact_type is None:
                continue
            types_by_frame.setdefault(float(point.co.x), set()).add(contact_type)

    # One Track Bar cell can aggregate Contact keys from several selected limbs.
    # A mixed semantic frame must stay semantic instead of falling through to the
    # ordinary green Rotation color. Use the strongest visible Contact state for
    # aggregate display: Planted > Sliding > Free. A5 currently exercises only
    # Free/Sliding; Planted remains reserved for A6.
    priority = (
        ContactKeyType.PLANTED,
        ContactKeyType.SLIDING,
        ContactKeyType.FREE,
    )
    return {
        frame: next(contact_type for contact_type in priority if contact_type in contact_types)
        for frame, contact_types in types_by_frame.items()
        if contact_types
    }


def contact_trackbar_cells_for_context(context) -> dict[float, bool]:
    """Expose visible Contact semantic cells to generic Track Bar hit-testing.

    Sliding keys may be authoritative on hidden IK/state carriers without a
    matching Hand/Foot transform FCurve on that frame. The Track Bar still
    draws those semantic cells, so they must also be selectable/editable.
    """

    contact_types = contact_key_types_for_context(context)
    if not contact_types:
        return {}

    selected_by_frame = {float(frame): False for frame in contact_types}
    for owner, state_path in _contact_display_rows(context):
        try:
            bag = assigned_channelbag(owner)
            state_curve = _find_fcurve(bag, state_path, 0)
        except (AttributeError, ReferenceError, RuntimeError):
            continue
        if state_curve is None:
            continue
        for point in state_curve.keyframe_points:
            frame = float(point.co.x)
            if frame not in selected_by_frame:
                continue
            selected_by_frame[frame] = selected_by_frame[frame] or bool(
                point.select_control_point
                or point.select_left_handle
                or point.select_right_handle
            )
    return selected_by_frame


def select_contact_trackbar_frames(
    context,
    *,
    frames: tuple[float, ...],
    mode: str,
) -> bool:
    """Mirror Track Bar selection onto complete Contact semantic bundles."""

    if mode not in {"SET", "ADD", "TOGGLE", "SUB"}:
        raise ValueError(f"Unsupported Contact Track Bar selection mode: {mode}")

    rows = _contact_display_rows(context)
    contact_frames = tuple(float(frame) for frame in contact_key_types_for_context(context))
    targets = {
        contact_frame
        for contact_frame in contact_frames
        if any(abs(contact_frame - float(frame)) <= 1e-4 for frame in frames)
    }
    selected_before = contact_trackbar_cells_for_context(context)
    found = bool(targets)
    changed = False

    # SET is global replacement semantics.  Clear stale/hidden Contact key
    # selection across every authored generated limb before selecting the
    # current limb's requested frames.  This keeps one Track Bar selection
    # instead of four independent arm/leg selection histories.
    if mode == "SET":
        changed = _set_contact_bundle_selection(
            _all_contact_bundle_selection_rows(context),
            None,
        ) or changed

    if not rows or not targets:
        return found or changed

    frame_states: dict[float, bool] = {}
    for target_frame in targets:
        currently_selected = bool(selected_before.get(target_frame, False))
        if mode in {"SET", "ADD"}:
            frame_states[target_frame] = True
        elif mode == "SUB":
            frame_states[target_frame] = False
        else:
            frame_states[target_frame] = not currently_selected

    bundle_rows = _contact_bundle_edit_rows_for_frames(
        context,
        tuple(sorted(targets)),
    )
    changed = _set_contact_bundle_selection(bundle_rows, frame_states) or changed
    return found or changed


def contact_point_preset_for_selection(
    scene,
    control_context,
    preset: str,
    *,
    selector_character_id: str | None = None,
) -> tuple[float, float, float]:
    """Map one thin 3x3 Foot/Hand preset onto the generalized terminal-local point."""

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        detail = "; ".join(issue.detail for issue in resolution.issues) or "Contact preset target is unresolved."
        raise ContactAuthoringError(detail)
    target = resolution.target
    view = resolve_character(scene, target.character_id)
    mapping, capability = _selected_mapping(view, target.selected_binding_ids)
    if mapping is None or capability is None:
        raise ContactAuthoringError("Select exactly one generated hand/foot Contact domain for a Contact point preset.")
    by_id = {contract.binding_id: contract for contract in target.controls}
    terminal = by_id.get(capability.authored_terminal_binding_id)
    if terminal is None or terminal.semantic_key not in {"awb.hand", "awb.foot"}:
        raise ContactAuthoringError("Contact point presets are limited to generated Hand/Foot terminals.")
    length = float(getattr(capability.result_terminal.target.bone, "length", 0.0) or 0.0)
    if not math.isfinite(length) or length <= 1e-8:
        raise ContactAuthoringError("Selected terminal has no usable rest length for a Contact point preset.")
    grid = {
        "REAR_LEFT": (-1.0, 0.0),
        "REAR_CENTER": (0.0, 0.0),
        "REAR_RIGHT": (1.0, 0.0),
        "MID_LEFT": (-1.0, 0.5),
        "MID_CENTER": (0.0, 0.5),
        "MID_RIGHT": (1.0, 0.5),
        "FRONT_LEFT": (-1.0, 1.0),
        "FRONT_CENTER": (0.0, 1.0),
        "FRONT_RIGHT": (1.0, 1.0),
    }
    if preset not in grid:
        raise ContactAuthoringError(f"Unknown Contact point preset: {preset!r}.")
    lateral, longitudinal = grid[preset]
    return (float(lateral * length * 0.25), float(longitudinal * length), 0.0)


def _scene_contains_object(scene, obj) -> bool:
    key = _runtime_pointer(obj)
    return key is not None and any(_runtime_pointer(candidate) == key for candidate in scene.objects)


def _is_external_placeholder(obj) -> bool:
    return bool(obj is not None and obj.get(AWB_CONTACT_EXTERNAL_PLACEHOLDER_PROPERTY, False))


def _external_placeholder_for_owner(owner):
    candidates = tuple(
        obj
        for collection in getattr(owner, "users_collection", ())
        for obj in collection.objects
        if _is_external_placeholder(obj)
    )
    unique = {int(_runtime_pointer(obj) or 0): obj for obj in candidates if _runtime_pointer(obj) is not None}
    if len(unique) != 1:
        raise ContactAuthoringError(
            "Generated Rigped must resolve exactly one owned external Contact placeholder Object."
        )
    return next(iter(unique.values()))


def _object_depends_on_rig(obj, rig) -> bool:
    """Conservatively reject obvious external->Rigped dependency cycles."""

    rig_key = _runtime_pointer(rig)
    if rig_key is None:
        return True
    cursor = getattr(obj, "parent", None)
    seen: set[int] = set()
    while cursor is not None:
        key = _runtime_pointer(cursor)
        if key is None or key in seen:
            break
        if key == rig_key:
            return True
        seen.add(key)
        cursor = getattr(cursor, "parent", None)

    for constraint in getattr(obj, "constraints", ()):
        if _runtime_pointer(getattr(constraint, "target", None)) == rig_key:
            return True

    animation_data = getattr(obj, "animation_data", None)
    for fcurve in getattr(animation_data, "drivers", ()) if animation_data is not None else ():
        driver = getattr(fcurve, "driver", None)
        for variable in getattr(driver, "variables", ()) if driver is not None else ():
            for target in getattr(variable, "targets", ()):
                if _runtime_pointer(getattr(target, "id", None)) == rig_key:
                    return True
    return False


def _validate_external_contact_target(scene, rig, external_object) -> None:
    if external_object is None or not _scene_contains_object(scene, external_object):
        raise ContactAuthoringError("Object-space Contact requires one live Object from the active Scene.")
    if _is_external_placeholder(external_object):
        raise ContactAuthoringError("Object-space Contact requires an explicitly bound external user Object.")
    if _runtime_pointer(external_object) == _runtime_pointer(rig):
        raise ContactAuthoringError("Rigped cannot use itself as an external Contact Object.")
    if _object_depends_on_rig(external_object, rig):
        raise ContactAuthoringError(
            "External Contact Object depends on this Rigped and would create an unsupported dependency cycle."
        )


def _external_contact_keys_exist(owner, hold: ContactHoldCapability) -> bool:
    bag = assigned_channelbag(owner)
    path = hold.external_constraint.path_from_id("influence")
    curve = _find_fcurve(bag, path, 0)
    if curve is None:
        return False
    return any(_same_float(float(point.co.y), 1.0) for point in curve.keyframe_points)


def bind_selected_contact_object(
    scene,
    control_context,
    external_object,
    *,
    operation_id: str = "baw-bind-contact-object",
    selector_character_id: str | None = None,
) -> ContactAuthoringResult:
    """Bind one fixed external Object identity to the selected limb Contact domain.

    I18 intentionally supports one external Object identity per limb. Once an
    Object-space Contact key exists, changing that identity would rewrite
    history, so retargeting fails closed until those keys are removed.
    """

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return ContactAuthoringResult(
            False,
            diagnostics=tuple(_diagnostic(operation_id, issue.code, issue.detail) for issue in resolution.issues),
        )
    target = resolution.target
    view = resolve_character(scene, target.character_id)
    mapping, capability = _selected_mapping(view, target.selected_binding_ids)
    if mapping is None or capability is None:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    operation_id,
                    "I18_CONTACT_SELECTION_AMBIGUOUS",
                    "Select exactly one generated hand/foot Contact domain before binding an external Object.",
                    character_id=target.character_id,
                ),
            ),
        )
    try:
        hold = _resolve_contact_hold(view, target, mapping, capability)
        owner = capability.native_ik.solver_owner.owner_object
        _validate_external_contact_target(scene, owner, external_object)
        current = hold.external_constraint.target
        if _runtime_pointer(current) == _runtime_pointer(external_object):
            return ContactAuthoringResult(True)
        if _external_contact_keys_exist(owner, hold):
            raise ContactAuthoringError(
                "This limb already owns Object-space Contact keys; changing its external Object would rewrite history."
            )
        if not _same_float(float(hold.external_constraint.influence), 0.0):
            raise ContactAuthoringError(
                "External Contact target can only change while its Object-space influence is zero."
            )
        hold.external_constraint.target = external_object
        try:
            hold.external_constraint.inverse_matrix = external_object.matrix_world.inverted()
        except ValueError as exc:
            raise ContactAuthoringError(
                "External Contact Object world transform is singular and cannot define a rigid attachment space."
            ) from exc
        bpy.context.view_layer.update()
        if _runtime_pointer(hold.external_constraint.target) != _runtime_pointer(external_object):
            raise ContactAuthoringError("External Contact Object binding did not persist.")
        return ContactAuthoringResult(True)
    except ContactAuthoringError as exc:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(operation_id, "I18_EXTERNAL_OBJECT_BIND_REJECTED", str(exc), character_id=target.character_id),
            ),
        )


def clear_selected_contact_object(
    scene,
    control_context,
    *,
    operation_id: str = "baw-clear-contact-object",
    selector_character_id: str | None = None,
) -> ContactAuthoringResult:
    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return ContactAuthoringResult(
            False,
            diagnostics=tuple(_diagnostic(operation_id, issue.code, issue.detail) for issue in resolution.issues),
        )
    target = resolution.target
    view = resolve_character(scene, target.character_id)
    mapping, capability = _selected_mapping(view, target.selected_binding_ids)
    if mapping is None or capability is None:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    operation_id,
                    "I18_CONTACT_SELECTION_AMBIGUOUS",
                    "Select exactly one generated hand/foot Contact domain before clearing its external Object.",
                    character_id=target.character_id,
                ),
            ),
        )
    try:
        hold = _resolve_contact_hold(view, target, mapping, capability)
        owner = capability.native_ik.solver_owner.owner_object
        if _external_contact_keys_exist(owner, hold):
            raise ContactAuthoringError(
                "This limb still owns Object-space Contact keys; clear is blocked to preserve historical replay."
            )
        if not _same_float(float(hold.external_constraint.influence), 0.0):
            raise ContactAuthoringError("Cannot clear an external Contact Object while Object-space influence is active.")
        hold.external_constraint.target = _external_placeholder_for_owner(owner)
        hold.external_constraint.inverse_matrix = Matrix.Identity(4)
        bpy.context.view_layer.update()
        return ContactAuthoringResult(True)
    except ContactAuthoringError as exc:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(operation_id, "I18_EXTERNAL_OBJECT_CLEAR_REJECTED", str(exc), character_id=target.character_id),
            ),
        )


def _contact_bundle_edit_rows_for_frames(
    context,
    frames: tuple[float, ...],
) -> tuple[tuple[Any, tuple[tuple[str, int], ...]], ...]:
    """Resolve complete generated-limb Contact curve bundles for Track Bar edits.

    The public Track Bar cell is an aggregate semantic frame, while Contact owns
    hidden IK/pole/result/hold curves on the same Armature ChannelBag. Generic
    Move/Clone/Delete must therefore mutate the whole generated limb bundle or
    not mutate it at all. Only frames with an authored Contact-state key expand.
    """

    if not frames:
        return ()
    scene = getattr(context, "scene", None)
    if scene is None:
        return ()
    control_context = control_context_for_context(context)
    resolution = resolve_rigped_target(scene, control_context)
    target = resolution.target
    if target is None or not target.selected_binding_ids:
        return ()

    view = resolve_character(scene, target.character_id)
    mappings = _selected_contact_owner_mappings(view, target)
    if not mappings:
        mappings = _selected_mappings(view, target.selected_binding_ids)
    if not mappings:
        return ()

    grouped: dict[int, tuple[Any, dict[tuple[str, int], None]]] = {}
    for mapping, capability in mappings:
        try:
            hold = _resolve_contact_hold(view, target, mapping, capability)
            state_path = _contact_paths(capability, hold)[0]
        except ContactAuthoringError:
            continue
        owner = capability.native_ik.solver_owner.owner_object
        bag = assigned_channelbag(owner)
        if bag is None:
            continue
        state_curve = _find_fcurve(bag, state_path, 0)
        if state_curve is None or not any(
            _key_at_frame(state_curve, float(frame)) is not None for frame in frames
        ):
            continue

        controls = [
            *capability.fk_controls,
            *capability.result_controls,
            capability.authored_terminal,
            capability.result_terminal,
            capability.native_ik.solver_owner,
            capability.native_ik.ik_target,
            capability.native_ik.pole_target,
            hold.control.target,
            hold.point_control.target,
        ]
        prefixes: set[str] = set()
        for resolved in controls:
            if resolved is None:
                continue
            native = resolved.target
            if isinstance(native, bpy.types.PoseBone):
                prefixes.add(str(native.path_from_id()))

        owner_key = int(owner.as_pointer())
        row_owner, curve_keys = grouped.setdefault(owner_key, (owner, {}))
        del row_owner
        for curve in bag.fcurves:
            data_path = str(curve.data_path)
            if any(data_path.startswith(prefix) for prefix in prefixes):
                curve_keys.setdefault(
                    (data_path, int(getattr(curve, "array_index", 0))),
                    None,
                )

    return tuple(
        (owner, tuple(curve_keys))
        for owner, curve_keys in grouped.values()
        if curve_keys
    )


def _all_contact_bundle_selection_rows(
    context,
) -> tuple[tuple[Any, tuple[tuple[str, int], ...]], ...]:
    """Resolve every authored generated-limb Contact bundle on the active Rigped.

    Track Bar key selection is one global UI state, not one hidden selection set
    per limb.  A plain SET/clear operation therefore needs to clear Contact key
    selection on limbs that are not currently selected in Pose Mode as well.
    """

    scene = getattr(context, "scene", None)
    if scene is None:
        return ()
    resolution = resolve_rigped_target(scene, control_context_for_context(context))
    target = resolution.target
    if target is None:
        return ()
    view = resolve_character(scene, target.character_id)

    mappings: list[tuple[Any, Any]] = []
    for mapping in view.definition.kinematics:
        capability = resolve_limb_representation_capability(view, mapping.mapping_id).capability
        if capability is not None:
            mappings.append((mapping, capability))

    grouped: dict[int, tuple[Any, dict[tuple[str, int], None]]] = {}
    for mapping, capability in mappings:
        try:
            hold = _resolve_contact_hold(view, target, mapping, capability)
            state_path = _contact_paths(capability, hold)[0]
        except ContactAuthoringError:
            continue
        owner = capability.native_ik.solver_owner.owner_object
        bag = assigned_channelbag(owner)
        if bag is None:
            continue
        state_curve = _find_fcurve(bag, state_path, 0)
        if state_curve is None or not state_curve.keyframe_points:
            continue

        controls = [
            *capability.fk_controls,
            *capability.result_controls,
            capability.authored_terminal,
            capability.result_terminal,
            capability.native_ik.solver_owner,
            capability.native_ik.ik_target,
            capability.native_ik.pole_target,
            hold.control.target,
            hold.point_control.target,
        ]
        prefixes: set[str] = set()
        for resolved in controls:
            if resolved is None:
                continue
            native = resolved.target
            if isinstance(native, bpy.types.PoseBone):
                prefixes.add(str(native.path_from_id()))

        owner_key = int(owner.as_pointer())
        _row_owner, curve_keys = grouped.setdefault(owner_key, (owner, {}))
        for curve in bag.fcurves:
            data_path = str(curve.data_path)
            if any(data_path.startswith(prefix) for prefix in prefixes):
                curve_keys.setdefault(
                    (data_path, int(getattr(curve, "array_index", 0))),
                    None,
                )

    return tuple(
        (owner, tuple(curve_keys))
        for owner, curve_keys in grouped.values()
        if curve_keys
    )


def _set_contact_bundle_selection(
    rows,
    frame_states: dict[float, bool] | None,
) -> bool:
    """Set Contact bundle key/handle selection, or clear every point when None."""

    changed = False
    for owner, curve_keys in rows:
        bag = assigned_channelbag(owner)
        if bag is None:
            continue
        expected = {(str(path), int(index)) for path, index in curve_keys}
        for curve in bag.fcurves:
            curve_key = (str(curve.data_path), int(getattr(curve, "array_index", 0)))
            if curve_key not in expected:
                continue
            for point in curve.keyframe_points:
                if frame_states is None:
                    selected = False
                else:
                    selected = None
                    point_frame = float(point.co.x)
                    for frame, frame_selected in frame_states.items():
                        if abs(point_frame - float(frame)) <= 1e-4:
                            selected = bool(frame_selected)
                            break
                    if selected is None:
                        continue
                previous = bool(
                    point.select_control_point
                    or point.select_left_handle
                    or point.select_right_handle
                )
                changed = changed or previous != selected
                point.select_control_point = selected
                point.select_left_handle = selected
                point.select_right_handle = selected
    return changed


def expand_generic_trackbar_edit_for_contact(
    context,
    *,
    frames: tuple[float, ...],
):
    """Add hidden generated Contact curves to aggregate Track Bar mutations."""

    return _contact_bundle_edit_rows_for_frames(context, tuple(float(frame) for frame in frames))


def clear_contact_bundle_key_selection_for_context(
    context,
    *,
    frames: tuple[float, ...],
) -> bool:
    """Deselect every authored key in the selected generated Contact limb bundle.

    Contact authoring writes dependency rotations on the whole two-bone chain in
    addition to the visible terminal key. Blender selects newly inserted points
    by default, so clearing only the currently selected Hand/Foot leaves hidden
    UpperArm/ForeArm or Thigh/Calf points selected. When those controls are later
    selected, Track Bar incorrectly enters Selection Range mode. Clear the exact
    Contact bundle rows at the authored frame instead.
    """

    frozen_frames = tuple(float(frame) for frame in frames)
    if not frozen_frames:
        return False
    changed = False
    for owner, curve_keys in _contact_bundle_edit_rows_for_frames(context, frozen_frames):
        bag = assigned_channelbag(owner)
        if bag is None:
            continue
        expected = {(str(path), int(index)) for path, index in curve_keys}
        for curve in bag.fcurves:
            curve_key = (str(curve.data_path), int(getattr(curve, "array_index", 0)))
            if curve_key not in expected:
                continue
            for point in curve.keyframe_points:
                if not any(abs(float(point.co.x) - frame) <= 1e-4 for frame in frozen_frames):
                    continue
                selected = bool(
                    point.select_control_point
                    or point.select_left_handle
                    or point.select_right_handle
                )
                point.select_control_point = False
                point.select_left_handle = False
                point.select_right_handle = False
                changed = changed or selected
    return changed


def allow_generic_trackbar_edit_for_contact(
    context,
    *,
    operation: str,
    source_frames: tuple[float, ...],
    target_frames: tuple[float, ...],
) -> bool:
    """Protect Contact frames while allowing bundle-aware basic Track Bar edits.

    Move/Clone/Delete are safe when the source frame owns an explicit Contact
    state key because the registered Contact edit expander adds every generated
    limb/IK/pole/hold curve from that ChannelBag to the same mutation target.
    Selection Range move/scale use the same complete bundle expansion; unrelated
    property edits remain fail-closed, and an ordinary source frame may not be
    moved/cloned on top of an existing Contact frame.
    """

    source_rows = _contact_bundle_edit_rows_for_frames(
        context,
        tuple(float(frame) for frame in source_frames),
    )
    target_rows = _contact_bundle_edit_rows_for_frames(
        context,
        tuple(float(frame) for frame in target_frames),
    )
    if not source_rows and not target_rows:
        return True

    operation = str(operation).upper()
    return bool(
        operation
        in {
            "MOVE",
            "CLONE",
            "DELETE",
            "PREVIEW_MOVE",
            "PREVIEW_CLONE",
            "SELECTION_RANGE_MOVE",
            "SELECTION_RANGE_SCALE",
        }
        and source_rows
    )


def finalize_generic_trackbar_delete_for_contact(
    context,
    *,
    frames: tuple[float, ...],
) -> bool:
    """Return a limb to pristine unkeyed Contact state after its last key is deleted.

    Generic key deletion removes keyframe points but Blender leaves empty FCurves
    and the last evaluated custom-property/constraint values in memory. Contact
    preflight intentionally rejects that half-authored state. When the selected
    generated limb has no Contact state keys left and all Contact-owned companion
    curves are likewise empty, remove those empty authority curves and restore
    the generated UNINITIALIZED/physical-Free runtime state.
    """

    del frames  # Selection determines the generated limb domain after deletion.
    scene = getattr(context, "scene", None)
    if scene is None:
        return False
    resolution = resolve_rigped_target(scene, control_context_for_context(context))
    target = resolution.target
    if target is None or not target.selected_binding_ids:
        return False
    view = resolve_character(scene, target.character_id)
    mappings = _selected_contact_owner_mappings(view, target)
    if not mappings:
        mappings = _selected_mappings(view, target.selected_binding_ids)
    if not mappings:
        return False

    changed = False
    for mapping, capability in mappings:
        try:
            hold = _resolve_contact_hold(view, target, mapping, capability)
            (
                state_path,
                ik_path,
                terminal_path,
                pole_angle_path,
                hold_influence_path,
                pivot_influence_path,
                external_influence_path,
            ) = _contact_paths(capability, hold)
        except ContactAuthoringError:
            continue

        owner = capability.native_ik.solver_owner.owner_object
        bag = assigned_channelbag(owner)
        if bag is None:
            continue
        state_curve = _find_fcurve(bag, state_path, 0)
        if state_curve is not None and len(state_curve.keyframe_points) > 0:
            continue

        authority_paths = {
            state_path,
            ik_path,
            terminal_path,
            pole_angle_path,
            hold_influence_path,
            pivot_influence_path,
            external_influence_path,
            control_property_path(hold.control.target, "location"),
            control_property_path(hold.point_control.target, "location"),
        }
        authority_curves = tuple(
            curve
            for curve in bag.fcurves
            if str(curve.data_path) in authority_paths
        )
        # If any Contact-owned authority row still contains keys, this is not a
        # clean "last Contact key deleted" state and must not be silently reset.
        if any(len(curve.keyframe_points) > 0 for curve in authority_curves):
            continue

        for curve in authority_curves:
            bag.fcurves.remove(curve)

        solver_bone = capability.native_ik.solver_owner.target
        solver_bone[AWB_CONTACT_STATE_PROPERTY] = float(ContactStateValue.UNINITIALIZED)
        _reset_contact_authoring_latch(capability)
        capability.native_ik.constraint.influence = 0.0
        capability.terminal_ik_constraint.influence = 0.0
        hold.constraint.influence = 0.0
        hold.pivot_constraint.influence = 0.0
        hold.external_constraint.influence = 0.0
        owner.update_tag(refresh={"OBJECT", "DATA", "TIME"})
        changed = True

    if changed:
        view_layer = getattr(context, "view_layer", None)
        if view_layer is not None:
            view_layer.update()
    return changed


def build_contact_intent_plan(
    scene,
    control_context,
    *,
    operation_id: str,
    mode: ContactAuthoringMode = ContactAuthoringMode.CYCLE,
    enabled_types: tuple[ContactKeyType, ...] = (
        ContactKeyType.FREE,
        ContactKeyType.SLIDING,
        ContactKeyType.PLANTED,
    ),
    plant_space: ContactPlantSpace = ContactPlantSpace.WORLD,
    contact_point_local: tuple[float, float, float] | None = None,
    mapping_id: str | None = None,
    selector_character_id: str | None = None,
    forced_cycle_type: ContactKeyType | None = None,
) -> ContactPlanBuildResult:
    enabled = tuple(dict.fromkeys(ContactKeyType(item) for item in enabled_types))
    forced_cycle_type = (
        ContactKeyType(forced_cycle_type)
        if forced_cycle_type is not None
        else None
    )
    plant_space = ContactPlantSpace(plant_space)
    point_local = tuple(float(component) for component in (contact_point_local or (0.0, 0.0, 0.0)))
    if len(point_local) != 3 or any(not math.isfinite(component) for component in point_local):
        return ContactPlanBuildResult(
            None,
            (_diagnostic(operation_id, "I19_INVALID_CONTACT_POINT", "Contact point must contain three finite local coordinates."),),
        )
    if mode is ContactAuthoringMode.CYCLE and not enabled:
        return ContactPlanBuildResult(
            None,
            (_diagnostic(operation_id, "I13_NO_ENABLED_CONTACT_TYPES", "No Contact types are enabled; C is a no-op."),),
        )
    if (
        mode is ContactAuthoringMode.CYCLE
        and forced_cycle_type is not None
        and forced_cycle_type not in enabled
    ):
        return ContactPlanBuildResult(
            None,
            (
                _diagnostic(
                    operation_id,
                    "I13_FORCED_CONTACT_TYPE_DISABLED",
                    "Forced multi-limb Contact target is not enabled in the current C cycle.",
                ),
            ),
        )

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return ContactPlanBuildResult(
            None,
            tuple(_diagnostic(operation_id, issue.code, issue.detail) for issue in resolution.issues),
        )
    target = resolution.target
    view = resolve_character(scene, target.character_id)
    if mapping_id is None:
        mapping, capability = _selected_mapping(view, target.selected_binding_ids)
    else:
        mapping = next(
            (item for item in view.definition.kinematics if item.mapping_id == mapping_id),
            None,
        )
        capability = (
            resolve_limb_representation_capability(view, mapping_id).capability
            if mapping is not None
            else None
        )
        if (
            mapping is not None
            and capability is not None
            and not set(target.selected_binding_ids).intersection(
                _mapping_selection_binding_ids(mapping, capability)
            )
        ):
            mapping = None
            capability = None
    if mapping is None or capability is None:
        return ContactPlanBuildResult(
            None,
            (
                _diagnostic(
                    operation_id,
                    "I13_CONTACT_SELECTION_AMBIGUOUS",
                    "Current native selection must resolve the requested generated limb Contact domain.",
                    character_id=target.character_id,
                ),
            ),
        )

    base_result = build_kinematic_dependency_plan(
        scene,
        control_context,
        operation_id=f"{operation_id}:snap",
        mapping_id=mapping.mapping_id,
        selector_character_id=selector_character_id,
    )
    if not base_result.ok or base_result.plan is None:
        return ContactPlanBuildResult(None, base_result.diagnostics)
    snap_plan = base_result.plan

    owner = capability.native_ik.solver_owner.owner_object
    owner_runtime_key = _runtime_pointer(owner)
    if owner_runtime_key is None:
        return ContactPlanBuildResult(
            None,
            (_diagnostic(operation_id, "I13_CONTACT_OWNER_IDENTITY_MISSING", "Contact animation owner has no live runtime identity."),),
        )
    if getattr(owner, "library", None) is not None or getattr(owner, "override_library", None) is not None:
        return ContactPlanBuildResult(
            None,
            (_diagnostic(operation_id, "I13_READ_ONLY_CONTACT_OWNER", "Linked/override Contact animation owners are fail-closed in I13."),),
        )

    try:
        hold = _resolve_contact_hold(view, target, mapping, capability)
        (
            state_path,
            ik_path,
            terminal_path,
            pole_angle_path,
            hold_influence_path,
            pivot_influence_path,
            external_influence_path,
        ) = _contact_paths(capability, hold)
        bag = assigned_channelbag(owner)
        state_curve = _find_fcurve(bag, state_path, 0)
        ik_curve = _find_fcurve(bag, ik_path, 0)
        terminal_curve = _find_fcurve(bag, terminal_path, 0)
        pole_curve = _find_fcurve(bag, pole_angle_path, 0)
        hold_curve = _find_fcurve(bag, hold_influence_path, 0)
        pivot_curve = _find_fcurve(bag, pivot_influence_path, 0)
        external_curve = _find_fcurve(bag, external_influence_path, 0)
        _validate_existing_track(
            scene,
            target,
            capability,
            hold,
            state_curve,
            ik_curve,
            terminal_curve,
            pole_curve,
            hold_curve,
            pivot_curve,
            external_curve,
        )
        time = float(scene.frame_current) + float(getattr(scene, "frame_subframe", 0.0))
        exact_type, effective_type = _resolve_state_types(state_curve, time)

        physical_ik = _evaluated_scalar(ik_curve, time, float(capability.native_ik.constraint.influence))
        physical_terminal = _evaluated_scalar(
            terminal_curve,
            time,
            float(capability.terminal_ik_constraint.influence),
        )
        physical_hold = _evaluated_scalar(hold_curve, time, float(hold.constraint.influence))
        physical_pivot = _evaluated_scalar(pivot_curve, time, float(hold.pivot_constraint.influence))
        physical_external = _evaluated_scalar(
            external_curve,
            time,
            float(hold.external_constraint.influence),
        )
        if not _same_float(physical_ik, physical_terminal):
            raise ContactAuthoringError("Current Contact influence companions disagree at the operation time.")
        if not (_same_float(physical_pivot, 0.0) or _same_float(physical_pivot, 1.0)):
            raise ContactAuthoringError("Current continuous Contact-point pivot influence is not discrete.")
        if not (_same_float(physical_external, 0.0) or _same_float(physical_external, 1.0)):
            raise ContactAuthoringError("Current external Contact-space influence is not discrete.")
        physical_tuple = (
            round(physical_ik),
            round(physical_hold),
            round(physical_pivot),
            round(physical_external),
        )
        if physical_tuple not in {
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (1, 1, 1, 0),
            (1, 1, 0, 1),
        }:
            raise ContactAuthoringError("Current Contact influence is outside the exact Free/Sliding/Planted domain.")
        if _same_float(physical_ik, 0.0):
            physical_type = ContactKeyType.FREE
        elif _same_float(physical_hold, 1.0):
            physical_type = ContactKeyType.PLANTED
        else:
            physical_type = ContactKeyType.SLIDING
        physical_plant_space = (
            ContactPlantSpace.OBJECT
            if physical_type is ContactKeyType.PLANTED and _same_float(physical_external, 1.0)
            else ContactPlantSpace.WORLD
        )

        if mode is ContactAuthoringMode.REPLANT:
            if physical_type is not ContactKeyType.PLANTED:
                raise ContactAuthoringError("Replant requires the selected Contact domain to be currently Planted.")
            target_type = ContactKeyType.PLANTED
        elif mode is ContactAuthoringMode.ANCHOR:
            # KEY/All Key must preserve native authored authority even if the
            # animator later removes that type from the C cycle membership.
            target_type = exact_type or effective_type or physical_type
        elif forced_cycle_type is not None:
            target_type = forced_cycle_type
        elif exact_type is not None and exact_type in enabled:
            target_type = _next_enabled(exact_type, enabled)
        else:
            # New-frame C does not derive its semantic from timeline direction.
            # It repeats the last semantic explicitly confirmed by Contact
            # authoring for this limb, whether the insertion frame is before or
            # after existing keys. The left/effective state is only a migration
            # fallback for files authored before the latch property existed.
            insertion_type = _contact_authoring_latch_type(capability) or effective_type
            target_type = insertion_type if insertion_type in enabled else enabled[0]

        apply_contact_point = (
            target_type is ContactKeyType.PLANTED
            and (mode is ContactAuthoringMode.REPLANT or physical_type is not ContactKeyType.PLANTED)
        )
        target_plant_space = ContactPlantSpace.WORLD
        external_target_runtime_key: int | None = None
        if target_type is ContactKeyType.PLANTED:
            if mode is ContactAuthoringMode.ANCHOR or (
                mode is ContactAuthoringMode.CYCLE
                and physical_type is ContactKeyType.PLANTED
                and not apply_contact_point
            ):
                target_plant_space = physical_plant_space
            else:
                target_plant_space = plant_space
            if target_plant_space is ContactPlantSpace.OBJECT:
                _validate_external_contact_target(
                    scene,
                    owner,
                    hold.external_constraint.target,
                )
                if apply_contact_point and any(abs(component) > 1e-8 for component in point_local):
                    raise ContactAuthoringError(
                        "I19 arbitrary non-origin Contact points are currently proven only in World space; "
                        "Object-space Replant/Plant requires the local origin preset."
                    )
                external_target_runtime_key = _runtime_pointer(hold.external_constraint.target)
                if external_target_runtime_key is None:
                    raise ContactAuthoringError("External Contact Object has no live runtime identity.")

        transition_payload = None
        if physical_type is ContactKeyType.FREE and target_type is not ContactKeyType.FREE:
            transition_payload = build_representation_snap_payload(capability, SnapDirection.FK_TO_IK)
        elif physical_type is not ContactKeyType.FREE and target_type is ContactKeyType.FREE:
            transition_payload = build_representation_snap_payload(capability, SnapDirection.IK_TO_FK)
        representation_identity = build_representation_snap_payload(capability, SnapDirection.FK_TO_IK)

        pole_cleanup_fcurve_token: int | None = None
        pole_cleanup_snapshot: FCurveSnapshot | None = None
        if (
            mode is ContactAuthoringMode.CYCLE
            and exact_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}
            and target_type is ContactKeyType.FREE
        ):
            pole_curve = _find_fcurve(bag, pole_angle_path, 0)
            if pole_curve is None or _curve_key_at(pole_curve, time) is None:
                raise ContactAuthoringError(
                    "Same-frame Sliding→Free replacement is missing its authoritative pole-angle key."
                )
            pole_cleanup_fcurve_token = _runtime_pointer(pole_curve)
            if pole_cleanup_fcurve_token is None:
                raise ContactAuthoringError("Sliding pole-angle FCurve has no live runtime identity.")
            pole_cleanup_snapshot = _capture_fcurve_snapshot(pole_curve)

        hold_cleanup_curves: list[ContactCurveCleanup] = []
        if (
            mode is ContactAuthoringMode.CYCLE
            and exact_type is ContactKeyType.PLANTED
            and target_type is not ContactKeyType.PLANTED
        ):
            cleanup_paths = (
                (control_property_path(hold.control.target, "location"), "point-hold"),
                (control_property_path(hold.point_control.target, "location"), "continuous-point"),
            )
            for location_path, label in cleanup_paths:
                for index in range(3):
                    curve = _find_fcurve(bag, location_path, index)
                    if curve is None or _curve_key_at(curve, time) is None:
                        raise ContactAuthoringError(
                            f"Same-frame Planted replacement is missing its authoritative {label} anchor key."
                        )
                    token = _runtime_pointer(curve)
                    if token is None:
                        raise ContactAuthoringError(f"{label.title()} anchor FCurve has no live runtime identity.")
                    hold_cleanup_curves.append(
                        ContactCurveCleanup(int(token), _capture_fcurve_snapshot(curve))
                    )

        scalar_channels = (
            _build_scalar_channel(
                owner,
                label="CONTACT_STATE",
                data_path=state_path,
                value=float(state_value_for_type(target_type)),
            ),
            _build_scalar_channel(
                owner,
                label="IK_INFLUENCE",
                data_path=ik_path,
                value=0.0 if target_type is ContactKeyType.FREE else 1.0,
            ),
            _build_scalar_channel(
                owner,
                label="TERMINAL_IK_INFLUENCE",
                data_path=terminal_path,
                value=0.0 if target_type is ContactKeyType.FREE else 1.0,
            ),
            _build_scalar_channel(
                owner,
                label="POINT_HOLD_INFLUENCE",
                data_path=hold_influence_path,
                value=1.0 if target_type is ContactKeyType.PLANTED else 0.0,
            ),
            _build_scalar_channel(
                owner,
                label="CONTACT_POINT_PIVOT_INFLUENCE",
                data_path=pivot_influence_path,
                value=(
                    1.0
                    if target_type is ContactKeyType.PLANTED
                    and target_plant_space is ContactPlantSpace.WORLD
                    else 0.0
                ),
            ),
            _build_scalar_channel(
                owner,
                label="EXTERNAL_OBJECT_SPACE_INFLUENCE",
                data_path=external_influence_path,
                value=(
                    1.0
                    if target_type is ContactKeyType.PLANTED
                    and target_plant_space is ContactPlantSpace.OBJECT
                    else 0.0
                ),
            ),
        )
        if target_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}:
            scalar_channels = scalar_channels + (
                _build_scalar_channel(
                    owner,
                    label="IK_POLE_ANGLE",
                    data_path=pole_angle_path,
                    value=float(capability.native_ik.constraint.pole_angle),
                ),
            )
        baseline_required = state_curve is None and time > float(scene.frame_start)
    except (ContactAuthoringError, ValueError) as exc:
        return ContactPlanBuildResult(
            None,
            (_diagnostic(operation_id, "I13_CONTACT_PREFLIGHT_REJECTED", str(exc), character_id=target.character_id),),
        )

    return ContactPlanBuildResult(
        ContactIntentPlan(
            operation_id=operation_id,
            snap_plan=snap_plan,
            mapping_id=mapping.mapping_id,
            target_type=target_type,
            target_plant_space=target_plant_space,
            contact_point_local=point_local if apply_contact_point else None,
            apply_contact_point=apply_contact_point,
            external_target_runtime_key=(
                int(external_target_runtime_key)
                if external_target_runtime_key is not None
                else None
            ),
            effective_type=effective_type,
            exact_current_type=exact_type,
            owner_runtime_key=int(owner_runtime_key),
            owner_binding_token=channel_binding_token(owner),
            representation_identity=representation_identity,
            transition_payload=transition_payload,
            hold_capability=hold,
            scalar_channels=scalar_channels,
            pole_cleanup_fcurve_token=pole_cleanup_fcurve_token,
            pole_cleanup_snapshot=pole_cleanup_snapshot,
            hold_cleanup_curves=tuple(hold_cleanup_curves),
            baseline_required=baseline_required,
            enabled_types=enabled,
        )
    )


def build_contact_batch_intent_plan(
    scene,
    control_context,
    *,
    operation_id: str,
    mode: ContactAuthoringMode = ContactAuthoringMode.CYCLE,
    enabled_types: tuple[ContactKeyType, ...] = (
        ContactKeyType.FREE,
        ContactKeyType.SLIDING,
        ContactKeyType.PLANTED,
    ),
    plant_space: ContactPlantSpace = ContactPlantSpace.WORLD,
    contact_point_local: tuple[float, float, float] | None = None,
    selector_character_id: str | None = None,
    mapping_ids: tuple[str, ...] | None = None,
    forced_mapping_types: dict[str, ContactKeyType] | None = None,
) -> ContactBatchPlanBuildResult:
    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return ContactBatchPlanBuildResult(
            None,
            tuple(_diagnostic(operation_id, issue.code, issue.detail) for issue in resolution.issues),
        )
    target = resolution.target
    view = resolve_character(scene, target.character_id)
    if mapping_ids is None:
        mappings = _selected_contact_command_mappings(view, target)
    else:
        requested = tuple(dict.fromkeys(str(mapping_id) for mapping_id in mapping_ids))
        resolved_mappings: list[tuple[Any, Any]] = []
        for mapping_id in requested:
            mapping = next(
                (
                    item
                    for item in view.definition.kinematics
                    if str(item.mapping_id) == mapping_id
                ),
                None,
            )
            if mapping is None:
                return ContactBatchPlanBuildResult(
                    None,
                    (
                        _diagnostic(
                            operation_id,
                            "I20_UNKNOWN_MAPPING",
                            f"Unknown generated limb mapping {mapping_id!r}.",
                            character_id=target.character_id,
                        ),
                    ),
                )
            capability = resolve_limb_representation_capability(
                view,
                mapping.mapping_id,
            ).capability
            if capability is None:
                return ContactBatchPlanBuildResult(
                    None,
                    (
                        _diagnostic(
                            operation_id,
                            "I20_MAPPING_CAPABILITY_UNAVAILABLE",
                            f"Generated limb mapping {mapping_id!r} has no runtime capability.",
                            character_id=target.character_id,
                        ),
                    ),
                )
            resolved_mappings.append((mapping, capability))
        mappings = tuple(resolved_mappings)
    if len(mappings) < 2:
        return ContactBatchPlanBuildResult(
            None,
            (
                _diagnostic(
                    operation_id,
                    "I20_MULTI_LIMB_SELECTION_REQUIRED",
                    "I20 batch Contact requires native selection spanning at least two unambiguous generated limb domains.",
                    character_id=target.character_id,
                ),
            ),
        )

    forced_targets: dict[str, ContactKeyType] | None = None
    if forced_mapping_types is not None:
        if mode is not ContactAuthoringMode.CYCLE:
            return ContactBatchPlanBuildResult(
                None,
                (
                    _diagnostic(
                        operation_id,
                        "I20_FORCED_MAPPING_TYPES_REQUIRE_CYCLE",
                        "Explicit per-mapping Contact targets are supported only for CYCLE planning.",
                        character_id=target.character_id,
                    ),
                ),
            )
        try:
            forced_targets = {
                str(mapping_id): ContactKeyType(contact_type)
                for mapping_id, contact_type in forced_mapping_types.items()
            }
        except (TypeError, ValueError) as exc:
            return ContactBatchPlanBuildResult(
                None,
                (
                    _diagnostic(
                        operation_id,
                        "I20_INVALID_FORCED_MAPPING_TYPE",
                        f"Explicit per-mapping Contact target is invalid: {exc}",
                        character_id=target.character_id,
                    ),
                ),
            )
        resolved_mapping_ids = tuple(str(mapping.mapping_id) for mapping, _capability in mappings)
        missing = tuple(
            mapping_id
            for mapping_id in resolved_mapping_ids
            if mapping_id not in forced_targets
        )
        extra = tuple(
            mapping_id
            for mapping_id in forced_targets
            if mapping_id not in set(resolved_mapping_ids)
        )
        if missing or extra:
            return ContactBatchPlanBuildResult(
                None,
                (
                    _diagnostic(
                        operation_id,
                        "I20_FORCED_MAPPING_COVERAGE_MISMATCH",
                        "Explicit per-mapping Contact targets must exactly cover the frozen batch "
                        f"(missing={missing!r}, extra={extra!r}).",
                        character_id=target.character_id,
                    ),
                ),
            )

    shared_cycle_type: ContactKeyType | None = None
    if mode is ContactAuthoringMode.CYCLE and forced_targets is None:
        active_binding_id = target.active_binding_id
        active_mapping = next(
            (
                mapping
                for mapping, capability in mappings
                if active_binding_id in _mapping_selection_binding_ids(mapping, capability)
            ),
            mappings[0][0],
        )
        active_built = build_contact_intent_plan(
            scene,
            control_context,
            operation_id=f"{operation_id}:active:{active_mapping.mapping_id}",
            mode=mode,
            enabled_types=enabled_types,
            plant_space=plant_space,
            contact_point_local=contact_point_local,
            mapping_id=active_mapping.mapping_id,
            selector_character_id=selector_character_id,
        )
        if not active_built.ok or active_built.plan is None:
            return ContactBatchPlanBuildResult(None, active_built.diagnostics)
        shared_cycle_type = active_built.plan.target_type

    intents: list[ContactIntentPlan] = []
    diagnostics: list[Diagnostic] = []
    for mapping, _capability in mappings:
        built = build_contact_intent_plan(
            scene,
            control_context,
            operation_id=f"{operation_id}:{mapping.mapping_id}",
            mode=mode,
            enabled_types=enabled_types,
            plant_space=plant_space,
            contact_point_local=contact_point_local,
            mapping_id=mapping.mapping_id,
            selector_character_id=selector_character_id,
            forced_cycle_type=(
                forced_targets[str(mapping.mapping_id)]
                if forced_targets is not None
                else shared_cycle_type
            ),
        )
        if not built.ok or built.plan is None:
            diagnostics.extend(built.diagnostics)
            continue
        intents.append(built.plan)
    if diagnostics or len(intents) != len(mappings):
        return ContactBatchPlanBuildResult(None, tuple(diagnostics))

    owner_keys = {intent.owner_runtime_key for intent in intents}
    owner_tokens = {intent.owner_binding_token for intent in intents}
    if len(owner_keys) != 1 or len(owner_tokens) != 1:
        return ContactBatchPlanBuildResult(
            None,
            (
                _diagnostic(
                    operation_id,
                    "I20_MULTIPLE_ANIMATION_OWNERS_UNSUPPORTED",
                    "Initial multi-limb Contact batch requires every limb to share one Rigped animation owner/binding.",
                    character_id=target.character_id,
                ),
            ),
        )

    return ContactBatchPlanBuildResult(
        ContactBatchIntentPlan(
            operation_id,
            tuple(intents),
            tuple(intent.mapping_id for intent in intents),
        )
    )


def _contract_by_runtime_key(target, key: tuple[int, int]):
    return next(
        (contract for contract in target.controls if runtime_control_key(contract.target) == key),
        None,
    )


def _channels_for_state(contract, state: SnapControlState, *, location: bool, rotation: bool) -> list[PlannedChannel]:
    rows: list[PlannedChannel] = []
    if location:
        for index, value in enumerate(state.location):
            rows.append(_build_channel(contract, ChannelFamily.POSITION, "location", index, value))
    if rotation:
        property_name, representation, indices, _values, mode = _rotation_descriptor(contract.target.target)
        if property_name != state.rotation_property:
            raise ContactAuthoringError("Snap rotation representation changed before Contact write planning.")
        for index in indices:
            rows.append(
                _build_channel(
                    contract,
                    ChannelFamily.ROTATION,
                    property_name,
                    index,
                    state.rotation[index],
                    rotation_mode=mode,
                    rotation_representation=representation,
                )
            )
    return rows


def _current_state(contract) -> SnapControlState:
    resolved = contract.target
    property_name, _representation, _indices, values, _mode = _rotation_descriptor(resolved.target)
    return SnapControlState(
        runtime_control_key(resolved),
        tuple(float(component) for component in resolved.target.location),
        property_name,
        tuple(float(component) for component in values),
    )


def _quaternion_state_compatible_with_existing_keys(
    contract,
    state: SnapControlState,
    *,
    time: float | None,
) -> SnapControlState:
    """Keep quaternion Contact keys in the same hemisphere as the nearest authored key.

    Blender interpolates quaternion FCurve components independently. Equivalent
    q/-q endpoints therefore take a destructive path through near-zero when their
    signs differ. Preserve the exact pose while choosing the sign compatible with
    the nearest existing quaternion key before the immutable write plan is built.
    """

    if time is None or state.rotation_property != "rotation_quaternion" or len(state.rotation) != 4:
        return state
    resolved = contract.target
    bag = assigned_channelbag(resolved.owner_object)
    if bag is None:
        return state
    data_path = control_property_path(resolved, "rotation_quaternion")
    curves = tuple(_find_fcurve(bag, data_path, index) for index in range(4))
    if any(curve is None for curve in curves):
        return state

    candidate_times = sorted(
        {
            float(point.co.x)
            for point in curves[0].keyframe_points
            if all(_key_at_frame(curve, float(point.co.x)) is not None for curve in curves[1:])
        }
    )
    if not candidate_times:
        return state
    reference_time = min(
        candidate_times,
        key=lambda key_time: (abs(key_time - float(time)), 0 if key_time <= float(time) else 1),
    )
    reference_values = tuple(
        float(_key_at_frame(curve, reference_time).co.y)
        for curve in curves
    )
    current = Quaternion(tuple(float(value) for value in state.rotation))
    reference = Quaternion(reference_values)
    current_norm = math.sqrt(sum(float(value) * float(value) for value in current))
    reference_norm = math.sqrt(sum(float(value) * float(value) for value in reference))
    if current_norm <= 1e-12 or reference_norm <= 1e-12:
        return state
    current.normalize()
    reference.normalize()
    if float(current.dot(reference)) >= 0.0:
        return state
    return replace(state, rotation=tuple(-float(value) for value in state.rotation))


def _state_for_pose_matrix(
    contract,
    desired_matrix,
    *,
    parent_pose_matrix=None,
) -> SnapControlState:
    """Resolve raw PoseBone channels that reproduce one desired armature-space matrix."""

    resolved = contract.target
    pose_bone = resolved.target
    rest = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        basis = rest.inverted() @ desired_matrix
    else:
        parent_rest = pose_bone.parent.bone.matrix_local.copy()
        parent_pose = (
            parent_pose_matrix.copy()
            if parent_pose_matrix is not None
            else pose_bone.parent.matrix.copy()
        )
        basis = rest.inverted() @ parent_rest @ parent_pose.inverted() @ desired_matrix

    property_name, _representation, _indices, _values, mode = _rotation_descriptor(pose_bone)
    quaternion = basis.to_quaternion().normalized()
    if property_name == "rotation_quaternion":
        rotation = (quaternion.w, quaternion.x, quaternion.y, quaternion.z)
    elif property_name == "rotation_axis_angle":
        axis = quaternion.axis
        rotation = (quaternion.angle, axis.x, axis.y, axis.z)
    else:
        euler = quaternion.to_euler(mode)
        rotation = tuple(float(component) for component in euler)
    return SnapControlState(
        runtime_control_key(resolved),
        tuple(float(component) for component in basis.to_translation()),
        property_name,
        tuple(float(component) for component in rotation),
    )


def _hold_state_for_contact_point(
    hold: ContactHoldCapability,
    contact_matrix,
    local_point: tuple[float, float, float],
) -> SnapControlState:
    """Calibrate the pre-Pivot hold origin so enabling Pivot preserves the current terminal pose."""

    resolved = hold.control.target
    armature = resolved.owner_object
    hold_pose = resolved.target
    desired_world = armature.matrix_world @ contact_matrix
    pivot_world = desired_world @ Vector(local_point)
    desired_origin_world = desired_world.to_translation()
    rotation = desired_world.to_quaternion().normalized().to_matrix()
    raw_origin_world = pivot_world + (rotation.inverted() @ (desired_origin_world - pivot_world))
    try:
        raw_origin_armature = armature.matrix_world.inverted() @ raw_origin_world
    except ValueError as exc:
        raise ContactAuthoringError(
            "Rigped world transform is singular and cannot calibrate the continuous Contact pivot."
        ) from exc
    desired = hold_pose.bone.matrix_local.copy()
    desired.translation = raw_origin_armature
    return _state_for_pose_matrix(hold.control, desired)


def _hold_state_for_external_contact_point(
    hold: ContactHoldCapability,
    contact_matrix,
) -> SnapControlState:
    """Resolve raw hold channels that evaluate to the current point in external Object space."""

    external = hold.external_constraint.target
    if external is None:
        raise ContactAuthoringError("Object-space Planted Contact has no bound external Object.")
    resolved = hold.control.target
    armature = resolved.owner_object
    hold_pose = resolved.target
    desired_armature = hold_pose.bone.matrix_local.copy()
    desired_armature.translation = contact_matrix.to_translation()
    try:
        desired_world = armature.matrix_world @ desired_armature
        external_delta = external.matrix_world @ hold.external_constraint.inverse_matrix
        raw_world = external_delta.inverted() @ desired_world
        raw_armature = armature.matrix_world.inverted() @ raw_world
    except ValueError as exc:
        raise ContactAuthoringError(
            "External Contact Object or Rigped world transform is singular and cannot define a rigid attachment space."
        ) from exc
    return _state_for_pose_matrix(hold.control, raw_armature)


def _point_state_for_local_contact(
    hold: ContactHoldCapability,
    terminal_matrix,
    local_point: tuple[float, float, float],
) -> SnapControlState:
    point_pose = hold.point_control.target.target
    desired = point_pose.bone.matrix_local.copy()
    desired.translation = terminal_matrix @ Vector(local_point)
    return _state_for_pose_matrix(hold.point_control, desired)


def _build_transform_rows(
    target,
    capability,
    target_type: ContactKeyType,
    snap_result,
    *,
    hold: ContactHoldCapability | None = None,
    hold_state: SnapControlState | None = None,
    point_state: SnapControlState | None = None,
    ik_target_state_override: SnapControlState | None = None,
    write_time: float | None = None,
) -> tuple[PlannedChannel, ...]:
    rows: list[PlannedChannel] = []
    by_binding = {contract.binding_id: contract for contract in target.controls}

    if target_type is ContactKeyType.FREE:
        if snap_result is not None:
            fk_states = snap_result.fk_control_states
            terminal_state = snap_result.authored_terminal_state
            if len(fk_states) != len(capability.fk_controls) or terminal_state is None:
                raise ContactAuthoringError("IK→FK snap did not return the complete authored Free dependency state.")
        else:
            fk_states = tuple(_current_state(by_binding[binding_id]) for binding_id in capability.fk_binding_ids)
            terminal_state = _current_state(by_binding[capability.authored_terminal_binding_id])
        for binding_id, state in zip(capability.fk_binding_ids, fk_states, strict=True):
            rows.extend(_channels_for_state(by_binding[binding_id], state, location=False, rotation=True))
        rows.extend(
            _channels_for_state(
                by_binding[capability.authored_terminal_binding_id],
                terminal_state,
                location=False,
                rotation=True,
            )
        )
        return tuple(rows)

    for binding_id in capability.fk_binding_ids:
        contract = by_binding[binding_id]
        rows.extend(_channels_for_state(contract, _current_state(contract), location=False, rotation=True))

    if target_type is ContactKeyType.SLIDING:
        # Sliding is IK-authoritative during playback, but the authored terminal
        # FK curve is still the release/transition authority on the immediately
        # adjacent Free frame. Persist the public terminal rotation produced by
        # the solved Sliding preview so the Contact boundary does not fall back
        # to a stale pre-Sliding Hand/Foot key.
        terminal_contract = by_binding[capability.authored_terminal_binding_id]
        rows.extend(
            _channels_for_state(
                terminal_contract,
                _current_state(terminal_contract),
                location=False,
                rotation=True,
            )
        )

    ik_binding_id = capability.native_ik.mapping.ik_target_binding_id if hasattr(capability.native_ik, "mapping") else None
    if not ik_binding_id:
        ik_contract = _contract_by_runtime_key(target, runtime_control_key(capability.native_ik.ik_target))
    else:
        ik_contract = by_binding.get(ik_binding_id)
    pole_contract = (
        _contract_by_runtime_key(target, runtime_control_key(capability.native_ik.pole_target))
        if capability.native_ik.pole_target is not None
        else None
    )
    if ik_contract is None or pole_contract is None:
        raise ContactAuthoringError("IK-authoritative Contact cannot resolve authored IK target/pole contracts.")

    ik_state = ik_target_state_override
    if ik_state is None:
        ik_state = snap_result.ik_target_state if snap_result is not None else _current_state(ik_contract)
    pole_state = snap_result.pole_target_state if snap_result is not None else _current_state(pole_contract)
    if ik_state is None or pole_state is None:
        raise ContactAuthoringError("IK-authoritative Contact lacks a complete target/pole state.")
    ik_state = _quaternion_state_compatible_with_existing_keys(
        ik_contract,
        ik_state,
        time=write_time,
    )
    rows.extend(_channels_for_state(ik_contract, ik_state, location=True, rotation=True))
    rows.extend(_channels_for_state(pole_contract, pole_state, location=True, rotation=False))
    if target_type is ContactKeyType.PLANTED:
        if hold is None or hold_state is None or point_state is None:
            raise ContactAuthoringError(
                "Planted Contact requires resolved point-hold and continuous-point carrier states."
            )
        rows.extend(
            _channels_for_state(
                hold.control,
                hold_state,
                location=True,
                rotation=False,
            )
        )
        rows.extend(
            _channels_for_state(
                hold.point_control,
                point_state,
                location=True,
                rotation=False,
            )
        )
    return tuple(rows)


def snapshot_contact_transform_rows(
    scene,
    control_context,
    intent: ContactIntentPlan,
    *,
    selector_character_id: str | None = None,
) -> ContactTransformRowSnapshot:
    """Read the current semantic transform rows using the existing Contact closure builder."""

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        detail = "; ".join(issue.detail for issue in resolution.issues)
        raise ContactAuthoringError(detail or "Rigped Contact target is unavailable.")
    target = resolution.target
    if target.character_id != intent.snap_plan.character_id:
        raise ContactAuthoringError("Contact transform snapshot character changed after planning.")

    view = resolve_character(scene, target.character_id)
    capability = resolve_limb_representation_capability(view, intent.mapping_id).capability
    if capability is None or not representation_payload_matches(
        intent.representation_identity,
        capability,
    ):
        raise ContactAuthoringError("Contact transform snapshot representation changed after planning.")
    if intent.transition_payload is not None:
        raise ContactAuthoringError(
            "State-changing Contact transitions cannot be used as an AUTO baseline snapshot."
        )

    rows = _build_transform_rows(
        target,
        capability,
        intent.target_type,
        None,
        write_time=float(intent.snap_plan.frame) + float(intent.snap_plan.subframe),
    )
    owner = capability.native_ik.solver_owner.owner_object
    bag = assigned_channelbag(owner)
    keyed = 0
    for row in rows:
        fcurve = _find_fcurve(
            bag,
            row.row_key.data_path,
            row.row_key.array_index,
        )
        if fcurve is not None and len(fcurve.keyframe_points) > 0:
            keyed += 1
    return ContactTransformRowSnapshot(tuple(rows), keyed)


def _contact_operation_plan(
    intent: ContactIntentPlan,
    target,
    transform_rows: tuple[PlannedChannel, ...],
    *,
    closure_plan: OperationPlan | None = None,
) -> OperationPlan:
    base = intent.snap_plan
    if closure_plan is not None:
        identity_fields = (
            (closure_plan.character_id, base.character_id, "Character"),
            (closure_plan.setup_revision, base.setup_revision, "setup revision"),
            (closure_plan.setup_signature, base.setup_signature, "setup signature"),
            (closure_plan.character_source_stamp, base.character_source_stamp, "source stamp"),
            (closure_plan.frame, base.frame, "frame"),
            (closure_plan.subframe, base.subframe, "subframe"),
            (closure_plan.selected_binding_ids, base.selected_binding_ids, "selection"),
            (closure_plan.active_binding_id, base.active_binding_id, "active binding"),
        )
        mismatched = tuple(label for actual, expected, label in identity_fields if actual != expected)
        if mismatched:
            raise ContactAuthoringError(
                f"Contact closure plan identity disagrees with Contact preflight: {mismatched!r}."
            )

    rows_by_key: dict[Any, PlannedChannel] = {}
    if closure_plan is not None:
        for group in closure_plan.owner_groups:
            for row in group.channels:
                rows_by_key[row.row_key] = row
    # Contact dependency values are authoritative on overlapping limb rows.
    for row in transform_rows:
        rows_by_key[row.row_key] = row
    merged_rows = tuple(sorted(rows_by_key.values(), key=channel_sort_key))
    if not merged_rows:
        raise ContactAuthoringError("Contact persistent write requires a non-empty transform dependency bundle.")

    owner = merged_rows[0].owner_runtime_key
    binding_token = merged_rows[0].owner_binding_token
    if any(row.owner_runtime_key != owner for row in merged_rows):
        raise ContactAuthoringError("Initial I13 Contact closure requires one animation owner.")
    if any(row.owner_binding_token != binding_token for row in merged_rows):
        raise ContactAuthoringError("Contact closure rows disagree on the frozen owner binding token.")
    if binding_token != intent.owner_binding_token:
        raise ContactAuthoringError("Contact closure owner token disagrees with Contact scalar ownership.")
    group = PlannedOwnerGroup(owner, binding_token, merged_rows)

    semantic_binding_ids = tuple(
        dict.fromkeys(
            (
                *(closure_plan.dependency_footprint.semantic_binding_ids if closure_plan is not None else ()),
                *base.dependency_footprint.semantic_binding_ids,
                intent.hold_capability.binding_id,
                intent.hold_capability.point_binding_id,
            )
        )
    )
    contracts_by_id = {contract.binding_id: contract for contract in target.controls}
    if any(binding_id not in contracts_by_id for binding_id in semantic_binding_ids):
        raise ContactAuthoringError("Contact closure dependency binding disappeared before planning.")
    control_runtime_keys = tuple(
        runtime_control_key(contracts_by_id[binding_id].target)
        for binding_id in semantic_binding_ids
    )

    plan = OperationPlan(
        operation_id=intent.operation_id,
        operation_type=OperationType.CONTACT,
        character_id=base.character_id,
        setup_revision=base.setup_revision,
        setup_signature=base.setup_signature,
        character_source_stamp=base.character_source_stamp,
        frame=base.frame,
        subframe=base.subframe,
        selected_binding_ids=base.selected_binding_ids,
        active_binding_id=base.active_binding_id,
        owner_groups=(group,),
        read_footprint=ReadFootprint(
            character_source_stamp=base.read_footprint.character_source_stamp,
            setup_revision=base.read_footprint.setup_revision,
            setup_signature=base.read_footprint.setup_signature,
            selected_binding_ids=base.read_footprint.selected_binding_ids,
            active_binding_id=base.read_footprint.active_binding_id,
            selection_runtime_keys=base.read_footprint.selection_runtime_keys,
            active_runtime_key=base.read_footprint.active_runtime_key,
            owner_binding_tokens=(intent.owner_binding_token,),
            control_runtime_keys=control_runtime_keys,
        ),
        dependency_footprint=DependencyFootprint(
            semantic_binding_ids=semantic_binding_ids,
            owner_binding_tokens=(intent.owner_binding_token,),
            kinematic=base.dependency_footprint.kinematic,
        ),
        write_footprint=PlanWriteFootprint(tuple(row.row_key for row in merged_rows), ()),
        kinematic_dependency=base.kinematic_dependency,
    )
    issues = validate_plan_structure(plan)
    if issues:
        raise ContactAuthoringError("; ".join(issue.detail for issue in issues))
    return plan


def _contact_batch_operation_plan(
    batch: ContactBatchIntentPlan,
    target,
    prepared: tuple[ContactPreparedIntent, ...],
    *,
    closure_plan: OperationPlan | None = None,
) -> OperationPlan:
    if len(prepared) < 2:
        raise ContactAuthoringError("I20 combined Contact plan requires at least two prepared limb domains.")
    base = prepared[0].intent.snap_plan
    for item in prepared[1:]:
        plan = item.intent.snap_plan
        identity_fields = (
            (plan.character_id, base.character_id, "Character"),
            (plan.setup_revision, base.setup_revision, "setup revision"),
            (plan.setup_signature, base.setup_signature, "setup signature"),
            (plan.character_source_stamp, base.character_source_stamp, "source stamp"),
            (plan.frame, base.frame, "frame"),
            (plan.subframe, base.subframe, "subframe"),
            (plan.selected_binding_ids, base.selected_binding_ids, "selection"),
            (plan.active_binding_id, base.active_binding_id, "active binding"),
        )
        mismatched = tuple(label for actual, expected, label in identity_fields if actual != expected)
        if mismatched:
            raise ContactAuthoringError(
                f"I20 limb preflight identities disagree before batching: {mismatched!r}."
            )

    if closure_plan is not None:
        identity_fields = (
            (closure_plan.character_id, base.character_id, "Character"),
            (closure_plan.setup_revision, base.setup_revision, "setup revision"),
            (closure_plan.setup_signature, base.setup_signature, "setup signature"),
            (closure_plan.character_source_stamp, base.character_source_stamp, "source stamp"),
            (closure_plan.frame, base.frame, "frame"),
            (closure_plan.subframe, base.subframe, "subframe"),
            (closure_plan.selected_binding_ids, base.selected_binding_ids, "selection"),
            (closure_plan.active_binding_id, base.active_binding_id, "active binding"),
        )
        mismatched = tuple(label for actual, expected, label in identity_fields if actual != expected)
        if mismatched:
            raise ContactAuthoringError(
                f"I20 closure plan identity disagrees with batch preflight: {mismatched!r}."
            )

    owner_keys = {item.intent.owner_runtime_key for item in prepared}
    owner_tokens = {item.intent.owner_binding_token for item in prepared}
    if len(owner_keys) != 1 or len(owner_tokens) != 1:
        raise ContactAuthoringError("I20 initial batch supports one shared animation owner/binding only.")
    owner_runtime_key = next(iter(owner_keys))
    owner_binding_token = next(iter(owner_tokens))

    binding_writer: dict[str, str] = {}
    transform_rows_by_key: dict[Any, PlannedChannel] = {}
    scalar_rows: set[tuple[str, int]] = set()
    constraint_tokens: set[int] = set()
    cleanup_tokens: set[int] = set()
    for item in prepared:
        mapping_id = item.intent.mapping_id
        for row in item.transform_rows:
            previous_mapping = binding_writer.get(row.row_key.binding_id)
            if previous_mapping is not None and previous_mapping != mapping_id:
                raise ContactAuthoringError(
                    "I20 dependency conflict: multiple selected limbs would write the same semantic binding "
                    f"{row.row_key.binding_id!r}."
                )
            binding_writer[row.row_key.binding_id] = mapping_id
            previous_row = transform_rows_by_key.get(row.row_key)
            if previous_row is not None:
                raise ContactAuthoringError(
                    f"I20 dependency conflict: duplicate writable channel row {row.row_key!r}."
                )
            transform_rows_by_key[row.row_key] = row

        for scalar in item.intent.scalar_channels:
            scalar_key = (scalar.row_key.data_path, int(scalar.row_key.array_index))
            if scalar_key in scalar_rows:
                raise ContactAuthoringError(
                    f"I20 dependency conflict: duplicate Contact scalar row {scalar_key!r}."
                )
            scalar_rows.add(scalar_key)

        tokens = (
            _runtime_pointer(item.capability.native_ik.constraint),
            _runtime_pointer(item.capability.terminal_ik_constraint),
            item.hold.constraint_runtime_key,
            item.hold.pivot_constraint_runtime_key,
            item.hold.external_constraint_runtime_key,
        )
        for token in tokens:
            if token is None:
                raise ContactAuthoringError("I20 writable constraint lost runtime identity during dependency analysis.")
            token = int(token)
            if token in constraint_tokens:
                raise ContactAuthoringError(
                    "I20 dependency conflict: selected limbs share a writable native constraint."
                )
            constraint_tokens.add(token)

        candidate_cleanup = (
            *((item.intent.pole_cleanup_fcurve_token,) if item.intent.pole_cleanup_fcurve_token is not None else ()),
            *(cleanup.fcurve_token for cleanup in item.intent.hold_cleanup_curves),
        )
        for token in candidate_cleanup:
            token = int(token)
            if token in cleanup_tokens:
                raise ContactAuthoringError(
                    "I20 dependency conflict: selected limbs share a writable cleanup FCurve."
                )
            cleanup_tokens.add(token)

    rows_by_key: dict[Any, PlannedChannel] = {}
    if closure_plan is not None:
        for group in closure_plan.owner_groups:
            if group.owner_runtime_key != owner_runtime_key or group.owner_binding_token != owner_binding_token:
                raise ContactAuthoringError(
                    "I20 initial Contact closure must use the same animation owner/binding as every limb."
                )
            for row in group.channels:
                rows_by_key[row.row_key] = row
    # Contact dependency values own overlapping limb closure rows.
    rows_by_key.update(transform_rows_by_key)
    merged_rows = tuple(sorted(rows_by_key.values(), key=channel_sort_key))
    if not merged_rows:
        raise ContactAuthoringError("I20 persistent batch requires a non-empty transform dependency bundle.")
    group = PlannedOwnerGroup(owner_runtime_key, owner_binding_token, merged_rows)

    semantic_binding_ids = tuple(
        dict.fromkeys(
            (
                *(closure_plan.dependency_footprint.semantic_binding_ids if closure_plan is not None else ()),
                *(
                    binding_id
                    for item in prepared
                    for binding_id in (
                        *item.intent.snap_plan.dependency_footprint.semantic_binding_ids,
                        item.hold.binding_id,
                        item.hold.point_binding_id,
                    )
                ),
            )
        )
    )
    contracts_by_id = {contract.binding_id: contract for contract in target.controls}
    if any(binding_id not in contracts_by_id for binding_id in semantic_binding_ids):
        raise ContactAuthoringError("I20 dependency binding disappeared before combined planning.")
    control_runtime_keys = tuple(
        runtime_control_key(contracts_by_id[binding_id].target)
        for binding_id in semantic_binding_ids
    )

    plan = OperationPlan(
        operation_id=batch.operation_id,
        operation_type=OperationType.CONTACT,
        character_id=base.character_id,
        setup_revision=base.setup_revision,
        setup_signature=base.setup_signature,
        character_source_stamp=base.character_source_stamp,
        frame=base.frame,
        subframe=base.subframe,
        selected_binding_ids=base.selected_binding_ids,
        active_binding_id=base.active_binding_id,
        owner_groups=(group,),
        read_footprint=ReadFootprint(
            character_source_stamp=base.read_footprint.character_source_stamp,
            setup_revision=base.read_footprint.setup_revision,
            setup_signature=base.read_footprint.setup_signature,
            selected_binding_ids=base.read_footprint.selected_binding_ids,
            active_binding_id=base.read_footprint.active_binding_id,
            selection_runtime_keys=base.read_footprint.selection_runtime_keys,
            active_runtime_key=base.read_footprint.active_runtime_key,
            owner_binding_tokens=(owner_binding_token,),
            control_runtime_keys=control_runtime_keys,
        ),
        dependency_footprint=DependencyFootprint(
            semantic_binding_ids=semantic_binding_ids,
            owner_binding_tokens=(owner_binding_token,),
            kinematic=None,
        ),
        write_footprint=PlanWriteFootprint(tuple(row.row_key for row in merged_rows), ()),
        kinematic_dependency=None,
    )
    issues = validate_plan_structure(plan)
    if issues:
        raise ContactAuthoringError("; ".join(issue.detail for issue in issues))
    return plan


def _scalar_tokens_match(owner, channels: tuple[ContactScalarChannel, ...]) -> bool:
    bag = assigned_channelbag(owner)
    for channel in channels:
        curve = _find_fcurve(bag, channel.row_key.data_path, channel.row_key.array_index)
        token = _runtime_pointer(curve) if curve is not None else None
        if channel.existing_fcurve_token != token:
            return False
    return True


def _pole_cleanup_snapshot_matches(owner, intent: ContactIntentPlan) -> bool:
    token = intent.pole_cleanup_fcurve_token
    snapshot = intent.pole_cleanup_snapshot
    if token is None or snapshot is None:
        return token is None and snapshot is None
    bag = assigned_channelbag(owner)
    curve = _find_fcurve(bag, snapshot.data_path, snapshot.array_index)
    if curve is None or _runtime_pointer(curve) != token:
        return False
    return _capture_fcurve_snapshot(curve) == snapshot


def _hold_cleanup_snapshots_match(owner, intent: ContactIntentPlan) -> bool:
    bag = assigned_channelbag(owner)
    for cleanup in intent.hold_cleanup_curves:
        snapshot = cleanup.snapshot
        curve = _find_fcurve(bag, snapshot.data_path, snapshot.array_index)
        if curve is None or _runtime_pointer(curve) != cleanup.fcurve_token:
            return False
        if _capture_fcurve_snapshot(curve) != snapshot:
            return False
    return True


def _set_constant_at(fcurve, time: float) -> None:
    point = _key_at_frame(fcurve, time)
    if point is None:
        raise ContactAuthoringError("Contact scalar key disappeared immediately after write.")
    point.interpolation = "CONSTANT"
    fcurve.update()


def _pose_residual(actual, expected) -> tuple[float, float]:
    position = float((actual.to_translation() - expected.to_translation()).length)
    angle = float(actual.to_quaternion().normalized().rotation_difference(expected.to_quaternion().normalized()).angle)
    if angle > math.pi:
        angle = abs((2.0 * math.pi) - angle)
    return position, abs(angle)


def _sliding_public_pose_overlay_active(capability) -> bool:
    """Return True when direct FK Rotate left a visible pose ahead of Sliding IK authority."""

    scale = max(
        1e-6,
        float(getattr(capability.native_ik.solver_owner.owner_object.dimensions, "length", 0.0) or 0.0),
    )
    position_tolerance = max(1e-7, scale * 1e-5)
    rotation_tolerance = 1e-5
    pairs = tuple(
        zip(capability.fk_controls, capability.result_controls, strict=True)
    ) + ((capability.authored_terminal, capability.result_terminal),)
    for public_control, result_control in pairs:
        position, rotation = _pose_residual(
            public_control.target.matrix,
            result_control.target.matrix,
        )
        if position > position_tolerance or rotation > rotation_tolerance:
            return True
    return False


def _probe_sliding_public_pose_to_ik(
    scene,
    control_context,
    intent: ContactIntentPlan,
    capability,
    *,
    selector_character_id: str | None,
    hook: StageHook,
):
    """Transiently promote a direct-Rotate public pose into Sliding IK state.

    Sliding replay authority is the hidden IK target/pole. Direct Rotate is
    intentionally native on the public Rigped controls, so a later C stamp must
    convert that visible FK overlay into equivalent IK authority instead of
    re-keying the previous target/pole pose.
    """

    if intent.target_type is not ContactKeyType.SLIDING:
        return None
    native_ik = capability.native_ik.constraint
    terminal_ik = capability.terminal_ik_constraint
    if not (
        _same_float(float(native_ik.influence), 1.0)
        and _same_float(float(terminal_ik.influence), 1.0)
    ):
        return None
    if not _sliding_public_pose_overlay_active(capability):
        return None

    native_before = float(native_ik.influence)
    terminal_before = float(terminal_ik.influence)
    feedback_constraints = (
        *capability.fk_copy_constraints,
        capability.terminal_fk_constraint,
    )
    feedback_before = tuple(bool(constraint.mute) for constraint in feedback_constraints)
    try:
        native_ik.influence = 0.0
        terminal_ik.influence = 0.0
        # B1 active-input bridge: Sliding normally disconnects public FK from
        # the hidden result chain.  While native IK is disabled, temporarily
        # expose the user's current public overlay so the FK->IK probe samples
        # the actual rotated pose rather than the stale hidden result.
        for constraint in feedback_constraints:
            constraint.mute = False
        bpy.context.view_layer.update()
        desired_result = tuple(
            control.target.matrix.copy() for control in capability.result_controls
        )
        desired_terminal = capability.result_terminal.target.matrix.copy()
        snap_result = execute_representation_snap(
            scene,
            control_context,
            intent.snap_plan,
            intent.representation_identity,
            selector_character_id=selector_character_id,
            hook=hook,
        )
        return snap_result, desired_result, desired_terminal
    finally:
        for constraint, muted in zip(
            feedback_constraints,
            feedback_before,
            strict=True,
        ):
            constraint.mute = bool(muted)
        bpy.context.view_layer.update()
        native_ik.influence = native_before
        terminal_ik.influence = terminal_before
        bpy.context.view_layer.update()


def _prepare_contact_intent_for_batch(
    scene,
    control_context,
    intent: ContactIntentPlan,
    *,
    selector_character_id: str | None = None,
    hook: StageHook | None = None,
) -> tuple[ContactPreparedIntent | None, tuple[Diagnostic, ...]]:
    """Resolve one limb through transient snap without leaving persistent/raw partial state."""

    hook = hook or NoopStageHook()
    freshness = validate_plan_fresh(
        scene,
        control_context,
        intent.snap_plan,
        selector_character_id=selector_character_id,
    )
    if not freshness.ok:
        return None, freshness.diagnostics
    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return None, tuple(
            _diagnostic(intent.operation_id, issue.code, issue.detail)
            for issue in resolution.issues
        )
    target = resolution.target
    view = resolve_character(scene, target.character_id)
    rep_resolution = resolve_limb_representation_capability(view, intent.mapping_id)
    capability = rep_resolution.capability
    if capability is None or not representation_payload_matches(intent.representation_identity, capability):
        return None, (
            _diagnostic(
                intent.operation_id,
                "I20_STALE_REPRESENTATION_IDENTITY",
                "Multi-limb Contact representation identity changed after planning.",
                character_id=target.character_id,
            ),
        )
    mapping = next(
        (item for item in view.definition.kinematics if item.mapping_id == intent.mapping_id),
        None,
    )
    if mapping is None:
        return None, (
            _diagnostic(
                intent.operation_id,
                "I20_CONTACT_MAPPING_MISSING",
                "Multi-limb Contact mapping disappeared after planning.",
                character_id=target.character_id,
            ),
        )
    try:
        hold = _resolve_contact_hold(view, target, mapping, capability)
    except ContactAuthoringError as exc:
        return None, (
            _diagnostic(
                intent.operation_id,
                "I20_CONTACT_HOLD_STALE",
                str(exc),
                character_id=target.character_id,
            ),
        )
    if (
        hold.binding_id != intent.hold_capability.binding_id
        or hold.point_binding_id != intent.hold_capability.point_binding_id
        or hold.control_runtime_key != intent.hold_capability.control_runtime_key
        or hold.point_control_runtime_key != intent.hold_capability.point_control_runtime_key
        or hold.constraint_runtime_key != intent.hold_capability.constraint_runtime_key
        or hold.pivot_constraint_runtime_key != intent.hold_capability.pivot_constraint_runtime_key
        or hold.external_constraint_runtime_key != intent.hold_capability.external_constraint_runtime_key
    ):
        return None, (
            _diagnostic(
                intent.operation_id,
                "I20_CONTACT_HOLD_IDENTITY_CHANGED",
                "Multi-limb Contact hold/point/constraint identity changed after planning.",
                character_id=target.character_id,
            ),
        )
    owner = capability.native_ik.solver_owner.owner_object
    if intent.external_target_runtime_key is not None:
        if _runtime_pointer(hold.external_constraint.target) != intent.external_target_runtime_key:
            return None, (
                _diagnostic(
                    intent.operation_id,
                    "I20_STALE_EXTERNAL_OBJECT_IDENTITY",
                    "External Contact Object identity changed after batch planning.",
                    character_id=target.character_id,
                ),
            )
        try:
            _validate_external_contact_target(scene, owner, hold.external_constraint.target)
        except ContactAuthoringError as exc:
            return None, (
                _diagnostic(
                    intent.operation_id,
                    "I20_EXTERNAL_OBJECT_STALE",
                    str(exc),
                    character_id=target.character_id,
                ),
            )
    if (
        _runtime_pointer(owner) != intent.owner_runtime_key
        or channel_binding_token(owner) != intent.owner_binding_token
    ):
        return None, (
            _diagnostic(
                intent.operation_id,
                "I20_STALE_CONTACT_OWNER_BINDING",
                "Multi-limb Contact animation owner binding changed after planning.",
                character_id=target.character_id,
            ),
        )
    if not _scalar_tokens_match(owner, intent.scalar_channels):
        return None, (
            _diagnostic(
                intent.operation_id,
                "I20_STALE_CONTACT_CURVE_BINDING",
                "Multi-limb Contact scalar FCurve identity changed after planning.",
                character_id=target.character_id,
            ),
        )
    if not _pole_cleanup_snapshot_matches(owner, intent) or not _hold_cleanup_snapshots_match(owner, intent):
        return None, (
            _diagnostic(
                intent.operation_id,
                "I20_STALE_CONTACT_CLEANUP",
                "Multi-limb Contact same-frame cleanup target changed after planning.",
                character_id=target.character_id,
            ),
        )

    expected_result = tuple(control.target.matrix.copy() for control in capability.result_controls)
    expected_terminal = capability.result_terminal.target.matrix.copy()
    snap_result = None
    if intent.transition_payload is not None:
        snap_result = execute_representation_snap(
            scene,
            control_context,
            intent.snap_plan,
            intent.transition_payload,
            selector_character_id=selector_character_id,
            hook=hook,
        )
        if not snap_result.success:
            return None, tuple(snap_result.diagnostics)
    else:
        overlay_probe = _probe_sliding_public_pose_to_ik(
            scene,
            control_context,
            intent,
            capability,
            selector_character_id=selector_character_id,
            hook=hook,
        )
        if overlay_probe is not None:
            snap_result, expected_result, expected_terminal = overlay_probe
            if not snap_result.success:
                return None, tuple(snap_result.diagnostics)

    freshness = validate_plan_fresh(
        scene,
        control_context,
        intent.snap_plan,
        selector_character_id=selector_character_id,
    )
    if not freshness.ok:
        return None, freshness.diagnostics
    if (
        channel_binding_token(owner) != intent.owner_binding_token
        or not _scalar_tokens_match(owner, intent.scalar_channels)
        or not _pole_cleanup_snapshot_matches(owner, intent)
        or not _hold_cleanup_snapshots_match(owner, intent)
    ):
        return None, (
            _diagnostic(
                intent.operation_id,
                "I20_BINDING_CHANGED_AFTER_TRANSIENT_PROBE",
                "Multi-limb Contact binding changed during a transient representation probe.",
                character_id=target.character_id,
            ),
        )

    if intent.target_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED} and snap_result is not None:
        pole_channels = tuple(
            channel
            for channel in intent.scalar_channels
            if channel.row_key.label == "IK_POLE_ANGLE"
        )
        if len(pole_channels) != 1 or snap_result.pole_angle is None:
            raise ContactAuthoringError("I20 IK-authoritative transition lacks a frozen pole-angle write row.")
        intent = replace(
            intent,
            scalar_channels=tuple(
                replace(channel, target_value=float(snap_result.pole_angle))
                if channel.row_key.label == "IK_POLE_ANGLE"
                else channel
                for channel in intent.scalar_channels
            ),
        )

    hold_state = None
    point_state = None
    ik_target_state_override = None
    if intent.target_type is ContactKeyType.PLANTED:
        if not intent.apply_contact_point and _same_float(float(hold.constraint.influence), 1.0):
            hold_state = _current_state(hold.control)
            point_state = _current_state(hold.point_control)
        else:
            local_point = intent.contact_point_local
            if local_point is None:
                raise ContactAuthoringError("I20 Planted/Replant operation lost its frozen local Contact point.")
            if intent.target_plant_space is ContactPlantSpace.OBJECT:
                hold_state = _hold_state_for_external_contact_point(hold, expected_terminal)
            else:
                hold_state = _hold_state_for_contact_point(hold, expected_terminal, local_point)
            point_state = _point_state_for_local_contact(hold, expected_terminal, local_point)
    elif intent.target_type is ContactKeyType.SLIDING and _same_float(float(hold.constraint.influence), 1.0):
        ik_contract = _contract_by_runtime_key(
            target,
            runtime_control_key(capability.native_ik.ik_target),
        )
        if ik_contract is None:
            raise ContactAuthoringError("I20 Planted→Sliding transition lost its authored IK target contract.")
        ik_target_state_override = _state_for_pose_matrix(
            ik_contract,
            capability.native_ik.ik_target.target.matrix.copy(),
        )

    transform_rows = _build_transform_rows(
        target,
        capability,
        intent.target_type,
        snap_result,
        hold=hold,
        hold_state=hold_state,
        point_state=point_state,
        ik_target_state_override=ik_target_state_override,
        write_time=float(intent.snap_plan.frame) + float(intent.snap_plan.subframe),
    )
    return (
        ContactPreparedIntent(
            intent,
            capability,
            hold,
            transform_rows,
            expected_result,
            expected_terminal,
            (snap_result.hinge_branch if snap_result is not None else None),
            hold_state,
            point_state,
        ),
        (),
    )


def execute_contact_intent_plan(
    scene,
    control_context,
    intent: ContactIntentPlan,
    *,
    closure_plan: OperationPlan | None = None,
    trigger: WriterTrigger = WriterTrigger.CONTACT_AUTHORING,
    auto_baseline_rows: tuple[PlannedChannel, ...] = (),
    auto_baseline_time: float | None = None,
    selector_character_id: str | None = None,
    hook: StageHook | None = None,
) -> ContactAuthoringResult:
    hook = hook or NoopStageHook()
    trace_event(
        "WRITER",
        "CONTACT_WRITE_BEGIN",
        operation_id=intent.operation_id,
        context=bpy.context,
        trigger=trigger.value,
        frame=float(intent.snap_plan.frame) + float(intent.snap_plan.subframe),
        mapping_id=intent.mapping_id,
        target_type=intent.target_type.value,
        baseline_required=bool(intent.baseline_required),
        auto_baseline_rows=len(auto_baseline_rows),
        auto_baseline_time=auto_baseline_time,
    )
    hook.enter(OperationStage.PREFLIGHT, operation=intent.operation_id)
    freshness = validate_plan_fresh(
        scene,
        control_context,
        intent.snap_plan,
        selector_character_id=selector_character_id,
    )
    if not freshness.ok:
        return ContactAuthoringResult(False, diagnostics=freshness.diagnostics)
    if closure_plan is not None:
        closure_freshness = validate_plan_fresh(
            scene,
            control_context,
            closure_plan,
            selector_character_id=selector_character_id,
        )
        if not closure_freshness.ok:
            return ContactAuthoringResult(False, diagnostics=closure_freshness.diagnostics)
    resolution = resolve_rigped_target(scene, control_context, selector_character_id=selector_character_id)
    if resolution.target is None:
        return ContactAuthoringResult(
            False,
            diagnostics=tuple(_diagnostic(intent.operation_id, issue.code, issue.detail) for issue in resolution.issues),
        )
    target = resolution.target
    view = resolve_character(scene, target.character_id)
    rep_resolution = resolve_limb_representation_capability(view, intent.mapping_id)
    capability = rep_resolution.capability
    if capability is None or not representation_payload_matches(intent.representation_identity, capability):
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    intent.operation_id,
                    "I13_STALE_REPRESENTATION_IDENTITY",
                    "Contact limb representation identity changed after planning.",
                    character_id=target.character_id,
                ),
            ),
        )
    mapping = next(
        (item for item in view.definition.kinematics if item.mapping_id == intent.mapping_id),
        None,
    )
    if mapping is None:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    intent.operation_id,
                    "I15_CONTACT_MAPPING_MISSING",
                    "Contact kinematic mapping disappeared after planning.",
                    character_id=target.character_id,
                ),
            ),
        )
    try:
        hold = _resolve_contact_hold(view, target, mapping, capability)
    except ContactAuthoringError as exc:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    intent.operation_id,
                    "I15_CONTACT_HOLD_STALE",
                    str(exc),
                    character_id=target.character_id,
                ),
            ),
        )
    if (
        hold.binding_id != intent.hold_capability.binding_id
        or hold.point_binding_id != intent.hold_capability.point_binding_id
        or hold.control_runtime_key != intent.hold_capability.control_runtime_key
        or hold.point_control_runtime_key != intent.hold_capability.point_control_runtime_key
        or hold.constraint_runtime_key != intent.hold_capability.constraint_runtime_key
        or hold.pivot_constraint_runtime_key != intent.hold_capability.pivot_constraint_runtime_key
        or hold.external_constraint_runtime_key != intent.hold_capability.external_constraint_runtime_key
    ):
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    intent.operation_id,
                    "I15_CONTACT_HOLD_IDENTITY_CHANGED",
                    "Contact point-hold carrier/constraint identity changed after planning.",
                    character_id=target.character_id,
                ),
            ),
        )
    owner = capability.native_ik.solver_owner.owner_object
    if intent.external_target_runtime_key is not None:
        if _runtime_pointer(hold.external_constraint.target) != intent.external_target_runtime_key:
            return ContactAuthoringResult(
                False,
                diagnostics=(
                    _diagnostic(
                        intent.operation_id,
                        "I18_STALE_EXTERNAL_OBJECT_IDENTITY",
                        "External Contact Object identity changed after planning.",
                        character_id=target.character_id,
                    ),
                ),
            )
        try:
            _validate_external_contact_target(scene, owner, hold.external_constraint.target)
        except ContactAuthoringError as exc:
            return ContactAuthoringResult(
                False,
                diagnostics=(
                    _diagnostic(
                        intent.operation_id,
                        "I18_EXTERNAL_OBJECT_STALE",
                        str(exc),
                        character_id=target.character_id,
                    ),
                ),
            )
    if _runtime_pointer(owner) != intent.owner_runtime_key or channel_binding_token(owner) != intent.owner_binding_token:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    intent.operation_id,
                    "I13_STALE_CONTACT_OWNER_BINDING",
                    "Contact animation owner Action/Slot/ChannelBag binding changed after planning.",
                    character_id=target.character_id,
                ),
            ),
        )
    if not _scalar_tokens_match(owner, intent.scalar_channels):
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(intent.operation_id, "I13_STALE_CONTACT_CURVE_BINDING", "Contact scalar FCurve identity changed after planning."),
            ),
        )
    if not _pole_cleanup_snapshot_matches(owner, intent):
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    intent.operation_id,
                    "I13_STALE_POLE_CLEANUP_CURVE",
                    "Same-frame Contact pole-angle FCurve changed after planning.",
                ),
            ),
        )
    if not _hold_cleanup_snapshots_match(owner, intent):
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    intent.operation_id,
                    "I15_STALE_HOLD_CLEANUP_CURVE",
                    "Same-frame Contact point-hold FCurve changed after planning.",
                ),
            ),
        )

    expected_result = tuple(control.target.matrix.copy() for control in capability.result_controls)
    expected_terminal = capability.result_terminal.target.matrix.copy()
    snap_result = None
    if intent.transition_payload is not None:
        snap_result = execute_representation_snap(
            scene,
            control_context,
            intent.snap_plan,
            intent.transition_payload,
            selector_character_id=selector_character_id,
            hook=hook,
        )
        if not snap_result.success:
            return ContactAuthoringResult(False, diagnostics=tuple(snap_result.diagnostics))
    else:
        overlay_probe = _probe_sliding_public_pose_to_ik(
            scene,
            control_context,
            intent,
            capability,
            selector_character_id=selector_character_id,
            hook=hook,
        )
        if overlay_probe is not None:
            snap_result, expected_result, expected_terminal = overlay_probe
            if not snap_result.success:
                return ContactAuthoringResult(False, diagnostics=tuple(snap_result.diagnostics))

    freshness = validate_plan_fresh(
        scene,
        control_context,
        intent.snap_plan,
        selector_character_id=selector_character_id,
    )
    if not freshness.ok:
        return ContactAuthoringResult(False, diagnostics=freshness.diagnostics)
    if (
        channel_binding_token(owner) != intent.owner_binding_token
        or not _scalar_tokens_match(owner, intent.scalar_channels)
        or not _pole_cleanup_snapshot_matches(owner, intent)
        or not _hold_cleanup_snapshots_match(owner, intent)
    ):
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(intent.operation_id, "I13_CONTACT_BINDING_CHANGED_AFTER_SNAP", "Contact animation binding changed during transient snap."),
            ),
        )

    if intent.target_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED} and snap_result is not None:
        pole_channels = tuple(channel for channel in intent.scalar_channels if channel.row_key.label == "IK_POLE_ANGLE")
        if len(pole_channels) != 1 or snap_result.pole_angle is None:
            raise ContactAuthoringError("IK-authoritative transition lacks a frozen pole-angle write row.")
        intent = replace(
            intent,
            scalar_channels=tuple(
                replace(channel, target_value=float(snap_result.pole_angle))
                if channel.row_key.label == "IK_POLE_ANGLE"
                else channel
                for channel in intent.scalar_channels
            ),
        )

    hold_state = None
    point_state = None
    ik_target_state_override = None
    if intent.target_type is ContactKeyType.PLANTED:
        if not intent.apply_contact_point and _same_float(float(hold.constraint.influence), 1.0):
            hold_state = _current_state(hold.control)
            point_state = _current_state(hold.point_control)
        else:
            local_point = intent.contact_point_local
            if local_point is None:
                raise ContactAuthoringError("Planted/Replant operation lost its frozen local Contact point.")
            if intent.target_plant_space is ContactPlantSpace.OBJECT:
                hold_state = _hold_state_for_external_contact_point(hold, expected_terminal)
            else:
                hold_state = _hold_state_for_contact_point(hold, expected_terminal, local_point)
            point_state = _point_state_for_local_contact(hold, expected_terminal, local_point)
    elif (
        intent.target_type is ContactKeyType.SLIDING
        and _same_float(float(hold.constraint.influence), 1.0)
    ):
        ik_contract = _contract_by_runtime_key(
            target,
            runtime_control_key(capability.native_ik.ik_target),
        )
        if ik_contract is None:
            raise ContactAuthoringError(
                "Planted→Sliding transition cannot resolve the authored IK target contract."
            )
        ik_target_state_override = _state_for_pose_matrix(
            ik_contract,
            capability.native_ik.ik_target.target.matrix.copy(),
        )

    transform_rows = _build_transform_rows(
        target,
        capability,
        intent.target_type,
        snap_result,
        hold=hold,
        hold_state=hold_state,
        point_state=point_state,
        ik_target_state_override=ik_target_state_override,
        write_time=float(intent.snap_plan.frame) + float(intent.snap_plan.subframe),
    )
    if auto_baseline_rows:
        if trigger is not WriterTrigger.AUTO_TRANSFORM:
            raise ContactAuthoringError("AUTO baseline rows require the AUTO_TRANSFORM trigger.")
        if intent.target_type is not ContactKeyType.FREE:
            raise ContactAuthoringError("AUTO first-key transform baseline is supported only for Free authority.")
        if auto_baseline_time is None:
            raise ContactAuthoringError("AUTO first-key transform baseline is missing its baseline time.")
        current_by_key = {row.row_key: row for row in transform_rows}
        if {row.row_key for row in auto_baseline_rows} != set(current_by_key):
            raise ContactAuthoringError("AUTO baseline transform footprint differs from the current Free closure.")
        for baseline in auto_baseline_rows:
            current = current_by_key[baseline.row_key]
            if (
                baseline.owner_runtime_key != current.owner_runtime_key
                or baseline.control_runtime_key != current.control_runtime_key
                or baseline.existing_fcurve_token != current.existing_fcurve_token
                or baseline.allocation_intent is not current.allocation_intent
            ):
                raise ContactAuthoringError("AUTO baseline transform storage identity changed during the gesture.")

    contact_plan = _contact_operation_plan(
        intent,
        target,
        transform_rows,
        closure_plan=closure_plan,
    )
    transform_rows = contact_plan.owner_groups[0].channels
    contact_freshness = validate_plan_fresh(
        scene,
        control_context,
        contact_plan,
        selector_character_id=selector_character_id,
    )
    if not contact_freshness.ok:
        return ContactAuthoringResult(False, diagnostics=contact_freshness.diagnostics)

    group = contact_plan.owner_groups[0]
    _check_scope_not_quarantined(contact_plan, group)
    journal = MutationJournal(
        contact_plan.operation_id,
        contact_plan.character_id,
        contact_plan.setup_revision,
        contact_plan.setup_signature,
        BlenderRollbackExecutor(),
    )
    mutation_ordinal = 0

    def stage_write(detail: str) -> None:
        nonlocal mutation_ordinal
        mutation_ordinal += 1
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=mutation_ordinal,
            operation=contact_plan.operation_id,
            detail=detail,
        )

    rows_written = 0
    created_fcurves = 0
    semantic_properties_prepared: set[tuple[int, str]] = set()
    time = float(contact_plan.frame) + float(contact_plan.subframe)
    try:
        bag = _ensure_owner_storage(owner, group, contact_plan, journal, stage_write)
        if auto_baseline_rows:
            assert auto_baseline_time is not None
            for baseline in auto_baseline_rows:
                existing_curve = _find_fcurve(
                    bag,
                    baseline.row_key.data_path,
                    baseline.row_key.array_index,
                )
                if baseline.existing_fcurve_token is None:
                    if existing_curve is not None:
                        raise ContactAuthoringError(
                            "AUTO baseline target FCurve appeared after gesture begin."
                        )
                else:
                    if (
                        existing_curve is None
                        or _runtime_pointer(existing_curve) != baseline.existing_fcurve_token
                        or len(existing_curve.keyframe_points) != 0
                    ):
                        raise ContactAuthoringError(
                            "AUTO baseline target FCurve changed after gesture begin."
                        )
                _prepare_semantic_state_property(
                    owner,
                    baseline,
                    contact_plan,
                    journal,
                    stage_write,
                    semantic_properties_prepared,
                )
                fcurve, created = _ensure_fcurve(
                    owner,
                    bag,
                    baseline,
                    contact_plan,
                    journal,
                    stage_write,
                )
                created_fcurves += int(created)
                _write_channel_key(
                    owner,
                    bag,
                    fcurve,
                    baseline,
                    float(auto_baseline_time),
                    contact_plan,
                    journal,
                    stage_write,
                )
                rows_written += 1

        hold_location_path = control_property_path(hold.control.target, "location")
        point_location_path = control_property_path(hold.point_control.target, "location")
        anchor_location_paths = {hold_location_path, point_location_path}
        for channel in transform_rows:
            _prepare_semantic_state_property(
                owner,
                channel,
                contact_plan,
                journal,
                stage_write,
                semantic_properties_prepared,
            )
            fcurve, created = _ensure_fcurve(owner, bag, channel, contact_plan, journal, stage_write)
            created_fcurves += int(created)
            _write_channel_key(owner, bag, fcurve, channel, time, contact_plan, journal, stage_write)
            if (
                intent.target_type is ContactKeyType.PLANTED
                and channel.row_key.data_path in anchor_location_paths
            ):
                fcurve.extrapolation = "CONSTANT"
                _set_constant_at(fcurve, time)
            rows_written += 1

        if intent.pole_cleanup_snapshot is not None:
            pole_curve = _find_fcurve(
                bag,
                intent.pole_cleanup_snapshot.data_path,
                intent.pole_cleanup_snapshot.array_index,
            )
            pole_curve_ptr = _runtime_pointer(pole_curve) if pole_curve is not None else None
            bag_ptr = _runtime_pointer(bag)
            if (
                pole_curve is None
                or pole_curve_ptr != intent.pole_cleanup_fcurve_token
                or bag_ptr is None
                or _capture_fcurve_snapshot(pole_curve) != intent.pole_cleanup_snapshot
            ):
                raise ContactAuthoringError("Same-frame pole-angle cleanup target changed before mutation.")
            pole_key = _key_at_frame(pole_curve, time)
            if pole_key is None:
                raise ContactAuthoringError("Same-frame Sliding→Free pole-angle key disappeared before cleanup.")
            journal.record(
                FCurveMutationReceipt(
                    journal.next_ordinal(),
                    _scope(contact_plan, owner_ptr=group.owner_runtime_key, bag_ptr=int(bag_ptr)),
                    int(bag_ptr),
                    int(pole_curve_ptr),
                    intent.pole_cleanup_snapshot,
                )
            )
            stage_write(f"remove Contact pole angle @ {time}")
            pole_curve.keyframe_points.remove(pole_key, fast=True)
            pole_curve.update()

        for cleanup in intent.hold_cleanup_curves:
            snapshot = cleanup.snapshot
            hold_curve = _find_fcurve(bag, snapshot.data_path, snapshot.array_index)
            hold_curve_ptr = _runtime_pointer(hold_curve) if hold_curve is not None else None
            bag_ptr = _runtime_pointer(bag)
            if (
                hold_curve is None
                or hold_curve_ptr != cleanup.fcurve_token
                or bag_ptr is None
                or _capture_fcurve_snapshot(hold_curve) != snapshot
            ):
                raise ContactAuthoringError("Same-frame point-hold cleanup target changed before mutation.")
            hold_key = _key_at_frame(hold_curve, time)
            if hold_key is None:
                raise ContactAuthoringError("Same-frame Planted replacement anchor key disappeared before cleanup.")
            journal.record(
                FCurveMutationReceipt(
                    journal.next_ordinal(),
                    _scope(contact_plan, owner_ptr=group.owner_runtime_key, bag_ptr=int(bag_ptr)),
                    int(bag_ptr),
                    int(hold_curve_ptr),
                    snapshot,
                )
            )
            stage_write(
                f"remove Contact point-hold {snapshot.data_path}[{snapshot.array_index}] @ {time}"
            )
            hold_curve.keyframe_points.remove(hold_key, fast=True)
            hold_curve.update()

        for scalar in intent.scalar_channels:
            fcurve, created = _ensure_fcurve(owner, bag, scalar, contact_plan, journal, stage_write)
            created_fcurves += int(created)
            if intent.baseline_required and scalar.row_key.label in {
                "CONTACT_STATE",
                "IK_INFLUENCE",
                "TERMINAL_IK_INFLUENCE",
                "POINT_HOLD_INFLUENCE",
                "CONTACT_POINT_PIVOT_INFLUENCE",
                "EXTERNAL_OBJECT_SPACE_INFLUENCE",
            }:
                baseline_time = (
                    float(auto_baseline_time)
                    if trigger is WriterTrigger.AUTO_TRANSFORM
                    and auto_baseline_rows
                    and auto_baseline_time is not None
                    else float(scene.frame_start)
                )
                baseline_value = 0.0
                if (
                    trigger is WriterTrigger.AUTO_TRANSFORM
                    and auto_baseline_rows
                    and intent.target_type is ContactKeyType.FREE
                    and scalar.row_key.label == "CONTACT_STATE"
                ):
                    baseline_value = float(state_value_for_type(ContactKeyType.FREE))
                baseline = replace(scalar, target_value=baseline_value)
                _write_channel_key(
                    owner,
                    bag,
                    fcurve,
                    baseline,
                    baseline_time,
                    contact_plan,
                    journal,
                    stage_write,
                )
                _set_constant_at(fcurve, baseline_time)
                rows_written += 1
            _write_channel_key(owner, bag, fcurve, scalar, time, contact_plan, journal, stage_write)
            _set_constant_at(fcurve, time)
            rows_written += 1

        feedback_constraints = (
            *capability.fk_copy_constraints,
            capability.terminal_fk_constraint,
        )
        hook.enter(OperationStage.REEVALUATE, operation=contact_plan.operation_id)
        _reevaluate_contact_frame_preserving_public_pose(
            scene,
            contact_plan.frame,
            contact_plan.subframe,
            feedback_constraints=feedback_constraints,
        )
        hook.enter(OperationStage.VERIFY, operation=contact_plan.operation_id)

        state_value = float(capability.native_ik.solver_owner.target[AWB_CONTACT_STATE_PROPERTY])
        expected_state = float(state_value_for_type(intent.target_type))
        expected_influence = 0.0 if intent.target_type is ContactKeyType.FREE else 1.0
        expected_hold_influence = 1.0 if intent.target_type is ContactKeyType.PLANTED else 0.0
        expected_pivot_influence = (
            1.0
            if intent.target_type is ContactKeyType.PLANTED
            and intent.target_plant_space is ContactPlantSpace.WORLD
            else 0.0
        )
        expected_external_influence = (
            1.0
            if intent.target_type is ContactKeyType.PLANTED
            and intent.target_plant_space is ContactPlantSpace.OBJECT
            else 0.0
        )
        if not _same_float(state_value, expected_state):
            raise ContactAuthoringError("Persisted Contact state did not replay to the requested type.")
        if not (
            _same_float(capability.native_ik.constraint.influence, expected_influence)
            and _same_float(capability.terminal_ik_constraint.influence, expected_influence)
            and _same_float(hold.constraint.influence, expected_hold_influence)
            and _same_float(hold.pivot_constraint.influence, expected_pivot_influence)
            and _same_float(hold.external_constraint.influence, expected_external_influence)
        ):
            raise ContactAuthoringError("Persisted Contact influence bundle did not replay atomically.")
        if intent.target_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}:
            pole_channel = next(
                (channel for channel in intent.scalar_channels if channel.row_key.label == "IK_POLE_ANGLE"),
                None,
            )
            if pole_channel is None:
                raise ContactAuthoringError("IK-authoritative Contact verification is missing its pole-angle row.")
            pole_curve = _find_fcurve(bag, pole_channel.row_key.data_path, pole_channel.row_key.array_index)
            pole_value = _exact_scalar_value(pole_curve, time)
            if pole_value is None or not _same_float(pole_value, pole_channel.target_value):
                raise ContactAuthoringError("Persisted IK-authoritative pole-angle key did not match the requested bundle.")
        elif intent.pole_cleanup_snapshot is not None:
            pole_curve = _find_fcurve(
                bag,
                intent.pole_cleanup_snapshot.data_path,
                intent.pole_cleanup_snapshot.array_index,
            )
            if pole_curve is None or _key_at_frame(pole_curve, time) is not None:
                raise ContactAuthoringError("Free replacement retained an authoritative same-frame pole-angle key.")

        hold_location_path = control_property_path(hold.control.target, "location")
        point_location_path = control_property_path(hold.point_control.target, "location")
        if intent.target_type is ContactKeyType.PLANTED:
            if hold_state is None or point_state is None:
                raise ContactAuthoringError("Planted verification lost its frozen Contact anchor states.")
            for location_path, state, label in (
                (hold_location_path, hold_state, "point-hold"),
                (point_location_path, point_state, "continuous-point"),
            ):
                for index, expected_value in enumerate(state.location):
                    curve = _find_fcurve(bag, location_path, index)
                    actual_value = _exact_scalar_value(curve, time)
                    if actual_value is None or not _same_float(actual_value, expected_value):
                        raise ContactAuthoringError(
                            f"Persisted Planted {label} anchor did not match the requested bundle."
                        )
        elif intent.hold_cleanup_curves:
            for cleanup in intent.hold_cleanup_curves:
                curve = _find_fcurve(
                    bag,
                    cleanup.snapshot.data_path,
                    cleanup.snapshot.array_index,
                )
                if curve is None or _key_at_frame(curve, time) is not None:
                    raise ContactAuthoringError(
                        "Non-Planted replacement retained an authoritative same-frame point-hold anchor key."
                    )

        scale = max(1e-6, float(getattr(owner.dimensions, "length", 0.0) or 0.0))
        # Blender's two-bone IK is mildly ill-conditioned when a limb is almost
        # straight (the generated arms intentionally are). Re-evaluating the
        # same target/pole can shift the intermediate joint by a few 1e-6 scene
        # units while terminal position and rotation remain continuous. Keep a
        # tight scale-relative positional guard without treating that solver
        # noise as an authored pose jump. Rotation stays on the stricter guard.
        position_tolerance = max(1e-7, scale * 1e-5)
        rotation_tolerance = 1e-6

        if trigger is not WriterTrigger.AUTO_TRANSFORM:
            _journal_contact_authoring_latch(
                capability,
                intent.target_type,
                contact_plan,
                journal,
                stage_write,
            )
        feedback_muted = intent.target_type is not ContactKeyType.FREE
        _journal_limb_fk_feedback_muted(
            capability,
            feedback_muted,
            contact_plan,
            journal,
            stage_write,
            expected_result=expected_result,
            expected_terminal=expected_terminal,
            hinge_branch=(snap_result.hinge_branch if snap_result is not None else None),
        )
        bpy.context.view_layer.update()
        feedback_constraints = (
            *capability.fk_copy_constraints,
            capability.terminal_fk_constraint,
        )
        if any(bool(constraint.mute) != feedback_muted for constraint in feedback_constraints):
            raise ContactAuthoringError(
                "Persistent Contact transition did not apply the requested FK-feedback authority."
            )
        for control, expected in zip(capability.result_controls, expected_result, strict=True):
            position, rotation = _pose_residual(control.target.matrix, expected)
            if position > position_tolerance or rotation > rotation_tolerance:
                raise ContactAuthoringError(
                    "Contact feedback-authority transition changed the evaluated result-chain pose "
                    f"at {getattr(control.target, 'name', '<unknown>')} "
                    f"(position={position:.9g}, rotation={rotation:.9g})."
                )
        terminal_position, terminal_rotation = _pose_residual(
            capability.result_terminal.target.matrix,
            expected_terminal,
        )
        if terminal_position > position_tolerance or terminal_rotation > rotation_tolerance:
            raise ContactAuthoringError(
                "Contact feedback-authority transition changed the evaluated terminal pose."
            )
        journal.commit()
        hook.enter(OperationStage.COMMIT, operation=contact_plan.operation_id)
        trace_event(
            "WRITER",
            "CONTACT_WRITE_COMMIT",
            operation_id=contact_plan.operation_id,
            context=bpy.context,
            trigger=trigger.value,
            mapping_id=intent.mapping_id,
            target_type=intent.target_type.value,
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        return ContactAuthoringResult(
            True,
            intent.target_type,
            rows_written,
            created_fcurves,
            mapping_contact_types=((intent.mapping_id, intent.target_type),),
        )
    except Exception as original_exc:
        trace_exception(
            "WRITER",
            "CONTACT_WRITE_FAIL",
            original_exc,
            operation_id=contact_plan.operation_id,
            context=bpy.context,
            trigger=trigger.value,
            mapping_id=intent.mapping_id,
            target_type=intent.target_type.value,
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        hook.enter(
            OperationStage.ROLLBACK_WRITE,
            operation=contact_plan.operation_id,
            detail="rollback failed Contact bundle",
        )
        _trace_contact_rollback_report(
            "CONTACT_ROLLBACK_BEGIN",
            operation_id=contact_plan.operation_id,
            journal=journal,
            context=bpy.context,
            trigger=trigger.value,
            mapping_id=intent.mapping_id,
            target_type=intent.target_type.value,
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        rollback = journal.rollback()
        _trace_contact_rollback_report(
            "CONTACT_ROLLBACK_END",
            operation_id=contact_plan.operation_id,
            journal=journal,
            context=bpy.context,
            rollback=rollback,
            trigger=trigger.value,
            mapping_id=intent.mapping_id,
            target_type=intent.target_type.value,
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        _reevaluate_contact_frame_preserving_public_pose(
            scene,
            contact_plan.frame,
            contact_plan.subframe,
        )
        hook.enter(OperationStage.ROLLBACK_VERIFY, operation=contact_plan.operation_id)
        if rollback.residue_receipts:
            detail = " | ".join(item.detail for item in rollback.diagnostics)
            raise ContactAuthoringError(
                f"I13 rollback incomplete after {type(original_exc).__name__}: {detail}"
            ) from original_exc
        raise


def execute_contact_batch_intent_plan(
    scene,
    control_context,
    batch: ContactBatchIntentPlan,
    *,
    closure_plan: OperationPlan | None = None,
    trigger: WriterTrigger = WriterTrigger.CONTACT_AUTHORING,
    auto_baseline_rows_by_mapping: tuple[
        tuple[str, tuple[PlannedChannel, ...]], ...
    ] = (),
    auto_baseline_time: float | None = None,
    selector_character_id: str | None = None,
    hook: StageHook | None = None,
) -> ContactAuthoringResult:
    """Commit dependency-independent limb Contact bundles in one persistent journal."""

    hook = hook or NoopStageHook()
    trace_event(
        "WRITER",
        "CONTACT_BATCH_WRITE_BEGIN",
        operation_id=batch.operation_id,
        context=bpy.context,
        trigger=trigger.value,
        mappings=tuple(
            (intent.mapping_id, intent.target_type.value)
            for intent in batch.intents
        ),
        auto_baseline_mappings=tuple(
            mapping_id for mapping_id, _rows in auto_baseline_rows_by_mapping
        ),
        auto_baseline_time=auto_baseline_time,
    )
    if len(batch.intents) < 2:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    batch.operation_id,
                    "I20_MULTI_LIMB_BATCH_TOO_SMALL",
                    "Multi-limb Contact execution requires at least two planned limb domains.",
                ),
            ),
        )
    if closure_plan is not None:
        closure_freshness = validate_plan_fresh(
            scene,
            control_context,
            closure_plan,
            selector_character_id=selector_character_id,
        )
        if not closure_freshness.ok:
            return ContactAuthoringResult(False, diagnostics=closure_freshness.diagnostics)
    prepared_items: list[ContactPreparedIntent] = []
    for intent in batch.intents:
        try:
            prepared, diagnostics = _prepare_contact_intent_for_batch(
                scene,
                control_context,
                intent,
                selector_character_id=selector_character_id,
                hook=hook,
            )
        except ContactAuthoringError as exc:
            return ContactAuthoringResult(
                False,
                diagnostics=(
                    _diagnostic(
                        batch.operation_id,
                        "I20_LIMB_PREPARE_REJECTED",
                        str(exc),
                        character_id=intent.snap_plan.character_id,
                    ),
                ),
            )
        if prepared is None:
            return ContactAuthoringResult(False, diagnostics=diagnostics)
        prepared_items.append(prepared)
    prepared_tuple = tuple(prepared_items)

    if auto_baseline_rows_by_mapping:
        if trigger is not WriterTrigger.AUTO_TRANSFORM:
            raise ContactAuthoringError(
                "AUTO batch baselines require the AUTO_TRANSFORM trigger."
            )
        if auto_baseline_time is None:
            raise ContactAuthoringError("AUTO batch baseline time is missing.")
        prepared_by_mapping = {
            item.intent.mapping_id: item for item in prepared_tuple
        }
        for mapping_id, baseline_rows in auto_baseline_rows_by_mapping:
            item = prepared_by_mapping.get(mapping_id)
            if item is None:
                raise ContactAuthoringError(
                    f"AUTO batch baseline mapping disappeared: {mapping_id}."
                )
            if item.intent.target_type is not ContactKeyType.FREE:
                raise ContactAuthoringError(
                    "AUTO first-key batch baseline is supported only for Free authority."
                )
            current_by_key = {
                row.row_key: row for row in item.transform_rows
            }
            if {row.row_key for row in baseline_rows} != set(current_by_key):
                raise ContactAuthoringError(
                    f"AUTO batch baseline footprint changed for {mapping_id}."
                )
            for baseline in baseline_rows:
                current = current_by_key[baseline.row_key]
                if (
                    baseline.owner_runtime_key != current.owner_runtime_key
                    or baseline.control_runtime_key != current.control_runtime_key
                    or baseline.existing_fcurve_token != current.existing_fcurve_token
                    or baseline.allocation_intent is not current.allocation_intent
                ):
                    raise ContactAuthoringError(
                        f"AUTO batch baseline storage identity changed for {mapping_id}."
                    )

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return ContactAuthoringResult(
            False,
            diagnostics=tuple(
                _diagnostic(batch.operation_id, issue.code, issue.detail)
                for issue in resolution.issues
            ),
        )
    target = resolution.target
    try:
        contact_plan = _contact_batch_operation_plan(
            batch,
            target,
            prepared_tuple,
            closure_plan=closure_plan,
        )
    except ContactAuthoringError as exc:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                _diagnostic(
                    batch.operation_id,
                    "I20_DEPENDENCY_CONFLICT",
                    str(exc),
                    character_id=target.character_id,
                ),
            ),
        )

    contact_freshness = validate_plan_fresh(
        scene,
        control_context,
        contact_plan,
        selector_character_id=selector_character_id,
    )
    if not contact_freshness.ok:
        return ContactAuthoringResult(False, diagnostics=contact_freshness.diagnostics)

    owner = prepared_tuple[0].capability.native_ik.solver_owner.owner_object
    for item in prepared_tuple:
        if (
            _runtime_pointer(item.capability.native_ik.solver_owner.owner_object)
            != contact_plan.owner_groups[0].owner_runtime_key
            or not _scalar_tokens_match(owner, item.intent.scalar_channels)
            or not _pole_cleanup_snapshot_matches(owner, item.intent)
            or not _hold_cleanup_snapshots_match(owner, item.intent)
        ):
            return ContactAuthoringResult(
                False,
                diagnostics=(
                    _diagnostic(
                        batch.operation_id,
                        "I20_BATCH_STALE_BEFORE_COMMIT",
                        "A selected limb Contact binding changed after combined planning.",
                        character_id=target.character_id,
                    ),
                ),
            )

    group = contact_plan.owner_groups[0]
    _check_scope_not_quarantined(contact_plan, group)
    journal = MutationJournal(
        contact_plan.operation_id,
        contact_plan.character_id,
        contact_plan.setup_revision,
        contact_plan.setup_signature,
        BlenderRollbackExecutor(),
    )
    mutation_ordinal = 0

    def stage_write(detail: str) -> None:
        nonlocal mutation_ordinal
        mutation_ordinal += 1
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=mutation_ordinal,
            operation=contact_plan.operation_id,
            detail=detail,
        )

    rows_written = 0
    created_fcurves = 0
    semantic_properties_prepared: set[tuple[int, str]] = set()
    time = float(contact_plan.frame) + float(contact_plan.subframe)
    constant_anchor_paths = {
        control_property_path(item.hold.control.target, "location")
        for item in prepared_tuple
        if item.intent.target_type is ContactKeyType.PLANTED
    }
    constant_anchor_paths.update(
        control_property_path(item.hold.point_control.target, "location")
        for item in prepared_tuple
        if item.intent.target_type is ContactKeyType.PLANTED
    )

    try:
        bag = _ensure_owner_storage(owner, group, contact_plan, journal, stage_write)
        if auto_baseline_rows_by_mapping:
            assert auto_baseline_time is not None
            prepared_by_mapping = {
                item.intent.mapping_id: item for item in prepared_tuple
            }
            for mapping_id, baseline_rows in auto_baseline_rows_by_mapping:
                item = prepared_by_mapping[mapping_id]
                current_by_key = {
                    row.row_key: row for row in item.transform_rows
                }
                for baseline in baseline_rows:
                    current = current_by_key[baseline.row_key]
                    existing_curve = _find_fcurve(
                        bag,
                        current.row_key.data_path,
                        current.row_key.array_index,
                    )
                    if baseline.existing_fcurve_token is None:
                        if existing_curve is not None:
                            raise ContactAuthoringError(
                                "AUTO batch baseline FCurve appeared after gesture begin."
                            )
                    else:
                        if (
                            existing_curve is None
                            or _runtime_pointer(existing_curve)
                            != baseline.existing_fcurve_token
                            or len(existing_curve.keyframe_points) != 0
                        ):
                            raise ContactAuthoringError(
                                "AUTO batch baseline FCurve changed after gesture begin."
                            )
                    fcurve, created = _ensure_fcurve(
                        owner,
                        bag,
                        current,
                        contact_plan,
                        journal,
                        stage_write,
                    )
                    created_fcurves += int(created)
                    baseline_current = replace(
                        current,
                        target_value=baseline.target_value,
                    )
                    _write_channel_key(
                        owner,
                        bag,
                        fcurve,
                        baseline_current,
                        float(auto_baseline_time),
                        contact_plan,
                        journal,
                        stage_write,
                    )
                    rows_written += 1

        for channel in group.channels:
            _prepare_semantic_state_property(
                owner,
                channel,
                contact_plan,
                journal,
                stage_write,
                semantic_properties_prepared,
            )
            fcurve, created = _ensure_fcurve(
                owner,
                bag,
                channel,
                contact_plan,
                journal,
                stage_write,
            )
            created_fcurves += int(created)
            _write_channel_key(
                owner,
                bag,
                fcurve,
                channel,
                time,
                contact_plan,
                journal,
                stage_write,
            )
            if channel.row_key.data_path in constant_anchor_paths:
                fcurve.extrapolation = "CONSTANT"
                _set_constant_at(fcurve, time)
            rows_written += 1

        for item in prepared_tuple:
            intent = item.intent
            if intent.pole_cleanup_snapshot is not None:
                snapshot = intent.pole_cleanup_snapshot
                pole_curve = _find_fcurve(bag, snapshot.data_path, snapshot.array_index)
                pole_curve_ptr = _runtime_pointer(pole_curve) if pole_curve is not None else None
                bag_ptr = _runtime_pointer(bag)
                if (
                    pole_curve is None
                    or pole_curve_ptr != intent.pole_cleanup_fcurve_token
                    or bag_ptr is None
                    or _capture_fcurve_snapshot(pole_curve) != snapshot
                ):
                    raise ContactAuthoringError("I20 pole-angle cleanup target changed before mutation.")
                pole_key = _key_at_frame(pole_curve, time)
                if pole_key is None:
                    raise ContactAuthoringError("I20 same-frame pole-angle key disappeared before cleanup.")
                journal.record(
                    FCurveMutationReceipt(
                        journal.next_ordinal(),
                        _scope(contact_plan, owner_ptr=group.owner_runtime_key, bag_ptr=int(bag_ptr)),
                        int(bag_ptr),
                        int(pole_curve_ptr),
                        snapshot,
                    )
                )
                stage_write(f"I20 remove Contact pole angle {intent.mapping_id} @ {time}")
                pole_curve.keyframe_points.remove(pole_key, fast=True)
                pole_curve.update()

            for cleanup in intent.hold_cleanup_curves:
                snapshot = cleanup.snapshot
                cleanup_curve = _find_fcurve(bag, snapshot.data_path, snapshot.array_index)
                cleanup_ptr = _runtime_pointer(cleanup_curve) if cleanup_curve is not None else None
                bag_ptr = _runtime_pointer(bag)
                if (
                    cleanup_curve is None
                    or cleanup_ptr != cleanup.fcurve_token
                    or bag_ptr is None
                    or _capture_fcurve_snapshot(cleanup_curve) != snapshot
                ):
                    raise ContactAuthoringError("I20 Contact anchor cleanup target changed before mutation.")
                cleanup_key = _key_at_frame(cleanup_curve, time)
                if cleanup_key is None:
                    raise ContactAuthoringError("I20 same-frame Planted anchor key disappeared before cleanup.")
                journal.record(
                    FCurveMutationReceipt(
                        journal.next_ordinal(),
                        _scope(contact_plan, owner_ptr=group.owner_runtime_key, bag_ptr=int(bag_ptr)),
                        int(bag_ptr),
                        int(cleanup_ptr),
                        snapshot,
                    )
                )
                stage_write(
                    f"I20 remove Contact anchor {intent.mapping_id} "
                    f"{snapshot.data_path}[{snapshot.array_index}] @ {time}"
                )
                cleanup_curve.keyframe_points.remove(cleanup_key, fast=True)
                cleanup_curve.update()

        auto_baseline_mapping_ids = {
            mapping_id for mapping_id, _rows in auto_baseline_rows_by_mapping
        }
        for item in prepared_tuple:
            intent = item.intent
            for scalar in intent.scalar_channels:
                fcurve, created = _ensure_fcurve(
                    owner,
                    bag,
                    scalar,
                    contact_plan,
                    journal,
                    stage_write,
                )
                created_fcurves += int(created)
                if intent.baseline_required and scalar.row_key.label in {
                    "CONTACT_STATE",
                    "IK_INFLUENCE",
                    "TERMINAL_IK_INFLUENCE",
                    "POINT_HOLD_INFLUENCE",
                    "CONTACT_POINT_PIVOT_INFLUENCE",
                    "EXTERNAL_OBJECT_SPACE_INFLUENCE",
                }:
                    uses_auto_baseline = (
                        trigger is WriterTrigger.AUTO_TRANSFORM
                        and intent.mapping_id in auto_baseline_mapping_ids
                        and auto_baseline_time is not None
                    )
                    baseline_time = (
                        float(auto_baseline_time)
                        if uses_auto_baseline
                        else float(scene.frame_start)
                    )
                    baseline_value = 0.0
                    if (
                        uses_auto_baseline
                        and intent.target_type is ContactKeyType.FREE
                        and scalar.row_key.label == "CONTACT_STATE"
                    ):
                        baseline_value = float(
                            state_value_for_type(ContactKeyType.FREE)
                        )
                    baseline = replace(
                        scalar,
                        target_value=baseline_value,
                    )
                    _write_channel_key(
                        owner,
                        bag,
                        fcurve,
                        baseline,
                        baseline_time,
                        contact_plan,
                        journal,
                        stage_write,
                    )
                    _set_constant_at(fcurve, baseline_time)
                    rows_written += 1
                _write_channel_key(
                    owner,
                    bag,
                    fcurve,
                    scalar,
                    time,
                    contact_plan,
                    journal,
                    stage_write,
                )
                _set_constant_at(fcurve, time)
                rows_written += 1

        reevaluate_feedback_constraints = tuple(
            constraint
            for item in prepared_tuple
            for constraint in (
                *item.capability.fk_copy_constraints,
                item.capability.terminal_fk_constraint,
            )
        )
        hook.enter(OperationStage.REEVALUATE, operation=contact_plan.operation_id)
        _reevaluate_contact_frame_preserving_public_pose(
            scene,
            contact_plan.frame,
            contact_plan.subframe,
            feedback_constraints=reevaluate_feedback_constraints,
        )
        hook.enter(OperationStage.VERIFY, operation=contact_plan.operation_id)

        scale = max(1e-6, float(getattr(owner.dimensions, "length", 0.0) or 0.0))
        position_tolerance = max(1e-7, scale * 1e-5)
        rotation_tolerance = 1e-6
        for item in prepared_tuple:
            intent = item.intent
            capability = item.capability
            hold = item.hold
            state_value = float(
                capability.native_ik.solver_owner.target[AWB_CONTACT_STATE_PROPERTY]
            )
            expected_state = float(state_value_for_type(intent.target_type))
            expected_influence = 0.0 if intent.target_type is ContactKeyType.FREE else 1.0
            expected_hold = 1.0 if intent.target_type is ContactKeyType.PLANTED else 0.0
            expected_pivot = (
                1.0
                if intent.target_type is ContactKeyType.PLANTED
                and intent.target_plant_space is ContactPlantSpace.WORLD
                else 0.0
            )
            expected_external = (
                1.0
                if intent.target_type is ContactKeyType.PLANTED
                and intent.target_plant_space is ContactPlantSpace.OBJECT
                else 0.0
            )
            if not (
                _same_float(state_value, expected_state)
                and _same_float(capability.native_ik.constraint.influence, expected_influence)
                and _same_float(capability.terminal_ik_constraint.influence, expected_influence)
                and _same_float(hold.constraint.influence, expected_hold)
                and _same_float(hold.pivot_constraint.influence, expected_pivot)
                and _same_float(hold.external_constraint.influence, expected_external)
            ):
                raise ContactAuthoringError(
                    f"I20 persisted Contact influence/state bundle disagrees for {intent.mapping_id}."
                )

            if intent.target_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}:
                pole_channel = next(
                    (
                        channel
                        for channel in intent.scalar_channels
                        if channel.row_key.label == "IK_POLE_ANGLE"
                    ),
                    None,
                )
                if pole_channel is None:
                    raise ContactAuthoringError("I20 IK-authoritative bundle lost pole-angle row.")
                pole_curve = _find_fcurve(
                    bag,
                    pole_channel.row_key.data_path,
                    pole_channel.row_key.array_index,
                )
                pole_value = _exact_scalar_value(pole_curve, time)
                if pole_value is None or not _same_float(pole_value, pole_channel.target_value):
                    raise ContactAuthoringError(
                        f"I20 persisted pole angle disagrees for {intent.mapping_id}."
                    )

            if intent.target_type is ContactKeyType.PLANTED:
                if item.hold_state is None or item.point_state is None:
                    raise ContactAuthoringError("I20 Planted verification lost frozen anchor states.")
                for contract, state, label in (
                    (hold.control, item.hold_state, "point-hold"),
                    (hold.point_control, item.point_state, "continuous-point"),
                ):
                    path = control_property_path(contract.target, "location")
                    for index, expected_value in enumerate(state.location):
                        curve = _find_fcurve(bag, path, index)
                        actual_value = _exact_scalar_value(curve, time)
                        if actual_value is None or not _same_float(actual_value, expected_value):
                            raise ContactAuthoringError(
                                f"I20 persisted {label} anchor disagrees for {intent.mapping_id}."
                            )

        if trigger is not WriterTrigger.AUTO_TRANSFORM:
            for item in prepared_tuple:
                _journal_contact_authoring_latch(
                    item.capability,
                    item.intent.target_type,
                    contact_plan,
                    journal,
                    stage_write,
                )
        for item in prepared_tuple:
            _journal_limb_fk_feedback_muted(
                item.capability,
                item.intent.target_type is not ContactKeyType.FREE,
                contact_plan,
                journal,
                stage_write,
                expected_result=item.expected_result,
                expected_terminal=item.expected_terminal,
                hinge_branch=item.hinge_branch,
            )
        bpy.context.view_layer.update()
        for item in prepared_tuple:
            feedback_muted = item.intent.target_type is not ContactKeyType.FREE
            feedback_constraints = (
                *item.capability.fk_copy_constraints,
                item.capability.terminal_fk_constraint,
            )
            if any(
                bool(constraint.mute) != feedback_muted
                for constraint in feedback_constraints
            ):
                raise ContactAuthoringError(
                    f"I20 feedback authority disagrees for {item.intent.mapping_id}."
                )
            for control, expected in zip(
                item.capability.result_controls,
                item.expected_result,
                strict=True,
            ):
                position, rotation = _pose_residual(control.target.matrix, expected)
                if position > position_tolerance or rotation > rotation_tolerance:
                    raise ContactAuthoringError(
                        f"I20 feedback-authority transition changed result pose for "
                        f"{item.intent.mapping_id}."
                    )
            terminal_position, terminal_rotation = _pose_residual(
                item.capability.result_terminal.target.matrix,
                item.expected_terminal,
            )
            if terminal_position > position_tolerance or terminal_rotation > rotation_tolerance:
                raise ContactAuthoringError(
                    f"I20 feedback-authority transition changed terminal pose for "
                    f"{item.intent.mapping_id}."
                )
        journal.commit()
        hook.enter(OperationStage.COMMIT, operation=contact_plan.operation_id)
        mapping_types = tuple(
            (item.intent.mapping_id, item.intent.target_type)
            for item in prepared_tuple
        )
        common_type = (
            mapping_types[0][1]
            if all(contact_type is mapping_types[0][1] for _mapping, contact_type in mapping_types)
            else None
        )
        trace_event(
            "WRITER",
            "CONTACT_BATCH_WRITE_COMMIT",
            operation_id=contact_plan.operation_id,
            context=bpy.context,
            trigger=trigger.value,
            mappings=tuple(
                (mapping_id, contact_type.value)
                for mapping_id, contact_type in mapping_types
            ),
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        return ContactAuthoringResult(
            True,
            common_type,
            rows_written,
            created_fcurves,
            mapping_contact_types=mapping_types,
        )
    except Exception as original_exc:
        trace_exception(
            "WRITER",
            "CONTACT_BATCH_WRITE_FAIL",
            original_exc,
            operation_id=contact_plan.operation_id,
            context=bpy.context,
            trigger=trigger.value,
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        hook.enter(
            OperationStage.ROLLBACK_WRITE,
            operation=contact_plan.operation_id,
            detail="rollback failed I20 multi-limb Contact bundle",
        )
        _trace_contact_rollback_report(
            "CONTACT_BATCH_ROLLBACK_BEGIN",
            operation_id=contact_plan.operation_id,
            journal=journal,
            context=bpy.context,
            trigger=trigger.value,
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        rollback = journal.rollback()
        _trace_contact_rollback_report(
            "CONTACT_BATCH_ROLLBACK_END",
            operation_id=contact_plan.operation_id,
            journal=journal,
            context=bpy.context,
            rollback=rollback,
            trigger=trigger.value,
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        _reevaluate_contact_frame_preserving_public_pose(
            scene,
            contact_plan.frame,
            contact_plan.subframe,
        )
        hook.enter(OperationStage.ROLLBACK_VERIFY, operation=contact_plan.operation_id)
        if rollback.residue_receipts:
            detail = " | ".join(item.detail for item in rollback.diagnostics)
            raise ContactAuthoringError(
                f"I20 rollback incomplete after {type(original_exc).__name__}: {detail}"
            ) from original_exc
        raise


def execute_contact_command(
    scene,
    control_context,
    *,
    operation_id: str,
    enabled_types: tuple[ContactKeyType, ...] = (ContactKeyType.FREE, ContactKeyType.SLIDING),
    plant_space: ContactPlantSpace = ContactPlantSpace.WORLD,
    contact_point_local: tuple[float, float, float] = (0.0, 0.0, 0.0),
    selector_character_id: str | None = None,
    hook: StageHook | None = None,
) -> ContactAuthoringResult:
    """Coordinate one explicit C operation from one frozen selection domain.

    E3 keeps Contact as the transaction owner.  Supported direct controls are
    planned independently through the E2 subset API, then supplied as Contact
    closure rows so one Contact journal owns every persistent mutation, rollback,
    verification pass, and commit.  Direct-only C remains fail-closed.
    """

    operation_domain = resolve_operation_domain(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    snapshot = operation_domain.snapshot
    if snapshot is None or operation_domain.issues:
        character_id = snapshot.character_id if snapshot is not None else None
        return ContactAuthoringResult(
            False,
            diagnostics=tuple(
                _diagnostic(
                    operation_id,
                    issue.code,
                    issue.detail,
                    character_id=character_id,
                )
                for issue in operation_domain.issues
            ),
        )

    mapping_ids = tuple(snapshot.contact_mapping_ids)
    if not mapping_ids:
        # Direct-only C is intentionally not part of this stabilization.
        return ContactAuthoringResult(False)

    direct_plan: OperationPlan | None = None
    if snapshot.supported_direct_binding_ids:
        direct_built = build_direct_key_plan(
            scene,
            control_context,
            operation_id=operation_id,
            selector_character_id=selector_character_id,
            include_rigped_free_marker=True,
            binding_ids=tuple(snapshot.supported_direct_binding_ids),
            operation_domain=operation_domain,
        )
        if not direct_built.ok or direct_built.plan is None:
            return ContactAuthoringResult(False, diagnostics=direct_built.diagnostics)
        direct_plan = direct_built.plan

    try:
        if len(mapping_ids) > 1:
            batch_build = build_contact_batch_intent_plan(
                scene,
                control_context,
                operation_id=operation_id,
                mode=ContactAuthoringMode.CYCLE,
                enabled_types=enabled_types,
                plant_space=plant_space,
                contact_point_local=contact_point_local,
                selector_character_id=selector_character_id,
                mapping_ids=mapping_ids,
            )
            if not batch_build.ok or batch_build.plan is None:
                return ContactAuthoringResult(False, diagnostics=batch_build.diagnostics)
            return execute_contact_batch_intent_plan(
                scene,
                control_context,
                batch_build.plan,
                closure_plan=direct_plan,
                selector_character_id=selector_character_id,
                hook=hook,
            )

        planned = build_contact_intent_plan(
            scene,
            control_context,
            operation_id=operation_id,
            mode=ContactAuthoringMode.CYCLE,
            enabled_types=enabled_types,
            plant_space=plant_space,
            contact_point_local=contact_point_local,
            mapping_id=mapping_ids[0],
            selector_character_id=selector_character_id,
        )
        if not planned.ok or planned.plan is None:
            return ContactAuthoringResult(False, diagnostics=planned.diagnostics)
        return execute_contact_intent_plan(
            scene,
            control_context,
            planned.plan,
            closure_plan=direct_plan,
            selector_character_id=selector_character_id,
            hook=hook,
        )
    except DirectWriterError as exc:
        raise ContactAuthoringError(str(exc)) from exc

def execute_contact_anchor(
    scene,
    control_context,
    *,
    operation_id: str,
    selector_character_id: str | None = None,
    hook: StageHook | None = None,
) -> ContactAuthoringResult:
    """Key the current authoritative Contact state without cycling it."""

    mapping_ids = selected_contact_mapping_ids(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if len(mapping_ids) > 1:
        batch_build = build_contact_batch_intent_plan(
            scene,
            control_context,
            operation_id=operation_id,
            mode=ContactAuthoringMode.ANCHOR,
            selector_character_id=selector_character_id,
        )
        if not batch_build.ok or batch_build.plan is None:
            return ContactAuthoringResult(False, diagnostics=batch_build.diagnostics)
        return execute_contact_batch_intent_plan(
            scene,
            control_context,
            batch_build.plan,
            selector_character_id=selector_character_id,
            hook=hook,
        )

    planned = build_contact_intent_plan(
        scene,
        control_context,
        operation_id=operation_id,
        mode=ContactAuthoringMode.ANCHOR,
        selector_character_id=selector_character_id,
    )
    if not planned.ok or planned.plan is None:
        return ContactAuthoringResult(False, diagnostics=planned.diagnostics)
    try:
        return execute_contact_intent_plan(
            scene,
            control_context,
            planned.plan,
            selector_character_id=selector_character_id,
            hook=hook,
        )
    except DirectWriterError as exc:
        raise ContactAuthoringError(str(exc)) from exc


def execute_contact_replant(
    scene,
    control_context,
    *,
    operation_id: str,
    plant_space: ContactPlantSpace = ContactPlantSpace.WORLD,
    contact_point_local: tuple[float, float, float] = (0.0, 0.0, 0.0),
    selector_character_id: str | None = None,
    hook: StageHook | None = None,
) -> ContactAuthoringResult:
    """Explicitly replace the current Planted anchor/point without implicit historical edits."""

    mapping_ids = selected_contact_mapping_ids(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if len(mapping_ids) > 1:
        batch_build = build_contact_batch_intent_plan(
            scene,
            control_context,
            operation_id=operation_id,
            mode=ContactAuthoringMode.REPLANT,
            plant_space=plant_space,
            contact_point_local=contact_point_local,
            selector_character_id=selector_character_id,
        )
        if not batch_build.ok or batch_build.plan is None:
            return ContactAuthoringResult(False, diagnostics=batch_build.diagnostics)
        return execute_contact_batch_intent_plan(
            scene,
            control_context,
            batch_build.plan,
            selector_character_id=selector_character_id,
            hook=hook,
        )

    planned = build_contact_intent_plan(
        scene,
        control_context,
        operation_id=operation_id,
        mode=ContactAuthoringMode.REPLANT,
        plant_space=plant_space,
        contact_point_local=contact_point_local,
        selector_character_id=selector_character_id,
    )
    if not planned.ok or planned.plan is None:
        return ContactAuthoringResult(False, diagnostics=planned.diagnostics)
    try:
        return execute_contact_intent_plan(
            scene,
            control_context,
            planned.plan,
            selector_character_id=selector_character_id,
            hook=hook,
        )
    except DirectWriterError as exc:
        raise ContactAuthoringError(str(exc)) from exc
