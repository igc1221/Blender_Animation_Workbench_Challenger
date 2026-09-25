from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, pi, radians, sqrt
from typing import Any, ClassVar
from uuid import uuid4

import bpy
from bpy.app.handlers import persistent
from bpy.props import EnumProperty, FloatProperty
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Matrix, Quaternion, Vector

from .character_metadata import resolve_character
from .debug_trace import (
    TraceRouteOutcome,
    TraceTerminalStatus,
    bind_operation_causal_context,
    link_trace_operation,
    new_trace_causal_root,
    new_trace_operation_id,
    register_debug_state_probe,
    trace_event,
    trace_exception,
    unregister_debug_state_probe,
)
from .gizmo_preferences import (
    MAX_LINEAR_ROTATION_RADIANS_PER_PIXEL,
    arcball_world_step,
    clear_rotation_angle,
    gizmo_zoom_scale,
    linear_roll_screen_tangent,
    projection_world_per_pixel,
    show_rotation_angle,
    snapped_rotation_angle,
)
from .kinematic_runtime import evaluated_chain_tip_world_position
from .phase4_contact_authoring import (
    ContactAuthoringError,
    ContactAuthoringMode,
    ContactAuthoringResult,
    ContactIntentPlan,
    build_contact_intent_plan,
    clear_contact_bundle_key_selection_for_context,
    execute_contact_intent_plan,
    selected_contact_limb_control_keys,
    selected_contact_mapping_ids,
)
from .phase4_contact_model import (
    AWB_CONTACT_STATE_PROPERTY,
    ContactKeyType,
    type_for_state_value,
)
from .phase4_representation_snap import (
    LimbRepresentationCapability,
    SnapControlState,
    SnapDirection,
    _canonical_generated_rigped_pole,
    _derived_pole_angle,
    _generated_lower_hinge_x_angle,
    _generated_rigped_hinge_branch,
    _initial_pole_solution,
    build_representation_snap_payload,
    execute_representation_snap,
    resolve_limb_representation_capability,
    set_limb_fk_feedback_muted,
)
from .phase4_writer import WriterTrigger
from .rigped_auto_key import (
    RigpedAutoAnchorPlan,
    RigpedAutoContactBatchPlan,
    RigpedAutoDirectMovePlan,
    RigpedAutoDirectRotatePlan,
    commit_rigped_auto_anchor,
    commit_rigped_auto_contact_batch,
    commit_rigped_auto_direct_move,
    commit_rigped_auto_direct_rotate,
    commit_rigped_auto_writer_results,
    plan_rigped_auto_anchor,
    plan_rigped_auto_contact_batch,
    plan_rigped_auto_direct_move,
    plan_rigped_auto_direct_rotate,
)
from .rigped_auto_key import (
    rollback_rigped_auto_writer_results as _rollback_rigped_auto_writer_results_or_raise,
)
from .rigped_contract import (
    RIGPED_SETUP_PROPERTY,
    RigpedTransformGesture,
    RigpedTransformRoute,
    resolve_rigped_target,
    resolve_rigped_transform,
)
from .rigped_humanoid_builder import configure_generated_rigped_ik_hinge_branch
from .rigped_limb_math import preferred_two_bone_bend
from .rigped_operation_domain import OperationDomainSnapshot, resolve_operation_domain
from .rigped_sliding_evaluation import (
    SlidingAuthoredReference,
    build_transient_solver_seed,
    capture_native_solved_result,
    capture_sliding_authored_reference,
    derive_public_display,
    native_pose_basis_from_matrix,
)
from .semantic_adapter import (
    ResolvedControl,
    control_context_for_context,
    rotation_property,
    runtime_control_key,
)
from .ui_language import text
from .viewport_transform_math import axis_point as _axis_point
from .viewport_transform_math import plane_point as _plane_point
from .viewport_transform_math import rotation_vector as _rotation_vector


class RigpedSemanticMoveError(RuntimeError):
    pass


def _report_operator_error(operator, context, exc: BaseException, **data) -> None:
    """Mirror Blender-visible operator errors into the Flight Recorder."""
    trace_exception(
        "ERROR",
        "OPERATOR_ERROR",
        exc,
        operation_id=getattr(operator, "_trace_operation_id", None),
        context=context,
        operator_id=str(getattr(operator, "bl_idname", type(operator).__name__)),
        stage="ERROR_REPORT",
        **data,
    )
    operator.report({"ERROR"}, str(exc))


def rollback_rigped_auto_writer_results(results) -> BaseException | None:
    """Best-effort AUTO rollback wrapper that never skips gesture cleanup.

    The rigped-auto helper raises only after every open journal has attempted
    rollback. Transform failure paths still must restore pose/preview state even
    when that aggregate rollback reports residue, so record it here and let the
    caller continue its cleanup sequence.
    """

    try:
        _rollback_rigped_auto_writer_results_or_raise(tuple(results))
    except Exception as exc:  # noqa: BLE001 - cleanup must continue after aggregate rollback failure
        operation_id = next(
            (
                str(result.operation_id)
                for result in results
                if getattr(result, "operation_id", None)
            ),
            None,
        )
        trace_exception(
            "ERROR",
            "AUTO_ROLLBACK_AGGREGATE_FAILURE",
            exc,
            operation_id=operation_id,
        )
        return exc
    return None


@dataclass(frozen=True, slots=True)
class SemanticMoveSelection:
    character_id: str
    binding_id: str
    capability: LimbRepresentationCapability
    active_control: ResolvedControl


@dataclass(frozen=True, slots=True)
class SemanticMoveResolution:
    character_id: str
    binding_id: str
    contact_type: ContactKeyType
    capability: LimbRepresentationCapability
    active_control: ResolvedControl


@dataclass(frozen=True, slots=True)
class DirectTransformSelection:
    active_control: ResolvedControl
    move_native: bool
    rotate_native: bool


@dataclass(slots=True)
class DirectMoveControlState:
    control: ResolvedControl
    start_location: tuple[float, float, float]
    world_to_local_delta: Matrix


@dataclass(slots=True)
class DirectRotateControlState:
    control: ResolvedControl
    start_basis: Matrix
    start_matrix: Matrix
    pivot_local: Vector
    center_pivot: bool
    hinge_state: tuple[float, float, Quaternion] | None = None


@dataclass(slots=True)
class SlidingRotateSyncSession:
    intent: ContactIntentPlan
    capability: LimbRepresentationCapability
    terminal_selected: bool
    start_fk_states: tuple[SnapControlState, ...]
    start_terminal_state: SnapControlState
    start_solver_state: SnapControlState
    start_ik_state: SnapControlState
    start_pole_state: SnapControlState
    start_pole_angle: float
    start_ik_influence: float
    start_terminal_ik_influence: float
    start_feedback_mutes: tuple[bool, ...]
    start_hinge_settings: tuple[Any, ...]


@dataclass(slots=True)
class SlidingHiddenSeedSnapshot:
    capability: LimbRepresentationCapability
    result_states: tuple[SnapControlState, ...]
    terminal_state: SnapControlState


@dataclass(slots=True)
class SlidingDependencyGuard:
    capability: LimbRepresentationCapability
    reference: SlidingAuthoredReference
    start_ik_state: SnapControlState
    start_pole_state: SnapControlState | None
    start_ik_influence: float
    start_ik_mute: bool
    start_terminal_ik_influence: float
    start_terminal_ik_mute: bool
    start_hinge_settings: tuple[Any, ...]
    start_result_terminal: Matrix
    start_public_terminal: Matrix
    ik_runtime_key: tuple[int, int]
    pole_runtime_key: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class FkJointMoveCandidate:
    """Static selection-to-operation routing cached across viewport redraws."""

    active_control: ResolvedControl
    semantic_selection: SemanticMoveSelection | None
    domain_key: str
    kind: str


_SEMANTIC_MOVE_SELECTION_CACHE_KEY: tuple[Any, ...] | None = None
_SEMANTIC_MOVE_SELECTION_CACHE_VALUE: SemanticMoveSelection | None = None
_DIRECT_TRANSFORM_SELECTION_CACHE_KEY: tuple[Any, ...] | None = None
_DIRECT_TRANSFORM_SELECTION_CACHE_VALUE: DirectTransformSelection | None = None
_CLAVICLE_SELECTION_CACHE_KEY: tuple[Any, ...] | None = None
_CLAVICLE_SELECTION_CACHE_VALUE: ResolvedControl | None = None
_FK_JOINT_MOVE_CANDIDATE_CACHE_KEY: tuple[Any, ...] | None = None
_FK_JOINT_MOVE_CANDIDATE_CACHE_VALUE: tuple[FkJointMoveCandidate, ...] = ()
_SLIDING_MOVE_SELECTION_CACHE_KEY: tuple[Any, ...] | None = None
_SLIDING_MOVE_SELECTION_CACHE_VALUE: tuple[SemanticMoveSelection, ...] = ()


@dataclass(slots=True)
class SlidingMoveSession:
    intent: ContactIntentPlan
    capability: LimbRepresentationCapability
    ik_target: ResolvedControl
    pole_target: ResolvedControl
    start_location: tuple[float, float, float]
    pole_start_location: tuple[float, float, float]
    local_delta_matrix: Matrix
    pole_local_delta_matrix: Matrix
    start_pivot_world: Vector
    start_pole_world: Vector
    start_axis_world: Vector
    pole_perp_world: Vector
    pole_axial_offset: float
    pole_perp_distance: float
    start_fk_states: tuple[SnapControlState, ...]
    start_terminal_state: SnapControlState
    analytic_fk_session: FkTwoBoneMoveSession | None = None


@dataclass(slots=True)
class FreeMoveSession:
    intent: ContactIntentPlan
    capability: LimbRepresentationCapability
    ik_target: ResolvedControl
    pole_target: ResolvedControl
    local_delta_matrix: Matrix
    start_pivot_world: Vector
    start_fk_states: tuple[SnapControlState, ...]
    start_terminal_state: SnapControlState
    start_ik_state: SnapControlState
    start_pole_state: SnapControlState
    start_ik_influence: float
    start_terminal_ik_influence: float
    start_pole_angle: float
    calibrated_ik_state: SnapControlState
    calibrated_pole_state: SnapControlState
    calibrated_pole_angle: float
    last_issue: str | None = None


@dataclass(frozen=True, slots=True)
class FkJointMoveTarget:
    active_control: ResolvedControl
    resolution: SemanticMoveResolution | None
    domain_key: str
    kind: str


@dataclass(slots=True)
class FkSingleLinkMoveSession:
    active_control: ResolvedControl
    start_basis: Matrix
    start_matrix: Matrix
    start_head_world: Vector
    start_tail_world: Vector
    start_pivot_world: Vector
    center_pivot: bool = False
    visual_only: bool = False


@dataclass(slots=True)
class FkTwoBoneMoveSession:
    capability: LimbRepresentationCapability
    active_control: ResolvedControl
    first_control: ResolvedControl
    second_control: ResolvedControl
    terminal_control: ResolvedControl
    first_start_basis: Matrix
    second_start_basis: Matrix
    terminal_start_basis: Matrix
    first_start_matrix: Matrix
    second_start_matrix: Matrix
    terminal_start_matrix: Matrix
    root_world: Vector
    joint_world: Vector
    end_world: Vector
    baseline_perp_world: Vector
    bend_plane_normal_world: Vector
    last_direction_world: Vector
    last_bend_world: Vector
    first_length: float
    second_length: float


@dataclass(frozen=True, slots=True)
class SemanticMoveCommitResult:
    success: bool
    keyed: bool
    diagnostics: tuple[Any, ...] = ()


def _world_position(resolved: ResolvedControl) -> Vector:
    if isinstance(resolved.target, bpy.types.PoseBone):
        return (resolved.owner_object.matrix_world @ resolved.target.matrix).to_translation().copy()
    return resolved.target.matrix_world.to_translation().copy()


def _uses_center_pivot(resolved: ResolvedControl) -> bool:
    target = resolved.target
    return bool(
        isinstance(target, bpy.types.PoseBone)
        and str(target.name).upper() in {"COM", "PELVIS"}
    )


def _control_pivot_world(resolved: ResolvedControl) -> Vector:
    target = resolved.target
    if _uses_center_pivot(resolved) and isinstance(target, bpy.types.PoseBone):
        center_local = (Vector(target.head) + Vector(target.tail)) * 0.5
        return Vector(resolved.owner_object.matrix_world @ center_local)
    return _world_position(resolved)


def _world_to_control_location_delta_matrix(resolved: ResolvedControl) -> Matrix | None:
    target = resolved.target
    owner = resolved.owner_object
    if not isinstance(target, bpy.types.PoseBone):
        if owner.parent is None:
            return Matrix.Identity(3)
        parent_space = owner.parent.matrix_world @ owner.matrix_parent_inverse
        return parent_space.to_3x3().inverted_safe()
    if bool(target.bone.use_connect):
        return None

    base = owner.convert_space(
        pose_bone=target,
        matrix=Matrix.Identity(4),
        from_space="LOCAL",
        to_space="WORLD",
    ).translation
    columns: list[Vector] = []
    for axis in range(3):
        unit = Vector((0.0, 0.0, 0.0))
        unit[axis] = 1.0
        world = owner.convert_space(
            pose_bone=target,
            matrix=Matrix.Translation(unit),
            from_space="LOCAL",
            to_space="WORLD",
        ).translation
        columns.append(Vector(world) - Vector(base))
    local_to_world = Matrix(
        tuple(tuple(float(value) for value in column) for column in columns)
    ).transposed()
    if abs(float(local_to_world.determinant())) <= 1e-10:
        return None
    return local_to_world.inverted_safe()


def _resolved_control_role_name(resolved: ResolvedControl) -> str:
    role = getattr(resolved.control, "role", None)
    if role is not None:
        return str(getattr(role, "value", role))
    return str(getattr(resolved.target, "name", ""))


def _capture_control_state(resolved: ResolvedControl) -> SnapControlState:
    property_name = rotation_property(resolved.target)
    return SnapControlState(
        runtime_control_key(resolved),
        tuple(float(value) for value in resolved.target.location),
        property_name,
        tuple(float(value) for value in getattr(resolved.target, property_name)),
    )


def _apply_control_state(
    resolved: ResolvedControl,
    state: SnapControlState,
    *,
    location: bool = True,
    rotation: bool = True,
) -> None:
    if runtime_control_key(resolved) != state.runtime_key:
        raise RigpedSemanticMoveError("Rigped semantic Move control identity changed during the modal operation.")
    if location:
        resolved.target.location = state.location
    if rotation:
        if rotation_property(resolved.target) != state.rotation_property:
            raise RigpedSemanticMoveError("Rigped semantic Move rotation representation changed during the modal operation.")
        setattr(resolved.target, state.rotation_property, state.rotation)


_RIGPED_SLIDING_REPLAY_SYNC_ACTIVE = False
_RIGPED_JOINT_LIMIT_SYNC_ACTIVE = False
_RIGPED_SLIDING_REPLAY_LIMBS = (
    ("MCH_ForeArm.L", ("UpperArm.L", "ForeArm.L", "Hand.L"), ("MCH_UpperArm.L", "MCH_ForeArm.L", "MCH_Hand.L")),
    ("MCH_ForeArm.R", ("UpperArm.R", "ForeArm.R", "Hand.R"), ("MCH_UpperArm.R", "MCH_ForeArm.R", "MCH_Hand.R")),
    ("MCH_Calf.L", ("Thigh.L", "Calf.L", "Foot.L"), ("MCH_Thigh.L", "MCH_Calf.L", "MCH_Foot.L")),
    ("MCH_Calf.R", ("Thigh.R", "Calf.R", "Foot.R"), ("MCH_Thigh.R", "MCH_Calf.R", "MCH_Foot.R")),
)


def _apply_pose_bone_rotation_from_matrix(
    pose_bone,
    desired_matrix: Matrix,
    *,
    parent_pose_matrix: Matrix | None = None,
) -> None:
    """Show one solved result matrix on a generated public control without keying it."""

    basis = native_pose_basis_from_matrix(
        pose_bone,
        desired_matrix,
        parent_pose_matrix=parent_pose_matrix,
    )
    quaternion = basis.to_quaternion().normalized()
    if pose_bone.rotation_mode == "QUATERNION":
        pose_bone.rotation_quaternion = quaternion
    elif pose_bone.rotation_mode == "AXIS_ANGLE":
        axis = quaternion.axis
        pose_bone.rotation_axis_angle = (quaternion.angle, axis.x, axis.y, axis.z)
    else:
        pose_bone.rotation_euler = quaternion.to_euler(pose_bone.rotation_mode)


def _sync_generated_sliding_hinge_branch_from_public_pose(
    solver_owner,
    public_lower,
) -> bool:
    """Restore the generated hidden hinge branch from evaluated authored FK data."""

    name = str(getattr(solver_owner, "name", ""))
    if name in {"MCH_ForeArm.L", "MCH_ForeArm.R", "MCH_Calf.L", "MCH_Calf.R"}:
        quaternion = public_lower.matrix_basis.to_quaternion()
        angle = _generated_lower_hinge_x_angle(quaternion)
        fallback = -1 if name.startswith("MCH_ForeArm.") else 1
        branch_sign = (
            fallback
            if abs(angle) <= radians(0.01)
            else (1 if angle > 0.0 else -1)
        )
    else:
        return False
    return bool(configure_generated_rigped_ik_hinge_branch(solver_owner, branch_sign))


def _sync_generated_sliding_lower_roll_from_public_pose(
    solver_owner,
    public_lower,
) -> bool:
    """Restore keyed ForeArm/Calf axial roll as a native Sliding IK input."""

    if str(getattr(solver_owner, "name", "")) not in {
        "MCH_ForeArm.L",
        "MCH_ForeArm.R",
        "MCH_Calf.L",
        "MCH_Calf.R",
    }:
        return False
    if str(getattr(public_lower, "name", "")) not in {
        "ForeArm.L",
        "ForeArm.R",
        "Calf.L",
        "Calf.R",
    }:
        return False
    if solver_owner.rotation_mode != "QUATERNION":
        return False
    hinge_state = _hinge_state_from_basis(public_lower, public_lower.matrix_basis)
    if hinge_state is None:
        return False
    desired_long = float(hinge_state[1])
    current = solver_owner.rotation_quaternion.copy().normalized()
    current_long = _signed_twist_angle(current, _RIGPED_LOCAL_AXES["Y"])
    delta = ((desired_long - current_long + pi) % (2.0 * pi)) - pi
    if abs(delta) <= 1e-7:
        return False
    solver_owner.rotation_quaternion = (
        current @ Quaternion(_RIGPED_LOCAL_AXES["Y"], delta)
    ).normalized()
    return True


def _set_replay_limb_fk_feedback_muted(
    rig,
    pose,
    public_names: tuple[str, ...],
    result_names: tuple[str, ...],
    muted: bool,
) -> bool:
    changed = False
    for public_name, result_name in zip(public_names, result_names, strict=True):
        result_bone = pose.get(result_name)
        if result_bone is None:
            continue
        for constraint in result_bone.constraints:
            if (
                constraint.type != "COPY_ROTATION"
                or constraint.target is not rig
                or str(constraint.subtarget) != str(public_name)
            ):
                continue
            target = bool(muted)
            if bool(constraint.mute) == target:
                break
            constraint.mute = target
            changed = True
            break
    return changed


def _replay_capabilities_by_solver_name(scene, rig) -> dict[str, LimbRepresentationCapability]:
    """Resolve generated limb capabilities for one live Rigped replay owner."""

    setup = rig.get(RIGPED_SETUP_PROPERTY) if hasattr(rig, "get") else None
    getter = getattr(setup, "get", None)
    if not callable(getter):
        return {}
    character_id = str(getter("character_id", "") or "")
    if not character_id:
        return {}
    try:
        view = resolve_character(scene, character_id)
    except (KeyError, RuntimeError, TypeError, ValueError, ReferenceError):
        return {}

    capabilities: dict[str, LimbRepresentationCapability] = {}
    for mapping in view.definition.kinematics:
        capability = resolve_limb_representation_capability(
            view,
            mapping.mapping_id,
        ).capability
        if capability is None or capability.native_ik.solver_owner.owner_object is not rig:
            continue
        capabilities[str(capability.native_ik.solver_owner.target.name)] = capability
    return capabilities


@persistent
def _sync_rigped_sliding_replay_display(scene, _depsgraph=None) -> None:
    """Keep public Contact chains visually welded to their hidden IK solve.

    Sliding and Planted replay authority is the hidden IK target/pole plus
    Contact state. Public FK curves remain transition/release data, but while
    either IK-authoritative state is active they must not present a second
    independently interpolated limb to the user.
    """

    global _RIGPED_SLIDING_REPLAY_SYNC_ACTIVE
    if _RIGPED_SLIDING_REPLAY_SYNC_ACTIVE or scene is None:
        return

    _RIGPED_SLIDING_REPLAY_SYNC_ACTIVE = True
    touched = False
    try:
        for rig in getattr(scene, "objects", ()):
            if getattr(rig, "type", None) != "ARMATURE" or getattr(rig, "pose", None) is None:
                continue
            pose = rig.pose.bones
            capabilities_by_solver = _replay_capabilities_by_solver_name(scene, rig)
            for state_name, public_names, result_names in _RIGPED_SLIDING_REPLAY_LIMBS:
                state_bone = pose.get(state_name)
                if state_bone is None or AWB_CONTACT_STATE_PROPERTY not in state_bone:
                    continue
                try:
                    contact_type = type_for_state_value(
                        float(state_bone[AWB_CONTACT_STATE_PROPERTY])
                    )
                except (TypeError, ValueError):
                    continue

                branch_changed = False
                roll_changed = False
                if contact_type is ContactKeyType.SLIDING:
                    public_lower = pose.get(public_names[1])
                    if public_lower is not None:
                        branch_changed = _sync_generated_sliding_hinge_branch_from_public_pose(
                            state_bone,
                            public_lower,
                        )
                        roll_changed = _sync_generated_sliding_lower_roll_from_public_pose(
                            state_bone,
                            public_lower,
                        )
                feedback_changed = _set_replay_limb_fk_feedback_muted(
                    rig,
                    pose,
                    public_names,
                    result_names,
                    contact_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED},
                )
                view_layer = getattr(bpy.context, "view_layer", None)
                live_context = (
                    view_layer is not None
                    and getattr(bpy.context, "scene", None) is scene
                )
                if (branch_changed or roll_changed or feedback_changed) and live_context:
                    view_layer.update()

                # A reachable near-straight Planted leg can remain stuck at full
                # extension after its Root moves even though the held IK target is
                # now inside reach. Reuse the accepted Sliding singularity seed:
                # it perturbs only the hidden result pose, never Contact authority.
                seed_changed = False
                if contact_type is ContactKeyType.PLANTED and live_context:
                    capability = capabilities_by_solver.get(state_name)
                    if capability is not None:
                        seed_changed = _seed_stalled_sliding_native_ik(
                            bpy.context,
                            capability,
                        )
                touched = branch_changed or roll_changed or feedback_changed or seed_changed or touched
                if contact_type not in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}:
                    continue

                public = tuple(pose.get(name) for name in public_names)
                result = tuple(pose.get(name) for name in result_names)
                if any(item is None for item in (*public, *result)):
                    continue
                desired = tuple(item.matrix.copy() for item in result)
                _apply_pose_bone_rotation_from_matrix(public[0], desired[0])
                _apply_pose_bone_rotation_from_matrix(
                    public[1],
                    desired[1],
                    parent_pose_matrix=desired[0],
                )
                _apply_pose_bone_rotation_from_matrix(
                    public[2],
                    desired[2],
                    parent_pose_matrix=desired[1],
                )
                touched = True

        if touched:
            view_layer = getattr(bpy.context, "view_layer", None)
            if view_layer is not None and getattr(bpy.context, "scene", None) is scene:
                view_layer.update()
    finally:
        _RIGPED_SLIDING_REPLAY_SYNC_ACTIVE = False


@persistent
def _sync_rigped_joint_limits(scene, _depsgraph=None) -> None:
    """Project evaluated generated Rigped joints back into their legal envelopes."""

    global _RIGPED_JOINT_LIMIT_SYNC_ACTIVE
    if _RIGPED_JOINT_LIMIT_SYNC_ACTIVE or scene is None:
        return

    _RIGPED_JOINT_LIMIT_SYNC_ACTIVE = True
    touched = False
    try:
        for rig in getattr(scene, "objects", ()):
            if (
                getattr(rig, "type", None) != "ARMATURE"
                or getattr(rig, "pose", None) is None
                or not hasattr(rig, "get")
                or rig.get(RIGPED_SETUP_PROPERTY) is None
            ):
                continue
            touched = bool(apply_rigped_joint_limits_to_rig(rig)) or touched

        if touched:
            view_layer = getattr(bpy.context, "view_layer", None)
            if view_layer is not None and getattr(bpy.context, "scene", None) is scene:
                view_layer.update()
    finally:
        _RIGPED_JOINT_LIMIT_SYNC_ACTIVE = False


@persistent
def _sync_rigped_replay_load_post(*_args: object) -> None:
    """Reconcile derived Rigped replay display immediately after a .blend load."""

    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        scenes = tuple(getattr(bpy.data, "scenes", ()) or ())
        scene = scenes[0] if scenes else None
    if scene is None:
        return
    _sync_rigped_sliding_replay_display(scene)
    _sync_rigped_joint_limits(scene)


def _rigped_debug_state_probe(context) -> dict[str, Any] | None:
    """Return bounded Rigped semantic state for the generic debugger."""
    context = context or getattr(bpy, "context", None)
    active = getattr(context, "active_object", None) if context is not None else None
    if (
        active is None
        or getattr(active, "type", None) != "ARMATURE"
        or not hasattr(active, "get")
        or active.get(RIGPED_SETUP_PROPERTY) is None
    ):
        return None

    scene = getattr(context, "scene", None)
    try:
        selected = tuple(getattr(context, "selected_pose_bones", ()) or ())
    except (ReferenceError, RuntimeError):
        selected = ()
    bones: list[dict[str, Any]] = []
    for bone in sorted(selected, key=lambda item: str(getattr(item, "name", "") or ""))[:32]:
        row: dict[str, Any] = {"bone": str(getattr(bone, "name", "") or "")}
        try:
            if AWB_CONTACT_STATE_PROPERTY in bone:
                raw = float(bone[AWB_CONTACT_STATE_PROPERTY])
                row["contact_state_raw"] = raw
                row["contact_type"] = type_for_state_value(raw).value
        except (ReferenceError, RuntimeError, TypeError, ValueError):
            row["contact_state_unavailable"] = True
        bones.append(row)

    setup = active.get(RIGPED_SETUP_PROPERTY)
    getter = getattr(setup, "get", None)
    character_id = str(getter("character_id", "") or "") if callable(getter) else ""
    return {
        "rig_object": str(getattr(active, "name", "") or ""),
        "character_id": character_id or None,
        "semantic_transform_mode": (
            str(getattr(scene, "baw_rigped_semantic_transform_mode", "NONE") or "NONE")
            if scene is not None
            else "NONE"
        ),
        "selected_pose_bones": bones,
    }


def register_rigped_sliding_replay_handler() -> None:
    register_debug_state_probe("rigped", _rigped_debug_state_probe)
    frame_handlers = bpy.app.handlers.frame_change_post
    if _sync_rigped_sliding_replay_display not in frame_handlers:
        frame_handlers.append(_sync_rigped_sliding_replay_display)
    if _sync_rigped_joint_limits not in frame_handlers:
        frame_handlers.append(_sync_rigped_joint_limits)
    load_handlers = bpy.app.handlers.load_post
    if _sync_rigped_replay_load_post not in load_handlers:
        load_handlers.append(_sync_rigped_replay_load_post)

    # Do not run the public FK clamp from depsgraph_update_post. That handler
    # fires during modal Sliding/IK edits and can mutate the visible chain while
    # the semantic move operator is still solving it. Frame changes are the
    # playback boundary; live authoring operators enforce limits themselves.
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        _sync_rigped_sliding_replay_display(scene)
        _sync_rigped_joint_limits(scene)


def unregister_rigped_sliding_replay_handler() -> None:
    unregister_debug_state_probe("rigped")
    frame_handlers = bpy.app.handlers.frame_change_post
    if _sync_rigped_sliding_replay_display in frame_handlers:
        frame_handlers.remove(_sync_rigped_sliding_replay_display)
    if _sync_rigped_joint_limits in frame_handlers:
        frame_handlers.remove(_sync_rigped_joint_limits)
    load_handlers = bpy.app.handlers.load_post
    if _sync_rigped_replay_load_post in load_handlers:
        load_handlers.remove(_sync_rigped_replay_load_post)



def _state_for_pose_basis(
    resolved: ResolvedControl,
    basis: Matrix,
) -> SnapControlState:
    pose_bone = resolved.target
    if not isinstance(pose_bone, bpy.types.PoseBone):
        raise RigpedSemanticMoveError("Rigped pose reconstruction currently requires PoseBone controls.")

    property_name = rotation_property(pose_bone)
    quaternion = basis.to_quaternion().normalized()
    if property_name == "rotation_quaternion":
        rotation = (quaternion.w, quaternion.x, quaternion.y, quaternion.z)
    elif property_name == "rotation_axis_angle":
        axis = quaternion.axis
        rotation = (quaternion.angle, axis.x, axis.y, axis.z)
    else:
        rotation = tuple(float(value) for value in quaternion.to_euler(pose_bone.rotation_mode))
    return SnapControlState(
        runtime_control_key(resolved),
        tuple(float(value) for value in basis.to_translation()),
        property_name,
        tuple(float(value) for value in rotation),
    )


def _state_for_pose_matrix(
    resolved: ResolvedControl,
    desired_matrix: Matrix,
    *,
    parent_pose_matrix: Matrix | None = None,
) -> SnapControlState:
    pose_bone = resolved.target
    if not isinstance(pose_bone, bpy.types.PoseBone):
        raise RigpedSemanticMoveError("Rigped pose reconstruction currently requires PoseBone controls.")
    basis = native_pose_basis_from_matrix(
        pose_bone,
        desired_matrix,
        parent_pose_matrix=parent_pose_matrix,
    )
    return _state_for_pose_basis(resolved, basis)


def _pose_residual(actual: Matrix, expected: Matrix) -> tuple[float, float]:
    position = float((actual.to_translation() - expected.to_translation()).length)
    angle = float(
        actual.to_quaternion()
        .normalized()
        .rotation_difference(expected.to_quaternion().normalized())
        .angle
    )
    if angle > 3.141592653589793:
        angle = abs((2.0 * 3.141592653589793) - angle)
    return position, abs(angle)


def _rna_pointer(value) -> int:
    as_pointer = getattr(value, "as_pointer", None)
    return int(as_pointer()) if callable(as_pointer) else id(value)


def _semantic_move_selection_cache_key(context) -> tuple[Any, ...] | None:
    if getattr(context, "mode", "") != "POSE":
        return None
    scene = getattr(context, "scene", None)
    active_object = getattr(context, "active_object", None)
    if scene is None or active_object is None:
        return None
    selected_pose_bones = tuple(getattr(context, "selected_pose_bones", ()) or ())
    active_pose_bone = getattr(context, "active_pose_bone", None)
    store = getattr(scene, "awb_characters", None)
    character_revisions = tuple(
        (str(row.character_id), int(row.revision))
        for row in getattr(store, "characters", ())
    )
    return (
        _rna_pointer(scene),
        _rna_pointer(active_object),
        tuple(sorted(_rna_pointer(bone) for bone in selected_pose_bones)),
        _rna_pointer(active_pose_bone) if active_pose_bone is not None else 0,
        int(getattr(store, "schema_version", 0)) if store is not None else 0,
        character_revisions,
    )


def _mapping_may_contain_binding(view, mapping, binding_id: str) -> bool:
    if binding_id in {mapping.ik_target_binding_id, mapping.pole_binding_id}:
        return True
    chain_by_id = {chain.chain_id: chain for chain in view.definition.chains}
    fk_chain = chain_by_id.get(mapping.fk_chain_id or "")
    if fk_chain is not None and binding_id in fk_chain.members:
        return True

    binding_by_id = {binding.binding_id: binding for binding in view.definition.bindings}
    selected_binding = binding_by_id.get(binding_id)
    ik_binding = binding_by_id.get(mapping.ik_target_binding_id or "")
    if selected_binding is None or ik_binding is None:
        return False
    return (
        selected_binding.semantic_key == ik_binding.semantic_key
        and selected_binding.side == ik_binding.side
        and selected_binding.usage.value == "PRIMARY"
        and selected_binding.kind.value == "BONE"
    )


def _active_control_for_capability(
    capability: LimbRepresentationCapability,
    mapping,
    binding_id: str,
) -> ResolvedControl | None:
    for candidate_id, control in zip(
        capability.fk_binding_ids,
        capability.fk_controls,
        strict=True,
    ):
        if candidate_id == binding_id:
            return control
    if capability.authored_terminal_binding_id == binding_id:
        return capability.authored_terminal
    if mapping.ik_target_binding_id == binding_id:
        return capability.native_ik.ik_target
    if mapping.pole_binding_id == binding_id:
        return capability.native_ik.pole_target
    return None


def _resolve_semantic_move_binding(
    context,
    target,
    binding_id: str,
) -> SemanticMoveSelection | None:
    """Resolve one selected binding without requiring the whole selection to be singular."""

    view = resolve_character(context.scene, target.character_id)
    candidates: list[SemanticMoveSelection] = []
    for mapping in view.definition.kinematics:
        if not _mapping_may_contain_binding(view, mapping, binding_id):
            continue
        representation = resolve_limb_representation_capability(view, mapping.mapping_id)
        capability = representation.capability
        if capability is None:
            continue
        active_control = _active_control_for_capability(capability, mapping, binding_id)
        if active_control is not None:
            candidates.append(
                SemanticMoveSelection(
                    target.character_id,
                    binding_id,
                    capability,
                    active_control,
                )
            )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _resolve_semantic_move_selection(context) -> SemanticMoveSelection | None:
    control_context = control_context_for_context(context)
    resolution = resolve_rigped_target(context.scene, control_context)
    target = resolution.target
    if target is None or len(target.selected_binding_ids) != 1:
        return None
    decision = resolve_rigped_transform(target, RigpedTransformGesture.MOVE)
    if decision.route not in {
        RigpedTransformRoute.SEMANTIC_KINEMATIC,
        RigpedTransformRoute.SEMANTIC_CONTACT,
    }:
        return None
    return _resolve_semantic_move_binding(context, target, target.selected_binding_ids[0])


def _selected_semantic_selection(context) -> SemanticMoveSelection | None:
    global _SEMANTIC_MOVE_SELECTION_CACHE_KEY, _SEMANTIC_MOVE_SELECTION_CACHE_VALUE

    cache_key = _semantic_move_selection_cache_key(context)
    if cache_key is None:
        return None
    if cache_key == _SEMANTIC_MOVE_SELECTION_CACHE_KEY:
        return _SEMANTIC_MOVE_SELECTION_CACHE_VALUE
    resolved = _resolve_semantic_move_selection(context)
    _SEMANTIC_MOVE_SELECTION_CACHE_KEY = cache_key
    _SEMANTIC_MOVE_SELECTION_CACHE_VALUE = resolved
    return resolved


def _resolve_direct_transform_selection(context) -> DirectTransformSelection | None:
    control_context = control_context_for_context(context)
    # Native Pose transforms are allowed to operate on multiple selected Rigped
    # controls. The active control owns gizmo pivot/orientation, matching Blender,
    # while resolve_rigped_transform verifies that the whole selection is safe for
    # the requested native gesture.
    if not control_context.controls or control_context.active is None:
        return None
    resolution = resolve_rigped_target(context.scene, control_context)
    target = resolution.target
    if target is None or not target.selected_binding_ids:
        return None
    move_decision = resolve_rigped_transform(target, RigpedTransformGesture.MOVE)
    rotate_decision = resolve_rigped_transform(target, RigpedTransformGesture.ROTATE)
    return DirectTransformSelection(
        active_control=control_context.active,
        move_native=move_decision.route is RigpedTransformRoute.NATIVE,
        rotate_native=rotate_decision.route is RigpedTransformRoute.NATIVE,
    )


def _selected_direct_transform(context) -> DirectTransformSelection | None:
    global _DIRECT_TRANSFORM_SELECTION_CACHE_KEY, _DIRECT_TRANSFORM_SELECTION_CACHE_VALUE

    cache_key = _semantic_move_selection_cache_key(context)
    if cache_key is None:
        return None
    if cache_key == _DIRECT_TRANSFORM_SELECTION_CACHE_KEY:
        return _DIRECT_TRANSFORM_SELECTION_CACHE_VALUE
    resolved = _resolve_direct_transform_selection(context)
    _DIRECT_TRANSFORM_SELECTION_CACHE_KEY = cache_key
    _DIRECT_TRANSFORM_SELECTION_CACHE_VALUE = resolved
    return resolved


def _orientation_axes_for_control(context, active: ResolvedControl) -> dict[str, Vector] | None:
    orientation = str(context.scene.transform_orientation_slots[0].type)
    if orientation == "GLOBAL":
        basis = Matrix.Identity(3)
    elif orientation == "LOCAL":
        if isinstance(active.target, bpy.types.PoseBone):
            basis = (active.owner_object.matrix_world @ active.target.matrix).to_3x3().normalized()
        else:
            basis = active.target.matrix_world.to_3x3().normalized()
    elif orientation == "VIEW":
        region_data = getattr(context, "region_data", None)
        if region_data is None:
            region_data = getattr(getattr(context, "space_data", None), "region_3d", None)
        if region_data is None:
            return None
        basis = region_data.view_matrix.inverted().to_3x3().normalized()
    else:
        return None
    return {
        "X": Vector(basis.col[0]).normalized(),
        "Y": Vector(basis.col[1]).normalized(),
        "Z": Vector(basis.col[2]).normalized(),
    }


def direct_transform_pivot(context) -> Vector | None:
    selected = _selected_direct_transform(context)
    if selected is None:
        return None
    return _control_pivot_world(selected.active_control)


def direct_transform_axes(context) -> dict[str, Vector] | None:
    selected = _selected_direct_transform(context)
    if selected is None:
        return None
    return _orientation_axes_for_control(context, selected.active_control)


def direct_move_available(context) -> bool:
    selected = _selected_direct_transform(context)
    return selected is not None and selected.move_native


def direct_rotate_available(context) -> bool:
    selected = _selected_direct_transform(context)
    return selected is not None and selected.rotate_native


def _semantic_move_resolution_for_selection(
    selected: SemanticMoveSelection | None,
) -> SemanticMoveResolution | None:
    if selected is None:
        return None
    capability = selected.capability
    solver = capability.native_ik.solver_owner.target
    if AWB_CONTACT_STATE_PROPERTY not in solver:
        return None
    state_type = type_for_state_value(float(solver[AWB_CONTACT_STATE_PROPERTY]))
    if state_type is None and abs(float(capability.native_ik.constraint.influence)) <= 1e-8:
        state_type = ContactKeyType.FREE
    if state_type not in {ContactKeyType.FREE, ContactKeyType.SLIDING, ContactKeyType.PLANTED}:
        return None
    return SemanticMoveResolution(
        selected.character_id,
        selected.binding_id,
        state_type,
        capability,
        selected.active_control,
    )


def _selected_semantic_mapping(context) -> SemanticMoveResolution | None:
    return _semantic_move_resolution_for_selection(_selected_semantic_selection(context))


def _selected_rotational_move_control(context) -> ResolvedControl | None:
    global _CLAVICLE_SELECTION_CACHE_KEY, _CLAVICLE_SELECTION_CACHE_VALUE

    cache_key = _semantic_move_selection_cache_key(context)
    if cache_key is None:
        return None
    if cache_key == _CLAVICLE_SELECTION_CACHE_KEY:
        return _CLAVICLE_SELECTION_CACHE_VALUE

    control_context = control_context_for_context(context)
    result = None
    if len(control_context.controls) == 1 and control_context.active is not None:
        resolution = resolve_rigped_target(context.scene, control_context)
        target = resolution.target
        if target is not None and len(target.selected_binding_ids) == 1:
            binding_id = target.selected_binding_ids[0]
            view = resolve_character(context.scene, target.character_id)
            binding = next(
                (item for item in view.definition.bindings if item.binding_id == binding_id),
                None,
            )
            if binding is not None and binding.semantic_key in {
                "awb.clavicle",
                "awb.spine",
                "awb.neck",
                "awb.head",
            }:
                result = control_context.active

    _CLAVICLE_SELECTION_CACHE_KEY = cache_key
    _CLAVICLE_SELECTION_CACHE_VALUE = result
    return result


def _fk_joint_move_kind(resolved: SemanticMoveResolution | None) -> str | None:
    if resolved is None or resolved.contact_type is not ContactKeyType.FREE:
        return None
    active_key = runtime_control_key(resolved.active_control)
    for index, control in enumerate(resolved.capability.fk_controls):
        if runtime_control_key(control) == active_key:
            return "FIRST" if index == 0 else "CHAIN_END"
    if runtime_control_key(resolved.capability.authored_terminal) == active_key:
        return "CHAIN_END"
    return None


def _fk_joint_move_kind_for_selection(selected: SemanticMoveSelection | None) -> str | None:
    """Return the static FK role without reading animated Contact state."""

    if selected is None:
        return None
    active_key = runtime_control_key(selected.active_control)
    for index, control in enumerate(selected.capability.fk_controls):
        if runtime_control_key(control) == active_key:
            return "FIRST" if index == 0 else "CHAIN_END"
    if runtime_control_key(selected.capability.authored_terminal) == active_key:
        return "CHAIN_END"
    return None


def _resolve_fk_joint_move_candidates(context) -> tuple[FkJointMoveCandidate, ...]:
    """Resolve selection routing once; no view or pose-matrix data is cached here."""

    control_context = control_context_for_context(context)
    resolution = resolve_rigped_target(context.scene, control_context)
    target = resolution.target
    if target is None or not target.selected_binding_ids or control_context.active is None:
        return ()

    contracts = {contract.binding_id: contract for contract in target.controls}
    active_binding_id = target.active_binding_id
    rotational_semantics = {"awb.pelvis", "awb.clavicle", "awb.spine", "awb.neck", "awb.head"}
    singles: list[FkJointMoveCandidate] = []
    limb_groups: dict[str, list[FkJointMoveCandidate]] = {}

    for binding_id in target.selected_binding_ids:
        contract = contracts.get(binding_id)
        if contract is None:
            return ()
        if contract.semantic_key in rotational_semantics:
            singles.append(
                FkJointMoveCandidate(
                    contract.target,
                    None,
                    f"single:{binding_id}",
                    "SINGLE",
                )
            )
            continue

        selected = _resolve_semantic_move_binding(context, target, binding_id)
        kind = _fk_joint_move_kind_for_selection(selected)
        if selected is None or kind is None:
            return ()
        mapping_id = str(selected.capability.native_ik.mapping_id)
        limb_groups.setdefault(mapping_id, []).append(
            FkJointMoveCandidate(
                selected.active_control,
                selected,
                f"limb:{mapping_id}",
                kind,
            )
        )

    chosen: list[FkJointMoveCandidate] = list(singles)
    for candidates in limb_groups.values():
        active_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.semantic_selection is not None
                and candidate.semantic_selection.binding_id == active_binding_id
            ),
            None,
        )
        if active_candidate is not None:
            chosen.append(active_candidate)
            continue
        chosen.append(
            next(
                (candidate for candidate in candidates if candidate.kind == "CHAIN_END"),
                candidates[0],
            )
        )

    active_key = runtime_control_key(control_context.active)
    if not any(runtime_control_key(item.active_control) == active_key for item in chosen):
        return ()
    chosen.sort(
        key=lambda item: 0 if runtime_control_key(item.active_control) == active_key else 1
    )
    return tuple(chosen)


def _selected_fk_joint_move_candidates(context) -> tuple[FkJointMoveCandidate, ...]:
    """Cache invariant selection routing across gizmo poll/draw redraws.

    The key changes when selected/active PoseBones or Rigped character revisions
    change. Contact state deliberately stays out of this cache because it may be
    animated or changed at the current frame; that small dynamic check happens
    in _selected_fk_joint_move_targets instead.
    """

    global _FK_JOINT_MOVE_CANDIDATE_CACHE_KEY, _FK_JOINT_MOVE_CANDIDATE_CACHE_VALUE

    cache_key = _semantic_move_selection_cache_key(context)
    if cache_key is None:
        return ()
    if cache_key == _FK_JOINT_MOVE_CANDIDATE_CACHE_KEY:
        return _FK_JOINT_MOVE_CANDIDATE_CACHE_VALUE
    resolved = _resolve_fk_joint_move_candidates(context)
    _FK_JOINT_MOVE_CANDIDATE_CACHE_KEY = cache_key
    _FK_JOINT_MOVE_CANDIDATE_CACHE_VALUE = resolved
    return resolved


def _selected_fk_joint_move_targets(context) -> tuple[FkJointMoveTarget, ...]:
    """Materialize current FK targets from cached static routing plus live Contact state."""

    candidates = _selected_fk_joint_move_candidates(context)
    if not candidates:
        return ()
    targets: list[FkJointMoveTarget] = []
    for candidate in candidates:
        if candidate.semantic_selection is None:
            targets.append(
                FkJointMoveTarget(
                    candidate.active_control,
                    None,
                    candidate.domain_key,
                    candidate.kind,
                )
            )
            continue
        resolved = _semantic_move_resolution_for_selection(candidate.semantic_selection)
        if resolved is None or resolved.contact_type is not ContactKeyType.FREE:
            return ()
        targets.append(
            FkJointMoveTarget(
                candidate.active_control,
                resolved,
                candidate.domain_key,
                candidate.kind,
            )
        )
    return tuple(targets)


def _fk_two_bone_is_leg(capability: LimbRepresentationCapability) -> bool:
    terminal = capability.authored_terminal.target
    return bool(
        isinstance(terminal, bpy.types.PoseBone)
        and str(terminal.name).upper().startswith("FOOT.")
    )


def _fk_two_bone_is_arm(capability: LimbRepresentationCapability) -> bool:
    terminal = capability.authored_terminal.target
    return bool(
        isinstance(terminal, bpy.types.PoseBone)
        and str(terminal.name).upper().startswith("HAND.")
    )


def _fk_joint_move_active_control(context) -> ResolvedControl | None:
    targets = _selected_fk_joint_move_targets(context)
    return targets[0].active_control if targets else None


def fk_joint_move_available(context) -> bool:
    return _fk_joint_move_active_control(context) is not None


def fk_joint_move_pivot(context) -> Vector | None:
    # Kinematic solve targets (elbow/wrist/knee/ankle) are internal operation
    # points and must not move the visible gizmo. COM/Pelvis are body-center
    # controls, so their animator-facing pivot is the displayed link center.
    active = _fk_joint_move_active_control(context)
    return _control_pivot_world(active) if active is not None else None


def fk_joint_move_axes(context) -> dict[str, Vector] | None:
    active = _fk_joint_move_active_control(context)
    return _orientation_axes_for_control(context, active) if active is not None else None


def _begin_fk_single_link_move_for_control(active: ResolvedControl) -> FkSingleLinkMoveSession:
    pose_bone = active.target
    if not isinstance(pose_bone, bpy.types.PoseBone):
        raise RigpedSemanticMoveError("FK joint Move requires a PoseBone control.")
    head_world = Vector(active.owner_object.matrix_world @ pose_bone.head)
    tail_world = Vector(active.owner_object.matrix_world @ pose_bone.tail)
    return FkSingleLinkMoveSession(
        active_control=active,
        start_basis=pose_bone.matrix_basis.copy(),
        start_matrix=pose_bone.matrix.copy(),
        start_head_world=head_world,
        start_tail_world=tail_world,
        start_pivot_world=_control_pivot_world(active),
        center_pivot=_uses_center_pivot(active),
        # Pelvis W is intentionally display-only. Biped-style Pelvis posing is
        # Rotate-driven; keeping the Move gizmo visible preserves AWB's tool
        # continuity without inventing a Move->Rotate gesture.
        visual_only=str(pose_bone.name).upper() == "PELVIS",
    )


def begin_fk_single_link_move(context) -> FkSingleLinkMoveSession:
    targets = _selected_fk_joint_move_targets(context)
    if len(targets) != 1 or targets[0].kind not in {"SINGLE", "FIRST"}:
        raise RigpedSemanticMoveError(
            "Current selection is not one supported single-link FK Move control."
        )
    return _begin_fk_single_link_move_for_control(targets[0].active_control)


def _aligned_pose_matrix(
    start_matrix: Matrix,
    start_head: Vector,
    start_tail: Vector,
    desired_head: Vector,
    desired_tail: Vector,
) -> Matrix | None:
    start_direction = Vector(start_tail) - Vector(start_head)
    desired_direction = Vector(desired_tail) - Vector(desired_head)
    if start_direction.length <= 1e-9 or desired_direction.length <= 1e-9:
        return None
    rotation = (
        start_direction.normalized()
        .rotation_difference(desired_direction.normalized())
        .to_matrix()
        .to_4x4()
    )
    return (
        Matrix.Translation(desired_head)
        @ rotation
        @ Matrix.Translation(-Vector(start_head))
        @ start_matrix
    )


def _plane_aligned_pose_matrix(
    start_matrix: Matrix,
    start_head: Vector,
    start_tail: Vector,
    start_plane_normal: Vector,
    desired_head: Vector,
    desired_tail: Vector,
    desired_plane_normal: Vector,
) -> Matrix | None:
    """Build one history-independent link frame from direction + solver plane.

    The bone local Y axis follows the segment direction. The remaining roll is
    fixed by the limb solver-plane normal, which acts like Biped/HI IK swivel:
    identical goal geometry and swivel reference always reconstruct the same
    orientation, regardless of the cursor path taken to reach it.
    """

    start_y = Vector(start_tail) - Vector(start_head)
    desired_y = Vector(desired_tail) - Vector(desired_head)
    if start_y.length <= 1e-9 or desired_y.length <= 1e-9:
        return None
    start_y.normalize()
    desired_y.normalize()

    start_z = Vector(start_plane_normal)
    desired_z = Vector(desired_plane_normal)
    start_z -= start_y * float(start_z.dot(start_y))
    desired_z -= desired_y * float(desired_z.dot(desired_y))
    if start_z.length <= 1e-8 or desired_z.length <= 1e-8:
        return None
    start_z.normalize()
    desired_z.normalize()

    start_x = start_y.cross(start_z)
    desired_x = desired_y.cross(desired_z)
    if start_x.length <= 1e-8 or desired_x.length <= 1e-8:
        return None
    start_x.normalize()
    desired_x.normalize()
    start_z = start_x.cross(start_y).normalized()
    desired_z = desired_x.cross(desired_y).normalized()

    start_frame = Matrix((start_x, start_y, start_z)).transposed()
    desired_frame = Matrix((desired_x, desired_y, desired_z)).transposed()
    rotation = (desired_frame @ start_frame.transposed()).to_4x4()
    return (
        Matrix.Translation(desired_head)
        @ rotation
        @ Matrix.Translation(-Vector(start_head))
        @ start_matrix
    )


def apply_fk_single_link_move_delta(
    context,
    session: FkSingleLinkMoveSession,
    world_delta: Vector,
) -> bool:
    active = session.active_control
    pose_bone = active.target
    if not isinstance(pose_bone, bpy.types.PoseBone):
        return False
    if session.visual_only:
        return True
    owner_inverse = active.owner_object.matrix_world.inverted_safe()
    start_head = Vector(owner_inverse @ session.start_head_world)
    start_tail = Vector(owner_inverse @ session.start_tail_world)
    desired_tail = Vector(owner_inverse @ (session.start_tail_world + Vector(world_delta)))

    if session.center_pivot:
        pivot = Vector(owner_inverse @ session.start_pivot_world)
        baseline = start_tail - pivot
        desired = desired_tail - pivot
        if baseline.length <= 1e-9 or desired.length <= 1e-9:
            return False
        rotation = (
            baseline.normalized()
            .rotation_difference(desired.normalized())
            .to_matrix()
            .to_4x4()
        )
        desired_matrix = (
            Matrix.Translation(pivot)
            @ rotation
            @ Matrix.Translation(-pivot)
            @ session.start_matrix
        )
        desired_state = _state_for_pose_matrix(active, desired_matrix)
        _apply_control_state(active, desired_state, location=True, rotation=True)
        return True

    # A fixed-length FK link must never let its dragged endpoint cross through
    # the parent pivot. Near that singularity rotation_difference has no stable
    # axis and the clavicle/first limb link can suddenly flip by ~180 degrees.
    start_vector = start_tail - start_head
    if start_vector.length <= 1e-9:
        return False
    start_axis = start_vector.normalized()
    desired_vector = desired_tail - start_head
    minimum_forward = max(1e-6, float(start_vector.length) * 0.05)
    forward = float(desired_vector.dot(start_axis))
    if forward < minimum_forward:
        desired_tail += start_axis * (minimum_forward - forward)

    desired_matrix = _aligned_pose_matrix(
        session.start_matrix,
        start_head,
        start_tail,
        start_head,
        desired_tail,
    )
    if desired_matrix is None:
        return False
    desired_state = _state_for_pose_matrix(active, desired_matrix)
    _apply_control_state(active, desired_state, location=False, rotation=True)
    # Rotation properties are written from the mouse-down baseline instead of
    # assigning PoseBone.matrix against a changing evaluated parent. This keeps
    # tiny W drags monotonic without forcing a depsgraph update per pixel.
    return True


def cancel_fk_single_link_move(context, session: FkSingleLinkMoveSession) -> None:
    active = session.active_control
    if isinstance(active.target, bpy.types.PoseBone):
        active.target.matrix_basis = session.start_basis.copy()
        context.view_layer.update()


def _begin_fk_two_bone_move_for_resolution(
    resolved: SemanticMoveResolution,
    *,
    allow_sliding_lower_swivel: bool = False,
) -> FkTwoBoneMoveSession:
    sliding_lower_swivel = (
        allow_sliding_lower_swivel
        and resolved.contact_type is ContactKeyType.SLIDING
        and len(resolved.capability.fk_controls) == 2
        and str(resolved.active_control.target.name)
        in {"ForeArm.L", "ForeArm.R", "Calf.L", "Calf.R"}
        and runtime_control_key(resolved.active_control)
        == runtime_control_key(resolved.capability.fk_controls[1])
    )
    if _fk_joint_move_kind(resolved) != "CHAIN_END" and not sliding_lower_swivel:
        raise RigpedSemanticMoveError("Current selection is not a supported two-bone FK Move control.")
    capability = resolved.capability
    if len(capability.fk_controls) != 2:
        raise RigpedSemanticMoveError("FK joint Move requires exactly two FK links.")
    first, second = capability.fk_controls
    terminal = capability.authored_terminal
    if not all(
        isinstance(control.target, bpy.types.PoseBone)
        for control in (first, second, terminal)
    ):
        raise RigpedSemanticMoveError("FK joint Move requires PoseBone controls.")
    owner = first.owner_object
    root = Vector(owner.matrix_world @ first.target.head)
    joint = Vector(owner.matrix_world @ first.target.tail)
    end = Vector(owner.matrix_world @ second.target.tail)
    first_length = float((joint - root).length)
    second_length = float((end - joint).length)
    if first_length <= 1e-8 or second_length <= 1e-8:
        raise RigpedSemanticMoveError("FK joint Move found a zero-length limb link.")

    # Freeze the bend plane once at mouse-down, matching the accepted Fit
    # solver. Re-deriving a pole direction from the moving target every mouse
    # event is discontinuous near a straight chain and causes elbow/knee flips.
    baseline_target = end - root
    baseline_distance = float(baseline_target.length)
    if baseline_distance <= 1e-9:
        raise RigpedSemanticMoveError("FK joint Move found a collapsed limb chain.")
    baseline_dir = baseline_target.normalized()
    baseline_along = (
        first_length * first_length
        - second_length * second_length
        + baseline_distance * baseline_distance
    ) / (2.0 * baseline_distance)
    baseline_perp = joint - (root + baseline_dir * baseline_along)
    owner_world3 = owner.matrix_world.to_3x3()
    is_leg = _fk_two_bone_is_leg(capability)
    is_arm = _fk_two_bone_is_arm(capability)

    if is_leg:
        straight_seed = Vector(owner_world3 @ Vector((0.0, -1.0, 0.0)))
    elif is_arm and capability.native_ik.pole_target is not None:
        pole_world = _world_position(capability.native_ik.pole_target)
        straight_seed = pole_world - (
            root + baseline_dir * float((pole_world - root).dot(baseline_dir))
        )
    else:
        straight_seed = Vector(owner_world3 @ first.target.z_axis)

    fallback = Vector(owner_world3 @ Vector((1.0, 0.0, 0.0)))
    if fallback.length > 1e-9 and abs(float(fallback.normalized().dot(baseline_dir))) > 0.9:
        fallback = Vector(owner_world3 @ Vector((0.0, 0.0, 1.0)))
    resolved_bend = preferred_two_bone_bend(
        baseline_perp,
        baseline_dir,
        first_length,
        second_length,
        is_arm=is_arm,
        straight_seed=straight_seed,
        fallback=fallback,
    )
    if resolved_bend is None:
        raise RigpedSemanticMoveError("FK joint Move could not establish a stable bend plane.")
    baseline_perp = resolved_bend
    bend_plane_normal = baseline_dir.cross(baseline_perp)
    if bend_plane_normal.length <= 1e-7:
        raise RigpedSemanticMoveError("FK joint Move bend plane is singular.")
    bend_plane_normal.normalize()

    return FkTwoBoneMoveSession(
        capability=capability,
        active_control=resolved.active_control,
        first_control=first,
        second_control=second,
        terminal_control=terminal,
        first_start_basis=first.target.matrix_basis.copy(),
        second_start_basis=second.target.matrix_basis.copy(),
        terminal_start_basis=terminal.target.matrix_basis.copy(),
        first_start_matrix=first.target.matrix.copy(),
        second_start_matrix=second.target.matrix.copy(),
        terminal_start_matrix=terminal.target.matrix.copy(),
        root_world=root,
        joint_world=joint,
        end_world=end,
        baseline_perp_world=baseline_perp.copy(),
        bend_plane_normal_world=bend_plane_normal.copy(),
        last_direction_world=baseline_dir.copy(),
        last_bend_world=baseline_perp.copy(),
        first_length=first_length,
        second_length=second_length,
    )


def begin_fk_two_bone_move(context) -> FkTwoBoneMoveSession:
    targets = _selected_fk_joint_move_targets(context)
    if len(targets) != 1 or targets[0].kind != "CHAIN_END" or targets[0].resolution is None:
        raise RigpedSemanticMoveError("Current selection is not one supported two-bone FK Move control.")
    return _begin_fk_two_bone_move_for_resolution(targets[0].resolution)


def _solve_two_bone_fk_points(
    session: FkTwoBoneMoveSession,
    desired_end_world: Vector,
    *,
    allow_full_extension: bool = False,
    minimum_bend_radians: float = 0.0,
) -> tuple[Vector, Vector] | None:
    root = Vector(session.root_world)
    requested = Vector(desired_end_world)
    direction = requested - root
    distance = float(direction.length)
    if distance <= 1e-9:
        direction = Vector(session.end_world) - root
        distance = float(direction.length)
    if distance <= 1e-9:
        return None
    direction.normalize()

    first_length = float(session.first_length)
    second_length = float(session.second_length)
    # Exact 180-degree extension is a two-bone singularity: elbow/knee side is
    # undefined even with a pole target. Biped-style solvers retain a preferred
    # bend direction, so Sliding can request a tiny explicit bend reserve while
    # remaining visually near full reach. This keeps Blender's native IK away
    # from the branch-loss boundary without introducing stretch.
    reach_epsilon = max(1e-5, (first_length + second_length) * 0.001)
    minimum = abs(first_length - second_length) + reach_epsilon
    preferred_bend = max(0.0, float(minimum_bend_radians))
    if preferred_bend > 0.0:
        preferred_bend = min(preferred_bend, 3.141592653589793 - 1e-6)
        preferred_maximum = sqrt(
            max(
                0.0,
                first_length * first_length
                + second_length * second_length
                + 2.0 * first_length * second_length * cos(preferred_bend),
            )
        )
        maximum = max(minimum, min(first_length + second_length, preferred_maximum))
    else:
        maximum = max(
            minimum,
            first_length + second_length
            if allow_full_extension
            else first_length + second_length - reach_epsilon,
        )
    solved_distance = max(minimum, min(maximum, distance))
    end = root + direction * solved_distance

    along = (
        first_length * first_length
        - second_length * second_length
        + solved_distance * solved_distance
    ) / (2.0 * solved_distance)
    height_sq = max(0.0, first_length * first_length - along * along)
    height = sqrt(height_sq)

    bend = Vector(session.bend_plane_normal_world).cross(direction)

    # Preserve the accepted continuous branch behavior for both arms and legs.
    # The arm's human-friendly preference is chosen once at gesture start when
    # the chain is nearly straight; it never swivels as a function of hand height.
    # Transport the previous solved bend only to choose the sign of the two
    # mathematical bend solutions; final orientation remains history-independent.
    previous_direction = Vector(session.last_direction_world)
    previous_bend = Vector(session.last_bend_world)
    continuity_bend = previous_bend.copy()
    if previous_direction.length > 1e-7 and previous_bend.length > 1e-7:
        previous_direction.normalize()
        transport = previous_direction.rotation_difference(direction)
        continuity_bend = Vector(transport @ previous_bend)
    continuity_bend -= direction * float(continuity_bend.dot(direction))

    reference_bend = Vector(session.baseline_perp_world)
    reference_bend -= direction * float(reference_bend.dot(direction))
    if bend.length <= 1e-7:
        bend = continuity_bend.copy()
    if bend.length <= 1e-7:
        bend = reference_bend.copy()
    if bend.length <= 1e-7:
        return None
    bend.normalize()

    sign_reference = continuity_bend if continuity_bend.length > 1e-7 else reference_bend
    if sign_reference.length > 1e-7 and float(bend.dot(sign_reference)) < 0.0:
        bend.negate()

    session.last_direction_world = direction.copy()
    session.last_bend_world = bend.copy()
    joint = root + direction * along + bend * height
    return joint, end


def _apply_solved_two_bone_fk_pose(
    session: FkTwoBoneMoveSession,
    desired_joint_world: Vector,
    desired_end_world: Vector,
    *,
    terminal_follows_second: bool = False,
) -> bool:
    """Apply one already-solved two-bone pose to a two-bone control chain.

    Free FK gestures apply this to the public controls. Sliding singularity
    recovery applies the same solved pose transiently to the hidden MCH result
    controls while native IK is muted, then hands final authority back to
    Blender's native IK solver.
    """

    owner = session.first_control.owner_object
    owner_inverse = owner.matrix_world.inverted_safe()
    owner_basis_inverse = owner.matrix_world.to_3x3().inverted_safe()
    root = Vector(owner_inverse @ session.root_world)
    start_joint = Vector(owner_inverse @ session.joint_world)
    start_end = Vector(owner_inverse @ session.end_world)
    desired_joint = Vector(owner_inverse @ desired_joint_world)
    desired_end = Vector(owner_inverse @ desired_end_world)

    # Biped-style requirement: the solve must be history-independent. Use the
    # current solved limb plane as the absolute roll/swivel reference rather than
    # accumulating orientation from the previous mouse event. A closed cursor
    # path that returns the hand/foot to the same goal therefore reconstructs the
    # same ForeArm/Calf orientation instead of retaining geometric holonomy/twist.
    start_plane = Vector(owner_basis_inverse @ session.bend_plane_normal_world)
    desired_plane_world = (
        Vector(desired_end_world) - Vector(session.root_world)
    ).cross(Vector(desired_joint_world) - Vector(session.root_world))
    if desired_plane_world.length <= 1e-8:
        desired_plane_world = Vector(session.bend_plane_normal_world)
    desired_plane = Vector(owner_basis_inverse @ desired_plane_world)

    first_matrix = _plane_aligned_pose_matrix(
        session.first_start_matrix,
        root,
        start_joint,
        start_plane,
        root,
        desired_joint,
        desired_plane,
    )
    second_matrix = _plane_aligned_pose_matrix(
        session.second_start_matrix,
        start_joint,
        start_end,
        start_plane,
        desired_joint,
        desired_end,
        desired_plane,
    )
    if first_matrix is None or second_matrix is None:
        return False

    # Ordinary FK Move keeps terminal world orientation independent while the
    # endpoint is repositioned. ForeArm swivel is the narrow exception: wrist
    # position stays fixed, but Hand orientation follows the ForeArm by keeping
    # the mouse-down terminal transform local to the second link.
    if terminal_follows_second:
        terminal_relative = (
            session.second_start_matrix.inverted_safe()
            @ session.terminal_start_matrix
        )
        terminal_matrix = second_matrix @ terminal_relative
    else:
        terminal_matrix = session.terminal_start_matrix.copy()
    terminal_matrix.translation = desired_end

    # Reconstruct rotation channels from the absolute desired matrices. Direct
    # PoseBone.matrix assignment makes parent/child pose evaluation participate in
    # every MOUSEMOVE and can visibly tremble on 1-2 px input, so write rotation
    # properties against the frozen desired parent matrices instead.
    first_state = _state_for_pose_matrix(session.first_control, first_matrix)
    second_state = _state_for_pose_matrix(
        session.second_control,
        second_matrix,
        parent_pose_matrix=first_matrix,
    )
    terminal_state = _state_for_pose_matrix(
        session.terminal_control,
        terminal_matrix,
        parent_pose_matrix=second_matrix,
    )
    _apply_control_state(session.first_control, first_state, location=False, rotation=True)
    _apply_control_state(session.second_control, second_state, location=False, rotation=True)
    _apply_control_state(session.terminal_control, terminal_state, location=False, rotation=True)
    return True


def apply_fk_two_bone_move_delta(
    context,
    session: FkTwoBoneMoveSession,
    world_delta: Vector,
    *,
    allow_full_extension: bool = False,
) -> bool:
    solved = _solve_two_bone_fk_points(
        session,
        Vector(session.end_world) + Vector(world_delta),
        allow_full_extension=allow_full_extension,
    )
    if solved is None:
        return False
    desired_joint_world, desired_end_world = solved
    if not _apply_solved_two_bone_fk_pose(
        session,
        desired_joint_world,
        desired_end_world,
    ):
        return False

    # Free FK Move must obey the same generated Rigped envelopes during live
    # preview that frame replay enforces later. Public elbow/knee limits are
    # branch-neutral, so valid FK bends on either side of local zero survive
    # unchanged while only excessive/off-axis motion is projected.
    for control in (
        session.first_control,
        session.second_control,
        session.terminal_control,
    ):
        target = control.target
        if isinstance(target, bpy.types.PoseBone):
            apply_rigped_joint_limits_to_pose_bone(target)
    return True


def cancel_fk_two_bone_move(context, session: FkTwoBoneMoveSession) -> None:
    session.first_control.target.matrix_basis = session.first_start_basis.copy()
    session.second_control.target.matrix_basis = session.second_start_basis.copy()
    session.terminal_control.target.matrix_basis = session.terminal_start_basis.copy()
    context.view_layer.update()


def _selected_sliding_move_selections(context) -> tuple[SemanticMoveSelection, ...]:
    """Cache static selection-to-limb routing across gizmo redraws."""

    global _SLIDING_MOVE_SELECTION_CACHE_KEY, _SLIDING_MOVE_SELECTION_CACHE_VALUE

    cache_key = _semantic_move_selection_cache_key(context)
    if cache_key is None:
        return ()
    if cache_key == _SLIDING_MOVE_SELECTION_CACHE_KEY:
        return _SLIDING_MOVE_SELECTION_CACHE_VALUE

    control_context = control_context_for_context(context)
    target_resolution = resolve_rigped_target(context.scene, control_context)
    target = target_resolution.target
    resolved_selections: tuple[SemanticMoveSelection, ...] = ()
    if target is not None and target.selected_binding_ids:
        by_mapping: dict[str, SemanticMoveSelection] = {}
        active_mapping_id: str | None = None
        valid = True
        for binding_id in target.selected_binding_ids:
            selected = _resolve_semantic_move_binding(context, target, binding_id)
            if selected is None:
                valid = False
                break
            mapping_id = str(selected.capability.native_ik.mapping_id)
            by_mapping.setdefault(mapping_id, selected)
            if binding_id == target.active_binding_id:
                active_mapping_id = mapping_id
        if valid and by_mapping:
            ordered = sorted(
                by_mapping.items(),
                key=lambda item: (0 if item[0] == active_mapping_id else 1, item[0]),
            )
            resolved_selections = tuple(item[1] for item in ordered)

    _SLIDING_MOVE_SELECTION_CACHE_KEY = cache_key
    _SLIDING_MOVE_SELECTION_CACHE_VALUE = resolved_selections
    return resolved_selections


def _selected_semantic_move_resolutions(context) -> tuple[SemanticMoveResolution, ...]:
    candidates = _selected_fk_joint_move_candidates(context)
    if not candidates:
        return ()

    resolutions: list[SemanticMoveResolution] = []
    for candidate in candidates:
        selected = candidate.semantic_selection
        if selected is None:
            # Direct rotational controls (Head/Neck/Spine/Clavicle/Pelvis) are
            # valid mixed Move domains and reuse the existing FK single-link path.
            continue
        resolved = _semantic_move_resolution_for_selection(selected)
        if resolved is None or resolved.contact_type is ContactKeyType.PLANTED:
            return ()
        if resolved.contact_type not in {ContactKeyType.FREE, ContactKeyType.SLIDING}:
            return ()
        resolutions.append(resolved)
    return tuple(resolutions)


def _selected_sliding_move_resolutions(context) -> tuple[SemanticMoveResolution, ...]:
    candidates = _selected_fk_joint_move_candidates(context)
    resolutions = _selected_semantic_move_resolutions(context)
    if (
        not resolutions
        or len(candidates) != len(resolutions)
        or any(
            resolved.contact_type is not ContactKeyType.SLIDING
            for resolved in resolutions
        )
    ):
        return ()
    return resolutions


def semantic_move_available(context) -> bool:
    # FK_MOVE owns the all-Free case. Semantic Move owns any valid selection
    # containing Sliding authority, including Sliding + Free limbs and direct
    # rotational controls such as Head in the same W gesture.
    resolutions = _selected_semantic_move_resolutions(context)
    return bool(resolutions) and any(
        resolved.contact_type is ContactKeyType.SLIDING
        for resolved in resolutions
    )


def _semantic_move_active_control(context) -> ResolvedControl | None:
    candidates = _selected_fk_joint_move_candidates(context)
    if not candidates or not semantic_move_available(context):
        return None
    return candidates[0].active_control


def semantic_move_pivot(context) -> Vector | None:
    active = _semantic_move_active_control(context)
    return _control_pivot_world(active) if active is not None else None


def semantic_move_axes(context) -> dict[str, Vector] | None:
    active = _semantic_move_active_control(context)
    return _orientation_axes_for_control(context, active) if active is not None else None


def sliding_semantic_move_available(context) -> bool:
    return bool(_selected_sliding_move_resolutions(context))


def _begin_sliding_semantic_move_for_resolution(
    context,
    resolved: SemanticMoveResolution,
) -> SlidingMoveSession:
    planned = build_contact_intent_plan(
        context.scene,
        control_context_for_context(context),
        operation_id=f"ak4:sliding-move-preview:{uuid4().hex}",
        mode=ContactAuthoringMode.ANCHOR,
        mapping_id=str(resolved.capability.native_ik.mapping_id),
    )
    if not planned.ok or planned.plan is None:
        detail = "; ".join(item.detail for item in planned.diagnostics)
        raise RigpedSemanticMoveError(detail or "Sliding semantic Move preflight failed.")
    if planned.plan.target_type is not ContactKeyType.SLIDING:
        raise RigpedSemanticMoveError("Sliding semantic Move requires authoritative Sliding state.")
    if planned.plan.mapping_id != resolved.capability.native_ik.mapping_id:
        raise RigpedSemanticMoveError("Sliding semantic Move mapping changed during preflight.")

    ik_target = resolved.capability.native_ik.ik_target
    local_delta_matrix = _world_to_control_location_delta_matrix(ik_target)
    if local_delta_matrix is None:
        raise RigpedSemanticMoveError("Sliding IK target cannot resolve a writable world/local location basis.")

    pole_target = resolved.capability.native_ik.pole_target
    if pole_target is None:
        raise RigpedSemanticMoveError("Sliding semantic Move requires the generated pole target.")
    pole_local_delta_matrix = _world_to_control_location_delta_matrix(pole_target)
    if pole_local_delta_matrix is None:
        raise RigpedSemanticMoveError("Sliding pole target cannot resolve a writable world/local location basis.")

    # Sliding uses the hidden native IK chain as authority, but a fixed world-space
    # pole can cross the moving chain plane and make Blender choose the opposite
    # two-bone solution. Build one gesture-local analytic session for BOTH arms
    # and legs, then transport the hidden pole along the same continuous bend
    # branch for the whole drag.
    analytic_resolution = SemanticMoveResolution(
        resolved.character_id,
        str(resolved.capability.authored_terminal_binding_id),
        ContactKeyType.FREE,
        resolved.capability,
        resolved.capability.authored_terminal,
    )
    analytic_fk_session = _begin_fk_two_bone_move_for_resolution(
        analytic_resolution
    )
    start_pole_world = _world_position(pole_target)
    start_axis = Vector(analytic_fk_session.end_world) - Vector(analytic_fk_session.root_world)
    if start_axis.length <= 1e-9:
        raise RigpedSemanticMoveError("Sliding semantic Move found a collapsed two-bone axis.")
    start_axis.normalize()
    pole_offset = start_pole_world - Vector(analytic_fk_session.joint_world)
    pole_axial_offset = float(pole_offset.dot(start_axis))
    pole_perp = pole_offset - start_axis * pole_axial_offset
    if pole_perp.length <= 1e-8:
        raise RigpedSemanticMoveError("Sliding semantic Move found a singular pole direction.")
    pole_perp_distance = float(pole_perp.length)
    pole_perp.normalize()

    # Keep two different concepts separate. The analytic session already owns
    # the human preferred bend branch (or the current meaningful bend). The
    # authored Blender pole vector is *not* that bend direction when pole_angle
    # is non-zero; it is the native solver reference before pole-angle rotation.
    # Treating it as the knee/elbow bend direction causes an immediate branch
    # jump on generated legs whose pole angle is roughly +/- 90 degrees.

    return SlidingMoveSession(
        intent=planned.plan,
        capability=resolved.capability,
        ik_target=ik_target,
        pole_target=pole_target,
        start_location=tuple(float(value) for value in ik_target.target.location),
        pole_start_location=tuple(float(value) for value in pole_target.target.location),
        local_delta_matrix=local_delta_matrix,
        pole_local_delta_matrix=pole_local_delta_matrix,
        start_pivot_world=_world_position(resolved.capability.result_terminal),
        start_pole_world=start_pole_world,
        start_axis_world=start_axis.copy(),
        pole_perp_world=pole_perp.copy(),
        pole_axial_offset=pole_axial_offset,
        pole_perp_distance=pole_perp_distance,
        start_fk_states=tuple(
            _capture_control_state(control)
            for control in resolved.capability.fk_controls
        ),
        start_terminal_state=_capture_control_state(
            resolved.capability.authored_terminal
        ),
        analytic_fk_session=analytic_fk_session,
    )


def begin_sliding_semantic_moves(context) -> tuple[SlidingMoveSession, ...]:
    resolutions = _selected_sliding_move_resolutions(context)
    if not resolutions:
        raise RigpedSemanticMoveError(
            "Current selection does not resolve supported Sliding Contact limbs."
        )
    sessions: list[SlidingMoveSession] = []
    try:
        for resolved in resolutions:
            sessions.append(_begin_sliding_semantic_move_for_resolution(context, resolved))
    except Exception:
        for session in sessions:
            cancel_sliding_semantic_move(context, session)
        raise
    return tuple(sessions)


def begin_sliding_semantic_move(context) -> SlidingMoveSession:
    sessions = begin_sliding_semantic_moves(context)
    if len(sessions) != 1:
        for session in sessions:
            cancel_sliding_semantic_move(context, session)
        raise RigpedSemanticMoveError(
            "Single Sliding Move requires exactly one selected limb domain."
        )
    return sessions[0]


def _begin_semantic_move_domains(
    context,
) -> tuple[
    tuple[SemanticMoveResolution, ...],
    tuple[SlidingMoveSession, ...],
    tuple[FkJointMoveSession, ...],
]:
    candidates = _selected_fk_joint_move_candidates(context)
    if not candidates:
        raise RigpedSemanticMoveError(
            "Current selection does not resolve supported mixed Move domains."
        )
    resolutions: list[SemanticMoveResolution] = []
    sliding_sessions: list[SlidingMoveSession] = []
    fk_sessions: list[FkJointMoveSession] = []
    try:
        for candidate in candidates:
            selected = candidate.semantic_selection
            if selected is None:
                if candidate.kind != "SINGLE":
                    raise RigpedSemanticMoveError(
                        "Direct mixed Move control is not a supported FK single-link domain."
                    )
                fk_sessions.append(
                    _begin_fk_single_link_move_for_control(candidate.active_control)
                )
                continue

            resolved = _semantic_move_resolution_for_selection(selected)
            if resolved is None or resolved.contact_type is ContactKeyType.PLANTED:
                raise RigpedSemanticMoveError(
                    "Mixed semantic Move supports Free and Sliding authority only."
                )
            resolutions.append(resolved)
            if resolved.contact_type is ContactKeyType.SLIDING:
                sliding_sessions.append(
                    _begin_sliding_semantic_move_for_resolution(context, resolved)
                )
                continue
            if resolved.contact_type is not ContactKeyType.FREE:
                raise RigpedSemanticMoveError(
                    "Mixed semantic Move supports Free and Sliding authority only."
                )
            if candidate.kind == "CHAIN_END":
                fk_sessions.append(_begin_fk_two_bone_move_for_resolution(resolved))
            elif candidate.kind == "FIRST":
                fk_sessions.append(
                    _begin_fk_single_link_move_for_control(resolved.active_control)
                )
            else:
                raise RigpedSemanticMoveError(
                    "Free limb selection is not a supported FK Move control."
                )
    except Exception:
        cancel_sliding_semantic_moves(context, tuple(sliding_sessions))
        cancel_fk_joint_moves(context, tuple(fk_sessions))
        raise
    return tuple(resolutions), tuple(sliding_sessions), tuple(fk_sessions)


def _mapped_mixed_semantic_move_delta(
    active_control: ResolvedControl,
    target_control: ResolvedControl,
    world_delta: Vector,
    *,
    orientation: str,
) -> Vector:
    if orientation in {"GLOBAL", "VIEW"}:
        return Vector(world_delta)
    active_target = active_control.target
    target_target = target_control.target
    if isinstance(active_target, bpy.types.PoseBone) and isinstance(target_target, bpy.types.PoseBone):
        opposite = _opposite_control_name(str(active_target.name))
        if opposite is not None and str(target_target.name) == opposite:
            return _mirror_control_world_vector(target_control, world_delta, axial=False)
    return Vector(world_delta)


def _rebase_sliding_semantic_move_session(
    context,
    session: SlidingMoveSession,
) -> SlidingMoveSession:
    capability = session.capability
    analytic_resolution = SemanticMoveResolution(
        character_id="",
        binding_id=str(capability.authored_terminal_binding_id),
        contact_type=ContactKeyType.FREE,
        capability=capability,
        active_control=capability.authored_terminal,
    )
    analytic_fk_session = _begin_fk_two_bone_move_for_resolution(analytic_resolution)

    local_delta_matrix = _world_to_control_location_delta_matrix(session.ik_target)
    pole_local_delta_matrix = _world_to_control_location_delta_matrix(session.pole_target)
    if local_delta_matrix is None or pole_local_delta_matrix is None:
        raise RigpedSemanticMoveError(
            "Sliding mixed Move could not rebase the hidden IK controls."
        )

    start_pole_world = _world_position(session.pole_target)
    start_axis = (
        Vector(analytic_fk_session.end_world)
        - Vector(analytic_fk_session.root_world)
    )
    if start_axis.length <= 1e-9:
        raise RigpedSemanticMoveError(
            "Sliding mixed Move found a collapsed rebased two-bone axis."
        )
    start_axis.normalize()
    pole_offset = start_pole_world - Vector(analytic_fk_session.joint_world)
    pole_axial_offset = float(pole_offset.dot(start_axis))
    pole_perp = pole_offset - start_axis * pole_axial_offset
    if pole_perp.length <= 1e-8:
        raise RigpedSemanticMoveError(
            "Sliding mixed Move found a singular rebased pole direction."
        )
    pole_perp_distance = float(pole_perp.length)
    pole_perp.normalize()

    return SlidingMoveSession(
        intent=session.intent,
        capability=capability,
        ik_target=session.ik_target,
        pole_target=session.pole_target,
        start_location=tuple(float(value) for value in session.ik_target.target.location),
        pole_start_location=tuple(float(value) for value in session.pole_target.target.location),
        local_delta_matrix=local_delta_matrix,
        pole_local_delta_matrix=pole_local_delta_matrix,
        start_pivot_world=_world_position(capability.result_terminal),
        start_pole_world=start_pole_world,
        start_axis_world=start_axis.copy(),
        pole_perp_world=pole_perp.copy(),
        pole_axial_offset=pole_axial_offset,
        pole_perp_distance=pole_perp_distance,
        start_fk_states=tuple(
            _capture_control_state(control)
            for control in capability.fk_controls
        ),
        start_terminal_state=_capture_control_state(capability.authored_terminal),
        analytic_fk_session=analytic_fk_session,
    )


def apply_semantic_move_domains_delta(
    context,
    resolutions: tuple[SemanticMoveResolution, ...],
    sliding_sessions: tuple[SlidingMoveSession, ...],
    fk_sessions: tuple[FkJointMoveSession, ...],
    world_delta: Vector,
) -> bool:
    if not resolutions:
        return False
    orientation = str(context.scene.transform_orientation_slots[0].type)
    active_control = _semantic_move_active_control(context)
    if active_control is None:
        return False

    sliding_rows = tuple(
        (
            session,
            _mapped_mixed_semantic_move_delta(
                active_control,
                session.capability.authored_terminal,
                world_delta,
                orientation=orientation,
            ),
        )
        for session in sliding_sessions
    )
    fk_rows = tuple(
        (
            session,
            _mapped_mixed_semantic_move_delta(
                active_control,
                _fk_session_active_control(session),
                world_delta,
                orientation=orientation,
            ),
        )
        for session in fk_sessions
    )
    single_rows = tuple(
        row for row in fk_rows if isinstance(row[0], FkSingleLinkMoveSession)
    )
    limb_rows = tuple(
        row for row in fk_rows if isinstance(row[0], FkTwoBoneMoveSession)
    )

    try:
        # Mixed ancestor + limb movement must be an absolute function of the
        # current cursor delta. Restore the gesture baseline first so a selected
        # Spine/Head never leaves child solvers one depsgraph event behind.
        cancel_semantic_move_domains(context, sliding_sessions, fk_sessions)

        # Direct/single FK controls own the parent pose first.
        for session, mapped_delta in single_rows:
            if not apply_fk_joint_move_delta(context, session, mapped_delta):
                raise RigpedSemanticMoveError(
                    "Direct FK domain could not solve the requested mixed Move delta."
                )
            context.view_layer.update()

        # Rebase Free two-bone limbs from the now-current parent pose while
        # preserving the animator-facing absolute end-point target.
        for session, mapped_delta in limb_rows:
            current_session = _begin_fk_two_bone_move_for_resolution(
                SemanticMoveResolution(
                    character_id="",
                    binding_id="",
                    contact_type=ContactKeyType.FREE,
                    capability=session.capability,
                    active_control=session.active_control,
                )
            )
            desired_end_world = Vector(session.end_world) + Vector(mapped_delta)
            compensating_delta = desired_end_world - Vector(current_session.end_world)
            if not apply_fk_two_bone_move_delta(
                context, current_session, compensating_delta
            ):
                raise RigpedSemanticMoveError(
                    "Free FK domain could not solve the requested mixed Move delta."
                )
            context.view_layer.update()

        # Sliding is solved last from the evaluated parent pose. This keeps the
        # hidden IK target, public FK reconstruction, and release-time residual
        # gate describing the exact same final pose when Spine/Head also moves.
        for session, mapped_delta in sliding_rows:
            current_session = _rebase_sliding_semantic_move_session(context, session)
            desired_end_world = (
                Vector(session.start_pivot_world) + Vector(mapped_delta)
            )
            compensating_delta = (
                desired_end_world - Vector(current_session.start_pivot_world)
            )
            apply_sliding_semantic_move_delta(
                context, current_session, compensating_delta
            )
            context.view_layer.update()

        # Applying a public FK reconstruction can itself change the evaluated
        # child result by a depsgraph settle when an ancestor such as Spine
        # moved in the same gesture. Converge each Sliding public/result pair
        # before release so the Contact writer sees the same stable pose.
        for session, _mapped_delta in sliding_rows:
            _converge_sliding_public_pose_from_result(
                context,
                session.capability,
            )
    except Exception:
        cancel_semantic_move_domains(context, sliding_sessions, fk_sessions)
        raise
    return True


def cancel_semantic_move_domains(
    context,
    sliding_sessions: tuple[SlidingMoveSession, ...],
    fk_sessions: tuple[FkJointMoveSession, ...],
) -> None:
    cancel_sliding_semantic_moves(context, sliding_sessions)
    cancel_fk_joint_moves(context, fk_sessions)


def _sync_sliding_public_pose_from_result(
    context,
    capability: LimbRepresentationCapability,
    *,
    update: bool = True,
) -> None:
    solved_result = capture_native_solved_result(capability)
    public_display = derive_public_display(capability, solved_result)

    first_state = _state_for_pose_basis(
        capability.fk_controls[0],
        public_display.first_basis,
    )
    second_state = _state_for_pose_basis(
        capability.fk_controls[1],
        public_display.second_basis,
    )
    terminal_state = _state_for_pose_basis(
        capability.authored_terminal,
        public_display.terminal_basis,
    )
    _apply_control_state(
        capability.fk_controls[0],
        first_state,
        location=False,
        rotation=True,
    )
    _apply_control_state(
        capability.fk_controls[1],
        second_state,
        location=False,
        rotation=True,
    )
    _apply_control_state(
        capability.authored_terminal,
        terminal_state,
        location=False,
        rotation=True,
    )
    if update:
        context.view_layer.update()


def _sliding_public_pose_residual(
    capability: LimbRepresentationCapability,
) -> tuple[float, float]:
    max_position = 0.0
    max_rotation = 0.0
    pairs = tuple(
        zip(capability.fk_controls, capability.result_controls, strict=True)
    ) + ((capability.authored_terminal, capability.result_terminal),)
    for public_control, result_control in pairs:
        position, rotation = _pose_residual(
            public_control.target.matrix,
            result_control.target.matrix,
        )
        max_position = max(max_position, position)
        max_rotation = max(max_rotation, rotation)
    return max_position, max_rotation


def _converge_sliding_public_pose_from_result(
    context,
    capability: LimbRepresentationCapability,
    *,
    max_passes: int = 6,
) -> None:
    # Public Sliding display projection writes rotation channels only.
    # Position residual remains valuable evidence, but it is not a writable
    # degree of freedom for this projection and therefore must not reject an
    # otherwise converged native-result display.
    rotation_tolerance = 5e-7
    last_position = 0.0
    last_rotation = 0.0
    for _pass_index in range(max(1, int(max_passes))):
        _sync_sliding_public_pose_from_result(context, capability)
        last_position, last_rotation = _sliding_public_pose_residual(capability)
        if last_rotation <= rotation_tolerance:
            return
    raise RigpedSemanticMoveError(
        "Sliding public/result convergence failed: "
        f"position={last_position:.9g} rotation={last_rotation:.9g}"
    )


def apply_sliding_semantic_move_delta(
    context,
    session: SlidingMoveSession,
    world_delta: Vector,
) -> None:
    requested_delta = Vector(world_delta)
    analytic = session.analytic_fk_session
    solver_seed = None
    if analytic is not None:
        # Solve one continuous two-bone branch in world space first. This is a
        # pure geometry solve: it clamps fixed reach and updates only the
        # gesture-local continuity state. The hidden Blender IK chain remains the
        # actual Sliding authority.
        solved = _solve_two_bone_fk_points(
            analytic,
            Vector(session.start_pivot_world) + requested_delta,
            allow_full_extension=False,
            minimum_bend_radians=radians(8.0),
        )
        if solved is not None:
            desired_joint_world, desired_end_world = solved
            requested_delta = Vector(desired_end_world) - Vector(session.start_pivot_world)

            direction = Vector(desired_end_world) - Vector(analytic.root_world)
            deterministic_bend = (
                Vector(analytic.bend_plane_normal_world).cross(direction)
            )
            solver_seed = build_transient_solver_seed(
                root_world=Vector(analytic.root_world),
                target_world=Vector(desired_end_world),
                preferred_bend_world=deterministic_bend,
                first_length=float(analytic.first_length),
                second_length=float(analytic.second_length),
                minimum_bend_radians=radians(8.0),
            )
            pole_reference = Vector(session.pole_perp_world)
            start_axis = Vector(session.start_axis_world)
            if (
                direction.length > 1e-9
                and pole_reference.length > 1e-9
                and start_axis.length > 1e-9
            ):
                direction.normalize()
                start_axis.normalize()
                # Transport the native pole reference with the changing chain
                # axis, preserving its authored relationship to pole_angle.
                # The analytic bend direction stays separate and is used only
                # to choose the human knee/elbow solution branch.
                transport = start_axis.rotation_difference(direction)
                pole_reference = Vector(transport @ pole_reference)
                pole_reference -= direction * float(pole_reference.dot(direction))
                if pole_reference.length > 1e-9:
                    pole_reference.normalize()
                    desired_pole_world = (
                        Vector(desired_joint_world)
                        + direction * float(session.pole_axial_offset)
                        + pole_reference * float(session.pole_perp_distance)
                    )
                    pole_world_delta = desired_pole_world - Vector(session.start_pole_world)
                    pole_local_delta = session.pole_local_delta_matrix @ pole_world_delta
                    session.pole_target.target.location = (
                        Vector(session.pole_start_location) + pole_local_delta
                    )

    local_delta = session.local_delta_matrix @ requested_delta
    session.ik_target.target.location = Vector(session.start_location) + local_delta

    # Blender's native two-bone IK can remain locked on an exactly straight
    # chain even when the goal moves slightly inside maximum reach. Seed the
    # MCH input pose on the analytic preferred branch with IK transiently muted,
    # then re-enable the native solve from that non-singular starting pose.
    native_ik = session.capability.native_ik.constraint
    if analytic is not None and solver_seed is not None:
        was_muted = bool(native_ik.mute)
        try:
            native_ik.mute = True
            if not _apply_solved_two_bone_fk_pose(
                analytic,
                Vector(solver_seed.joint_world),
                Vector(solver_seed.end_world),
            ):
                raise RigpedSemanticMoveError(
                    "Sliding preferred-bend seed could not reconstruct the public limb pose."
                )
            context.view_layer.update()
        finally:
            native_ik.mute = was_muted
    context.view_layer.update()

    # Native IK can hit its anatomical envelope before the requested Sliding goal.
    # Never leave the hidden target beyond the actually solved chain tip: clamp the
    # target back to the reachable result so Hand/Foot and the red IK pivot stay
    # welded instead of accumulating a visible residual at extreme compound poses.
    capability = session.capability
    actual_tip_world = Vector(evaluated_chain_tip_world_position(capability.native_ik))
    requested_tip_world = Vector(session.start_pivot_world) + requested_delta
    reach_error = actual_tip_world - requested_tip_world
    reach_tolerance = max(
        1e-6,
        float((analytic.first_length + analytic.second_length) if analytic is not None else 1.0)
        * 1e-5,
    )
    if reach_error.length > reach_tolerance:
        session.ik_target.target.location = (
            Vector(session.ik_target.target.location)
            + session.local_delta_matrix @ reach_error
        )
        context.view_layer.update()

    # Sliding authority lives on the hidden two-bone IK chain, but the animator
    # must still see the public Rigped controls follow that evaluated result.
    _sync_sliding_public_pose_from_result(context, capability)


def _mapped_sliding_move_delta(
    active_session: SlidingMoveSession,
    target_session: SlidingMoveSession,
    world_delta: Vector,
    *,
    orientation: str,
) -> Vector:
    if orientation in {"GLOBAL", "VIEW"}:
        return Vector(world_delta)
    active = active_session.capability.authored_terminal
    target = target_session.capability.authored_terminal
    active_target = active.target
    target_target = target.target
    if isinstance(active_target, bpy.types.PoseBone) and isinstance(target_target, bpy.types.PoseBone):
        opposite = _opposite_control_name(str(active_target.name))
        if opposite is not None and str(target_target.name) == opposite:
            return _mirror_control_world_vector(target, world_delta, axial=False)
    return Vector(world_delta)


def apply_sliding_semantic_moves_delta(
    context,
    sessions: tuple[SlidingMoveSession, ...],
    world_delta: Vector,
) -> bool:
    if not sessions:
        return False
    orientation = str(context.scene.transform_orientation_slots[0].type)
    active_session = sessions[0]
    try:
        for session in sessions:
            mapped = _mapped_sliding_move_delta(
                active_session,
                session,
                world_delta,
                orientation=orientation,
            )
            apply_sliding_semantic_move_delta(context, session, mapped)
    except Exception:
        for session in sessions:
            cancel_sliding_semantic_move(context, session)
        raise
    return True


def cancel_sliding_semantic_moves(
    context,
    sessions: tuple[SlidingMoveSession, ...],
) -> None:
    for session in sessions:
        cancel_sliding_semantic_move(context, session)


def cancel_sliding_semantic_move(context, session: SlidingMoveSession) -> None:
    session.ik_target.target.location = session.start_location
    session.pole_target.target.location = session.pole_start_location
    for control, state in zip(
        session.capability.fk_controls,
        session.start_fk_states,
        strict=True,
    ):
        _apply_control_state(control, state, location=True, rotation=True)
    _apply_control_state(
        session.capability.authored_terminal,
        session.start_terminal_state,
        location=True,
        rotation=True,
    )
    context.view_layer.update()


def commit_sliding_semantic_move(
    context,
    session: SlidingMoveSession,
    auto_plan: RigpedAutoAnchorPlan | None = None,
) -> SemanticMoveCommitResult:
    if not bool(getattr(context.scene, "baw_auto_key_enabled", False)):
        return SemanticMoveCommitResult(True, False)
    if auto_plan is None:
        planned = plan_rigped_auto_anchor(
            context.scene,
            control_context_for_context(context),
            operation_id=f"ak4:sliding-move-auto:{uuid4().hex}",
            mapping_id=session.intent.mapping_id,
        )
        if not planned.ok or planned.plan is None:
            detail = (
                planned.diagnostics[0].detail
                if planned.diagnostics
                else "Rigped Sliding Move Auto planning failed."
            )
            cancel_sliding_semantic_move(context, session)
            raise RigpedSemanticMoveError(detail)
        auto_plan = planned.plan
    if auto_plan.intent.target_type is not ContactKeyType.SLIDING:
        cancel_sliding_semantic_move(context, session)
        raise RigpedSemanticMoveError("Sliding Move Auto must preserve Sliding authority.")
    try:
        result: ContactAuthoringResult = commit_rigped_auto_anchor(
            context.scene,
            control_context_for_context(context),
            auto_plan,
        )
    except (ContactAuthoringError, RuntimeError, ValueError, ReferenceError) as exc:
        cancel_sliding_semantic_move(context, session)
        raise RigpedSemanticMoveError(str(exc)) from exc
    if not result.applied:
        cancel_sliding_semantic_move(context, session)
        return SemanticMoveCommitResult(False, False, tuple(result.diagnostics))
    return SemanticMoveCommitResult(True, True, tuple(result.diagnostics))


def _restore_free_move_start(
    context,
    session: FreeMoveSession,
    *,
    update: bool = True,
) -> None:
    capability = session.capability
    capability.native_ik.constraint.influence = session.start_ik_influence
    capability.terminal_ik_constraint.influence = session.start_terminal_ik_influence
    capability.native_ik.constraint.pole_angle = session.start_pole_angle
    _apply_control_state(session.ik_target, session.start_ik_state)
    _apply_control_state(session.pole_target, session.start_pole_state)
    for control, state in zip(capability.fk_controls, session.start_fk_states, strict=True):
        _apply_control_state(control, state)
    _apply_control_state(capability.authored_terminal, session.start_terminal_state)
    if update:
        context.view_layer.update()


def begin_free_semantic_move(context) -> FreeMoveSession:
    resolved = _selected_semantic_mapping(context)
    if resolved is None or resolved.contact_type is not ContactKeyType.FREE:
        raise RigpedSemanticMoveError(
            "Free semantic Move requires exactly one generated FK-authoritative limb."
        )
    capability = resolved.capability
    pole_target = capability.native_ik.pole_target
    if pole_target is None:
        raise RigpedSemanticMoveError("Free semantic Move requires the generated authored pole target.")

    planned = build_contact_intent_plan(
        context.scene,
        control_context_for_context(context),
        operation_id="i16:free-semantic-move",
        mode=ContactAuthoringMode.ANCHOR,
    )
    if not planned.ok or planned.plan is None:
        detail = "; ".join(item.detail for item in planned.diagnostics)
        raise RigpedSemanticMoveError(detail or "Free semantic Move preflight failed.")
    intent = planned.plan
    if intent.target_type is not ContactKeyType.FREE:
        raise RigpedSemanticMoveError("Free semantic Move preflight did not preserve Free authority.")
    if intent.mapping_id != capability.native_ik.mapping_id:
        raise RigpedSemanticMoveError("Free semantic Move mapping changed during preflight.")

    payload = build_representation_snap_payload(capability, SnapDirection.FK_TO_IK)
    try:
        calibration = execute_representation_snap(
            context.scene,
            control_context_for_context(context),
            intent.snap_plan,
            payload,
        )
    except (RuntimeError, ValueError, ReferenceError) as exc:
        raise RigpedSemanticMoveError(str(exc)) from exc
    if (
        not calibration.success
        or not calibration.restored_exactly
        or calibration.ik_target_state is None
        or calibration.pole_target_state is None
        or calibration.pole_angle is None
    ):
        detail = "; ".join(item.detail for item in calibration.diagnostics)
        raise RigpedSemanticMoveError(
            detail or "Free semantic Move could not calibrate an exact operation-local IK solve."
        )

    ik_target = capability.native_ik.ik_target
    local_delta_matrix = _world_to_control_location_delta_matrix(ik_target)
    if local_delta_matrix is None:
        raise RigpedSemanticMoveError("Free IK target cannot resolve a writable world/local location basis.")
    return FreeMoveSession(
        intent=intent,
        capability=capability,
        ik_target=ik_target,
        pole_target=pole_target,
        local_delta_matrix=local_delta_matrix,
        start_pivot_world=Vector(evaluated_chain_tip_world_position(capability.native_ik)),
        start_fk_states=tuple(_capture_control_state(control) for control in capability.fk_controls),
        start_terminal_state=_capture_control_state(capability.authored_terminal),
        start_ik_state=_capture_control_state(ik_target),
        start_pole_state=_capture_control_state(pole_target),
        start_ik_influence=float(capability.native_ik.constraint.influence),
        start_terminal_ik_influence=float(capability.terminal_ik_constraint.influence),
        start_pole_angle=float(capability.native_ik.constraint.pole_angle),
        calibrated_ik_state=calibration.ik_target_state,
        calibrated_pole_state=calibration.pole_target_state,
        calibrated_pole_angle=float(calibration.pole_angle),
    )


def apply_free_semantic_move_delta(
    context,
    session: FreeMoveSession,
    world_delta: Vector,
) -> bool:
    capability = session.capability
    session.last_issue = None

    # Batch the transient FK->IK preview setup before one depsgraph evaluation.
    # The previous implementation forced several full view-layer updates for
    # every mouse event, which became visibly laggy once animation/paths existed.
    _restore_free_move_start(context, session, update=False)
    _apply_control_state(session.ik_target, session.calibrated_ik_state)
    _apply_control_state(session.pole_target, session.calibrated_pole_state)
    capability.native_ik.constraint.pole_angle = session.calibrated_pole_angle
    capability.native_ik.constraint.influence = 1.0
    capability.terminal_ik_constraint.influence = 1.0

    scale = max(
        1e-6,
        float(getattr(capability.result_controls[0].owner_object.dimensions, "length", 0.0) or 0.0),
    )
    requested = Vector(session.start_pivot_world) + Vector(world_delta)
    target_tolerance = max(1e-6, scale * 1e-5)
    local_delta = session.local_delta_matrix @ Vector(world_delta)
    session.ik_target.target.location = Vector(session.calibrated_ik_state.location) + local_delta
    context.view_layer.update()
    actual = Vector(evaluated_chain_tip_world_position(capability.native_ik))
    target_error = float((actual - requested).length)
    if target_error > target_tolerance:
        session.last_issue = f"CHAIN_TIP_RESIDUAL:{target_error:.9g}"
        _restore_free_move_start(context, session)
        return False

    desired_result = tuple(control.target.matrix.copy() for control in capability.result_controls)
    desired_terminal = capability.result_terminal.target.matrix.copy()

    # Return to FK authority in one reset evaluation, then derive each local
    # rotation from the captured desired parent pose instead of reevaluating the
    # whole constrained rig after every bone in the chain.
    _restore_free_move_start(context, session, update=False)
    context.view_layer.update()
    desired_fk_pose: dict[str, Matrix] = {}
    for control, desired in zip(capability.fk_controls, desired_result, strict=True):
        parent = getattr(control.target, "parent", None)
        parent_pose = desired_fk_pose.get(parent.name) if parent is not None else None
        state = _state_for_pose_matrix(
            control,
            desired,
            parent_pose_matrix=parent_pose,
        )
        _apply_control_state(control, state, location=False, rotation=True)
        desired_fk_pose[str(control.target.name)] = desired
    terminal_parent = getattr(capability.authored_terminal.target, "parent", None)
    terminal_parent_pose = (
        desired_fk_pose.get(terminal_parent.name) if terminal_parent is not None else None
    )
    terminal_state = _state_for_pose_matrix(
        capability.authored_terminal,
        desired_terminal,
        parent_pose_matrix=terminal_parent_pose,
    )
    _apply_control_state(
        capability.authored_terminal,
        terminal_state,
        location=False,
        rotation=True,
    )
    context.view_layer.update()

    position_tolerance = max(1e-7, scale * 1e-6)
    for control, expected in zip(capability.result_controls, desired_result, strict=True):
        position_error, rotation_error = _pose_residual(control.target.matrix, expected)
        if position_error > position_tolerance or rotation_error > 1e-6:
            session.last_issue = (
                f"FK_RECONSTRUCTION_RESIDUAL:{position_error:.9g}:{rotation_error:.9g}"
            )
            _restore_free_move_start(context, session)
            return False
    terminal_position, terminal_rotation = _pose_residual(
        capability.result_terminal.target.matrix,
        desired_terminal,
    )
    if terminal_position > position_tolerance or terminal_rotation > 1e-6:
        session.last_issue = (
            f"TERMINAL_RECONSTRUCTION_RESIDUAL:{terminal_position:.9g}:{terminal_rotation:.9g}"
        )
        _restore_free_move_start(context, session)
        return False
    return True


def cancel_free_semantic_move(context, session: FreeMoveSession) -> None:
    _restore_free_move_start(context, session)


def commit_free_semantic_move(
    context,
    session: FreeMoveSession,
) -> SemanticMoveCommitResult:
    if not bool(getattr(context.scene, "baw_auto_key_enabled", False)):
        return SemanticMoveCommitResult(True, False)
    try:
        result: ContactAuthoringResult = execute_contact_intent_plan(
            context.scene,
            control_context_for_context(context),
            session.intent,
            trigger=WriterTrigger.AUTO_TRANSFORM,
        )
    except (ContactAuthoringError, RuntimeError, ValueError, ReferenceError) as exc:
        cancel_free_semantic_move(context, session)
        raise RigpedSemanticMoveError(str(exc)) from exc
    if not result.applied:
        cancel_free_semantic_move(context, session)
        return SemanticMoveCommitResult(False, False, tuple(result.diagnostics))
    return SemanticMoveCommitResult(True, True, tuple(result.diagnostics))


FkJointMoveSession = FkSingleLinkMoveSession | FkTwoBoneMoveSession


def _fk_session_active_control(session: FkJointMoveSession) -> ResolvedControl:
    return session.active_control


def _opposite_control_name(name: str) -> str | None:
    if name.endswith(".L"):
        return f"{name[:-2]}.R"
    if name.endswith(".R"):
        return f"{name[:-2]}.L"
    return None


def _mirror_control_world_vector(
    control: ResolvedControl,
    vector: Vector,
    *,
    axial: bool = False,
) -> Vector:
    world3 = control.owner_object.matrix_world.to_3x3()
    local = Vector(control.owner_object.matrix_world.inverted_safe().to_3x3() @ Vector(vector))
    local.x = -local.x
    if axial:
        local.negate()
    return Vector(world3 @ local)


def _mapped_fk_move_delta(
    active_session: FkJointMoveSession,
    target_session: FkJointMoveSession,
    world_delta: Vector,
) -> Vector:
    active = _fk_session_active_control(active_session)
    target = _fk_session_active_control(target_session)
    active_target = active.target
    target_target = target.target
    if isinstance(active_target, bpy.types.PoseBone) and isinstance(target_target, bpy.types.PoseBone):
        opposite = _opposite_control_name(str(active_target.name))
        if opposite is not None and str(target_target.name) == opposite:
            return _mirror_control_world_vector(target, world_delta, axial=False)
    return Vector(world_delta)


def begin_fk_joint_moves(context) -> tuple[FkJointMoveSession, ...]:
    targets = _selected_fk_joint_move_targets(context)
    if not targets:
        raise RigpedSemanticMoveError("Current selection has no supported Free/FK Move operation domain.")
    sessions: list[FkJointMoveSession] = []
    for target in targets:
        if target.kind in {"SINGLE", "FIRST"}:
            sessions.append(_begin_fk_single_link_move_for_control(target.active_control))
        elif target.kind == "CHAIN_END" and target.resolution is not None:
            sessions.append(_begin_fk_two_bone_move_for_resolution(target.resolution))
        else:
            raise RigpedSemanticMoveError("Current selection contains an unsupported FK Move target.")
    return tuple(sessions)


def begin_fk_joint_move(context) -> FkJointMoveSession:
    sessions = begin_fk_joint_moves(context)
    if len(sessions) != 1:
        raise RigpedSemanticMoveError("Current selection resolves multiple FK Move operation domains.")
    return sessions[0]


def apply_fk_joint_move_delta(
    context,
    session: FkJointMoveSession,
    world_delta: Vector,
) -> bool:
    if isinstance(session, FkSingleLinkMoveSession):
        return apply_fk_single_link_move_delta(context, session, world_delta)
    return apply_fk_two_bone_move_delta(context, session, world_delta)


def apply_fk_joint_moves_delta(
    context,
    sessions: tuple[FkJointMoveSession, ...],
    world_delta: Vector,
) -> bool:
    if not sessions:
        return False
    active_session = sessions[0]
    orientation = str(context.scene.transform_orientation_slots[0].type)
    mapped_deltas = tuple(
        (
            Vector(world_delta)
            if orientation in {"GLOBAL", "VIEW"}
            else _mapped_fk_move_delta(active_session, session, world_delta)
        )
        for session in sessions
    )
    single_rows = tuple(
        (session, mapped_delta)
        for session, mapped_delta in zip(sessions, mapped_deltas, strict=True)
        if isinstance(session, FkSingleLinkMoveSession)
    )
    limb_rows = tuple(
        (session, mapped_delta)
        for session, mapped_delta in zip(sessions, mapped_deltas, strict=True)
        if isinstance(session, FkTwoBoneMoveSession)
    )

    if single_rows and limb_rows:
        # Mixed direct + limb Move must be an absolute function of the current
        # gesture delta. Otherwise a selected ancestor such as Spine leaves its
        # evaluated matrix one mouse event behind while a child arm rebuilds
        # local FK channels, so release can jump when it replays final_delta.
        cancel_fk_joint_moves(context, sessions)

        for session, mapped_delta in single_rows:
            if not apply_fk_joint_move_delta(context, session, mapped_delta):
                cancel_fk_joint_moves(context, sessions)
                return False
            # Child limb roots must see this gesture's ancestor pose rather than
            # the previous mouse event's evaluated parent matrix.
            context.view_layer.update()

        for session, mapped_delta in limb_rows:
            try:
                current_session = _begin_fk_two_bone_move_for_resolution(
                    SemanticMoveResolution(
                        character_id="",
                        binding_id="",
                        contact_type=ContactKeyType.FREE,
                        capability=session.capability,
                        active_control=session.active_control,
                    )
                )
            except RigpedSemanticMoveError:
                cancel_fk_joint_moves(context, sessions)
                return False

            desired_end_world = Vector(session.end_world) + Vector(mapped_delta)
            compensating_delta = desired_end_world - Vector(current_session.end_world)
            if not apply_fk_two_bone_move_delta(
                context,
                current_session,
                compensating_delta,
            ):
                cancel_fk_joint_moves(context, sessions)
                return False

        context.view_layer.update()
        return True

    for session, mapped_delta in zip(sessions, mapped_deltas, strict=True):
        if not apply_fk_joint_move_delta(context, session, mapped_delta):
            cancel_fk_joint_moves(context, sessions)
            return False
    context.view_layer.update()
    return True


def _fk_move_motion_response_samples(
    context,
    sessions: tuple[FkJointMoveSession, ...],
    world_delta: Vector,
) -> tuple[dict[str, Any], ...]:
    """Measure requested-vs-actual movement without emitting per-mouse-event logs."""

    if not sessions:
        return ()
    active_session = sessions[0]
    orientation = str(context.scene.transform_orientation_slots[0].type)
    rows: list[dict[str, Any]] = []
    for session in sessions:
        mapped_delta = (
            Vector(world_delta)
            if orientation in {"GLOBAL", "VIEW"}
            else _mapped_fk_move_delta(active_session, session, world_delta)
        )
        requested_length = float(mapped_delta.length)
        if isinstance(session, FkTwoBoneMoveSession):
            start_world = Vector(session.end_world)
            actual_world = _world_position(session.terminal_control)
            reach_epsilon = max(
                1e-5,
                float(session.first_length + session.second_length) * 0.001,
            )
            max_reach = max(
                reach_epsilon,
                float(session.first_length + session.second_length) - reach_epsilon,
            )
            reach_ratio = float(
                (Vector(actual_world) - Vector(session.root_world)).length / max_reach
            )
            kind = "CHAIN_END"
        else:
            if session.center_pivot:
                start_world = Vector(session.start_pivot_world)
                actual_world = _control_pivot_world(session.active_control)
            else:
                start_world = Vector(session.start_tail_world)
                pose_bone = session.active_control.target
                actual_world = Vector(
                    session.active_control.owner_object.matrix_world @ pose_bone.tail
                )
            reach_ratio = None
            kind = "DIRECT"
        actual_delta = Vector(actual_world) - start_world
        actual_length = float(actual_delta.length)
        along_ratio = None
        lateral_error = 0.0
        if requested_length > 1e-9:
            direction = Vector(mapped_delta) / requested_length
            along = float(actual_delta.dot(direction))
            along_ratio = float(along / requested_length)
            lateral_error = float((actual_delta - direction * along).length)
        rows.append(
            {
                "domain": str(getattr(session.active_control.target, "name", "")),
                "kind": kind,
                "requested_delta": tuple(float(value) for value in mapped_delta),
                "actual_delta": tuple(float(value) for value in actual_delta),
                "requested_distance": requested_length,
                "actual_distance": actual_length,
                "along_ratio": along_ratio,
                "lateral_error": lateral_error,
                "reach_ratio": reach_ratio,
            }
        )
    return tuple(rows)


def cancel_fk_joint_move(context, session: FkJointMoveSession) -> None:
    if isinstance(session, FkSingleLinkMoveSession):
        cancel_fk_single_link_move(context, session)
    else:
        cancel_fk_two_bone_move(context, session)


def cancel_fk_joint_moves(
    context,
    sessions: tuple[FkJointMoveSession, ...],
) -> None:
    for session in reversed(sessions):
        if isinstance(session, FkSingleLinkMoveSession):
            session.active_control.target.matrix_basis = session.start_basis.copy()
        else:
            session.first_control.target.matrix_basis = session.first_start_basis.copy()
            session.second_control.target.matrix_basis = session.second_start_basis.copy()
            session.terminal_control.target.matrix_basis = session.terminal_start_basis.copy()
    context.view_layer.update()


def _fk_session_controls(session: FkJointMoveSession) -> tuple[ResolvedControl, ...]:
    if isinstance(session, FkSingleLinkMoveSession):
        return (session.active_control,)
    return (
        session.first_control,
        session.second_control,
        session.terminal_control,
    )


def commit_fk_joint_moves(
    context,
    sessions: tuple[FkJointMoveSession, ...],
) -> SemanticMoveCommitResult:
    # FK Move itself is a pose edit. Explicit C authoring and optional AUTO
    # authoring are orchestrated outside this no-op completion hook so the
    # solver never owns key creation.
    return SemanticMoveCommitResult(True, False)


def commit_fk_joint_move(
    context,
    session: FkJointMoveSession,
) -> SemanticMoveCommitResult:
    return commit_fk_joint_moves(context, (session,))


SemanticMoveSession = SlidingMoveSession | FreeMoveSession


def begin_semantic_move(context) -> SemanticMoveSession:
    resolved = _selected_semantic_mapping(context)
    if resolved is None:
        raise RigpedSemanticMoveError("Current selection does not resolve one supported semantic Move limb.")
    if resolved.contact_type is ContactKeyType.SLIDING:
        return begin_sliding_semantic_move(context)
    if resolved.contact_type is ContactKeyType.FREE:
        raise RigpedSemanticMoveError("Free is FK-authoritative; set Sliding before moving Hand/Foot as IK.")
    raise RigpedSemanticMoveError("Planted semantic Move is fail-closed; release or slide the Contact first.")


def apply_semantic_move_delta(
    context,
    session: SemanticMoveSession,
    world_delta: Vector,
) -> bool:
    if isinstance(session, SlidingMoveSession):
        apply_sliding_semantic_move_delta(context, session, world_delta)
        return True
    return apply_free_semantic_move_delta(context, session, world_delta)


def cancel_semantic_move(context, session: SemanticMoveSession) -> None:
    if isinstance(session, SlidingMoveSession):
        cancel_sliding_semantic_move(context, session)
    else:
        cancel_free_semantic_move(context, session)


def commit_semantic_move(
    context,
    session: SemanticMoveSession,
    auto_plan: RigpedAutoAnchorPlan | None = None,
) -> SemanticMoveCommitResult:
    if isinstance(session, SlidingMoveSession):
        return commit_sliding_semantic_move(context, session, auto_plan=auto_plan)
    return commit_free_semantic_move(context, session)


def activate_rigped_fk_joint_move_tool(context) -> bool:
    if not fk_joint_move_available(context):
        return False
    context.scene.baw_rigped_semantic_transform_mode = "FK_MOVE"
    context.space_data.show_gizmo = True
    if hasattr(context.space_data, "show_gizmo_tool"):
        context.space_data.show_gizmo_tool = False
    context.area.tag_redraw()
    return True


def activate_rigped_semantic_move_tool(context) -> bool:
    if not semantic_move_available(context):
        context.scene.baw_rigped_semantic_transform_mode = "NONE"
        return False
    context.scene.baw_rigped_semantic_transform_mode = "MOVE"
    context.space_data.show_gizmo = True
    if hasattr(context.space_data, "show_gizmo_tool"):
        context.space_data.show_gizmo_tool = False
    context.area.tag_redraw()
    return True


def deactivate_rigped_semantic_tool(context) -> None:
    scene = getattr(context, "scene", None)
    if scene is not None and hasattr(scene, "baw_rigped_semantic_transform_mode"):
        scene.baw_rigped_semantic_transform_mode = "NONE"


def _set_semantic_move_drag_active(context, active: bool) -> None:
    window_manager = getattr(context, "window_manager", None)
    if window_manager is not None and hasattr(
        window_manager, "baw_rigped_semantic_move_drag_active"
    ):
        window_manager.baw_rigped_semantic_move_drag_active = bool(active)


def _refresh_after_semantic_move(context) -> None:
    """Refresh expensive animation state once after the modal drag."""

    try:
        if bool(getattr(context.scene, "baw_auto_key_enabled", False)):
            from .trackbar_keying import resync_awb_auto_key_after_context_change

            resync_awb_auto_key_after_context_change(context)
        from .viewport_trajectory import refresh_active_trajectory_live

        refresh_active_trajectory_live(context, force=True)
    except (RuntimeError, ReferenceError, TypeError):
        pass
    if context.area is not None:
        context.area.tag_redraw()


class BAW_OT_rigped_semantic_move_axis(bpy.types.Operator):
    bl_idname = "baw.rigped_semantic_move_axis"
    bl_label = "Move Rigped Semantic Limb"
    bl_description = "Move one supported Rigped limb through the AWB semantic transform boundary"
    # This is a custom modal transform. Blender's automatic UNDO option did not
    # reliably create a separate history step for the live pose edits performed
    # during MOUSEMOVE; Ctrl+Z could therefore jump back to the preceding C
    # Contact-authoring operation and turn Sliding back into Free. Commit one
    # explicit checkpoint on successful release instead.
    bl_options: ClassVar[set[str]] = {"REGISTER", "BLOCKING"}

    axis: EnumProperty(
        items=(
            ("X", "X", "Move along X"),
            ("Y", "Y", "Move along Y"),
            ("Z", "Z", "Move along Z"),
            ("XY", "XY", "Move in the XY plane"),
            ("XZ", "XZ", "Move in the XZ plane"),
            ("YZ", "YZ", "Move in the YZ plane"),
            ("FREE", "Free", "Move in the current view plane"),
        ),
        default="X",
    )

    _session: SemanticMoveSession | None = None
    _sessions: tuple[SlidingMoveSession, ...] = ()
    _fk_sessions: tuple[FkJointMoveSession, ...] = ()
    _resolutions: tuple[SemanticMoveResolution, ...] = ()
    _auto_plan: RigpedAutoAnchorPlan | None = None
    _auto_direct_plan: RigpedAutoDirectRotatePlan | None = None
    _auto_contact_batch_plan: RigpedAutoContactBatchPlan | None = None
    _axis = Vector((1.0, 0.0, 0.0))
    _pivot = Vector((0.0, 0.0, 0.0))
    _move_screen_axis = Vector((1.0, 0.0))
    _move_start_screen_projection = 0.0
    _move_world_per_pixel = 0.0
    _move_use_screen_axis = False
    _move_use_screen_plane = False
    _move_start_mouse = Vector((0.0, 0.0))
    _move_plane_u = Vector((1.0, 0.0, 0.0))
    _move_plane_v = Vector((0.0, 1.0, 0.0))
    _move_plane_screen_u = Vector((1.0, 0.0))
    _move_plane_screen_v = Vector((0.0, 1.0))
    _move_plane_reference = 1.0
    _move_plane_det = 1.0
    _last_delta = Vector((0.0, 0.0, 0.0))
    _move_last_mouse = Vector((0.0, 0.0))
    _trace_operation_id: str | None = None

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "mode", "") == "POSE"
            and getattr(
                context.scene,
                "baw_rigped_semantic_transform_mode",
                "NONE",
            )
            in {"MOVE", "FK_MOVE", "DIRECT_MOVE"}
            and semantic_move_available(context)
        )

    def invoke(self, context, event):
        axes = semantic_move_axes(context)
        pivot = semantic_move_pivot(context)
        if axes is None or pivot is None:
            self.report({"WARNING"}, text("rigped.transform.orientation_unsupported", context))
            return {"CANCELLED"}
        try:
            resolutions, sessions, fk_sessions = _begin_semantic_move_domains(context)
        except RigpedSemanticMoveError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        if not sessions:
            cancel_fk_joint_moves(context, fk_sessions)
            self.report({"WARNING"}, "Semantic Move requires at least one Sliding limb domain.")
            return {"CANCELLED"}
        session = sessions[0]

        self._auto_plan = None
        self._auto_direct_plan = None
        self._auto_contact_batch_plan = None
        if bool(getattr(context.scene, "baw_auto_key_enabled", False)):
            auto_context = control_context_for_context(context)
            auto_limb_context, auto_direct_context = _fk_move_auto_contexts(
                context.scene,
                auto_context,
                fk_sessions,
            )
            mapping_ids = tuple(
                str(resolved.capability.native_ik.mapping_id)
                for resolved in resolutions
            )
            if len(mapping_ids) >= 2:
                planned_batch = plan_rigped_auto_contact_batch(
                    context.scene,
                    auto_limb_context,
                    operation_id=f"ak4:mixed-move-auto:{uuid4().hex}",
                    mapping_ids=mapping_ids,
                )
                if not planned_batch.ok or planned_batch.plan is None:
                    cancel_semantic_move_domains(context, sessions, fk_sessions)
                    detail = (
                        planned_batch.diagnostics[0].detail
                        if planned_batch.diagnostics
                        else "Rigped mixed Free/Sliding Move Auto planning failed."
                    )
                    self.report({"WARNING"}, detail)
                    return {"CANCELLED"}
                self._auto_contact_batch_plan = planned_batch.plan
            else:
                planned = plan_rigped_auto_anchor(
                    context.scene,
                    auto_limb_context,
                    operation_id=f"ak4:sliding-move-auto:{uuid4().hex}",
                    mapping_id=session.intent.mapping_id,
                )
                if not planned.ok or planned.plan is None:
                    cancel_semantic_move_domains(context, sessions, fk_sessions)
                    detail = (
                        planned.diagnostics[0].detail
                        if planned.diagnostics
                        else "Rigped Sliding Move Auto planning failed."
                    )
                    self.report({"WARNING"}, detail)
                    return {"CANCELLED"}
                if planned.plan.intent.target_type is not ContactKeyType.SLIDING:
                    cancel_semantic_move_domains(context, sessions, fk_sessions)
                    self.report({"WARNING"}, "Sliding Move Auto must preserve Sliding authority.")
                    return {"CANCELLED"}
                self._auto_plan = planned.plan

            if auto_direct_context is not None:
                direct = plan_rigped_auto_direct_rotate(
                    context.scene,
                    auto_direct_context,
                    operation_id=f"ak4:mixed-move-auto:{uuid4().hex}:direct",
                    active_only=len(auto_direct_context.controls) == 1,
                )
                if not direct.ok or direct.plan is None:
                    cancel_semantic_move_domains(context, sessions, fk_sessions)
                    detail = (
                        direct.diagnostics[0].detail
                        if direct.diagnostics
                        else "Rigped direct mixed Move Auto planning failed."
                    )
                    self.report({"WARNING"}, detail)
                    return {"CANCELLED"}
                self._auto_direct_plan = direct.plan

        self._sessions = sessions
        self._fk_sessions = fk_sessions
        self._resolutions = resolutions
        self._session = session
        self._pivot = Vector(pivot)
        self._last_delta = Vector((0.0, 0.0, 0.0))
        self._move_last_mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self._move_use_screen_axis = False
        self._move_use_screen_plane = False
        self._move_world_per_pixel = 0.0

        reference_world = 0.3
        active_control = _semantic_move_active_control(context)
        if active_control is not None and isinstance(active_control.target, bpy.types.PoseBone):
            pose_bone = active_control.target
            reference_world = float(
                Vector(
                    active_control.owner_object.matrix_world.to_3x3()
                    @ (pose_bone.tail - pose_bone.head)
                ).length
            )
        reference_world = max(0.05, reference_world)

        region_data = getattr(context, "region_data", None)
        if region_data is None:
            region_data = getattr(getattr(context, "space_data", None), "region_3d", None)

        if self.axis in {"X", "Y", "Z"}:
            self._axis = Vector(axes[self.axis]).normalized()
            if context.region is not None and region_data is not None:
                pivot_2d = location_3d_to_region_2d(context.region, region_data, self._pivot)
                sample_2d = location_3d_to_region_2d(
                    context.region,
                    region_data,
                    self._pivot + self._axis * reference_world,
                )
                if pivot_2d is not None and sample_2d is not None:
                    screen_delta = Vector(sample_2d) - Vector(pivot_2d)
                    if screen_delta.length >= 6.0:
                        self._move_screen_axis = screen_delta.normalized()
                        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
                        self._move_start_screen_projection = float(
                            mouse.dot(self._move_screen_axis)
                        )
                        world_per_pixel = projection_world_per_pixel(
                            context, self._pivot
                        )
                        if world_per_pixel is not None:
                            self._move_world_per_pixel = float(world_per_pixel)
                            self._move_use_screen_axis = True
            if not self._move_use_screen_axis:
                cancel_semantic_move_domains(context, sessions, fk_sessions)
                self._session = None
                self._sessions = ()
                self._fk_sessions = ()
                self._resolutions = ()
                self.report({"WARNING"}, f"Rigped {self.axis} move axis is edge-on to this view")
                return {"CANCELLED"}
        else:
            if self.axis == "FREE":
                if region_data is None:
                    cancel_semantic_move_domains(context, sessions, fk_sessions)
                    self._session = None
                    self._sessions = ()
                    self._fk_sessions = ()
                    self._resolutions = ()
                    return {"CANCELLED"}
                view_basis = region_data.view_matrix.inverted().to_3x3().normalized()
                first = Vector(view_basis.col[0]).normalized()
                second = Vector(view_basis.col[1]).normalized()
            else:
                plane_axes = {
                    "XY": ("X", "Y"),
                    "XZ": ("X", "Z"),
                    "YZ": ("Y", "Z"),
                }
                first_name, second_name = plane_axes[self.axis]
                first = Vector(axes[first_name]).normalized()
                second = Vector(axes[second_name]).normalized()

            self._move_plane_u = first
            self._move_plane_v = second
            self._move_plane_reference = reference_world
            self._move_start_mouse = Vector((event.mouse_region_x, event.mouse_region_y))

            if context.region is not None and region_data is not None:
                pivot_2d = location_3d_to_region_2d(context.region, region_data, self._pivot)
                u_2d = location_3d_to_region_2d(
                    context.region,
                    region_data,
                    self._pivot + first * reference_world,
                )
                v_2d = location_3d_to_region_2d(
                    context.region,
                    region_data,
                    self._pivot + second * reference_world,
                )
                if pivot_2d is not None and u_2d is not None and v_2d is not None:
                    screen_u = Vector(u_2d) - Vector(pivot_2d)
                    screen_v = Vector(v_2d) - Vector(pivot_2d)
                    det = float(screen_u.x * screen_v.y - screen_u.y * screen_v.x)
                    lengths_2d = float(screen_u.length * screen_v.length)
                    conditioning = abs(det) / max(lengths_2d, 1e-9)
                    if (
                        screen_u.length >= 6.0
                        and screen_v.length >= 6.0
                        and conditioning >= 0.18
                    ):
                        self._move_plane_screen_u = screen_u
                        self._move_plane_screen_v = screen_v
                        self._move_plane_det = det
                        self._move_use_screen_plane = True
            if not self._move_use_screen_plane:
                cancel_semantic_move_domains(context, sessions, fk_sessions)
                self._session = None
                self._sessions = ()
                self._fk_sessions = ()
                self._resolutions = ()
                return {"CANCELLED"}

        route_name = "HYBRID_MOVE" if fk_sessions else "SLIDING_MOVE"
        self._trace_operation_id = new_trace_operation_id(
            "hybrid-move" if fk_sessions else "sliding-move"
        )
        bind_operation_causal_context(
            self._trace_operation_id,
            new_trace_causal_root("hybrid-move" if fk_sessions else "sliding-move"),
        )
        trace_event(
            "OPERATION",
            "TRANSFORM_BEGIN",
            operation_id=self._trace_operation_id,
            subsystem="modal",
            lifecycle_phase="invoke",
            route_outcome=TraceRouteOutcome.CLAIMED,
            context=context,
            tool="MOVE",
            route=route_name,
            axis=self.axis,
            authority="MIXED" if fk_sessions else "SLIDING",
            mapping_ids=tuple(
                str(resolved.capability.native_ik.mapping_id)
                for resolved in resolutions
            ),
            auto_key=bool(getattr(context.scene, "baw_auto_key_enabled", False)),
        )
        _set_semantic_move_drag_active(context, True)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        session = self._session
        sessions = self._sessions
        fk_sessions = self._fk_sessions
        resolutions = self._resolutions
        if session is None or not sessions or not resolutions:
            _set_semantic_move_drag_active(context, False)
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            mouse_now = Vector((event.mouse_region_x, event.mouse_region_y))
            pixel_step = float((mouse_now - self._move_last_mouse).length)
            delta = None
            if self.axis in {"X", "Y", "Z"}:
                projection = float(mouse_now.dot(self._move_screen_axis))
                distance = (
                    projection - self._move_start_screen_projection
                ) * self._move_world_per_pixel
                delta = self._axis * distance
            else:
                mouse_delta = mouse_now - self._move_start_mouse
                screen_u = self._move_plane_screen_u
                screen_v = self._move_plane_screen_v
                det = self._move_plane_det
                alpha = float(
                    (mouse_delta.x * screen_v.y - mouse_delta.y * screen_v.x) / det
                )
                beta = float(
                    (screen_u.x * mouse_delta.y - screen_u.y * mouse_delta.x) / det
                )
                delta = (
                    self._move_plane_u * (self._move_plane_reference * alpha)
                    + self._move_plane_v * (self._move_plane_reference * beta)
                )

            stable_delta = Vector(delta)
            if pixel_step <= 3.0:
                stable_delta = self._last_delta.lerp(stable_delta, 0.68)
            try:
                applied = apply_semantic_move_domains_delta(
                    context,
                    resolutions,
                    sessions,
                    fk_sessions,
                    stable_delta,
                )
            except RigpedSemanticMoveError as exc:
                self.report({"WARNING"}, str(exc))
                applied = False
            if applied:
                self._last_delta = stable_delta
            else:
                self.report({"WARNING"}, "Rigped Sliding Move target is outside the supported exact solve domain")
            self._move_last_mouse = mouse_now
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self._last_delta.length <= 1e-9:
                cancel_semantic_move_domains(context, sessions, fk_sessions)
                self._session = None
                self._sessions = ()
                self._fk_sessions = ()
                self._resolutions = ()
                self._auto_plan = None
                self._auto_direct_plan = None
                self._auto_contact_batch_plan = None
                _set_semantic_move_drag_active(context, False)
                _refresh_after_semantic_move(context)
                trace_event(
                    "OPERATION",
                    "TRANSFORM_CANCEL",
                    operation_id=self._trace_operation_id,
                    subsystem="modal",
                    lifecycle_phase="cancel",
                    terminal_status=TraceTerminalStatus.CANCELLED,
                    route_outcome=TraceRouteOutcome.CLAIMED,
                    context=context,
                    reason="NO_DELTA",
                )
                self._trace_operation_id = None
                return {"CANCELLED"}

            success = True
            keyed = False
            diagnostics: tuple[Any, ...] = ()
            auto_enabled_at_release = bool(
                getattr(context.scene, "baw_auto_key_enabled", False)
            )
            auto_contact_batch_plan = (
                self._auto_contact_batch_plan if auto_enabled_at_release else None
            )
            auto_plan = self._auto_plan if auto_enabled_at_release else None
            auto_direct_plan = (
                self._auto_direct_plan if auto_enabled_at_release else None
            )
            deferred_auto_results: list[Any] = []
            auto_context = control_context_for_context(context)
            auto_limb_context, auto_direct_context = _fk_move_auto_contexts(
                context.scene,
                auto_context,
                fk_sessions,
            )
            try:
                if auto_contact_batch_plan is not None:
                    link_trace_operation(
                        auto_contact_batch_plan.batch.operation_id,
                        self._trace_operation_id,
                    )
                    auto_result = commit_rigped_auto_contact_batch(
                        context.scene,
                        auto_limb_context,
                        auto_contact_batch_plan,
                        defer_commit=True,
                    )
                    success = bool(auto_result.applied)
                    keyed = bool(auto_result.applied)
                    diagnostics = tuple(auto_result.diagnostics)
                    if auto_result.applied:
                        deferred_auto_results.append(auto_result)
                elif auto_plan is not None:
                    link_trace_operation(
                        auto_plan.intent.operation_id,
                        self._trace_operation_id,
                    )
                    auto_result = commit_rigped_auto_anchor(
                        context.scene,
                        auto_limb_context,
                        auto_plan,
                        defer_commit=True,
                    )
                    success = bool(auto_result.applied)
                    keyed = bool(auto_result.applied)
                    diagnostics = tuple(auto_result.diagnostics)
                    if auto_result.applied:
                        deferred_auto_results.append(auto_result)

                if success and auto_direct_plan is not None:
                    direct_plan = auto_direct_plan
                    if auto_plan is not None or auto_contact_batch_plan is not None:
                        direct_plan = replace(direct_plan, allow_storage_rebind=True)
                    link_trace_operation(
                        direct_plan.begin_plan.operation_id,
                        self._trace_operation_id,
                    )
                    direct_result = commit_rigped_auto_direct_rotate(
                        context.scene,
                        auto_direct_context,
                        direct_plan,
                        defer_commit=True,
                    )
                    success = bool(direct_result.applied)
                    keyed = keyed or bool(direct_result.applied)
                    diagnostics = diagnostics + tuple(direct_result.diagnostics)
                    if direct_result.applied:
                        deferred_auto_results.append(direct_result)
                    if success:
                        _refresh_current_sliding_public_overlays(context)

                if success:
                    commit_rigped_auto_writer_results(tuple(deferred_auto_results))
            except (ContactAuthoringError, RuntimeError, ValueError, ReferenceError) as exc:
                rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                cancel_semantic_move_domains(context, sessions, fk_sessions)
                _report_operator_error(self, context, exc)
                success = False

            if not success:
                rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                cancel_semantic_move_domains(context, sessions, fk_sessions)

            self._session = None
            self._sessions = ()
            self._fk_sessions = ()
            self._resolutions = ()
            self._auto_plan = None
            self._auto_direct_plan = None
            self._auto_contact_batch_plan = None
            _set_semantic_move_drag_active(context, False)
            if keyed:
                frame = float(context.scene.frame_current) + float(
                    getattr(context.scene, "frame_subframe", 0.0)
                )
                clear_contact_bundle_key_selection_for_context(context, frames=(frame,))
                from .trackbar_model import clear_key_selection_for_context

                clear_key_selection_for_context(context)
                context.scene.baw_has_selected_key = False
            _refresh_after_semantic_move(context)
            if not success:
                detail = "; ".join(item.detail for item in diagnostics)
                trace_event(
                    "OPERATION",
                    "TRANSFORM_FAIL",
                    operation_id=self._trace_operation_id,
                    subsystem="modal",
                    lifecycle_phase="fail",
                    terminal_status=TraceTerminalStatus.CANCELLED,
                    route_outcome=TraceRouteOutcome.CLAIMED,
                    context=context,
                    detail=detail or "Rigped Sliding Move commit failed",
                    final_delta=tuple(float(value) for value in self._last_delta),
                )
                self._trace_operation_id = None
                self.report({"WARNING"}, detail or "Rigped Sliding Move commit failed")
                return {"CANCELLED"}
            route_name = "HYBRID_MOVE" if fk_sessions else "SLIDING_MOVE"
            bpy.ops.ed.undo_push(
                message=(
                    "AWB Rigped Mixed Move"
                    if fk_sessions
                    else "AWB Rigped Sliding Move"
                )
            )
            replay_delta = tuple(float(value) for value in self._last_delta)
            trace_event(
                "OPERATION",
                "TRANSFORM_COMMIT",
                operation_id=self._trace_operation_id,
                subsystem="modal",
                lifecycle_phase="commit",
                terminal_status=TraceTerminalStatus.FINISHED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
                tool="MOVE",
                route=route_name,
                final_delta=replay_delta,
                keyed=keyed,
                replay_action={
                    "kind": "MOVE",
                    "route": route_name,
                    "frame": int(context.scene.frame_current),
                    "subframe": float(getattr(context.scene, "frame_subframe", 0.0)),
                    "controls": tuple(
                        str(item.name)
                        for item in tuple(getattr(context, "selected_pose_bones", ()) or ())
                    ),
                    "axis": self.axis,
                    "delta_world": replay_delta,
                },
            )
            self._trace_operation_id = None
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            cancel_semantic_move_domains(context, sessions, fk_sessions)
            self._session = None
            self._sessions = ()
            self._fk_sessions = ()
            self._resolutions = ()
            self._auto_plan = None
            self._auto_contact_batch_plan = None
            _set_semantic_move_drag_active(context, False)
            _refresh_after_semantic_move(context)
            trace_event(
                "OPERATION",
                "TRANSFORM_CANCEL",
                operation_id=self._trace_operation_id,
                subsystem="modal",
                lifecycle_phase="cancel",
                terminal_status=TraceTerminalStatus.CANCELLED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
                reason=event.type,
            )
            self._trace_operation_id = None
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


def _view_axis(context) -> Vector | None:
    region_data = getattr(context, "region_data", None)
    if region_data is None:
        region_data = getattr(getattr(context, "space_data", None), "region_3d", None)
    if region_data is None:
        return None
    return Vector(region_data.view_matrix.inverted().to_3x3().col[2]).normalized()


def _subset_control_context(control_context, controls):
    controls = tuple(controls)
    if not controls:
        return None
    selected_keys = {runtime_control_key(control) for control in controls}
    active = control_context.active
    if active is None or runtime_control_key(active) not in selected_keys:
        active = controls[0]
    return replace(control_context, controls=controls, active=active)


def _fk_move_auto_contexts(scene, control_context, sessions):
    """Partition mixed FK Move selection into limb-contact and direct domains."""

    limb_keys = selected_contact_limb_control_keys(scene, control_context)
    visual_only_keys = {
        runtime_control_key(session.active_control)
        for session in sessions
        if isinstance(session, FkSingleLinkMoveSession) and session.visual_only
    }
    limb_controls = tuple(
        control
        for control in control_context.controls
        if runtime_control_key(control) in limb_keys
    )
    direct_controls = tuple(
        control
        for control in control_context.controls
        if runtime_control_key(control) not in limb_keys
        and runtime_control_key(control) not in visual_only_keys
    )
    return (
        _subset_control_context(control_context, limb_controls),
        _subset_control_context(control_context, direct_controls),
    )


def _direct_rotate_auto_contexts(scene, control_context):
    """Partition mixed Rotate selection into Contact-limb and direct domains."""

    limb_keys = selected_contact_limb_control_keys(scene, control_context)
    limb_controls = tuple(
        control
        for control in control_context.controls
        if runtime_control_key(control) in limb_keys
    )
    direct_controls = tuple(
        control
        for control in control_context.controls
        if runtime_control_key(control) not in limb_keys
    )
    return (
        _subset_control_context(control_context, limb_controls),
        _subset_control_context(control_context, direct_controls),
    )


class BAW_OT_rigped_fk_joint_move_axis(bpy.types.Operator):
    """Biped-style Free/FK limb Move without authoring raw Location."""

    bl_idname = "baw.rigped_fk_joint_move_axis"
    bl_label = "Move Rigped FK Joint"
    # This custom modal owns both live FK pose edits and release-time Auto/Contact
    # authoring. Blender's automatic UNDO boundary can split those writes into
    # separate history states, so commit one explicit checkpoint after the whole
    # gesture succeeds instead.
    bl_options: ClassVar[set[str]] = {"REGISTER", "BLOCKING"}

    axis: EnumProperty(
        items=(
            ("X", "X", "Move along X"),
            ("Y", "Y", "Move along Y"),
            ("Z", "Z", "Move along Z"),
            ("XY", "XY", "Move in the XY plane"),
            ("XZ", "XZ", "Move in the XZ plane"),
            ("YZ", "YZ", "Move in the YZ plane"),
            ("FREE", "Free", "Move in the current view plane"),
        ),
        default="X",
    )

    _sessions: tuple[FkJointMoveSession, ...] = ()
    _auto_plan: RigpedAutoAnchorPlan | None = None
    _auto_direct_plan: RigpedAutoDirectRotatePlan | None = None
    _auto_contact_batch_plan: RigpedAutoContactBatchPlan | None = None
    _deferred_auto_contact_mapping_ids: tuple[str, ...] = ()
    _axis = Vector((1.0, 0.0, 0.0))
    _pivot = Vector((0.0, 0.0, 0.0))
    _start_axis_point = Vector((0.0, 0.0, 0.0))
    _move_screen_axis = Vector((1.0, 0.0))
    _move_start_screen_projection = 0.0
    _move_world_per_pixel = 0.0
    _move_use_screen_axis = False
    _move_use_screen_plane = False
    _move_start_mouse = Vector((0.0, 0.0))
    _move_plane_u = Vector((1.0, 0.0, 0.0))
    _move_plane_v = Vector((0.0, 1.0, 0.0))
    _move_plane_screen_u = Vector((1.0, 0.0))
    _move_plane_screen_v = Vector((0.0, 1.0))
    _move_plane_reference = 1.0
    _move_plane_det = 1.0
    _plane_normal = Vector((0.0, 0.0, 1.0))
    _start_plane_point = Vector((0.0, 0.0, 0.0))
    _last_delta = Vector((0.0, 0.0, 0.0))
    _move_last_mouse = Vector((0.0, 0.0))
    _trace_operation_id: str | None = None
    _motion_previous: ClassVar[
        dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]
    ] = {}
    _motion_min_step_ratio: ClassVar[dict[str, float]] = {}
    _motion_max_reach_ratio: ClassVar[dict[str, float]] = {}

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "mode", "") == "POSE"
            and getattr(
                context.scene,
                "baw_rigped_semantic_transform_mode",
                "NONE",
            )
            in {"MOVE", "FK_MOVE", "DIRECT_MOVE"}
            and fk_joint_move_available(context)
        )

    def invoke(self, context, event):
        axes = fk_joint_move_axes(context)
        pivot = fk_joint_move_pivot(context)
        if axes is None or pivot is None:
            return {"CANCELLED"}
        try:
            sessions = begin_fk_joint_moves(context)
        except RigpedSemanticMoveError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}

        self._auto_plan = None
        self._auto_direct_plan = None
        self._auto_contact_batch_plan = None
        self._deferred_auto_contact_mapping_ids = ()
        if bool(getattr(context.scene, "baw_auto_key_enabled", False)):
            targets = _selected_fk_joint_move_targets(context)
            if len(targets) != len(sessions):
                self.report({"WARNING"}, "Rigped FK Move Auto target changed during preflight.")
                return {"CANCELLED"}

            auto_context = control_context_for_context(context)
            operation_id = f"ak3:fk-move-auto:{uuid4().hex}"
            limb_context, direct_context = _fk_move_auto_contexts(
                context.scene,
                auto_context,
                sessions,
            )
            mapping_ids = (
                selected_contact_mapping_ids(context.scene, limb_context)
                if limb_context is not None
                else ()
            )
            if limb_context is not None and not mapping_ids:
                self.report(
                    {"WARNING"},
                    "Rigped FK Move could not resolve the selected limb domains.",
                )
                return {"CANCELLED"}

            if len(mapping_ids) >= 2:
                # AUTO must never block the already-valid FK preview. For a
                # multi-limb gesture, defer semantic batch planning until the
                # transform has actually succeeded on release so the writer
                # sees the final solved pose (especially the first 0F key).
                self._deferred_auto_contact_mapping_ids = tuple(mapping_ids)
            elif mapping_ids:
                auto = plan_rigped_auto_anchor(
                    context.scene,
                    limb_context,
                    operation_id=f"{operation_id}:contact",
                    mapping_id=mapping_ids[0],
                )
                if not auto.ok or auto.plan is None:
                    detail = (
                        auto.diagnostics[0].detail
                        if auto.diagnostics
                        else "Rigped Free FK Move Auto planning failed."
                    )
                    self.report({"WARNING"}, detail)
                    return {"CANCELLED"}
                if auto.plan.intent.target_type is not ContactKeyType.FREE:
                    self.report({"WARNING"}, "AK3 FK Move Auto accepts Free limb authority only.")
                    return {"CANCELLED"}
                self._auto_plan = auto.plan

            if direct_context is not None:
                direct = plan_rigped_auto_direct_rotate(
                    context.scene,
                    direct_context,
                    operation_id=f"{operation_id}:direct",
                    active_only=False,
                )
                if not direct.ok or direct.plan is None:
                    detail = (
                        direct.diagnostics[0].detail
                        if direct.diagnostics
                        else "Rigped direct FK Move Auto planning failed."
                    )
                    self.report({"WARNING"}, detail)
                    return {"CANCELLED"}
                self._auto_direct_plan = direct.plan

        self._sessions = sessions
        active_session = sessions[0]
        self._pivot = Vector(pivot)
        self._last_delta = Vector((0.0, 0.0, 0.0))
        self._move_last_mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self._motion_previous = {}
        self._motion_min_step_ratio = {}
        self._motion_max_reach_ratio = {}
        self._move_use_screen_axis = False
        self._move_use_screen_plane = False
        self._move_world_per_pixel = 0.0
        # This custom kinematic modal owns the full FK closure transaction.
        # Suppress the ordinary per-depsgraph Auto Key hot path until release.
        _set_semantic_move_drag_active(context, True)

        reference_world = (
            float((active_session.start_tail_world - active_session.start_head_world).length)
            if isinstance(active_session, FkSingleLinkMoveSession)
            else max(float(active_session.first_length), float(active_session.second_length))
        )
        reference_world = max(0.05, reference_world)
        region_data = getattr(context, "region_data", None)
        if region_data is None:
            region_data = getattr(getattr(context, "space_data", None), "region_3d", None)

        if self.axis in {"X", "Y", "Z"}:
            self._axis = Vector(axes[self.axis]).normalized()
            if context.region is not None and region_data is not None:
                pivot_2d = location_3d_to_region_2d(
                    context.region,
                    region_data,
                    self._pivot,
                )
                sample_2d = location_3d_to_region_2d(
                    context.region,
                    region_data,
                    self._pivot + self._axis * reference_world,
                )
                if pivot_2d is not None and sample_2d is not None:
                    screen_delta = Vector(sample_2d) - Vector(pivot_2d)
                    if screen_delta.length >= 6.0:
                        self._move_screen_axis = screen_delta.normalized()
                        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
                        self._move_start_screen_projection = float(
                            mouse.dot(self._move_screen_axis)
                        )
                        world_per_pixel = projection_world_per_pixel(
                            context, self._pivot
                        )
                        if world_per_pixel is not None:
                            self._move_world_per_pixel = float(world_per_pixel)
                            self._move_use_screen_axis = True
            if not self._move_use_screen_axis:
                # Do not fall back to ray-vs-axis closest points here. When the
                # axis is nearly view-aligned that solution is ill-conditioned:
                # a one-pixel mouse change can become a large world-space jump.
                # Refuse the edge-on drag instead of letting W chatter or flip.
                cancel_fk_joint_moves(context, sessions)
                self._sessions = ()
                _set_semantic_move_drag_active(context, False)
                return {"CANCELLED"}
        else:
            if self.axis == "FREE":
                if region_data is None:
                    cancel_fk_joint_moves(context, sessions)
                    self._sessions = ()
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                view_basis = region_data.view_matrix.inverted().to_3x3().normalized()
                first = Vector(view_basis.col[0]).normalized()
                second = Vector(view_basis.col[1]).normalized()
            else:
                plane_axes = {
                    "XY": ("X", "Y"),
                    "XZ": ("X", "Z"),
                    "YZ": ("Y", "Z"),
                }
                first_name, second_name = plane_axes[self.axis]
                first = Vector(axes[first_name]).normalized()
                second = Vector(axes[second_name]).normalized()
            normal = first.cross(second)
            if normal.length <= 1e-9:
                cancel_fk_joint_moves(context, sessions)
                self._sessions = ()
                _set_semantic_move_drag_active(context, False)
                return {"CANCELLED"}
            self._plane_normal = normal.normalized()
            self._move_plane_u = first
            self._move_plane_v = second
            self._move_plane_reference = reference_world
            self._move_start_mouse = Vector((event.mouse_region_x, event.mouse_region_y))

            if context.region is not None and region_data is not None:
                pivot_2d = location_3d_to_region_2d(context.region, region_data, self._pivot)
                u_2d = location_3d_to_region_2d(
                    context.region,
                    region_data,
                    self._pivot + first * reference_world,
                )
                v_2d = location_3d_to_region_2d(
                    context.region,
                    region_data,
                    self._pivot + second * reference_world,
                )
                if pivot_2d is not None and u_2d is not None and v_2d is not None:
                    screen_u = Vector(u_2d) - Vector(pivot_2d)
                    screen_v = Vector(v_2d) - Vector(pivot_2d)
                    det = float(screen_u.x * screen_v.y - screen_u.y * screen_v.x)
                    lengths = float(screen_u.length * screen_v.length)
                    conditioning = abs(det) / max(lengths, 1e-9)
                    if (
                        screen_u.length >= 6.0
                        and screen_v.length >= 6.0
                        and conditioning >= 0.18
                    ):
                        self._move_plane_screen_u = screen_u
                        self._move_plane_screen_v = screen_v
                        self._move_plane_det = det
                        self._move_use_screen_plane = True

            if not self._move_use_screen_plane:
                # Inverting two nearly collinear projected axes magnifies tiny
                # cursor noise. Ray-plane fallback has the same singularity when
                # the plane is edge-on, so fail closed and let the user orbit a
                # little instead of producing a visible W tremble/jump.
                cancel_fk_joint_moves(context, sessions)
                self._sessions = ()
                _set_semantic_move_drag_active(context, False)
                return {"CANCELLED"}

        self._trace_operation_id = new_trace_operation_id("fk-move")
        bind_operation_causal_context(
            self._trace_operation_id,
            new_trace_causal_root("fk-move"),
        )
        trace_event(
            "OPERATION",
            "TRANSFORM_BEGIN",
            operation_id=self._trace_operation_id,
            subsystem="modal",
            lifecycle_phase="invoke",
            route_outcome=TraceRouteOutcome.CLAIMED,
            context=context,
            tool="MOVE",
            route="FK_MOVE",
            axis=self.axis,
            authority="FREE",
            auto_key=bool(getattr(context.scene, "baw_auto_key_enabled", False)),
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        sessions = self._sessions
        if not sessions:
            _set_semantic_move_drag_active(context, False)
            return {"CANCELLED"}
        if event.type == "MOUSEMOVE":
            mouse_now = Vector((event.mouse_region_x, event.mouse_region_y))
            pixel_step = float((mouse_now - self._move_last_mouse).length)
            delta = None
            if self.axis in {"X", "Y", "Z"}:
                if self._move_use_screen_axis:
                    mouse = Vector((event.mouse_region_x, event.mouse_region_y))
                    projection = float(mouse.dot(self._move_screen_axis))
                    distance = (
                        projection - self._move_start_screen_projection
                    ) * self._move_world_per_pixel
                    delta = self._axis * distance
                else:
                    current = _axis_point(
                        context,
                        self._pivot,
                        self._axis,
                        event.mouse_region_x,
                        event.mouse_region_y,
                    )
                    if current is not None:
                        distance = float(
                            (Vector(current) - self._start_axis_point).dot(self._axis)
                        )
                        delta = self._axis * distance
            else:
                if self._move_use_screen_plane:
                    mouse_delta = Vector(
                        (event.mouse_region_x, event.mouse_region_y)
                    ) - self._move_start_mouse
                    screen_u = self._move_plane_screen_u
                    screen_v = self._move_plane_screen_v
                    det = self._move_plane_det
                    alpha = float(
                        (mouse_delta.x * screen_v.y - mouse_delta.y * screen_v.x)
                        / det
                    )
                    beta = float(
                        (screen_u.x * mouse_delta.y - screen_u.y * mouse_delta.x)
                        / det
                    )
                    delta = (
                        self._move_plane_u * (self._move_plane_reference * alpha)
                        + self._move_plane_v * (self._move_plane_reference * beta)
                    )
                else:
                    current = _plane_point(
                        context,
                        self._pivot,
                        self._plane_normal,
                        event.mouse_region_x,
                        event.mouse_region_y,
                    )
                    if current is not None:
                        delta = Vector(current) - self._start_plane_point
            if delta is not None:
                stable_delta = Vector(delta)
                if pixel_step <= 3.0:
                    stable_delta = self._last_delta.lerp(stable_delta, 0.68)
                if apply_fk_joint_moves_delta(context, sessions, stable_delta):
                    self._last_delta = stable_delta
                    for sample in _fk_move_motion_response_samples(
                        context,
                        sessions,
                        stable_delta,
                    ):
                        domain = str(sample["domain"])
                        requested = Vector(sample["requested_delta"])
                        actual = Vector(sample["actual_delta"])
                        previous = self._motion_previous.get(domain)
                        if previous is not None:
                            previous_requested = Vector(previous[0])
                            previous_actual = Vector(previous[1])
                            requested_step = requested - previous_requested
                            actual_step = actual - previous_actual
                            requested_step_length = float(requested_step.length)
                            if requested_step_length > 1e-5:
                                direction = requested_step / requested_step_length
                                step_ratio = float(
                                    actual_step.dot(direction) / requested_step_length
                                )
                                previous_min = self._motion_min_step_ratio.get(domain)
                                if previous_min is None or step_ratio < previous_min:
                                    self._motion_min_step_ratio[domain] = step_ratio
                        self._motion_previous[domain] = (
                            tuple(float(value) for value in requested),
                            tuple(float(value) for value in actual),
                        )
                        reach_ratio = sample.get("reach_ratio")
                        if reach_ratio is not None:
                            self._motion_max_reach_ratio[domain] = max(
                                float(reach_ratio),
                                self._motion_max_reach_ratio.get(domain, 0.0),
                            )
                self._move_last_mouse = mouse_now
                if context.area is not None:
                    context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self._last_delta.length <= 1e-9:
                cancel_fk_joint_moves(context, sessions)
                self._sessions = ()
                self._auto_plan = None
                self._auto_direct_plan = None
                self._auto_contact_batch_plan = None
                _set_semantic_move_drag_active(context, False)
                trace_event(
                    "OPERATION",
                    "TRANSFORM_CANCEL",
                    operation_id=self._trace_operation_id,
                    subsystem="modal",
                    lifecycle_phase="cancel",
                    terminal_status=TraceTerminalStatus.CANCELLED,
                    route_outcome=TraceRouteOutcome.CLAIMED,
                    context=context,
                    reason="NO_DELTA",
                )
                self._trace_operation_id = None
                return {"CANCELLED"}

            # Capture the exact pre-gesture Blender state only for a successful
            # release. Frame navigation and the AWB AUTO toggle intentionally do
            # not create undo checkpoints, so an older global checkpoint can
            # otherwise make Ctrl+Z jump back past the current frame/AUTO state.
            # Restore the preview, checkpoint that exact state, then replay the
            # already-validated final delta before release-time authoring.
            final_delta = Vector(self._last_delta)
            cancel_fk_joint_moves(context, sessions)
            bpy.ops.ed.undo_push(message="AWB Rigped FK Move Start")
            if not apply_fk_joint_moves_delta(context, sessions, final_delta):
                self._sessions = ()
                self._auto_plan = None
                self._auto_direct_plan = None
                self._auto_contact_batch_plan = None
                _set_semantic_move_drag_active(context, False)
                self.report({"WARNING"}, "Rigped FK Move final replay failed.")
                return {"CANCELLED"}
            self._last_delta = final_delta
            replay_delta = tuple(float(value) for value in final_delta)
            replay_action = {
                "kind": "MOVE",
                "route": "FK_MOVE",
                "frame": int(context.scene.frame_current),
                "subframe": float(getattr(context.scene, "frame_subframe", 0.0)),
                "controls": tuple(
                    str(item.name)
                    for item in tuple(getattr(context, "selected_pose_bones", ()) or ())
                ),
                "axis": self.axis,
                "delta_world": replay_delta,
            }
            motion_response = []
            for sample in _fk_move_motion_response_samples(
                context,
                sessions,
                final_delta,
            ):
                domain = str(sample["domain"])
                row = dict(sample)
                row["min_step_ratio"] = self._motion_min_step_ratio.get(domain)
                row["max_reach_ratio"] = self._motion_max_reach_ratio.get(
                    domain,
                    sample.get("reach_ratio"),
                )
                row["response_ratio"] = (
                    row["min_step_ratio"]
                    if row["min_step_ratio"] is not None
                    else row.get("along_ratio")
                )
                motion_response.append(row)
            trace_event(
                "DIAGNOSTIC",
                "MOTION_RESPONSE_SUMMARY",
                operation_id=self._trace_operation_id,
                context=context,
                tool="MOVE",
                route="FK_MOVE",
                axis=self.axis,
                domains=tuple(motion_response),
            )
            stalled = tuple(
                row
                for row in motion_response
                if float(row.get("requested_distance") or 0.0) > 1e-5
                and row.get("response_ratio") is not None
                and float(row["response_ratio"]) <= 0.15
            )
            if stalled:
                trace_event(
                    "DIAGNOSTIC",
                    "MOTION_STALL",
                    operation_id=self._trace_operation_id,
                    context=context,
                    tool="MOVE",
                    route="FK_MOVE",
                    domains=stalled,
                )
            reach_clamped = tuple(
                row
                for row in motion_response
                if row.get("max_reach_ratio") is not None
                and float(row["max_reach_ratio"]) >= 0.995
                and row.get("response_ratio") is not None
                and float(row["response_ratio"]) < 0.95
            )
            if reach_clamped:
                trace_event(
                    "DIAGNOSTIC",
                    "REACH_CLAMP",
                    operation_id=self._trace_operation_id,
                    context=context,
                    tool="MOVE",
                    route="FK_MOVE",
                    domains=reach_clamped,
                )

            auto_enabled_at_release = bool(
                getattr(context.scene, "baw_auto_key_enabled", False)
            )
            auto_plan = self._auto_plan if auto_enabled_at_release else None
            auto_direct_plan = (
                self._auto_direct_plan if auto_enabled_at_release else None
            )
            auto_contact_batch_plan = (
                self._auto_contact_batch_plan if auto_enabled_at_release else None
            )
            deferred_auto_contact_mapping_ids = (
                self._deferred_auto_contact_mapping_ids
                if auto_enabled_at_release
                else ()
            )
            deferred_auto_results: list[Any] = []
            auto_context = control_context_for_context(context)
            auto_limb_context, auto_direct_context = _fk_move_auto_contexts(
                context.scene,
                auto_context,
                sessions,
            )
            if (
                auto_contact_batch_plan is None
                and deferred_auto_contact_mapping_ids
            ):
                batch = plan_rigped_auto_contact_batch(
                    context.scene,
                    auto_limb_context,
                    operation_id=f"ak3:fk-move-auto-release:{uuid4().hex}",
                    mapping_ids=deferred_auto_contact_mapping_ids,
                )
                if not batch.ok or batch.plan is None:
                    cancel_fk_joint_moves(context, sessions)
                    detail = (
                        batch.diagnostics[0].detail
                        if batch.diagnostics
                        else "Rigped multi-limb Free FK Move Auto planning failed."
                    )
                    self.report({"WARNING"}, detail)
                    self._sessions = ()
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                auto_contact_batch_plan = batch.plan
            if auto_plan is not None:
                link_trace_operation(
                    auto_plan.intent.operation_id,
                    self._trace_operation_id,
                )
                try:
                    auto_result = commit_rigped_auto_anchor(
                        context.scene,
                        auto_limb_context,
                        auto_plan,
                        defer_commit=True,
                    )
                except (ContactAuthoringError, RuntimeError, ValueError, ReferenceError) as exc:
                    cancel_fk_joint_moves(context, sessions)
                    _report_operator_error(self, context, exc, replay_action=replay_action)
                    self._sessions = ()
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                if not auto_result.applied:
                    cancel_fk_joint_moves(context, sessions)
                    detail = "; ".join(item.detail for item in auto_result.diagnostics)
                    self.report({"WARNING"}, detail or "Rigped Free FK Move Auto commit failed.")
                    self._sessions = ()
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                deferred_auto_results.append(auto_result)

                frames = [
                    float(context.scene.frame_current)
                    + float(getattr(context.scene, "frame_subframe", 0.0))
                ]
                if auto_plan.baseline_time is not None:
                    frames.append(float(auto_plan.baseline_time))
                clear_contact_bundle_key_selection_for_context(context, frames=tuple(frames))
                from .trackbar_model import clear_key_selection_for_context

                clear_key_selection_for_context(context)
                context.scene.baw_has_selected_key = False

            elif auto_contact_batch_plan is not None:
                link_trace_operation(
                    auto_contact_batch_plan.batch.operation_id,
                    self._trace_operation_id,
                )
                try:
                    auto_result = commit_rigped_auto_contact_batch(
                        context.scene,
                        auto_limb_context,
                        auto_contact_batch_plan,
                        defer_commit=True,
                    )
                except (ContactAuthoringError, RuntimeError, ValueError, ReferenceError) as exc:
                    cancel_fk_joint_moves(context, sessions)
                    _report_operator_error(self, context, exc, replay_action=replay_action)
                    self._sessions = ()
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                if not auto_result.applied:
                    cancel_fk_joint_moves(context, sessions)
                    detail = "; ".join(item.detail for item in auto_result.diagnostics)
                    self.report(
                        {"WARNING"},
                        detail or "Rigped multi-limb Free FK Move Auto commit failed.",
                    )
                    self._sessions = ()
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                deferred_auto_results.append(auto_result)
                from .trackbar_model import clear_key_selection_for_context

                clear_key_selection_for_context(context)
                context.scene.baw_has_selected_key = False

            if auto_direct_plan is not None:
                if auto_plan is not None or auto_contact_batch_plan is not None:
                    auto_direct_plan = replace(
                        auto_direct_plan,
                        allow_storage_rebind=True,
                    )
                link_trace_operation(
                    auto_direct_plan.begin_plan.operation_id,
                    self._trace_operation_id,
                )
                try:
                    auto_result = commit_rigped_auto_direct_rotate(
                        context.scene,
                        auto_direct_context,
                        auto_direct_plan,
                        defer_commit=True,
                    )
                except (RuntimeError, ValueError, ReferenceError) as exc:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    cancel_fk_joint_moves(context, sessions)
                    _report_operator_error(self, context, exc, replay_action=replay_action)
                    self._sessions = ()
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                if not auto_result.applied:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    cancel_fk_joint_moves(context, sessions)
                    detail = "; ".join(item.detail for item in auto_result.diagnostics)
                    self.report({"WARNING"}, detail or "Rigped direct FK Move Auto commit failed.")
                    self._sessions = ()
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                deferred_auto_results.append(auto_result)
                _refresh_current_sliding_public_overlays(context)
                from .trackbar_model import clear_key_selection_for_context

                clear_key_selection_for_context(context)
                context.scene.baw_has_selected_key = False

            try:
                result = commit_fk_joint_moves(context, sessions)
            except RigpedSemanticMoveError as exc:
                rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                cancel_fk_joint_moves(context, sessions)
                _report_operator_error(self, context, exc, replay_action=replay_action)
                self._sessions = ()
                self._auto_plan = None
                self._auto_direct_plan = None
                self._auto_contact_batch_plan = None
                self._deferred_auto_contact_mapping_ids = ()
                _set_semantic_move_drag_active(context, False)
                return {"CANCELLED"}

            if not result.success:
                rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                cancel_fk_joint_moves(context, sessions)
                detail = "; ".join(item.detail for item in result.diagnostics)
                trace_event(
                    "OPERATION",
                    "TRANSFORM_FAIL",
                    operation_id=self._trace_operation_id,
                    subsystem="modal",
                    lifecycle_phase="fail",
                    terminal_status=TraceTerminalStatus.CANCELLED,
                    route_outcome=TraceRouteOutcome.CLAIMED,
                    context=context,
                    detail=detail or "Rigped FK Move commit failed",
                    final_delta=tuple(float(value) for value in self._last_delta),
                )
                self._trace_operation_id = None
                self.report({"WARNING"}, detail or "Rigped FK Move commit failed")
                self._sessions = ()
                self._auto_plan = None
                self._auto_direct_plan = None
                self._auto_contact_batch_plan = None
                self._deferred_auto_contact_mapping_ids = ()
                _set_semantic_move_drag_active(context, False)
                return {"CANCELLED"}

            try:
                commit_rigped_auto_writer_results(tuple(deferred_auto_results))
            except (RuntimeError, ValueError, ReferenceError) as exc:
                rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                cancel_fk_joint_moves(context, sessions)
                _report_operator_error(self, context, exc, replay_action=replay_action)
                self._sessions = ()
                self._auto_plan = None
                self._auto_direct_plan = None
                self._auto_contact_batch_plan = None
                self._deferred_auto_contact_mapping_ids = ()
                _set_semantic_move_drag_active(context, False)
                return {"CANCELLED"}

            self._sessions = ()
            self._auto_plan = None
            self._auto_direct_plan = None
            self._auto_contact_batch_plan = None
            self._deferred_auto_contact_mapping_ids = ()
            _set_semantic_move_drag_active(context, False)
            if context.area is not None:
                context.area.tag_redraw()
            # One successful FK Move gesture owns one undo step, including all
            # release-time Auto/Contact baseline/state/key authoring.
            bpy.ops.ed.undo_push(message="AWB Rigped FK Move")
            trace_event(
                "OPERATION",
                "TRANSFORM_COMMIT",
                operation_id=self._trace_operation_id,
                subsystem="modal",
                lifecycle_phase="commit",
                terminal_status=TraceTerminalStatus.FINISHED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
                tool="MOVE",
                route="FK_MOVE",
                final_delta=replay_delta,
                auto_key=bool(getattr(context.scene, "baw_auto_key_enabled", False)),
                replay_action=replay_action,
            )
            self._trace_operation_id = None
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            cancel_fk_joint_moves(context, sessions)
            self._sessions = ()
            self._deferred_auto_contact_mapping_ids = ()
            _set_semantic_move_drag_active(context, False)
            if context.area is not None:
                context.area.tag_redraw()
            trace_event(
                "OPERATION",
                "TRANSFORM_CANCEL",
                operation_id=self._trace_operation_id,
                subsystem="modal",
                lifecycle_phase="cancel",
                terminal_status=TraceTerminalStatus.CANCELLED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
                reason=event.type,
            )
            self._trace_operation_id = None
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_GGT_rigped_fk_joint_move(bpy.types.GizmoGroup):
    bl_idname = "BAW_GGT_rigped_fk_joint_move"
    bl_label = "AWB Rigped FK Joint Move Gizmo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options: ClassVar[set[str]] = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "mode", "") == "POSE"
            and getattr(context.scene, "baw_rigped_semantic_transform_mode", "NONE")
            == "FK_MOVE"
            and fk_joint_move_available(context)
        )

    def setup(self, _context):
        self._zoom_reference_distance = None
        self.axis_gizmos = {}
        self.axis_hit_gizmos = {}
        self.plane_gizmos = {}
        self.plane_hit_gizmos = {}
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        for axis_name in ("X", "Y", "Z"):
            visible = self.gizmos.new("GIZMO_GT_arrow_3d")
            visible.draw_style = "NORMAL"
            visible.draw_options = {"STEM"}
            visible.color = colors[axis_name]
            visible.alpha = 0.9
            visible.color_highlight = colors[axis_name]
            visible.alpha_highlight = 0.9
            visible.line_width = 1.0
            visible.scale_basis = 1.0
            visible.hide_select = True
            visible.use_draw_modal = True
            self.axis_gizmos[axis_name] = visible

            hit = self.gizmos.new("GIZMO_GT_arrow_3d")
            props = hit.target_set_operator(BAW_OT_rigped_fk_joint_move_axis.bl_idname)
            props.axis = axis_name
            hit.draw_style = "NORMAL"
            hit.draw_options = {"STEM"}
            hit.color = colors[axis_name]
            hit.alpha = 0.001
            hit.color_highlight = (1.0, 1.0, 1.0)
            hit.alpha_highlight = 1.0
            hit.line_width = 1.0
            hit.scale_basis = 1.0
            hit.select_bias = 2.0
            hit.use_draw_modal = False
            self.axis_hit_gizmos[axis_name] = hit

        plane_colors = {"XY": colors["Z"], "XZ": colors["Y"], "YZ": colors["X"]}
        for plane_name in ("XY", "XZ", "YZ"):
            visible = self.gizmos.new("GIZMO_GT_primitive_3d")
            visible.draw_style = "PLANE"
            visible.draw_inner = True
            visible.color = plane_colors[plane_name]
            visible.alpha = 0.5
            visible.color_highlight = plane_colors[plane_name]
            visible.alpha_highlight = 0.5
            visible.line_width = 1.0
            visible.scale_basis = 0.11
            visible.hide_select = True
            visible.use_draw_offset_scale = True
            visible.use_draw_modal = True
            self.plane_gizmos[plane_name] = visible

            hit = self.gizmos.new("GIZMO_GT_primitive_3d")
            props = hit.target_set_operator(BAW_OT_rigped_fk_joint_move_axis.bl_idname)
            props.axis = plane_name
            hit.draw_style = "PLANE"
            hit.draw_inner = True
            hit.color = plane_colors[plane_name]
            hit.alpha = 0.001
            hit.color_highlight = (1.0, 1.0, 1.0)
            hit.alpha_highlight = 0.95
            hit.line_width = 1.0
            hit.scale_basis = 0.14
            hit.select_bias = 2.0
            hit.use_draw_offset_scale = True
            hit.use_draw_modal = False
            self.plane_hit_gizmos[plane_name] = hit

        center = self.gizmos.new("GIZMO_GT_dial_3d")
        center.color = (0.82, 0.82, 0.82)
        center.alpha = 0.9
        center.color_highlight = (0.82, 0.82, 0.82)
        center.alpha_highlight = 0.9
        center.line_width = 1.5
        center.scale_basis = 0.18
        center.hide_select = True
        center.use_draw_modal = True
        self.center_gizmo = center

        center_hit = self.gizmos.new("GIZMO_GT_dial_3d")
        props = center_hit.target_set_operator(BAW_OT_rigped_fk_joint_move_axis.bl_idname)
        props.axis = "FREE"
        center_hit.color = (0.82, 0.82, 0.82)
        center_hit.alpha = 0.001
        center_hit.color_highlight = (1.0, 1.0, 1.0)
        center_hit.alpha_highlight = 1.0
        center_hit.line_width = 1.5
        center_hit.scale_basis = 0.20
        center_hit.select_bias = 2.0
        center_hit.use_draw_modal = False
        self.center_hit_gizmo = center_hit

    def draw_prepare(self, context):
        active = _fk_joint_move_active_control(context)
        pivot = _control_pivot_world(active) if active is not None else None
        axes = (
            _orientation_axes_for_control(context, active)
            if active is not None
            else None
        )
        if pivot is None or axes is None:
            for gizmo in (
                *self.axis_gizmos.values(),
                *self.axis_hit_gizmos.values(),
                *self.plane_gizmos.values(),
                *self.plane_hit_gizmos.values(),
                self.center_gizmo,
                self.center_hit_gizmo,
            ):
                gizmo.hide = True
            return

        gizmo_scale, self._zoom_reference_distance = gizmo_zoom_scale(
            context,
            self._zoom_reference_distance,
        )
        self.center_gizmo.scale_basis = 0.18 * gizmo_scale
        self.center_hit_gizmo.scale_basis = 0.20 * gizmo_scale
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        for axis_name, visible in self.axis_gizmos.items():
            hit = self.axis_hit_gizmos[axis_name]
            visible.scale_basis = gizmo_scale
            hit.scale_basis = gizmo_scale
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(axes[axis_name])
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            visible.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else colors[axis_name]
            )
            visible.matrix_basis = matrix
            hit.matrix_basis = matrix
            visible.hide = False
            hit.hide = False

        world_length = 0.3
        if active is not None and isinstance(active.target, bpy.types.PoseBone):
            pose_bone = active.target
            world_length = float(
                Vector(
                    active.owner_object.matrix_world.to_3x3()
                    @ (pose_bone.tail - pose_bone.head)
                ).length
            )
        plane_offset = max(0.025, min(0.07, world_length * 0.14))
        plane_axes = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}
        plane_colors = {"XY": colors["Z"], "XZ": colors["Y"], "YZ": colors["X"]}
        for plane_name, visible in self.plane_gizmos.items():
            hit = self.plane_hit_gizmos[plane_name]
            visible.scale_basis = 0.11 * gizmo_scale
            hit.scale_basis = 0.14 * gizmo_scale
            first_name, second_name = plane_axes[plane_name]
            first = Vector(axes[first_name])
            second = Vector(axes[second_name])
            normal = first.cross(second).normalized()
            matrix = Matrix((first, second, normal)).transposed().to_4x4()
            matrix.translation = pivot + (first + second) * plane_offset
            visible.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else plane_colors[plane_name]
            )
            visible.matrix_basis = matrix
            hit.matrix_basis = matrix
            visible.hide = False
            hit.hide = False

        view_axis = _view_axis(context)
        if view_axis is None:
            self.center_gizmo.hide = True
            self.center_hit_gizmo.hide = True
        else:
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(view_axis)
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            self.center_gizmo.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(self.center_hit_gizmo, "is_highlight", False))
                else (0.82, 0.82, 0.82)
            )
            self.center_gizmo.matrix_basis = matrix
            self.center_hit_gizmo.matrix_basis = matrix
            self.center_gizmo.hide = False
            self.center_hit_gizmo.hide = False


def _selected_direct_move_controls(context) -> tuple[ResolvedControl, ...]:
    selected = _selected_direct_transform(context)
    if selected is None or not selected.move_native:
        return ()
    control_context = control_context_for_context(context)
    controls = tuple(
        control
        for control in control_context.controls
        if isinstance(control.target, bpy.types.PoseBone)
    )
    if not controls or control_context.active is None:
        return ()
    active_key = runtime_control_key(control_context.active)
    if not any(runtime_control_key(control) == active_key for control in controls):
        return ()
    return tuple(
        sorted(
            controls,
            key=lambda control: 0 if runtime_control_key(control) == active_key else 1,
        )
    )


def _begin_direct_move_states(
    control_context,
    operation_domain: OperationDomainSnapshot,
) -> tuple[DirectMoveControlState, ...]:
    if operation_domain.selector_was_stale:
        raise RigpedSemanticMoveError(
            "Direct Move cannot start from a stale Character selector."
        )
    if operation_domain.contact_mapping_ids:
        raise RigpedSemanticMoveError(
            "Direct Move cannot mix body translation with a Contact-limb operation domain."
        )
    if operation_domain.rejected_control_runtime_keys:
        raise RigpedSemanticMoveError(
            "Direct Move operation domain contains rejected controls."
        )
    if set(operation_domain.selected_binding_ids) != set(
        operation_domain.supported_direct_binding_ids
    ):
        raise RigpedSemanticMoveError(
            "Direct Move requires one fully supported frozen direct-control domain."
        )

    controls_by_key = {
        runtime_control_key(control): control
        for control in control_context.controls
        if isinstance(control.target, bpy.types.PoseBone)
    }
    selected_keys = tuple(operation_domain.selected_control_runtime_keys)
    if set(controls_by_key) != set(selected_keys):
        raise RigpedSemanticMoveError(
            "Direct Move native selection changed while freezing the operation domain."
        )
    active_key = operation_domain.active_control_runtime_key
    if active_key is None or active_key not in controls_by_key:
        raise RigpedSemanticMoveError(
            "Direct Move lost its frozen active control."
        )
    controls = tuple(
        sorted(
            (controls_by_key[key] for key in selected_keys),
            key=lambda control: 0 if runtime_control_key(control) == active_key else 1,
        )
    )

    states: list[DirectMoveControlState] = []
    for control in controls:
        matrix = _world_to_control_location_delta_matrix(control)
        if matrix is None:
            raise RigpedSemanticMoveError(
                f"Direct Move cannot translate connected control {control.target.name!r}."
            )
        states.append(
            DirectMoveControlState(
                control=control,
                start_location=tuple(float(value) for value in control.target.location),
                world_to_local_delta=matrix.copy(),
            )
        )
    if not states:
        raise RigpedSemanticMoveError("Current selection has no supported direct Move control.")
    return tuple(states)


def _apply_direct_move_delta(
    context,
    states: tuple[DirectMoveControlState, ...],
    world_delta: Vector,
    *,
    sliding_capabilities: tuple[LimbRepresentationCapability, ...] | None = None,
    dependency_guards: tuple[SlidingDependencyGuard, ...] = (),
    operation_id: str | None = None,
) -> None:
    for state in states:
        local_delta = state.world_to_local_delta @ Vector(world_delta)
        state.control.target.location = Vector(state.start_location) + local_delta
    context.view_layer.update()
    # Sliding authority is character-wide, not selection-wide. A direct
    # ancestor Move may change the evaluated FK hierarchy while the hidden IK
    # hierarchy follows a different parent space. Keep every Sliding public
    # control as a derived overlay of the current solved result during preview.
    if dependency_guards:
        _refresh_frozen_sliding_dependency_overlays(
            context,
            dependency_guards,
            operation_id=operation_id,
            phase="MOVE_PREVIEW",
        )
    else:
        _refresh_current_sliding_public_overlays(
            context,
            capabilities=sliding_capabilities,
        )


def _restore_direct_move_states(
    context,
    states: tuple[DirectMoveControlState, ...],
    *,
    sliding_capabilities: tuple[LimbRepresentationCapability, ...] | None = None,
    seed_snapshots: tuple[SlidingHiddenSeedSnapshot, ...] = (),
    dependency_guards: tuple[SlidingDependencyGuard, ...] = (),
    operation_id: str | None = None,
) -> None:
    for state in states:
        state.control.target.location = state.start_location
    _restore_sliding_dependency_guards(
        context,
        dependency_guards,
        update=False,
    )
    _restore_sliding_hidden_seed_snapshots(
        context,
        seed_snapshots,
        update=False,
    )
    context.view_layer.update()
    _refresh_current_sliding_public_overlays(
        context,
        capabilities=sliding_capabilities,
        allow_seed=False,
    )
    _assert_sliding_dependency_authority(
        context,
        dependency_guards,
        operation_id=operation_id,
        phase="MOVE_RESTORE",
    )


class BAW_OT_rigped_direct_move_axis(bpy.types.Operator):
    """AWB-owned direct Move modal so generated Rigped AUTO can key position."""

    bl_idname = "baw.rigped_direct_move_axis"
    bl_label = "Move Rigped Control"
    # This modal modifies Blender pose data and must use Blender's native
    # operator Undo contract. Manual ed.undo_push bookkeeping produced asymmetric
    # history in live Blender (either first Undo was a no-op or Redo lost the
    # committed pose), so keep one native operator boundary and no manual pushes.
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}

    axis: EnumProperty(
        items=(
            ("X", "X", "Move along X"),
            ("Y", "Y", "Move along Y"),
            ("Z", "Z", "Move along Z"),
            ("XY", "XY", "Move in XY plane"),
            ("XZ", "XZ", "Move in XZ plane"),
            ("YZ", "YZ", "Move in YZ plane"),
            ("FREE", "Free", "Move in view plane"),
        ),
        default="X",
    )
    orient_type: EnumProperty(
        items=(
            ("GLOBAL", "Global", ""),
            ("LOCAL", "Local", ""),
            ("VIEW", "View", ""),
        ),
        default="GLOBAL",
    )

    _states: tuple[DirectMoveControlState, ...] = ()
    _frozen_control_context: Any = None
    _operation_domain: OperationDomainSnapshot | None = None
    _sliding_guard_capabilities: tuple[LimbRepresentationCapability, ...] = ()
    _sliding_seed_snapshots: tuple[SlidingHiddenSeedSnapshot, ...] = ()
    _sliding_dependency_guards: tuple[SlidingDependencyGuard, ...] = ()
    _auto_plan: RigpedAutoDirectMovePlan | None = None
    _pivot = Vector((0.0, 0.0, 0.0))
    _axis = Vector((1.0, 0.0, 0.0))
    _move_screen_axis = Vector((1.0, 0.0))
    _move_start_screen_projection = 0.0
    _move_world_per_pixel = 0.0
    _move_use_screen_axis = False
    _move_use_screen_plane = False
    _move_start_mouse = Vector((0.0, 0.0))
    _move_plane_u = Vector((1.0, 0.0, 0.0))
    _move_plane_v = Vector((0.0, 1.0, 0.0))
    _move_plane_screen_u = Vector((1.0, 0.0))
    _move_plane_screen_v = Vector((0.0, 1.0))
    _move_plane_reference = 1.0
    _move_plane_det = 1.0
    _plane_normal = Vector((0.0, 0.0, 1.0))
    _last_delta = Vector((0.0, 0.0, 0.0))
    _move_last_mouse = Vector((0.0, 0.0))
    _trace_operation_id: str | None = None

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "mode", "") == "POSE"
            and getattr(
                context.scene,
                "baw_rigped_semantic_transform_mode",
                "NONE",
            )
            in {"MOVE", "FK_MOVE", "DIRECT_MOVE"}
            and direct_move_available(context)
        )

    def invoke(self, context, event):
        control_context = control_context_for_context(context)
        domain_resolution = resolve_operation_domain(
            context.scene,
            control_context,
        )
        if not domain_resolution.ok or domain_resolution.snapshot is None:
            detail = (
                domain_resolution.issues[0].detail
                if domain_resolution.issues
                else "Rigped Move could not freeze the operation domain."
            )
            self.report({"WARNING"}, detail)
            return {"CANCELLED"}
        operation_domain = domain_resolution.snapshot

        try:
            states = _begin_direct_move_states(
                control_context,
                operation_domain,
            )
        except RigpedSemanticMoveError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        active = states[0].control
        axes = _orientation_axes_for_control(context, active)
        pivot = _control_pivot_world(active)
        if axes is None or pivot is None:
            return {"CANCELLED"}

        self._frozen_control_context = control_context
        self._operation_domain = operation_domain
        self._sliding_guard_capabilities = _sliding_capabilities_for_character(
            context.scene,
            operation_domain.character_id,
        )
        self._sliding_seed_snapshots = _capture_sliding_hidden_seed_snapshots(
            self._sliding_guard_capabilities
        )
        e7_body_move = all(
            _resolved_control_role_name(state.control) == "COM"
            for state in states
        )
        self._sliding_dependency_guards = (
            _capture_sliding_dependency_guards(self._sliding_guard_capabilities)
            if e7_body_move
            else ()
        )
        self._auto_plan = None
        if bool(getattr(context.scene, "baw_auto_key_enabled", False)):
            planned = plan_rigped_auto_direct_move(
                context.scene,
                control_context,
                operation_id=f"rigped-auto-move:{uuid4().hex}",
                active_only=len(states) == 1,
                contact_activation_mapping_ids=(
                    _sliding_capability_mapping_ids(self._sliding_guard_capabilities)
                    if e7_body_move
                    else ()
                ),
            )
            if not planned.ok or planned.plan is None:
                detail = (
                    planned.diagnostics[0].detail
                    if planned.diagnostics
                    else "Rigped direct Move Auto planning failed."
                )
                self.report({"WARNING"}, detail)
                return {"CANCELLED"}
            self._auto_plan = planned.plan

        self._states = states
        self._pivot = Vector(pivot)
        self._last_delta = Vector((0.0, 0.0, 0.0))
        self._move_last_mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self._move_use_screen_axis = False
        self._move_use_screen_plane = False
        self._move_world_per_pixel = 0.0
        _set_semantic_move_drag_active(context, True)

        active = states[0].control
        pose_bone = active.target
        reference_world = 0.3
        if isinstance(pose_bone, bpy.types.PoseBone):
            reference_world = float(
                Vector(
                    active.owner_object.matrix_world.to_3x3()
                    @ (pose_bone.tail - pose_bone.head)
                ).length
            )
        reference_world = max(0.05, reference_world)
        region_data = getattr(context, "region_data", None)
        if region_data is None:
            region_data = getattr(getattr(context, "space_data", None), "region_3d", None)

        if self.axis in {"X", "Y", "Z"}:
            self._axis = Vector(axes[self.axis]).normalized()
            if context.region is not None and region_data is not None:
                pivot_2d = location_3d_to_region_2d(context.region, region_data, self._pivot)
                sample_2d = location_3d_to_region_2d(
                    context.region,
                    region_data,
                    self._pivot + self._axis * reference_world,
                )
                if pivot_2d is not None and sample_2d is not None:
                    screen_delta = Vector(sample_2d) - Vector(pivot_2d)
                    if screen_delta.length >= 6.0:
                        self._move_screen_axis = screen_delta.normalized()
                        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
                        self._move_start_screen_projection = float(mouse.dot(self._move_screen_axis))
                        world_per_pixel = projection_world_per_pixel(context, self._pivot)
                        if world_per_pixel is not None:
                            self._move_world_per_pixel = float(world_per_pixel)
                            self._move_use_screen_axis = True
            if not self._move_use_screen_axis:
                self._states = ()
                _set_semantic_move_drag_active(context, False)
                return {"CANCELLED"}
        else:
            if self.axis == "FREE":
                if region_data is None:
                    self._states = ()
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                view_basis = region_data.view_matrix.inverted().to_3x3().normalized()
                first = Vector(view_basis.col[0]).normalized()
                second = Vector(view_basis.col[1]).normalized()
            else:
                plane_axes = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}
                first_name, second_name = plane_axes[self.axis]
                first = Vector(axes[first_name]).normalized()
                second = Vector(axes[second_name]).normalized()
            normal = first.cross(second)
            if normal.length <= 1e-9:
                self._states = ()
                _set_semantic_move_drag_active(context, False)
                return {"CANCELLED"}
            self._plane_normal = normal.normalized()
            self._move_plane_u = first
            self._move_plane_v = second
            self._move_plane_reference = reference_world
            self._move_start_mouse = Vector((event.mouse_region_x, event.mouse_region_y))
            if context.region is not None and region_data is not None:
                pivot_2d = location_3d_to_region_2d(context.region, region_data, self._pivot)
                u_2d = location_3d_to_region_2d(
                    context.region, region_data, self._pivot + first * reference_world
                )
                v_2d = location_3d_to_region_2d(
                    context.region, region_data, self._pivot + second * reference_world
                )
                if pivot_2d is not None and u_2d is not None and v_2d is not None:
                    screen_u = Vector(u_2d) - Vector(pivot_2d)
                    screen_v = Vector(v_2d) - Vector(pivot_2d)
                    det = float(screen_u.x * screen_v.y - screen_u.y * screen_v.x)
                    lengths = float(screen_u.length * screen_v.length)
                    conditioning = abs(det) / max(lengths, 1e-9)
                    if screen_u.length >= 6.0 and screen_v.length >= 6.0 and conditioning >= 0.18:
                        self._move_plane_screen_u = screen_u
                        self._move_plane_screen_v = screen_v
                        self._move_plane_det = det
                        self._move_use_screen_plane = True
            if not self._move_use_screen_plane:
                self._states = ()
                _set_semantic_move_drag_active(context, False)
                return {"CANCELLED"}

        self._trace_operation_id = new_trace_operation_id("direct-move")
        bind_operation_causal_context(
            self._trace_operation_id,
            new_trace_causal_root("direct-move"),
        )
        trace_event(
            "OPERATION",
            "TRANSFORM_BEGIN",
            operation_id=self._trace_operation_id,
            subsystem="modal",
            lifecycle_phase="invoke",
            route_outcome=TraceRouteOutcome.CLAIMED,
            context=context,
            tool="MOVE",
            route="DIRECT_MOVE",
            axis=self.axis,
            auto_key=bool(getattr(context.scene, "baw_auto_key_enabled", False)),
            sliding_guard_count=len(self._sliding_guard_capabilities),
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        states = self._states
        if not states:
            _set_semantic_move_drag_active(context, False)
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            mouse_now = Vector((event.mouse_region_x, event.mouse_region_y))
            pixel_step = float((mouse_now - self._move_last_mouse).length)
            delta = None
            if self.axis in {"X", "Y", "Z"}:
                mouse = Vector((event.mouse_region_x, event.mouse_region_y))
                projection = float(mouse.dot(self._move_screen_axis))
                distance = (
                    projection - self._move_start_screen_projection
                ) * self._move_world_per_pixel
                delta = self._axis * distance
            else:
                mouse_delta = mouse_now - self._move_start_mouse
                screen_u = self._move_plane_screen_u
                screen_v = self._move_plane_screen_v
                det = self._move_plane_det
                alpha = float((mouse_delta.x * screen_v.y - mouse_delta.y * screen_v.x) / det)
                beta = float((screen_u.x * mouse_delta.y - screen_u.y * mouse_delta.x) / det)
                delta = (
                    self._move_plane_u * (self._move_plane_reference * alpha)
                    + self._move_plane_v * (self._move_plane_reference * beta)
                )
            stable_delta = Vector(delta)
            if pixel_step <= 3.0:
                stable_delta = self._last_delta.lerp(stable_delta, 0.68)
            try:
                _apply_direct_move_delta(
                    context,
                    states,
                    stable_delta,
                    sliding_capabilities=self._sliding_guard_capabilities,
                    dependency_guards=self._sliding_dependency_guards,
                    operation_id=self._trace_operation_id,
                )
            except (RigpedSemanticMoveError, RuntimeError, ValueError, ReferenceError) as exc:
                _restore_direct_move_states(
                    context,
                    states,
                    sliding_capabilities=self._sliding_guard_capabilities,
                    seed_snapshots=self._sliding_seed_snapshots,
                    dependency_guards=self._sliding_dependency_guards,
                    operation_id=self._trace_operation_id,
                )
                _report_operator_error(
                    self,
                    context,
                    exc,
                    tool="MOVE",
                    route="DIRECT_MOVE",
                    axis=self.axis,
                    phase="MOUSEMOVE",
                    attempted_delta=tuple(float(value) for value in stable_delta),
                )
                self._states = ()
                self._sliding_guard_capabilities = ()
                self._auto_plan = None
                _set_semantic_move_drag_active(context, False)
                self._trace_operation_id = None
                if context.area is not None:
                    context.area.tag_redraw()
                return {"CANCELLED"}
            self._last_delta = stable_delta
            self._move_last_mouse = mouse_now
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self._last_delta.length <= 1e-9:
                _restore_direct_move_states(
                    context,
                    states,
                    sliding_capabilities=self._sliding_guard_capabilities,
                    seed_snapshots=self._sliding_seed_snapshots,
                    dependency_guards=self._sliding_dependency_guards,
                    operation_id=self._trace_operation_id,
                )
                self._states = ()
                self._sliding_guard_capabilities = ()
                self._auto_plan = None
                _set_semantic_move_drag_active(context, False)
                trace_event(
                    "OPERATION",
                    "TRANSFORM_CANCEL",
                    operation_id=self._trace_operation_id,
                    subsystem="modal",
                    lifecycle_phase="cancel",
                    terminal_status=TraceTerminalStatus.CANCELLED,
                    route_outcome=TraceRouteOutcome.CLAIMED,
                    context=context,
                    reason="NO_DELTA",
                )
                self._trace_operation_id = None
                return {"CANCELLED"}

            final_delta = Vector(self._last_delta)
            replay_delta = tuple(float(value) for value in final_delta)
            replay_action = {
                "kind": "MOVE",
                "route": "DIRECT_MOVE",
                "frame": int(context.scene.frame_current),
                "subframe": float(getattr(context.scene, "frame_subframe", 0.0)),
                "controls": tuple(
                    str(item.name)
                    for item in tuple(getattr(context, "selected_pose_bones", ()) or ())
                ),
                "axis": self.axis,
                "delta_world": replay_delta,
            }
            _restore_direct_move_states(
                context,
                states,
                sliding_capabilities=self._sliding_guard_capabilities,
                seed_snapshots=self._sliding_seed_snapshots,
                dependency_guards=self._sliding_dependency_guards,
                operation_id=self._trace_operation_id,
            )
            _apply_direct_move_delta(
                context,
                states,
                final_delta,
                sliding_capabilities=self._sliding_guard_capabilities,
                dependency_guards=self._sliding_dependency_guards,
                operation_id=self._trace_operation_id,
            )

            auto_enabled_at_release = bool(
                getattr(context.scene, "baw_auto_key_enabled", False)
            )
            auto_plan = self._auto_plan if auto_enabled_at_release else None
            if auto_plan is not None:
                deferred_auto_results: list[Any] = []
                link_trace_operation(auto_plan.begin_plan.operation_id, self._trace_operation_id)
                try:
                    result = commit_rigped_auto_direct_move(
                        context.scene,
                        self._frozen_control_context,
                        auto_plan,
                        defer_commit=True,
                    )
                    if result.applied:
                        deferred_auto_results.append(result)
                        _refresh_current_sliding_public_overlays(
                            context,
                            capabilities=self._sliding_guard_capabilities,
                        )
                        commit_rigped_auto_writer_results(tuple(deferred_auto_results))
                        context.view_layer.update()
                        # Committing the direct AUTO writer can immediately
                        # reevaluate the Action at this same frame. Re-project
                        # the IK-authoritative public chains after that commit
                        # so Planted/Sliding visible controls remain welded to
                        # their hidden result instead of inheriting the moved
                        # body hierarchy.
                        _refresh_current_sliding_public_overlays(
                            context,
                            capabilities=self._sliding_guard_capabilities,
                            allow_seed=False,
                        )
                except (RuntimeError, ValueError, ReferenceError) as exc:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    _restore_direct_move_states(
                        context,
                        states,
                        sliding_capabilities=self._sliding_guard_capabilities,
                        seed_snapshots=self._sliding_seed_snapshots,
                        dependency_guards=self._sliding_dependency_guards,
                        operation_id=self._trace_operation_id,
                    )
                    _report_operator_error(self, context, exc, replay_action=replay_action)
                    self._states = ()
                    self._sliding_guard_capabilities = ()
                    self._auto_plan = None
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}
                if not result.applied:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    _restore_direct_move_states(
                        context,
                        states,
                        sliding_capabilities=self._sliding_guard_capabilities,
                        seed_snapshots=self._sliding_seed_snapshots,
                        dependency_guards=self._sliding_dependency_guards,
                        operation_id=self._trace_operation_id,
                    )
                    detail = "; ".join(item.detail for item in result.diagnostics)
                    self.report({"WARNING"}, detail or "Rigped direct Move Auto commit failed.")
                    self._states = ()
                    self._sliding_guard_capabilities = ()
                    self._auto_plan = None
                    _set_semantic_move_drag_active(context, False)
                    return {"CANCELLED"}

                from .trackbar_model import clear_key_selection_for_context

                clear_key_selection_for_context(context)
                context.scene.baw_has_selected_key = False

            self._states = ()
            self._sliding_guard_capabilities = ()
            self._auto_plan = None
            _set_semantic_move_drag_active(context, False)
            trace_event(
                "OPERATION",
                "TRANSFORM_COMMIT",
                operation_id=self._trace_operation_id,
                subsystem="modal",
                lifecycle_phase="commit",
                terminal_status=TraceTerminalStatus.FINISHED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
                tool="MOVE",
                route="DIRECT_MOVE",
                final_delta=replay_delta,
                auto_key=bool(getattr(context.scene, "baw_auto_key_enabled", False)),
                replay_action=replay_action,
            )
            self._trace_operation_id = None
            if context.area is not None:
                context.area.tag_redraw()
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            _restore_direct_move_states(
                context,
                states,
                sliding_capabilities=self._sliding_guard_capabilities,
                seed_snapshots=self._sliding_seed_snapshots,
                dependency_guards=self._sliding_dependency_guards,
                operation_id=self._trace_operation_id,
            )
            self._states = ()
            self._sliding_guard_capabilities = ()
            self._auto_plan = None
            _set_semantic_move_drag_active(context, False)
            trace_event(
                "OPERATION",
                "TRANSFORM_CANCEL",
                operation_id=self._trace_operation_id,
                subsystem="modal",
                lifecycle_phase="cancel",
                terminal_status=TraceTerminalStatus.CANCELLED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
                reason=event.type,
            )
            self._trace_operation_id = None
            if context.area is not None:
                context.area.tag_redraw()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_GGT_rigped_direct_move(bpy.types.GizmoGroup):
    """Anchored Blender-style Move shell for native Rigped controls.

    The visible handles never own Blender's transform operator, so they cannot
    accumulate gizmo offset during drag. Invisible proxies invoke the native
    transform operator, preserving Blender FK hierarchy, Auto Key and Undo.
    """

    bl_idname = "BAW_GGT_rigped_direct_move"
    bl_label = "AWB Rigped Direct Move Gizmo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options: ClassVar[set[str]] = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "mode", "") == "POSE"
            and getattr(context.scene, "baw_rigped_semantic_transform_mode", "NONE")
            == "DIRECT_MOVE"
            and direct_move_available(context)
        )

    def setup(self, _context):
        self._zoom_reference_distance = None
        self.axis_gizmos = {}
        self.axis_hit_gizmos = {}
        self.axis_hit_props = {}
        self.plane_gizmos = {}
        self.plane_hit_gizmos = {}
        self.plane_hit_props = {}
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        for axis_name in ("X", "Y", "Z"):
            visible = self.gizmos.new("GIZMO_GT_arrow_3d")
            visible.draw_style = "NORMAL"
            visible.draw_options = {"STEM"}
            visible.color = colors[axis_name]
            visible.alpha = 0.9
            visible.color_highlight = colors[axis_name]
            visible.alpha_highlight = 0.9
            visible.line_width = 1.0
            visible.scale_basis = 1.0
            visible.hide_select = True
            visible.use_draw_modal = True
            self.axis_gizmos[axis_name] = visible

            hit = self.gizmos.new("GIZMO_GT_arrow_3d")
            props = hit.target_set_operator(BAW_OT_rigped_direct_move_axis.bl_idname)
            props.axis = axis_name
            hit.draw_style = "NORMAL"
            hit.draw_options = {"STEM"}
            hit.color = colors[axis_name]
            hit.alpha = 0.001
            hit.color_highlight = (1.0, 1.0, 1.0)
            hit.alpha_highlight = 1.0
            hit.line_width = 1.0
            hit.scale_basis = 1.0
            hit.select_bias = 2.0
            hit.use_draw_modal = False
            self.axis_hit_gizmos[axis_name] = hit
            self.axis_hit_props[axis_name] = props

        plane_colors = {
            "XY": colors["Z"],
            "XZ": colors["Y"],
            "YZ": colors["X"],
        }
        for plane_name in ("XY", "XZ", "YZ"):
            visible = self.gizmos.new("GIZMO_GT_primitive_3d")
            visible.draw_style = "PLANE"
            visible.draw_inner = True
            visible.color = plane_colors[plane_name]
            visible.alpha = 0.5
            visible.color_highlight = plane_colors[plane_name]
            visible.alpha_highlight = 0.5
            visible.line_width = 1.0
            visible.scale_basis = 0.11
            visible.hide_select = True
            visible.use_draw_offset_scale = True
            visible.use_draw_modal = True
            self.plane_gizmos[plane_name] = visible

            hit = self.gizmos.new("GIZMO_GT_primitive_3d")
            props = hit.target_set_operator(BAW_OT_rigped_direct_move_axis.bl_idname)
            props.axis = plane_name
            hit.draw_style = "PLANE"
            hit.draw_inner = True
            hit.color = plane_colors[plane_name]
            hit.alpha = 0.001
            hit.color_highlight = (1.0, 1.0, 1.0)
            hit.alpha_highlight = 0.95
            hit.line_width = 1.0
            hit.scale_basis = 0.14
            hit.select_bias = 2.0
            hit.use_draw_offset_scale = True
            hit.use_draw_modal = False
            self.plane_hit_gizmos[plane_name] = hit
            self.plane_hit_props[plane_name] = props

        center = self.gizmos.new("GIZMO_GT_dial_3d")
        center.color = (0.82, 0.82, 0.82)
        center.alpha = 0.9
        center.color_highlight = (0.82, 0.82, 0.82)
        center.alpha_highlight = 0.9
        center.line_width = 1.5
        center.scale_basis = 0.18
        center.hide_select = True
        center.use_draw_modal = True
        self.center_gizmo = center

        center_hit = self.gizmos.new("GIZMO_GT_dial_3d")
        center_props = center_hit.target_set_operator(BAW_OT_rigped_direct_move_axis.bl_idname)
        center_props.axis = "FREE"
        center_hit.color = (0.82, 0.82, 0.82)
        center_hit.alpha = 0.001
        center_hit.color_highlight = (1.0, 1.0, 1.0)
        center_hit.alpha_highlight = 1.0
        center_hit.line_width = 1.5
        center_hit.scale_basis = 0.20
        center_hit.select_bias = 2.0
        center_hit.use_draw_modal = False
        self.center_hit_gizmo = center_hit
        self.center_hit_props = center_props

    def draw_prepare(self, context):
        pivot = direct_transform_pivot(context)
        axes = direct_transform_axes(context)
        if pivot is None or axes is None or not direct_move_available(context):
            for gizmo in (
                *self.axis_gizmos.values(),
                *self.axis_hit_gizmos.values(),
                *self.plane_gizmos.values(),
                *self.plane_hit_gizmos.values(),
                self.center_gizmo,
                self.center_hit_gizmo,
            ):
                gizmo.hide = True
            return

        orientation = str(context.scene.transform_orientation_slots[0].type)
        for props in self.axis_hit_props.values():
            props.orient_type = orientation
        for props in self.plane_hit_props.values():
            props.orient_type = orientation
        self.center_hit_props.orient_type = orientation

        gizmo_scale, self._zoom_reference_distance = gizmo_zoom_scale(
            context,
            self._zoom_reference_distance,
        )
        self.center_gizmo.scale_basis = 0.18 * gizmo_scale
        self.center_hit_gizmo.scale_basis = 0.20 * gizmo_scale
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        for axis_name, visible in self.axis_gizmos.items():
            hit = self.axis_hit_gizmos[axis_name]
            visible.scale_basis = gizmo_scale
            hit.scale_basis = gizmo_scale
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(axes[axis_name])
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            visible.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else colors[axis_name]
            )
            visible.matrix_basis = matrix
            hit.matrix_basis = matrix
            visible.hide = False
            hit.hide = False

        selected = _selected_direct_transform(context)
        world_length = 0.3
        if selected is not None and isinstance(selected.active_control.target, bpy.types.PoseBone):
            pose_bone = selected.active_control.target
            world_length = Vector(
                selected.active_control.owner_object.matrix_world.to_3x3()
                @ (pose_bone.tail - pose_bone.head)
            ).length
        plane_offset = max(0.025, min(0.07, float(world_length) * 0.14))
        plane_axes = {
            "XY": ("X", "Y"),
            "XZ": ("X", "Z"),
            "YZ": ("Y", "Z"),
        }
        plane_colors = {
            "XY": colors["Z"],
            "XZ": colors["Y"],
            "YZ": colors["X"],
        }
        for plane_name, visible in self.plane_gizmos.items():
            hit = self.plane_hit_gizmos[plane_name]
            visible.scale_basis = 0.11 * gizmo_scale
            hit.scale_basis = 0.14 * gizmo_scale
            first_name, second_name = plane_axes[plane_name]
            first = Vector(axes[first_name])
            second = Vector(axes[second_name])
            normal = first.cross(second).normalized()
            matrix = Matrix((first, second, normal)).transposed().to_4x4()
            matrix.translation = pivot + (first + second) * plane_offset
            visible.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else plane_colors[plane_name]
            )
            visible.matrix_basis = matrix
            hit.matrix_basis = matrix
            visible.hide = False
            hit.hide = False

        view_axis = _view_axis(context)
        if view_axis is None:
            self.center_gizmo.hide = True
            self.center_hit_gizmo.hide = True
        else:
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(view_axis)
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            self.center_gizmo.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(self.center_hit_gizmo, "is_highlight", False))
                else (0.82, 0.82, 0.82)
            )
            self.center_gizmo.matrix_basis = matrix
            self.center_hit_gizmo.matrix_basis = matrix
            self.center_gizmo.hide = False
            self.center_hit_gizmo.hide = False


def _selected_direct_rotate_controls(context) -> tuple[ResolvedControl, ...]:
    selected = _selected_direct_transform(context)
    if selected is None or not selected.rotate_native:
        return ()
    control_context = control_context_for_context(context)
    controls = tuple(
        control
        for control in control_context.controls
        if isinstance(control.target, bpy.types.PoseBone)
    )
    if not controls or control_context.active is None:
        return ()
    active_key = runtime_control_key(control_context.active)
    if not any(runtime_control_key(control) == active_key for control in controls):
        return ()
    return tuple(
        sorted(
            controls,
            key=lambda control: 0 if runtime_control_key(control) == active_key else 1,
        )
    )


# Biped keeps upper links and terminals broadly rotatable, but elbows/knees are
# hinge-dominant and human limbs do not pass through arbitrary 180-degree local
# rotations. These are deliberately permissive animation envelopes rather than
# clinical ROM values: the goal is to block visibly broken poses while keeping
# the slightly exaggerated freedom animators expect from Biped.
_RIGPED_BALL_JOINT_LIMITS = {
    "UpperArm.L": (radians(170.0), radians(130.0)),
    "UpperArm.R": (radians(170.0), radians(130.0)),
    "Thigh.L": (radians(165.0), radians(110.0)),
    "Thigh.R": (radians(165.0), radians(110.0)),
}
# Hand/Foot are true local 3DOF terminals. Use a gimbal-free swing/twist
# envelope instead of clamping XYZ Euler coordinates: local-Y is axial twist,
# while X/Z share the terminal swing cone. Direct Rotate follows the actual
# current local gizmo axis, so an extreme forward/back foot still has a sane Y
# roll and cannot inherit Euler branch coupling from the other two axes.
_RIGPED_TERMINAL_SWING_TWIST_LIMITS = {
    "Hand.L": (radians(95.0), radians(80.0)),
    "Hand.R": (radians(95.0), radians(80.0)),
    "Foot.L": (radians(80.0), radians(70.0)),
    "Foot.R": (radians(80.0), radians(70.0)),
}
_RIGPED_HINGE_JOINT_LIMITS = {
    # Fitted Blender ForeArm local X is the elbow bend-plane normal; local Y
    # remains the long-axis forearm roll. Local Z is reserved for shoulder-wrist
    # swivel in the animator-facing Rotate contract.
    # Public FK hinge controls are branch-neutral. The generated hidden IK/MCH
    # solver owns the active bend branch, and a valid authored FK pose may cross
    # the generated local hinge zero. Limit magnitude and off-axis twist here,
    # but never hard-code left/right bend sign on the public controls.
    "ForeArm.L": ("X", -radians(155.0), radians(155.0), radians(100.0)),
    "ForeArm.R": ("X", -radians(155.0), radians(155.0), radians(100.0)),
    # Generated calf local X is the knee hinge on both sides. LOCAL Y shares
    # the animator-facing lower-link axial-roll contract with ForeArm, so keep
    # the same usable long-axis range instead of hard-stopping after 20 degrees.
    "Calf.L": ("X", -radians(155.0), radians(155.0), radians(100.0)),
    "Calf.R": ("X", -radians(155.0), radians(155.0), radians(100.0)),
}
_RIGPED_LOCAL_AXES = {
    "X": Vector((1.0, 0.0, 0.0)),
    "Y": Vector((0.0, 1.0, 0.0)),
    "Z": Vector((0.0, 0.0, 1.0)),
}


def _clamp_scalar(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _signed_twist_angle(rotation: Quaternion, axis: Vector) -> float:
    """Return shortest signed twist of ``rotation`` around one local unit axis."""

    q = rotation.normalized()
    if q.w < 0.0:
        q = Quaternion((-q.w, -q.x, -q.y, -q.z))
    vector = Vector((q.x, q.y, q.z))
    projected = float(vector.dot(axis))
    angle = 2.0 * atan2(projected, float(q.w))
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return float(angle)


def _clamp_ball_joint_rotation(
    rotation: Quaternion,
    *,
    swing_limit: float,
    twist_limit: float,
) -> Quaternion:
    long_axis = _RIGPED_LOCAL_AXES["Y"]
    q = rotation.normalized()
    twist_angle = _signed_twist_angle(q, long_axis)
    twist_full = Quaternion(long_axis, twist_angle)
    swing = (q @ twist_full.inverted()).normalized()
    swing_angle = float(swing.angle)
    if swing_angle > float(swing_limit) and swing_angle > 1e-9:
        swing = Quaternion(Vector(swing.axis).normalized(), float(swing_limit))
    twist = Quaternion(
        long_axis,
        _clamp_scalar(twist_angle, -float(twist_limit), float(twist_limit)),
    )
    return (swing @ twist).normalized()


def _clamp_hinge_joint_rotation(
    rotation: Quaternion,
    *,
    hinge_axis_name: str,
    hinge_min: float,
    hinge_max: float,
    long_twist_limit: float,
) -> Quaternion:
    long_axis = _RIGPED_LOCAL_AXES["Y"]
    hinge_axis = _RIGPED_LOCAL_AXES[hinge_axis_name]
    q = rotation.normalized()

    # Keep axial forearm/lower-leg roll as a separate limited DOF, then project
    # everything else onto the anatomical hinge. Any off-plane swing is removed
    # instead of allowing an elbow/knee to become a ball joint.
    long_angle = _signed_twist_angle(q, long_axis)
    long_full = Quaternion(long_axis, long_angle)
    without_long = (q @ long_full.inverted()).normalized()
    hinge_angle = _signed_twist_angle(without_long, hinge_axis)
    hinge = Quaternion(
        hinge_axis,
        _clamp_scalar(hinge_angle, float(hinge_min), float(hinge_max)),
    )
    long_twist = Quaternion(
        long_axis,
        _clamp_scalar(long_angle, -float(long_twist_limit), float(long_twist_limit)),
    )
    return (hinge @ long_twist).normalized()


def _clamped_generated_rigped_rotation(
    pose_bone,
    rotation: Quaternion,
) -> Quaternion:
    name = str(getattr(pose_bone, "name", ""))
    terminal = _RIGPED_TERMINAL_SWING_TWIST_LIMITS.get(name)
    if terminal is not None:
        return _clamp_ball_joint_rotation(
            rotation,
            swing_limit=terminal[0],
            twist_limit=terminal[1],
        )
    ball = _RIGPED_BALL_JOINT_LIMITS.get(name)
    if ball is not None:
        return _clamp_ball_joint_rotation(
            rotation,
            swing_limit=ball[0],
            twist_limit=ball[1],
        )
    hinge = _RIGPED_HINGE_JOINT_LIMITS.get(name)
    if hinge is not None:
        return _clamp_hinge_joint_rotation(
            rotation,
            hinge_axis_name=hinge[0],
            hinge_min=hinge[1],
            hinge_max=hinge[2],
            long_twist_limit=hinge[3],
        )
    return rotation.normalized()


def apply_rigped_joint_limits_to_pose_bone(pose_bone) -> bool:
    """Clamp one generated Rigped pose bone in-place while preserving storage mode."""

    if not isinstance(pose_bone, bpy.types.PoseBone):
        return False
    name = str(getattr(pose_bone, "name", ""))
    if (
        name not in _RIGPED_TERMINAL_SWING_TWIST_LIMITS
        and name not in _RIGPED_BALL_JOINT_LIMITS
        and name not in _RIGPED_HINGE_JOINT_LIMITS
    ):
        return False

    current = pose_bone.matrix_basis.to_quaternion().normalized()
    clamped = _clamped_generated_rigped_rotation(pose_bone, current)
    if float(current.rotation_difference(clamped).angle) <= 1e-6:
        return False

    if pose_bone.rotation_mode == "QUATERNION":
        pose_bone.rotation_quaternion = clamped
    elif pose_bone.rotation_mode == "AXIS_ANGLE":
        axis = Vector(clamped.axis)
        pose_bone.rotation_axis_angle = (
            float(clamped.angle),
            float(axis.x),
            float(axis.y),
            float(axis.z),
        )
    else:
        pose_bone.rotation_euler = clamped.to_euler(pose_bone.rotation_mode)
    return True


def apply_rigped_joint_limits_to_rig(rig) -> int:
    """Clamp public FK joints without fighting IK-authoritative Contact replay."""

    pose = getattr(rig, "pose", None)
    if pose is None:
        return 0

    # While a limb is Sliding or Planted, its visible public chain is only a
    # replay shell for the hidden IK solve. Re-clamping that public shell after
    # replay moves it away from MCH/IK and breaks the authoritative Contact pose.
    sliding_public_names: set[str] = set()
    for state_name, public_names, _result_names in _RIGPED_SLIDING_REPLAY_LIMBS:
        state_bone = pose.bones.get(state_name)
        if state_bone is None or AWB_CONTACT_STATE_PROPERTY not in state_bone:
            continue
        try:
            contact_type = type_for_state_value(
                float(state_bone[AWB_CONTACT_STATE_PROPERTY])
            )
        except (TypeError, ValueError):
            continue
        if contact_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}:
            sliding_public_names.update(str(name) for name in public_names)

    changed = 0
    for name in (
        *_RIGPED_BALL_JOINT_LIMITS,
        *_RIGPED_TERMINAL_SWING_TWIST_LIMITS,
        *_RIGPED_HINGE_JOINT_LIMITS,
    ):
        if str(name) in sliding_public_names:
            continue
        pose_bone = pose.bones.get(str(name))
        if pose_bone is not None and apply_rigped_joint_limits_to_pose_bone(pose_bone):
            changed += 1
    return changed


def _generated_rigped_local_rotation(
    resolved: ResolvedControl,
    desired_matrix: Matrix,
    *,
    parent_pose_matrix: Matrix | None = None,
) -> Quaternion:
    pose_bone = resolved.target
    if not isinstance(pose_bone, bpy.types.PoseBone):
        return Quaternion()
    rest = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        basis = rest.inverted_safe() @ desired_matrix
    else:
        parent_rest = pose_bone.parent.bone.matrix_local.copy()
        parent_pose = (
            parent_pose_matrix.copy()
            if parent_pose_matrix is not None
            else pose_bone.parent.matrix.copy()
        )
        basis = rest.inverted_safe() @ parent_rest @ parent_pose.inverted_safe() @ desired_matrix
    return basis.to_quaternion().normalized()


def _generated_rigped_rotation_within_limits(
    pose_bone,
    rotation: Quaternion,
    *,
    epsilon: float = 1e-5,
) -> bool:
    """Test one generated local rotation without projecting it to another pose."""

    name = str(getattr(pose_bone, "name", ""))
    terminal = _RIGPED_TERMINAL_SWING_TWIST_LIMITS.get(name)
    if terminal is not None:
        long_axis = _RIGPED_LOCAL_AXES["Y"]
        q = rotation.normalized()
        twist_angle = _signed_twist_angle(q, long_axis)
        twist = Quaternion(long_axis, twist_angle)
        swing = (q @ twist.inverted()).normalized()
        swing_angle = float(swing.angle)
        return (
            swing_angle <= float(terminal[0]) + epsilon
            and abs(twist_angle) <= float(terminal[1]) + epsilon
        )

    ball = _RIGPED_BALL_JOINT_LIMITS.get(name)
    if ball is not None:
        long_axis = _RIGPED_LOCAL_AXES["Y"]
        q = rotation.normalized()
        twist_angle = _signed_twist_angle(q, long_axis)
        twist = Quaternion(long_axis, twist_angle)
        swing = (q @ twist.inverted()).normalized()
        swing_angle = float(swing.angle)
        return (
            swing_angle <= float(ball[0]) + epsilon
            and abs(twist_angle) <= float(ball[1]) + epsilon
        )

    hinge = _RIGPED_HINGE_JOINT_LIMITS.get(name)
    if hinge is not None:
        hinge_axis = _RIGPED_LOCAL_AXES[hinge[0]]
        long_axis = _RIGPED_LOCAL_AXES["Y"]
        q = rotation.normalized()
        long_angle = _signed_twist_angle(q, long_axis)
        long_twist = Quaternion(long_axis, long_angle)
        without_long = (q @ long_twist.inverted()).normalized()
        hinge_angle = _signed_twist_angle(without_long, hinge_axis)
        hinge_rotation = Quaternion(hinge_axis, hinge_angle)
        off_hinge = float(without_long.rotation_difference(hinge_rotation).angle)
        return (
            float(hinge[1]) - epsilon <= hinge_angle <= float(hinge[2]) + epsilon
            and abs(long_angle) <= float(hinge[3]) + epsilon
            and off_hinge <= radians(4.0) + epsilon
        )
    return True


def _raw_direct_rotate_desired(
    active: ResolvedControl,
    states: tuple[DirectRotateControlState, ...],
    axis_world: Vector,
    angle: float,
    *,
    mirror_opposites: bool = True,
) -> tuple[dict[int, Matrix], dict[int, DirectRotateControlState]]:
    desired_by_bone: dict[int, Matrix] = {}
    state_by_bone: dict[int, DirectRotateControlState] = {}
    for state in states:
        control = state.control
        pose_bone = control.target
        if not isinstance(pose_bone, bpy.types.PoseBone):
            continue
        mapped_axis_world = _mapped_direct_rotate_axis(
            active,
            control,
            axis_world,
            mirror_opposites=mirror_opposites,
        )
        axis_local = control.owner_object.matrix_world.to_3x3().inverted_safe() @ mapped_axis_world
        if axis_local.length <= 1e-9:
            continue
        axis_local.normalize()
        pivot_local = Vector(state.pivot_local)
        rotation = (
            Matrix.Translation(pivot_local)
            @ Matrix.Rotation(float(angle), 4, axis_local)
            @ Matrix.Translation(-pivot_local)
        )
        pointer = int(pose_bone.as_pointer())
        desired_by_bone[pointer] = rotation @ state.start_matrix
        state_by_bone[pointer] = state
    return desired_by_bone, state_by_bone


def _direct_rotate_angle_is_valid(
    active: ResolvedControl,
    states: tuple[DirectRotateControlState, ...],
    axis_world: Vector,
    angle: float,
    *,
    mirror_opposites: bool = True,
) -> bool:
    desired_by_bone, state_by_bone = _raw_direct_rotate_desired(
        active,
        states,
        axis_world,
        angle,
        mirror_opposites=mirror_opposites,
    )
    for pointer, state in state_by_bone.items():
        pose_bone = state.control.target
        if not isinstance(pose_bone, bpy.types.PoseBone):
            continue
        name = str(pose_bone.name)
        if (
            name not in _RIGPED_TERMINAL_SWING_TWIST_LIMITS
            and name not in _RIGPED_BALL_JOINT_LIMITS
            and name not in _RIGPED_HINGE_JOINT_LIMITS
        ):
            continue
        parent_desired = None
        if pose_bone.parent is not None:
            parent_desired = desired_by_bone.get(int(pose_bone.parent.as_pointer()))
        rotation = _generated_rigped_local_rotation(
            state.control,
            desired_by_bone[pointer],
            parent_pose_matrix=parent_desired,
        )
        if not _generated_rigped_rotation_within_limits(pose_bone, rotation):
            return False
    return True


def _bounded_direct_rotate_angle(
    active: ResolvedControl,
    states: tuple[DirectRotateControlState, ...],
    axis_world: Vector,
    requested_angle: float,
    *,
    mirror_opposites: bool = True,
) -> float:
    """Stop a Rotate gesture at the first anatomical boundary; never wrap through it."""

    if not any(
        isinstance(state.control.target, bpy.types.PoseBone)
        and (
            str(state.control.target.name) in _RIGPED_TERMINAL_SWING_TWIST_LIMITS
            or str(state.control.target.name) in _RIGPED_BALL_JOINT_LIMITS
            or str(state.control.target.name) in _RIGPED_HINGE_JOINT_LIMITS
        )
        for state in states
    ):
        return float(requested_angle)
    requested = float(requested_angle)
    if abs(requested) <= 1e-10:
        return 0.0

    # No constrained Biped-style limb joint is allowed to make a complete half
    # turn in one gesture. More importantly, search the path in small increments
    # instead of testing only the final quaternion. A compound rotation can leave
    # an envelope and later re-enter it through an equivalent representation;
    # the animator must stop at the *first* invalid pose, not tunnel through it.
    direction = 1.0 if requested > 0.0 else -1.0
    target_magnitude = min(abs(requested), pi - radians(0.25))
    if not _direct_rotate_angle_is_valid(
        active,
        states,
        axis_world,
        0.0,
        mirror_opposites=mirror_opposites,
    ):
        return 0.0

    step_limit = radians(2.0)
    steps = max(1, int(target_magnitude / step_limit) + 1)
    last_valid = 0.0
    first_invalid: float | None = None
    for index in range(1, steps + 1):
        magnitude = target_magnitude * float(index) / float(steps)
        if _direct_rotate_angle_is_valid(
            active,
            states,
            axis_world,
            direction * magnitude,
            mirror_opposites=mirror_opposites,
        ):
            last_valid = magnitude
            continue
        first_invalid = magnitude
        break

    if first_invalid is None:
        return direction * target_magnitude if abs(requested) > target_magnitude else requested

    low = last_valid
    high = first_invalid
    for _index in range(18):
        middle = (low + high) * 0.5
        if _direct_rotate_angle_is_valid(
            active,
            states,
            axis_world,
            direction * middle,
            mirror_opposites=mirror_opposites,
        ):
            low = middle
        else:
            high = middle
    # Stay a hair inside the mathematical boundary. Matrix->quaternion->matrix
    # round-trips can otherwise put the applied pose a few ulps back outside the
    # predicate and cause visible hard-stop chatter on the next mouse event.
    safe_low = max(0.0, low - radians(0.02))
    return direction * safe_low


def _pose_matrix_from_basis(
    resolved: ResolvedControl,
    basis: Matrix,
    *,
    parent_pose_matrix: Matrix | None = None,
) -> Matrix:
    pose_bone = resolved.target
    if not isinstance(pose_bone, bpy.types.PoseBone):
        return basis.copy()
    rest = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        return rest @ basis
    parent_rest = pose_bone.parent.bone.matrix_local.copy()
    parent_pose = (
        parent_pose_matrix.copy()
        if parent_pose_matrix is not None
        else pose_bone.parent.matrix.copy()
    )
    return parent_pose @ parent_rest.inverted_safe() @ rest @ basis


def _hinge_state_from_basis(
    pose_bone,
    basis: Matrix,
) -> tuple[float, float, Quaternion] | None:
    hinge = _RIGPED_HINGE_JOINT_LIMITS.get(str(getattr(pose_bone, "name", "")))
    if hinge is None:
        return None
    hinge_axis = _RIGPED_LOCAL_AXES[hinge[0]]
    long_axis = _RIGPED_LOCAL_AXES["Y"]
    q = basis.to_quaternion().normalized()
    long_angle = _signed_twist_angle(q, long_axis)
    long_full = Quaternion(long_axis, long_angle)
    without_long = (q @ long_full.inverted()).normalized()
    hinge_angle = _signed_twist_angle(without_long, hinge_axis)
    hinge_full = Quaternion(hinge_axis, hinge_angle)
    residual = (without_long @ hinge_full.inverted()).normalized()
    return (
        _clamp_scalar(hinge_angle, float(hinge[1]), float(hinge[2])),
        _clamp_scalar(long_angle, -float(hinge[3]), float(hinge[3])),
        residual,
    )


def _lower_link_local_x_branch_sign(
    pose_bone_name: str,
    hinge_angle: float,
) -> float:
    """Return the semantic bend hemisphere for a Rigped lower-link LOCAL X."""

    name = str(pose_bone_name)
    fallback = -1.0 if name.startswith("ForeArm.") else 1.0
    if abs(float(hinge_angle)) <= radians(0.25):
        return fallback
    return 1.0 if float(hinge_angle) > 0.0 else -1.0


def _hinge_axis_weights(
    active: ResolvedControl,
    state: DirectRotateControlState,
    axis_world: Vector,
    *,
    mirror_opposites: bool = True,
    axis_name: str | None = None,
) -> tuple[float, float]:
    pose_bone = state.control.target
    if not isinstance(pose_bone, bpy.types.PoseBone):
        return (0.0, 0.0)
    hinge = _RIGPED_HINGE_JOINT_LIMITS.get(str(pose_bone.name))
    if hinge is None:
        return (0.0, 0.0)
    mapped_axis = _mapped_direct_rotate_axis(
        active,
        state.control,
        axis_world,
        mirror_opposites=mirror_opposites,
        axis_name=axis_name,
    )
    if mapped_axis.length <= 1e-9:
        return (0.0, 0.0)
    mapped_axis.normalize()
    basis_world = (
        state.control.owner_object.matrix_world.to_3x3()
        @ state.start_matrix.to_3x3().normalized()
    )
    hinge_world = basis_world @ _RIGPED_LOCAL_AXES[hinge[0]]
    long_world = basis_world @ _RIGPED_LOCAL_AXES["Y"]
    if hinge_world.length > 1e-9:
        hinge_world.normalize()
    if long_world.length > 1e-9:
        long_world.normalize()
    return (
        float(mapped_axis.dot(hinge_world)),
        float(mapped_axis.dot(long_world)),
    )


def _bounded_hinge_direct_rotate_angle(
    active: ResolvedControl,
    states: tuple[DirectRotateControlState, ...],
    axis_world: Vector,
    requested_angle: float,
    *,
    mirror_opposites: bool = True,
    axis_name: str | None = None,
) -> float:
    lower = -float("inf")
    upper = float("inf")
    found = False
    effective = False
    for state in states:
        pose_bone = state.control.target
        hinge_state = state.hinge_state
        if not isinstance(pose_bone, bpy.types.PoseBone) or hinge_state is None:
            continue
        limits = _RIGPED_HINGE_JOINT_LIMITS.get(str(pose_bone.name))
        if limits is None:
            continue
        found = True
        hinge_weight, long_weight = _hinge_axis_weights(
            active,
            state,
            axis_world,
            mirror_opposites=mirror_opposites,
            axis_name=axis_name,
        )
        for start, weight, minimum, maximum in (
            (hinge_state[0], hinge_weight, float(limits[1]), float(limits[2])),
            (hinge_state[1], long_weight, -float(limits[3]), float(limits[3])),
        ):
            if abs(weight) <= 1e-7:
                continue
            effective = True
            a = (minimum - float(start)) / weight
            b = (maximum - float(start)) / weight
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
    if not found:
        return float(requested_angle)
    if not effective:
        return 0.0
    if lower > upper:
        return 0.0
    return _clamp_scalar(float(requested_angle), lower, upper)


def _hinge_direct_rotate_desired(
    active: ResolvedControl,
    state: DirectRotateControlState,
    axis_world: Vector,
    angle: float,
    *,
    mirror_opposites: bool = True,
    axis_name: str | None = None,
    parent_pose_matrix: Matrix | None = None,
) -> Matrix | None:
    pose_bone = state.control.target
    hinge_state = state.hinge_state
    if not isinstance(pose_bone, bpy.types.PoseBone) or hinge_state is None:
        return None
    limits = _RIGPED_HINGE_JOINT_LIMITS.get(str(pose_bone.name))
    if limits is None:
        return None
    hinge_weight, long_weight = _hinge_axis_weights(
        active,
        state,
        axis_world,
        mirror_opposites=mirror_opposites,
        axis_name=axis_name,
    )
    hinge_angle = _clamp_scalar(
        float(hinge_state[0]) + float(angle) * hinge_weight,
        float(limits[1]),
        float(limits[2]),
    )
    long_angle = _clamp_scalar(
        float(hinge_state[1]) + float(angle) * long_weight,
        -float(limits[3]),
        float(limits[3]),
    )
    residual = hinge_state[2]
    rotation = (
        residual
        @ Quaternion(_RIGPED_LOCAL_AXES[limits[0]], hinge_angle)
        @ Quaternion(_RIGPED_LOCAL_AXES["Y"], long_angle)
    ).normalized()
    location, _rotation, scale = state.start_basis.decompose()
    basis = Matrix.LocRotScale(location, rotation, scale)
    return _pose_matrix_from_basis(
        state.control,
        basis,
        parent_pose_matrix=parent_pose_matrix,
    )


def _mapped_direct_rotate_axis(
    active: ResolvedControl,
    target: ResolvedControl,
    axis_world: Vector,
    *,
    mirror_opposites: bool = True,
    axis_name: str | None = None,
) -> Vector:
    active_target = active.target
    target_target = target.target
    if (
        mirror_opposites
        and isinstance(active_target, bpy.types.PoseBone)
        and isinstance(target_target, bpy.types.PoseBone)
    ):
        opposite = _opposite_control_name(str(active_target.name))
        if opposite is not None and str(target_target.name) == opposite:
            lower_link_pair = {
                str(active_target.name),
                str(target_target.name),
            } <= {"ForeArm.L", "ForeArm.R", "Calf.L", "Calf.R"}
            mirrored = _mirror_control_world_vector(
                target,
                axis_world,
                axial=not (lower_link_pair and axis_name == "Y"),
            )
            if mirrored.length > 1e-9:
                mirrored.normalize()
            return mirrored
    return Vector(axis_world).normalized()


def _capture_hinge_settings(owner) -> tuple[Any, ...]:
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


def _restore_hinge_settings(owner, snapshot: tuple[Any, ...]) -> None:
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


def _direct_rotate_sliding_sync_sessions(
    context,
    operation_domain: OperationDomainSnapshot,
) -> tuple[SlidingRotateSyncSession, ...]:
    control_context = control_context_for_context(context)
    view = resolve_character(context.scene, operation_domain.character_id)
    selected_binding_ids = frozenset(operation_domain.selected_binding_ids)

    sessions: list[SlidingRotateSyncSession] = []
    for mapping_id in operation_domain.contact_mapping_ids:
        representation = resolve_limb_representation_capability(view, mapping_id)
        capability = representation.capability
        if capability is None:
            raise RigpedSemanticMoveError(
                "Sliding Rotate lost a selected limb representation capability."
            )
        solver = capability.native_ik.solver_owner.target
        if AWB_CONTACT_STATE_PROPERTY not in solver:
            continue
        contact_type = type_for_state_value(float(solver[AWB_CONTACT_STATE_PROPERTY]))
        if contact_type is not ContactKeyType.SLIDING:
            continue

        pole_target = capability.native_ik.pole_target
        if pole_target is None:
            raise RigpedSemanticMoveError(
                "Sliding Rotate requires the generated IK pole target."
            )
        planned = build_contact_intent_plan(
            context.scene,
            control_context,
            operation_id=f"rigped-direct-rotate-preview:{mapping_id}",
            mode=ContactAuthoringMode.ANCHOR,
            enabled_types=(ContactKeyType.FREE, ContactKeyType.SLIDING),
            mapping_id=mapping_id,
        )
        if (
            not planned.ok
            or planned.plan is None
            or planned.plan.target_type is not ContactKeyType.SLIDING
        ):
            raise RigpedSemanticMoveError(
                "Sliding Rotate could not prepare a coherent IK preview for the selected limb."
            )
        sessions.append(
            SlidingRotateSyncSession(
                intent=planned.plan,
                capability=capability,
                terminal_selected=(
                    str(capability.authored_terminal_binding_id)
                    in selected_binding_ids
                ),
                start_fk_states=tuple(
                    _capture_control_state(control)
                    for control in capability.fk_controls
                ),
                start_terminal_state=_capture_control_state(
                    capability.authored_terminal
                ),
                start_solver_state=_capture_control_state(
                    capability.native_ik.solver_owner
                ),
                start_ik_state=_capture_control_state(capability.native_ik.ik_target),
                start_pole_state=_capture_control_state(pole_target),
                start_pole_angle=float(capability.native_ik.constraint.pole_angle),
                start_ik_influence=float(capability.native_ik.constraint.influence),
                start_terminal_ik_influence=float(
                    capability.terminal_ik_constraint.influence
                ),
                start_feedback_mutes=tuple(
                    bool(constraint.mute)
                    for constraint in (
                        *capability.fk_copy_constraints,
                        capability.terminal_fk_constraint,
                    )
                ),
                start_hinge_settings=_capture_hinge_settings(
                    capability.native_ik.solver_owner.target
                ),
            )
        )
    return tuple(sessions)


def _sliding_capabilities_for_character(
    scene,
    character_id: str,
) -> tuple[LimbRepresentationCapability, ...]:
    """Freeze current IK-authoritative Contact membership for live body overlays."""

    view = resolve_character(scene, character_id)
    capabilities: list[LimbRepresentationCapability] = []
    for mapping in view.definition.kinematics:
        capability = resolve_limb_representation_capability(
            view,
            mapping.mapping_id,
        ).capability
        if capability is None:
            continue
        solver_owner = capability.native_ik.solver_owner.target
        if AWB_CONTACT_STATE_PROPERTY not in solver_owner:
            continue
        contact_type = type_for_state_value(
            float(solver_owner[AWB_CONTACT_STATE_PROPERTY])
        )
        if contact_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}:
            capabilities.append(capability)
    return tuple(capabilities)


def _current_sliding_capabilities(
    context,
) -> tuple[LimbRepresentationCapability, ...]:
    control_context = control_context_for_context(context)
    resolution = resolve_rigped_target(context.scene, control_context)
    target = resolution.target
    if target is None:
        return ()
    return _sliding_capabilities_for_character(context.scene, target.character_id)


def _capture_sliding_hidden_seed_snapshots(
    capabilities: tuple[LimbRepresentationCapability, ...],
) -> tuple[SlidingHiddenSeedSnapshot, ...]:
    return tuple(
        SlidingHiddenSeedSnapshot(
            capability=capability,
            result_states=tuple(
                _capture_control_state(control)
                for control in capability.result_controls
            ),
            terminal_state=_capture_control_state(capability.result_terminal),
        )
        for capability in capabilities
    )


def _restore_sliding_hidden_seed_snapshots(
    context,
    snapshots: tuple[SlidingHiddenSeedSnapshot, ...],
    *,
    update: bool = True,
) -> None:
    for snapshot in snapshots:
        for control, state in zip(
            snapshot.capability.result_controls,
            snapshot.result_states,
            strict=True,
        ):
            _apply_control_state(control, state)
        _apply_control_state(
            snapshot.capability.result_terminal,
            snapshot.terminal_state,
        )
    if update:
        context.view_layer.update()


def _trace_matrix(matrix: Matrix | None):
    if matrix is None:
        return None
    return tuple(
        tuple(float(matrix[row][column]) for column in range(4))
        for row in range(4)
    )


def _angle_distance(left: float, right: float) -> float:
    return abs(((float(left) - float(right) + pi) % (2.0 * pi)) - pi)


def _capture_sliding_dependency_guards(
    capabilities: tuple[LimbRepresentationCapability, ...],
) -> tuple[SlidingDependencyGuard, ...]:
    guards: list[SlidingDependencyGuard] = []
    for capability in capabilities:
        pole_target = capability.native_ik.pole_target
        if pole_target is None:
            raise RigpedSemanticMoveError(
                "Sliding body dependency requires the generated IK pole target."
            )
        native_ik = capability.native_ik.constraint
        terminal_ik = capability.terminal_ik_constraint
        guards.append(
            SlidingDependencyGuard(
                capability=capability,
                reference=capture_sliding_authored_reference(capability),
                start_ik_state=_capture_control_state(capability.native_ik.ik_target),
                start_pole_state=_capture_control_state(pole_target),
                start_ik_influence=float(native_ik.influence),
                start_ik_mute=bool(native_ik.mute),
                start_terminal_ik_influence=float(terminal_ik.influence),
                start_terminal_ik_mute=bool(terminal_ik.mute),
                start_hinge_settings=_capture_hinge_settings(
                    capability.native_ik.solver_owner.target
                ),
                start_result_terminal=capability.result_terminal.target.matrix.copy(),
                start_public_terminal=capability.authored_terminal.target.matrix.copy(),
                ik_runtime_key=runtime_control_key(capability.native_ik.ik_target),
                pole_runtime_key=runtime_control_key(pole_target),
            )
        )
    return tuple(guards)


def _restore_sliding_dependency_guards(
    context,
    guards: tuple[SlidingDependencyGuard, ...],
    *,
    update: bool = True,
) -> None:
    for guard in guards:
        capability = guard.capability
        _apply_control_state(capability.native_ik.ik_target, guard.start_ik_state)
        pole_target = capability.native_ik.pole_target
        if guard.start_pole_state is not None:
            if pole_target is None:
                raise RigpedSemanticMoveError(
                    "Sliding body dependency lost its pole target during restore."
                )
            _apply_control_state(pole_target, guard.start_pole_state)
        native_ik = capability.native_ik.constraint
        terminal_ik = capability.terminal_ik_constraint
        native_ik.influence = float(guard.start_ik_influence)
        native_ik.mute = bool(guard.start_ik_mute)
        terminal_ik.influence = float(guard.start_terminal_ik_influence)
        terminal_ik.mute = bool(guard.start_terminal_ik_mute)
        _restore_hinge_settings(
            capability.native_ik.solver_owner.target,
            guard.start_hinge_settings,
        )
    if update:
        context.view_layer.update()


def _sliding_dependency_guard_diagnostics(
    guards: tuple[SlidingDependencyGuard, ...],
) -> tuple[tuple[dict[str, Any], ...], bool]:
    diagnostics: list[dict[str, Any]] = []
    violated = False
    matrix_position_tolerance = 1e-7
    matrix_rotation_tolerance = 1e-7
    scalar_tolerance = 1e-9

    for guard in guards:
        capability = guard.capability
        try:
            mapping_id = str(capability.native_ik.mapping_id)
            pole_target = capability.native_ik.pole_target
            live_ik_key = runtime_control_key(capability.native_ik.ik_target)
            live_pole_key = runtime_control_key(pole_target) if pole_target is not None else None
            live_reference = capture_sliding_authored_reference(capability)
            native_ik = capability.native_ik.constraint
            terminal_ik = capability.terminal_ik_constraint
            hinge_settings = _capture_hinge_settings(
                capability.native_ik.solver_owner.target
            )
        except (ReferenceError, RuntimeError, ValueError, TypeError):
            diagnostics.append(
                {
                    "mapping_id": guard.reference.mapping_id,
                    "stale": True,
                }
            )
            violated = True
            continue

        target_position, target_rotation = _pose_residual(
            live_reference.target_matrix,
            guard.reference.target_matrix,
        )
        if guard.reference.pole_matrix is None or live_reference.pole_matrix is None:
            pole_position = 0.0
            pole_rotation = 0.0
            pole_presence_changed = (
                guard.reference.pole_matrix is None
            ) != (
                live_reference.pole_matrix is None
            )
        else:
            pole_position, pole_rotation = _pose_residual(
                live_reference.pole_matrix,
                guard.reference.pole_matrix,
            )
            pole_presence_changed = False
        pole_angle_delta = _angle_distance(
            live_reference.pole_angle,
            guard.reference.pole_angle,
        )
        runtime_identity_changed = (
            mapping_id != guard.reference.mapping_id
            or live_ik_key != guard.ik_runtime_key
            or live_pole_key != guard.pole_runtime_key
        )
        ik_raw_changed = (
            abs(float(native_ik.influence) - guard.start_ik_influence) > scalar_tolerance
            or bool(native_ik.mute) != guard.start_ik_mute
            or abs(float(terminal_ik.influence) - guard.start_terminal_ik_influence)
            > scalar_tolerance
            or bool(terminal_ik.mute) != guard.start_terminal_ik_mute
            or hinge_settings != guard.start_hinge_settings
        )
        authority_changed = (
            runtime_identity_changed
            or pole_presence_changed
            or target_position > matrix_position_tolerance
            or target_rotation > matrix_rotation_tolerance
            or pole_position > matrix_position_tolerance
            or pole_rotation > matrix_rotation_tolerance
            or pole_angle_delta > scalar_tolerance
            or ik_raw_changed
        )
        violated = authority_changed or violated
        diagnostics.append(
            {
                "mapping_id": mapping_id,
                "stale": False,
                "runtime_identity_changed": runtime_identity_changed,
                "target_position_delta": float(target_position),
                "target_rotation_delta": float(target_rotation),
                "pole_position_delta": float(pole_position),
                "pole_rotation_delta": float(pole_rotation),
                "pole_angle_delta": float(pole_angle_delta),
                "ik_raw_changed": bool(ik_raw_changed),
                "target_before": _trace_matrix(guard.reference.target_matrix),
                "target_after": _trace_matrix(live_reference.target_matrix),
                "pole_before": _trace_matrix(guard.reference.pole_matrix),
                "pole_after": _trace_matrix(live_reference.pole_matrix),
            }
        )
    return tuple(diagnostics), violated


def _assert_sliding_dependency_authority(
    context,
    guards: tuple[SlidingDependencyGuard, ...],
    *,
    operation_id: str | None,
    phase: str,
) -> tuple[dict[str, Any], ...]:
    diagnostics, violated = _sliding_dependency_guard_diagnostics(guards)
    if violated:
        trace_event(
            "ERROR",
            "SLIDING_BODY_AUTHORITY_DRIFT",
            operation_id=operation_id,
            context=context,
            phase=phase,
            diagnostics=diagnostics,
        )
        raise RigpedSemanticMoveError(
            "Sliding body dependency changed frozen target/pole authority."
        )
    return diagnostics


def _refresh_frozen_sliding_dependency_overlays(
    context,
    guards: tuple[SlidingDependencyGuard, ...],
    *,
    operation_id: str | None,
    phase: str,
) -> int:
    if not guards:
        return 0

    _assert_sliding_dependency_authority(
        context,
        guards,
        operation_id=operation_id,
        phase=f"{phase}:PRE",
    )
    result_before = tuple(
        capture_native_solved_result(guard.capability)
        for guard in guards
    )
    seed_fired = tuple(
        bool(_seed_stalled_sliding_native_ik(context, guard.capability))
        for guard in guards
    )
    capabilities = tuple(guard.capability for guard in guards)
    refreshed = _refresh_current_sliding_public_overlays(
        context,
        capabilities=capabilities,
        allow_seed=False,
    )
    authority_diagnostics = _assert_sliding_dependency_authority(
        context,
        guards,
        operation_id=operation_id,
        phase=f"{phase}:POST",
    )
    result_after = tuple(
        capture_native_solved_result(guard.capability)
        for guard in guards
    )

    per_limb = []
    for guard, before, after, did_seed, authority in zip(
        guards,
        result_before,
        result_after,
        seed_fired,
        authority_diagnostics,
        strict=True,
    ):
        reference = capture_sliding_authored_reference(guard.capability)
        target_position, target_rotation = _pose_residual(
            after.terminal_pose,
            reference.target_matrix,
        )
        public_position, public_rotation = _sliding_public_pose_residual(
            guard.capability
        )
        per_limb.append(
            {
                "mapping_id": guard.reference.mapping_id,
                "seed_fired": bool(did_seed),
                "authority": authority,
                "result_before": _trace_matrix(before.terminal_pose),
                "result_after_projection": _trace_matrix(after.terminal_pose),
                "result_target_position": float(target_position),
                "result_target_rotation": float(target_rotation),
                "public_result_position": float(public_position),
                "public_result_rotation": float(public_rotation),
            }
        )
    trace_event(
        "DIAGNOSTIC",
        "SLIDING_BODY_DEPENDENCY_PREVIEW",
        operation_id=operation_id,
        context=context,
        phase=phase,
        mapping_ids=tuple(guard.reference.mapping_id for guard in guards),
        limbs=tuple(per_limb),
    )
    return refreshed


def _passive_sliding_capabilities(
    current_sliding: tuple[LimbRepresentationCapability, ...],
    active_syncs: tuple[SlidingRotateSyncSession, ...],
) -> tuple[LimbRepresentationCapability, ...]:
    active_mapping_ids = {
        str(session.capability.native_ik.mapping_id)
        for session in active_syncs
    }
    return tuple(
        capability
        for capability in current_sliding
        if str(capability.native_ik.mapping_id) not in active_mapping_ids
    )


def _sliding_capability_mapping_ids(
    capabilities: tuple[LimbRepresentationCapability, ...],
) -> tuple[str, ...]:
    return tuple(
        str(capability.native_ik.mapping_id)
        for capability in capabilities
    )


def _configured_generated_hinge_branch_sign(solver_owner) -> int | None:
    name = str(getattr(solver_owner, "name", ""))
    if name.startswith(("MCH_ForeArm", "MCH_Calf")):
        if not bool(solver_owner.use_ik_limit_x):
            return None
        minimum = float(solver_owner.ik_min_x)
        maximum = float(solver_owner.ik_max_x)
    else:
        return None

    epsilon = 1e-6
    if minimum >= -epsilon and maximum > epsilon:
        return 1
    if maximum <= epsilon and minimum < -epsilon:
        return -1
    return None


def _transient_solver_seed_branch_sign(
    capability: LimbRepresentationCapability,
    solved_result,
    *,
    root_world: Vector,
    joint_world: Vector,
    end_world: Vector,
    desired_joint_world: Vector,
    desired_end_world: Vector,
    bend_plane_normal_world: Vector,
) -> int | None:
    owner = capability.result_controls[0].owner_object
    owner_inverse = owner.matrix_world.inverted_safe()
    owner_basis_inverse = owner.matrix_world.to_3x3().inverted_safe()

    root = Vector(owner_inverse @ root_world)
    start_joint = Vector(owner_inverse @ joint_world)
    start_end = Vector(owner_inverse @ end_world)
    desired_joint = Vector(owner_inverse @ desired_joint_world)
    desired_end = Vector(owner_inverse @ desired_end_world)
    start_plane = Vector(owner_basis_inverse @ bend_plane_normal_world)
    desired_plane_world = (
        Vector(desired_end_world) - Vector(root_world)
    ).cross(Vector(desired_joint_world) - Vector(root_world))
    if desired_plane_world.length <= 1e-8:
        desired_plane_world = Vector(bend_plane_normal_world)
    desired_plane = Vector(owner_basis_inverse @ desired_plane_world)

    first_matrix = _plane_aligned_pose_matrix(
        solved_result.first_pose,
        root,
        start_joint,
        start_plane,
        root,
        desired_joint,
        desired_plane,
    )
    second_matrix = _plane_aligned_pose_matrix(
        solved_result.second_pose,
        start_joint,
        start_end,
        start_plane,
        desired_joint,
        desired_end,
        desired_plane,
    )
    if first_matrix is None or second_matrix is None:
        return None

    second_basis = native_pose_basis_from_matrix(
        capability.result_controls[1].target,
        second_matrix,
        parent_pose_matrix=first_matrix,
    )
    quaternion = second_basis.to_quaternion().normalized()
    solver_name = str(
        getattr(capability.native_ik.solver_owner.target, "name", "")
    )
    if solver_name.startswith(("MCH_ForeArm", "MCH_Calf")):
        component = float(quaternion.x)
    else:
        return None

    angle = 2.0 * atan2(component, float(quaternion.w))
    angle = ((angle + pi) % (2.0 * pi)) - pi
    if abs(angle) <= radians(0.25):
        return None
    return 1 if angle > 0.0 else -1


def _seed_stalled_sliding_native_ik(
    context,
    capability: LimbRepresentationCapability,
) -> bool:
    """Escape a reachable near-straight native IK stall without moving authority."""

    native_ik = capability.native_ik.constraint
    if bool(native_ik.mute) or float(native_ik.influence) <= 1e-6:
        return False
    pole_target = capability.native_ik.pole_target
    if pole_target is None:
        return False

    first_result = capability.result_controls[0].target
    second_result = capability.result_controls[1].target
    if not all(
        isinstance(item, bpy.types.PoseBone)
        for item in (first_result, second_result)
    ):
        return False

    owner = capability.result_controls[0].owner_object
    root_world = Vector(owner.matrix_world @ first_result.head)
    joint_world = Vector(owner.matrix_world @ first_result.tail)
    end_world = Vector(owner.matrix_world @ second_result.tail)
    first_length = float((joint_world - root_world).length)
    second_length = float((end_world - joint_world).length)
    if first_length <= 1e-8 or second_length <= 1e-8:
        return False

    target_world = _world_position(capability.native_ik.ik_target)
    target_axis = target_world - root_world
    target_distance = float(target_axis.length)
    total_length = first_length + second_length
    reach_epsilon = max(1e-6, total_length * 1e-5)
    if (
        target_distance <= abs(first_length - second_length) + reach_epsilon
        or target_distance >= total_length - reach_epsilon
    ):
        # Folded or reach-saturated chains are not straight-singularity stalls.
        return False
    target_direction = target_axis.normalized()

    actual_tip_world = Vector(
        evaluated_chain_tip_world_position(capability.native_ik)
    )
    target_error = float((actual_tip_world - target_world).length)
    target_tolerance = max(1e-6, total_length * 1e-5)
    if target_error <= target_tolerance:
        return False

    current_axis = end_world - root_world
    current_distance = float(current_axis.length)
    if current_distance <= 1e-9:
        return False
    current_direction = current_axis.normalized()
    current_along = (
        first_length * first_length
        - second_length * second_length
        + current_distance * current_distance
    ) / (2.0 * current_distance)
    current_perp = joint_world - (
        root_world + current_direction * current_along
    )
    if current_perp.length > max(1e-5, total_length * 0.01):
        return False

    pole_world = _world_position(pole_target)
    preferred_bend = pole_world - (
        root_world
        + target_direction
        * float((pole_world - root_world).dot(target_direction))
    )
    if preferred_bend.length <= 1e-8:
        return False
    preferred_bend.normalize()

    solved_result = capture_native_solved_result(capability)
    configured_branch = _configured_generated_hinge_branch_sign(
        capability.native_ik.solver_owner.target
    )
    selected_seed = None
    candidate_bends = (
        preferred_bend.copy(),
        -preferred_bend.copy(),
    )
    for candidate_bend in candidate_bends:
        solver_seed = build_transient_solver_seed(
            root_world=root_world,
            target_world=target_world,
            preferred_bend_world=candidate_bend,
            first_length=first_length,
            second_length=second_length,
            minimum_bend_radians=radians(8.0),
        )
        if solver_seed is None:
            continue

        bend_plane_normal = target_direction.cross(candidate_bend)
        if bend_plane_normal.length <= 1e-8:
            continue
        bend_plane_normal.normalize()

        if configured_branch is not None:
            candidate_branch = _transient_solver_seed_branch_sign(
                capability,
                solved_result,
                root_world=root_world,
                joint_world=joint_world,
                end_world=end_world,
                desired_joint_world=Vector(solver_seed.joint_world),
                desired_end_world=Vector(solver_seed.end_world),
                bend_plane_normal_world=bend_plane_normal,
            )
            if candidate_branch != configured_branch:
                continue

        selected_seed = (
            candidate_bend,
            solver_seed,
            bend_plane_normal,
        )
        break

    if selected_seed is None:
        return False
    preferred_bend, solver_seed, bend_plane_normal = selected_seed
    seed_session = FkTwoBoneMoveSession(
        capability=capability,
        active_control=capability.result_terminal,
        first_control=capability.result_controls[0],
        second_control=capability.result_controls[1],
        terminal_control=capability.result_terminal,
        first_start_basis=capability.result_controls[0].target.matrix_basis.copy(),
        second_start_basis=capability.result_controls[1].target.matrix_basis.copy(),
        terminal_start_basis=capability.result_terminal.target.matrix_basis.copy(),
        first_start_matrix=solved_result.first_pose.copy(),
        second_start_matrix=solved_result.second_pose.copy(),
        terminal_start_matrix=solved_result.terminal_pose.copy(),
        root_world=root_world,
        joint_world=joint_world,
        end_world=end_world,
        baseline_perp_world=preferred_bend.copy(),
        bend_plane_normal_world=bend_plane_normal.copy(),
        last_direction_world=target_direction.copy(),
        last_bend_world=preferred_bend.copy(),
        first_length=first_length,
        second_length=second_length,
    )

    was_muted = bool(native_ik.mute)
    try:
        native_ik.mute = True
        if not _apply_solved_two_bone_fk_pose(
            seed_session,
            Vector(solver_seed.joint_world),
            Vector(solver_seed.end_world),
        ):
            return False
        context.view_layer.update()
    finally:
        native_ik.mute = was_muted
    context.view_layer.update()
    return True


def _refresh_current_sliding_public_overlays(
    context,
    *,
    capabilities: tuple[LimbRepresentationCapability, ...] | None = None,
    max_passes: int = 6,
    allow_seed: bool = True,
) -> int:
    if capabilities is None:
        capabilities = _current_sliding_capabilities(context)
    if not capabilities:
        return 0

    feedback_changed = False
    for capability in capabilities:
        feedback_changed = (
            bool(set_limb_fk_feedback_muted(capability, True))
            or feedback_changed
        )
    if feedback_changed:
        context.view_layer.update()

    # Public FK is a derived cache while Sliding. Rebuild all Sliding limbs as
    # one depsgraph batch and settle the writable public/result relation inside
    # the same preview event, rather than leaking one convergence step into the
    # next mouse event or release. The projection writes rotation channels only;
    # position residual is observed for diagnostics but is not a failure gate.
    rotation_tolerance = 5e-7

    # SolverSeed is singularity initialization only. It must not be
    # re-injected on every public/result convergence pass or it becomes a
    # competing iterative solver and can repeatedly perturb near-straight
    # chains under non-identity ancestor transforms. Cancel/restore paths pass
    # allow_seed=False so operation-local hidden seed inputs return exactly to
    # the gesture-start snapshot instead of being immediately re-authored.
    if allow_seed:
        for capability in capabilities:
            _seed_stalled_sliding_native_ik(context, capability)

    last_position = 0.0
    last_rotation = 0.0
    residual_history: list[tuple[int, float, float]] = []
    for _pass_index in range(max(1, int(max_passes))):
        for capability in capabilities:
            _sync_sliding_public_pose_from_result(
                context,
                capability,
                update=False,
            )
        context.view_layer.update()

        converged = True
        last_position = 0.0
        last_rotation = 0.0
        for capability in capabilities:
            position, rotation = _sliding_public_pose_residual(capability)
            last_position = max(last_position, position)
            last_rotation = max(last_rotation, rotation)
            if rotation > rotation_tolerance:
                converged = False
        residual_history.append(
            (_pass_index + 1, float(last_position), float(last_rotation))
        )
        if converged:
            # Constraint feedback can look converged in the same depsgraph tick
            # that wrote the public overlay, then drift on the next evaluation
            # because the always-on FK Copy Rotation inputs feed the public pose
            # back into the native IK chain. Require one no-write stability
            # update before accepting convergence.
            context.view_layer.update()
            stable = True
            for capability in capabilities:
                position, rotation = _sliding_public_pose_residual(capability)
                last_position = max(last_position, position)
                last_rotation = max(last_rotation, rotation)
                if rotation > rotation_tolerance:
                    stable = False
            if stable:
                return len(capabilities)

    trace_event(
        "DIAGNOSTIC",
        "SLIDING_CONVERGENCE_FAILURE",
        context=context,
        mapping_ids=tuple(
            str(capability.native_ik.mapping_id)
            for capability in capabilities
        ),
        max_passes=max(1, int(max_passes)),
        residual_history=tuple(residual_history),
        final_position=float(last_position),
        final_rotation=float(last_rotation),
    )
    raise RigpedSemanticMoveError(
        "Sliding public/result batch convergence failed: "
        f"position={last_position:.9g} rotation={last_rotation:.9g}"
    )


def _restore_direct_rotate_sliding_sync_session(
    context,
    session: SlidingRotateSyncSession,
    *,
    update: bool = True,
) -> None:
    capability = session.capability
    pole_target = capability.native_ik.pole_target
    if pole_target is None:
        return
    native_ik = capability.native_ik.constraint
    terminal_ik = capability.terminal_ik_constraint
    native_ik.influence = 0.0
    terminal_ik.influence = 0.0
    _apply_control_state(
        capability.native_ik.solver_owner,
        session.start_solver_state,
    )
    _apply_control_state(capability.native_ik.ik_target, session.start_ik_state)
    _apply_control_state(pole_target, session.start_pole_state)
    native_ik.pole_angle = session.start_pole_angle
    _restore_hinge_settings(
        capability.native_ik.solver_owner.target,
        session.start_hinge_settings,
    )
    for control, state in zip(
        capability.fk_controls,
        session.start_fk_states,
        strict=True,
    ):
        _apply_control_state(control, state)
    _apply_control_state(
        capability.authored_terminal,
        session.start_terminal_state,
    )
    feedback_constraints = (
        *capability.fk_copy_constraints,
        capability.terminal_fk_constraint,
    )
    for constraint, start_mute in zip(
        feedback_constraints,
        session.start_feedback_mutes,
        strict=True,
    ):
        constraint.mute = bool(start_mute)
    native_ik.influence = session.start_ik_influence
    terminal_ik.influence = session.start_terminal_ik_influence
    if update:
        context.view_layer.update()


def _apply_direct_rotate_sliding_sync(context, session: SlidingRotateSyncSession) -> None:
    capability = session.capability
    pole_target = capability.native_ik.pole_target
    if pole_target is None:
        raise RigpedSemanticMoveError("Sliding Rotate lost its generated IK pole target.")
    native_ik = capability.native_ik.constraint
    terminal_ik = capability.terminal_ik_constraint
    solver_owner = capability.native_ik.solver_owner.target

    previous_ik_influence = float(native_ik.influence)
    previous_terminal_influence = float(terminal_ik.influence)
    feedback_constraints = (
        *capability.fk_copy_constraints,
        capability.terminal_fk_constraint,
    )
    previous_feedback_mutes = tuple(
        bool(constraint.mute) for constraint in feedback_constraints
    )

    try:
        # Preview is intentionally lighter than the persistent C gate. Direct
        # Rotate first exposes the current public FK pose, then derives the
        # equivalent generated IK target/pole bundle without running the exact
        # residual acceptance check on every mouse move. C remains the strict
        # persistent validation boundary.
        #
        # Sliding normally mutes public->MCH FK feedback. Active Rotate is the
        # one bounded exception: while native IK is disabled, expose the user's
        # temporary public FK pose to the hidden result chain, derive the new
        # target/pole intent, then restore the frozen feedback-mute authority
        # before native IK is re-enabled. Passive limbs never enter this phase.
        native_ik.influence = 0.0
        terminal_ik.influence = 0.0
        for constraint in feedback_constraints:
            constraint.mute = False
        context.view_layer.update()

        desired_terminal = capability.result_terminal.target.matrix.copy()
        branch_sign = _generated_rigped_hinge_branch(capability)
        pole_solution = _initial_pole_solution(capability)
        if pole_solution is None or not pole_solution.ok or pole_solution.position is None:
            raise RigpedSemanticMoveError(
                "Sliding Rotate could not derive a stable preview pole from the current limb pose."
            )
        pole_angle = _derived_pole_angle(capability, pole_solution.position)
        pole_position, pole_angle = _canonical_generated_rigped_pole(
            capability,
            pole_solution.position,
            float(pole_angle),
            branch_sign,
        )

        # Active Sliding Rotate is an authored target-transport gesture: the
        # current public FK preview defines the new terminal target while Sliding
        # remains IK-authoritative.  Body/ancestor dependency in E7 is the
        # separate pinned-target case; do not import that semantics here.
        ik_state = _state_for_pose_matrix(
            capability.native_ik.ik_target,
            desired_terminal,
        )
        pole_matrix = pole_target.target.matrix.copy()
        pole_matrix.translation = (
            pole_target.owner_object.matrix_world.inverted_safe() @ Vector(pole_position)
        )
        pole_state = _state_for_pose_matrix(pole_target, pole_matrix)
        for constraint, previous_mute in zip(
            feedback_constraints,
            previous_feedback_mutes,
            strict=True,
        ):
            constraint.mute = bool(previous_mute)
        context.view_layer.update()
        _apply_control_state(
            capability.native_ik.ik_target,
            ik_state,
            location=True,
            rotation=True,
        )
        _apply_control_state(
            pole_target,
            pole_state,
            location=True,
            rotation=False,
        )
        native_ik.pole_angle = float(pole_angle)
        if branch_sign is not None:
            configure_generated_rigped_ik_hinge_branch(solver_owner, branch_sign)
        native_ik.influence = previous_ik_influence
        terminal_ik.influence = previous_terminal_influence
        context.view_layer.update()

        # Sliding is IK-authoritative between keys. Show the actual native IK
        # result immediately so the public limb and red pivot remain one
        # coherent preview instead of leaving an FK-only overlay behind.
        desired_first = capability.result_controls[0].target.matrix.copy()
        desired_second = capability.result_controls[1].target.matrix.copy()
        solved_terminal = capability.result_terminal.target.matrix.copy()
        first_state = _state_for_pose_matrix(capability.fk_controls[0], desired_first)
        second_state = _state_for_pose_matrix(
            capability.fk_controls[1],
            desired_second,
            parent_pose_matrix=desired_first,
        )
        terminal_state = _state_for_pose_matrix(
            capability.authored_terminal,
            solved_terminal,
            parent_pose_matrix=desired_second,
        )
        _apply_control_state(capability.fk_controls[0], first_state, location=False, rotation=True)
        _apply_control_state(capability.fk_controls[1], second_state, location=False, rotation=True)
        _apply_control_state(capability.authored_terminal, terminal_state, location=False, rotation=True)
        context.view_layer.update()
    except Exception as exc:
        _restore_direct_rotate_sliding_sync_session(context, session)
        if isinstance(exc, RigpedSemanticMoveError):
            raise
        raise RigpedSemanticMoveError(str(exc)) from exc


def _apply_direct_rotate_sliding_syncs(
    context,
    sessions: tuple[SlidingRotateSyncSession, ...],
    *,
    pin_terminal: bool = True,
) -> None:
    """Convert all active Sliding Rotate previews to IK as one dependency batch.

    Per-limb conversion is unsafe for multi-selection because each depsgraph
    update can re-evaluate still-unsynced Sliding limbs and overwrite their
    temporary public FK preview. Expose every active limb first, capture every
    target/pole intent from the same evaluated preview, then restore IK authority
    for the whole operation domain together.
    """

    if not sessions:
        return

    snapshots = []
    prepared = []
    try:
        # Phase 1: expose every selected public FK preview to its hidden result
        # chain before *any* limb is converted back to IK authority.
        for session in sessions:
            capability = session.capability
            pole_target = capability.native_ik.pole_target
            if pole_target is None:
                raise RigpedSemanticMoveError(
                    "Sliding Rotate lost its generated IK pole target."
                )
            native_ik = capability.native_ik.constraint
            terminal_ik = capability.terminal_ik_constraint
            feedback_constraints = (
                *capability.fk_copy_constraints,
                capability.terminal_fk_constraint,
            )
            snapshots.append(
                (
                    session,
                    float(native_ik.influence),
                    float(terminal_ik.influence),
                    tuple(bool(constraint.mute) for constraint in feedback_constraints),
                )
            )
            native_ik.influence = 0.0
            terminal_ik.influence = 0.0
            for constraint in feedback_constraints:
                constraint.mute = False
        context.view_layer.update()

        # Phase 2: every limb now observes the same coherent multi-preview.
        # Capture target/pole intent without interleaving depsgraph updates.
        for session, previous_ik_influence, previous_terminal_influence, previous_mutes in snapshots:
            capability = session.capability
            pole_target = capability.native_ik.pole_target
            if pole_target is None:
                raise RigpedSemanticMoveError(
                    "Sliding Rotate lost its generated IK pole target."
                )
            desired_terminal = capability.result_terminal.target.matrix.copy()
            branch_sign = _generated_rigped_hinge_branch(capability)
            pole_solution = _initial_pole_solution(capability)
            if pole_solution is None or not pole_solution.ok or pole_solution.position is None:
                raise RigpedSemanticMoveError(
                    "Sliding Rotate could not derive a stable preview pole from the current limb pose."
                )
            pole_angle = _derived_pole_angle(capability, pole_solution.position)
            pole_position, pole_angle = _canonical_generated_rigped_pole(
                capability,
                pole_solution.position,
                float(pole_angle),
                branch_sign,
            )
            ik_state = (
                _state_for_pose_matrix(
                    capability.native_ik.ik_target,
                    desired_terminal,
                )
                if session.terminal_selected or not pin_terminal
                else session.start_ik_state
            )
            pole_matrix = pole_target.target.matrix.copy()
            pole_matrix.translation = (
                pole_target.owner_object.matrix_world.inverted_safe()
                @ Vector(pole_position)
            )
            pole_state = _state_for_pose_matrix(pole_target, pole_matrix)
            prepared.append(
                (
                    session,
                    ik_state,
                    pole_state,
                    float(pole_angle),
                    branch_sign,
                    previous_ik_influence,
                    previous_terminal_influence,
                    previous_mutes,
                )
            )

        # Phase 3: freeze public->MCH feedback for the whole batch before IK is
        # re-enabled, so no already-prepared limb can clobber a later one.
        for session, _previous_ik, _previous_terminal, previous_mutes in snapshots:
            capability = session.capability
            feedback_constraints = (
                *capability.fk_copy_constraints,
                capability.terminal_fk_constraint,
            )
            for constraint, previous_mute in zip(
                feedback_constraints,
                previous_mutes,
                strict=True,
            ):
                constraint.mute = bool(previous_mute)
        context.view_layer.update()

        # Phase 4: apply every hidden target/pole bundle, then restore native IK
        # authority for every selected limb in one evaluation boundary.
        for (
            session,
            ik_state,
            pole_state,
            pole_angle,
            branch_sign,
            previous_ik_influence,
            previous_terminal_influence,
            _previous_mutes,
        ) in prepared:
            capability = session.capability
            pole_target = capability.native_ik.pole_target
            if pole_target is None:
                raise RigpedSemanticMoveError(
                    "Sliding Rotate lost its generated IK pole target."
                )
            _apply_control_state(
                capability.native_ik.ik_target,
                ik_state,
                location=True,
                rotation=True,
            )
            _apply_control_state(
                pole_target,
                pole_state,
                location=True,
                rotation=False,
            )
            capability.native_ik.constraint.pole_angle = float(pole_angle)
            if branch_sign is not None:
                configure_generated_rigped_ik_hinge_branch(
                    capability.native_ik.solver_owner.target,
                    branch_sign,
                )
            capability.native_ik.constraint.influence = previous_ik_influence
            capability.terminal_ik_constraint.influence = previous_terminal_influence
        context.view_layer.update()

        _refresh_current_sliding_public_overlays(
            context,
            capabilities=tuple(session.capability for session in sessions),
            allow_seed=False,
        )
    except Exception as exc:
        for session in sessions:
            _restore_direct_rotate_sliding_sync_session(
                context,
                session,
                update=False,
            )
        context.view_layer.update()
        if isinstance(exc, RigpedSemanticMoveError):
            raise
        raise RigpedSemanticMoveError(str(exc)) from exc


def _apply_sliding_lower_long_roll(
    context,
    session: SlidingRotateSyncSession,
    angle: float,
) -> None:
    """Roll the native lower IK link while keeping the terminal world transform fixed."""

    solver_control = session.capability.native_ik.solver_owner
    if session.start_solver_state.rotation_property != "rotation_quaternion":
        raise RigpedSemanticMoveError("Sliding lower-link roll requires quaternion solver rotation.")
    start_rotation = Quaternion(session.start_solver_state.rotation).normalized()
    desired_rotation = (
        start_rotation @ Quaternion(_RIGPED_LOCAL_AXES["Y"], float(angle))
    ).normalized()
    _apply_control_state(
        solver_control,
        replace(
            session.start_solver_state,
            rotation=tuple(float(value) for value in desired_rotation),
        ),
        location=False,
        rotation=True,
    )
    context.view_layer.update()
    _refresh_current_sliding_public_overlays(
        context,
        capabilities=(session.capability,),
        allow_seed=False,
    )


def _restore_direct_rotate_sliding_syncs(
    context,
    sessions: tuple[SlidingRotateSyncSession, ...],
) -> None:
    for session in sessions:
        _restore_direct_rotate_sliding_sync_session(
            context,
            session,
            update=False,
        )
    context.view_layer.update()


def _lower_limb_special_z_session(context, active: ResolvedControl, axis_world: Vector) -> FkTwoBoneMoveSession | None:
    if str(context.scene.transform_orientation_slots[0].type) != "LOCAL":
        return None
    pose_bone = active.target
    if not isinstance(pose_bone, bpy.types.PoseBone):
        return None
    if str(pose_bone.name) not in {"ForeArm.L", "ForeArm.R", "Calf.L", "Calf.R"}:
        return None
    local_basis = (active.owner_object.matrix_world @ pose_bone.matrix).to_3x3().normalized()
    local_z = Vector(local_basis.col[2])
    if local_z.length <= 1e-9:
        return None
    local_z.normalize()
    requested = Vector(axis_world)
    if requested.length <= 1e-9:
        return None
    requested.normalize()
    if abs(float(requested.dot(local_z))) < 0.999:
        return None
    resolved = _selected_semantic_mapping(context)
    if resolved is None or resolved.contact_type not in {
        ContactKeyType.FREE,
        ContactKeyType.SLIDING,
    }:
        return None
    if runtime_control_key(resolved.active_control) != runtime_control_key(active):
        return None
    return _begin_fk_two_bone_move_for_resolution(
        resolved,
        allow_sliding_lower_swivel=(resolved.contact_type is ContactKeyType.SLIDING),
    )


def _lower_limb_special_z_axis_sign(
    session: FkTwoBoneMoveSession,
    input_axis_world: Vector,
) -> float:
    """Map LOCAL-Z input onto the frozen semantic swivel direction once per gesture."""

    root = Vector(session.root_world)
    joint_offset = Vector(session.joint_world) - root
    semantic_axis = Vector(session.end_world) - root
    input_axis = Vector(input_axis_world)
    if (
        joint_offset.length <= 1e-9
        or semantic_axis.length <= 1e-9
        or input_axis.length <= 1e-9
    ):
        return 1.0
    semantic_axis.normalize()
    input_axis.normalize()
    input_tangent = input_axis.cross(joint_offset)
    semantic_tangent = semantic_axis.cross(joint_offset)
    if input_tangent.length <= 1e-7 or semantic_tangent.length <= 1e-7:
        return 1.0
    input_tangent.normalize()
    semantic_tangent.normalize()
    alignment = float(input_tangent.dot(semantic_tangent))
    if abs(alignment) <= 1e-4:
        return 1.0
    return 1.0 if alignment > 0.0 else -1.0


def _apply_lower_limb_special_z_rotation(
    context,
    session: FkTwoBoneMoveSession,
    angle: float,
    *,
    terminal_follows_second: bool,
    update: bool = True,
) -> bool:
    root = Vector(session.root_world)
    end = Vector(session.end_world)
    axis = end - root
    if axis.length <= 1e-9:
        return False
    axis.normalize()
    limited = max(-pi + radians(0.25), min(pi - radians(0.25), float(angle)))
    joint_offset = Vector(session.joint_world) - root
    desired_joint = root + Quaternion(axis, limited) @ joint_offset
    applied = _apply_solved_two_bone_fk_pose(
        session,
        desired_joint,
        end,
        terminal_follows_second=terminal_follows_second,
    )
    if applied and update:
        context.view_layer.update()
    return applied


def _quaternion_rotation_vector_components(
    rotation: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Return a branch-preserving rotation vector for an (w, x, y, z) quaternion."""

    w, x, y, z = (float(component) for component in rotation)
    magnitude = sqrt(w * w + x * x + y * y + z * z)
    if magnitude <= 1e-12:
        return (0.0, 0.0, 0.0)
    w, x, y, z = (component / magnitude for component in (w, x, y, z))
    vector_length = sqrt(x * x + y * y + z * z)
    if vector_length <= 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * atan2(vector_length, w)
    scale = angle / vector_length
    return (x * scale, y * scale, z * scale)


def _damped_two_axis_rotation_step(
    swivel_error: float,
    roll_error: float,
    generator_alignment: float,
    damping: float,
) -> tuple[float, float]:
    """Solve one bounded-size DLS update from two unit world generators."""

    alignment = max(-1.0, min(1.0, float(generator_alignment)))
    damping_sq = max(1e-8, float(damping) * float(damping))
    diagonal = 1.0 + damping_sq
    determinant = diagonal * diagonal - alignment * alignment
    if determinant <= 1e-12:
        return (0.0, 0.0)
    first = (
        diagonal * float(swivel_error) - alignment * float(roll_error)
    ) / determinant
    second = (
        diagonal * float(roll_error) - alignment * float(swivel_error)
    ) / determinant
    return (float(first), float(second))


@dataclass(frozen=True, slots=True)
class SlidingGlobalRotateProjection:
    swivel_angle: float
    roll_angle: float
    residual_radians: float
    conditioning: float


def _project_global_rotation_to_sliding_lower_dofs(
    session: FkTwoBoneMoveSession,
    state: DirectRotateControlState,
    axis_world: Vector,
    angle: float,
) -> SlidingGlobalRotateProjection:
    """Project one GLOBAL Rotate gesture onto Sliding swivel + axial-roll DOF.

    Refine the two actual world rotation generators from the zero-DOF branch.
    The swivel is around root-to-terminal; axial roll is around the lower-link
    local Y after that swivel. Damping leaves unreachable orientation as a
    measured residual instead of amplifying a near-singular inverse.
    """

    requested_axis = Vector(axis_world)
    if requested_axis.length <= 1e-9:
        return SlidingGlobalRotateProjection(0.0, 0.0, 0.0, 0.0)
    requested_axis.normalize()
    requested_rotation = Quaternion(requested_axis, float(angle))

    root = Vector(session.root_world)
    end = Vector(session.end_world)
    swivel_axis = end - root
    if swivel_axis.length <= 1e-9:
        return SlidingGlobalRotateProjection(
            0.0,
            0.0,
            min(2.0 * pi, abs(float(angle))),
            0.0,
        )
    swivel_axis.normalize()

    owner_world3 = session.second_control.owner_object.matrix_world.to_3x3()
    start_world_basis = (
        owner_world3 @ session.second_start_matrix.to_3x3()
    ).normalized()
    start_world_rotation = start_world_basis.to_quaternion().normalized()
    desired_world_rotation = (
        requested_rotation @ start_world_rotation
    ).normalized()

    pose_bone = state.control.target
    hinge_state = state.hinge_state
    limits = (
        _RIGPED_HINGE_JOINT_LIMITS.get(str(pose_bone.name))
        if isinstance(pose_bone, bpy.types.PoseBone)
        else None
    )
    if hinge_state is None or limits is None:
        return SlidingGlobalRotateProjection(
            0.0,
            0.0,
            min(2.0 * pi, abs(float(angle))),
            0.0,
        )

    swivel_bound = pi - radians(0.25)
    start_long = float(hinge_state[1])
    roll_limit = float(limits[3])
    roll_minimum = -roll_limit - start_long
    roll_maximum = roll_limit - start_long
    local_y = _RIGPED_LOCAL_AXES["Y"]
    swivel_angle = 0.0
    roll_angle = 0.0
    conditioning = 0.0

    for _iteration in range(24):
        swivel_world_rotation = Quaternion(swivel_axis, swivel_angle)
        post_swivel_rotation = (
            swivel_world_rotation @ start_world_rotation
        ).normalized()
        roll_generator = post_swivel_rotation @ local_y
        if roll_generator.length <= 1e-9:
            break
        roll_generator.normalize()
        alignment = max(
            -1.0,
            min(1.0, float(swivel_axis.dot(roll_generator))),
        )
        alignment_magnitude = abs(alignment)
        conditioning = sqrt(
            max(0.0, 1.0 - alignment_magnitude)
            / max(1e-12, 1.0 + alignment_magnitude)
        )

        predicted_world_rotation = (
            post_swivel_rotation
            @ Quaternion(local_y, roll_angle)
        ).normalized()
        error_rotation = (
            desired_world_rotation @ predicted_world_rotation.conjugated()
        ).normalized()
        error = Vector(
            _quaternion_rotation_vector_components(
                tuple(float(component) for component in error_rotation)
            )
        )
        if error.length <= 1e-6:
            break

        damping = 0.02 + (1.0 - conditioning) * 0.28
        swivel_step, roll_step = _damped_two_axis_rotation_step(
            float(swivel_axis.dot(error)),
            float(roll_generator.dot(error)),
            alignment,
            damping,
        )
        largest_step = max(abs(swivel_step), abs(roll_step))
        if largest_step > 0.35:
            scale = 0.35 / largest_step
            swivel_step *= scale
            roll_step *= scale

        gesture_bound = max(1e-6, abs(float(angle)) * 2.0)
        next_swivel = _clamp_scalar(
            swivel_angle + swivel_step,
            max(-swivel_bound, -gesture_bound),
            min(swivel_bound, gesture_bound),
        )
        next_roll = _clamp_scalar(
            roll_angle + roll_step,
            max(roll_minimum, -gesture_bound),
            min(roll_maximum, gesture_bound),
        )
        if (
            abs(next_swivel - swivel_angle) <= 1e-8
            and abs(next_roll - roll_angle) <= 1e-8
        ):
            break
        swivel_angle = next_swivel
        roll_angle = next_roll

    final_swivel_rotation = Quaternion(swivel_axis, swivel_angle)
    final_post_swivel_rotation = (
        final_swivel_rotation @ start_world_rotation
    ).normalized()
    final_rotation = (
        final_post_swivel_rotation @ Quaternion(local_y, roll_angle)
    ).normalized()
    final_error = (
        desired_world_rotation @ final_rotation.conjugated()
    ).normalized()
    residual = Vector(
        _quaternion_rotation_vector_components(
            tuple(float(component) for component in final_error)
        )
    ).length

    return SlidingGlobalRotateProjection(
        float(swivel_angle),
        float(roll_angle),
        min(2.0 * pi, max(0.0, float(residual))),
        max(0.0, min(1.0, float(conditioning))),
    )


class BAW_OT_rigped_direct_rotate_axis(bpy.types.Operator):
    """Fit-style visual Rotate modal with simultaneous Rigped multi-preview."""

    bl_idname = "baw.rigped_direct_rotate_axis"
    bl_label = "Rotate Rigped Control"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}

    axis: EnumProperty(
        items=(
            ("X", "X", "Rotate around X"),
            ("Y", "Y", "Rotate around Y"),
            ("Z", "Z", "Rotate around Z"),
            ("VIEW", "View", "Rotate around the current view axis"),
            ("FREE", "Free", "Free virtual-trackball rotation"),
        ),
        default="X",
    )

    _active: ResolvedControl | None = None
    _states: tuple[DirectRotateControlState, ...] = ()
    _sliding_syncs: tuple[SlidingRotateSyncSession, ...] = ()
    _sliding_guard_capabilities: tuple[LimbRepresentationCapability, ...] = ()
    _sliding_affected_capabilities: tuple[LimbRepresentationCapability, ...] = ()
    _sliding_seed_snapshots: tuple[SlidingHiddenSeedSnapshot, ...] = ()
    _sliding_dependency_guards: tuple[SlidingDependencyGuard, ...] = ()
    _axis_world = Vector((0.0, 0.0, 1.0))
    _pivot = Vector((0.0, 0.0, 0.0))
    _start_rotation_vector = Vector((1.0, 0.0, 0.0))
    _previous_rotation_vector = Vector((1.0, 0.0, 0.0))
    _previous_rotation_mouse = Vector((0.0, 0.0))
    _rotate_use_screen_tangent = False
    _rotate_screen_tangent = Vector((1.0, 0.0))
    _rotate_reference_radius = 0.0
    _previous_free_mouse = Vector((0.0, 0.0))
    _free_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    _raw_angle = 0.0
    _current_angle = 0.0
    _angle_sign = 1.0
    _orientation = "GLOBAL"
    _auto_plan: RigpedAutoAnchorPlan | None = None
    _auto_direct_plan: RigpedAutoDirectRotatePlan | None = None
    _auto_contact_batch_plan: RigpedAutoContactBatchPlan | None = None
    _deferred_auto_contact_mapping_ids: tuple[str, ...] = ()
    _forearm_special_session: FkTwoBoneMoveSession | None = None
    _single_lower_link_z_axis_sign: float = 1.0
    _lower_link_z_sessions: tuple[tuple[FkTwoBoneMoveSession, float, bool], ...] = ()
    _sliding_global_lower_sessions: tuple[
        tuple[FkTwoBoneMoveSession, SlidingRotateSyncSession], ...
    ] = ()
    _sliding_global_projected_dofs: tuple[tuple[str, float, float], ...] = ()
    _sliding_global_projection_diagnostics: tuple[tuple[str, float, float], ...] = ()
    _trace_operation_id: str | None = None

    trackball_radius_px: FloatProperty(
        name="",
        description="",
        default=64.0,
        min=8.0,
    )

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "mode", "") == "POSE"
            and getattr(context.scene, "baw_rigped_semantic_transform_mode", "NONE")
            == "DIRECT_ROTATE"
            and direct_rotate_available(context)
        )

    def invoke(self, context, event):
        if self.axis == "FREE":
            from .viewport_keymap import prioritize_awb_selection_over_free_rotate

            if prioritize_awb_selection_over_free_rotate(context, event):
                if context.area is not None:
                    context.area.tag_redraw()
                return {"FINISHED"}

        controls = _selected_direct_rotate_controls(context)
        axes = direct_transform_axes(context)
        if not controls or axes is None:
            return {"CANCELLED"}

        domain_resolution = resolve_operation_domain(
            context.scene,
            control_context_for_context(context),
        )
        if not domain_resolution.ok or domain_resolution.snapshot is None:
            detail = (
                domain_resolution.issues[0].detail
                if domain_resolution.issues
                else "Rigped Rotate could not freeze the operation domain."
            )
            self.report({"WARNING"}, detail)
            return {"CANCELLED"}
        operation_domain = domain_resolution.snapshot
        frozen_current_sliding = _sliding_capabilities_for_character(
            context.scene,
            operation_domain.character_id,
        )

        # Historical/baked keys may predate the current joint-limit policy.
        # Repair them before capturing the modal start state so Rotate can move
        # back inward instead of freezing because angle=0 is already invalid.
        repaired = False
        for control in controls:
            pose_bone = control.target
            if isinstance(pose_bone, bpy.types.PoseBone):
                repaired = apply_rigped_joint_limits_to_pose_bone(pose_bone) or repaired
        if repaired:
            context.view_layer.update()

        active = controls[0]
        if not isinstance(active.target, bpy.types.PoseBone):
            return {"CANCELLED"}
        axis = (
            _view_axis(context)
            if self.axis in {"VIEW", "FREE"}
            else axes.get(self.axis)
        )
        if axis is None or axis.length <= 1e-9:
            return {"CANCELLED"}

        selected_lower_names = tuple(
            sorted(
                str(control.target.name)
                for control in controls
                if isinstance(control.target, bpy.types.PoseBone)
            )
        )
        full_four_local_x_gesture = (
            str(context.scene.transform_orientation_slots[0].type) == "LOCAL"
            and self.axis == "X"
            and selected_lower_names
            == ("Calf.L", "Calf.R", "ForeArm.L", "ForeArm.R")
        )
        if full_four_local_x_gesture:
            active_hinge_state = _hinge_state_from_basis(
                active.target,
                active.target.matrix_basis,
            )
            if active_hinge_state is not None:
                axis = Vector(axis) * _lower_link_local_x_branch_sign(
                    str(active.target.name),
                    float(active_hinge_state[0]),
                )
        pivot = _control_pivot_world(active)
        if self.axis == "FREE":
            region_data = getattr(context, "region_data", None)
            if region_data is None:
                return {"CANCELLED"}
            view_basis = region_data.view_matrix.inverted().to_3x3()
            start = Vector(view_basis.col[0]).normalized()
        else:
            start = _rotation_vector(
                context,
                pivot,
                Vector(axis),
                event.mouse_region_x,
                event.mouse_region_y,
            )
            if start is None:
                return {"CANCELLED"}
        self._active = active
        states: list[DirectRotateControlState] = []
        for control in controls:
            pose_bone = control.target
            if not isinstance(pose_bone, bpy.types.PoseBone):
                continue
            center_pivot = _uses_center_pivot(control)
            pivot_local = (
                (Vector(pose_bone.head) + Vector(pose_bone.tail)) * 0.5
                if center_pivot
                else pose_bone.matrix.to_translation().copy()
            )
            start_basis = pose_bone.matrix_basis.copy()
            states.append(
                DirectRotateControlState(
                    control=control,
                    start_basis=start_basis,
                    start_matrix=pose_bone.matrix.copy(),
                    pivot_local=pivot_local,
                    center_pivot=center_pivot,
                    hinge_state=_hinge_state_from_basis(pose_bone, start_basis),
                )
            )
        self._states = tuple(states)
        self._forearm_special_session = None
        self._single_lower_link_z_axis_sign = 1.0
        self._lower_link_z_sessions = ()
        self._sliding_global_lower_sessions = ()
        self._sliding_global_projected_dofs = ()
        self._sliding_global_projection_diagnostics = ()
        try:
            self._forearm_special_session = _lower_limb_special_z_session(
                context,
                active,
                Vector(axis),
            )
        except RigpedSemanticMoveError as exc:
            _report_operator_error(self, context, exc)
            self._states = ()
            self._active = None
            return {"CANCELLED"}
        if self._forearm_special_session is not None:
            self._single_lower_link_z_axis_sign = (
                _lower_limb_special_z_axis_sign(
                    self._forearm_special_session,
                    Vector(axis),
                )
            )
        if self._forearm_special_session is not None and len(self._states) > 1:
            # The dedicated single-ForeArm swivel preview cannot own a multi
            # selection. Fall back to the shared multi-rotate path; Sliding sync
            # will keep each terminal pinned and convert the resulting bend-plane
            # change into its own IK pole/swivel.
            self._forearm_special_session = None
        try:
            self._sliding_syncs = _direct_rotate_sliding_sync_sessions(
                context,
                operation_domain,
            )
        except RigpedSemanticMoveError as exc:
            _report_operator_error(self, context, exc)
            self._states = ()
            self._sliding_syncs = ()
            self._sliding_guard_capabilities = ()
            self._sliding_affected_capabilities = ()
            self._active = None
            self._auto_plan = None
            self._forearm_special_session = None
            return {"CANCELLED"}

        lower_link_resolutions = _selected_semantic_move_resolutions(context)
        lower_link_names = tuple(
            sorted(
                str(state.control.target.name)
                for state in self._states
                if isinstance(state.control.target, bpy.types.PoseBone)
            )
        )
        lower_link_set = {"ForeArm.L", "ForeArm.R", "Calf.L", "Calf.R"}
        sliding_global_supported_selection = (
            (
                len(lower_link_names) == 1
                and lower_link_names[0] in lower_link_set
            )
            or lower_link_names
            in {
                ("Calf.L", "Calf.R"),
                ("ForeArm.L", "ForeArm.R"),
            }
        )
        if (
            str(context.scene.transform_orientation_slots[0].type) == "GLOBAL"
            and self.axis in {"X", "Y", "Z"}
            and sliding_global_supported_selection
            and len(self._states) == len(self._sliding_syncs)
            and len(self._states) == len(lower_link_resolutions)
            and len(lower_link_names) == len(self._states)
        ):
            sync_by_mapping = {
                str(session.capability.native_ik.mapping_id): session
                for session in self._sliding_syncs
            }
            global_rows: list[
                tuple[FkTwoBoneMoveSession, SlidingRotateSyncSession]
            ] = []
            try:
                for resolved in lower_link_resolutions:
                    if resolved.contact_type is not ContactKeyType.SLIDING:
                        global_rows = []
                        break
                    sync_session = sync_by_mapping.get(
                        str(resolved.capability.native_ik.mapping_id)
                    )
                    if sync_session is None:
                        global_rows = []
                        break
                    global_rows.append(
                        (
                            _begin_fk_two_bone_move_for_resolution(
                                resolved,
                                allow_sliding_lower_swivel=True,
                            ),
                            sync_session,
                        )
                    )
            except RigpedSemanticMoveError as exc:
                _report_operator_error(self, context, exc)
                global_rows = []
            if len(global_rows) == len(self._states):
                self._sliding_global_lower_sessions = tuple(global_rows)

        if (
            str(context.scene.transform_orientation_slots[0].type) == "LOCAL"
            and self.axis == "Z"
            and len(self._states) > 1
            and len(self._states) == len(lower_link_resolutions)
            and len(lower_link_names) == len(self._states)
            and all(name in lower_link_set for name in lower_link_names)
        ):
            active_basis_world = (
                active.owner_object.matrix_world @ active.target.matrix
            ).to_3x3().normalized()
            active_local_z = Vector(active_basis_world.col[2])
            requested_axis = Vector(axis)
            if active_local_z.length > 1e-9 and requested_axis.length > 1e-9:
                active_local_z.normalize()
                requested_axis.normalize()
                if abs(float(requested_axis.dot(active_local_z))) >= 0.999:
                    state_by_key = {
                        runtime_control_key(state.control): state
                        for state in self._states
                    }
                    multi_rows: list[tuple[FkTwoBoneMoveSession, float, bool]] = []
                    try:
                        for resolved in lower_link_resolutions:
                            if resolved.contact_type not in {
                                ContactKeyType.FREE,
                                ContactKeyType.SLIDING,
                            }:
                                multi_rows = []
                                break
                            lower_control = resolved.active_control
                            lower_name = str(lower_control.target.name)
                            state = state_by_key.get(runtime_control_key(lower_control))
                            if state is None or lower_name not in lower_link_set:
                                multi_rows = []
                                break
                            special_session = _begin_fk_two_bone_move_for_resolution(
                                resolved,
                                allow_sliding_lower_swivel=(
                                    resolved.contact_type is ContactKeyType.SLIDING
                                ),
                            )
                            mapped_axis = _mapped_direct_rotate_axis(
                                active,
                                lower_control,
                                Vector(axis),
                                mirror_opposites=True,
                                axis_name="Z",
                            )
                            axis_sign = _lower_limb_special_z_axis_sign(
                                special_session,
                                mapped_axis,
                            )
                            multi_rows.append(
                                (
                                    special_session,
                                    axis_sign,
                                    resolved.contact_type is ContactKeyType.FREE,
                                )
                            )
                    except RigpedSemanticMoveError as exc:
                        _report_operator_error(self, context, exc)
                        multi_rows = []
                    if len(multi_rows) == len(self._states):
                        self._lower_link_z_sessions = tuple(multi_rows)

        active_capabilities = tuple(
            session.capability
            for session in self._sliding_syncs
        )
        self._sliding_guard_capabilities = _passive_sliding_capabilities(
            frozen_current_sliding,
            self._sliding_syncs,
        )
        self._sliding_affected_capabilities = frozen_current_sliding
        self._sliding_seed_snapshots = _capture_sliding_hidden_seed_snapshots(
            self._sliding_guard_capabilities
        )
        e7_body_rotate_roles = {
            "COM",
            "Pelvis",
            "Spine",
            "Head",
            "Clavicle.L",
            "Clavicle.R",
        }
        e7_body_rotate = (
            not self._sliding_syncs
            and bool(controls)
            and all(
                _resolved_control_role_name(control) in e7_body_rotate_roles
                for control in controls
            )
        )
        self._sliding_dependency_guards = (
            _capture_sliding_dependency_guards(self._sliding_guard_capabilities)
            if e7_body_rotate
            else ()
        )
        active_ids = set(_sliding_capability_mapping_ids(active_capabilities))
        passive_ids = set(
            _sliding_capability_mapping_ids(self._sliding_guard_capabilities)
        )
        affected_ids = set(
            _sliding_capability_mapping_ids(self._sliding_affected_capabilities)
        )
        if active_ids & passive_ids or active_ids | passive_ids != affected_ids:
            self.report(
                {"WARNING"},
                "Rigped Rotate Sliding ownership coverage changed at gesture start.",
            )
            self._states = ()
            self._sliding_syncs = ()
            self._sliding_guard_capabilities = ()
            self._sliding_affected_capabilities = ()
            self._active = None
            self._forearm_special_session = None
            return {"CANCELLED"}

        self._auto_plan = None
        self._auto_direct_plan = None
        self._auto_contact_batch_plan = None
        self._deferred_auto_contact_mapping_ids = ()
        if bool(getattr(context.scene, "baw_auto_key_enabled", False)):
            auto_context = control_context_for_context(context)
            auto_limb_context, auto_direct_context = _direct_rotate_auto_contexts(
                context.scene,
                auto_context,
            )
            operation_id = f"rigped-auto-rotate:{uuid4().hex}"
            mapping_ids = (
                selected_contact_mapping_ids(context.scene, auto_limb_context)
                if auto_limb_context is not None
                else ()
            )

            if len(mapping_ids) >= 2:
                # Keep Rotate preview independent from AUTO preflight. The
                # semantic batch is planned only after the gesture produced its
                # final solved pose, so enabling AUTO cannot disable a valid
                # multi-Foot/Hand Rotate at frame 0.
                self._deferred_auto_contact_mapping_ids = tuple(mapping_ids)
            elif mapping_ids:
                auto = plan_rigped_auto_anchor(
                    context.scene,
                    auto_limb_context,
                    operation_id=f"{operation_id}:contact",
                    mapping_id=mapping_ids[0],
                )
                if not auto.ok or auto.plan is None:
                    detail = (
                        auto.diagnostics[0].detail
                        if auto.diagnostics
                        else "Rigped semantic Rotate Auto planning failed."
                    )
                    self.report({"WARNING"}, detail)
                    self._states = ()
                    self._active = None
                    self._forearm_special_session = None
                    return {"CANCELLED"}
                if auto.plan.intent.target_type not in {
                    ContactKeyType.FREE,
                    ContactKeyType.SLIDING,
                }:
                    self.report(
                        {"WARNING"},
                        "Rigped Rotate Auto currently supports Free or Sliding authority.",
                    )
                    self._states = ()
                    self._active = None
                    self._forearm_special_session = None
                    return {"CANCELLED"}
                self._auto_plan = auto.plan
            if auto_direct_context is not None:
                direct = plan_rigped_auto_direct_rotate(
                    context.scene,
                    auto_direct_context,
                    operation_id=f"{operation_id}:direct",
                    active_only=len(auto_direct_context.controls) == 1,
                )
                if not direct.ok or direct.plan is None:
                    detail = (
                        direct.diagnostics[0].detail
                        if direct.diagnostics
                        else "Rigped direct Rotate Auto planning failed."
                    )
                    self.report({"WARNING"}, detail)
                    self._states = ()
                    self._active = None
                    self._forearm_special_session = None
                    return {"CANCELLED"}
                self._auto_direct_plan = direct.plan
                if self._auto_plan is not None or self._deferred_auto_contact_mapping_ids:
                    self._auto_direct_plan = replace(
                        self._auto_direct_plan,
                        allow_storage_rebind=True,
                    )
        self._axis_world = Vector(axis).normalized()
        self._pivot = Vector(pivot)
        self._start_rotation_vector = Vector(start)
        self._previous_rotation_vector = Vector(start)
        self._previous_rotation_mouse = Vector(
            (float(event.mouse_region_x), float(event.mouse_region_y))
        )
        self._previous_free_mouse = Vector(self._previous_rotation_mouse)
        self._free_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        if self.axis != "FREE":
            tangent = linear_roll_screen_tangent(
                context,
                self._pivot,
                self._axis_world,
                self._previous_rotation_mouse,
                self._start_rotation_vector,
            )
            if tangent is None:
                self._states = ()
                self._sliding_syncs = ()
                self._active = None
                return {"CANCELLED"}
            self._rotate_screen_tangent = Vector(tangent)
            self._rotate_use_screen_tangent = True
        self._rotate_reference_radius = 0.0
        self._raw_angle = 0.0
        self._current_angle = 0.0
        self._orientation = str(context.scene.transform_orientation_slots[0].type)
        selected_pose_names = tuple(
            sorted(
                str(state.control.target.name)
                for state in self._states
                if isinstance(state.control.target, bpy.types.PoseBone)
            )
        )
        self._angle_sign = (
            -1.0
            if self._orientation == "LOCAL"
            and self.axis == "Z"
            and selected_pose_names == ("Clavicle.L", "Clavicle.R")
            else 1.0
        )
        if self.axis != "FREE":
            show_rotation_angle(context, 0.0, self._pivot)

        # Match the accepted Fit Rotate feel for ordinary FK controls. Near an
        # edge-on ring, repeated mouse-ray/rotation-plane intersections become
        # numerically unstable and can turn 1-2 px input into a visible jump.
        # Calibrate one screen-space tangent at mouse-down and accumulate the
        # angular delta from actual pixel motion instead.
        if (
            self.axis != "FREE"
            and context.region is not None
            and context.region_data is not None
        ):
            pivot_2d = location_3d_to_region_2d(
                context.region,
                context.region_data,
                self._pivot,
            )
            world_length = float(
                Vector(
                    active.owner_object.matrix_world.to_3x3()
                    @ (active.target.tail - active.target.head)
                ).length
            )
            reference_radius = max(0.05, world_length)
            reference_2d = location_3d_to_region_2d(
                context.region,
                context.region_data,
                self._pivot + Vector(start) * reference_radius,
            )
            mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
            if pivot_2d is not None and reference_2d is not None:
                projected_radius = float((Vector(reference_2d) - Vector(pivot_2d)).length)
                mouse_radius = float((mouse - Vector(pivot_2d)).length)
                if projected_radius >= 1.0 and mouse_radius >= 6.0:
                    reference_radius *= mouse_radius / projected_radius
                    probe_angle = 0.05
                    probe_start = location_3d_to_region_2d(
                        context.region,
                        context.region_data,
                        self._pivot + Vector(start) * reference_radius,
                    )
                    probe_vector = Matrix.Rotation(
                        probe_angle,
                        3,
                        self._axis_world,
                    ) @ Vector(start)
                    probe_end = location_3d_to_region_2d(
                        context.region,
                        context.region_data,
                        self._pivot + probe_vector * reference_radius,
                    )
                    if probe_start is not None and probe_end is not None:
                        tangent_pixels = Vector(probe_end) - Vector(probe_start)
                        if tangent_pixels.length >= 0.5:
                            self._rotate_reference_radius = reference_radius
                            self._rotate_use_screen_tangent = True

        self._trace_operation_id = new_trace_operation_id("direct-rotate")
        bind_operation_causal_context(
            self._trace_operation_id,
            new_trace_causal_root("direct-rotate"),
        )
        authority = (
            self._auto_plan.intent.target_type.value
            if self._auto_plan is not None
            else ("SLIDING" if self._sliding_syncs else "DIRECT")
        )
        trace_event(
            "OPERATION",
            "TRANSFORM_BEGIN",
            operation_id=self._trace_operation_id,
            subsystem="modal",
            lifecycle_phase="invoke",
            route_outcome=TraceRouteOutcome.CLAIMED,
            context=context,
            tool="ROTATE",
            route="DIRECT_ROTATE",
            axis=self.axis,
            authority=authority,
            auto_key=bool(getattr(context.scene, "baw_auto_key_enabled", False)),
            sliding_sync_count=len(self._sliding_syncs),
            sliding_guard_count=len(self._sliding_guard_capabilities),
            sliding_sync_ids=tuple(
                str(session.capability.native_ik.mapping_id)
                for session in self._sliding_syncs
            ),
            sliding_guard_ids=_sliding_capability_mapping_ids(
                self._sliding_guard_capabilities
            ),
            sliding_affected_ids=_sliding_capability_mapping_ids(
                self._sliding_affected_capabilities
            ),
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _restore_preview(self, context) -> None:
        if self._lower_link_z_sessions:
            for (
                special_session,
                _axis_sign,
                _terminal_follows_second,
            ) in self._lower_link_z_sessions:
                special_session.first_control.target.matrix_basis = (
                    special_session.first_start_basis.copy()
                )
                special_session.second_control.target.matrix_basis = (
                    special_session.second_start_basis.copy()
                )
                special_session.terminal_control.target.matrix_basis = (
                    special_session.terminal_start_basis.copy()
                )
            context.view_layer.update()
        elif self._forearm_special_session is not None:
            cancel_fk_two_bone_move(context, self._forearm_special_session)
        else:
            for state in self._states:
                if isinstance(state.control.target, bpy.types.PoseBone):
                    state.control.target.matrix_basis = state.start_basis.copy()
        if self._sliding_syncs:
            _restore_direct_rotate_sliding_syncs(context, self._sliding_syncs)
        else:
            context.view_layer.update()
        _restore_sliding_dependency_guards(
            context,
            self._sliding_dependency_guards,
            update=False,
        )
        _restore_sliding_hidden_seed_snapshots(
            context,
            self._sliding_seed_snapshots,
            update=False,
        )
        context.view_layer.update()
        _refresh_current_sliding_public_overlays(
            context,
            capabilities=self._sliding_guard_capabilities,
            allow_seed=False,
        )
        _assert_sliding_dependency_authority(
            context,
            self._sliding_dependency_guards,
            operation_id=self._trace_operation_id,
            phase="ROTATE_RESTORE",
        )

    def _apply_preview(self, context) -> None:
        active = self._active
        if active is None or not self._states:
            return

        if self._sliding_syncs:
            # The angle is absolute from mouse-down. Rebuild its FK/Hand input
            # from that same snapshot before each Sliding FK-to-IK conversion;
            # the previous tick's solved public pose is display only.
            self._restore_preview(context)
            if abs(self._current_angle) <= 1e-9:
                return

        if self._lower_link_z_sessions:
            for (
                special_session,
                axis_sign,
                terminal_follows_second,
            ) in self._lower_link_z_sessions:
                if not _apply_lower_limb_special_z_rotation(
                    context,
                    special_session,
                    self._current_angle * axis_sign,
                    terminal_follows_second=terminal_follows_second,
                    update=False,
                ):
                    raise RigpedSemanticMoveError(
                        "Multi-limb lower-link Z swivel could not preserve the terminal chain."
                    )
            context.view_layer.update()
            _apply_direct_rotate_sliding_syncs(context, self._sliding_syncs)
            if self._sliding_dependency_guards:
                _refresh_frozen_sliding_dependency_overlays(
                    context,
                    self._sliding_dependency_guards,
                    operation_id=self._trace_operation_id,
                    phase="ROTATE_PREVIEW",
                )
            else:
                _refresh_current_sliding_public_overlays(
                    context,
                    capabilities=self._sliding_guard_capabilities,
                )
            return

        if self._forearm_special_session is not None:
            if not _apply_lower_limb_special_z_rotation(
                context,
                self._forearm_special_session,
                self._current_angle * self._single_lower_link_z_axis_sign,
                terminal_follows_second=not bool(self._sliding_syncs),
            ):
                raise RigpedSemanticMoveError(
                    "Lower-limb Z swivel could not preserve the terminal chain."
                )
            _apply_direct_rotate_sliding_syncs(context, self._sliding_syncs)
            if self._sliding_dependency_guards:
                _refresh_frozen_sliding_dependency_overlays(
                    context,
                    self._sliding_dependency_guards,
                    operation_id=self._trace_operation_id,
                    phase="ROTATE_PREVIEW",
                )
            else:
                _refresh_current_sliding_public_overlays(
                    context,
                    capabilities=self._sliding_guard_capabilities,
                )
            return

        sliding_lower_names = tuple(
            sorted(
                str(state.control.target.name)
                for state in self._states
                if isinstance(state.control.target, bpy.types.PoseBone)
            )
        )
        sliding_lower_pair = sliding_lower_names in {
            ("Calf.L", "Calf.R"),
            ("ForeArm.L", "ForeArm.R"),
        }
        sliding_lower_full_four = sliding_lower_names == (
            "Calf.L",
            "Calf.R",
            "ForeArm.L",
            "ForeArm.R",
        )
        full_four_local_x = (
            self._orientation == "LOCAL"
            and self.axis == "X"
            and sliding_lower_full_four
            and all(state.hinge_state is not None for state in self._states)
        )
        sliding_lower_single = (
            len(sliding_lower_names) == 1
            and sliding_lower_names[0] in {"ForeArm.L", "ForeArm.R", "Calf.L", "Calf.R"}
        )
        sliding_lower_local_x = (
            self._orientation == "LOCAL"
            and self.axis == "X"
            and len(self._states) == len(self._sliding_syncs)
            and bool(sliding_lower_names)
            and (
                sliding_lower_single
                or sliding_lower_pair
                or sliding_lower_full_four
            )
            and all(
                name in {"ForeArm.L", "ForeArm.R", "Calf.L", "Calf.R"}
                for name in sliding_lower_names
            )
        )
        sliding_lower_global = (
            self._orientation == "GLOBAL"
            and self.axis in {"X", "Y", "Z"}
            and len(self._states) == len(self._sliding_syncs)
            and bool(sliding_lower_names)
            and (sliding_lower_single or sliding_lower_pair)
            and all(
                name in {"ForeArm.L", "ForeArm.R", "Calf.L", "Calf.R"}
                for name in sliding_lower_names
            )
        )
        # Sliding GLOBAL lower-link Rotate must keep the authored terminal
        # Hand/Foot world transform pinned. LOCAL X is the one supported
        # lower-link exception that transports the terminal so bend can change
        # without inventing an impossible fixed-end two-bone solve.
        sliding_lower_pins_terminal = sliding_lower_global
        sliding_lower_transports_terminal = (
            sliding_lower_local_x and not sliding_lower_pins_terminal
        )
        full_four_local_x_axes: dict[int, Vector] = {}
        if full_four_local_x:
            for state in self._states:
                pose_bone = state.control.target
                hinge_state = state.hinge_state
                if not isinstance(pose_bone, bpy.types.PoseBone) or hinge_state is None:
                    continue
                lower_name = str(pose_bone.name)
                hinge_angle = float(hinge_state[0])
                branch_sign = _lower_link_local_x_branch_sign(
                    lower_name,
                    hinge_angle,
                )
                basis_world = (
                    state.control.owner_object.matrix_world.to_3x3()
                    @ state.start_matrix.to_3x3().normalized()
                )
                bend_axis = basis_world @ _RIGPED_LOCAL_AXES["X"]
                if bend_axis.length <= 1e-9:
                    raise RigpedSemanticMoveError(
                        "Four-limb LOCAL X found a collapsed bend axis."
                    )
                bend_axis.normalize()
                bend_axis *= float(branch_sign)
                full_four_local_x_axes[int(pose_bone.as_pointer())] = bend_axis
        if (
            self._orientation == "LOCAL"
            and self.axis == "Y"
            and len(self._states) == len(self._sliding_syncs)
            and len(self._states) in {1, 2}
            and all(state.hinge_state is not None for state in self._states)
            and (sliding_lower_single or sliding_lower_pair)
        ):
            states_by_key = {
                runtime_control_key(state.control): state
                for state in self._states
            }
            session_states = tuple(
                (
                    session,
                    states_by_key.get(
                        runtime_control_key(session.capability.fk_controls[1])
                    ),
                )
                for session in self._sliding_syncs
            )
            if all(state is not None for _session, state in session_states):
                self._current_angle = _bounded_hinge_direct_rotate_angle(
                    active,
                    self._states,
                    self._axis_world,
                    self._current_angle,
                    axis_name=self.axis,
                )
                if abs(self._current_angle) > 1e-9:
                    if sliding_lower_pair:
                        pair_capabilities = []
                        active_name = str(active.target.name)
                        active_opposite = _opposite_control_name(active_name)
                        for session, state in session_states:
                            assert state is not None
                            solver_control = session.capability.native_ik.solver_owner
                            if session.start_solver_state.rotation_property != "rotation_quaternion":
                                raise RigpedSemanticMoveError(
                                    "Sliding lower-link roll requires quaternion solver rotation."
                                )
                            start_rotation = Quaternion(
                                session.start_solver_state.rotation
                            ).normalized()
                            lower_name = str(
                                session.capability.fk_controls[1].target.name
                            )
                            roll_angle = (
                                -self._current_angle
                                if active_opposite is not None
                                and lower_name == active_opposite
                                else self._current_angle
                            )
                            desired_rotation = (
                                start_rotation
                                @ Quaternion(_RIGPED_LOCAL_AXES["Y"], roll_angle)
                            ).normalized()
                            _apply_control_state(
                                solver_control,
                                replace(
                                    session.start_solver_state,
                                    rotation=tuple(float(value) for value in desired_rotation),
                                ),
                                location=False,
                                rotation=True,
                            )
                            pair_capabilities.append(session.capability)
                        context.view_layer.update()
                        _refresh_current_sliding_public_overlays(
                            context,
                            capabilities=tuple(pair_capabilities),
                            allow_seed=False,
                        )
                    else:
                        session, state = session_states[0]
                        assert state is not None
                        _apply_sliding_lower_long_roll(
                            context,
                            session,
                            self._current_angle,
                        )
                _refresh_current_sliding_public_overlays(
                    context,
                    capabilities=self._sliding_guard_capabilities,
                )
                return

        hinge_states = tuple(state for state in self._states if state.hinge_state is not None)
        generic_states = tuple(state for state in self._states if state.hinge_state is None)
        # GLOBAL and VIEW are shared frame-space gestures: every selected control
        # receives the same visible/world rotation direction. LOCAL retains the
        # anatomical opposite-side mirror behavior for paired controls.
        mirror_opposites = self._orientation == "LOCAL" and self.axis != "FREE"
        sliding_global_projection = bool(self._sliding_global_lower_sessions)
        hinge_axis_effective = sliding_global_projection or sliding_lower_local_x or any(
            abs(weight) > 1e-7
            for state in hinge_states
            for weight in _hinge_axis_weights(
                active,
                state,
                self._axis_world,
                mirror_opposites=mirror_opposites,
                axis_name=self.axis,
            )
        )
        if hinge_states and not generic_states and not hinge_axis_effective:
            # A forbidden hinge axis is a true no-op. In particular, do not run
            # Sliding FK->IK synchronization for an axis that cannot change the
            # authored hinge/roll state; re-solving an unchanged pose can move
            # the pole/branch numerically and produce a visible direction jump.
            self._current_angle = 0.0
            return

        # ForeArm/Calf are true Biped-style hinge + axial-roll joints in every
        # gizmo orientation. Hand/Foot and broad ball joints follow the actual
        # rotate axis and stop at the first invalid swing/twist pose along that
        # path, avoiding both Euler coupling and quaternion re-entry flips.
        if hinge_states and not sliding_global_projection:
            if full_four_local_x_axes:
                bounded_angle = float(self._current_angle)
                for state in hinge_states:
                    pose_bone = state.control.target
                    assert isinstance(pose_bone, bpy.types.PoseBone)
                    bend_axis = full_four_local_x_axes.get(int(pose_bone.as_pointer()))
                    if bend_axis is None:
                        raise RigpedSemanticMoveError(
                            "Four-limb Sliding LOCAL X lost a semantic bend axis."
                        )
                    bounded_angle = _bounded_hinge_direct_rotate_angle(
                        state.control,
                        (state,),
                        bend_axis,
                        bounded_angle,
                        mirror_opposites=False,
                        axis_name="X",
                    )
                self._current_angle = bounded_angle
            else:
                self._current_angle = _bounded_hinge_direct_rotate_angle(
                    active,
                    hinge_states,
                    self._axis_world,
                    self._current_angle,
                    mirror_opposites=mirror_opposites,
                    axis_name=self.axis,
                )
        if generic_states:
            self._current_angle = _bounded_direct_rotate_angle(
                active,
                generic_states,
                self._axis_world,
                self._current_angle,
                mirror_opposites=mirror_opposites,
            )

        raw_desired_by_bone, state_by_bone = _raw_direct_rotate_desired(
            active,
            self._states,
            self._axis_world,
            self._current_angle,
            mirror_opposites=mirror_opposites,
        )
        resolved_desired_by_bone: dict[int, Matrix] = {}

        def resolve_desired(state: DirectRotateControlState) -> Matrix:
            pose_bone = state.control.target
            assert isinstance(pose_bone, bpy.types.PoseBone)
            pointer = int(pose_bone.as_pointer())
            cached = resolved_desired_by_bone.get(pointer)
            if cached is not None:
                return cached
            parent_desired = None
            if pose_bone.parent is not None:
                parent_pointer = int(pose_bone.parent.as_pointer())
                parent_state = state_by_bone.get(parent_pointer)
                parent_desired = (
                    resolve_desired(parent_state)
                    if parent_state is not None
                    else pose_bone.parent.matrix.copy()
                )
            desired = None
            if state.hinge_state is not None:
                semantic_bend_axis = full_four_local_x_axes.get(pointer)
                if semantic_bend_axis is not None:
                    desired = _hinge_direct_rotate_desired(
                        state.control,
                        state,
                        semantic_bend_axis,
                        self._current_angle,
                        mirror_opposites=False,
                        axis_name="X",
                        parent_pose_matrix=parent_desired,
                    )
                else:
                    desired = _hinge_direct_rotate_desired(
                        active,
                        state,
                        self._axis_world,
                        self._current_angle,
                        mirror_opposites=mirror_opposites,
                        axis_name=self.axis,
                        parent_pose_matrix=parent_desired,
                    )
            if desired is None:
                desired = raw_desired_by_bone[pointer]
            resolved_desired_by_bone[pointer] = desired
            return desired

        if self._sliding_global_lower_sessions:
            projected_rows = []
            any_swivel = False
            for special_session, sync_session in self._sliding_global_lower_sessions:
                second_target = special_session.second_control.target
                if not isinstance(second_target, bpy.types.PoseBone):
                    raise RigpedSemanticMoveError(
                        "Sliding GLOBAL lower-link Rotate lost its PoseBone target."
                    )
                state = state_by_bone.get(int(second_target.as_pointer()))
                if state is None:
                    raise RigpedSemanticMoveError(
                        "Sliding GLOBAL lower-link Rotate lost its selected lower-link state."
                    )
                projection = _project_global_rotation_to_sliding_lower_dofs(
                    special_session,
                    state,
                    self._axis_world,
                    self._current_angle,
                )
                projected_rows.append(
                    (special_session, sync_session, projection)
                )
                if abs(projection.swivel_angle) <= 1e-9:
                    continue
                root = Vector(special_session.root_world)
                end = Vector(special_session.end_world)
                swivel_axis = end - root
                if swivel_axis.length <= 1e-9:
                    raise RigpedSemanticMoveError(
                        "Sliding GLOBAL lower-link Rotate found a collapsed swivel axis."
                    )
                swivel_axis.normalize()
                desired_joint = root + Quaternion(
                    swivel_axis,
                    projection.swivel_angle,
                ) @ (
                    Vector(special_session.joint_world) - root
                )
                if not _apply_solved_two_bone_fk_pose(
                    special_session,
                    desired_joint,
                    end,
                    terminal_follows_second=False,
                ):
                    raise RigpedSemanticMoveError(
                        "Sliding GLOBAL lower-link Rotate could not apply its projected swivel."
                    )
                any_swivel = True

            self._sliding_global_projected_dofs = tuple(
                (
                    str(row[0].second_control.target.name),
                    float(row[2].swivel_angle),
                    float(row[2].roll_angle),
                )
                for row in projected_rows
            )
            self._sliding_global_projection_diagnostics = tuple(
                (
                    str(row[0].second_control.target.name),
                    float(row[2].residual_radians),
                    float(row[2].conditioning),
                )
                for row in projected_rows[:4]
            )

            if any_swivel:
                context.view_layer.update()
                _apply_direct_rotate_sliding_syncs(
                    context,
                    tuple(row[1] for row in projected_rows),
                )

            roll_capabilities = []
            for _special_session, sync_session, projection in projected_rows:
                roll_angle = projection.roll_angle
                if abs(roll_angle) <= 1e-9:
                    continue
                if sync_session.start_solver_state.rotation_property != "rotation_quaternion":
                    raise RigpedSemanticMoveError(
                        "Sliding GLOBAL lower-link roll requires quaternion solver rotation."
                    )
                solver_control = sync_session.capability.native_ik.solver_owner
                current_solver_state = _capture_control_state(solver_control)
                if current_solver_state.rotation_property != "rotation_quaternion":
                    raise RigpedSemanticMoveError(
                        "Sliding GLOBAL lower-link roll requires quaternion solver rotation."
                    )
                current_rotation = Quaternion(
                    current_solver_state.rotation
                ).normalized()
                desired_rotation = (
                    current_rotation
                    @ Quaternion(_RIGPED_LOCAL_AXES["Y"], roll_angle)
                ).normalized()
                _apply_control_state(
                    solver_control,
                    replace(
                        current_solver_state,
                        rotation=tuple(float(value) for value in desired_rotation),
                    ),
                    location=False,
                    rotation=True,
                )
                roll_capabilities.append(sync_session.capability)

            if roll_capabilities:
                context.view_layer.update()
                _refresh_current_sliding_public_overlays(
                    context,
                    capabilities=tuple(roll_capabilities),
                    allow_seed=False,
                )

            if self._sliding_dependency_guards:
                _refresh_frozen_sliding_dependency_overlays(
                    context,
                    self._sliding_dependency_guards,
                    operation_id=self._trace_operation_id,
                    phase="ROTATE_PREVIEW",
                )
            else:
                _refresh_current_sliding_public_overlays(
                    context,
                    capabilities=self._sliding_guard_capabilities,
                )
            return

        for state in state_by_bone.values():
            pose_bone = state.control.target
            assert isinstance(pose_bone, bpy.types.PoseBone)
            desired = resolve_desired(state)
            parent_desired = None
            if pose_bone.parent is not None:
                parent_desired = resolved_desired_by_bone.get(int(pose_bone.parent.as_pointer()))
            desired_state = _state_for_pose_matrix(
                state.control,
                desired,
                parent_pose_matrix=parent_desired,
            )
            _apply_control_state(
                state.control,
                desired_state,
                location=state.center_pivot,
                rotation=True,
            )
        context.view_layer.update()
        _apply_direct_rotate_sliding_syncs(
            context,
            self._sliding_syncs,
            pin_terminal=not sliding_lower_transports_terminal,
        )
        if self._sliding_dependency_guards:
            _refresh_frozen_sliding_dependency_overlays(
                context,
                self._sliding_dependency_guards,
                operation_id=self._trace_operation_id,
                phase="ROTATE_PREVIEW",
            )
        else:
            _refresh_current_sliding_public_overlays(
                context,
                capabilities=self._sliding_guard_capabilities,
            )

    def modal(self, context, event):
        if self._active is None:
            return {"CANCELLED"}
        if event.type == "MOUSEMOVE":
            mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
            applied = False
            if self.axis == "FREE":
                rotation_step = arcball_world_step(
                    context,
                    self._pivot,
                    self._previous_free_mouse,
                    mouse,
                    float(self.trackball_radius_px),
                )
                if abs(float(rotation_step.angle)) > 1e-12:
                    self._free_rotation = (
                        rotation_step @ self._free_rotation
                    ).normalized()
                    axis_world = Vector(self._free_rotation.axis)
                    angle = float(self._free_rotation.angle)
                    if axis_world.length > 1e-9 and angle > 1e-12:
                        axis_world.normalize()
                        self._axis_world = axis_world
                        self._current_angle = angle
                        applied = True
                self._previous_free_mouse = Vector(mouse)
            elif self._rotate_use_screen_tangent:
                mouse_delta = mouse - self._previous_rotation_mouse
                raw_step = (
                    float(mouse_delta.dot(self._rotate_screen_tangent))
                    * MAX_LINEAR_ROTATION_RADIANS_PER_PIXEL
                )
                self._raw_angle += raw_step
                self._previous_rotation_mouse = mouse
                applied = True
            else:
                current = _rotation_vector(
                    context,
                    self._pivot,
                    self._axis_world,
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
                if current is not None:
                    previous = self._previous_rotation_vector
                    raw_step = atan2(
                        float(self._axis_world.dot(previous.cross(current))),
                        float(previous.dot(current)),
                    )
                    pixel_step = float((mouse - self._previous_rotation_mouse).length)
                    max_step = min(0.35, max(0.006, pixel_step * 0.025))
                    step = (
                        0.0
                        if abs(raw_step) > max_step * 4.0
                        else max(-max_step, min(max_step, raw_step))
                    )
                    self._raw_angle += step
                    self._previous_rotation_vector = Vector(current)
                    self._previous_rotation_mouse = mouse
                    applied = True
            if applied:
                if self.axis == "FREE":
                    candidate = float(self._current_angle)
                else:
                    gesture_angle = snapped_rotation_angle(
                        context, event, self._raw_angle
                    )
                    self._current_angle = float(gesture_angle) * float(self._angle_sign)
                    candidate = float(self._current_angle)
                try:
                    self._apply_preview(context)
                except RigpedSemanticMoveError as exc:
                    self._restore_preview(context)
                    _report_operator_error(
                        self,
                        context,
                        exc,
                        tool="ROTATE",
                        route="DIRECT_ROTATE",
                        phase="MOUSEMOVE",
                        axis=self.axis,
                        attempted_angle=float(candidate),
                        orientation=str(self._orientation),
                    )
                    self._states = ()
                    self._sliding_syncs = ()
                    self._active = None
                    self._auto_plan = None
                    self._forearm_special_session = None
                    clear_rotation_angle(context)
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}
                if (
                    abs(self._current_angle - candidate) > 1e-9
                    and self.axis == "FREE"
                ):
                    self._free_rotation = Quaternion(
                        self._axis_world,
                        float(self._current_angle),
                    )
                # Keep constrained axis drags anchored to the original raw
                # mouse gesture. Feeding a bounded/semantic angle back into
                # _raw_angle changes the next MOUSEMOVE reference and causes
                # visible jumps when Sliding roll or joint limits engage.
                if self.axis != "FREE":
                    show_rotation_angle(
                        context, self._current_angle, self._pivot
                    )
                if context.area is not None:
                    context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            angle = float(self._current_angle)
            if abs(angle) <= 1e-9:
                self._restore_preview(context)
                self._states = ()
                self._sliding_syncs = ()
                self._active = None
                self._auto_plan = None
                self._auto_direct_plan = None
                self._forearm_special_session = None
                clear_rotation_angle(context)
                trace_event(
                    "OPERATION",
                    "TRANSFORM_CANCEL",
                    operation_id=self._trace_operation_id,
                    subsystem="modal",
                    lifecycle_phase="cancel",
                    terminal_status=TraceTerminalStatus.CANCELLED,
                    route_outcome=TraceRouteOutcome.CLAIMED,
                    context=context,
                    reason="NO_ANGLE",
                )
                self._trace_operation_id = None
                return {"CANCELLED"}

            auto_enabled_at_release = bool(
                getattr(context.scene, "baw_auto_key_enabled", False)
            )
            auto_plan = self._auto_plan if auto_enabled_at_release else None
            auto_direct_plan = (
                self._auto_direct_plan if auto_enabled_at_release else None
            )
            auto_contact_batch_plan = (
                self._auto_contact_batch_plan if auto_enabled_at_release else None
            )
            deferred_auto_contact_mapping_ids = (
                self._deferred_auto_contact_mapping_ids
                if auto_enabled_at_release
                else ()
            )
            deferred_auto_results: list[Any] = []
            auto_context = control_context_for_context(context)
            auto_limb_context, auto_direct_context = _direct_rotate_auto_contexts(
                context.scene,
                auto_context,
            )
            if (
                auto_contact_batch_plan is None
                and deferred_auto_contact_mapping_ids
            ):
                batch = plan_rigped_auto_contact_batch(
                    context.scene,
                    auto_limb_context,
                    operation_id=f"rigped-auto-rotate-release:{uuid4().hex}",
                    mapping_ids=deferred_auto_contact_mapping_ids,
                )
                if not batch.ok or batch.plan is None:
                    self._restore_preview(context)
                    detail = (
                        batch.diagnostics[0].detail
                        if batch.diagnostics
                        else "Rigped multi-limb Rotate Auto planning failed."
                    )
                    self.report({"WARNING"}, detail)
                    self._states = ()
                    self._sliding_syncs = ()
                    self._active = None
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    self._forearm_special_session = None
                    clear_rotation_angle(context)
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}
                auto_contact_batch_plan = batch.plan

            if auto_plan is not None:
                link_trace_operation(
                    auto_plan.intent.operation_id,
                    self._trace_operation_id,
                )
                try:
                    result = commit_rigped_auto_anchor(
                        context.scene,
                        auto_limb_context,
                        auto_plan,
                        defer_commit=True,
                    )
                except (ContactAuthoringError, RuntimeError, ValueError, ReferenceError) as exc:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    self._restore_preview(context)
                    _report_operator_error(self, context, exc)
                    self._states = ()
                    self._sliding_syncs = ()
                    self._active = None
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    self._forearm_special_session = None
                    clear_rotation_angle(context)
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}
                if not result.applied:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    self._restore_preview(context)
                    detail = "; ".join(item.detail for item in result.diagnostics)
                    self.report({"WARNING"}, detail or "Rigped semantic Rotate Auto commit failed.")
                    self._states = ()
                    self._sliding_syncs = ()
                    self._active = None
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    self._forearm_special_session = None
                    clear_rotation_angle(context)
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}
                deferred_auto_results.append(result)

            elif auto_contact_batch_plan is not None:
                link_trace_operation(
                    auto_contact_batch_plan.batch.operation_id,
                    self._trace_operation_id,
                )
                try:
                    result = commit_rigped_auto_contact_batch(
                        context.scene,
                        auto_limb_context,
                        auto_contact_batch_plan,
                        defer_commit=True,
                    )
                except (ContactAuthoringError, RuntimeError, ValueError, ReferenceError) as exc:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    self._restore_preview(context)
                    _report_operator_error(self, context, exc)
                    self._states = ()
                    self._sliding_syncs = ()
                    self._active = None
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    self._forearm_special_session = None
                    clear_rotation_angle(context)
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}
                if not result.applied:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    self._restore_preview(context)
                    detail = "; ".join(item.detail for item in result.diagnostics)
                    self.report({"WARNING"}, detail or "Rigped multi-limb Rotate Auto commit failed.")
                    self._states = ()
                    self._sliding_syncs = ()
                    self._active = None
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    self._forearm_special_session = None
                    clear_rotation_angle(context)
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}
                deferred_auto_results.append(result)

            if auto_direct_plan is not None:
                if auto_plan is not None or auto_contact_batch_plan is not None:
                    auto_direct_plan = replace(
                        auto_direct_plan,
                        allow_storage_rebind=True,
                    )
                link_trace_operation(
                    auto_direct_plan.begin_plan.operation_id,
                    self._trace_operation_id,
                )
                try:
                    result = commit_rigped_auto_direct_rotate(
                        context.scene,
                        auto_direct_context,
                        auto_direct_plan,
                        defer_commit=True,
                    )
                    if result.applied:
                        deferred_auto_results.append(result)

                        # A direct writer can trigger Action reevaluation across
                        # both actively selected Sliding limbs and passive guarded
                        # Sliding limbs. Final commit reconciliation therefore covers
                        # the gesture-start frozen affected set inside the same
                        # rollback-capable release boundary, before commit.
                        _refresh_current_sliding_public_overlays(
                            context,
                            capabilities=self._sliding_affected_capabilities,
                        )
                except (RuntimeError, ValueError, ReferenceError) as exc:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    self._restore_preview(context)
                    _report_operator_error(self, context, exc)
                    self._states = ()
                    self._sliding_syncs = ()
                    self._active = None
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    self._forearm_special_session = None
                    clear_rotation_angle(context)
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}
                if not result.applied:
                    rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                    self._restore_preview(context)
                    detail = "; ".join(item.detail for item in result.diagnostics)
                    self.report(
                        {"WARNING"},
                        detail or "Rigped direct Rotate Auto commit failed.",
                    )
                    self._states = ()
                    self._sliding_syncs = ()
                    self._active = None
                    self._auto_plan = None
                    self._auto_direct_plan = None
                    self._auto_contact_batch_plan = None
                    self._deferred_auto_contact_mapping_ids = ()
                    self._forearm_special_session = None
                    clear_rotation_angle(context)
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}

            try:
                commit_rigped_auto_writer_results(tuple(deferred_auto_results))
            except (RuntimeError, ValueError, ReferenceError) as exc:
                rollback_rigped_auto_writer_results(tuple(deferred_auto_results))
                self._restore_preview(context)
                _report_operator_error(self, context, exc)
                self._states = ()
                self._sliding_syncs = ()
                self._active = None
                self._auto_plan = None
                self._auto_direct_plan = None
                self._auto_contact_batch_plan = None
                self._deferred_auto_contact_mapping_ids = ()
                self._forearm_special_session = None
                clear_rotation_angle(context)
                if context.area is not None:
                    context.area.tag_redraw()
                return {"CANCELLED"}

            if deferred_auto_results:
                if auto_plan is not None:
                    frames = [
                        float(context.scene.frame_current)
                        + float(getattr(context.scene, "frame_subframe", 0.0))
                    ]
                    if auto_plan.baseline_time is not None:
                        frames.append(float(auto_plan.baseline_time))
                    clear_contact_bundle_key_selection_for_context(
                        context,
                        frames=tuple(frames),
                    )
                from .trackbar_model import clear_key_selection_for_context

                clear_key_selection_for_context(context)
                context.scene.baw_has_selected_key = False

            self._states = ()
            self._sliding_syncs = ()
            self._sliding_guard_capabilities = ()
            self._active = None
            self._auto_plan = None
            self._auto_direct_plan = None
            self._auto_contact_batch_plan = None
            self._deferred_auto_contact_mapping_ids = ()
            self._forearm_special_session = None
            clear_rotation_angle(context)
            if context.area is not None:
                context.area.tag_redraw()
            projected_dofs = self._sliding_global_projected_dofs
            projection_diagnostics = self._sliding_global_projection_diagnostics
            replay_rotation = (
                self._free_rotation.copy()
                if self.axis == "FREE"
                else Quaternion(self._axis_world, angle)
            )
            replay_quaternion = tuple(float(value) for value in replay_rotation)
            trace_event(
                "OPERATION",
                "TRANSFORM_COMMIT",
                operation_id=self._trace_operation_id,
                subsystem="modal",
                lifecycle_phase="commit",
                terminal_status=TraceTerminalStatus.FINISHED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
                tool="ROTATE",
                route="DIRECT_ROTATE",
                axis=self.axis,
                final_angle=angle,
                sliding_global_projected_dofs=projected_dofs,
                sliding_global_projection_diagnostics=projection_diagnostics,
                auto_key=bool(getattr(context.scene, "baw_auto_key_enabled", False)),
                replay_action={
                    "kind": "ROTATE",
                    "route": "DIRECT_ROTATE",
                    "frame": int(context.scene.frame_current),
                    "subframe": float(getattr(context.scene, "frame_subframe", 0.0)),
                    "controls": tuple(
                        str(item.name)
                        for item in tuple(getattr(context, "selected_pose_bones", ()) or ())
                    ),
                    "axis": self.axis,
                    "orientation": self._orientation,
                    "delta_world_quaternion": replay_quaternion,
                    "final_angle": float(angle),
                },
            )
            self._trace_operation_id = None
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._restore_preview(context)
            self._states = ()
            self._sliding_syncs = ()
            self._sliding_guard_capabilities = ()
            self._active = None
            self._auto_plan = None
            self._auto_direct_plan = None
            self._auto_contact_batch_plan = None
            self._deferred_auto_contact_mapping_ids = ()
            self._forearm_special_session = None
            clear_rotation_angle(context)
            if context.area is not None:
                context.area.tag_redraw()
            trace_event(
                "OPERATION",
                "TRANSFORM_CANCEL",
                operation_id=self._trace_operation_id,
                subsystem="modal",
                lifecycle_phase="cancel",
                terminal_status=TraceTerminalStatus.CANCELLED,
                route_outcome=TraceRouteOutcome.CLAIMED,
                context=context,
                reason=event.type,
            )
            self._trace_operation_id = None
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}


class BAW_GGT_rigped_direct_rotate(bpy.types.GizmoGroup):
    """Anchored Fit-style Rotate shell with Blender-native FK authority."""

    bl_idname = "BAW_GGT_rigped_direct_rotate"
    bl_label = "AWB Rigped Direct Rotate Gizmo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options: ClassVar[set[str]] = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "mode", "") == "POSE"
            and getattr(context.scene, "baw_rigped_semantic_transform_mode", "NONE")
            == "DIRECT_ROTATE"
            and direct_rotate_available(context)
        )

    def setup(self, _context):
        self._zoom_reference_distance = None
        self.axis_gizmos = {}
        self.axis_hit_gizmos = {}
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        active_color = (1.0, 0.82, 0.05)
        for axis_name in ("X", "Y", "Z"):
            visible = self.gizmos.new("GIZMO_GT_dial_3d")
            visible.color = colors[axis_name]
            visible.alpha = 0.65
            visible.color_highlight = active_color
            visible.alpha_highlight = 0.95
            visible.line_width = 2.0
            visible.scale_basis = 1.0
            visible.draw_options = {"CLIP"}
            visible.hide_select = True
            visible.use_draw_modal = True
            self.axis_gizmos[axis_name] = visible

            hit = self.gizmos.new("GIZMO_GT_dial_3d")
            props = hit.target_set_operator(BAW_OT_rigped_direct_rotate_axis.bl_idname)
            props.axis = axis_name
            hit.color = colors[axis_name]
            hit.alpha = 0.001
            hit.color_highlight = active_color
            hit.alpha_highlight = 0.95
            hit.line_width = 2.0
            hit.scale_basis = 1.0
            hit.draw_options = {"CLIP"}
            hit.select_bias = 2.0
            hit.use_draw_modal = False
            self.axis_hit_gizmos[axis_name] = hit

        view_visible = self.gizmos.new("GIZMO_GT_dial_3d")
        view_visible.color = (0.85, 0.85, 0.85)
        view_visible.alpha = 0.45
        view_visible.color_highlight = active_color
        view_visible.alpha_highlight = 0.95
        view_visible.line_width = 1.5
        view_visible.scale_basis = 1.15
        view_visible.hide_select = True
        view_visible.use_draw_modal = True
        self.view_gizmo = view_visible

        view_hit = self.gizmos.new("GIZMO_GT_dial_3d")
        view_props = view_hit.target_set_operator(BAW_OT_rigped_direct_rotate_axis.bl_idname)
        view_props.axis = "VIEW"
        view_hit.color = (0.85, 0.85, 0.85)
        view_hit.alpha = 0.001
        view_hit.color_highlight = active_color
        view_hit.alpha_highlight = 0.95
        view_hit.line_width = 1.5
        view_hit.scale_basis = 1.15
        view_hit.select_bias = 2.0
        view_hit.use_draw_modal = False
        self.view_hit_gizmo = view_hit

    def draw_prepare(self, context):
        pivot = direct_transform_pivot(context)
        axes = direct_transform_axes(context)
        if pivot is None or axes is None or not direct_rotate_available(context):
            for gizmo in (
                *self.axis_gizmos.values(),
                *self.axis_hit_gizmos.values(),
                self.view_gizmo,
                self.view_hit_gizmo,
            ):
                gizmo.hide = True
            return

        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        active_color = (1.0, 0.82, 0.05)
        gizmo_scale, self._zoom_reference_distance = gizmo_zoom_scale(
            context,
            self._zoom_reference_distance,
        )
        for axis_name, visible in self.axis_gizmos.items():
            hit = self.axis_hit_gizmos[axis_name]
            visible.scale_basis = gizmo_scale
            hit.scale_basis = gizmo_scale
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(axes[axis_name])
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            visible.color = (
                active_color
                if bool(getattr(hit, "is_highlight", False))
                or bool(getattr(hit, "is_modal", False))
                else colors[axis_name]
            )
            visible.alpha = (
                0.95
                if bool(getattr(hit, "is_highlight", False))
                or bool(getattr(hit, "is_modal", False))
                else 0.65
            )
            visible.matrix_basis = matrix
            hit.matrix_basis = matrix
            visible.hide = False
            hit.hide = False

        view_axis = _view_axis(context)
        if view_axis is None:
            self.view_gizmo.hide = True
            self.view_hit_gizmo.hide = True
        else:
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(view_axis)
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            view_active = bool(getattr(self.view_hit_gizmo, "is_highlight", False)) or bool(
                getattr(self.view_hit_gizmo, "is_modal", False)
            )
            self.view_gizmo.color = active_color if view_active else (0.85, 0.85, 0.85)
            self.view_gizmo.alpha = 0.95 if view_active else 0.45
            self.view_gizmo.scale_basis = 1.15 * gizmo_scale
            self.view_hit_gizmo.scale_basis = 1.15 * gizmo_scale
            self.view_gizmo.matrix_basis = matrix
            self.view_hit_gizmo.matrix_basis = matrix
            self.view_gizmo.hide = False
            self.view_hit_gizmo.hide = False


class BAW_GGT_rigped_semantic_move(bpy.types.GizmoGroup):
    """Anchored Blender-style Move shell for Sliding Hand/Foot IK.

    Visible handles are display-only. Invisible hit proxies own the custom
    semantic move modal so the visible gizmo cannot accumulate Blender gizmo
    offset and drift away from the actual Hand/Foot + IK pivot during drag.
    """

    bl_idname = "BAW_GGT_rigped_semantic_move"
    bl_label = "AWB Rigped Semantic Move Gizmo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options: ClassVar[set[str]] = {"3D", "PERSISTENT", "SHOW_MODAL_ALL"}

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "mode", "") == "POSE"
            and getattr(context.scene, "baw_rigped_semantic_transform_mode", "NONE") == "MOVE"
            and semantic_move_available(context)
        )

    def setup(self, _context):
        self._zoom_reference_distance = None
        self.axis_gizmos = {}
        self.axis_hit_gizmos = {}
        self.plane_gizmos = {}
        self.plane_hit_gizmos = {}
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        for axis_name in ("X", "Y", "Z"):
            visible = self.gizmos.new("GIZMO_GT_arrow_3d")
            visible.draw_style = "NORMAL"
            visible.draw_options = {"STEM"}
            visible.color = colors[axis_name]
            visible.alpha = 0.9
            visible.color_highlight = colors[axis_name]
            visible.alpha_highlight = 0.9
            visible.line_width = 1.0
            visible.scale_basis = 1.0
            visible.hide_select = True
            visible.use_draw_modal = True
            self.axis_gizmos[axis_name] = visible

            hit = self.gizmos.new("GIZMO_GT_arrow_3d")
            props = hit.target_set_operator(BAW_OT_rigped_semantic_move_axis.bl_idname)
            props.axis = axis_name
            hit.draw_style = "NORMAL"
            hit.draw_options = {"STEM"}
            hit.color = colors[axis_name]
            hit.alpha = 0.001
            hit.color_highlight = (1.0, 1.0, 1.0)
            hit.alpha_highlight = 1.0
            hit.line_width = 1.0
            hit.scale_basis = 1.0
            hit.select_bias = 2.0
            hit.use_draw_modal = False
            self.axis_hit_gizmos[axis_name] = hit

        plane_colors = {"XY": colors["Z"], "XZ": colors["Y"], "YZ": colors["X"]}
        for plane_name in ("XY", "XZ", "YZ"):
            visible = self.gizmos.new("GIZMO_GT_primitive_3d")
            visible.draw_style = "PLANE"
            visible.draw_inner = True
            visible.color = plane_colors[plane_name]
            visible.alpha = 0.5
            visible.color_highlight = plane_colors[plane_name]
            visible.alpha_highlight = 0.5
            visible.line_width = 1.0
            visible.scale_basis = 0.11
            visible.hide_select = True
            visible.use_draw_offset_scale = True
            visible.use_draw_modal = True
            self.plane_gizmos[plane_name] = visible

            hit = self.gizmos.new("GIZMO_GT_primitive_3d")
            props = hit.target_set_operator(BAW_OT_rigped_semantic_move_axis.bl_idname)
            props.axis = plane_name
            hit.draw_style = "PLANE"
            hit.draw_inner = True
            hit.color = plane_colors[plane_name]
            hit.alpha = 0.001
            hit.color_highlight = (1.0, 1.0, 1.0)
            hit.alpha_highlight = 0.95
            hit.line_width = 1.0
            hit.scale_basis = 0.14
            hit.select_bias = 2.0
            hit.use_draw_offset_scale = True
            hit.use_draw_modal = False
            self.plane_hit_gizmos[plane_name] = hit

        center = self.gizmos.new("GIZMO_GT_dial_3d")
        center.color = (0.82, 0.82, 0.82)
        center.alpha = 0.9
        center.color_highlight = (0.82, 0.82, 0.82)
        center.alpha_highlight = 0.9
        center.line_width = 1.5
        center.scale_basis = 0.18
        center.hide_select = True
        center.use_draw_modal = True
        self.center_gizmo = center

        center_hit = self.gizmos.new("GIZMO_GT_dial_3d")
        props = center_hit.target_set_operator(BAW_OT_rigped_semantic_move_axis.bl_idname)
        props.axis = "FREE"
        center_hit.color = (0.82, 0.82, 0.82)
        center_hit.alpha = 0.001
        center_hit.color_highlight = (1.0, 1.0, 1.0)
        center_hit.alpha_highlight = 1.0
        center_hit.line_width = 1.5
        center_hit.scale_basis = 0.20
        center_hit.select_bias = 2.0
        center_hit.use_draw_modal = False
        self.center_hit_gizmo = center_hit

    def draw_prepare(self, context):
        pivot = semantic_move_pivot(context)
        axes = semantic_move_axes(context)
        if pivot is None or axes is None:
            for gizmo in (
                *self.axis_gizmos.values(),
                *self.axis_hit_gizmos.values(),
                *self.plane_gizmos.values(),
                *self.plane_hit_gizmos.values(),
                self.center_gizmo,
                self.center_hit_gizmo,
            ):
                gizmo.hide = True
            return

        gizmo_scale, self._zoom_reference_distance = gizmo_zoom_scale(
            context,
            self._zoom_reference_distance,
        )
        self.center_gizmo.scale_basis = 0.18 * gizmo_scale
        self.center_hit_gizmo.scale_basis = 0.20 * gizmo_scale
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        for axis_name, visible in self.axis_gizmos.items():
            hit = self.axis_hit_gizmos[axis_name]
            visible.scale_basis = gizmo_scale
            hit.scale_basis = gizmo_scale
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(axes[axis_name])
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            visible.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else colors[axis_name]
            )
            visible.matrix_basis = matrix
            hit.matrix_basis = matrix
            visible.hide = False
            hit.hide = False

        resolved = _selected_semantic_mapping(context)
        world_length = 0.3
        if resolved is not None:
            active = resolved.active_control
            if isinstance(active.target, bpy.types.PoseBone):
                world_length = float(
                    Vector(
                        active.owner_object.matrix_world.to_3x3()
                        @ (active.target.tail - active.target.head)
                    ).length
                )
        plane_offset = max(0.025, min(0.07, world_length * 0.14))
        plane_axes = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}
        plane_colors = {"XY": colors["Z"], "XZ": colors["Y"], "YZ": colors["X"]}
        for plane_name, visible in self.plane_gizmos.items():
            hit = self.plane_hit_gizmos[plane_name]
            visible.scale_basis = 0.11 * gizmo_scale
            hit.scale_basis = 0.14 * gizmo_scale
            first_name, second_name = plane_axes[plane_name]
            first = Vector(axes[first_name])
            second = Vector(axes[second_name])
            normal = first.cross(second).normalized()
            matrix = Matrix((first, second, normal)).transposed().to_4x4()
            matrix.translation = pivot + (first + second) * plane_offset
            visible.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else plane_colors[plane_name]
            )
            visible.matrix_basis = matrix
            hit.matrix_basis = matrix
            visible.hide = False
            hit.hide = False

        view_axis = _view_axis(context)
        if view_axis is None:
            self.center_gizmo.hide = True
            self.center_hit_gizmo.hide = True
        else:
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(view_axis)
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            self.center_gizmo.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(self.center_hit_gizmo, "is_highlight", False))
                else (0.82, 0.82, 0.82)
            )
            self.center_gizmo.matrix_basis = matrix
            self.center_hit_gizmo.matrix_basis = matrix
            self.center_gizmo.hide = False
            self.center_hit_gizmo.hide = False
