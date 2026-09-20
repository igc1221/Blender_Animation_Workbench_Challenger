from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .phase4_verification import Diagnostic, DiagnosticSeverity, OperationStage

type RuntimeControlKey = tuple[int, int]
type OwnerBindingToken = tuple[int | None, int | None, int | None, int | None]


class OperationType(StrEnum):
    DIRECT_KEY = "DIRECT_KEY"
    ALL_KEY = "ALL_KEY"
    KINEMATIC_MOVE = "KINEMATIC_MOVE"
    CONTACT = "CONTACT"


class ChannelFamily(StrEnum):
    POSITION = "POSITION"
    ROTATION = "ROTATION"
    SEMANTIC_STATE = "SEMANTIC_STATE"


class RotationRepresentation(StrEnum):
    EULER = "EULER"
    QUATERNION = "QUATERNION"
    AXIS_ANGLE = "AXIS_ANGLE"


class AllocationIntent(StrEnum):
    EXISTING_FCURVE = "EXISTING_FCURVE"
    NEED_FCURVE = "NEED_FCURVE"
    NEED_ASSIGNED_BAG = "NEED_ASSIGNED_BAG"
    NEED_ACTION_AND_ASSIGNED_SLOT = "NEED_ACTION_AND_ASSIGNED_SLOT"
    NEED_ANIMATION_DATA_ACTION_SLOT_BAG = "NEED_ANIMATION_DATA_ACTION_SLOT_BAG"


@dataclass(frozen=True, slots=True, order=True)
class ChannelRowKey:
    binding_id: str
    family: ChannelFamily
    data_path: str
    array_index: int


@dataclass(frozen=True, slots=True)
class PlannedChannel:
    row_key: ChannelRowKey
    semantic_key: str
    owner_runtime_key: int
    control_runtime_key: RuntimeControlKey
    owner_binding_token: OwnerBindingToken
    rotation_mode: str | None
    rotation_representation: RotationRepresentation | None
    existing_fcurve_token: int | None
    allocation_intent: AllocationIntent
    target_value: float


@dataclass(frozen=True, slots=True)
class PlannedOwnerGroup:
    owner_runtime_key: int
    owner_binding_token: OwnerBindingToken
    channels: tuple[PlannedChannel, ...]


@dataclass(frozen=True, slots=True)
class MissingAllocation:
    owner_runtime_key: int
    owner_binding_token: OwnerBindingToken
    intent: AllocationIntent
    row_keys: tuple[ChannelRowKey, ...]


@dataclass(frozen=True, slots=True)
class ReadFootprint:
    character_source_stamp: tuple
    setup_revision: int
    setup_signature: str
    selected_binding_ids: tuple[str, ...]
    active_binding_id: str | None
    selection_runtime_keys: tuple[RuntimeControlKey, ...]
    active_runtime_key: RuntimeControlKey | None
    owner_binding_tokens: tuple[OwnerBindingToken, ...]
    control_runtime_keys: tuple[RuntimeControlKey, ...]
    native_selection_runtime_keys: tuple[RuntimeControlKey, ...] = ()
    native_active_runtime_key: RuntimeControlKey | None = None


@dataclass(frozen=True, slots=True)
class KinematicDependency:
    mapping_id: str
    driven_binding_ids: tuple[str, ...]
    ik_target_binding_id: str
    pole_binding_id: str | None
    solver_owner_binding_id: str
    constraint_runtime_token: int
    solver_owner_runtime_key: RuntimeControlKey
    ik_target_runtime_key: RuntimeControlKey
    pole_runtime_key: RuntimeControlKey | None
    chain_count: int
    use_tail: bool
    use_stretch: bool
    use_rotation: bool


@dataclass(frozen=True, slots=True)
class DependencyFootprint:
    semantic_binding_ids: tuple[str, ...]
    owner_binding_tokens: tuple[OwnerBindingToken, ...]
    kinematic: KinematicDependency | None = None


@dataclass(frozen=True, slots=True)
class PlanWriteFootprint:
    row_keys: tuple[ChannelRowKey, ...]
    missing_allocations: tuple[MissingAllocation, ...]


@dataclass(frozen=True, slots=True)
class OperationPlan:
    operation_id: str
    operation_type: OperationType
    character_id: str
    setup_revision: int
    setup_signature: str
    character_source_stamp: tuple
    frame: int
    subframe: float
    selected_binding_ids: tuple[str, ...]
    active_binding_id: str | None
    owner_groups: tuple[PlannedOwnerGroup, ...]
    read_footprint: ReadFootprint
    dependency_footprint: DependencyFootprint
    write_footprint: PlanWriteFootprint
    kinematic_dependency: KinematicDependency | None = None


@dataclass(frozen=True, slots=True)
class PlanBuildResult:
    plan: OperationPlan | None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None and not any(
            diagnostic.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.BLOCKER}
            for diagnostic in self.diagnostics
        )


@dataclass(frozen=True, slots=True)
class LiveFreshnessFacts:
    character_source_stamp: tuple
    setup_revision: int
    setup_signature: str
    selected_binding_ids: tuple[str, ...]
    active_binding_id: str | None
    selection_runtime_keys: tuple[RuntimeControlKey, ...]
    active_runtime_key: RuntimeControlKey | None
    owner_binding_tokens: tuple[OwnerBindingToken, ...]
    control_runtime_keys: tuple[RuntimeControlKey, ...]
    existing_fcurve_tokens: tuple[tuple[ChannelRowKey, int | None], ...]
    kinematic_constraint_token: int | None = None
    kinematic_solver_owner_key: RuntimeControlKey | None = None
    kinematic_target_key: RuntimeControlKey | None = None
    kinematic_pole_key: RuntimeControlKey | None = None
    native_selection_runtime_keys: tuple[RuntimeControlKey, ...] = ()
    native_active_runtime_key: RuntimeControlKey | None = None


@dataclass(frozen=True, slots=True)
class FreshnessResult:
    ok: bool
    diagnostics: tuple[Diagnostic, ...]


def diagnostic(code: str, detail: str) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        stage=OperationStage.PREFLIGHT,
        detail=detail,
    )


def channel_sort_key(channel: PlannedChannel) -> tuple:
    row = channel.row_key
    return (
        channel.owner_runtime_key,
        row.binding_id,
        row.family.value,
        row.data_path,
        row.array_index,
    )


def validate_plan_structure(plan: OperationPlan) -> tuple[Diagnostic, ...]:
    issues: list[Diagnostic] = []
    seen_rows: set[ChannelRowKey] = set()
    for group in plan.owner_groups:
        if not group.channels:
            issues.append(diagnostic("I3_EMPTY_OWNER_GROUP", "Operation plan contains an empty owner group."))
        for channel in group.channels:
            if channel.row_key in seen_rows:
                issues.append(
                    diagnostic(
                        "I3_DUPLICATE_CHANNEL_ROW",
                        f"Duplicate planned channel row: {channel.row_key!r}",
                    )
                )
            seen_rows.add(channel.row_key)
            if channel.owner_runtime_key != group.owner_runtime_key:
                issues.append(
                    diagnostic(
                        "I3_OWNER_GROUP_MISMATCH",
                        f"Channel owner does not match containing group: {channel.row_key!r}",
                    )
                )
            if channel.owner_binding_token != group.owner_binding_token:
                issues.append(
                    diagnostic(
                        "I3_OWNER_BINDING_TOKEN_MISMATCH",
                        f"Channel binding token does not match containing group: {channel.row_key!r}",
                    )
                )
            if channel.row_key.family is ChannelFamily.POSITION:
                if channel.row_key.array_index not in (0, 1, 2):
                    issues.append(
                        diagnostic(
                            "I3_BAD_POSITION_INDEX",
                            f"Position row has invalid array index: {channel.row_key!r}",
                        )
                    )
                if channel.rotation_representation is not None or channel.rotation_mode is not None:
                    issues.append(
                        diagnostic(
                            "I3_POSITION_HAS_ROTATION_METADATA",
                            f"Position row contains rotation metadata: {channel.row_key!r}",
                        )
                    )
            elif channel.row_key.family is ChannelFamily.ROTATION:
                rep = channel.rotation_representation
                if rep is None or channel.rotation_mode is None:
                    issues.append(
                        diagnostic(
                            "I3_ROTATION_METADATA_MISSING",
                            f"Rotation row lacks frozen representation metadata: {channel.row_key!r}",
                        )
                    )
                max_index = 3 if rep in {RotationRepresentation.QUATERNION, RotationRepresentation.AXIS_ANGLE} else 2
                if channel.row_key.array_index < 0 or channel.row_key.array_index > max_index:
                    issues.append(
                        diagnostic(
                            "I3_BAD_ROTATION_INDEX",
                            f"Rotation row has invalid array index: {channel.row_key!r}",
                        )
                    )
            elif channel.row_key.family is ChannelFamily.SEMANTIC_STATE:
                if channel.row_key.array_index != 0:
                    issues.append(
                        diagnostic(
                            "I3_BAD_SEMANTIC_STATE_INDEX",
                            f"Semantic state row must use scalar index 0: {channel.row_key!r}",
                        )
                    )
                if channel.rotation_representation is not None or channel.rotation_mode is not None:
                    issues.append(
                        diagnostic(
                            "I3_SEMANTIC_STATE_HAS_ROTATION_METADATA",
                            f"Semantic state row contains rotation metadata: {channel.row_key!r}",
                        )
                    )
            if (
                channel.allocation_intent is AllocationIntent.EXISTING_FCURVE
                and channel.existing_fcurve_token is None
            ):
                issues.append(
                    diagnostic(
                        "I3_EXISTING_FCURVE_TOKEN_MISSING",
                        f"Existing FCurve row lacks runtime token: {channel.row_key!r}",
                    )
                )
            if (
                channel.allocation_intent is not AllocationIntent.EXISTING_FCURVE
                and channel.existing_fcurve_token is not None
            ):
                issues.append(
                    diagnostic(
                        "I3_MISSING_FCURVE_HAS_TOKEN",
                        f"Missing-allocation row unexpectedly has FCurve token: {channel.row_key!r}",
                    )
                )
    if tuple(sorted(plan.write_footprint.row_keys)) != tuple(sorted(seen_rows)):
        issues.append(
            diagnostic(
                "I3_WRITE_FOOTPRINT_MISMATCH",
                "Write footprint does not exactly match planned channel rows.",
            )
        )
    return tuple(issues)


def validate_freshness(plan: OperationPlan, live: LiveFreshnessFacts) -> FreshnessResult:
    issues: list[Diagnostic] = []

    def mismatch(code: str, detail: str) -> None:
        issues.append(diagnostic(code, detail))

    if live.character_source_stamp != plan.character_source_stamp:
        mismatch("I3_STALE_CHARACTER_STAMP", "Character source stamp changed after planning.")
    if live.setup_revision != plan.setup_revision:
        mismatch("I3_STALE_SETUP_REVISION", "Rigped setup revision changed after planning.")
    if live.setup_signature != plan.setup_signature:
        mismatch("I3_STALE_SETUP_SIGNATURE", "Rigped setup signature changed after planning.")
    if live.selected_binding_ids != plan.selected_binding_ids:
        mismatch("I3_SELECTION_CHANGED", "Native selected semantic bindings changed after planning.")
    if live.active_binding_id != plan.active_binding_id:
        mismatch("I3_ACTIVE_CHANGED", "Native active semantic binding changed after planning.")
    read = plan.read_footprint
    if live.selection_runtime_keys != read.selection_runtime_keys:
        mismatch("I3_SELECTION_RUNTIME_CHANGED", "Selected runtime control identity changed after planning.")
    if live.active_runtime_key != read.active_runtime_key:
        mismatch("I3_ACTIVE_RUNTIME_CHANGED", "Active runtime control identity changed after planning.")
    if read.native_selection_runtime_keys and (
        live.native_selection_runtime_keys != read.native_selection_runtime_keys
    ):
        mismatch(
            "E2_NATIVE_SELECTION_RUNTIME_CHANGED",
            "Raw native selected-control identity changed after direct-subset planning.",
        )
    if read.native_selection_runtime_keys and (
        live.native_active_runtime_key != read.native_active_runtime_key
    ):
        mismatch(
            "E2_NATIVE_ACTIVE_RUNTIME_CHANGED",
            "Raw native active-control identity changed after direct-subset planning.",
        )
    if live.owner_binding_tokens != read.owner_binding_tokens:
        mismatch("I3_OWNER_BINDING_CHANGED", "Owner Action/Slot/ChannelBag binding changed after planning.")
    if live.control_runtime_keys != read.control_runtime_keys:
        mismatch("I3_CONTROL_RUNTIME_CHANGED", "Resolved native control identity changed after planning.")

    live_fcurves = dict(live.existing_fcurve_tokens)
    for group in plan.owner_groups:
        for channel in group.channels:
            if channel.allocation_intent is not AllocationIntent.EXISTING_FCURVE:
                continue
            if live_fcurves.get(channel.row_key) != channel.existing_fcurve_token:
                mismatch(
                    "I3_FCURVE_BINDING_CHANGED",
                    f"Existing FCurve identity changed after planning: {channel.row_key!r}",
                )

    dependency = plan.kinematic_dependency
    if dependency is not None:
        if live.kinematic_constraint_token != dependency.constraint_runtime_token:
            mismatch("I3_KINEMATIC_CONSTRAINT_CHANGED", "IK constraint identity changed after planning.")
        if live.kinematic_solver_owner_key != dependency.solver_owner_runtime_key:
            mismatch("I3_KINEMATIC_SOLVER_OWNER_CHANGED", "IK solver owner changed after planning.")
        if live.kinematic_target_key != dependency.ik_target_runtime_key:
            mismatch("I3_KINEMATIC_TARGET_CHANGED", "IK target changed after planning.")
        if live.kinematic_pole_key != dependency.pole_runtime_key:
            mismatch("I3_KINEMATIC_POLE_CHANGED", "IK pole changed after planning.")

    return FreshnessResult(not issues, tuple(issues))
