from __future__ import annotations

from dataclasses import dataclass, replace

from .debug_trace import trace_event
from .phase4_contact_authoring import (
    ContactAuthoringMode,
    ContactAuthoringResult,
    ContactBatchIntentPlan,
    ContactIntentPlan,
    build_contact_batch_intent_plan,
    build_contact_intent_plan,
    execute_contact_batch_intent_plan,
    execute_contact_intent_plan,
    finalize_deferred_contact_authoring_result,
    snapshot_contact_transform_rows,
)
from .phase4_contact_model import ContactKeyType
from .phase4_mutation_journal import MutationJournal
from .phase4_operation_plan import ChannelFamily, OperationPlan, PlannedChannel
from .phase4_preflight import build_direct_key_plan
from .phase4_verification import (
    Diagnostic,
    DiagnosticSeverity,
    OperationStage,
)
from .phase4_writer import (
    DirectWriterResult,
    WriterTrigger,
    execute_direct_key_plan,
    finalize_deferred_direct_writer_result,
)
from .semantic_adapter import assigned_channelbag


@dataclass(frozen=True, slots=True)
class RigpedAutoDirectRotatePlan:
    """One direct-control Rotate gesture using the existing direct planner/writer."""

    begin_plan: OperationPlan
    baseline_plan: OperationPlan | None = None
    baseline_time: float | None = None
    active_only: bool = True
    requested_families: tuple[ChannelFamily, ...] = (ChannelFamily.ROTATION,)
    allow_storage_rebind: bool = False
    trigger: WriterTrigger = WriterTrigger.AUTO_TRANSFORM


@dataclass(frozen=True, slots=True)
class RigpedAutoDirectRotatePlanResult:
    plan: RigpedAutoDirectRotatePlan | None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.diagnostics


@dataclass(frozen=True, slots=True)
class RigpedAutoDirectMovePlan:
    """One direct-control Move gesture using the existing direct planner/writer."""

    begin_plan: OperationPlan
    baseline_plan: OperationPlan | None = None
    baseline_time: float | None = None
    active_only: bool = True
    trigger: WriterTrigger = WriterTrigger.AUTO_TRANSFORM


@dataclass(frozen=True, slots=True)
class RigpedAutoDirectMovePlanResult:
    plan: RigpedAutoDirectMovePlan | None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.diagnostics


AutoWriterResult = DirectWriterResult | ContactAuthoringResult


def rollback_rigped_auto_writer_results(
    results: tuple[AutoWriterResult, ...],
) -> None:
    """Rollback every still-open deferred AUTO writer in reverse order."""

    for result in reversed(results):
        journal = result.pending_journal
        if journal is None:
            continue
        if journal.state.value == "OPEN":
            journal.rollback_or_raise()


def commit_rigped_auto_writer_results(
    results: tuple[AutoWriterResult, ...],
) -> None:
    """Finalize a verified set of deferred AUTO writers as one gesture."""

    pending = tuple(result for result in results if result.pending_journal is not None)
    for result in pending:
        if (
            not result.applied
            or result.pending_journal is None
            or result.operation_id is None
        ):
            raise RuntimeError("AUTO writer result is not commit-ready.")
        if result.pending_journal.state.value != "OPEN":
            raise RuntimeError(
                "AUTO writer journal is not open: "
                f"{result.pending_journal.state.value}."
            )

    MutationJournal.commit_group(
        tuple(
            result.pending_journal
            for result in pending
            if result.pending_journal is not None
        )
    )

    # The persistent rollback boundary is crossed exactly once above. The
    # per-result finalizers now perform only direct-writer bookkeeping and
    # non-blocking trace emission for already-COMMITTED journals.
    for result in pending:
        if isinstance(result, DirectWriterResult):
            finalize_deferred_direct_writer_result(result)
        else:
            finalize_deferred_contact_authoring_result(result)

    if pending:
        trace_event(
            "WRITER",
            "AUTO_TRANSACTION_COMMIT",
            operation_ids=tuple(
                str(result.operation_id)
                for result in pending
                if result.operation_id is not None
            ),
            writer_count=len(pending),
        )


def _direct_plan_family_channels(
    plan: OperationPlan,
    families: tuple[ChannelFamily, ...],
) -> tuple[PlannedChannel, ...]:
    family_set = set(families)
    return tuple(
        channel
        for group in plan.owner_groups
        for channel in group.channels
        if getattr(channel.row_key, "family", None) in family_set
    )


def _direct_plan_keyed_row_keys(
    control_context,
    plan: OperationPlan,
    *,
    families: tuple[ChannelFamily, ...],
) -> frozenset:
    owners = {
        int(resolved.owner_object.as_pointer()): resolved.owner_object
        for resolved in tuple(getattr(control_context, "controls", ()) or ())
    }
    family_set = set(families)
    keyed = set()
    for group in plan.owner_groups:
        owner = owners.get(group.owner_runtime_key)
        bag = assigned_channelbag(owner) if owner is not None else None
        for channel in group.channels:
            if getattr(channel.row_key, "family", None) not in family_set:
                continue
            curve = None
            if bag is not None:
                curve = next(
                    (
                        candidate
                        for candidate in bag.fcurves
                        if str(candidate.data_path) == channel.row_key.data_path
                        and int(candidate.array_index) == channel.row_key.array_index
                    ),
                    None,
                )
            if curve is not None and len(curve.keyframe_points) > 0:
                keyed.add(channel.row_key)
    return frozenset(keyed)


def _direct_plan_keyed_row_count(
    control_context,
    plan: OperationPlan,
    *,
    families: tuple[ChannelFamily, ...],
) -> int:
    return len(
        _direct_plan_keyed_row_keys(
            control_context,
            plan,
            families=families,
        )
    )


def _filter_direct_plan_families(
    plan: OperationPlan,
    families: tuple[ChannelFamily, ...],
) -> OperationPlan:
    family_set = set(families)
    owner_groups = tuple(
        replace(
            group,
            channels=tuple(
                channel
                for channel in group.channels
                if channel.row_key.family in family_set
            ),
        )
        for group in plan.owner_groups
    )
    row_keys = tuple(
        row_key
        for row_key in plan.write_footprint.row_keys
        if row_key.family in family_set
    )
    missing_allocations = []
    for missing in plan.write_footprint.missing_allocations:
        filtered_keys = tuple(
            row_key for row_key in missing.row_keys if row_key.family in family_set
        )
        if filtered_keys:
            missing_allocations.append(replace(missing, row_keys=filtered_keys))
    return replace(
        plan,
        owner_groups=owner_groups,
        write_footprint=replace(
            plan.write_footprint,
            row_keys=row_keys,
            missing_allocations=tuple(missing_allocations),
        ),
    )


def _filter_direct_plan_row_keys(
    plan: OperationPlan,
    row_keys,
) -> OperationPlan:
    row_key_set = set(row_keys)
    owner_groups = tuple(
        replace(
            group,
            channels=tuple(
                channel
                for channel in group.channels
                if channel.row_key in row_key_set
            ),
        )
        for group in plan.owner_groups
    )
    owner_groups = tuple(group for group in owner_groups if group.channels)
    missing_allocations = []
    for missing in plan.write_footprint.missing_allocations:
        filtered_keys = tuple(
            row_key for row_key in missing.row_keys if row_key in row_key_set
        )
        if filtered_keys:
            missing_allocations.append(replace(missing, row_keys=filtered_keys))
    return replace(
        plan,
        owner_groups=owner_groups,
        write_footprint=replace(
            plan.write_footprint,
            row_keys=tuple(
                row_key
                for row_key in plan.write_footprint.row_keys
                if row_key in row_key_set
            ),
            missing_allocations=tuple(missing_allocations),
        ),
    )


def _direct_plan_baseline_rows_by_control(
    control_context,
    plan: OperationPlan,
    *,
    families: tuple[ChannelFamily, ...],
    operation_id: str,
):
    keyed_rows = _direct_plan_keyed_row_keys(
        control_context,
        plan,
        families=families,
    )
    baseline_rows = set()
    baseline_controls = set()
    diagnostics = []
    for family in families:
        family_channels = _direct_plan_family_channels(plan, (family,))
        grouped = {}
        for channel in family_channels:
            grouped.setdefault(channel.control_runtime_key, []).append(channel)
        for control_key, channels in grouped.items():
            keyed_count = sum(channel.row_key in keyed_rows for channel in channels)
            if 0 < keyed_count < len(channels):
                family_name = "rotation" if family is ChannelFamily.ROTATION else "position"
                family_code = (
                    "AK_AUTO_PARTIAL_DIRECT_ROTATION"
                    if family is ChannelFamily.ROTATION
                    else "AK_AUTO_PARTIAL_DIRECT_POSITION"
                )
                diagnostics.append(
                    Diagnostic(
                        code=family_code,
                        severity=DiagnosticSeverity.ERROR,
                        stage=OperationStage.PREFLIGHT,
                        operation=operation_id,
                        detail=(
                            f"Existing direct {family_name} animation is only partially authored "
                            f"for binding {channels[0].row_key.binding_id!r}; AUTO will not repair "
                            f"or reinterpret that {family_name} family."
                        ),
                        character_id=plan.character_id,
                    )
                )
                continue
            if channels and keyed_count == 0:
                baseline_controls.add(control_key)
                baseline_rows.update(channel.row_key for channel in channels)

    if diagnostics:
        return (), tuple(diagnostics)

    for group in plan.owner_groups:
        for channel in group.channels:
            if (
                channel.row_key.family is ChannelFamily.SEMANTIC_STATE
                and channel.control_runtime_key in baseline_controls
            ):
                baseline_rows.add(channel.row_key)
    return tuple(baseline_rows), ()


def _direct_rotate_requires_position(control_context, *, active_only: bool) -> bool:
    controls = tuple(getattr(control_context, "controls", ()) or ())
    active = getattr(control_context, "active", None)
    if active_only and active is not None:
        controls = (active,)
    return any(
        str(getattr(getattr(control, "target", None), "name", "")).upper()
        in {"COM", "PELVIS"}
        for control in controls
    )


def plan_rigped_auto_direct_move(
    scene,
    control_context,
    *,
    operation_id: str,
    selector_character_id: str | None = None,
    active_only: bool = True,
) -> RigpedAutoDirectMovePlanResult:
    """Plan one non-Contact direct Rigped Move using existing direct authority."""

    if not bool(getattr(scene, "baw_auto_key_enabled", False)):
        return RigpedAutoDirectMovePlanResult(None)

    built = build_direct_key_plan(
        scene,
        control_context,
        operation_id=operation_id,
        requested_families=(ChannelFamily.POSITION,),
        selector_character_id=selector_character_id,
        active_only=active_only,
        include_rigped_free_marker=True,
    )
    if not built.ok or built.plan is None:
        return RigpedAutoDirectMovePlanResult(None, built.diagnostics)

    requested_families = (ChannelFamily.POSITION,)
    baseline_rows, diagnostics = _direct_plan_baseline_rows_by_control(
        control_context,
        built.plan,
        families=requested_families,
        operation_id=operation_id,
    )
    if diagnostics:
        return RigpedAutoDirectMovePlanResult(None, diagnostics)

    current_time = float(scene.frame_current) + float(
        getattr(scene, "frame_subframe", 0.0)
    )
    baseline_plan = None
    baseline_time = None
    if baseline_rows and current_time > 0.0:
        baseline_plan = _filter_direct_plan_row_keys(built.plan, baseline_rows)
        baseline_time = 0.0

    return RigpedAutoDirectMovePlanResult(
        RigpedAutoDirectMovePlan(
            begin_plan=built.plan,
            baseline_plan=baseline_plan,
            baseline_time=baseline_time,
            active_only=active_only,
        ),
    )


def commit_rigped_auto_direct_move(
    scene,
    control_context,
    session: RigpedAutoDirectMovePlan,
    *,
    selector_character_id: str | None = None,
    defer_commit: bool = False,
) -> DirectWriterResult:
    """Commit final direct position through the existing phase4 direct writer."""

    built = build_direct_key_plan(
        scene,
        control_context,
        operation_id=session.begin_plan.operation_id,
        requested_families=(ChannelFamily.POSITION,),
        selector_character_id=selector_character_id,
        active_only=session.active_only,
        include_rigped_free_marker=True,
    )
    if not built.ok or built.plan is None:
        return DirectWriterResult(False, 0, 0, built.diagnostics)

    return execute_direct_key_plan(
        scene,
        control_context,
        built.plan,
        trigger=session.trigger,
        auto_enabled=True,
        auto_begin_plan=session.begin_plan,
        auto_baseline_plan=session.baseline_plan,
        auto_baseline_time=session.baseline_time,
        selector_character_id=selector_character_id,
        defer_commit=defer_commit,
    )


def plan_rigped_auto_direct_rotate(
    scene,
    control_context,
    *,
    operation_id: str,
    selector_character_id: str | None = None,
    active_only: bool = True,
) -> RigpedAutoDirectRotatePlanResult:
    """Plan one non-Contact direct Rigped Rotate using existing direct authority."""

    if not bool(getattr(scene, "baw_auto_key_enabled", False)):
        return RigpedAutoDirectRotatePlanResult(None)

    requested_families = (
        (ChannelFamily.ROTATION, ChannelFamily.POSITION)
        if _direct_rotate_requires_position(control_context, active_only=active_only)
        else (ChannelFamily.ROTATION,)
    )
    built = build_direct_key_plan(
        scene,
        control_context,
        operation_id=operation_id,
        requested_families=requested_families,
        selector_character_id=selector_character_id,
        active_only=active_only,
        include_rigped_free_marker=True,
    )
    if not built.ok or built.plan is None:
        return RigpedAutoDirectRotatePlanResult(None, built.diagnostics)

    baseline_rows, diagnostics = _direct_plan_baseline_rows_by_control(
        control_context,
        built.plan,
        families=requested_families,
        operation_id=operation_id,
    )
    if diagnostics:
        return RigpedAutoDirectRotatePlanResult(None, diagnostics)

    current_time = float(scene.frame_current) + float(
        getattr(scene, "frame_subframe", 0.0)
    )
    baseline_plan = None
    baseline_time = None
    if baseline_rows and current_time > 0.0:
        baseline_plan = _filter_direct_plan_row_keys(built.plan, baseline_rows)
        baseline_time = 0.0

    return RigpedAutoDirectRotatePlanResult(
        RigpedAutoDirectRotatePlan(
            begin_plan=built.plan,
            baseline_plan=baseline_plan,
            baseline_time=baseline_time,
            active_only=active_only,
            requested_families=requested_families,
        ),
    )


def commit_rigped_auto_direct_rotate(
    scene,
    control_context,
    session: RigpedAutoDirectRotatePlan,
    *,
    selector_character_id: str | None = None,
    defer_commit: bool = False,
) -> DirectWriterResult:
    """Commit final direct rotation through the existing phase4 direct writer."""

    built = build_direct_key_plan(
        scene,
        control_context,
        operation_id=session.begin_plan.operation_id,
        requested_families=session.requested_families,
        selector_character_id=selector_character_id,
        active_only=session.active_only,
        include_rigped_free_marker=True,
    )
    if not built.ok or built.plan is None:
        return DirectWriterResult(False, 0, 0, built.diagnostics)

    auto_begin_plan = session.begin_plan
    if session.allow_storage_rebind:
        current_groups = {
            group.owner_runtime_key: group for group in built.plan.owner_groups
        }
        rebased_groups = []
        for begin_group in session.begin_plan.owner_groups:
            current_group = current_groups.get(begin_group.owner_runtime_key)
            if current_group is None:
                raise RuntimeError("AUTO direct owner disappeared while rebasing sibling writer storage.")
            current_channels = {
                channel.row_key: channel for channel in current_group.channels
            }
            rebased_channels = []
            for begin_channel in begin_group.channels:
                current_channel = current_channels.get(begin_channel.row_key)
                if current_channel is None:
                    raise RuntimeError("AUTO direct row disappeared while rebasing sibling writer storage.")
                rebased_channels.append(
                    replace(
                        begin_channel,
                        owner_binding_token=current_channel.owner_binding_token,
                        existing_fcurve_token=current_channel.existing_fcurve_token,
                        allocation_intent=current_channel.allocation_intent,
                    )
                )
            rebased_groups.append(
                replace(
                    begin_group,
                    owner_binding_token=current_group.owner_binding_token,
                    channels=tuple(rebased_channels),
                )
            )
        auto_begin_plan = replace(
            session.begin_plan,
            owner_groups=tuple(rebased_groups),
        )

    return execute_direct_key_plan(
        scene,
        control_context,
        built.plan,
        trigger=session.trigger,
        auto_enabled=True,
        auto_begin_plan=auto_begin_plan,
        auto_baseline_plan=session.baseline_plan,
        auto_baseline_time=session.baseline_time,
        selector_character_id=selector_character_id,
        defer_commit=defer_commit,
    )


@dataclass(frozen=True, slots=True)
class RigpedAutoContactBatchPlan:
    """Multiple independent limb domains committed by the existing Contact batch writer."""

    batch: ContactBatchIntentPlan
    baseline_rows_by_mapping: tuple[
        tuple[str, tuple[PlannedChannel, ...]], ...
    ] = ()
    baseline_time: float | None = None
    trigger: WriterTrigger = WriterTrigger.AUTO_TRANSFORM


@dataclass(frozen=True, slots=True)
class RigpedAutoContactBatchPlanResult:
    plan: RigpedAutoContactBatchPlan | None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.diagnostics


def plan_rigped_auto_contact_batch(
    scene,
    control_context,
    *,
    operation_id: str,
    selector_character_id: str | None = None,
    mapping_ids: tuple[str, ...] | None = None,
) -> RigpedAutoContactBatchPlanResult:
    if not bool(getattr(scene, "baw_auto_key_enabled", False)):
        return RigpedAutoContactBatchPlanResult(None)

    built = build_contact_batch_intent_plan(
        scene,
        control_context,
        operation_id=operation_id,
        mode=ContactAuthoringMode.ANCHOR,
        enabled_types=(
            ContactKeyType.FREE,
            ContactKeyType.SLIDING,
            ContactKeyType.PLANTED,
        ),
        selector_character_id=selector_character_id,
        mapping_ids=mapping_ids,
    )
    if not built.ok or built.plan is None:
        return RigpedAutoContactBatchPlanResult(None, built.diagnostics)

    current_time = float(scene.frame_current) + float(
        getattr(scene, "frame_subframe", 0.0)
    )
    baseline_rows: list[tuple[str, tuple[PlannedChannel, ...]]] = []
    for intent in built.plan.intents:
        if intent.target_type is ContactKeyType.PLANTED:
            return RigpedAutoContactBatchPlanResult(
                None,
                (
                    Diagnostic(
                        code="AK_AUTO_MULTI_PLANTED_DEFERRED",
                        severity=DiagnosticSeverity.ERROR,
                        stage=OperationStage.PREFLIGHT,
                        operation=operation_id,
                        detail="Multi-limb Auto does not author Planted authority yet.",
                        character_id=intent.snap_plan.character_id,
                    ),
                ),
            )
        if intent.target_type is ContactKeyType.SLIDING:
            # Sliding already owns a valid Contact/IK authority bundle. AUTO
            # anchors only the current solved bundle; it must not invent a 0F
            # Sliding baseline.
            continue
        if intent.target_type is not ContactKeyType.FREE:
            return RigpedAutoContactBatchPlanResult(
                None,
                (
                    Diagnostic(
                        code="AK_AUTO_MULTI_AUTHORITY_UNSUPPORTED",
                        severity=DiagnosticSeverity.ERROR,
                        stage=OperationStage.PREFLIGHT,
                        operation=operation_id,
                        detail="Multi-limb Auto found an unsupported Contact authority.",
                        character_id=intent.snap_plan.character_id,
                    ),
                ),
            )
        if current_time <= 0.0:
            continue
        try:
            snapshot = snapshot_contact_transform_rows(
                scene,
                control_context,
                intent,
                selector_character_id=selector_character_id,
            )
        except Exception as exc:  # noqa: BLE001 -- snapshot failures become preflight diagnostics
            return RigpedAutoContactBatchPlanResult(
                None,
                (
                    Diagnostic(
                        code="AK_AUTO_MULTI_BASELINE_SNAPSHOT_FAILED",
                        severity=DiagnosticSeverity.ERROR,
                        stage=OperationStage.PREFLIGHT,
                        operation=operation_id,
                        detail=str(exc),
                        character_id=intent.snap_plan.character_id,
                    ),
                ),
            )
        if 0 < snapshot.keyed_row_count < len(snapshot.rows):
            return RigpedAutoContactBatchPlanResult(
                None,
                (
                    Diagnostic(
                        code="AK_AUTO_MULTI_PARTIAL_FREE_BUNDLE",
                        severity=DiagnosticSeverity.ERROR,
                        stage=OperationStage.PREFLIGHT,
                        operation=operation_id,
                        detail=(
                            "Existing Free transform animation is only partially "
                            "authored in one selected limb domain."
                        ),
                        character_id=intent.snap_plan.character_id,
                    ),
                ),
            )
        if snapshot.rows and snapshot.keyed_row_count == 0:
            baseline_rows.append((intent.mapping_id, snapshot.rows))

    return RigpedAutoContactBatchPlanResult(
        RigpedAutoContactBatchPlan(
            batch=built.plan,
            baseline_rows_by_mapping=tuple(baseline_rows),
            baseline_time=0.0 if baseline_rows else None,
        ),
    )


def commit_rigped_auto_contact_batch(
    scene,
    control_context,
    session: RigpedAutoContactBatchPlan,
    *,
    selector_character_id: str | None = None,
    defer_commit: bool = False,
) -> ContactAuthoringResult:
    """Rebuild the final Contact batch after the transform, then persist it.

    The begin-session plan is a guard/baseline snapshot only. Transform operators
    must never persist the stale pre-transform Contact intents themselves.
    """

    built = build_contact_batch_intent_plan(
        scene,
        control_context,
        operation_id=session.batch.operation_id,
        mode=ContactAuthoringMode.ANCHOR,
        enabled_types=(
            ContactKeyType.FREE,
            ContactKeyType.SLIDING,
            ContactKeyType.PLANTED,
        ),
        selector_character_id=selector_character_id,
        mapping_ids=session.batch.mapping_ids,
    )
    if not built.ok or built.plan is None:
        return ContactAuthoringResult(False, diagnostics=built.diagnostics)

    begin_types = {
        intent.mapping_id: intent.target_type
        for intent in session.batch.intents
    }
    final_types = {
        intent.mapping_id: intent.target_type
        for intent in built.plan.intents
    }
    if begin_types != final_types:
        character_id = (
            built.plan.intents[0].snap_plan.character_id
            if built.plan.intents
            else None
        )
        return ContactAuthoringResult(
            False,
            diagnostics=(
                Diagnostic(
                    code="AK_AUTO_CONTACT_AUTHORITY_CHANGED",
                    severity=DiagnosticSeverity.ERROR,
                    stage=OperationStage.PREFLIGHT,
                    operation=session.batch.operation_id,
                    detail=(
                        "Contact authority changed during the transform; "
                        "AUTO refused to reinterpret the gesture."
                    ),
                    character_id=character_id,
                ),
            ),
        )

    return execute_contact_batch_intent_plan(
        scene,
        control_context,
        built.plan,
        trigger=session.trigger,
        auto_baseline_rows_by_mapping=session.baseline_rows_by_mapping,
        auto_baseline_time=session.baseline_time,
        selector_character_id=selector_character_id,
        defer_commit=defer_commit,
    )


@dataclass(frozen=True, slots=True)
class RigpedAutoAnchorPlan:
    """Read-only AK1 ownership wrapper around the existing Contact ANCHOR plan.

    This module intentionally does not resolve Free/Sliding state, build FK/IK
    closure rows, allocate animation storage, or write keys. Those responsibilities
    remain owned by phase4_contact_authoring.
    """

    intent: ContactIntentPlan
    baseline_rows: tuple[PlannedChannel, ...] = ()
    baseline_time: float | None = None
    trigger: WriterTrigger = WriterTrigger.AUTO_TRANSFORM


@dataclass(frozen=True, slots=True)
class RigpedAutoAnchorPlanResult:
    plan: RigpedAutoAnchorPlan | None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.diagnostics


def plan_rigped_auto_anchor(
    scene,
    control_context,
    *,
    operation_id: str,
    mapping_id: str | None = None,
    selector_character_id: str | None = None,
) -> RigpedAutoAnchorPlanResult:
    """Plan one state-preserving Rigped semantic Auto write without mutating data.

    AUTO does not own a second state resolver or closure planner. The existing
    Contact ANCHOR planner is the source of truth for effective/physical state
    and for the semantic domain that a later writer commit will persist.
    """

    if not bool(getattr(scene, "baw_auto_key_enabled", False)):
        return RigpedAutoAnchorPlanResult(None)

    built = build_contact_intent_plan(
        scene,
        control_context,
        operation_id=operation_id,
        mode=ContactAuthoringMode.ANCHOR,
        enabled_types=(
            ContactKeyType.FREE,
            ContactKeyType.SLIDING,
            ContactKeyType.PLANTED,
        ),
        mapping_id=mapping_id,
        selector_character_id=selector_character_id,
    )
    if not built.ok or built.plan is None:
        return RigpedAutoAnchorPlanResult(None, built.diagnostics)

    baseline_rows: tuple[PlannedChannel, ...] = ()
    baseline_time: float | None = None
    current_time = float(scene.frame_current) + float(
        getattr(scene, "frame_subframe", 0.0)
    )
    if built.plan.target_type is ContactKeyType.FREE and current_time > 0.0:
        try:
            snapshot = snapshot_contact_transform_rows(
                scene,
                control_context,
                built.plan,
                selector_character_id=selector_character_id,
            )
        except Exception as exc:  # noqa: BLE001 -- snapshot failures become preflight diagnostics
            return RigpedAutoAnchorPlanResult(
                None,
                (
                    Diagnostic(
                        code="AK_AUTO_BASELINE_SNAPSHOT_FAILED",
                        severity=DiagnosticSeverity.ERROR,
                        stage=OperationStage.PREFLIGHT,
                        operation=operation_id,
                        detail=str(exc),
                        character_id=built.plan.snap_plan.character_id,
                    ),
                ),
            )
        if 0 < snapshot.keyed_row_count < len(snapshot.rows):
            return RigpedAutoAnchorPlanResult(
                None,
                (
                    Diagnostic(
                        code="AK_AUTO_PARTIAL_FREE_BUNDLE",
                        severity=DiagnosticSeverity.ERROR,
                        stage=OperationStage.PREFLIGHT,
                        operation=operation_id,
                        detail=(
                            "Existing Free transform animation is only partially authored; "
                            "AUTO will not repair or reinterpret the semantic bundle."
                        ),
                        character_id=built.plan.snap_plan.character_id,
                    ),
                ),
            )
        if snapshot.rows and snapshot.keyed_row_count == 0:
            baseline_rows = snapshot.rows
            baseline_time = 0.0

    return RigpedAutoAnchorPlanResult(
        RigpedAutoAnchorPlan(
            intent=built.plan,
            baseline_rows=baseline_rows,
            baseline_time=baseline_time,
        ),
    )


def commit_rigped_auto_anchor(
    scene,
    control_context,
    session: RigpedAutoAnchorPlan,
    *,
    selector_character_id: str | None = None,
    defer_commit: bool = False,
) -> ContactAuthoringResult:
    """Persist a single Contact Auto gesture from a fresh post-transform plan.

    The session captured at gesture begin owns only authority/baseline guards.
    Final transform rows and Contact scalar values are rebuilt after the preview
    has succeeded so AUTO cannot replay a stale pre-transform intent.
    """

    built = build_contact_intent_plan(
        scene,
        control_context,
        operation_id=session.intent.operation_id,
        mode=ContactAuthoringMode.ANCHOR,
        enabled_types=(
            ContactKeyType.FREE,
            ContactKeyType.SLIDING,
            ContactKeyType.PLANTED,
        ),
        mapping_id=session.intent.mapping_id,
        selector_character_id=selector_character_id,
    )
    if not built.ok or built.plan is None:
        return ContactAuthoringResult(False, diagnostics=built.diagnostics)

    if built.plan.target_type is not session.intent.target_type:
        return ContactAuthoringResult(
            False,
            diagnostics=(
                Diagnostic(
                    code="AK_AUTO_CONTACT_AUTHORITY_CHANGED",
                    severity=DiagnosticSeverity.ERROR,
                    stage=OperationStage.PREFLIGHT,
                    operation=session.intent.operation_id,
                    detail=(
                        "Contact authority changed during the transform; "
                        "AUTO refused to reinterpret the gesture."
                    ),
                    character_id=built.plan.snap_plan.character_id,
                ),
            ),
        )

    return execute_contact_intent_plan(
        scene,
        control_context,
        built.plan,
        trigger=session.trigger,
        auto_baseline_rows=session.baseline_rows,
        auto_baseline_time=session.baseline_time,
        selector_character_id=selector_character_id,
        defer_commit=defer_commit,
    )
