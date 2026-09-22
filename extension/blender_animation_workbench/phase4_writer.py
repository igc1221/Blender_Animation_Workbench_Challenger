from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import bpy

from .debug_trace import trace_event, trace_exception
from .phase4_mutation_journal import (
    CreatedActionReceipt,
    CreatedAnimationDataReceipt,
    CreatedChannelBagReceipt,
    CreatedFCurveReceipt,
    CreatedSlotReceipt,
    FCurveMutationReceipt,
    FCurveSnapshot,
    IDPropertyMutationReceipt,
    KeyMutationReceipt,
    KeySnapshot,
    MutationJournal,
    QuarantineKey,
    is_quarantined,
)
from .phase4_mutation_journal_blender import BlenderRollbackExecutor
from .phase4_operation_plan import (
    AllocationIntent,
    ChannelFamily,
    OperationPlan,
    OperationType,
    PlannedChannel,
    PlannedOwnerGroup,
)
from .phase4_preflight import validate_plan_fresh
from .phase4_verification import (
    Diagnostic,
    DiagnosticSeverity,
    NoopStageHook,
    OperationStage,
    StageHook,
)
from .semantic_adapter import assigned_channelbag, channel_binding_token


class WriterTrigger(StrEnum):
    EXPLICIT_KEY = "EXPLICIT_KEY"
    CONTACT_AUTHORING = "CONTACT_AUTHORING"
    AUTO_TRANSFORM = "AUTO_TRANSFORM"


class DirectWriterError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DirectWriterResult:
    applied: bool
    rows_written: int
    created_fcurves: int
    diagnostics: tuple[Diagnostic, ...] = ()
    pending_journal: MutationJournal | None = None
    operation_id: str | None = None


_COMMITTED_OPERATION_IDS: set[str] = set()


def clear_committed_operation_ids_for_file_lifecycle() -> None:
    _COMMITTED_OPERATION_IDS.clear()


def finalize_deferred_direct_writer_result(result: DirectWriterResult) -> None:
    """Finalize one already-verified deferred direct writer transaction."""

    journal = result.pending_journal
    if not result.applied or journal is None or result.operation_id is None:
        raise DirectWriterError("Deferred direct writer result is not commit-ready.")
    if journal.state.value != "OPEN":
        raise DirectWriterError(
            f"Deferred direct writer journal is not open: {journal.state.value}."
        )
    journal.commit()
    _COMMITTED_OPERATION_IDS.add(result.operation_id)
    trace_event(
        "WRITER",
        "DIRECT_WRITE_COMMIT",
        operation_id=result.operation_id,
        context=bpy.context,
        deferred=True,
        rows_written=result.rows_written,
        created_fcurves=result.created_fcurves,
    )


def _pointer(value: Any) -> int | None:
    if value is None:
        return None
    as_pointer = getattr(value, "as_pointer", None)
    if not callable(as_pointer):
        return None
    try:
        result = int(as_pointer())
    except ReferenceError:
        return None
    return result or None


def _find_object(pointer: int):
    return next((obj for obj in bpy.data.objects if _pointer(obj) == pointer), None)


def _find_control_target(owner, pointer: int):
    if _pointer(owner) == pointer:
        return owner
    pose = getattr(owner, "pose", None)
    if pose is not None:
        return next(
            (
                bone
                for bone in getattr(pose, "bones", ())
                if _pointer(bone) == pointer
            ),
            None,
        )
    return None


def _semantic_property_name(channel: PlannedChannel) -> str:
    path = str(channel.row_key.data_path)
    marker = '["'
    start = path.rfind(marker)
    if start < 0 or not path.endswith('"]'):
        raise DirectWriterError(
            f"Semantic-state row has unsupported data path: {path!r}"
        )
    return path[start + len(marker) : -2]


def _prepare_semantic_state_property(
    owner,
    channel: PlannedChannel,
    plan: OperationPlan,
    journal: MutationJournal,
    stage_write,
    prepared: set[tuple[int, str]],
) -> None:
    if getattr(channel.row_key, "family", None) is not ChannelFamily.SEMANTIC_STATE:
        return
    target_ptr = int(channel.control_runtime_key[1])
    property_name = _semantic_property_name(channel)
    key = (target_ptr, property_name)
    if key in prepared:
        return
    target = _find_control_target(owner, target_ptr)
    if target is None:
        raise DirectWriterError("Semantic-state target no longer exists.")
    before_exists = property_name in target
    before_value = target.get(property_name) if before_exists else None
    journal.record(
        IDPropertyMutationReceipt(
            journal.next_ordinal(),
            _scope(plan, owner_ptr=_pointer(owner), bag_ptr=None),
            int(_pointer(owner)),
            target_ptr,
            property_name,
            bool(before_exists),
            before_value,
        )
    )
    stage_write(f"set semantic property {property_name}")
    target[property_name] = round(channel.target_value)
    prepared.add(key)

def _find_fcurve(channelbag, channel: PlannedChannel):
    if channelbag is None:
        return None
    curves = getattr(channelbag, "fcurves", None)
    if curves is None:
        return None
    finder = getattr(curves, "find", None)
    if callable(finder):
        try:
            found = finder(channel.row_key.data_path, index=channel.row_key.array_index)
        except TypeError:
            found = None
        if found is not None:
            return found
    return next(
        (
            curve
            for curve in curves
            if str(curve.data_path) == channel.row_key.data_path
            and int(curve.array_index) == channel.row_key.array_index
        ),
        None,
    )


def _key_at_frame(fcurve, frame: float):
    matches = tuple(
        point
        for point in fcurve.keyframe_points
        if float(point.co.x) == float(frame)
    )
    if len(matches) > 1:
        raise DirectWriterError(f"Multiple keys exist at exact write time {frame!r}.")
    return matches[0] if matches else None


def _capture_key_snapshot(point) -> KeySnapshot:
    if point is None:
        return KeySnapshot(False)
    return KeySnapshot(
        True,
        co=(float(point.co.x), float(point.co.y)),
        handle_left=(float(point.handle_left.x), float(point.handle_left.y)),
        handle_right=(float(point.handle_right.x), float(point.handle_right.y)),
        handle_left_type=str(point.handle_left_type),
        handle_right_type=str(point.handle_right_type),
        interpolation=str(point.interpolation),
        easing=str(point.easing),
        back=float(point.back),
        amplitude=float(point.amplitude),
        period=float(point.period),
        keyframe_type=str(point.type),
        select_control_point=bool(point.select_control_point),
        select_left_handle=bool(point.select_left_handle),
        select_right_handle=bool(point.select_right_handle),
    )


def _capture_fcurve_snapshot(fcurve) -> FCurveSnapshot:
    group = getattr(fcurve, "group", None)
    return FCurveSnapshot(
        True,
        data_path=str(fcurve.data_path),
        array_index=int(fcurve.array_index),
        group_name=str(group.name) if group is not None else None,
        extrapolation=str(fcurve.extrapolation),
        keys=tuple(_capture_key_snapshot(point) for point in fcurve.keyframe_points),
    )


def _scope(plan: OperationPlan, *, owner_ptr: int | None, bag_ptr: int | None) -> QuarantineKey:
    return QuarantineKey(
        plan.character_id,
        plan.setup_revision,
        plan.setup_signature,
        owner_ptr,
        bag_ptr,
    )


def _check_scope_not_quarantined(plan: OperationPlan, group: PlannedOwnerGroup) -> None:
    owner_scope = _scope(plan, owner_ptr=group.owner_runtime_key, bag_ptr=group.owner_binding_token[3])
    broad_scope = _scope(plan, owner_ptr=group.owner_runtime_key, bag_ptr=None)
    if is_quarantined(owner_scope) or is_quarantined(broad_scope):
        raise DirectWriterError(
            "I5 mutation scope is quarantined after an incomplete rollback; explicit recovery is required."
        )


def _record_created_animdata(
    journal: MutationJournal,
    plan: OperationPlan,
    owner,
    before_token,
) -> None:
    journal.record(
        CreatedAnimationDataReceipt(
            journal.next_ordinal(),
            _scope(plan, owner_ptr=_pointer(owner), bag_ptr=None),
            int(_pointer(owner)),
            before_token,
            channel_binding_token(owner),
        )
    )


def _record_created_action(
    journal: MutationJournal,
    plan: OperationPlan,
    owner,
    action,
    before_token,
) -> None:
    owner_ptr = int(_pointer(owner))
    action_ptr = int(_pointer(action))
    journal.record(
        CreatedActionReceipt(
            journal.next_ordinal(),
            _scope(plan, owner_ptr=owner_ptr, bag_ptr=None),
            owner_ptr,
            action_ptr,
            before_token,
            (owner_ptr, action_ptr, None, None),
        )
    )


def _record_created_slot(
    journal: MutationJournal,
    plan: OperationPlan,
    owner,
    action,
    slot,
    before_token,
) -> None:
    owner_ptr = int(_pointer(owner))
    action_ptr = int(_pointer(action))
    slot_ptr = int(_pointer(slot))
    journal.record(
        CreatedSlotReceipt(
            journal.next_ordinal(),
            _scope(plan, owner_ptr=owner_ptr, bag_ptr=None),
            owner_ptr,
            action_ptr,
            slot_ptr,
            before_token,
            (owner_ptr, action_ptr, slot_ptr, None),
        )
    )


def _record_created_bag(
    journal: MutationJournal,
    plan: OperationPlan,
    owner,
    action,
    slot,
    bag,
    before_token,
) -> None:
    journal.record(
        CreatedChannelBagReceipt(
            journal.next_ordinal(),
            _scope(plan, owner_ptr=_pointer(owner), bag_ptr=_pointer(bag)),
            int(_pointer(owner)),
            int(_pointer(action)),
            int(_pointer(slot)),
            int(_pointer(bag)),
            before_token,
            channel_binding_token(owner),
        )
    )


def _ensure_owner_storage(
    owner,
    group: PlannedOwnerGroup,
    plan: OperationPlan,
    journal: MutationJournal,
    stage_write,
):
    current_bag = assigned_channelbag(owner)
    if current_bag is not None:
        return current_bag

    animation_data = getattr(owner, "animation_data", None)
    if animation_data is not None and getattr(animation_data, "action", None) is not None:
        raise DirectWriterError(
            "I5 first slice does not mutate a pre-existing assigned Action that has no ChannelBag."
        )

    if any(
        channel.allocation_intent is AllocationIntent.NEED_ASSIGNED_BAG
        for channel in group.channels
    ):
        raise DirectWriterError(
            "I5 NEED_ASSIGNED_BAG is intentionally fail-closed until layer/strip ownership rollback is proven."
        )

    if animation_data is None:
        before = channel_binding_token(owner)
        stage_write("create animation_data")
        animation_data = owner.animation_data_create()
        if animation_data is None:
            raise DirectWriterError("Blender could not create AnimData for the planned owner.")
        _record_created_animdata(journal, plan, owner, before)

    before_action = channel_binding_token(owner)
    stage_write("create Action")
    action = bpy.data.actions.new(name=f"AWB {owner.name} Action")
    _record_created_action(journal, plan, owner, action, before_action)
    animation_data.action = action

    before_slot = channel_binding_token(owner)
    stage_write("create Action Slot")
    slot = action.slots.new(owner.id_type, owner.name)
    _record_created_slot(journal, plan, owner, action, slot, before_slot)
    animation_data.action_slot = slot

    stage_write("create Action layer/strip/ChannelBag")
    layer = action.layers.new("AWB")
    strip = layer.strips.new(type="KEYFRAME")
    before_bag = channel_binding_token(owner)
    bag = strip.channelbag(slot, ensure=True)
    if bag is None:
        raise DirectWriterError("Blender could not create the assigned ChannelBag.")
    _record_created_bag(journal, plan, owner, action, slot, bag, before_bag)
    return bag


def _ensure_fcurve(
    owner,
    bag,
    channel: PlannedChannel,
    plan: OperationPlan,
    journal: MutationJournal,
    stage_write,
):
    fcurve = _find_fcurve(bag, channel)
    if fcurve is not None:
        if (
            channel.allocation_intent is AllocationIntent.EXISTING_FCURVE
            and _pointer(fcurve) != channel.existing_fcurve_token
        ):
            raise DirectWriterError("Existing FCurve identity changed after freshness validation.")
        return fcurve, False
    if channel.allocation_intent is AllocationIntent.EXISTING_FCURVE:
        raise DirectWriterError("Planned existing FCurve disappeared before write.")
    stage_write(f"create FCurve {channel.row_key.data_path}[{channel.row_key.array_index}]")
    fcurve = bag.fcurves.new(
        channel.row_key.data_path,
        index=channel.row_key.array_index,
    )
    journal.record(
        CreatedFCurveReceipt(
            journal.next_ordinal(),
            _scope(plan, owner_ptr=_pointer(owner), bag_ptr=_pointer(bag)),
            int(_pointer(bag)),
            int(_pointer(fcurve)),
            channel.row_key.data_path,
            channel.row_key.array_index,
        )
    )
    return fcurve, True


def _write_channel_key(
    owner,
    bag,
    fcurve,
    channel: PlannedChannel,
    time: float,
    plan: OperationPlan,
    journal: MutationJournal,
    stage_write,
) -> None:
    existing = _key_at_frame(fcurve, time)
    if channel.allocation_intent is AllocationIntent.EXISTING_FCURVE:
        journal.record(
            FCurveMutationReceipt(
                journal.next_ordinal(),
                _scope(plan, owner_ptr=_pointer(owner), bag_ptr=_pointer(bag)),
                int(_pointer(bag)),
                int(_pointer(fcurve)),
                _capture_fcurve_snapshot(fcurve),
            )
        )
    before = _capture_key_snapshot(existing)
    journal.record(
        KeyMutationReceipt(
            journal.next_ordinal(),
            _scope(plan, owner_ptr=_pointer(owner), bag_ptr=_pointer(bag)),
            int(_pointer(bag)),
            int(_pointer(fcurve)),
            time,
            before,
        )
    )
    stage_write(f"write key {channel.row_key.data_path}[{channel.row_key.array_index}] @ {time}")
    if existing is None:
        point = fcurve.keyframe_points.insert(
            time,
            channel.target_value,
            options={"FAST"},
            keyframe_type="KEYFRAME",
        )
        if point is None:
            raise DirectWriterError("Blender failed to insert the planned key.")
    else:
        delta = channel.target_value - float(existing.co.y)
        existing.co.y = channel.target_value
        existing.handle_left.y = float(existing.handle_left.y) + delta
        existing.handle_right.y = float(existing.handle_right.y) + delta
        point = existing
    if getattr(channel.row_key, "family", None) is ChannelFamily.SEMANTIC_STATE:
        point.interpolation = "CONSTANT"
    fcurve.update()


def _no_auto_diagnostic(plan: OperationPlan) -> Diagnostic:
    return Diagnostic(
        code="I5_AUTO_GATE_OFF",
        severity=DiagnosticSeverity.INFO,
        stage=OperationStage.PREFLIGHT,
        operation=plan.operation_id,
        detail="AUTO transform-time preference is disabled; no semantic writer mutation was performed.",
        character_id=plan.character_id,
    )


def execute_direct_key_plan(
    scene,
    control_context,
    plan: OperationPlan,
    *,
    trigger: WriterTrigger = WriterTrigger.EXPLICIT_KEY,
    auto_enabled: bool = False,
    auto_begin_plan: OperationPlan | None = None,
    auto_baseline_plan: OperationPlan | None = None,
    auto_baseline_time: float | None = None,
    selector_character_id: str | None = None,
    hook: StageHook | None = None,
    defer_commit: bool = False,
) -> DirectWriterResult:
    if plan.operation_type not in {OperationType.DIRECT_KEY, OperationType.ALL_KEY}:
        raise DirectWriterError("I5/I8 P/R writer accepts only DIRECT_KEY or ALL_KEY plans.")
    if not plan.owner_groups or not plan.write_footprint.row_keys:
        raise DirectWriterError("I5 direct writer refuses an empty write plan.")
    if plan.operation_id in _COMMITTED_OPERATION_IDS:
        raise DirectWriterError("This semantic operation_id has already committed in the current session.")
    trace_event(
        "WRITER",
        "DIRECT_WRITE_BEGIN",
        operation_id=plan.operation_id,
        context=bpy.context,
        trigger=trigger.value,
        operation_type=plan.operation_type.value,
        frame=float(plan.frame) + float(plan.subframe),
        row_count=len(plan.write_footprint.row_keys),
        auto_baseline=auto_baseline_plan is not None,
        auto_baseline_time=auto_baseline_time,
    )
    if (
        plan.operation_type is OperationType.DIRECT_KEY
        and trigger is WriterTrigger.AUTO_TRANSFORM
        and not auto_enabled
    ):
        return DirectWriterResult(False, 0, 0, (_no_auto_diagnostic(plan),))

    hook = hook or NoopStageHook()
    hook.enter(OperationStage.PREFLIGHT, operation=plan.operation_id)
    freshness = validate_plan_fresh(
        scene,
        control_context,
        plan,
        selector_character_id=selector_character_id,
    )
    if not freshness.ok:
        return DirectWriterResult(False, 0, 0, freshness.diagnostics)

    auto_guard_plan = auto_begin_plan or auto_baseline_plan
    if auto_guard_plan is not None:
        if trigger is not WriterTrigger.AUTO_TRANSFORM:
            raise DirectWriterError("AUTO begin plan requires AUTO_TRANSFORM trigger.")
        if auto_guard_plan.operation_type is not OperationType.DIRECT_KEY:
            raise DirectWriterError("AUTO begin plan accepts only DIRECT_KEY plans.")
        if (
            auto_guard_plan.character_id != plan.character_id
            or auto_guard_plan.setup_revision != plan.setup_revision
            or auto_guard_plan.setup_signature != plan.setup_signature
            or auto_guard_plan.selected_binding_ids != plan.selected_binding_ids
            or auto_guard_plan.active_binding_id != plan.active_binding_id
            or auto_guard_plan.write_footprint.row_keys != plan.write_footprint.row_keys
        ):
            raise DirectWriterError("AUTO direct footprint changed during the gesture.")
        baseline_channels = {
            channel.row_key: channel
            for group in auto_guard_plan.owner_groups
            for channel in group.channels
        }
        current_channels = {
            channel.row_key: channel
            for group in plan.owner_groups
            for channel in group.channels
        }
        for row_key, baseline in baseline_channels.items():
            current = current_channels.get(row_key)
            if current is None:
                raise DirectWriterError("AUTO baseline row disappeared during the gesture.")
            if (
                baseline.owner_runtime_key != current.owner_runtime_key
                or baseline.control_runtime_key != current.control_runtime_key
                or baseline.owner_binding_token != current.owner_binding_token
                or baseline.existing_fcurve_token != current.existing_fcurve_token
                or baseline.allocation_intent is not current.allocation_intent
            ):
                raise DirectWriterError("AUTO baseline storage identity changed during the gesture.")

    if auto_baseline_plan is not None and auto_baseline_time is None:
        raise DirectWriterError("AUTO baseline time is required.")

    for group in plan.owner_groups:
        _check_scope_not_quarantined(plan, group)

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

    rows_written = 0
    created_fcurves = 0
    semantic_properties_prepared: set[tuple[int, str]] = set()
    time = float(plan.frame) + float(plan.subframe)
    try:
        for group in plan.owner_groups:
            owner = _find_object(group.owner_runtime_key)
            if owner is None:
                raise DirectWriterError("Planned animation owner no longer exists.")
            if channel_binding_token(owner) != group.owner_binding_token:
                raise DirectWriterError("Owner animation binding changed after freshness validation.")
            bag = _ensure_owner_storage(owner, group, plan, journal, stage_write)
            if auto_baseline_plan is not None:
                assert auto_baseline_time is not None
                baseline_group = next(
                    (
                        candidate
                        for candidate in auto_baseline_plan.owner_groups
                        if candidate.owner_runtime_key == group.owner_runtime_key
                    ),
                    None,
                )
                if baseline_group is None:
                    raise DirectWriterError("AUTO baseline owner group disappeared.")
                current_by_key = {channel.row_key: channel for channel in group.channels}
                for baseline in baseline_group.channels:
                    current = current_by_key.get(baseline.row_key)
                    if current is None:
                        raise DirectWriterError("AUTO baseline row disappeared.")
                    _prepare_semantic_state_property(
                        owner,
                        current,
                        plan,
                        journal,
                        stage_write,
                        semantic_properties_prepared,
                    )
                    fcurve, created = _ensure_fcurve(
                        owner,
                        bag,
                        current,
                        plan,
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
                        plan,
                        journal,
                        stage_write,
                    )
                    rows_written += 1
            for channel in group.channels:
                _prepare_semantic_state_property(
                    owner,
                    channel,
                    plan,
                    journal,
                    stage_write,
                    semantic_properties_prepared,
                )
                fcurve, created = _ensure_fcurve(
                    owner,
                    bag,
                    channel,
                    plan,
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
                    plan,
                    journal,
                    stage_write,
                )
                rows_written += 1

        hook.enter(OperationStage.REEVALUATE, operation=plan.operation_id)
        bpy.context.view_layer.update()
        hook.enter(OperationStage.VERIFY, operation=plan.operation_id)
        expected_rows = len(plan.write_footprint.row_keys)
        if auto_baseline_plan is not None:
            expected_rows += len(auto_baseline_plan.write_footprint.row_keys)
        if rows_written != expected_rows:
            raise DirectWriterError(
                f"Writer row count mismatch: wrote {rows_written}, planned {expected_rows}."
            )
        if defer_commit:
            trace_event(
                "WRITER",
                "DIRECT_WRITE_PREPARED",
                operation_id=plan.operation_id,
                context=bpy.context,
                trigger=trigger.value,
                rows_written=rows_written,
                created_fcurves=created_fcurves,
            )
            return DirectWriterResult(
                True,
                rows_written,
                created_fcurves,
                pending_journal=journal,
                operation_id=plan.operation_id,
            )

        journal.commit()
        _COMMITTED_OPERATION_IDS.add(plan.operation_id)
        hook.enter(OperationStage.COMMIT, operation=plan.operation_id)
        trace_event(
            "WRITER",
            "DIRECT_WRITE_COMMIT",
            operation_id=plan.operation_id,
            context=bpy.context,
            trigger=trigger.value,
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        return DirectWriterResult(
            True,
            rows_written,
            created_fcurves,
            operation_id=plan.operation_id,
        )
    except Exception as exc:
        trace_exception(
            "WRITER",
            "DIRECT_WRITE_FAIL",
            exc,
            operation_id=plan.operation_id,
            context=bpy.context,
            trigger=trigger.value,
            rows_written=rows_written,
            created_fcurves=created_fcurves,
        )
        hook.enter(
            OperationStage.ROLLBACK_WRITE,
            operation=plan.operation_id,
            detail=f"rollback after {type(exc).__name__}",
        )
        journal.rollback_or_raise()
        hook.enter(OperationStage.ROLLBACK_VERIFY, operation=plan.operation_id)
        raise
