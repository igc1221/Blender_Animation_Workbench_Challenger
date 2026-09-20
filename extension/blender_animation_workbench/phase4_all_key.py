from __future__ import annotations

from dataclasses import dataclass

from .phase4_contact_authoring import (
    ContactAuthoringMode,
    ContactAuthoringResult,
    build_contact_batch_intent_plan,
    build_contact_intent_plan,
    execute_contact_batch_intent_plan,
    execute_contact_intent_plan,
    selected_contact_mapping_ids,
    selected_contact_owner_binding_ids,
)
from .phase4_operation_plan import OperationPlan
from .phase4_preflight import build_all_key_plan
from .phase4_verification import Diagnostic, NoopStageHook, StageHook
from .phase4_writer import DirectWriterResult, WriterTrigger, execute_direct_key_plan
from .rigped_contract import resolve_rigped_target


@dataclass(frozen=True, slots=True)
class AllKeyResult:
    applied: bool
    plan: OperationPlan | None
    writer_result: DirectWriterResult | None
    diagnostics: tuple[Diagnostic, ...] = ()
    contact_result: ContactAuthoringResult | None = None


def execute_all_key(
    scene,
    control_context,
    *,
    operation_id: str,
    selector_character_id: str | None = None,
    hook: StageHook | None = None,
) -> AllKeyResult:
    """Plan and author one complete direct-control P/R Rigped pose anchor.

    All Key is always explicit authoring. AUTO is deliberately not part of
    this API. The planner resolves the current Rigped from fresh native
    selection, then expands to every supported authored control while preserving
    the original selection in the immutable freshness footprint.
    """

    build = build_all_key_plan(
        scene,
        control_context,
        operation_id=operation_id,
        selector_character_id=selector_character_id,
    )
    if not build.ok or build.plan is None:
        return AllKeyResult(False, None, None, build.diagnostics)

    target_resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    target = target_resolution.target
    contact_owner_ids = selected_contact_owner_binding_ids(target) if target is not None else ()
    if contact_owner_ids:
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
                return AllKeyResult(False, build.plan, None, batch_build.diagnostics)
            contact = execute_contact_batch_intent_plan(
                scene,
                control_context,
                batch_build.plan,
                closure_plan=build.plan,
                selector_character_id=selector_character_id,
                hook=hook or NoopStageHook(),
            )
        else:
            contact_build = build_contact_intent_plan(
                scene,
                control_context,
                operation_id=operation_id,
                mode=ContactAuthoringMode.ANCHOR,
                selector_character_id=selector_character_id,
            )
            if not contact_build.ok or contact_build.plan is None:
                return AllKeyResult(False, build.plan, None, contact_build.diagnostics)
            contact = execute_contact_intent_plan(
                scene,
                control_context,
                contact_build.plan,
                closure_plan=build.plan,
                selector_character_id=selector_character_id,
                hook=hook or NoopStageHook(),
            )
        return AllKeyResult(
            applied=contact.applied,
            plan=build.plan,
            writer_result=None,
            diagnostics=contact.diagnostics,
            contact_result=contact,
        )

    writer = execute_direct_key_plan(
        scene,
        control_context,
        build.plan,
        trigger=WriterTrigger.EXPLICIT_KEY,
        auto_enabled=False,
        selector_character_id=selector_character_id,
        hook=hook or NoopStageHook(),
    )
    return AllKeyResult(
        applied=writer.applied,
        plan=build.plan,
        writer_result=writer,
        diagnostics=writer.diagnostics,
    )
