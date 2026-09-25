from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import bpy
from mathutils import Matrix, Quaternion, Vector

from .debug_trace import (
    capture_debug_state_checkpoint,
    read_recent_trace_events,
    replay_execution_active,
    replay_path,
    set_replay_execution_active,
    trace_scrub_begin,
    trace_scrub_end,
)
from .phase4_contact_authoring import (
    ContactAuthoringMode,
    build_contact_batch_intent_plan,
    build_contact_intent_plan,
    execute_contact_batch_intent_plan,
    execute_contact_command,
    execute_contact_intent_plan,
    selected_contact_mapping_ids,
)
from .phase4_contact_model import (
    AWB_CONTACT_STATE_PROPERTY,
    ContactKeyType,
    ContactPlantSpace,
    type_for_state_value,
)
from .phase4_contact_ui import (
    contact_enabled_types,
    contact_plant_space,
)
from .rigped_auto_key import (
    commit_rigped_auto_anchor,
    commit_rigped_auto_contact_batch,
    commit_rigped_auto_direct_move,
    commit_rigped_auto_direct_rotate,
    plan_rigped_auto_anchor,
    plan_rigped_auto_contact_batch,
    plan_rigped_auto_direct_move,
    plan_rigped_auto_direct_rotate,
)
from .rigped_operation_domain import resolve_operation_domain
from .rigped_transform import (
    _RIGPED_SLIDING_REPLAY_LIMBS,
    _apply_control_state,
    _apply_direct_move_delta,
    _apply_direct_rotate_sliding_sync,
    _begin_direct_move_states,
    _begin_semantic_move_domains,
    _capture_sliding_dependency_guards,
    _control_pivot_world,
    _current_sliding_capabilities,
    _direct_rotate_auto_contexts,
    _direct_rotate_sliding_sync_sessions,
    _fk_move_auto_contexts,
    _passive_sliding_capabilities,
    _refresh_current_sliding_public_overlays,
    _resolved_control_role_name,
    _selected_direct_rotate_controls,
    _sliding_capabilities_for_character,
    _state_for_pose_matrix,
    _uses_center_pivot,
    apply_fk_joint_moves_delta,
    apply_semantic_move_domains_delta,
    begin_fk_joint_moves,
    cancel_fk_joint_moves,
    cancel_semantic_move_domains,
    commit_fk_joint_moves,
)
from .semantic_adapter import control_context_for_context
from .trackbar_keying import set_awb_auto_key


class ReplayMode(StrEnum):
    RECORDED_RESULT = "RECORDED_RESULT"
    COMMAND = "COMMAND"


_GUI_REPLAY_SCHEMA = "awb-gui-replay/v1"
_GUI_REPLAY_RESULT_SCHEMA = "awb-gui-replay-result/v1"
_GUI_REPLAY_TICKET_SCHEMA = "awb-gui-replay-ticket/v1"
_GUI_REPLAY_ALLOWED_KEYS = {"W", "E", "R"}
_GUI_REPLAY_ALLOWED_VALUES = {"PRESS", "RELEASE"}
_GUI_REPLAY_DEFAULT_TIMEOUT_SECONDS = 60.0
_GUI_REPLAY_NAMESPACE_KEY = "_awb_gui_replay_execution_id"

_RECORDED_CONTACT_TYPES = tuple(ContactKeyType)


def _load_replay_script(path: str | None = None) -> dict[str, Any]:
    target = Path(path or replay_path())
    if not target.exists():
        raise RuntimeError(f"AWB replay script does not exist: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "awb-semantic-replay/v1":
        raise RuntimeError("AWB replay script schema is unsupported.")
    return data


def _active_armature(context):
    active = getattr(context, "active_object", None)
    if active is not None and getattr(active, "type", None) == "ARMATURE":
        return active
    scene = getattr(context, "scene", None)
    if scene is None:
        return None
    return next(
        (
            obj
            for obj in tuple(getattr(scene, "objects", ()) or ())
            if getattr(obj, "type", None) == "ARMATURE"
            and getattr(obj, "pose", None) is not None
        ),
        None,
    )


def _select_pose_controls(context, names: tuple[str, ...]) -> None:
    rig = _active_armature(context)
    if rig is None or getattr(rig, "pose", None) is None:
        return
    if getattr(context, "active_object", None) is not rig:
        context.view_layer.objects.active = rig
    if getattr(context, "mode", "") != "POSE":
        try:
            bpy.ops.object.mode_set(mode="POSE")
        except RuntimeError:
            return

    requested = tuple(str(name) for name in names if str(name))
    for pose_bone in rig.pose.bones:
        pose_bone.select = False
    active_data_bone = None
    for name in requested:
        pose_bone = rig.pose.bones.get(name)
        if pose_bone is None:
            continue
        pose_bone.select = True
        if active_data_bone is None:
            active_data_bone = pose_bone.bone
    if active_data_bone is not None:
        rig.data.bones.active = active_data_bone


def _quat_delta_deg(first: Quaternion, second: Quaternion) -> float:
    return float(first.rotation_difference(second).angle) * 57.29577951308232


def _sliding_divergences(context) -> list[dict[str, Any]]:
    scene = getattr(context, "scene", None)
    if scene is None:
        return []
    rows: list[dict[str, Any]] = []
    for rig in tuple(getattr(scene, "objects", ()) or ()):
        if getattr(rig, "type", None) != "ARMATURE" or getattr(rig, "pose", None) is None:
            continue
        pose = rig.pose.bones
        for state_name, public_names, result_names in _RIGPED_SLIDING_REPLAY_LIMBS:
            state_bone = pose.get(state_name)
            if state_bone is None or AWB_CONTACT_STATE_PROPERTY not in state_bone:
                continue
            try:
                contact_type = type_for_state_value(float(state_bone[AWB_CONTACT_STATE_PROPERTY]))
            except (TypeError, ValueError):
                continue
            if contact_type is not ContactKeyType.SLIDING:
                continue

            for public_name, result_name in zip(public_names, result_names, strict=True):
                public_bone = pose.get(public_name)
                result_bone = pose.get(result_name)
                if public_bone is None or result_bone is None:
                    continue
                public_world = rig.matrix_world @ public_bone.matrix
                result_world = rig.matrix_world @ result_bone.matrix
                rotation_error_deg = _quat_delta_deg(
                    public_world.to_quaternion().normalized(),
                    result_world.to_quaternion().normalized(),
                )
                location_error = float(
                    (Vector(public_world.to_translation()) - Vector(result_world.to_translation())).length
                )
                rows.append(
                    {
                        "rig": str(rig.name),
                        "state_bone": str(state_name),
                        "public_bone": str(public_name),
                        "result_bone": str(result_name),
                        "rotation_error_deg": rotation_error_deg,
                        "location_error": location_error,
                    }
                )
    return rows


def _selected_pose_world_snapshot(context) -> dict[str, tuple[float, float, float, float]]:
    rig = _active_armature(context)
    if rig is None or getattr(rig, "pose", None) is None:
        return {}
    result: dict[str, tuple[float, float, float, float]] = {}
    for pose_bone in tuple(getattr(context, "selected_pose_bones", ()) or ()):
        world = rig.matrix_world @ pose_bone.matrix
        quat = world.to_quaternion().normalized()
        result[str(pose_bone.name)] = tuple(float(value) for value in quat)
    return result


def _execute_auto_key_action(context, action: dict[str, Any]) -> dict[str, Any]:
    scene = context.scene
    scene.baw_key_position = bool(action.get("position", True))
    scene.baw_key_rotation = bool(action.get("rotation", True))
    scene.baw_key_scale = bool(action.get("scale", True))
    enabled = bool(action.get("enabled", False))
    if not set_awb_auto_key(context, enabled):
        raise RuntimeError(f"AWB semantic replay could not set Auto Key to {enabled}.")
    return {"kind": "AUTO_KEY", "enabled": bool(scene.baw_auto_key_enabled)}


def _prepare_contact_action_context(
    context,
    action: dict[str, Any],
) -> tuple[Any, int, tuple[str, ...]]:
    scene = context.scene
    frame = int(action.get("frame", scene.frame_current))
    subframe = float(action.get("subframe", 0.0))
    scene.frame_set(frame, subframe=subframe)
    controls = tuple(str(name) for name in tuple(action.get("controls") or ()) if str(name))
    if controls:
        _select_pose_controls(context, controls)
    return scene, frame, controls


def _recorded_contact_type(value: Any, *, field_name: str) -> ContactKeyType | None:
    if value is None:
        return None
    try:
        return ContactKeyType(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"AWB semantic replay Contact INVALID_RECORDED_RESULT: "
            f"{field_name}={value!r} is not a valid Contact type."
        ) from exc


def _recorded_mapping_contact_types(
    action: dict[str, Any],
) -> tuple[tuple[str, ContactKeyType], ...] | None:
    raw = action.get("mapping_contact_types")
    if raw is None:
        return None
    if isinstance(raw, dict):
        items = tuple(raw.items())
    else:
        try:
            raw_items = tuple(raw)
        except TypeError as exc:
            raise RuntimeError(
                "AWB semantic replay Contact INVALID_RECORDED_RESULT: "
                "mapping_contact_types is not an iterable mapping result."
            ) from exc
        if any(
            not isinstance(item, (list, tuple)) or len(item) != 2
            for item in raw_items
        ):
            raise RuntimeError(
                "AWB semantic replay Contact INVALID_RECORDED_RESULT: "
                "mapping_contact_types must contain exact (mapping_id, contact_type) pairs."
            )
        items = tuple((item[0], item[1]) for item in raw_items)

    normalized: list[tuple[str, ContactKeyType]] = []
    seen: set[str] = set()
    for mapping_id, contact_type in items:
        normalized_id = str(mapping_id)
        if not normalized_id or normalized_id in seen:
            raise RuntimeError(
                "AWB semantic replay Contact INVALID_RECORDED_RESULT: "
                f"mapping_contact_types has a missing/duplicate mapping id {normalized_id!r}."
            )
        seen.add(normalized_id)
        normalized.append(
            (
                normalized_id,
                _recorded_contact_type(
                    contact_type,
                    field_name=f"mapping_contact_types[{normalized_id!r}]",
                ),
            )
        )
    return tuple(
        (mapping_id, contact_type)
        for mapping_id, contact_type in normalized
        if contact_type is not None
    )


def _recorded_direct_binding_ids(action: dict[str, Any]) -> tuple[str, ...] | None:
    raw = action.get("direct_binding_ids")
    if raw is None:
        return None
    if isinstance(raw, str):
        raise TypeError(
            "AWB semantic replay Contact INVALID_RECORDED_RESULT: "
            "direct_binding_ids must be a sequence, not a string."
        )
    try:
        values = tuple(str(value) for value in tuple(raw))
    except TypeError as exc:
        raise RuntimeError(
            "AWB semantic replay Contact INVALID_RECORDED_RESULT: "
            "direct_binding_ids must be an iterable sequence."
        ) from exc
    if any(not value for value in values) or len(set(values)) != len(values):
        raise RuntimeError(
            "AWB semantic replay Contact INVALID_RECORDED_RESULT: "
            "direct_binding_ids contains an empty or duplicate binding id."
        )
    return values


def _contact_result_payload(
    *,
    frame: int,
    controls: tuple[str, ...],
    replay_mode: ReplayMode,
    result,
) -> dict[str, Any]:
    actual = result.contact_type.value if result.contact_type is not None else None
    actual_mappings = tuple(
        (str(mapping_id), getattr(contact_type, "value", str(contact_type)))
        for mapping_id, contact_type in getattr(result, "mapping_contact_types", ())
    )
    payload: dict[str, Any] = {
        "kind": "CONTACT",
        "frame": frame,
        "controls": controls,
        "replay_mode": replay_mode.value,
        "contact_type": actual,
    }
    if actual_mappings:
        payload["mapping_contact_types"] = actual_mappings
    return payload


def _execute_contact_command_action(
    context,
    action: dict[str, Any],
) -> dict[str, Any]:
    scene, frame, controls = _prepare_contact_action_context(context, action)
    result = execute_contact_command(
        scene,
        control_context_for_context(context),
        operation_id=f"replay-contact-command:{uuid4().hex}",
        enabled_types=contact_enabled_types(context),
        plant_space=contact_plant_space(context),
        contact_point_local=tuple(float(value) for value in scene.baw_contact_point_local),
    )
    if not result.applied:
        detail = "; ".join(item.detail for item in result.diagnostics)
        raise RuntimeError(detail or "AWB semantic replay Contact command action failed.")

    payload = _contact_result_payload(
        frame=frame,
        controls=controls,
        replay_mode=ReplayMode.COMMAND,
        result=result,
    )
    if "contact_type" in action:
        payload["recorded_contact_type"] = action.get("contact_type")
    if "mapping_contact_types" in action:
        payload["recorded_mapping_contact_types"] = action.get("mapping_contact_types")
    if "direct_binding_ids" in action:
        payload["recorded_direct_binding_ids"] = action.get("direct_binding_ids")
    return payload


def _execute_contact_recorded_result_action(
    context,
    action: dict[str, Any],
) -> dict[str, Any]:
    scene, frame, controls = _prepare_contact_action_context(context, action)
    control_context = control_context_for_context(context)
    operation_domain = resolve_operation_domain(scene, control_context)
    snapshot = operation_domain.snapshot
    if snapshot is None or operation_domain.issues:
        detail = "; ".join(issue.detail for issue in operation_domain.issues)
        raise RuntimeError(
            "AWB semantic replay Contact INCOMPLETE_CONTEXT: "
            + (detail or "the frozen Contact operation domain is unavailable.")
        )

    mapping_ids = tuple(str(mapping_id) for mapping_id in snapshot.contact_mapping_ids)
    if not mapping_ids:
        raise RuntimeError(
            "AWB semantic replay Contact INCOMPLETE_CONTEXT: "
            "the recorded selection resolves no Contact mapping."
        )

    domain_direct_ids = tuple(str(binding_id) for binding_id in snapshot.supported_direct_binding_ids)
    recorded_direct_ids = _recorded_direct_binding_ids(action)
    if domain_direct_ids:
        if recorded_direct_ids is None:
            raise RuntimeError(
                "AWB semantic replay Contact INCOMPLETE_CONTEXT: "
                "legacy broad-C history has direct controls but no recorded direct-key coverage."
            )
        if set(recorded_direct_ids) != set(domain_direct_ids):
            raise RuntimeError(
                "AWB semantic replay Contact DIRECT_COVERAGE_MISMATCH: "
                f"recorded={recorded_direct_ids!r}, frozen={domain_direct_ids!r}."
            )
        raise RuntimeError(
            "AWB semantic replay Contact UNSUPPORTED_DIRECT_REPLAY: "
            "direct binding coverage is recorded, but deterministic direct transform payload "
            "is not part of semantic replay v1; refusing to synthesize current broad-C keys."
        )
    if recorded_direct_ids:
        raise RuntimeError(
            "AWB semantic replay Contact DIRECT_COVERAGE_MISMATCH: "
            f"recorded direct bindings {recorded_direct_ids!r} are outside the frozen operation domain."
        )

    scalar_type = _recorded_contact_type(
        action.get("contact_type"),
        field_name="contact_type",
    )
    mapping_types = _recorded_mapping_contact_types(action)

    operation_id = f"replay-contact-recorded:{uuid4().hex}"
    if len(mapping_ids) == 1:
        mapping_id = mapping_ids[0]
        if mapping_types is None:
            if scalar_type is None:
                raise RuntimeError(
                    "AWB semantic replay Contact INCOMPLETE_CONTEXT: "
                    "single-mapping recorded replay is missing contact_type."
                )
            target_type = scalar_type
        else:
            if len(mapping_types) != 1 or mapping_types[0][0] != mapping_id:
                raise RuntimeError(
                    "AWB semantic replay Contact MAPPING_COVERAGE_MISMATCH: "
                    f"recorded={tuple(item[0] for item in mapping_types)!r}, frozen={mapping_ids!r}."
                )
            target_type = mapping_types[0][1]
            if scalar_type is not None and scalar_type is not target_type:
                raise RuntimeError(
                    "AWB semantic replay Contact RECORDED_RESULT_MISMATCH: "
                    f"contact_type={scalar_type.value!r} disagrees with mapping result "
                    f"{target_type.value!r}."
                )
        if target_type is ContactKeyType.PLANTED:
            raise RuntimeError(
                "AWB semantic replay Contact INCOMPLETE_CONTEXT: "
                "recorded Planted replay lacks a frozen plant-space/contact-point payload."
            )
        planned = build_contact_intent_plan(
            scene,
            control_context,
            operation_id=operation_id,
            mode=ContactAuthoringMode.CYCLE,
            enabled_types=_RECORDED_CONTACT_TYPES,
            plant_space=ContactPlantSpace.WORLD,
            contact_point_local=(0.0, 0.0, 0.0),
            mapping_id=mapping_id,
            forced_cycle_type=target_type,
        )
        if not planned.ok or planned.plan is None:
            detail = "; ".join(item.detail for item in planned.diagnostics)
            raise RuntimeError(
                detail or "AWB semantic replay Contact recorded-result planning failed."
            )
        result = execute_contact_intent_plan(
            scene,
            control_context,
            planned.plan,
            closure_plan=None,
        )
        if not result.applied:
            detail = "; ".join(item.detail for item in result.diagnostics)
            raise RuntimeError(
                detail or "AWB semantic replay Contact recorded-result execution failed."
            )
        if result.contact_type is not target_type:
            actual = result.contact_type.value if result.contact_type is not None else None
            raise RuntimeError(
                "AWB semantic replay Contact RECORDED_RESULT_MISMATCH: "
                f"expected {target_type.value}, got {actual}."
            )
        actual_mappings = tuple(
            (str(actual_mapping_id), actual_type)
            for actual_mapping_id, actual_type in result.mapping_contact_types
        )
        expected_mapping_result = ((mapping_id, target_type),)
        if actual_mappings != expected_mapping_result:
            raise RuntimeError(
                "AWB semantic replay Contact MAPPING_COVERAGE_MISMATCH: "
                f"expected={expected_mapping_result!r}, actual={actual_mappings!r}."
            )
        return _contact_result_payload(
            frame=frame,
            controls=controls,
            replay_mode=ReplayMode.RECORDED_RESULT,
            result=result,
        )

    if mapping_types is None:
        raise RuntimeError(
            "AWB semantic replay Contact INCOMPLETE_CONTEXT: "
            "multi-mapping recorded replay requires explicit mapping_contact_types coverage."
        )
    recorded_by_id = dict(mapping_types)
    missing = tuple(mapping_id for mapping_id in mapping_ids if mapping_id not in recorded_by_id)
    extra = tuple(mapping_id for mapping_id in recorded_by_id if mapping_id not in set(mapping_ids))
    if missing or extra:
        raise RuntimeError(
            "AWB semantic replay Contact MAPPING_COVERAGE_MISMATCH: "
            f"missing={missing!r}, extra={extra!r}, frozen={mapping_ids!r}."
        )
    ordered_targets = tuple((mapping_id, recorded_by_id[mapping_id]) for mapping_id in mapping_ids)
    if any(contact_type is ContactKeyType.PLANTED for _mapping_id, contact_type in ordered_targets):
        raise RuntimeError(
            "AWB semantic replay Contact INCOMPLETE_CONTEXT: "
            "recorded Planted replay lacks a frozen plant-space/contact-point payload."
        )
    common_type = (
        ordered_targets[0][1]
        if all(contact_type is ordered_targets[0][1] for _mapping_id, contact_type in ordered_targets)
        else None
    )
    if scalar_type is not None and scalar_type is not common_type:
        raise RuntimeError(
            "AWB semantic replay Contact RECORDED_RESULT_MISMATCH: "
            "scalar contact_type disagrees with explicit per-mapping results."
        )

    batch = build_contact_batch_intent_plan(
        scene,
        control_context,
        operation_id=operation_id,
        mode=ContactAuthoringMode.CYCLE,
        enabled_types=_RECORDED_CONTACT_TYPES,
        plant_space=ContactPlantSpace.WORLD,
        contact_point_local=(0.0, 0.0, 0.0),
        mapping_ids=mapping_ids,
        forced_mapping_types=dict(ordered_targets),
    )
    if not batch.ok or batch.plan is None:
        detail = "; ".join(item.detail for item in batch.diagnostics)
        raise RuntimeError(
            detail or "AWB semantic replay Contact recorded-result batch planning failed."
        )
    result = execute_contact_batch_intent_plan(
        scene,
        control_context,
        batch.plan,
        closure_plan=None,
    )
    if not result.applied:
        detail = "; ".join(item.detail for item in result.diagnostics)
        raise RuntimeError(
            detail or "AWB semantic replay Contact recorded-result batch execution failed."
        )
    actual_mappings = tuple(
        (str(mapping_id), contact_type)
        for mapping_id, contact_type in result.mapping_contact_types
    )
    if actual_mappings != ordered_targets:
        raise RuntimeError(
            "AWB semantic replay Contact RECORDED_RESULT_MISMATCH: "
            f"expected={ordered_targets!r}, actual={actual_mappings!r}."
        )
    return _contact_result_payload(
        frame=frame,
        controls=controls,
        replay_mode=ReplayMode.RECORDED_RESULT,
        result=result,
    )


def _execute_contact_action(
    context,
    action: dict[str, Any],
    *,
    replay_mode: ReplayMode,
) -> dict[str, Any]:
    if replay_mode is ReplayMode.COMMAND:
        return _execute_contact_command_action(context, action)
    return _execute_contact_recorded_result_action(context, action)


def _execute_free_direct_rotate_action(context, action: dict[str, Any]) -> dict[str, Any]:
    scene = context.scene
    frame = int(action.get("frame", scene.frame_current))
    subframe = float(action.get("subframe", 0.0))
    scene.frame_set(frame, subframe=subframe)
    controls = tuple(str(name) for name in tuple(action.get("controls") or ()) if str(name))
    if controls:
        _select_pose_controls(context, controls)

    orientation = str(action.get("orientation") or "GLOBAL")
    if orientation in {"GLOBAL", "LOCAL", "VIEW", "NORMAL", "GIMBAL", "CURSOR"}:
        try:
            scene.transform_orientation_slots[0].type = orientation
        except (AttributeError, TypeError, ValueError):
            pass

    resolved = _selected_direct_rotate_controls(context)
    if not resolved:
        raise RuntimeError(
            "AWB semantic replay v1 direct Rotate requires at least one selected Rigped control."
        )

    domain_resolution = resolve_operation_domain(
        scene,
        control_context_for_context(context),
    )
    if not domain_resolution.ok or domain_resolution.snapshot is None:
        detail = (
            domain_resolution.issues[0].detail
            if domain_resolution.issues
            else "AWB semantic replay Rotate could not freeze the operation domain."
        )
        raise RuntimeError(detail)
    operation_domain = domain_resolution.snapshot
    frozen_current_sliding = _current_sliding_capabilities(context)
    sliding_syncs = _direct_rotate_sliding_sync_sessions(
        context,
        operation_domain,
    )
    sliding_guard_capabilities = _passive_sliding_capabilities(
        frozen_current_sliding,
        sliding_syncs,
    )

    auto_plan = None
    auto_contact_batch_plan = None
    auto_direct_plan = None
    auto_limb_context = None
    auto_direct_context = None
    deferred_mapping_ids: tuple[str, ...] = ()
    if bool(getattr(scene, "baw_auto_key_enabled", False)):
        auto_context = control_context_for_context(context)
        auto_limb_context, auto_direct_context = _direct_rotate_auto_contexts(
            scene,
            auto_context,
        )
        mapping_ids = (
            selected_contact_mapping_ids(scene, auto_limb_context)
            if auto_limb_context is not None
            else ()
        )
        if len(mapping_ids) >= 2:
            deferred_mapping_ids = tuple(mapping_ids)
        elif mapping_ids:
            planned = plan_rigped_auto_anchor(
                scene,
                auto_limb_context,
                operation_id=f"replay-auto-rotate:{uuid4().hex}:contact",
                mapping_id=mapping_ids[0],
            )
            if not planned.ok or planned.plan is None:
                detail = planned.diagnostics[0].detail if planned.diagnostics else "Auto Rotate plan failed."
                raise RuntimeError(detail)
            if planned.plan.intent.target_type not in {
                ContactKeyType.FREE,
                ContactKeyType.SLIDING,
            }:
                raise RuntimeError(
                    "AWB semantic replay v1 direct Rotate supports Free or Sliding Contact authority only."
                )
            auto_plan = planned.plan
        if auto_direct_context is not None:
            planned_direct = plan_rigped_auto_direct_rotate(
                scene,
                auto_direct_context,
                operation_id=f"replay-auto-direct-rotate:{uuid4().hex}:direct",
                active_only=len(auto_direct_context.controls) == 1,
            )
            if not planned_direct.ok or planned_direct.plan is None:
                detail = (
                    planned_direct.diagnostics[0].detail
                    if planned_direct.diagnostics
                    else "Direct Rotate Auto plan failed."
                )
                raise RuntimeError(detail)
            auto_direct_plan = planned_direct.plan
            if auto_plan is not None or deferred_mapping_ids:
                auto_direct_plan = replace(
                    auto_direct_plan,
                    allow_storage_rebind=True,
                )

    quaternion_values = tuple(float(value) for value in tuple(action.get("delta_world_quaternion") or ()))
    if len(quaternion_values) != 4:
        raise RuntimeError("AWB semantic replay Rotate is missing its world delta quaternion.")
    delta_world = Quaternion(quaternion_values).normalized()

    rotation_world = delta_world.to_matrix().to_4x4()
    for control in resolved:
        pose_bone = control.target
        owner = control.owner_object
        start_pose = pose_bone.matrix.copy()
        start_world = owner.matrix_world @ start_pose
        pivot_world = Vector(_control_pivot_world(control))
        desired_world = (
            Matrix.Translation(pivot_world)
            @ rotation_world
            @ Matrix.Translation(-pivot_world)
            @ start_world
        )
        desired_pose = owner.matrix_world.inverted_safe() @ desired_world
        desired_state = _state_for_pose_matrix(control, desired_pose)
        _apply_control_state(
            control,
            desired_state,
            location=_uses_center_pivot(control),
            rotation=True,
        )
    context.view_layer.update()
    for session in sliding_syncs:
        _apply_direct_rotate_sliding_sync(context, session)
    _refresh_current_sliding_public_overlays(
        context,
        capabilities=sliding_guard_capabilities,
    )

    if deferred_mapping_ids:
        batch = plan_rigped_auto_contact_batch(
            scene,
            auto_limb_context,
            operation_id=f"replay-auto-rotate-release:{uuid4().hex}",
            mapping_ids=deferred_mapping_ids,
        )
        if not batch.ok or batch.plan is None:
            detail = batch.diagnostics[0].detail if batch.diagnostics else "Auto Rotate batch plan failed."
            raise RuntimeError(detail)
        auto_contact_batch_plan = batch.plan

    if auto_plan is not None:
        committed = commit_rigped_auto_anchor(
            scene,
            auto_limb_context,
            auto_plan,
        )
        if not committed.applied:
            detail = "; ".join(item.detail for item in committed.diagnostics)
            raise RuntimeError(detail or "AWB semantic replay Auto Rotate commit failed.")
    if auto_contact_batch_plan is not None:
        committed = commit_rigped_auto_contact_batch(
            scene,
            auto_limb_context,
            auto_contact_batch_plan,
        )
        if not committed.applied:
            detail = "; ".join(item.detail for item in committed.diagnostics)
            raise RuntimeError(detail or "AWB semantic replay multi-limb Rotate Auto commit failed.")
    if auto_direct_plan is not None:
        committed = commit_rigped_auto_direct_rotate(
            scene,
            auto_direct_context,
            auto_direct_plan,
        )
        if not committed.applied:
            detail = "; ".join(item.detail for item in committed.diagnostics)
            raise RuntimeError(detail or "AWB semantic replay direct Rotate Auto commit failed.")
        _refresh_current_sliding_public_overlays(context)

    return {
        "kind": "ROTATE",
        "frame": frame,
        "controls": controls,
        "axis": action.get("axis"),
        "orientation": orientation,
        "delta_world_quaternion": quaternion_values,
    }


def _execute_sliding_move_action(context, action: dict[str, Any]) -> dict[str, Any]:
    scene = context.scene
    frame = int(action.get("frame", scene.frame_current))
    subframe = float(action.get("subframe", 0.0))
    scene.frame_set(frame, subframe=subframe)
    controls = tuple(str(name) for name in tuple(action.get("controls") or ()) if str(name))
    if controls:
        _select_pose_controls(context, controls)

    resolutions, sessions, fk_sessions = _begin_semantic_move_domains(context)
    if not sessions or fk_sessions:
        cancel_semantic_move_domains(context, sessions, fk_sessions)
        raise RuntimeError(
            "AWB semantic replay Sliding Move requires Sliding limb domains only."
        )
    session = sessions[0]

    auto_plan = None
    auto_contact_batch_plan = None
    auto_limb_context = None
    if bool(getattr(scene, "baw_auto_key_enabled", False)):
        auto_context = control_context_for_context(context)
        auto_limb_context, _auto_direct_context = _fk_move_auto_contexts(
            scene,
            auto_context,
            fk_sessions,
        )
        mapping_ids = tuple(
            str(resolved.capability.native_ik.mapping_id)
            for resolved in resolutions
        )
        if len(mapping_ids) >= 2:
            planned_batch = plan_rigped_auto_contact_batch(
                scene,
                auto_limb_context,
                operation_id=f"replay-auto-sliding-move:{uuid4().hex}",
                mapping_ids=mapping_ids,
            )
            if not planned_batch.ok or planned_batch.plan is None:
                cancel_semantic_move_domains(context, sessions, fk_sessions)
                detail = (
                    planned_batch.diagnostics[0].detail
                    if planned_batch.diagnostics
                    else "Auto Sliding Move batch plan failed."
                )
                raise RuntimeError(detail)
            auto_contact_batch_plan = planned_batch.plan
        else:
            planned = plan_rigped_auto_anchor(
                scene,
                auto_limb_context,
                operation_id=f"replay-auto-sliding-move:{uuid4().hex}",
                mapping_id=session.intent.mapping_id,
            )
            if not planned.ok or planned.plan is None:
                cancel_semantic_move_domains(context, sessions, fk_sessions)
                detail = planned.diagnostics[0].detail if planned.diagnostics else "Auto Sliding Move plan failed."
                raise RuntimeError(detail)
            if planned.plan.intent.target_type is not ContactKeyType.SLIDING:
                cancel_semantic_move_domains(context, sessions, fk_sessions)
                raise RuntimeError(
                    "AWB semantic replay v1 Sliding Move requires Sliding Contact authority."
                )
            auto_plan = planned.plan

    delta_values = tuple(float(value) for value in tuple(action.get("delta_world") or ()))
    if len(delta_values) != 3:
        cancel_semantic_move_domains(context, sessions, fk_sessions)
        raise RuntimeError("AWB semantic replay Sliding Move is missing its world delta.")

    try:
        if not apply_semantic_move_domains_delta(
            context,
            resolutions,
            sessions,
            fk_sessions,
            Vector(delta_values),
        ):
            raise RuntimeError("AWB semantic replay Sliding Move solve failed.")

        if auto_contact_batch_plan is not None:
            committed = commit_rigped_auto_contact_batch(
                scene,
                auto_limb_context,
                auto_contact_batch_plan,
            )
            if not committed.applied:
                detail = "; ".join(item.detail for item in committed.diagnostics)
                raise RuntimeError(detail or "AWB semantic replay Sliding Move batch commit failed.")
        elif auto_plan is not None:
            committed = commit_rigped_auto_anchor(
                scene,
                auto_limb_context,
                auto_plan,
            )
            if not committed.applied:
                detail = "; ".join(item.detail for item in committed.diagnostics)
                raise RuntimeError(detail or "AWB semantic replay Sliding Move commit failed.")
    except Exception:
        cancel_semantic_move_domains(context, sessions, fk_sessions)
        raise

    return {
        "kind": "MOVE",
        "frame": frame,
        "controls": controls,
        "axis": action.get("axis"),
        "route": "SLIDING_MOVE",
        "delta_world": delta_values,
        "domain_count": len(sessions),
    }


def _execute_direct_move_action(context, action: dict[str, Any]) -> dict[str, Any]:
    scene = context.scene
    frame = int(action.get("frame", scene.frame_current))
    subframe = float(action.get("subframe", 0.0))
    scene.frame_set(frame, subframe=subframe)
    controls = tuple(str(name) for name in tuple(action.get("controls") or ()) if str(name))
    if controls:
        _select_pose_controls(context, controls)

    control_context = control_context_for_context(context)
    domain_resolution = resolve_operation_domain(scene, control_context)
    if not domain_resolution.ok or domain_resolution.snapshot is None:
        detail = (
            domain_resolution.issues[0].detail
            if domain_resolution.issues
            else "Direct Move replay could not freeze the operation domain."
        )
        raise RuntimeError(detail)
    operation_domain = domain_resolution.snapshot
    states = _begin_direct_move_states(control_context, operation_domain)
    frozen_sliding = _sliding_capabilities_for_character(
        scene,
        operation_domain.character_id,
    )
    e7_body_move = all(
        _resolved_control_role_name(state.control) == "COM"
        for state in states
    )
    dependency_guards = (
        _capture_sliding_dependency_guards(frozen_sliding)
        if e7_body_move
        else ()
    )

    auto_plan = None
    if bool(getattr(scene, "baw_auto_key_enabled", False)):
        planned = plan_rigped_auto_direct_move(
            scene,
            control_context,
            operation_id=f"replay-auto-direct-move:{uuid4().hex}",
            active_only=len(states) == 1,
        )
        if not planned.ok or planned.plan is None:
            detail = planned.diagnostics[0].detail if planned.diagnostics else "Direct Move Auto plan failed."
            raise RuntimeError(detail)
        auto_plan = planned.plan

    delta_values = tuple(float(value) for value in tuple(action.get("delta_world") or ()))
    if len(delta_values) != 3:
        raise RuntimeError("AWB semantic replay Direct Move is missing its world delta.")
    _apply_direct_move_delta(
        context,
        states,
        Vector(delta_values),
        sliding_capabilities=frozen_sliding,
        dependency_guards=dependency_guards,
        operation_id="semantic-replay-direct-move",
    )

    if auto_plan is not None:
        committed = commit_rigped_auto_direct_move(
            scene,
            control_context,
            auto_plan,
        )
        if not committed.applied:
            detail = "; ".join(item.detail for item in committed.diagnostics)
            raise RuntimeError(detail or "AWB semantic replay Direct Move Auto commit failed.")
        _refresh_current_sliding_public_overlays(
            context,
            capabilities=frozen_sliding,
        )

    return {
        "kind": "MOVE",
        "frame": frame,
        "controls": controls,
        "axis": action.get("axis"),
        "route": "DIRECT_MOVE",
        "delta_world": delta_values,
    }


def _execute_free_fk_move_action(context, action: dict[str, Any]) -> dict[str, Any]:
    scene = context.scene
    frame = int(action.get("frame", scene.frame_current))
    subframe = float(action.get("subframe", 0.0))
    scene.frame_set(frame, subframe=subframe)
    controls = tuple(str(name) for name in tuple(action.get("controls") or ()) if str(name))
    if controls:
        _select_pose_controls(context, controls)

    sessions = begin_fk_joint_moves(context)
    if not sessions:
        raise RuntimeError("AWB semantic replay Move could not resolve any Free FK operation domain.")

    auto_plan = None
    auto_contact_batch_plan = None
    auto_direct_plan = None
    auto_limb_context = None
    auto_direct_context = None
    deferred_mapping_ids: tuple[str, ...] = ()
    if bool(getattr(scene, "baw_auto_key_enabled", False)):
        auto_context = control_context_for_context(context)
        auto_limb_context, auto_direct_context = _fk_move_auto_contexts(
            scene,
            auto_context,
            sessions,
        )
        mapping_ids = (
            selected_contact_mapping_ids(scene, auto_limb_context)
            if auto_limb_context is not None
            else ()
        )
        operation_id = f"replay-auto-move:{uuid4().hex}"
        if len(mapping_ids) >= 2:
            deferred_mapping_ids = tuple(mapping_ids)
        elif len(mapping_ids) == 1:
            planned = plan_rigped_auto_anchor(
                scene,
                auto_limb_context,
                operation_id=f"{operation_id}:contact",
                mapping_id=mapping_ids[0],
            )
            if not planned.ok or planned.plan is None:
                cancel_fk_joint_moves(context, sessions)
                detail = planned.diagnostics[0].detail if planned.diagnostics else "Auto Move plan failed."
                raise RuntimeError(detail)
            if planned.plan.intent.target_type is not ContactKeyType.FREE:
                cancel_fk_joint_moves(context, sessions)
                raise RuntimeError(
                    "AWB semantic replay v1 Move currently supports Free Contact authority only."
                )
            auto_plan = planned.plan

        if auto_direct_context is not None:
            direct = plan_rigped_auto_direct_rotate(
                scene,
                auto_direct_context,
                operation_id=f"{operation_id}:direct",
                active_only=False,
            )
            if not direct.ok or direct.plan is None:
                cancel_fk_joint_moves(context, sessions)
                detail = (
                    direct.diagnostics[0].detail
                    if direct.diagnostics
                    else "AWB semantic replay direct FK Move Auto planning failed."
                )
                raise RuntimeError(detail)
            auto_direct_plan = direct.plan

    delta_values = tuple(float(value) for value in tuple(action.get("delta_world") or ()))
    if len(delta_values) != 3:
        cancel_fk_joint_moves(context, sessions)
        raise RuntimeError("AWB semantic replay Move is missing its world delta.")

    if not apply_fk_joint_moves_delta(context, sessions, Vector(delta_values)):
        cancel_fk_joint_moves(context, sessions)
        raise RuntimeError("AWB semantic replay Free FK Move solve failed.")

    if deferred_mapping_ids:
        batch = plan_rigped_auto_contact_batch(
            scene,
            auto_limb_context,
            operation_id=f"replay-auto-move-release:{uuid4().hex}",
            mapping_ids=deferred_mapping_ids,
        )
        if not batch.ok or batch.plan is None:
            cancel_fk_joint_moves(context, sessions)
            detail = (
                batch.diagnostics[0].detail
                if batch.diagnostics
                else "AWB semantic replay multi-limb FK Move Auto planning failed."
            )
            raise RuntimeError(detail)
        auto_contact_batch_plan = batch.plan

    if auto_plan is not None:
        committed = commit_rigped_auto_anchor(scene, auto_limb_context, auto_plan)
        if not committed.applied:
            cancel_fk_joint_moves(context, sessions)
            detail = "; ".join(item.detail for item in committed.diagnostics)
            raise RuntimeError(detail or "AWB semantic replay Auto Move commit failed.")
    elif auto_contact_batch_plan is not None:
        committed = commit_rigped_auto_contact_batch(
            scene,
            auto_limb_context,
            auto_contact_batch_plan,
        )
        if not committed.applied:
            cancel_fk_joint_moves(context, sessions)
            detail = "; ".join(item.detail for item in committed.diagnostics)
            raise RuntimeError(detail or "AWB semantic replay multi-limb Auto Move commit failed.")

    if auto_direct_plan is not None:
        if auto_plan is not None or auto_contact_batch_plan is not None:
            auto_direct_plan = replace(auto_direct_plan, allow_storage_rebind=True)
        committed = commit_rigped_auto_direct_rotate(
            scene,
            auto_direct_context,
            auto_direct_plan,
        )
        if not committed.applied:
            cancel_fk_joint_moves(context, sessions)
            detail = "; ".join(item.detail for item in committed.diagnostics)
            raise RuntimeError(detail or "AWB semantic replay direct FK Move Auto commit failed.")
        _refresh_current_sliding_public_overlays(context)

    finalized = commit_fk_joint_moves(context, sessions)
    if not finalized.success:
        cancel_fk_joint_moves(context, sessions)
        detail = "; ".join(item.detail for item in finalized.diagnostics)
        raise RuntimeError(detail or "AWB semantic replay Free FK Move finalize failed.")

    return {
        "kind": "MOVE",
        "frame": frame,
        "controls": controls,
        "axis": action.get("axis"),
        "route": action.get("route"),
        "delta_world": delta_values,
        "domain_count": len(sessions),
        "mixed_direct": auto_direct_plan is not None,
    }


def _execute_hybrid_move_action(context, action: dict[str, Any]) -> dict[str, Any]:
    scene = context.scene
    frame = int(action.get("frame", scene.frame_current))
    subframe = float(action.get("subframe", 0.0))
    scene.frame_set(frame, subframe=subframe)
    controls = tuple(str(name) for name in tuple(action.get("controls") or ()) if str(name))
    if controls:
        _select_pose_controls(context, controls)

    resolutions, sliding_sessions, fk_sessions = _begin_semantic_move_domains(context)
    if not sliding_sessions:
        cancel_semantic_move_domains(context, sliding_sessions, fk_sessions)
        raise RuntimeError("AWB semantic replay Hybrid Move requires at least one Sliding domain.")

    auto_plan = None
    auto_contact_batch_plan = None
    auto_direct_plan = None
    auto_limb_context = None
    auto_direct_context = None
    if bool(getattr(scene, "baw_auto_key_enabled", False)):
        auto_context = control_context_for_context(context)
        auto_limb_context, auto_direct_context = _fk_move_auto_contexts(
            scene,
            auto_context,
            fk_sessions,
        )
        mapping_ids = tuple(
            str(resolved.capability.native_ik.mapping_id)
            for resolved in resolutions
        )
        if len(mapping_ids) >= 2:
            planned = plan_rigped_auto_contact_batch(
                scene,
                auto_limb_context,
                operation_id=f"replay-auto-hybrid-move:{uuid4().hex}",
                mapping_ids=mapping_ids,
            )
            if not planned.ok or planned.plan is None:
                cancel_semantic_move_domains(context, sliding_sessions, fk_sessions)
                detail = planned.diagnostics[0].detail if planned.diagnostics else "Hybrid Move Auto batch plan failed."
                raise RuntimeError(detail)
            auto_contact_batch_plan = planned.plan
        else:
            planned = plan_rigped_auto_anchor(
                scene,
                auto_limb_context,
                operation_id=f"replay-auto-hybrid-move:{uuid4().hex}:contact",
                mapping_id=sliding_sessions[0].intent.mapping_id,
            )
            if not planned.ok or planned.plan is None:
                cancel_semantic_move_domains(context, sliding_sessions, fk_sessions)
                detail = planned.diagnostics[0].detail if planned.diagnostics else "Hybrid Move Auto plan failed."
                raise RuntimeError(detail)
            auto_plan = planned.plan

        if auto_direct_context is not None:
            direct = plan_rigped_auto_direct_rotate(
                scene,
                auto_direct_context,
                operation_id=f"replay-auto-hybrid-move:{uuid4().hex}:direct",
                active_only=len(auto_direct_context.controls) == 1,
            )
            if not direct.ok or direct.plan is None:
                cancel_semantic_move_domains(context, sliding_sessions, fk_sessions)
                detail = direct.diagnostics[0].detail if direct.diagnostics else "Hybrid Move direct Auto plan failed."
                raise RuntimeError(detail)
            auto_direct_plan = direct.plan

    delta_values = tuple(float(value) for value in tuple(action.get("delta_world") or ()))
    if len(delta_values) != 3:
        cancel_semantic_move_domains(context, sliding_sessions, fk_sessions)
        raise RuntimeError("AWB semantic replay Hybrid Move is missing its world delta.")

    try:
        if not apply_semantic_move_domains_delta(
            context,
            resolutions,
            sliding_sessions,
            fk_sessions,
            Vector(delta_values),
        ):
            raise RuntimeError("AWB semantic replay Hybrid Move solve failed.")

        if auto_contact_batch_plan is not None:
            committed = commit_rigped_auto_contact_batch(
                scene,
                auto_limb_context,
                auto_contact_batch_plan,
            )
            if not committed.applied:
                detail = "; ".join(item.detail for item in committed.diagnostics)
                raise RuntimeError(detail or "AWB semantic replay Hybrid Move batch commit failed.")
        elif auto_plan is not None:
            committed = commit_rigped_auto_anchor(scene, auto_limb_context, auto_plan)
            if not committed.applied:
                detail = "; ".join(item.detail for item in committed.diagnostics)
                raise RuntimeError(detail or "AWB semantic replay Hybrid Move commit failed.")

        if auto_direct_plan is not None:
            if auto_plan is not None or auto_contact_batch_plan is not None:
                auto_direct_plan = replace(auto_direct_plan, allow_storage_rebind=True)
            committed = commit_rigped_auto_direct_rotate(
                scene,
                auto_direct_context,
                auto_direct_plan,
            )
            if not committed.applied:
                detail = "; ".join(item.detail for item in committed.diagnostics)
                raise RuntimeError(detail or "AWB semantic replay Hybrid Move direct commit failed.")
            _refresh_current_sliding_public_overlays(context)
    except Exception:
        cancel_semantic_move_domains(context, sliding_sessions, fk_sessions)
        raise

    return {
        "kind": "MOVE",
        "frame": frame,
        "controls": controls,
        "axis": action.get("axis"),
        "route": "HYBRID_MOVE",
        "delta_world": delta_values,
        "sliding_domain_count": len(sliding_sessions),
        "fk_domain_count": len(fk_sessions),
        "mixed_direct": auto_direct_plan is not None,
    }


def _execute_scrub_action(
    context,
    action: dict[str, Any],
    *,
    frame_budget: list[int],
    max_frames: int,
) -> dict[str, Any]:
    scene = context.scene
    frames = tuple(action.get("frames") or ())
    if not frames:
        return {"kind": "SCRUB", "frames": 0, "samples": []}
    controls = tuple(str(name) for name in tuple(action.get("controls") or ()) if str(name))
    if controls:
        _select_pose_controls(context, controls)

    first = frames[0]
    scene.frame_set(int(first.get("frame", 0)), subframe=float(first.get("subframe", 0.0)))
    trace_scrub_begin(context, source="REPLAY_EXECUTOR")
    samples: list[dict[str, Any]] = []
    try:
        for index, frame_item in enumerate(frames):
            if index > 0:
                scene.frame_set(
                    int(frame_item.get("frame", 0)),
                    subframe=float(frame_item.get("subframe", 0.0)),
                )
            frame_budget[0] += 1
            if frame_budget[0] > int(max_frames):
                raise RuntimeError("AWB semantic replay exceeded its bounded frame budget.")
            samples.append(
                {
                    "frame": int(scene.frame_current),
                    "subframe": float(getattr(scene, "frame_subframe", 0.0)),
                    "selected_world_quaternions": _selected_pose_world_snapshot(context),
                }
            )
    finally:
        trace_scrub_end(context, cancelled=False)
    return {
        "kind": "SCRUB",
        "source_seq": action.get("source_seq"),
        "frames": len(frames),
        "samples": samples,
    }


def _semantic_replay_action_route(
    action: dict[str, Any],
    *,
    replay_mode: ReplayMode,
) -> str:
    kind = str(action.get("kind") or "")
    if kind == "AUTO_KEY":
        return "AUTO_KEY"
    if kind == "CONTACT":
        return f"CONTACT_{replay_mode.value}"
    if kind == "ROTATE":
        return "FREE_DIRECT_ROTATE"
    if kind == "MOVE":
        route = str(action.get("route") or "")
        if route in {"SLIDING_MOVE", "HYBRID_MOVE", "DIRECT_MOVE"}:
            return route
        return "FREE_FK_MOVE"
    if kind == "SCRUB":
        return "SCRUB"
    raise RuntimeError(f"AWB semantic replay v1 does not support action kind {kind!r}.")


def _semantic_replay_step_manifest(
    actions: tuple[Any, ...],
    *,
    replay_mode: ReplayMode,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for step_index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            raise TypeError(
                f"AWB semantic replay action {step_index} is not a mapping; stable alignment is unavailable."
            )
        kind = str(action.get("kind") or "")
        route = _semantic_replay_action_route(action, replay_mode=replay_mode)
        manifest.append(
            {
                "step_index": step_index,
                "step_id": f"semantic:{step_index:04d}",
                "source_seq": action.get("source_seq"),
                "action_kind": kind,
                "action_route": route,
                "checkpoint_boundaries": ["BEFORE_OPERATION", "AFTER_OPERATION"],
            }
        )
    return manifest


def run_semantic_replay(
    context=None,
    *,
    replay_file: str | None = None,
    script: dict[str, Any] | None = None,
    replay_mode: ReplayMode | str,
    replay_execution_id: str | None = None,
    max_actions: int = 1000,
    max_frames: int = 12000,
) -> dict[str, Any]:
    """Execute replay-grade user intent against the current clean scene.

    BB-6 alignment is authored here, at execution time: every semantic action
    gets a stable step manifest entry plus explicit BEFORE/AFTER checkpoints.
    Runtime UUIDs remain provenance only and are never required for cross-run
    equality/alignment.
    """

    context = context or bpy.context
    if getattr(context, "scene", None) is None:
        raise RuntimeError("AWB semantic replay requires an active scene.")
    replay = dict(script) if script is not None else _load_replay_script(replay_file)
    try:
        resolved_replay_mode = ReplayMode(replay_mode)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"AWB semantic replay mode is unsupported: {replay_mode!r}.") from exc
    actions = tuple(replay.get("actions") or ())
    if len(actions) > int(max_actions):
        raise RuntimeError("AWB semantic replay exceeded its bounded action budget.")

    step_manifest = _semantic_replay_step_manifest(
        actions,
        replay_mode=resolved_replay_mode,
    )
    execution_id = str(replay_execution_id or f"semantic-replay:{uuid4().hex}")
    results: list[dict[str, Any]] = []
    frame_budget = [0]
    set_replay_execution_active(
        True,
        execution_id=execution_id,
    )
    try:
        for action, step in zip(actions, step_manifest, strict=True):
            checkpoint_args = {
                "context": context,
                "source_event": "SEMANTIC_REPLAY_STEP",
                "subsystem": "replay",
                "evaluation_phase": "semantic_replay",
                "replay_execution_id": execution_id,
                "replay_step_id": step["step_id"],
                "replay_step_index": step["step_index"],
                "replay_action_kind": step["action_kind"],
                "replay_action_route": step["action_route"],
                "replay_boundary_ordinal": 1,
            }
            capture_debug_state_checkpoint(
                "BEFORE_OPERATION",
                lifecycle_phase="before",
                **checkpoint_args,
            )
            try:
                kind = str(action.get("kind") or "")
                if kind == "AUTO_KEY":
                    result = _execute_auto_key_action(context, action)
                elif kind == "CONTACT":
                    result = _execute_contact_action(
                        context,
                        action,
                        replay_mode=resolved_replay_mode,
                    )
                elif kind == "ROTATE":
                    result = _execute_free_direct_rotate_action(context, action)
                elif kind == "MOVE":
                    route = str(action.get("route") or "")
                    if route == "SLIDING_MOVE":
                        result = _execute_sliding_move_action(context, action)
                    elif route == "HYBRID_MOVE":
                        result = _execute_hybrid_move_action(context, action)
                    elif route == "DIRECT_MOVE":
                        result = _execute_direct_move_action(context, action)
                    else:
                        result = _execute_free_fk_move_action(context, action)
                elif kind == "SCRUB":
                    result = _execute_scrub_action(
                        context,
                        action,
                        frame_budget=frame_budget,
                        max_frames=max_frames,
                    )
                else:
                    raise RuntimeError(
                        f"AWB semantic replay v1 does not support action kind {kind!r}."
                    )
            except Exception:
                capture_debug_state_checkpoint(
                    "FAILURE",
                    lifecycle_phase="fail",
                    **checkpoint_args,
                )
                raise
            results.append(result)
            capture_debug_state_checkpoint(
                "AFTER_OPERATION",
                lifecycle_phase="after",
                **checkpoint_args,
            )
    finally:
        set_replay_execution_active(False)

    return {
        "schema": "awb-semantic-replay-result/v1",
        "source_session_id": replay.get("source_session_id"),
        "replay_mode": resolved_replay_mode.value,
        "replay_execution_id": execution_id,
        "action_count": len(actions),
        "executed_frame_count": frame_budget[0],
        "step_manifest": step_manifest,
        "results": results,
    }



def run_checkpoint_scrub_replay(
    context=None,
    *,
    replay_file: str | None = None,
    source_seqs: tuple[int, ...] | None = None,
    last_scrubs: int | None = None,
    max_frames: int = 12000,
) -> dict[str, Any]:
    """Replay recorded scrub paths against an already-authored repro checkpoint.

    This executor intentionally starts with scrub playback only. The authored
    checkpoint preserves the user's exact key/Contact state, while the replay
    script supplies deterministic frame navigation. This is the safest first
    loop for frame-evaluation bugs because it does not rewrite animation.
    """

    context = context or bpy.context
    scene = getattr(context, "scene", None)
    if scene is None:
        raise RuntimeError("AWB replay requires an active Blender scene.")

    script = _load_replay_script(replay_file)
    actions = [
        item
        for item in tuple(script.get("actions") or ())
        if isinstance(item, dict) and item.get("kind") == "SCRUB"
    ]
    if source_seqs is not None:
        wanted = {int(value) for value in source_seqs}
        actions = [item for item in actions if int(item.get("source_seq") or -1) in wanted]
    if last_scrubs is not None:
        actions = actions[-max(0, int(last_scrubs)):]

    original_frame = int(scene.frame_current)
    original_subframe = float(getattr(scene, "frame_subframe", 0.0))
    samples: list[dict[str, Any]] = []
    executed_frames = 0

    set_replay_execution_active(True)
    try:
        for action in actions:
            frames = tuple(action.get("frames") or ())
            if not frames:
                continue
            controls = tuple(str(name) for name in tuple(action.get("controls") or ()) if str(name))
            if controls:
                _select_pose_controls(context, controls)

            first = frames[0]
            scene.frame_set(
                int(first.get("frame", 0)),
                subframe=float(first.get("subframe", 0.0)),
            )
            trace_scrub_begin(context, source="REPLAY_EXECUTOR")
            try:
                for index, frame_item in enumerate(frames):
                    if index > 0:
                        scene.frame_set(
                            int(frame_item.get("frame", 0)),
                            subframe=float(frame_item.get("subframe", 0.0)),
                        )
                    executed_frames += 1
                    if executed_frames > int(max_frames):
                        raise RuntimeError("AWB replay exceeded its bounded frame budget.")
                    for row in _sliding_divergences(context):
                        samples.append(
                            {
                                "source_seq": int(action.get("source_seq") or -1),
                                "frame": int(scene.frame_current),
                                "subframe": float(getattr(scene, "frame_subframe", 0.0)),
                                **row,
                            }
                        )
            finally:
                trace_scrub_end(context, cancelled=False)
    finally:
        scene.frame_set(original_frame, subframe=original_subframe)
        set_replay_execution_active(False)

    worst_by_bone: dict[str, dict[str, Any]] = {}
    divergence_count = 0
    for row in samples:
        if float(row["rotation_error_deg"]) > 0.1 or float(row["location_error"]) > 1e-4:
            divergence_count += 1
        name = str(row["public_bone"])
        existing = worst_by_bone.get(name)
        if existing is None or float(row["rotation_error_deg"]) > float(existing["rotation_error_deg"]):
            worst_by_bone[name] = row

    worst = sorted(
        worst_by_bone.values(),
        key=lambda item: float(item["rotation_error_deg"]),
        reverse=True,
    )
    return {
        "schema": "awb-checkpoint-replay-result/v1",
        "source_session_id": script.get("source_session_id"),
        "scrub_action_count": len(actions),
        "executed_frames": executed_frames,
        "sliding_sample_count": len(samples),
        "divergence_count": divergence_count,
        "worst_public_result_errors": worst[:16],
    }

def _gui_replay_failure(
    reason: str,
    *,
    execution_id: str | None = None,
    details: dict[str, Any] | None = None,
    matched_routes: list[dict[str, Any]] | None = None,
    expected_action_count: int = 0,
) -> dict[str, Any]:
    routes = list(matched_routes or ())
    return {
        "schema": _GUI_REPLAY_RESULT_SCHEMA,
        "execution_id": execution_id,
        "status": "FAILED",
        "reason": reason,
        "details": dict(details or {}),
        "matched_action_count": len(routes),
        "expected_action_count": int(expected_action_count),
        "matched_routes": routes,
    }


def _gui_replay_view3d_target(context):
    window = getattr(context, "window", None)
    screen = getattr(context, "screen", None)
    if window is None or screen is None:
        return None
    area = next(
        (
            item
            for item in tuple(getattr(screen, "areas", ()) or ())
            if getattr(item, "type", None) == "VIEW_3D"
        ),
        None,
    )
    if area is None:
        return None
    return (
        window,
        area,
        int(area.x + area.width // 2),
        int(area.y + area.height // 2),
    )


def _validate_gui_replay_payload(
    payload: dict[str, Any],
    *,
    max_events: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if payload.get("schema") != _GUI_REPLAY_SCHEMA:
        raise RuntimeError("AWB GUI replay payload schema is unsupported.")
    if payload.get("target_area") != "VIEW_3D":
        raise RuntimeError("AWB GUI replay v1 only supports VIEW_3D targets.")

    raw_events = payload.get("events")
    raw_actions = payload.get("actions")
    if (
        not isinstance(raw_events, list)
        or not isinstance(raw_actions, list)
        or not raw_events
        or not raw_actions
    ):
        raise RuntimeError("AWB GUI replay requires bounded actions and events.")
    if len(raw_events) > max(1, int(max_events)):
        raise RuntimeError("AWB GUI replay exceeded its bounded event budget.")
    if len(raw_events) != len(raw_actions) * 2:
        raise RuntimeError("AWB GUI replay requires one PRESS/RELEASE pair per action.")

    events: list[dict[str, Any]] = []
    for item in raw_events:
        if not isinstance(item, dict):
            raise TypeError("AWB GUI replay event must be an object.")
        event_type = str(item.get("type") or "")
        value = str(item.get("value") or "")
        if event_type not in _GUI_REPLAY_ALLOWED_KEYS:
            raise RuntimeError(f"AWB GUI replay key is unsupported: {event_type!r}.")
        if value not in _GUI_REPLAY_ALLOWED_VALUES:
            raise RuntimeError(f"AWB GUI replay value is unsupported: {value!r}.")
        events.append(dict(item))

    for index in range(0, len(events), 2):
        press = events[index]
        release = events[index + 1]
        if press.get("value") != "PRESS" or release.get("value") != "RELEASE":
            raise RuntimeError("AWB GUI replay requires PRESS followed by RELEASE.")
        for field in ("stroke_id", "type", "requested_mode"):
            if press.get(field) != release.get(field):
                raise RuntimeError(
                    f"AWB GUI replay PRESS/RELEASE mismatch for {field}."
                )

    actions: list[dict[str, Any]] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            raise TypeError("AWB GUI replay action must be an object.")
        requested_mode = str(item.get("requested_mode") or "")
        key = str(item.get("key") or "")
        if requested_mode not in {"MOVE", "ROTATE", "SCALE"}:
            raise RuntimeError(
                f"AWB GUI replay requested_mode is unsupported: {requested_mode!r}."
            )
        if key not in _GUI_REPLAY_ALLOWED_KEYS:
            raise RuntimeError(f"AWB GUI replay action key is unsupported: {key!r}.")
        actions.append(dict(item))
    return tuple(actions), tuple(events)


def _gui_replay_active_id() -> str | None:
    value = bpy.app.driver_namespace.get(_GUI_REPLAY_NAMESPACE_KEY)
    return str(value) if value else None


def _clear_gui_replay_execution(execution_id: str) -> None:
    if _gui_replay_active_id() == execution_id:
        bpy.app.driver_namespace.pop(_GUI_REPLAY_NAMESPACE_KEY, None)
        set_replay_execution_active(False)


def _gui_replay_timeout_cleanup(execution_id: str):
    _clear_gui_replay_execution(execution_id)


def run_gui_input_replay(
    context=None,
    *,
    gui_replay: dict[str, Any],
    replay_execution_id: str | None = None,
    max_events: int = 128,
    timeout_seconds: float = _GUI_REPLAY_DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Queue evidence-derived GUI events and return a stateless confirmation ticket."""

    context = context or bpy.context
    active_id = _gui_replay_active_id()
    if active_id is not None or replay_execution_active():
        return _gui_replay_failure(
            "GUI_REPLAY_ALREADY_ACTIVE",
            details={"active_execution_id": active_id},
        )

    target = _gui_replay_view3d_target(context)
    if target is None:
        return _gui_replay_failure("EXECUTION_UNAVAILABLE_HEADLESS")
    window, _area, x, y = target
    if not callable(getattr(window, "event_simulate", None)):
        return _gui_replay_failure("EVENT_SIMULATION_UNAVAILABLE")

    try:
        actions, events = _validate_gui_replay_payload(
            gui_replay,
            max_events=max_events,
        )
    except Exception as exc:  # noqa: BLE001 - structured diagnostic failure
        return _gui_replay_failure(
            "INVALID_GUI_REPLAY_PAYLOAD",
            details={"error": f"{type(exc).__name__}: {exc}"},
        )

    expected_blend = str(gui_replay.get("expected_blend") or "")
    actual_blend = str(getattr(bpy.data, "filepath", "") or "")
    if expected_blend and (
        os.path.normcase(os.path.normpath(expected_blend))
        != os.path.normcase(os.path.normpath(actual_blend))
    ):
        return _gui_replay_failure(
            "BASELINE_BLEND_MISMATCH",
            details={"expected": expected_blend, "observed": actual_blend},
            expected_action_count=len(actions),
        )
    expected_mode = str(gui_replay.get("expected_mode") or "")
    actual_mode = str(getattr(context, "mode", "") or "")
    if expected_mode and expected_mode != actual_mode:
        return _gui_replay_failure(
            "BASELINE_MODE_MISMATCH",
            details={"expected": expected_mode, "observed": actual_mode},
            expected_action_count=len(actions),
        )

    expected_context = gui_replay.get("expected_context")
    if isinstance(expected_context, dict):
        expected_active = expected_context.get("active_object")
        if isinstance(expected_active, dict):
            expected_active = expected_active.get("name")
        actual_active = getattr(getattr(context, "active_object", None), "name", None)
        if expected_active and str(expected_active) != str(actual_active or ""):
            return _gui_replay_failure(
                "BASELINE_ACTIVE_OBJECT_MISMATCH",
                details={"expected": expected_active, "observed": actual_active},
                expected_action_count=len(actions),
            )

        expected_pose = tuple(
            str(item.get("bone") if isinstance(item, dict) else item)
            for item in tuple(expected_context.get("selected_pose_bones") or ())
        )
        actual_pose = tuple(
            str(item.name)
            for item in tuple(getattr(context, "selected_pose_bones", ()) or ())
        )
        if expected_pose and expected_pose != actual_pose:
            return _gui_replay_failure(
                "BASELINE_POSE_SELECTION_MISMATCH",
                details={"expected": list(expected_pose), "observed": list(actual_pose)},
                expected_action_count=len(actions),
            )

    recent = read_recent_trace_events(512, latest_session_only=True)
    start_seq = int(recent[-1].get("seq") or 0) if recent else 0
    execution_id = (
        str(replay_execution_id)
        if replay_execution_id
        else f"gui-replay:{uuid4().hex}"
    )
    timeout = max(1.0, float(timeout_seconds))
    deadline = time.monotonic() + timeout

    bpy.app.driver_namespace[_GUI_REPLAY_NAMESPACE_KEY] = execution_id
    set_replay_execution_active(True, execution_id=execution_id)
    try:
        for event in events:
            window.event_simulate(
                type=str(event["type"]),
                value=str(event["value"]),
                x=x,
                y=y,
            )
        bpy.app.timers.register(
            lambda: _gui_replay_timeout_cleanup(execution_id),
            first_interval=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - never leave replay flag active
        _clear_gui_replay_execution(execution_id)
        return _gui_replay_failure(
            "EVENT_SIMULATION_FAILED",
            execution_id=execution_id,
            details={"error": f"{type(exc).__name__}: {exc}"},
            expected_action_count=len(actions),
        )

    return {
        "schema": _GUI_REPLAY_TICKET_SCHEMA,
        "execution_id": execution_id,
        "status": "QUEUED",
        "start_seq": start_seq,
        "deadline_monotonic": deadline,
        "expected_action_count": len(actions),
        "event_count": len(events),
        "target_xy": [x, y],
        "actions": [dict(item) for item in actions],
    }


def _gui_replay_causal_matches(
    ticket: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    execution_id = str(ticket.get("execution_id") or "")
    if not execution_id:
        return [], "INVALID_GUI_REPLAY_TICKET"
    rows = [
        item
        for item in read_recent_trace_events(1024, latest_session_only=True)
        if bool(item.get("replay_execution", False))
        and item.get("replay_execution_id") == execution_id
    ]
    rows.sort(key=lambda item: int(item.get("seq") or 0))

    matched: list[dict[str, Any]] = []
    cursor_seq = 0
    for expected in tuple(ticket.get("actions") or ()):
        if not isinstance(expected, dict):
            return matched, "INVALID_GUI_REPLAY_TICKET"
        requested_mode = str(expected.get("requested_mode") or "")
        ingress = next(
            (
                item
                for item in rows
                if int(item.get("seq") or 0) > cursor_seq
                and item.get("event") == "TRANSFORM_TOOL_INGRESS"
                and str((item.get("data") or {}).get("requested_mode") or "")
                == requested_mode
            ),
            None,
        )
        if ingress is None:
            return matched, None

        trace_id = ingress.get("trace_id")
        terminal = next(
            (
                item
                for item in rows
                if int(item.get("seq") or 0) > int(ingress.get("seq") or 0)
                and item.get("event") == "TRANSFORM_TOOL_TERMINAL"
                and item.get("trace_id") == trace_id
            ),
            None,
        )
        if terminal is None:
            return matched, None

        expected_terminal = expected.get("expected_terminal_status")
        expected_route = expected.get("expected_route_outcome")
        if (
            expected_terminal is not None
            and terminal.get("terminal_status") != expected_terminal
        ):
            return matched, "CAUSAL_TERMINAL_STATUS_MISMATCH"
        if expected_route is not None and terminal.get("route_outcome") != expected_route:
            return matched, "CAUSAL_ROUTE_OUTCOME_MISMATCH"

        matched.append(
            {
                "requested_mode": requested_mode,
                "ingress_seq": ingress.get("seq"),
                "terminal_seq": terminal.get("seq"),
                "observed_trace_id": trace_id,
                "terminal_status": terminal.get("terminal_status"),
                "route_outcome": terminal.get("route_outcome"),
            }
        )
        cursor_seq = int(terminal.get("seq") or cursor_seq)

    return matched, None


def confirm_gui_input_replay(ticket: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ticket, dict) or ticket.get("schema") != _GUI_REPLAY_TICKET_SCHEMA:
        return _gui_replay_failure("INVALID_GUI_REPLAY_TICKET")

    execution_id = str(ticket.get("execution_id") or "")
    expected_count = int(ticket.get("expected_action_count") or 0)
    matched, mismatch = _gui_replay_causal_matches(ticket)

    if mismatch is not None:
        _clear_gui_replay_execution(execution_id)
        return _gui_replay_failure(
            mismatch,
            execution_id=execution_id,
            matched_routes=matched,
            expected_action_count=expected_count,
        )
    if len(matched) == expected_count and expected_count > 0:
        _clear_gui_replay_execution(execution_id)
        return {
            "schema": _GUI_REPLAY_RESULT_SCHEMA,
            "execution_id": execution_id,
            "status": "CONFIRMED",
            "reason": None,
            "details": {},
            "matched_action_count": len(matched),
            "expected_action_count": expected_count,
            "matched_routes": matched,
        }

    deadline = float(ticket.get("deadline_monotonic") or 0.0)
    active_id = _gui_replay_active_id()
    if deadline and time.monotonic() >= deadline:
        _clear_gui_replay_execution(execution_id)
        return _gui_replay_failure(
            "CAUSAL_CONFIRMATION_TIMEOUT",
            execution_id=execution_id,
            matched_routes=matched,
            expected_action_count=expected_count,
        )
    if active_id != execution_id:
        return _gui_replay_failure(
            "GUI_REPLAY_EXECUTION_EXPIRED",
            execution_id=execution_id,
            matched_routes=matched,
            expected_action_count=expected_count,
        )

    return {
        "schema": _GUI_REPLAY_RESULT_SCHEMA,
        "execution_id": execution_id,
        "status": "PENDING",
        "reason": None,
        "details": {},
        "matched_action_count": len(matched),
        "expected_action_count": expected_count,
        "matched_routes": matched,
    }

