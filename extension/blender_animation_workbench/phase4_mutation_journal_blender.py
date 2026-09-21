from __future__ import annotations

from typing import Any

import bpy

from .phase4_mutation_journal import (
    ConstraintMutationReceipt,
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
    RawFieldKind,
    RawFieldMutationReceipt,
    Receipt,
)
from .phase4_verification import (
    Diagnostic,
    DiagnosticSeverity,
    OperationStage,
    RollbackStatus,
)
from .semantic_adapter import channel_binding_token


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


def _find_pointer(collection, pointer: int):
    return next((value for value in collection if _pointer(value) == pointer), None)


def _find_object(pointer: int):
    return _find_pointer(bpy.data.objects, pointer)


def _find_action(pointer: int):
    return _find_pointer(bpy.data.actions, pointer)


def _find_slot(action, pointer: int):
    return _find_pointer(getattr(action, "slots", ()), pointer) if action is not None else None


def _iter_channelbags():
    for action in bpy.data.actions:
        for layer in getattr(action, "layers", ()):
            for strip in getattr(layer, "strips", ()):
                for bag in getattr(strip, "channelbags", ()):
                    yield action, layer, strip, bag


def _find_bag(pointer: int):
    return next(
        ((action, layer, strip, bag) for action, layer, strip, bag in _iter_channelbags() if _pointer(bag) == pointer),
        None,
    )


def _find_fcurve(pointer: int):
    for action, layer, strip, bag in _iter_channelbags():
        curve = _find_pointer(getattr(bag, "fcurves", ()), pointer)
        if curve is not None:
            return action, layer, strip, bag, curve
    return None


def _keys_at_frame(fcurve, frame: float) -> tuple[Any, ...]:
    return tuple(
        point
        for point in getattr(fcurve, "keyframe_points", ())
        if float(point.co.x) == float(frame)
    )


def _assign_key_snapshot_base(point, snapshot: KeySnapshot) -> None:
    snapshot.validate()
    if not snapshot.exists or snapshot.co is None:
        raise RuntimeError("Cannot assign an absent KeySnapshot to a KeyframePoint.")
    point.co = snapshot.co
    if snapshot.handle_left_type is not None:
        point.handle_left_type = snapshot.handle_left_type
    if snapshot.handle_right_type is not None:
        point.handle_right_type = snapshot.handle_right_type
    if snapshot.interpolation is not None:
        point.interpolation = snapshot.interpolation
    if snapshot.easing is not None:
        point.easing = snapshot.easing
    if snapshot.back is not None:
        point.back = snapshot.back
    if snapshot.amplitude is not None:
        point.amplitude = snapshot.amplitude
    if snapshot.period is not None:
        point.period = snapshot.period
    if snapshot.keyframe_type is not None:
        point.type = snapshot.keyframe_type
    if snapshot.select_control_point is not None:
        point.select_control_point = snapshot.select_control_point
    if snapshot.select_left_handle is not None:
        point.select_left_handle = snapshot.select_left_handle
    if snapshot.select_right_handle is not None:
        point.select_right_handle = snapshot.select_right_handle


def _assign_key_snapshot_handles(point, snapshot: KeySnapshot) -> None:
    if snapshot.handle_left is not None:
        point.handle_left = snapshot.handle_left
    if snapshot.handle_right is not None:
        point.handle_right = snapshot.handle_right


def _assign_key_snapshot_post_update(point, snapshot: KeySnapshot) -> None:
    if snapshot.interpolation is not None:
        point.interpolation = snapshot.interpolation
    if snapshot.easing is not None:
        point.easing = snapshot.easing
    if snapshot.back is not None:
        point.back = snapshot.back
    if snapshot.amplitude is not None:
        point.amplitude = snapshot.amplitude
    if snapshot.period is not None:
        point.period = snapshot.period
    if snapshot.keyframe_type is not None:
        point.type = snapshot.keyframe_type
    if snapshot.select_control_point is not None:
        point.select_control_point = snapshot.select_control_point
    if snapshot.select_left_handle is not None:
        point.select_left_handle = snapshot.select_left_handle
    if snapshot.select_right_handle is not None:
        point.select_right_handle = snapshot.select_right_handle


def _assign_key_snapshot(point, snapshot: KeySnapshot) -> None:
    _assign_key_snapshot_base(point, snapshot)
    _assign_key_snapshot_handles(point, snapshot)


def _restore_key_snapshot(fcurve, frame: float, snapshot: KeySnapshot) -> None:
    snapshot.validate()
    matches = _keys_at_frame(fcurve, frame)
    if snapshot.exists:
        if len(matches) > 1:
            raise RuntimeError(f"Multiple keys exist at rollback frame {frame!r}.")
        if matches:
            point = matches[0]
        else:
            assert snapshot.co is not None
            point = fcurve.keyframe_points.insert(
                float(snapshot.co[0]),
                float(snapshot.co[1]),
                options={"FAST"},
                keyframe_type=snapshot.keyframe_type or "KEYFRAME",
            )
            if point is None:
                raise RuntimeError("Blender did not return a recreated KeyframePoint.")
        _assign_key_snapshot(point, snapshot)
        return
    for point in matches:
        fcurve.keyframe_points.remove(point, fast=True)
    if matches:
        fcurve.update()


def _restore_fcurve_key_snapshots(
    fcurve,
    snapshots: tuple[KeySnapshot, ...],
) -> None:
    # Blender 5.2 can invalidate Python KeyframePoint wrappers after a fast
    # removal. Re-resolve the current collection tail for every deletion rather
    # than iterating a tuple of wrappers that may become stale mid-loop.
    while fcurve.keyframe_points:
        fcurve.keyframe_points.remove(fcurve.keyframe_points[-1], fast=True)

    rebuilt: list[tuple[Any, KeySnapshot]] = []
    for snapshot in snapshots:
        snapshot.validate()
        if not snapshot.exists or snapshot.co is None:
            raise RuntimeError("FCurve raw snapshot contains an absent key.")
        point = fcurve.keyframe_points.insert(
            float(snapshot.co[0]),
            float(snapshot.co[1]),
            options={"FAST"},
            keyframe_type=snapshot.keyframe_type or "KEYFRAME",
        )
        if point is None:
            raise RuntimeError("Blender did not return a rebuilt KeyframePoint.")
        rebuilt.append((point, snapshot))

    # Insertions and AUTO/VECTOR handle-type assignments can recalculate
    # neighbouring handles. Freeze all non-handle metadata first, then restore
    # every raw handle coordinate in a final pass after no more type changes can
    # perturb adjacent points.
    for point, snapshot in rebuilt:
        _assign_key_snapshot_base(point, snapshot)
    # Let Blender settle AUTO/VECTOR handles against the fully restored key set.
    # AUTO handles ignore arbitrary raw coordinate writes until this evaluation
    # step, while FREE/ALIGNED coordinates can still be restored exactly after.
    fcurve.update()
    for index, (_point, snapshot) in enumerate(rebuilt):
        point = fcurve.keyframe_points[index]
        _assign_key_snapshot_post_update(point, snapshot)
        _assign_key_snapshot_handles(point, snapshot)


def _point_snapshot_matches(point, snapshot: KeySnapshot) -> bool:
    snapshot.validate()
    if not snapshot.exists or snapshot.co is None:
        return False
    return (
        tuple(float(value) for value in point.co) == snapshot.co
        and (
            snapshot.handle_left is None
            or tuple(float(value) for value in point.handle_left) == snapshot.handle_left
        )
        and (
            snapshot.handle_right is None
            or tuple(float(value) for value in point.handle_right) == snapshot.handle_right
        )
        and (
            snapshot.handle_left_type is None
            or str(point.handle_left_type) == snapshot.handle_left_type
        )
        and (
            snapshot.handle_right_type is None
            or str(point.handle_right_type) == snapshot.handle_right_type
        )
        and (snapshot.interpolation is None or str(point.interpolation) == snapshot.interpolation)
        and (snapshot.easing is None or str(point.easing) == snapshot.easing)
        and (snapshot.back is None or float(point.back) == snapshot.back)
        and (snapshot.amplitude is None or float(point.amplitude) == snapshot.amplitude)
        and (snapshot.period is None or float(point.period) == snapshot.period)
        and (snapshot.keyframe_type is None or str(point.type) == snapshot.keyframe_type)
        and (
            snapshot.select_control_point is None
            or bool(point.select_control_point) is snapshot.select_control_point
        )
        and (
            snapshot.select_left_handle is None
            or bool(point.select_left_handle) is snapshot.select_left_handle
        )
        and (
            snapshot.select_right_handle is None
            or bool(point.select_right_handle) is snapshot.select_right_handle
        )
    )


def _key_snapshot_matches(fcurve, frame: float, snapshot: KeySnapshot) -> bool:
    matches = _keys_at_frame(fcurve, frame)
    if not snapshot.exists:
        return not matches
    return len(matches) == 1 and _point_snapshot_matches(matches[0], snapshot)


def _fcurve_snapshot_matches(fcurve, snapshot: FCurveSnapshot) -> bool:
    snapshot.validate()
    if not snapshot.exists:
        return fcurve is None
    if fcurve is None:
        return False
    group = getattr(fcurve, "group", None)
    points = tuple(fcurve.keyframe_points)
    return (
        str(fcurve.data_path) == snapshot.data_path
        and int(fcurve.array_index) == snapshot.array_index
        and (str(group.name) if group is not None else None) == snapshot.group_name
        and str(fcurve.extrapolation) == snapshot.extrapolation
        and len(points) == len(snapshot.keys)
        and all(
            _point_snapshot_matches(point, key_snapshot)
            for point, key_snapshot in zip(points, snapshot.keys, strict=True)
        )
    )


def _fcurve_snapshot_mismatch_detail(fcurve, snapshot: FCurveSnapshot) -> str:
    if fcurve is None:
        return "FCurve missing"
    points = tuple(fcurve.keyframe_points)
    if len(points) != len(snapshot.keys):
        return f"key count {len(points)} != {len(snapshot.keys)}"
    for index, (point, key) in enumerate(zip(points, snapshot.keys, strict=True)):
        checks = (
            ("co", tuple(float(value) for value in point.co), key.co),
            ("handle_left", tuple(float(value) for value in point.handle_left), key.handle_left),
            ("handle_right", tuple(float(value) for value in point.handle_right), key.handle_right),
            ("handle_left_type", str(point.handle_left_type), key.handle_left_type),
            ("handle_right_type", str(point.handle_right_type), key.handle_right_type),
            ("interpolation", str(point.interpolation), key.interpolation),
            ("easing", str(point.easing), key.easing),
            ("back", float(point.back), key.back),
            ("amplitude", float(point.amplitude), key.amplitude),
            ("period", float(point.period), key.period),
            ("type", str(point.type), key.keyframe_type),
            ("select_control_point", bool(point.select_control_point), key.select_control_point),
            ("select_left_handle", bool(point.select_left_handle), key.select_left_handle),
            ("select_right_handle", bool(point.select_right_handle), key.select_right_handle),
        )
        for field, actual, expected in checks:
            if expected is not None and actual != expected:
                return f"key[{index}] {field}: {actual!r} != {expected!r}"
    return "FCurve metadata/group/extrapolation mismatch"


def _find_target(owner, target_ptr: int):
    if _pointer(owner) == target_ptr:
        return owner
    pose = getattr(owner, "pose", None)
    if pose is not None:
        target = _find_pointer(getattr(pose, "bones", ()), target_ptr)
        if target is not None:
            return target
    return None


def _rotation_attribute(field: RawFieldKind) -> str | None:
    if field in {RawFieldKind.OBJECT_ROTATION_EULER, RawFieldKind.POSE_BONE_ROTATION_EULER}:
        return "rotation_euler"
    if field in {
        RawFieldKind.OBJECT_ROTATION_QUATERNION,
        RawFieldKind.POSE_BONE_ROTATION_QUATERNION,
    }:
        return "rotation_quaternion"
    if field in {
        RawFieldKind.OBJECT_ROTATION_AXIS_ANGLE,
        RawFieldKind.POSE_BONE_ROTATION_AXIS_ANGLE,
    }:
        return "rotation_axis_angle"
    return None


def _raw_attribute(field: RawFieldKind) -> str:
    if field in {RawFieldKind.OBJECT_LOCATION, RawFieldKind.POSE_BONE_LOCATION}:
        return "location"
    rotation = _rotation_attribute(field)
    if rotation is not None:
        return rotation
    if field in {RawFieldKind.OBJECT_SCALE, RawFieldKind.POSE_BONE_SCALE}:
        return "scale"
    pose_ik_fields = {
        RawFieldKind.POSE_BONE_LOCK_IK_X: "lock_ik_x",
        RawFieldKind.POSE_BONE_LOCK_IK_Y: "lock_ik_y",
        RawFieldKind.POSE_BONE_LOCK_IK_Z: "lock_ik_z",
        RawFieldKind.POSE_BONE_USE_IK_LIMIT_X: "use_ik_limit_x",
        RawFieldKind.POSE_BONE_USE_IK_LIMIT_Y: "use_ik_limit_y",
        RawFieldKind.POSE_BONE_USE_IK_LIMIT_Z: "use_ik_limit_z",
        RawFieldKind.POSE_BONE_IK_MIN_X: "ik_min_x",
        RawFieldKind.POSE_BONE_IK_MAX_X: "ik_max_x",
        RawFieldKind.POSE_BONE_IK_MIN_Y: "ik_min_y",
        RawFieldKind.POSE_BONE_IK_MAX_Y: "ik_max_y",
        RawFieldKind.POSE_BONE_IK_MIN_Z: "ik_min_z",
        RawFieldKind.POSE_BONE_IK_MAX_Z: "ik_max_z",
    }
    if field in pose_ik_fields:
        return pose_ik_fields[field]
    if field is RawFieldKind.CONSTRAINT_INFLUENCE:
        return "influence"
    if field is RawFieldKind.CONSTRAINT_MUTE:
        return "mute"
    if field is RawFieldKind.CONSTRAINT_POLE_ANGLE:
        return "pole_angle"
    raise ValueError(f"Unsupported raw field kind: {field.value}")


def _assign_raw(target, field: RawFieldKind, value) -> None:
    attribute = _raw_attribute(field)
    current = getattr(target, attribute)
    if isinstance(value, tuple):
        current[:] = value
    else:
        setattr(target, attribute, value)


def _read_raw(target, field: RawFieldKind):
    value = getattr(target, _raw_attribute(field))
    if isinstance(value, (bool, int, float)):
        return bool(value) if isinstance(value, bool) else float(value)
    return tuple(float(component) for component in value)


def _find_constraint(owner, pointer: int):
    constraint = _find_pointer(getattr(owner, "constraints", ()), pointer)
    if constraint is not None:
        return constraint
    pose = getattr(owner, "pose", None)
    if pose is not None:
        for bone in getattr(pose, "bones", ()):
            constraint = _find_pointer(getattr(bone, "constraints", ()), pointer)
            if constraint is not None:
                return constraint
    return None


def _verify_diag(receipt: Receipt, detail: str) -> Diagnostic:
    return Diagnostic(
        code="I4_ROLLBACK_VERIFY_MISMATCH",
        severity=DiagnosticSeverity.BLOCKER,
        stage=OperationStage.ROLLBACK_VERIFY,
        detail=f"ordinal={receipt.ordinal} {type(receipt).__name__}: {detail}",
        character_id=receipt.scope.character_id,
        rollback_status=RollbackStatus.INCOMPLETE,
    )


class BlenderRollbackExecutor:
    """Blender 5.2 rollback adapter for bounded animation/native I4 receipts."""

    def __init__(self) -> None:
        # Key receipts are verified after parent operation-created FCurves are
        # removed. Remember only FCurves this executor itself successfully
        # removed so a missing parent means verified rollback, never silent loss
        # of a pre-existing user curve.
        self._rolled_back_created_fcurves: set[int] = set()

    def rollback_receipt(self, receipt: Receipt) -> None:
        if isinstance(receipt, KeyMutationReceipt):
            found = _find_fcurve(receipt.fcurve_ptr)
            if found is None:
                raise RuntimeError("Target FCurve no longer exists for key rollback.")
            _action, _layer, _strip, bag, fcurve = found
            if _pointer(bag) != receipt.bag_ptr:
                raise RuntimeError("Target FCurve moved to a different ChannelBag.")
            _restore_key_snapshot(fcurve, receipt.frame, receipt.before)
            return

        if isinstance(receipt, CreatedFCurveReceipt):
            found = _find_fcurve(receipt.fcurve_ptr)
            if found is None:
                self._rolled_back_created_fcurves.add(receipt.fcurve_ptr)
                return
            _action, _layer, _strip, bag, fcurve = found
            if _pointer(bag) != receipt.bag_ptr:
                raise RuntimeError("Operation-created FCurve moved to a different ChannelBag.")
            if str(fcurve.data_path) != receipt.data_path or int(fcurve.array_index) != receipt.array_index:
                raise RuntimeError("Operation-created FCurve address changed before rollback.")
            bag.fcurves.remove(fcurve)
            self._rolled_back_created_fcurves.add(receipt.fcurve_ptr)
            return

        if isinstance(receipt, CreatedChannelBagReceipt):
            found = _find_bag(receipt.bag_ptr)
            if found is None:
                return
            action, _layer, strip, bag = found
            if _pointer(action) != receipt.action_ptr or _pointer(getattr(bag, "slot", None)) != receipt.slot_ptr:
                raise RuntimeError("Operation-created ChannelBag identity/binding changed before rollback.")
            strip.channelbags.remove(bag)
            return

        if isinstance(receipt, CreatedSlotReceipt):
            action = _find_action(receipt.action_ptr)
            slot = _find_slot(action, receipt.slot_ptr)
            if action is None or slot is None:
                return
            owner = _find_object(receipt.owner_ptr)
            if owner is None:
                raise RuntimeError("Animation owner no longer exists for Slot rollback.")
            current = channel_binding_token(owner)
            if current[0] != receipt.owner_binding_after[0] or current[1] != receipt.action_ptr:
                raise RuntimeError("Animation owner no longer uses the operation-created Action.")
            action.slots.remove(slot)
            return

        if isinstance(receipt, CreatedActionReceipt):
            action = _find_action(receipt.action_ptr)
            if action is None:
                return
            owner = _find_object(receipt.owner_ptr)
            if owner is None:
                raise RuntimeError("Animation owner no longer exists for Action rollback.")
            animation_data = getattr(owner, "animation_data", None)
            current_action_ptr = _pointer(animation_data.action) if animation_data is not None else None
            users = int(getattr(action, "users", 0))
            if current_action_ptr == receipt.action_ptr:
                if users > 1:
                    raise RuntimeError("Operation-created Action has been adopted by another user.")
                animation_data.action = None
                bpy.data.actions.remove(action)
                return
            if channel_binding_token(owner) == receipt.owner_binding_before and users == 0:
                bpy.data.actions.remove(action)
                return
            raise RuntimeError("Operation-created Action identity/ownership changed before rollback.")

        if isinstance(receipt, CreatedAnimationDataReceipt):
            owner = _find_object(receipt.owner_ptr)
            if owner is None:
                raise RuntimeError("Animation owner no longer exists for AnimData rollback.")
            if getattr(owner, "animation_data", None) is None:
                return
            if channel_binding_token(owner) != receipt.owner_binding_before:
                raise RuntimeError(
                    "Animation owner has non-original state; refusing to clear operation-created AnimData."
                )
            owner.animation_data_clear()
            return

        if isinstance(receipt, FCurveMutationReceipt):
            found = _find_fcurve(receipt.fcurve_ptr)
            if found is None:
                raise RuntimeError("Mutated FCurve no longer exists for rollback.")
            _action, _layer, _strip, _bag, fcurve = found
            receipt.before.validate()
            if not receipt.before.exists:
                raise RuntimeError("FCurveMutationReceipt requires an existing before-state.")
            if str(fcurve.data_path) != receipt.before.data_path or int(fcurve.array_index) != receipt.before.array_index:
                raise RuntimeError("Mutated FCurve address changed before rollback.")
            if receipt.before.extrapolation is not None:
                fcurve.extrapolation = receipt.before.extrapolation
            before_group = receipt.before.group_name
            current_group = getattr(fcurve, "group", None)
            current_group_name = str(current_group.name) if current_group is not None else None
            if current_group_name != before_group:
                bag = found[3]
                if before_group is None:
                    fcurve.group = None
                else:
                    group = getattr(bag, "groups", None)
                    target_group = group.get(before_group) if group is not None else None
                    if target_group is None:
                        raise RuntimeError("Original FCurve group no longer exists for exact rollback.")
                    fcurve.group = target_group
            _restore_fcurve_key_snapshots(fcurve, receipt.before.keys)
            return

        if isinstance(receipt, RawFieldMutationReceipt):
            owner = _find_object(receipt.owner_ptr)
            if owner is None:
                raise RuntimeError("Raw-field owner no longer exists for rollback.")
            target = _find_target(owner, receipt.target_ptr)
            if target is None:
                raise RuntimeError("Raw-field target no longer exists for rollback.")
            _assign_raw(target, receipt.before.field, receipt.before.value)
            return

        if isinstance(receipt, IDPropertyMutationReceipt):
            owner = _find_object(receipt.owner_ptr)
            if owner is None:
                raise RuntimeError("ID-property owner no longer exists for rollback.")
            target = _find_target(owner, receipt.target_ptr)
            if target is None:
                raise RuntimeError("ID-property target no longer exists for rollback.")
            if receipt.before_exists:
                target[receipt.property_name] = receipt.before_value
            elif receipt.property_name in target:
                del target[receipt.property_name]
            return

        if isinstance(receipt, ConstraintMutationReceipt):
            owner = _find_object(receipt.owner_ptr)
            if owner is None:
                raise RuntimeError("Constraint owner no longer exists for rollback.")
            constraint = _find_constraint(owner, receipt.constraint_ptr)
            if constraint is None:
                raise RuntimeError("Constraint no longer exists for rollback.")
            _assign_raw(constraint, receipt.before.field, receipt.before.value)
            return

        raise TypeError(f"Unsupported I4 receipt: {type(receipt).__name__}")

    def verify_receipt_rolled_back(self, receipt: Receipt) -> tuple[Diagnostic, ...]:
        ok = False
        detail = "unknown receipt verification"

        if isinstance(receipt, KeyMutationReceipt):
            found = _find_fcurve(receipt.fcurve_ptr)
            if receipt.fcurve_ptr in self._rolled_back_created_fcurves:
                ok = found is None
                detail = "parent operation-created FCurve still exists after key rollback"
            else:
                ok = found is not None and _key_snapshot_matches(found[-1], receipt.frame, receipt.before)
                detail = "key raw state does not match before-state"
        elif isinstance(receipt, CreatedFCurveReceipt):
            ok = _find_fcurve(receipt.fcurve_ptr) is None
            detail = "operation-created FCurve still exists"
        elif isinstance(receipt, CreatedChannelBagReceipt):
            ok = _find_bag(receipt.bag_ptr) is None
            detail = "operation-created ChannelBag still exists"
        elif isinstance(receipt, CreatedSlotReceipt):
            action = _find_action(receipt.action_ptr)
            ok = action is None or _find_slot(action, receipt.slot_ptr) is None
            detail = "operation-created Action Slot still exists"
        elif isinstance(receipt, CreatedActionReceipt):
            ok = _find_action(receipt.action_ptr) is None
            detail = "operation-created Action still exists"
        elif isinstance(receipt, CreatedAnimationDataReceipt):
            owner = _find_object(receipt.owner_ptr)
            ok = owner is not None and getattr(owner, "animation_data", None) is None
            detail = "operation-created AnimData still exists"
        elif isinstance(receipt, FCurveMutationReceipt):
            found = _find_fcurve(receipt.fcurve_ptr)
            fcurve = found[-1] if found is not None else None
            ok = _fcurve_snapshot_matches(fcurve, receipt.before)
            detail = _fcurve_snapshot_mismatch_detail(fcurve, receipt.before)
        elif isinstance(receipt, RawFieldMutationReceipt):
            owner = _find_object(receipt.owner_ptr)
            target = _find_target(owner, receipt.target_ptr) if owner is not None else None
            ok = target is not None and _read_raw(target, receipt.before.field) == receipt.before.value
            detail = "raw target field does not match before-state"
        elif isinstance(receipt, IDPropertyMutationReceipt):
            owner = _find_object(receipt.owner_ptr)
            target = _find_target(owner, receipt.target_ptr) if owner is not None else None
            if target is None:
                ok = False
            elif receipt.before_exists:
                ok = (
                    receipt.property_name in target
                    and target.get(receipt.property_name) == receipt.before_value
                )
            else:
                ok = receipt.property_name not in target
            detail = "ID property does not match before-state"
        elif isinstance(receipt, ConstraintMutationReceipt):
            owner = _find_object(receipt.owner_ptr)
            constraint = _find_constraint(owner, receipt.constraint_ptr) if owner is not None else None
            ok = constraint is not None and _read_raw(constraint, receipt.before.field) == receipt.before.value
            detail = "constraint raw field does not match before-state"

        return () if ok else (_verify_diag(receipt, detail),)
