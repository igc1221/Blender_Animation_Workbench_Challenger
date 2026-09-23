from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import bpy
from mathutils import Matrix, Quaternion, Vector

from .character_metadata import resolve_character
from .debug_trace import trace_event, trace_exception
from .rigped_contract import RIGPED_SETUP_PROPERTY
from .rigped_fit_runtime import (
    FitCommitResult,
    RigpedFitRuntimeError,
    _animation_signature,
    _descriptor_carrier,
    publish_fit_session_after_native_commit,
)
from .rigped_fit_session import validate_fit_semantic_session
from .rigped_fit_state import (
    FIT_REST_ROUNDTRIP_TOLERANCE,
    DerivedRestPart,
    FitDraft,
    derive_rest_parts,
    validate_fit_draft,
)


class FitCommitTransactionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FitBoneTarget:
    name: str
    head: tuple[float, float, float]
    tail: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    width: float
    depth: float
    use_connect: bool


@dataclass(frozen=True, slots=True)
class FitCommitManifest:
    character_id: str
    targets: tuple[FitBoneTarget, ...]


@dataclass(frozen=True, slots=True)
class FitEditBoneBefore:
    name: str
    head: tuple[float, float, float]
    tail: tuple[float, float, float]
    roll: float
    bbone_x: float
    bbone_z: float
    parent_name: str | None
    use_connect: bool


@dataclass(frozen=True, slots=True)
class FitNativeBeforeImage:
    edit_bones: tuple[FitEditBoneBefore, ...]
    descriptor_exists: bool
    descriptor_value: Any
    constraint_signature: tuple
    animation_signature: tuple
    object_matrix_signature: tuple[float, ...]
    pose_signature: tuple
    active_object: Any
    selected_objects: tuple[Any, ...]


_DIRECT_PREFIXES = ("MCH_", "DEF_", "IK_", "EXP_")


def _vec(value) -> tuple[float, float, float]:
    return tuple(float(item) for item in value)


def _quat(value) -> tuple[float, float, float, float]:
    q = Quaternion(tuple(float(item) for item in value)).normalized()
    return tuple(float(item) for item in q)


def _clone_idproperty(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _clone_idproperty(item) for key, item in value.items()}
    keys = getattr(value, "keys", None)
    getter = getattr(value, "__getitem__", None)
    if callable(keys) and callable(getter):
        return {
            str(key): _clone_idproperty(value[key])
            for key in keys()
        }
    if isinstance(value, (tuple, list)):
        return [_clone_idproperty(item) for item in value]
    return deepcopy(value)


def _primary_maps(draft: FitDraft):
    derived = {part.part_id: part for part in derive_rest_parts(draft)}
    definitions = {part.part_id: part for part in draft.document.parts}
    by_name = {
        definitions[part_id].name_hint: rest
        for part_id, rest in derived.items()
        if definitions[part_id].name_hint
    }
    return definitions, derived, by_name


def _baseline_draft(draft: FitDraft) -> FitDraft:
    return FitDraft(draft.document, draft.document.baseline_values)


def _strip_direct_prefix(name: str) -> str:
    for prefix in _DIRECT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _contact_source_name(name: str, side: str) -> str | None:
    suffix = ".L" if side == "LEFT" else ".R" if side == "RIGHT" else ""
    if "Hand" in name:
        return f"Hand{suffix}"
    if "Foot" in name:
        return f"Foot{suffix}"
    return None


def _endpoint_matches(point, baseline_by_name) -> tuple[tuple[str, str], ...]:
    point_vector = Vector(point)
    matches: list[tuple[str, str]] = []
    for name, part in baseline_by_name.items():
        for endpoint in ("head", "tail"):
            if (point_vector - Vector(getattr(part, endpoint))).length <= FIT_REST_ROUNDTRIP_TOLERANCE:
                matches.append((name, endpoint))
    return tuple(matches)


def _mapped_endpoint(point, baseline_by_name, current_by_name):
    matches = _endpoint_matches(point, baseline_by_name)
    if not matches:
        return None
    current_points = [
        Vector(getattr(current_by_name[name], endpoint))
        for name, endpoint in matches
        if name in current_by_name
    ]
    if not current_points:
        return None
    first = current_points[0]
    for candidate in current_points[1:]:
        if (candidate - first).length > FIT_REST_ROUNDTRIP_TOLERANCE:
            raise FitCommitTransactionError("FIT_F4_ENDPOINT_CORRESPONDENCE_AMBIGUOUS")
    return first


def _exact_primary_driver(bone, baseline_by_name) -> str | None:
    head = Vector(bone.head_local)
    tail = Vector(bone.tail_local)
    matches = [
        name
        for name, part in baseline_by_name.items()
        if (head - Vector(part.head)).length <= FIT_REST_ROUNDTRIP_TOLERANCE
        and (tail - Vector(part.tail)).length <= FIT_REST_ROUNDTRIP_TOLERANCE
    ]
    return matches[0] if len(matches) == 1 else None


def _derived_driver_name(
    bone,
    baseline_by_name,
    current_by_name,
    *,
    side: str,
) -> str | None:
    direct_name = _strip_direct_prefix(str(bone.name))
    if direct_name in baseline_by_name and direct_name in current_by_name:
        return direct_name

    contact_name = _contact_source_name(str(bone.name), side)
    if contact_name in baseline_by_name and contact_name in current_by_name:
        return contact_name

    return _exact_primary_driver(bone, baseline_by_name)


def _derived_orientation(
    bone,
    *,
    target_head: Vector,
    target_tail: Vector,
    baseline_by_name,
    current_by_name,
    driver_name: str | None,
) -> Quaternion:
    base_q = _bone_rest_orientation(bone)
    candidate = base_q
    if driver_name is not None:
        base_driver = Quaternion(baseline_by_name[driver_name].orientation).normalized()
        current_driver = Quaternion(current_by_name[driver_name].orientation).normalized()
        candidate = (current_driver @ base_driver.conjugated() @ base_q).normalized()

    target_direction = target_tail - target_head
    if target_direction.length <= 1e-9:
        raise FitCommitTransactionError(f"FIT_F4_ZERO_LENGTH_TARGET:{bone.name}")
    candidate_y = candidate @ Vector((0.0, 1.0, 0.0))
    if candidate_y.length <= 1e-9:
        raise FitCommitTransactionError(f"FIT_F4_ORIENTATION_INVALID:{bone.name}")
    swing = candidate_y.normalized().rotation_difference(target_direction.normalized())
    return (swing @ candidate).normalized()


def _derived_correspondence_target(
    bone,
    baseline_by_name,
    current_by_name,
    *,
    usage: str,
    semantic_key: str,
    side: str,
    definition_by_name,
) -> FitBoneTarget | None:
    driver_name = _derived_driver_name(
        bone,
        baseline_by_name,
        current_by_name,
        side=side,
    )
    direct_driver = current_by_name.get(driver_name) if driver_name is not None else None
    if direct_driver is not None:
        head = Vector(direct_driver.head)
        tail = Vector(direct_driver.tail)
    else:
        head = _mapped_endpoint(bone.head_local, baseline_by_name, current_by_name)
        tail = _mapped_endpoint(bone.tail_local, baseline_by_name, current_by_name)
        if head is None or tail is None:
            return None
    orientation = _derived_orientation(
        bone,
        target_head=head,
        target_tail=tail,
        baseline_by_name=baseline_by_name,
        current_by_name=current_by_name,
        driver_name=driver_name,
    )
    owned_clone = (
        driver_name is not None
        and usage in {"MECHANISM", "DEFORM", "REFERENCE"}
        and driver_name in definition_by_name
        and str(definition_by_name[driver_name].semantic_key) == semantic_key
    )
    driver = current_by_name.get(driver_name) if owned_clone else None
    return FitBoneTarget(
        str(bone.name),
        _vec(head),
        _vec(tail),
        _quat(orientation),
        float(driver.width) if driver is not None else float(bone.bbone_x),
        float(driver.depth) if driver is not None else float(bone.bbone_z),
        bool(driver.connected) if driver is not None else bool(bone.use_connect),
    )


def _primary_name_for(definitions, *, semantic_key: str, side: str) -> str:
    matches = [
        str(definition.name_hint)
        for definition in definitions.values()
        if str(definition.semantic_key) == semantic_key
        and str(definition.side) == side
        and definition.name_hint
    ]
    if len(matches) != 1:
        raise FitCommitTransactionError(
            f"FIT_F4_PRIMARY_ROLE_AMBIGUOUS:{semantic_key}:{side}"
        )
    return matches[0]


def _pole_source_names(definitions, side: str, semantic_key: str):
    if semantic_key == "awb.arm":
        return (
            _primary_name_for(
                definitions,
                semantic_key="awb.upper_arm",
                side=side,
            ),
            _primary_name_for(
                definitions,
                semantic_key="awb.forearm",
                side=side,
            ),
        )
    if semantic_key == "awb.leg":
        return (
            _primary_name_for(
                definitions,
                semantic_key="awb.thigh",
                side=side,
            ),
            _primary_name_for(
                definitions,
                semantic_key="awb.calf",
                side=side,
            ),
        )
    return None


def _bone_rest_orientation(bone) -> Quaternion:
    return bone.matrix_local.to_quaternion().normalized()


def _canonical_forearm_target_orientation(
    parent_target: FitBoneTarget,
    forearm_target: FitBoneTarget,
) -> tuple[float, float, float, float]:
    """Return the fitted ForeArm X-hinge frame, or preserve input near singularity."""

    upper = Vector(forearm_target.head) - Vector(parent_target.head)
    lower = Vector(forearm_target.tail) - Vector(forearm_target.head)
    if upper.length <= 1e-6 or lower.length <= 1e-6:
        return forearm_target.orientation

    bend_normal = upper.cross(lower)
    if bend_normal.length <= 1e-5:
        return forearm_target.orientation

    local_y = lower.normalized()
    # Geometry sign is deliberate: local X = -N makes negative-X rotation
    # increase the already-authored elbow bend on both mirrored sides.
    local_x = -bend_normal.normalized()
    local_x = local_x - local_y * float(local_x.dot(local_y))
    if local_x.length <= 1e-5:
        return forearm_target.orientation
    local_x.normalize()

    local_z = local_x.cross(local_y)
    if local_z.length <= 1e-5:
        return forearm_target.orientation
    local_z.normalize()

    # mathutils.Matrix consumes rows; write local basis axes as columns.
    basis = Matrix(
        (
            (local_x.x, local_y.x, local_z.x),
            (local_x.y, local_y.y, local_z.y),
            (local_x.z, local_y.z, local_z.z),
        )
    )
    return _quat(basis.to_quaternion().normalized())


def _length(part: DerivedRestPart) -> float:
    return (Vector(part.tail) - Vector(part.head)).length


def _pole_target(
    *,
    bone,
    baseline_upper: DerivedRestPart,
    baseline_lower: DerivedRestPart,
    target_upper: DerivedRestPart,
    target_lower: DerivedRestPart,
) -> FitBoneTarget:
    base_upper_q = Quaternion(baseline_upper.orientation).normalized()
    target_upper_q = Quaternion(target_upper.orientation).normalized()
    pole_q = _bone_rest_orientation(bone)

    base_joint = Vector(baseline_lower.head)
    target_joint = Vector(target_lower.head)
    base_length = max(_length(baseline_upper), 1e-9)
    scale = _length(target_upper) / base_length

    local_head = base_upper_q.conjugated() @ (Vector(bone.head_local) - base_joint)
    local_tail = base_upper_q.conjugated() @ (Vector(bone.tail_local) - base_joint)
    head = target_joint + (target_upper_q @ (local_head * scale))
    tail = target_joint + (target_upper_q @ (local_tail * scale))
    delta_q = (target_upper_q @ base_upper_q.conjugated()).normalized()
    orientation = (delta_q @ pole_q).normalized()
    return FitBoneTarget(
        str(bone.name),
        _vec(head),
        _vec(tail),
        _quat(orientation),
        float(bone.bbone_x),
        float(bone.bbone_z),
        bool(bone.use_connect),
    )


def compute_fit_commit_manifest(scene, semantic_session) -> FitCommitManifest:
    validate_fit_draft(semantic_session.draft)
    view = resolve_character(scene, semantic_session.character_id)
    if view.issues:
        raise FitCommitTransactionError("FIT_F4_CHARACTER_INCOHERENT")

    definitions, current_by_id, current_by_name = _primary_maps(semantic_session.draft)
    definition_by_name = {
        definition.name_hint: definition
        for definition in definitions.values()
        if definition.name_hint
    }
    _base_definitions, _baseline_by_id, baseline_by_name = _primary_maps(
        _baseline_draft(semantic_session.draft)
    )
    resolved = dict(view.resolved_bindings)
    targets: dict[str, FitBoneTarget] = {}
    forearm_target_names: set[str] = set()

    for binding in view.definition.bindings:
        resolved_binding = resolved.get(binding.binding_id)
        target = getattr(resolved_binding, "target", None) if resolved_binding is not None else None
        bone = getattr(target, "bone", None) if target is not None else None
        if bone is None:
            continue

        usage = str(getattr(binding.usage, "value", binding.usage))
        side = str(getattr(binding.side, "value", binding.side))
        semantic_key = str(binding.semantic_key)
        if semantic_key == "awb.forearm":
            forearm_target_names.add(str(bone.name))

        if usage == "PRIMARY":
            source = current_by_id.get(str(binding.binding_id))
            if source is None:
                continue
            targets[str(bone.name)] = FitBoneTarget(
                str(bone.name),
                tuple(float(value) for value in source.head),
                tuple(float(value) for value in source.tail),
                tuple(float(value) for value in source.orientation),
                float(source.width),
                float(source.depth),
                bool(source.connected),
            )
            continue

        if usage == "POLE":
            pair = _pole_source_names(definitions, side, semantic_key)
            if pair is not None:
                upper_name, lower_name = pair
                if (
                    upper_name in baseline_by_name
                    and lower_name in baseline_by_name
                    and upper_name in current_by_name
                    and lower_name in current_by_name
                ):
                    targets[str(bone.name)] = _pole_target(
                        bone=bone,
                        baseline_upper=baseline_by_name[upper_name],
                        baseline_lower=baseline_by_name[lower_name],
                        target_upper=current_by_name[upper_name],
                        target_lower=current_by_name[lower_name],
                    )
                    continue

        derived_target = _derived_correspondence_target(
            bone,
            baseline_by_name,
            current_by_name,
            usage=usage,
            semantic_key=semantic_key,
            side=side,
            definition_by_name=definition_by_name,
        )
        if derived_target is not None:
            targets[str(bone.name)] = derived_target

    required_names = {
        str(bone.name)
        for binding in view.definition.bindings
        if (
            (resolved_binding := resolved.get(binding.binding_id)) is not None
            and (target := getattr(resolved_binding, "target", None)) is not None
            and (bone := getattr(target, "bone", None)) is not None
        )
    }
    missing_targets = sorted(required_names - set(targets))
    if missing_targets:
        raise FitCommitTransactionError(
            "FIT_F4_DERIVED_TARGET_UNRESOLVED:" + ",".join(missing_targets)
        )

    primary_names = {definition.name_hint for definition in definitions.values()}
    missing_primary = sorted(name for name in primary_names if name and name not in targets)
    if missing_primary:
        raise FitCommitTransactionError(
            "FIT_F4_PRIMARY_TARGET_MISSING:" + ",".join(missing_primary)
        )

    rig_bones = semantic_session.rig_object.data.bones
    for forearm_name in sorted(forearm_target_names):
        forearm_target = targets.get(forearm_name)
        rest_bone = rig_bones.get(forearm_name)
        parent_name = (
            str(rest_bone.parent.name)
            if rest_bone is not None and rest_bone.parent is not None
            else ""
        )
        parent_target = targets.get(parent_name)
        if forearm_target is None or parent_target is None:
            raise FitCommitTransactionError(
                f"FIT_F4_FOREARM_FRAME_PARENT_MISSING:{forearm_name}"
            )
        targets[forearm_name] = FitBoneTarget(
            forearm_target.name,
            forearm_target.head,
            forearm_target.tail,
            _canonical_forearm_target_orientation(parent_target, forearm_target),
            forearm_target.width,
            forearm_target.depth,
            forearm_target.use_connect,
        )

    return FitCommitManifest(
        character_id=semantic_session.character_id,
        targets=tuple(sorted(targets.values(), key=lambda item: item.name)),
    )


def _constraint_signature(rig) -> tuple:
    rows = []
    pose = getattr(rig, "pose", None)
    for pose_bone in getattr(pose, "bones", ()) if pose is not None else ():
        for index, constraint in enumerate(getattr(pose_bone, "constraints", ())):
            rows.append(
                (
                    str(pose_bone.name),
                    int(index),
                    str(getattr(constraint, "type", "")),
                    str(getattr(constraint, "name", "")),
                    str(getattr(getattr(constraint, "target", None), "name", "")),
                    str(getattr(constraint, "subtarget", "") or ""),
                    str(getattr(getattr(constraint, "pole_target", None), "name", "")),
                    str(getattr(constraint, "pole_subtarget", "") or ""),
                    float(getattr(constraint, "pole_angle", 0.0)),
                    float(getattr(constraint, "influence", 1.0)),
                    int(getattr(constraint, "chain_count", 0) or 0),
                    bool(getattr(constraint, "use_tail", False)),
                    bool(getattr(constraint, "use_stretch", False)),
                    bool(getattr(constraint, "use_rotation", False)),
                    str(getattr(constraint, "target_space", "")),
                    str(getattr(constraint, "owner_space", "")),
                    str(getattr(constraint, "mix_mode", "")),
                )
            )
    return tuple(rows)


def _matrix_signature(matrix) -> tuple[float, ...]:
    return tuple(
        round(float(matrix[row][column]), 9)
        for row in range(4)
        for column in range(4)
    )


def _pose_signature(rig) -> tuple:
    pose = getattr(rig, "pose", None)
    rows = []
    for bone in getattr(pose, "bones", ()) if pose is not None else ():
        custom_shape = getattr(bone, "custom_shape", None)
        rows.append(
            (
                str(bone.name),
                str(getattr(bone, "rotation_mode", "")),
                tuple(float(value) for value in bone.location),
                tuple(float(value) for value in bone.rotation_euler),
                tuple(float(value) for value in bone.rotation_quaternion),
                tuple(float(value) for value in bone.rotation_axis_angle),
                tuple(float(value) for value in bone.scale),
                tuple(bool(value) for value in bone.lock_location),
                tuple(bool(value) for value in bone.lock_rotation),
                tuple(bool(value) for value in bone.lock_scale),
                bool(getattr(bone, "lock_rotation_w", False)),
                str(getattr(custom_shape, "name", "")) if custom_shape is not None else None,
            )
        )
    return tuple(rows)


def _select_only(rig) -> None:
    if bpy.context.object is not None and bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in tuple(bpy.context.selected_objects):
        obj.select_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig


def _capture_edit_bones(rig) -> tuple[FitEditBoneBefore, ...]:
    _select_only(rig)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        return tuple(
            FitEditBoneBefore(
                name=str(bone.name),
                head=_vec(bone.head),
                tail=_vec(bone.tail),
                roll=float(bone.roll),
                bbone_x=float(bone.bbone_x),
                bbone_z=float(bone.bbone_z),
                parent_name=str(bone.parent.name) if bone.parent is not None else None,
                use_connect=bool(bone.use_connect),
            )
            for bone in rig.data.edit_bones
        )
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def capture_fit_native_before_image(scene, semantic_session) -> FitNativeBeforeImage:
    rig = semantic_session.rig_object
    view = resolve_character(scene, semantic_session.character_id)
    carrier = _descriptor_carrier(view)
    exists = RIGPED_SETUP_PROPERTY in carrier
    value = _clone_idproperty(carrier.get(RIGPED_SETUP_PROPERTY)) if exists else None
    active = bpy.context.view_layer.objects.active
    selected = tuple(bpy.context.selected_objects)
    constraints = _constraint_signature(rig)
    animation = _animation_signature(view)
    object_matrix_signature = _matrix_signature(rig.matrix_world)
    pose_signature = _pose_signature(rig)
    try:
        edit_bones = _capture_edit_bones(rig)
    finally:
        _restore_context_objects(active, selected)
    return FitNativeBeforeImage(
        edit_bones=edit_bones,
        descriptor_exists=exists,
        descriptor_value=value,
        constraint_signature=constraints,
        animation_signature=animation,
        object_matrix_signature=object_matrix_signature,
        pose_signature=pose_signature,
        active_object=active,
        selected_objects=selected,
    )


def _restore_context_objects(active, selected) -> None:
    if bpy.context.object is not None and bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in tuple(bpy.context.selected_objects):
        obj.select_set(False)
    for obj in selected:
        try:
            obj.select_set(True)
        except ReferenceError:
            continue
    try:
        bpy.context.view_layer.objects.active = active
    except ReferenceError:
        bpy.context.view_layer.objects.active = None


def _restore_edit_bones(rig, snapshot: tuple[FitEditBoneBefore, ...]) -> None:
    _select_only(rig)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        edit_bones = rig.data.edit_bones
        if {str(bone.name) for bone in edit_bones} != {item.name for item in snapshot}:
            raise FitCommitTransactionError("FIT_F4_ROLLBACK_TOPOLOGY_CHANGED")
        for item in snapshot:
            bone = edit_bones.get(item.name)
            if bone is None:
                raise FitCommitTransactionError("FIT_F4_ROLLBACK_BONE_MISSING")
            bone.use_connect = False
            bone.head = item.head
            bone.tail = item.tail
            bone.roll = item.roll
            bone.bbone_x = item.bbone_x
            bone.bbone_z = item.bbone_z
        for item in snapshot:
            bone = edit_bones[item.name]
            bone.parent = edit_bones.get(item.parent_name) if item.parent_name else None
            bone.use_connect = item.use_connect
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def _edit_bone_before_images_match(
    current: tuple[FitEditBoneBefore, ...],
    expected: tuple[FitEditBoneBefore, ...],
) -> bool:
    current_by_name = {item.name: item for item in current}
    expected_by_name = {item.name: item for item in expected}
    if set(current_by_name) != set(expected_by_name):
        return False
    for name, wanted in expected_by_name.items():
        actual = current_by_name[name]
        if not _vector_close(actual.head, wanted.head):
            return False
        if not _vector_close(actual.tail, wanted.tail):
            return False
        if not math.isclose(float(actual.roll), float(wanted.roll), abs_tol=1e-6):
            return False
        if not math.isclose(float(actual.bbone_x), float(wanted.bbone_x), abs_tol=1e-6):
            return False
        if not math.isclose(float(actual.bbone_z), float(wanted.bbone_z), abs_tol=1e-6):
            return False
        if actual.parent_name != wanted.parent_name:
            return False
        if bool(actual.use_connect) != bool(wanted.use_connect):
            return False
    return True


def _verify_restored_before_image(scene, semantic_session, before: FitNativeBeforeImage) -> None:
    rig = semantic_session.rig_object
    current_edit_bones = _capture_edit_bones(rig)
    if not _edit_bone_before_images_match(current_edit_bones, before.edit_bones):
        raise FitCommitTransactionError("FIT_F4_ROLLBACK_REST_MISMATCH")

    view = resolve_character(scene, semantic_session.character_id)
    carrier = _descriptor_carrier(view)
    descriptor_exists = RIGPED_SETUP_PROPERTY in carrier
    descriptor_value = (
        _clone_idproperty(carrier.get(RIGPED_SETUP_PROPERTY))
        if descriptor_exists
        else None
    )
    if descriptor_exists != before.descriptor_exists:
        raise FitCommitTransactionError("FIT_F4_ROLLBACK_DESCRIPTOR_PRESENCE_MISMATCH")
    if descriptor_value != before.descriptor_value:
        raise FitCommitTransactionError("FIT_F4_ROLLBACK_DESCRIPTOR_MISMATCH")
    if _constraint_signature(rig) != before.constraint_signature:
        raise FitCommitTransactionError("FIT_F4_ROLLBACK_CONSTRAINT_MISMATCH")
    if _animation_signature(view) != before.animation_signature:
        raise FitCommitTransactionError("FIT_F4_ROLLBACK_ANIMATION_MISMATCH")
    if _matrix_signature(rig.matrix_world) != before.object_matrix_signature:
        raise FitCommitTransactionError("FIT_F4_ROLLBACK_OBJECT_MATRIX_MISMATCH")
    if _pose_signature(rig) != before.pose_signature:
        raise FitCommitTransactionError("FIT_F4_ROLLBACK_POSE_MISMATCH")


def restore_fit_native_before_image(scene, semantic_session, before: FitNativeBeforeImage) -> None:
    rig = semantic_session.rig_object
    _restore_edit_bones(rig, before.edit_bones)
    view = resolve_character(scene, semantic_session.character_id)
    carrier = _descriptor_carrier(view)
    if before.descriptor_exists:
        carrier[RIGPED_SETUP_PROPERTY] = _clone_idproperty(before.descriptor_value)
    elif RIGPED_SETUP_PROPERTY in carrier:
        del carrier[RIGPED_SETUP_PROPERTY]
    bpy.context.view_layer.update()
    try:
        _verify_restored_before_image(scene, semantic_session, before)
    finally:
        _restore_context_objects(before.active_object, before.selected_objects)


def _apply_manifest(rig, manifest: FitCommitManifest, before: FitNativeBeforeImage) -> None:
    target_by_name = {target.name: target for target in manifest.targets}
    before_by_name = {item.name: item for item in before.edit_bones}
    for item in before.edit_bones:
        if (
            item.use_connect
            and item.name not in target_by_name
            and item.parent_name in target_by_name
        ):
            parent_before = before_by_name.get(item.parent_name)
            parent_target = target_by_name[item.parent_name]
            if (
                parent_before is not None
                and not _vector_close(parent_before.tail, parent_target.tail)
            ):
                raise FitCommitTransactionError(
                    f"FIT_F4_NON_TARGET_CONNECTED_CHILD:{item.name}"
                )

    _select_only(rig)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        edit_bones = rig.data.edit_bones
        for bone in edit_bones:
            bone.use_connect = False

        for target in manifest.targets:
            bone = edit_bones.get(target.name)
            if bone is None:
                raise FitCommitTransactionError(f"FIT_F4_TARGET_BONE_MISSING:{target.name}")

        for target in manifest.targets:
            bone = edit_bones[target.name]
            bone.head = target.head
            bone.tail = target.tail
            direction = Vector(target.tail) - Vector(target.head)
            if direction.length <= 1e-9:
                raise FitCommitTransactionError(f"FIT_F4_ZERO_LENGTH_TARGET:{target.name}")
            target_q = Quaternion(target.orientation).normalized()
            target_z = target_q @ Vector((0.0, 0.0, 1.0))
            bone.align_roll(target_z)
            bone.bbone_x = max(float(target.width), 1e-6)
            bone.bbone_z = max(float(target.depth), 1e-6)

        for item in before.edit_bones:
            if item.name not in target_by_name:
                edit_bones[item.name].use_connect = bool(item.use_connect)
        for target in manifest.targets:
            edit_bones[target.name].use_connect = bool(target.use_connect)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def _vector_close(a, b, tolerance=FIT_REST_ROUNDTRIP_TOLERANCE) -> bool:
    return (Vector(a) - Vector(b)).length <= float(tolerance)


def _orientation_close(a, b, tolerance=FIT_REST_ROUNDTRIP_TOLERANCE) -> bool:
    qa = Quaternion(a).normalized()
    qb = Quaternion(b).normalized()
    dot = abs(float(qa.dot(qb)))
    return abs(1.0 - min(1.0, dot)) <= float(tolerance)


def _verify_manifest(scene, semantic_session, manifest: FitCommitManifest, before: FitNativeBeforeImage) -> None:
    rig = semantic_session.rig_object
    target_names = {target.name for target in manifest.targets}
    current_edit_bones = _capture_edit_bones(rig)
    current_non_targets = tuple(
        item for item in current_edit_bones if item.name not in target_names
    )
    before_non_targets = tuple(
        item for item in before.edit_bones if item.name not in target_names
    )
    if not _edit_bone_before_images_match(
        current_non_targets,
        before_non_targets,
    ):
        raise FitCommitTransactionError("FIT_F4_NON_TARGET_REST_CHANGED")

    bones = rig.data.bones
    for target in manifest.targets:
        bone = bones.get(target.name)
        if bone is None:
            raise FitCommitTransactionError(f"FIT_F4_VERIFY_BONE_MISSING:{target.name}")
        if not _vector_close(bone.head_local, target.head):
            raise FitCommitTransactionError(f"FIT_F4_VERIFY_HEAD_MISMATCH:{target.name}")
        if not _vector_close(bone.tail_local, target.tail):
            raise FitCommitTransactionError(f"FIT_F4_VERIFY_TAIL_MISMATCH:{target.name}")
        if not _orientation_close(
            bone.matrix_local.to_quaternion(),
            target.orientation,
        ):
            raise FitCommitTransactionError(
                f"FIT_F4_VERIFY_ORIENTATION_MISMATCH:{target.name}"
            )
        if not math.isclose(float(bone.bbone_x), float(target.width), abs_tol=1e-6):
            raise FitCommitTransactionError(f"FIT_F4_VERIFY_WIDTH_MISMATCH:{target.name}")
        if not math.isclose(float(bone.bbone_z), float(target.depth), abs_tol=1e-6):
            raise FitCommitTransactionError(f"FIT_F4_VERIFY_DEPTH_MISMATCH:{target.name}")
        if bool(bone.use_connect) != bool(target.use_connect):
            raise FitCommitTransactionError(
                f"FIT_F4_VERIFY_CONNECT_MISMATCH:{target.name}"
            )

    if _constraint_signature(rig) != before.constraint_signature:
        raise FitCommitTransactionError("FIT_F4_CONSTRAINT_CHANGED")
    view = resolve_character(scene, semantic_session.character_id)
    if _animation_signature(view) != before.animation_signature:
        raise FitCommitTransactionError("FIT_F4_ANIMATION_CHANGED")
    if _matrix_signature(rig.matrix_world) != before.object_matrix_signature:
        raise FitCommitTransactionError("FIT_F4_OBJECT_MATRIX_CHANGED")
    if _pose_signature(rig) != before.pose_signature:
        raise FitCommitTransactionError("FIT_F4_POSE_CHANGED")


def commit_fit_semantic_session_atomic(context, semantic_session) -> FitCommitResult:
    issues = validate_fit_semantic_session(context, semantic_session)
    if issues:
        raise FitCommitTransactionError(
            "FIT_F4_SESSION_STALE:" + ",".join(str(issue) for issue in issues)
        )
    if (
        semantic_session.active_move_gesture is not None
        or semantic_session.active_rotate_gesture is not None
        or semantic_session.active_scale_gesture is not None
    ):
        raise FitCommitTransactionError("FIT_F4_GESTURE_ACTIVE")

    operation_id = (
        f"fit-f4:{semantic_session.character_id}:"
        f"{semantic_session.revision}:{semantic_session.preview_serial}"
    )
    trace_event(
        "FIT",
        "FIT_F4_COMMIT_BEGIN",
        operation_id=operation_id,
        context=context,
        fit_revision=int(semantic_session.revision),
        preview_serial=int(semantic_session.preview_serial),
    )
    manifest = compute_fit_commit_manifest(context.scene, semantic_session)
    before = capture_fit_native_before_image(context.scene, semantic_session)
    issues = validate_fit_semantic_session(context, semantic_session)
    if issues:
        raise FitCommitTransactionError(
            "FIT_F4_BEFORE_IMAGE_STALE:" + ",".join(str(issue) for issue in issues)
        )

    try:
        _apply_manifest(semantic_session.rig_object, manifest, before)
        context.view_layer.update()
        _verify_manifest(context.scene, semantic_session, manifest, before)
        trace_event(
            "FIT",
            "FIT_F4_NATIVE_VERIFY_PASS",
            operation_id=operation_id,
            context=context,
            target_count=len(manifest.targets),
        )
        result = publish_fit_session_after_native_commit(
            context.scene,
            semantic_session.token,
        )
    except Exception as exc:
        trace_exception(
            "FIT",
            "FIT_F4_COMMIT_FAIL",
            exc,
            operation_id=operation_id,
            context=context,
        )
        try:
            restore_fit_native_before_image(context.scene, semantic_session, before)
        except (
            FitCommitTransactionError,
            RigpedFitRuntimeError,
            RuntimeError,
            ValueError,
            TypeError,
            ReferenceError,
        ) as rollback_exc:
            trace_exception(
                "FIT",
                "FIT_F4_ROLLBACK_FAIL",
                rollback_exc,
                operation_id=operation_id,
                context=context,
            )
            raise FitCommitTransactionError(
                f"FIT_F4_ROLLBACK_FAILED:{type(rollback_exc).__name__}:{rollback_exc}"
            ) from exc
        trace_event(
            "FIT",
            "FIT_F4_ROLLBACK_SUCCESS",
            operation_id=operation_id,
            context=context,
        )
        if isinstance(exc, (FitCommitTransactionError, RigpedFitRuntimeError)):
            raise
        raise FitCommitTransactionError(
            f"FIT_F4_COMMIT_FAILED:{type(exc).__name__}:{exc}"
        ) from exc

    trace_event(
        "FIT",
        "FIT_F4_COMMIT_SUCCESS",
        operation_id=operation_id,
        context=context,
        target_count=len(manifest.targets),
        setup_revision=int(result.setup_revision),
        structural_change=bool(result.decision.structural_change),
    )
    return result
