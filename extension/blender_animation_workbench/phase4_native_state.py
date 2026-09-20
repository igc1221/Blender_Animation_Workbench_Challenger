from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import bpy

from .phase4_verification import (
    CanonicalRecord,
    CanonicalSnapshot,
    FingerprintProfile,
    FingerprintRequest,
)
from .semantic_adapter import assigned_channelbag, channel_binding_token, runtime_control_key


def _pointer(value: Any) -> int:
    if value is None:
        return 0
    as_pointer = getattr(value, "as_pointer", None)
    if callable(as_pointer):
        result = int(as_pointer())
        if result:
            return result
    return id(value)


def _key(*parts: Any) -> tuple[str, ...]:
    return tuple(str(part) for part in parts)


def _vector(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(component) for component in value)
    except TypeError:
        return None


def _matrix(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(component) for row in value for component in row)
    except TypeError:
        return None


def _library_name(value: Any) -> str | None:
    library = getattr(value, "library", None)
    return str(getattr(library, "filepath", "") or getattr(library, "name", "")) or None


def _dedupe_runtime(values: Iterable[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[int] = set()
    for value in values:
        if value is None:
            continue
        pointer = _pointer(value)
        if pointer in seen:
            continue
        seen.add(pointer)
        result.append(value)
    return tuple(result)


def _request_objects(request: FingerprintRequest) -> tuple[Any, ...]:
    values: list[Any] = list(request.animation_owners)
    values.extend(request.include_objects)
    values.extend(
        getattr(control, "owner_object", None)
        for control in request.controls
    )
    return _dedupe_runtime(values)


def _global_identity_records() -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    for kind, collection in (
        ("OBJECT", bpy.data.objects),
        ("ARMATURE", bpy.data.armatures),
        ("ACTION", bpy.data.actions),
        ("COLLECTION", bpy.data.collections),
    ):
        rows = tuple(
            sorted(
                (
                    _pointer(item),
                    str(item.name),
                    _library_name(item),
                )
                for item in collection
            )
        )
        records.append((_key("global", kind), rows))
    return records


def _identity_records(request: FingerprintRequest) -> list[CanonicalRecord]:
    records = _global_identity_records()
    for obj in _request_objects(request):
        data = getattr(obj, "data", None)
        records.append(
            (
                _key("object", _pointer(obj)),
                (
                    str(getattr(obj, "name", "")),
                    str(getattr(obj, "type", "")),
                    _pointer(data) if data is not None else 0,
                    _pointer(getattr(obj, "parent", None)),
                    str(getattr(obj, "parent_type", "")),
                    str(getattr(obj, "parent_bone", "")),
                    _library_name(obj),
                    bool(getattr(obj, "is_editable", True)),
                ),
            )
        )
        if getattr(obj, "type", None) != "ARMATURE" or data is None:
            continue
        for bone in getattr(data, "bones", ()):
            records.append(
                (
                    _key("bone", _pointer(obj), _pointer(bone)),
                    (
                        str(bone.name),
                        _pointer(getattr(bone, "parent", None)),
                        bool(getattr(bone, "use_connect", False)),
                        bool(getattr(bone, "use_deform", False)),
                        _vector(getattr(bone, "head_local", None)),
                        _vector(getattr(bone, "tail_local", None)),
                        _matrix(getattr(bone, "matrix_local", None)),
                    ),
                )
            )
    return records


def _keyframe_payload(point: Any) -> tuple:
    return (
        _vector(getattr(point, "co", None)),
        _vector(getattr(point, "handle_left", None)),
        _vector(getattr(point, "handle_right", None)),
        str(getattr(point, "handle_left_type", "")),
        str(getattr(point, "handle_right_type", "")),
        str(getattr(point, "interpolation", "")),
        str(getattr(point, "easing", "")),
        float(getattr(point, "back", 0.0)),
        float(getattr(point, "amplitude", 0.0)),
        float(getattr(point, "period", 0.0)),
        str(getattr(point, "type", "")),
    )


def _fcurve_modifiers(curve: Any) -> tuple:
    return tuple(
        (
            index,
            str(getattr(modifier, "type", "")),
            bool(getattr(modifier, "active", False)),
            bool(getattr(modifier, "mute", False)),
        )
        for index, modifier in enumerate(getattr(curve, "modifiers", ()))
    )


def _raw_animation_records(request: FingerprintRequest) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    for owner in _dedupe_runtime(request.animation_owners):
        anim_data = getattr(owner, "animation_data", None)
        action = getattr(anim_data, "action", None) if anim_data is not None else None
        slot = getattr(anim_data, "action_slot", None) if anim_data is not None else None
        bag = assigned_channelbag(owner) if action is not None else None
        records.append(
            (
                _key("animation-binding", _pointer(owner)),
                (
                    channel_binding_token(owner),
                    str(getattr(action, "name", "")) if action is not None else None,
                    str(getattr(slot, "identifier", "")) if slot is not None else None,
                ),
            )
        )
        if bag is None:
            continue
        for curve in getattr(bag, "fcurves", ()):
            data_path = str(curve.data_path)
            array_index = int(curve.array_index)
            records.append(
                (
                    _key("fcurve", _pointer(owner), data_path, array_index),
                    (
                        str(getattr(getattr(curve, "group", None), "name", "")),
                        str(getattr(curve, "extrapolation", "")),
                        bool(getattr(curve, "mute", False)),
                        bool(getattr(curve, "lock", False)),
                        tuple(_keyframe_payload(point) for point in curve.keyframe_points),
                        _fcurve_modifiers(curve),
                    ),
                )
            )
    return records


def _constraint_payload(constraint: Any) -> tuple:
    return (
        str(getattr(constraint, "type", "")),
        str(getattr(constraint, "name", "")),
        _pointer(getattr(constraint, "target", None)),
        str(getattr(constraint, "subtarget", "") or ""),
        _pointer(getattr(constraint, "pole_target", None)),
        str(getattr(constraint, "pole_subtarget", "") or ""),
        float(getattr(constraint, "influence", 1.0)),
        int(getattr(constraint, "chain_count", 0) or 0),
        bool(getattr(constraint, "use_tail", False)),
        bool(getattr(constraint, "use_stretch", False)),
        bool(getattr(constraint, "use_rotation", False)),
    )


def _raw_pose_records(request: FingerprintRequest) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    seen: set[tuple[int, int]] = set()
    for resolved in request.controls:
        key = runtime_control_key(resolved)
        if key in seen:
            continue
        seen.add(key)
        target = resolved.target
        records.append(
            (
                _key("control", key[0], key[1]),
                (
                    str(getattr(resolved.control.kind, "value", resolved.control.kind)),
                    str(getattr(target, "rotation_mode", "")),
                    _vector(getattr(target, "location", None)),
                    _vector(getattr(target, "rotation_euler", None)),
                    _vector(getattr(target, "rotation_quaternion", None)),
                    _vector(getattr(target, "rotation_axis_angle", None)),
                    _vector(getattr(target, "scale", None)),
                    _matrix(getattr(target, "matrix_basis", None)),
                    tuple(
                        _constraint_payload(constraint)
                        for constraint in getattr(target, "constraints", ())
                    ),
                ),
            )
        )
    return records


def _global_ui_records(scene: Any) -> list[CanonicalRecord]:
    context = bpy.context
    active_object = getattr(context.view_layer.objects, "active", None)
    active_pose_bone = getattr(context, "active_pose_bone", None)
    selected_objects = tuple(sorted(_pointer(obj) for obj in (context.selected_objects or ())))
    selected_pose_bones = tuple(
        sorted(_pointer(bone) for bone in (context.selected_pose_bones or ()))
    )
    tool_settings = scene.tool_settings
    return [
        (
            _key("scene-time"),
            (
                int(scene.frame_current),
                float(scene.frame_subframe),
                int(scene.frame_start),
                int(scene.frame_end),
            ),
        ),
        (
            _key("selection"),
            (
                str(context.mode),
                _pointer(active_object),
                selected_objects,
                _pointer(active_pose_bone),
                selected_pose_bones,
            ),
        ),
        (
            _key("auto-key"),
            (
                bool(getattr(tool_settings, "use_keyframe_insert_auto", False)),
                bool(getattr(tool_settings, "use_keyframe_insert_keying", False)),
                str(getattr(tool_settings, "keyframe_insert_auto_type", "")),
                bool(getattr(scene, "baw_auto_key_enabled", False)),
            ),
        ),
    ]


def _canonical_idprop(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _canonical_idprop(value.to_dict())
    if isinstance(value, dict):
        return tuple(
            (str(key), _canonical_idprop(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    try:
        return tuple(_canonical_idprop(item) for item in value)
    except TypeError:
        return str(value)


def _metadata_records(scene: Any, request: FingerprintRequest) -> list[CanonicalRecord]:
    from .character_metadata import character_ids, character_stamp
    from .rigped_contract import RIGPED_SETUP_PROPERTY

    records: list[CanonicalRecord] = [
        (_key("character-ids"), tuple(character_ids(scene)))
    ]
    for character_id in request.character_ids:
        records.append(
            (
                _key("character-stamp", character_id),
                _canonical_idprop(character_stamp(scene, character_id)),
            )
        )
    for obj in _request_objects(request):
        getter = getattr(obj, "get", None)
        if not callable(getter):
            continue
        descriptor = getter(RIGPED_SETUP_PROPERTY)
        if descriptor is not None:
            records.append(
                (
                    _key("rigped-setup", _pointer(obj)),
                    (_canonical_idprop(descriptor),),
                )
            )
    return records


def _evaluated_records(request: FingerprintRequest) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    seen: set[tuple[int, int]] = set()
    for resolved in request.controls:
        key = runtime_control_key(resolved)
        if key in seen:
            continue
        seen.add(key)
        target = resolved.target
        if target is resolved.owner_object:
            world_matrix = getattr(target, "matrix_world", None)
        else:
            matrix = getattr(target, "matrix", None)
            world_matrix = (
                resolved.owner_object.matrix_world @ matrix
                if matrix is not None
                else None
            )
        records.append(
            (
                _key("evaluated", key[0], key[1]),
                (_matrix(world_matrix),),
            )
        )
    return records


_COLLECTORS = {
    FingerprintProfile.IDENTITY: lambda scene, request: _identity_records(request),
    FingerprintProfile.RAW_ANIM: lambda scene, request: _raw_animation_records(request),
    FingerprintProfile.RAW_POSE: lambda scene, request: _raw_pose_records(request),
    FingerprintProfile.GLOBAL_UI: lambda scene, request: _global_ui_records(scene),
    FingerprintProfile.METADATA: lambda scene, request: _metadata_records(scene, request),
    FingerprintProfile.EVAL_TOLERANT: lambda scene, request: _evaluated_records(request),
}


def capture_native_state(scene: Any, request: FingerprintRequest) -> CanonicalSnapshot:
    if not request.profiles:
        raise ValueError("FingerprintRequest must declare at least one profile.")
    if len(set(request.profiles)) != len(request.profiles):
        raise ValueError("FingerprintRequest profiles must be unique.")

    profiles: list[tuple[FingerprintProfile, tuple[CanonicalRecord, ...]]] = []
    for profile in request.profiles:
        records = _COLLECTORS[profile](scene, request)
        records.sort(key=lambda record: record[0])
        keys = [record[0] for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Recorder produced duplicate keys for {profile.value}.")
        profiles.append((profile, tuple(records)))
    profiles.sort(key=lambda item: tuple(FingerprintProfile).index(item[0]))
    return CanonicalSnapshot(request.label, tuple(profiles))
