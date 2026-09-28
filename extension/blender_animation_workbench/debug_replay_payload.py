from __future__ import annotations

from typing import Any

from .phase4_contact_model import ContactKeyType

DIRECT_ROTATE_RECORDED_RESULT_SCHEMA = "awb-direct-rotate-recorded-result/v1"

_RECORDED_POSE_REQUIRED_FIELDS = {
    "object",
    "bone",
    "location",
    "rotation_mode",
    "rotation_property",
    "rotation",
}
_RECORDED_SLIDING_REQUIRED_FIELDS = {
    "mapping_id",
    "solver_object",
    "solver_bone",
    "pole_angle",
    "ik_influence",
    "ik_mute",
    "terminal_ik_influence",
    "terminal_ik_mute",
    "feedback_mutes",
    "hinge_settings",
    "contact_state",
}
_ROTATION_PROPERTY_BY_MODE = {
    "QUATERNION": "rotation_quaternion",
    "AXIS_ANGLE": "rotation_axis_angle",
    "XYZ": "rotation_euler",
    "XZY": "rotation_euler",
    "YXZ": "rotation_euler",
    "YZX": "rotation_euler",
    "ZXY": "rotation_euler",
    "ZYX": "rotation_euler",
}


def recorded_contact_type(value: Any, *, field_name: str) -> ContactKeyType | None:
    if value is None:
        return None
    try:
        return ContactKeyType(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "AWB semantic replay Contact INVALID_RECORDED_RESULT: "
            f"{field_name}={value!r} is not a valid Contact type."
        ) from exc


def recorded_mapping_contact_types(
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
                recorded_contact_type(
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


def recorded_direct_binding_ids(action: dict[str, Any]) -> tuple[str, ...] | None:
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


def validate_recorded_pose_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(_RECORDED_POSE_REQUIRED_FIELDS.difference(payload))
    if missing:
        raise RuntimeError(
            f"AWB recorded Rotate result pose state is incomplete: {missing!r}."
        )

    object_name = str(payload.get("object") or "")
    bone_name = str(payload.get("bone") or "")
    if not object_name or not bone_name:
        raise RuntimeError("AWB recorded Rotate result has an empty pose identity.")

    try:
        location = tuple(float(value) for value in tuple(payload["location"]))
        rotation = tuple(float(value) for value in tuple(payload["rotation"]))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"AWB recorded Rotate result has non-numeric pose data for {bone_name!r}."
        ) from exc
    if len(location) != 3:
        raise RuntimeError(
            f"AWB recorded Rotate result has invalid location for {bone_name!r}."
        )

    rotation_mode = str(payload.get("rotation_mode") or "")
    rotation_property = str(payload.get("rotation_property") or "")
    expected_property = _ROTATION_PROPERTY_BY_MODE.get(rotation_mode)
    if expected_property is None or rotation_property != expected_property:
        raise RuntimeError(
            "AWB recorded Rotate result has inconsistent rotation representation "
            f"for {bone_name!r}."
        )
    expected_size = 4 if rotation_property != "rotation_euler" else 3
    if len(rotation) != expected_size:
        raise RuntimeError(
            f"AWB recorded Rotate result has invalid rotation for {bone_name!r}."
        )

    return {
        "key": (object_name, bone_name),
        "object": object_name,
        "bone": bone_name,
        "location": location,
        "rotation_mode": rotation_mode,
        "rotation_property": rotation_property,
        "rotation": rotation,
    }


def _validate_recorded_sliding_payload(item: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(_RECORDED_SLIDING_REQUIRED_FIELDS.difference(item))
    if missing:
        raise RuntimeError(
            f"AWB recorded Rotate Sliding state is incomplete: {missing!r}."
        )

    mapping_id = str(item["mapping_id"])
    feedback_mutes = tuple(item["feedback_mutes"])
    if any(not isinstance(value, bool) for value in feedback_mutes):
        raise RuntimeError(
            f"AWB recorded Rotate feedback coverage mismatch for {mapping_id!r}."
        )

    hinge_settings = tuple(item["hinge_settings"])
    if (
        len(hinge_settings) != 12
        or any(not isinstance(value, bool) for value in hinge_settings[:6])
    ):
        raise RuntimeError(
            f"AWB recorded Rotate hinge settings are incomplete for {mapping_id!r}."
        )

    try:
        normalized_hinge_settings = (
            *hinge_settings[:6],
            *(float(value) for value in hinge_settings[6:]),
        )
        pole_angle = float(item["pole_angle"])
        ik_influence = float(item["ik_influence"])
        terminal_ik_influence = float(item["terminal_ik_influence"])
        contact_state = (
            None
            if item["contact_state"] is None
            else float(item["contact_state"])
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"AWB recorded Rotate Sliding numeric state is invalid for {mapping_id!r}."
        ) from exc

    if not isinstance(item["ik_mute"], bool) or not isinstance(
        item["terminal_ik_mute"],
        bool,
    ):
        raise TypeError(
            f"AWB recorded Rotate Sliding mute state is invalid for {mapping_id!r}."
        )

    return {
        "mapping_id": mapping_id,
        "solver_object": str(item.get("solver_object") or ""),
        "solver_bone": str(item.get("solver_bone") or ""),
        "feedback_mutes": feedback_mutes,
        "hinge_settings": normalized_hinge_settings,
        "contact_state": contact_state,
        "pole_angle": pole_angle,
        "ik_influence": ik_influence,
        "ik_mute": item["ik_mute"],
        "terminal_ik_influence": terminal_ik_influence,
        "terminal_ik_mute": item["terminal_ik_mute"],
    }


def validate_direct_rotate_recorded_result_payload(
    payload: dict[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != DIRECT_ROTATE_RECORDED_RESULT_SCHEMA
    ):
        raise RuntimeError("AWB recorded Rotate result schema is unsupported.")

    raw_states = tuple(payload.get("pose_states") or ())
    if not raw_states:
        raise RuntimeError("AWB recorded Rotate result has no pose states.")
    if any(not isinstance(item, dict) for item in raw_states):
        raise RuntimeError("AWB recorded Rotate result pose state is malformed.")

    pose_rows = tuple(validate_recorded_pose_state_payload(item) for item in raw_states)
    pose_keys = tuple(row["key"] for row in pose_rows)
    if len(set(pose_keys)) != len(pose_keys):
        raise RuntimeError("AWB recorded Rotate result has duplicate pose identities.")

    raw_sliding = tuple(payload.get("sliding") or ())
    if any(not isinstance(item, dict) for item in raw_sliding):
        raise RuntimeError("AWB recorded Rotate Sliding state is malformed.")

    recorded_mapping_ids = tuple(
        str(item.get("mapping_id") or "")
        for item in raw_sliding
    )
    if any(not mapping_id for mapping_id in recorded_mapping_ids):
        raise RuntimeError("AWB recorded Rotate Sliding mapping identity is empty.")
    if len(set(recorded_mapping_ids)) != len(recorded_mapping_ids):
        raise RuntimeError("AWB recorded Rotate result has duplicate Sliding mappings.")

    sliding_rows = tuple(_validate_recorded_sliding_payload(item) for item in raw_sliding)
    return pose_rows, sliding_rows
