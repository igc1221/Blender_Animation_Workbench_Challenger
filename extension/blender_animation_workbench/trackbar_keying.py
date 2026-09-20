from __future__ import annotations

from time import monotonic
from typing import ClassVar

import bpy
from bpy.app.handlers import persistent

from .semantic_adapter import (
    assigned_channelbag,
    channel_binding_token,
    channels_for_property,
    control_context_for_context,
    control_property_path,
    keying_controls_for_context,
    resolve_control_target,
    rotation_property,
    runtime_control_key,
)
from .trackbar_model import clear_key_selection_for_context

AWB_AUTO_KEYING_SET_ID = "BAW_AutoKey"
_RIGPED_SETUP_PROPERTY = "awb_rigped_setup"

_KEYMAP_ITEMS: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []
_AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
_AUTO_KEY_POSTPROCESS_DELAY = 0.06
_AUTO_KEY_LAST_TRANSFORM_CHANGE = 0.0
_AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False
_AUTO_KEY_DATA_SIGNATURE: tuple | None = None
_AUTO_KEY_TRANSFORM_SNAPSHOT: dict[tuple[int, int], tuple] = {}
_AUTO_KEY_ANIMATED_FAMILIES: set[tuple[tuple[int, int], str]] = set()
_AUTO_KEY_FRAME_BASELINE_TRANSFORM: dict[tuple[int, int], tuple] = {}
_AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES: set[tuple[tuple[int, int], str]] = set()
_AUTO_KEY_FRAME_BASELINE_FRAME: int | None = None
_AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE: tuple | None = None
_AUTO_KEY_INSERT_GUARD = False


def _rotation_path(target) -> str:
    """Compatibility wrapper for the centralized Blender addressing contract."""
    return rotation_property(target)


def _auto_key_controls_for_context(context):
    """Return ordinary AWB Auto Key controls, excluding generated Rigped controls.

    During the Rigped Animate rebuild, generated Rigped controls do not use the
    general AWB Auto Key path. Custom/external bones on the same Armature remain
    eligible because they are not owned by the generated Rigped binding set.
    """
    controls = tuple(keying_controls_for_context(context))
    if getattr(context, "mode", "") != "POSE" or not controls:
        return controls
    from .rigped_animation_baseline import split_rigped_controls

    _rigped, external = split_rigped_controls(context, controls)
    return external


def _transform_family_signature(target) -> tuple:
    rotation_path = _rotation_path(target)
    rotation_value = getattr(target, rotation_path)
    return (
        tuple(round(float(value), 9) for value in target.location),
        str(rotation_path),
        tuple(round(float(value), 9) for value in rotation_value),
        tuple(round(float(value), 9) for value in target.scale),
    )


def _capture_auto_key_transform_snapshot(context) -> dict[tuple[int, int], tuple]:
    return {
        runtime_control_key(resolved): _transform_family_signature(resolved.target)
        for resolved in _auto_key_controls_for_context(context)
    }


def _changed_transform_families(previous: tuple, current: tuple) -> tuple[bool, bool, bool]:
    position = previous[0] != current[0]
    rotation = previous[1] != current[1] or previous[2] != current[2]
    scale = previous[3] != current[3]
    return position, rotation, scale


def _active_operator_identifier(context) -> str:
    operator = getattr(context, "active_operator", None)
    if operator is None:
        return ""
    identifier = str(getattr(operator, "bl_idname", ""))
    if identifier:
        return identifier
    return str(getattr(getattr(operator, "bl_rna", None), "identifier", ""))


def _active_operator_is_awb_trajectory_edit(context) -> bool:
    identifier = _active_operator_identifier(context)
    return identifier.startswith("BAW_OT_") and "trajectory" in identifier.lower()


def _insert_transform_key(
    target,
    frame: int,
    *,
    position: bool,
    rotation: bool,
    scale: bool,
) -> None:
    if position:
        target.keyframe_insert(data_path="location", frame=frame)
    if rotation:
        target.keyframe_insert(data_path=_rotation_path(target), frame=frame)
    if scale:
        target.keyframe_insert(data_path="scale", frame=frame)


def _full_transform_data_path(target, property_name: str) -> str:
    resolved = resolve_control_target(target)
    if resolved is None:
        return property_name
    return control_property_path(resolved, property_name)


def _animated_families_from_signature(
    signature: tuple | None,
) -> set[tuple[tuple[int, int], str]]:
    return {
        (row[0], str(row[1]))
        for row in signature or ()
        if row[3]
    }


def _animated_families_for_context(context) -> set[tuple[tuple[int, int], str]]:
    """Return existing animated transform families without scanning key values.

    Frame navigation only needs to know whether a transform family already owns
    an FCurve before the next edit.  The previous baseline path rebuilt the full
    Auto-Key data signature, including every keyframe point value, on every
    frame change.  That made simple time scrubbing scale with animation size.
    """
    families: set[tuple[tuple[int, int], str]] = set()
    seen: set[tuple[int, int]] = set()
    for resolved in _auto_key_controls_for_context(context):
        owner = resolved.owner_object
        prefix = str(resolved.data_path_prefix or "")
        control_key = runtime_control_key(resolved)
        if control_key in seen:
            continue
        seen.add(control_key)
        channelbag = assigned_channelbag(owner)
        if channelbag is None:
            continue
        for fcurve in channelbag.fcurves:
            if not fcurve.keyframe_points:
                continue
            data_path = str(fcurve.data_path)
            if prefix:
                if not data_path.startswith(f"{prefix}."):
                    continue
            elif data_path.startswith('pose.bones['):
                continue
            families.add((control_key, data_path))
    return families


def _auto_key_context_binding_signature(context) -> tuple:
    """Capture the selected control set plus each control's current animation binding."""
    rows = []
    for resolved in _auto_key_controls_for_context(context):
        rows.append(
            (
                runtime_control_key(resolved),
                channel_binding_token(resolved.owner_object),
            )
        )
    rows.sort(key=lambda row: row[0])
    return tuple(rows)


def _auto_key_control_signature_from_binding_signature(signature: tuple | None) -> tuple:
    """Strip Action/Slot identity while preserving the selected runtime controls."""
    return tuple(row[0] for row in signature or ())


def _auto_key_is_first_binding_creation(
    baseline_signature: tuple | None,
    current_signature: tuple | None,
) -> bool:
    """Return True only for same-control transitions from no Action to a new Action."""
    baseline = tuple(baseline_signature or ())
    current = tuple(current_signature or ())
    if len(baseline) != len(current):
        return False

    changed = False
    for before, after in zip(baseline, current, strict=True):
        if before[0] != after[0]:
            return False
        before_binding = before[1]
        after_binding = after[1]
        if before_binding == after_binding:
            continue
        changed = True
        if before_binding[1] is not None or after_binding[1] is None:
            return False
    return changed


def _reset_auto_key_frame_baseline(
    context,
    *,
    transform_snapshot: dict[tuple[int, int], tuple] | None = None,
) -> None:
    global _AUTO_KEY_FRAME_BASELINE_TRANSFORM
    global _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES, _AUTO_KEY_FRAME_BASELINE_FRAME
    global _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE

    scene = getattr(context, "scene", None)
    if scene is None:
        _AUTO_KEY_FRAME_BASELINE_TRANSFORM = {}
        _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES = set()
        _AUTO_KEY_FRAME_BASELINE_FRAME = None
        _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE = None
        return
    _AUTO_KEY_FRAME_BASELINE_TRANSFORM = (
        dict(transform_snapshot)
        if transform_snapshot is not None
        else _capture_auto_key_transform_snapshot(context)
    )
    _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES = _animated_families_for_context(context)
    _AUTO_KEY_FRAME_BASELINE_FRAME = int(scene.frame_current)
    _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE = _auto_key_context_binding_signature(context)


def _rewrite_key_values_at_frame(
    resolved,
    property_name: str,
    frame: int,
    values: tuple,
) -> None:
    for channel in channels_for_property(resolved, property_name):
        fcurve = channel.fcurve
        axis = channel.array_index
        if axis < 0 or axis >= len(values):
            continue
        for key in fcurve.keyframe_points:
            if abs(float(key.co.x) - float(frame)) > 1e-4:
                continue
            value = float(values[axis])
            delta = value - float(key.co.y)
            key.co.y = value
            key.handle_left.y = float(key.handle_left.y) + delta
            key.handle_right.y = float(key.handle_right.y) + delta
            break
        fcurve.update()


def _seed_auto_key_default_frame(
    resolved,
    animated_families: set[tuple[tuple[int, int], str]],
    previous_transform: tuple,
    *,
    position: bool,
    rotation: bool,
    scale: bool,
    frame: int,
) -> bool:
    """Match 3ds Max first-key behavior by preserving the pre-transform value at 0F."""
    if frame == 0:
        return False

    control_key = runtime_control_key(resolved)
    rotation_property = _rotation_path(resolved.target)
    location_path = _full_transform_data_path(resolved.target, "location")
    rotation_path = _full_transform_data_path(resolved.target, rotation_property)
    scale_path = _full_transform_data_path(resolved.target, "scale")

    seed_position = bool(
        position and (control_key, location_path) not in animated_families
    )
    seed_rotation = bool(
        rotation and (control_key, rotation_path) not in animated_families
    )
    seed_scale = bool(
        scale and (control_key, scale_path) not in animated_families
    )
    if not (seed_position or seed_rotation or seed_scale):
        return False

    _insert_transform_key(
        resolved.target,
        0,
        position=seed_position,
        rotation=seed_rotation,
        scale=seed_scale,
    )
    if seed_position:
        _rewrite_key_values_at_frame(resolved, "location", 0, previous_transform[0])
    if seed_rotation:
        _rewrite_key_values_at_frame(resolved, rotation_property, 0, previous_transform[2])
    if seed_scale:
        _rewrite_key_values_at_frame(resolved, "scale", 0, previous_transform[3])
    return True


def _add_keying_set_path(ks, data, property_name: str) -> None:
    resolved = resolve_control_target(data)
    if resolved is None:
        return
    owner_object = resolved.owner_object
    path = control_property_path(resolved, property_name)
    group_name = str(getattr(resolved.target, "name", ""))
    if group_name:
        ks.paths.add(
            owner_object,
            path,
            index=-1,
            group_method="NAMED",
            group_name=group_name,
        )
    else:
        ks.paths.add(owner_object, path, index=-1)


def _redraw_view3d(context) -> None:
    screen = getattr(context, "screen", None)
    if screen is None:
        area = getattr(context, "area", None)
        if area is not None:
            area.tag_redraw()
        return
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()


def _awb_auto_keying_set(scene):
    for keying_set in scene.keying_sets_all:
        type_info = getattr(keying_set, "type_info", None)
        if type_info is not None and type_info.bl_idname == AWB_AUTO_KEYING_SET_ID:
            return keying_set
    return None


def _animation_preferences():
    preferences = getattr(bpy.context, "preferences", None)
    return getattr(preferences, "edit", None) if preferences is not None else None


_KEY_CHANNEL_BITS = {
    "LOCATION": 1 << 0,
    "ROTATION": 1 << 1,
    "SCALE": 1 << 2,
    "ROTATE_MODE": 1 << 3,
    "CUSTOM_PROPS": 1 << 4,
}


def _encode_key_insert_channels(channels) -> int:
    mask = 0
    for channel in channels:
        mask |= _KEY_CHANNEL_BITS.get(str(channel), 0)
    return mask


def _decode_key_insert_channels(mask: int) -> set[str]:
    return {
        channel
        for channel, bit in _KEY_CHANNEL_BITS.items()
        if int(mask) & int(bit)
    }


def _awb_key_insert_channels(scene) -> set[str]:
    channels: set[str] = set()
    if scene.baw_key_position:
        channels.add("LOCATION")
    if scene.baw_key_rotation:
        channels.add("ROTATION")
    if scene.baw_key_scale:
        channels.add("SCALE")
    return channels


def _apply_awb_auto_key_runtime(scene) -> bool:
    """Arm native Auto Key only for controls owned by the ordinary AWB path.

    Generated Rigped pose controls are explicitly excluded during the Animate
    rebuild.  Their public key authoring is owned by the Rigped C command, so
    Blender's native Auto Key must be physically disabled whenever any selected
    Pose control belongs to the generated Rigped binding set.  External/custom
    bones on the same Armature re-enable the ordinary native Auto Key path when
    selected without Rigped controls.
    """
    tool_settings = scene.tool_settings
    if bool(tool_settings.use_keyframe_insert_keyingset):
        tool_settings.use_keyframe_insert_keyingset = False
    if str(tool_settings.auto_keying_mode) != "ADD_REPLACE_KEYS":
        tool_settings.auto_keying_mode = "ADD_REPLACE_KEYS"

    context = bpy.context
    native_auto_allowed = True
    if getattr(context, "scene", None) is scene:
        active = getattr(context, "active_object", None)
        getter = getattr(active, "get", None) if active is not None else None
        if (
            active is not None
            and getattr(active, "type", None) == "ARMATURE"
            and callable(getter)
            and bool(getter(_RIGPED_SETUP_PROPERTY))
        ):
            # Hard rebuild boundary: while a generated Rigped is the active
            # animation object, Blender's global native Auto Key must stay off.
            # Rigped animation keys are authored only by the dedicated Rigped
            # path; ordinary/custom-bone Auto Key will be reintroduced after
            # the baseline contract is closed.
            native_auto_allowed = False
        elif getattr(context, "mode", "") == "POSE":
            controls = tuple(keying_controls_for_context(context))
            if controls:
                from .rigped_animation_baseline import split_rigped_controls

                rigped, _external = split_rigped_controls(context, controls)
                native_auto_allowed = not bool(rigped)
    if bool(tool_settings.use_keyframe_insert_auto) != bool(native_auto_allowed):
        tool_settings.use_keyframe_insert_auto = native_auto_allowed

    edit_preferences = _animation_preferences()
    if edit_preferences is not None:
        if hasattr(edit_preferences, "key_insert_channels"):
            desired_channels = _awb_key_insert_channels(scene)
            if set(edit_preferences.key_insert_channels) != desired_channels:
                edit_preferences.key_insert_channels = desired_channels
        if (
            hasattr(edit_preferences, "use_keyframe_insert_available")
            and bool(edit_preferences.use_keyframe_insert_available)
        ):
            edit_preferences.use_keyframe_insert_available = False
        if (
            hasattr(edit_preferences, "use_auto_keyframe_insert_needed")
            and not bool(edit_preferences.use_auto_keyframe_insert_needed)
        ):
            edit_preferences.use_auto_keyframe_insert_needed = True
    return True


def _restore_previous_auto_key_settings(scene) -> None:
    tool_settings = scene.tool_settings
    if not getattr(scene, "baw_auto_key_prev_valid", False):
        return

    tool_settings.use_keyframe_insert_keyingset = bool(
        scene.baw_auto_key_prev_use_keying_set
    )
    tool_settings.auto_keying_mode = (
        "REPLACE_KEYS"
        if scene.baw_auto_key_prev_replace_only
        else "ADD_REPLACE_KEYS"
    )
    keying_sets_all = scene.keying_sets_all
    if hasattr(keying_sets_all, "active_index"):
        try:
            keying_sets_all.active_index = int(scene.baw_auto_key_prev_keying_set_index)
        except (TypeError, ValueError):
            pass

    edit_preferences = _animation_preferences()
    if edit_preferences is not None:
        if hasattr(edit_preferences, "key_insert_channels"):
            edit_preferences.key_insert_channels = _decode_key_insert_channels(
                scene.baw_auto_key_prev_insert_channels_mask
            )
        if hasattr(edit_preferences, "use_keyframe_insert_available"):
            edit_preferences.use_keyframe_insert_available = bool(
                scene.baw_auto_key_prev_insert_available
            )
        if hasattr(edit_preferences, "use_auto_keyframe_insert_needed"):
            edit_preferences.use_auto_keyframe_insert_needed = bool(
                scene.baw_auto_key_prev_insert_needed
            )
    scene.baw_auto_key_prev_valid = False


def resync_awb_auto_key_after_external_evaluation(context) -> bool:
    """Restore AWB's native Auto Key runtime without discarding transform work.

    External evaluation must never cancel a pending transform transaction or
    replace its pre-transform baseline. Only repair the Blender ToolSettings
    runtime when it actually diverged from AWB's visible AUTO state.
    """
    scene = getattr(context, "scene", None)
    if scene is None or not bool(getattr(scene, "baw_auto_key_enabled", False)):
        return False

    tool_settings = scene.tool_settings
    runtime_matches = (
        bool(tool_settings.use_keyframe_insert_auto)
        and str(tool_settings.auto_keying_mode) == "ADD_REPLACE_KEYS"
        and not bool(tool_settings.use_keyframe_insert_keyingset)
    )
    if runtime_matches:
        return True
    return _apply_awb_auto_key_runtime(scene)


def resync_awb_auto_key_after_context_change(context) -> bool:
    """Rebase Auto Key on the current selected controls and animation bindings.

    Character/native selection and Action/Slot reassignment are context changes,
    not transforms. Any pending timer from the old context must be discarded so
    it cannot supplement keys onto the newly selected/bound controls.
    """
    global _AUTO_KEY_POSTPROCESS_TIMER_PENDING, _AUTO_KEY_DATA_SIGNATURE
    global _AUTO_KEY_TRANSFORM_SNAPSHOT, _AUTO_KEY_ANIMATED_FAMILIES
    global _AUTO_KEY_LAST_TRANSFORM_CHANGE, _AUTO_KEY_NATIVE_TRANSFORM_LATCHED

    scene = getattr(context, "scene", None)
    if scene is None or not bool(getattr(scene, "baw_auto_key_enabled", False)):
        return False
    if bpy.app.timers.is_registered(_run_auto_key_postprocess_timer):
        bpy.app.timers.unregister(_run_auto_key_postprocess_timer)
    _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
    _AUTO_KEY_LAST_TRANSFORM_CHANGE = 0.0
    _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False
    _apply_awb_auto_key_runtime(scene)
    _AUTO_KEY_DATA_SIGNATURE = _auto_key_data_signature(context)
    _AUTO_KEY_ANIMATED_FAMILIES = _animated_families_from_signature(
        _AUTO_KEY_DATA_SIGNATURE
    )
    _AUTO_KEY_TRANSFORM_SNAPSHOT = _capture_auto_key_transform_snapshot(context)
    _reset_auto_key_frame_baseline(context)
    return True


def set_awb_auto_key(context, enabled: bool) -> bool:
    global _AUTO_KEY_DATA_SIGNATURE, _AUTO_KEY_TRANSFORM_SNAPSHOT
    global _AUTO_KEY_ANIMATED_FAMILIES
    global _AUTO_KEY_FRAME_BASELINE_TRANSFORM
    global _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES, _AUTO_KEY_FRAME_BASELINE_FRAME
    global _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE
    global _AUTO_KEY_NATIVE_TRANSFORM_LATCHED

    _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False
    scene = getattr(context, "scene", None)
    if scene is None:
        return False
    tool_settings = scene.tool_settings

    if enabled:
        if not (
            scene.baw_key_position
            or scene.baw_key_rotation
            or scene.baw_key_scale
        ):
            return False

        if not getattr(scene, "baw_auto_key_enabled", False):
            keying_sets_all = scene.keying_sets_all
            scene.baw_auto_key_prev_valid = True
            scene.baw_auto_key_prev_use_keying_set = bool(
                tool_settings.use_keyframe_insert_keyingset
            )
            scene.baw_auto_key_prev_replace_only = (
                tool_settings.auto_keying_mode == "REPLACE_KEYS"
            )
            scene.baw_auto_key_prev_keying_set_index = int(
                getattr(keying_sets_all, "active_index", 0)
            )
            edit_preferences = _animation_preferences()
            if edit_preferences is not None:
                scene.baw_auto_key_prev_insert_channels_mask = _encode_key_insert_channels(
                    getattr(edit_preferences, "key_insert_channels", set())
                )
                scene.baw_auto_key_prev_insert_available = bool(
                    getattr(edit_preferences, "use_keyframe_insert_available", False)
                )
                scene.baw_auto_key_prev_insert_needed = bool(
                    getattr(edit_preferences, "use_auto_keyframe_insert_needed", True)
                )

        if not _apply_awb_auto_key_runtime(scene):
            _restore_previous_auto_key_settings(scene)
            scene.baw_auto_key_enabled = False
            return False
        scene.baw_auto_key_enabled = True
        _AUTO_KEY_DATA_SIGNATURE = _auto_key_data_signature(context)
        _AUTO_KEY_ANIMATED_FAMILIES = _animated_families_from_signature(
            _AUTO_KEY_DATA_SIGNATURE
        )
        _AUTO_KEY_TRANSFORM_SNAPSHOT = _capture_auto_key_transform_snapshot(context)
        _reset_auto_key_frame_baseline(context)
    else:
        tool_settings.use_keyframe_insert_auto = False
        _restore_previous_auto_key_settings(scene)
        scene.baw_auto_key_enabled = False
        _AUTO_KEY_DATA_SIGNATURE = None
        _AUTO_KEY_TRANSFORM_SNAPSHOT = {}
        _AUTO_KEY_ANIMATED_FAMILIES = set()
        _AUTO_KEY_FRAME_BASELINE_TRANSFORM = {}
        _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES = set()
        _AUTO_KEY_FRAME_BASELINE_FRAME = None
        _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE = None

    if bool(getattr(scene, "baw_trajectory_edit_mode", False)):
        # Edit Path owns the viewport transform affordance. AUTO must not
        # unhide Blender's original object/bone tool gizmo while path editing.
        from .trajectory_edit import _set_native_tool_gizmo_hidden

        _set_native_tool_gizmo_hidden(context, hidden=True)

    _redraw_view3d(context)
    return True


def _auto_key_data_signature(context) -> tuple:
    rows: list[tuple] = []
    seen: set[tuple[int, int]] = set()
    for resolved in _auto_key_controls_for_context(context):
        owner = resolved.owner_object
        prefix = str(resolved.data_path_prefix or "")
        control_key = runtime_control_key(resolved)
        if control_key in seen:
            continue
        seen.add(control_key)
        channelbag = assigned_channelbag(owner)
        if channelbag is None:
            continue
        for fcurve in channelbag.fcurves:
            data_path = str(fcurve.data_path)
            if prefix:
                if not data_path.startswith(f"{prefix}."):
                    continue
            elif data_path.startswith('pose.bones['):
                continue
            rows.append(
                (
                    control_key,
                    data_path,
                    int(fcurve.array_index),
                    tuple(
                        (round(float(key.co.x), 6), round(float(key.co.y), 9))
                        for key in fcurve.keyframe_points
                    ),
                )
            )
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return tuple(rows)


def _transform_modal_is_active(context) -> bool:
    """Return True while Blender still owns a native transform/gizmo modal."""
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return False
    if bool(getattr(wm, "baw_trackbar_frame_drag_active", False)):
        # The Track Bar current-time control is also a Blender gizmo and appears
        # as GIZMO_OT_tweak. It is navigation, never a transform edit.
        return False
    for window in wm.windows:
        for operator in window.modal_operators:
            identifiers = {
                str(getattr(operator, "bl_idname", "")),
                str(getattr(operator, "idname", "")),
                str(getattr(getattr(operator, "bl_rna", None), "identifier", "")),
            }
            lowered = {identifier.lower() for identifier in identifiers if identifier}
            if any(
                identifier in {
                    "transform_ot_translate",
                    "transform_ot_rotate",
                    "transform_ot_resize",
                    "transform_ot_trackball",
                }
                or "gizmo_tweak" in identifier
                or "rigped_fk_joint_move_axis" in identifier
                for identifier in lowered
            ):
                return True
    return False


def _active_generated_rigped(context) -> bool:
    active = getattr(context, "active_object", None)
    if active is None or getattr(active, "type", None) != "ARMATURE":
        return False
    getter = getattr(active, "get", None)
    return bool(callable(getter) and getter(_RIGPED_SETUP_PROPERTY))


def _last_operator_is_transform(context) -> bool:
    operator = getattr(context, "active_operator", None)
    if operator is None:
        operators = getattr(getattr(context, "window_manager", None), "operators", None)
        if operators:
            operator = operators[-1]
    if operator is None:
        return False
    identifiers = {
        str(getattr(operator, "bl_idname", "")),
        str(getattr(getattr(operator, "bl_rna", None), "identifier", "")),
    }
    return any(
        identifier in {
            "TRANSFORM_OT_translate",
            "TRANSFORM_OT_rotate",
            "TRANSFORM_OT_resize",
            "TRANSFORM_OT_trackball",
        }
        or "rigped_fk_joint_move_axis" in identifier.lower()
        for identifier in identifiers
    )


def _seed_first_animated_families_after_transform(context) -> bool:
    global _AUTO_KEY_INSERT_GUARD

    scene = getattr(context, "scene", None)
    current_context_signature = _auto_key_context_binding_signature(context)
    baseline_controls = _auto_key_control_signature_from_binding_signature(
        _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE
    )
    current_controls = _auto_key_control_signature_from_binding_signature(
        current_context_signature
    )
    if (
        scene is None
        or _AUTO_KEY_INSERT_GUARD
        or not _AUTO_KEY_FRAME_BASELINE_TRANSFORM
        or _AUTO_KEY_FRAME_BASELINE_FRAME != int(scene.frame_current)
        or baseline_controls != current_controls
    ):
        return False

    current = _capture_auto_key_transform_snapshot(context)
    changed_any = False
    _AUTO_KEY_INSERT_GUARD = True
    try:
        for resolved in _auto_key_controls_for_context(context):
            control_key = runtime_control_key(resolved)
            before = _AUTO_KEY_FRAME_BASELINE_TRANSFORM.get(control_key)
            after = current.get(control_key)
            if before is None or after is None:
                continue
            position, rotation, scale = _changed_transform_families(before, after)
            position = bool(position and scene.baw_key_position)
            rotation = bool(rotation and scene.baw_key_rotation)
            scale = bool(scale and scene.baw_key_scale)
            if not (position or rotation or scale):
                continue
            if _seed_auto_key_default_frame(
                resolved,
                _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES,
                before,
                position=position,
                rotation=rotation,
                scale=scale,
                frame=int(scene.frame_current),
            ):
                changed_any = True
            if position:
                _AUTO_KEY_ANIMATED_FAMILIES.add(
                    (control_key, _full_transform_data_path(resolved.target, "location"))
                )
            if rotation:
                _AUTO_KEY_ANIMATED_FAMILIES.add(
                    (
                        control_key,
                        _full_transform_data_path(
                            resolved.target,
                            _rotation_path(resolved.target),
                        ),
                    )
                )
            if scale:
                _AUTO_KEY_ANIMATED_FAMILIES.add(
                    (control_key, _full_transform_data_path(resolved.target, "scale"))
                )
    finally:
        _AUTO_KEY_INSERT_GUARD = False

    if changed_any:
        context.view_layer.update()
    return changed_any


def _run_auto_key_postprocess_timer():
    global _AUTO_KEY_POSTPROCESS_TIMER_PENDING, _AUTO_KEY_DATA_SIGNATURE
    global _AUTO_KEY_TRANSFORM_SNAPSHOT, _AUTO_KEY_ANIMATED_FAMILIES
    global _AUTO_KEY_NATIVE_TRANSFORM_LATCHED

    context = bpy.context
    scene = getattr(context, "scene", None)
    if scene is None or not bool(getattr(scene, "baw_auto_key_enabled", False)):
        _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
        _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False
        return None

    if _active_generated_rigped(context):
        # Generated Rigped Auto is transaction-owned by the semantic transform
        # operator. The ordinary depsgraph Auto pipeline must never run a second
        # delayed writer/selection cleanup for the same gesture.
        _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
        _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False
        return None

    # Debounce the heavy post-transform work. During a mouse gizmo drag Blender
    # can emit many depsgraph updates per second; only the final settled value
    # should trigger key supplementation, FCurve scanning and selection sync.
    elapsed = monotonic() - _AUTO_KEY_LAST_TRANSFORM_CHANGE
    if elapsed < _AUTO_KEY_POSTPROCESS_DELAY:
        return max(0.01, _AUTO_KEY_POSTPROCESS_DELAY - elapsed)
    if _transform_modal_is_active(context):
        return 0.03

    # The native-transform cause is latched when the depsgraph first sees the
    # transform-value change. Do not re-read context.active_operator here:
    # Trajectory refresh can legally run Blender motion-path operators before
    # this delayed callback, replacing the last/active operator even though the
    # change originated from a real native gizmo transform.
    if not _AUTO_KEY_NATIVE_TRANSFORM_LATCHED:
        _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
        return None

    # Blender can replace/clear active keying-set state across frame/context
    # changes. AWB's visible AUTO state is authoritative while it is enabled.
    if not _apply_awb_auto_key_runtime(scene):
        scene.baw_auto_key_enabled = False
        _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
        _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False
        return None

    current_context_signature = _auto_key_context_binding_signature(context)
    baseline_controls = _auto_key_control_signature_from_binding_signature(
        _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE
    )
    current_controls = _auto_key_control_signature_from_binding_signature(
        current_context_signature
    )
    if (
        _AUTO_KEY_FRAME_BASELINE_FRAME != int(scene.frame_current)
        or baseline_controls != current_controls
    ):
        # The timer belongs to an older frame/control generation. Never let it
        # write into a newly selected control set. A binding-only change is
        # allowed here because Blender's first native Auto Key transform can
        # create its Action/Slot after the pre-transform baseline was captured.
        resync_awb_auto_key_after_context_change(context)
        return None

    current_transform = _capture_auto_key_transform_snapshot(context)
    _seed_first_animated_families_after_transform(context)
    if _AUTO_KEY_FRAME_BASELINE_TRANSFORM:
        _supplement_changed_transform_keys(
            context,
            _AUTO_KEY_FRAME_BASELINE_TRANSFORM,
            current_transform,
        )

    _AUTO_KEY_DATA_SIGNATURE = _auto_key_data_signature(context)
    _AUTO_KEY_ANIMATED_FAMILIES = _animated_families_from_signature(
        _AUTO_KEY_DATA_SIGNATURE
    )
    _AUTO_KEY_TRANSFORM_SNAPSHOT = _capture_auto_key_transform_snapshot(context)
    _reset_auto_key_frame_baseline(context)
    clear_key_selection_for_context(context)
    scene.baw_has_selected_key = False
    from .viewport_trajectory import sync_active_trajectory_key_selection

    sync_active_trajectory_key_selection(context)
    _redraw_view3d(context)
    _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
    _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False
    return None


def _queue_auto_key_postprocess() -> None:
    global _AUTO_KEY_POSTPROCESS_TIMER_PENDING
    if bpy.app.background or _AUTO_KEY_POSTPROCESS_TIMER_PENDING:
        return
    _AUTO_KEY_POSTPROCESS_TIMER_PENDING = True
    bpy.app.timers.register(
        _run_auto_key_postprocess_timer,
        first_interval=_AUTO_KEY_POSTPROCESS_DELAY,
    )


def _supplement_changed_transform_keys(
    context,
    previous: dict[tuple[int, int], tuple],
    current: dict[tuple[int, int], tuple],
) -> bool:
    """Insert only transform families that actually changed since the frame baseline.

    This fills Blender Auto Key's gap where a later Rotate/Scale transform does
    not create a brand-new FCurve after Location already exists. AWB trajectory
    operators edit FCurves directly and are deliberately excluded.
    """
    global _AUTO_KEY_INSERT_GUARD

    if _AUTO_KEY_INSERT_GUARD:
        return False

    scene = context.scene
    frame = int(scene.frame_current)
    changed_any = False
    _AUTO_KEY_INSERT_GUARD = True
    try:
        for resolved in _auto_key_controls_for_context(context):
            control_key = runtime_control_key(resolved)
            before = previous.get(control_key)
            after = current.get(control_key)
            if before is None or after is None:
                continue
            position, rotation, scale = _changed_transform_families(before, after)
            position = bool(position and scene.baw_key_position)
            rotation = bool(rotation and scene.baw_key_rotation)
            scale = bool(scale and scene.baw_key_scale)
            if not (position or rotation or scale):
                continue
            _insert_transform_key(
                resolved.target,
                frame,
                position=position,
                rotation=rotation,
                scale=scale,
            )
            if position:
                _AUTO_KEY_ANIMATED_FAMILIES.add(
                    (control_key, _full_transform_data_path(resolved.target, "location"))
                )
            if rotation:
                _AUTO_KEY_ANIMATED_FAMILIES.add(
                    (
                        control_key,
                        _full_transform_data_path(
                            resolved.target,
                            _rotation_path(resolved.target),
                        ),
                    )
                )
            if scale:
                _AUTO_KEY_ANIMATED_FAMILIES.add(
                    (control_key, _full_transform_data_path(resolved.target, "scale"))
                )
            changed_any = True
    finally:
        _AUTO_KEY_INSERT_GUARD = False

    if changed_any:
        context.view_layer.update()
        clear_key_selection_for_context(context)
        scene.baw_has_selected_key = False
        from .viewport_trajectory import sync_active_trajectory_key_selection

        sync_active_trajectory_key_selection(context)
        _redraw_view3d(context)
    return changed_any


@persistent
def _awb_auto_key_depsgraph_update_post(scene, depsgraph) -> None:
    global _AUTO_KEY_TRANSFORM_SNAPSHOT, _AUTO_KEY_LAST_TRANSFORM_CHANGE
    global _AUTO_KEY_NATIVE_TRANSFORM_LATCHED, _AUTO_KEY_POSTPROCESS_TIMER_PENDING

    if (
        bpy.app.background
        or _AUTO_KEY_INSERT_GUARD
        or not bool(getattr(scene, "baw_auto_key_enabled", False))
    ):
        return
    context = bpy.context
    if getattr(context, "scene", None) is not scene:
        return
    window_manager = getattr(context, "window_manager", None)
    if window_manager is not None and bool(
        getattr(window_manager, "baw_trackbar_frame_drag_active", False)
    ):
        # Current-time scrubbing is navigation, not a transform edit. The
        # frame-change handler rebases Auto Key once per evaluated frame, so the
        # depsgraph hot path has nothing to do during the drag.
        return
    if window_manager is not None and getattr(
        window_manager, "baw_rigped_semantic_move_drag_active", False
    ):
        # AWB semantic Move owns its transaction and keys exactly once on
        # release. Per-depsgraph Auto Key tracking would only duplicate the hot
        # path while the constrained preview is moving.
        return
    if _active_generated_rigped(context):
        # Hard generated-Rigped boundary: C/semantic transform operators own
        # authoring. Frame replay, key selection, depsgraph evaluation and stale
        # Blender transform operator state are never Auto authoring triggers.
        if bpy.app.timers.is_registered(_run_auto_key_postprocess_timer):
            bpy.app.timers.unregister(_run_auto_key_postprocess_timer)
        _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
        _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False
        _AUTO_KEY_LAST_TRANSFORM_CHANGE = 0.0
        return

    current_context_signature = _auto_key_context_binding_signature(context)
    baseline_controls = _auto_key_control_signature_from_binding_signature(
        _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE
    )
    current_controls = _auto_key_control_signature_from_binding_signature(
        current_context_signature
    )
    current_transform = _capture_auto_key_transform_snapshot(context)
    transform_changed = current_transform != _AUTO_KEY_TRANSFORM_SNAPSHOT
    native_transform_modal = _transform_modal_is_active(context)
    last_operator_is_transform = _last_operator_is_transform(context)
    native_transform_change = native_transform_modal or last_operator_is_transform

    if current_context_signature != _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE:
        controls_changed = current_controls != baseline_controls
        first_binding_creation = _auto_key_is_first_binding_creation(
            _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE,
            current_context_signature,
        )
        binding_created_during_transform = bool(
            not controls_changed
            and first_binding_creation
            and (
                _AUTO_KEY_NATIVE_TRANSFORM_LATCHED
                or _AUTO_KEY_POSTPROCESS_TIMER_PENDING
                or (transform_changed and native_transform_change)
            )
        )
        if controls_changed or not binding_created_during_transform:
            # Selection/control changes and ordinary external Action/Slot swaps are
            # context changes. Rebase before transform detection so no stale
            # baseline can ever write into a newly selected/bound control set.
            resync_awb_auto_key_after_context_change(context)
            return

    # Hot path: do not scan FCurves or insert/sync keys on every depsgraph tick.
    # Just notice a real transform-value change and debounce the expensive work
    # until the mouse transform settles.
    if not transform_changed:
        return
    _AUTO_KEY_TRANSFORM_SNAPSHOT = current_transform

    if _active_operator_is_awb_trajectory_edit(context) and not native_transform_modal:
        # A finished trajectory selection operator can remain Blender's
        # ``active_operator`` even after the user starts the next native gizmo
        # transform. Only suppress genuine trajectory-edit updates; an active
        # native transform modal is authoritative evidence of user P/R/S input.
        return

    if not native_transform_change:
        return

    _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = True
    _AUTO_KEY_LAST_TRANSFORM_CHANGE = monotonic()
    _queue_auto_key_postprocess()


@persistent
def _awb_auto_key_frame_change_post(scene, *_args) -> None:
    global _AUTO_KEY_TRANSFORM_SNAPSHOT, _AUTO_KEY_POSTPROCESS_TIMER_PENDING
    global _AUTO_KEY_LAST_TRANSFORM_CHANGE, _AUTO_KEY_NATIVE_TRANSFORM_LATCHED

    if bpy.app.background or not bool(getattr(scene, "baw_auto_key_enabled", False)):
        return

    # A frame evaluation can change transform values without user input. Cancel
    # any stale transform debounce before capturing the new frame baseline.
    if bpy.app.timers.is_registered(_run_auto_key_postprocess_timer):
        bpy.app.timers.unregister(_run_auto_key_postprocess_timer)
    _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
    _AUTO_KEY_LAST_TRANSFORM_CHANGE = 0.0
    _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False

    # Frame changes must keep AUTO armed and reset the transform baseline so
    # evaluated animation on the new frame is never mistaken for a user edit.
    _apply_awb_auto_key_runtime(scene)
    context = bpy.context
    _AUTO_KEY_TRANSFORM_SNAPSHOT = _capture_auto_key_transform_snapshot(context)
    _reset_auto_key_frame_baseline(
        context,
        transform_snapshot=_AUTO_KEY_TRANSFORM_SNAPSHOT,
    )


@persistent
def _awb_auto_key_lifecycle_post(*_args) -> None:
    """Discard stale Auto Key timers/baselines across Undo/Redo/load generations."""
    global _AUTO_KEY_POSTPROCESS_TIMER_PENDING, _AUTO_KEY_DATA_SIGNATURE
    global _AUTO_KEY_TRANSFORM_SNAPSHOT, _AUTO_KEY_ANIMATED_FAMILIES
    global _AUTO_KEY_LAST_TRANSFORM_CHANGE, _AUTO_KEY_NATIVE_TRANSFORM_LATCHED
    global _AUTO_KEY_FRAME_BASELINE_TRANSFORM
    global _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES, _AUTO_KEY_FRAME_BASELINE_FRAME
    global _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE

    if bpy.app.timers.is_registered(_run_auto_key_postprocess_timer):
        bpy.app.timers.unregister(_run_auto_key_postprocess_timer)
    _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
    _AUTO_KEY_LAST_TRANSFORM_CHANGE = 0.0
    _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False

    context = bpy.context
    scene = getattr(context, "scene", None)
    if scene is not None and bool(getattr(scene, "baw_auto_key_enabled", False)):
        _apply_awb_auto_key_runtime(scene)
        _AUTO_KEY_DATA_SIGNATURE = _auto_key_data_signature(context)
        _AUTO_KEY_ANIMATED_FAMILIES = _animated_families_from_signature(
            _AUTO_KEY_DATA_SIGNATURE
        )
        _AUTO_KEY_TRANSFORM_SNAPSHOT = _capture_auto_key_transform_snapshot(context)
        _reset_auto_key_frame_baseline(context)
        return

    _AUTO_KEY_DATA_SIGNATURE = None
    _AUTO_KEY_TRANSFORM_SNAPSHOT = {}
    _AUTO_KEY_ANIMATED_FAMILIES = set()
    _AUTO_KEY_FRAME_BASELINE_TRANSFORM = {}
    _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES = set()
    _AUTO_KEY_FRAME_BASELINE_FRAME = None
    _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE = None


def register_auto_key_handlers() -> None:
    if _awb_auto_key_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_awb_auto_key_depsgraph_update_post)
    if _awb_auto_key_frame_change_post not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_awb_auto_key_frame_change_post)
    for handler_name in ("undo_post", "redo_post", "load_post"):
        handlers = getattr(bpy.app.handlers, handler_name, None)
        if handlers is not None and _awb_auto_key_lifecycle_post not in handlers:
            handlers.append(_awb_auto_key_lifecycle_post)


def unregister_auto_key_handlers() -> None:
    global _AUTO_KEY_POSTPROCESS_TIMER_PENDING, _AUTO_KEY_DATA_SIGNATURE
    global _AUTO_KEY_TRANSFORM_SNAPSHOT, _AUTO_KEY_ANIMATED_FAMILIES
    global _AUTO_KEY_FRAME_BASELINE_TRANSFORM
    global _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES, _AUTO_KEY_FRAME_BASELINE_FRAME
    global _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE
    global _AUTO_KEY_INSERT_GUARD, _AUTO_KEY_NATIVE_TRANSFORM_LATCHED
    for handlers, callback in (
        (bpy.app.handlers.depsgraph_update_post, _awb_auto_key_depsgraph_update_post),
        (bpy.app.handlers.frame_change_post, _awb_auto_key_frame_change_post),
    ):
        if callback in handlers:
            handlers.remove(callback)
    for handler_name in ("undo_post", "redo_post", "load_post"):
        handlers = getattr(bpy.app.handlers, handler_name, None)
        if handlers is not None and _awb_auto_key_lifecycle_post in handlers:
            handlers.remove(_awb_auto_key_lifecycle_post)
    if bpy.app.timers.is_registered(_run_auto_key_postprocess_timer):
        bpy.app.timers.unregister(_run_auto_key_postprocess_timer)
    _AUTO_KEY_POSTPROCESS_TIMER_PENDING = False
    _AUTO_KEY_DATA_SIGNATURE = None
    _AUTO_KEY_TRANSFORM_SNAPSHOT = {}
    _AUTO_KEY_ANIMATED_FAMILIES = set()
    _AUTO_KEY_FRAME_BASELINE_TRANSFORM = {}
    _AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES = set()
    _AUTO_KEY_FRAME_BASELINE_FRAME = None
    _AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE = None
    _AUTO_KEY_INSERT_GUARD = False
    _AUTO_KEY_NATIVE_TRANSFORM_LATCHED = False


def disable_awb_auto_key_if_active(context) -> None:
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    if getattr(scene, "baw_auto_key_enabled", False):
        set_awb_auto_key(context, False)


def _active_transform_key_group(context) -> str | None:
    operator = getattr(context, "active_operator", None)
    if operator is None:
        return None
    identifiers = {
        str(getattr(operator, "bl_idname", "")),
        str(getattr(getattr(operator, "bl_rna", None), "identifier", "")),
    }
    if "TRANSFORM_OT_translate" in identifiers:
        return "POSITION"
    if "TRANSFORM_OT_rotate" in identifiers:
        return "ROTATION"
    if "TRANSFORM_OT_resize" in identifiers:
        return "SCALE"
    return None


class BAW_KSI_auto_key(bpy.types.KeyingSetInfo):
    bl_idname = AWB_AUTO_KEYING_SET_ID
    bl_label = "AWB Auto Key"
    bl_description = "Auto-key AWB transform channels for the current object or selected pose bones"

    def poll(self, context):
        scene = getattr(context, "scene", None)
        if scene is None:
            return False
        if not (
            scene.baw_key_position
            or scene.baw_key_rotation
            or scene.baw_key_scale
        ):
            return False
        return bool(_auto_key_controls_for_context(context))

    def iterator(self, context, ks):
        for resolved in _auto_key_controls_for_context(context):
            self.generate(context, ks, resolved.target)

    def generate(self, context, ks, data):
        resolved = resolve_control_target(data)
        if resolved is None:
            return
        target = resolved.target
        scene = context.scene
        transform_group = _active_transform_key_group(context)
        position = scene.baw_key_position and transform_group in {None, "POSITION"}
        rotation = scene.baw_key_rotation and transform_group in {None, "ROTATION"}
        scale = scene.baw_key_scale and transform_group in {None, "SCALE"}
        if position:
            _add_keying_set_path(ks, data, "location")
        if rotation:
            _add_keying_set_path(ks, data, _rotation_path(target))
        if scale:
            _add_keying_set_path(ks, data, "scale")


class BAW_OT_toggle_auto_key(bpy.types.Operator):
    bl_idname = "baw.toggle_auto_key"
    bl_label = "Toggle Auto Key"
    bl_description = "Toggle AWB Auto Key using the current Position/Rotation/Scale filters"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return getattr(context, "scene", None) is not None

    def execute(self, context):
        scene = context.scene
        enable = not bool(scene.baw_auto_key_enabled)
        if enable and not (
            scene.baw_key_position
            or scene.baw_key_rotation
            or scene.baw_key_scale
        ):
            self.report({"WARNING"}, "Auto Key Filter: no channels enabled")
            return {"CANCELLED"}
        if not set_awb_auto_key(context, enable):
            from .debug_trace import trace_event

            trace_event(
                "INPUT",
                "AUTO_KEY_TOGGLE_FAIL",
                context=context,
                requested=enable,
            )
            self.report({"ERROR"}, "Unable to switch AWB Auto Key state")
            return {"CANCELLED"}
        from .debug_trace import trace_event

        trace_event(
            "INPUT",
            "AUTO_KEY_TOGGLE",
            context=context,
            enabled=enable,
            position=bool(scene.baw_key_position),
            rotation=bool(scene.baw_key_rotation),
            scale=bool(scene.baw_key_scale),
            replay_action={
                "kind": "AUTO_KEY",
                "enabled": bool(enable),
                "position": bool(scene.baw_key_position),
                "rotation": bool(scene.baw_key_rotation),
                "scale": bool(scene.baw_key_scale),
            },
        )
        return {"FINISHED"}


class BAW_OT_set_key(bpy.types.Operator):
    bl_idname = "baw.set_key"
    bl_label = "Set Key"
    bl_description = "Set transform keys on the selected object or pose bones"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(control_context_for_context(context).controls)

    def invoke(self, context, event):
        scene = context.scene
        if (
            event.type == "K"
            and context.mode in {"OBJECT", "POSE"}
            and bool(getattr(scene, "baw_trajectory_edit_mode", False))
        ):
            from .trajectory_edit import _add_trajectory_position_keys

            frame = int(scene.frame_current)
            control_context = control_context_for_context(context)
            identities = tuple(
                (resolved.control.control_id, frame)
                for resolved in control_context.controls
            )
            _changed, selected = _add_trajectory_position_keys(context, identities)
            if selected:
                return {"FINISHED"}
            self.report(
                {"WARNING"},
                "Trajectory Add Key requires an unconstrained XYZ Position track",
            )
            return {"CANCELLED"}
        return self.execute(context)

    def execute(self, context):
        scene = context.scene
        frame = scene.frame_current
        position = scene.baw_key_position
        rotation = scene.baw_key_rotation
        scale = scene.baw_key_scale

        if not (position or rotation or scale):
            self.report({"WARNING"}, "Set Key Filter: no channels enabled")
            return {"CANCELLED"}

        control_context = control_context_for_context(context)
        controls = tuple(control_context.controls)
        if not controls:
            self.report({"WARNING"}, "No AWB animation control selected")
            return {"CANCELLED"}

        # Rigped owns its own C-key authoring grammar. K must never write the
        # generated Rigped controls themselves, but custom/external bones added
        # to the same Armature remain ordinary AWB controls and keep normal K.
        if context.mode == "POSE":
            from .rigped_animation_baseline import split_rigped_controls

            _rigped_controls, controls = split_rigped_controls(context, controls)
            if not controls:
                self.report({"INFO"}, "Rigped uses C for Free keys; K made no changes")
                return {"CANCELLED"}

        for resolved in controls:
            _insert_transform_key(
                resolved.target,
                frame,
                position=position,
                rotation=rotation,
                scale=scale,
            )

        # Key authoring must not implicitly enter Track Bar edit-selection mode.
        # Blender selects newly inserted KeyframePoints by default, which made the
        # Selection Range UI appear immediately after ordinary K authoring.
        clear_key_selection_for_context(context)
        scene.baw_has_selected_key = False

        if context.mode == "POSE":
            self.report({"INFO"}, f"Set Key: {len(controls)} control(s) at frame {frame}")
        elif len(controls) > 1:
            self.report({"INFO"}, f"Set Key: {len(controls)} objects at frame {frame}")
        else:
            self.report(
                {"INFO"},
                f"Set Key: {controls[0].control.display_name} at frame {frame}",
            )

        # Import locally to keep keying semantics independent from viewport
        # presentation while still refreshing an already-visible trajectory.
        from .viewport_trajectory import refresh_active_trajectory_live

        refresh_active_trajectory_live(context, force=True, exact_range=True)
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


def register_keymaps() -> None:
    wm = bpy.context.window_manager
    if wm is None or wm.keyconfigs.addon is None:
        return

    unregister_keymaps()
    kc = wm.keyconfigs.addon
    keymaps = (
        ("3D View", "VIEW_3D", "WINDOW"),
        ("Object Mode", "EMPTY", "WINDOW"),
        ("Pose", "EMPTY", "WINDOW"),
    )
    for name, space_type, region_type in keymaps:
        km = kc.keymaps.get(name)
        if km is None:
            km = kc.keymaps.new(
                name=name,
                space_type=space_type,
                region_type=region_type,
            )
        kmi = km.keymap_items.new(
            BAW_OT_set_key.bl_idname,
            "K",
            "PRESS",
            head=True,
        )
        _KEYMAP_ITEMS.append((km, kmi))


def unregister_keymaps() -> None:
    for km, kmi in reversed(_KEYMAP_ITEMS):
        if kmi.id != -1:
            km.keymap_items.remove(kmi)
    _KEYMAP_ITEMS.clear()
