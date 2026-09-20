from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from .semantic_model import (
    AWBChannel,
    AWBControl,
    AWBControlRole,
    parse_control_role,
    transform_channel,
)

AWB_ROLE_PROPERTY = "awb_role"


@dataclass(frozen=True, slots=True)
class ResolvedControl:
    """AWB control identity paired with its current Blender runtime target."""

    control: AWBControl
    owner_object: Any
    target: Any
    data_path_prefix: str | None = None


@dataclass(frozen=True, slots=True)
class ControlContext:
    """One runtime snapshot of AWB's effective selected control context."""

    mode: str
    controls: tuple[ResolvedControl, ...]
    active: ResolvedControl | None


@dataclass(frozen=True, slots=True)
class AnimationOwnerGroup:
    """Resolved controls sharing one Blender animation owner."""

    owner_object: Any
    controls: tuple[ResolvedControl, ...]
    prefixes: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class ResolvedChannel:
    """One concrete FCurve routed to one resolved AWB control."""

    resolved: ResolvedControl
    family: AWBChannel
    data_path: str
    array_index: int
    fcurve: Any


def runtime_control_key(resolved: ResolvedControl) -> tuple[int, int]:
    """Return an operation-local owner/target identity for Blender runtime data.

    Blender ``as_pointer()`` values are runtime memory addresses. This key is
    only valid while the referenced RNA data remains alive; it must not be
    persisted or reused across undo/redo, reload, file load, or deletion.
    """
    return (resolved.owner_object.as_pointer(), resolved.target.as_pointer())


def assigned_channelbag(owner) -> Any | None:
    """Return only the ChannelBag currently assigned to ``owner``.

    This is lookup-only: no Action, Slot, layer, strip, ChannelBag, or FCurve is
    created as a side effect.
    """
    anim_data = getattr(owner, "animation_data", None)
    if anim_data is None or getattr(anim_data, "action", None) is None:
        return None

    # Local import keeps semantic_adapter importable in pure unit tests and
    # avoids making Blender initialization part of the semantic contract.
    from bpy_extras.anim_utils import animdata_get_channelbag_for_assigned_slot

    return animdata_get_channelbag_for_assigned_slot(anim_data)


def _runtime_pointer_or_none(value) -> int | None:
    if value is None:
        return None
    as_pointer = getattr(value, "as_pointer", None)
    if not callable(as_pointer):
        return None
    pointer = int(as_pointer())
    return pointer if pointer else None


def channel_binding_token(owner) -> tuple[int | None, int | None, int | None, int | None]:
    """Return an operation-local owner/Action/Slot/ChannelBag binding token.

    The token is deliberately runtime-only. It is suitable for validating one
    in-flight edit transaction, not for persistence or identity across undo,
    reload, file load, deletion, or Blender restart.
    """
    anim_data = getattr(owner, "animation_data", None)
    action = getattr(anim_data, "action", None) if anim_data is not None else None
    slot = getattr(anim_data, "action_slot", None) if anim_data is not None else None
    channelbag = assigned_channelbag(owner) if action is not None else None
    return (
        _runtime_pointer_or_none(owner),
        _runtime_pointer_or_none(action),
        _runtime_pointer_or_none(slot),
        _runtime_pointer_or_none(channelbag),
    )


def group_animation_owners(
    controls: tuple[ResolvedControl, ...] | list[ResolvedControl],
) -> tuple[AnimationOwnerGroup, ...]:
    """Group controls by runtime animation owner while preserving control order."""
    grouped: list[list[Any]] = []
    owner_indexes: dict[int, int] = {}

    for resolved in controls:
        owner = resolved.owner_object
        owner_key = int(owner.as_pointer())
        index = owner_indexes.get(owner_key)
        if index is None:
            owner_indexes[owner_key] = len(grouped)
            grouped.append([owner, [resolved], [resolved.data_path_prefix]])
            continue
        grouped[index][1].append(resolved)
        grouped[index][2].append(resolved.data_path_prefix)

    return tuple(
        AnimationOwnerGroup(
            owner_object=owner,
            controls=tuple(group_controls),
            prefixes=tuple(dict.fromkeys(prefixes)),
        )
        for owner, group_controls, prefixes in grouped
    )


def control_property_path(resolved: ResolvedControl, property_name: str) -> str:
    """Return the concrete Blender data path for one control property."""
    prefix = str(resolved.data_path_prefix or "")
    return f"{prefix}.{property_name}" if prefix else str(property_name)


def rotation_property(target) -> str:
    """Return the Blender transform property used by the target's rotation mode."""
    rotation_mode = str(getattr(target, "rotation_mode", ""))
    if rotation_mode == "QUATERNION":
        return "rotation_quaternion"
    if rotation_mode == "AXIS_ANGLE":
        return "rotation_axis_angle"
    return "rotation_euler"


def matches_control_path(data_path: str, data_path_prefix) -> bool:
    """Match an FCurve data path to the existing AWB control-prefix contract."""
    if data_path_prefix is None:
        return True
    if isinstance(data_path_prefix, (tuple, list, set, frozenset)):
        return any(matches_control_path(data_path, prefix) for prefix in data_path_prefix)
    if data_path_prefix == "":
        return not str(data_path).startswith('pose.bones[')
    return str(data_path).startswith(str(data_path_prefix))


def channels_for_controls(
    controls: tuple[ResolvedControl, ...] | list[ResolvedControl],
) -> tuple[ResolvedChannel, ...]:
    """Resolve current assigned-slot FCurves with one indexed FCurve scan per owner.

    Pose-mode whole-rig selection can contain dozens of controls while one shared
    ChannelBag contains hundreds of curves. The old nested scan tested every curve
    against every selected control, making Track Bar redraw/edit cost grow as
    ``FCurves * selected controls``. Bone control prefixes are mutually exclusive,
    so a sorted prefix index resolves the owning selected control in logarithmic
    time while preserving the existing wildcard/object-prefix contracts.
    """
    channels: list[ResolvedChannel] = []
    seen: set[tuple[tuple[int, int], str, int]] = set()

    for group in group_animation_owners(controls):
        channelbag = assigned_channelbag(group.owner_object)
        if channelbag is None:
            continue

        wildcard_controls: list[ResolvedControl] = []
        object_controls: list[ResolvedControl] = []
        prefix_controls: dict[str, list[ResolvedControl]] = {}
        for resolved in group.controls:
            prefix = resolved.data_path_prefix
            if prefix is None:
                wildcard_controls.append(resolved)
            elif prefix == "":
                object_controls.append(resolved)
            else:
                prefix_controls.setdefault(str(prefix), []).append(resolved)
        sorted_prefixes = sorted(prefix_controls)

        for fcurve in channelbag.fcurves:
            data_path = str(fcurve.data_path)
            array_index = int(getattr(fcurve, "array_index", 0))
            family = transform_channel(data_path)
            matched_controls: list[ResolvedControl] = list(wildcard_controls)

            if not data_path.startswith('pose.bones['):
                matched_controls.extend(object_controls)

            if sorted_prefixes:
                index = bisect_right(sorted_prefixes, data_path) - 1
                if index >= 0:
                    prefix = sorted_prefixes[index]
                    if data_path.startswith(prefix):
                        matched_controls.extend(prefix_controls[prefix])

            for resolved in matched_controls:
                identity = (runtime_control_key(resolved), data_path, array_index)
                if identity in seen:
                    continue
                seen.add(identity)
                channels.append(
                    ResolvedChannel(
                        resolved=resolved,
                        family=family,
                        data_path=data_path,
                        array_index=array_index,
                        fcurve=fcurve,
                    )
                )
    return tuple(channels)


def channels_for_property(
    resolved: ResolvedControl,
    property_name: str,
) -> tuple[ResolvedChannel, ...]:
    """Resolve exact assigned-slot channels for one control property."""
    expected_path = control_property_path(resolved, property_name)
    return tuple(
        sorted(
            (
                channel
                for channel in channels_for_controls((resolved,))
                if channel.data_path == expected_path
            ),
            key=lambda channel: channel.array_index,
        )
    )


def key_at_frame(
    channel: ResolvedChannel,
    frame: float,
    *,
    epsilon: float = 1e-4,
) -> Any | None:
    """Resolve a concrete key immediately from its current FCurve."""
    target = float(frame)
    tolerance = abs(float(epsilon))
    return next(
        (
            point
            for point in channel.fcurve.keyframe_points
            if abs(float(point.co.x) - target) <= tolerance
        ),
        None,
    )


def _role_from_sources(*sources) -> AWBControlRole | None:
    for source in sources:
        if source is None or not hasattr(source, "get"):
            continue
        role = parse_control_role(source.get(AWB_ROLE_PROPERTY))
        if role is not None:
            return role
    return None


def _object_control(obj) -> ResolvedControl:
    role = _role_from_sources(obj)
    return ResolvedControl(
        control=AWBControl.object(obj.name, role=role),
        owner_object=obj,
        target=obj,
        # Empty prefix is an explicit object-level selector. Track Bar model/edit
        # helpers interpret it as "exclude pose.bones[...] curves" so an
        # Armature object in Object Mode never edits its bones' animation.
        data_path_prefix="",
    )


def _bone_control(obj, pose_bone) -> ResolvedControl:
    role = _role_from_sources(pose_bone, getattr(pose_bone, "bone", None))
    return ResolvedControl(
        control=AWBControl.bone(obj.name, pose_bone.name, role=role),
        owner_object=obj,
        target=pose_bone,
        data_path_prefix=pose_bone.path_from_id(),
    )


def resolve_control_target(target) -> ResolvedControl | None:
    """Resolve a raw Blender Object/PoseBone without changing selection policy."""
    if target is None:
        return None

    # Local import preserves pure-module importability outside Blender. This
    # helper is runtime decoding only; edit eligibility still comes from
    # :func:`control_context_for_context`.
    import bpy

    if isinstance(target, bpy.types.PoseBone):
        owner = getattr(target, "id_data", None)
        if owner is None:
            return None
        return _bone_control(owner, target)
    if isinstance(target, bpy.types.Object):
        return _object_control(target)
    return None


def active_control_for_context(context) -> ResolvedControl | None:
    obj = context.active_object
    if obj is None:
        return None

    if obj.type == "ARMATURE" and context.mode == "POSE":
        pose_bone = context.active_pose_bone
        if pose_bone is None or not pose_bone.select:
            return None
        return _bone_control(obj, pose_bone)

    if context.mode == "OBJECT" and not obj.select_get():
        return None
    return _object_control(obj)


def keying_controls_for_context(context) -> list[ResolvedControl]:
    obj = context.active_object
    if obj is None:
        return []

    if context.mode == "POSE":
        if obj.type != "ARMATURE":
            return []
        pose_bones = list(context.selected_pose_bones or [])
        controls = []
        for pose_bone in pose_bones:
            owner = getattr(pose_bone, "id_data", None)
            if owner is None or getattr(owner, "type", None) != "ARMATURE":
                continue
            controls.append(_bone_control(owner, pose_bone))
        return controls

    if context.mode == "OBJECT":
        selected_objects = list(context.selected_objects or [])
        if not selected_objects:
            return []
        # Keep the active object first for deterministic primary-control UI,
        # then include every other selected object as an equal AWB control.
        ordered = []
        if obj in selected_objects:
            ordered.append(obj)
        ordered.extend(
            sorted(
                (selected for selected in selected_objects if selected != obj),
                key=lambda selected: selected.name,
            )
        )
        return [_object_control(selected) for selected in ordered]

    return []


def trackbar_controls_for_context(context) -> list[ResolvedControl]:
    """Resolve the current Track Bar control set.

    Phase 1J intentionally shares Object/Pose ownership with Set Key / Auto Key:
    Object Mode is the selected object set; Pose Mode is the actually selected
    pose-bone set. An active-but-deselected object/bone is not an AWB edit target.
    Keeping this as a named adapter boundary lets a later semantic-character
    phase diverge without changing Track Bar callers.
    """
    return keying_controls_for_context(context)


def control_context_for_context(context) -> ControlContext:
    """Build the additive Phase 2 façade without changing resolver semantics."""
    controls = tuple(keying_controls_for_context(context))
    active = active_control_for_context(context)
    if active is not None:
        control_keys = {runtime_control_key(resolved) for resolved in controls}
        if runtime_control_key(active) not in control_keys:
            active = None
    return ControlContext(mode=context.mode, controls=controls, active=active)
