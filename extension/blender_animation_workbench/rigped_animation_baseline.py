from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import bpy

from .character_metadata import BONE_TOKEN_PROPERTY, read_character
from .rigped_contract import RIGPED_SETUP_PROPERTY
from .semantic_adapter import (
    ResolvedControl,
    assigned_channelbag,
    control_context_for_context,
    runtime_control_key,
)


@dataclass(frozen=True, slots=True)
class RigpedFreeKeyResult:
    applied: bool
    keyed_controls: int = 0


RIGPED_KEY_STATE_PROPERTY = "awb_rigped_key_state"


class RigpedKeyState(IntEnum):
    FREE = 1
    SLIDING = 2
    PLANTED = 3


_FREE_KEYMAP_ITEMS: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []
_RIGPED_BONE_TOKEN_CACHE_KEY: tuple | None = None
_RIGPED_BONE_TOKEN_CACHE: frozenset[str] = frozenset()


def active_rigped_character_id(context) -> str | None:
    """Return the generated Rigped character owned by the active Armature."""

    rig = getattr(context, "active_object", None)
    if rig is None or getattr(rig, "type", None) != "ARMATURE":
        return None
    raw = rig.get(RIGPED_SETUP_PROPERTY) if hasattr(rig, "get") else None
    getter = getattr(raw, "get", None)
    if not callable(getter):
        return None
    character_id = str(getter("character_id", "") or "")
    return character_id or None


def _rigped_bone_token_cache_key(context) -> tuple | None:
    rig = getattr(context, "active_object", None)
    if rig is None or getattr(rig, "type", None) != "ARMATURE":
        return None
    raw = rig.get(RIGPED_SETUP_PROPERTY) if hasattr(rig, "get") else None
    getter = getattr(raw, "get", None)
    if not callable(getter):
        return None
    character_id = str(getter("character_id", "") or "")
    if not character_id:
        return None
    scene = getattr(context, "scene", None)
    armature = getattr(rig, "data", None)
    if scene is None or armature is None:
        return None
    return (
        int(scene.as_pointer()),
        int(rig.as_pointer()),
        int(armature.as_pointer()),
        character_id,
        int(getter("revision", 0) or 0),
        str(getter("signature", "") or ""),
        len(getattr(armature, "bones", ())),
    )


def _rigped_owned_bone_tokens(context) -> frozenset[str]:
    """Return generated Rigped Bone tokens without resolving native targets.

    Auto Key asks this question on high-frequency depsgraph/frame-change paths.
    Resolving the whole Character graph there made merely selecting one Rigped
    PoseBone expensive. The generated binding set is structural, so cache the
    persistent Bone-token set until the active generated rig identity changes.
    """
    global _RIGPED_BONE_TOKEN_CACHE_KEY, _RIGPED_BONE_TOKEN_CACHE

    cache_key = _rigped_bone_token_cache_key(context)
    if cache_key is None:
        return frozenset()
    if cache_key == _RIGPED_BONE_TOKEN_CACHE_KEY:
        return _RIGPED_BONE_TOKEN_CACHE

    character_id = str(cache_key[3])
    try:
        definition = read_character(context.scene, character_id)
    except Exception:  # noqa: BLE001 -- malformed metadata must degrade to an empty cache
        definition = None
    tokens = frozenset(
        str(binding.bone_id)
        for binding in (() if definition is None else definition.bindings)
        if str(binding.bone_id or "")
    )
    _RIGPED_BONE_TOKEN_CACHE_KEY = cache_key
    _RIGPED_BONE_TOKEN_CACHE = tokens
    return tokens


def _resolved_bone_token(resolved: ResolvedControl) -> str | None:
    bone = getattr(getattr(resolved, "target", None), "bone", None)
    if bone is None or not hasattr(bone, "get"):
        return None
    value = bone.get(BONE_TOKEN_PROPERTY)
    token = str(value).strip() if value is not None else ""
    return token or None


def is_rigped_owned_control(context, resolved: ResolvedControl) -> bool:
    token = _resolved_bone_token(resolved)
    return bool(token and token in _rigped_owned_bone_tokens(context))


def split_rigped_controls(
    context,
    controls: tuple[ResolvedControl, ...],
) -> tuple[tuple[ResolvedControl, ...], tuple[ResolvedControl, ...]]:
    """Split selected controls into generated Rigped controls and external controls.

    External controls include custom bones added to the same Armature later, such
    as cloth, skirt, cape, accessory, or secondary deformation controls, because
    they are not part of the generated Rigped character binding set.
    """

    owned_tokens = _rigped_owned_bone_tokens(context)
    rigped: list[ResolvedControl] = []
    external: list[ResolvedControl] = []
    for resolved in controls:
        token = _resolved_bone_token(resolved)
        if token and token in owned_tokens:
            rigped.append(resolved)
        else:
            external.append(resolved)
    return tuple(rigped), tuple(external)


def selected_rigped_controls(context) -> tuple[ResolvedControl, ...]:
    controls = tuple(control_context_for_context(context).controls)
    rigped, _external = split_rigped_controls(context, controls)
    return rigped


def _rigped_state_data_path(target: bpy.types.PoseBone) -> str:
    return target.path_from_id(f'["{RIGPED_KEY_STATE_PROPERTY}"]')


def _set_state_key_interpolation_constant(
    resolved: ResolvedControl,
    frame: int,
) -> None:
    channelbag = assigned_channelbag(resolved.owner_object)
    if channelbag is None:
        return
    expected_path = _rigped_state_data_path(resolved.target)
    for fcurve in channelbag.fcurves:
        if str(fcurve.data_path) != expected_path:
            continue
        for key in fcurve.keyframe_points:
            if abs(float(key.co.x) - float(frame)) <= 1e-4:
                key.interpolation = "CONSTANT"
                return


def rigped_key_types_for_context(context) -> dict[float, RigpedKeyState]:
    """Return semantic Rigped key states for the currently selected Rigped controls."""

    controls = selected_rigped_controls(context)
    if not controls:
        return {}

    paths_by_owner: dict[int, tuple[object, set[str]]] = {}
    for resolved in controls:
        target = resolved.target
        if not isinstance(target, bpy.types.PoseBone):
            continue
        # Contact-owned limb controls intentionally may have no ordinary Rigped
        # semantic-state property at all. Absence means "no ordinary semantic
        # override" and must not make Track Bar color aggregation fail-soft.
        if RIGPED_KEY_STATE_PROPERTY not in target:
            continue
        owner_key = int(resolved.owner_object.as_pointer())
        row = paths_by_owner.get(owner_key)
        if row is None:
            row = (resolved.owner_object, set())
            paths_by_owner[owner_key] = row
        row[1].add(_rigped_state_data_path(target))

    states: dict[float, RigpedKeyState] = {}
    for owner, expected_paths in paths_by_owner.values():
        channelbag = assigned_channelbag(owner)
        if channelbag is None:
            continue
        for fcurve in channelbag.fcurves:
            if str(fcurve.data_path) not in expected_paths:
                continue
            for key in fcurve.keyframe_points:
                try:
                    state = RigpedKeyState(round(float(key.co.y)))
                except ValueError:
                    continue
                states[float(key.co.x)] = state
    return states


def rigped_free_display_active(context) -> bool:
    """True only when the current Track Bar selection is purely Rigped-owned."""

    if getattr(context, "mode", "") != "POSE":
        return False
    controls = tuple(control_context_for_context(context).controls)
    if not controls:
        return False
    rigped, external = split_rigped_controls(context, controls)
    return bool(rigped and not external)


def _ensure_quaternion_rotation(pose_bone: bpy.types.PoseBone) -> None:
    """Switch one Rigped pose control to quaternion storage without changing pose."""

    if str(pose_bone.rotation_mode) == "QUATERNION":
        return
    matrix_basis = pose_bone.matrix_basis.copy()
    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.matrix_basis = matrix_basis


def _insert_free_key_controls(
    context,
    controls: tuple[ResolvedControl, ...],
    *,
    clear_selection: bool,
) -> RigpedFreeKeyResult:
    """Author ordinary Rigped Free keys only on the supplied generated controls."""

    if not controls:
        return RigpedFreeKeyResult(False, 0)

    from .rigped_transform import apply_rigped_joint_limits_to_pose_bone

    frame = int(context.scene.frame_current)
    keyed = 0
    for resolved in controls:
        target = resolved.target
        if not isinstance(target, bpy.types.PoseBone):
            continue
        _ensure_quaternion_rotation(target)
        apply_rigped_joint_limits_to_pose_bone(target)
        target.keyframe_insert(data_path="location", frame=frame)
        target.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        target[RIGPED_KEY_STATE_PROPERTY] = int(RigpedKeyState.FREE)
        target.keyframe_insert(
            data_path=f'["{RIGPED_KEY_STATE_PROPERTY}"]',
            frame=frame,
        )
        _set_state_key_interpolation_constant(resolved, frame)
        keyed += 1

    if keyed and clear_selection:
        # Rigped C authors a semantic key but must leave Track Bar edit selection
        # empty. Blender selects newly inserted points by default, which otherwise
        # exposes the Selection Range UI before the user explicitly selects keys.
        from .trackbar_model import clear_key_selection_for_context

        clear_key_selection_for_context(context)
        context.scene.baw_has_selected_key = False
        if context.area is not None:
            context.area.tag_redraw()
    return RigpedFreeKeyResult(bool(keyed), keyed)


def insert_free_key(context) -> RigpedFreeKeyResult:
    """Author the minimal clean Rigped Free key baseline.

    Free keys intentionally own the complete XYZ position and complete quaternion
    rotation sample. Scale is Figure/Fit authority and is not authored here.
    Contact/IK state is not consulted in this baseline implementation.
    """

    return _insert_free_key_controls(
        context,
        selected_rigped_controls(context),
        clear_selection=True,
    )


class BAW_OT_rigped_free_key(bpy.types.Operator):
    bl_idname = "baw.rigped_free_key"
    bl_label = "Rigped Free Key"
    bl_description = "Set a Rigped FK Free key (XYZ position + quaternion rotation)"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(
            getattr(context, "mode", "") == "POSE"
            and selected_rigped_controls(context)
        )

    def invoke(self, context, event):
        if bool(getattr(event, "is_repeat", False)):
            return {"CANCELLED"}
        return self.execute(context)

    def execute(self, context):
        # A5 resolves C through the generated arm/leg limb domain rather than
        # only from the terminal Hand/Foot. Thigh/Calf/Foot all address the same
        # leg Contact domain; UpperArm/ForeArm/Hand all address the same arm
        # Contact domain. Non-limb Rigped controls keep the accepted Free path.
        from .phase4_contact_authoring import (
            clear_contact_bundle_key_selection_for_context,
            execute_contact_command,
            selected_contact_command_control_keys,
            selected_contact_command_mapping_ids,
        )

        control_context = control_context_for_context(context)
        contact_mapping_ids = selected_contact_command_mapping_ids(
            context.scene,
            control_context,
        )
        contact_control_keys = selected_contact_command_control_keys(
            context.scene,
            control_context,
        )

        if contact_mapping_ids:
            # A5 boundary: supported arm/leg C uses the production Contact
            # authoring backend, but only Free <-> Sliding is enabled. Planted
            # remains deliberately unavailable until A6 is accepted.
            from .phase4_contact_model import ContactKeyType, ContactPlantSpace
            from .trackbar_model import clear_key_selection_for_context

            contact_result = execute_contact_command(
                context.scene,
                control_context_for_context(context),
                operation_id="baw-rigped-a5-contact",
                enabled_types=(ContactKeyType.FREE, ContactKeyType.SLIDING),
                plant_space=ContactPlantSpace.WORLD,
                contact_point_local=(0.0, 0.0, 0.0),
            )
            if not contact_result.applied:
                detail = (
                    contact_result.diagnostics[0].detail
                    if contact_result.diagnostics
                    else "Rigped Sliding authoring failed"
                )
                self.report({"WARNING"}, detail)
                return {"CANCELLED"}

            # Broad Rigped selections (for example box-selecting the whole rig)
            # may include Root/Spine/Head alongside Contact limbs. Contact owns
            # the selected arm/leg controls; preserve ordinary C-key behavior on
            # the remaining generated controls without writing a competing
            # Rigped FREE state onto limb controls.
            residual_controls = tuple(
                resolved
                for resolved in selected_rigped_controls(context)
                if runtime_control_key(resolved) not in contact_control_keys
            )
            _insert_free_key_controls(
                context,
                residual_controls,
                clear_selection=False,
            )

            frame = float(context.scene.frame_current) + float(getattr(context.scene, "frame_subframe", 0.0))
            clear_contact_bundle_key_selection_for_context(context, frames=(frame,))
            clear_key_selection_for_context(context)
            context.scene.baw_has_selected_key = False
            if (
                contact_result.contact_type is ContactKeyType.SLIDING
                and str(getattr(context.scene, "baw_rigped_semantic_transform_mode", "NONE"))
                in {"FK_MOVE", "MOVE"}
            ):
                context.scene.baw_rigped_semantic_transform_mode = "MOVE"
            elif (
                contact_result.contact_type is ContactKeyType.FREE
                and str(getattr(context.scene, "baw_rigped_semantic_transform_mode", "NONE"))
                in {"FK_MOVE", "MOVE"}
            ):
                context.scene.baw_rigped_semantic_transform_mode = "FK_MOVE"
            if context.area is not None:
                context.area.tag_redraw()
            label = contact_result.contact_type.value.title() if contact_result.contact_type else "Contact"
            self.report({"INFO"}, f"Rigped {label} Key")
            return {"FINISHED"}

        result = insert_free_key(context)
        if not result.applied:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Free Key: {result.keyed_controls} Rigped control(s)")
        return {"FINISHED"}


def register_rigped_free_keymap() -> None:
    unregister_rigped_free_keymap()
    wm = bpy.context.window_manager
    if wm is None or wm.keyconfigs.addon is None:
        return
    kc = wm.keyconfigs.addon
    km = kc.keymaps.get("Pose")
    if km is None:
        km = kc.keymaps.new(name="Pose", space_type="EMPTY", region_type="WINDOW")
    kmi = km.keymap_items.new(
        BAW_OT_rigped_free_key.bl_idname,
        "C",
        "PRESS",
        head=True,
    )
    _FREE_KEYMAP_ITEMS.append((km, kmi))


def unregister_rigped_free_keymap() -> None:
    for km, kmi in reversed(_FREE_KEYMAP_ITEMS):
        if kmi.id != -1:
            km.keymap_items.remove(kmi)
    _FREE_KEYMAP_ITEMS.clear()
