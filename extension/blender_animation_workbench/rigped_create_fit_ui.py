from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import blf
import bpy
from bpy.app.handlers import persistent
from bpy_extras import view3d_utils
from mathutils import Vector

from .character_metadata import (
    CharacterMetadataError,
    character_ids,
    prune_stale_generated_rigped_metadata,
    resolve_character,
)
from .rigped_contract import (
    RIGPED_SETUP_PROPERTY,
    RigpedCapability,
    control_contracts_for_view,
    read_setup_descriptor,
)
from .rigped_create_policy import (
    fitted_humanoid_spec,
    height_from_free_mouse,
)
from .rigped_fit_runtime import (
    RigpedFitRuntimeError,
    commit_fit_session,
    inspect_fit_entry,
)
from .rigped_humanoid_builder import (
    build_generated_rigped_humanoid,
    discard_generated_rigped_humanoid,
    ensure_generated_rigped_ik_hinge_limits,
)
from .rigped_rigify_reference import rigify_reference_humanoid_spec
from .semantic_adapter import control_context_for_context, runtime_control_key
from .ui_language import language_for_context
from .ui_language import text as ui_text


@dataclass(frozen=True, slots=True)
class _BoneRestSnapshot:
    name: str
    head: tuple[float, float, float]
    tail: tuple[float, float, float]
    roll: float
    bbone_x: float
    bbone_z: float
    parent_name: str | None
    use_connect: bool


@dataclass(frozen=True, slots=True)
class _FitHistoryEntry:
    before: tuple[_BoneRestSnapshot, ...]
    after: tuple[_BoneRestSnapshot, ...]


@dataclass(slots=True)
class _FitUiState:
    character_id: str
    session: Any
    rig_object: Any
    rest_snapshot: tuple[_BoneRestSnapshot, ...]
    original_active: Any | None
    original_selected: tuple[Any, ...]
    original_mode: str
    original_pivot_point: str
    original_orientation: str
    transform_mode: str = "NONE"
    orientation_mode: str = "LOCAL"
    history: list[_FitHistoryEntry] = field(default_factory=list)
    history_cursor: int = 0


_FIT_STATES: dict[int, _FitUiState] = {}
_INTERNAL_HELPERS_HIDDEN_PROPERTY = "awb_internal_helpers_hidden_v1"
_BIPED_BOX_WIRE_PROPERTY = "awb_biped_box_wire_v1"
_BIPED_BOX_WIRE_SHAPE_OBJECT = "AWB_BipedBoxWireShape"
_BIPED_BOX_WIRE_DEFAULT_WIDTH = 1.5
_BIPED_BOX_WIRE_DEFAULT_LEFT = (28.0 / 255.0, 28.0 / 255.0, 177.0 / 255.0)
_BIPED_BOX_WIRE_DEFAULT_RIGHT = (6.0 / 255.0, 134.0 / 255.0, 6.0 / 255.0)
_BIPED_BOX_WIRE_DEFAULT_CENTER = (8.0 / 255.0, 110.0 / 255.0, 134.0 / 255.0)
_BIPED_BOX_WIRE_DEFAULT_PELVIS = (224.0 / 255.0, 198.0 / 255.0, 87.0 / 255.0)
_BIPED_BOX_WIRE_DEFAULT_HEAD = (166.0 / 255.0, 202.0 / 255.0, 240.0 / 255.0)
_BIPED_BOX_WIRE_DEFAULT_COM = (0.92, 0.72, 0.18)
_BIPED_BOX_WIRE_DEFAULT_SELECTED = (1.0, 0.72, 0.18)
_BIPED_BOX_WIRE_DEFAULT_ACTIVE = (1.0, 0.92, 0.35)


def _safe_pointer(value) -> int | None:
    if value is None:
        return None
    try:
        pointer = getattr(value, "as_pointer", None)
        if not callable(pointer):
            return None
        result = int(pointer())
    except ReferenceError:
        return None
    return result or None


def _window_key(context) -> int | None:
    return _safe_pointer(getattr(context, "window", None))


def fit_ui_state(context) -> _FitUiState | None:
    key = _window_key(context)
    return _FIT_STATES.get(key) if key is not None else None


def set_fit_transform_mode(context, mode: str) -> bool:
    state = fit_ui_state(context)
    if state is None:
        return False
    state.transform_mode = str(mode)
    if context.area is not None:
        context.area.tag_redraw()
    return True


def fit_transform_mode(context) -> str:
    state = fit_ui_state(context)
    return state.transform_mode if state is not None else "NONE"


def set_fit_orientation_mode(context, mode: str) -> bool:
    state = fit_ui_state(context)
    value = str(mode).upper()
    if state is None or value not in {"WORLD", "LOCAL"}:
        return False
    state.orientation_mode = value
    # Keep Blender's visible orientation indicator aligned with the Fit gizmo
    # state. Fit computes true bone-local axes itself; do not relabel Local as
    # Blender's NORMAL orientation.
    context.scene.transform_orientation_slots[0].type = "GLOBAL" if value == "WORLD" else "LOCAL"
    if context.area is not None:
        context.area.tag_redraw()
    return True


def fit_orientation_mode(context) -> str:
    state = fit_ui_state(context)
    return state.orientation_mode if state is not None else "LOCAL"


def _rig_owner_for_character(scene, character_id: str):
    view = resolve_character(scene, character_id)
    owners: list[Any] = []
    seen: set[int] = set()
    for _binding_id, resolved in view.resolved_bindings:
        owner = resolved.owner_object
        pointer = _safe_pointer(owner)
        if pointer is None or pointer in seen:
            continue
        seen.add(pointer)
        owners.append(owner)
    armatures = tuple(owner for owner in owners if getattr(owner, "type", None) == "ARMATURE")
    if len(armatures) != 1:
        raise RigpedFitRuntimeError("FIT_UI_ARMATURE_OWNER_AMBIGUOUS")
    return armatures[0]


def rigped_box_wire_enabled(rig_object) -> bool:
    return bool(
        rig_object is not None
        and hasattr(rig_object, "get")
        and rig_object.get(_BIPED_BOX_WIRE_PROPERTY, False)
    )


def _authored_pose_bones(rig_object) -> tuple[Any, ...]:
    collection = rig_object.data.collections.get("Authored")
    if collection is None:
        return ()
    names = {bone.name for bone in collection.bones}
    return tuple(
        pose_bone
        for pose_bone in rig_object.pose.bones
        if pose_bone.name in names and not pose_bone.bone.hide
    )


def _scene_wire_color(scene, pose_bone) -> tuple[float, float, float]:
    name = str(pose_bone.name)
    if name == "Pelvis":
        value = getattr(scene, "baw_rigped_box_wire_pelvis_color", _BIPED_BOX_WIRE_DEFAULT_PELVIS)
    elif name == "COM":
        value = getattr(scene, "baw_rigped_box_wire_com_color", _BIPED_BOX_WIRE_DEFAULT_COM)
    elif name == "Head":
        value = getattr(scene, "baw_rigped_box_wire_head_color", _BIPED_BOX_WIRE_DEFAULT_HEAD)
    elif name.endswith(".L"):
        value = getattr(scene, "baw_rigped_box_wire_left_color", _BIPED_BOX_WIRE_DEFAULT_LEFT)
    elif name.endswith(".R"):
        value = getattr(scene, "baw_rigped_box_wire_right_color", _BIPED_BOX_WIRE_DEFAULT_RIGHT)
    else:
        value = getattr(scene, "baw_rigped_box_wire_center_color", _BIPED_BOX_WIRE_DEFAULT_CENTER)
    return tuple(float(component) for component in value[:3])


def _lighter(color: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    return tuple(min(1.0, component + ((1.0 - component) * amount)) for component in color)


def _apply_biped_box_wire_colors(rig_object, scene) -> None:
    rig_object.data.show_bone_colors = True
    for pose_bone in _authored_pose_bones(rig_object):
        normal = _scene_wire_color(scene, pose_bone)
        selected = tuple(
            float(component)
            for component in getattr(
                scene,
                "baw_rigped_box_wire_selected_color",
                _BIPED_BOX_WIRE_DEFAULT_SELECTED,
            )[:3]
        )
        active = tuple(
            float(component)
            for component in getattr(
                scene,
                "baw_rigped_box_wire_active_color",
                _BIPED_BOX_WIRE_DEFAULT_ACTIVE,
            )[:3]
        )
        for color in (pose_bone.bone.color, pose_bone.color):
            color.palette = "CUSTOM"
            color.custom.normal = normal
            color.custom.select = selected
            color.custom.active = active


def set_rigped_box_wire_display(rig_object, enabled: bool, scene=None) -> None:
    if rig_object is None or getattr(rig_object, "type", None) != "ARMATURE":
        raise RigpedFitRuntimeError("RIGPED_BOX_WIRE_REQUIRES_ARMATURE")

    scene = scene or bpy.context.scene
    _apply_biped_box_wire_colors(rig_object, scene)
    authored = _authored_pose_bones(rig_object)
    shape_object = bpy.data.objects.get(_BIPED_BOX_WIRE_SHAPE_OBJECT)
    if shape_object is not None:
        for pose_bone in authored:
            if pose_bone.custom_shape is shape_object:
                pose_bone.custom_shape = None

    # Box Display is presentation-only. Do not mutate Blender's native
    # armature display mode or custom-shape visibility. Suppress only Blender's
    # native bone viewport overlay so the authoritative bones stay usable while
    # AWB Box Wire becomes the sole visible rig representation.
    rig_object[_BIPED_BOX_WIRE_PROPERTY] = bool(enabled)
    from .rigped_box_wire_overlay import sync_native_bone_overlay_visibility

    sync_native_bone_overlay_visibility()


def update_rigped_box_wire_options(_owner, context) -> None:
    if context is None or getattr(context, "scene", None) is None:
        return
    character_id = active_rigped_character_id(context)
    if character_id is None:
        return
    try:
        rig = _rig_owner_for_character(context.scene, character_id)
        set_rigped_box_wire_display(
            rig,
            rigped_box_wire_enabled(rig),
            scene=context.scene,
        )
    except (CharacterMetadataError, RigpedFitRuntimeError, RuntimeError, ReferenceError):
        return


def active_rigped_character_id(context) -> str | None:
    active = getattr(context, "active_object", None)
    if _safe_pointer(active) is None:
        return None

    # Generated Rigped carries its descriptor on the Armature Object. This is
    # the hot UI path: avoid resolving the full Character graph on every N-panel
    # redraw while a bone is being transformed.
    if hasattr(active, "get"):
        raw = active.get(RIGPED_SETUP_PROPERTY)
        getter = getattr(raw, "get", None)
        if callable(getter):
            character_id = str(getter("character_id", "") or "")
            if character_id and character_id in character_ids(context.scene):
                return character_id

    active_ptr = _safe_pointer(active)
    matches: list[str] = []
    for character_id in character_ids(context.scene):
        view = resolve_character(context.scene, character_id)
        if any(
            _safe_pointer(resolved.owner_object) == active_ptr
            for _binding_id, resolved in view.resolved_bindings
        ):
            descriptor, issues = read_setup_descriptor(view)
            if descriptor is not None and not issues:
                matches.append(character_id)
    return matches[0] if len(matches) == 1 else None


def _ensure_object_mode() -> None:
    if getattr(bpy.context, "object", None) is not None and str(bpy.context.mode) != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def _select_only_object(context, obj) -> None:
    _ensure_object_mode()
    for candidate in context.scene.objects:
        candidate.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


def _capture_edit_rest(rig) -> tuple[_BoneRestSnapshot, ...]:
    return tuple(
        _BoneRestSnapshot(
            name=str(bone.name),
            head=tuple(float(value) for value in bone.head),
            tail=tuple(float(value) for value in bone.tail),
            roll=float(bone.roll),
            bbone_x=float(bone.bbone_x),
            bbone_z=float(bone.bbone_z),
            parent_name=str(bone.parent.name) if bone.parent is not None else None,
            use_connect=bool(bone.use_connect),
        )
        for bone in rig.data.edit_bones
    )


def _restore_edit_rest(rig, snapshot: tuple[_BoneRestSnapshot, ...]) -> None:
    edit_bones = rig.data.edit_bones
    current_names = {str(bone.name) for bone in edit_bones}
    expected_names = {entry.name for entry in snapshot}
    if current_names != expected_names:
        raise RigpedFitRuntimeError("FIT_CANCEL_TOPOLOGY_CHANGED_UNSUPPORTED")
    for entry in snapshot:
        bone = edit_bones.get(entry.name)
        if bone is None:
            raise RigpedFitRuntimeError("FIT_CANCEL_BONE_MISSING")
        bone.use_connect = False
        bone.head = entry.head
        bone.tail = entry.tail
        bone.roll = entry.roll
        bone.bbone_x = entry.bbone_x
        bone.bbone_z = entry.bbone_z
    for entry in snapshot:
        bone = edit_bones[entry.name]
        bone.parent = edit_bones.get(entry.parent_name) if entry.parent_name else None
        bone.use_connect = entry.use_connect


def capture_fit_structural_state(context) -> tuple[_BoneRestSnapshot, ...] | None:
    state = fit_ui_state(context)
    if state is None or str(getattr(context, "mode", "")) != "EDIT_ARMATURE":
        return None
    if _safe_pointer(getattr(context, "active_object", None)) != _safe_pointer(state.rig_object):
        return None
    return _capture_edit_rest(state.rig_object)


def restore_fit_structural_state(
    context,
    snapshot: tuple[_BoneRestSnapshot, ...],
) -> bool:
    state = fit_ui_state(context)
    if state is None or str(getattr(context, "mode", "")) != "EDIT_ARMATURE":
        return False
    if _safe_pointer(getattr(context, "active_object", None)) != _safe_pointer(state.rig_object):
        return False
    try:
        _restore_edit_rest(state.rig_object, snapshot)
    except RigpedFitRuntimeError:
        return False
    if context.area is not None:
        context.area.tag_redraw()
    return True


def record_fit_structural_change(
    context,
    before: tuple[_BoneRestSnapshot, ...],
) -> bool:
    state = fit_ui_state(context)
    after = capture_fit_structural_state(context)
    if state is None or after is None or after == before:
        return False
    if state.history_cursor < len(state.history):
        del state.history[state.history_cursor :]
    state.history.append(_FitHistoryEntry(before=before, after=after))
    state.history_cursor = len(state.history)
    return True


def undo_fit_structural_change(context) -> bool:
    state = fit_ui_state(context)
    if state is None or state.history_cursor <= 0:
        return False
    entry = state.history[state.history_cursor - 1]
    if not restore_fit_structural_state(context, entry.before):
        return False
    state.history_cursor -= 1
    return True


def redo_fit_structural_change(context) -> bool:
    state = fit_ui_state(context)
    if state is None or state.history_cursor >= len(state.history):
        return False
    entry = state.history[state.history_cursor]
    if not restore_fit_structural_state(context, entry.after):
        return False
    state.history_cursor += 1
    return True

def _enter_fit_box_wire_display(rig, scene) -> None:
    _apply_biped_box_wire_colors(rig, scene)
    if not rigped_box_wire_enabled(rig):
        return

    # Fit rollback: make Blender's native Edit bones visible first. The custom
    # Box Wire renderer is intentionally disabled in Edit mode until a stable
    # Fit-specific display path is proven.
    rig.data.display_type = "BBONE"
    from .rigped_box_wire_overlay import show_native_bone_overlays_for_fit

    show_native_bone_overlays_for_fit()


def _begin_fit(context, character_id: str) -> _FitUiState:
    window_key = _window_key(context)
    if window_key is None:
        raise RigpedFitRuntimeError("FIT_UI_WINDOW_UNAVAILABLE")
    if window_key in _FIT_STATES:
        raise RigpedFitRuntimeError("FIT_UI_ALREADY_ACTIVE")
    if any(state.character_id == character_id for state in _FIT_STATES.values()):
        raise RigpedFitRuntimeError("FIT_UI_CHARACTER_ACTIVE_IN_OTHER_WINDOW")

    inspection = inspect_fit_entry(context.scene, character_id)
    if not inspection.decision.allowed or inspection.session is None:
        codes = (*inspection.decision.issue_codes, *inspection.rigped_issue_codes, *inspection.runtime_issue_codes)
        raise RigpedFitRuntimeError(",".join(dict.fromkeys(codes)) or "FIT_UI_ENTRY_BLOCKED")

    rig = _rig_owner_for_character(context.scene, character_id)
    original_active = context.view_layer.objects.active
    original_selected = tuple(context.selected_objects)
    original_mode = str(context.mode)
    original_pivot_point = str(context.scene.tool_settings.transform_pivot_point)
    orientation_slot = context.scene.transform_orientation_slots[0]
    original_orientation = str(orientation_slot.type)
    _enter_fit_box_wire_display(rig, context.scene)
    _select_only_object(context, rig)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.wm.tool_set_by_id(name="baw.select_edit_armature")
    bpy.ops.armature.select_all(action="DESELECT")
    context.scene.tool_settings.transform_pivot_point = "ACTIVE_ELEMENT"
    # Fit owns its gizmo orientation state. Keep Blender's visible label on
    # LOCAL while the custom Fit gizmo computes the active bone's local basis.
    orientation_slot.type = "LOCAL"
    snapshot = _capture_edit_rest(rig)
    state = _FitUiState(
        character_id=character_id,
        session=inspection.session,
        rig_object=rig,
        rest_snapshot=snapshot,
        original_active=original_active,
        original_selected=original_selected,
        original_mode=original_mode,
        original_pivot_point=original_pivot_point,
        original_orientation=original_orientation,
    )
    _FIT_STATES[window_key] = state
    return state


def _restore_fit_transform_settings(context, state: _FitUiState) -> None:
    context.scene.tool_settings.transform_pivot_point = state.original_pivot_point
    context.scene.transform_orientation_slots[0].type = state.original_orientation
    if rigped_box_wire_enabled(state.rig_object):
        from .rigped_box_wire_overlay import sync_native_bone_overlay_visibility

        sync_native_bone_overlay_visibility()


def _semantic_root_pose_bone(scene, character_id: str):
    view = resolve_character(scene, character_id)
    matches = []
    binding_by_id = {binding.binding_id: binding for binding in view.definition.bindings}
    for binding_id, resolved in view.resolved_bindings:
        binding = binding_by_id.get(binding_id)
        if binding is None:
            continue
        if binding.semantic_key == "awb.root" and binding.usage.value == "PRIMARY":
            matches.append(resolved.target)
    return matches[0] if len(matches) == 1 else None


def _hide_animator_internal_pose_helpers(scene, character_id: str) -> None:
    rig = _rig_owner_for_character(scene, character_id)
    helper_collection = rig.data.collections.get("AWB Internal IK")
    if helper_collection is None:
        helper_collection = rig.data.collections.new("AWB Internal IK")
    helper_collection.is_visible = False

    helper_prefixes = ("IK_Hand.", "IK_Foot.", "IK_Elbow.", "IK_Knee.")
    for bone in rig.data.bones:
        if not str(bone.name).startswith(helper_prefixes):
            continue
        helper_collection.assign(bone)
        for collection in tuple(rig.data.collections):
            if collection == helper_collection:
                continue
            try:
                collection.unassign(bone)
            except RuntimeError:
                pass
        bone.hide = True
        bone.hide_select = True
    rig[_INTERNAL_HELPERS_HIDDEN_PROPERTY] = True


def _hide_rigped_internal_helpers_now() -> None:
    for scene in bpy.data.scenes:
        try:
            prune_stale_generated_rigped_metadata(scene)
            live_character_ids = character_ids(scene)
        except (CharacterMetadataError, RuntimeError, ReferenceError):
            continue
        for character_id in live_character_ids:
            try:
                _hide_animator_internal_pose_helpers(scene, character_id)
            except (CharacterMetadataError, RuntimeError, ReferenceError):
                continue


@persistent
def _hide_rigped_internal_helpers_after_load(*_args) -> None:
    _hide_rigped_internal_helpers_now()


def _hide_rigped_internal_helpers_timer():
    try:
        _hide_rigped_internal_helpers_now()
    except (AttributeError, RuntimeError, ReferenceError):
        return 0.1
    return None


def register_rigped_internal_visibility_handlers() -> None:
    handlers = bpy.app.handlers.load_post
    if _hide_rigped_internal_helpers_after_load not in handlers:
        handlers.append(_hide_rigped_internal_helpers_after_load)
    if not bpy.app.timers.is_registered(_hide_rigped_internal_helpers_timer):
        bpy.app.timers.register(_hide_rigped_internal_helpers_timer, first_interval=0.0)


def unregister_rigped_internal_visibility_handlers() -> None:
    handlers = bpy.app.handlers.load_post
    if _hide_rigped_internal_helpers_after_load in handlers:
        handlers.remove(_hide_rigped_internal_helpers_after_load)
    if bpy.app.timers.is_registered(_hide_rigped_internal_helpers_timer):
        bpy.app.timers.unregister(_hide_rigped_internal_helpers_timer)


def _transition_to_pose_root(context, character_id: str, rig) -> None:
    _select_only_object(context, rig)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.wm.tool_set_by_id(name="baw.select_pose")
    # Animate reset baseline: always enter Pose mode with no stale semantic
    # transform/IK tool state carried over from the previous implementation.
    if hasattr(context.scene, "baw_rigped_semantic_transform_mode"):
        context.scene.baw_rigped_semantic_transform_mode = "NONE"
    # Body Picker is opt-in because resolving/drawing the full picker on every
    # 3D View redraw is unnecessary during ordinary animation scrubbing.
    if hasattr(context.scene, "baw_rigped_body_picker_enabled"):
        context.scene.baw_rigped_body_picker_enabled = False
    _hide_animator_internal_pose_helpers(context.scene, character_id)
    bpy.ops.pose.select_all(action="DESELECT")
    root = _semantic_root_pose_bone(context.scene, character_id)
    if root is not None:
        root.select = True
        rig.data.bones.active = root.bone


def _transition_to_rig_selection(context, rig=None) -> None:
    _ensure_object_mode()
    bpy.ops.wm.tool_set_by_id(name="baw.select_object")
    if rig is not None:
        _select_only_object(context, rig)


def _restore_object_selection(context, state: _FitUiState) -> None:
    _ensure_object_mode()
    for obj in context.scene.objects:
        obj.select_set(False)
    for obj in state.original_selected:
        pointer = _safe_pointer(obj)
        if pointer is None:
            continue
        if any(_safe_pointer(candidate) == pointer for candidate in context.scene.objects):
            obj.select_set(True)
    active_ptr = _safe_pointer(state.original_active)
    context.view_layer.objects.active = next(
        (
            candidate
            for candidate in context.scene.objects
            if active_ptr is not None and _safe_pointer(candidate) == active_ptr
        ),
        None,
    )


def _viewport_placement_location(context, event, region) -> tuple[float, float, float]:
    region_3d = getattr(context.space_data, "region_3d", None)
    if region_3d is None:
        raise RigpedFitRuntimeError("CREATE_VIEW3D_REGION_STATE_UNAVAILABLE")
    coord = (float(event.mouse_x - region.x), float(event.mouse_y - region.y))
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, region_3d, coord)
    ray_direction = view3d_utils.region_2d_to_vector_3d(region, region_3d, coord)
    hit, location, _normal, _face, _obj, _matrix = context.scene.ray_cast(
        context.evaluated_depsgraph_get(),
        ray_origin,
        ray_direction,
    )
    if hit:
        return tuple(float(value) for value in location)
    if abs(float(ray_direction.z)) > 1e-8:
        distance = -float(ray_origin.z) / float(ray_direction.z)
        if distance >= 0.0:
            location = ray_origin + ray_direction * distance
            return (float(location.x), float(location.y), 0.0)
    fallback = view3d_utils.region_2d_to_location_3d(
        region,
        region_3d,
        coord,
        Vector((0.0, 0.0, 0.0)),
    )
    return (float(fallback.x), float(fallback.y), 0.0)


class BAW_OT_create_rigped_drag(bpy.types.Operator):
    bl_idname = "baw.create_rigped_drag"
    bl_label = "New Rigped"
    bl_description = ui_text("tooltip.rigped.new", language="EN")
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def description(cls, context, _properties):
        return ui_text("tooltip.rigped.new", context)

    _draw_handle = None
    _area = None
    _window_region = None
    _phase = "PLACE"
    _start_mouse_y = 0
    _mouse_y = 0
    _initial_height = 1.8
    _preview_height = 1.8
    _size_anchor_ready = False
    _created_result = None
    _original_active = None
    _original_selected = ()
    _original_mode = "OBJECT"
    _language = "EN"

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return bool(
            getattr(context, "area", None) is not None
            and context.area.type == "VIEW_3D"
            and str(getattr(context, "mode", "")) == "OBJECT"
            and fit_ui_state(context) is None
            and scene is not None
            and getattr(scene, "library", None) is None
            and getattr(scene, "is_editable", True)
        )

    def _set_header(self) -> None:
        if self._area is None:
            return
        if self._phase == "PLACE":
            text = ui_text("rigped.create_header.place", language=self._language)
        elif self._phase == "PLACED_WAIT_RELEASE":
            text = ui_text("rigped.create_header.release", language=self._language)
        else:
            text = ui_text(
                "rigped.create_header.size",
                language=self._language,
                height=self._preview_height,
            )
        self._area.header_text_set(text)

    def _cleanup(self) -> None:
        if self._draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, "WINDOW")
            self._draw_handle = None
        if self._area is not None:
            self._area.header_text_set(None)
            self._area.tag_redraw()

    def _draw_preview(self) -> None:
        font_id = 0
        blf.size(font_id, 14)
        blf.position(font_id, 24, 48, 0)
        if self._phase == "PLACE":
            blf.draw(font_id, ui_text("rigped.create_preview.place", language=self._language))
            return
        blf.draw(
            font_id,
            ui_text(
                "rigped.create_preview.height",
                language=self._language,
                height=self._preview_height,
            ),
        )
        blf.position(font_id, 24, 28, 0)
        blf.draw(font_id, ui_text("rigped.create_preview.controls", language=self._language))

    def invoke(self, context, event):
        self._area = context.area
        self._window_region = next(
            (region for region in context.area.regions if region.type == "WINDOW"),
            None,
        )
        if self._window_region is None:
            self.report({"ERROR"}, "Create Rigped requires a 3D View window region")
            return {"CANCELLED"}
        self._phase = "PLACE"
        self._start_mouse_y = int(event.mouse_y)
        self._mouse_y = int(event.mouse_y)
        self._initial_height = 1.8
        self._preview_height = 1.8
        self._size_anchor_ready = False
        self._created_result = None
        self._original_active = context.view_layer.objects.active
        self._original_selected = tuple(context.selected_objects)
        self._original_mode = str(context.mode)
        self._language = language_for_context(context)
        _ensure_object_mode()
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_preview,
            (),
            "WINDOW",
            "POST_PIXEL",
        )
        self._set_header()
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _mouse_in_window_region(self, event) -> bool:
        region = self._window_region
        return bool(
            region is not None
            and region.x <= event.mouse_x < region.x + region.width
            and region.y <= event.mouse_y < region.y + region.height
        )

    def _restore_original_selection(self, context) -> None:
        _ensure_object_mode()
        for obj in context.scene.objects:
            obj.select_set(False)
        for obj in self._original_selected:
            pointer = _safe_pointer(obj)
            if pointer is None:
                continue
            if any(_safe_pointer(candidate) == pointer for candidate in context.scene.objects):
                obj.select_set(True)
        active_ptr = _safe_pointer(self._original_active)
        context.view_layer.objects.active = next(
            (
                candidate
                for candidate in context.scene.objects
                if active_ptr is not None and _safe_pointer(candidate) == active_ptr
            ),
            None,
        )
        active = context.view_layer.objects.active
        if self._original_mode == "POSE" and active is not None and active.type == "ARMATURE":
            bpy.ops.object.mode_set(mode="POSE")

    def _cancel(self, context):
        self._cleanup()
        if self._created_result is not None:
            try:
                _ensure_object_mode()
                set_rigped_box_wire_display(
                    self._created_result.armature_object,
                    False,
                    scene=context.scene,
                )
                discard_generated_rigped_humanoid(context.scene, self._created_result)
            except RuntimeError as exc:
                self.report({"ERROR"}, f"Create Rigped cancel cleanup failed: {exc}")
                return {"CANCELLED"}
        self._restore_original_selection(context)
        return {"CANCELLED"}

    def _spawn_default(self, context, event):
        if self._window_region is None:
            return {"CANCELLED"}
        try:
            origin = _viewport_placement_location(context, event, self._window_region)
            box_size = float(getattr(context.scene, "baw_rigped_initial_box_size", 1.0))
            spec = fitted_humanoid_spec(
                self._initial_height,
                origin,
                base=rigify_reference_humanoid_spec(),
            )
            spec = replace(spec, display_scale=spec.display_scale * box_size)
            result = build_generated_rigped_humanoid(context.scene, spec)
            set_rigped_box_wire_display(
                result.armature_object,
                True,
                scene=context.scene,
            )
            self._created_result = result
            _select_only_object(context, result.armature_object)
        except (RuntimeError, ValueError) as exc:
            self._cleanup()
            self.report({"ERROR"}, f"Create Rigped failed: {exc}")
            return {"CANCELLED"}
        self._phase = "PLACED_WAIT_RELEASE"
        self._start_mouse_y = int(event.mouse_y)
        self._mouse_y = int(event.mouse_y)
        self._preview_height = self._initial_height
        self._set_header()
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _resize_created(self, context) -> None:
        result = self._created_result
        if result is None:
            return
        self._preview_height = height_from_free_mouse(
            self._initial_height,
            self._mouse_y - self._start_mouse_y,
        )
        scale = self._preview_height / self._initial_height
        result.armature_object.scale = (scale, scale, scale)
        self._set_header()
        context.area.tag_redraw()

    def _confirm(self, context):
        preview = self._created_result
        if preview is None:
            return {"CANCELLED"}
        self._cleanup()
        origin = tuple(float(value) for value in preview.armature_object.location)
        try:
            _ensure_object_mode()
            set_rigped_box_wire_display(
                preview.armature_object,
                False,
                scene=context.scene,
            )
            discard_generated_rigped_humanoid(context.scene, preview)
            final_spec = fitted_humanoid_spec(
                self._preview_height,
                origin,
                base=rigify_reference_humanoid_spec(),
            )
            final_spec = replace(
                final_spec,
                display_scale=final_spec.display_scale
                * float(getattr(context.scene, "baw_rigped_initial_box_size", 1.0)),
            )
            result = build_generated_rigped_humanoid(context.scene, final_spec)
            set_rigped_box_wire_display(
                result.armature_object,
                True,
                scene=context.scene,
            )
            self._created_result = result
            _select_only_object(context, result.armature_object)
        except (RuntimeError, ValueError) as exc:
            try:
                current = self._created_result
                if current is not None:
                    _ensure_object_mode()
                    set_rigped_box_wire_display(
                        current.armature_object,
                        False,
                        scene=context.scene,
                    )
                    discard_generated_rigped_humanoid(context.scene, current)
                self._restore_original_selection(context)
            except RuntimeError:
                pass
            self.report({"ERROR"}, f"Create Rigped confirm failed: {exc}")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Rigped created at height {self._preview_height:.3f}; choose Fit or Animate",
        )
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE", "WINDOW_DEACTIVATE"}:
            return self._cancel(context)

        if (
            event.type == "LEFTMOUSE"
            and event.value == "RELEASE"
            and self._phase == "PLACED_WAIT_RELEASE"
        ):
            self._phase = "SIZE"
            self._start_mouse_y = int(event.mouse_y)
            self._mouse_y = int(event.mouse_y)
            self._size_anchor_ready = False
            self._set_header()
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if (
            event.type == "LEFTMOUSE"
            and event.value == "PRESS"
            and self._mouse_in_window_region(event)
        ):
            if self._phase == "PLACE":
                return self._spawn_default(context, event)
            if self._phase == "SIZE":
                return self._confirm(context)
            return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE" and self._phase == "SIZE":
            self._mouse_y = int(event.mouse_y)
            if not self._size_anchor_ready:
                self._start_mouse_y = self._mouse_y
                self._size_anchor_ready = True
                return {"RUNNING_MODAL"}
            self._resize_created(context)
            return {"RUNNING_MODAL"}

        if self._phase in {"PLACED_WAIT_RELEASE", "SIZE"}:
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


class BAW_OT_rigped_selection_mode(bpy.types.Operator):
    bl_idname = "baw.rigped_selection_mode"
    bl_label = "Select"
    bl_description = ui_text("tooltip.rigped.select", language="EN")
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def description(cls, context, _properties):
        return ui_text("tooltip.rigped.select", context)

    @classmethod
    def poll(cls, context):
        return bool(
            getattr(context, "area", None) is not None
            and context.area.type == "VIEW_3D"
            and fit_ui_state(context) is None
            and str(getattr(context, "mode", "")) in {"OBJECT", "POSE"}
        )

    def execute(self, context):
        _transition_to_rig_selection(context)
        self.report({"INFO"}, "Select active")
        return {"FINISHED"}


class BAW_OT_rigped_box_wire_toggle(bpy.types.Operator):
    bl_idname = "baw.rigped_box_wire_toggle"
    bl_label = "Biped Box Wire"
    bl_description = "Toggle Biped-style wireframe box display for this Rigped"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if getattr(context, "area", None) is None or context.area.type != "VIEW_3D":
            return False
        character_id = active_rigped_character_id(context)
        return character_id is not None and fit_ui_state(context) is None

    def execute(self, context):
        character_id = active_rigped_character_id(context)
        if character_id is None:
            return {"CANCELLED"}
        try:
            rig = _rig_owner_for_character(context.scene, character_id)
            enabled = not rigped_box_wire_enabled(rig)
            set_rigped_box_wire_display(rig, enabled, scene=context.scene)
        except (CharacterMetadataError, RigpedFitRuntimeError, RuntimeError, ReferenceError) as exc:
            self.report({"ERROR"}, f"Biped Box Wire failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Biped Box Wire ON" if enabled else "Biped Box Wire OFF")
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_rigped_box_wire_reset(bpy.types.Operator):
    bl_idname = "baw.rigped_box_wire_reset"
    bl_label = "Reset Biped Colors"
    bl_description = "Restore the classic Biped wire width and color palette"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        scene.baw_rigped_box_wire_width = _BIPED_BOX_WIRE_DEFAULT_WIDTH
        scene.baw_rigped_box_wire_left_color = _BIPED_BOX_WIRE_DEFAULT_LEFT
        scene.baw_rigped_box_wire_right_color = _BIPED_BOX_WIRE_DEFAULT_RIGHT
        scene.baw_rigped_box_wire_center_color = _BIPED_BOX_WIRE_DEFAULT_CENTER
        scene.baw_rigped_box_wire_pelvis_color = _BIPED_BOX_WIRE_DEFAULT_PELVIS
        scene.baw_rigped_box_wire_head_color = _BIPED_BOX_WIRE_DEFAULT_HEAD
        scene.baw_rigped_box_wire_com_color = _BIPED_BOX_WIRE_DEFAULT_COM
        scene.baw_rigped_box_wire_selected_color = _BIPED_BOX_WIRE_DEFAULT_SELECTED
        scene.baw_rigped_box_wire_active_color = _BIPED_BOX_WIRE_DEFAULT_ACTIVE
        self.report({"INFO"}, "Biped Box Wire options reset")
        return {"FINISHED"}


class BAW_OT_rigped_animate_mode(bpy.types.Operator):
    bl_idname = "baw.rigped_animate_mode"
    bl_label = "Animate"
    bl_description = ui_text("tooltip.rigped.animate", language="EN")
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def description(cls, context, _properties):
        return ui_text("tooltip.rigped.animate", context)

    @classmethod
    def poll(cls, context):
        return bool(
            fit_ui_state(context) is None
            and str(getattr(context, "mode", "")) in {"OBJECT", "POSE"}
            and active_rigped_character_id(context) is not None
        )

    def execute(self, context):
        character_id = active_rigped_character_id(context)
        if character_id is None:
            self.report({"ERROR"}, "Select one Rigped first")
            return {"CANCELLED"}
        rig = _rig_owner_for_character(context.scene, character_id)
        ensure_generated_rigped_ik_hinge_limits(rig)
        try:
            _transition_to_pose_root(context, character_id, rig)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Animate Mode blocked: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Animate Mode active")
        return {"FINISHED"}


class BAW_OT_rigped_fit_on(bpy.types.Operator):
    bl_idname = "baw.rigped_fit_on"
    bl_label = "Fit"
    bl_description = ui_text("tooltip.rigped.fit", language="EN")
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def description(cls, context, _properties):
        return ui_text("tooltip.rigped.fit", context)

    @classmethod
    def poll(cls, context):
        return fit_ui_state(context) is None and active_rigped_character_id(context) is not None

    def execute(self, context):
        character_id = active_rigped_character_id(context)
        if character_id is None:
            self.report({"ERROR"}, "Select one Rigped first")
            return {"CANCELLED"}
        try:
            _begin_fit(context, character_id)
        except RigpedFitRuntimeError as exc:
            self.report({"ERROR"}, f"Fit Mode blocked: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_rigped_fit_off(bpy.types.Operator):
    bl_idname = "baw.rigped_fit_off"
    bl_label = "Apply Fit"
    bl_description = ui_text("tooltip.rigped.apply_fit", language="EN")
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def description(cls, context, _properties):
        return ui_text("tooltip.rigped.apply_fit", context)

    @classmethod
    def poll(cls, context):
        return fit_ui_state(context) is not None

    def execute(self, context):
        key = _window_key(context)
        state = fit_ui_state(context)
        if key is None or state is None:
            return {"CANCELLED"}
        try:
            _ensure_object_mode()
            result = commit_fit_session(context.scene, state.session)
        except (RigpedFitRuntimeError, RuntimeError) as exc:
            try:
                _select_only_object(context, state.rig_object)
                bpy.ops.object.mode_set(mode="EDIT")
            except RuntimeError:
                pass
            self.report({"ERROR"}, f"Fit commit failed: {exc}")
            return {"CANCELLED"}
        _FIT_STATES.pop(key, None)
        _restore_fit_transform_settings(context, state)
        _transition_to_rig_selection(context, state.rig_object)
        if result.decision.structural_change:
            self.report({"INFO"}, f"Fit committed; setup revision {result.setup_revision}")
        else:
            self.report({"INFO"}, "Fit closed with no structural change")
        return {"FINISHED"}


class BAW_OT_rigped_fit_cancel(bpy.types.Operator):
    bl_idname = "baw.rigped_fit_cancel"
    bl_label = "Reset Fit"
    bl_description = ui_text("tooltip.rigped.reset_fit", language="EN")
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def description(cls, context, _properties):
        return ui_text("tooltip.rigped.reset_fit", context)

    @classmethod
    def poll(cls, context):
        return fit_ui_state(context) is not None

    def execute(self, context):
        key = _window_key(context)
        state = fit_ui_state(context)
        if key is None or state is None:
            return {"CANCELLED"}
        try:
            _select_only_object(context, state.rig_object)
            bpy.ops.object.mode_set(mode="EDIT")
            _restore_edit_rest(state.rig_object, state.rest_snapshot)
            bpy.ops.object.mode_set(mode="OBJECT")
            view = resolve_character(context.scene, state.character_id)
            descriptor, issues = read_setup_descriptor(view)
            if descriptor is None or issues or descriptor.signature != state.session.setup_signature:
                raise RigpedFitRuntimeError("FIT_CANCEL_RESTORE_SIGNATURE_MISMATCH")
        except (RigpedFitRuntimeError, RuntimeError) as exc:
            self.report({"ERROR"}, f"Fit cancel could not restore exactly: {exc}")
            return {"CANCELLED"}
        _FIT_STATES.pop(key, None)
        _restore_fit_transform_settings(context, state)
        _transition_to_rig_selection(context, state.rig_object)
        self.report({"INFO"}, "Fit changes cancelled; setup revision unchanged")
        return {"FINISHED"}


_PICKER_ROLE_TEXT = {
    "awb.root": "rigped.picker.root",
    "awb.com": "rigped.picker.com",
    "awb.pelvis": "rigped.picker.pelvis",
    "awb.spine": "rigped.picker.spine",
    "awb.neck": "rigped.picker.neck",
    "awb.head": "rigped.picker.head",
    "awb.clavicle": "rigped.picker.clavicle",
    "awb.arm": "rigped.picker.arm",
    "awb.upper_arm": "rigped.picker.upper_arm",
    "awb.forearm": "rigped.picker.forearm",
    "awb.hand": "rigped.picker.hand",
    "awb.leg": "rigped.picker.leg",
    "awb.thigh": "rigped.picker.thigh",
    "awb.calf": "rigped.picker.calf",
    "awb.foot": "rigped.picker.foot",
    "awb.toe": "rigped.picker.toe",
}


def _picker_label(contract, context) -> str:
    role_key = _PICKER_ROLE_TEXT.get(contract.semantic_key)
    role = ui_text(role_key, context) if role_key is not None else contract.semantic_key
    side = {"LEFT": "L", "RIGHT": "R"}.get(contract.side.value, "")
    usage = contract.usage.value
    if usage == "TARGET":
        suffix = f" {ui_text('rigped.picker.suffix.ik', context)}"
    elif usage == "POLE":
        suffix = f" {ui_text('rigped.picker.suffix.pole', context)}"
    else:
        suffix = ""
    return " ".join(part for part in (side, f"{role}{suffix}") if part)


def _draw_body_picker(layout, context, character_id: str) -> None:
    try:
        view = resolve_character(context.scene, character_id)
    except CharacterMetadataError:
        return
    contracts, issues = control_contracts_for_view(view)
    if issues:
        return
    selectable = tuple(
        contract
        for contract in contracts
        if RigpedCapability.ANIMATOR_SELECTABLE in contract.capabilities
        and contract.usage.value not in {"TARGET", "POLE"}
    )
    if not selectable:
        return

    # Body Picker draw only needs to mirror current native selection. Do not run
    # full Rigped target validation here: that path recomputes the generated-rig
    # structural signature (bones/constraints/rest data + SHA256) and was being
    # executed on every N-panel redraw during frame scrubbing.
    control_context = control_context_for_context(context)
    selected_runtime_keys = {
        runtime_control_key(control)
        for control in control_context.controls
    }
    selected_ids = {
        str(binding_id)
        for binding_id, resolved in view.resolved_bindings
        if runtime_control_key(resolved) in selected_runtime_keys
    }

    picker = layout.box()
    picker.label(text=ui_text("rigped.picker.title", context), icon="BONE_DATA")
    grid = picker.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
    for contract in selectable:
        op = grid.operator(
            "baw.select_character_binding",
            text=_picker_label(contract, context),
            depress=contract.binding_id in selected_ids,
        )
        op.character_id = character_id
        op.binding_id = contract.binding_id
        op.operation = "REPLACE"


def draw_rigped_workflow(layout, context) -> None:
    box = layout.box()
    header = box.row(align=True)
    header.label(text=ui_text("rigped.workflow", context), icon="ARMATURE_DATA")

    state = fit_ui_state(context)
    if state is not None:
        status = box.row(align=True)
        status.label(text=ui_text("rigped.fit_mode", context), icon="EDITMODE_HLT")
        status.label(text=ui_text("rigped.fit_hint", context))
        actions = box.row(align=True)
        actions.operator(
            "baw.rigped_fit_off",
            text=ui_text("rigped.apply_fit", context),
            icon="CHECKMARK",
        )
        actions.operator(
            "baw.rigped_fit_cancel",
            text=ui_text("rigped.reset_fit", context),
            icon="LOOP_BACK",
        )
        return

    mode = str(getattr(context, "mode", ""))
    character_id = active_rigped_character_id(context)
    if mode == "POSE" and character_id is not None:
        active = getattr(context, "active_object", None)
        hidden_ready = bool(
            active is not None
            and hasattr(active, "get")
            and active.get(_INTERNAL_HELPERS_HIDDEN_PROPERTY, False)
        )
        if not hidden_ready:
            try:
                _hide_animator_internal_pose_helpers(context.scene, character_id)
            except (CharacterMetadataError, RuntimeError, ReferenceError):
                pass

    mode_row = box.row(align=True)
    mode_row.operator(
        "baw.rigped_selection_mode",
        text=ui_text("rigped.select", context),
        icon="OBJECT_DATA",
    )
    mode_row.operator(
        "baw.rigped_fit_on",
        text=ui_text("rigped.fit", context),
        icon="EDITMODE_HLT",
    )
    mode_row.operator(
        "baw.rigped_animate_mode",
        text=ui_text("rigped.animate", context),
        icon="POSE_HLT",
    )

    if character_id is not None:
        try:
            rig = _rig_owner_for_character(context.scene, character_id)
        except (CharacterMetadataError, RigpedFitRuntimeError, RuntimeError, ReferenceError):
            rig = None
        if rig is not None:
            wire_row = box.row(align=True)
            wire_row.operator(
                "baw.rigped_box_wire_toggle",
                text="Biped Box Wire",
                depress=rigped_box_wire_enabled(rig),
            )
            wire_row.prop(
                context.scene,
                "baw_rigped_box_wire_options_expanded",
                text="",
                icon="PREFERENCES",
                toggle=True,
            )
            if context.scene.baw_rigped_box_wire_options_expanded:
                options = box.box()
                options.prop(context.scene, "baw_rigped_box_wire_width", text="Wire Width")
                colors = options.grid_flow(
                    row_major=True,
                    columns=2,
                    even_columns=True,
                    align=True,
                )
                colors.prop(context.scene, "baw_rigped_box_wire_left_color", text="Left")
                colors.prop(context.scene, "baw_rigped_box_wire_right_color", text="Right")
                colors.prop(context.scene, "baw_rigped_box_wire_center_color", text="Center")
                colors.prop(context.scene, "baw_rigped_box_wire_pelvis_color", text="Pelvis")
                colors.prop(context.scene, "baw_rigped_box_wire_head_color", text="Head")
                colors.prop(context.scene, "baw_rigped_box_wire_com_color", text="COM")
                colors.prop(context.scene, "baw_rigped_box_wire_selected_color", text="Selected")
                colors.prop(context.scene, "baw_rigped_box_wire_active_color", text="Active")
                options.operator(
                    "baw.rigped_box_wire_reset",
                    text="Reset Biped Defaults",
                    icon="LOOP_BACK",
                )

    status = box.row(align=True)
    if mode == "POSE" and character_id is not None:
        status.label(text=ui_text("rigped.status.animate", context), icon="POSE_HLT")
    elif mode == "OBJECT":
        status.label(text=ui_text("rigped.status.select", context), icon="OBJECT_DATA")
    if character_id is not None:
        status.label(text=ui_text("rigped.current", context), icon="OUTLINER_OB_ARMATURE")

    if mode == "OBJECT":
        tune = box.row(align=True)
        tune.prop(
            context.scene,
            "baw_rigped_initial_box_size",
            text="Initial Box Size (Temp)",
        )
        box.operator(
            "baw.create_rigped_drag",
            text=ui_text("rigped.new", context),
            icon="ARMATURE_DATA",
        )
        box.label(text=ui_text("rigped.create_hint", context))
    elif mode == "POSE":
        if character_id is not None:
            picker_row = box.row(align=True)
            picker_row.prop(
                context.scene,
                "baw_rigped_body_picker_enabled",
                text="Body Picker",
                icon="BONE_DATA",
                toggle=True,
            )
            if bool(getattr(context.scene, "baw_rigped_body_picker_enabled", False)):
                _draw_body_picker(box, context, character_id)
        box.label(text=ui_text("rigped.select_hint", context), icon="INFO")
