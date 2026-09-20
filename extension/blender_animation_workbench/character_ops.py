from __future__ import annotations

from typing import ClassVar

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from .character_metadata import (
    SCHEMA_VERSION_V1,
    CharacterMetadataError,
    CharacterNotFound,
    UnsupportedCharacterSchema,
    add_character_chain,
    add_character_group,
    add_kinematic_mapping,
    add_opposite_pair,
    bind_object,
    bind_pose_bone,
    character_stamp,
    clone_character,
    create_character,
    migrate_store_v1_to_v2,
    rebind_pose_bone,
    remove_character,
    remove_character_binding,
    remove_character_chain,
    remove_character_group,
    remove_kinematic_mapping,
    remove_opposite_pair,
    resolve_character,
    set_binding_semantics,
    set_character_chain,
    set_character_group,
    set_kinematic_mapping,
)
from .character_model import CharacterSide, ControlMode, ControlUsage
from .character_query import selection_targets
from .semantic_adapter import control_context_for_context, runtime_control_key
from .semantic_model import AWBControlKind

_SELECTION_OPERATION_ITEMS = (
    ("REPLACE", "Replace", "Replace the current native selection with the semantic scope"),
    ("ADD", "Add", "Add the semantic scope to the current native selection"),
)
_SIDE_ITEMS = tuple((item.value, item.value.title(), f"Character side: {item.value}") for item in CharacterSide)
_MODE_ITEMS = tuple((item.value, item.value.title(), f"Control mode: {item.value}") for item in ControlMode)
_USAGE_ITEMS = tuple((item.value, item.value.title(), f"Control usage: {item.value}") for item in ControlUsage)


class CharacterSelectionError(RuntimeError):
    pass


class BAW_PG_clone_owner_map_item(bpy.types.PropertyGroup):
    source_owner_id: StringProperty(name="Source Owner ID")
    source_label: StringProperty(name="Source Owner")
    target_object: PointerProperty(name="Target Object", type=bpy.types.Object)


def _runtime_keys(controls) -> set[tuple[int, int]]:
    return {runtime_control_key(control) for control in controls}


def _first_target_or_active(targets, active, operation: str):
    target_by_key = {runtime_control_key(target): target for target in targets}
    if active is not None:
        active_key = runtime_control_key(active)
        if operation == "ADD" or active_key in target_by_key:
            return active
    return targets[0]


def _preflight_object_targets(context, targets) -> tuple:
    if getattr(context, "mode", "") != "OBJECT":
        raise CharacterSelectionError("Object Character targets require Object Mode.")
    if any(target.control.kind != AWBControlKind.OBJECT for target in targets):
        raise CharacterSelectionError("Object Mode selection cannot include PoseBone targets.")

    selectable = getattr(context, "selectable_objects", None)
    selectable_keys = (
        {int(obj.as_pointer()) for obj in selectable} if selectable is not None else None
    )
    objects = []
    for resolved in targets:
        obj = resolved.target
        if selectable_keys is not None and int(obj.as_pointer()) not in selectable_keys:
            raise CharacterSelectionError(f"Object {obj.name!r} is hidden or unselectable.")
        if bool(getattr(obj, "hide_select", False)):
            raise CharacterSelectionError(f"Object {obj.name!r} is not selectable.")
        if not obj.visible_get(view_layer=context.view_layer):
            raise CharacterSelectionError(f"Object {obj.name!r} is not visible in the current View Layer.")
        objects.append(obj)
    return tuple(objects)


def _preflight_pose_targets(context, targets) -> tuple[object, tuple]:
    if getattr(context, "mode", "") != "POSE":
        raise CharacterSelectionError("PoseBone Character targets require Pose Mode.")
    if any(target.control.kind != AWBControlKind.BONE for target in targets):
        raise CharacterSelectionError("Pose Mode selection cannot include Object targets.")

    owners = {int(target.owner_object.as_pointer()): target.owner_object for target in targets}
    if len(owners) != 1:
        raise CharacterSelectionError("Initial Character Pose selection supports one Armature Object.")
    owner = next(iter(owners.values()))
    if getattr(context, "active_object", None) is not owner:
        raise CharacterSelectionError("The Character Armature must already be the active Pose Object.")

    visible = getattr(context, "visible_pose_bones", None)
    visible_keys = (
        {int(pose_bone.as_pointer()) for pose_bone in visible} if visible is not None else None
    )
    pose_bones = []
    for resolved in targets:
        pose_bone = resolved.target
        if visible_keys is not None and int(pose_bone.as_pointer()) not in visible_keys:
            raise CharacterSelectionError(f"PoseBone {pose_bone.name!r} is hidden.")
        if bool(getattr(pose_bone, "hide", False)) or bool(
            getattr(pose_bone.bone, "hide_select", False)
        ):
            raise CharacterSelectionError(f"PoseBone {pose_bone.name!r} is hidden or unselectable.")
        pose_bones.append(pose_bone)
    return owner, tuple(pose_bones)


def _restore_object_selection(context, selected_objects, active_object) -> None:
    for obj in tuple(getattr(context, "selected_objects", ()) or ()):
        obj.select_set(False, view_layer=context.view_layer)
    for obj in selected_objects:
        obj.select_set(True, view_layer=context.view_layer)
    context.view_layer.objects.active = active_object


def _restore_pose_selection(owner, selected_pose_bones, active_pose_bone) -> None:
    for pose_bone in owner.pose.bones:
        pose_bone.select = False
    for pose_bone in selected_pose_bones:
        pose_bone.select = True
    owner.data.bones.active = active_pose_bone.bone if active_pose_bone is not None else None


def _verify_native_selection(context, before_context, targets, operation: str, expected_active) -> None:
    after_context = control_context_for_context(context)
    planned_keys = _runtime_keys(targets)
    before_keys = _runtime_keys(before_context.controls)
    expected_keys = planned_keys if operation == "REPLACE" else before_keys | planned_keys
    after_keys = _runtime_keys(after_context.controls)
    if after_keys != expected_keys:
        raise CharacterSelectionError("Native selection did not round-trip through Phase 2 ControlContext.")

    if expected_active is not None and (
        after_context.active is None
        or runtime_control_key(after_context.active) != runtime_control_key(expected_active)
    ):
        raise CharacterSelectionError("Native active-control state did not match the selection policy.")


def _apply_object_selection(context, targets, operation: str) -> None:
    objects = _preflight_object_targets(context, targets)
    before_context = control_context_for_context(context)
    selected_before = tuple(getattr(context, "selected_objects", ()) or ())
    active_before = getattr(context, "active_object", None)
    expected_active = _first_target_or_active(targets, before_context.active, operation)

    try:
        if operation == "REPLACE":
            for obj in selected_before:
                obj.select_set(False, view_layer=context.view_layer)
        for obj in objects:
            obj.select_set(True, view_layer=context.view_layer)
        context.view_layer.objects.active = expected_active.target
        _verify_native_selection(context, before_context, targets, operation, expected_active)
    except Exception:
        _restore_object_selection(context, selected_before, active_before)
        raise


def _apply_pose_selection(context, targets, operation: str) -> None:
    owner, pose_bones = _preflight_pose_targets(context, targets)
    before_context = control_context_for_context(context)
    selected_before = tuple(getattr(context, "selected_pose_bones", ()) or ())
    active_before = getattr(context, "active_pose_bone", None)
    expected_active = _first_target_or_active(targets, before_context.active, operation)

    try:
        if operation == "REPLACE":
            for pose_bone in owner.pose.bones:
                pose_bone.select = False
        for pose_bone in pose_bones:
            pose_bone.select = True
        owner.data.bones.active = expected_active.target.bone
        _verify_native_selection(context, before_context, targets, operation, expected_active)
    except Exception:
        _restore_pose_selection(owner, selected_before, active_before)
        raise


def _execute_semantic_selection(
    operator,
    context,
    *,
    character_id: str,
    operation: str,
    include_helpers: bool,
    binding_id: str | None = None,
    semantic_key: str | None = None,
    group_id: str | None = None,
    chain_id: str | None = None,
    mapping_id: str | None = None,
    current_mode_only: bool = False,
):
    character_id = character_id.strip()
    if not character_id:
        operator.report({"ERROR"}, "Character ID is required.")
        return {"CANCELLED"}

    try:
        view = resolve_character(context.scene, character_id)
        expansion = selection_targets(
            view,
            binding_id=binding_id,
            semantic_key=semantic_key,
            group_id=group_id,
            chain_id=chain_id,
            mapping_id=mapping_id,
            include_helpers=include_helpers,
        )
        targets = expansion.targets
        issues = expansion.issues
        if current_mode_only:
            mode_kind = {
                "OBJECT": AWBControlKind.OBJECT,
                "POSE": AWBControlKind.BONE,
            }.get(getattr(context, "mode", ""))
            if mode_kind is None:
                raise CharacterSelectionError("Character selection requires Object or Pose Mode.")
            binding_by_id = {
                binding.binding_id: binding for binding in view.definition.bindings
            }
            applicable_ids = {
                binding_id
                for binding_id in expansion.binding_ids
                if binding_by_id.get(binding_id) is not None
                and binding_by_id[binding_id].kind == mode_kind
            }
            targets = tuple(target for target in targets if target.control.kind == mode_kind)
            issues = tuple(
                issue
                for issue in issues
                if issue.record_id is None
                or issue.record_id in applicable_ids
                or issue.record_id not in binding_by_id
            )

        if issues:
            codes = ", ".join(dict.fromkeys(issue.code for issue in issues))
            raise CharacterSelectionError(f"Character selection scope is not applicable: {codes}")
        if not targets:
            raise CharacterSelectionError("Character selection scope resolved to no native targets.")

        kinds = {target.control.kind for target in targets}
        if kinds == {AWBControlKind.OBJECT}:
            _apply_object_selection(context, targets, operation)
        elif kinds == {AWBControlKind.BONE}:
            _apply_pose_selection(context, targets, operation)
        else:
            raise CharacterSelectionError(
                "Initial Character selection does not automatically cross Object/Pose modes."
            )
    except (CharacterMetadataError, CharacterNotFound, CharacterSelectionError) as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}

    operator.report({"INFO"}, f"Selected {len(targets)} Character target(s)")
    return {"FINISHED"}


def _selected_binding_ids_for_character(context, character_id: str) -> tuple[str, ...]:
    """Return selected Character bindings in authored binding order, never UI selection order."""

    view = resolve_character(context.scene, character_id)
    selected = control_context_for_context(context).controls
    if not selected:
        raise CharacterMetadataError("Select at least one Character control first.")

    selected_keys = _runtime_keys(selected)
    resolved_by_key: dict[tuple[int, int], list[str]] = {}
    for binding_id, resolved in view.resolved_bindings:
        resolved_by_key.setdefault(runtime_control_key(resolved), []).append(binding_id)

    missing = [key for key in selected_keys if key not in resolved_by_key]
    if missing:
        raise CharacterMetadataError(
            "All selected controls must already belong to the target Character before editing Group/Chain membership."
        )
    ambiguous = [ids for key, ids in resolved_by_key.items() if key in selected_keys and len(ids) != 1]
    if ambiguous:
        raise CharacterMetadataError(
            "Selected Character controls resolve to ambiguous bindings; repair membership before editing Group/Chain."
        )

    selected_ids = {
        ids[0]
        for key, ids in resolved_by_key.items()
        if key in selected_keys
    }
    return tuple(
        binding.binding_id
        for binding in view.definition.bindings
        if binding.binding_id in selected_ids
    )


def _group_by_id(view, group_id: str):
    return next((group for group in view.definition.groups if group.group_id == group_id), None)


def _chain_by_id(view, chain_id: str):
    return next((chain for chain in view.definition.chains if chain.chain_id == chain_id), None)


def _kinematic_by_id(view, mapping_id: str):
    return next(
        (mapping for mapping in view.definition.kinematics if mapping.mapping_id == mapping_id),
        None,
    )


def _write_kinematic_mapping(scene, view, mapping, **overrides) -> None:
    values = {
        "semantic_key": mapping.semantic_key,
        "side": mapping.side,
        "fk_chain_id": mapping.fk_chain_id,
        "ik_target_binding_id": mapping.ik_target_binding_id,
        "pole_binding_id": mapping.pole_binding_id,
        "reference_chain_id": mapping.reference_chain_id,
        "extras": mapping.extras,
    }
    values.update(overrides)
    set_kinematic_mapping(
        scene,
        view.definition.character_id,
        mapping.mapping_id,
        expected_stamp=view.source_stamp,
        **values,
    )


def _active_binding_id_for_character(context, character_id: str) -> str:
    active = control_context_for_context(context).active
    if active is None:
        raise CharacterMetadataError("Select one active Character control first.")
    active_key = runtime_control_key(active)
    view = resolve_character(context.scene, character_id)
    matches = [
        binding_id
        for binding_id, resolved in view.resolved_bindings
        if runtime_control_key(resolved) == active_key
    ]
    if len(matches) != 1:
        raise CharacterMetadataError(
            "The active control must resolve to exactly one binding in the target Character."
        )
    return matches[0]


def _opposite_pair_from_selection(context, character_id: str) -> tuple[str, str]:
    members = _selected_binding_ids_for_character(context, character_id)
    if len(members) != 2:
        raise CharacterMetadataError("Select exactly two Character controls for an opposite pair.")
    view = resolve_character(context.scene, character_id)
    binding_by_id = {binding.binding_id: binding for binding in view.definition.bindings}
    left = [binding_id for binding_id in members if binding_by_id[binding_id].side == CharacterSide.LEFT]
    right = [binding_id for binding_id in members if binding_by_id[binding_id].side == CharacterSide.RIGHT]
    if len(left) != 1 or len(right) != 1:
        raise CharacterMetadataError(
            "Opposite creation requires one selected LEFT binding and one selected RIGHT binding. "
            "Assign explicit side semantics first."
        )
    return left[0], right[0]


class BAW_OT_create_character(bpy.types.Operator):
    bl_idname = "baw.create_character"
    bl_label = "Create AWB Character"
    bl_description = "Create one Scene-local AWB Character from selected local Objects"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    label: StringProperty(name="Character Name", default="Character")

    @classmethod
    def poll(cls, context):
        if getattr(context, "mode", "") != "OBJECT":
            return False
        semantic_context = control_context_for_context(context)
        return any(
            resolved.control.kind == AWBControlKind.OBJECT
            for resolved in semantic_context.controls
        )

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        semantic_context = control_context_for_context(context)
        targets = tuple(
            resolved.target
            for resolved in semantic_context.controls
            if resolved.control.kind == AWBControlKind.OBJECT
        )
        if not targets:
            self.report({"WARNING"}, "Select at least one local Object.")
            return {"CANCELLED"}

        try:
            character_id = create_character(context.scene, self.label, targets)
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report({"INFO"}, f"Created AWB Character {character_id}")
        return {"FINISHED"}


class BAW_OT_clone_character(bpy.types.Operator):
    bl_idname = "baw.clone_character"
    bl_label = "Clone AWB Character"
    bl_description = "Clone one Character semantic graph through an explicit owner-to-Object map"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    source_character_id: StringProperty(name="Source Character ID", options={"HIDDEN"})
    source_revision: IntProperty(name="Source Revision", options={"HIDDEN"})
    label: StringProperty(name="Clone Name", default="Character Copy")

    def invoke(self, context, _event):
        try:
            view = resolve_character(context.scene, self.source_character_id)
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.source_revision = int(view.definition.revision)
        self.label = f"{view.definition.label} Copy"
        owner_map = context.window_manager.baw_clone_owner_map
        owner_map.clear()
        active_object = getattr(context, "active_object", None)
        for owner in view.definition.owners:
            item = owner_map.add()
            item.source_owner_id = owner.owner_id
            item.source_label = owner.label_hint
            if len(view.definition.owners) == 1 and active_object is not None:
                item.target_object = active_object
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "label")
        layout.label(text="Owners")
        for item in context.window_manager.baw_clone_owner_map:
            row = layout.row(align=True)
            row.label(text=item.source_label or item.source_owner_id[:8])
            row.prop(item, "target_object", text="")

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.source_character_id)
            if int(view.definition.revision) != int(self.source_revision):
                raise CharacterMetadataError(
                    "Source Character changed while the clone dialog was open; reopen Clone."
                )
            owner_map = {}
            for item in context.window_manager.baw_clone_owner_map:
                if item.target_object is None:
                    raise CharacterMetadataError(
                        f"Choose a target Object for source owner {item.source_label!r}."
                    )
                owner_map[item.source_owner_id] = item.target_object
            clone_id = clone_character(
                context.scene,
                self.source_character_id,
                owner_map,
                label=self.label,
                expected_stamp=view.source_stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        context.window_manager.baw_clone_owner_map.clear()
        self.report({"INFO"}, f"Cloned AWB Character {clone_id}")
        return {"FINISHED"}

    def cancel(self, context):
        context.window_manager.baw_clone_owner_map.clear()


class BAW_OT_remove_character(bpy.types.Operator):
    bl_idname = "baw.remove_character"
    bl_label = "Remove AWB Character"
    bl_description = "Remove one AWB Character metadata definition without deleting animation"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")

    def execute(self, context):
        character_id = self.character_id.strip()
        if not character_id:
            self.report({"ERROR"}, "Character ID is required.")
            return {"CANCELLED"}

        try:
            stamp = character_stamp(context.scene, character_id)
            remove_character(context.scene, character_id, expected_stamp=stamp)
        except CharacterNotFound:
            self.report({"ERROR"}, f"AWB Character {character_id!r} was not found.")
            return {"CANCELLED"}
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report({"INFO"}, f"Removed AWB Character {character_id}")
        return {"FINISHED"}


class BAW_OT_bind_character_active(bpy.types.Operator):
    bl_idname = "baw.bind_character_active"
    bl_label = "Bind Active to AWB Character"
    bl_description = "Explicitly bind the active selected Object or PoseBone to an existing AWB Character"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")

    @classmethod
    def poll(cls, context):
        active = control_context_for_context(context).active
        return active is not None and getattr(context, "mode", "") in {"OBJECT", "POSE"}

    def execute(self, context):
        character_id = self.character_id.strip()
        if not character_id:
            self.report({"ERROR"}, "Character ID is required.")
            return {"CANCELLED"}

        active = control_context_for_context(context).active
        if active is None:
            self.report({"ERROR"}, "Select one active Object or PoseBone to bind.")
            return {"CANCELLED"}

        try:
            stamp = character_stamp(context.scene, character_id)
            if active.control.kind == AWBControlKind.OBJECT:
                binding_id = bind_object(
                    context.scene,
                    character_id,
                    active.target,
                    expected_stamp=stamp,
                )
            elif active.control.kind == AWBControlKind.BONE:
                binding_id = bind_pose_bone(
                    context.scene,
                    character_id,
                    active.target,
                    expected_stamp=stamp,
                )
            else:
                self.report({"ERROR"}, "Unsupported active control kind.")
                return {"CANCELLED"}
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report({"INFO"}, f"Bound active control as AWB Character binding {binding_id}")
        return {"FINISHED"}


class BAW_OT_assign_character_semantics(bpy.types.Operator):
    bl_idname = "baw.assign_character_semantics"
    bl_label = "Edit AWB Character Semantics"
    bl_description = "Explicitly edit semantic metadata for one existing AWB Character binding"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID", options={"HIDDEN"})
    binding_id: StringProperty(name="Binding ID", options={"HIDDEN"})
    semantic_key: StringProperty(name="Semantic Key", default="")
    side: EnumProperty(name="Side", items=_SIDE_ITEMS, default=CharacterSide.NONE.value)
    mode: EnumProperty(name="Mode", items=_MODE_ITEMS, default=ControlMode.NEUTRAL.value)
    usage: EnumProperty(name="Usage", items=_USAGE_ITEMS, default=ControlUsage.PRIMARY.value)

    def invoke(self, context, _event):
        character_id = self.character_id.strip()
        binding_id = self.binding_id.strip()
        try:
            view = resolve_character(context.scene, character_id)
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        binding = next(
            (item for item in view.definition.bindings if item.binding_id == binding_id),
            None,
        )
        if binding is None:
            self.report({"ERROR"}, f"Character binding {binding_id!r} was not found.")
            return {"CANCELLED"}
        self.semantic_key = binding.semantic_key
        self.side = binding.side.value
        self.mode = binding.mode.value
        self.usage = binding.usage.value
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "semantic_key")
        layout.prop(self, "side")
        layout.prop(self, "mode")
        layout.prop(self, "usage")

    def execute(self, context):
        character_id = self.character_id.strip()
        binding_id = self.binding_id.strip()
        if not character_id or not binding_id:
            self.report({"ERROR"}, "Character ID and Binding ID are required.")
            return {"CANCELLED"}
        try:
            stamp = character_stamp(context.scene, character_id)
            set_binding_semantics(
                context.scene,
                character_id,
                binding_id,
                semantic_key=self.semantic_key.strip(),
                side=self.side,
                mode=self.mode,
                usage=self.usage,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Updated AWB Character binding {binding_id}")
        return {"FINISHED"}


class BAW_OT_repair_character_bone_binding(bpy.types.Operator):
    bl_idname = "baw.repair_character_bone_binding"
    bl_label = "Repair AWB Character Bone Binding"
    bl_description = "Explicitly repair one missing or ambiguous Bone binding onto the active PoseBone"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    binding_id: StringProperty(name="Binding ID")

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "mode", "") == "POSE"
            and getattr(context, "active_pose_bone", None) is not None
            and bool(getattr(context.active_pose_bone, "select", False))
        )

    def execute(self, context):
        try:
            stamp = character_stamp(context.scene, self.character_id)
            token = rebind_pose_bone(
                context.scene,
                self.character_id,
                self.binding_id,
                context.active_pose_bone,
                expected_stamp=stamp,
                repair_duplicate_token=True,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Repaired AWB Bone binding with token {token}")
        return {"FINISHED"}


class BAW_OT_remove_character_binding(bpy.types.Operator):
    bl_idname = "baw.remove_character_binding"
    bl_label = "Remove AWB Character Binding"
    bl_description = "Remove one AWB Character membership binding without deleting native data"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    binding_id: StringProperty(name="Binding ID")

    def execute(self, context):
        character_id = self.character_id.strip()
        binding_id = self.binding_id.strip()
        if not character_id or not binding_id:
            self.report({"ERROR"}, "Character ID and Binding ID are required.")
            return {"CANCELLED"}
        try:
            stamp = character_stamp(context.scene, character_id)
            remove_character_binding(
                context.scene,
                character_id,
                binding_id,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Removed AWB Character binding {binding_id}")
        return {"FINISHED"}


class BAW_OT_create_character_group_from_selection(bpy.types.Operator):
    bl_idname = "baw.create_character_group_from_selection"
    bl_label = "Create AWB Character Group"
    bl_description = "Create an unordered Character group from currently selected controls"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID", options={"HIDDEN"})
    label: StringProperty(name="Label", default="Group")
    semantic_key: StringProperty(name="Semantic Key", default="custom.group")
    side: EnumProperty(name="Side", items=_SIDE_ITEMS, default=CharacterSide.NONE.value)

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop(self, "label")
        self.layout.prop(self, "semantic_key")
        self.layout.prop(self, "side")

    def execute(self, context):
        try:
            stamp = character_stamp(context.scene, self.character_id)
            members = _selected_binding_ids_for_character(context, self.character_id)
            group_id = add_character_group(
                context.scene,
                self.character_id,
                semantic_key=self.semantic_key.strip(),
                label=self.label.strip() or "Group",
                side=self.side,
                members=members,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created AWB Character group {group_id}")
        return {"FINISHED"}


class BAW_OT_edit_character_group(bpy.types.Operator):
    bl_idname = "baw.edit_character_group"
    bl_label = "Edit AWB Character Group"
    bl_description = "Edit group metadata while preserving its explicit member set"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID", options={"HIDDEN"})
    group_id: StringProperty(name="Group ID", options={"HIDDEN"})
    label: StringProperty(name="Label")
    semantic_key: StringProperty(name="Semantic Key")
    side: EnumProperty(name="Side", items=_SIDE_ITEMS, default=CharacterSide.NONE.value)

    def invoke(self, context, _event):
        try:
            group = _group_by_id(resolve_character(context.scene, self.character_id), self.group_id)
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if group is None:
            self.report({"ERROR"}, f"Character group {self.group_id!r} was not found.")
            return {"CANCELLED"}
        self.label = group.label
        self.semantic_key = group.semantic_key
        self.side = group.side.value
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop(self, "label")
        self.layout.prop(self, "semantic_key")
        self.layout.prop(self, "side")

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            group = _group_by_id(view, self.group_id)
            if group is None:
                raise CharacterMetadataError(f"Character group {self.group_id!r} was not found.")
            set_character_group(
                context.scene,
                self.character_id,
                self.group_id,
                semantic_key=self.semantic_key.strip(),
                label=self.label.strip() or "Group",
                side=self.side,
                parent_group_id=group.parent_group_id,
                members=group.members,
                expected_stamp=view.source_stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_set_character_group_members_from_selection(bpy.types.Operator):
    bl_idname = "baw.set_character_group_members_from_selection"
    bl_label = "Set Group Members from Selection"
    bl_description = "Replace group members with currently selected Character controls"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    group_id: StringProperty(name="Group ID")

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            group = _group_by_id(view, self.group_id)
            if group is None:
                raise CharacterMetadataError(f"Character group {self.group_id!r} was not found.")
            members = _selected_binding_ids_for_character(context, self.character_id)
            set_character_group(
                context.scene,
                self.character_id,
                self.group_id,
                semantic_key=group.semantic_key,
                label=group.label,
                side=group.side,
                parent_group_id=group.parent_group_id,
                members=members,
                expected_stamp=view.source_stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_remove_character_group(bpy.types.Operator):
    bl_idname = "baw.remove_character_group"
    bl_label = "Remove AWB Character Group"
    bl_description = "Remove one Character group only when no semantic record still references it"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    group_id: StringProperty(name="Group ID")

    def execute(self, context):
        try:
            stamp = character_stamp(context.scene, self.character_id)
            remove_character_group(
                context.scene,
                self.character_id,
                self.group_id,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_create_character_chain_from_selection(bpy.types.Operator):
    bl_idname = "baw.create_character_chain_from_selection"
    bl_label = "Create AWB Character Chain"
    bl_description = "Create a chain from selected Character controls in authored binding order; reorder explicitly afterward"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID", options={"HIDDEN"})
    label: StringProperty(name="Label", default="Chain")
    semantic_key: StringProperty(name="Semantic Key", default="custom.chain")
    side: EnumProperty(name="Side", items=_SIDE_ITEMS, default=CharacterSide.NONE.value)
    mode: EnumProperty(name="Mode", items=_MODE_ITEMS, default=ControlMode.NEUTRAL.value)

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop(self, "label")
        self.layout.prop(self, "semantic_key")
        self.layout.prop(self, "side")
        self.layout.prop(self, "mode")

    def execute(self, context):
        try:
            stamp = character_stamp(context.scene, self.character_id)
            members = _selected_binding_ids_for_character(context, self.character_id)
            chain_id = add_character_chain(
                context.scene,
                self.character_id,
                semantic_key=self.semantic_key.strip(),
                label=self.label.strip() or "Chain",
                side=self.side,
                mode=self.mode,
                members=members,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created AWB Character chain {chain_id}")
        return {"FINISHED"}


class BAW_OT_edit_character_chain(bpy.types.Operator):
    bl_idname = "baw.edit_character_chain"
    bl_label = "Edit AWB Character Chain"
    bl_description = "Edit chain metadata while preserving its explicit authored member order"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID", options={"HIDDEN"})
    chain_id: StringProperty(name="Chain ID", options={"HIDDEN"})
    label: StringProperty(name="Label")
    semantic_key: StringProperty(name="Semantic Key")
    side: EnumProperty(name="Side", items=_SIDE_ITEMS, default=CharacterSide.NONE.value)
    mode: EnumProperty(name="Mode", items=_MODE_ITEMS, default=ControlMode.NEUTRAL.value)

    def invoke(self, context, _event):
        try:
            chain = _chain_by_id(resolve_character(context.scene, self.character_id), self.chain_id)
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if chain is None:
            self.report({"ERROR"}, f"Character chain {self.chain_id!r} was not found.")
            return {"CANCELLED"}
        self.label = chain.label
        self.semantic_key = chain.semantic_key
        self.side = chain.side.value
        self.mode = chain.mode.value
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop(self, "label")
        self.layout.prop(self, "semantic_key")
        self.layout.prop(self, "side")
        self.layout.prop(self, "mode")

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            chain = _chain_by_id(view, self.chain_id)
            if chain is None:
                raise CharacterMetadataError(f"Character chain {self.chain_id!r} was not found.")
            set_character_chain(
                context.scene,
                self.character_id,
                self.chain_id,
                semantic_key=self.semantic_key.strip(),
                label=self.label.strip() or "Chain",
                side=self.side,
                mode=self.mode,
                members=chain.members,
                expected_stamp=view.source_stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_set_character_chain_members_from_selection(bpy.types.Operator):
    bl_idname = "baw.set_character_chain_members_from_selection"
    bl_label = "Reset Chain Members from Selection"
    bl_description = "Replace chain members in authored Character binding order; use Up/Down to author final order"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    chain_id: StringProperty(name="Chain ID")

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            chain = _chain_by_id(view, self.chain_id)
            if chain is None:
                raise CharacterMetadataError(f"Character chain {self.chain_id!r} was not found.")
            members = _selected_binding_ids_for_character(context, self.character_id)
            set_character_chain(
                context.scene,
                self.character_id,
                self.chain_id,
                semantic_key=chain.semantic_key,
                label=chain.label,
                side=chain.side,
                mode=chain.mode,
                members=members,
                expected_stamp=view.source_stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_move_character_chain_member(bpy.types.Operator):
    bl_idname = "baw.move_character_chain_member"
    bl_label = "Move AWB Chain Member"
    bl_description = "Move one chain member up or down in the authoritative authored order"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    chain_id: StringProperty(name="Chain ID")
    member_index: IntProperty(name="Member Index", min=0)
    direction: EnumProperty(
        name="Direction",
        items=(("UP", "Up", "Move earlier"), ("DOWN", "Down", "Move later")),
        default="UP",
    )

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            chain = _chain_by_id(view, self.chain_id)
            if chain is None:
                raise CharacterMetadataError(f"Character chain {self.chain_id!r} was not found.")
            members = list(chain.members)
            index = int(self.member_index)
            target_index = index - 1 if self.direction == "UP" else index + 1
            if index >= len(members) or target_index < 0 or target_index >= len(members):
                raise CharacterMetadataError("Chain member is already at that order boundary.")
            members[index], members[target_index] = members[target_index], members[index]
            set_character_chain(
                context.scene,
                self.character_id,
                self.chain_id,
                semantic_key=chain.semantic_key,
                label=chain.label,
                side=chain.side,
                mode=chain.mode,
                members=tuple(members),
                expected_stamp=view.source_stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_remove_character_chain(bpy.types.Operator):
    bl_idname = "baw.remove_character_chain"
    bl_label = "Remove AWB Character Chain"
    bl_description = "Remove one Character chain only when no kinematic mapping still references it"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    chain_id: StringProperty(name="Chain ID")

    def execute(self, context):
        try:
            stamp = character_stamp(context.scene, self.character_id)
            remove_character_chain(
                context.scene,
                self.character_id,
                self.chain_id,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_create_opposite_from_selection(bpy.types.Operator):
    bl_idname = "baw.create_opposite_from_selection"
    bl_label = "Create AWB Opposite Pair"
    bl_description = "Create an explicit LEFT/RIGHT opposite relation from exactly two selected Character controls"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")

    def execute(self, context):
        try:
            stamp = character_stamp(context.scene, self.character_id)
            left_binding_id, right_binding_id = _opposite_pair_from_selection(
                context,
                self.character_id,
            )
            add_opposite_pair(
                context.scene,
                self.character_id,
                left_binding_id,
                right_binding_id,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Created explicit AWB opposite pair")
        return {"FINISHED"}


class BAW_OT_remove_opposite_pair(bpy.types.Operator):
    bl_idname = "baw.remove_opposite_pair"
    bl_label = "Remove AWB Opposite Pair"
    bl_description = "Remove one explicit Character opposite relation"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    left_binding_id: StringProperty(name="Left Binding ID")
    right_binding_id: StringProperty(name="Right Binding ID")

    def execute(self, context):
        try:
            stamp = character_stamp(context.scene, self.character_id)
            remove_opposite_pair(
                context.scene,
                self.character_id,
                self.left_binding_id,
                self.right_binding_id,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_create_kinematic_mapping(bpy.types.Operator):
    bl_idname = "baw.create_kinematic_mapping"
    bl_label = "Create AWB Kinematic Mapping"
    bl_description = "Create a generic authored kinematic relationship without solver behavior"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID", options={"HIDDEN"})
    fk_chain_id: StringProperty(name="FK Chain ID", options={"HIDDEN"})
    ik_target_binding_id: StringProperty(name="IK Target Binding ID", options={"HIDDEN"})
    semantic_key: StringProperty(name="Semantic Key", default="custom.kinematic")
    side: EnumProperty(name="Side", items=_SIDE_ITEMS, default=CharacterSide.NONE.value)

    def invoke(self, context, _event):
        try:
            view = resolve_character(context.scene, self.character_id)
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if self.fk_chain_id:
            chain = _chain_by_id(view, self.fk_chain_id)
            if chain is None:
                self.report({"ERROR"}, f"Character chain {self.fk_chain_id!r} was not found.")
                return {"CANCELLED"}
            self.semantic_key = chain.semantic_key
            self.side = chain.side.value
        elif self.ik_target_binding_id:
            binding = next(
                (
                    item
                    for item in view.definition.bindings
                    if item.binding_id == self.ik_target_binding_id
                ),
                None,
            )
            if binding is None:
                self.report({"ERROR"}, "IK target binding was not found.")
                return {"CANCELLED"}
            self.semantic_key = binding.semantic_key or "custom.kinematic"
            self.side = binding.side.value
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop(self, "semantic_key")
        self.layout.prop(self, "side")

    def execute(self, context):
        if not self.fk_chain_id and not self.ik_target_binding_id:
            self.report({"ERROR"}, "Seed the mapping from an FK chain or active IK target.")
            return {"CANCELLED"}
        try:
            stamp = character_stamp(context.scene, self.character_id)
            mapping_id = add_kinematic_mapping(
                context.scene,
                self.character_id,
                semantic_key=self.semantic_key.strip(),
                side=self.side,
                fk_chain_id=self.fk_chain_id or None,
                ik_target_binding_id=self.ik_target_binding_id or None,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created AWB kinematic mapping {mapping_id}")
        return {"FINISHED"}


class BAW_OT_edit_kinematic_mapping(bpy.types.Operator):
    bl_idname = "baw.edit_kinematic_mapping"
    bl_label = "Edit AWB Kinematic Mapping"
    bl_description = "Edit semantic metadata for a generic kinematic relationship"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID", options={"HIDDEN"})
    mapping_id: StringProperty(name="Mapping ID", options={"HIDDEN"})
    semantic_key: StringProperty(name="Semantic Key")
    side: EnumProperty(name="Side", items=_SIDE_ITEMS, default=CharacterSide.NONE.value)

    def invoke(self, context, _event):
        try:
            mapping = _kinematic_by_id(
                resolve_character(context.scene, self.character_id),
                self.mapping_id,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if mapping is None:
            self.report({"ERROR"}, f"Kinematic mapping {self.mapping_id!r} was not found.")
            return {"CANCELLED"}
        self.semantic_key = mapping.semantic_key
        self.side = mapping.side.value
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop(self, "semantic_key")
        self.layout.prop(self, "side")

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            mapping = _kinematic_by_id(view, self.mapping_id)
            if mapping is None:
                raise CharacterMetadataError(f"Kinematic mapping {self.mapping_id!r} was not found.")
            _write_kinematic_mapping(
                context.scene,
                view,
                mapping,
                semantic_key=self.semantic_key.strip(),
                side=self.side,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_set_kinematic_chain_role(bpy.types.Operator):
    bl_idname = "baw.set_kinematic_chain_role"
    bl_label = "Set AWB Kinematic Chain Role"
    bl_description = "Assign an authored Character chain as FK or reference chain"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    mapping_id: StringProperty(name="Mapping ID")
    chain_id: StringProperty(name="Chain ID")
    role: EnumProperty(
        name="Role",
        items=(
            ("FK_CHAIN", "FK Chain", "Use this chain as the FK chain"),
            ("REFERENCE_CHAIN", "Reference Chain", "Use this chain as the reference chain"),
        ),
        default="FK_CHAIN",
    )

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            mapping = _kinematic_by_id(view, self.mapping_id)
            if mapping is None:
                raise CharacterMetadataError(f"Kinematic mapping {self.mapping_id!r} was not found.")
            if _chain_by_id(view, self.chain_id) is None:
                raise CharacterMetadataError(f"Character chain {self.chain_id!r} was not found.")
            key = "fk_chain_id" if self.role == "FK_CHAIN" else "reference_chain_id"
            _write_kinematic_mapping(context.scene, view, mapping, **{key: self.chain_id})
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_set_kinematic_active_role(bpy.types.Operator):
    bl_idname = "baw.set_kinematic_active_role"
    bl_label = "Set AWB Kinematic Active Role"
    bl_description = "Use the active Character control as IK target or pole"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    mapping_id: StringProperty(name="Mapping ID")
    role: EnumProperty(
        name="Role",
        items=(
            ("IK_TARGET", "IK Target", "Use active control as IK target"),
            ("POLE", "Pole", "Use active control as pole"),
        ),
        default="IK_TARGET",
    )

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            mapping = _kinematic_by_id(view, self.mapping_id)
            if mapping is None:
                raise CharacterMetadataError(f"Kinematic mapping {self.mapping_id!r} was not found.")
            binding_id = _active_binding_id_for_character(context, self.character_id)
            key = "ik_target_binding_id" if self.role == "IK_TARGET" else "pole_binding_id"
            _write_kinematic_mapping(context.scene, view, mapping, **{key: binding_id})
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_set_kinematic_extras_from_selection(bpy.types.Operator):
    bl_idname = "baw.set_kinematic_extras_from_selection"
    bl_label = "Set AWB Kinematic Extras"
    bl_description = "Replace kinematic extras with selected Character controls"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    mapping_id: StringProperty(name="Mapping ID")

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            mapping = _kinematic_by_id(view, self.mapping_id)
            if mapping is None:
                raise CharacterMetadataError(f"Kinematic mapping {self.mapping_id!r} was not found.")
            extras = _selected_binding_ids_for_character(context, self.character_id)
            _write_kinematic_mapping(context.scene, view, mapping, extras=extras)
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_clear_kinematic_role(bpy.types.Operator):
    bl_idname = "baw.clear_kinematic_role"
    bl_label = "Clear AWB Kinematic Role"
    bl_description = "Clear one optional kinematic relationship field"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    mapping_id: StringProperty(name="Mapping ID")
    role: EnumProperty(
        name="Role",
        items=(
            ("FK_CHAIN", "FK Chain", "Clear FK chain"),
            ("IK_TARGET", "IK Target", "Clear IK target"),
            ("POLE", "Pole", "Clear pole"),
            ("REFERENCE_CHAIN", "Reference Chain", "Clear reference chain"),
            ("EXTRAS", "Extras", "Clear extras"),
        ),
        default="POLE",
    )

    def execute(self, context):
        try:
            view = resolve_character(context.scene, self.character_id)
            mapping = _kinematic_by_id(view, self.mapping_id)
            if mapping is None:
                raise CharacterMetadataError(f"Kinematic mapping {self.mapping_id!r} was not found.")
            overrides = {
                "FK_CHAIN": {"fk_chain_id": None},
                "IK_TARGET": {"ik_target_binding_id": None},
                "POLE": {"pole_binding_id": None},
                "REFERENCE_CHAIN": {"reference_chain_id": None},
                "EXTRAS": {"extras": ()},
            }[self.role]
            _write_kinematic_mapping(context.scene, view, mapping, **overrides)
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_remove_kinematic_mapping(bpy.types.Operator):
    bl_idname = "baw.remove_kinematic_mapping"
    bl_label = "Remove AWB Kinematic Mapping"
    bl_description = "Remove one generic kinematic relationship without touching animation data"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    mapping_id: StringProperty(name="Mapping ID")

    def execute(self, context):
        try:
            stamp = character_stamp(context.scene, self.character_id)
            remove_kinematic_mapping(
                context.scene,
                self.character_id,
                self.mapping_id,
                expected_stamp=stamp,
            )
        except CharacterMetadataError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BAW_OT_migrate_character_schema_v2(bpy.types.Operator):
    bl_idname = "baw.migrate_character_schema_v2"
    bl_label = "Migrate AWB Characters to Schema v2"
    bl_description = "Explicitly upgrade locator-only Character metadata to semantic-graph schema v2"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        store = getattr(getattr(context, "scene", None), "awb_characters", None)
        return store is not None and int(getattr(store, "schema_version", 0)) == SCHEMA_VERSION_V1

    def execute(self, context):
        try:
            migrated = migrate_store_v1_to_v2(context.scene)
        except (CharacterMetadataError, UnsupportedCharacterSchema) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Migrated {len(migrated)} AWB Character(s) to schema v2")
        return {"FINISHED"}


def _selection_operator_poll(context) -> bool:
    return getattr(context, "mode", "") in {"OBJECT", "POSE"}


class BAW_OT_select_character(bpy.types.Operator):
    bl_idname = "baw.select_character"
    bl_label = "Select AWB Character"
    bl_description = "Apply one AWB Character definition to Blender native selection"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    operation: EnumProperty(name="Operation", items=_SELECTION_OPERATION_ITEMS, default="REPLACE")
    include_helpers: BoolProperty(name="Include Helpers", default=False)

    @classmethod
    def poll(cls, context):
        return _selection_operator_poll(context)

    def execute(self, context):
        return _execute_semantic_selection(
            self,
            context,
            character_id=self.character_id,
            operation=self.operation,
            include_helpers=self.include_helpers,
            current_mode_only=True,
        )


class BAW_OT_select_character_binding(bpy.types.Operator):
    bl_idname = "baw.select_character_binding"
    bl_label = "Select AWB Character Control"
    bl_description = "Resolve one authored Character binding into Blender native selection"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    binding_id: StringProperty(name="Binding ID")
    operation: EnumProperty(name="Operation", items=_SELECTION_OPERATION_ITEMS, default="REPLACE")

    @classmethod
    def poll(cls, context):
        return _selection_operator_poll(context)

    def execute(self, context):
        binding_id = self.binding_id.strip()
        if not binding_id:
            self.report({"ERROR"}, "Character Binding ID is required.")
            return {"CANCELLED"}
        return _execute_semantic_selection(
            self,
            context,
            character_id=self.character_id,
            operation=self.operation,
            include_helpers=False,
            binding_id=binding_id,
            current_mode_only=True,
        )


class BAW_OT_select_character_group(bpy.types.Operator):
    bl_idname = "baw.select_character_group"
    bl_label = "Select AWB Character Group"
    bl_description = "Apply one authored AWB Character group to Blender native selection"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    group_id: StringProperty(name="Group ID")
    operation: EnumProperty(name="Operation", items=_SELECTION_OPERATION_ITEMS, default="REPLACE")
    include_helpers: BoolProperty(name="Include Helpers", default=False)

    @classmethod
    def poll(cls, context):
        return _selection_operator_poll(context)

    def execute(self, context):
        group_id = self.group_id.strip()
        if not group_id:
            self.report({"ERROR"}, "Character Group ID is required.")
            return {"CANCELLED"}
        return _execute_semantic_selection(
            self,
            context,
            character_id=self.character_id,
            operation=self.operation,
            include_helpers=self.include_helpers,
            group_id=group_id,
        )


class BAW_OT_select_character_chain(bpy.types.Operator):
    bl_idname = "baw.select_character_chain"
    bl_label = "Select AWB Character Chain"
    bl_description = "Apply one ordered AWB Character chain to Blender native selection"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    chain_id: StringProperty(name="Chain ID")
    operation: EnumProperty(name="Operation", items=_SELECTION_OPERATION_ITEMS, default="REPLACE")
    include_helpers: BoolProperty(name="Include Helpers", default=False)

    @classmethod
    def poll(cls, context):
        return _selection_operator_poll(context)

    def execute(self, context):
        chain_id = self.chain_id.strip()
        if not chain_id:
            self.report({"ERROR"}, "Character Chain ID is required.")
            return {"CANCELLED"}
        return _execute_semantic_selection(
            self,
            context,
            character_id=self.character_id,
            operation=self.operation,
            include_helpers=self.include_helpers,
            chain_id=chain_id,
        )


class BAW_OT_select_character_kinematic(bpy.types.Operator):
    bl_idname = "baw.select_character_kinematic"
    bl_label = "Select AWB Character Kinematic Mapping"
    bl_description = "Apply one AWB kinematic relationship to Blender native selection"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    character_id: StringProperty(name="Character ID")
    mapping_id: StringProperty(name="Mapping ID")
    operation: EnumProperty(name="Operation", items=_SELECTION_OPERATION_ITEMS, default="REPLACE")
    include_helpers: BoolProperty(name="Include Helpers", default=False)

    @classmethod
    def poll(cls, context):
        return _selection_operator_poll(context)

    def execute(self, context):
        mapping_id = self.mapping_id.strip()
        if not mapping_id:
            self.report({"ERROR"}, "Kinematic Mapping ID is required.")
            return {"CANCELLED"}
        return _execute_semantic_selection(
            self,
            context,
            character_id=self.character_id,
            operation=self.operation,
            include_helpers=self.include_helpers,
            mapping_id=mapping_id,
        )
