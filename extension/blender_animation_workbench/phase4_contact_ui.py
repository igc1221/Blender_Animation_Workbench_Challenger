from __future__ import annotations

from typing import ClassVar

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty

from .gizmo_preferences import (
    draw_global_gizmo_preferences,
    schedule_user_preferences_save,
    sync_global_gizmo_size,
    sync_global_rotation_step,
)
from .phase4_contact_authoring import (
    ContactAuthoringError,
    bind_selected_contact_object,
    clear_contact_bundle_key_selection_for_context,
    clear_selected_contact_object,
    contact_point_preset_for_selection,
    execute_contact_command,
    execute_contact_replant,
)
from .phase4_contact_model import ContactKeyType, ContactPlantSpace
from .rigped_operation_domain import resolve_operation_domain
from .semantic_adapter import control_context_for_context
from .trackbar_model import clear_key_selection_for_context
from .ui_language import text

_CONTACT_KEYMAP_ITEMS: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []

_ORIENTATION_PREFERENCES = (
    ("GLOBAL", "orientation_global_enabled"),
    ("LOCAL", "orientation_local_enabled"),
    ("VIEW", "orientation_view_enabled"),
)

_SHORTCUT_ITEMS = tuple(
    (letter, letter, f"Use {letter} for the AWB Contact command")
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
) + tuple(
    (f"F{index}", f"F{index}", f"Use F{index} for the AWB Contact command")
    for index in range(1, 13)
)

_CONTACT_POINT_PRESETS = (
    ("REAR_LEFT", "RL", "Rear-left terminal-local Contact point"),
    ("REAR_CENTER", "RC", "Rear-center/origin terminal-local Contact point"),
    ("REAR_RIGHT", "RR", "Rear-right terminal-local Contact point"),
    ("MID_LEFT", "ML", "Mid-left terminal-local Contact point"),
    ("MID_CENTER", "MC", "Mid-center terminal-local Contact point"),
    ("MID_RIGHT", "MR", "Mid-right terminal-local Contact point"),
    ("FRONT_LEFT", "FL", "Front-left terminal-local Contact point"),
    ("FRONT_CENTER", "FC", "Front-center terminal-local Contact point"),
    ("FRONT_RIGHT", "FR", "Front-right terminal-local Contact point"),
)


def _addon_preferences(context):
    preferences = getattr(context, "preferences", None)
    addons = getattr(preferences, "addons", None) if preferences is not None else None
    if addons is None:
        return None
    addon = addons.get(__package__)
    if addon is not None:
        return getattr(addon, "preferences", None)
    package_tail = __package__.split(".")[-1]
    for key, candidate in addons.items():
        if str(key).split(".")[-1] == package_tail:
            return getattr(candidate, "preferences", None)
    return None


def transform_orientation_cycle(context) -> tuple[str, ...]:
    preferences = _addon_preferences(context)
    if preferences is None:
        return ("GLOBAL", "LOCAL", "VIEW")
    enabled = tuple(
        orientation
        for orientation, attribute in _ORIENTATION_PREFERENCES
        if bool(getattr(preferences, attribute, False))
    )
    return enabled or ("GLOBAL",)


def contact_enabled_types(context) -> tuple[ContactKeyType, ...]:
    """Return the enabled A6 Free/Sliding/Planted C-cycle states."""

    default_cycle = (
        ContactKeyType.FREE,
        ContactKeyType.SLIDING,
        ContactKeyType.PLANTED,
    )
    preferences = _addon_preferences(context)
    if preferences is None:
        return default_cycle
    enabled: list[ContactKeyType] = []
    if bool(preferences.contact_free_enabled):
        enabled.append(ContactKeyType.FREE)
    sliding_enabled = bool(preferences.contact_sliding_enabled)
    if sliding_enabled:
        enabled.append(ContactKeyType.SLIDING)
    if sliding_enabled and bool(preferences.contact_planted_enabled):
        enabled.append(ContactKeyType.PLANTED)
    return tuple(enabled) or default_cycle


def contact_shortcut_event_type(context) -> str:
    preferences = _addon_preferences(context)
    value = str(getattr(preferences, "contact_shortcut", "C")) if preferences is not None else "C"
    return value or "C"


def contact_plant_space(context) -> ContactPlantSpace:
    scene = getattr(context, "scene", None)
    value = str(getattr(scene, "baw_contact_plant_space", ContactPlantSpace.WORLD.value)) if scene is not None else ContactPlantSpace.WORLD.value
    try:
        return ContactPlantSpace(value)
    except ValueError:
        return ContactPlantSpace.WORLD


def _contact_preferences_updated(_self, context) -> None:
    if bpy.app.background:
        return
    register_contact_keymaps()
    area = getattr(context, "area", None)
    if area is not None:
        area.tag_redraw()


def _gizmo_preferences_updated(self, context) -> None:
    if bpy.app.background:
        return
    sync_global_gizmo_size(context, self)
    sync_global_rotation_step(context, self)
    schedule_user_preferences_save()
    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        return
    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class BAW_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    contact_shortcut: EnumProperty(
        name="Contact",
        description="Shortcut for authoring/cycling the current Rigped Contact key",
        items=_SHORTCUT_ITEMS,
        default="C",
        update=_contact_preferences_updated,
    )
    contact_free_enabled: BoolProperty(
        name="Free",
        description="Include Free in the Contact cycle",
        default=True,
    )
    contact_sliding_enabled: BoolProperty(
        name="Sliding",
        description="Include Sliding in the Contact cycle",
        default=True,
    )
    contact_planted_enabled: BoolProperty(
        name="Planted",
        description="Include Planted in the Contact cycle",
        default=True,
    )
    orientation_global_enabled: BoolProperty(
        name="Global",
        description="Include Global in repeated W/E/R Transform Orientation cycling",
        default=True,
    )
    orientation_local_enabled: BoolProperty(
        name="Local",
        description="Include Local in repeated W/E/R Transform Orientation cycling",
        default=True,
    )
    orientation_view_enabled: BoolProperty(
        name="View",
        description="Include View in repeated W/E/R Transform Orientation cycling",
        default=True,
    )
    gizmo_size_px: FloatProperty(
        name="",
        description="",
        default=75.0,
        min=10.0,
        max=200.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_zoom_compensation_percent: FloatProperty(
        name="",
        description="",
        default=50.0,
        min=0.0,
        max=200.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_zoom_out_max_percent: FloatProperty(
        name="",
        description="",
        default=400.0,
        min=100.0,
        max=400.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotate_axis_width: FloatProperty(
        name="",
        description="",
        default=2.0,
        min=1.0,
        max=6.0,
        precision=1,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotate_view_width: FloatProperty(
        name="",
        description="",
        default=1.5,
        min=1.0,
        max=6.0,
        precision=1,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotate_view_scale_percent: FloatProperty(
        name="",
        description="",
        default=115.0,
        min=100.0,
        max=160.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_free_rotate_sensitivity_percent: FloatProperty(
        name="",
        description="",
        default=100.0,
        min=25.0,
        max=300.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotate_x_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.86, 0.16, 0.12),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotate_y_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.20, 0.72, 0.18),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotate_z_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.16, 0.38, 0.92),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotate_view_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.85, 0.85, 0.85),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotate_axis_hit_width: FloatProperty(
        name="Axis Ring Hit Width",
        description="Invisible mouse hit width for the X/Y/Z rotation rings",
        default=8.0,
        min=3.0,
        max=18.0,
        precision=1,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotate_view_hit_width: FloatProperty(
        name="View Ring Hit Width",
        description="Invisible mouse hit width for the outer View rotation ring",
        default=6.0,
        min=3.0,
        max=18.0,
        precision=1,
        update=_gizmo_preferences_updated,
    )
    gizmo_rotation_step_degrees: FloatProperty(
        name="",
        description="",
        default=1.0,
        min=0.1,
        max=180.0,
        precision=1,
        update=_gizmo_preferences_updated,
    )
    gizmo_show_rotation_angle: BoolProperty(
        name="",
        description="",
        default=True,
        update=_gizmo_preferences_updated,
    )
    gizmo_section_common_open: BoolProperty(
        name="",
        description="",
        default=True,
        update=_gizmo_preferences_updated,
    )
    gizmo_section_rotate_open: BoolProperty(
        name="",
        description="",
        default=True,
        update=_gizmo_preferences_updated,
    )
    gizmo_section_move_open: BoolProperty(
        name="",
        description="",
        default=True,
        update=_gizmo_preferences_updated,
    )
    gizmo_section_scale_open: BoolProperty(
        name="",
        description="",
        default=False,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_axis_width: FloatProperty(
        name="",
        description="",
        default=1.0,
        min=0.5,
        max=6.0,
        precision=1,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_arrow_length_percent: FloatProperty(
        name="",
        description="",
        default=100.0,
        min=40.0,
        max=220.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_arrow_head_size_percent: FloatProperty(
        name="",
        description="",
        default=100.0,
        min=50.0,
        max=200.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_plane_size_percent: FloatProperty(
        name="",
        description="",
        default=100.0,
        min=40.0,
        max=220.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_plane_width: FloatProperty(
        name="Move Plane Outline Width",
        description="Visible outline width of the Move plane handles",
        default=1.2,
        min=0.5,
        max=6.0,
        precision=1,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_plane_color: FloatVectorProperty(
        name="Move Plane Color",
        description="Outline color shared by all Move plane handles",
        subtype="COLOR",
        size=3,
        default=(0.52, 0.52, 0.52),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_center_size_percent: FloatProperty(
        name="Move Center Size (%)",
        description="Size of the free Move center handle",
        default=100.0,
        min=40.0,
        max=220.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_x_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.86, 0.16, 0.12),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_y_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.20, 0.72, 0.18),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_z_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.16, 0.38, 0.92),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_move_center_color: FloatVectorProperty(
        name="Move Center Color",
        description="Color of the free Move center handle",
        subtype="COLOR",
        size=3,
        default=(0.82, 0.82, 0.82),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_scale_axis_width: FloatProperty(
        name="",
        description="",
        default=1.0,
        min=0.5,
        max=6.0,
        precision=1,
        update=_gizmo_preferences_updated,
    )
    gizmo_scale_axis_length_percent: FloatProperty(
        name="",
        description="",
        default=100.0,
        min=40.0,
        max=220.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_scale_handle_size_percent: FloatProperty(
        name="",
        description="",
        default=100.0,
        min=50.0,
        max=200.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_scale_uniform_size_percent: FloatProperty(
        name="",
        description="",
        default=100.0,
        min=40.0,
        max=220.0,
        precision=0,
        update=_gizmo_preferences_updated,
    )
    gizmo_scale_x_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.86, 0.16, 0.12),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_scale_y_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.20, 0.72, 0.18),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )
    gizmo_scale_z_color: FloatVectorProperty(
        name="",
        description="",
        subtype="COLOR",
        size=3,
        default=(0.16, 0.38, 0.92),
        min=0.0,
        max=1.0,
        update=_gizmo_preferences_updated,
    )

    def draw(self, context):
        layout = self.layout
        shortcuts = layout.box()
        shortcuts.label(text="Shortcuts")
        shortcuts.prop(self, "contact_shortcut")

        contact = layout.box()
        contact.label(text="Contact Cycle")
        row = contact.row(align=True)
        row.prop(self, "contact_free_enabled", toggle=True)
        row.prop(self, "contact_sliding_enabled", toggle=True)
        row.prop(self, "contact_planted_enabled", toggle=True)

        orientation = layout.box()
        orientation.label(text=text("rigped.orientation_cycle", context))
        first = orientation.row(align=True)
        first.prop(self, "orientation_global_enabled", text=text("rigped.orientation.global", context), toggle=True)
        first.prop(self, "orientation_local_enabled", text=text("rigped.orientation.local", context), toggle=True)
        first.prop(self, "orientation_view_enabled", text=text("rigped.orientation.view", context), toggle=True)

        draw_global_gizmo_preferences(layout, context, self)


class BAW_OT_bind_contact_object(bpy.types.Operator):
    bl_idname = "baw.bind_contact_object"
    bl_label = "Bind Contact Object"
    bl_description = "Bind the selected Rigped hand/foot Contact domain to one rigid external Object"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (
            context.mode == "POSE"
            and scene is not None
            and getattr(scene, "baw_contact_object_target", None) is not None
            and bool(control_context_for_context(context).controls)
        )

    def execute(self, context):
        result = bind_selected_contact_object(
            context.scene,
            control_context_for_context(context),
            context.scene.baw_contact_object_target,
        )
        if not result.applied:
            detail = result.diagnostics[0].detail if result.diagnostics else "Contact Object bind failed"
            self.report({"WARNING"}, detail)
            return {"CANCELLED"}
        self.report({"INFO"}, text("rigped.contact.object_bound", context))
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_clear_contact_object(bpy.types.Operator):
    bl_idname = "baw.clear_contact_object"
    bl_label = "Clear Contact Object"
    bl_description = "Clear the selected Rigped limb external Contact Object when no Object-space Contact keys depend on it"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "POSE" and bool(control_context_for_context(context).controls)

    def execute(self, context):
        result = clear_selected_contact_object(
            context.scene,
            control_context_for_context(context),
        )
        if not result.applied:
            detail = result.diagnostics[0].detail if result.diagnostics else "Contact Object clear failed"
            self.report({"WARNING"}, detail)
            return {"CANCELLED"}
        self.report({"INFO"}, text("rigped.contact.object_cleared", context))
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_set_contact_point_preset(bpy.types.Operator):
    bl_idname = "baw.set_contact_point_preset"
    bl_label = "Set Contact Point Preset"
    bl_description = "Set the generalized terminal-local Contact point without changing animation until Plant/Replant"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    preset: EnumProperty(items=_CONTACT_POINT_PRESETS)

    @classmethod
    def poll(cls, context):
        return context.mode == "POSE" and bool(control_context_for_context(context).controls)

    def execute(self, context):
        try:
            value = contact_point_preset_for_selection(
                context.scene,
                control_context_for_context(context),
                self.preset,
            )
        except ContactAuthoringError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        context.scene.baw_contact_point_local = value
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_replant_contact(bpy.types.Operator):
    bl_idname = "baw.replant_contact"
    bl_label = "Replant Contact"
    bl_description = "Explicitly replace the current Planted anchor using the displayed continuous Contact point"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "POSE" and bool(control_context_for_context(context).controls)

    def execute(self, context):
        result = execute_contact_replant(
            context.scene,
            control_context_for_context(context),
            operation_id="baw-contact-replant",
            plant_space=contact_plant_space(context),
            contact_point_local=tuple(float(value) for value in context.scene.baw_contact_point_local),
        )
        if not result.applied:
            detail = result.diagnostics[0].detail if result.diagnostics else "Contact Replant failed"
            self.report({"WARNING"}, detail)
            return {"CANCELLED"}
        self.report({"INFO"}, text("rigped.contact.replanted", context))
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


def draw_contact_workflow(layout, context) -> None:
    if str(getattr(context, "mode", "")) != "POSE":
        return
    scene = getattr(context, "scene", None)
    if scene is None:
        return

    box = layout.box()
    box.label(text=text("rigped.contact.title", context), icon="CONSTRAINT_BONE")
    space = box.row(align=True)
    space.prop(scene, "baw_contact_plant_space", expand=True)

    box.prop(scene, "baw_contact_point_local", text=text("rigped.contact.point", context))
    presets = box.column(align=True)
    presets.enabled = contact_plant_space(context) is ContactPlantSpace.WORLD
    for row_presets in (_CONTACT_POINT_PRESETS[0:3], _CONTACT_POINT_PRESETS[3:6], _CONTACT_POINT_PRESETS[6:9]):
        row = presets.row(align=True)
        for preset_id, label, _description in row_presets:
            operator = row.operator(BAW_OT_set_contact_point_preset.bl_idname, text=label)
            operator.preset = preset_id
    box.operator(
        BAW_OT_replant_contact.bl_idname,
        text=text("rigped.contact.replant", context),
        icon="PIVOT_CURSOR",
    )

    if contact_plant_space(context) is ContactPlantSpace.OBJECT:
        box.prop(scene, "baw_contact_object_target", text=text("rigped.contact.object", context))
        actions = box.row(align=True)
        actions.operator(
            BAW_OT_bind_contact_object.bl_idname,
            text=text("rigped.contact.bind", context),
            icon="LINKED",
        )
        actions.operator(
            BAW_OT_clear_contact_object.bl_idname,
            text=text("rigped.contact.clear", context),
            icon="UNLINKED",
        )
        box.label(text=text("rigped.contact.object_hint", context), icon="INFO")
        box.label(text=text("rigped.contact.object_point_hint", context), icon="INFO")


class BAW_OT_contact(bpy.types.Operator):
    bl_idname = "baw.contact"
    bl_label = "Contact"
    bl_description = "Author the current Rigped Contact key; repeat on the same frame to cycle enabled Contact types"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "POSE" and bool(control_context_for_context(context).controls)

    def invoke(self, context, event):
        # Blender emits repeated PRESS events while a physical key is held.
        # I13 treats one deliberate physical press as exactly one Contact author.
        if bool(getattr(event, "is_repeat", False)):
            return {"CANCELLED"}
        return self.execute(context)

    def execute(self, context):
        from .debug_trace import link_trace_operation, new_trace_operation_id, trace_event

        trace_operation_id = new_trace_operation_id("contact")
        trace_event(
            "INPUT",
            "CONTACT_BEGIN",
            operation_id=trace_operation_id,
            context=context,
            enabled_types=tuple(item.value for item in contact_enabled_types(context)),
            plant_space=contact_plant_space(context).value,
            contact_point_local=tuple(
                float(value) for value in context.scene.baw_contact_point_local
            ),
        )

        # Joint limits are Rigped pose authority, not just gizmo guards. Repair
        # any stale/imported/out-of-range evaluated pose before Contact snapshots
        # or keys can bake it into the Action.
        from .rigped_transform import apply_rigped_joint_limits_to_rig

        rig = getattr(context, "active_object", None)
        if (
            rig is not None
            and getattr(rig, "type", None) == "ARMATURE"
            and apply_rigped_joint_limits_to_rig(rig)
        ):
            context.view_layer.update()

        control_context = control_context_for_context(context)
        replay_domain = resolve_operation_domain(context.scene, control_context)
        replay_direct_binding_ids = (
            tuple(
                str(binding_id)
                for binding_id in replay_domain.snapshot.supported_direct_binding_ids
            )
            if replay_domain.snapshot is not None and not replay_domain.issues
            else ()
        )
        link_trace_operation("baw-contact", trace_operation_id)
        result = execute_contact_command(
            context.scene,
            control_context,
            operation_id="baw-contact",
            enabled_types=contact_enabled_types(context),
            plant_space=contact_plant_space(context),
            contact_point_local=tuple(float(value) for value in context.scene.baw_contact_point_local),
        )
        if not result.applied:
            detail = result.diagnostics[0].detail if result.diagnostics else "Contact authoring failed"
            trace_event(
                "OPERATION",
                "CONTACT_FAIL",
                operation_id=trace_operation_id,
                context=context,
                writer_operation_id="baw-contact",
                diagnostics=tuple(
                    {"code": item.code, "detail": item.detail}
                    for item in result.diagnostics
                ),
            )
            self.report({"WARNING"}, detail)
            return {"CANCELLED"}

        replay_contact_type = result.contact_type.value if result.contact_type is not None else None
        replay_action = {
            "kind": "CONTACT",
            "frame": int(context.scene.frame_current),
            "subframe": float(getattr(context.scene, "frame_subframe", 0.0)),
            "controls": tuple(
                str(item.name)
                for item in tuple(getattr(context, "selected_pose_bones", ()) or ())
            ),
            "contact_type": replay_contact_type,
        }
        if result.mapping_contact_types:
            replay_action["mapping_contact_types"] = tuple(
                (mapping_id, contact_type.value)
                for mapping_id, contact_type in result.mapping_contact_types
            )
        if replay_direct_binding_ids:
            replay_action["direct_binding_ids"] = replay_direct_binding_ids
        trace_event(
            "OPERATION",
            "CONTACT_COMMIT",
            operation_id=trace_operation_id,
            context=context,
            writer_operation_id="baw-contact",
            contact_type=replay_contact_type,
            mapping_contact_types=tuple(
                (mapping_id, contact_type.value)
                for mapping_id, contact_type in result.mapping_contact_types
            ),
            direct_binding_ids=replay_direct_binding_ids,
            rows_written=int(result.rows_written),
            created_fcurves=int(result.created_fcurves),
            replay_action=replay_action,
        )

        if result.contact_type is None:
            labels = ", ".join(
                f"{mapping_id}: {contact_type.value.title()}"
                for mapping_id, contact_type in result.mapping_contact_types
            ) or "Mixed"
            self.report({"INFO"}, f"Contact: {labels}")
        else:
            self.report({"INFO"}, f"Contact: {result.contact_type.value.title()}")

            # Keep an already-active W semantic tool coherent across a C state
            # transition. Without this, Free -> Sliding leaves FK_MOVE active and
            # Sliding -> Free leaves MOVE active; the old gizmo then fails its poll
            # and appears to vanish until W is pressed again.
            semantic_mode = str(
                getattr(context.scene, "baw_rigped_semantic_transform_mode", "NONE")
            )
            if semantic_mode in {"FK_MOVE", "MOVE"}:
                if result.contact_type is ContactKeyType.FREE:
                    context.scene.baw_rigped_semantic_transform_mode = "FK_MOVE"
                elif result.contact_type is ContactKeyType.SLIDING:
                    context.scene.baw_rigped_semantic_transform_mode = "MOVE"
                else:
                    context.scene.baw_rigped_semantic_transform_mode = "NONE"
                space = getattr(context, "space_data", None)
                if space is not None and getattr(space, "type", None) == "VIEW_3D":
                    space.show_gizmo = True
                    if hasattr(space, "show_gizmo_tool"):
                        space.show_gizmo_tool = False

        # C authoring must not leave freshly-created Contact/transform keys
        # selected. A selected key at the previous frame makes the next C press
        # look like a Track Bar selection range, which is not part of the C
        # authoring contract.
        frame = float(context.scene.frame_current) + float(
            getattr(context.scene, "frame_subframe", 0.0)
        )
        clear_contact_bundle_key_selection_for_context(context, frames=(frame,))
        clear_key_selection_for_context(context)
        context.scene.baw_has_selected_key = False

        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


def register_contact_keymaps() -> None:
    unregister_contact_keymaps()
    wm = bpy.context.window_manager
    if wm is None or wm.keyconfigs.addon is None:
        return
    kc = wm.keyconfigs.addon
    # Contact is a Pose-mode Rigped command. Keep its binding in Blender's Pose
    # keymap and leave the ordinary Camera C fallback in the generic 3D View
    # keymap. Putting both C items in the same 3D View map made their priority
    # depend on registration order and could route a valid Rigped C press to the
    # camera action instead of Contact.
    km = kc.keymaps.get("Pose")
    if km is None:
        km = kc.keymaps.new(name="Pose", space_type="EMPTY")
    kmi = km.keymap_items.new(
        BAW_OT_contact.bl_idname,
        contact_shortcut_event_type(bpy.context),
        "PRESS",
        head=True,
    )
    _CONTACT_KEYMAP_ITEMS.append((km, kmi))


def unregister_contact_keymaps() -> None:
    for km, kmi in reversed(_CONTACT_KEYMAP_ITEMS):
        if kmi.id != -1:
            km.keymap_items.remove(kmi)
    _CONTACT_KEYMAP_ITEMS.clear()
