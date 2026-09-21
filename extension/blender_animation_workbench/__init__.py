import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
)

from . import character_metadata
from .character_ops import (
    BAW_OT_assign_character_semantics,
    BAW_OT_bind_character_active,
    BAW_OT_clear_kinematic_role,
    BAW_OT_clone_character,
    BAW_OT_create_character,
    BAW_OT_create_character_chain_from_selection,
    BAW_OT_create_character_group_from_selection,
    BAW_OT_create_kinematic_mapping,
    BAW_OT_create_opposite_from_selection,
    BAW_OT_edit_character_chain,
    BAW_OT_edit_character_group,
    BAW_OT_edit_kinematic_mapping,
    BAW_OT_migrate_character_schema_v2,
    BAW_OT_move_character_chain_member,
    BAW_OT_remove_character,
    BAW_OT_remove_character_binding,
    BAW_OT_remove_character_chain,
    BAW_OT_remove_character_group,
    BAW_OT_remove_kinematic_mapping,
    BAW_OT_remove_opposite_pair,
    BAW_OT_repair_character_bone_binding,
    BAW_OT_select_character,
    BAW_OT_select_character_binding,
    BAW_OT_select_character_chain,
    BAW_OT_select_character_group,
    BAW_OT_select_character_kinematic,
    BAW_OT_set_character_chain_members_from_selection,
    BAW_OT_set_character_group_members_from_selection,
    BAW_OT_set_kinematic_active_role,
    BAW_OT_set_kinematic_chain_role,
    BAW_OT_set_kinematic_extras_from_selection,
    BAW_PG_clone_owner_map_item,
)
from .character_ui import draw_character_assignment
from .debug_trace import (
    register_debug_trace_handlers,
    unregister_debug_trace_handlers,
)
from .development_workspace import (
    register_development_workspace,
    unregister_development_workspace,
)
from .gizmo_preferences import (
    BAW_OT_gizmo_label_help,
    draw_global_gizmo_preferences,
    initialize_global_gizmo_preferences,
    register_global_gizmo_load_handler,
    register_global_gizmo_theme,
    unregister_global_gizmo_load_handler,
    unregister_global_gizmo_theme,
)
from .global_transform_gizmo import (
    BAW_GGT_global_transform,
    BAW_GT_free_rotate_disk,
    BAW_GT_uniform_scale_box,
    BAW_OT_figure_fit_move_axis,
    BAW_OT_figure_fit_rotate_axis,
    BAW_OT_figure_fit_scale_axis,
    BAW_OT_global_native_transform_axis,
)
from .phase4_contact_authoring import (
    allow_generic_trackbar_edit_for_contact,
    contact_trackbar_cells_for_context,
    expand_generic_trackbar_edit_for_contact,
    finalize_generic_trackbar_delete_for_contact,
    select_contact_trackbar_frames,
)
from .phase4_contact_ui import (
    BAW_AP_preferences,
    BAW_OT_bind_contact_object,
    BAW_OT_clear_contact_object,
    BAW_OT_contact,
    BAW_OT_replant_contact,
    BAW_OT_set_contact_point_preset,
    register_contact_keymaps,
    unregister_contact_keymaps,
)
from .rigped_animation_baseline import (
    BAW_OT_rigped_free_key,
    unregister_rigped_free_keymap,
)
from .rigped_box_wire_overlay import (
    register_rigped_box_wire_overlay,
    unregister_rigped_box_wire_overlay,
)
from .rigped_create_fit_ui import (
    BAW_OT_create_rigped_drag,
    BAW_OT_rigped_animate_mode,
    BAW_OT_rigped_box_wire_reset,
    BAW_OT_rigped_box_wire_toggle,
    BAW_OT_rigped_fit_cancel,
    BAW_OT_rigped_fit_off,
    BAW_OT_rigped_fit_on,
    BAW_OT_rigped_selection_mode,
    draw_rigped_workflow,
    register_rigped_internal_visibility_handlers,
    unregister_rigped_internal_visibility_handlers,
    update_rigped_box_wire_options,
)
from .rigped_fit_transform import (
    BAW_OT_rigped_fit_commit_transform,
    BAW_OT_rigped_fit_history_redo,
    BAW_OT_rigped_fit_history_undo,
    BAW_OT_rigped_fit_scale_axis,
    BAW_OT_rigped_fit_transform_axis,
)
from .rigped_ik_pivot_overlay import (
    register_rigped_ik_pivot_overlay,
    unregister_rigped_ik_pivot_overlay,
)
from .rigped_transform import (
    BAW_OT_rigped_direct_move_axis,
    BAW_OT_rigped_direct_rotate_axis,
    BAW_OT_rigped_fk_joint_move_axis,
    BAW_OT_rigped_semantic_move_axis,
    register_rigped_sliding_replay_handler,
    unregister_rigped_sliding_replay_handler,
)
from .trackbar_gizmo import (
    BAW_GGT_trackbar,
    BAW_GT_trackbar,
    BAW_MT_trackbar_handles,
    BAW_MT_trackbar_interpolation,
    BAW_MT_trackbar_key_context,
    BAW_OT_block_empty_trackbar_keyboard,
    BAW_OT_block_trackbar_mouse,
    BAW_OT_box_select_trackbar,
    BAW_OT_delete_trackbar_key,
    BAW_OT_fit_trackbar_keys,
    BAW_OT_pan_trackbar_drag,
    BAW_OT_resize_end_frame_drag,
    BAW_OT_trackbar_add_key,
    BAW_OT_trackbar_key_properties,
    BAW_OT_zoom_trackbar_wheel,
    register_interaction_keymaps,
    unregister_interaction_keymaps,
)
from .trackbar_keying import (
    BAW_KSI_auto_key,
    BAW_OT_set_key,
    BAW_OT_toggle_auto_key,
    disable_awb_auto_key_if_active,
    register_auto_key_handlers,
    register_keymaps,
    unregister_auto_key_handlers,
    unregister_keymaps,
)
from .trackbar_model import (
    register_trackbar_cell_provider,
    register_trackbar_delete_finalizer,
    register_trackbar_edit_expander,
    register_trackbar_edit_guard,
    register_trackbar_selection_handler,
    unregister_trackbar_cell_provider,
    unregister_trackbar_delete_finalizer,
    unregister_trackbar_edit_expander,
    unregister_trackbar_edit_guard,
    unregister_trackbar_selection_handler,
)
from .trackbar_overlay import register_draw_handler, unregister_draw_handler
from .trajectory_edit import (
    BAW_GGT_trajectory_move,
    BAW_GT_trajectory_slide,
    BAW_OT_add_trajectory_key_current_frame,
    BAW_OT_add_trajectory_key_on_path,
    BAW_OT_delete_trajectory_position_keys,
    BAW_OT_edit_trajectory_key,
    BAW_OT_move_trajectory_axis,
    BAW_OT_move_trajectory_plane,
    BAW_OT_rotate_trajectory_axis,
    BAW_OT_scale_trajectory_axis,
    BAW_OT_scale_trajectory_uniform,
    BAW_OT_toggle_trajectory_add_key,
    BAW_OT_toggle_trajectory_edit,
    BAW_OT_toggle_trajectory_tangent,
    register_trajectory_edit_keymaps,
    register_trajectory_tangent_draw_handler,
    restore_trajectory_edit_view_state,
    unregister_trajectory_edit_keymaps,
    unregister_trajectory_tangent_draw_handler,
)
from .ui_language import SUPPORTED_LANGUAGES
from .ui_language import text as ui_text
from .viewport_keymap import (
    BAW_OT_awb_select_click,
    BAW_OT_block_native_select_click,
    BAW_OT_set_transform_tool,
    BAW_OT_set_view_projection,
    BAW_OT_toggle_edged_faces,
    BAW_OT_toggle_selected_faces,
    BAW_OT_toggle_wireframe,
    register_viewport_keymaps,
    unregister_viewport_keymaps,
)
from .viewport_trajectory import (
    BAW_OT_refresh_trajectory,
    BAW_OT_toggle_trajectory_visibility,
    register_trajectory_change_handlers,
    register_trajectory_draw_handler,
    unregister_trajectory_change_handlers,
    unregister_trajectory_draw_handler,
)


def _disable_continuous_mouse_grab() -> None:
    preferences = getattr(bpy.context, "preferences", None)
    inputs = getattr(preferences, "inputs", None) if preferences is not None else None
    if inputs is not None and hasattr(inputs, "use_mouse_continuous"):
        inputs.use_mouse_continuous = False


def _contact_plant_space_items(_self, context):
    return (
        ("WORLD", ui_text("rigped.contact.world", context), "Plant at a fixed world-space point"),
        ("OBJECT", ui_text("rigped.contact.object_space", context), "Plant relative to the bound external Object"),
    )


def _get_start_frame(scene):
    return int(scene.frame_start)


def _set_start_frame(scene, value):
    scene.frame_start = min(int(value), int(scene.frame_end))
    if scene.frame_current < scene.frame_start:
        scene.frame_set(scene.frame_start)


def _get_end_frame(scene):
    return int(scene.frame_end)


def _set_end_frame(scene, value):
    scene.frame_end = max(int(scene.frame_start), int(value))
    if scene.frame_current > scene.frame_end:
        scene.frame_set(scene.frame_end)


class BAW_PT_development_panel(bpy.types.Panel):
    bl_label = "Animation Workbench"
    bl_idname = "BAW_PT_development_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Animation WB"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        language_row = layout.row(align=True)
        language_row.prop(scene, "baw_ui_language", text=ui_text("ui.language", context))

        top = layout.row(align=True)
        top.prop(scene, "baw_trackbar_enabled", text=ui_text("panel.track_bar", context), toggle=True)
        top.prop(scene, "baw_start_frame", text=ui_text("panel.start", context))
        top.prop(scene, "baw_end_frame", text=ui_text("panel.end", context))

        key_header, key_body = layout.panel("BAW_UI_keying", default_closed=False)
        key_header.label(text=ui_text("panel.keying", context))
        if key_body is not None:
            channels = key_body.row(align=True)
            channels.prop(scene, "baw_key_position", text=ui_text("panel.position_short", context), toggle=True)
            channels.prop(scene, "baw_key_rotation", text=ui_text("panel.rotation_short", context), toggle=True)
            channels.prop(scene, "baw_key_scale", text=ui_text("panel.scale", context), toggle=True)

            actions = key_body.row(align=True)
            actions.operator("baw.set_key", text=ui_text("panel.set_key", context))
            actions.prop(scene, "baw_key_mode", text=ui_text("panel.key_mode", context), toggle=True)
            actions.operator(
                "baw.toggle_auto_key",
                text=ui_text("panel.auto", context),
                depress=scene.baw_auto_key_enabled,
            )

        traj_header, traj_body = layout.panel("BAW_UI_trajectory", default_closed=True)
        traj_header.label(text=ui_text("panel.trajectory", context))
        traj_header.operator(
            "baw.toggle_trajectory_visibility",
            text=ui_text("ui.on", context) if scene.baw_trajectory_visible else ui_text("ui.off", context),
            depress=scene.baw_trajectory_visible,
        )
        if traj_body is not None:
            row = traj_body.row(align=True)
            row.enabled = bool(scene.baw_trajectory_visible)
            row.prop(scene, "baw_trajectory_live_preview", text=ui_text("panel.live", context), toggle=True)
            row.operator("baw.refresh_trajectory", text=ui_text("panel.refresh", context))

            edit = traj_body.row(align=True)
            edit.enabled = bool(scene.baw_trajectory_visible)
            edit.operator(
                "baw.toggle_trajectory_edit",
                text=ui_text("panel.edit", context),
                depress=scene.baw_trajectory_edit_mode,
            )
            tangent = edit.row(align=True)
            tangent.enabled = bool(scene.baw_trajectory_edit_mode)
            tangent.operator(
                "baw.toggle_trajectory_tangent",
                text=ui_text("panel.tangents", context),
                depress=scene.baw_trajectory_transform_mode == "TANGENT",
            )
            add_key = edit.row(align=True)
            add_key.enabled = bool(scene.baw_trajectory_edit_mode)
            add_key.operator(
                "baw.toggle_trajectory_add_key",
                text=ui_text("panel.add_key", context),
                depress=scene.baw_trajectory_add_key_mode,
            )

        layout.separator()
        addon_entry = context.preferences.addons.get(__package__)
        addon_preferences = getattr(addon_entry, "preferences", None) if addon_entry is not None else None
        draw_global_gizmo_preferences(layout, context, addon_preferences)

        layout.separator()
        draw_rigped_workflow(layout, context)
        # Animate reset baseline: Contact/IK UI is intentionally hidden until
        # plain FK keying and Track Bar behavior are rebuilt and user-accepted.
        layout.separator()
        draw_character_assignment(layout, context)


_CLASSES = (
    BAW_OT_gizmo_label_help,
    BAW_AP_preferences,
    BAW_KSI_auto_key,
    BAW_PG_clone_owner_map_item,
    BAW_OT_create_character,
    BAW_OT_clone_character,
    BAW_OT_remove_character,
    BAW_OT_bind_character_active,
    BAW_OT_assign_character_semantics,
    BAW_OT_repair_character_bone_binding,
    BAW_OT_remove_character_binding,
    BAW_OT_create_character_group_from_selection,
    BAW_OT_edit_character_group,
    BAW_OT_set_character_group_members_from_selection,
    BAW_OT_remove_character_group,
    BAW_OT_create_character_chain_from_selection,
    BAW_OT_edit_character_chain,
    BAW_OT_set_character_chain_members_from_selection,
    BAW_OT_move_character_chain_member,
    BAW_OT_remove_character_chain,
    BAW_OT_create_opposite_from_selection,
    BAW_OT_remove_opposite_pair,
    BAW_OT_create_kinematic_mapping,
    BAW_OT_edit_kinematic_mapping,
    BAW_OT_set_kinematic_chain_role,
    BAW_OT_set_kinematic_active_role,
    BAW_OT_set_kinematic_extras_from_selection,
    BAW_OT_clear_kinematic_role,
    BAW_OT_remove_kinematic_mapping,
    BAW_OT_migrate_character_schema_v2,
    BAW_OT_select_character,
    BAW_OT_select_character_binding,
    BAW_OT_select_character_group,
    BAW_OT_select_character_chain,
    BAW_OT_select_character_kinematic,
    BAW_OT_create_rigped_drag,
    BAW_OT_rigped_selection_mode,
    BAW_OT_rigped_box_wire_toggle,
    BAW_OT_rigped_box_wire_reset,
    BAW_OT_rigped_fit_on,
    BAW_OT_rigped_fit_off,
    BAW_OT_rigped_fit_cancel,
    BAW_OT_rigped_animate_mode,
    BAW_OT_rigped_free_key,
    BAW_OT_toggle_auto_key,
    BAW_OT_set_key,
    BAW_OT_bind_contact_object,
    BAW_OT_clear_contact_object,
    BAW_OT_set_contact_point_preset,
    BAW_OT_replant_contact,
    BAW_OT_contact,
    BAW_OT_rigped_direct_move_axis,
    BAW_OT_rigped_direct_rotate_axis,
    BAW_OT_rigped_fk_joint_move_axis,
    BAW_OT_rigped_semantic_move_axis,
    BAW_OT_rigped_fit_commit_transform,
    BAW_OT_rigped_fit_history_undo,
    BAW_OT_rigped_fit_history_redo,
    BAW_OT_rigped_fit_scale_axis,
    BAW_OT_rigped_fit_transform_axis,
    BAW_OT_awb_select_click,
    BAW_OT_block_native_select_click,
    BAW_OT_set_transform_tool,
    BAW_OT_set_view_projection,
    BAW_OT_toggle_wireframe,
    BAW_OT_toggle_edged_faces,
    BAW_OT_toggle_selected_faces,
    BAW_MT_trackbar_key_context,
    BAW_MT_trackbar_interpolation,
    BAW_MT_trackbar_handles,
    BAW_OT_block_empty_trackbar_keyboard,
    BAW_OT_block_trackbar_mouse,
    BAW_OT_box_select_trackbar,
    BAW_OT_trackbar_add_key,
    BAW_OT_delete_trackbar_key,
    BAW_OT_fit_trackbar_keys,
    BAW_OT_pan_trackbar_drag,
    BAW_OT_resize_end_frame_drag,
    BAW_OT_trackbar_key_properties,
    BAW_OT_zoom_trackbar_wheel,
    BAW_OT_toggle_trajectory_visibility,
    BAW_OT_refresh_trajectory,
    BAW_OT_toggle_trajectory_edit,
    BAW_OT_toggle_trajectory_tangent,
    BAW_OT_toggle_trajectory_add_key,
    BAW_OT_add_trajectory_key_on_path,
    BAW_OT_add_trajectory_key_current_frame,
    BAW_OT_delete_trajectory_position_keys,
    BAW_OT_edit_trajectory_key,
    BAW_OT_move_trajectory_axis,
    BAW_OT_move_trajectory_plane,
    BAW_OT_rotate_trajectory_axis,
    BAW_OT_scale_trajectory_axis,
    BAW_OT_scale_trajectory_uniform,
    BAW_GT_trajectory_slide,
    BAW_GGT_trajectory_move,
    BAW_OT_figure_fit_move_axis,
    BAW_OT_figure_fit_rotate_axis,
    BAW_OT_figure_fit_scale_axis,
    BAW_OT_global_native_transform_axis,
    BAW_GT_free_rotate_disk,
    BAW_GT_uniform_scale_box,
    BAW_GGT_global_transform,
    BAW_GT_trackbar,
    BAW_GGT_trackbar,
    BAW_PT_development_panel,
)


def register():
    _disable_continuous_mouse_grab()
    bpy.types.Scene.baw_ui_language = EnumProperty(
        name="Language",
        description="Choose the Animation Workbench interface language",
        items=SUPPORTED_LANGUAGES,
        default="EN",
    )
    bpy.types.Scene.baw_trackbar_enabled = BoolProperty(
        name="Show Animation Workbench Track Bar",
        description="Show the Animation Workbench Track Bar in the 3D Viewport",
        default=True,
    )
    bpy.types.Scene.baw_trackbar_view_center = FloatProperty(
        name="Track Bar View Center",
        description="Center frame of the independently panned Track Bar view",
        default=0.0,
    )
    bpy.types.Scene.baw_trackbar_view_span = IntProperty(
        name="Track Bar View Span",
        description="Number of frames currently visible across the Track Bar",
        default=0,
        min=0,
    )
    bpy.types.Scene.baw_key_position = BoolProperty(
        name="Position",
        description="Set position keys when using Animation Workbench Set Key",
        default=True,
    )
    bpy.types.Scene.baw_key_rotation = BoolProperty(
        name="Rotation",
        description="Set rotation keys when using Animation Workbench Set Key",
        default=True,
    )
    bpy.types.Scene.baw_key_scale = BoolProperty(
        name="Scale",
        description="Set scale keys when using Animation Workbench Set Key",
        default=True,
    )
    bpy.types.Scene.baw_key_mode = BoolProperty(
        name="AWB Key Mode",
        description="Make Track Bar previous/next controls jump between semantic keys",
        default=False,
    )
    bpy.types.Scene.baw_auto_key_enabled = BoolProperty(
        name="AWB Auto Key",
        description="Whether Animation Workbench owns Blender Auto Key for transform recording",
        default=False,
    )
    bpy.types.Scene.baw_trajectory_visible = BoolProperty(
        name="AWB Trajectory Visible",
        description="Show AWB Motion Paths for the current animation control selection",
        default=True,
    )
    bpy.types.Scene.baw_trajectory_live_preview = BoolProperty(
        name="AWB Trajectory Live Preview",
        description=(
            "Preview the affected trajectory arc while a native transform gizmo is dragged; "
            "when disabled, the path refreshes after the transform is released"
        ),
        default=True,
    )
    bpy.types.Scene.baw_trajectory_edit_mode = BoolProperty(
        name="AWB Trajectory Edit Mode",
        description="Allow direct spatial editing of selected Object position keys on Motion Paths",
        default=False,
    )
    bpy.types.Scene.baw_contact_plant_space = EnumProperty(
        name="Contact Plant Space",
        description="Space used the next time Contact enters Planted",
        items=_contact_plant_space_items,
    )
    bpy.types.Scene.baw_contact_point_local = FloatVectorProperty(
        name="Contact Point Local",
        description="Generalized terminal-local Contact point used only for a new World Planted episode or explicit Replant",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="XYZ",
    )
    bpy.types.Scene.baw_contact_object_target = PointerProperty(
        name="Contact Object",
        description="External rigid Object to bind to the selected Rigped hand/foot Contact domain",
        type=bpy.types.Object,
    )
    bpy.types.Scene.baw_rigped_body_picker_enabled = BoolProperty(
        name="Rigped Body Picker",
        description="Show the Rigped Body Picker while Animate mode is active",
        default=False,
        options={"HIDDEN", "SKIP_SAVE"},
    )
    bpy.types.Scene.baw_rigped_box_wire_options_expanded = BoolProperty(
        name="Biped Box Wire Options",
        description="Show Biped Box Wire display options",
        default=False,
    )
    bpy.types.Scene.baw_rigped_box_wire_width = FloatProperty(
        name="Wire Width",
        description="Line width for Biped Box Wire custom bone shapes",
        default=1.5,
        min=1.0,
        max=16.0,
        update=update_rigped_box_wire_options,
    )
    bpy.types.Scene.baw_rigped_box_wire_left_color = FloatVectorProperty(
        name="Left",
        description="Classic Biped-style color for left-side controls",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(28.0 / 255.0, 28.0 / 255.0, 177.0 / 255.0),
        update=update_rigped_box_wire_options,
    )
    bpy.types.Scene.baw_rigped_box_wire_right_color = FloatVectorProperty(
        name="Right",
        description="Classic Biped-style color for right-side controls",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(6.0 / 255.0, 134.0 / 255.0, 6.0 / 255.0),
        update=update_rigped_box_wire_options,
    )
    bpy.types.Scene.baw_rigped_box_wire_center_color = FloatVectorProperty(
        name="Center",
        description="Classic Biped-style color for Root, COM, Spine and Neck",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(8.0 / 255.0, 110.0 / 255.0, 134.0 / 255.0),
        update=update_rigped_box_wire_options,
    )
    bpy.types.Scene.baw_rigped_box_wire_pelvis_color = FloatVectorProperty(
        name="Pelvis",
        description="Classic Biped-style pelvis color",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(224.0 / 255.0, 198.0 / 255.0, 87.0 / 255.0),
        update=update_rigped_box_wire_options,
    )
    bpy.types.Scene.baw_rigped_box_wire_head_color = FloatVectorProperty(
        name="Head",
        description="Classic Biped-style head color",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(166.0 / 255.0, 202.0 / 255.0, 240.0 / 255.0),
        update=update_rigped_box_wire_options,
    )
    bpy.types.Scene.baw_rigped_box_wire_com_color = FloatVectorProperty(
        name="COM",
        description="Rigped COM display color used in Select, Fit, and Animate modes",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(0.92, 0.72, 0.18),
        update=update_rigped_box_wire_options,
    )
    bpy.types.Scene.baw_rigped_box_wire_selected_color = FloatVectorProperty(
        name="Selected",
        description="Color used by selected Rigped controls in every Rigped mode",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(1.0, 0.72, 0.18),
        update=update_rigped_box_wire_options,
    )
    bpy.types.Scene.baw_rigped_box_wire_active_color = FloatVectorProperty(
        name="Active",
        description="Color used by the active Rigped control in every Rigped mode",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(1.0, 0.92, 0.35),
        update=update_rigped_box_wire_options,
    )
    bpy.types.Scene.baw_rigped_semantic_transform_mode = EnumProperty(
        name="AWB Rigped Semantic Transform Mode",
        description="Transient animator tool state for AWB-owned Rigped transforms",
        items=(
            ("NONE", "None", "No AWB-owned Rigped transform tool"),
            ("MOVE", "Move", "Use the AWB-owned semantic Rigped Move gizmo"),
            ("FK_MOVE", "FK Move", "Use the Biped-style Free/FK limb Move gizmo without raw Location"),
            ("DIRECT_MOVE", "Direct Move", "Use the anchored AWB Move shell with Blender-native transform authority"),
            ("DIRECT_ROTATE", "Direct Rotate", "Use the anchored AWB Rotate shell with Blender-native FK transform authority"),
        ),
        default="NONE",
        options={"HIDDEN", "SKIP_SAVE"},
    )
    bpy.types.Scene.baw_trajectory_transform_mode = EnumProperty(
        name="AWB Trajectory Transform Mode",
        description="Trajectory editing gizmo used for selected Position keys",
        items=(
            ("MOVE", "Slide", "Slide trajectory Position keys along their existing path"),
            ("ROTATE", "Rotate", "Reserved spatial Rotate mode"),
            ("SCALE", "Retime", "Scale selected Position-key timing around the time center"),
            ("TANGENT", "Tangent", "Edit selected Position-key Bezier tangents in the viewport"),
        ),
        default="MOVE",
    )
    bpy.types.Scene.baw_trajectory_add_key_mode = BoolProperty(
        name="AWB Trajectory Add Key Mode",
        description="Keep trajectory Add Key active for repeated Position-key insertion clicks",
        default=False,
    )
    bpy.types.Scene.baw_auto_key_prev_valid = BoolProperty(
        name="AWB Auto Key Previous Settings Valid",
        default=False,
        options={"HIDDEN"},
    )
    bpy.types.Scene.baw_auto_key_prev_use_keying_set = BoolProperty(
        name="AWB Previous Auto Key Keying Set Restriction",
        default=False,
        options={"HIDDEN"},
    )
    bpy.types.Scene.baw_auto_key_prev_replace_only = BoolProperty(
        name="AWB Previous Auto Key Replace Only",
        default=False,
        options={"HIDDEN"},
    )
    bpy.types.Scene.baw_auto_key_prev_keying_set_index = IntProperty(
        name="AWB Previous Active Keying Set Index",
        default=0,
        options={"HIDDEN"},
    )
    bpy.types.Scene.baw_auto_key_prev_insert_channels_mask = IntProperty(
        name="AWB Previous Default Key Channels",
        default=0,
        options={"HIDDEN"},
    )
    bpy.types.Scene.baw_auto_key_prev_insert_available = BoolProperty(
        name="AWB Previous Auto Key Only Insert Available",
        default=False,
        options={"HIDDEN"},
    )
    bpy.types.Scene.baw_auto_key_prev_insert_needed = BoolProperty(
        name="AWB Previous Auto Key Only Insert Needed",
        default=True,
        options={"HIDDEN"},
    )
    bpy.types.Scene.baw_start_frame = IntProperty(
        name="Start Frame",
        description="Set the first frame shown across the full Animation Workbench Track Bar",
        get=_get_start_frame,
        set=_set_start_frame,
    )
    bpy.types.Scene.baw_end_frame = IntProperty(
        name="End Frame",
        description="Set the animation end frame and keep it visible in the Track Bar",
        get=_get_end_frame,
        set=_set_end_frame,
    )
    bpy.types.Scene.baw_has_selected_key = BoolProperty(
        name="Track Bar Has Selected Key",
        default=False,
    )
    bpy.types.Scene.baw_selected_key_frame = IntProperty(
        name="Track Bar Selected Key Frame",
        default=0,
    )
    character_metadata.register()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    initialize_global_gizmo_preferences(bpy.context)
    register_global_gizmo_load_handler()
    register_global_gizmo_theme(bpy.context)
    bpy.types.WindowManager.baw_clone_owner_map = CollectionProperty(
        type=BAW_PG_clone_owner_map_item,
        options={"HIDDEN"},
    )
    bpy.types.WindowManager.baw_rigped_semantic_move_drag_active = BoolProperty(
        name="AWB Rigped Semantic Move Drag Active",
        description="Transient guard used to keep animation overlays out of the semantic Move hot path",
        default=False,
        options={"HIDDEN", "SKIP_SAVE"},
    )
    bpy.types.WindowManager.baw_trackbar_frame_drag_active = BoolProperty(
        name="AWB Track Bar Frame Drag Active",
        description="Transient guard distinguishing time scrubbing from transform gizmo drags",
        default=False,
        options={"HIDDEN", "SKIP_SAVE"},
    )
    register_keymaps()
    register_auto_key_handlers()
    register_viewport_keymaps()
    # Rigped Contact cells are aggregate semantic bundles. Generic Track Bar
    # Move/Clone/Delete must expand to every hidden Contact/IK companion curve.
    register_trackbar_edit_guard(allow_generic_trackbar_edit_for_contact)
    register_trackbar_edit_expander(expand_generic_trackbar_edit_for_contact)
    register_trackbar_delete_finalizer(finalize_generic_trackbar_delete_for_contact)
    register_trackbar_cell_provider(contact_trackbar_cells_for_context)
    register_trackbar_selection_handler(select_contact_trackbar_frames)
    # Rigped C is owned by the production Contact command. The legacy
    # rigped_free_key operator remains available only for explicit/internal use;
    # it must not own the C shortcut because broad selections would also author
    # unrelated residual Free keys.
    register_contact_keymaps()
    register_interaction_keymaps()
    register_trajectory_edit_keymaps()
    register_draw_handler()
    register_trajectory_draw_handler()
    register_trajectory_tangent_draw_handler()
    register_trajectory_change_handlers()
    register_rigped_internal_visibility_handlers()
    # Sliding has one replay authority: hidden IK target/pole + Contact state.
    # Mirror that evaluated result onto public controls after each frame change.
    register_rigped_sliding_replay_handler()
    # Precision trace must observe the final post-replay public pose, not the
    # transient pre-sync FCurve evaluation. Register it after replay handlers.
    register_debug_trace_handlers()
    # A5 Sliding reintroduces the Biped-style visible red IK/contact pivot cue.
    # The overlay remains state-driven: Free hides it, Sliding shows it.
    register_rigped_ik_pivot_overlay()
    register_rigped_box_wire_overlay()
    register_development_workspace()


def unregister():
    disable_awb_auto_key_if_active(bpy.context)
    unregister_debug_trace_handlers()
    unregister_global_gizmo_load_handler()
    unregister_global_gizmo_theme(bpy.context)
    unregister_rigped_box_wire_overlay()
    unregister_rigped_ik_pivot_overlay()
    unregister_rigped_sliding_replay_handler()
    unregister_rigped_internal_visibility_handlers()
    unregister_auto_key_handlers()
    restore_trajectory_edit_view_state()
    unregister_development_workspace()
    unregister_trajectory_change_handlers()
    unregister_trajectory_tangent_draw_handler()
    unregister_trajectory_draw_handler()
    unregister_draw_handler()
    unregister_trajectory_edit_keymaps()
    unregister_trackbar_selection_handler(select_contact_trackbar_frames)
    unregister_trackbar_cell_provider(contact_trackbar_cells_for_context)
    unregister_trackbar_delete_finalizer(finalize_generic_trackbar_delete_for_contact)
    unregister_trackbar_edit_expander(expand_generic_trackbar_edit_for_contact)
    unregister_trackbar_edit_guard(allow_generic_trackbar_edit_for_contact)
    unregister_interaction_keymaps()
    unregister_rigped_free_keymap()
    unregister_contact_keymaps()
    unregister_viewport_keymaps()
    unregister_keymaps()
    if hasattr(bpy.types.WindowManager, "baw_trackbar_frame_drag_active"):
        del bpy.types.WindowManager.baw_trackbar_frame_drag_active
    if hasattr(bpy.types.WindowManager, "baw_rigped_semantic_move_drag_active"):
        del bpy.types.WindowManager.baw_rigped_semantic_move_drag_active
    if hasattr(bpy.types.WindowManager, "baw_clone_owner_map"):
        del bpy.types.WindowManager.baw_clone_owner_map
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    character_metadata.unregister()
    for name in (
        "baw_selected_key_frame",
        "baw_has_selected_key",
        "baw_end_frame",
        "baw_start_frame",
        "baw_auto_key_prev_insert_needed",
        "baw_auto_key_prev_insert_available",
        "baw_auto_key_prev_insert_channels_mask",
        "baw_auto_key_prev_keying_set_index",
        "baw_auto_key_prev_replace_only",
        "baw_auto_key_prev_use_keying_set",
        "baw_auto_key_prev_valid",
        "baw_auto_key_enabled",
        "baw_trajectory_visible",
        "baw_trajectory_live_preview",
        "baw_trajectory_add_key_mode",
        "baw_trajectory_transform_mode",
        "baw_contact_object_target",
        "baw_contact_point_local",
        "baw_contact_plant_space",
        "baw_rigped_body_picker_enabled",
        "baw_rigped_box_wire_active_color",
        "baw_rigped_box_wire_selected_color",
        "baw_rigped_box_wire_com_color",
        "baw_rigped_box_wire_head_color",
        "baw_rigped_box_wire_pelvis_color",
        "baw_rigped_box_wire_center_color",
        "baw_rigped_box_wire_right_color",
        "baw_rigped_box_wire_left_color",
        "baw_rigped_box_wire_width",
        "baw_rigped_box_wire_options_expanded",
        "baw_rigped_semantic_transform_mode",
        "baw_trajectory_edit_mode",
        "baw_key_mode",
        "baw_key_scale",
        "baw_key_rotation",
        "baw_key_position",
        "baw_trackbar_view_span",
        "baw_trackbar_view_center",
        "baw_trackbar_enabled",
        "baw_ui_language",
    ):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
