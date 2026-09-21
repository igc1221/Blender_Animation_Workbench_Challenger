from __future__ import annotations

from math import cos, exp, pi, sin
from typing import ClassVar

import bmesh
import bpy
from bpy.props import EnumProperty, FloatProperty
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Matrix, Quaternion, Vector

from .debug_trace import trace_event
from .gizmo_preferences import (
    GIZMO_AXIS_COLORS,
    GIZMO_HIGHLIGHT_COLOR,
    MAX_LINEAR_ROTATION_RADIANS_PER_PIXEL,
    addon_gizmo_preferences,
    arcball_world_step,
    clear_rotation_angle,
    gizmo_zoom_scale,
    linear_roll_screen_tangent,
    projected_axis_screen_direction,
    projection_world_per_pixel,
    show_rotation_angle,
    snapped_rotation_angle,
)
from .rigped_create_fit_ui import (
    fit_orientation_mode,
    fit_transform_mode,
    fit_ui_state,
    fit_ui_state_present,
)
from .rigped_fit_session import (
    FitSemanticSessionError,
    apply_fit_move_preview,
    begin_fit_move_gesture,
    cancel_fit_move_gesture,
    commit_fit_move_gesture,
    fit_active_part_world_pivot_axes,
    fit_figure_move_available,
    fit_semantic_session,
)
from .rigped_fit_transform import (
    BAW_OT_rigped_fit_scale_axis,
    BAW_OT_rigped_fit_transform_axis,
    _active_edit_bone,
    _fit_orientation_axes,
    _fit_pivot_local,
    _plane_point,
    _rotation_vector,
)
from .rigped_transform import (
    BAW_OT_rigped_direct_move_axis,
    BAW_OT_rigped_direct_rotate_axis,
    BAW_OT_rigped_fk_joint_move_axis,
    BAW_OT_rigped_semantic_move_axis,
    _control_pivot_world,
    _fk_joint_move_active_control,
    _orientation_axes_for_control,
    direct_move_available,
    direct_rotate_available,
    direct_transform_axes,
    direct_transform_pivot,
    fk_joint_move_available,
    semantic_move_available,
    semantic_move_axes,
    semantic_move_pivot,
)


def _active_tool_id(context) -> str:
    workspace = getattr(context, "workspace", None)
    tools = getattr(workspace, "tools", None) if workspace is not None else None
    if tools is None:
        return ""
    try:
        tool = tools.from_space_view3d_mode(context.mode, create=False)
    except (RuntimeError, TypeError):
        return ""
    return str(getattr(tool, "idname", "") or "")


def _force_figure_safe_workspace_tool(context) -> None:
    if not fit_ui_state_present(context) or getattr(context, "mode", "") != "OBJECT":
        return
    active_tool = _active_tool_id(context)
    if active_tool not in {"builtin.move", "builtin.rotate", "builtin.scale"}:
        return
    safe_tool = "baw.move_object" if fit_transform_mode(context) == "MOVE" else "baw.select_object"
    try:
        bpy.ops.wm.tool_set_by_id(name=safe_tool)
    except RuntimeError:
        return
    if hasattr(context.space_data, "show_gizmo_tool"):
        context.space_data.show_gizmo_tool = False


def _native_mode(context) -> str:
    return {
        "builtin.move": "MOVE",
        "builtin.rotate": "ROTATE",
        "builtin.scale": "SCALE",
        "baw.move_object": "MOVE",
        "baw.rotate_object": "ROTATE",
        "baw.scale_object": "SCALE",
        "baw.move_pose": "MOVE",
        "baw.rotate_pose": "ROTATE",
        "baw.scale_pose": "SCALE",
    }.get(_active_tool_id(context), "")


def _route_and_mode(context) -> tuple[str, str]:
    if bool(getattr(getattr(context, "scene", None), "baw_trajectory_edit_mode", False)):
        return "", ""

    if fit_ui_state_present(context) and getattr(context, "mode", "") == "OBJECT":
        # A raw Figure/Fit host suppresses native Object transforms even if the
        # semantic session became stale/missing. Only a fully valid Figure
        # state may expose the migrated semantic Move route.
        state = fit_ui_state(context)
        if state is None:
            return "", ""
        mode = str(fit_transform_mode(context) or "")
        if mode == "MOVE" and fit_figure_move_available(context):
            return "FIGURE", "MOVE"
        return "", ""

    # Fit deliberately keeps the AWB Select workspace tool active and owns its
    # transform mode internally, so resolve Fit before the Select-tool guard.
    if getattr(context, "mode", "") == "EDIT_ARMATURE" and fit_ui_state(context) is not None:
        return "FIT", str(fit_transform_mode(context) or "")

    active_tool = _active_tool_id(context)
    if active_tool in {
        "baw.select_object",
        "baw.select_pose",
        "baw.select_edit_armature",
    }:
        # Explicit Select owns plain LMB click/drag completely. Any stale
        # semantic transform state is ignored until W/E/R activates an AWB
        # transform workspace tool again.
        return "", ""

    state = str(
        getattr(
            getattr(context, "scene", None),
            "baw_rigped_semantic_transform_mode",
            "NONE",
        )
    )
    if getattr(context, "mode", "") == "POSE":
        if state == "DIRECT_ROTATE" and direct_rotate_available(context):
            return "DIRECT_ROTATE", "ROTATE"
        if state in {"DIRECT_MOVE", "FK_MOVE", "MOVE"}:
            # W means "Move" as a persistent tool intent. Re-resolve the
            # semantic route from the current selection on every gizmo refresh
            # so switching from a Sliding terminal to COM/Pelvis does not fall
            # through to Blender's native transform path with stale state.
            if direct_move_available(context):
                return "DIRECT_MOVE", "MOVE"
            if fk_joint_move_available(context):
                return "FK_MOVE", "MOVE"
            if semantic_move_available(context):
                return "SEMANTIC_MOVE", "MOVE"

    mode = _native_mode(context)
    if mode and getattr(context, "active_object", None) is not None:
        # Native transform tools exist in many View3D contexts (Object, Pose,
        # Mesh/Curve/Surface/Lattice/Metaball/Armature edit, paint/sculpt, etc.).
        # The AWB shell is deliberately mode-agnostic here: Blender's native
        # transform operator remains the authority while one visual gizmo
        # replaces the built-in tool gizmo everywhere it is available.
        return "NATIVE", mode
    return "", ""


def _view_axes(context) -> dict[str, Vector] | None:
    region_data = getattr(context, "region_data", None)
    if region_data is None:
        region_data = getattr(getattr(context, "space_data", None), "region_3d", None)
    if region_data is None:
        return None
    basis = region_data.view_matrix.inverted().to_3x3()
    return {
        "X": Vector(basis.col[0]).normalized(),
        "Y": Vector(basis.col[1]).normalized(),
        "Z": Vector(basis.col[2]).normalized(),
    }


def _edit_selection_world_points(context) -> tuple[Vector, ...]:
    points: list[Vector] = []
    objects = tuple(getattr(context, "objects_in_mode_unique_data", ()) or ())
    if not objects:
        active = getattr(context, "active_object", None)
        objects = (active,) if active is not None else ()

    for obj in objects:
        if obj is None:
            continue
        matrix = obj.matrix_world
        if obj.type == "MESH":
            try:
                mesh = bmesh.from_edit_mesh(obj.data)
            except (RuntimeError, TypeError, ValueError):
                continue
            points.extend(Vector(matrix @ vert.co) for vert in mesh.verts if vert.select)
        elif obj.type in {"CURVE", "SURFACE"}:
            for spline in obj.data.splines:
                for point in getattr(spline, "bezier_points", ()):
                    if bool(point.select_control_point or point.select_left_handle or point.select_right_handle):
                        points.append(Vector(matrix @ point.co))
                for point in getattr(spline, "points", ()):
                    if bool(point.select):
                        points.append(Vector(matrix @ Vector(point.co[:3])))
        elif obj.type == "LATTICE":
            points.extend(Vector(matrix @ point.co_deform) for point in obj.data.points if point.select)
        elif obj.type == "META":
            points.extend(Vector(matrix @ element.co) for element in obj.data.elements if element.select)
        elif obj.type == "ARMATURE":
            for bone in obj.data.edit_bones:
                if bone.select_head:
                    points.append(Vector(matrix @ bone.head))
                if bone.select_tail:
                    points.append(Vector(matrix @ bone.tail))
    return tuple(points)


def _native_selection_world_points(context) -> tuple[Vector, ...]:
    mode = getattr(context, "mode", "")
    if mode.startswith("EDIT_"):
        return _edit_selection_world_points(context)
    if mode == "POSE":
        obj = getattr(context, "active_object", None)
        bones = tuple(getattr(context, "selected_pose_bones", ()) or ())
        if obj is None:
            return ()
        return tuple(Vector(obj.matrix_world @ bone.matrix.translation) for bone in bones)
    objects = tuple(getattr(context, "selected_objects", ()) or ())
    return tuple(Vector(obj.matrix_world.translation) for obj in objects)


def _selection_pivot(context, fallback: Vector) -> Vector:
    tool_settings = getattr(getattr(context, "scene", None), "tool_settings", None)
    pivot_mode = str(getattr(tool_settings, "transform_pivot_point", "MEDIAN_POINT"))
    if pivot_mode == "CURSOR":
        return Vector(context.scene.cursor.location)

    points = _native_selection_world_points(context)
    if not points:
        return fallback
    if pivot_mode == "BOUNDING_BOX_CENTER":
        minimum = Vector(min(point[i] for point in points) for i in range(3))
        maximum = Vector(max(point[i] for point in points) for i in range(3))
        return (minimum + maximum) * 0.5
    if pivot_mode == "ACTIVE_ELEMENT":
        return fallback
    # MEDIAN_POINT and INDIVIDUAL_ORIGINS both use the shared shell center.
    return sum(points, Vector((0.0, 0.0, 0.0))) / len(points)


def _orientation_axes(context, local_basis: Matrix) -> dict[str, Vector] | None:
    slot = context.scene.transform_orientation_slots[0]
    orientation = str(slot.type)
    if orientation == "VIEW":
        return _view_axes(context)
    if orientation == "CURSOR":
        basis = context.scene.cursor.matrix.to_3x3()
    elif orientation == "CUSTOM" and getattr(slot, "custom_orientation", None) is not None:
        basis = slot.custom_orientation.matrix.to_3x3()
    elif orientation in {"LOCAL", "NORMAL", "GIMBAL"}:
        basis = local_basis
    else:
        basis = Matrix.Identity(3)
    return {
        "X": Vector(basis.col[0]).normalized(),
        "Y": Vector(basis.col[1]).normalized(),
        "Z": Vector(basis.col[2]).normalized(),
    }


def _native_pivot_axes(context) -> tuple[Vector | None, dict[str, Vector] | None]:
    obj = getattr(context, "active_object", None)
    if obj is None:
        return None, None

    mode = getattr(context, "mode", "")
    # One global visibility rule: a transform gizmo only exists while there is
    # an actual transform selection. Do not fall back to a stale active object,
    # bone, or edit element after empty-space deselection.
    if not _native_selection_world_points(context):
        return None, None

    if mode == "POSE":
        bone = getattr(context, "active_pose_bone", None)
        if bone is None:
            return None, None
        fallback = Vector(obj.matrix_world @ bone.matrix.translation)
        local_basis = (obj.matrix_world @ bone.matrix).to_3x3()
    elif mode == "EDIT_ARMATURE":
        bone = _active_edit_bone(context)
        if bone is None:
            return None, None
        fallback = Vector(obj.matrix_world @ bone.head)
        local_basis = (obj.matrix_world @ bone.matrix).to_3x3()
    else:
        fallback = Vector(obj.matrix_world.translation)
        local_basis = obj.matrix_world.to_3x3()

    pivot = _selection_pivot(context, fallback)
    return pivot, _orientation_axes(context, local_basis)


def _pivot_axes(context, route: str) -> tuple[Vector | None, dict[str, Vector] | None]:
    if route == "FIGURE":
        return fit_active_part_world_pivot_axes(
            context,
            orientation_mode=fit_orientation_mode(context),
        )
    if route == "FIT":
        rig = getattr(context, "active_object", None)
        bone = _active_edit_bone(context)
        axes = _fit_orientation_axes(context)
        if rig is None or bone is None or axes is None:
            return None, None
        return Vector(rig.matrix_world @ _fit_pivot_local(bone)), axes

    # Pose mode can retain an active pose bone after clicking empty viewport
    # space, even though the actual selection has been cleared. Never let that
    # stale active element keep the shared transform gizmo alive: Animate and
    # ordinary Pose mode should follow Fit's selection contract exactly.
    if getattr(context, "mode", "") == "POSE":
        selected_pose_bones = tuple(getattr(context, "selected_pose_bones", ()) or ())
        if not selected_pose_bones:
            return None, None

    if route in {"DIRECT_ROTATE", "NATIVE"} and getattr(context, "mode", "") == "POSE":
        pivot = direct_transform_pivot(context)
        axes = direct_transform_axes(context)
        if pivot is not None and axes is not None:
            return pivot, axes
    if route == "SEMANTIC_MOVE":
        return semantic_move_pivot(context), semantic_move_axes(context)
    if route == "FK_MOVE":
        active = _fk_joint_move_active_control(context)
        if active is None:
            return None, None
        return _control_pivot_world(active), _orientation_axes_for_control(context, active)
    return _native_pivot_axes(context)


def _axis_matrix(axis: Vector, pivot: Vector) -> Matrix:
    rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(axis)
    matrix = rotation.to_matrix().to_4x4()
    matrix.translation = pivot
    return matrix


def _new_invisible_hit(group, gizmo_type: str, operator: str):
    gizmo = group.gizmos.new(gizmo_type)
    props = gizmo.target_set_operator(operator)
    gizmo.alpha = 0.001
    gizmo.alpha_highlight = 0.001
    gizmo.select_bias = 2.0
    gizmo.use_draw_modal = False
    return gizmo, props


class BAW_OT_global_native_transform_axis(bpy.types.Operator):
    """Guide-free modal wrapper around Blender's native transform execution."""

    bl_idname = "baw.global_native_transform_axis"
    bl_label = "AWB Global Native Transform"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "BLOCKING"}

    mode: EnumProperty(
        items=(("MOVE", "Move", "Move selection"), ("ROTATE", "Rotate", "Rotate selection"), ("SCALE", "Scale", "Scale selection")),
        default="MOVE",
    )
    axis: EnumProperty(
        items=(
            ("X", "X", "Use X axis"),
            ("Y", "Y", "Use Y axis"),
            ("Z", "Z", "Use Z axis"),
            ("XY", "XY", "Use XY plane"),
            ("XZ", "XZ", "Use XZ plane"),
            ("YZ", "YZ", "Use YZ plane"),
            ("VIEW", "View", "Use view axis"),
            ("FREE", "Free", "Free virtual-trackball rotation"),
            ("XYZ", "XYZ", "Uniform scale"),
        ),
        default="X",
    )

    _pivot = Vector((0.0, 0.0, 0.0))
    _axes: dict[str, Vector] | None = None
    _orientation = "GLOBAL"
    _orientation_matrix = Matrix.Identity(3)
    _screen_axis = Vector((1.0, 0.0))
    _pivot_2d = Vector((0.0, 0.0))
    _start_projection = 0.0
    _world_per_pixel = 0.0
    _start_plane_point: Vector | None = None
    _plane_normal = Vector((0.0, 0.0, 1.0))
    _previous_rotation_vector = Vector((1.0, 0.0, 0.0))
    _previous_rotation_mouse = Vector((0.0, 0.0))
    _rotate_screen_tangent = Vector((1.0, 0.0))
    _previous_free_mouse = Vector((0.0, 0.0))
    _free_start_mouse = Vector((0.0, 0.0))
    _free_dragging = False
    _free_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    _raw_angle = 0.0
    _applied_angle = 0.0
    _applied_move = Vector((0.0, 0.0, 0.0))
    _start_scale_measure = 1.0
    _start_scale_projection = 0.0
    _applied_scale = 1.0

    trackball_radius_px: FloatProperty(
        name="",
        description="",
        default=64.0,
        min=8.0,
    )

    @classmethod
    def poll(cls, context):
        route, mode = _route_and_mode(context)
        return route == "NATIVE" and mode in {"MOVE", "ROTATE", "SCALE"}

    def _axis_world(self) -> Vector | None:
        if self._axes is None:
            return None
        if self.axis == "VIEW":
            view_axes = _view_axes(bpy.context)
            return Vector(view_axes["Z"]) if view_axes is not None else None
        axis = self._axes.get(self.axis)
        return Vector(axis) if axis is not None else None

    def _orientation_kwargs(self) -> dict:
        orientation = self._orientation if self._orientation else "GLOBAL"
        return {
            "orient_type": orientation,
            "orient_matrix": self._orientation_matrix,
            "orient_matrix_type": orientation,
        }

    def _exec_move(self, step: Vector) -> None:
        if step.length <= 1e-12:
            return
        bpy.ops.transform.translate(
            value=tuple(float(value) for value in step),
            orient_type="GLOBAL",
            release_confirm=False,
        )

    def _exec_rotate(self, step: float) -> None:
        if abs(step) <= 1e-12:
            return
        axis_name = "Z" if self.axis == "VIEW" else self.axis
        kwargs = self._orientation_kwargs()
        if self.axis == "VIEW":
            kwargs["orient_type"] = "VIEW"
            kwargs["orient_matrix_type"] = "VIEW"
        # Blender's native rotate operator applies this incremental value with
        # the opposite sign from the shared cursor/ring angle convention used
        # by Fit and Animate. Convert only at the native execution boundary so
        # the displayed/snapped angle semantics stay identical everywhere.
        bpy.ops.transform.rotate(
            value=-float(step),
            orient_axis=axis_name,
            center_override=tuple(float(value) for value in self._pivot),
            release_confirm=False,
            **kwargs,
        )

    def _exec_free_rotate(self, rotation: Quaternion) -> None:
        angle = float(rotation.angle)
        if angle <= 1e-12:
            return
        axis = Vector(rotation.axis)
        if axis.length <= 1e-9:
            return
        axis.normalize()
        orient_matrix = _axis_matrix(axis, self._pivot).to_3x3()
        # Match the same native execution-boundary sign conversion used
        # by constrained rotation so Object/Edit/Pose Free Rotate follows
        # Fit/Animate drag direction.
        bpy.ops.transform.rotate(
            value=-angle,
            orient_axis="Z",
            orient_type="GLOBAL",
            orient_matrix=orient_matrix,
            orient_matrix_type="GLOBAL",
            center_override=tuple(float(value) for value in self._pivot),
            release_confirm=False,
        )

    def _exec_scale(self, factor: float) -> None:
        if abs(factor - 1.0) <= 1e-12:
            return
        constraints = {
            "X": (True, False, False),
            "Y": (False, True, False),
            "Z": (False, False, True),
            "XYZ": (False, False, False),
        }
        bpy.ops.transform.resize(
            value=(float(factor), float(factor), float(factor)),
            constraint_axis=constraints.get(self.axis, (False, False, False)),
            center_override=tuple(float(value) for value in self._pivot),
            release_confirm=False,
            **self._orientation_kwargs(),
        )

    def invoke(self, context, event):
        if self.mode == "ROTATE" and self.axis == "FREE":
            from .viewport_keymap import prioritize_awb_selection_over_free_rotate

            if prioritize_awb_selection_over_free_rotate(context, event):
                if context.area is not None:
                    context.area.tag_redraw()
                return {"FINISHED"}

        pivot, axes = _native_pivot_axes(context)
        if pivot is None or axes is None:
            return {"CANCELLED"}
        self._pivot = Vector(pivot)
        self._axes = {name: Vector(vector).normalized() for name, vector in axes.items()}
        self._orientation = str(context.scene.transform_orientation_slots[0].type or "GLOBAL")
        self._orientation_matrix = Matrix(
            (self._axes["X"], self._axes["Y"], self._axes["Z"])
        ).transposed()
        self._raw_angle = 0.0
        self._applied_angle = 0.0
        self._applied_move = Vector((0.0, 0.0, 0.0))
        self._applied_scale = 1.0
        self._free_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))

        if context.region is None or context.region_data is None:
            return {"CANCELLED"}
        pivot_2d = location_3d_to_region_2d(context.region, context.region_data, self._pivot)
        if pivot_2d is None:
            return {"CANCELLED"}
        self._pivot_2d = Vector(pivot_2d)

        if self.mode == "MOVE":
            if self.axis in {"X", "Y", "Z"}:
                axis = self._axis_world()
                if axis is None:
                    return {"CANCELLED"}
                screen_axis = projected_axis_screen_direction(
                    context, self._pivot, axis
                )
                world_per_pixel = projection_world_per_pixel(
                    context, self._pivot
                )
                if screen_axis is None or world_per_pixel is None:
                    return {"CANCELLED"}
                self._screen_axis = screen_axis
                self._world_per_pixel = float(world_per_pixel)
                self._start_projection = float(mouse.dot(self._screen_axis))
            else:
                first, second = {
                    "XY": ("X", "Y"),
                    "XZ": ("X", "Z"),
                    "YZ": ("Y", "Z"),
                }[self.axis]
                normal = self._axes[first].cross(self._axes[second])
                if normal.length <= 1e-9:
                    return {"CANCELLED"}
                self._plane_normal = normal.normalized()
                point = _plane_point(
                    context,
                    self._pivot,
                    self._plane_normal,
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
                if point is None:
                    return {"CANCELLED"}
                self._start_plane_point = Vector(point)

        elif self.mode == "ROTATE":
            if self.axis == "FREE":
                self._previous_free_mouse = Vector(mouse)
                self._free_start_mouse = Vector(mouse)
                self._free_dragging = False
                clear_rotation_angle(context)
            else:
                axis = self._axis_world()
                if axis is None:
                    return {"CANCELLED"}
                start = _rotation_vector(
                    context,
                    self._pivot,
                    axis,
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
                tangent = linear_roll_screen_tangent(
                    context,
                    self._pivot,
                    axis,
                    mouse,
                    Vector(start) if start is not None else None,
                )
                if tangent is None:
                    return {"CANCELLED"}
                if start is not None:
                    self._previous_rotation_vector = Vector(start)
                self._previous_rotation_mouse = Vector(mouse)
                self._rotate_screen_tangent = Vector(tangent)
                show_rotation_angle(context, 0.0, self._pivot)

        elif self.mode == "SCALE":
            if self.axis == "XYZ":
                # Uniform scale locks an outward screen direction at mouse-down.
                # The direction never flips later, even if the cursor crosses
                # the origin, so one continuous inward drag keeps shrinking.
                radial = mouse - self._pivot_2d
                if abs(radial.x) >= abs(radial.y):
                    self._screen_axis = Vector((1.0 if radial.x >= 0.0 else -1.0, 0.0))
                else:
                    self._screen_axis = Vector((0.0, 1.0 if radial.y >= 0.0 else -1.0))
                self._start_scale_projection = float(mouse.dot(self._screen_axis))
            else:
                axis = self._axis_world()
                if axis is None:
                    return {"CANCELLED"}
                screen_axis = projected_axis_screen_direction(
                    context, self._pivot, axis
                )
                if screen_axis is None:
                    return {"CANCELLED"}
                self._screen_axis = screen_axis
                self._start_scale_projection = float(mouse.dot(self._screen_axis))

        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _cancel_applied(self, context) -> None:
        try:
            if self.mode == "MOVE" and self._applied_move.length > 1e-12:
                self._exec_move(-self._applied_move)
            elif self.mode == "ROTATE":
                if self.axis == "FREE":
                    if abs(float(self._free_rotation.angle)) > 1e-12:
                        self._exec_free_rotate(self._free_rotation.inverted())
                elif abs(self._applied_angle) > 1e-12:
                    self._exec_rotate(-self._applied_angle)
            elif self.mode == "SCALE" and abs(self._applied_scale) > 1e-9:
                self._exec_scale(1.0 / self._applied_scale)
        finally:
            clear_rotation_angle(context)

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
            try:
                if self.mode == "MOVE":
                    if self.axis in {"X", "Y", "Z"}:
                        axis = self._axis_world()
                        if axis is None:
                            return {"CANCELLED"}
                        desired_scalar = (
                            float(mouse.dot(self._screen_axis)) - self._start_projection
                        ) * self._world_per_pixel
                        desired = axis * desired_scalar
                    else:
                        point = _plane_point(
                            context,
                            self._pivot,
                            self._plane_normal,
                            event.mouse_region_x,
                            event.mouse_region_y,
                        )
                        if point is None or self._start_plane_point is None:
                            return {"RUNNING_MODAL"}
                        desired = Vector(point) - self._start_plane_point
                    step = desired - self._applied_move
                    self._exec_move(step)
                    self._applied_move = desired

                elif self.mode == "ROTATE":
                    if self.axis == "FREE":
                        if not self._free_dragging:
                            if (mouse - self._free_start_mouse).length < 6.0:
                                return {"RUNNING_MODAL"}
                            self._free_dragging = True
                        rotation_step = arcball_world_step(
                            context,
                            self._pivot,
                            self._previous_free_mouse,
                            mouse,
                            float(self.trackball_radius_px),
                        )
                        if abs(float(rotation_step.angle)) > 1e-12:
                            self._exec_free_rotate(rotation_step)
                            # Native execution applies the inverse-sign
                            # quaternion step. Accumulate logical steps in
                            # gesture order so ESC can undo the exact
                            # non-commutative world-rotation sequence.
                            self._free_rotation = (
                                self._free_rotation @ rotation_step
                            ).normalized()
                        self._previous_free_mouse = Vector(mouse)
                    else:
                        axis = self._axis_world()
                        if axis is None:
                            return {"CANCELLED"}
                        mouse_delta = mouse - self._previous_rotation_mouse
                        raw_step = (
                            float(mouse_delta.dot(self._rotate_screen_tangent))
                            * MAX_LINEAR_ROTATION_RADIANS_PER_PIXEL
                        )
                        self._raw_angle += raw_step
                        self._previous_rotation_mouse = Vector(mouse)
                        desired = snapped_rotation_angle(context, event, self._raw_angle)
                        step = float(desired - self._applied_angle)
                        self._exec_rotate(step)
                        self._applied_angle = float(desired)
                        show_rotation_angle(context, self._applied_angle, self._pivot)

                elif self.mode == "SCALE":
                    current_projection = float(mouse.dot(self._screen_axis))
                    pixel_delta = current_projection - self._start_scale_projection
                    desired = max(0.02, exp(pixel_delta / 260.0))
                    incremental = desired / self._applied_scale
                    self._exec_scale(incremental)
                    self._applied_scale = float(desired)
            except RuntimeError as exc:
                self.report({"WARNING"}, str(exc))
                self._cancel_applied(context)
                return {"CANCELLED"}
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            clear_rotation_angle(context)
            if self.mode == "ROTATE" and self.axis == "FREE" and not self._free_dragging:
                from .viewport_keymap import apply_awb_click_selection

                action = "REMOVE" if event.alt else ("ADD" if event.ctrl else "SET")
                result = apply_awb_click_selection(
                    context,
                    (int(event.mouse_region_x), int(event.mouse_region_y)),
                    (int(event.mouse_x), int(event.mouse_y)),
                    action,
                )
                if context.area is not None:
                    context.area.tag_redraw()
                return result
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._cancel_applied(context)
            if context.area is not None:
                context.area.tag_redraw()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}



class BAW_OT_figure_fit_move_axis(bpy.types.Operator):
    """Object-hosted Figure Move preview that mutates FitDraft only."""

    bl_idname = "baw.figure_fit_move_axis"
    bl_label = "Figure Move"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    axis: EnumProperty(
        items=(
            ("X", "X", "Move on X"),
            ("Y", "Y", "Move on Y"),
            ("Z", "Z", "Move on Z"),
            ("XY", "XY", "Move on XY"),
            ("XZ", "XZ", "Move on XZ"),
            ("YZ", "YZ", "Move on YZ"),
        ),
        default="X",
    )

    @classmethod
    def poll(cls, context):
        route, mode = _route_and_mode(context)
        return route == "FIGURE" and mode == "MOVE"

    def _axis_world(self) -> Vector | None:
        axis = self._axes.get(self.axis)
        return Vector(axis) if axis is not None else None

    def invoke(self, context, event):
        session = fit_semantic_session(context)
        if session is None or session.active_part_id is None:
            return {"CANCELLED"}
        pivot, axes = _pivot_axes(context, "FIGURE")
        if pivot is None or axes is None:
            return {"CANCELLED"}

        self._part_id = str(session.active_part_id)
        self._pivot = Vector(pivot)
        self._axes = {
            name: Vector(vector).normalized()
            for name, vector in axes.items()
        }
        self._applied_move = Vector((0.0, 0.0, 0.0))
        mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))

        if self.axis in {"X", "Y", "Z"}:
            axis = self._axis_world()
            if axis is None:
                return {"CANCELLED"}
            screen_axis = projected_axis_screen_direction(context, self._pivot, axis)
            world_per_pixel = projection_world_per_pixel(context, self._pivot)
            if screen_axis is None or world_per_pixel is None:
                return {"CANCELLED"}
            self._screen_axis = Vector(screen_axis)
            self._world_per_pixel = float(world_per_pixel)
            self._start_projection = float(mouse.dot(self._screen_axis))
        else:
            first, second = {
                "XY": ("X", "Y"),
                "XZ": ("X", "Z"),
                "YZ": ("Y", "Z"),
            }[self.axis]
            normal = self._axes[first].cross(self._axes[second])
            if normal.length <= 1e-9:
                return {"CANCELLED"}
            self._plane_normal = normal.normalized()
            point = _plane_point(
                context,
                self._pivot,
                self._plane_normal,
                event.mouse_region_x,
                event.mouse_region_y,
            )
            if point is None:
                return {"CANCELLED"}
            self._start_plane_point = Vector(point)

        try:
            self._gesture = begin_fit_move_gesture(
                context,
                part_id=self._part_id,
            )
        except FitSemanticSessionError as exc:
            trace_event(
                "INPUT",
                "FIT_F3_MOVE_REFUSED",
                context=context,
                part_id=self._part_id,
                axis=self.axis,
                reason=str(exc),
            )
            return {"CANCELLED"}

        trace_event(
            "INPUT",
            "FIT_F3_MOVE_BEGIN",
            context=context,
            part_id=self._part_id,
            axis=self.axis,
            fit_revision=self._gesture.revision,
            preview_serial=self._gesture.preview_serial,
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _cancel_gesture(self, context) -> None:
        gesture = getattr(self, "_gesture", None)
        if gesture is None:
            return
        try:
            cancel_fit_move_gesture(context, gesture)
        except FitSemanticSessionError:
            pass

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
            if self.axis in {"X", "Y", "Z"}:
                axis = self._axis_world()
                if axis is None:
                    return {"CANCELLED"}
                desired_scalar = (
                    float(mouse.dot(self._screen_axis)) - self._start_projection
                ) * self._world_per_pixel
                desired = axis * desired_scalar
            else:
                point = _plane_point(
                    context,
                    self._pivot,
                    self._plane_normal,
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
                if point is None:
                    return {"RUNNING_MODAL"}
                desired = Vector(point) - self._start_plane_point

            try:
                receipt = apply_fit_move_preview(
                    context,
                    gesture=self._gesture,
                    world_delta=desired,
                )
            except FitSemanticSessionError as exc:
                self._cancel_gesture(context)
                trace_event(
                    "ERROR",
                    "FIT_F3_MOVE_FAIL",
                    context=context,
                    part_id=self._part_id,
                    axis=self.axis,
                    reason=str(exc),
                )
                self.report({"WARNING"}, str(exc))
                return {"CANCELLED"}

            self._applied_move = Vector(desired)
            trace_event(
                "INPUT",
                "FIT_F3_MOVE_PREVIEW",
                context=context,
                part_id=self._part_id,
                axis=self.axis,
                world_delta=receipt.world_delta,
                rig_delta=receipt.rig_delta,
                fit_revision=receipt.revision_after,
                preview_serial=receipt.preview_serial,
                changed=receipt.changed,
            )
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            try:
                changed = commit_fit_move_gesture(context, self._gesture)
            except FitSemanticSessionError as exc:
                self._cancel_gesture(context)
                trace_event(
                    "ERROR",
                    "FIT_F3_MOVE_FAIL",
                    context=context,
                    part_id=self._part_id,
                    axis=self.axis,
                    reason=str(exc),
                )
                self.report({"WARNING"}, str(exc))
                return {"CANCELLED"}
            session = fit_semantic_session(context)
            trace_event(
                "INPUT",
                "FIT_F3_MOVE_COMMIT",
                context=context,
                part_id=self._part_id,
                axis=self.axis,
                world_delta=tuple(float(value) for value in self._applied_move),
                fit_revision=(None if session is None else int(session.revision)),
                preview_serial=(None if session is None else int(session.preview_serial)),
                changed=changed,
            )
            return {"FINISHED"}

        if event.type in {"RIGHTMOUSE", "ESC"}:
            self._cancel_gesture(context)
            session = fit_semantic_session(context)
            trace_event(
                "INPUT",
                "FIT_F3_MOVE_CANCEL",
                context=context,
                part_id=self._part_id,
                axis=self.axis,
                fit_revision=(None if session is None else int(session.revision)),
                preview_serial=(None if session is None else int(session.preview_serial)),
            )
            if context.area is not None:
                context.area.tag_redraw()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_GT_free_rotate_disk(bpy.types.Gizmo):
    """Invisible circular hit surface for Max-style free rotation."""

    bl_idname = "BAW_GT_free_rotate_disk"

    def setup(self):
        segments = 40
        vertices = []
        center = (0.0, 0.0, 0.0)
        for index in range(segments):
            angle_a = (2.0 * pi * index) / segments
            angle_b = (2.0 * pi * (index + 1)) / segments
            vertices.extend(
                (
                    center,
                    (cos(angle_a), sin(angle_a), 0.0),
                    (cos(angle_b), sin(angle_b), 0.0),
                )
            )
        self._shape = self.new_custom_shape("TRIS", tuple(vertices))

    def draw(self, _context):
        self.draw_custom_shape(self._shape)

    def draw_select(self, _context, select_id):
        self.draw_custom_shape(self._shape, select_id=select_id)


class BAW_GT_uniform_scale_box(bpy.types.Gizmo):
    """Centered 3D cube used by the global uniform-scale handle."""

    bl_idname = "BAW_GT_uniform_scale_box"

    def setup(self):
        h = 0.5
        p000 = (-h, -h, -h)
        p001 = (-h, -h, h)
        p010 = (-h, h, -h)
        p011 = (-h, h, h)
        p100 = (h, -h, -h)
        p101 = (h, -h, h)
        p110 = (h, h, -h)
        p111 = (h, h, h)
        self._shape = self.new_custom_shape(
            "TRIS",
            (
                p000, p100, p110, p000, p110, p010,
                p001, p011, p111, p001, p111, p101,
                p000, p001, p101, p000, p101, p100,
                p010, p110, p111, p010, p111, p011,
                p000, p010, p011, p000, p011, p001,
                p100, p101, p111, p100, p111, p110,
            ),
        )

    def draw(self, _context):
        self.draw_custom_shape(self._shape)

    def draw_select(self, _context, select_id):
        self.draw_custom_shape(self._shape, select_id=select_id)


class BAW_GGT_global_transform(bpy.types.GizmoGroup):
    """One AWB transform gizmo shell for Object, Fit and Animate contexts."""

    bl_idname = "BAW_GGT_global_transform"
    bl_label = "AWB Global Transform Gizmo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options: ClassVar[set[str]] = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context):
        if fit_ui_state_present(context) and getattr(context, "mode", "") == "OBJECT":
            # Keep the persistent shell alive throughout the raw Figure/Fit host
            # lifetime, including stale semantic-session states, so native Object
            # transform routes remain suppressed fail-closed.
            return True
        route, mode = _route_and_mode(context)
        return bool(route and mode in {"MOVE", "ROTATE", "SCALE"})

    def setup(self, _context):
        self._zoom_reference_distance = None
        self.move_axes = {}
        self.move_planes = {}
        self.rotate_axes = {}
        self.scale_axes = {}
        self.hit_routes = {"MOVE": {}, "ROTATE": {}, "SCALE": {}}

        for axis in ("X", "Y", "Z"):
            move = self.gizmos.new("GIZMO_GT_arrow_3d")
            move.draw_style = "NORMAL"
            move.draw_options = {"STEM"}
            move.color = GIZMO_AXIS_COLORS[axis]
            move.alpha = 0.9
            move.line_width = 1.0
            move.hide_select = True
            # Keep the shared Move arrow visible normally, but do not let the
            # gizmo itself draw Blender's modal drag guide/ghost feedback.
            # Axis selection/dragging is still owned by the invisible hit gizmo.
            move.use_draw_modal = False
            self.move_axes[axis] = move

            rotate = self.gizmos.new("GIZMO_GT_dial_3d")
            rotate.color = GIZMO_AXIS_COLORS[axis]
            rotate.alpha = 0.65
            rotate.line_width = 2.0
            rotate.draw_options = {"CLIP"}
            rotate.hide_select = True
            rotate.use_draw_modal = True
            self.rotate_axes[axis] = rotate

            scale = self.gizmos.new("GIZMO_GT_arrow_3d")
            scale.draw_style = "BOX"
            scale.draw_options = {"STEM"}
            scale.color = GIZMO_AXIS_COLORS[axis]
            scale.alpha = 0.8
            scale.line_width = 1.0
            scale.hide_select = True
            scale.use_draw_modal = True
            self.scale_axes[axis] = scale

        for plane in ("XY", "XZ", "YZ"):
            gizmo = self.gizmos.new("GIZMO_GT_primitive_3d")
            gizmo.draw_style = "PLANE"
            gizmo.draw_inner = False
            gizmo.color = (0.52, 0.52, 0.52)
            gizmo.alpha = 0.9
            gizmo.line_width = 1.2
            gizmo.scale_basis = 0.11
            gizmo.hide_select = True
            gizmo.use_draw_offset_scale = True
            gizmo.use_draw_modal = True
            self.move_planes[plane] = gizmo

        base = self.gizmos.new("GIZMO_GT_dial_3d")
        base.color = (0.42, 0.42, 0.42)
        base.alpha = 0.75
        base.line_width = 1.2
        base.scale_basis = 1.0
        base.hide_select = True
        base.use_draw_modal = True
        self.rotate_base = base

        view = self.gizmos.new("GIZMO_GT_dial_3d")
        view.color = (0.85, 0.85, 0.85)
        view.alpha = 0.45
        view.line_width = 1.5
        view.scale_basis = 1.15
        view.hide_select = True
        view.use_draw_modal = True
        self.rotate_view = view

        # Max-style uniform scale handle: a custom cube whose geometry is
        # mathematically centered on local (0, 0, 0), so its center is always
        # the actual transform origin regardless of camera or orientation.
        uniform = self.gizmos.new(BAW_GT_uniform_scale_box.bl_idname)
        uniform.color = (0.88, 0.88, 0.88)
        uniform.alpha = 0.8
        uniform.scale_basis = 0.275
        uniform.hide_select = True
        uniform.use_draw_modal = True
        self.scale_uniform = uniform

        self._setup_move_hits()
        self._setup_rotate_hits()
        self._setup_scale_hits()

    def _setup_move_hits(self):
        handles = ("X", "Y", "Z", "XY", "XZ", "YZ")
        for route, operator in (
            ("NATIVE", BAW_OT_global_native_transform_axis.bl_idname),
            ("FIT", BAW_OT_rigped_fit_transform_axis.bl_idname),
            ("FIGURE", BAW_OT_figure_fit_move_axis.bl_idname),
            ("DIRECT_MOVE", BAW_OT_rigped_direct_move_axis.bl_idname),
            ("FK_MOVE", BAW_OT_rigped_fk_joint_move_axis.bl_idname),
            ("SEMANTIC_MOVE", BAW_OT_rigped_semantic_move_axis.bl_idname),
        ):
            route_hits = {}
            for handle in handles:
                gizmo_type = (
                    "GIZMO_GT_arrow_3d"
                    if handle in {"X", "Y", "Z"}
                    else "GIZMO_GT_primitive_3d"
                )
                hit, props = _new_invisible_hit(self, gizmo_type, operator)
                if handle in {"X", "Y", "Z"}:
                    hit.draw_style = "NORMAL"
                    hit.draw_options = {"STEM"}
                    hit.scale_basis = 1.0
                    hit.select_bias = 0.75
                else:
                    hit.draw_style = "PLANE"
                    hit.draw_inner = True
                    hit.use_draw_offset_scale = True
                    hit.scale_basis = 0.16
                    hit.select_bias = 5.0
                if route == "NATIVE":
                    props.mode = "MOVE"
                    props.axis = handle
                else:
                    if route == "FIT":
                        props.mode = "MOVE"
                    props.axis = handle
                route_hits[handle] = (hit, props)
            self.hit_routes["MOVE"][route] = route_hits

    def _setup_rotate_hits(self):
        for route, operator in (
            ("NATIVE", BAW_OT_global_native_transform_axis.bl_idname),
            ("FIT", BAW_OT_rigped_fit_transform_axis.bl_idname),
            ("DIRECT_ROTATE", BAW_OT_rigped_direct_rotate_axis.bl_idname),
        ):
            route_hits = {}
            for handle in ("X", "Y", "Z", "VIEW", "FREE"):
                gizmo_type = (
                    BAW_GT_free_rotate_disk.bl_idname
                    if handle == "FREE"
                    else "GIZMO_GT_dial_3d"
                )
                hit, props = _new_invisible_hit(self, gizmo_type, operator)
                if handle == "FREE":
                    hit.scale_basis = 0.92
                    hit.select_bias = 0.1
                else:
                    hit.scale_basis = 1.15 if handle == "VIEW" else 1.0
                    if handle != "VIEW":
                        hit.draw_options = {"CLIP"}
                if route == "NATIVE":
                    props.mode = "ROTATE"
                    props.axis = handle
                else:
                    if route == "FIT":
                        props.mode = "ROTATE"
                    props.axis = handle
                route_hits[handle] = (hit, props)
            self.hit_routes["ROTATE"][route] = route_hits

    def _setup_scale_hits(self):
        for route, operator in (("NATIVE", BAW_OT_global_native_transform_axis.bl_idname), ("FIT", BAW_OT_rigped_fit_scale_axis.bl_idname)):
            route_hits = {}
            for handle in ("X", "Y", "Z", "XYZ"):
                gizmo_type = (
                    "GIZMO_GT_arrow_3d"
                    if handle != "XYZ"
                    else BAW_GT_uniform_scale_box.bl_idname
                )
                hit, props = _new_invisible_hit(self, gizmo_type, operator)
                if handle != "XYZ":
                    hit.draw_style = "BOX"
                    hit.draw_options = set()
                    hit.scale_basis = 1.0
                    hit.select_bias = 0.8
                else:
                    hit.scale_basis = 3.0
                    hit.select_bias = 2.0
                if route == "NATIVE":
                    props.mode = "SCALE"
                    props.axis = handle
                else:
                    props.axis = handle
                route_hits[handle] = (hit, props)
            self.hit_routes["SCALE"][route] = route_hits

    def _hide_all(self):
        visible = [*self.move_axes.values(), *self.move_planes.values(), *self.rotate_axes.values(), self.rotate_base, self.rotate_view, *self.scale_axes.values(), self.scale_uniform]
        for gizmo in visible:
            gizmo.hide = True
        for routes in self.hit_routes.values():
            for route_hits in routes.values():
                for hit, _props in route_hits.values():
                    hit.hide = True

    def draw_prepare(self, context):
        self._hide_all()
        _force_figure_safe_workspace_tool(context)
        if (
            fit_ui_state_present(context)
            and getattr(context, "mode", "") == "OBJECT"
            and hasattr(context.space_data, "show_gizmo_tool")
        ):
            context.space_data.show_gizmo_tool = False
        route, mode = _route_and_mode(context)
        if not route or not mode:
            return
        if route == "NATIVE" and hasattr(context.space_data, "show_gizmo_tool"):
            # The active Blender transform tool still owns semantics/operators;
            # only its built-in drawing is suppressed so the single AWB shell
            # is the visible gizmo in every supported View3D mode.
            context.space_data.show_gizmo_tool = False
        pivot, axes = _pivot_axes(context, route)
        if pivot is None or axes is None:
            return

        hits = self.hit_routes.get(mode, {}).get(route)
        if hits is None:
            return

        zoom_scale, self._zoom_reference_distance = gizmo_zoom_scale(context, self._zoom_reference_distance)
        addon_preferences = addon_gizmo_preferences(context)
        axis_ring_width = float(getattr(addon_preferences, "gizmo_rotate_axis_width", 2.0))
        view_ring_width = float(getattr(addon_preferences, "gizmo_rotate_view_width", 1.5))
        axis_ring_hit_width = float(
            getattr(addon_preferences, "gizmo_rotate_axis_hit_width", 8.0)
        )
        view_ring_hit_width = float(
            getattr(addon_preferences, "gizmo_rotate_view_hit_width", 6.0)
        )
        rotate_axis_colors = {
            "X": tuple(getattr(addon_preferences, "gizmo_rotate_x_color", GIZMO_AXIS_COLORS["X"])),
            "Y": tuple(getattr(addon_preferences, "gizmo_rotate_y_color", GIZMO_AXIS_COLORS["Y"])),
            "Z": tuple(getattr(addon_preferences, "gizmo_rotate_z_color", GIZMO_AXIS_COLORS["Z"])),
        }
        rotate_view_color = tuple(
            getattr(addon_preferences, "gizmo_rotate_view_color", (0.85, 0.85, 0.85))
        )
        move_axis_width = float(getattr(addon_preferences, "gizmo_move_axis_width", 1.0))
        move_arrow_length = max(
            0.4,
            min(
                2.2,
                float(getattr(addon_preferences, "gizmo_move_arrow_length_percent", 100.0))
                / 100.0,
            ),
        )
        move_arrow_head_scale = max(
            0.5,
            min(
                2.0,
                float(getattr(addon_preferences, "gizmo_move_arrow_head_size_percent", 100.0))
                / 100.0,
            ),
        )
        move_plane_scale = max(
            0.8,
            min(
                4.4,
                float(getattr(addon_preferences, "gizmo_move_plane_size_percent", 100.0))
                / 50.0,
            ),
        )
        move_axis_colors = {
            "X": tuple(getattr(addon_preferences, "gizmo_move_x_color", GIZMO_AXIS_COLORS["X"])),
            "Y": tuple(getattr(addon_preferences, "gizmo_move_y_color", GIZMO_AXIS_COLORS["Y"])),
            "Z": tuple(getattr(addon_preferences, "gizmo_move_z_color", GIZMO_AXIS_COLORS["Z"])),
        }
        scale_axis_width = float(getattr(addon_preferences, "gizmo_scale_axis_width", 1.0))
        scale_axis_length = max(
            0.4,
            min(
                2.2,
                float(getattr(addon_preferences, "gizmo_scale_axis_length_percent", 100.0))
                / 100.0,
            ),
        )
        scale_handle_scale = max(
            0.5,
            min(
                2.0,
                float(getattr(addon_preferences, "gizmo_scale_handle_size_percent", 100.0))
                / 100.0,
            ),
        )
        scale_uniform_scale = max(
            0.4,
            min(
                2.2,
                float(getattr(addon_preferences, "gizmo_scale_uniform_size_percent", 100.0))
                / 100.0,
            ),
        )
        scale_axis_colors = {
            "X": tuple(getattr(addon_preferences, "gizmo_scale_x_color", GIZMO_AXIS_COLORS["X"])),
            "Y": tuple(getattr(addon_preferences, "gizmo_scale_y_color", GIZMO_AXIS_COLORS["Y"])),
            "Z": tuple(getattr(addon_preferences, "gizmo_scale_z_color", GIZMO_AXIS_COLORS["Z"])),
        }
        view_ring_scale = max(
            1.0,
            min(
                1.6,
                float(getattr(addon_preferences, "gizmo_rotate_view_scale_percent", 115.0))
                / 100.0,
            ),
        )
        orientation = str(context.scene.transform_orientation_slots[0].type)
        if route in {"NATIVE", "DIRECT_MOVE"}:
            for handle, (_hit, props) in hits.items():
                if hasattr(props, "orient_type"):
                    props.orient_type = "VIEW" if mode == "ROTATE" and handle == "VIEW" else orientation

        if mode == "ROTATE":
            view_axes = _view_axes(context)
            if view_axes is not None:
                base_matrix = _axis_matrix(view_axes["Z"], pivot)
                free_hit, free_props = hits["FREE"]
                free_hit.scale_basis = 0.92 * zoom_scale
                free_hit.matrix_basis = base_matrix
                free_hit.select_bias = 0.1
                free_hit.hide = False

                view_preferences = getattr(
                    getattr(context, "preferences", None), "view", None
                )
                radius_px = max(
                    8.0,
                    float(getattr(view_preferences, "gizmo_size", 75.0))
                    * float(zoom_scale),
                )
                if hasattr(free_props, "trackball_radius_px"):
                    free_props.trackball_radius_px = radius_px

                free_active = bool(
                    getattr(free_hit, "is_highlight", False)
                ) or bool(getattr(free_hit, "is_modal", False))
                self.rotate_base.color = (
                    GIZMO_HIGHLIGHT_COLOR
                    if free_active
                    else (0.42, 0.42, 0.42)
                )
                self.rotate_base.alpha = 0.95 if free_active else 0.75
                self.rotate_base.scale_basis = zoom_scale
                self.rotate_base.matrix_basis = base_matrix
                self.rotate_base.hide = False

            for axis, visible in self.rotate_axes.items():
                hit, _props = hits[axis]
                visible.line_width = axis_ring_width
                hit.line_width = max(axis_ring_hit_width, axis_ring_width)
                hit.select_bias = 2.4
                visible.scale_basis = zoom_scale
                hit.scale_basis = zoom_scale
                matrix = _axis_matrix(axes[axis], pivot)
                visible.matrix_basis = matrix
                hit.matrix_basis = matrix
                active = bool(getattr(hit, "is_highlight", False)) or bool(getattr(hit, "is_modal", False))
                visible.color = GIZMO_HIGHLIGHT_COLOR if active else rotate_axis_colors[axis]
                visible.alpha = 0.95 if active else 0.65
                visible.hide = False
                hit.hide = False
            view_axes = _view_axes(context)
            if view_axes is not None:
                hit, _props = hits["VIEW"]
                matrix = _axis_matrix(view_axes["Z"], pivot)
                self.rotate_view.line_width = view_ring_width
                hit.line_width = max(view_ring_hit_width, view_ring_width)
                hit.select_bias = 1.4
                self.rotate_view.scale_basis = view_ring_scale * zoom_scale
                hit.scale_basis = view_ring_scale * zoom_scale
                self.rotate_view.matrix_basis = matrix
                hit.matrix_basis = matrix
                active = bool(getattr(hit, "is_highlight", False)) or bool(getattr(hit, "is_modal", False))
                self.rotate_view.color = GIZMO_HIGHLIGHT_COLOR if active else rotate_view_color
                self.rotate_view.alpha = 0.95 if active else 0.45
                self.rotate_view.hide = False
                hit.hide = False
            return

        if mode == "MOVE":
            for axis, visible in self.move_axes.items():
                hit, _props = hits[axis]
                visible.line_width = move_axis_width
                hit.line_width = max(4.0, move_axis_width + 3.0)
                # Arrow length and head size are independently adjustable: the
                # whole gizmo scales the head, while the intrinsic arrow length
                # is divided by that scale to preserve the requested total reach.
                arrow_scale = move_arrow_head_scale * zoom_scale
                arrow_length = move_arrow_length / move_arrow_head_scale
                visible.scale_basis = arrow_scale
                hit.scale_basis = arrow_scale
                if hasattr(visible, "length"):
                    visible.length = arrow_length
                if hasattr(hit, "length"):
                    hit.length = arrow_length
                matrix = _axis_matrix(axes[axis], pivot)
                visible.matrix_basis = matrix
                hit.matrix_basis = matrix
                visible.color = (
                    GIZMO_HIGHLIGHT_COLOR
                    if bool(getattr(hit, "is_highlight", False))
                    else move_axis_colors[axis]
                )
                visible.hide = False
                hit.hide = False
            plane_axes = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}
            plane_colors = {
                "XY": move_axis_colors["Z"],
                "XZ": move_axis_colors["Y"],
                "YZ": move_axis_colors["X"],
            }
            view_axes = _view_axes(context)
            view_toward_user = view_axes["Z"] if view_axes is not None else None
            panel_depth: dict[str, float] = {}
            if view_toward_user is not None:
                for plane, (first_name, second_name) in plane_axes.items():
                    center_direction = Vector(axes[first_name]) + Vector(axes[second_name])
                    panel_depth[plane] = float(center_direction.dot(view_toward_user))
            depth_order = sorted(
                plane_axes,
                key=lambda name: panel_depth.get(name, 0.0),
                reverse=True,
            )
            panel_bias = {
                plane: (10.0, 4.0, 0.5)[min(index, 2)]
                for index, plane in enumerate(depth_order)
            }

            for plane, visible in self.move_planes.items():
                hit, _props = hits[plane]
                first_name, second_name = plane_axes[plane]
                first, second = Vector(axes[first_name]), Vector(axes[second_name])
                normal = first.cross(second).normalized()
                matrix = Matrix((first, second, normal)).transposed().to_4x4()
                matrix.translation = pivot
                # GIZMO_GT_primitive_3d PLANE spans -1..+1 in local X/Y.
                # A +1/+1 local offset puts one square corner exactly on the
                # transform origin, so the three Max-style panels meet only at
                # that corner instead of overlapping around the pivot.
                offset = Matrix.Translation((1.0, 1.0, 0.0))
                visible.scale_basis = 0.11 * move_plane_scale * zoom_scale
                hit.scale_basis = 0.16 * move_plane_scale * zoom_scale
                visible.matrix_basis = matrix
                hit.matrix_basis = matrix
                visible.matrix_offset = offset
                hit.matrix_offset = offset
                active = bool(getattr(hit, "is_highlight", False)) or bool(
                    getattr(hit, "is_modal", False)
                )
                visible.draw_inner = True
                visible.line_width = 0.0
                visible.alpha = 0.65 if active else 0.5
                # When plane projections overlap, prefer the panel whose center
                # is closest to the viewer. Axis lines remain lower priority.
                hit.select_bias = panel_bias.get(plane, 3.5)
                visible.color = GIZMO_HIGHLIGHT_COLOR if active else plane_colors[plane]
                visible.hide = False
                hit.hide = False
            return

        for axis, visible in self.scale_axes.items():
            hit, _props = hits[axis]
            visible.line_width = scale_axis_width
            hit.line_width = max(4.0, scale_axis_width + 3.0)
            handle_scale = scale_handle_scale * zoom_scale
            axis_length = scale_axis_length / scale_handle_scale
            visible.scale_basis = handle_scale
            hit.scale_basis = handle_scale
            if hasattr(visible, "length"):
                visible.length = axis_length
            if hasattr(hit, "length"):
                hit.length = axis_length
            matrix = _axis_matrix(axes[axis], pivot)
            visible.matrix_basis = matrix
            hit.matrix_basis = matrix
            active = bool(getattr(hit, "is_highlight", False)) or bool(
                getattr(hit, "is_modal", False)
            )
            visible.color = GIZMO_HIGHLIGHT_COLOR if active else scale_axis_colors[axis]
            visible.alpha = 0.95 if active else 0.8
            visible.hide = False
            hit.hide = False
        hit, _props = hits["XYZ"]
        # The uniform box belongs to the gizmo basis itself, not to the camera.
        # Keeping the full X/Y/Z basis here makes it stay centered on the true
        # transform origin and prevents view-dependent billboard reshaping.
        matrix = Matrix(
            (Vector(axes["X"]), Vector(axes["Y"]), Vector(axes["Z"]))
        ).transposed().to_4x4()
        matrix.translation = pivot
        self.scale_uniform.scale_basis = 0.275 * scale_uniform_scale * zoom_scale
        hit.scale_basis = 0.375 * scale_uniform_scale * zoom_scale
        self.scale_uniform.matrix_basis = matrix
        hit.matrix_basis = matrix
        active = bool(getattr(hit, "is_highlight", False)) or bool(
            getattr(hit, "is_modal", False)
        )
        self.scale_uniform.color = (
            GIZMO_HIGHLIGHT_COLOR if active else (0.88, 0.88, 0.88)
        )
        self.scale_uniform.alpha = 0.95 if active else 0.8
        self.scale_uniform.hide = False
        hit.hide = False
