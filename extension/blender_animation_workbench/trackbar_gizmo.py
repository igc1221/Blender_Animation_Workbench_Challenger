from __future__ import annotations

import math
from typing import ClassVar

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from .debug_trace import trace_scrub_begin, trace_scrub_end
from .semantic_adapter import trackbar_controls_for_context
from .trackbar_drawing import (
    CURRENT_FRAME_CONTROL_Y_MAX,
    CURRENT_FRAME_CONTROL_Y_MIN,
    KEY_MARKER_BOTTOM,
    KEY_MARKER_HEIGHT,
    TRACK_LINE_Y,
    TRACKBAR_HEIGHT,
    auto_key_button_bounds,
    clear_box_selection_overlay,
    frame_cell_width,
    frame_control_bounds,
    key_mode_button_bounds,
    selection_range_bounds,
    set_box_selection_overlay,
    trackbar_bounds,
    transport_endpoint_button_bounds,
    visible_frame_window,
)
from .trackbar_math import frame_delta_from_pixel_drag, x_to_cell_frame
from .trackbar_model import (
    adjacent_key_frame_for_context,
    apply_key_properties_for_context,
    clone_key_frame_for_context,
    clone_selected_key_frames_for_context,
    delete_selected_keys_for_context,
    has_key_at_frame,
    key_channels_for_context,
    key_property_state_for_targets,
    key_property_targets_for_context,
    move_key_frame_for_context,
    move_selected_key_frames_for_context,
    remove_cloned_key_frame_for_context,
    restore_active_key_state_for_context,
    restore_key_frame_snapshots_for_context,
    restore_multi_key_preview_for_context,
    restore_selection_range_scale_preview_for_context,
    scale_selected_key_frames_for_context,
    select_key_frame_for_context,
    select_key_frame_range_for_context,
    selected_key_frames_for_context,
    snapshot_active_key_state_for_context,
    snapshot_multi_key_preview_for_context,
    snapshot_replaced_key_frame_for_context,
    snapshot_selection_range_scale_for_context,
    update_multi_key_preview_for_context,
    update_selection_range_scale_preview_for_context,
)
from .viewport_trajectory import (
    refresh_active_trajectory_live,
    sync_active_trajectory_key_selection,
)

RULER_RESIZE_Y_MIN = 0.0
RULER_RESIZE_Y_MAX = float(TRACK_LINE_Y)

_INTERACTION_KEYMAP_ITEMS: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []
_TRACKBAR_KEY_CONTEXT_FRAME: int | None = None


def _set_trackbar_view_bounds(scene, start: int, end: int) -> None:
    if end <= start:
        end = start + 1
    scene.baw_trackbar_view_center = (start + end) * 0.5
    scene.baw_trackbar_view_span = end - start + 1


class BAW_OT_resize_end_frame_drag(bpy.types.Operator):
    bl_idname = "baw.resize_end_frame_drag"
    bl_label = "Resize Track View"
    bl_description = "Alt-drag the lower ruler to expand or contract the visible Track Bar view from either side"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.scene is not None
            and getattr(context.scene, "baw_trackbar_enabled", True)
        )

    def invoke(self, context, event):
        region = context.region
        scene = context.scene
        if region is None or scene is None:
            return {"PASS_THROUGH"}

        visible_start, visible_end, visible_span, x_min, x_max = visible_frame_window(
            scene,
            region,
        )
        if not (
            x_min <= event.mouse_region_x <= x_max
            and RULER_RESIZE_Y_MIN <= event.mouse_region_y <= RULER_RESIZE_Y_MAX
        ):
            return {"PASS_THROUGH"}

        midpoint = (x_min + x_max) * 0.5
        self._resize_side = "START" if event.mouse_region_x <= midpoint else "END"
        self._initial_visible_start = int(visible_start)
        self._initial_visible_end = int(visible_end)
        self._initial_view_center_prop = float(scene.baw_trackbar_view_center)
        self._initial_view_span_prop = int(scene.baw_trackbar_view_span)
        self._initial_mouse_x = int(event.mouse_region_x)
        self._initial_cell_width = max(0.01, frame_cell_width(scene, region))

        # Alt-drag edits the independent Track Bar view itself. Playback
        # Start/End are intentionally untouched so the view can extend freely
        # through negative frames and remain consistent with MMB/wheel.
        scene.baw_trackbar_view_center = (visible_start + visible_end) * 0.5
        scene.baw_trackbar_view_span = int(visible_span)

        if context.window is not None:
            context.window.cursor_modal_set("MOVE_X")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        scene = context.scene
        if event.type == "MOUSEMOVE":
            delta_x = event.mouse_region_x - self._initial_mouse_x
            raw_delta = frame_delta_from_pixel_drag(delta_x, self._initial_cell_width)
            if self._resize_side == "START":
                new_start = min(
                    self._initial_visible_end - 1,
                    self._initial_visible_start - raw_delta,
                )
                _set_trackbar_view_bounds(scene, new_start, self._initial_visible_end)
            else:
                new_end = max(
                    self._initial_visible_start + 1,
                    self._initial_visible_end - raw_delta,
                )
                _set_trackbar_view_bounds(scene, self._initial_visible_start, new_end)
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if context.window is not None:
                context.window.cursor_modal_restore()
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            scene.baw_trackbar_view_center = self._initial_view_center_prop
            scene.baw_trackbar_view_span = self._initial_view_span_prop
            if context.window is not None:
                context.window.cursor_modal_restore()
            if context.area is not None:
                context.area.tag_redraw()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_OT_pan_trackbar_drag(bpy.types.Operator):
    bl_idname = "baw.pan_trackbar_drag"
    bl_label = "Pan Track Bar"
    bl_description = "Pan the visible Track Bar frame window with Middle Mouse"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.scene is not None
            and getattr(context.scene, "baw_trackbar_enabled", True)
        )

    def invoke(self, context, event):
        region = context.region
        scene = context.scene
        if region is None or scene is None:
            return {"PASS_THROUGH"}

        x_min, x_max, y_min, y_max = trackbar_bounds(region)
        if not (
            x_min <= event.mouse_region_x <= x_max
            and y_min <= event.mouse_region_y <= y_max
        ):
            return {"PASS_THROUGH"}

        visible_start, visible_end, visible_span, _x_min, _x_max = visible_frame_window(
            scene,
            region,
        )
        self._initial_mouse_x = int(event.mouse_region_x)
        self._initial_view_center = (visible_start + visible_end) * 0.5
        self._initial_view_span = int(visible_span)
        self._initial_cell_width = max(0.01, frame_cell_width(scene, region))
        scene.baw_trackbar_view_center = self._initial_view_center
        scene.baw_trackbar_view_span = self._initial_view_span

        if context.window is not None:
            context.window.cursor_modal_set("MOVE_X")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        scene = context.scene
        if event.type == "MOUSEMOVE":
            delta_x = event.mouse_region_x - self._initial_mouse_x
            delta_frames = frame_delta_from_pixel_drag(delta_x, self._initial_cell_width)
            scene.baw_trackbar_view_center = self._initial_view_center - delta_frames
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "MIDDLEMOUSE" and event.value == "RELEASE":
            if context.window is not None:
                context.window.cursor_modal_restore()
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            scene.baw_trackbar_view_center = self._initial_view_center
            scene.baw_trackbar_view_span = self._initial_view_span
            if context.window is not None:
                context.window.cursor_modal_restore()
            if context.area is not None:
                context.area.tag_redraw()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_OT_zoom_trackbar_wheel(bpy.types.Operator):
    bl_idname = "baw.zoom_trackbar_wheel"
    bl_label = "Zoom Track Bar"
    bl_description = "Zoom the visible Track Bar frame window around its current center"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.scene is not None
            and getattr(context.scene, "baw_trackbar_enabled", True)
        )

    def invoke(self, context, event):
        region = context.region
        scene = context.scene
        if region is None or scene is None:
            return {"PASS_THROUGH"}

        x_min, x_max, y_min, y_max = trackbar_bounds(region)
        if not (
            x_min <= event.mouse_region_x <= x_max
            and y_min <= event.mouse_region_y <= y_max
        ):
            return {"PASS_THROUGH"}

        visible_start, visible_end, visible_span, _x_min, _x_max = visible_frame_window(
            scene,
            region,
        )
        center = (visible_start + visible_end) * 0.5
        span = max(2, int(visible_span))

        if event.type == "WHEELUPMOUSE":
            new_span = max(2, round(span * 0.85))
            if new_span == span and span > 2:
                new_span = span - 1
        elif event.type == "WHEELDOWNMOUSE":
            new_span = min(100000, round(span * 1.18))
            if new_span == span:
                new_span = min(100000, span + 1)
        else:
            return {"PASS_THROUGH"}

        scene.baw_trackbar_view_center = center
        scene.baw_trackbar_view_span = new_span
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


_KEY_MENU_BG = (28.0 / 255.0, 28.0 / 255.0, 28.0 / 255.0, 1.0)
_KEY_MENU_BORDER = (0.24, 0.24, 0.24, 1.0)
_KEY_MENU_HOVER = (0.19, 0.19, 0.19, 1.0)
_KEY_MENU_TEXT = (0.90, 0.90, 0.90, 1.0)
_KEY_MENU_DIM = (0.62, 0.62, 0.62, 1.0)
_KEY_MENU_WIDTH = 188.0
_KEY_MENU_SUB_WIDTH = 164.0
_KEY_MENU_ROW_HEIGHT = 24.0
_KEY_MENU_TITLE_HEIGHT = 24.0
_KEY_MENU_PADDING = 6.0
_KEY_MENU_TRACK_GAP = 8.0


def _draw_key_menu_rect(x: float, y: float, width: float, height: float, color) -> None:
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    vertices = (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )
    batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": vertices},
        indices=((0, 1, 2), (0, 2, 3)),
    )
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_key_menu_text(x: float, y: float, text: str, color=_KEY_MENU_TEXT) -> None:
    font_id = 0
    blf.size(font_id, 12)
    blf.color(font_id, *color)
    blf.position(font_id, x, y, 0)
    blf.draw(font_id, text)


def _point_in_key_menu_rect(x: float, y: float, rect) -> bool:
    if not isinstance(rect, (tuple, list)) or len(rect) != 4:
        return False
    left, bottom, right, top = rect
    return left <= x <= right and bottom <= y <= top


class BAW_OT_block_trackbar_mouse(bpy.types.Operator):
    bl_idname = "baw.block_trackbar_mouse"
    bl_label = "Key Properties"
    bl_description = "Open the Track Bar key context menu or consume blank Track Bar RMB input"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.scene is not None
            and getattr(context.scene, "baw_trackbar_enabled", True)
        )

    def _finish_menu(self, context):
        handle = getattr(self, "_draw_handle", None)
        if handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
            self._draw_handle = None
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}

    def _main_layout(self):
        rows = ["ADD_KEY"]
        if self._has_key:
            rows.extend(("INTERPOLATION", "HANDLES"))
        height = (
            _KEY_MENU_PADDING * 2.0
            + _KEY_MENU_TITLE_HEIGHT
            + len(rows) * _KEY_MENU_ROW_HEIGHT
        )
        x = self._menu_x
        y = self._menu_y
        top = y + height - _KEY_MENU_PADDING - _KEY_MENU_TITLE_HEIGHT
        rects = {}
        for index, row in enumerate(rows):
            row_top = top - index * _KEY_MENU_ROW_HEIGHT
            rects[row] = (
                x + _KEY_MENU_PADDING,
                row_top - _KEY_MENU_ROW_HEIGHT,
                x + _KEY_MENU_WIDTH - _KEY_MENU_PADDING,
                row_top,
            )
        return (x, y, x + _KEY_MENU_WIDTH, y + height), rects

    def _submenu_layout(self):
        if self._submenu == "INTERPOLATION":
            items = (
                ("BEZIER", "Bezier"),
                ("LINEAR", "Linear"),
                ("CONSTANT", "Constant"),
            )
            anchor = self._main_rows.get("INTERPOLATION")
        elif self._submenu == "HANDLES":
            items = (
                ("AUTO_CLAMPED", "Auto Clamped"),
                ("AUTO", "Auto"),
                ("VECTOR", "Vector"),
                ("ALIGNED", "Aligned"),
                ("FREE", "Free"),
            )
            anchor = self._main_rows.get("HANDLES")
        else:
            return None, (), {}
        if anchor is None:
            return None, (), {}

        height = _KEY_MENU_PADDING * 2.0 + len(items) * _KEY_MENU_ROW_HEIGHT
        right_x = self._main_rect[2] + 4.0
        if right_x + _KEY_MENU_SUB_WIDTH <= self._region_width - 4.0:
            x = right_x
        else:
            x = max(4.0, self._main_rect[0] - _KEY_MENU_SUB_WIDTH - 4.0)
        top = min(self._region_height - 4.0, anchor[3] + _KEY_MENU_PADDING)
        y = max(4.0, top - height)
        rects = {}
        row_top = y + height - _KEY_MENU_PADDING
        for index, (value, _label) in enumerate(items):
            item_top = row_top - index * _KEY_MENU_ROW_HEIGHT
            rects[value] = (
                x + _KEY_MENU_PADDING,
                item_top - _KEY_MENU_ROW_HEIGHT,
                x + _KEY_MENU_SUB_WIDTH - _KEY_MENU_PADDING,
                item_top,
            )
        return (x, y, x + _KEY_MENU_SUB_WIDTH, y + height), items, rects

    def _draw_menu(self):
        gpu.state.blend_set("ALPHA")
        try:
            self._main_rect, self._main_rows = self._main_layout()
            x0, y0, x1, y1 = self._main_rect
            _draw_key_menu_rect(x0, y0, x1 - x0, y1 - y0, _KEY_MENU_BORDER)
            _draw_key_menu_rect(x0 + 1.0, y0 + 1.0, x1 - x0 - 2.0, y1 - y0 - 2.0, _KEY_MENU_BG)
            _draw_key_menu_text(
                x0 + _KEY_MENU_PADDING + 4.0,
                y1 - _KEY_MENU_PADDING - 17.0,
                "Key Properties",
            )

            for row, label in (
                ("ADD_KEY", "Add Key"),
                ("INTERPOLATION", "Interpolation"),
                ("HANDLES", "Handles"),
            ):
                rect = self._main_rows.get(row)
                if rect is None:
                    continue
                if self._hover_main == row:
                    _draw_key_menu_rect(
                        rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1], _KEY_MENU_HOVER
                    )
                _draw_key_menu_text(rect[0] + 8.0, rect[1] + 6.0, label)
                if row in {"INTERPOLATION", "HANDLES"}:
                    _draw_key_menu_text(rect[2] - 14.0, rect[1] + 6.0, ">", _KEY_MENU_DIM)

            self._submenu_rect, items, self._submenu_rows = self._submenu_layout()
            if self._submenu_rect is None:
                return
            sx0, sy0, sx1, sy1 = self._submenu_rect
            _draw_key_menu_rect(sx0, sy0, sx1 - sx0, sy1 - sy0, _KEY_MENU_BORDER)
            _draw_key_menu_rect(
                sx0 + 1.0, sy0 + 1.0, sx1 - sx0 - 2.0, sy1 - sy0 - 2.0, _KEY_MENU_BG
            )
            for value, label in items:
                rect = self._submenu_rows[value]
                if self._hover_sub == value:
                    _draw_key_menu_rect(
                        rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1], _KEY_MENU_HOVER
                    )
                current = (
                    self._interpolation_state
                    if self._submenu == "INTERPOLATION"
                    else self._handle_state
                )
                if current == value:
                    _draw_key_menu_text(rect[0] + 6.0, rect[1] + 6.0, "✓")
                    text_x = rect[0] + 24.0
                else:
                    text_x = rect[0] + 8.0
                _draw_key_menu_text(text_x, rect[1] + 6.0, label)
        finally:
            gpu.state.blend_set("NONE")

    def invoke(self, context, event):
        global _TRACKBAR_KEY_CONTEXT_FRAME

        region = context.region
        scene = context.scene
        if region is None or scene is None:
            return {"PASS_THROUGH"}

        x_min, x_max, y_min, y_max = trackbar_bounds(region)
        if not (
            x_min <= event.mouse_region_x <= x_max
            and y_min <= event.mouse_region_y <= y_max
        ):
            return {"PASS_THROUGH"}

        visible_start, visible_end, _count, key_x_min, key_x_max = visible_frame_window(
            scene,
            region,
        )
        in_key_row = (
            key_x_min <= event.mouse_region_x <= key_x_max
            and KEY_MARKER_BOTTOM
            <= event.mouse_region_y
            <= KEY_MARKER_BOTTOM + KEY_MARKER_HEIGHT
        )
        if not in_key_row:
            return {"FINISHED"}

        visible_count = max(1, visible_end - visible_start + 1)
        frame = x_to_cell_frame(
            event.mouse_region_x,
            visible_start,
            key_x_min,
            frame_cell_width(scene, region),
            visible_count,
        )
        if frame is None:
            return {"FINISHED"}

        frame = int(frame)
        _TRACKBAR_KEY_CONTEXT_FRAME = frame
        targets = key_property_targets_for_context(context, hit_cell=frame)
        state = key_property_state_for_targets(targets) if targets else None
        self._frame = frame
        self._has_key = bool(targets)
        self._interpolation_state = state.interpolation_state if state else "EMPTY"
        self._handle_state = state.handle_state if state else "N/A"
        self._region_width = float(region.width)
        self._region_height = float(region.height)
        self._menu_x = min(
            max(4.0, float(event.mouse_region_x) - 18.0),
            max(4.0, float(region.width) - _KEY_MENU_WIDTH - 4.0),
        )
        self._menu_y = float(TRACKBAR_HEIGHT) + _KEY_MENU_TRACK_GAP
        self._submenu = None
        self._hover_main = None
        self._hover_sub = None
        self._main_rect = None
        self._main_rows = {}
        self._submenu_rect = None
        self._submenu_rows = {}
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_menu,
            (),
            "WINDOW",
            "POST_PIXEL",
        )
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if context.area is None:
            return self._finish_menu(context)

        x = float(event.mouse_region_x)
        y = float(event.mouse_region_y)
        if event.type == "MOUSEMOVE":
            self._hover_main = None
            self._hover_sub = None
            for row, rect in self._main_rows.items():
                if _point_in_key_menu_rect(x, y, rect):
                    self._hover_main = row
                    break
            if self._hover_main == "INTERPOLATION":
                self._submenu = "INTERPOLATION"
            elif self._hover_main == "HANDLES":
                self._submenu = "HANDLES"
            elif self._hover_main == "ADD_KEY":
                self._submenu = None

            if self._submenu_rect is not None and _point_in_key_menu_rect(
                x, y, self._submenu_rect
            ):
                for value, rect in self._submenu_rows.items():
                    if _point_in_key_menu_rect(x, y, rect):
                        self._hover_sub = value
                        break
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type in {"ESC"} or (event.type == "RIGHTMOUSE" and event.value == "PRESS"):
            return self._finish_menu(context)

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            for row, rect in self._main_rows.items():
                if not _point_in_key_menu_rect(x, y, rect):
                    continue
                if row == "ADD_KEY":
                    bpy.ops.baw.trackbar_add_key(hit_cell=self._frame)
                    return self._finish_menu(context)
                if row == "INTERPOLATION":
                    self._submenu = "INTERPOLATION"
                    context.area.tag_redraw()
                    return {"RUNNING_MODAL"}
                if row == "HANDLES":
                    self._submenu = "HANDLES"
                    context.area.tag_redraw()
                    return {"RUNNING_MODAL"}

            if self._submenu_rect is not None and _point_in_key_menu_rect(
                x, y, self._submenu_rect
            ):
                for value, rect in self._submenu_rows.items():
                    if not _point_in_key_menu_rect(x, y, rect):
                        continue
                    property_kind = (
                        "INTERPOLATION" if self._submenu == "INTERPOLATION" else "HANDLE"
                    )
                    bpy.ops.baw.trackbar_key_properties(
                        hit_cell=self._frame,
                        property_kind=property_kind,
                        value=value,
                    )
                    return self._finish_menu(context)

            region = context.region
            scene = context.scene
            if region is not None and scene is not None:
                visible_start, visible_end, _count, key_x_min, key_x_max = visible_frame_window(
                    scene,
                    region,
                )
                in_trackbar_rail = (
                    key_x_min <= x <= key_x_max
                    and 0.0 <= y <= TRACKBAR_HEIGHT
                )
                control_rects = frame_control_bounds(scene, region)
                over_frame_control = bool(
                    control_rects
                    and any(_point_in_key_menu_rect(x, y, rect) for rect in control_rects.values())
                )
                endpoint_rects = transport_endpoint_button_bounds(region)
                over_endpoint = bool(
                    endpoint_rects
                    and any(_point_in_key_menu_rect(x, y, rect) for rect in endpoint_rects.values())
                )
                over_fixed_button = (
                    over_frame_control
                    or over_endpoint
                    or _point_in_key_menu_rect(x, y, key_mode_button_bounds(region))
                    or _point_in_key_menu_rect(x, y, auto_key_button_bounds(region))
                )
                if in_trackbar_rail and not over_fixed_button:
                    visible_count = max(1, visible_end - visible_start + 1)
                    frame = x_to_cell_frame(
                        x,
                        visible_start,
                        key_x_min,
                        frame_cell_width(scene, region),
                        visible_count,
                    )
                    if frame is not None:
                        frame = int(frame)
                        scene.frame_set(frame)
                        has_selected_key = select_key_frame_for_context(
                            context,
                            frame,
                            mode="SET",
                        )
                        scene.baw_has_selected_key = bool(has_selected_key)
                        if has_selected_key:
                            scene.baw_selected_key_frame = frame
                        sync_active_trajectory_key_selection(context)
                        refresh_active_trajectory_live(context)

            return self._finish_menu(context)

        return {"RUNNING_MODAL"}


class BAW_OT_trackbar_add_key(bpy.types.Operator):
    bl_idname = "baw.trackbar_add_key"
    bl_label = "Add Key"
    bl_description = "Set an AWB transform key at the clicked Track Bar frame"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    hit_cell: bpy.props.IntProperty(options={"HIDDEN"}, default=0)

    def execute(self, context):
        scene = context.scene
        frame = int(self.hit_cell)
        scene.frame_set(frame)
        result = bpy.ops.baw.set_key()
        if result != {"FINISHED"}:
            return result
        select_key_frame_for_context(context, frame, mode="SET")
        sync_active_trajectory_key_selection(context)
        refresh_active_trajectory_live(context, force=True, exact_range=True)
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_trackbar_key_properties(bpy.types.Operator):
    bl_idname = "baw.trackbar_key_properties"
    bl_label = "Set Track Bar Key Property"
    bl_description = "Apply one native Blender key property to the Track Bar key context"
    bl_options: ClassVar[set[str]] = {"REGISTER", "UNDO", "INTERNAL"}

    hit_cell: bpy.props.IntProperty(options={"HIDDEN"}, default=0)
    property_kind: bpy.props.EnumProperty(
        options={"HIDDEN"},
        items=(
            ("INTERPOLATION", "Interpolation", "Set key interpolation"),
            ("HANDLE", "Handle", "Set both Bezier handle types"),
        ),
        default="INTERPOLATION",
    )
    value: bpy.props.StringProperty(options={"HIDDEN"}, default="")

    def execute(self, context):
        targets = key_property_targets_for_context(context, hit_cell=int(self.hit_cell))
        if not targets:
            return {"CANCELLED"}

        interpolation = None
        handle_type = None
        if self.property_kind == "INTERPOLATION":
            if self.value not in {"CONSTANT", "LINEAR", "BEZIER"}:
                return {"CANCELLED"}
            interpolation = self.value
        elif self.property_kind == "HANDLE":
            if self.value not in {"AUTO", "AUTO_CLAMPED", "VECTOR", "ALIGNED", "FREE"}:
                return {"CANCELLED"}
            handle_type = self.value
        else:
            return {"CANCELLED"}

        try:
            result = apply_key_properties_for_context(
                context,
                targets,
                interpolation=interpolation,
                handle_type=handle_type,
            )
        except (RuntimeError, ValueError) as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}

        if not result.changed_points:
            return {"CANCELLED"}

        sync_active_trajectory_key_selection(context)
        refresh_active_trajectory_live(context, force=True, exact_range=True)
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_MT_trackbar_key_context(bpy.types.Menu):
    bl_idname = "BAW_MT_trackbar_key_context"
    bl_label = "Key Properties"

    def draw(self, context):
        layout = self.layout
        frame = _TRACKBAR_KEY_CONTEXT_FRAME
        if frame is None:
            layout.label(text="No frame")
            return

        add_key = layout.operator(
            BAW_OT_trackbar_add_key.bl_idname,
            text="Add Key",
            icon="KEY_HLT",
        )
        add_key.hit_cell = int(frame)

        targets = key_property_targets_for_context(context, hit_cell=int(frame))
        if not targets:
            return

        layout.separator()
        layout.menu(BAW_MT_trackbar_interpolation.bl_idname, text="Interpolation")
        layout.menu(BAW_MT_trackbar_handles.bl_idname, text="Handles")


class BAW_MT_trackbar_interpolation(bpy.types.Menu):
    bl_idname = "BAW_MT_trackbar_interpolation"
    bl_label = "Interpolation"

    def draw(self, context):
        frame = _TRACKBAR_KEY_CONTEXT_FRAME
        if frame is None:
            self.layout.label(text="No key")
            return
        state = key_property_state_for_targets(
            key_property_targets_for_context(context, hit_cell=int(frame))
        )
        for value, label in (
            ("BEZIER", "Bezier"),
            ("LINEAR", "Linear"),
            ("CONSTANT", "Constant"),
        ):
            op = self.layout.operator(
                BAW_OT_trackbar_key_properties.bl_idname,
                text=label,
                icon="CHECKMARK" if state.interpolation_state == value else "NONE",
            )
            op.hit_cell = int(frame)
            op.property_kind = "INTERPOLATION"
            op.value = value


class BAW_MT_trackbar_handles(bpy.types.Menu):
    bl_idname = "BAW_MT_trackbar_handles"
    bl_label = "Handles"

    def draw(self, context):
        frame = _TRACKBAR_KEY_CONTEXT_FRAME
        if frame is None:
            self.layout.label(text="No key")
            return
        state = key_property_state_for_targets(
            key_property_targets_for_context(context, hit_cell=int(frame))
        )
        for value, label in (
            ("AUTO_CLAMPED", "Auto Clamped"),
            ("AUTO", "Auto"),
            ("VECTOR", "Vector"),
            ("ALIGNED", "Aligned"),
            ("FREE", "Free"),
        ):
            op = self.layout.operator(
                BAW_OT_trackbar_key_properties.bl_idname,
                text=label,
                icon="CHECKMARK" if state.handle_state == value else "NONE",
            )
            op.hit_cell = int(frame)
            op.property_kind = "HANDLE"
            op.value = value


class BAW_OT_block_empty_trackbar_keyboard(bpy.types.Operator):
    bl_idname = "baw.block_empty_trackbar_keyboard"
    bl_label = "Block Empty Track Bar Keyboard Input"
    bl_description = (
        "Consume viewport keyboard shortcuts while the pointer is inside an empty AWB Track Bar"
    )
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.scene is not None
            and getattr(context.scene, "baw_trackbar_enabled", True)
        )

    def invoke(self, context, event):
        region = context.region
        if region is None:
            return {"PASS_THROUGH"}

        x_min, x_max, y_min, y_max = trackbar_bounds(region)
        if not (
            x_min <= event.mouse_region_x <= x_max
            and y_min <= event.mouse_region_y <= y_max
        ):
            return {"PASS_THROUGH"}

        if trackbar_controls_for_context(context):
            return {"PASS_THROUGH"}

        # The empty Track Bar owns its keyboard surface. Future AWB shortcuts
        # are intentionally added ahead of this fallback one by one; until then
        # no Blender viewport shortcut/menu may leak through from this region.
        return {"FINISHED"}


class BAW_OT_box_select_trackbar(bpy.types.Operator):
    bl_idname = "baw.box_select_trackbar"
    bl_label = "Box Select Track Bar Keys"
    bl_description = "Drag from an empty Track Bar key cell to select a range of keys"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.scene is not None
            and getattr(context.scene, "baw_trackbar_enabled", True)
        )

    @staticmethod
    def _frame_from_mouse(context, mouse_x):
        scene = context.scene
        region = context.region
        if scene is None or region is None:
            return None
        visible_start, visible_end, _count, x_min, _x_max = visible_frame_window(
            scene,
            region,
        )
        visible_count = max(1, visible_end - visible_start + 1)
        return x_to_cell_frame(
            mouse_x,
            visible_start,
            x_min,
            frame_cell_width(scene, region),
            visible_count,
        )

    @staticmethod
    def _sync_legacy_marker(context, preferred_frame: int | None = None) -> None:
        scene = context.scene
        selected_frames = [round(frame) for frame in selected_key_frames_for_context(context)]
        scene.baw_has_selected_key = bool(selected_frames)
        if selected_frames:
            if preferred_frame is not None and preferred_frame in selected_frames:
                scene.baw_selected_key_frame = preferred_frame
            else:
                scene.baw_selected_key_frame = selected_frames[-1]
        sync_active_trajectory_key_selection(context)

    def invoke(self, context, event):
        region = context.region
        scene = context.scene
        if region is None or scene is None:
            return {"PASS_THROUGH"}

        _start, _end, _count, _x_min, _x_max = visible_frame_window(scene, region)
        # Keep the padded gutters at both Track Bar ends box-selectable. This
        # gives frame-start/frame-end keys an empty drag origin even when the
        # endpoint cell itself is keyed and therefore owned by key Move.
        in_key_row = (
            0.0 <= event.mouse_region_x <= float(region.width)
            and KEY_MARKER_BOTTOM
            <= event.mouse_region_y
            <= KEY_MARKER_BOTTOM + KEY_MARKER_HEIGHT
        )
        if not in_key_row:
            return {"PASS_THROUGH"}

        frame = self._frame_from_mouse(context, event.mouse_region_x)
        if frame is None or has_key_at_frame(context, frame):
            # Keyed cells stay owned by the existing gizmo so single-key
            # selection, Move, and Shift+Drag Clone do not change behavior.
            return {"PASS_THROUGH"}

        # Empty key-rail clicks must move current time immediately on press.
        # If the user keeps dragging, this same interaction can still become
        # a box-select; a plain click therefore never feels swallowed.
        scene.frame_set(int(frame))
        if region is not None:
            region.tag_redraw()

        self._start_frame = int(frame)
        self._current_frame = int(frame)
        self._start_x = float(event.mouse_region_x)
        self._current_x = float(event.mouse_region_x)
        self._has_moved = False
        if event.shift:
            self._selection_mode = "ADD"
        elif event.ctrl:
            self._selection_mode = "TOGGLE"
        else:
            self._selection_mode = "SET"

        clear_box_selection_overlay()
        if context.window is not None:
            context.window.cursor_modal_set("CROSSHAIR")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        scene = context.scene
        region = context.region

        if event.type == "MOUSEMOVE":
            if region is None:
                return {"RUNNING_MODAL"}
            _start, _end, _count, x_min, x_max = visible_frame_window(scene, region)
            mouse_x = max(x_min, min(x_max, float(event.mouse_region_x)))
            frame = self._frame_from_mouse(context, mouse_x)
            if frame is not None:
                self._current_x = mouse_x
                self._current_frame = int(frame)
                if (
                    self._current_frame != self._start_frame
                    or abs(self._current_x - self._start_x) >= 4.0
                ):
                    self._has_moved = True
                    set_box_selection_overlay(self._start_x, self._current_x)
                    region.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self._has_moved:
                select_key_frame_range_for_context(
                    context,
                    self._start_frame,
                    self._current_frame,
                    mode=self._selection_mode,
                )
            else:
                # Plain empty-cell click keeps the existing Track Bar behavior:
                # move current time there and clear key selection. Modified
                # empty clicks only move time and preserve the current selection.
                scene.frame_set(self._start_frame)
                if self._selection_mode == "SET":
                    select_key_frame_for_context(context, self._start_frame, mode="SET")

            self._sync_legacy_marker(context)
            clear_box_selection_overlay()
            if context.window is not None:
                context.window.cursor_modal_restore()
            if region is not None:
                region.tag_redraw()
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            clear_box_selection_overlay()
            if context.window is not None:
                context.window.cursor_modal_restore()
            if region is not None:
                region.tag_redraw()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_OT_delete_trackbar_key(bpy.types.Operator):
    bl_idname = "baw.delete_trackbar_key"
    bl_label = "Delete Track Bar Key"
    bl_description = "Delete the selected frame cell of the active control"
    bl_options: ClassVar[set[str]] = {"UNDO", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None and context.area.type == "VIEW_3D"
            and context.region is not None and context.region.type == "WINDOW"
            and context.scene is not None
            and getattr(context.scene, "baw_trackbar_enabled", True)
        )

    def invoke(self, context, event):
        x_min, x_max, y_min, y_max = trackbar_bounds(context.region)
        inside_trackbar = (
            x_min <= event.mouse_region_x <= x_max
            and y_min <= event.mouse_region_y <= y_max
        )
        # Once Track Bar keys are selected, Delete/X own that key selection even
        # when the cursor is over the viewport object. This prevents Blender's
        # Object Delete from leaking through while the user is editing keys.
        has_selected_keys = bool(selected_key_frames_for_context(context))
        if not inside_trackbar and not has_selected_keys:
            return {"PASS_THROUGH"}
        if bool(getattr(context.scene, "baw_trajectory_edit_mode", False)) and not inside_trackbar:
            # Trajectory edit has a stricter Position-only delete operator.
            return {"PASS_THROUGH"}
        return self.execute(context)

    def execute(self, context):
        deleted = delete_selected_keys_for_context(context)
        if deleted:
            sync_active_trajectory_key_selection(context)
            refresh_active_trajectory_live(context, force=True, exact_range=True)
        context.area.tag_redraw()
        # Never let Delete inside the Track Bar fall through to object deletion.
        return {"FINISHED"} if deleted else {"CANCELLED"}


class BAW_OT_fit_trackbar_keys(bpy.types.Operator):
    bl_idname = "baw.fit_trackbar_keys"
    bl_label = "Fit Track Bar to Keys"
    bl_description = "Fit the visible Track Bar range to the active object's or pose bone's keyed frames"
    bl_options: ClassVar[set[str]] = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.scene is not None
            and getattr(context.scene, "baw_trackbar_enabled", True)
        )

    def invoke(self, context, event):
        region = context.region
        scene = context.scene
        if region is None or scene is None:
            return {"PASS_THROUGH"}

        x_min, x_max, _y_min, _y_max = trackbar_bounds(region)
        if not (
            x_min <= event.mouse_region_x <= x_max
            and 0.0 <= event.mouse_region_y <= TRACKBAR_HEIGHT
        ):
            return {"PASS_THROUGH"}

        keyed_frames = sorted(round(frame) for frame in key_channels_for_context(context))
        if keyed_frames:
            first = keyed_frames[0]
            last = keyed_frames[-1]
            if first == last:
                span = 5
                center = float(first)
            else:
                span = last - first + 1
                center = (first + last) * 0.5
            scene.baw_trackbar_view_center = center
            scene.baw_trackbar_view_span = max(2, span)
            if context.area is not None:
                context.area.tag_redraw()

        # Consume the double-click even when no keys exist so viewport tools
        # never receive mouse input originating inside the Track Bar.
        return {"FINISHED"}


class BAW_GT_trackbar(bpy.types.Gizmo):
    bl_idname = "BAW_GT_trackbar"

    def setup(self):
        self.use_draw_hover = False
        self.use_draw_modal = False
        self.use_event_handle_all = False
        self.use_grab_cursor = False
        self._initial_frame = 1
        self._drag_initial_mouse_x = 0
        self._drag_cell_width = 1.0
        self._dragging_frame_control = False
        self._dragging_frame_rail = False
        self._dragging_range_resize = False
        self._dragging_selection_range = False
        self._selection_range_mode: str | None = None
        self._selection_range_source_frames: list[float] = []
        self._selection_range_anchor_frame = 0
        self._selection_range_current_frame = 0
        self._selection_range_pivot_frame = 0
        self._selection_range_source_handle_frame = 0
        self._selection_range_initial_frame = 0
        self._selection_range_initial_snapshot = []
        self._selection_range_preview_transaction = None
        self._selection_range_has_moved = False
        self._dragging_key = False
        self._key_drag_clone = False
        self._key_drag_source_frame = 0
        self._key_drag_current_frame = 0
        self._key_drag_source_frames: list[float] = []
        self._key_drag_initial_snapshot = []
        self._key_drag_preview_transaction = None
        self._key_drag_has_moved = False
        self._key_drag_group = False
        self._key_drag_plain_preserved_selection = False
        self._key_drag_replaced_snapshots = []
        self._key_drag_selection_mode = "SET"
        self._blocking_track_click = False

    def draw(self, context):
        # Drawing is handled by a persistent SpaceView3D draw handler.
        # This gizmo is interaction-only so the Track Bar stays visible
        # even when the mouse is elsewhere in the viewport.
        pass

    def draw_select(self, context, select_id):
        # Selection uses test_select(); no off-screen selection geometry is needed.
        pass

    @staticmethod
    def _point_in_rect(x: float, y: float, rect) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _scene_time(scene) -> float:
        return float(scene.frame_current) + float(getattr(scene, "frame_subframe", 0.0))

    @staticmethod
    def _set_scene_time(scene, frame_time: float) -> None:
        base_frame = math.floor(float(frame_time))
        subframe = float(frame_time) - float(base_frame)
        scene.frame_set(int(base_frame), subframe=subframe)

    @staticmethod
    def _delete_selected_keys_from_modal(context) -> bool:
        """Delete Track Bar selection even while the click gizmo owns modal input."""
        deleted = delete_selected_keys_for_context(context)
        if deleted:
            sync_active_trajectory_key_selection(context)
            refresh_active_trajectory_live(context, force=True, exact_range=True)
        if context.area is not None:
            context.area.tag_redraw()
        return deleted

    def _step_frame(self, context, event, delta: int, control_name: str) -> None:
        scene = context.scene
        region = context.region
        old_controls = frame_control_bounds(scene, region) if region is not None else None
        old_time = self._scene_time(scene)

        if bool(getattr(scene, "baw_key_mode", False)):
            target = adjacent_key_frame_for_context(
                context,
                old_time,
                next=delta > 0,
            )
            if target is None:
                if region is not None:
                    region.tag_redraw()
                return
            self._set_scene_time(scene, target)
        else:
            old_frame = int(scene.frame_current)
            frame = max(int(scene.frame_start), min(int(scene.frame_end), old_frame + delta))
            scene.frame_set(frame)

        if region is not None:
            region.tag_redraw()

        if (
            context.window is None
            or region is None
            or old_controls is None
            or abs(self._scene_time(scene) - old_time) <= 1e-9
        ):
            return

        new_controls = frame_control_bounds(scene, region)
        if new_controls is None:
            return
        old_rect = old_controls[control_name]
        new_rect = new_controls[control_name]
        old_cx = (old_rect[0] + old_rect[2]) * 0.5
        old_cy = (old_rect[1] + old_rect[3]) * 0.5
        new_cx = (new_rect[0] + new_rect[2]) * 0.5
        new_cy = (new_rect[1] + new_rect[3]) * 0.5
        context.window.cursor_warp(
            round(event.mouse_x + (new_cx - old_cx)),
            round(event.mouse_y + (new_cy - old_cy)),
        )

    def test_select(self, context, location):
        region = context.region
        scene = context.scene
        if region is None or scene is None:
            return -1
        _start, _end, _frame_count, x_min, track_x_max = visible_frame_window(scene, region)
        _x_min, _x_max, y_min, _y_max = trackbar_bounds(region)
        x, y = location

        in_lower_work_area = (
            x_min <= x <= track_x_max
            and y_min <= y < KEY_MARKER_BOTTOM
        )
        in_key_row = (
            x_min <= x <= track_x_max
            and KEY_MARKER_BOTTOM <= y <= KEY_MARKER_BOTTOM + KEY_MARKER_HEIGHT
        )
        in_frame_rail = (
            x_min <= x <= track_x_max
            and CURRENT_FRAME_CONTROL_Y_MIN <= y <= CURRENT_FRAME_CONTROL_Y_MAX
        )

        controls = frame_control_bounds(scene, region)
        in_controls = False
        if controls is not None:
            in_controls = any(
                self._point_in_rect(x, y, controls[name])
                for name in ("left", "value", "right")
            )

        # Only keyed cells belong to the single-key gizmo in the key row.
        # Empty key-row cells are deliberately left unhittable so the
        # dedicated box-select LMB operator can own them before viewport input.
        keyed_cell = False
        if in_key_row:
            frame = self._frame_from_mouse(context, x)
            keyed_cell = frame is not None and has_key_at_frame(context, frame)

        if in_lower_work_area or keyed_cell or in_frame_rail or in_controls:
            return 0
        return -1

    def _frame_from_mouse(self, context, mouse_x):
        scene = context.scene
        region = context.region
        if scene is None or region is None:
            return None

        visible_start, visible_end, _frame_count, x_min, _track_x_max = visible_frame_window(scene, region)
        visible_count = max(1, visible_end - visible_start + 1)
        cell_width = frame_cell_width(scene, region)
        return x_to_cell_frame(
            mouse_x,
            visible_start,
            x_min,
            cell_width,
            visible_count,
        )

    def _set_frame_from_mouse(self, context, mouse_x):
        frame = self._frame_from_mouse(context, mouse_x)
        if frame is None:
            return
        context.scene.frame_set(frame)
        if context.region is not None:
            context.region.tag_redraw()

    def _activate_frame_from_mouse(self, context, mouse_x):
        frame = self._frame_from_mouse(context, mouse_x)
        if frame is None:
            return False

        scene = context.scene
        scene.frame_set(frame)
        if has_key_at_frame(context, frame):
            select_key_frame_for_context(context, frame)
            scene.baw_has_selected_key = True
            scene.baw_selected_key_frame = frame
        else:
            select_key_frame_for_context(context, frame)
            scene.baw_has_selected_key = False

        sync_active_trajectory_key_selection(context)
        if context.region is not None:
            context.region.tag_redraw()
        return True

    def _sync_legacy_selected_key_marker(self, context, preferred_frame: int | None = None) -> None:
        scene = context.scene
        selected_frames = [round(frame) for frame in selected_key_frames_for_context(context)]
        scene.baw_has_selected_key = bool(selected_frames)
        if selected_frames:
            if preferred_frame is not None and preferred_frame in selected_frames:
                scene.baw_selected_key_frame = preferred_frame
            else:
                scene.baw_selected_key_frame = selected_frames[-1]
        sync_active_trajectory_key_selection(context)

    def _selection_range_hit_mode(self, context, x: float, y: float) -> str | None:
        region = context.region
        if region is None:
            return None
        selected_frames = selected_key_frames_for_context(context)
        bounds = selection_range_bounds(context.scene, region, selected_frames)
        if bounds is None:
            return None
        # Endpoint handles deliberately win over the overlapping center segment.
        if self._point_in_rect(x, y, bounds["left_handle"]):
            return "LEFT_HANDLE"
        if self._point_in_rect(x, y, bounds["right_handle"]):
            return "RIGHT_HANDLE"
        if self._point_in_rect(x, y, bounds["center"]):
            return "CENTER"
        return None

    def _start_selection_range_center(self, context, mouse_x: float) -> bool:
        source_frames = selected_key_frames_for_context(context)
        if len({round(frame) for frame in source_frames}) < 2:
            return False
        anchor_frame = self._frame_from_mouse(context, mouse_x)
        if anchor_frame is None:
            return False

        initial_snapshot = snapshot_active_key_state_for_context(context)
        preview_transaction = snapshot_multi_key_preview_for_context(
            context,
            list(source_frames),
            mode="MOVE",
            guard_operation="SELECTION_RANGE_MOVE",
        )
        if not initial_snapshot or preview_transaction is None:
            return False

        self._dragging_selection_range = True
        self._selection_range_mode = "CENTER"
        self._selection_range_source_frames = list(source_frames)
        self._selection_range_anchor_frame = int(anchor_frame)
        self._selection_range_current_frame = int(anchor_frame)
        self._selection_range_initial_frame = int(context.scene.frame_current)
        self._selection_range_initial_snapshot = initial_snapshot
        self._selection_range_preview_transaction = preview_transaction
        self._selection_range_has_moved = False
        if context.window is not None:
            context.window.cursor_modal_set("MOVE_X")
        if context.region is not None:
            context.region.tag_redraw()
        return True

    def _start_selection_range_scale(self, context, mode: str) -> bool:
        source_frames = selected_key_frames_for_context(context)
        rounded_frames = sorted({round(frame) for frame in source_frames})
        if len(rounded_frames) < 2 or mode not in {"LEFT_HANDLE", "RIGHT_HANDLE"}:
            return False

        left_frame = rounded_frames[0]
        right_frame = rounded_frames[-1]
        if mode == "LEFT_HANDLE":
            source_handle_frame = left_frame
            pivot_frame = right_frame
        else:
            source_handle_frame = right_frame
            pivot_frame = left_frame

        initial_snapshot = snapshot_active_key_state_for_context(context)
        preview_transaction = snapshot_selection_range_scale_for_context(
            context,
            list(source_frames),
            pivot_frame=pivot_frame,
            source_handle_frame=source_handle_frame,
        )
        if not initial_snapshot or preview_transaction is None:
            return False

        self._dragging_selection_range = True
        self._selection_range_mode = mode
        self._selection_range_source_frames = list(source_frames)
        self._selection_range_anchor_frame = int(source_handle_frame)
        self._selection_range_current_frame = int(source_handle_frame)
        self._selection_range_pivot_frame = int(pivot_frame)
        self._selection_range_source_handle_frame = int(source_handle_frame)
        self._selection_range_initial_frame = int(context.scene.frame_current)
        self._selection_range_initial_snapshot = initial_snapshot
        self._selection_range_preview_transaction = preview_transaction
        self._selection_range_has_moved = False
        if context.window is not None:
            context.window.cursor_modal_set("MOVE_X")
        if context.region is not None:
            context.region.tag_redraw()
        return True

    def _restore_selection_range_center(self, context, *, exact: bool) -> None:
        if self._selection_range_preview_transaction is not None:
            restore_multi_key_preview_for_context(
                context,
                self._selection_range_preview_transaction,
            )
        if exact and self._selection_range_initial_snapshot:
            restore_active_key_state_for_context(
                context,
                self._selection_range_initial_snapshot,
            )
        context.scene.frame_set(self._selection_range_initial_frame)
        self._sync_legacy_selected_key_marker(context)
        if context.region is not None:
            context.region.tag_redraw()

    def _update_selection_range_center(self, context, target_frame: int) -> bool:
        transaction = self._selection_range_preview_transaction
        if transaction is None:
            return False
        delta = int(target_frame - self._selection_range_anchor_frame)
        changed = update_multi_key_preview_for_context(
            context,
            transaction,
            delta,
        )
        if changed:
            self._selection_range_current_frame = int(target_frame)
            self._selection_range_has_moved = (
                self._selection_range_has_moved or delta != 0
            )
            context.scene.frame_set(self._selection_range_initial_frame)
            self._sync_legacy_selected_key_marker(context)
            refresh_active_trajectory_live(context)
            if context.region is not None:
                context.region.tag_redraw()
        return changed

    def _restore_selection_range_scale(self, context, *, exact: bool) -> None:
        if self._selection_range_preview_transaction is not None:
            restore_selection_range_scale_preview_for_context(
                context,
                self._selection_range_preview_transaction,
            )
        if exact and self._selection_range_initial_snapshot:
            restore_active_key_state_for_context(
                context,
                self._selection_range_initial_snapshot,
            )
        context.scene.frame_set(self._selection_range_initial_frame)
        self._sync_legacy_selected_key_marker(context)
        if context.region is not None:
            context.region.tag_redraw()

    def _update_selection_range_scale(self, context, target_frame: int) -> bool:
        transaction = self._selection_range_preview_transaction
        if transaction is None:
            return False
        changed = update_selection_range_scale_preview_for_context(
            context,
            transaction,
            int(target_frame),
        )
        if changed:
            self._selection_range_current_frame = int(target_frame)
            self._selection_range_has_moved = (
                self._selection_range_has_moved
                or int(target_frame) != self._selection_range_source_handle_frame
            )
            context.scene.frame_set(self._selection_range_initial_frame)
            self._sync_legacy_selected_key_marker(context)
            refresh_active_trajectory_live(context)
            if context.region is not None:
                context.region.tag_redraw()
        return changed

    def _clear_selection_range_drag(self, context) -> None:
        if context.window is not None:
            context.window.cursor_modal_restore()
        self._dragging_selection_range = False
        self._selection_range_mode = None
        self._selection_range_source_frames = []
        self._selection_range_anchor_frame = 0
        self._selection_range_current_frame = 0
        self._selection_range_pivot_frame = 0
        self._selection_range_source_handle_frame = 0
        self._selection_range_initial_frame = 0
        self._selection_range_initial_snapshot = []
        self._selection_range_preview_transaction = None
        self._selection_range_has_moved = False

    def _finish_selection_range_center(self, context, *, cancel: bool = False) -> None:
        delta = int(
            self._selection_range_current_frame - self._selection_range_anchor_frame
        )
        if cancel:
            self._restore_selection_range_center(context, exact=True)
        elif self._selection_range_has_moved and delta != 0:
            self._restore_selection_range_center(context, exact=True)
            bpy.ops.ed.undo_push(message="AWB Selection Range Move Start")
            committed = move_selected_key_frames_for_context(
                context,
                self._selection_range_source_frames,
                delta,
            )
            if committed:
                context.scene.frame_set(self._selection_range_initial_frame)
                self._sync_legacy_selected_key_marker(context)
                bpy.ops.ed.undo_push(message="AWB Selection Range Move")
        refresh_active_trajectory_live(context, force=True, exact_range=True)
        self._clear_selection_range_drag(context)

    def _finish_selection_range_scale(self, context, *, cancel: bool = False) -> None:
        target_frame = int(self._selection_range_current_frame)
        source_handle_frame = int(self._selection_range_source_handle_frame)
        pivot_frame = int(self._selection_range_pivot_frame)

        if cancel:
            self._restore_selection_range_scale(context, exact=True)
        elif self._selection_range_has_moved and target_frame != source_handle_frame:
            self._restore_selection_range_scale(context, exact=True)
            bpy.ops.ed.undo_push(message="AWB Selection Range Scale Start")
            committed = scale_selected_key_frames_for_context(
                context,
                self._selection_range_source_frames,
                pivot_frame=pivot_frame,
                source_handle_frame=source_handle_frame,
                target_handle_frame=target_frame,
            )
            if committed:
                context.scene.frame_set(self._selection_range_initial_frame)
                self._sync_legacy_selected_key_marker(context)
                bpy.ops.ed.undo_push(message="AWB Selection Range Scale")
        refresh_active_trajectory_live(context, force=True, exact_range=True)
        self._clear_selection_range_drag(context)

    def _restore_multi_key_drag(self, context, *, exact: bool = False) -> None:
        if self._key_drag_preview_transaction is not None:
            restore_multi_key_preview_for_context(
                context,
                self._key_drag_preview_transaction,
            )
        if exact and self._key_drag_initial_snapshot:
            restore_active_key_state_for_context(
                context,
                self._key_drag_initial_snapshot,
            )
        scene = context.scene
        scene.frame_set(self._key_drag_source_frame)
        self._sync_legacy_selected_key_marker(
            context,
            preferred_frame=self._key_drag_source_frame,
        )
        if context.region is not None:
            context.region.tag_redraw()

    def _apply_multi_key_drag_preview(self, context, target_frame: int) -> bool:
        source_frame = self._key_drag_source_frame
        delta = int(target_frame - source_frame)
        transaction = self._key_drag_preview_transaction
        if transaction is None:
            return False

        changed = update_multi_key_preview_for_context(
            context,
            transaction,
            delta,
        )
        if changed:
            self._key_drag_current_frame = target_frame
            self._key_drag_has_moved = self._key_drag_has_moved or target_frame != source_frame
            context.scene.frame_set(target_frame)
            self._sync_legacy_selected_key_marker(context, preferred_frame=target_frame)
            refresh_active_trajectory_live(context)
            if context.region is not None:
                context.region.tag_redraw()
        return changed

    def _finish_multi_key_drag(self, context, *, cancel: bool = False) -> None:
        source_frame = self._key_drag_source_frame
        target_frame = self._key_drag_current_frame

        if cancel:
            self._restore_multi_key_drag(context, exact=True)
        elif self._key_drag_has_moved and target_frame != source_frame:
            self._restore_multi_key_drag(context, exact=True)
            operation_name = "Multi Clone" if self._key_drag_clone else "Multi Move"
            bpy.ops.ed.undo_push(message=f"AWB Track Bar {operation_name} Start")
            delta = int(target_frame - source_frame)
            if self._key_drag_clone:
                committed = clone_selected_key_frames_for_context(
                    context,
                    self._key_drag_source_frames,
                    delta,
                )
            else:
                committed = move_selected_key_frames_for_context(
                    context,
                    self._key_drag_source_frames,
                    delta,
                )
            if committed:
                context.scene.frame_set(target_frame)
                self._sync_legacy_selected_key_marker(context, preferred_frame=target_frame)
                bpy.ops.ed.undo_push(message=f"AWB Track Bar {operation_name}")
        elif self._key_drag_plain_preserved_selection:
            select_key_frame_for_context(context, source_frame, mode="SET")
            self._sync_legacy_selected_key_marker(context, preferred_frame=source_frame)

        refresh_active_trajectory_live(context, force=True, exact_range=True)
        if context.window is not None:
            context.window.cursor_modal_restore()
        self._dragging_key = False
        self._key_drag_clone = False
        self._key_drag_group = False
        self._key_drag_plain_preserved_selection = False
        self._key_drag_source_frames = []
        self._key_drag_initial_snapshot = []
        self._key_drag_preview_transaction = None
        self._key_drag_has_moved = False
        self._key_drag_selection_mode = "SET"
        self._key_drag_replaced_snapshots = []

    def _restore_key_drag(self, context) -> None:
        scene = context.scene
        source_frame = self._key_drag_source_frame
        current_frame = self._key_drag_current_frame
        if current_frame != source_frame:
            if self._key_drag_clone:
                remove_cloned_key_frame_for_context(
                    context,
                    source_frame,
                    current_frame,
                )
            else:
                move_key_frame_for_context(context, current_frame, source_frame)
        restore_key_frame_snapshots_for_context(
            context,
            self._key_drag_replaced_snapshots,
        )
        scene.frame_set(source_frame)
        select_key_frame_for_context(context, source_frame)
        scene.baw_has_selected_key = True
        scene.baw_selected_key_frame = source_frame
        if context.region is not None:
            context.region.tag_redraw()

    def _finish_key_drag(self, context, *, cancel: bool = False) -> None:
        source_frame = self._key_drag_source_frame
        target_frame = self._key_drag_current_frame

        if cancel:
            self._restore_key_drag(context)
        elif target_frame != source_frame:
            # The gizmo owns only the live preview. Restore the exact pre-drag
            # state first, then store explicit pre/post snapshots in Blender's
            # global undo history. Cancel never reaches this commit path.
            self._restore_key_drag(context)
            operation_name = "Clone" if self._key_drag_clone else "Move"
            bpy.ops.ed.undo_push(message=f"AWB Track Bar {operation_name} Start")
            if self._key_drag_clone:
                committed = clone_key_frame_for_context(
                    context,
                    source_frame,
                    target_frame,
                )
            else:
                committed = move_key_frame_for_context(
                    context,
                    source_frame,
                    target_frame,
                )
            if committed:
                scene = context.scene
                scene.frame_set(target_frame)
                select_key_frame_for_context(context, target_frame)
                scene.baw_has_selected_key = True
                scene.baw_selected_key_frame = target_frame
                bpy.ops.ed.undo_push(message=f"AWB Track Bar {operation_name}")

        self._sync_legacy_selected_key_marker(context, preferred_frame=target_frame)
        refresh_active_trajectory_live(context, force=True, exact_range=True)
        if (
            not cancel
            and target_frame == source_frame
            and self._key_drag_selection_mode == "SET"
        ):
            # A plain click enters the same modal path as a potential key drag.
            # When no frame-to-frame movement occurred, commit the click
            # selection explicitly on release. This also collapses a preserved
            # multi-selection to the clicked key, matching the intended click
            # grammar, while Shift/Ctrl selection modes remain untouched.
            select_key_frame_for_context(context, source_frame, mode="SET")
            self._sync_legacy_selected_key_marker(
                context,
                preferred_frame=source_frame,
            )
        if context.window is not None:
            context.window.cursor_modal_restore()
        self._dragging_key = False
        self._key_drag_clone = False
        self._key_drag_selection_mode = "SET"
        self._key_drag_replaced_snapshots = []

    def invoke(self, context, event):
        if event.type != "LEFTMOUSE":
            return {"PASS_THROUGH"}

        scene = context.scene
        self._initial_frame = scene.frame_current

        region = context.region
        if region is not None and event.alt:
            visible_start, visible_end, visible_span, x_min, x_max = visible_frame_window(
                scene,
                region,
            )
            in_lower_ruler = (
                x_min <= event.mouse_region_x <= x_max
                and RULER_RESIZE_Y_MIN <= event.mouse_region_y <= RULER_RESIZE_Y_MAX
            )
            if in_lower_ruler:
                midpoint = (x_min + x_max) * 0.5
                self._range_resize_side = (
                    "START" if event.mouse_region_x <= midpoint else "END"
                )
                self._range_visible_start = int(visible_start)
                self._range_visible_end = int(visible_end)
                self._range_initial_mouse_x = int(event.mouse_region_x)
                self._range_cell_width = max(0.01, frame_cell_width(scene, region))
                self._range_view_center_prop = float(scene.baw_trackbar_view_center)
                self._range_view_span_prop = int(scene.baw_trackbar_view_span)

                # Use the same independent view model as MMB pan and wheel zoom.
                scene.baw_trackbar_view_center = (visible_start + visible_end) * 0.5
                scene.baw_trackbar_view_span = int(visible_span)
                self._dragging_range_resize = True
                if context.window is not None:
                    context.window.cursor_modal_set("MOVE_X")
                return {"RUNNING_MODAL"}

        if context.region is not None:
            x = float(event.mouse_region_x)
            y = float(event.mouse_region_y)

            endpoint_bounds = transport_endpoint_button_bounds(context.region)
            if endpoint_bounds:
                if self._point_in_rect(x, y, endpoint_bounds["start"]):
                    scene.frame_set(int(scene.frame_start))
                    self._blocking_track_click = True
                    context.region.tag_redraw()
                    return {"RUNNING_MODAL"}
                if self._point_in_rect(x, y, endpoint_bounds["end"]):
                    scene.frame_set(int(scene.frame_end))
                    self._blocking_track_click = True
                    context.region.tag_redraw()
                    return {"RUNNING_MODAL"}

            key_bounds = key_mode_button_bounds(context.region)
            if key_bounds is not None and self._point_in_rect(x, y, key_bounds):
                scene.baw_key_mode = not bool(scene.baw_key_mode)
                self._blocking_track_click = True
                context.region.tag_redraw()
                return {"RUNNING_MODAL"}

            auto_bounds = auto_key_button_bounds(context.region)
            if self._point_in_rect(x, y, auto_bounds):
                bpy.ops.baw.toggle_auto_key()
                self._blocking_track_click = True
                context.region.tag_redraw()
                return {"RUNNING_MODAL"}

        controls = frame_control_bounds(scene, context.region) if context.region is not None else None
        if controls is not None:
            x = float(event.mouse_region_x)
            y = float(event.mouse_region_y)
            if self._point_in_rect(x, y, controls["left"]):
                self._step_frame(context, event, -1, "left")
                self._blocking_track_click = True
                return {"RUNNING_MODAL"}
            if self._point_in_rect(x, y, controls["right"]):
                self._step_frame(context, event, 1, "right")
                self._blocking_track_click = True
                return {"RUNNING_MODAL"}
            if self._point_in_rect(x, y, controls["value"]):
                self._drag_initial_mouse_x = int(event.mouse_region_x)
                # Keep the current-frame control spatially locked to the visible
                # Track Bar cells: one cell of mouse travel advances one frame.
                # A fixed px/frame value makes the control outrun the pointer when
                # the view is zoomed in (for example 0..10), which feels like
                # acceleration even though the numeric frame delta is linear.
                self._drag_cell_width = max(1.0, frame_cell_width(scene, context.region))
                self._dragging_frame_control = True
                if context.window_manager is not None:
                    context.window_manager.baw_trackbar_frame_drag_active = True
                trace_scrub_begin(context, source="FRAME_CONTROL")
                if context.window is not None:
                    context.window.cursor_modal_set("MOVE_X")
                return {"RUNNING_MODAL"}

        if context.region is not None:
            _start, _end, _frame_count, x_min, track_x_max = visible_frame_window(
                scene,
                context.region,
            )
            in_frame_rail = (
                x_min <= event.mouse_region_x <= track_x_max
                and CURRENT_FRAME_CONTROL_Y_MIN
                <= event.mouse_region_y
                <= CURRENT_FRAME_CONTROL_Y_MAX
            )
            if in_frame_rail:
                self._dragging_frame_rail = True
                if context.window_manager is not None:
                    context.window_manager.baw_trackbar_frame_drag_active = True
                trace_scrub_begin(context, source="FRAME_RAIL")
                if not self._activate_frame_from_mouse(context, event.mouse_region_x):
                    self._dragging_frame_rail = False
                    if context.window_manager is not None:
                        context.window_manager.baw_trackbar_frame_drag_active = False
                    trace_scrub_end(context, cancelled=True)
                    return {"CANCELLED"}
                return {"RUNNING_MODAL"}

            if not (
                x_min <= event.mouse_region_x <= track_x_max
                and 0.0 <= event.mouse_region_y <= TRACKBAR_HEIGHT
            ):
                return {"PASS_THROUGH"}

            range_hit = None
            if not (event.alt or event.ctrl or event.shift):
                range_hit = self._selection_range_hit_mode(
                    context,
                    float(event.mouse_region_x),
                    float(event.mouse_region_y),
                )
                if range_hit == "CENTER":
                    if self._start_selection_range_center(
                        context,
                        float(event.mouse_region_x),
                    ):
                        return {"RUNNING_MODAL"}
                elif range_hit in {"LEFT_HANDLE", "RIGHT_HANDLE"}:
                    if self._start_selection_range_scale(context, range_hit):
                        return {"RUNNING_MODAL"}
                    self._blocking_track_click = True
                    return {"RUNNING_MODAL"}

            # The lower rail is reserved for explicit AWB buttons. Blank space
            # there must never change current time or key selection. Selection
            # Range interactions above already returned, and fixed buttons are
            # handled earlier in this invoke path.
            if float(event.mouse_region_y) < KEY_MARKER_BOTTOM:
                self._blocking_track_click = True
                if context.region is not None:
                    context.region.tag_redraw()
                return {"RUNNING_MODAL"}

            frame = self._frame_from_mouse(context, event.mouse_region_x)
            in_key_area = (
                KEY_MARKER_BOTTOM
                <= event.mouse_region_y
                <= KEY_MARKER_BOTTOM + KEY_MARKER_HEIGHT
            )
            if frame is not None and in_key_area and has_key_at_frame(context, frame):
                selected_before = [
                    round(value) for value in selected_key_frames_for_context(context)
                ]
                hit_was_selected = frame in selected_before
                scene.frame_set(frame)

                if event.ctrl:
                    # Ctrl is selection-only. Do not start a drag transaction.
                    select_key_frame_for_context(context, frame, mode="TOGGLE")
                    self._sync_legacy_selected_key_marker(context, preferred_frame=frame)
                    self._blocking_track_click = True
                    if context.region is not None:
                        context.region.tag_redraw()
                    return {"RUNNING_MODAL"}

                plain_preserved = (
                    not event.shift
                    and hit_was_selected
                    and len(selected_before) > 1
                )
                if event.shift:
                    selection_mode = "ADD"
                    select_key_frame_for_context(context, frame, mode=selection_mode)
                elif plain_preserved:
                    # Preserve the group during press so a drag can move it.
                    # A click with no drag collapses to this frame on release.
                    selection_mode = "SET"
                else:
                    selection_mode = "SET"
                    select_key_frame_for_context(context, frame, mode=selection_mode)

                self._sync_legacy_selected_key_marker(context, preferred_frame=frame)
                source_frames = selected_key_frames_for_context(context)
                self._key_drag_selection_mode = selection_mode
                self._key_drag_clone = bool(event.shift)
                self._key_drag_source_frame = frame
                self._key_drag_current_frame = frame
                self._key_drag_source_frames = list(source_frames)
                self._key_drag_group = len(source_frames) > 1
                self._key_drag_plain_preserved_selection = plain_preserved
                # Use the lightweight captured preview transaction for both one
                # key and multi-key drags. The old single-key path re-resolved
                # and mutated the complete semantic/Contact bundle on every
                # mousemove, which made Track Bar Move/Clone visibly stall.
                self._key_drag_initial_snapshot = snapshot_active_key_state_for_context(context)
                self._key_drag_preview_transaction = snapshot_multi_key_preview_for_context(
                    context,
                    self._key_drag_source_frames,
                    mode="CLONE" if self._key_drag_clone else "MOVE",
                )
                self._key_drag_has_moved = False
                self._key_drag_replaced_snapshots = []
                self._dragging_key = True
                if context.window is not None:
                    context.window.cursor_modal_set("COPY" if self._key_drag_clone else "MOVE_X")
                if context.region is not None:
                    context.region.tag_redraw()
                return {"RUNNING_MODAL"}

            if frame is not None and (event.ctrl or event.shift):
                # Modified click outside the key row changes time but preserves
                # the existing key selection. Plain empty click still clears it.
                scene.frame_set(frame)
                self._sync_legacy_selected_key_marker(context)
                if context.region is not None:
                    context.region.tag_redraw()
                self._blocking_track_click = True
                return {"RUNNING_MODAL"}

        if not self._activate_frame_from_mouse(context, event.mouse_region_x):
            return {"CANCELLED"}
        self._blocking_track_click = True
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        scene = context.scene

        if self._dragging_selection_range and self._selection_range_mode == "CENTER":
            if event.type == "MOUSEMOVE":
                target_frame = self._frame_from_mouse(context, event.mouse_region_x)
                if (
                    target_frame is not None
                    and target_frame != self._selection_range_current_frame
                ):
                    self._update_selection_range_center(context, target_frame)
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._finish_selection_range_center(context)
                return {"FINISHED"}
            if event.type in {"ESC", "RIGHTMOUSE"}:
                self._finish_selection_range_center(context, cancel=True)
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}

        if (
            self._dragging_selection_range
            and self._selection_range_mode in {"LEFT_HANDLE", "RIGHT_HANDLE"}
        ):
            if event.type == "MOUSEMOVE":
                target_frame = self._frame_from_mouse(context, event.mouse_region_x)
                if (
                    target_frame is not None
                    and target_frame != self._selection_range_current_frame
                ):
                    self._update_selection_range_scale(context, target_frame)
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._finish_selection_range_scale(context)
                return {"FINISHED"}
            if event.type in {"ESC", "RIGHTMOUSE"}:
                self._finish_selection_range_scale(context, cancel=True)
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}

        if self._dragging_range_resize:
            if event.type == "MOUSEMOVE":
                delta_x = event.mouse_region_x - self._range_initial_mouse_x
                raw_delta = frame_delta_from_pixel_drag(delta_x, self._range_cell_width)
                if self._range_resize_side == "START":
                    new_start = min(
                        self._range_visible_end - 1,
                        self._range_visible_start - raw_delta,
                    )
                    _set_trackbar_view_bounds(scene, new_start, self._range_visible_end)
                else:
                    new_end = max(
                        self._range_visible_start + 1,
                        self._range_visible_end - raw_delta,
                    )
                    _set_trackbar_view_bounds(scene, self._range_visible_start, new_end)
                if context.area is not None:
                    context.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._dragging_range_resize = False
                if context.window is not None:
                    context.window.cursor_modal_restore()
                return {"FINISHED"}
            if event.type in {"ESC", "RIGHTMOUSE"}:
                scene.baw_trackbar_view_center = self._range_view_center_prop
                scene.baw_trackbar_view_span = self._range_view_span_prop
                self._dragging_range_resize = False
                if context.window is not None:
                    context.window.cursor_modal_restore()
                if context.area is not None:
                    context.area.tag_redraw()
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}

        if self._dragging_key:
            if event.type in {"DEL", "X"} and event.value == "PRESS":
                if self._key_drag_preview_transaction is not None:
                    self._finish_multi_key_drag(context)
                else:
                    self._finish_key_drag(context)
                self._delete_selected_keys_from_modal(context)
                return {"FINISHED"}

            if self._key_drag_preview_transaction is not None:
                if event.type == "MOUSEMOVE":
                    target_frame = self._frame_from_mouse(context, event.mouse_region_x)
                    if (
                        target_frame is not None
                        and target_frame != self._key_drag_current_frame
                    ):
                        self._apply_multi_key_drag_preview(context, target_frame)
                    return {"RUNNING_MODAL"}
                if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                    self._finish_multi_key_drag(context)
                    return {"FINISHED"}
                if event.type in {"ESC", "RIGHTMOUSE"}:
                    self._finish_multi_key_drag(context, cancel=True)
                    return {"CANCELLED"}
                return {"RUNNING_MODAL"}

            if event.type == "MOUSEMOVE":
                target_frame = self._frame_from_mouse(context, event.mouse_region_x)
                current_frame = self._key_drag_current_frame
                if target_frame is not None and target_frame != current_frame:
                    source_frame = self._key_drag_source_frame
                    if (
                        current_frame == source_frame
                        and self._key_drag_selection_mode != "SET"
                    ):
                        # Ctrl/Shift click can preserve a multi-selection, but this
                        # phase still edits only one dragged key. Collapse to the
                        # drag source only when a real frame-to-frame drag begins.
                        select_key_frame_for_context(context, source_frame, mode="SET")
                        scene.baw_has_selected_key = True
                        scene.baw_selected_key_frame = source_frame
                        self._key_drag_selection_mode = "SET"
                    if self._key_drag_clone:
                        if current_frame != source_frame:
                            remove_cloned_key_frame_for_context(
                                context,
                                source_frame,
                                current_frame,
                            )
                            restore_key_frame_snapshots_for_context(
                                context,
                                self._key_drag_replaced_snapshots,
                            )
                            self._key_drag_replaced_snapshots = []
                            self._key_drag_current_frame = source_frame

                        if target_frame == source_frame:
                            scene.frame_set(source_frame)
                            select_key_frame_for_context(context, source_frame)
                            scene.baw_has_selected_key = True
                            scene.baw_selected_key_frame = source_frame
                        else:
                            replaced_snapshots = snapshot_replaced_key_frame_for_context(
                                context,
                                source_frame,
                                target_frame,
                            )
                            if clone_key_frame_for_context(
                                context,
                                source_frame,
                                target_frame,
                            ):
                                self._key_drag_replaced_snapshots = replaced_snapshots
                                self._key_drag_current_frame = target_frame
                                scene.frame_set(target_frame)
                                select_key_frame_for_context(context, target_frame)
                                scene.baw_has_selected_key = True
                                scene.baw_selected_key_frame = target_frame
                    else:
                        replaced_snapshots = snapshot_replaced_key_frame_for_context(
                            context,
                            current_frame,
                            target_frame,
                        )
                        if move_key_frame_for_context(context, current_frame, target_frame):
                            restore_key_frame_snapshots_for_context(
                                context,
                                self._key_drag_replaced_snapshots,
                            )
                            self._key_drag_replaced_snapshots = replaced_snapshots
                            self._key_drag_current_frame = target_frame
                            scene.frame_set(target_frame)
                            scene.baw_selected_key_frame = target_frame
                    if self._key_drag_current_frame != current_frame:
                        self._sync_legacy_selected_key_marker(
                            context,
                            preferred_frame=self._key_drag_current_frame,
                        )
                        refresh_active_trajectory_live(context)
                if context.region is not None:
                    context.region.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._finish_key_drag(context)
                return {"FINISHED"}
            if event.type in {"ESC", "RIGHTMOUSE"}:
                self._finish_key_drag(context, cancel=True)
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}

        if self._dragging_frame_rail:
            if event.type == "MOUSEMOVE":
                self._set_frame_from_mouse(context, event.mouse_region_x)
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._activate_frame_from_mouse(context, event.mouse_region_x)
                self._dragging_frame_rail = False
                if context.window_manager is not None:
                    context.window_manager.baw_trackbar_frame_drag_active = False
                trace_scrub_end(context, cancelled=False)
                return {"FINISHED"}
            if event.type in {"ESC", "RIGHTMOUSE"}:
                scene.frame_set(self._initial_frame)
                self._dragging_frame_rail = False
                if context.window_manager is not None:
                    context.window_manager.baw_trackbar_frame_drag_active = False
                trace_scrub_end(context, cancelled=True)
                if context.region is not None:
                    context.region.tag_redraw()
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}

        if self._blocking_track_click:
            if (
                event.type in {"DEL", "X"}
                and event.value == "PRESS"
                and selected_key_frames_for_context(context)
            ):
                self._blocking_track_click = False
                self._delete_selected_keys_from_modal(context)
                return {"FINISHED"}
            if event.type == "MOUSEMOVE":
                # Consume drag motion inside the Track Bar without scrubbing.
                # This keeps viewport tools from seeing the held LMB drag.
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._blocking_track_click = False
                return {"FINISHED"}
            if event.type in {"ESC", "RIGHTMOUSE"}:
                self._blocking_track_click = False
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if self._dragging_frame_control:
            if event.type == "MOUSEMOVE":
                delta_x = event.mouse_region_x - self._drag_initial_mouse_x
                delta_frames = frame_delta_from_pixel_drag(delta_x, self._drag_cell_width)
                # Clamp the moving current-frame control to the Track Bar view,
                # not Scene Start/End. This preserves the independent Track Bar
                # pan/zoom model while preventing a visible 0..60 rail from
                # drifting to frame 61+ when the pointer reaches the right edge.
                visible_start, visible_end, _span, _x_min, _x_max = visible_frame_window(
                    scene,
                    context.region,
                )
                frame = max(
                    int(visible_start),
                    min(int(visible_end), self._initial_frame + delta_frames),
                )
                scene.frame_set(frame)
                if context.region is not None:
                    context.region.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._dragging_frame_control = False
                if context.window_manager is not None:
                    context.window_manager.baw_trackbar_frame_drag_active = False
                trace_scrub_end(context, cancelled=False)
                if context.window is not None:
                    context.window.cursor_modal_restore()
                return {"FINISHED"}
            if event.type in {"ESC", "RIGHTMOUSE"}:
                scene.frame_set(self._initial_frame)
                self._dragging_frame_control = False
                if context.window_manager is not None:
                    context.window_manager.baw_trackbar_frame_drag_active = False
                trace_scrub_end(context, cancelled=True)
                if context.window is not None:
                    context.window.cursor_modal_restore()
                if context.region is not None:
                    context.region.tag_redraw()
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def exit(self, context, cancel):
        if self._dragging_selection_range:
            if self._selection_range_mode == "CENTER":
                self._finish_selection_range_center(context, cancel=cancel)
            elif self._selection_range_mode in {"LEFT_HANDLE", "RIGHT_HANDLE"}:
                self._finish_selection_range_scale(context, cancel=cancel)
        if self._dragging_key:
            if self._key_drag_preview_transaction is not None:
                self._finish_multi_key_drag(context, cancel=cancel)
            else:
                self._finish_key_drag(context, cancel=cancel)
        if self._dragging_frame_control and context.window is not None:
            context.window.cursor_modal_restore()
            self._dragging_frame_control = False
        if context.window_manager is not None:
            context.window_manager.baw_trackbar_frame_drag_active = False
        trace_scrub_end(context, cancelled=bool(cancel))
        if self._dragging_range_resize and context.window is not None:
            context.window.cursor_modal_restore()
        self._dragging_range_resize = False
        self._dragging_frame_rail = False
        self._blocking_track_click = False
        if cancel and context.scene is not None:
            context.scene.frame_set(self._initial_frame)


class BAW_GGT_trackbar(bpy.types.GizmoGroup):
    bl_idname = "BAW_GGT_trackbar"
    bl_label = "Animation Workbench Track Bar"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options: ClassVar[set[str]] = {"PERSISTENT", "SHOW_MODAL_ALL"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        window_manager = getattr(context, "window_manager", None)
        semantic_drag = bool(
            window_manager is not None
            and getattr(window_manager, "baw_rigped_semantic_move_drag_active", False)
        )
        return (
            not semantic_drag
            and scene is not None
            and getattr(scene, "baw_trackbar_enabled", True)
        )

    def setup(self, context):
        gizmo = self.gizmos.new(BAW_GT_trackbar.bl_idname)
        gizmo.use_tooltip = False
        self.trackbar = gizmo


def _keyboard_event_types() -> tuple[str, ...]:
    """Return Blender key event identifiers without pointer/system events."""
    named_keys = {
        "ESC", "TAB", "RET", "LINE_FEED", "SPACE", "BACK_SPACE", "DEL",
        "SEMI_COLON", "PERIOD", "COMMA", "QUOTE", "ACCENT_GRAVE",
        "MINUS", "PLUS", "SLASH", "BACK_SLASH", "EQUAL",
        "LEFT_BRACKET", "RIGHT_BRACKET", "LEFT_ARROW", "DOWN_ARROW",
        "RIGHT_ARROW", "UP_ARROW", "PAUSE", "INSERT", "HOME",
        "PAGE_UP", "PAGE_DOWN", "END", "LEFT_CTRL", "LEFT_ALT",
        "LEFT_SHIFT", "RIGHT_CTRL", "RIGHT_ALT", "RIGHT_SHIFT", "OSKEY",
        "APP", "GRLESS", "CAPSLOCK",
    }
    identifiers: list[str] = []
    enum_items = bpy.types.KeyMapItem.bl_rna.properties["type"].enum_items
    for item in enum_items:
        identifier = str(item.identifier)
        is_letter = len(identifier) == 1 and "A" <= identifier <= "Z"
        is_digit_name = identifier in {
            "ZERO", "ONE", "TWO", "THREE", "FOUR",
            "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
        }
        is_function = (
            identifier.startswith("F")
            and identifier[1:].isdigit()
            and 1 <= int(identifier[1:]) <= 24
        )
        is_numpad = identifier.startswith("NUMPAD_")
        is_media = identifier.startswith("MEDIA")
        if (
            is_letter
            or is_digit_name
            or is_function
            or is_numpad
            or is_media
            or identifier in named_keys
        ):
            identifiers.append(identifier)
    return tuple(dict.fromkeys(identifiers))


def register_interaction_keymaps() -> None:
    wm = bpy.context.window_manager
    if wm is None or wm.keyconfigs.addon is None:
        return

    unregister_interaction_keymaps()
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
        resize_kmi = km.keymap_items.new(
            BAW_OT_resize_end_frame_drag.bl_idname,
            "LEFTMOUSE",
            "PRESS",
            alt=True,
            head=True,
        )
        _INTERACTION_KEYMAP_ITEMS.append((km, resize_kmi))

        box_select_kmi = km.keymap_items.new(
            BAW_OT_box_select_trackbar.bl_idname,
            "LEFTMOUSE",
            "PRESS",
            any=True,
            head=True,
        )
        _INTERACTION_KEYMAP_ITEMS.append((km, box_select_kmi))

        fit_kmi = km.keymap_items.new(
            BAW_OT_fit_trackbar_keys.bl_idname,
            "LEFTMOUSE",
            "DOUBLE_CLICK",
            head=True,
        )
        _INTERACTION_KEYMAP_ITEMS.append((km, fit_kmi))

        pan_kmi = km.keymap_items.new(
            BAW_OT_pan_trackbar_drag.bl_idname,
            "MIDDLEMOUSE",
            "PRESS",
            any=True,
            head=True,
        )
        _INTERACTION_KEYMAP_ITEMS.append((km, pan_kmi))

        for wheel_type in ("WHEELUPMOUSE", "WHEELDOWNMOUSE"):
            zoom_kmi = km.keymap_items.new(
                BAW_OT_zoom_trackbar_wheel.bl_idname,
                wheel_type,
                "PRESS",
                any=True,
                head=True,
            )
            _INTERACTION_KEYMAP_ITEMS.append((km, zoom_kmi))

        block_rmb_kmi = km.keymap_items.new(
            BAW_OT_block_trackbar_mouse.bl_idname,
            "RIGHTMOUSE",
            "PRESS",
            any=True,
            head=True,
        )
        _INTERACTION_KEYMAP_ITEMS.append((km, block_rmb_kmi))

        for delete_event in ("DEL", "X"):
            delete_kmi = km.keymap_items.new(
                BAW_OT_delete_trackbar_key.bl_idname,
                delete_event,
                "PRESS",
                head=True,
            )
            _INTERACTION_KEYMAP_ITEMS.append((km, delete_kmi))

        for event_type in _keyboard_event_types():
            if event_type in {"DEL", "X"}:
                # Delete/X are owned by BAW_OT_delete_trackbar_key above. The
                # generic Track Bar keyboard blocker must never shadow them.
                continue
            keyboard_block_kmi = km.keymap_items.new(
                BAW_OT_block_empty_trackbar_keyboard.bl_idname,
                event_type,
                "PRESS",
                any=True,
                head=True,
            )
            _INTERACTION_KEYMAP_ITEMS.append((km, keyboard_block_kmi))


def unregister_interaction_keymaps() -> None:
    for km, kmi in reversed(_INTERACTION_KEYMAP_ITEMS):
        if kmi.id != -1:
            km.keymap_items.remove(kmi)
    _INTERACTION_KEYMAP_ITEMS.clear()
