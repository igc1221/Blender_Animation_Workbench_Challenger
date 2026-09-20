from __future__ import annotations

import bpy

from .trackbar_drawing import draw_trackbar

_DRAW_HANDLE = None
_RIGPED_SETUP_PROPERTY = "awb_rigped_setup"


def _native_rigped_transform_active(context) -> bool:
    active = getattr(context, "active_object", None)
    getter = getattr(active, "get", None) if active is not None else None
    if (
        active is None
        or getattr(active, "type", None) != "ARMATURE"
        or not callable(getter)
        or not getter(_RIGPED_SETUP_PROPERTY)
    ):
        return False
    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        return False
    semantic_mode = str(
        getattr(getattr(context, "scene", None), "baw_rigped_semantic_transform_mode", "NONE")
    )
    direct_gizmo_active = semantic_mode in {"DIRECT_MOVE", "DIRECT_ROTATE"}
    for window in window_manager.windows:
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
                }
                or (direct_gizmo_active and "gizmo_tweak" in identifier)
                for identifier in lowered
            ):
                return True
    return False


def _draw_callback() -> None:
    context = bpy.context
    if context.area is None or context.area.type != "VIEW_3D":
        return
    scene = context.scene
    if scene is None or not getattr(scene, "baw_trackbar_enabled", True):
        return
    # Track Bar is persistent UI: never hide it just because Rigped is being
    # transformed or Fit is active. trackbar_drawing already uses a lightweight
    # Edit-Armature path, so visibility must remain stable across all Rigped modes.
    draw_trackbar(context)


def tag_all_view3d_redraw() -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def register_draw_handler() -> None:
    global _DRAW_HANDLE
    if _DRAW_HANDLE is not None:
        return
    _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        _draw_callback,
        (),
        "WINDOW",
        "POST_PIXEL",
    )
    tag_all_view3d_redraw()


def unregister_draw_handler() -> None:
    global _DRAW_HANDLE
    if _DRAW_HANDLE is None:
        return
    bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
    _DRAW_HANDLE = None
    tag_all_view3d_redraw()
