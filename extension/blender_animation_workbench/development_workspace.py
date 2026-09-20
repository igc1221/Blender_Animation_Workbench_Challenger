from __future__ import annotations

import bpy

WORKSPACE_NAME = "AWB Development"
AWB_VIEWPORT_BACKGROUND = (28.0 / 255.0, 28.0 / 255.0, 28.0 / 255.0)
_TIMER_DELAY_SECONDS = 0.25
_timer_registered = False


def _window_region(area):
    return next((region for region in area.regions if region.type == "WINDOW"), None)


def _configure_view3d(area) -> None:
    if area.type != "VIEW_3D":
        area.type = "VIEW_3D"
    area.show_menus = False
    space = area.spaces.active
    shading = getattr(space, "shading", None)
    if shading is not None:
        shading.background_type = "VIEWPORT"
        shading.background_color = AWB_VIEWPORT_BACKGROUND
    for attr, value in (
        ("show_region_toolbar", False),
        ("show_region_ui", False),
        ("show_region_hud", False),
        ("show_region_tool_header", False),
        ("show_region_header", False),
    ):
        if hasattr(space, attr):
            setattr(space, attr, value)


def ensure_development_workspace() -> bool:
    """Create/activate the AWB development workspace and enter a clean View3D focus view."""
    if bpy.app.background:
        return False

    context = bpy.context
    wm = context.window_manager
    if wm is None or not wm.windows:
        return False

    window = context.window or wm.windows[0]
    workspace = window.workspace
    if workspace is None:
        return False

    # The active workspace and its area layout belong to the saved manual
    # baseline. Startup must not switch to a stale AWB Development workspace.

    screen = window.screen
    if screen is None or not screen.areas:
        return False

    # Configure the largest existing area in place, but preserve the saved
    # multi-area screen layout. In particular, never maximize View3D here.
    area = max(screen.areas, key=lambda item: item.width * item.height)
    _configure_view3d(area)
    if hasattr(screen, "show_statusbar"):
        screen.show_statusbar = False

    return True


def _timer_entry():
    global _timer_registered
    _timer_registered = False
    try:
        ensure_development_workspace()
    except RuntimeError as exc:  # pragma: no cover - Blender UI diagnostics only
        print(f"AWB development workspace setup failed: {exc}")


def register_development_workspace() -> None:
    global _timer_registered
    if bpy.app.background or _timer_registered:
        return
    bpy.app.timers.register(_timer_entry, first_interval=_TIMER_DELAY_SECONDS)
    _timer_registered = True


def unregister_development_workspace() -> None:
    global _timer_registered
    if _timer_registered and bpy.app.timers.is_registered(_timer_entry):
        bpy.app.timers.unregister(_timer_entry)
    _timer_registered = False
