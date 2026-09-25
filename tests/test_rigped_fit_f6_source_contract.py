from pathlib import Path

ROOT = Path(__file__).parents[1]
EXT = ROOT / "extension" / "blender_animation_workbench"


def _source(name: str) -> str:
    return (EXT / name).read_text(encoding="utf-8")


def test_f6_load_cleanup_is_transient_only_and_has_no_save_hooks():
    ui = _source("rigped_create_fit_ui.py")
    cleanup_start = ui.index("def _clear_fit_transient_state_after_load")
    cleanup_end = ui.index("def _hide_rigped_internal_helpers_timer", cleanup_start)
    cleanup = ui[cleanup_start:cleanup_end]

    assert "_FIT_STATES.clear()" in cleanup
    assert "clear_fit_semantic_sessions()" in cleanup
    assert "bpy.data" not in cleanup
    assert "save_pre" not in ui
    assert "save_post" not in ui


def test_f6_dead_window_pruning_and_pointer_reuse_are_explicit():
    ui = _source("rigped_create_fit_ui.py")
    session = _source("rigped_fit_session.py")

    assert "def prune_dead_fit_window_state" in ui
    assert "prune_fit_semantic_sessions(live_windows)" in ui
    assert "window_screen_pointer" in ui
    assert "window_screen_pointer" in session
    assert "scene_pointer" in ui
    assert "scene_pointer" in session
    assert "def _window_screen_pointer" in ui
    assert "def _window_screen_pointer" in session
    assert "def _window_scene_pointer" in ui
    assert "def _scene_pointer" in session
    assert "def _live_window_generations" in ui
    assert "_FIT_WINDOW_MANAGER" not in ui
    assert "_FIT_WINDOW_COUNT" not in ui
    assert "window_object" not in ui
    assert "window_object" not in session
    assert "def clear_fit_semantic_sessions" in session


def test_f6_reset_remains_available_for_present_but_stale_state():
    ui = _source("rigped_create_fit_ui.py")
    start = ui.index("class BAW_OT_rigped_fit_cancel")
    end = ui.index("_PICKER_ROLE_TEXT", start)
    block = ui[start:end]

    assert "return fit_ui_state_present(context)" in block
    assert "Stale Fit state discarded" in block
    assert "native file state unchanged" in block
    assert "_FIT_STATES.pop(key, None)" in block
    assert "_restore_fit_transform_settings(context, state)" in block
    assert "_transition_to_rig_selection(context, state.rig_object)" in block


def test_f6_native_guard_and_figure_ownership_remain_fail_closed():
    keymap = _source("viewport_keymap.py")
    ui = _source("rigped_create_fit_ui.py")

    guard = keymap[keymap.index("class BAW_OT_guard_figure_native_edit") :]
    assert "return fit_ui_state_present(context)" in guard
    assert "if any(state.character_id == character_id for state in _FIT_STATES.values())" in ui
    assert "if fit_host_present:" in keymap
    assert "FIT_F5_NON_OBJECT_TRANSFORM_BLOCKED" in keymap


def test_f6_no_legacy_live_editbone_authority_or_save_mutation_is_reintroduced():
    product_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in EXT.glob("*.py")
        if path.name not in {"debug_trace.py", "debug_causal.py"}
    )

    assert "rigped_fit_transform" not in product_sources
    assert "save_pre" not in product_sources
    assert "save_post" not in product_sources

    debug_trace = _source("debug_trace.py")
    save_pre_start = debug_trace.index("def _trace_save_pre")
    save_pre_end = debug_trace.index("def _trace_save_post", save_pre_start)
    save_pre_block = debug_trace[save_pre_start:save_pre_end]
    assert "bpy.data" not in save_pre_block
    assert "bpy.ops" not in save_pre_block

    commit = _source("rigped_fit_commit.py")
    assert "def commit_fit_semantic_session_atomic" in commit
    assert "_apply_manifest(semantic_session.rig_object, manifest, before)" in commit

def test_f6_box_display_and_draw_hot_paths_do_not_become_fit_authority():
    ui = _source("rigped_create_fit_ui.py")
    overlay = _source("rigped_box_wire_overlay.py")

    raw_start = ui.index("def _fit_ui_state_raw")
    raw_end = ui.index("def fit_ui_state_present", raw_start)
    raw_block = ui[raw_start:raw_end]

    assert "validate_fit_semantic_session" not in raw_block
    assert "resolve_character" not in raw_block
    assert "resolve_character" not in overlay
    assert "validate_fit_semantic_session" not in overlay
    assert "fit_geometry_snapshot_for_rig" in overlay

    display_start = ui.index("def set_rigped_box_wire_display")
    display_end = ui.index("def update_rigped_box_wire_options", display_start)
    display_block = ui[display_start:display_end]
    assert "_BIPED_BOX_WIRE_PROPERTY" in display_block
    assert "edit_bones" not in display_block
    assert "commit_fit_semantic_session_atomic" not in display_block

def test_f6_screen_mismatch_keeps_ui_stale_present_until_explicit_reset():
    ui = _source("rigped_create_fit_ui.py")
    session = _source("rigped_fit_session.py")

    prune_start = ui.index("def prune_dead_fit_window_state")
    prune_end = ui.index("def _fit_ui_state_raw", prune_start)
    prune_block = ui[prune_start:prune_end]
    state_start = ui.index("def fit_ui_state(context)")
    state_end = ui.index("def set_fit_transform_mode", state_start)
    state_block = ui[state_start:state_end]

    assert "generation[1] != state.scene_pointer" in prune_block
    assert "generation[0] != state.window_screen_pointer" not in prune_block
    assert "_window_screen_pointer(window) != state.window_screen_pointer" in state_block
    assert "screen_pointer != session.window_screen_pointer" in session

