import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "awb_keying_context_safety_pkg"


class _RNAType:
    pass


class _Timers:
    def __init__(self):
        self.registered = set()

    def is_registered(self, callback):
        return callback in self.registered

    def register(self, callback, **_kwargs):
        self.registered.add(callback)

    def unregister(self, callback):
        self.registered.remove(callback)


def _load_keying_module(monkeypatch):
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(
        KeyMap=_RNAType,
        KeyMapItem=_RNAType,
        KeyingSetInfo=_RNAType,
        Operator=_RNAType,
    )

    bpy_app = ModuleType("bpy.app")
    bpy_handlers = ModuleType("bpy.app.handlers")
    bpy_handlers.persistent = lambda function: function
    bpy_handlers.depsgraph_update_post = []
    bpy_handlers.frame_change_post = []
    bpy_handlers.undo_post = []
    bpy_handlers.redo_post = []
    bpy_handlers.load_post = []
    timers = _Timers()
    bpy_app.handlers = bpy_handlers
    bpy_app.timers = timers
    bpy.app = SimpleNamespace(background=False, handlers=bpy_handlers, timers=timers)
    bpy.context = SimpleNamespace(scene=None)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.app", bpy_app)
    monkeypatch.setitem(sys.modules, "bpy.app.handlers", bpy_handlers)

    package = ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_DIR)]
    monkeypatch.setitem(sys.modules, PACKAGE_NAME, package)

    semantic_adapter = ModuleType(f"{PACKAGE_NAME}.semantic_adapter")
    for name in (
        "assigned_channelbag",
        "channel_binding_token",
        "channels_for_property",
        "control_context_for_context",
        "control_property_path",
        "keying_controls_for_context",
        "resolve_control_target",
        "rotation_property",
        "runtime_control_key",
    ):
        setattr(semantic_adapter, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, semantic_adapter.__name__, semantic_adapter)

    trackbar_model = ModuleType(f"{PACKAGE_NAME}.trackbar_model")
    trackbar_model.clear_key_selection_for_context = lambda _context: None
    monkeypatch.setitem(sys.modules, trackbar_model.__name__, trackbar_model)

    module_path = PACKAGE_DIR / "trackbar_keying.py"
    module_name = f"{PACKAGE_NAME}.trackbar_keying"
    spec = spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _row(control_key, owner, action=None, slot=None, channelbag=None):
    return (control_key, (owner, action, slot, channelbag))


def test_first_binding_creation_accepts_same_control_none_to_action(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    baseline = (_row((10, 11), 10),)
    current = (_row((10, 11), 10, 20, 30, 40),)

    assert keying._auto_key_is_first_binding_creation(baseline, current)


def test_first_binding_creation_rejects_existing_action_reassignment(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    baseline = (_row((10, 11), 10, 20, 30, 40),)
    current = (_row((10, 11), 10, 21, 31, 41),)

    assert not keying._auto_key_is_first_binding_creation(baseline, current)


def test_first_binding_creation_rejects_control_change(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    baseline = (_row((10, 11), 10),)
    current = (_row((12, 13), 12, 20, 30, 40),)

    assert not keying._auto_key_is_first_binding_creation(baseline, current)


def test_first_binding_creation_rejects_binding_only_slot_change(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    baseline = (_row((10, 11), 10, 20, 30, 40),)
    current = (_row((10, 11), 10, 20, 31, 41),)

    assert not keying._auto_key_is_first_binding_creation(baseline, current)


def test_first_binding_creation_requires_an_actual_change(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    baseline = (_row((10, 11), 10),)

    assert not keying._auto_key_is_first_binding_creation(baseline, baseline)


def test_latched_transform_preserves_baseline_on_late_first_action_creation(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    scene = SimpleNamespace(baw_auto_key_enabled=True)
    context = SimpleNamespace(scene=scene)
    baseline = (_row((10, 11), 10),)
    current = (_row((10, 11), 10, 20, 30, 40),)
    snapshot = {(10, 11): ("settled",)}
    keying.bpy.context = context
    keying._AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE = baseline
    keying._AUTO_KEY_TRANSFORM_SNAPSHOT = snapshot
    keying._AUTO_KEY_NATIVE_TRANSFORM_LATCHED = True
    keying._AUTO_KEY_POSTPROCESS_TIMER_PENDING = True
    monkeypatch.setattr(keying, "_auto_key_context_binding_signature", lambda _context: current)
    monkeypatch.setattr(keying, "_capture_auto_key_transform_snapshot", lambda _context: snapshot)
    monkeypatch.setattr(keying, "_transform_modal_is_active", lambda _context: False)
    monkeypatch.setattr(keying, "_last_operator_is_transform", lambda _context: True)
    resync_calls = []
    monkeypatch.setattr(
        keying,
        "resync_awb_auto_key_after_context_change",
        lambda _context: resync_calls.append("resync"),
    )

    keying._awb_auto_key_depsgraph_update_post(scene, None)

    assert resync_calls == []
    assert keying._AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE == baseline
    assert keying._AUTO_KEY_NATIVE_TRANSFORM_LATCHED
    assert keying._AUTO_KEY_POSTPROCESS_TIMER_PENDING


def test_latched_transform_still_rebases_existing_action_reassignment(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    scene = SimpleNamespace(baw_auto_key_enabled=True)
    context = SimpleNamespace(scene=scene)
    baseline = (_row((10, 11), 10, 20, 30, 40),)
    current = (_row((10, 11), 10, 21, 31, 41),)
    snapshot = {(10, 11): ("settled",)}
    keying.bpy.context = context
    keying._AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE = baseline
    keying._AUTO_KEY_TRANSFORM_SNAPSHOT = snapshot
    keying._AUTO_KEY_NATIVE_TRANSFORM_LATCHED = True
    keying._AUTO_KEY_POSTPROCESS_TIMER_PENDING = True
    monkeypatch.setattr(keying, "_auto_key_context_binding_signature", lambda _context: current)
    monkeypatch.setattr(keying, "_capture_auto_key_transform_snapshot", lambda _context: snapshot)
    monkeypatch.setattr(keying, "_transform_modal_is_active", lambda _context: False)
    monkeypatch.setattr(keying, "_last_operator_is_transform", lambda _context: True)
    resync_calls = []
    monkeypatch.setattr(
        keying,
        "resync_awb_auto_key_after_context_change",
        lambda _context: resync_calls.append("resync"),
    )

    keying._awb_auto_key_depsgraph_update_post(scene, None)

    assert resync_calls == ["resync"]


def test_lifecycle_post_discards_stale_timer_and_auto_key_caches(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    timer = keying._run_auto_key_postprocess_timer
    keying.bpy.app.timers.register(timer)
    keying._AUTO_KEY_POSTPROCESS_TIMER_PENDING = True
    keying._AUTO_KEY_LAST_TRANSFORM_CHANGE = 123.0
    keying._AUTO_KEY_NATIVE_TRANSFORM_LATCHED = True
    keying._AUTO_KEY_DATA_SIGNATURE = ("stale",)
    keying._AUTO_KEY_TRANSFORM_SNAPSHOT = {(1, 2): ("stale",)}
    keying._AUTO_KEY_ANIMATED_FAMILIES = {((1, 2), "location")}
    keying._AUTO_KEY_FRAME_BASELINE_TRANSFORM = {(1, 2): ("stale",)}
    keying._AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES = {((1, 2), "location")}
    keying._AUTO_KEY_FRAME_BASELINE_FRAME = 20
    keying._AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE = (_row((1, 2), 1, 2, 3, 4),)

    keying._awb_auto_key_lifecycle_post()

    assert not keying.bpy.app.timers.is_registered(timer)
    assert not keying._AUTO_KEY_POSTPROCESS_TIMER_PENDING
    assert keying._AUTO_KEY_LAST_TRANSFORM_CHANGE == 0.0
    assert not keying._AUTO_KEY_NATIVE_TRANSFORM_LATCHED
    assert keying._AUTO_KEY_DATA_SIGNATURE is None
    assert keying._AUTO_KEY_TRANSFORM_SNAPSHOT == {}
    assert keying._AUTO_KEY_ANIMATED_FAMILIES == set()
    assert keying._AUTO_KEY_FRAME_BASELINE_TRANSFORM == {}
    assert keying._AUTO_KEY_FRAME_BASELINE_ANIMATED_FAMILIES == set()
    assert keying._AUTO_KEY_FRAME_BASELINE_FRAME is None
    assert keying._AUTO_KEY_FRAME_BASELINE_CONTEXT_SIGNATURE is None


def test_undo_redo_load_handlers_register_and_unregister_cleanly(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    handlers = keying.bpy.app.handlers

    keying.register_auto_key_handlers()

    assert keying._awb_auto_key_depsgraph_update_post in handlers.depsgraph_update_post
    assert keying._awb_auto_key_frame_change_post in handlers.frame_change_post
    for handler_list in (handlers.undo_post, handlers.redo_post, handlers.load_post):
        assert keying._awb_auto_key_lifecycle_post in handler_list

    keying.bpy.app.timers.register(keying._run_auto_key_postprocess_timer)
    keying._AUTO_KEY_POSTPROCESS_TIMER_PENDING = True
    keying._AUTO_KEY_NATIVE_TRANSFORM_LATCHED = True
    keying.unregister_auto_key_handlers()

    assert keying._awb_auto_key_depsgraph_update_post not in handlers.depsgraph_update_post
    assert keying._awb_auto_key_frame_change_post not in handlers.frame_change_post
    for handler_list in (handlers.undo_post, handlers.redo_post, handlers.load_post):
        assert keying._awb_auto_key_lifecycle_post not in handler_list
    assert not keying.bpy.app.timers.is_registered(keying._run_auto_key_postprocess_timer)
    assert not keying._AUTO_KEY_POSTPROCESS_TIMER_PENDING
    assert not keying._AUTO_KEY_NATIVE_TRANSFORM_LATCHED


def test_first_auto_key_transform_preserves_frame_zero_seed(monkeypatch):
    keying = _load_keying_module(monkeypatch)
    resolved = SimpleNamespace(target=object())
    monkeypatch.setattr(keying, "runtime_control_key", lambda _resolved: (10, 11))
    monkeypatch.setattr(keying, "_rotation_path", lambda _target: "rotation_euler")
    monkeypatch.setattr(
        keying,
        "_full_transform_data_path",
        lambda _target, property_name: property_name,
    )

    inserted = []
    rewritten = []
    monkeypatch.setattr(
        keying,
        "_insert_transform_key",
        lambda target, frame, **kwargs: inserted.append((target, frame, kwargs)),
    )
    monkeypatch.setattr(
        keying,
        "_rewrite_key_values_at_frame",
        lambda resolved, property_name, frame, values: rewritten.append(
            (resolved, property_name, frame, values)
        ),
    )

    previous = ((1.0, 2.0, 3.0), "ROT", (0.1, 0.2, 0.3), (1.0, 1.0, 1.0))
    changed = keying._seed_auto_key_default_frame(
        resolved,
        set(),
        previous,
        position=True,
        rotation=True,
        scale=True,
        frame=20,
    )

    assert changed
    assert inserted == [
        (
            resolved.target,
            0,
            {"position": True, "rotation": True, "scale": True},
        )
    ]
    assert rewritten == [
        (resolved, "location", 0, previous[0]),
        (resolved, "rotation_euler", 0, previous[2]),
        (resolved, "scale", 0, previous[3]),
    ]


def test_last_operator_accepts_trackball_but_rejects_generic_transform_wrappers(monkeypatch):
    keying = _load_keying_module(monkeypatch)

    operator = SimpleNamespace(
        bl_idname="TRANSFORM_OT_trackball",
        bl_rna=SimpleNamespace(identifier="TRANSFORM_OT_trackball"),
    )
    context = SimpleNamespace(
        active_operator=operator,
        window_manager=SimpleNamespace(operators=[operator]),
    )
    assert keying._last_operator_is_transform(context)

    for identifier in ("TRANSFORM_OT_transform", "TRANSFORM_OT_from_gizmo"):
        generic = SimpleNamespace(
            bl_idname=identifier,
            bl_rna=SimpleNamespace(identifier=identifier),
        )
        generic_context = SimpleNamespace(
            active_operator=generic,
            window_manager=SimpleNamespace(operators=[generic]),
        )
        assert not keying._last_operator_is_transform(generic_context), identifier


def test_modal_operator_accepts_trackball_and_gizmo_tweak_only(monkeypatch):
    keying = _load_keying_module(monkeypatch)

    for identifier in ("TRANSFORM_OT_trackball", "GIZMOGROUP_OT_gizmo_tweak"):
        operator = SimpleNamespace(
            bl_idname=identifier,
            idname=identifier,
            bl_rna=SimpleNamespace(identifier=identifier),
        )
        context = SimpleNamespace(
            window_manager=SimpleNamespace(
                windows=[SimpleNamespace(modal_operators=[operator])]
            )
        )
        assert keying._transform_modal_is_active(context), identifier

    for identifier in ("TRANSFORM_OT_transform", "TRANSFORM_OT_from_gizmo"):
        generic = SimpleNamespace(
            bl_idname=identifier,
            idname=identifier,
            bl_rna=SimpleNamespace(identifier=identifier),
        )
        generic_context = SimpleNamespace(
            window_manager=SimpleNamespace(
                windows=[SimpleNamespace(modal_operators=[generic])]
            )
        )
        assert not keying._transform_modal_is_active(generic_context), identifier
