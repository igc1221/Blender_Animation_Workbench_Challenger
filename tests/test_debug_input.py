from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "extension" / "blender_animation_workbench" / "debug_input.py"


class FakeOperator:
    pass


def _load_module(monkeypatch):
    package_name = "awb_debug_input_testpkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MODULE_PATH.parent)]
    trace_stub = types.ModuleType(f"{package_name}.debug_trace")

    def replay_inactive():
        return False

    def no_replay_id():
        return None

    def debug_root():
        return str(ROOT / "debug")

    trace_stub.replay_execution_active = replay_inactive
    trace_stub.replay_execution_id = no_replay_id
    trace_stub.debug_root_path = debug_root
    package.debug_trace = trace_stub
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, f"{package_name}.debug_trace", trace_stub)

    bpy_module = types.ModuleType("bpy")
    bpy_module.types = types.SimpleNamespace(Operator=FakeOperator)
    bpy_module.context = types.SimpleNamespace(window_manager=None)
    monkeypatch.setitem(sys.modules, "bpy", bpy_module)

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.debug_input",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.reset_raw_input_trace()

    def ignore_file_record(_record):
        return None

    module._append_record_to_file = ignore_file_record
    return module, trace_stub, bpy_module


def _event(event_type="A", value="PRESS", **overrides):
    values = {
        "type": event_type,
        "value": value,
        "unicode": "a" if event_type == "A" else "",
        "ascii": 97 if event_type == "A" else 0,
        "shift": False,
        "ctrl": False,
        "alt": False,
        "oskey": False,
        "key_modifier": "NONE",
        "direction": "ANY",
        "is_repeat": False,
        "mouse_x": 10,
        "mouse_y": 20,
        "mouse_region_x": 4,
        "mouse_region_y": 5,
        "x": 10,
        "y": 20,
        "pressure": 0.0,
        "tilt": 0.0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_ring_is_bounded_and_reports_dropped_records(monkeypatch):
    module, _, _ = _load_module(monkeypatch)

    for index in range(module._RING_MAX_RECORDS + 1):
        module._emit("TEST", source="TEST", fields={"index": index})

    events = module.read_recent_raw_input_events(module._RING_MAX_RECORDS + 8)
    assert len(events) == module._RING_MAX_RECORDS
    assert events[0]["raw_input_seq"] == 2
    assert events[-1]["raw_input_seq"] == module._RING_MAX_RECORDS + 1
    assert all(event["schema"] == "awb-raw-input-trace/v1" for event in events)
    assert [event["raw_input_seq"] for event in events] == sorted(
        event["raw_input_seq"] for event in events
    )
    assert module.raw_input_stats()["ring_dropped"] == 1


def test_mousemove_is_sampled_with_hard_per_owner_bound(monkeypatch):
    module, _, _ = _load_module(monkeypatch)
    monkeypatch.setattr(module, "_MOUSEMOVE_MAX_PER_OWNER", 2)
    owner_id = module._new_modal_owner("baw.drag")
    context = types.SimpleNamespace(mode="POSE")

    for index in range(module._MOUSEMOVE_SAMPLE_EVERY * 4):
        event = _event(
            "MOUSEMOVE",
            "NOTHING",
            mouse_x=index,
            mouse_region_x=index,
        )
        module._record_modal_raw_event("baw.drag", owner_id, context, event)

    stats = module.raw_input_stats()
    records = module.read_recent_raw_input_events(100)
    mouse_records = [record for record in records if record["event_name"] == "MODAL_RAW_EVENT"]
    assert stats["mousemove_seen"] == 32
    assert stats["mousemove_sampled"] == 2
    assert stats["mousemove_dropped"] == 30
    assert len(mouse_records) == 2


def test_keymap_probe_fifo_is_consumed_once(monkeypatch):
    module, _, _ = _load_module(monkeypatch)
    event = _event()
    probe = module.record_keymap_probe_event(event)

    first = module.record_operator_ingress("baw.sample", None, event)
    second = module.record_operator_ingress("baw.sample", None, event)

    assert first["correlation_status"] == "CORRELATED"
    assert first["keymap_raw_input_seq"] == probe["raw_input_seq"]
    assert second["correlation_status"] == "UNCORRELATED"
    assert second["keymap_raw_input_seq"] is None


def test_replay_execution_mismatch_does_not_consume_live_probe(monkeypatch):
    module, trace_stub, _ = _load_module(monkeypatch)
    event = _event()
    probe = module.record_keymap_probe_event(event)

    def replay_active():
        return True

    def replay_id():
        return "replay-17"

    def still_inactive():
        return False

    def no_replay_id():
        return None

    trace_stub.replay_execution_active = replay_active
    trace_stub.replay_execution_id = replay_id
    mismatch = module.record_operator_ingress("baw.sample", None, event)
    trace_stub.replay_execution_active = still_inactive
    trace_stub.replay_execution_id = no_replay_id
    matching = module.record_operator_ingress("baw.sample", None, event)

    assert mismatch["correlation_status"] == "UNCORRELATED"
    assert mismatch["replay_execution_id"] == "replay-17"
    assert matching["correlation_status"] == "CORRELATED"
    assert matching["keymap_raw_input_seq"] == probe["raw_input_seq"]

    trace_stub.replay_execution_active = replay_active
    trace_stub.replay_execution_id = replay_id
    replay_probe = module.record_keymap_probe_event(event)
    trace_stub.replay_execution_active = still_inactive
    trace_stub.replay_execution_id = no_replay_id
    live_mismatch = module.record_operator_ingress("baw.sample", None, event)
    trace_stub.replay_execution_active = replay_active
    trace_stub.replay_execution_id = replay_id
    replay_match = module.record_operator_ingress("baw.sample", None, event)
    assert live_mismatch["correlation_status"] == "UNCORRELATED"
    assert replay_match["correlation_status"] == "CORRELATED"
    assert replay_match["keymap_raw_input_seq"] == replay_probe["raw_input_seq"]


def test_operator_wrappers_capture_invocation_modal_route_and_terminal(monkeypatch):
    module, _, _ = _load_module(monkeypatch)

    class SampleOperator(FakeOperator):
        bl_idname = "baw.sample_modal"

        def invoke(self, _context, _event):
            return {"RUNNING_MODAL"}

        def modal(self, _context, _event):
            return {"FINISHED"}

    module.instrument_operator_classes((SampleOperator,))
    wrapped_invoke = SampleOperator.invoke
    module.instrument_operator_classes((SampleOperator,))
    assert SampleOperator.invoke is wrapped_invoke
    operator = SampleOperator()
    context = types.SimpleNamespace(mode="POSE")
    assert operator.invoke(context, _event()) == {"RUNNING_MODAL"}
    owner_id = operator._awb_raw_input_owner_id
    assert operator.modal(context, _event("MOUSEMOVE", "NOTHING")) == {"FINISHED"}

    records = module.read_recent_raw_input_events(32)
    names = [record["event_name"] for record in records]
    assert "OPERATOR_INGRESS" in names
    assert "MODAL_OWNER_BEGIN" in names
    assert "MODAL_RAW_EVENT" in names
    assert "MODAL_ROUTE" in names
    assert "MODAL_OWNER_TERMINAL" in names
    raw_modal = next(record for record in records if record["event_name"] == "MODAL_RAW_EVENT")
    route = next(record for record in records if record["event_name"] == "MODAL_ROUTE")
    terminal = next(record for record in records if record["event_name"] == "MODAL_OWNER_TERMINAL")
    ingress = next(record for record in records if record["event_name"] == "OPERATOR_INGRESS")
    assert ingress["correlation_status"] == "UNCORRELATED"
    assert raw_modal["source"] == "MODAL_OWNER"
    assert raw_modal["owner_id"] == owner_id
    assert route["result"] == ["FINISHED"]
    assert terminal["status"] == "FINISHED"
    assert terminal["mousemove_owner_seen"] == 1
    assert terminal["mousemove_owner_sampled"] == 1
    module.restore_operator_instrumentation((SampleOperator,))
    assert SampleOperator.invoke.__name__ == "invoke"


class FakeKeymapItems(list):
    def new(self, idname, event_type, value, **options):
        item = types.SimpleNamespace(
            idname=idname,
            type=event_type,
            value=value,
            any=options.get("any", False),
            shift=options.get("shift", False),
            ctrl=options.get("ctrl", False),
            alt=options.get("alt", False),
            oskey=options.get("oskey", False),
            key_modifier=options.get("key_modifier", "NONE"),
            direction=options.get("direction", "ANY"),
            head=options.get("head", False),
        )
        self.append(item)
        return item

    def remove(self, item):
        super().remove(item)


def test_probe_registration_is_idempotent_and_skips_global_mousemove(monkeypatch):
    module, _, bpy_module = _load_module(monkeypatch)
    keymap_items = FakeKeymapItems(
        [
            types.SimpleNamespace(
                idname="baw.sample",
                type="A",
                value="PRESS",
                any=False,
                shift=False,
                ctrl=True,
                alt=False,
                oskey=False,
                key_modifier="NONE",
                direction="ANY",
            ),
            types.SimpleNamespace(
                idname="baw.drag",
                type="MOUSEMOVE",
                value="NOTHING",
                any=True,
                shift=False,
                ctrl=False,
                alt=False,
                oskey=False,
                key_modifier="NONE",
                direction="ANY",
            ),
            types.SimpleNamespace(
                idname="view3d.rotate",
                type="MIDDLEMOUSE",
                value="PRESS",
                any=False,
                shift=False,
                ctrl=False,
                alt=False,
                oskey=False,
                key_modifier="NONE",
                direction="ANY",
            ),
        ]
    )
    keymap = types.SimpleNamespace(keymap_items=keymap_items)
    keyconfig = types.SimpleNamespace(keymaps=[keymap])
    bpy_module.context.window_manager = types.SimpleNamespace(
        keyconfigs=types.SimpleNamespace(addon=keyconfig)
    )

    module.register_keymap_probes()
    module.register_keymap_probes()

    probes = [item for item in keymap_items if item.idname == module._KEYMAP_PROBE_ID]
    assert len(probes) == 4
    assert {(item.type, item.value) for item in probes} == {
        ("A", "PRESS"),
        ("A", "RELEASE"),
        ("MIDDLEMOUSE", "PRESS"),
        ("MIDDLEMOUSE", "RELEASE"),
    }
    assert all(item.type != "MOUSEMOVE" for item in probes)
    module.unregister_keymap_probes()
    assert all(item.idname != module._KEYMAP_PROBE_ID for item in keymap_items)


def test_workspace_tool_probe_is_head_idempotent_and_restored(monkeypatch):
    module, _, _ = _load_module(monkeypatch)

    class FakeTool:
        bl_keymap = (
            ("baw.sample", {"type": "LEFTMOUSE", "value": "PRESS"}, {}),
            ("baw.drag", {"type": "MOUSEMOVE", "value": "NOTHING"}, {}),
        )

    original = FakeTool.bl_keymap
    module.install_workspace_tool_keymap_probes((FakeTool,))
    first = FakeTool.bl_keymap
    module.install_workspace_tool_keymap_probes((FakeTool,))

    probes = [entry for entry in FakeTool.bl_keymap if entry[0] == module._KEYMAP_PROBE_ID]
    assert len(probes) == 1
    assert probes[0][1]["head"] is True
    assert FakeTool.bl_keymap == first
    assert all(entry[1].get("type") != "MOUSEMOVE" for entry in probes)
    module.restore_workspace_tool_keymap_probes((FakeTool,))
    assert FakeTool.bl_keymap == original

    assert module.BAW_OT_raw_input_probe().invoke(None, _event()) == {"PASS_THROUGH"}


def test_modal_exception_emits_failure_and_reraises(monkeypatch):
    module, _, _ = _load_module(monkeypatch)

    class FailingOperator(FakeOperator):
        bl_idname = "baw.failing_modal"

        def invoke(self, _context, _event):
            return {"RUNNING_MODAL"}

        def modal(self, _context, _event):
            raise RuntimeError("modal failure")

    module.instrument_operator_classes((FailingOperator,))
    operator = FailingOperator()
    context = types.SimpleNamespace(mode="POSE")
    operator.invoke(context, _event())
    with pytest.raises(RuntimeError, match="modal failure"):
        operator.modal(context, _event("LEFTMOUSE", "RELEASE"))

    records = module.read_recent_raw_input_events(16)
    failure = next(record for record in records if record["event_name"] == "MODAL_OWNER_FAILURE")
    assert failure["owner_id"] == operator._awb_raw_input_owner_id
