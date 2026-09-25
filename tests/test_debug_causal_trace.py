from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "extension" / "blender_animation_workbench" / "debug_causal.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("awb_debug_causal_tested", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_trace_module(monkeypatch):
    package_name = "awb_debug_trace_testpkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MODULE_PATH.parent)]
    monkeypatch.setitem(sys.modules, package_name, package)

    causal_name = f"{package_name}.debug_causal"
    causal_spec = importlib.util.spec_from_file_location(causal_name, MODULE_PATH)
    assert causal_spec is not None and causal_spec.loader is not None
    causal_module = importlib.util.module_from_spec(causal_spec)
    monkeypatch.setitem(sys.modules, causal_name, causal_module)
    causal_spec.loader.exec_module(causal_module)

    handlers_module = types.ModuleType("bpy.app.handlers")
    handlers_module.persistent = lambda callback: callback
    for handler_name in (
        "frame_change_post",
        "load_pre",
        "load_post",
        "load_post_fail",
        "save_pre",
        "save_post",
        "save_post_fail",
        "undo_pre",
        "undo_post",
        "redo_pre",
        "redo_post",
        "depsgraph_update_pre",
        "depsgraph_update_post",
    ):
        setattr(handlers_module, handler_name, [])

    app_module = types.ModuleType("bpy.app")
    app_module.handlers = handlers_module
    app_module.tempdir = ""

    bpy_module = types.ModuleType("bpy")
    bpy_module.app = app_module
    bpy_module.context = None
    bpy_module.data = types.SimpleNamespace(filepath="")
    bpy_module.types = types.SimpleNamespace(PoseBone=type("PoseBone", (), {}))

    monkeypatch.setitem(sys.modules, "bpy", bpy_module)
    monkeypatch.setitem(sys.modules, "bpy.app", app_module)
    monkeypatch.setitem(sys.modules, "bpy.app.handlers", handlers_module)

    trace_name = f"{package_name}.debug_trace"
    trace_path = MODULE_PATH.parent / "debug_trace.py"
    trace_spec = importlib.util.spec_from_file_location(trace_name, trace_path)
    assert trace_spec is not None and trace_spec.loader is not None
    trace_module = importlib.util.module_from_spec(trace_spec)
    monkeypatch.setitem(sys.modules, trace_name, trace_module)
    trace_spec.loader.exec_module(trace_module)
    return trace_module


def _configure_trace_io(module, tmp_path: Path, monkeypatch) -> Path:
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(module, "_trace_path", lambda: trace_path)
    monkeypatch.setattr(module, "_context_snapshot", lambda _context: {})
    monkeypatch.setattr(module, "_write_runtime_context_snapshot", lambda _record: None)
    monkeypatch.setattr(module, "persist_latest_replay_script", lambda: None)
    return trace_path


def test_root_and_child_share_trace_but_not_span():
    module = _load_module()
    root = module.new_trace_root("selection")
    child = module.new_trace_child(root, "operator")

    assert root.trace_id == child.trace_id
    assert root.span_id != child.span_id
    assert root.parent_span_id is None
    assert child.parent_span_id == root.span_id


def test_registry_inherits_only_from_named_bound_parent():
    module = _load_module()
    registry = module.OperationCausalRegistry()
    root = module.new_trace_root("transform")
    registry.bind("parent-op", root)

    child = registry.link("child-op", "parent-op")

    assert child is not None
    assert child.trace_id == root.trace_id
    assert child.parent_span_id == root.span_id
    assert registry.get("child-op") == child
    assert registry.link("orphan", "missing-parent") is None
    assert registry.get("orphan") is None


def test_explicit_child_binding_wins_over_auto_inheritance():
    module = _load_module()
    registry = module.OperationCausalRegistry()
    parent = module.new_trace_root("parent")
    explicit = module.new_trace_root("explicit")
    registry.bind("parent-op", parent)
    registry.bind("child-op", explicit)

    resolved = registry.link("child-op", "parent-op")

    assert resolved == explicit
    assert resolved.trace_id != parent.trace_id


def test_registry_reset_refuses_stale_parent_inheritance():
    module = _load_module()
    registry = module.OperationCausalRegistry()
    root = module.new_trace_root("old")
    registry.bind("parent-op", root)
    old_epoch = registry.epoch

    registry.reset()

    assert registry.epoch != old_epoch
    assert registry.get("parent-op") is None
    assert registry.link("child-op", "parent-op") is None


def test_interleaved_roots_never_cross_contaminate():
    module = _load_module()
    registry = module.OperationCausalRegistry()
    first = module.new_trace_root("first")
    second = module.new_trace_root("second")
    registry.bind("a", first)
    registry.bind("b", second)

    a_child = registry.link("a-child", "a")
    b_child = registry.link("b-child", "b")

    assert a_child is not None and b_child is not None
    assert a_child.trace_id == first.trace_id
    assert b_child.trace_id == second.trace_id
    assert a_child.trace_id != b_child.trace_id


def test_registry_is_bounded():
    module = _load_module()
    registry = module.OperationCausalRegistry(max_bindings=16)
    for index in range(40):
        registry.bind(f"op-{index}", module.new_trace_root(f"root-{index}"))

    assert len(registry) == 16
    assert registry.get("op-0") is None
    assert registry.get("op-39") is not None


def test_registry_retire_cascades_descendants_and_parent_links():
    module = _load_module()
    registry = module.OperationCausalRegistry()
    root = module.new_trace_root("root")
    registry.bind("parent", root)
    child = registry.link("child", "parent")
    grandchild = registry.link("grandchild", "child")

    assert child is not None and grandchild is not None
    assert registry.parent("child") == "parent"
    assert registry.parent("grandchild") == "child"

    retired = registry.retire("parent")

    assert set(retired) == {"parent", "child", "grandchild"}
    assert registry.get("parent") is None
    assert registry.get("child") is None
    assert registry.get("grandchild") is None
    assert registry.parent("child") is None
    assert registry.parent("grandchild") is None


def test_registry_link_to_missing_parent_keeps_no_stale_parent_relation():
    module = _load_module()
    registry = module.OperationCausalRegistry()
    explicit = module.new_trace_root("explicit")
    registry.bind("child", explicit)

    assert registry.link("child", "missing-parent") is None
    assert registry.parent("child") is None
    assert registry.get("child") == explicit


def test_handler_recursion_guard_fails_closed_until_exit():
    module = _load_module()
    guard = module.LifecycleHandlerRecursionGuard()

    assert guard.enter() is True
    assert guard.active is True
    assert guard.enter() is False
    guard.exit()
    assert guard.active is False
    assert guard.enter() is True
    guard.exit()


def test_vocabulary_and_terminal_enums_are_frozen():
    module = _load_module()

    assert module.validate_trace_subsystem("KEYMAP") == "keymap"
    assert module.validate_trace_lifecycle_phase("MODAL_TICK") == "modal_tick"
    assert module.validate_trace_evaluation_phase("POST_HANDLER") == "post_handler"
    assert module.TraceTerminalStatus.FINISHED.value == "FINISHED"
    assert module.TraceRouteOutcome.NATIVE_FALLTHROUGH.value == "NATIVE_FALLTHROUGH"

    with pytest.raises(ValueError):
        module.validate_trace_subsystem("rigped")
    with pytest.raises(ValueError):
        module.validate_trace_lifecycle_phase("maybe")
    with pytest.raises(ValueError):
        module.validate_trace_evaluation_phase("guess")


def test_auto_link_records_operation_span_origin():
    module = _load_module()
    registry = module.OperationCausalRegistry()
    root = module.new_trace_root("root")
    registry.bind("parent", root)

    child = registry.link("child", "parent")

    assert child is not None
    assert child.span_origin == "AUTO_OPERATION_LINK"


def test_debug_trace_core_resets_epochs_and_keeps_depsgraph_opt_in():
    source = (
        ROOT
        / "extension"
        / "blender_animation_workbench"
        / "debug_trace.py"
    ).read_text(encoding="utf-8")

    assert source.count("_OPERATION_CAUSAL.reset()") >= 2
    assert '_register_handler_list("undo_pre", _trace_undo_pre)' in source
    assert '_register_handler_list("save_pre", _trace_save_pre)' in source
    assert "Depsgraph tracing is intentionally opt-in" in source
    assert "if not _DEPSGRAPH_TRACE_ENABLED" in source
    assert "_LIFECYCLE_HANDLER_GUARD.enter()" in source
    assert '"last_observed_causal"' in source
    assert "_OPERATION_PARENTS" not in source
    assert source.count("_OPERATION_CAUSAL.retire(") >= 3
    assert "_RUNTIME_ERROR_CAUSAL_MAX_AGE_NS" in source
    assert '"causal_correlation"' in source


@pytest.mark.parametrize("mode", ["MOVE", "ROTATE", "SCALE"])
def test_production_trace_wrapper_generic_wer_keeps_one_causal_chain(
    mode,
    tmp_path: Path,
    monkeypatch,
):
    module = _load_trace_module(monkeypatch)
    trace_path = _configure_trace_io(module, tmp_path, monkeypatch)

    causal_root = module.new_trace_causal_root("transform-tool")
    module.trace_lifecycle_event(
        "TRANSFORM_TOOL_INGRESS",
        subsystem="keymap",
        phase="ingress",
        causal=causal_root,
        route_outcome=module.TraceRouteOutcome.CLAIMED,
        requested_mode=mode,
    )
    module.trace_event(
        "INPUT",
        "TRANSFORM_TOOL_REQUEST",
        causal=causal_root,
        subsystem="input",
        lifecycle_phase="ingress",
        requested_mode=mode,
    )
    route_causal = module.new_trace_causal_child(
        causal_root,
        "transform-tool-route",
    )
    module.trace_lifecycle_event(
        "TRANSFORM_TOOL_GENERIC_ROUTE",
        subsystem="operator",
        phase="routing",
        causal=route_causal,
        route_outcome=module.TraceRouteOutcome.CLAIMED,
        requested_mode=mode,
        route="GENERIC_WORKSPACE_TOOL",
    )
    module.trace_lifecycle_event(
        "TRANSFORM_TOOL_GENERIC_FINISHED",
        subsystem="operator",
        phase="commit",
        causal=route_causal,
        terminal_status=module.TraceTerminalStatus.FINISHED,
        route_outcome=module.TraceRouteOutcome.CLAIMED,
        requested_mode=mode,
        route="GENERIC_WORKSPACE_TOOL",
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["event"] for row in rows] == [
        "TRANSFORM_TOOL_INGRESS",
        "TRANSFORM_TOOL_REQUEST",
        "TRANSFORM_TOOL_GENERIC_ROUTE",
        "TRANSFORM_TOOL_GENERIC_FINISHED",
    ]
    assert {row["trace_id"] for row in rows} == {causal_root.trace_id}
    assert rows[0]["span_id"] == causal_root.span_id
    assert rows[1]["span_id"] == causal_root.span_id
    assert rows[2]["span_id"] == route_causal.span_id
    assert rows[2]["parent_span_id"] == causal_root.span_id
    assert rows[3]["span_id"] == route_causal.span_id
    assert rows[3]["terminal_status"] == "FINISHED"


@pytest.mark.parametrize(
    ("root_prefix", "begin_event", "writer_prefix", "writer_begin", "writer_terminal", "terminal_event"),
    [
        (
            "contact",
            "CONTACT_BEGIN",
            "baw-contact",
            "CONTACT_WRITE_BEGIN",
            "CONTACT_WRITE_COMMIT",
            "CONTACT_COMMIT",
        ),
        (
            "direct-rotate",
            "TRANSFORM_BEGIN",
            "auto-writer",
            "DIRECT_WRITE_BEGIN",
            "DIRECT_WRITE_COMMIT",
            "TRANSFORM_COMMIT",
        ),
    ],
)
def test_production_trace_wrapper_contact_and_rigped_bind_writer_children(
    root_prefix,
    begin_event,
    writer_prefix,
    writer_begin,
    writer_terminal,
    terminal_event,
    tmp_path: Path,
    monkeypatch,
):
    module = _load_trace_module(monkeypatch)
    trace_path = _configure_trace_io(module, tmp_path, monkeypatch)

    operation_id = f"{root_prefix}:parent"
    writer_operation_id = f"{writer_prefix}:child"
    causal_root = module.new_trace_causal_root(root_prefix)
    module.bind_operation_causal_context(operation_id, causal_root)
    module.trace_event(
        "OPERATION",
        begin_event,
        operation_id=operation_id,
        subsystem="modal",
        lifecycle_phase="invoke",
    )
    child_causal = module.link_trace_operation(writer_operation_id, operation_id)
    assert child_causal is not None
    module.trace_event(
        "WRITER",
        writer_begin,
        operation_id=writer_operation_id,
    )
    module.trace_event(
        "WRITER",
        writer_terminal,
        operation_id=writer_operation_id,
    )
    module.trace_event(
        "OPERATION",
        terminal_event,
        operation_id=operation_id,
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    writer_rows = [
        row for row in rows if row["operation_id"] == writer_operation_id
    ]
    assert {row["trace_id"] for row in rows} == {causal_root.trace_id}
    assert writer_rows
    assert all(row["parent_operation_id"] == operation_id for row in writer_rows)
    assert all(row["span_id"] == child_causal.span_id for row in writer_rows)
    assert all(row["parent_span_id"] == causal_root.span_id for row in writer_rows)
    assert writer_rows[-1]["terminal_status"] == "FINISHED"
    assert module.operation_causal_context(writer_operation_id) is None
    assert module.operation_causal_context(operation_id) is None
    assert module._OPERATION_CAUSAL.parent(writer_operation_id) is None


@pytest.mark.parametrize(
    ("event", "terminal_status"),
    [
        ("TRANSFORM_COMMIT", None),
        ("TRANSFORM_CANCEL", None),
        ("TRANSFORM_FAIL", None),
        ("SYNTHETIC_ERROR", "ERROR"),
    ],
)
def test_terminal_cleanup_survives_trace_persistence_failure(
    event,
    terminal_status,
    monkeypatch,
):
    module = _load_trace_module(monkeypatch)
    operation_id = "parent-op"
    child_operation_id = "child-op"
    module.bind_operation_causal_context(
        operation_id,
        module.new_trace_causal_root("terminal"),
    )
    assert module.link_trace_operation(child_operation_id, operation_id) is not None

    def fail_trace_path():
        raise OSError("synthetic trace path failure")

    monkeypatch.setattr(module, "_trace_path", fail_trace_path)
    resolved_status = (
        module.TraceTerminalStatus.ERROR if terminal_status == "ERROR" else None
    )
    module.trace_event(
        "OPERATION",
        event,
        operation_id=operation_id,
        terminal_status=resolved_status,
    )

    assert module.operation_causal_context(operation_id) is None
    assert module.operation_causal_context(child_operation_id) is None
    assert module._OPERATION_CAUSAL.parent(child_operation_id) is None


def test_runtime_error_correlation_drops_stale_last_trace(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_trace_module(monkeypatch)
    error_path = tmp_path / "runtime_errors.jsonl"
    monkeypatch.setattr(module, "_runtime_errors_path", lambda: error_path)
    monkeypatch.setattr(module, "_context_snapshot", lambda _context: {})
    module._LAST_TRACE_RECORD = {
        "session_id": module._SESSION_ID,
        "seq": 41,
        "event": "OLD_EVENT",
        "trace_id": "trace:old",
        "span_id": "span:old",
        "parent_span_id": None,
        "monotonic_ns": 1_000,
    }
    monkeypatch.setattr(
        module,
        "monotonic_ns",
        lambda: 1_000 + module._RUNTIME_ERROR_CAUSAL_MAX_AGE_NS + 1,
    )

    module._append_runtime_error_record(
        "sys.excepthook",
        "RuntimeError",
        "synthetic",
        "traceback",
    )

    record = json.loads(error_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["causal_correlation"]["status"] == "STALE"
    assert record["causal_correlation"]["candidate_seq"] == 41
    assert record["last_trace"]["trace_id"] is None
    assert record["last_observed_causal"]["trace_id"] is None


def test_contact_root_and_writer_child_share_one_causal_tree():
    source = (
        ROOT
        / "extension"
        / "blender_animation_workbench"
        / "phase4_contact_ui.py"
    ).read_text(encoding="utf-8")

    assert 'new_trace_causal_root("contact")' in source
    assert "bind_operation_causal_context(trace_operation_id, causal_root)" in source
    assert "link_trace_operation(writer_operation_id, trace_operation_id)" in source
    assert 'event_type=str(event.type)' not in source[source.index("def execute(self, context):"):]
    assert "terminal_status=TraceTerminalStatus.FINISHED" in source
    assert source.count("terminal_status=TraceTerminalStatus.CANCELLED") >= 3


def test_rigped_transform_roots_and_terminals_are_causally_tagged():
    source = (
        ROOT
        / "extension"
        / "blender_animation_workbench"
        / "rigped_transform.py"
    ).read_text(encoding="utf-8")

    assert source.count("bind_operation_causal_context(") >= 4
    for prefix in ("sliding-move", "fk-move", "direct-move", "direct-rotate"):
        assert prefix in source
    assert source.count("terminal_status=TraceTerminalStatus.FINISHED") == 4
    assert source.count("terminal_status=TraceTerminalStatus.CANCELLED") == 10
    assert source.count('lifecycle_phase="commit"') == 4
    assert source.count('lifecycle_phase="cancel"') == 8
    assert source.count('lifecycle_phase="fail"') == 2


def test_generic_selection_modal_has_explicit_ingress_route_and_terminal():
    source = (
        ROOT
        / "extension"
        / "blender_animation_workbench"
        / "viewport_keymap.py"
    ).read_text(encoding="utf-8")

    assert 'new_trace_causal_root("selection")' in source
    assert '"SELECTION_CLICK_INGRESS"' in source
    assert '"SELECTION_MODAL_STARTED"' in source
    assert '"SELECTION_MODAL_FINISHED"' in source
    assert '"SELECTION_MODAL_CANCELLED"' in source
    assert "TraceRouteOutcome.NATIVE_FALLTHROUGH" in source
    assert "terminal_status=TraceTerminalStatus.FINISHED" in source
    assert "terminal_status=TraceTerminalStatus.CANCELLED" in source
