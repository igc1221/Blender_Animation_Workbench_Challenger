from __future__ import annotations

import importlib.util
import sys
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
    assert 'if terminal_status is not None and operation_id is not None:' in source


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
