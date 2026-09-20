from __future__ import annotations

import ast
from pathlib import Path

SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "rigped_transform.py"
)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _class(name: str) -> ast.ClassDef:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _source(node: ast.AST) -> str:
    segment = ast.get_source_segment(SOURCE, node)
    assert segment is not None
    return segment


def test_e5_rotate_uses_frozen_operation_domain_for_active_sliding() -> None:
    helper = _function("_direct_rotate_sliding_sync_sessions")
    source = _source(helper)
    assert "operation_domain: OperationDomainSnapshot" in source
    assert "operation_domain.contact_mapping_ids" in source
    assert "operation_domain.selected_binding_ids" in source
    assert "resolve_rigped_target(" not in source

    operator = _class("BAW_OT_rigped_direct_rotate_axis")
    invoke = next(
        node
        for node in operator.body
        if isinstance(node, ast.FunctionDef) and node.name == "invoke"
    )
    invoke_source = _source(invoke)
    assert "resolve_operation_domain(" in invoke_source
    assert "operation_domain = domain_resolution.snapshot" in invoke_source
    assert "_direct_rotate_sliding_sync_sessions(" in invoke_source
    assert "operation_domain," in invoke_source


def test_e5_active_passive_ownership_is_frozen_disjoint_and_traceable() -> None:
    passive = _source(_function("_passive_sliding_capabilities"))
    assert "current_sliding:" in passive
    assert "_current_sliding_capabilities(" not in passive

    operator = _class("BAW_OT_rigped_direct_rotate_axis")
    invoke = next(
        node
        for node in operator.body
        if isinstance(node, ast.FunctionDef) and node.name == "invoke"
    )
    source = _source(invoke)
    assert "frozen_current_sliding = _current_sliding_capabilities(context)" in source
    assert "active_ids & passive_ids" in source
    assert "active_ids | passive_ids != affected_ids" in source
    for field in (
        "sliding_sync_ids",
        "sliding_guard_ids",
        "sliding_affected_ids",
    ):
        assert field in source


def test_e5_cancel_restores_public_and_hidden_sliding_state() -> None:
    session = _class("SlidingRotateSyncSession")
    session_source = _source(session)
    assert "start_fk_states" in session_source
    assert "start_terminal_state" in session_source

    restore = _source(_function("_restore_direct_rotate_sliding_sync_session"))
    assert "session.start_ik_state" in restore
    assert "session.start_pole_state" in restore
    assert "session.start_pole_angle" in restore
    assert "session.start_hinge_settings" in restore
    assert "session.start_fk_states" in restore
    assert "session.start_terminal_state" in restore
    assert "session.start_ik_influence" in restore
    assert "session.start_terminal_ik_influence" in restore

    apply = _source(_function("_apply_direct_rotate_sliding_sync"))
    assert "_restore_direct_rotate_sliding_sync_session(context, session)" in apply


def test_e5_final_reconciliation_uses_frozen_affected_set() -> None:
    operator = _class("BAW_OT_rigped_direct_rotate_axis")
    modal = next(
        node
        for node in operator.body
        if isinstance(node, ast.FunctionDef) and node.name == "modal"
    )
    source = _source(modal)
    marker = "Final commit reconciliation therefore covers"
    marker_index = source.index(marker)
    tail = source[marker_index : marker_index + 700]
    assert "_refresh_current_sliding_public_overlays(" in tail
    assert "capabilities=self._sliding_affected_capabilities" in tail
