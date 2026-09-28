"""H3 replay ownership contracts around the frozen R1/R2 observations."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "extension" / "blender_animation_workbench"
REPLAY = ast.parse((PACKAGE / "debug_replay.py").read_text(encoding="utf-8"))
PAYLOAD = ast.parse((PACKAGE / "debug_replay_payload.py").read_text(encoding="utf-8"))


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_name(child) == name
    ]


def _rotate_entry() -> ast.FunctionDef:
    functions = _functions(REPLAY)
    runner = functions["run_semantic_replay"]
    branches = [
        branch
        for branch in ast.walk(runner)
        if isinstance(branch, ast.If)
        and isinstance(branch.test, ast.Compare)
        and any(
            isinstance(value, ast.Constant) and value.value == "ROTATE"
            for value in ast.walk(branch.test)
        )
    ]
    assert len(branches) == 1
    targets = {
        _call_name(call)
        for statement in branches[0].body
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and _call_name(call) in functions
    }
    assert len(targets) == 1, "ROTATE dispatch has one replay execution entry"
    return functions[targets.pop()]


def _rotate_graph() -> dict[str, ast.FunctionDef]:
    functions = _functions(REPLAY)
    found: dict[str, ast.FunctionDef] = {}
    pending = [_rotate_entry().name]
    while pending:
        name = pending.pop()
        if name in found or name not in functions:
            continue
        function = functions[name]
        found[name] = function
        pending.extend(
            _call_name(call)
            for call in ast.walk(function)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        )
    return found


def _live_validator() -> ast.FunctionDef:
    candidates = [
        function
        for function in _rotate_graph().values()
        if _calls(function, "validate_direct_rotate_recorded_result_payload")
    ]
    assert len(candidates) == 1
    return candidates[0]


def _recorded_apply() -> ast.FunctionDef:
    validator = _live_validator()
    candidates = [
        function
        for function in _rotate_graph().values()
        if _calls(function, validator.name)
    ]
    assert len(candidates) == 1
    return candidates[0]


def _top_level_step(function: ast.FunctionDef, call_name: str) -> int:
    matches = [
        index
        for index, statement in enumerate(function.body)
        if _calls(statement, call_name)
    ]
    assert len(matches) == 1, f"expected one {call_name} step in {function.name}"
    return matches[0]


def _has_state_assignment(node: ast.AST) -> bool:
    return any(
        isinstance(target, ast.Attribute)
        for child in ast.walk(node)
        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (child.targets if isinstance(child, ast.Assign) else [child.target])
    ) or bool(_calls(node, "setattr") or _calls(node, "delattr"))


def _production_writer_owner() -> ast.FunctionDef:
    candidates = [
        function
        for function in _rotate_graph().values()
        if _calls(function, "commit_rigped_auto_anchor")
        and _calls(function, "commit_rigped_auto_direct_rotate")
    ]
    assert len(candidates) == 1
    return candidates[0]


def test_v1_rotate_modes_are_explicit_and_dispatch_to_distinct_semantics() -> None:
    replay_mode = next(
        node for node in REPLAY.body if isinstance(node, ast.ClassDef) and node.name == "ReplayMode"
    )
    values = {
        target.id: statement.value.value
        for statement in replay_mode.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Constant)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    assert values["COMMAND"] == "COMMAND"
    assert values["RECORDED_RESULT"] == "RECORDED_RESULT"

    runner = _functions(REPLAY)["run_semantic_replay"]
    mode_index = [arg.arg for arg in runner.args.kwonlyargs].index("replay_mode")
    assert runner.args.kw_defaults[mode_index] is None
    assert _calls(runner, "ReplayMode")
    rotate = _rotate_entry()
    assert any(
        keyword.arg == "replay_mode"
        for call in _calls(runner, rotate.name)
        for keyword in call.keywords
    )
    mode_tests = [
        ast.unparse(branch.test)
        for branch in ast.walk(rotate)
        if isinstance(branch, ast.If)
    ]
    assert any("ReplayMode.RECORDED_RESULT" in test for test in mode_tests)
    assert any("ReplayMode.COMMAND" in test for test in mode_tests)
    recorded_branches = [
        branch
        for branch in rotate.body
        if isinstance(branch, ast.If)
        and "ReplayMode.RECORDED_RESULT" in ast.unparse(branch.test)
        and any(_calls(statement, _recorded_apply().name) for statement in branch.body)
    ]
    assert len(recorded_branches) == 1
    assert not any(
        _calls(statement, "_apply_control_state")
        for statement in recorded_branches[0].body
    )
    graph = _rotate_graph()
    assert any(_calls(function, _recorded_apply().name) for function in graph.values())
    assert any(_calls(function, "_apply_control_state") for function in graph.values())
    assert any(
        _calls(function, "_apply_direct_rotate_sliding_sync")
        for function in graph.values()
    )


def test_v2_recorded_result_validation_precedes_pose_and_writer_mutation() -> None:
    live = _live_validator()
    apply = _recorded_apply()
    rotate = _rotate_entry()
    assert _top_level_step(live, "validate_direct_rotate_recorded_result_payload") < (
        _top_level_step(live, "_bind_recorded_pose_state")
    )
    assert not _has_state_assignment(live)
    assert not any(
        _calls(live, writer)
        for writer in (
            "commit_rigped_auto_anchor",
            "commit_rigped_auto_contact_batch",
            "commit_rigped_auto_direct_rotate",
        )
    )
    assert _top_level_step(apply, live.name) < min(
        index
        for index, statement in enumerate(apply.body)
        if _has_state_assignment(statement)
    )
    apply_step = _top_level_step(rotate, apply.name)
    for writer in (
        "commit_rigped_auto_anchor",
        "commit_rigped_auto_contact_batch",
        "commit_rigped_auto_direct_rotate",
    ):
        assert apply_step < _top_level_step(rotate, writer)


def test_v3_w3_pure_validator_remains_the_structural_schema_owner() -> None:
    imported = {
        alias.name
        for node in REPLAY.body
        if isinstance(node, ast.ImportFrom) and node.module == "debug_replay_payload"
        for alias in node.names
    }
    assert "validate_direct_rotate_recorded_result_payload" in imported
    assert _calls(_live_validator(), "validate_direct_rotate_recorded_result_payload")
    assert "validate_direct_rotate_recorded_result_payload" in _functions(PAYLOAD)
    assert "validate_direct_rotate_recorded_result_payload" not in _functions(REPLAY)

    forbidden_imports = {
        "bpy",
        "debug_replay",
        "rigped_transform",
        "rigped_auto_key",
        "phase4_contact_authoring",
        "phase4_writer",
    }
    imported_modules = {
        node.module
        for node in PAYLOAD.body
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in PAYLOAD.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imported_modules & forbidden_imports
    assert not any(
        _calls(PAYLOAD, name)
        for name in (
            "keyframe_insert",
            "commit_rigped_auto_anchor",
            "commit_rigped_auto_direct_rotate",
            "execute_contact_intent_plan",
        )
    )


def test_v4_recorded_pose_is_auto_independent_and_storage_uses_auto_writers() -> None:
    rotate = _rotate_entry()
    auto_guard = [
        branch
        for branch in rotate.body
        if isinstance(branch, ast.If)
        and "baw_auto_key_enabled" in ast.unparse(branch.test)
    ]
    assert len(auto_guard) == 1
    assert _calls(auto_guard[0], "plan_rigped_auto_anchor")
    assert _calls(auto_guard[0], "plan_rigped_auto_direct_rotate")
    recorded_step = _top_level_step(rotate, _recorded_apply().name)
    assert recorded_step > rotate.body.index(auto_guard[0])
    assert all(
        _top_level_step(rotate, writer) > recorded_step
        for writer in (
            "commit_rigped_auto_anchor",
            "commit_rigped_auto_direct_rotate",
        )
    )
    production_imports = {
        alias.name
        for node in REPLAY.body
        if isinstance(node, ast.ImportFrom) and node.module == "rigped_auto_key"
        for alias in node.names
    }
    assert {"commit_rigped_auto_anchor", "commit_rigped_auto_direct_rotate"} <= (
        production_imports
    )
    writer_owner = _production_writer_owner()
    for writer in (
        "commit_rigped_auto_anchor",
        "commit_rigped_auto_contact_batch",
        "commit_rigped_auto_direct_rotate",
    ):
        guards = [
            branch
            for branch in writer_owner.body
            if isinstance(branch, ast.If)
            and any(_calls(statement, writer) for statement in branch.body)
        ]
        assert len(guards) == 1
        assert isinstance(guards[0].test, ast.Compare)
        assert any(isinstance(operator, ast.IsNot) for operator in guards[0].test.ops)
        assert any(
            isinstance(value, ast.Constant) and value.value is None
            for value in guards[0].test.comparators
        )


def test_v5_r2_mixed_rotate_keeps_anchor_before_real_direct_commit() -> None:
    writer_owner = _production_writer_owner()
    assert _top_level_step(writer_owner, "commit_rigped_auto_anchor") < (
        _top_level_step(writer_owner, "commit_rigped_auto_direct_rotate")
    )
    direct_call = _calls(writer_owner, "commit_rigped_auto_direct_rotate")
    assert len(direct_call) == 1
    assert not any(
        keyword.arg == "defer_commit" and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in direct_call[0].keywords
    )
    assert not any(
        "rollback" in _call_name(call).lower()
        for call in ast.walk(writer_owner)
        if isinstance(call, ast.Call)
    )


def test_v6_replay_rotate_does_not_own_modal_preview_cancel_or_native_undo() -> None:
    forbidden = (
        "modal",
        "_apply_preview",
        "_restore_preview",
        "_restore_direct_rotate_states",
        "_restore_direct_move_states",
        "undo_push",
        "gizmo",
    )
    for function in _rotate_graph().values():
        called = {
            _call_name(call).lower()
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
        }
        assert not any(part in name for name in called for part in forbidden)


def test_v7_replay_rotate_has_no_duplicate_native_storage_writer() -> None:
    for function in _rotate_graph().values():
        for call in ast.walk(function):
            if not isinstance(call, ast.Call):
                continue
            assert _call_name(call) not in {
                "keyframe_insert",
                "keyframe_delete",
                "animation_data_create",
            }
            callee = ast.unparse(call.func).lower()
            assert not any(
                storage in callee and callee.endswith(suffix)
                for storage in ("fcurves", "channelbags", "keyframe_points")
                for suffix in (".new", ".insert", ".remove")
            )
        assert not any(
            isinstance(target, (ast.Attribute, ast.Subscript))
            and any(
                storage in ast.unparse(target).lower()
                for storage in ("fcurves", "channelbags", "keyframe_points")
            )
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
