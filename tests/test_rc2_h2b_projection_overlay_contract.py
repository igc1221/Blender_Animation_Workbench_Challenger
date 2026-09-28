"""H2B ownership contracts for passive projection and authored ForeArm roll."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "extension" / "blender_animation_workbench"
TRANSFORM = ast.parse((PACKAGE / "rigped_transform.py").read_text(encoding="utf-8"))
EVALUATION = ast.parse(
    (PACKAGE / "rigped_sliding_evaluation.py").read_text(encoding="utf-8")
)
GEOMETRY = ast.parse(
    (PACKAGE / "rigped_solver_geometry.py").read_text(encoding="utf-8")
)


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return _functions(tree)[name]


def _call_name(call: ast.Call) -> str:
    callee = call.func
    if isinstance(callee, ast.Name):
        return callee.id
    if isinstance(callee, ast.Attribute):
        return callee.attr
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_name(child) == name
    ]


def _attributes(node: ast.AST, root: str) -> set[str]:
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == root
    }


def _reachable_transform_functions(start: str) -> dict[str, ast.FunctionDef]:
    functions = _functions(TRANSFORM)
    found: dict[str, ast.FunctionDef] = {}
    pending = [start]
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


def _overlay_function() -> ast.FunctionDef:
    reachable = _reachable_transform_functions("_sync_sliding_public_pose_from_result")
    candidates = [
        function
        for function in reachable.values()
        if _calls(function, "_hinge_state_from_basis")
        and _calls(function, "_basis_with_local_y_twist")
    ]
    assert len(candidates) == 1, "one transform-local authored roll composition path"
    return candidates[0]


def _assert_no_authoring_authority(function: ast.FunctionDef) -> None:
    forbidden_exact = {
        "keyframe_insert",
        "keyframe_delete",
        "MutationJournal",
        "undo_push",
        "setattr",
        "delattr",
    }
    forbidden_fragments = (
        "contact",
        "auto",
        "writer",
        "journal",
        "undo",
        "replay",
    )
    names = {
        _call_name(call)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
    }
    assert not names & forbidden_exact
    assert not {
        name
        for name in names
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    }
    assert not _calls(function, "update"), "projection must not advance depsgraph"
    assert not _calls(function, "_apply_control_state"), "projection must not write controls"
    assert not any(
        isinstance(target, (ast.Attribute, ast.Subscript))
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
    ), "projection must not assign Blender state"


def test_v1_native_result_projection_is_passive() -> None:
    derive = _function(EVALUATION, "derive_public_display")
    assert {"first_pose", "second_pose", "terminal_pose"} <= _attributes(
        derive, "solved_result"
    )
    assert len(_calls(derive, "native_pose_basis_from_matrix")) == 3
    assert len(_calls(derive, "SlidingPublicDisplay")) == 1
    assert not _calls(derive, "_hinge_state_from_basis")
    assert not _calls(derive, "_basis_with_local_y_twist")
    assert not {
        child.value
        for child in ast.walk(derive)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    } & {"ForeArm.L", "ForeArm.R"}
    _assert_no_authoring_authority(derive)


def test_v2_forearm_overlay_stays_transform_local_and_rebuilds_terminal() -> None:
    overlay = _overlay_function()
    assert overlay.name not in _functions(EVALUATION)
    guarded_names = {
        child.value
        for branch in ast.walk(overlay)
        if isinstance(branch, ast.If)
        for child in ast.walk(branch.test)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }
    assert {"ForeArm.L", "ForeArm.R"} <= guarded_names
    assert _calls(overlay, "_hinge_state_from_basis")
    twists = _calls(overlay, "_basis_with_local_y_twist")
    assert any("second_basis" in _attributes(call, "public_display") for call in twists)
    pose_assignments = {
        target.id
        for statement in ast.walk(overlay)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and _call_name(statement.value) == "_pose_matrix_from_basis"
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    terminal_rebuilds = [
        call
        for call in _calls(overlay, "native_pose_basis_from_matrix")
        if "terminal_pose" in _attributes(call, "solved_result")
        and any(
            keyword.arg == "parent_pose_matrix"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id in pose_assignments
            for keyword in call.keywords
        )
    ]
    assert terminal_rebuilds, "terminal basis follows composed lower pose"
    _assert_no_authoring_authority(overlay)


def test_v3_sync_owns_three_rotation_only_public_writes_and_update() -> None:
    sync = _function(TRANSFORM, "_sync_sliding_public_pose_from_result")
    reachable = _reachable_transform_functions(sync.name)
    assert any(_calls(function, "derive_public_display") for function in reachable.values())
    assert _overlay_function().name in reachable
    display_functions = [
        function
        for function in reachable.values()
        if _calls(function, "derive_public_display")
    ]
    assert all(
        not _calls(function, "_apply_control_state")
        and not _calls(function, "update")
        for function in display_functions
    )
    for function in display_functions:
        if function is not sync:
            _assert_no_authoring_authority(function)
    assert len(_calls(sync, "capture_native_solved_result")) == 1
    writes = _calls(sync, "_apply_control_state")
    assert len(writes) == 3
    capture_step = next(
        index
        for index, statement in enumerate(sync.body)
        if _calls(statement, "capture_native_solved_result")
    )
    first_write_step = next(
        index
        for index, statement in enumerate(sync.body)
        if _calls(statement, "_apply_control_state")
    )
    projection_names = {function.name for function in display_functions}
    projection_names.add("derive_public_display")
    projection_step = next(
        index
        for index, statement in enumerate(sync.body)
        if any(
            isinstance(call, ast.Call) and _call_name(call) in projection_names
            for call in ast.walk(statement)
        )
    )
    last_write_step = max(
        index
        for index, statement in enumerate(sync.body)
        if _calls(statement, "_apply_control_state")
    )
    update_step = next(
        index
        for index, statement in enumerate(sync.body)
        if isinstance(statement, ast.If) and _calls(statement, "update")
    )
    assert capture_step < projection_step < first_write_step < update_step
    assert last_write_step < update_step
    assert all(
        {
            keyword.arg: keyword.value.value
            for keyword in call.keywords
            if isinstance(keyword.value, ast.Constant)
        }.items()
        >= {"location": False, "rotation": True}.items()
        for call in writes
    )
    assert len(_calls(sync, "_state_for_pose_basis")) == 2
    assert len(_calls(sync, "_state_for_pose_matrix")) == 1
    assert any(
        isinstance(branch.test, ast.Name)
        and branch.test.id == "update"
        and _calls(branch, "update")
        for branch in ast.walk(sync)
        if isinstance(branch, ast.If)
    )
    assert not _calls(_overlay_function(), "_apply_control_state")


def test_v4_seed_lifetime_and_restore_no_reseed_boundary() -> None:
    seed = _function(TRANSFORM, "_seed_stalled_sliding_native_ik")
    refresh = _function(TRANSFORM, "_refresh_current_sliding_public_overlays")
    restore = _function(TRANSFORM, "_restore_direct_move_states")
    assert _calls(seed, "build_transient_solver_seed")
    assert not _calls(_overlay_function(), "build_transient_solver_seed")
    assert any(
        isinstance(branch.test, ast.Name)
        and branch.test.id == "allow_seed"
        and len(_calls(branch, seed.name)) == 1
        for branch in ast.walk(refresh)
        if isinstance(branch, ast.If)
    )
    assert _calls(restore, "_restore_sliding_hidden_seed_snapshots")
    assert any(
        keyword.arg == "allow_seed"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for call in _calls(restore, refresh.name)
        for keyword in call.keywords
    )


def test_v5_shared_geometry_does_not_return_to_transient_snap() -> None:
    shared = {
        "canonical_generated_rigped_pole",
        "derived_pole_angle",
        "generated_lower_hinge_x_angle",
        "generated_rigped_hinge_branch",
        "initial_pole_solution",
    }
    transform_imports = {
        (node.module, alias.name)
        for node in TRANSFORM.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {("rigped_solver_geometry", name) for name in shared} <= transform_imports
    assert not {("phase4_representation_snap", name) for name in shared} & transform_imports
    assert shared <= _functions(GEOMETRY).keys()
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module in {"rigped_transform", "phase4_representation_snap"}
        for node in GEOMETRY.body
    )


def test_v6_projection_and_overlay_have_no_writer_or_replay_authority() -> None:
    _assert_no_authoring_authority(_function(EVALUATION, "derive_public_display"))
    _assert_no_authoring_authority(_overlay_function())
