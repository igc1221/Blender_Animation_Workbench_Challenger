from __future__ import annotations

import ast
from pathlib import Path

SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "rigped_ik_pivot_overlay.py"
)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

EXPECTED_LIMB_TARGETS = (
    ("MCH_ForeArm.L", "IK_Hand.L"),
    ("MCH_ForeArm.R", "IK_Hand.R"),
    ("MCH_Calf.L", "IK_Foot.L"),
    ("MCH_Calf.R", "IK_Foot.R"),
)


def _assignment_literal(name: str):
    for node in TREE.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name}")


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def test_e8_red_cue_maps_state_owner_to_hidden_ik_target_exactly() -> None:
    assert _assignment_literal("_LIMB_TARGETS") == EXPECTED_LIMB_TARGETS


def test_e8_red_cue_never_uses_solved_terminal_as_target() -> None:
    targets = {target for _state_owner, target in EXPECTED_LIMB_TARGETS}
    assert targets == {"IK_Hand.L", "IK_Hand.R", "IK_Foot.L", "IK_Foot.R"}
    assert not (targets & {"MCH_Hand.L", "MCH_Hand.R", "MCH_Foot.L", "MCH_Foot.R"})


def test_e8_overlay_uses_target_pose_position_in_world_space() -> None:
    function = _function("_ik_pivot_world_positions")

    target_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "target_bone"
            for target in node.targets
        )
    )
    target_call = target_assignment.value
    assert isinstance(target_call, ast.Call)
    assert isinstance(target_call.func, ast.Attribute)
    assert isinstance(target_call.func.value, ast.Name)
    assert target_call.func.value.id == "pose"
    assert target_call.func.attr == "get"
    assert len(target_call.args) == 1
    assert isinstance(target_call.args[0], ast.Name)
    assert target_call.args[0].id == "target_name"

    world_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "world"
            for target in node.targets
        )
    )
    world_expr = world_assignment.value
    assert isinstance(world_expr, ast.BinOp)
    assert isinstance(world_expr.op, ast.MatMult)
    assert isinstance(world_expr.left, ast.Attribute)
    assert isinstance(world_expr.left.value, ast.Name)
    assert world_expr.left.value.id == "rig"
    assert world_expr.left.attr == "matrix_world"
    assert isinstance(world_expr.right, ast.Attribute)
    assert world_expr.right.attr == "translation"
    pose_matrix = world_expr.right.value
    assert isinstance(pose_matrix, ast.Attribute)
    assert isinstance(pose_matrix.value, ast.Name)
    assert pose_matrix.value.id == "target_bone"
    assert pose_matrix.attr == "matrix"
