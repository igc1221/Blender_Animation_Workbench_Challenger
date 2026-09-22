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
    source = ast.get_source_segment(SOURCE, function)
    assert source is not None
    assert "target_bone = pose.get(target_name)" in source
    assert "rig.matrix_world @ target_bone.matrix.translation" in source
