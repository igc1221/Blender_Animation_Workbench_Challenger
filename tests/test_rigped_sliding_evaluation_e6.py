from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = (
    ROOT
    / "extension"
    / "blender_animation_workbench"
    / "rigped_sliding_evaluation.py"
)
TRANSFORM_PATH = (
    ROOT
    / "extension"
    / "blender_animation_workbench"
    / "rigped_transform.py"
)
SERVICE_SOURCE = SERVICE_PATH.read_text(encoding="utf-8")
TRANSFORM_SOURCE = TRANSFORM_PATH.read_text(encoding="utf-8")
SERVICE_TREE = ast.parse(SERVICE_SOURCE)
TRANSFORM_TREE = ast.parse(TRANSFORM_SOURCE)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _source(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def test_e6_service_separates_reference_seed_result_and_public_display() -> None:
    for name in (
        "SlidingAuthoredReference",
        "SlidingSolverSeed",
        "SlidingSolvedResult",
        "SlidingPublicDisplay",
    ):
        _class(SERVICE_TREE, name)

    derive = _source(
        SERVICE_SOURCE,
        _function(SERVICE_TREE, "derive_public_display"),
    )
    assert "solved_result: SlidingSolvedResult" in derive
    assert "SlidingSolverSeed" not in derive
    assert "capture_sliding_authored_reference" not in derive


def test_e6_service_is_writer_free() -> None:
    forbidden = (
        "phase4_writer",
        "execute_representation_snap",
        "keyframe_insert",
        "keyframe_delete",
        ".fcurves",
        "channelbag",
        "animation_data_create",
    )
    lowered = SERVICE_SOURCE.lower()
    for token in forbidden:
        assert token.lower() not in lowered


def test_e6_native_conversion_uses_blender_bone_conversion() -> None:
    inverse = _source(
        SERVICE_SOURCE,
        _function(SERVICE_TREE, "native_pose_basis_from_matrix"),
    )
    assert "convert_local_to_pose(" in inverse
    assert "invert=True" in inverse
    assert "parent_matrix=parent_pose" in inverse
    assert "parent_matrix_local=pose_bone.parent.bone.matrix_local" in inverse
    assert "parent_pose.inverted" not in inverse

    forward = _source(
        SERVICE_SOURCE,
        _function(SERVICE_TREE, "native_pose_matrix_from_basis"),
    )
    assert "convert_local_to_pose(" in forward
    assert "invert=True" not in forward
    assert "parent_matrix=parent_pose" in forward
    assert "parent_matrix_local=pose_bone.parent.bone.matrix_local" in forward


def test_e6_seed_is_pure_and_keeps_authored_target_separate() -> None:
    seed = _source(
        SERVICE_SOURCE,
        _function(SERVICE_TREE, "build_transient_solver_seed"),
    )
    assert "target_world: Vector" in seed
    assert "seed_end =" in seed
    assert "target_world=tuple(float(value) for value in target)" in seed
    assert "end_world=tuple(float(value) for value in seed_end)" in seed
    for token in (
        ".location =",
        ".matrix_basis =",
        ".influence =",
        ".mute =",
        "keyframe",
    ):
        assert token not in seed


def test_e6_public_result_paths_use_native_conversion_service() -> None:
    state = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_state_for_pose_matrix"),
    )
    replay = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_apply_pose_bone_rotation_from_matrix"),
    )
    sync = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_sync_sliding_public_pose_from_result"),
    )

    assert "native_pose_basis_from_matrix(" in state
    assert "parent_rest" not in state
    assert "native_pose_basis_from_matrix(" in replay
    assert "parent_rest" not in replay
    assert "capture_native_solved_result(capability)" in sync
    assert "derive_public_display(capability, solved_result)" in sync
    assert "public_display.first_basis" in sync
    assert "public_display.second_basis" in sync
    assert "public_display.terminal_basis" in sync


def test_e6_sliding_seed_is_transient_before_native_ik_update() -> None:
    apply = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "apply_sliding_semantic_move_delta"),
    )
    assert "build_transient_solver_seed(" in apply
    assert "Vector(solver_seed.joint_world)" in apply
    assert "Vector(solver_seed.end_world)" in apply
    assert "native_ik.mute = True" in apply
    assert "native_ik.mute = was_muted" in apply
    assert "context.view_layer.update()" in apply
    assert apply.index("native_ik.mute = was_muted") < apply.rindex(
        "context.view_layer.update()"
    )
