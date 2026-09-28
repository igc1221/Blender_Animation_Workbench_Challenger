from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORM_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_transform.py"
SNAP_PATH = ROOT / "extension" / "blender_animation_workbench" / "phase4_representation_snap.py"
GEOMETRY_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_solver_geometry.py"
SLIDING_EVALUATION_PATH = (
    ROOT / "extension" / "blender_animation_workbench" / "rigped_sliding_evaluation.py"
)


def _source(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _function(path: Path, name: str) -> str:
    source, tree = _source(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"missing function {name} in {path}")


def _import_block(source: str, module: str) -> str:
    marker = f"from .{module} import ("
    start = source.index(marker)
    return source[start : source.index(")", start) + 1]


def test_h2_active_solver_stops_consuming_transient_snap_private_helpers() -> None:
    source, _tree = _source(TRANSFORM_PATH)
    snap_import = _import_block(source, "phase4_representation_snap")
    _source_text, tree = _source(TRANSFORM_PATH)

    for name in (
        "_canonical_generated_rigped_pole",
        "_derived_pole_angle",
        "_generated_lower_hinge_x_angle",
        "_generated_rigped_hinge_branch",
        "_initial_pole_solution",
    ):
        assert name not in snap_import

    geometry_aliases = {
        (alias.name, alias.asname)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "rigped_solver_geometry"
        for alias in node.names
    }
    assert geometry_aliases == {
        ("canonical_generated_rigped_pole", "_canonical_generated_rigped_pole"),
        ("derived_pole_angle", "_derived_pole_angle"),
        ("generated_lower_hinge_x_angle", "_generated_lower_hinge_x_angle"),
        ("generated_rigped_hinge_branch", "_generated_rigped_hinge_branch"),
        ("initial_pole_solution", "_initial_pole_solution"),
    }


def test_h2_shared_geometry_has_no_transient_snap_or_transform_import_cycle() -> None:
    _source_text, tree = _source(GEOMETRY_PATH)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "phase4_representation_snap" not in imported_modules
    assert "rigped_transform" not in imported_modules

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "LimbRepresentationCapability" not in imported_names


def test_h2_transient_snap_keeps_journal_restore_and_error_adapter_authority() -> None:
    source, _tree = _source(SNAP_PATH)
    execute = _function(SNAP_PATH, "execute_representation_snap")
    pole_adapter = _function(SNAP_PATH, "_derived_pole_angle")

    assert "MutationJournal(" in execute
    assert "BlenderRollbackExecutor()" in execute
    assert "validate_plan_fresh(" in execute
    assert "representation_payload_matches(" in execute
    assert "_generated_rigped_hinge_branch(capability)" in execute
    assert "_initial_pole_solution(capability)" in execute
    assert "_derived_pole_angle(" in execute
    assert "_canonical_generated_rigped_pole(" in execute
    assert "rigped_solver_geometry" in source
    assert "RigpedSolverGeometryError" in pole_adapter
    assert "raise RepresentationSnapError(str(exc)) from exc" in pole_adapter


def test_h2_b_separates_passive_projection_from_forearm_authored_overlay() -> None:
    display = _function(TRANSFORM_PATH, "_sliding_public_display_from_result")
    overlay = _function(TRANSFORM_PATH, "_compose_forearm_post_solve_roll_overlay")
    sync = _function(TRANSFORM_PATH, "_sync_sliding_public_pose_from_result")

    assert "derive_public_display(capability, solved_result)" in display
    assert "_compose_forearm_post_solve_roll_overlay(" in display
    assert display.index("derive_public_display(capability, solved_result)") < display.index(
        "_compose_forearm_post_solve_roll_overlay("
    )

    assert 'not in {"ForeArm.L", "ForeArm.R"}' in overlay
    assert "authored_lower_basis is None and not preserve_forearm_roll" in overlay
    assert "authored_hinge is None" in overlay
    assert overlay.count("return public_display") >= 2
    assert "_hinge_state_from_basis(" in overlay
    assert "_basis_with_local_y_twist(" in overlay
    assert "parent_pose_matrix=solved_result.first_pose" in overlay
    assert "solved_result.terminal_pose" in overlay
    assert "parent_pose_matrix=second_pose" in overlay
    assert "return replace(" in overlay
    assert "second_basis=second_basis" in overlay
    assert "terminal_basis=terminal_basis" in overlay

    for forbidden in (
        "keyframe_insert",
        "phase4_writer",
        "commit_rigped_auto",
        "execute_contact",
        "_apply_control_state(",
        "view_layer.update",
        "context.",
        "solved_result.",
    ):
        if forbidden == "solved_result.":
            continue
        assert forbidden not in overlay
    for mutation_token in (
        "solved_result.first_pose =",
        "solved_result.second_pose =",
        "solved_result.terminal_pose =",
        "capability.native_ik.",
        "capability.result_controls[",
        "capability.result_terminal.",
    ):
        assert mutation_token not in overlay

    snap_source = SNAP_PATH.read_text(encoding="utf-8")
    sliding_source = SLIDING_EVALUATION_PATH.read_text(encoding="utf-8")
    assert "_compose_forearm_post_solve_roll_overlay" not in snap_source
    assert "_compose_forearm_post_solve_roll_overlay" not in sliding_source

    assert sync.count("_apply_control_state(") == 3
    assert sync.count("location=False") == 3
    assert sync.count("rotation=True") == 3
    assert "authored_lower_basis=authored_lower_basis" in sync
    assert "preserve_forearm_roll=preserve_forearm_roll" in sync


def test_h2_seed_lifetime_remains_transform_owned_and_restore_disables_reseed() -> None:
    seed = _function(TRANSFORM_PATH, "_seed_stalled_sliding_native_ik")
    refresh = _function(TRANSFORM_PATH, "_refresh_current_sliding_public_overlays")
    restore = _function(TRANSFORM_PATH, "_restore_direct_move_states")

    assert "build_transient_solver_seed(" in seed
    assert "_seed_stalled_sliding_native_ik(context, capability)" in refresh
    assert "if allow_seed:" in refresh
    assert "_restore_sliding_hidden_seed_snapshots(" in restore
    assert "allow_seed=False" in restore
