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
GIZMO_PATH = (
    ROOT
    / "extension"
    / "blender_animation_workbench"
    / "global_transform_gizmo.py"
)
VIEWPORT_KEYMAP_PATH = (
    ROOT
    / "extension"
    / "blender_animation_workbench"
    / "viewport_keymap.py"
)
REPRESENTATION_SNAP_PATH = (
    ROOT
    / "extension"
    / "blender_animation_workbench"
    / "phase4_representation_snap.py"
)
CONTACT_AUTHORING_PATH = (
    ROOT
    / "extension"
    / "blender_animation_workbench"
    / "phase4_contact_authoring.py"
)
SERVICE_SOURCE = SERVICE_PATH.read_text(encoding="utf-8")
TRANSFORM_SOURCE = TRANSFORM_PATH.read_text(encoding="utf-8")
GIZMO_SOURCE = GIZMO_PATH.read_text(encoding="utf-8")
VIEWPORT_KEYMAP_SOURCE = VIEWPORT_KEYMAP_PATH.read_text(encoding="utf-8")
REPRESENTATION_SNAP_SOURCE = REPRESENTATION_SNAP_PATH.read_text(encoding="utf-8")
CONTACT_AUTHORING_SOURCE = CONTACT_AUTHORING_PATH.read_text(encoding="utf-8")
SERVICE_TREE = ast.parse(SERVICE_SOURCE)
TRANSFORM_TREE = ast.parse(TRANSFORM_SOURCE)
GIZMO_TREE = ast.parse(GIZMO_SOURCE)
VIEWPORT_KEYMAP_TREE = ast.parse(VIEWPORT_KEYMAP_SOURCE)
REPRESENTATION_SNAP_TREE = ast.parse(REPRESENTATION_SNAP_SOURCE)
CONTACT_AUTHORING_TREE = ast.parse(CONTACT_AUTHORING_SOURCE)


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


def test_e6_body_overlay_refresh_reseeds_only_stalled_reachable_straight_ik() -> None:
    reseed = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_seed_stalled_sliding_native_ik"),
    )
    refresh = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_refresh_current_sliding_public_overlays"),
    )

    assert "evaluated_chain_tip_world_position(capability.native_ik)" in reseed
    assert "target_distance >= total_length - reach_epsilon" in reseed
    assert "current_perp.length >" in reseed
    assert "build_transient_solver_seed(" in reseed
    assert "native_ik.mute = True" in reseed
    assert "native_ik.mute = was_muted" in reseed
    assert "first_control=capability.result_controls[0]" in reseed
    assert "second_control=capability.result_controls[1]" in reseed
    assert "terminal_control=capability.result_terminal" in reseed
    assert "_configured_generated_hinge_branch_sign(" in reseed
    assert "_transient_solver_seed_branch_sign(" in reseed
    assert "preferred_bend.copy()" in reseed
    assert "-preferred_bend.copy()" in reseed
    assert "candidate_branch != configured_branch" in reseed
    assert "capability.native_ik.ik_target.target.location =" not in reseed
    assert "pole_target.target.location =" not in reseed
    assert refresh.count("_seed_stalled_sliding_native_ik(context, capability)") == 1
    assert refresh.index("_seed_stalled_sliding_native_ik(context, capability)") < (
        refresh.index("for _pass_index in range")
    )
    assert refresh.index("_seed_stalled_sliding_native_ik(context, capability)") < (
        refresh.index("_sync_sliding_public_pose_from_result(")
    )
    assert "Require one no-write stability" in refresh
    assert "context.view_layer.update()" in refresh
    assert "if stable:" in refresh
    assert "set_limb_fk_feedback_muted(capability, True)" in refresh
    assert "_sync_generated_sliding_hinge_branch_from_pole(" in refresh
    assert refresh.index("_sync_generated_sliding_hinge_branch_from_pole(") < refresh.index(
        "_seed_stalled_sliding_native_ik(context, capability)"
    )


def test_e6_sliding_runtime_mutes_public_fk_feedback_without_changing_influence() -> None:
    helper = _source(
        REPRESENTATION_SNAP_SOURCE,
        _function(REPRESENTATION_SNAP_TREE, "set_limb_fk_feedback_muted"),
    )
    single = _source(
        CONTACT_AUTHORING_SOURCE,
        _function(CONTACT_AUTHORING_TREE, "execute_contact_intent_plan"),
    )
    batch = _source(
        CONTACT_AUTHORING_SOURCE,
        _function(CONTACT_AUTHORING_TREE, "execute_contact_batch_intent_plan"),
    )
    replay = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_sync_rigped_sliding_replay_display"),
    )
    snap = _source(
        REPRESENTATION_SNAP_SOURCE,
        _function(REPRESENTATION_SNAP_TREE, "execute_representation_snap"),
    )

    assert "constraint.mute = target" in helper
    assert "constraint.influence =" not in helper
    assert "intent.target_type is not ContactKeyType.FREE" in single
    assert "item.intent.target_type is not ContactKeyType.FREE" in batch
    assert "contact_type is not ContactKeyType.FREE" in replay
    assert "_sync_generated_sliding_hinge_branch_from_pole(state_bone)" in replay
    branch_sync = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_sync_generated_sliding_hinge_branch_from_pole"),
    )
    assert 'name.startswith("MCH_ForeArm")' in branch_sync
    assert "MCH_Calf" not in branch_sync
    assert '"restore FK feedback for Free authority"' in snap
    assert "RawFieldKind.CONSTRAINT_MUTE" in snap


def test_e6_persistent_move_tool_reroutes_after_selection_changes() -> None:
    route = _source(
        GIZMO_SOURCE,
        _function(GIZMO_TREE, "_route_and_mode"),
    )

    assert 'state in {"DIRECT_MOVE", "FK_MOVE", "MOVE"}' in route
    assert route.index("direct_move_available(context)") < route.index(
        "fk_joint_move_available(context)"
    )
    assert route.index("fk_joint_move_available(context)") < route.index(
        "semantic_move_available(context)"
    )
    move_group = route[route.index('state in {"DIRECT_MOVE", "FK_MOVE", "MOVE"}') :]
    assert 'return "DIRECT_MOVE", "MOVE"' in move_group
    assert 'return "FK_MOVE", "MOVE"' in move_group
    assert 'return "SEMANTIC_MOVE", "MOVE"' in move_group


def test_e6_move_operator_polls_accept_persistent_w_intent() -> None:
    expected_modes = '{"MOVE", "FK_MOVE", "DIRECT_MOVE"}'
    for class_name, availability in (
        ("BAW_OT_rigped_semantic_move_axis", "semantic_move_available(context)"),
        ("BAW_OT_rigped_fk_joint_move_axis", "fk_joint_move_available(context)"),
        ("BAW_OT_rigped_direct_move_axis", "direct_move_available(context)"),
    ):
        source = _source(TRANSFORM_SOURCE, _class(TRANSFORM_TREE, class_name))
        assert expected_modes in source
        assert availability in source


def test_e6_orientation_cycle_preserves_active_semantic_route() -> None:
    helper = _source(
        VIEWPORT_KEYMAP_SOURCE,
        _function(
            VIEWPORT_KEYMAP_TREE,
            "_cycle_transform_orientation_preserving_semantic_mode",
        ),
    )
    operator = _source(
        VIEWPORT_KEYMAP_SOURCE,
        _class(VIEWPORT_KEYMAP_TREE, "BAW_OT_set_transform_tool"),
    )

    assert "_cycle_transform_orientation(context)" in helper
    assert "scene.baw_rigped_semantic_transform_mode = str(semantic_mode)" in helper
    assert (
        "_cycle_transform_orientation_preserving_semantic_mode("
        in operator
    )
    assert 'orientation_after=next_orientation' in operator


def test_e6_direct_move_mousemove_errors_reach_flight_recorder() -> None:
    source = _source(
        TRANSFORM_SOURCE,
        _class(TRANSFORM_TREE, "BAW_OT_rigped_direct_move_axis"),
    )

    assert "_report_operator_error(" in source
    assert 'phase="MOUSEMOVE"' in source
    assert 'route="DIRECT_MOVE"' in source
    assert "_set_semantic_move_drag_active(context, False)" in source


def test_e6_sliding_overlay_convergence_gates_only_writable_rotation() -> None:
    source = _source(
        TRANSFORM_SOURCE,
        _class(TRANSFORM_TREE, "BAW_OT_rigped_direct_rotate_axis"),
    )
    refresh = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_refresh_current_sliding_public_overlays"),
    )
    converge = _source(
        TRANSFORM_SOURCE,
        _function(TRANSFORM_TREE, "_converge_sliding_public_pose_from_result"),
    )

    assert "max_passes=12" not in source
    assert "position residual is observed for diagnostics" in refresh
    assert "if rotation > rotation_tolerance:" in refresh
    assert "position >" not in refresh
    assert "if last_rotation <= rotation_tolerance:" in converge
    assert "last_position <=" not in converge
    assert "Require one no-write stability" in refresh
    assert '"SLIDING_CONVERGENCE_FAILURE"' in refresh
    assert "residual_history=tuple(residual_history)" in refresh
    assert 'phase="MOUSEMOVE"' in source
    assert 'attempted_angle=float(candidate)' in source
