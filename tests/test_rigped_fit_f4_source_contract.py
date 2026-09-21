from pathlib import Path

ROOT = Path(__file__).parents[1]
EXT = ROOT / "extension" / "blender_animation_workbench"


def _source(name: str) -> str:
    return (EXT / name).read_text(encoding="utf-8")


def test_f4_dirty_apply_routes_through_atomic_transaction_only():
    ui = _source("rigped_create_fit_ui.py")
    commit = _source("rigped_fit_commit.py")

    assert "dirty_semantic_draft" in ui
    assert "commit_fit_semantic_session_atomic(context, semantic_session)" in ui
    assert "Fit Apply blocked until semantic draft commit is implemented (F4)" not in ui
    assert "def commit_fit_semantic_session_atomic" in commit
    assert "validate_fit_semantic_session(context, semantic_session)" in commit
    assert "FIT_F4_SESSION_STALE" in commit
    assert "FIT_F4_GESTURE_ACTIVE" in commit
    assert "compute_fit_commit_manifest(context.scene, semantic_session)" in commit
    assert "capture_fit_native_before_image(context.scene, semantic_session)" in commit
    assert "FIT_F4_BEFORE_IMAGE_STALE" in commit


def test_f4_manifest_covers_primary_owned_carriers_targets_contacts_and_poles():
    commit = _source("rigped_fit_commit.py")

    assert '_DIRECT_PREFIXES = ("MCH_", "DEF_", "IK_", "EXP_")' in commit
    assert 'if usage == "PRIMARY":' in commit
    assert 'if usage == "POLE":' in commit
    assert "def _mapped_endpoint" in commit
    assert "def _derived_correspondence_target" in commit
    assert "FIT_F4_ENDPOINT_CORRESPONDENCE_AMBIGUOUS" in commit
    assert "FIT_F4_DERIVED_TARGET_UNRESOLVED" in commit
    assert "FIT_F4_PRIMARY_TARGET_MISSING" in commit


def test_f4_before_image_owns_every_native_field_that_commit_mutates():
    commit = _source("rigped_fit_commit.py")

    for field in (
        "head:",
        "tail:",
        "roll:",
        "bbone_x:",
        "bbone_z:",
        "parent_name:",
        "use_connect:",
    ):
        assert field in commit
    assert "descriptor_exists:" in commit
    assert "descriptor_value:" in commit
    assert "constraint_signature:" in commit
    assert "animation_signature:" in commit
    assert "object_matrix_signature:" in commit
    assert "pose_signature:" in commit
    assert "FIT_F4_OBJECT_MATRIX_CHANGED" in commit
    assert "FIT_F4_POSE_CHANGED" in commit
    assert "FIT_F4_ROLLBACK_OBJECT_MATRIX_MISMATCH" in commit
    assert "FIT_F4_ROLLBACK_POSE_MISMATCH" in commit
    assert "pole_angle" in commit


def test_f4_internal_failure_uses_explicit_restore_not_blender_undo():
    commit = _source("rigped_fit_commit.py")

    block = commit[
        commit.index("def commit_fit_semantic_session_atomic") :
    ]
    assert "restore_fit_native_before_image(context.scene, semantic_session, before)" in block
    assert "bpy.ops.ed.undo" not in block
    assert "FIT_F4_ROLLBACK_FAILED" in block
    assert "_restore_edit_bones(rig, before.edit_bones)" in commit
    assert "FIT_F4_ROLLBACK_REST_MISMATCH" in commit
    assert "FIT_F4_ROLLBACK_DESCRIPTOR_PRESENCE_MISMATCH" in commit
    assert "FIT_F4_ROLLBACK_DESCRIPTOR_MISMATCH" in commit
    assert "_verify_restored_before_image(scene, semantic_session, before)" in commit


def test_f4_publish_happens_only_after_native_verify():
    commit = _source("rigped_fit_commit.py")
    block = commit[
        commit.index("def commit_fit_semantic_session_atomic") :
    ]

    apply_pos = block.index("_apply_manifest")
    verify_pos = block.index("_verify_manifest")
    publish_pos = block.index("publish_fit_session_after_native_commit")
    assert apply_pos < verify_pos < publish_pos


def test_f4_constraints_and_animation_are_read_only_in_commit_transaction():
    commit = _source("rigped_fit_commit.py")

    assert 'raise FitCommitTransactionError("FIT_F4_CONSTRAINT_CHANGED")' in commit
    assert 'raise FitCommitTransactionError("FIT_F4_ANIMATION_CHANGED")' in commit
    assert ".constraints.new(" not in commit
    assert ".constraints.remove(" not in commit
