from __future__ import annotations

import math
import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_fit_commands_tests"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package

fit_spec = spec_from_file_location(
    f"{PACKAGE_NAME}.rigped_fit_state",
    PACKAGE_PATH / "rigped_fit_state.py",
)
assert fit_spec is not None and fit_spec.loader is not None
fit = module_from_spec(fit_spec)
sys.modules[fit_spec.name] = fit
fit_spec.loader.exec_module(fit)

commands_spec = spec_from_file_location(
    f"{PACKAGE_NAME}.rigped_fit_commands",
    PACKAGE_PATH / "rigped_fit_commands.py",
)
assert commands_spec is not None and commands_spec.loader is not None
commands = module_from_spec(commands_spec)
sys.modules[commands_spec.name] = commands
commands_spec.loader.exec_module(commands)

FitCommandError = commands.FitCommandError
fit_move_supported = commands.fit_move_supported
fit_rotate_supported = commands.fit_rotate_supported
fit_scale_supported = commands.fit_scale_supported
fit_operations_for_role = commands.fit_operations_for_role
move_fit_part_rig_local = commands.move_fit_part_rig_local
rotate_fit_part_rig_local = commands.rotate_fit_part_rig_local
scale_fit_part_local = commands.scale_fit_part_local
FitOperation = fit.FitOperation
FitPartKind = fit.FitPartKind
FitRestPartSnapshot = fit.FitRestPartSnapshot
derive_rest_parts = fit.derive_rest_parts
extract_fit_document = fit.extract_fit_document


def _draft():
    _document, draft = extract_fit_document(
        character_id="char",
        profile_id="rigped",
        schema_version=1,
        setup_revision=1,
        setup_signature="setup",
        owner_data_identity=("obj", "data"),
        animation_footprint_digest="anim",
        snapshots=(
            FitRestPartSnapshot(
                binding_id="root",
                semantic_key="awb.root",
                side="CENTER",
                parent_binding_id=None,
                connected=False,
                head=(0.0, 0.0, 0.0),
                tail=(0.0, 1.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                width=0.1,
                depth=0.1,
                name_hint="Root",
            ),
            FitRestPartSnapshot(
                binding_id="com",
                semantic_key="awb.com",
                side="CENTER",
                parent_binding_id="root",
                connected=False,
                head=(0.0, 0.2, 0.0),
                tail=(0.0, 0.7, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                width=0.2,
                depth=0.2,
                kind=FitPartKind.FRAME,
                allowed_operations=(FitOperation.MOVE, FitOperation.ROTATE, FitOperation.SCALE),
                name_hint="COM",
            ),
            FitRestPartSnapshot(
                binding_id="pelvis",
                semantic_key="awb.pelvis",
                side="CENTER",
                parent_binding_id="com",
                connected=False,
                head=(0.0, 0.2, 0.0),
                tail=(0.0, 0.8, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                width=0.2,
                depth=0.15,
                allowed_operations=(FitOperation.MOVE, FitOperation.ROTATE, FitOperation.SCALE),
                name_hint="Pelvis",
            ),
        ),
    )
    return draft


def _rest_map(draft):
    return {part.part_id: part for part in derive_rest_parts(draft)}


def test_com_move_translates_com_subtree_only():
    draft = _draft()
    before = _rest_map(draft)
    delta = (0.3, -0.2, 0.5)

    moved = move_fit_part_rig_local(draft, "com", delta)
    after = _rest_map(moved)

    assert after["root"].head == pytest.approx(before["root"].head)
    assert after["root"].tail == pytest.approx(before["root"].tail)
    assert after["com"].head == pytest.approx(
        tuple(before["com"].head[index] + delta[index] for index in range(3))
    )
    assert after["pelvis"].head == pytest.approx(
        tuple(before["pelvis"].head[index] + delta[index] for index in range(3))
    )
    assert after["com"].orientation == pytest.approx(before["com"].orientation)
    assert after["pelvis"].orientation == pytest.approx(before["pelvis"].orientation)


def test_pelvis_move_is_explicit_consumed_noop():
    draft = _draft()

    assert fit_move_supported(draft, "com")
    assert fit_move_supported(draft, "pelvis")
    assert move_fit_part_rig_local(draft, "pelvis", (0.1, 0.0, 0.0)) is draft


def test_zero_delta_is_exact_noop():
    draft = _draft()

    assert move_fit_part_rig_local(draft, "com", (0.0, 0.0, 0.0)) is draft


def test_unsupported_root_move_fails_even_for_zero_delta():
    draft = _draft()

    with pytest.raises(FitCommandError, match="FIT_F3_MOVE_UNSUPPORTED_PART"):
        move_fit_part_rig_local(draft, "root", (0.0, 0.0, 0.0))


def test_com_rotate_keeps_center_fixed_and_rotates_subtree_only():
    draft = _draft()
    before = _rest_map(draft)
    before_values = {record.part_id: record.value for record in draft.values}
    half = math.sqrt(0.5)

    rotated = rotate_fit_part_rig_local(
        draft,
        "com",
        (half, 0.0, 0.0, half),
    )
    after = _rest_map(rotated)
    after_values = {record.part_id: record.value for record in rotated.values}

    before_center = tuple(
        (before["com"].head[index] + before["com"].tail[index]) * 0.5
        for index in range(3)
    )
    after_center = tuple(
        (after["com"].head[index] + after["com"].tail[index]) * 0.5
        for index in range(3)
    )

    assert after["root"].head == pytest.approx(before["root"].head)
    assert after["root"].tail == pytest.approx(before["root"].tail)
    assert after_center == pytest.approx(before_center)
    assert after["com"].head == pytest.approx((0.25, 0.45, 0.0))
    assert after["com"].tail == pytest.approx((-0.25, 0.45, 0.0))
    assert after["pelvis"].head == pytest.approx(after["com"].head)
    assert after["com"].orientation == pytest.approx((half, 0.0, 0.0, half))
    assert after["pelvis"].orientation == pytest.approx((half, 0.0, 0.0, half))
    assert after_values["root"] == before_values["root"]
    assert after_values["pelvis"] == before_values["pelvis"]


def test_pelvis_rotate_is_capability_enabled_but_root_is_not():
    draft = _draft()

    assert fit_rotate_supported(draft, "com")
    assert fit_rotate_supported(draft, "pelvis")
    assert not fit_rotate_supported(draft, "root")

    with pytest.raises(FitCommandError, match="FIT_F3_ROTATE_UNSUPPORTED_PART"):
        rotate_fit_part_rig_local(draft, "root", (1.0, 0.0, 0.0, 0.0))


def test_identity_rotate_is_exact_noop_after_capability_check():
    draft = _draft()

    assert rotate_fit_part_rig_local(
        draft,
        "com",
        (1.0, 0.0, 0.0, 0.0),
    ) is draft


def test_com_rotate_parent_local_conversion_with_rotated_parent():
    draft = _draft()
    half = math.sqrt(0.5)
    root_record = next(record for record in draft.values if record.part_id == "root")
    rotated_parent = fit.replace_draft_value(
        draft,
        "root",
        replace(
            root_record.value,
            local_orientation=(half, 0.0, 0.0, half),
        ),
    )
    before = _rest_map(rotated_parent)
    before_values = {record.part_id: record.value for record in rotated_parent.values}

    rotated = rotate_fit_part_rig_local(
        rotated_parent,
        "com",
        (half, 0.0, 0.0, half),
    )
    after = _rest_map(rotated)
    after_values = {record.part_id: record.value for record in rotated.values}

    before_center = tuple(
        (before["com"].head[index] + before["com"].tail[index]) * 0.5
        for index in range(3)
    )
    after_center = tuple(
        (after["com"].head[index] + after["com"].tail[index]) * 0.5
        for index in range(3)
    )

    assert before["root"].tail == pytest.approx((-1.0, 0.0, 0.0))
    assert after["root"] == before["root"]
    assert before_center == pytest.approx((-0.45, 0.0, 0.0))
    assert after_center == pytest.approx(before_center)
    assert after["com"].head == pytest.approx((-0.45, 0.25, 0.0))
    assert after["com"].tail == pytest.approx((-0.45, -0.25, 0.0))
    assert after_values["com"].attachment_offset == pytest.approx((0.25, 0.45, 0.0))
    assert after_values["com"].local_orientation == pytest.approx(
        (half, 0.0, 0.0, half)
    )
    assert after_values["pelvis"] == before_values["pelvis"]


def _leg_draft():
    _document, draft = extract_fit_document(
        character_id="char",
        profile_id="rigped",
        schema_version=1,
        setup_revision=1,
        setup_signature="setup",
        owner_data_identity=("obj", "data"),
        animation_footprint_digest="anim",
        snapshots=(
            FitRestPartSnapshot(
                binding_id="root",
                semantic_key="awb.root",
                side="CENTER",
                parent_binding_id=None,
                connected=False,
                head=(0.0, 0.0, 0.0),
                tail=(0.0, 1.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                width=0.1,
                depth=0.1,
                name_hint="Root",
            ),
            FitRestPartSnapshot(
                binding_id="com",
                semantic_key="awb.com",
                side="CENTER",
                parent_binding_id="root",
                connected=False,
                head=(0.0, 1.0, 0.0),
                tail=(0.0, 1.5, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                width=0.2,
                depth=0.2,
                kind=FitPartKind.FRAME,
                allowed_operations=fit_operations_for_role("awb.com"),
                name_hint="COM",
            ),
            FitRestPartSnapshot(
                binding_id="pelvis",
                semantic_key="awb.pelvis",
                side="CENTER",
                parent_binding_id="com",
                connected=False,
                head=(0.0, 1.0, 0.0),
                tail=(0.0, 1.6, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                width=0.3,
                depth=0.2,
                allowed_operations=fit_operations_for_role("awb.pelvis"),
                name_hint="Pelvis",
            ),
            FitRestPartSnapshot(
                binding_id="thigh_l",
                semantic_key="awb.thigh",
                side="LEFT",
                parent_binding_id="pelvis",
                connected=False,
                head=(0.5, 1.0, 0.0),
                tail=(0.5, 0.0, 0.0),
                orientation=(0.0, 1.0, 0.0, 0.0),
                width=0.15,
                depth=0.15,
                allowed_operations=fit_operations_for_role("awb.thigh"),
                name_hint="Thigh.L",
            ),
            FitRestPartSnapshot(
                binding_id="calf_l",
                semantic_key="awb.calf",
                side="LEFT",
                parent_binding_id="thigh_l",
                connected=True,
                head=(0.5, 0.0, 0.0),
                tail=(1.5, 0.0, 0.0),
                orientation=(
                    math.sqrt(0.5),
                    0.0,
                    0.0,
                    -math.sqrt(0.5),
                ),
                width=0.12,
                depth=0.12,
                allowed_operations=fit_operations_for_role("awb.calf"),
                name_hint="Calf.L",
            ),
            FitRestPartSnapshot(
                binding_id="thigh_r",
                semantic_key="awb.thigh",
                side="RIGHT",
                parent_binding_id="pelvis",
                connected=False,
                head=(-0.5, 1.0, 0.0),
                tail=(-0.5, 0.0, 0.0),
                orientation=(0.0, 1.0, 0.0, 0.0),
                width=0.15,
                depth=0.15,
                allowed_operations=fit_operations_for_role("awb.thigh"),
                name_hint="Thigh.R",
            ),
            FitRestPartSnapshot(
                binding_id="calf_r",
                semantic_key="awb.calf",
                side="RIGHT",
                parent_binding_id="thigh_r",
                connected=True,
                head=(-0.5, 0.0, 0.0),
                tail=(-0.5, -1.0, 0.0),
                orientation=(0.0, 1.0, 0.0, 0.0),
                width=0.12,
                depth=0.12,
                allowed_operations=fit_operations_for_role("awb.calf"),
                name_hint="Calf.R",
            ),
        ),
    )
    return draft


def _length(part):
    return math.sqrt(
        sum((float(part.tail[i]) - float(part.head[i])) ** 2 for i in range(3))
    )


def test_role_capability_matrix_is_explicit_and_unknown_fails_closed():
    assert fit_operations_for_role("awb.com") == (
        FitOperation.MOVE,
        FitOperation.ROTATE,
        FitOperation.SCALE,
    )
    assert fit_operations_for_role(
        "awb.spine",
        parent_semantic_key="awb.spine",
    ) == (
        FitOperation.MOVE,
        FitOperation.ROTATE,
        FitOperation.SCALE,
    )
    assert fit_operations_for_role("awb.spine") == (
        FitOperation.ROTATE,
        FitOperation.SCALE,
    )
    assert fit_operations_for_role("awb.unknown") == ()


def test_com_scale_uses_local_axes_without_native_object_scale_semantics():
    draft = _draft()
    before = _rest_map(draft)
    before_values = {record.part_id: record.value for record in draft.values}

    scaled = scale_fit_part_local(draft, "com", "XYZ", 1.5)
    after = _rest_map(scaled)
    after_values = {record.part_id: record.value for record in scaled.values}

    assert after["com"].head == pytest.approx(before["com"].head)
    assert _length(after["com"]) == pytest.approx(_length(before["com"]) * 1.5)
    assert after["com"].width == pytest.approx(before["com"].width * 1.5)
    assert after["com"].depth == pytest.approx(before["com"].depth * 1.5)
    tail_delta = tuple(
        after["com"].tail[i] - before["com"].tail[i] for i in range(3)
    )
    assert after["pelvis"].head == pytest.approx(
        tuple(before["pelvis"].head[i] + tail_delta[i] for i in range(3))
    )
    assert after["pelvis"].tail == pytest.approx(
        tuple(before["pelvis"].tail[i] + tail_delta[i] for i in range(3))
    )
    assert after_values["pelvis"] != before_values["pelvis"]
    assert after_values["com"].frame_length_scale == pytest.approx(1.5)


def test_pelvis_x_scale_moves_leg_sockets_rigidly_and_preserves_lengths():
    draft = _leg_draft()
    before = _rest_map(draft)
    scaled = scale_fit_part_local(draft, "pelvis", "X", 1.5)
    after = _rest_map(scaled)

    assert after["pelvis"].width == pytest.approx(before["pelvis"].width * 1.5)
    assert after["thigh_l"].head[0] == pytest.approx(0.75)
    assert after["thigh_r"].head[0] == pytest.approx(-0.75)
    for part_id in ("thigh_l", "calf_l", "thigh_r", "calf_r"):
        assert _length(after[part_id]) == pytest.approx(_length(before[part_id]))
    assert tuple(
        after["calf_l"].head[i] - before["calf_l"].head[i] for i in range(3)
    ) == pytest.approx((0.25, 0.0, 0.0))
    assert tuple(
        after["calf_r"].head[i] - before["calf_r"].head[i] for i in range(3)
    ) == pytest.approx((-0.25, 0.0, 0.0))


def test_structural_y_scale_reanchors_connected_descendants_without_length_loss():
    draft = _leg_draft()
    before = _rest_map(draft)
    scaled = scale_fit_part_local(draft, "thigh_l", "Y", 1.25)
    after = _rest_map(scaled)

    assert _length(after["thigh_l"]) == pytest.approx(
        _length(before["thigh_l"]) * 1.25
    )
    assert after["calf_l"].head == pytest.approx(after["thigh_l"].tail)
    assert _length(after["calf_l"]) == pytest.approx(_length(before["calf_l"]))
    assert after["calf_l"].orientation == pytest.approx(
        before["calf_l"].orientation
    )


def test_two_bone_calf_move_preserves_segment_lengths():
    draft = _leg_draft()
    before = _rest_map(draft)
    moved = move_fit_part_rig_local(draft, "calf_l", (0.2, 0.15, 0.0))
    after = _rest_map(moved)

    assert _length(after["thigh_l"]) == pytest.approx(_length(before["thigh_l"]))
    assert _length(after["calf_l"]) == pytest.approx(_length(before["calf_l"]))
    assert after["calf_l"].head == pytest.approx(after["thigh_l"].tail)
    assert after["calf_l"].tail != pytest.approx(before["calf_l"].tail)


def test_pelvis_rotation_is_isolated_from_leg_absolute_pose():
    draft = _leg_draft()
    before = _rest_map(draft)
    half = math.sqrt(0.5)

    rotated = rotate_fit_part_rig_local(
        draft,
        "pelvis",
        (half, 0.0, 0.0, half),
    )
    after = _rest_map(rotated)

    assert after["pelvis"].orientation != pytest.approx(
        before["pelvis"].orientation
    )
    for part_id in ("thigh_l", "calf_l", "thigh_r", "calf_r"):
        assert after[part_id].head == pytest.approx(before[part_id].head)
        assert after[part_id].tail == pytest.approx(before[part_id].tail)
