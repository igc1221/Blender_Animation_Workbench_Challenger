from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "trackbar_key_edit.py"
)
SPEC = spec_from_file_location("baw_trackbar_key_edit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
trackbar_key_edit = module_from_spec(SPEC)
SPEC.loader.exec_module(trackbar_key_edit)


class _Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _Key:
    def __init__(self, frame: float, value: float):
        self.co = _Vector(frame, value)
        self.handle_left = _Vector(frame - 0.25, value)
        self.handle_right = _Vector(frame + 0.25, value)
        self.handle_left_type = "FREE"
        self.handle_right_type = "FREE"
        self.interpolation = "BEZIER"
        self.easing = "AUTO"
        self.amplitude = 0.0
        self.back = 0.0
        self.period = 0.0
        self.type = "KEYFRAME"
        self.select_control_point = False
        self.select_left_handle = False
        self.select_right_handle = False


class _FCurve:
    def __init__(self, data_path: str, *keys: _Key, array_index: int = 0):
        self.data_path = data_path
        self.array_index = array_index
        self.keyframe_points = list(keys)
        self.update_count = 0

    def update(self):
        self.update_count += 1


class _Points(list):
    def remove(self, value, *, fast=False):
        super().remove(value)

    def insert(self, frame, value, *, options=None, keyframe_type="KEYFRAME"):
        key = _Key(frame, value)
        key.type = keyframe_type
        self.append(key)
        return key


def test_delete_frame_preserves_other_frames_and_controls():
    curves = [
        _FCurve('pose.bones["Hand.L"].location', _Key(10, 1), _Key(20, 2)),
        _FCurve('pose.bones["Hand.L"].scale', _Key(10, 3)),
        _FCurve('pose.bones["Foot.L"].location', _Key(10, 4)),
    ]
    for curve in curves:
        curve.keyframe_points = _Points(curve.keyframe_points)
    assert trackbar_key_edit.delete_keyframe_points(
        curves, 10, data_path_prefix='pose.bones["Hand.L"]',
    )
    assert [[key.co.x for key in c.keyframe_points] for c in curves] == [[20], [], [10]]
    assert [c.update_count for c in curves] == [1, 1, 0]
    assert not trackbar_key_edit.delete_keyframe_points(curves, 99)


def test_move_keyframe_points_shifts_all_channels_and_handles():
    location = _Key(21.0, 1.0)
    rotation = _Key(21.0, 2.0)
    curves = [_FCurve("location", location), _FCurve("rotation_euler", rotation)]

    assert trackbar_key_edit.move_keyframe_points(curves, 21, 24)

    assert location.co.x == 24
    assert location.handle_left.x == 23.75
    assert location.handle_right.x == 24.25
    assert rotation.co.x == 24
    assert all(curve.update_count == 1 for curve in curves)


def test_move_keyframe_points_replaces_same_fcurve_destination():
    source = _Key(21.0, 1.0)
    occupied = _Key(24.0, 2.0)
    curve = _FCurve("location", source, occupied)
    curve.keyframe_points = _Points(curve.keyframe_points)

    assert trackbar_key_edit.move_keyframe_points([curve], 21, 24)
    assert source.co.x == 24
    assert source.co.y == 1.0
    assert occupied not in curve.keyframe_points
    assert list(curve.keyframe_points) == [source]
    assert curve.update_count == 1


def test_move_keyframe_points_replaces_only_matching_source_channels():
    source = _Key(21.0, 1.0)
    occupied_same_curve = _Key(24.0, 2.0)
    location = _FCurve(
        'pose.bones["Hand.L"].location',
        source,
        occupied_same_curve,
    )
    location.keyframe_points = _Points(location.keyframe_points)

    destination_only_channel = _Key(24.0, 4.0)
    rotation = _FCurve(
        'pose.bones["Hand.L"].rotation_euler',
        destination_only_channel,
    )
    unrelated = _FCurve('pose.bones["Foot.L"].location', _Key(24.0, 3.0))

    assert trackbar_key_edit.move_keyframe_points(
        [location, rotation, unrelated],
        21,
        24,
        data_path_prefix='pose.bones["Hand.L"]',
    )
    assert source.co.x == 24
    assert occupied_same_curve not in location.keyframe_points
    assert destination_only_channel.co.x == 24
    assert unrelated.keyframe_points[0].co.x == 24
    assert location.update_count == 1
    assert rotation.update_count == 0
    assert unrelated.update_count == 0


def test_clone_keyframe_points_preserves_source_and_key_shape():
    source = _Key(21.0, 3.5)
    source.interpolation = "LINEAR"
    source.handle_left_type = "VECTOR"
    source.handle_right_type = "FREE"
    source.handle_left = _Vector(20.5, 3.0)
    source.handle_right = _Vector(21.75, 4.0)
    curve = _FCurve("location", source, array_index=2)
    curve.keyframe_points = _Points(curve.keyframe_points)

    assert trackbar_key_edit.clone_keyframe_points([curve], 21, 24)
    by_frame = {round(key.co.x): key for key in curve.keyframe_points}
    assert sorted(by_frame) == [21, 24]
    assert by_frame[21] is source
    clone = by_frame[24]
    assert clone.co.y == 3.5
    assert clone.interpolation == "LINEAR"
    assert clone.handle_left_type == "VECTOR"
    assert clone.handle_right_type == "FREE"
    assert clone.handle_left.x == 23.5
    assert clone.handle_left.y == 3.0
    assert clone.handle_right.x == 24.75
    assert clone.handle_right.y == 4.0
    assert curve.update_count == 1


def test_clone_keyframe_points_replaces_only_same_source_fcurves():
    source = _Key(21.0, 1.0)
    occupied = _Key(24.0, 9.0)
    location = _FCurve("location", source, occupied)
    location.keyframe_points = _Points(location.keyframe_points)

    destination_only = _Key(24.0, 7.0)
    rotation = _FCurve("rotation_euler", destination_only)
    rotation.keyframe_points = _Points(rotation.keyframe_points)

    assert trackbar_key_edit.clone_keyframe_points([location, rotation], 21, 24)
    location_by_frame = {round(key.co.x): key for key in location.keyframe_points}
    assert sorted(location_by_frame) == [21, 24]
    assert location_by_frame[21] is source
    assert location_by_frame[24].co.y == 1.0
    assert occupied not in location.keyframe_points
    assert list(rotation.keyframe_points) == [destination_only]
    assert location.update_count == 1
    assert rotation.update_count == 0


def test_remove_clone_preview_preserves_source_and_destination_only_channels():
    source = _Key(21.0, 1.0)
    clone = _Key(24.0, 1.0)
    location = _FCurve("location", source, clone)
    location.keyframe_points = _Points(location.keyframe_points)

    destination_only = _Key(24.0, 5.0)
    rotation = _FCurve("rotation_euler", destination_only)
    rotation.keyframe_points = _Points(rotation.keyframe_points)

    assert trackbar_key_edit.remove_cloned_keyframe_points([location, rotation], 21, 24)
    assert list(location.keyframe_points) == [source]
    assert list(rotation.keyframe_points) == [destination_only]
    assert location.update_count == 1
    assert rotation.update_count == 0


def test_snapshot_and_exact_restore_roundtrip_after_preview_mutation():
    first = _Key(10.0, 1.0)
    first.select_control_point = True
    first.select_left_handle = True
    first.select_right_handle = True
    second = _Key(20.0, 2.0)
    second.interpolation = "LINEAR"
    curve = _FCurve("location", first, second, array_index=2)
    curve.keyframe_points = _Points(curve.keyframe_points)

    snapshots = trackbar_key_edit.snapshot_keyframe_points([curve])
    assert [snapshot.frame for snapshot in snapshots] == [10.0, 20.0]

    assert trackbar_key_edit.move_keyframe_points([curve], 10, 15)
    assert trackbar_key_edit.clone_keyframe_points([curve], 20, 25)
    assert sorted(round(key.co.x) for key in curve.keyframe_points) == [15, 20, 25]

    assert trackbar_key_edit.restore_keyframe_state([curve], snapshots)
    restored = {round(key.co.x): key for key in curve.keyframe_points}
    assert sorted(restored) == [10, 20]
    assert restored[10].co.y == 1.0
    assert restored[10].select_control_point
    assert restored[10].select_left_handle
    assert restored[10].select_right_handle
    assert restored[20].co.y == 2.0
    assert restored[20].interpolation == "LINEAR"


def test_delete_keyframe_points_at_frames_preserves_unrelated_controls():
    hand_location = _FCurve(
        'pose.bones["Hand.L"].location',
        _Key(10, 1),
        _Key(20, 2),
        _Key(30, 3),
    )
    hand_rotation = _FCurve(
        'pose.bones["Hand.L"].rotation_euler',
        _Key(10, 4),
        _Key(30, 5),
    )
    foot_location = _FCurve('pose.bones["Foot.L"].location', _Key(10, 6), _Key(20, 7))
    curves = [hand_location, hand_rotation, foot_location]
    for curve in curves:
        curve.keyframe_points = _Points(curve.keyframe_points)

    assert trackbar_key_edit.delete_keyframe_points_at_frames(
        curves,
        [10, 20],
        data_path_prefix='pose.bones["Hand.L"]',
    )
    assert [[round(key.co.x) for key in curve.keyframe_points] for curve in curves] == [
        [30],
        [30],
        [10, 20],
    ]


def test_multi_clone_is_simultaneous_when_sources_overlap_destinations():
    first = _Key(10.0, 1.0)
    second = _Key(15.0, 2.0)
    occupied = _Key(20.0, 9.0)
    for key in (first, second):
        key.select_control_point = True
        key.select_left_handle = True
        key.select_right_handle = True
    curve = _FCurve("location", first, second, occupied)
    curve.keyframe_points = _Points(curve.keyframe_points)

    assert trackbar_key_edit.clone_keyframe_points_at_frames([curve], [10, 15], 5)

    by_frame = {round(key.co.x): key for key in curve.keyframe_points}
    assert sorted(by_frame) == [10, 15, 20]
    assert by_frame[10].co.y == 1.0
    assert by_frame[15].co.y == 2.0
    assert by_frame[20].co.y == 2.0
    assert not by_frame[10].select_control_point
    assert by_frame[15].select_control_point
    assert by_frame[20].select_control_point
    assert occupied not in curve.keyframe_points


def test_multi_move_is_simultaneous_when_sources_overlap_destinations():
    first = _Key(10.0, 1.0)
    second = _Key(20.0, 2.0)
    occupied = _Key(30.0, 9.0)
    for key in (first, second):
        key.select_control_point = True
        key.select_left_handle = True
        key.select_right_handle = True
    curve = _FCurve("location", first, second, occupied)
    curve.keyframe_points = _Points(curve.keyframe_points)

    assert trackbar_key_edit.move_keyframe_points_at_frames([curve], [10, 20], 10)

    by_frame = {round(key.co.x): key for key in curve.keyframe_points}
    assert sorted(by_frame) == [20, 30]
    assert by_frame[20].co.y == 1.0
    assert by_frame[30].co.y == 2.0
    assert by_frame[20].select_control_point
    assert by_frame[30].select_control_point
    assert occupied not in curve.keyframe_points


def test_multi_move_preview_reapplies_from_pristine_sources_and_restores_exactly():
    first = _Key(10.0, 1.0)
    second = _Key(20.0, 2.0)
    occupied = _Key(30.0, 9.0)
    for key in (first, second):
        key.select_control_point = True
        key.select_left_handle = True
        key.select_right_handle = True
    curve = _FCurve("location", first, second, occupied)
    curve.keyframe_points = _Points(curve.keyframe_points)

    txn = trackbar_key_edit.snapshot_multi_key_preview_transaction(
        [curve], [10, 20], mode="MOVE"
    )
    assert txn is not None
    assert trackbar_key_edit.apply_multi_key_preview_transaction([curve], txn, -1)
    assert sorted(round(key.co.x) for key in curve.keyframe_points) == [9, 19, 30]

    assert trackbar_key_edit.apply_multi_key_preview_transaction([curve], txn, 3)
    by_frame = {round(key.co.x): key for key in curve.keyframe_points}
    assert sorted(by_frame) == [13, 23, 30]
    assert by_frame[13].co.y == 1.0
    assert by_frame[23].co.y == 2.0
    assert by_frame[30].co.y == 9.0

    assert trackbar_key_edit.restore_multi_key_preview_transaction([curve], txn)
    by_frame = {round(key.co.x): key for key in curve.keyframe_points}
    assert sorted(by_frame) == [10, 20, 30]
    assert by_frame[10].co.y == 1.0
    assert by_frame[20].co.y == 2.0
    assert by_frame[30].co.y == 9.0
    assert by_frame[10].select_control_point
    assert by_frame[20].select_control_point


def test_multi_clone_preview_protects_overlapping_selected_source():
    first = _Key(10.0, 1.0)
    second = _Key(15.0, 2.0)
    occupied = _Key(20.0, 9.0)
    for key in (first, second):
        key.select_control_point = True
        key.select_left_handle = True
        key.select_right_handle = True
    curve = _FCurve("location", first, second, occupied)
    curve.keyframe_points = _Points(curve.keyframe_points)

    txn = trackbar_key_edit.snapshot_multi_key_preview_transaction(
        [curve], [10, 15], mode="CLONE"
    )
    assert txn is not None
    assert trackbar_key_edit.apply_multi_key_preview_transaction([curve], txn, 5)
    by_frame = {round(key.co.x): key for key in curve.keyframe_points}
    assert sorted(by_frame) == [10, 15, 20]
    assert by_frame[10].co.y == 1.0
    assert by_frame[15].co.y == 2.0
    assert by_frame[20].co.y == 2.0
    assert not by_frame[10].select_control_point
    assert by_frame[15].select_control_point
    assert by_frame[20].select_control_point

    assert trackbar_key_edit.restore_multi_key_preview_transaction([curve], txn)
    by_frame = {round(key.co.x): key for key in curve.keyframe_points}
    assert sorted(by_frame) == [10, 15, 20]
    assert by_frame[10].co.y == 1.0
    assert by_frame[15].co.y == 2.0
    assert by_frame[20].co.y == 9.0


def test_multi_move_preserves_unrelated_frames_and_controls():
    hand = _FCurve(
        'pose.bones["Hand.L"].location',
        _Key(10, 1),
        _Key(20, 2),
        _Key(50, 5),
    )
    foot = _FCurve('pose.bones["Foot.L"].location', _Key(10, 7), _Key(20, 8))
    for curve in (hand, foot):
        curve.keyframe_points = _Points(curve.keyframe_points)

    assert trackbar_key_edit.move_keyframe_points_at_frames(
        [hand, foot],
        [10, 20],
        5,
        data_path_prefix='pose.bones["Hand.L"]',
    )
    assert sorted(round(key.co.x) for key in hand.keyframe_points) == [15, 25, 50]
    assert [round(key.co.x) for key in foot.keyframe_points] == [10, 20]
    assert foot.update_count == 0


def test_collision_snapshot_restores_overwritten_destination_key():
    source = _Key(21.0, 1.0)
    source.handle_left = _Vector(20.6, 0.8)
    source.handle_right = _Vector(21.4, 1.2)
    destination = _Key(24.0, 9.0)
    destination.interpolation = "LINEAR"
    destination.handle_left_type = "VECTOR"
    destination.handle_right_type = "VECTOR"
    destination.handle_left = _Vector(23.5, 8.0)
    destination.handle_right = _Vector(24.5, 10.0)
    curve = _FCurve("location", source, destination, array_index=2)
    curve.keyframe_points = _Points(curve.keyframe_points)

    snapshots = trackbar_key_edit.snapshot_replaced_keyframe_points([curve], 21, 24)
    assert len(snapshots) == 1
    assert snapshots[0].value == 9.0
    assert snapshots[0].array_index == 2

    assert trackbar_key_edit.move_keyframe_points([curve], 21, 24)
    assert [key.co.y for key in curve.keyframe_points] == [1.0]

    assert trackbar_key_edit.move_keyframe_points([curve], 24, 21)
    assert trackbar_key_edit.restore_keyframe_snapshots([curve], snapshots)

    by_frame = {round(key.co.x): key for key in curve.keyframe_points}
    assert sorted(by_frame) == [21, 24]
    assert by_frame[21].co.y == 1.0
    restored = by_frame[24]
    assert restored.co.y == 9.0
    assert restored.interpolation == "LINEAR"
    assert restored.handle_left_type == "VECTOR"
    assert restored.handle_right_type == "VECTOR"
    assert restored.handle_left.x == 23.5
    assert restored.handle_right.x == 24.5


def _range_curve(*keys: _Key) -> _FCurve:
    curve = _FCurve("location", *keys, array_index=2)
    curve.keyframe_points = _Points(curve.keyframe_points)
    return curve


def _keys_by_frame(curve: _FCurve) -> dict[int, _Key]:
    return {round(key.co.x): key for key in curve.keyframe_points}


def test_selection_range_scale_right_expansion_scales_key_and_handle_x_only():
    left = _Key(10.0, 1.0)
    middle = _Key(15.0, 5.0)
    right = _Key(20.0, 3.0)
    middle.handle_left = _Vector(14.0, 4.25)
    middle.handle_right = _Vector(16.0, 5.75)
    middle.interpolation = "LINEAR"
    curve = _range_curve(left, middle, right)

    txn = trackbar_key_edit.snapshot_selection_range_scale_transaction(
        [curve],
        [10, 15, 20],
        pivot_frame=10,
        source_handle_frame=20,
    )
    assert txn is not None
    assert trackbar_key_edit.apply_selection_range_scale_transaction([curve], txn, 30)

    by_frame = _keys_by_frame(curve)
    assert sorted(by_frame) == [10, 20, 30]
    scaled_middle = by_frame[20]
    assert scaled_middle.co.y == 5.0
    assert scaled_middle.handle_left.x == 18.0
    assert scaled_middle.handle_left.y == 4.25
    assert scaled_middle.handle_right.x == 22.0
    assert scaled_middle.handle_right.y == 5.75
    assert scaled_middle.interpolation == "LINEAR"
    assert all(key.select_control_point for key in by_frame.values())


def test_selection_range_scale_left_expansion_and_exact_restore():
    left = _Key(10.0, 1.0)
    middle = _Key(15.0, 2.0)
    middle.handle_left = _Vector(14.5, 1.5)
    middle.handle_right = _Vector(15.5, 2.5)
    right = _Key(20.0, 3.0)
    curve = _range_curve(left, middle, right)

    txn = trackbar_key_edit.snapshot_selection_range_scale_transaction(
        [curve],
        [10, 15, 20],
        pivot_frame=20,
        source_handle_frame=10,
    )
    assert txn is not None
    assert trackbar_key_edit.apply_selection_range_scale_transaction([curve], txn, 0)
    assert sorted(_keys_by_frame(curve)) == [0, 10, 20]

    assert trackbar_key_edit.restore_selection_range_scale_transaction([curve], txn)
    restored = _keys_by_frame(curve)
    assert sorted(restored) == [10, 15, 20]
    assert restored[15].co.y == 2.0
    assert restored[15].handle_left.x == 14.5
    assert restored[15].handle_left.y == 1.5
    assert restored[15].handle_right.x == 15.5
    assert restored[15].handle_right.y == 2.5


def test_selection_range_scale_down_rounds_to_snapped_integer_frames():
    curve = _range_curve(_Key(10, 1), _Key(14, 2), _Key(20, 3))
    txn = trackbar_key_edit.snapshot_selection_range_scale_transaction(
        [curve],
        [10, 14, 20],
        pivot_frame=10,
        source_handle_frame=20,
    )
    assert txn is not None
    assert trackbar_key_edit.apply_selection_range_scale_transaction([curve], txn, 15)
    assert sorted(_keys_by_frame(curve)) == [10, 12, 15]
    assert txn.selected_destination_frames == (10, 12, 15)


def test_selection_range_scale_replaces_non_source_same_fcurve_destination():
    source_left = _Key(10, 1)
    source_right = _Key(20, 2)
    occupied = _Key(30, 99)
    curve = _range_curve(source_left, source_right, occupied)

    txn = trackbar_key_edit.snapshot_selection_range_scale_transaction(
        [curve],
        [10, 20],
        pivot_frame=10,
        source_handle_frame=20,
    )
    assert txn is not None
    assert trackbar_key_edit.apply_selection_range_scale_transaction([curve], txn, 30)
    by_frame = _keys_by_frame(curve)
    assert sorted(by_frame) == [10, 30]
    assert by_frame[30].co.y == 2.0

    assert trackbar_key_edit.restore_selection_range_scale_transaction([curve], txn)
    restored = _keys_by_frame(curve)
    assert sorted(restored) == [10, 20, 30]
    assert restored[30].co.y == 99.0


def test_selection_range_scale_rounded_tie_prefers_outer_source():
    pivot = _Key(0, 0)
    inner = _Key(2, 2)
    outer = _Key(3, 3)
    handle = _Key(10, 10)
    curve = _range_curve(pivot, inner, outer, handle)

    txn = trackbar_key_edit.snapshot_selection_range_scale_transaction(
        [curve],
        [0, 2, 3, 10],
        pivot_frame=0,
        source_handle_frame=10,
    )
    assert txn is not None
    assert trackbar_key_edit.apply_selection_range_scale_transaction([curve], txn, 4)
    by_frame = _keys_by_frame(curve)
    assert sorted(by_frame) == [0, 1, 4]
    assert by_frame[0].co.y == 0.0
    assert by_frame[1].co.y == 3.0
    assert by_frame[4].co.y == 10.0


def test_selection_range_scale_actual_pivot_key_wins_rounded_collision():
    pivot = _Key(10.0, 1.0)
    subframe = _Key(10.4, 2.0)
    handle = _Key(20.0, 3.0)
    curve = _range_curve(pivot, subframe, handle)

    txn = trackbar_key_edit.snapshot_selection_range_scale_transaction(
        [curve],
        [10.0, 10.4, 20.0],
        pivot_frame=10,
        source_handle_frame=20,
    )
    assert txn is not None
    assert trackbar_key_edit.apply_selection_range_scale_transaction([curve], txn, 11)
    by_frame = _keys_by_frame(curve)
    assert sorted(by_frame) == [10, 11]
    assert by_frame[10].co.y == 1.0
    assert by_frame[11].co.y == 3.0


def test_selection_range_scale_target_is_clamped_before_pivot_crossing():
    curve = _range_curve(_Key(10, 1), _Key(20, 2))
    txn = trackbar_key_edit.snapshot_selection_range_scale_transaction(
        [curve],
        [10, 20],
        pivot_frame=10,
        source_handle_frame=20,
    )
    assert txn is not None
    assert trackbar_key_edit.apply_selection_range_scale_transaction([curve], txn, 5)
    assert sorted(_keys_by_frame(curve)) == [10, 11]
    assert txn.selected_destination_frames == (10, 11)


def test_key_property_state_empty_and_non_bezier_handle_na():
    assert trackbar_key_edit.summarize_key_property_rows([]) == (
        0,
        0,
        0,
        "EMPTY",
        "N/A",
        0,
        0,
    )

    rows = [
        trackbar_key_edit.KeyPropertyValueRow(
            "object:Cube", 10.0, "LINEAR", "VECTOR", "VECTOR"
        ),
        trackbar_key_edit.KeyPropertyValueRow(
            "object:Cube", 20.0, "LINEAR", "FREE", "FREE"
        ),
    ]
    state = trackbar_key_edit.summarize_key_property_rows(rows)
    assert state.target_count == 2
    assert state.control_count == 1
    assert state.frame_count == 2
    assert state.interpolation_state == "LINEAR"
    assert state.handle_state == "N/A"
    assert state.bezier_count == 0
    assert state.non_bezier_count == 2


def test_key_property_state_reports_common_bezier_handle_type():
    rows = [
        trackbar_key_edit.KeyPropertyValueRow(
            "object:A", 10.0, "BEZIER", "AUTO_CLAMPED", "AUTO_CLAMPED"
        ),
        trackbar_key_edit.KeyPropertyValueRow(
            "object:B", 10.0, "BEZIER", "AUTO_CLAMPED", "AUTO_CLAMPED"
        ),
    ]
    state = trackbar_key_edit.summarize_key_property_rows(rows)
    assert state.target_count == 2
    assert state.control_count == 2
    assert state.frame_count == 1
    assert state.interpolation_state == "BEZIER"
    assert state.handle_state == "AUTO_CLAMPED"
    assert state.bezier_count == 2
    assert state.non_bezier_count == 0


def test_key_property_state_reports_mixed_interpolation_and_handles():
    rows = [
        trackbar_key_edit.KeyPropertyValueRow(
            "bone:Rig:Hand.L", 10.0, "BEZIER", "AUTO", "AUTO"
        ),
        trackbar_key_edit.KeyPropertyValueRow(
            "bone:Rig:Hand.L", 10.0, "BEZIER", "ALIGNED", "FREE"
        ),
        trackbar_key_edit.KeyPropertyValueRow(
            "bone:Rig:Hand.R", 20.0, "CONSTANT", "VECTOR", "VECTOR"
        ),
    ]
    state = trackbar_key_edit.summarize_key_property_rows(rows)
    assert state.target_count == 3
    assert state.control_count == 2
    assert state.frame_count == 2
    assert state.interpolation_state == "MIXED"
    assert state.handle_state == "MIXED"
    assert state.bezier_count == 2
    assert state.non_bezier_count == 1


def test_set_keyframe_properties_applies_native_interpolation_and_handles_in_place():
    key = _Key(10.0, 3.0)
    key.select_control_point = True
    original_left = (key.handle_left.x, key.handle_left.y)
    original_right = (key.handle_right.x, key.handle_right.y)
    curve = _FCurve("location", key)

    result = trackbar_key_edit.set_keyframe_properties(
        [(curve, key)],
        interpolation="BEZIER",
        handle_type="AUTO_CLAMPED",
    )

    assert result.changed_points == 1
    assert result.changed_curves == 1
    assert result.skipped_non_bezier == 0
    assert key.interpolation == "BEZIER"
    assert key.handle_left_type == "AUTO_CLAMPED"
    assert key.handle_right_type == "AUTO_CLAMPED"
    assert key.co.x == 10.0
    assert key.co.y == 3.0
    assert key.select_control_point
    assert (key.handle_left.x, key.handle_left.y) == original_left
    assert (key.handle_right.x, key.handle_right.y) == original_right
    assert curve.update_count == 1


def test_set_keyframe_properties_handle_only_skips_non_bezier_without_conversion():
    key = _Key(10.0, 1.0)
    key.interpolation = "LINEAR"
    curve = _FCurve("location", key)

    result = trackbar_key_edit.set_keyframe_properties(
        [(curve, key)],
        handle_type="VECTOR",
    )

    assert result == (0, 0, 1)
    assert key.interpolation == "LINEAR"
    assert key.handle_left_type == "FREE"
    assert key.handle_right_type == "FREE"
    assert curve.update_count == 0


def test_set_keyframe_properties_explicit_bezier_then_handle_applies_same_pass():
    key = _Key(10.0, 1.0)
    key.interpolation = "CONSTANT"
    curve = _FCurve("location", key)

    result = trackbar_key_edit.set_keyframe_properties(
        [(curve, key)],
        interpolation="BEZIER",
        handle_type="VECTOR",
    )

    assert result == (1, 1, 0)
    assert key.interpolation == "BEZIER"
    assert key.handle_left_type == "VECTOR"
    assert key.handle_right_type == "VECTOR"
    assert curve.update_count == 1


def test_set_keyframe_properties_rejects_invalid_enum_before_mutation():
    key = _Key(10.0, 1.0)
    curve = _FCurve("location", key)

    try:
        trackbar_key_edit.set_keyframe_properties(
            [(curve, key)], interpolation="NOT_A_MODE"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid interpolation should raise ValueError")

    assert key.interpolation == "BEZIER"
    assert key.handle_left_type == "FREE"
    assert key.handle_right_type == "FREE"
    assert curve.update_count == 0
