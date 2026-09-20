from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "trackbar_math.py"
)
SPEC = spec_from_file_location("baw_trackbar_math", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
trackbar_math = module_from_spec(SPEC)
SPEC.loader.exec_module(trackbar_math)


def test_frame_to_x_maps_range_edges():
    assert trackbar_math.frame_to_x(0, 0, 100, 10, 210) == 10
    assert trackbar_math.frame_to_x(100, 0, 100, 10, 210) == 210
    assert trackbar_math.frame_to_x(50, 0, 100, 10, 210) == 110


def test_x_to_frame_clamps_outside_trackbar():
    assert trackbar_math.x_to_frame(-100, 1, 101, 10, 210) == 1
    assert trackbar_math.x_to_frame(999, 1, 101, 10, 210) == 101


def test_cell_width_matches_one_frame_step():
    left_10, right_10 = trackbar_math.frame_cell_bounds(10, 10, 16, 8)
    left_11, right_11 = trackbar_math.frame_cell_bounds(11, 10, 16, 8)
    assert (left_10, right_10) == (16, 24)
    assert (left_11, right_11) == (24, 32)
    assert right_10 == left_11


def test_x_to_cell_frame_uses_discrete_cells():
    assert trackbar_math.x_to_cell_frame(16, 10, 16, 8, 20) == 10
    assert trackbar_math.x_to_cell_frame(23.99, 10, 16, 8, 20) == 10
    assert trackbar_math.x_to_cell_frame(24, 10, 16, 8, 20) == 11


def test_visible_frame_capacity_uses_whole_cells_only():
    assert trackbar_math.visible_frame_capacity(80, 8) == 10
    assert trackbar_math.visible_frame_capacity(87, 8) == 10


def test_fitted_cell_width_fills_the_same_track_width_for_any_range():
    width = 1200
    cell_1_40 = trackbar_math.fitted_frame_cell_width(1, 40, width)
    cell_1_100 = trackbar_math.fitted_frame_cell_width(1, 100, width)
    cell_30_60 = trackbar_math.fitted_frame_cell_width(30, 60, width)

    assert cell_1_40 * 40 == width
    assert cell_1_100 * 100 == width
    assert cell_30_60 * 31 == width
    assert cell_1_40 > cell_1_100
    assert cell_30_60 > cell_1_40


def test_frame_delta_from_pixel_drag_matches_cell_width():
    assert trackbar_math.frame_delta_from_pixel_drag(0, 8) == 0
    assert trackbar_math.frame_delta_from_pixel_drag(8, 8) == 1
    assert trackbar_math.frame_delta_from_pixel_drag(24, 8) == 3
    assert trackbar_math.frame_delta_from_pixel_drag(-16, 8) == -2


def test_selection_range_frame_span_requires_two_distinct_frames():
    assert trackbar_math.selection_range_frame_span([]) is None
    assert trackbar_math.selection_range_frame_span([10]) is None
    assert trackbar_math.selection_range_frame_span([10, 10.0]) is None


def test_selection_range_frame_span_normalizes_and_snaps():
    assert trackbar_math.selection_range_frame_span([20, 10, 15]) == (10, 20)
    assert trackbar_math.selection_range_frame_span([20.2, 9.8, 15.1]) == (10, 20)


def test_selection_range_frame_span_supports_negative_frames():
    assert trackbar_math.selection_range_frame_span([-2, -10, -5]) == (-10, -2)


def test_nice_tick_step_stays_human_readable():
    assert trackbar_math.nice_tick_step(100, 800) == 10
    assert trackbar_math.nice_tick_step(1000, 800) == 100
    assert trackbar_math.nice_tick_step(10, 800) == 1
