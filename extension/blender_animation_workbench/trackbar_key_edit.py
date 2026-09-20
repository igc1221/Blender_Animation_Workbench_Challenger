from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple


class KeyPropertyValueRow(NamedTuple):
    """Read-only key-property values used to summarize a popup target set."""

    control_id: str
    frame: float
    interpolation: str
    handle_left_type: str
    handle_right_type: str


class KeyPropertyState(NamedTuple):
    target_count: int
    control_count: int
    frame_count: int
    interpolation_state: str
    handle_state: str
    bezier_count: int
    non_bezier_count: int


def summarize_key_property_rows(rows: Iterable[KeyPropertyValueRow]) -> KeyPropertyState:
    """Return common/MIXED/N/A state without mutating Blender animation data."""
    items = tuple(rows)
    if not items:
        return KeyPropertyState(0, 0, 0, "EMPTY", "N/A", 0, 0)

    interpolation_values = {str(row.interpolation) for row in items}
    interpolation_state = (
        next(iter(interpolation_values)) if len(interpolation_values) == 1 else "MIXED"
    )

    bezier_rows = [row for row in items if str(row.interpolation) == "BEZIER"]
    if not bezier_rows:
        handle_state = "N/A"
    else:
        handle_values = {
            (str(row.handle_left_type), str(row.handle_right_type))
            for row in bezier_rows
        }
        if len(handle_values) != 1:
            handle_state = "MIXED"
        else:
            left, right = next(iter(handle_values))
            handle_state = left if left == right else "MIXED"

    return KeyPropertyState(
        target_count=len(items),
        control_count=len({row.control_id for row in items}),
        frame_count=len({float(row.frame) for row in items}),
        interpolation_state=interpolation_state,
        handle_state=handle_state,
        bezier_count=len(bezier_rows),
        non_bezier_count=len(items) - len(bezier_rows),
    )


class KeyPropertyEditResult(NamedTuple):
    changed_points: int
    changed_curves: int
    skipped_non_bezier: int


def set_keyframe_properties(
    point_rows: Iterable[tuple[object, object]],
    *,
    interpolation: str | None = None,
    handle_type: str | None = None,
) -> KeyPropertyEditResult:
    """Apply native key properties in place and roll back on any write failure."""
    valid_interpolations = {"CONSTANT", "LINEAR", "BEZIER"}
    valid_handle_types = {"AUTO", "AUTO_CLAMPED", "VECTOR", "ALIGNED", "FREE"}
    if interpolation is not None and interpolation not in valid_interpolations:
        raise ValueError(f"Unsupported key interpolation: {interpolation}")
    if handle_type is not None and handle_type not in valid_handle_types:
        raise ValueError(f"Unsupported key handle type: {handle_type}")

    rows = tuple(point_rows)
    if not rows or (interpolation is None and handle_type is None):
        return KeyPropertyEditResult(0, 0, 0)

    snapshots = [
        (
            fcurve,
            key,
            str(key.interpolation),
            str(key.handle_left_type),
            str(key.handle_right_type),
        )
        for fcurve, key in rows
    ]
    changed_point_ids: set[int] = set()
    changed_curves: list[object] = []
    changed_curve_ids: set[int] = set()
    skipped_non_bezier = 0

    try:
        for fcurve, key in rows:
            point_changed = False
            if interpolation is not None and str(key.interpolation) != interpolation:
                key.interpolation = interpolation
                point_changed = True

            if handle_type is not None:
                if str(key.interpolation) == "BEZIER":
                    if str(key.handle_left_type) != handle_type:
                        key.handle_left_type = handle_type
                        point_changed = True
                    if str(key.handle_right_type) != handle_type:
                        key.handle_right_type = handle_type
                        point_changed = True
                else:
                    skipped_non_bezier += 1

            if point_changed:
                changed_point_ids.add(id(key))
                curve_id = id(fcurve)
                if curve_id not in changed_curve_ids:
                    changed_curve_ids.add(curve_id)
                    changed_curves.append(fcurve)

        for fcurve in changed_curves:
            fcurve.update()
    except Exception:
        rollback_curves: list[object] = []
        rollback_curve_ids: set[int] = set()
        for fcurve, key, old_interpolation, old_left, old_right in snapshots:
            key.interpolation = old_interpolation
            key.handle_left_type = old_left
            key.handle_right_type = old_right
            curve_id = id(fcurve)
            if curve_id not in rollback_curve_ids:
                rollback_curve_ids.add(curve_id)
                rollback_curves.append(fcurve)
        for fcurve in rollback_curves:
            fcurve.update()
        raise

    return KeyPropertyEditResult(
        changed_points=len(changed_point_ids),
        changed_curves=len(changed_curves),
        skipped_non_bezier=skipped_non_bezier,
    )


class KeyframeSnapshot(NamedTuple):
    data_path: str
    array_index: int
    frame: float
    value: float
    handle_left: tuple[float, float]
    handle_right: tuple[float, float]
    handle_left_type: str
    handle_right_type: str
    interpolation: str
    easing: str
    amplitude: float
    back: float
    period: float
    keyframe_type: str
    select_control_point: bool
    select_left_handle: bool
    select_right_handle: bool


def _matches_context(fcurve, data_path_prefix) -> bool:
    if data_path_prefix is None:
        return True
    data_path = str(fcurve.data_path)
    if isinstance(data_path_prefix, (tuple, list, set, frozenset)):
        return any(_matches_context(fcurve, prefix) for prefix in data_path_prefix)
    if data_path_prefix == "":
        return not data_path.startswith('pose.bones[')
    return data_path.startswith(str(data_path_prefix))


def _snapshot_keyframe(fcurve, key) -> KeyframeSnapshot:
    return KeyframeSnapshot(
        data_path=str(fcurve.data_path),
        array_index=int(getattr(fcurve, "array_index", 0)),
        frame=float(key.co.x),
        value=float(key.co.y),
        handle_left=(float(key.handle_left.x), float(key.handle_left.y)),
        handle_right=(float(key.handle_right.x), float(key.handle_right.y)),
        handle_left_type=str(key.handle_left_type),
        handle_right_type=str(key.handle_right_type),
        interpolation=str(key.interpolation),
        easing=str(key.easing),
        amplitude=float(key.amplitude),
        back=float(key.back),
        period=float(key.period),
        keyframe_type=str(key.type),
        select_control_point=bool(key.select_control_point),
        select_left_handle=bool(key.select_left_handle),
        select_right_handle=bool(key.select_right_handle),
    )


class MultiKeyPreviewCurveState:
    __slots__ = (
        "array_index",
        "collision_snapshots",
        "data_path",
        "destination_frames",
        "source_snapshots",
    )

    def __init__(
        self,
        data_path: str,
        array_index: int,
        source_snapshots: list[KeyframeSnapshot],
    ) -> None:
        self.data_path = data_path
        self.array_index = array_index
        self.source_snapshots = source_snapshots
        self.collision_snapshots: list[KeyframeSnapshot] = []
        self.destination_frames: tuple[float, ...] = ()


class MultiKeyPreviewTransaction:
    __slots__ = ("applied_delta", "curve_states", "mode", "source_frames")

    def __init__(
        self,
        mode: str,
        source_frames: tuple[float, ...],
        curve_states: list[MultiKeyPreviewCurveState],
    ) -> None:
        self.mode = mode
        self.source_frames = source_frames
        self.curve_states = curve_states
        self.applied_delta: int | None = None


def _insert_keyframe_snapshot(
    points,
    snapshot: KeyframeSnapshot,
    *,
    frame_delta: float = 0.0,
    selected_override: bool | None = None,
):
    target_frame = snapshot.frame + frame_delta
    key = points.insert(
        target_frame,
        snapshot.value,
        options={"FAST"},
        keyframe_type=snapshot.keyframe_type,
    )
    key.interpolation = snapshot.interpolation
    key.easing = snapshot.easing
    key.amplitude = snapshot.amplitude
    key.back = snapshot.back
    key.period = snapshot.period
    key.handle_left_type = "FREE"
    key.handle_right_type = "FREE"
    key.handle_left.x = snapshot.handle_left[0] + frame_delta
    key.handle_left.y = snapshot.handle_left[1]
    key.handle_right.x = snapshot.handle_right[0] + frame_delta
    key.handle_right.y = snapshot.handle_right[1]
    key.handle_left_type = snapshot.handle_left_type
    key.handle_right_type = snapshot.handle_right_type
    if selected_override is None:
        key.select_control_point = snapshot.select_control_point
        key.select_left_handle = snapshot.select_left_handle
        key.select_right_handle = snapshot.select_right_handle
    else:
        selected = bool(selected_override)
        key.select_control_point = selected
        key.select_left_handle = selected
        key.select_right_handle = selected
    return key


def _remove_keyframe_points_at_frames(points, frames: Iterable[float], epsilon: float) -> bool:
    targets = tuple(float(frame) for frame in frames)
    indices = [
        index
        for index, key in enumerate(points)
        if any(abs(float(key.co.x) - frame) <= epsilon for frame in targets)
    ]
    for index in reversed(indices):
        points.remove(points[index], fast=True)
    return bool(indices)


def snapshot_multi_key_preview_transaction(
    fcurves: Iterable,
    source_frames: Iterable[float],
    *,
    mode: str,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> MultiKeyPreviewTransaction | None:
    """Capture only participating source points for a lightweight modal preview."""
    edit_mode = str(mode).upper()
    if edit_mode not in {"MOVE", "CLONE"}:
        raise ValueError(f"Unsupported multi-key preview mode: {mode}")
    sources = tuple(float(frame) for frame in source_frames)
    if not sources:
        return None

    states: list[MultiKeyPreviewCurveState] = []
    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue
        snapshots = [
            _snapshot_keyframe(fcurve, key)
            for key in fcurve.keyframe_points
            if any(abs(float(key.co.x) - frame) <= epsilon for frame in sources)
        ]
        if snapshots:
            states.append(
                MultiKeyPreviewCurveState(
                    data_path=str(fcurve.data_path),
                    array_index=int(getattr(fcurve, "array_index", 0)),
                    source_snapshots=snapshots,
                )
            )
    if not states:
        return None
    return MultiKeyPreviewTransaction(edit_mode, sources, states)


def _preview_curve_for_state(fcurves: Iterable, state: MultiKeyPreviewCurveState):
    return next(
        (
            fcurve
            for fcurve in fcurves
            if str(fcurve.data_path) == state.data_path
            and int(getattr(fcurve, "array_index", 0)) == state.array_index
        ),
        None,
    )


def _preview_curve_index(fcurves: Iterable) -> dict[tuple[str, int], object]:
    """Index concrete preview curves once per modal update instead of per state."""
    return {
        (str(fcurve.data_path), int(getattr(fcurve, "array_index", 0))): fcurve
        for fcurve in fcurves
    }


def restore_multi_key_preview_transaction(
    fcurves: Iterable,
    transaction: MultiKeyPreviewTransaction,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Restore only source/destination cells touched by the current live preview."""
    matching_curves = [
        fcurve for fcurve in fcurves if _matches_context(fcurve, data_path_prefix)
    ]
    curve_index = _preview_curve_index(matching_curves)
    changed = False
    for state in transaction.curve_states:
        fcurve = curve_index.get((state.data_path, state.array_index))
        if fcurve is None:
            continue
        points = fcurve.keyframe_points
        source_values = tuple(snapshot.frame for snapshot in state.source_snapshots)
        touched_frames = source_values + tuple(state.destination_frames)
        curve_changed = _remove_keyframe_points_at_frames(points, touched_frames, epsilon)
        for snapshot in state.source_snapshots:
            _insert_keyframe_snapshot(points, snapshot)
            curve_changed = True
        for snapshot in state.collision_snapshots:
            _insert_keyframe_snapshot(points, snapshot)
            curve_changed = True
        if curve_changed:
            fcurve.update()
            changed = True
        state.collision_snapshots = []
        state.destination_frames = ()
    transaction.applied_delta = None
    return changed


def apply_multi_key_preview_transaction(
    fcurves: Iterable,
    transaction: MultiKeyPreviewTransaction,
    delta_frames: int,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Rebuild the current preview from pristine source snapshots without full-curve restore."""
    matching_curves = [
        fcurve for fcurve in fcurves if _matches_context(fcurve, data_path_prefix)
    ]
    curve_index = _preview_curve_index(matching_curves)
    if transaction.applied_delta is not None:
        restore_multi_key_preview_transaction(
            matching_curves,
            transaction,
            data_path_prefix=data_path_prefix,
            epsilon=epsilon,
        )

    delta = int(delta_frames)
    if delta == 0:
        return True

    changed = False
    for state in transaction.curve_states:
        fcurve = curve_index.get((state.data_path, state.array_index))
        if fcurve is None:
            continue
        points = fcurve.keyframe_points
        source_values = tuple(snapshot.frame for snapshot in state.source_snapshots)
        destination_values = tuple(snapshot.frame + delta for snapshot in state.source_snapshots)

        if transaction.mode == "CLONE":
            actual_destinations = tuple(
                frame
                for frame in destination_values
                if not any(abs(frame - source) <= epsilon for source in source_values)
            )
        else:
            actual_destinations = destination_values

        collision_frames = tuple(
            frame
            for frame in actual_destinations
            if not any(abs(frame - source) <= epsilon for source in source_values)
        )
        state.collision_snapshots = [
            _snapshot_keyframe(fcurve, key)
            for key in points
            if any(abs(float(key.co.x) - frame) <= epsilon for frame in collision_frames)
        ]
        state.destination_frames = actual_destinations

        _remove_keyframe_points_at_frames(points, source_values + collision_frames, epsilon)

        if transaction.mode == "CLONE":
            for snapshot in state.source_snapshots:
                source_is_destination = any(
                    abs(snapshot.frame - frame) <= epsilon for frame in destination_values
                )
                _insert_keyframe_snapshot(
                    points,
                    snapshot,
                    selected_override=source_is_destination,
                )
            for snapshot in state.source_snapshots:
                target_frame = snapshot.frame + delta
                if any(abs(target_frame - source) <= epsilon for source in source_values):
                    continue
                _insert_keyframe_snapshot(
                    points,
                    snapshot,
                    frame_delta=delta,
                    selected_override=True,
                )
        else:
            for snapshot in state.source_snapshots:
                _insert_keyframe_snapshot(
                    points,
                    snapshot,
                    frame_delta=delta,
                    selected_override=True,
                )

        fcurve.update()
        changed = True

    transaction.applied_delta = delta if changed else None
    return changed


class SelectionRangeScalePreviewCurveState:
    __slots__ = (
        "array_index",
        "collision_snapshots",
        "data_path",
        "destination_frames",
        "source_snapshots",
    )

    def __init__(
        self,
        data_path: str,
        array_index: int,
        source_snapshots: list[KeyframeSnapshot],
    ) -> None:
        self.data_path = data_path
        self.array_index = array_index
        self.source_snapshots = source_snapshots
        self.collision_snapshots: list[KeyframeSnapshot] = []
        self.destination_frames: tuple[float, ...] = ()


class SelectionRangeScalePreviewTransaction:
    __slots__ = (
        "applied_target_handle_frame",
        "curve_states",
        "pivot_frame",
        "selected_destination_frames",
        "source_frames",
        "source_handle_frame",
    )

    def __init__(
        self,
        source_frames: tuple[float, ...],
        *,
        pivot_frame: int,
        source_handle_frame: int,
        curve_states: list[SelectionRangeScalePreviewCurveState],
    ) -> None:
        self.source_frames = source_frames
        self.pivot_frame = int(pivot_frame)
        self.source_handle_frame = int(source_handle_frame)
        self.curve_states = curve_states
        self.applied_target_handle_frame: int | None = None
        self.selected_destination_frames: tuple[int, ...] = tuple(
            sorted({round(frame) for frame in source_frames})
        )


def _clamp_selection_range_target(
    pivot_frame: int,
    source_handle_frame: int,
    target_handle_frame: int,
) -> int | None:
    pivot = int(pivot_frame)
    source = int(source_handle_frame)
    target = int(target_handle_frame)
    if source == pivot:
        return None
    if source > pivot:
        return max(pivot + 1, target)
    return min(pivot - 1, target)


def _selection_range_scale_factor(
    pivot_frame: int,
    source_handle_frame: int,
    target_handle_frame: int,
) -> float | None:
    target = _clamp_selection_range_target(
        pivot_frame,
        source_handle_frame,
        target_handle_frame,
    )
    if target is None:
        return None
    return (target - pivot_frame) / (source_handle_frame - pivot_frame)


def _selection_range_scaled_frame(
    source_frame: float,
    pivot_frame: int,
    factor: float,
) -> int:
    return round(pivot_frame + (source_frame - pivot_frame) * factor)


def _selection_range_winner(
    candidates: list[KeyframeSnapshot],
    pivot_frame: int,
) -> KeyframeSnapshot:
    """Resolve rounded selected-selected ties independently of mutation order."""
    return min(
        candidates,
        key=lambda snapshot: (
            0 if snapshot.frame == pivot_frame else 1,
            -abs(snapshot.frame - pivot_frame),
            snapshot.frame,
        ),
    )


def _insert_scaled_keyframe_snapshot(
    points,
    snapshot: KeyframeSnapshot,
    *,
    pivot_frame: int,
    factor: float,
    target_frame: int,
    selected_override: bool = True,
):
    key = points.insert(
        target_frame,
        snapshot.value,
        options={"FAST"},
        keyframe_type=snapshot.keyframe_type,
    )
    key.interpolation = snapshot.interpolation
    key.easing = snapshot.easing
    key.amplitude = snapshot.amplitude
    key.back = snapshot.back
    key.period = snapshot.period
    key.handle_left_type = "FREE"
    key.handle_right_type = "FREE"
    key.handle_left.x = pivot_frame + (snapshot.handle_left[0] - pivot_frame) * factor
    key.handle_left.y = snapshot.handle_left[1]
    key.handle_right.x = pivot_frame + (snapshot.handle_right[0] - pivot_frame) * factor
    key.handle_right.y = snapshot.handle_right[1]
    key.handle_left_type = snapshot.handle_left_type
    key.handle_right_type = snapshot.handle_right_type
    if selected_override:
        key.select_control_point = True
        key.select_left_handle = True
        key.select_right_handle = True
    else:
        key.select_control_point = snapshot.select_control_point
        key.select_left_handle = snapshot.select_left_handle
        key.select_right_handle = snapshot.select_right_handle
    return key


def snapshot_selection_range_scale_transaction(
    fcurves: Iterable,
    source_frames: Iterable[float],
    *,
    pivot_frame: int,
    source_handle_frame: int,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> SelectionRangeScalePreviewTransaction | None:
    sources = tuple(sorted({float(frame) for frame in source_frames}))
    if len(sources) < 2 or int(pivot_frame) == int(source_handle_frame):
        return None

    states: list[SelectionRangeScalePreviewCurveState] = []
    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue
        source_snapshots = [
            _snapshot_keyframe(fcurve, key)
            for key in fcurve.keyframe_points
            if any(abs(float(key.co.x) - frame) <= epsilon for frame in sources)
        ]
        if source_snapshots:
            states.append(
                SelectionRangeScalePreviewCurveState(
                    data_path=str(fcurve.data_path),
                    array_index=int(getattr(fcurve, "array_index", 0)),
                    source_snapshots=source_snapshots,
                )
            )
    if not states:
        return None
    return SelectionRangeScalePreviewTransaction(
        sources,
        pivot_frame=int(pivot_frame),
        source_handle_frame=int(source_handle_frame),
        curve_states=states,
    )


def restore_selection_range_scale_transaction(
    fcurves: Iterable,
    transaction: SelectionRangeScalePreviewTransaction,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    matching_curves = [
        fcurve for fcurve in fcurves if _matches_context(fcurve, data_path_prefix)
    ]
    curve_index = _preview_curve_index(matching_curves)
    changed = False
    for state in transaction.curve_states:
        fcurve = curve_index.get((state.data_path, state.array_index))
        if fcurve is None:
            continue
        points = fcurve.keyframe_points
        source_values = tuple(snapshot.frame for snapshot in state.source_snapshots)
        touched_frames = source_values + tuple(state.destination_frames)
        curve_changed = _remove_keyframe_points_at_frames(points, touched_frames, epsilon)
        for snapshot in state.source_snapshots:
            _insert_keyframe_snapshot(points, snapshot)
            curve_changed = True
        for snapshot in state.collision_snapshots:
            _insert_keyframe_snapshot(points, snapshot)
            curve_changed = True
        if curve_changed:
            fcurve.update()
            changed = True
        state.collision_snapshots = []
        state.destination_frames = ()

    transaction.applied_target_handle_frame = None
    transaction.selected_destination_frames = tuple(
        sorted({round(frame) for frame in transaction.source_frames})
    )
    return changed


def apply_selection_range_scale_transaction(
    fcurves: Iterable,
    transaction: SelectionRangeScalePreviewTransaction,
    target_handle_frame: int,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    matching_curves = [
        fcurve for fcurve in fcurves if _matches_context(fcurve, data_path_prefix)
    ]
    curve_index = _preview_curve_index(matching_curves)
    if transaction.applied_target_handle_frame is not None:
        restore_selection_range_scale_transaction(
            matching_curves,
            transaction,
            data_path_prefix=data_path_prefix,
            epsilon=epsilon,
        )

    target = _clamp_selection_range_target(
        transaction.pivot_frame,
        transaction.source_handle_frame,
        int(target_handle_frame),
    )
    if target is None:
        return False
    factor = _selection_range_scale_factor(
        transaction.pivot_frame,
        transaction.source_handle_frame,
        target,
    )
    if factor is None:
        return False

    transaction.selected_destination_frames = tuple(
        sorted(
            {
                _selection_range_scaled_frame(
                    frame,
                    transaction.pivot_frame,
                    factor,
                )
                for frame in transaction.source_frames
            }
        )
    )

    if target == transaction.source_handle_frame:
        return True

    changed = False
    for state in transaction.curve_states:
        fcurve = curve_index.get((state.data_path, state.array_index))
        if fcurve is None:
            continue
        points = fcurve.keyframe_points
        source_values = tuple(snapshot.frame for snapshot in state.source_snapshots)
        candidates_by_destination: dict[int, list[KeyframeSnapshot]] = {}
        for snapshot in state.source_snapshots:
            destination = _selection_range_scaled_frame(
                snapshot.frame,
                transaction.pivot_frame,
                factor,
            )
            candidates_by_destination.setdefault(destination, []).append(snapshot)

        destination_values = tuple(sorted(candidates_by_destination))
        collision_frames = tuple(
            frame
            for frame in destination_values
            if not any(abs(frame - source) <= epsilon for source in source_values)
        )
        state.collision_snapshots = [
            _snapshot_keyframe(fcurve, key)
            for key in points
            if any(abs(float(key.co.x) - frame) <= epsilon for frame in collision_frames)
        ]
        state.destination_frames = destination_values

        _remove_keyframe_points_at_frames(
            points,
            source_values + collision_frames,
            epsilon,
        )
        for destination, candidates in sorted(candidates_by_destination.items()):
            winner = _selection_range_winner(candidates, transaction.pivot_frame)
            _insert_scaled_keyframe_snapshot(
                points,
                winner,
                pivot_frame=transaction.pivot_frame,
                factor=factor,
                target_frame=destination,
                selected_override=True,
            )
        fcurve.update()
        changed = True

    transaction.applied_target_handle_frame = target if changed else None
    return changed


def scale_keyframe_points_at_frames(
    fcurves: Iterable,
    source_frames: Iterable[float],
    *,
    pivot_frame: int,
    source_handle_frame: int,
    target_handle_frame: int,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    if int(source_handle_frame) == int(target_handle_frame):
        return False
    transaction = snapshot_selection_range_scale_transaction(
        fcurves,
        source_frames,
        pivot_frame=pivot_frame,
        source_handle_frame=source_handle_frame,
        data_path_prefix=data_path_prefix,
        epsilon=epsilon,
    )
    if transaction is None:
        return False
    return apply_selection_range_scale_transaction(
        fcurves,
        transaction,
        target_handle_frame,
        data_path_prefix=data_path_prefix,
        epsilon=epsilon,
    )


def snapshot_keyframe_points(
    fcurves: Iterable,
    *,
    data_path_prefix: str | None = None,
) -> list[KeyframeSnapshot]:
    """Capture the complete keyframe state for matching FCurves."""
    snapshots: list[KeyframeSnapshot] = []
    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue
        snapshots.extend(_snapshot_keyframe(fcurve, key) for key in fcurve.keyframe_points)
    return snapshots


def restore_keyframe_state(
    fcurves: Iterable,
    snapshots: Iterable[KeyframeSnapshot],
    *,
    data_path_prefix: str | None = None,
) -> bool:
    """Replace matching FCurve key contents with an exact captured state."""
    matching_curves = [
        fcurve for fcurve in fcurves if _matches_context(fcurve, data_path_prefix)
    ]
    snapshots_by_curve: dict[tuple[str, int], list[KeyframeSnapshot]] = {}
    for snapshot in snapshots:
        snapshots_by_curve.setdefault(
            (snapshot.data_path, snapshot.array_index),
            [],
        ).append(snapshot)

    changed = False
    for fcurve in matching_curves:
        points = fcurve.keyframe_points
        curve_changed = bool(points)
        for index in range(len(points) - 1, -1, -1):
            points.remove(points[index], fast=True)

        curve_snapshots = snapshots_by_curve.get(
            (str(fcurve.data_path), int(getattr(fcurve, "array_index", 0))),
            (),
        )
        curve_changed = curve_changed or bool(curve_snapshots)
        for snapshot in sorted(curve_snapshots, key=lambda item: item.frame):
            key = points.insert(
                snapshot.frame,
                snapshot.value,
                options={"FAST"},
                keyframe_type=snapshot.keyframe_type,
            )
            key.interpolation = snapshot.interpolation
            key.easing = snapshot.easing
            key.amplitude = snapshot.amplitude
            key.back = snapshot.back
            key.period = snapshot.period
            key.handle_left_type = snapshot.handle_left_type
            key.handle_right_type = snapshot.handle_right_type
            key.handle_left.x, key.handle_left.y = snapshot.handle_left
            key.handle_right.x, key.handle_right.y = snapshot.handle_right
            key.select_control_point = snapshot.select_control_point
            key.select_left_handle = snapshot.select_left_handle
            key.select_right_handle = snapshot.select_right_handle
        if curve_changed:
            fcurve.update()
            changed = True
    return changed


def snapshot_replaced_keyframe_points(
    fcurves: Iterable,
    source_frame: float,
    target_frame: float,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> list[KeyframeSnapshot]:
    """Capture destination keys that a source-frame move would replace."""
    snapshots: list[KeyframeSnapshot] = []
    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue
        points = fcurve.keyframe_points
        if not any(abs(float(key.co.x) - source_frame) <= epsilon for key in points):
            continue
        for key in points:
            if abs(float(key.co.x) - target_frame) <= epsilon:
                snapshots.append(_snapshot_keyframe(fcurve, key))
    return snapshots


def restore_keyframe_snapshots(
    fcurves: Iterable,
    snapshots: Iterable[KeyframeSnapshot],
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Restore key snapshots, replacing any current key on the same FCurve/frame."""
    matching_curves = [
        fcurve for fcurve in fcurves if _matches_context(fcurve, data_path_prefix)
    ]
    restored = False
    updated_curves = []

    for snapshot in snapshots:
        fcurve = next(
            (
                curve
                for curve in matching_curves
                if str(curve.data_path) == snapshot.data_path
                and int(getattr(curve, "array_index", 0)) == snapshot.array_index
            ),
            None,
        )
        if fcurve is None:
            continue

        points = fcurve.keyframe_points
        indices = [
            index
            for index, key in enumerate(points)
            if abs(float(key.co.x) - snapshot.frame) <= epsilon
        ]
        for index in reversed(indices):
            points.remove(points[index], fast=True)

        key = points.insert(
            snapshot.frame,
            snapshot.value,
            options={"FAST"},
            keyframe_type=snapshot.keyframe_type,
        )
        key.interpolation = snapshot.interpolation
        key.easing = snapshot.easing
        key.amplitude = snapshot.amplitude
        key.back = snapshot.back
        key.period = snapshot.period
        key.handle_left_type = snapshot.handle_left_type
        key.handle_right_type = snapshot.handle_right_type
        key.handle_left.x, key.handle_left.y = snapshot.handle_left
        key.handle_right.x, key.handle_right.y = snapshot.handle_right
        key.select_control_point = snapshot.select_control_point
        key.select_left_handle = snapshot.select_left_handle
        key.select_right_handle = snapshot.select_right_handle
        if fcurve not in updated_curves:
            updated_curves.append(fcurve)
        restored = True

    for fcurve in updated_curves:
        fcurve.update()
    return restored


def delete_keyframe_points_at_frames(
    fcurves: Iterable,
    frames: Iterable[float],
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Delete all matching keyframe points at any of the requested frames."""
    targets = tuple(float(frame) for frame in frames)
    if not targets:
        return False

    deleted = False
    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue
        points = fcurve.keyframe_points
        indices = [
            index
            for index, key in enumerate(points)
            if any(abs(float(key.co.x) - frame) <= epsilon for frame in targets)
        ]
        for index in reversed(indices):
            points.remove(points[index], fast=True)
        if indices:
            fcurve.update()
            deleted = True
    return deleted


def delete_keyframe_points(
    fcurves: Iterable,
    frame: float,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Delete one active-control frame cell, preserving other frames/controls."""
    deleted = False
    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue
        points = fcurve.keyframe_points
        indices = [i for i, key in enumerate(points) if abs(float(key.co.x) - frame) <= epsilon]
        # Re-resolve by descending index: RNA point references can be invalidated
        # when the collection storage changes during removal.
        for index in reversed(indices):
            points.remove(points[index], fast=True)
        if indices:
            fcurve.update()
            deleted = True
    return deleted


def clone_keyframe_points(
    fcurves: Iterable,
    source_frame: float,
    target_frame: float,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Clone one complete frame cell, replacing same-FCurve destination keys."""
    if abs(target_frame - source_frame) <= epsilon:
        return False

    delta = target_frame - source_frame
    cloned = False

    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue

        points = fcurve.keyframe_points
        source_snapshots = [
            _snapshot_keyframe(fcurve, key)
            for key in points
            if abs(float(key.co.x) - source_frame) <= epsilon
        ]
        if not source_snapshots:
            continue

        destination_indices = [
            index
            for index, key in enumerate(points)
            if abs(float(key.co.x) - target_frame) <= epsilon
        ]
        for index in reversed(destination_indices):
            points.remove(points[index], fast=True)

        for snapshot in source_snapshots:
            key = points.insert(
                target_frame,
                snapshot.value,
                options={"FAST"},
                keyframe_type=snapshot.keyframe_type,
            )
            key.interpolation = snapshot.interpolation
            key.easing = snapshot.easing
            key.amplitude = snapshot.amplitude
            key.back = snapshot.back
            key.period = snapshot.period
            key.handle_left_type = snapshot.handle_left_type
            key.handle_right_type = snapshot.handle_right_type
            key.handle_left.x = snapshot.handle_left[0] + delta
            key.handle_left.y = snapshot.handle_left[1]
            key.handle_right.x = snapshot.handle_right[0] + delta
            key.handle_right.y = snapshot.handle_right[1]
            key.select_control_point = snapshot.select_control_point
            key.select_left_handle = snapshot.select_left_handle
            key.select_right_handle = snapshot.select_right_handle

        fcurve.update()
        cloned = True

    return cloned


def remove_cloned_keyframe_points(
    fcurves: Iterable,
    source_frame: float,
    target_frame: float,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Remove a clone preview only from FCurves that have a source-frame key."""
    if abs(target_frame - source_frame) <= epsilon:
        return False

    removed = False
    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue
        points = fcurve.keyframe_points
        if not any(abs(float(key.co.x) - source_frame) <= epsilon for key in points):
            continue
        destination_indices = [
            index
            for index, key in enumerate(points)
            if abs(float(key.co.x) - target_frame) <= epsilon
        ]
        for index in reversed(destination_indices):
            points.remove(points[index], fast=True)
        if destination_indices:
            fcurve.update()
            removed = True
    return removed


def clone_keyframe_points_at_frames(
    fcurves: Iterable,
    source_frames: Iterable[float],
    delta: float,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Clone multiple frame cells simultaneously while preserving source key objects."""
    sources = tuple(float(frame) for frame in source_frames)
    if not sources or abs(delta) <= epsilon:
        return False

    cloned = False
    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue
        points = fcurve.keyframe_points
        source_keys = [
            key
            for key in points
            if any(abs(float(key.co.x) - frame) <= epsilon for frame in sources)
        ]
        if not source_keys:
            continue

        source_snapshots = [_snapshot_keyframe(fcurve, key) for key in source_keys]
        source_values = tuple(snapshot.frame for snapshot in source_snapshots)
        destination_values = tuple(snapshot.frame + delta for snapshot in source_snapshots)
        collision_destinations = tuple(
            frame
            for frame in destination_values
            if not any(abs(frame - source) <= epsilon for source in source_values)
        )

        # Clear source selection before removing any occupied destination points.
        # Blender KeyframePoint RNA wrappers are collection-index backed; deleting
        # an earlier destination (for example reverse clone 20 -> 15) can reindex
        # the source point and make a previously captured wrapper stale/misaligned.
        # Writing the source selection state first keeps reverse and forward
        # collision clones symmetric and prevents a false Selection Range.
        for key, snapshot in zip(source_keys, source_snapshots, strict=True):
            selected_as_destination = any(
                abs(snapshot.frame - frame) <= epsilon for frame in destination_values
            )
            key.select_control_point = selected_as_destination
            key.select_left_handle = selected_as_destination
            key.select_right_handle = selected_as_destination

        remove_indices = [
            index
            for index, key in enumerate(points)
            if key not in source_keys
            and any(
                abs(float(key.co.x) - frame) <= epsilon
                for frame in collision_destinations
            )
        ]
        for index in reversed(remove_indices):
            points.remove(points[index], fast=True)

        for snapshot in source_snapshots:
            target_frame = snapshot.frame + delta
            if any(abs(target_frame - source) <= epsilon for source in source_values):
                continue
            _insert_keyframe_snapshot(
                points,
                snapshot,
                frame_delta=delta,
                selected_override=True,
            )

        fcurve.update()
        cloned = True

    return cloned


def move_keyframe_points_at_frames(
    fcurves: Iterable,
    source_frames: Iterable[float],
    delta: float,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Move multiple frame cells atomically, replacing destination collisions."""
    sources = tuple(float(frame) for frame in source_frames)
    if not sources:
        return False
    if abs(delta) <= epsilon:
        return True

    moved = False
    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue
        points = fcurve.keyframe_points
        source_snapshots = [
            _snapshot_keyframe(fcurve, key)
            for key in points
            if any(abs(float(key.co.x) - frame) <= epsilon for frame in sources)
        ]
        if not source_snapshots:
            continue

        source_values = tuple(snapshot.frame for snapshot in source_snapshots)
        destination_values = tuple(frame + delta for frame in source_values)

        # Never mutate source RNA points in-place when selected source/destination
        # cells overlap. Blender may reorder/collapse FCurve points while their
        # coordinates are changing, which can send later source points to the
        # wrong frame. Rebuild the moved set atomically from snapshots instead.
        _remove_keyframe_points_at_frames(
            points,
            source_values + destination_values,
            epsilon,
        )
        for snapshot in source_snapshots:
            _insert_keyframe_snapshot(
                points,
                snapshot,
                frame_delta=delta,
                selected_override=True,
            )

        fcurve.update()
        moved = True

    return moved


def move_keyframe_points(
    fcurves: Iterable,
    source_frame: float,
    target_frame: float,
    *,
    data_path_prefix: str | None = None,
    epsilon: float = 1e-4,
) -> bool:
    """Move one complete frame cell, replacing same-FCurve destination keys."""
    if abs(target_frame - source_frame) <= epsilon:
        return True

    delta = target_frame - source_frame
    moved = False

    for fcurve in fcurves:
        if not _matches_context(fcurve, data_path_prefix):
            continue

        points = fcurve.keyframe_points
        source_indices = [
            index
            for index, key in enumerate(points)
            if abs(float(key.co.x) - source_frame) <= epsilon
        ]
        if not source_indices:
            continue

        destination_indices = [
            index
            for index, key in enumerate(points)
            if abs(float(key.co.x) - target_frame) <= epsilon
        ]

        # Match 3ds Max move-over-key behavior and Blender Auto-Merge semantics:
        # the moving key replaces only an existing key on the same FCurve.
        # Other channels at the same destination frame remain untouched.
        for index in source_indices:
            key = points[index]
            key.co.x = float(key.co.x) + delta
            key.handle_left.x = float(key.handle_left.x) + delta
            key.handle_right.x = float(key.handle_right.x) + delta

        # Moving the source does not reorder the RNA collection until update().
        # Remove the old destination by descending index before sorting/updating.
        for index in reversed(destination_indices):
            points.remove(points[index], fast=True)

        fcurve.update()
        moved = True

    return moved
