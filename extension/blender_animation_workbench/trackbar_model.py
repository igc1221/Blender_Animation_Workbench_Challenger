from __future__ import annotations

from dataclasses import dataclass

from .semantic_adapter import (
    active_control_for_context,
    assigned_channelbag,
    channel_binding_token,
    control_context_for_context,
    group_animation_owners,
    matches_control_path,
    trackbar_controls_for_context,
)
from .semantic_model import AWBControl
from .semantic_query import (
    contributors_for_frames,
    query_keys,
    resolve_contributor_key,
)
from .semantic_query import semantic_keys_for_context as _semantic_keys_for_context
from .trackbar_key_edit import (
    KeyPropertyEditResult,
    KeyPropertyState,
    KeyPropertyValueRow,
    apply_multi_key_preview_transaction,
    apply_selection_range_scale_transaction,
    clone_keyframe_points,
    clone_keyframe_points_at_frames,
    delete_keyframe_points,
    delete_keyframe_points_at_frames,
    move_keyframe_points,
    move_keyframe_points_at_frames,
    remove_cloned_keyframe_points,
    restore_keyframe_snapshots,
    restore_keyframe_state,
    restore_multi_key_preview_transaction,
    restore_selection_range_scale_transaction,
    set_keyframe_properties,
    snapshot_keyframe_points,
    snapshot_multi_key_preview_transaction,
    snapshot_replaced_keyframe_points,
    snapshot_selection_range_scale_transaction,
    summarize_key_property_rows,
)


@dataclass(frozen=True, slots=True)
class KeyPropertyTarget:
    """Operation-local address for one concrete keyframe contributor.

    The descriptor intentionally stores no Keyframe RNA reference. It captures
    the assigned animation binding identity and may only re-resolve while that
    exact Action / Slot / ChannelBag binding is still current.
    """

    owner_object: object
    control_id: str
    data_path_prefix: str | None
    data_path: str
    array_index: int
    frame: float
    binding_token: tuple[int | None, int | None, int | None, int | None]


@dataclass(frozen=True, slots=True)
class TrackbarOwnerEditTarget:
    """Operation-bound concrete ChannelBag/FCurve address set.

    Multiple selected owners may legitimately share one assigned ChannelBag. In
    that case this target owns exactly one mutation journal for the shared
    concrete curves while retaining every owner binding for stale-binding
    preflight and update tagging.
    """

    owner_object: object
    binding_token: tuple[int | None, int | None, int | None, int | None]
    curve_keys: tuple[tuple[str, int], ...]
    owner_bindings: tuple[
        tuple[object, tuple[int | None, int | None, int | None, int | None]], ...
    ]
    binding_key: tuple[int | None, int | None, int | None]


class ContextKeySnapshots:
    __slots__ = ("items",)

    def __init__(self, items) -> None:
        self.items = items

    def __bool__(self) -> bool:
        return any(item[-1] for item in self.items)


class ContextMultiKeyPreviewTransaction:
    __slots__ = ("captured_fcurves", "items", "mode", "source_frames")

    def __init__(
        self,
        mode: str,
        source_frames: tuple[float, ...],
        items,
        *,
        captured_fcurves=(),
    ) -> None:
        self.mode = mode
        self.source_frames = source_frames
        self.items = items
        self.captured_fcurves = tuple(captured_fcurves)


class ContextSelectionRangeScalePreviewTransaction:
    __slots__ = (
        "captured_fcurves",
        "items",
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
        items,
        captured_fcurves=(),
    ) -> None:
        self.source_frames = source_frames
        self.pivot_frame = int(pivot_frame)
        self.source_handle_frame = int(source_handle_frame)
        self.items = items
        self.captured_fcurves = tuple(captured_fcurves)
        self.selected_destination_frames = tuple(
            sorted({round(frame) for frame in source_frames})
        )


_TRACKBAR_EDIT_GUARDS: list[object] = []
_TRACKBAR_EDIT_EXPANDERS: list[object] = []
_TRACKBAR_DELETE_FINALIZERS: list[object] = []
_TRACKBAR_CELL_PROVIDERS: list[object] = []
_TRACKBAR_SELECTION_HANDLERS: list[object] = []


def register_trackbar_cell_provider(provider) -> None:
    """Register one higher-layer semantic Track Bar cell provider.

    Providers return ``{frame: selected}`` for semantic cells that may not have
    a directly visible FCurve on the currently selected control.
    """
    if provider not in _TRACKBAR_CELL_PROVIDERS:
        _TRACKBAR_CELL_PROVIDERS.append(provider)


def unregister_trackbar_cell_provider(provider) -> None:
    """Remove one previously registered semantic Track Bar cell provider."""
    if provider in _TRACKBAR_CELL_PROVIDERS:
        _TRACKBAR_CELL_PROVIDERS.remove(provider)


def register_trackbar_selection_handler(handler) -> None:
    """Register one higher-layer semantic Track Bar selection handler."""
    if handler not in _TRACKBAR_SELECTION_HANDLERS:
        _TRACKBAR_SELECTION_HANDLERS.append(handler)


def unregister_trackbar_selection_handler(handler) -> None:
    """Remove one previously registered semantic Track Bar selection handler."""
    if handler in _TRACKBAR_SELECTION_HANDLERS:
        _TRACKBAR_SELECTION_HANDLERS.remove(handler)


def _trackbar_extension_cells(context) -> dict[float, bool]:
    cells: dict[float, bool] = {}
    for provider in tuple(_TRACKBAR_CELL_PROVIDERS):
        provided = provider(context) or {}
        for frame, selected in provided.items():
            frame_value = float(frame)
            cells[frame_value] = cells.get(frame_value, False) or bool(selected)
    return cells


def trackbar_extension_cells_for_context(context) -> dict[float, bool]:
    """Return higher-layer semantic cells without re-querying ordinary FCurves."""
    return _trackbar_extension_cells(context)


def _select_trackbar_extension_frames(context, frames, *, mode: str) -> bool:
    frozen_frames = tuple(float(frame) for frame in frames)
    found = False
    for handler in tuple(_TRACKBAR_SELECTION_HANDLERS):
        found = bool(handler(context, frames=frozen_frames, mode=mode)) or found
    return found


def register_trackbar_edit_guard(guard) -> None:
    """Register one idempotent higher-layer preflight guard for generic Track Bar edits."""
    if guard not in _TRACKBAR_EDIT_GUARDS:
        _TRACKBAR_EDIT_GUARDS.append(guard)


def unregister_trackbar_edit_guard(guard) -> None:
    """Remove one previously registered Track Bar edit guard."""
    if guard in _TRACKBAR_EDIT_GUARDS:
        _TRACKBAR_EDIT_GUARDS.remove(guard)


def register_trackbar_edit_expander(expander) -> None:
    """Register one higher-layer semantic curve-bundle expander."""
    if expander not in _TRACKBAR_EDIT_EXPANDERS:
        _TRACKBAR_EDIT_EXPANDERS.append(expander)


def unregister_trackbar_edit_expander(expander) -> None:
    """Remove one previously registered Track Bar edit expander."""
    if expander in _TRACKBAR_EDIT_EXPANDERS:
        _TRACKBAR_EDIT_EXPANDERS.remove(expander)


def register_trackbar_delete_finalizer(finalizer) -> None:
    """Register one higher-layer post-delete semantic cleanup hook."""
    if finalizer not in _TRACKBAR_DELETE_FINALIZERS:
        _TRACKBAR_DELETE_FINALIZERS.append(finalizer)


def unregister_trackbar_delete_finalizer(finalizer) -> None:
    """Remove one previously registered Track Bar post-delete finalizer."""
    if finalizer in _TRACKBAR_DELETE_FINALIZERS:
        _TRACKBAR_DELETE_FINALIZERS.remove(finalizer)


def _finalize_trackbar_delete(context, frames) -> None:
    frozen_frames = tuple(float(frame) for frame in frames)
    for finalizer in tuple(_TRACKBAR_DELETE_FINALIZERS):
        finalizer(context, frames=frozen_frames)


def _trackbar_edit_allowed(
    context,
    operation: str,
    source_frames,
    target_frames=(),
) -> bool:
    """Run higher-layer guards before any generic key mutation.

    The Track Bar remains domain-agnostic: Phase 4 or later systems may protect
    semantic bundles without creating a reverse import from this foundational
    edit layer.
    """
    sources = tuple(float(frame) for frame in source_frames)
    targets = tuple(float(frame) for frame in target_frames)
    for guard in tuple(_TRACKBAR_EDIT_GUARDS):
        allowed = guard(
            context,
            operation=str(operation),
            source_frames=sources,
            target_frames=targets,
        )
        if not bool(allowed):
            return False
    return True


def animation_targets_for_context(context):
    """Return selected AWB controls grouped by animation owner object."""
    control_context = control_context_for_context(context)
    targets = []
    for group in group_animation_owners(control_context.controls):
        selector = group.prefixes[0] if len(group.prefixes) == 1 else group.prefixes
        targets.append((group.owner_object, selector))
    return targets


def _channelbag_for_object(obj):
    """Compatibility wrapper around centralized assigned-slot lookup."""
    return assigned_channelbag(obj)


def _target_channelbags_for_context(context):
    targets = animation_targets_for_context(context)
    return [
        (obj, selector, channelbag)
        for obj, selector in targets
        if (channelbag := _channelbag_for_object(obj)) is not None
    ]


def _animation_binding_key(owner) -> tuple[int | None, int | None, int | None]:
    """Return the stable concrete animation address shared by equivalent owners.

    ChannelBag Python wrappers are not a safe identity boundary in Blender 5.2:
    repeated lookups may produce different wrapper objects even for the same
    assigned Action/Slot. The binding token already uses native RNA pointers
    where available, so grouping on Action/Slot/ChannelBag address avoids a
    transient ``id(wrapper)`` false-stale result while still separating slots.
    """
    return channel_binding_token(owner)[1:]


def _edit_targets_for_frames(context, frames) -> tuple[TrackbarOwnerEditTarget, ...]:
    """Expand semantic frame cells to one mutation target per concrete animation binding."""
    semantic = query_keys(control_context_for_context(context))
    contributors = contributors_for_frames(semantic, frames)
    grouped: list[list] = []
    binding_indexes: dict[tuple[int | None, int | None, int | None], int] = {}

    def add_curve_keys(owner, curve_keys) -> None:
        if assigned_channelbag(owner) is None:
            return
        binding_key = _animation_binding_key(owner)
        index = binding_indexes.get(binding_key)
        if index is None:
            binding_indexes[binding_key] = len(grouped)
            grouped.append([owner, binding_key, {}, {}])
            index = len(grouped) - 1
        owner_pointer = _runtime_pointer(owner)
        grouped[index][2].setdefault(
            owner_pointer,
            (owner, channel_binding_token(owner)),
        )
        for data_path, array_index in curve_keys:
            grouped[index][3].setdefault((str(data_path), int(array_index)), None)

    for contributor in contributors:
        add_curve_keys(
            contributor.channel.resolved.owner_object,
            ((contributor.channel.data_path, contributor.channel.array_index),),
        )

    frozen_frames = tuple(float(frame) for frame in frames)
    for expander in tuple(_TRACKBAR_EDIT_EXPANDERS):
        rows = expander(context, frames=frozen_frames)
        for owner, curve_keys in rows or ():
            add_curve_keys(owner, curve_keys)

    return tuple(
        TrackbarOwnerEditTarget(
            owner_object=owner,
            binding_token=channel_binding_token(owner),
            curve_keys=tuple(curve_keys),
            owner_bindings=tuple(owner_bindings.values()),
            binding_key=binding_key,
        )
        for owner, binding_key, owner_bindings, curve_keys in grouped
        if curve_keys
    )


def _fcurves_for_edit_target(target: TrackbarOwnerEditTarget):
    """Re-resolve one concrete edit target only while every captured binding matches."""
    channelbag = None
    for owner, binding_token in target.owner_bindings:
        if channel_binding_token(owner) != binding_token:
            return None
        if _animation_binding_key(owner) != target.binding_key:
            return None
        owner_channelbag = assigned_channelbag(owner)
        if owner_channelbag is None:
            return None
        if channelbag is None:
            channelbag = owner_channelbag

    if channelbag is None:
        return None
    by_key = {
        (str(fcurve.data_path), int(getattr(fcurve, "array_index", 0))): fcurve
        for fcurve in channelbag.fcurves
    }
    if any(curve_key not in by_key for curve_key in target.curve_keys):
        return None
    return tuple(by_key[curve_key] for curve_key in target.curve_keys)


def _captured_fcurves_for_edit_target(target: TrackbarOwnerEditTarget, fcurves):
    """Validate preview-captured FCurves without rebuilding the ChannelBag index.

    Modal Track Bar previews never add/remove FCurves; they only rewrite points.
    Capture concrete curves once at drag start, then validate owner binding identity
    plus each curve's data-path/index on mousemove. If an external mutation removes
    an FCurve, its RNA wrapper raises ``ReferenceError`` and the preview fails closed.
    """
    for owner, binding_token in target.owner_bindings:
        if channel_binding_token(owner) != binding_token:
            return None
        if _animation_binding_key(owner) != target.binding_key:
            return None
    if len(fcurves) != len(target.curve_keys):
        return None
    try:
        for fcurve, curve_key in zip(fcurves, target.curve_keys, strict=True):
            current_key = (
                str(fcurve.data_path),
                int(getattr(fcurve, "array_index", 0)),
            )
            if current_key != curve_key:
                return None
    except (ReferenceError, RuntimeError):
        return None
    return fcurves


def _preview_fcurves_for_item(transaction, index: int, target: TrackbarOwnerEditTarget):
    captured = getattr(transaction, "captured_fcurves", ())
    if index < len(captured):
        return _captured_fcurves_for_edit_target(target, captured[index])
    # Compatibility for manually constructed/test transactions that predate the
    # preview cache. They retain the exact original fail-closed re-resolution path.
    return _fcurves_for_edit_target(target)


def _animation_targets_for_edit_target(target: TrackbarOwnerEditTarget):
    return tuple((owner, None) for owner, _binding_token in target.owner_bindings)


def _tag_animation_targets(context, targets, *, refresh_time: bool = True) -> None:
    tagged: set[int] = set()
    for obj, _selector in targets:
        pointer = int(obj.as_pointer())
        if pointer in tagged:
            continue
        tagged.add(pointer)
        obj.update_tag(refresh={"OBJECT", "DATA", "TIME"})
    if refresh_time:
        context.scene.frame_set(context.scene.frame_current)


def active_animation_target(context):
    """Return the single-owner target used by legacy/single-owner callers.

    Multi-object Object Mode intentionally returns ``(None, None)`` here;
    context-wide edit helpers use :func:`animation_targets_for_context` instead.
    """
    targets = animation_targets_for_context(context)
    if len(targets) != 1:
        return None, None
    return targets[0]


def active_awb_control(context) -> AWBControl | None:
    resolved = active_control_for_context(context)
    return resolved.control if resolved is not None else None


def _matches_active_control_path(data_path: str, data_path_prefix) -> bool:
    """Compatibility wrapper around the centralized semantic path matcher."""
    return matches_control_path(data_path, data_path_prefix)


def _runtime_pointer(value) -> int:
    as_pointer = getattr(value, "as_pointer", None)
    if callable(as_pointer):
        return int(as_pointer())
    return id(value)


def key_property_targets_for_context(
    context,
    *,
    hit_cell: int | None = None,
) -> tuple[KeyPropertyTarget, ...]:
    """Resolve concrete Track Bar key-property targets without changing selection.

    With ``hit_cell=None`` only actually selected control points are returned.
    With a clicked cell, an already-selected contributor keeps the whole actual
    selected-point set; otherwise only contributors in the clicked display cell
    are returned while the existing selection remains untouched.
    """
    resolved_controls = trackbar_controls_for_context(context)
    if not resolved_controls:
        return ()

    channelbags: dict[int, object | None] = {}
    candidates: list[tuple[KeyPropertyTarget, bool, int]] = []
    seen: set[tuple[int, int, float]] = set()

    for resolved in resolved_controls:
        owner = resolved.owner_object
        owner_pointer = _runtime_pointer(owner)
        if owner_pointer not in channelbags:
            channelbags[owner_pointer] = _channelbag_for_object(owner)
        channelbag = channelbags[owner_pointer]
        if channelbag is None:
            continue

        for fcurve in channelbag.fcurves:
            if not _matches_active_control_path(
                fcurve.data_path,
                resolved.data_path_prefix,
            ):
                continue
            curve_pointer = _runtime_pointer(fcurve)
            for key in fcurve.keyframe_points:
                frame = float(key.co.x)
                identity = (owner_pointer, curve_pointer, frame)
                if identity in seen:
                    continue
                seen.add(identity)
                target = KeyPropertyTarget(
                    owner_object=owner,
                    control_id=resolved.control.control_id,
                    data_path_prefix=resolved.data_path_prefix,
                    data_path=str(fcurve.data_path),
                    array_index=int(getattr(fcurve, "array_index", 0)),
                    frame=frame,
                    binding_token=channel_binding_token(owner),
                )
                candidates.append(
                    (target, bool(key.select_control_point), round(frame))
                )

    selected_targets = tuple(
        target for target, selected, _cell in candidates if selected
    )
    if hit_cell is None:
        return selected_targets

    cell = int(hit_cell)
    clicked = tuple(
        (target, selected)
        for target, selected, candidate_cell in candidates
        if candidate_cell == cell
    )
    if not clicked:
        return ()
    if any(selected for _target, selected in clicked):
        return selected_targets
    return tuple(target for target, _selected in clicked)


def _key_property_value_row(
    target: KeyPropertyTarget,
    *,
    epsilon: float = 1e-4,
) -> KeyPropertyValueRow | None:
    if channel_binding_token(target.owner_object) != target.binding_token:
        return None
    channelbag = _channelbag_for_object(target.owner_object)
    if channelbag is None:
        return None
    fcurve = next(
        (
            curve
            for curve in channelbag.fcurves
            if str(curve.data_path) == target.data_path
            and int(getattr(curve, "array_index", 0)) == target.array_index
        ),
        None,
    )
    if fcurve is None:
        return None
    key = next(
        (
            point
            for point in fcurve.keyframe_points
            if abs(float(point.co.x) - target.frame) <= epsilon
        ),
        None,
    )
    if key is None:
        return None
    return KeyPropertyValueRow(
        control_id=target.control_id,
        frame=target.frame,
        interpolation=str(key.interpolation),
        handle_left_type=str(key.handle_left_type),
        handle_right_type=str(key.handle_right_type),
    )


def key_property_state_for_targets(
    targets: tuple[KeyPropertyTarget, ...] | list[KeyPropertyTarget],
) -> KeyPropertyState:
    """Summarize current native key properties for operation-local targets."""
    rows = []
    for target in targets:
        row = _key_property_value_row(target)
        if row is not None:
            rows.append(row)
    return summarize_key_property_rows(rows)


def _resolve_key_property_point(
    target: KeyPropertyTarget,
    *,
    epsilon: float = 1e-4,
):
    if channel_binding_token(target.owner_object) != target.binding_token:
        return None
    channelbag = _channelbag_for_object(target.owner_object)
    if channelbag is None:
        return None
    fcurve = next(
        (
            curve
            for curve in channelbag.fcurves
            if str(curve.data_path) == target.data_path
            and int(getattr(curve, "array_index", 0)) == target.array_index
        ),
        None,
    )
    if fcurve is None:
        return None
    key = next(
        (
            point
            for point in fcurve.keyframe_points
            if abs(float(point.co.x) - target.frame) <= epsilon
        ),
        None,
    )
    if key is None:
        return None
    return fcurve, key


def apply_key_properties_for_context(
    context,
    targets: tuple[KeyPropertyTarget, ...] | list[KeyPropertyTarget],
    *,
    interpolation: str | None = None,
    handle_type: str | None = None,
) -> KeyPropertyEditResult:
    """Preflight operation-local targets, then apply one native property batch."""
    if not targets:
        return KeyPropertyEditResult(0, 0, 0)
    if not _trackbar_edit_allowed(
        context,
        "KEY_PROPERTIES",
        (target.frame for target in targets),
    ):
        return KeyPropertyEditResult(0, 0, 0)

    current_controls = trackbar_controls_for_context(context)
    allowed = {
        (
            _runtime_pointer(resolved.owner_object),
            resolved.control.control_id,
            resolved.data_path_prefix,
        )
        for resolved in current_controls
    }

    point_rows = []
    touched: list[tuple[object, str | None]] = []
    touched_ids: set[tuple[int, str | None]] = set()
    seen_points: set[int] = set()
    for target in targets:
        context_key = (
            _runtime_pointer(target.owner_object),
            target.control_id,
            target.data_path_prefix,
        )
        if context_key not in allowed:
            raise RuntimeError("Key Properties target is stale for the current AWB context")
        resolved = _resolve_key_property_point(target)
        if resolved is None:
            raise RuntimeError("Key Properties target no longer resolves in the assigned ChannelBag")
        fcurve, key = resolved
        point_id = id(key)
        if point_id in seen_points:
            continue
        seen_points.add(point_id)
        point_rows.append((fcurve, key))
        touched_key = (_runtime_pointer(target.owner_object), target.data_path_prefix)
        if touched_key not in touched_ids:
            touched_ids.add(touched_key)
            touched.append((target.owner_object, target.data_path_prefix))

    result = set_keyframe_properties(
        point_rows,
        interpolation=interpolation,
        handle_type=handle_type,
    )
    if result.changed_points:
        _tag_animation_targets(context, touched)
    return result


def semantic_keys_for_context(context):
    """Compatibility façade backed by the Phase 2 read-only semantic query layer."""
    return _semantic_keys_for_context(context)


def key_channels_for_context(context) -> dict[float, set[str]]:
    channels_by_frame: dict[float, set[str]] = {}
    for key in semantic_keys_for_context(context):
        channels_by_frame.setdefault(key.frame, set()).update(key.channel_names)
    return channels_by_frame


def key_frames_for_context(context) -> list[float]:
    return sorted(key_channels_for_context(context))


def navigation_key_frames_for_context(context) -> list[float]:
    """Return sorted unique transform-key times for current Track Bar controls."""
    navigation_channels = {"POSITION", "ROTATION", "SCALE"}
    return sorted(
        {
            key.frame
            for key in semantic_keys_for_context(context)
            if navigation_channels.intersection(key.channel_names)
        }
    )


def adjacent_key_frame_for_context(
    context,
    current: float,
    *,
    next: bool,
) -> float | None:
    """Return the strict previous/next semantic key time without wrapping."""
    frames = navigation_key_frames_for_context(context)
    if next:
        for frame in frames:
            if frame > current:
                return frame
        return None

    for frame in reversed(frames):
        if frame < current:
            return frame
    return None


def has_key_at_frame(context, frame: int, epsilon: float = 1e-4) -> bool:
    if any(abs(key_frame - frame) <= epsilon for key_frame in key_channels_for_context(context)):
        return True
    return any(
        abs(key_frame - frame) <= epsilon
        for key_frame in _trackbar_extension_cells(context)
    )


def selected_key_frames_for_context(context) -> list[float]:
    """Return unique selected semantic frame cells for current Track Bar controls."""
    selected = {
        key.frame
        for key in semantic_keys_for_context(context)
        if key.selected
    }
    selected.update(
        frame
        for frame, is_selected in _trackbar_extension_cells(context).items()
        if is_selected
    )
    return sorted(selected)


def snapshot_active_key_state_for_context(context) -> ContextKeySnapshots:
    """Capture complete selected-control key state for transactional previews."""
    items = []
    for obj, selector, channelbag in _target_channelbags_for_context(context):
        snapshots = snapshot_keyframe_points(
            channelbag.fcurves,
            data_path_prefix=selector,
        )
        items.append((obj, selector, channel_binding_token(obj), snapshots))
    return ContextKeySnapshots(items)


def restore_active_key_state_for_context(
    context,
    snapshots: ContextKeySnapshots,
) -> bool:
    """Restore exact selected-control state only to the captured animation bindings."""
    if not snapshots:
        return False

    resolved_items = []
    for obj, selector, binding_token, owner_snapshots in snapshots.items:
        if channel_binding_token(obj) != binding_token:
            return False
        channelbag = _channelbag_for_object(obj)
        if channelbag is None:
            return False
        resolved_items.append((obj, selector, channelbag, owner_snapshots))

    restored_any = False
    touched = []
    for obj, selector, channelbag, owner_snapshots in resolved_items:
        restored = restore_keyframe_state(
            channelbag.fcurves,
            owner_snapshots,
            data_path_prefix=selector,
        )
        if restored:
            restored_any = True
            touched.append((obj, selector))
    if restored_any:
        _tag_animation_targets(context, touched)
    return restored_any


def snapshot_multi_key_preview_for_context(
    context,
    source_frames: list[float],
    *,
    mode: str,
    guard_operation: str | None = None,
) -> ContextMultiKeyPreviewTransaction | None:
    """Capture one lightweight preview transaction per bound animation owner."""
    sources = tuple(float(frame) for frame in source_frames)
    operation = guard_operation or f"PREVIEW_{str(mode).upper()}"
    if not _trackbar_edit_allowed(context, operation, sources):
        return None
    items = []
    captured_fcurves = []
    for target in _edit_targets_for_frames(context, sources):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        transaction = snapshot_multi_key_preview_transaction(
            fcurves,
            sources,
            mode=mode,
            data_path_prefix=None,
        )
        if transaction is not None:
            items.append((target, transaction))
            captured_fcurves.append(fcurves)
    if not items:
        return None
    return ContextMultiKeyPreviewTransaction(
        str(mode).upper(),
        sources,
        items,
        captured_fcurves=captured_fcurves,
    )


def update_multi_key_preview_for_context(
    context,
    transaction: ContextMultiKeyPreviewTransaction,
    delta_frames: int,
) -> bool:
    """Update previews only when the complete captured binding batch is still valid."""
    resolved_items = []
    for index, (target, owner_transaction) in enumerate(transaction.items):
        fcurves = _preview_fcurves_for_item(transaction, index, target)
        if not fcurves:
            return False
        resolved_items.append((target, owner_transaction, fcurves))

    changed_any = False
    touched = []
    for target, owner_transaction, fcurves in resolved_items:
        changed = apply_multi_key_preview_transaction(
            fcurves,
            owner_transaction,
            delta_frames,
            data_path_prefix=None,
        )
        if changed:
            changed_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    if changed_any:
        # Preview insertion already writes the moved/cloned destination points
        # with the correct selected state. Re-running semantic clear/reselect on
        # every mousemove is redundant and especially expensive for Rigped
        # Contact because SET selection scans every authored limb bundle.
        # The gizmo immediately frame_set()s its preview time after this call,
        # so tag the owners here without forcing a second depsgraph time refresh.
        _tag_animation_targets(context, touched, refresh_time=False)
    return changed_any


def restore_multi_key_preview_for_context(
    context,
    transaction: ContextMultiKeyPreviewTransaction,
) -> bool:
    """Restore previews only when the complete captured binding batch is still valid."""
    resolved_items = []
    for index, (target, owner_transaction) in enumerate(transaction.items):
        fcurves = _preview_fcurves_for_item(transaction, index, target)
        if not fcurves:
            return False
        resolved_items.append((target, owner_transaction, fcurves))

    restored_any = False
    touched = []
    for target, owner_transaction, fcurves in resolved_items:
        restored = restore_multi_key_preview_transaction(
            fcurves,
            owner_transaction,
            data_path_prefix=None,
        )
        if restored:
            restored_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    clear_key_selection_for_context(context)
    for frame in transaction.source_frames:
        select_key_frame_for_context(context, frame, mode="ADD")
    if restored_any:
        _tag_animation_targets(context, touched)
    return restored_any


def snapshot_selection_range_scale_for_context(
    context,
    source_frames: list[float],
    *,
    pivot_frame: int,
    source_handle_frame: int,
) -> ContextSelectionRangeScalePreviewTransaction | None:
    """Capture one endpoint-scale preview transaction per bound animation owner."""
    sources = tuple(float(frame) for frame in source_frames)
    if not _trackbar_edit_allowed(context, "SELECTION_RANGE_SCALE", sources):
        return None
    items = []
    captured_fcurves = []
    for target in _edit_targets_for_frames(context, sources):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        transaction = snapshot_selection_range_scale_transaction(
            fcurves,
            sources,
            pivot_frame=pivot_frame,
            source_handle_frame=source_handle_frame,
            data_path_prefix=None,
        )
        if transaction is not None:
            items.append((target, transaction))
            captured_fcurves.append(fcurves)
    if not items:
        return None
    return ContextSelectionRangeScalePreviewTransaction(
        sources,
        pivot_frame=pivot_frame,
        source_handle_frame=source_handle_frame,
        items=items,
        captured_fcurves=captured_fcurves,
    )


def update_selection_range_scale_preview_for_context(
    context,
    transaction: ContextSelectionRangeScalePreviewTransaction,
    target_handle_frame: int,
) -> bool:
    """Recompute scale previews only when the complete captured binding batch is valid."""
    resolved_items = []
    for index, (target, owner_transaction) in enumerate(transaction.items):
        fcurves = _preview_fcurves_for_item(transaction, index, target)
        if not fcurves:
            return False
        resolved_items.append((target, owner_transaction, fcurves))

    changed_any = False
    touched = []
    destination_frames: set[int] = set()
    for target, owner_transaction, fcurves in resolved_items:
        changed = apply_selection_range_scale_transaction(
            fcurves,
            owner_transaction,
            target_handle_frame,
            data_path_prefix=None,
        )
        destination_frames.update(owner_transaction.selected_destination_frames)
        if changed:
            changed_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    if changed_any:
        transaction.selected_destination_frames = tuple(sorted(destination_frames))
        # Scaled preview insertion already marks every destination key selected.
        # Rebuilding semantic/Contact selection on every handle mousemove scans
        # all authored Rigped limb bundles and is redundant. Keep live preview
        # ownership tags lightweight; the gizmo owns any required time refresh.
        _tag_animation_targets(context, touched, refresh_time=False)
    return changed_any


def restore_selection_range_scale_preview_for_context(
    context,
    transaction: ContextSelectionRangeScalePreviewTransaction,
) -> bool:
    """Restore scale previews only when the complete captured binding batch is valid."""
    resolved_items = []
    for index, (target, owner_transaction) in enumerate(transaction.items):
        fcurves = _preview_fcurves_for_item(transaction, index, target)
        if not fcurves:
            return False
        resolved_items.append((target, owner_transaction, fcurves))

    restored_any = False
    touched = []
    for target, owner_transaction, fcurves in resolved_items:
        restored = restore_selection_range_scale_transaction(
            fcurves,
            owner_transaction,
            data_path_prefix=None,
        )
        if restored:
            restored_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    clear_key_selection_for_context(context)
    for frame in transaction.source_frames:
        select_key_frame_for_context(context, round(frame), mode="ADD")
    if restored_any:
        _tag_animation_targets(context, touched)
    return restored_any


def scale_selected_key_frames_for_context(
    context,
    source_frames: list[float],
    *,
    pivot_frame: int,
    source_handle_frame: int,
    target_handle_frame: int,
) -> bool:
    """Commit one simultaneous endpoint proportional time-scale edit."""
    transaction = snapshot_selection_range_scale_for_context(
        context,
        source_frames,
        pivot_frame=pivot_frame,
        source_handle_frame=source_handle_frame,
    )
    if transaction is None:
        return False
    return update_selection_range_scale_preview_for_context(
        context,
        transaction,
        target_handle_frame,
    )


def delete_selected_keys_for_context(context) -> bool:
    """Delete selected aggregate frame cells through semantic contributors."""
    selected_frames = selected_key_frames_for_context(context)
    if not selected_frames:
        return False
    if not _trackbar_edit_allowed(context, "DELETE", selected_frames):
        return False
    deleted_any = False
    touched = []
    for target in _edit_targets_for_frames(context, selected_frames):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        deleted = delete_keyframe_points_at_frames(
            fcurves,
            selected_frames,
            data_path_prefix=None,
        )
        if deleted:
            deleted_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    if deleted_any:
        context.scene.baw_has_selected_key = False
        _finalize_trackbar_delete(context, selected_frames)
        _tag_animation_targets(context, touched)
    return deleted_any


def clone_selected_key_frames_for_context(
    context,
    source_frames: list[float],
    delta_frames: int,
) -> bool:
    """Clone aggregate frame contributors across every selected control."""
    if not source_frames or delta_frames == 0:
        return False
    target_frames = tuple(float(frame) + float(delta_frames) for frame in source_frames)
    if not _trackbar_edit_allowed(context, "CLONE", source_frames, target_frames):
        return False
    cloned_any = False
    touched = []
    for target in _edit_targets_for_frames(context, source_frames):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        cloned = clone_keyframe_points_at_frames(
            fcurves,
            source_frames,
            float(delta_frames),
            data_path_prefix=None,
        )
        if cloned:
            cloned_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    if cloned_any:
        _tag_animation_targets(context, touched)
    return cloned_any


def move_selected_key_frames_for_context(
    context,
    source_frames: list[float],
    delta_frames: int,
) -> bool:
    """Move aggregate frame contributors across every selected control."""
    if not source_frames:
        return False
    target_frames = tuple(float(frame) + float(delta_frames) for frame in source_frames)
    if not _trackbar_edit_allowed(context, "MOVE", source_frames, target_frames):
        return False
    moved_any = False
    touched = []
    for target in _edit_targets_for_frames(context, source_frames):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        moved = move_keyframe_points_at_frames(
            fcurves,
            source_frames,
            float(delta_frames),
            data_path_prefix=None,
        )
        if moved:
            moved_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    if moved_any:
        _tag_animation_targets(context, touched)
    return moved_any


def select_key_frame_for_context(
    context,
    frame: int,
    epsilon: float = 1e-4,
    *,
    mode: str = "SET",
) -> bool:
    """Select one aggregate semantic frame across all selected contributors."""
    if mode not in {"SET", "ADD", "TOGGLE", "SUB"}:
        raise ValueError(f"Unsupported Track Bar key selection mode: {mode}")

    semantic = query_keys(control_context_for_context(context))
    tolerance = abs(float(epsilon))
    target_contributors = tuple(
        contributor
        for contributor in semantic.contributors
        if abs(contributor.frame - float(frame)) <= tolerance
    )

    if mode == "SET":
        for contributor in semantic.contributors:
            key = resolve_contributor_key(contributor, epsilon=epsilon)
            if key is None:
                continue
            is_target = abs(contributor.frame - float(frame)) <= tolerance
            key.select_control_point = is_target
            key.select_left_handle = is_target
            key.select_right_handle = is_target
    elif target_contributors:
        target_selected = any(contributor.selected for contributor in target_contributors)
        if mode == "ADD":
            select_target = True
        elif mode == "SUB":
            select_target = False
        else:
            select_target = not target_selected
        for contributor in target_contributors:
            key = resolve_contributor_key(contributor, epsilon=epsilon)
            if key is None:
                continue
            key.select_control_point = select_target
            key.select_left_handle = select_target
            key.select_right_handle = select_target

    extension_found = _select_trackbar_extension_frames(
        context,
        (float(frame),),
        mode=mode,
    )
    return bool(target_contributors) or extension_found


def clear_key_selection_for_context(context) -> bool:
    """Clear keyframe-point and handle selection for semantic contributors."""
    changed = False
    semantic = query_keys(control_context_for_context(context))
    for contributor in semantic.contributors:
        key = resolve_contributor_key(contributor)
        if key is None:
            continue
        changed = changed or bool(
            key.select_control_point or key.select_left_handle or key.select_right_handle
        )
        key.select_control_point = False
        key.select_left_handle = False
        key.select_right_handle = False
    extension_changed = _select_trackbar_extension_frames(context, (), mode="SET")
    return changed or extension_changed


def select_key_frame_range_for_context(
    context,
    start_frame: int,
    end_frame: int,
    *,
    mode: str = "SET",
) -> bool:
    """Select aggregate semantic frame cells inside an inclusive range."""
    if mode not in {"SET", "ADD", "TOGGLE", "SUB"}:
        raise ValueError(f"Unsupported Track Bar range selection mode: {mode}")

    first = min(int(start_frame), int(end_frame))
    last = max(int(start_frame), int(end_frame))
    semantic = query_keys(control_context_for_context(context))
    contributors_by_cell: dict[int, list] = {}
    selected_by_cell: dict[int, bool] = {}
    for contributor in semantic.contributors:
        cell_frame = round(contributor.frame)
        contributors_by_cell.setdefault(cell_frame, []).append(contributor)
        selected_by_cell[cell_frame] = selected_by_cell.get(cell_frame, False) or bool(
            contributor.selected
        )

    found = any(first <= frame <= last for frame in contributors_by_cell)
    for cell_frame, contributors in contributors_by_cell.items():
        in_range = first <= cell_frame <= last
        currently_selected = selected_by_cell.get(cell_frame, False)
        if mode == "SET":
            selected = in_range
        elif mode == "ADD":
            selected = currently_selected or in_range
        elif mode == "SUB":
            selected = currently_selected and not in_range
        else:
            selected = (not currently_selected) if in_range else currently_selected
        for contributor in contributors:
            key = resolve_contributor_key(contributor)
            if key is None:
                continue
            key.select_control_point = selected
            key.select_left_handle = selected
            key.select_right_handle = selected
    extension_found = _select_trackbar_extension_frames(
        context,
        tuple(float(frame) for frame in range(first, last + 1)),
        mode=mode,
    )
    return found or extension_found


def move_key_frame_for_context(
    context,
    source_frame: int,
    target_frame: int,
    epsilon: float = 1e-4,
) -> bool:
    """Move one aggregate frame through its concrete semantic contributors."""
    if not _trackbar_edit_allowed(context, "MOVE", (source_frame,), (target_frame,)):
        return False
    moved_any = False
    touched = []
    for target in _edit_targets_for_frames(context, (source_frame,)):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        moved = move_keyframe_points(
            fcurves,
            source_frame,
            target_frame,
            data_path_prefix=None,
            epsilon=epsilon,
        )
        if moved:
            moved_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    if moved_any:
        _tag_animation_targets(context, touched)
    return moved_any


def clone_key_frame_for_context(
    context,
    source_frame: int,
    target_frame: int,
    epsilon: float = 1e-4,
) -> bool:
    """Clone one aggregate frame through its concrete semantic contributors."""
    if not _trackbar_edit_allowed(context, "CLONE", (source_frame,), (target_frame,)):
        return False
    cloned_any = False
    touched = []
    for target in _edit_targets_for_frames(context, (source_frame,)):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        cloned = clone_keyframe_points(
            fcurves,
            source_frame,
            target_frame,
            data_path_prefix=None,
            epsilon=epsilon,
        )
        if cloned:
            cloned_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    if cloned_any:
        _tag_animation_targets(context, touched)
    return cloned_any


def remove_cloned_key_frame_for_context(
    context,
    source_frame: int,
    target_frame: int,
    epsilon: float = 1e-4,
) -> bool:
    """Remove aggregate clone previews through the original source contributors."""
    removed_any = False
    touched = []
    for target in _edit_targets_for_frames(context, (source_frame,)):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        removed = remove_cloned_keyframe_points(
            fcurves,
            source_frame,
            target_frame,
            data_path_prefix=None,
            epsilon=epsilon,
        )
        if removed:
            removed_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    if removed_any:
        _tag_animation_targets(context, touched)
    return removed_any


def snapshot_replaced_key_frame_for_context(
    context,
    source_frame: int,
    target_frame: int,
    epsilon: float = 1e-4,
) -> ContextKeySnapshots:
    """Capture destination collisions only on source-contributing FCurves."""
    items = []
    for target in _edit_targets_for_frames(context, (source_frame,)):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        snapshots = snapshot_replaced_keyframe_points(
            fcurves,
            source_frame,
            target_frame,
            data_path_prefix=None,
            epsilon=epsilon,
        )
        items.append((target.owner_object, target, snapshots))
    return ContextKeySnapshots(items)


def restore_key_frame_snapshots_for_context(
    context,
    snapshots: ContextKeySnapshots,
) -> bool:
    """Restore collision snapshots only to their captured animation binding."""
    if not snapshots:
        return False
    restored_any = False
    touched = []
    for obj, selector, owner_snapshots in snapshots.items:
        if not owner_snapshots:
            continue
        if isinstance(selector, TrackbarOwnerEditTarget):
            fcurves = _fcurves_for_edit_target(selector)
            data_path_prefix = None
        else:
            channelbag = _channelbag_for_object(obj)
            fcurves = channelbag.fcurves if channelbag is not None else None
            data_path_prefix = selector
        if not fcurves:
            continue
        restored = restore_keyframe_snapshots(
            fcurves,
            owner_snapshots,
            data_path_prefix=data_path_prefix,
        )
        if restored:
            restored_any = True
            if isinstance(selector, TrackbarOwnerEditTarget):
                touched.extend(_animation_targets_for_edit_target(selector))
            else:
                touched.append((obj, data_path_prefix))
    if restored_any:
        _tag_animation_targets(context, touched)
    return restored_any


def delete_selected_key_for_context(context) -> bool:
    scene = context.scene
    if not scene.baw_has_selected_key:
        return False
    frame = int(scene.baw_selected_key_frame)
    if not _trackbar_edit_allowed(context, "DELETE", (frame,)):
        return False
    # A scene-level marker alone is insufficient after changing selected controls.
    if not any(
        key.selected and abs(key.frame - frame) <= 1e-4
        for key in semantic_keys_for_context(context)
    ):
        return False

    deleted_any = False
    touched = []
    for target in _edit_targets_for_frames(context, (frame,)):
        fcurves = _fcurves_for_edit_target(target)
        if not fcurves:
            continue
        deleted = delete_keyframe_points(
            fcurves,
            frame,
            data_path_prefix=None,
        )
        if deleted:
            deleted_any = True
            touched.extend(_animation_targets_for_edit_target(target))
    if deleted_any:
        scene.baw_has_selected_key = False
        _finalize_trackbar_delete(context, (frame,))
        _tag_animation_targets(context, touched)
    return deleted_any
