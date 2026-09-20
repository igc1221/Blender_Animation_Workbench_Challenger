from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .semantic_adapter import (
    ControlContext,
    ResolvedChannel,
    channels_for_controls,
    control_context_for_context,
    key_at_frame,
    runtime_control_key,
)
from .semantic_model import (
    AWBChannel,
    AWBFrameAggregate,
    AWBKey,
    key_type_for_channels,
)


@dataclass(frozen=True, slots=True)
class KeyContributor:
    """Read-only summary of one concrete channel key at one exact time."""

    channel: ResolvedChannel
    frame: float
    selected: bool
    interpolation: str


@dataclass(frozen=True, slots=True)
class SemanticKeyQuery:
    """Call-scoped semantic read result; never persistent animation storage."""

    context: ControlContext
    channels: tuple[ResolvedChannel, ...]
    contributors: tuple[KeyContributor, ...]
    keys: tuple[AWBKey, ...]


def query_keys(control_context: ControlContext) -> SemanticKeyQuery:
    """Read current assigned-slot channels into AWB semantic key summaries."""
    channels = channels_for_controls(control_context.controls)
    contributors: list[KeyContributor] = []
    per_control: dict[tuple[int, int], dict[float, dict[str, object]]] = {}

    for channel in channels:
        control_key = runtime_control_key(channel.resolved)
        frames = per_control.setdefault(control_key, {})
        for point in channel.fcurve.keyframe_points:
            frame = float(point.co.x)
            selected = bool(point.select_control_point)
            interpolation = str(point.interpolation)
            contributors.append(
                KeyContributor(
                    channel=channel,
                    frame=frame,
                    selected=selected,
                    interpolation=interpolation,
                )
            )
            state = frames.setdefault(
                frame,
                {
                    "channels": set(),
                    "selected": False,
                    "interpolations": set(),
                },
            )
            state["channels"].add(channel.family)
            state["selected"] = bool(state["selected"]) or selected
            state["interpolations"].add(interpolation)

    semantic_keys: list[tuple[AWBKey, tuple[int, int]]] = []
    for resolved in control_context.controls:
        control_key = runtime_control_key(resolved)
        for frame, state in per_control.get(control_key, {}).items():
            frozen_channels = frozenset(state["channels"])
            interpolations = state["interpolations"]
            interpolation = (
                next(iter(interpolations)) if len(interpolations) == 1 else "MIXED"
            )
            semantic_keys.append(
                (
                    AWBKey(
                        frame=frame,
                        control=resolved.control,
                        channels=frozen_channels,
                        key_type=key_type_for_channels(frozen_channels),
                        interpolation=interpolation,
                        selected=bool(state["selected"]),
                    ),
                    control_key,
                )
            )

    semantic_keys.sort(
        key=lambda item: (
            item[0].frame,
            item[0].control.control_id,
            item[1],
        )
    )
    return SemanticKeyQuery(
        context=control_context,
        channels=channels,
        contributors=tuple(contributors),
        keys=tuple(item[0] for item in semantic_keys),
    )


def semantic_keys_for_context(context) -> list[AWBKey]:
    """Compatibility read façade for feature modules migrating incrementally."""
    return list(query_keys(control_context_for_context(context)).keys)


def resolve_contributor_key(
    contributor: KeyContributor,
    *,
    epsilon: float = 1e-4,
):
    """Resolve one contributor to a current KeyframePoint for immediate use."""
    return key_at_frame(
        contributor.channel,
        contributor.frame,
        epsilon=epsilon,
    )


def frame_aggregates(query: SemanticKeyQuery) -> tuple[AWBFrameAggregate, ...]:
    """Union control-level semantic keys per exact frame without losing contributors."""
    by_frame: dict[float, list[AWBKey]] = {}
    for key in query.keys:
        by_frame.setdefault(key.frame, []).append(key)

    aggregates: list[AWBFrameAggregate] = []
    for frame, contributors in sorted(by_frame.items()):
        channels = frozenset(
            channel
            for contributor in contributors
            for channel in contributor.channels
        )
        aggregates.append(
            AWBFrameAggregate(
                frame=frame,
                contributors=tuple(contributors),
                channels=channels,
                selected=any(contributor.selected for contributor in contributors),
            )
        )
    return tuple(aggregates)


def contributors_for_frames(
    query: SemanticKeyQuery,
    frames: Iterable[float],
    *,
    control_keys: frozenset[tuple[int, int]] | None = None,
    families: frozenset[AWBChannel] | None = None,
    epsilon: float = 1e-4,
) -> tuple[KeyContributor, ...]:
    """Expand exact semantic frames back to concrete read-only channel contributors."""
    requested_frames = tuple(float(frame) for frame in frames)
    if not requested_frames:
        return ()
    if control_keys is not None and not control_keys:
        return ()
    if families is not None and not families:
        return ()

    tolerance = abs(float(epsilon))
    matched: list[KeyContributor] = []
    for contributor in query.contributors:
        if (
            control_keys is not None
            and runtime_control_key(contributor.channel.resolved) not in control_keys
        ):
            continue
        if families is not None and contributor.channel.family not in families:
            continue
        if not any(
            abs(contributor.frame - requested_frame) <= tolerance
            for requested_frame in requested_frames
        ):
            continue
        matched.append(contributor)
    return tuple(matched)
