from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .character_metadata import resolve_character
from .rigped_contract import (
    RIGPED_SETUP_PROPERTY,
    RIGPED_SETUP_SCHEMA_VERSION,
    RigpedLifecycle,
    compute_setup_signature,
    read_setup_descriptor,
)
from .rigped_fit_policy import (
    FitCommitDecision,
    FitEntryDecision,
    FitEntryFacts,
    FitSessionToken,
    decide_fit_commit,
    evaluate_fit_entry,
    validate_fit_session_fresh,
)
from .semantic_adapter import assigned_channelbag, channel_binding_token


class RigpedFitRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FitRuntimeInspection:
    decision: FitEntryDecision
    session: FitSessionToken | None
    rigped_issue_codes: tuple[str, ...]
    runtime_issue_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FitCommitResult:
    decision: FitCommitDecision
    setup_revision: int
    setup_signature: str


def _mapping_value(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(key, default)
    getter = getattr(raw, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _descriptor_carrier(view):
    root_bindings = tuple(
        binding
        for binding in view.definition.bindings
        if binding.semantic_key == "awb.root" and binding.usage.value == "PRIMARY"
    )
    if len(root_bindings) != 1:
        raise RigpedFitRuntimeError("FIT_ROOT_BINDING_AMBIGUOUS")
    resolved = dict(view.resolved_bindings).get(root_bindings[0].binding_id)
    if resolved is None:
        raise RigpedFitRuntimeError("FIT_ROOT_BINDING_UNRESOLVED")
    return resolved.owner_object


def _unique_owner_objects(view) -> tuple[Any, ...]:
    owners: list[Any] = []
    seen: set[int] = set()
    for _binding_id, resolved in view.resolved_bindings:
        owner = resolved.owner_object
        pointer = int(owner.as_pointer())
        if pointer in seen:
            continue
        seen.add(pointer)
        owners.append(owner)
    return tuple(owners)


def _owner_writable(owner) -> bool:
    if getattr(owner, "library", None) is not None or getattr(owner, "is_editable", True) is False:
        return False
    data = getattr(owner, "data", None)
    return not (
        data is not None
        and (
            getattr(data, "library", None) is not None
            or getattr(data, "is_editable", True) is False
        )
    )


def _owner_tokens(view) -> tuple[tuple[int | None, int | None, int | None, int | None], ...]:
    return tuple(
        sorted(
            (channel_binding_token(owner) for owner in _unique_owner_objects(view)),
            key=repr,
        )
    )


def _animation_state(view) -> tuple[bool, tuple[str, ...]]:
    """Return current assigned-animation presence and ownership issues."""

    issues: list[str] = []
    has_animation = False
    for owner in _unique_owner_objects(view):
        animation_data = getattr(owner, "animation_data", None)
        action = getattr(animation_data, "action", None) if animation_data is not None else None
        if action is None:
            continue
        bag = assigned_channelbag(owner)
        if bag is None:
            issues.append("FIT_ANIMATION_BINDING_AMBIGUOUS")
            continue
        if any(tuple(fcurve.keyframe_points) for fcurve in bag.fcurves):
            has_animation = True
    return has_animation, tuple(dict.fromkeys(issues))


def _animation_signature(view) -> tuple:
    """Freeze the raw assigned F-Curve state for one Fit session.

    Fit may structurally change the Rigped rest pose, but it must never silently
    edit, add or remove existing animation keys. The signature is intentionally
    raw-FCurve based, so rest-pose evaluation changes do not look like key edits.
    """

    rows: list[tuple] = []
    for owner in _unique_owner_objects(view):
        binding_token = channel_binding_token(owner)
        animation_data = getattr(owner, "animation_data", None)
        action = getattr(animation_data, "action", None) if animation_data is not None else None
        if action is None:
            rows.append((binding_token, None))
            continue
        bag = assigned_channelbag(owner)
        if bag is None:
            rows.append((binding_token, "AMBIGUOUS"))
            continue
        curves = []
        for fcurve in bag.fcurves:
            points = tuple(
                (
                    round(float(point.co[0]), 9),
                    round(float(point.co[1]), 9),
                    str(point.interpolation),
                    str(point.handle_left_type),
                    str(point.handle_right_type),
                    tuple(round(float(value), 9) for value in point.handle_left),
                    tuple(round(float(value), 9) for value in point.handle_right),
                )
                for point in fcurve.keyframe_points
            )
            curves.append(
                (
                    str(fcurve.data_path),
                    int(fcurve.array_index),
                    str(fcurve.extrapolation),
                    points,
                )
            )
        rows.append((binding_token, tuple(sorted(curves, key=repr))))
    return tuple(sorted(rows, key=repr))


def inspect_fit_entry(scene, character_id: str) -> FitRuntimeInspection:
    view = resolve_character(scene, character_id)
    rigped_issues = tuple(issue.code for issue in view.issues)
    descriptor, descriptor_issues = read_setup_descriptor(view)
    rigped_issues += tuple(issue.code for issue in descriptor_issues)

    owners = _unique_owner_objects(view)
    writable = bool(owners) and all(_owner_writable(owner) for owner in owners)
    has_animation, runtime_issues = _animation_state(view)
    lifecycle = descriptor.lifecycle if descriptor is not None else None
    facts = FitEntryFacts(
        descriptor_valid=descriptor is not None and not descriptor_issues,
        lifecycle=lifecycle,
        has_animation=has_animation,
        writable=writable,
        character_coherent=not view.issues,
    )
    decision = evaluate_fit_entry(facts)
    combined_runtime = runtime_issues
    if runtime_issues and decision.allowed:
        decision = FitEntryDecision(False, decision.issue_codes + runtime_issues)

    if not decision.allowed or descriptor is None:
        return FitRuntimeInspection(decision, None, rigped_issues, combined_runtime)

    session = FitSessionToken(
        character_id=view.definition.character_id,
        profile_id=descriptor.profile_id,
        setup_revision=descriptor.revision,
        setup_signature=descriptor.signature,
        source_stamp=view.source_stamp,
        owner_binding_tokens=_owner_tokens(view),
        animation_signature=_animation_signature(view),
    )
    return FitRuntimeInspection(decision, session, rigped_issues, combined_runtime)


def validate_fit_runtime_session(scene, session: FitSessionToken) -> tuple[str, ...]:
    """Fail-closed freshness checks shared by semantic preview and final commit."""

    view = resolve_character(scene, session.character_id)
    issues: list[str] = []
    if view.issues:
        issues.append("FIT_SESSION_CHARACTER_INCOHERENT")
        return tuple(issues)

    try:
        _carrier, raw = _raw_descriptor(view)
    except RigpedFitRuntimeError as exc:
        return (str(exc),)

    if raw["schema_version"] != RIGPED_SETUP_SCHEMA_VERSION:
        issues.append("FIT_SESSION_SCHEMA_CHANGED")
    if raw["profile_id"] != session.profile_id:
        issues.append("FIT_SESSION_PROFILE_CHANGED")
    if raw["lifecycle"] != RigpedLifecycle.FITTED_UNBOUND.value:
        issues.append("FIT_SESSION_LIFECYCLE_CHANGED")

    descriptor, descriptor_issues = read_setup_descriptor(view)
    issues.extend(issue.code for issue in descriptor_issues)
    if descriptor is None:
        issues.append("FIT_SESSION_SETUP_SIGNATURE_INVALID")
    elif descriptor.signature != session.setup_signature:
        issues.append("FIT_SESSION_DESCRIPTOR_CHANGED")

    issues.extend(
        validate_fit_session_fresh(
            session,
            character_id=raw["character_id"],
            setup_revision=raw["revision"],
            stored_signature=raw["signature"],
            source_stamp=view.source_stamp,
            owner_binding_tokens=_owner_tokens(view),
        )
    )

    _has_animation, animation_issues = _animation_state(view)
    issues.extend(animation_issues)
    if _animation_signature(view) != session.animation_signature:
        issues.append("FIT_ANIMATION_CHANGED_DURING_SESSION")
    return tuple(dict.fromkeys(issues))


def _raw_descriptor(view) -> tuple[Any, dict[str, Any]]:
    carrier = _descriptor_carrier(view)
    getter = getattr(carrier, "get", None)
    raw = getter(RIGPED_SETUP_PROPERTY) if callable(getter) else None
    if raw is None:
        raise RigpedFitRuntimeError("FIT_DESCRIPTOR_MISSING_DURING_SESSION")
    try:
        payload = {
            "schema_version": int(_mapping_value(raw, "schema_version", 0) or 0),
            "character_id": str(_mapping_value(raw, "character_id", "") or ""),
            "profile_id": str(_mapping_value(raw, "profile_id", "") or ""),
            "revision": int(_mapping_value(raw, "revision", 0) or 0),
            "signature": str(_mapping_value(raw, "signature", "") or ""),
            "lifecycle": str(_mapping_value(raw, "lifecycle", "") or ""),
        }
    except (TypeError, ValueError) as exc:
        raise RigpedFitRuntimeError("FIT_DESCRIPTOR_INVALID_DURING_SESSION") from exc
    return carrier, payload


def commit_fit_session(
    scene,
    session: FitSessionToken,
    *,
    refresh_owned_derived: Callable[[Any], None] | None = None,
) -> FitCommitResult:
    """Publish a successful Fit revision after structural edits are already applied.

    This function intentionally does not own the native rest-structure snapshot.
    The future I7 modal Fit operation must restore that snapshot on cancel or on
    any exception raised here. Descriptor publication happens only after the
    optional generated-derived refresh succeeds.
    """

    view = resolve_character(scene, session.character_id)
    if view.issues:
        raise RigpedFitRuntimeError("FIT_SESSION_CHARACTER_INCOHERENT")
    descriptor, descriptor_issues = read_setup_descriptor(view)
    if descriptor is None or descriptor_issues:
        codes = tuple(issue.code for issue in descriptor_issues)
        raise RigpedFitRuntimeError(
            "FIT_SESSION_SETUP_SIGNATURE_INVALID"
            if not codes
            else ",".join(codes)
        )
    if (
        descriptor.revision != session.setup_revision
        or descriptor.signature != session.setup_signature
    ):
        raise RigpedFitRuntimeError("FIT_SESSION_DESCRIPTOR_CHANGED")
    carrier, raw = _raw_descriptor(view)
    if raw["schema_version"] != RIGPED_SETUP_SCHEMA_VERSION:
        raise RigpedFitRuntimeError("FIT_SESSION_SCHEMA_CHANGED")
    if raw["profile_id"] != session.profile_id:
        raise RigpedFitRuntimeError("FIT_SESSION_PROFILE_CHANGED")
    if raw["lifecycle"] != RigpedLifecycle.FITTED_UNBOUND.value:
        raise RigpedFitRuntimeError("FIT_SESSION_LIFECYCLE_CHANGED")

    freshness = validate_fit_session_fresh(
        session,
        character_id=raw["character_id"],
        setup_revision=raw["revision"],
        stored_signature=raw["signature"],
        source_stamp=view.source_stamp,
        owner_binding_tokens=_owner_tokens(view),
    )
    if freshness:
        raise RigpedFitRuntimeError(",".join(freshness))

    _has_animation, animation_issues = _animation_state(view)
    if animation_issues:
        raise RigpedFitRuntimeError(",".join(animation_issues))
    if _animation_signature(view) != session.animation_signature:
        raise RigpedFitRuntimeError("FIT_ANIMATION_CHANGED_DURING_SESSION")

    candidate_signature = compute_setup_signature(view)
    decision = decide_fit_commit(
        previous_revision=session.setup_revision,
        previous_signature=session.setup_signature,
        candidate_signature=candidate_signature,
    )
    if not decision.structural_change:
        return FitCommitResult(decision, session.setup_revision, session.setup_signature)

    if refresh_owned_derived is not None:
        refresh_owned_derived(view)
        view = resolve_character(scene, session.character_id)
        if view.issues:
            raise RigpedFitRuntimeError("FIT_DERIVED_REFRESH_INVALIDATED_CHARACTER")

    final_signature = compute_setup_signature(view)
    carrier[RIGPED_SETUP_PROPERTY] = {
        "schema_version": RIGPED_SETUP_SCHEMA_VERSION,
        "character_id": session.character_id,
        "profile_id": session.profile_id,
        "revision": decision.next_revision,
        "signature": final_signature,
        "lifecycle": RigpedLifecycle.FITTED_UNBOUND.value,
    }
    descriptor, issues = read_setup_descriptor(resolve_character(scene, session.character_id))
    if descriptor is None or issues:
        raise RigpedFitRuntimeError(f"FIT_DESCRIPTOR_PUBLISH_FAILED: {issues!r}")
    if descriptor.revision != decision.next_revision or descriptor.signature != final_signature:
        raise RigpedFitRuntimeError("FIT_DESCRIPTOR_PUBLISH_MISMATCH")
    return FitCommitResult(decision, descriptor.revision, descriptor.signature)
