from __future__ import annotations

from dataclasses import dataclass

from .rigped_contract import RigpedLifecycle


@dataclass(frozen=True, slots=True)
class FitEntryFacts:
    descriptor_valid: bool
    lifecycle: RigpedLifecycle | None
    has_animation: bool
    writable: bool
    character_coherent: bool


@dataclass(frozen=True, slots=True)
class FitEntryDecision:
    allowed: bool
    issue_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FitSessionToken:
    character_id: str
    profile_id: str
    setup_revision: int
    setup_signature: str
    source_stamp: tuple
    owner_binding_tokens: tuple[tuple[int | None, int | None, int | None, int | None], ...]
    animation_signature: tuple


@dataclass(frozen=True, slots=True)
class FitCommitDecision:
    structural_change: bool
    previous_revision: int
    next_revision: int
    previous_signature: str
    candidate_signature: str


def evaluate_fit_entry(facts: FitEntryFacts) -> FitEntryDecision:
    issues: list[str] = []
    if not facts.character_coherent:
        issues.append("FIT_CHARACTER_INCOHERENT")
    if not facts.descriptor_valid:
        issues.append("FIT_DESCRIPTOR_INVALID")
    if not facts.writable:
        issues.append("FIT_TARGET_READ_ONLY")
    if facts.lifecycle == RigpedLifecycle.BOUND:
        issues.append("FIT_AFTER_BIND_BLOCKED")
    elif facts.lifecycle is not None and facts.lifecycle != RigpedLifecycle.FITTED_UNBOUND:
        issues.append("FIT_UNSUPPORTED_LIFECYCLE")
    # Existing animation is allowed. Fit is a structural rest-edit workflow;
    # runtime entry/commit guards freeze the assigned animation signature for
    # the duration of the Fit session so keys cannot change underneath it.
    return FitEntryDecision(not issues, tuple(issues))


def decide_fit_commit(
    *,
    previous_revision: int,
    previous_signature: str,
    candidate_signature: str,
) -> FitCommitDecision:
    if previous_revision < 1:
        raise ValueError("Fit commit requires setup revision >= 1.")
    if not previous_signature or not candidate_signature:
        raise ValueError("Fit commit requires non-empty setup signatures.")
    changed = previous_signature != candidate_signature
    return FitCommitDecision(
        structural_change=changed,
        previous_revision=previous_revision,
        next_revision=(previous_revision + 1 if changed else previous_revision),
        previous_signature=previous_signature,
        candidate_signature=candidate_signature,
    )


def validate_fit_session_fresh(
    session: FitSessionToken,
    *,
    character_id: str,
    setup_revision: int,
    stored_signature: str,
    source_stamp: tuple,
    owner_binding_tokens: tuple[tuple[int | None, int | None, int | None, int | None], ...],
) -> tuple[str, ...]:
    issues: list[str] = []
    if character_id != session.character_id:
        issues.append("FIT_SESSION_CHARACTER_CHANGED")
    if setup_revision != session.setup_revision:
        issues.append("FIT_SESSION_REVISION_CHANGED")
    if stored_signature != session.setup_signature:
        issues.append("FIT_SESSION_DESCRIPTOR_CHANGED")
    if source_stamp != session.source_stamp:
        issues.append("FIT_SESSION_CHARACTER_GRAPH_CHANGED")
    if owner_binding_tokens != session.owner_binding_tokens:
        issues.append("FIT_SESSION_ANIMATION_BINDING_CHANGED")
    return tuple(issues)
