from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

TRACE_SUBSYSTEMS = frozenset(
    {
        "ui",
        "input",
        "keymap",
        "operator",
        "modal",
        "timer",
        "handler",
        "depsgraph",
        "animation",
        "domain",
        "writer",
        "runtime",
    }
)

TRACE_LIFECYCLE_PHASES = frozenset(
    {
        "ingress",
        "routing",
        "poll",
        "invoke",
        "execute",
        "modal_tick",
        "handler_pre",
        "handler_post",
        "depsgraph_pre",
        "depsgraph_post",
        "commit",
        "cancel",
        "fail",
    }
)

TRACE_EVALUATION_PHASES = frozenset(
    {
        "pre_handler",
        "post_handler",
        "evaluated",
        "live_context",
        "unknown",
    }
)


class TraceTerminalStatus(StrEnum):
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class TraceRouteOutcome(StrEnum):
    CLAIMED = "CLAIMED"
    REJECTED = "REJECTED"
    NATIVE_FALLTHROUGH = "NATIVE_FALLTHROUGH"
    NOT_REACHED = "NOT_REACHED"


@dataclass(frozen=True, slots=True)
class TraceCausalContext:
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    span_origin: str = "EXPLICIT"


def _safe_prefix(prefix: str, fallback: str) -> str:
    value = "".join(
        ch if ch.isalnum() or ch in "-_" else "-"
        for ch in str(prefix)
    ).strip("-")
    return value or fallback


def new_trace_id(prefix: str = "trace") -> str:
    return f"{_safe_prefix(prefix, 'trace')}:{uuid4().hex}"


def new_trace_span_id(prefix: str = "span") -> str:
    return f"{_safe_prefix(prefix, 'span')}:{uuid4().hex}"


def new_trace_root(prefix: str = "trace") -> TraceCausalContext:
    return TraceCausalContext(
        trace_id=new_trace_id(prefix),
        span_id=new_trace_span_id(f"{prefix}-root"),
        parent_span_id=None,
        span_origin="EXPLICIT_ROOT",
    )


def new_trace_child(
    parent: TraceCausalContext,
    prefix: str = "span",
    *,
    origin: str = "EXPLICIT_CHILD",
) -> TraceCausalContext:
    return TraceCausalContext(
        trace_id=parent.trace_id,
        span_id=new_trace_span_id(prefix),
        parent_span_id=parent.span_id,
        span_origin=str(origin),
    )


class OperationCausalRegistry:
    """Epoch-scoped operation causal bindings plus parent/child lifetime.

    There is deliberately no implicit "current trace". Parent links exist only
    when the named parent is live in the current epoch. Retiring an operation
    also retires its descendants so writer/modal children cannot survive their
    owning gesture and later cross-link a new operation.
    """

    def __init__(self, *, max_bindings: int = 2048) -> None:
        self._max_bindings = max(16, int(max_bindings))
        self._epoch = uuid4().hex
        self._bindings: dict[tuple[str, str], TraceCausalContext] = {}
        self._parents: dict[tuple[str, str], str] = {}

    @property
    def epoch(self) -> str:
        return self._epoch

    def reset(self, *, epoch: str | None = None) -> str:
        self._bindings.clear()
        self._parents.clear()
        self._epoch = str(epoch or uuid4().hex)
        return self._epoch

    def _key(self, operation_id: str) -> tuple[str, str]:
        return (self._epoch, str(operation_id))

    def _descendant_ids(self, operation_id: str) -> set[str]:
        descendants = {str(operation_id)}
        pending = [str(operation_id)]
        while pending:
            parent_id = pending.pop()
            for (epoch, child_id), linked_parent in tuple(self._parents.items()):
                if (
                    epoch == self._epoch
                    and linked_parent == parent_id
                    and child_id not in descendants
                ):
                    descendants.add(child_id)
                    pending.append(child_id)
        return descendants

    def _enforce_bound(self) -> None:
        while len(self._bindings) > self._max_bindings:
            oldest_epoch, oldest_id = next(iter(self._bindings))
            if oldest_epoch != self._epoch:
                self._bindings.pop((oldest_epoch, oldest_id), None)
                self._parents.pop((oldest_epoch, oldest_id), None)
                continue
            self.retire(oldest_id)

    def bind(
        self,
        operation_id: str | None,
        causal: TraceCausalContext,
    ) -> TraceCausalContext | None:
        if not operation_id:
            return None
        key = self._key(str(operation_id))
        # Rebinding starts a new explicit lifetime for this operation id.
        self._bindings.pop(key, None)
        self._parents.pop(key, None)
        self._bindings[key] = causal
        self._enforce_bound()
        return causal

    def get(self, operation_id: str | None) -> TraceCausalContext | None:
        if not operation_id:
            return None
        return self._bindings.get(self._key(str(operation_id)))

    def parent(self, operation_id: str | None) -> str | None:
        if not operation_id:
            return None
        return self._parents.get(self._key(str(operation_id)))

    def link(
        self,
        child_operation_id: str | None,
        parent_operation_id: str | None,
        *,
        child_prefix: str = "operation",
    ) -> TraceCausalContext | None:
        if (
            not child_operation_id
            or not parent_operation_id
            or str(child_operation_id) == str(parent_operation_id)
        ):
            return None
        child_id = str(child_operation_id)
        parent_id = str(parent_operation_id)
        parent = self.get(parent_id)
        if parent is None:
            # Fail closed: never retain a parent relation without a live parent.
            return None

        existing = self.get(child_id)
        if existing is None:
            existing = new_trace_child(
                parent,
                child_prefix,
                origin="AUTO_OPERATION_LINK",
            )
            self.bind(child_id, existing)

        self._parents[self._key(child_id)] = parent_id
        self._enforce_bound()
        return existing

    def retire(self, operation_id: str | None) -> tuple[str, ...]:
        if not operation_id:
            return ()
        retired = self._descendant_ids(str(operation_id))
        for retired_id in retired:
            key = self._key(retired_id)
            self._bindings.pop(key, None)
            self._parents.pop(key, None)
        return tuple(sorted(retired))

    def forget(self, operation_id: str | None) -> None:
        self.retire(operation_id)

    def __len__(self) -> int:
        return len(self._bindings)


class LifecycleHandlerRecursionGuard:
    """Thread-local recursion guard for Blender lifecycle callbacks."""

    def __init__(self) -> None:
        self._local = threading.local()

    def enter(self) -> bool:
        if bool(getattr(self._local, "active", False)):
            return False
        self._local.active = True
        return True

    def exit(self) -> None:
        self._local.active = False

    @property
    def active(self) -> bool:
        return bool(getattr(self._local, "active", False))


def validate_trace_subsystem(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in TRACE_SUBSYSTEMS:
        raise ValueError(f"Unsupported trace subsystem: {value!r}")
    return normalized


def validate_trace_lifecycle_phase(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in TRACE_LIFECYCLE_PHASES:
        raise ValueError(f"Unsupported trace lifecycle phase: {value!r}")
    return normalized


def validate_trace_evaluation_phase(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in TRACE_EVALUATION_PHASES:
        raise ValueError(f"Unsupported trace evaluation phase: {value!r}")
    return normalized
