from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

CanonicalValue = Any
CanonicalRecord = tuple[tuple[str, ...], tuple[CanonicalValue, ...]]


class DiagnosticSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    BLOCKER = "BLOCKER"


class OperationStage(StrEnum):
    RESOLVE = "RESOLVE"
    PREFLIGHT = "PREFLIGHT"
    SNAPSHOT = "SNAPSHOT"
    PLAN = "PLAN"
    APPLY_WRITE = "APPLY_WRITE"
    REEVALUATE = "REEVALUATE"
    VERIFY = "VERIFY"
    COMMIT = "COMMIT"
    ROLLBACK_WRITE = "ROLLBACK_WRITE"
    ROLLBACK_VERIFY = "ROLLBACK_VERIFY"


class FingerprintProfile(StrEnum):
    IDENTITY = "IDENTITY"
    RAW_ANIM = "RAW_ANIM"
    RAW_POSE = "RAW_POSE"
    GLOBAL_UI = "GLOBAL_UI"
    METADATA = "METADATA"
    EVAL_TOLERANT = "EVAL_TOLERANT"


class RollbackStatus(StrEnum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    NOT_REQUIRED = "NOT_REQUIRED"
    VERIFIED = "VERIFIED"
    INCOMPLETE = "INCOMPLETE"


class DiffKind(StrEnum):
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    CHANGED = "CHANGED"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    severity: DiagnosticSeverity
    stage: OperationStage | None = None
    operation: str | None = None
    detail: str = ""
    character_id: str | None = None
    record_id: str | None = None
    target_key: tuple[int, int] | None = None
    frame: float | None = None
    rollback_status: RollbackStatus | None = None


@dataclass(frozen=True, slots=True)
class FixtureManifest:
    fixture_id: str
    version: str = "1"
    description: str = ""
    expected_object_names: tuple[str, ...] = ()
    expected_character_ids: tuple[str, ...] = ()
    expected_control_keys: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def validate(self) -> tuple[Diagnostic, ...]:
        issues: list[Diagnostic] = []
        if not self.fixture_id or self.fixture_id != self.fixture_id.strip():
            issues.append(
                Diagnostic(
                    "FIXTURE_ID_INVALID",
                    DiagnosticSeverity.ERROR,
                    detail="fixture_id must be non-empty and trimmed.",
                )
            )
        for label, values in (
            ("expected_object_names", self.expected_object_names),
            ("expected_character_ids", self.expected_character_ids),
            ("expected_control_keys", self.expected_control_keys),
            ("tags", self.tags),
        ):
            if len(values) != len(set(values)):
                issues.append(
                    Diagnostic(
                        "FIXTURE_DUPLICATE_ENTRY",
                        DiagnosticSeverity.ERROR,
                        detail=f"{label} contains duplicate values.",
                    )
                )
            if any(not value or value != value.strip() for value in values):
                issues.append(
                    Diagnostic(
                        "FIXTURE_ENTRY_INVALID",
                        DiagnosticSeverity.ERROR,
                        detail=f"{label} contains an empty or untrimmed value.",
                    )
                )
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class FingerprintRequest:
    profiles: tuple[FingerprintProfile, ...]
    controls: tuple[Any, ...] = ()
    animation_owners: tuple[Any, ...] = ()
    character_ids: tuple[str, ...] = ()
    include_objects: tuple[Any, ...] = ()
    label: str = ""


@dataclass(frozen=True, slots=True)
class CanonicalSnapshot:
    label: str
    profiles: tuple[tuple[FingerprintProfile, tuple[CanonicalRecord, ...]], ...]

    def records(self, profile: FingerprintProfile) -> tuple[CanonicalRecord, ...]:
        return next((records for item, records in self.profiles if item == profile), ())

    def canonical(self) -> tuple:
        """Return state content only; label is diagnostic metadata."""

        return self.profiles


@dataclass(frozen=True, slots=True)
class DiffEntry:
    profile: FingerprintProfile
    key: tuple[str, ...]
    kind: DiffKind
    before: tuple[CanonicalValue, ...] | None
    after: tuple[CanonicalValue, ...] | None


@dataclass(frozen=True, slots=True)
class CanonicalDiff:
    entries: tuple[DiffEntry, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def entries_for(self, profile: FingerprintProfile) -> tuple[DiffEntry, ...]:
        return tuple(entry for entry in self.entries if entry.profile == profile)

    def unexpected_against(self, footprint: WriteFootprint) -> CanonicalDiff:
        return CanonicalDiff(
            tuple(
                entry
                for entry in self.entries
                if not footprint.matches(entry.profile, entry.key)
            )
        )


@dataclass(frozen=True, slots=True)
class WriteFootprint:
    allowed_prefixes: tuple[tuple[FingerprintProfile, tuple[str, ...]], ...] = ()
    require_match_all: bool = False

    def matches(self, profile: FingerprintProfile, key: tuple[str, ...]) -> bool:
        return any(
            allowed_profile == profile and key[: len(prefix)] == prefix
            for allowed_profile, prefix in self.allowed_prefixes
        )

    def all_matched(self, diff: CanonicalDiff) -> bool:
        if not self.require_match_all:
            return True
        return all(
            any(
                entry.profile == profile and entry.key[: len(prefix)] == prefix
                for entry in diff.entries
            )
            for profile, prefix in self.allowed_prefixes
        )


def _profile_order(profile: FingerprintProfile) -> int:
    return tuple(FingerprintProfile).index(profile)


def _profile_index(snapshot: CanonicalSnapshot, profile: FingerprintProfile) -> dict:
    result: dict[tuple[str, ...], tuple[CanonicalValue, ...]] = {}
    for key, payload in snapshot.records(profile):
        if key in result:
            raise ValueError(f"Duplicate canonical fingerprint key: {key!r}")
        result[key] = payload
    return result


def _profile_union(
    before: CanonicalSnapshot,
    after: CanonicalSnapshot,
) -> tuple[FingerprintProfile, ...]:
    profiles = {profile for profile, _records in before.profiles}
    profiles.update(profile for profile, _records in after.profiles)
    return tuple(sorted(profiles, key=_profile_order))


def diff_snapshots(before: CanonicalSnapshot, after: CanonicalSnapshot) -> CanonicalDiff:
    entries: list[DiffEntry] = []
    for profile in _profile_union(before, after):
        before_records = _profile_index(before, profile)
        after_records = _profile_index(after, profile)
        for key in sorted(set(before_records) | set(after_records)):
            if key not in before_records:
                entries.append(
                    DiffEntry(profile, key, DiffKind.ADDED, None, after_records[key])
                )
            elif key not in after_records:
                entries.append(
                    DiffEntry(profile, key, DiffKind.REMOVED, before_records[key], None)
                )
            elif before_records[key] != after_records[key]:
                entries.append(
                    DiffEntry(
                        profile,
                        key,
                        DiffKind.CHANGED,
                        before_records[key],
                        after_records[key],
                    )
                )
    return CanonicalDiff(tuple(entries))


def _tolerant_equal(left: CanonicalValue, right: CanonicalValue, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) <= tolerance
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _tolerant_equal(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    return left == right


def diff_snapshots_tolerant(
    before: CanonicalSnapshot,
    after: CanonicalSnapshot,
    tolerance: float,
    *,
    profiles: Iterable[FingerprintProfile] | None = None,
) -> CanonicalDiff:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    wanted = set(profiles) if profiles is not None else {FingerprintProfile.EVAL_TOLERANT}
    entries: list[DiffEntry] = []
    for profile in _profile_union(before, after):
        if profile not in wanted:
            continue
        before_records = _profile_index(before, profile)
        after_records = _profile_index(after, profile)
        for key in sorted(set(before_records) | set(after_records)):
            left = before_records.get(key)
            right = after_records.get(key)
            if left is None:
                entries.append(DiffEntry(profile, key, DiffKind.ADDED, None, right))
            elif right is None:
                entries.append(DiffEntry(profile, key, DiffKind.REMOVED, left, None))
            elif not _tolerant_equal(left, right, tolerance):
                entries.append(DiffEntry(profile, key, DiffKind.CHANGED, left, right))
    return CanonicalDiff(tuple(entries))


@dataclass(frozen=True, slots=True)
class StageEvent:
    stage: OperationStage
    ordinal: int = 1
    operation: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class FaultSpec:
    stage: OperationStage
    ordinal: int = 1
    error_type: type[BaseException] = RuntimeError
    message: str = "injected fault"


class StageHook:
    def enter(
        self,
        stage: OperationStage,
        *,
        ordinal: int = 1,
        operation: str = "",
        detail: str = "",
    ) -> None:
        raise NotImplementedError


class NoopStageHook(StageHook):
    def enter(
        self,
        stage: OperationStage,
        *,
        ordinal: int = 1,
        operation: str = "",
        detail: str = "",
    ) -> None:
        return None


class RecordingStageHook(StageHook):
    def __init__(self) -> None:
        self.events: list[StageEvent] = []

    def enter(
        self,
        stage: OperationStage,
        *,
        ordinal: int = 1,
        operation: str = "",
        detail: str = "",
    ) -> None:
        self.events.append(StageEvent(stage, ordinal, operation, detail))


class FaultInjectingHook(RecordingStageHook):
    def __init__(self, fault: FaultSpec) -> None:
        super().__init__()
        self.fault = fault
        self.fired = False

    def enter(
        self,
        stage: OperationStage,
        *,
        ordinal: int = 1,
        operation: str = "",
        detail: str = "",
    ) -> None:
        super().enter(stage, ordinal=ordinal, operation=operation, detail=detail)
        if self.fired or stage != self.fault.stage or ordinal != self.fault.ordinal:
            return
        self.fired = True
        raise self.fault.error_type(self.fault.message)


NOOP_STAGE_HOOK = NoopStageHook()


class RollbackIncompleteError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OperationOutcome:
    success: bool
    diagnostics: tuple[Diagnostic, ...] = ()
    rollback_status: RollbackStatus = RollbackStatus.NOT_ATTEMPTED


@dataclass(frozen=True, slots=True)
class TransactionResult:
    success: bool
    rollback_status: RollbackStatus
    diagnostics: tuple[Diagnostic, ...]
    stage_events: tuple[StageEvent, ...]
    before: CanonicalSnapshot | None = None
    after: CanonicalSnapshot | None = None
    diff: CanonicalDiff | None = None
    unexpected_diff: CanonicalDiff | None = None
    raised: BaseException | None = None

    def assert_zero_diff(self) -> None:
        if self.diff is None or not self.diff.is_empty:
            raise AssertionError(f"Expected zero diff, got {self.diff!r}")

    def assert_no_unexpected(self) -> None:
        if self.unexpected_diff is None or not self.unexpected_diff.is_empty:
            raise AssertionError(f"Unexpected diff: {self.unexpected_diff!r}")

    def assert_rollback_verified(self) -> None:
        if self.rollback_status != RollbackStatus.VERIFIED:
            raise AssertionError(
                f"Expected VERIFIED rollback, got {self.rollback_status.value}."
            )


class TransactionProbe:
    """Observation-only runner; rollback remains the production operation's job."""

    def __init__(
        self,
        scene: Any,
        snapshot_fn: Callable[[Any, FingerprintRequest], CanonicalSnapshot],
    ) -> None:
        self._scene = scene
        self._snapshot_fn = snapshot_fn

    def run(
        self,
        operation: Callable[[StageHook], OperationOutcome],
        *,
        fault: FaultSpec | None = None,
        before_request: FingerprintRequest | None = None,
        after_request: FingerprintRequest | None = None,
        expected_write_footprint: WriteFootprint | None = None,
        expect_zero_diff: bool = False,
        expect_success: bool | None = None,
    ) -> TransactionResult:
        hook: RecordingStageHook = (
            FaultInjectingHook(fault) if fault is not None else RecordingStageHook()
        )
        before = (
            self._snapshot_fn(self._scene, before_request)
            if before_request is not None
            else None
        )

        raised: BaseException | None = None
        try:
            outcome = operation(hook)
        except RollbackIncompleteError as exc:
            raised = exc
            outcome = OperationOutcome(
                success=False,
                diagnostics=(
                    Diagnostic(
                        "ROLLBACK_INCOMPLETE",
                        DiagnosticSeverity.BLOCKER,
                        stage=OperationStage.ROLLBACK_VERIFY,
                        detail=str(exc),
                        rollback_status=RollbackStatus.INCOMPLETE,
                    ),
                ),
                rollback_status=RollbackStatus.INCOMPLETE,
            )
        except BaseException as exc:  # noqa: BLE001 - verifier must capture operation failure
            raised = exc
            outcome = OperationOutcome(
                success=False,
                diagnostics=(
                    Diagnostic(
                        "OPERATION_RAISED",
                        DiagnosticSeverity.ERROR,
                        detail=f"{type(exc).__name__}: {exc}",
                    ),
                ),
            )

        after = (
            self._snapshot_fn(self._scene, after_request)
            if after_request is not None
            else None
        )
        diff = diff_snapshots(before, after) if before is not None and after is not None else None
        unexpected = (
            diff.unexpected_against(expected_write_footprint)
            if diff is not None and expected_write_footprint is not None
            else None
        )
        diagnostics = list(outcome.diagnostics)

        if expect_zero_diff and diff is not None and not diff.is_empty:
            diagnostics.append(
                Diagnostic(
                    "UNEXPECTED_DIFF",
                    DiagnosticSeverity.ERROR,
                    detail=f"Expected zero diff; observed {len(diff.entries)} entries.",
                )
            )
        if unexpected is not None and not unexpected.is_empty:
            diagnostics.append(
                Diagnostic(
                    "WRITE_FOOTPRINT_VIOLATION",
                    DiagnosticSeverity.ERROR,
                    detail=f"Observed {len(unexpected.entries)} undeclared diff entries.",
                )
            )
        if (
            expected_write_footprint is not None
            and diff is not None
            and not expected_write_footprint.all_matched(diff)
        ):
            diagnostics.append(
                Diagnostic(
                    "WRITE_FOOTPRINT_UNDERFILLED",
                    DiagnosticSeverity.ERROR,
                    detail="At least one required write prefix was not observed.",
                )
            )
        if expect_success is not None and outcome.success != expect_success:
            diagnostics.append(
                Diagnostic(
                    "EXPECTED_SUCCESS_MISMATCH",
                    DiagnosticSeverity.ERROR,
                    detail=f"Expected success={expect_success}, got {outcome.success}.",
                )
            )

        success = outcome.success and not any(
            diagnostic.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.BLOCKER}
            for diagnostic in diagnostics
        )
        return TransactionResult(
            success=success,
            rollback_status=outcome.rollback_status,
            diagnostics=tuple(diagnostics),
            stage_events=tuple(hook.events),
            before=before,
            after=after,
            diff=diff,
            unexpected_diff=unexpected,
            raised=raised,
        )
