from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .phase4_operation_plan import OwnerBindingToken
from .phase4_verification import (
    Diagnostic,
    DiagnosticSeverity,
    OperationStage,
    RollbackIncompleteError,
    RollbackStatus,
)


class JournalState(StrEnum):
    OPEN = "OPEN"
    COMMITTED = "COMMITTED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_INCOMPLETE = "ROLLBACK_INCOMPLETE"


class RawFieldKind(StrEnum):
    OBJECT_LOCATION = "OBJECT_LOCATION"
    OBJECT_ROTATION_EULER = "OBJECT_ROTATION_EULER"
    OBJECT_ROTATION_QUATERNION = "OBJECT_ROTATION_QUATERNION"
    OBJECT_ROTATION_AXIS_ANGLE = "OBJECT_ROTATION_AXIS_ANGLE"
    OBJECT_SCALE = "OBJECT_SCALE"
    POSE_BONE_LOCATION = "POSE_BONE_LOCATION"
    POSE_BONE_ROTATION_EULER = "POSE_BONE_ROTATION_EULER"
    POSE_BONE_ROTATION_QUATERNION = "POSE_BONE_ROTATION_QUATERNION"
    POSE_BONE_ROTATION_AXIS_ANGLE = "POSE_BONE_ROTATION_AXIS_ANGLE"
    POSE_BONE_SCALE = "POSE_BONE_SCALE"
    CONSTRAINT_INFLUENCE = "CONSTRAINT_INFLUENCE"
    CONSTRAINT_MUTE = "CONSTRAINT_MUTE"
    CONSTRAINT_POLE_ANGLE = "CONSTRAINT_POLE_ANGLE"


@dataclass(frozen=True, slots=True, order=True)
class QuarantineKey:
    character_id: str
    setup_revision: int
    setup_signature: str
    owner_ptr: int | None = None
    bag_ptr: int | None = None


@dataclass(frozen=True, slots=True)
class KeySnapshot:
    exists: bool
    co: tuple[float, float] | None = None
    handle_left: tuple[float, float] | None = None
    handle_right: tuple[float, float] | None = None
    handle_left_type: str | None = None
    handle_right_type: str | None = None
    interpolation: str | None = None
    easing: str | None = None
    back: float | None = None
    amplitude: float | None = None
    period: float | None = None
    keyframe_type: str | None = None
    select_control_point: bool | None = None
    select_left_handle: bool | None = None
    select_right_handle: bool | None = None

    def validate(self) -> None:
        if self.exists and self.co is None:
            raise ValueError("Existing key snapshot requires raw co.x/co.y.")
        if not self.exists and any(
            value is not None
            for value in (
                self.co,
                self.handle_left,
                self.handle_right,
                self.handle_left_type,
                self.handle_right_type,
                self.interpolation,
                self.easing,
                self.back,
                self.amplitude,
                self.period,
                self.keyframe_type,
                self.select_control_point,
                self.select_left_handle,
                self.select_right_handle,
            )
        ):
            raise ValueError("Absent key snapshot must not carry raw key fields.")


@dataclass(frozen=True, slots=True)
class FCurveSnapshot:
    exists: bool
    data_path: str | None = None
    array_index: int | None = None
    group_name: str | None = None
    extrapolation: str | None = None
    keys: tuple[KeySnapshot, ...] = ()

    def validate(self) -> None:
        if self.exists and (self.data_path is None or self.array_index is None):
            raise ValueError("Existing FCurve snapshot requires data_path and array_index.")
        if self.exists:
            for key in self.keys:
                key.validate()
                if not key.exists:
                    raise ValueError("FCurveSnapshot keys must describe existing keys.")
        if not self.exists and (
            self.keys
            or any(
                value is not None
                for value in (self.data_path, self.array_index, self.group_name, self.extrapolation)
            )
        ):
            raise ValueError("Absent FCurve snapshot must not carry raw FCurve fields.")


type RawFieldValue = float | bool | tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RawFieldSnapshot:
    field: RawFieldKind
    value: RawFieldValue


@dataclass(frozen=True, slots=True)
class ReceiptBase:
    ordinal: int
    scope: QuarantineKey


@dataclass(frozen=True, slots=True)
class CreatedAnimationDataReceipt(ReceiptBase):
    owner_ptr: int
    owner_binding_before: OwnerBindingToken
    owner_binding_after: OwnerBindingToken


@dataclass(frozen=True, slots=True)
class CreatedActionReceipt(ReceiptBase):
    owner_ptr: int
    action_ptr: int
    owner_binding_before: OwnerBindingToken
    owner_binding_after: OwnerBindingToken


@dataclass(frozen=True, slots=True)
class CreatedSlotReceipt(ReceiptBase):
    owner_ptr: int
    action_ptr: int
    slot_ptr: int
    owner_binding_before: OwnerBindingToken
    owner_binding_after: OwnerBindingToken


@dataclass(frozen=True, slots=True)
class CreatedChannelBagReceipt(ReceiptBase):
    owner_ptr: int
    action_ptr: int
    slot_ptr: int
    bag_ptr: int
    owner_binding_before: OwnerBindingToken
    owner_binding_after: OwnerBindingToken


@dataclass(frozen=True, slots=True)
class CreatedFCurveReceipt(ReceiptBase):
    bag_ptr: int
    fcurve_ptr: int
    data_path: str
    array_index: int


@dataclass(frozen=True, slots=True)
class KeyMutationReceipt(ReceiptBase):
    bag_ptr: int
    fcurve_ptr: int
    frame: float
    before: KeySnapshot


@dataclass(frozen=True, slots=True)
class FCurveMutationReceipt(ReceiptBase):
    bag_ptr: int
    fcurve_ptr: int
    before: FCurveSnapshot


@dataclass(frozen=True, slots=True)
class RawFieldMutationReceipt(ReceiptBase):
    owner_ptr: int
    target_ptr: int
    before: RawFieldSnapshot


@dataclass(frozen=True, slots=True)
class IDPropertyMutationReceipt(ReceiptBase):
    owner_ptr: int
    target_ptr: int
    property_name: str
    before_exists: bool
    before_value: int | float | bool | str | None


@dataclass(frozen=True, slots=True)
class ConstraintMutationReceipt(ReceiptBase):
    owner_ptr: int
    constraint_ptr: int
    before: RawFieldSnapshot


type Receipt = (
    CreatedAnimationDataReceipt
    | CreatedActionReceipt
    | CreatedSlotReceipt
    | CreatedChannelBagReceipt
    | CreatedFCurveReceipt
    | KeyMutationReceipt
    | FCurveMutationReceipt
    | RawFieldMutationReceipt
    | IDPropertyMutationReceipt
    | ConstraintMutationReceipt
)


@dataclass(frozen=True, slots=True)
class RollbackReport:
    status: RollbackStatus
    diagnostics: tuple[Diagnostic, ...]
    residue_receipts: tuple[Receipt, ...]
    quarantine_keys: tuple[QuarantineKey, ...]


class RollbackExecutor(Protocol):
    def rollback_receipt(self, receipt: Receipt) -> None: ...

    def verify_receipt_rolled_back(self, receipt: Receipt) -> tuple[Diagnostic, ...]: ...


_SESSION_QUARANTINE: set[QuarantineKey] = set()


def add_quarantine(keys: tuple[QuarantineKey, ...] | set[QuarantineKey]) -> None:
    _SESSION_QUARANTINE.update(keys)


def is_quarantined(key: QuarantineKey) -> bool:
    return key in _SESSION_QUARANTINE


def clear_quarantine_explicit_recovery(key: QuarantineKey) -> None:
    _SESSION_QUARANTINE.discard(key)


def clear_session_quarantine_for_file_lifecycle() -> None:
    """Clear runtime-only quarantine after an explicit file/reload lifecycle boundary."""
    _SESSION_QUARANTINE.clear()


def _rollback_diagnostic(
    code: str,
    detail: str,
    *,
    operation_id: str,
    character_id: str,
    stage: OperationStage,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=DiagnosticSeverity.BLOCKER,
        stage=stage,
        operation=operation_id,
        detail=detail,
        character_id=character_id,
        rollback_status=RollbackStatus.INCOMPLETE,
    )


class MutationJournal:
    def __init__(
        self,
        operation_id: str,
        character_id: str,
        setup_revision: int,
        setup_signature: str,
        executor: RollbackExecutor,
    ) -> None:
        if not operation_id or not character_id or setup_revision < 1 or not setup_signature:
            raise ValueError("MutationJournal requires a valid operation/Character/setup identity.")
        self.operation_id = operation_id
        self.character_id = character_id
        self.setup_revision = setup_revision
        self.setup_signature = setup_signature
        self._executor = executor
        self._state = JournalState.OPEN
        self._receipts: list[Receipt] = []

    @property
    def state(self) -> JournalState:
        return self._state

    @property
    def receipts(self) -> tuple[Receipt, ...]:
        return tuple(self._receipts)

    def next_ordinal(self) -> int:
        return len(self._receipts) + 1

    def record(self, receipt: Receipt) -> None:
        if self._state is not JournalState.OPEN:
            raise RuntimeError(f"Mutation journal is not open: {self._state.value}")
        if receipt.ordinal != self.next_ordinal():
            raise ValueError(
                f"Receipt ordinal {receipt.ordinal} does not match expected {self.next_ordinal()}."
            )
        if receipt.scope.character_id != self.character_id:
            raise ValueError("Receipt quarantine scope belongs to a different Character.")
        if receipt.scope.setup_revision != self.setup_revision:
            raise ValueError("Receipt quarantine scope has a different setup revision.")
        if receipt.scope.setup_signature != self.setup_signature:
            raise ValueError("Receipt quarantine scope has a different setup signature.")
        before = getattr(receipt, "before", None)
        validator = getattr(before, "validate", None)
        if callable(validator):
            validator()
        self._receipts.append(receipt)

    def commit(self) -> None:
        if self._state is not JournalState.OPEN:
            raise RuntimeError(f"Mutation journal cannot commit from {self._state.value}.")
        self._state = JournalState.COMMITTED

    def rollback(self) -> RollbackReport:
        if self._state is not JournalState.OPEN:
            raise RuntimeError(
                f"Mutation journal rollback is only valid before commit; state={self._state.value}."
            )
        self._state = JournalState.ROLLING_BACK
        rolled_back: list[Receipt] = []
        residue: list[Receipt] = []
        diagnostics: list[Diagnostic] = []
        quarantine: set[QuarantineKey] = set()

        for receipt in reversed(self._receipts):
            try:
                self._executor.rollback_receipt(receipt)
                rolled_back.append(receipt)
            except Exception as exc:  # noqa: BLE001 - rollback must continue best-effort
                residue.append(receipt)
                quarantine.add(receipt.scope)
                diagnostics.append(
                    _rollback_diagnostic(
                        "I4_ROLLBACK_RECEIPT_FAILED",
                        f"ordinal={receipt.ordinal} receipt={type(receipt).__name__}: {type(exc).__name__}: {exc}",
                        operation_id=self.operation_id,
                        character_id=self.character_id,
                        stage=OperationStage.ROLLBACK_WRITE,
                    )
                )

        for receipt in reversed(rolled_back):
            try:
                verification = self._executor.verify_receipt_rolled_back(receipt)
            except Exception as exc:  # noqa: BLE001 - verification failure is hard residue
                verification = (
                    _rollback_diagnostic(
                        "I4_ROLLBACK_VERIFY_EXCEPTION",
                        f"ordinal={receipt.ordinal} receipt={type(receipt).__name__}: {type(exc).__name__}: {exc}",
                        operation_id=self.operation_id,
                        character_id=self.character_id,
                        stage=OperationStage.ROLLBACK_VERIFY,
                    ),
                )
            if verification:
                residue.append(receipt)
                quarantine.add(receipt.scope)
                diagnostics.extend(verification)

        if residue:
            self._state = JournalState.ROLLBACK_INCOMPLETE
            ordered_quarantine = tuple(sorted(quarantine))
            add_quarantine(set(ordered_quarantine))
            return RollbackReport(
                RollbackStatus.INCOMPLETE,
                tuple(diagnostics),
                tuple(residue),
                ordered_quarantine,
            )

        self._state = JournalState.ROLLED_BACK
        return RollbackReport(RollbackStatus.VERIFIED, tuple(diagnostics), (), ())

    def rollback_or_raise(self) -> RollbackReport:
        report = self.rollback()
        if report.status is RollbackStatus.INCOMPLETE:
            raise RollbackIncompleteError(
                f"I4 rollback incomplete for {self.operation_id}: "
                f"{len(report.residue_receipts)} residue receipt(s)."
            )
        return report
