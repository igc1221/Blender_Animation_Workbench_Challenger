import sys
from dataclasses import FrozenInstanceError
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_phase4_mutation_journal_tests"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package


def _load(name: str):
    qualified = f"{PACKAGE_NAME}.{name}"
    spec = spec_from_file_location(qualified, PACKAGE_PATH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


verification = _load("phase4_verification")
operation_plan = _load("phase4_operation_plan")
journal_model = _load("phase4_mutation_journal")


class FakeExecutor:
    def __init__(self):
        self.rollback_order = []
        self.fail_rollback = set()
        self.verify_fail = set()

    def rollback_receipt(self, receipt):
        self.rollback_order.append(receipt.ordinal)
        if receipt.ordinal in self.fail_rollback:
            raise RuntimeError(f"fault {receipt.ordinal}")

    def verify_receipt_rolled_back(self, receipt):
        if receipt.ordinal not in self.verify_fail:
            return ()
        return (
            verification.Diagnostic(
                code="I4_TEST_VERIFY_MISMATCH",
                severity=verification.DiagnosticSeverity.BLOCKER,
                stage=verification.OperationStage.ROLLBACK_VERIFY,
                detail=f"verify mismatch {receipt.ordinal}",
                rollback_status=verification.RollbackStatus.INCOMPLETE,
            ),
        )


def _scope(owner_ptr=None, bag_ptr=None):
    return journal_model.QuarantineKey("char", 1, "sig", owner_ptr, bag_ptr)


def _journal(executor=None):
    return journal_model.MutationJournal("op", "char", 1, "sig", executor or FakeExecutor())


def _created_key(ordinal: int, frame: float):
    return journal_model.KeyMutationReceipt(
        ordinal,
        _scope(1, 4),
        4,
        5,
        frame,
        journal_model.KeySnapshot(False),
    )


def test_key_snapshot_requires_raw_state_for_existing_key():
    snapshot = journal_model.KeySnapshot(True)
    with pytest.raises(ValueError, match="co.x/co.y"):
        snapshot.validate()


def test_absent_key_snapshot_rejects_stale_raw_fields():
    snapshot = journal_model.KeySnapshot(False, co=(1.0, 2.0))
    with pytest.raises(ValueError, match="must not carry"):
        snapshot.validate()


def test_existing_key_snapshot_preserves_full_raw_contract():
    snapshot = journal_model.KeySnapshot(
        True,
        co=(1.25, 2.5),
        handle_left=(0.75, 2.0),
        handle_right=(1.75, 3.0),
        handle_left_type="FREE",
        handle_right_type="ALIGNED",
        interpolation="BEZIER",
        easing="EASE_IN_OUT",
        back=1.0,
        amplitude=2.0,
        period=3.0,
        keyframe_type="BREAKDOWN",
    )
    snapshot.validate()
    assert snapshot.co == (1.25, 2.5)
    assert snapshot.handle_left == (0.75, 2.0)
    assert snapshot.interpolation == "BEZIER"
    assert snapshot.keyframe_type == "BREAKDOWN"


def test_rotation_raw_field_contract_covers_all_blender_representations():
    values = {field.value for field in journal_model.RawFieldKind}
    assert "OBJECT_ROTATION_EULER" in values
    assert "OBJECT_ROTATION_QUATERNION" in values
    assert "OBJECT_ROTATION_AXIS_ANGLE" in values
    assert "POSE_BONE_ROTATION_EULER" in values
    assert "POSE_BONE_ROTATION_QUATERNION" in values
    assert "POSE_BONE_ROTATION_AXIS_ANGLE" in values


def test_receipts_are_immutable():
    receipt = _created_key(1, 12.0)
    with pytest.raises(FrozenInstanceError):
        receipt.frame = 13.0


def test_record_requires_exact_scope_and_ordinal():
    journal = _journal()
    with pytest.raises(ValueError, match="ordinal"):
        journal.record(_created_key(2, 12.0))

    wrong_scope = journal_model.KeyMutationReceipt(
        1,
        journal_model.QuarantineKey("other", 1, "sig"),
        4,
        5,
        12.0,
        journal_model.KeySnapshot(False),
    )
    with pytest.raises(ValueError, match="different Character"):
        journal.record(wrong_scope)


def test_rollback_is_reverse_order_and_verified():
    executor = FakeExecutor()
    journal = _journal(executor)
    journal.record(_created_key(1, 1.0))
    journal.record(_created_key(2, 2.0))
    report = journal.rollback()
    assert executor.rollback_order == [2, 1]
    assert report.status is verification.RollbackStatus.VERIFIED
    assert journal.state is journal_model.JournalState.ROLLED_BACK


def test_commit_is_terminal_and_cannot_be_rolled_back():
    journal = _journal()
    journal.commit()
    assert journal.state is journal_model.JournalState.COMMITTED
    with pytest.raises(RuntimeError, match="only valid before commit"):
        journal.rollback()
    with pytest.raises(RuntimeError, match="not open"):
        journal.record(_created_key(1, 1.0))


def test_commit_group_commits_all_after_full_prevalidation():
    first = _journal()
    second = _journal()

    journal_model.MutationJournal.commit_group((first, second))

    assert first.state is journal_model.JournalState.COMMITTED
    assert second.state is journal_model.JournalState.COMMITTED


def test_commit_group_rejects_non_open_member_before_any_new_commit():
    first = _journal()
    second = _journal()
    second.commit()

    with pytest.raises(RuntimeError, match="group cannot commit"):
        journal_model.MutationJournal.commit_group((first, second))

    assert first.state is journal_model.JournalState.OPEN
    assert second.state is journal_model.JournalState.COMMITTED


def test_commit_group_rejects_duplicate_without_crossing_commit_boundary():
    journal = _journal()

    with pytest.raises(ValueError, match="duplicate journal"):
        journal_model.MutationJournal.commit_group((journal, journal))

    assert journal.state is journal_model.JournalState.OPEN


def test_one_rollback_failure_does_not_stop_other_restores():
    executor = FakeExecutor()
    executor.fail_rollback = {2}
    journal = _journal(executor)
    journal.record(_created_key(1, 1.0))
    journal.record(_created_key(2, 2.0))
    journal.record(_created_key(3, 3.0))
    report = journal.rollback()
    assert executor.rollback_order == [3, 2, 1]
    assert report.status is verification.RollbackStatus.INCOMPLETE
    assert [receipt.ordinal for receipt in report.residue_receipts] == [2]
    assert any(diagnostic.code == "I4_ROLLBACK_RECEIPT_FAILED" for diagnostic in report.diagnostics)


def test_verification_mismatch_is_hard_failure_and_quarantines_scope():
    scope = _scope(1, 4)
    journal_model.clear_quarantine_explicit_recovery(scope)
    executor = FakeExecutor()
    executor.verify_fail = {1}
    journal = _journal(executor)
    journal.record(_created_key(1, 1.0))
    report = journal.rollback()
    assert report.status is verification.RollbackStatus.INCOMPLETE
    assert journal.state is journal_model.JournalState.ROLLBACK_INCOMPLETE
    assert report.quarantine_keys == (scope,)
    assert journal_model.is_quarantined(scope)
    journal_model.clear_quarantine_explicit_recovery(scope)


def test_rollback_or_raise_uses_shared_v3_hard_error():
    executor = FakeExecutor()
    executor.fail_rollback = {1}
    journal = _journal(executor)
    journal.record(_created_key(1, 1.0))
    with pytest.raises(verification.RollbackIncompleteError):
        journal.rollback_or_raise()


def test_created_animation_storage_receipt_freezes_before_and_after_binding_tokens():
    receipt = journal_model.CreatedChannelBagReceipt(
        1,
        _scope(1, 4),
        owner_ptr=1,
        action_ptr=2,
        slot_ptr=3,
        bag_ptr=4,
        owner_binding_before=(1, None, None, None),
        owner_binding_after=(1, 2, 3, 4),
    )
    assert receipt.owner_binding_before == (1, None, None, None)
    assert receipt.owner_binding_after == (1, 2, 3, 4)


def test_fcurve_snapshot_distinguishes_absence_from_existing_metadata():
    absent = journal_model.FCurveSnapshot(False)
    absent.validate()
    existing = journal_model.FCurveSnapshot(
        True,
        data_path='pose.bones["Root"].location',
        array_index=0,
        group_name="Root",
        extrapolation="CONSTANT",
    )
    existing.validate()
    assert existing.extrapolation == "CONSTANT"
