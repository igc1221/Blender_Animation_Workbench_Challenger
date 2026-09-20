import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "phase4_verification.py"
)
MODULE_NAME = "awb_phase4_verification_tests"

spec = spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert spec is not None and spec.loader is not None
verification = module_from_spec(spec)
sys.modules[MODULE_NAME] = verification
spec.loader.exec_module(verification)


def _record(key, *payload):
    return (tuple(key), tuple(payload))


def _snapshot(label, profile_records):
    profiles = tuple(
        (
            profile,
            tuple(sorted(records, key=lambda record: record[0])),
        )
        for profile, records in sorted(
            profile_records.items(), key=lambda item: tuple(verification.FingerprintProfile).index(item[0])
        )
    )
    return verification.CanonicalSnapshot(label, profiles)


def test_snapshot_label_is_not_part_of_state_canonicalization():
    records = {
        verification.FingerprintProfile.IDENTITY: [
            _record(("b",), 2),
            _record(("a",), 1),
        ]
    }
    left = _snapshot("before", records)
    right = _snapshot("after", records)
    assert left.canonical() == right.canonical()
    assert verification.diff_snapshots(left, right).is_empty


def test_diff_is_input_order_independent_and_detects_exact_change():
    before = _snapshot(
        "before",
        {
            verification.FingerprintProfile.RAW_ANIM: [
                _record(("owner", "location", "0", "0"), 1.0),
                _record(("owner", "rotation", "2", "0"), 0.25),
            ]
        },
    )
    after = verification.CanonicalSnapshot(
        "after",
        (
            (
                verification.FingerprintProfile.RAW_ANIM,
                (
                    _record(("owner", "rotation", "2", "0"), 0.25),
                    _record(("owner", "location", "0", "0"), 1.0000000000000002),
                ),
            ),
        ),
    )
    diff = verification.diff_snapshots(before, after)
    assert len(diff.entries) == 1
    assert diff.entries[0].kind == verification.DiffKind.CHANGED
    assert diff.entries[0].key == ("owner", "location", "0", "0")


def test_tolerant_comparison_is_recursive_and_explicit():
    before = _snapshot(
        "before",
        {
            verification.FingerprintProfile.EVAL_TOLERANT: [
                _record(("matrix",), ((1.0, 0.0), (0.0, 1.0)))
            ]
        },
    )
    after = _snapshot(
        "after",
        {
            verification.FingerprintProfile.EVAL_TOLERANT: [
                _record(("matrix",), ((1.0 + 1e-8, 0.0), (0.0, 1.0)))
            ]
        },
    )
    assert len(verification.diff_snapshots(before, after).entries) == 1
    assert verification.diff_snapshots_tolerant(before, after, 1e-6).is_empty
    assert len(verification.diff_snapshots_tolerant(before, after, 1e-10).entries) == 1


def test_tolerant_diff_defaults_to_evaluated_profile_only():
    before = _snapshot(
        "before",
        {
            verification.FingerprintProfile.RAW_POSE: [
                _record(("pose",), 1.0)
            ],
            verification.FingerprintProfile.EVAL_TOLERANT: [
                _record(("eval",), 1.0)
            ],
        },
    )
    after = _snapshot(
        "after",
        {
            verification.FingerprintProfile.RAW_POSE: [
                _record(("pose",), 1.0 + 1e-8)
            ],
            verification.FingerprintProfile.EVAL_TOLERANT: [
                _record(("eval",), 1.0 + 1e-8)
            ],
        },
    )
    assert verification.diff_snapshots_tolerant(before, after, 1e-6).is_empty
    exact = verification.diff_snapshots(before, after)
    assert {entry.profile for entry in exact.entries} == {
        verification.FingerprintProfile.RAW_POSE,
        verification.FingerprintProfile.EVAL_TOLERANT,
    }


def test_write_footprint_accepts_declared_prefix_and_rejects_other_channel():
    before = _snapshot(
        "before",
        {
            verification.FingerprintProfile.RAW_ANIM: [
                _record(("owner", "location", "0", "key", "1"), 1.0),
                _record(("owner", "rotation", "2", "key", "1"), 0.25),
            ]
        },
    )
    after = _snapshot(
        "after",
        {
            verification.FingerprintProfile.RAW_ANIM: [
                _record(("owner", "location", "0", "key", "1"), 2.0),
                _record(("owner", "rotation", "2", "key", "1"), 0.5),
            ]
        },
    )
    diff = verification.diff_snapshots(before, after)
    footprint = verification.WriteFootprint(
        (
            (
                verification.FingerprintProfile.RAW_ANIM,
                ("owner", "location"),
            ),
        )
    )
    unexpected = diff.unexpected_against(footprint)
    assert len(unexpected.entries) == 1
    assert unexpected.entries[0].key[1] == "rotation"


def test_write_footprint_require_match_all_detects_missing_expected_write():
    diff = verification.CanonicalDiff(
        (
            verification.DiffEntry(
                verification.FingerprintProfile.RAW_ANIM,
                ("root", "location", "0"),
                verification.DiffKind.CHANGED,
                (0.0,),
                (1.0,),
            ),
        )
    )
    footprint = verification.WriteFootprint(
        (
            (verification.FingerprintProfile.RAW_ANIM, ("root", "location")),
            (verification.FingerprintProfile.METADATA, ("descriptor",)),
        ),
        require_match_all=True,
    )
    assert footprint.all_matched(diff) is False


def test_fault_hook_fires_only_once_at_exact_stage_and_ordinal():
    hook = verification.FaultInjectingHook(
        verification.FaultSpec(verification.OperationStage.APPLY_WRITE, ordinal=2)
    )
    hook.enter(verification.OperationStage.APPLY_WRITE, ordinal=1)
    hook.enter(verification.OperationStage.PLAN)
    try:
        hook.enter(verification.OperationStage.APPLY_WRITE, ordinal=2)
    except RuntimeError as exc:
        assert "injected fault" in str(exc)
    else:
        raise AssertionError("Expected injected fault.")
    hook.enter(verification.OperationStage.APPLY_WRITE, ordinal=2)
    assert hook.fired is True
    assert len(hook.events) == 4


def test_transaction_probe_reports_verified_rollback_with_zero_diff():
    snapshot = _snapshot(
        "same",
        {verification.FingerprintProfile.RAW_POSE: [_record(("pose",), 1.0)]},
    )

    def snapshot_fn(_scene, request):
        return verification.CanonicalSnapshot(request.label, snapshot.profiles)

    request = verification.FingerprintRequest(
        (verification.FingerprintProfile.RAW_POSE,), label="before"
    )
    after_request = verification.FingerprintRequest(
        (verification.FingerprintProfile.RAW_POSE,), label="after"
    )
    probe = verification.TransactionProbe(object(), snapshot_fn)

    def operation(hook):
        hook.enter(verification.OperationStage.APPLY_WRITE)
        hook.enter(verification.OperationStage.ROLLBACK_WRITE)
        hook.enter(verification.OperationStage.ROLLBACK_VERIFY)
        return verification.OperationOutcome(
            False,
            rollback_status=verification.RollbackStatus.VERIFIED,
        )

    result = probe.run(
        operation,
        before_request=request,
        after_request=after_request,
        expect_zero_diff=True,
        expect_success=False,
    )
    assert result.rollback_status == verification.RollbackStatus.VERIFIED
    assert result.diff is not None and result.diff.is_empty
    assert not any(
        diagnostic.code == "UNEXPECTED_DIFF" for diagnostic in result.diagnostics
    )


def test_transaction_probe_keeps_rollback_incomplete_as_blocker():
    snapshot = _snapshot("same", {})

    def snapshot_fn(_scene, request):
        return verification.CanonicalSnapshot(request.label, snapshot.profiles)

    request = verification.FingerprintRequest(
        (verification.FingerprintProfile.IDENTITY,), label="state"
    )
    probe = verification.TransactionProbe(object(), snapshot_fn)

    def operation(_hook):
        raise verification.RollbackIncompleteError("restore verification failed")

    result = probe.run(operation, before_request=request, after_request=request)
    assert result.success is False
    assert result.rollback_status == verification.RollbackStatus.INCOMPLETE
    blockers = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.severity == verification.DiagnosticSeverity.BLOCKER
    ]
    assert [diagnostic.code for diagnostic in blockers] == ["ROLLBACK_INCOMPLETE"]


def test_fixture_manifest_validation_is_deterministic():
    clean = verification.FixtureManifest(
        "F-RootCOM",
        expected_object_names=("Root", "COM"),
        tags=("core",),
    )
    assert clean.validate() == ()

    invalid = verification.FixtureManifest(
        "",
        expected_object_names=("Root", "Root", " COM "),
    )
    codes = [diagnostic.code for diagnostic in invalid.validate()]
    assert codes == [
        "FIXTURE_ID_INVALID",
        "FIXTURE_DUPLICATE_ENTRY",
        "FIXTURE_ENTRY_INVALID",
    ]
