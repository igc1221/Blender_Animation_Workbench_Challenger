"""Focused read-only RC3 ownership verifier tests (no Blender import)."""

from __future__ import annotations

import shutil
from pathlib import Path

from scripts import verify_rc3_ownership_api_consistency as verifier


def _fixture(tmp_path: Path) -> Path:
    for module in (
        "rigped_operation_domain", "rigped_transform", "phase4_contact_authoring",
        "rigped_auto_key", "phase4_writer", "phase4_mutation_journal", "debug_replay",
    ):
        source = verifier.ROOT / verifier.PACKAGE / f"{module}.py"
        destination = tmp_path / verifier.PACKAGE / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for relative in verifier.ACTIVE_DOCS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verifier.ROOT / relative, destination)
    directory = tmp_path / verifier.RIGPED_PLANS
    directory.mkdir(parents=True)
    for name in verifier.RIGPED_CORE_PLAN_NAMES:
        (directory / name).write_text("# Plan\n", encoding="utf-8")
    return tmp_path


def _statuses(root: Path) -> dict[str, str]:
    return {item["id"]: item["status"] for item in verifier.verify(root)["checks"]}


def test_accepted_architecture_passes_with_rigped_core_plans(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    receipt = verifier.verify(root)
    assert receipt["schema"] == verifier.SCHEMA
    assert receipt["status"] == "PASS", receipt["checks"]
    assert len(receipt["checks"]) == 6


def test_missing_plan_and_missing_source_fail_closed(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    (root / verifier.RIGPED_PLANS / verifier.RIGPED_CORE_PLAN_NAMES[-1]).unlink()
    (root / verifier.PACKAGE / "phase4_writer.py").unlink()
    statuses = _statuses(root)
    assert statuses["rigped_core_plans"] == "FAIL"
    assert statuses["writer_boundary"] == "FAIL"
    assert verifier.verify(root)["status"] == "FAIL"


def test_legacy_operations_plan_directory_fails_closed(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    legacy = root / verifier.LEGACY_OPERATIONS_PLANS
    legacy.mkdir(parents=True)
    (legacy / "README.md").write_text("# stale\n", encoding="utf-8")
    assert _statuses(root)["rigped_core_plans"] == "FAIL"


def test_legacy_orchestrator_directory_fails_closed(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    legacy = root / verifier.LEGACY_ORCHESTRATOR
    legacy.mkdir(parents=True)
    (legacy / "README.md").write_text("# stale\n", encoding="utf-8")
    assert _statuses(root)["rigped_core_plans"] == "FAIL"


def test_missing_canonical_domain_api_fails_closed(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    domain = root / verifier.PACKAGE / "rigped_operation_domain.py"
    content = domain.read_text(encoding="utf-8")
    domain.write_text(content.replace("def candidate_limb_domain_binding_ids(", "def renamed_limb_candidates("), encoding="utf-8")
    assert _statuses(root)["canonical_domain"] == "FAIL"


def test_writer_domain_duplication_and_journal_policy_import_fail(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    writer = root / verifier.PACKAGE / "phase4_writer.py"
    writer.write_text(writer.read_text(encoding="utf-8") + "\ndef resolve_operation_domain():\n    pass\n", encoding="utf-8")
    journal = root / verifier.PACKAGE / "phase4_mutation_journal.py"
    journal.write_text(journal.read_text(encoding="utf-8") + "\nfrom .phase4_contact_authoring import execute_contact_intent_plan\n", encoding="utf-8")
    statuses = _statuses(root)
    assert statuses["writer_boundary"] == "FAIL"
    assert statuses["canonical_domain"] == "FAIL"
    assert statuses["generic_journal"] == "FAIL"


def test_replay_native_storage_writer_and_authority_drift_fail(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    replay = root / verifier.PACKAGE / "debug_replay.py"
    replay.write_text(replay.read_text(encoding="utf-8") + "\ndef rogue(bone):\n    bone.keyframe_insert('location')\n", encoding="utf-8")
    doc = root / verifier.ACTIVE_DOCS[0]
    doc.write_text(doc.read_text(encoding="utf-8") + "\nPermanent authority: docs/AGENT/OPERATIONS_PLANS/README.md\n", encoding="utf-8")
    statuses = _statuses(root)
    assert statuses["replay_boundary"] == "FAIL"
    assert statuses["active_docs_authority"] == "FAIL"


def test_cosmetic_changes_and_historical_plan_references_do_not_fail(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    writer = root / verifier.PACKAGE / "phase4_writer.py"
    writer.write_text("# moved without changing API\n\n" + writer.read_text(encoding="utf-8"), encoding="utf-8")
    doc = root / verifier.ACTIVE_DOCS[0]
    doc.write_text(doc.read_text(encoding="utf-8") + "\nHistorical: docs/AGENT/OPERATIONS_PLANS/RC1_CORE_HARDENING.md\n", encoding="utf-8")
    assert verifier.verify(root)["status"] == "PASS"
