from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_operation_domain_tests"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package


class FakeCapability(StrEnum):
    ANIMATOR_SELECTABLE = "ANIMATOR_SELECTABLE"
    DIRECT_MOVE = "DIRECT_MOVE"
    DIRECT_ROTATE = "DIRECT_ROTATE"


@dataclass(frozen=True)
class FakeControlContext:
    mode: str
    controls: tuple
    active: object | None


class FakeContract:
    pass


def _runtime_control_key(control):
    return control.runtime_key


character_metadata = ModuleType(f"{PACKAGE_NAME}.character_metadata")
character_metadata.resolve_character = lambda _scene, _character_id: None
sys.modules[character_metadata.__name__] = character_metadata

representation = ModuleType(f"{PACKAGE_NAME}.phase4_representation_snap")
representation.resolve_limb_representation_capability = lambda _view, _mapping_id: None
sys.modules[representation.__name__] = representation

rigped_contract = ModuleType(f"{PACKAGE_NAME}.rigped_contract")
rigped_contract.RigpedCapability = FakeCapability
rigped_contract.RigpedControlContract = FakeContract
rigped_contract.resolve_rigped_target = lambda _scene, _context, selector_character_id=None: None
sys.modules[rigped_contract.__name__] = rigped_contract

semantic_adapter = ModuleType(f"{PACKAGE_NAME}.semantic_adapter")
semantic_adapter.ControlContext = FakeControlContext
semantic_adapter.runtime_control_key = _runtime_control_key
sys.modules[semantic_adapter.__name__] = semantic_adapter

spec = spec_from_file_location(
    f"{PACKAGE_NAME}.rigped_operation_domain",
    PACKAGE_PATH / "rigped_operation_domain.py",
)
assert spec is not None and spec.loader is not None
domain = module_from_spec(spec)
sys.modules[spec.name] = domain
spec.loader.exec_module(domain)


def _control(key):
    return SimpleNamespace(runtime_key=key)


def _contract(binding_id, key, capabilities=()):
    return SimpleNamespace(
        binding_id=binding_id,
        capabilities=tuple(capabilities),
        target=_control(key),
    )


def _target(controls, selected_binding_ids, *, active_binding_id=None):
    return SimpleNamespace(
        character_id="character",
        descriptor=SimpleNamespace(revision=7, signature="setup-signature"),
        controls=tuple(controls),
        selected_binding_ids=tuple(selected_binding_ids),
        active_binding_id=active_binding_id,
        source_stamp=("source", 1),
        selector_character_id=None,
        selector_was_stale=False,
    )


def _mapping(mapping_id, *, ik_target="ik", pole="pole", fk_chain_id="fk-chain"):
    return SimpleNamespace(
        mapping_id=mapping_id,
        fk_chain_id=fk_chain_id,
        ik_target_binding_id=ik_target,
        pole_binding_id=pole,
    )


def _limb_capability(*fk_ids, terminal="hand"):
    return SimpleNamespace(
        fk_binding_ids=tuple(fk_ids),
        authored_terminal_binding_id=terminal,
    )


def _rigped_result(target):
    return SimpleNamespace(target=target, issues=())


def _rep_result(capability):
    return SimpleNamespace(capability=capability, issues=())


def test_same_limb_multi_selection_dedupes_to_one_mapping(monkeypatch):
    first = _control((1, 11))
    second = _control((1, 12))
    context = FakeControlContext("POSE", (first, second), second)
    target = _target(
        (
            _contract("upper", (1, 11)),
            _contract("lower", (1, 12)),
        ),
        ("upper", "lower"),
        active_binding_id="lower",
    )
    mapping = _mapping("arm.L")
    view = SimpleNamespace(definition=SimpleNamespace(kinematics=(mapping,)))
    calls = {"rigped": 0}

    def resolve_target(_scene, _context, *, selector_character_id=None):
        calls["rigped"] += 1
        return _rigped_result(target)

    monkeypatch.setattr(domain, "resolve_rigped_target", resolve_target)
    monkeypatch.setattr(domain, "resolve_character", lambda _scene, _id: view)
    monkeypatch.setattr(
        domain,
        "resolve_limb_representation_capability",
        lambda _view, _mapping_id: _rep_result(
            _limb_capability("upper", "lower")
        ),
    )

    result = domain.resolve_operation_domain(
        SimpleNamespace(frame_current=10, frame_subframe=0.25),
        context,
    )

    assert result.ok
    assert calls["rigped"] == 1
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.selected_control_runtime_keys == ((1, 11), (1, 12))
    assert snapshot.selected_binding_ids == ("upper", "lower")
    assert snapshot.active_control_runtime_key == (1, 12)
    assert snapshot.active_binding_id == "lower"
    assert snapshot.contact_mapping_ids == ("arm.L",)
    assert snapshot.supported_direct_binding_ids == ()
    assert tuple(item.kind for item in snapshot.binding_domains) == (
        domain.OperationDomainKind.CONTACT_LIMB,
        domain.OperationDomainKind.CONTACT_LIMB,
    )
    assert snapshot.binding_domains[0].mapping_id == "arm.L"
    assert snapshot.binding_domains[1].mapping_id == "arm.L"


def test_unmatched_raw_selected_control_is_explicitly_rejected(monkeypatch):
    matched = _control((2, 21))
    unmatched = _control((2, 99))
    context = FakeControlContext("POSE", (matched, unmatched), matched)
    target = _target((_contract("upper", (2, 21)),), ("upper",), active_binding_id="upper")
    mapping = _mapping("arm.L")
    view = SimpleNamespace(definition=SimpleNamespace(kinematics=(mapping,)))

    monkeypatch.setattr(
        domain,
        "resolve_rigped_target",
        lambda _scene, _context, *, selector_character_id=None: _rigped_result(target),
    )
    monkeypatch.setattr(domain, "resolve_character", lambda _scene, _id: view)
    monkeypatch.setattr(
        domain,
        "resolve_limb_representation_capability",
        lambda _view, _mapping_id: _rep_result(_limb_capability("upper")),
    )

    result = domain.resolve_operation_domain(SimpleNamespace(), context)

    assert result.ok is False
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.selected_control_runtime_keys == ((2, 21), (2, 99))
    assert snapshot.selected_binding_ids == ("upper",)
    assert snapshot.rejected_control_runtime_keys == ((2, 99),)
    assert result.issues[0].code == "E1_SELECTED_CONTROL_OUTSIDE_RIGPED_TARGET"
    assert snapshot.binding_domains[1].kind is domain.OperationDomainKind.REJECTED
    assert snapshot.binding_domains[1].selected_binding_id is None


def test_explicit_direct_capability_classifies_direct_only_after_limb_ownership(monkeypatch):
    root = _control((3, 31))
    context = FakeControlContext("POSE", (root,), root)
    target = _target(
        (
            _contract(
                "root",
                (3, 31),
                (
                    FakeCapability.ANIMATOR_SELECTABLE,
                    FakeCapability.DIRECT_MOVE,
                    FakeCapability.DIRECT_ROTATE,
                ),
            ),
        ),
        ("root",),
        active_binding_id="root",
    )
    view = SimpleNamespace(definition=SimpleNamespace(kinematics=()))

    monkeypatch.setattr(
        domain,
        "resolve_rigped_target",
        lambda _scene, _context, *, selector_character_id=None: _rigped_result(target),
    )
    monkeypatch.setattr(domain, "resolve_character", lambda _scene, _id: view)

    result = domain.resolve_operation_domain(SimpleNamespace(frame_current=3), context)

    assert result.ok
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.contact_mapping_ids == ()
    assert snapshot.supported_direct_binding_ids == ("root",)
    assert snapshot.binding_domains[0].kind is domain.OperationDomainKind.DIRECT


def test_multiple_limb_owners_are_rejected_not_silently_deduped(monkeypatch):
    selected = _control((4, 41))
    context = FakeControlContext("POSE", (selected,), selected)
    target = _target((_contract("shared", (4, 41)),), ("shared",), active_binding_id="shared")
    mappings = (_mapping("arm.L"), _mapping("arm.R"))
    view = SimpleNamespace(definition=SimpleNamespace(kinematics=mappings))

    monkeypatch.setattr(
        domain,
        "resolve_rigped_target",
        lambda _scene, _context, *, selector_character_id=None: _rigped_result(target),
    )
    monkeypatch.setattr(domain, "resolve_character", lambda _scene, _id: view)
    monkeypatch.setattr(
        domain,
        "resolve_limb_representation_capability",
        lambda _view, _mapping_id: _rep_result(
            _limb_capability("shared", terminal=f"terminal:{_mapping_id}")
        ),
    )

    result = domain.resolve_operation_domain(SimpleNamespace(), context)

    assert result.ok is False
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.contact_mapping_ids == ()
    assert snapshot.rejected_control_runtime_keys == ((4, 41),)
    assert result.issues[0].code == "E1_AMBIGUOUS_LIMB_DOMAIN"
    assert snapshot.binding_domains[0].kind is domain.OperationDomainKind.REJECTED


def test_unsupported_binding_is_rejected_instead_of_promoted_to_direct(monkeypatch):
    selected = _control((5, 51))
    context = FakeControlContext("POSE", (selected,), selected)
    target = _target(
        (
            _contract(
                "unsupported",
                (5, 51),
                (FakeCapability.ANIMATOR_SELECTABLE,),
            ),
        ),
        ("unsupported",),
        active_binding_id="unsupported",
    )
    view = SimpleNamespace(definition=SimpleNamespace(kinematics=()))

    monkeypatch.setattr(
        domain,
        "resolve_rigped_target",
        lambda _scene, _context, *, selector_character_id=None: _rigped_result(target),
    )
    monkeypatch.setattr(domain, "resolve_character", lambda _scene, _id: view)

    result = domain.resolve_operation_domain(SimpleNamespace(), context)

    assert result.ok is False
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.contact_mapping_ids == ()
    assert snapshot.supported_direct_binding_ids == ()
    assert snapshot.rejected_control_runtime_keys == ((5, 51),)
    assert result.issues[0].code == "E1_UNSUPPORTED_OPERATION_DOMAIN"
    assert snapshot.binding_domains[0].kind is domain.OperationDomainKind.REJECTED


def test_resolver_does_not_mutate_input_selection_or_resolved_identity(monkeypatch):
    selected = _control((6, 61))
    context = FakeControlContext("POSE", (selected,), selected)
    target = _target(
        (
            _contract(
                "root",
                (6, 61),
                (
                    FakeCapability.ANIMATOR_SELECTABLE,
                    FakeCapability.DIRECT_MOVE,
                ),
            ),
        ),
        ("root",),
        active_binding_id="root",
    )
    view = SimpleNamespace(definition=SimpleNamespace(kinematics=()))
    scene = SimpleNamespace(frame_current=8, frame_subframe=0.5)
    before = (
        context.controls,
        context.active,
        tuple(control.runtime_key for control in context.controls),
        target.selected_binding_ids,
        target.active_binding_id,
        target.source_stamp,
        scene.frame_current,
        scene.frame_subframe,
    )

    monkeypatch.setattr(
        domain,
        "resolve_rigped_target",
        lambda _scene, _context, *, selector_character_id=None: _rigped_result(target),
    )
    monkeypatch.setattr(domain, "resolve_character", lambda _scene, _id: view)

    result = domain.resolve_operation_domain(scene, context)

    after = (
        context.controls,
        context.active,
        tuple(control.runtime_key for control in context.controls),
        target.selected_binding_ids,
        target.active_binding_id,
        target.source_stamp,
        scene.frame_current,
        scene.frame_subframe,
    )
    assert result.ok
    assert before == after

def test_failed_limb_capability_rejects_before_direct_fallback(monkeypatch):
    hand = _control((7, 71))
    context = FakeControlContext("POSE", (hand,), hand)
    target = _target(
        (
            _contract(
                "hand",
                (7, 71),
                (
                    FakeCapability.ANIMATOR_SELECTABLE,
                    FakeCapability.DIRECT_ROTATE,
                ),
            ),
        ),
        ("hand",),
        active_binding_id="hand",
    )
    mapping = _mapping("arm.L")
    view = SimpleNamespace(
        definition=SimpleNamespace(
            kinematics=(mapping,),
            chains=(
                SimpleNamespace(
                    chain_id="fk-chain",
                    members=("upper", "lower"),
                ),
            ),
            bindings=(
                SimpleNamespace(
                    binding_id="ik",
                    semantic_key="awb.hand",
                    side="LEFT",
                    usage="TARGET",
                    kind="BONE",
                ),
                SimpleNamespace(
                    binding_id="hand",
                    semantic_key="awb.hand",
                    side="LEFT",
                    usage="PRIMARY",
                    kind="BONE",
                ),
            ),
        )
    )

    monkeypatch.setattr(
        domain,
        "resolve_rigped_target",
        lambda _scene, _context, *, selector_character_id=None: _rigped_result(target),
    )
    monkeypatch.setattr(domain, "resolve_character", lambda _scene, _id: view)
    monkeypatch.setattr(
        domain,
        "resolve_limb_representation_capability",
        lambda _view, _mapping_id: SimpleNamespace(
            capability=None,
            issues=(SimpleNamespace(code="BROKEN_LIMB"),),
        ),
    )

    result = domain.resolve_operation_domain(SimpleNamespace(), context)

    assert result.ok is False
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.supported_direct_binding_ids == ()
    assert snapshot.contact_mapping_ids == ()
    item = snapshot.binding_domains[0]
    assert item.kind is domain.OperationDomainKind.REJECTED
    assert item.mapping_id == "arm.L"
    assert item.rejection_code == "E1_UNSUPPORTED_LIMB_CAPABILITY"
    assert "BROKEN_LIMB" in item.detail


def test_selection_binding_reconciliation_is_symmetric(monkeypatch):
    root = _control((8, 81))
    context = FakeControlContext("POSE", (root,), root)
    target = _target(
        (
            _contract(
                "root",
                (8, 81),
                (
                    FakeCapability.ANIMATOR_SELECTABLE,
                    FakeCapability.DIRECT_MOVE,
                ),
            ),
        ),
        (),
        active_binding_id=None,
    )
    view = SimpleNamespace(definition=SimpleNamespace(kinematics=()))

    monkeypatch.setattr(
        domain,
        "resolve_rigped_target",
        lambda _scene, _context, *, selector_character_id=None: _rigped_result(target),
    )
    monkeypatch.setattr(domain, "resolve_character", lambda _scene, _id: view)

    result = domain.resolve_operation_domain(SimpleNamespace(), context)

    assert result.ok is False
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.supported_direct_binding_ids == ("root",)
    assert any(
        issue.code == "E1_SELECTION_BINDING_MISMATCH"
        for issue in result.issues
    )

def test_capability_domain_is_subset_of_topology_candidate_domain():
    mapping = _mapping("arm.L")
    view = SimpleNamespace(
        definition=SimpleNamespace(
            chains=(
                SimpleNamespace(
                    chain_id="fk-chain",
                    members=("upper", "lower"),
                ),
            ),
            bindings=(
                SimpleNamespace(
                    binding_id="ik",
                    semantic_key="awb.hand",
                    side="LEFT",
                    usage="TARGET",
                    kind="BONE",
                ),
                SimpleNamespace(
                    binding_id="hand",
                    semantic_key="awb.hand",
                    side="LEFT",
                    usage="PRIMARY",
                    kind="BONE",
                ),
            ),
        )
    )
    capability = _limb_capability("upper", "lower", terminal="hand")

    capability_ids = set(domain.limb_domain_binding_ids(mapping, capability))
    topology_ids = set(domain._candidate_limb_domain_binding_ids(view, mapping))

    assert capability_ids <= topology_ids
    assert capability_ids == {"upper", "lower", "hand", "ik", "pole"}


def test_selection_binding_reconciliation_catches_target_only_binding(monkeypatch):
    context = FakeControlContext("POSE", (), None)
    target = _target(
        (
            _contract(
                "root",
                (9, 91),
                (
                    FakeCapability.ANIMATOR_SELECTABLE,
                    FakeCapability.DIRECT_MOVE,
                ),
            ),
        ),
        ("root",),
        active_binding_id=None,
    )
    view = SimpleNamespace(definition=SimpleNamespace(kinematics=()))

    monkeypatch.setattr(
        domain,
        "resolve_rigped_target",
        lambda _scene, _context, *, selector_character_id=None: _rigped_result(target),
    )
    monkeypatch.setattr(domain, "resolve_character", lambda _scene, _id: view)

    result = domain.resolve_operation_domain(SimpleNamespace(), context)

    assert result.ok is False
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.binding_domains == ()
    assert snapshot.selected_binding_ids == ("root",)
    assert any(
        issue.code == "E1_SELECTION_BINDING_MISMATCH"
        for issue in result.issues
    )
