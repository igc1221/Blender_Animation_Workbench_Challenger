import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_character_query_tests"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package


def _load_module(name: str):
    qualified_name = f"{PACKAGE_NAME}.{name}"
    spec = spec_from_file_location(qualified_name, PACKAGE_PATH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


semantic_model = _load_module("semantic_model")
semantic_adapter = _load_module("semantic_adapter")
character_model = _load_module("character_model")
character_query = _load_module("character_query")


class FakeNative:
    def __init__(self, pointer: int):
        self.pointer = pointer

    def as_pointer(self):
        return self.pointer


def _resolved(name: str, pointer: int):
    native = FakeNative(pointer)
    return semantic_adapter.ResolvedControl(
        semantic_model.AWBControl.object(name),
        native,
        native,
    )


def _query_fixture():
    owner = character_model.CharacterOwner("owner", "Rig")
    bindings = (
        character_model.CharacterBinding(
            "fk-a",
            "owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.FK,
        ),
        character_model.CharacterBinding(
            "fk-b",
            "owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.FK,
        ),
        character_model.CharacterBinding(
            "ik",
            "owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.IK,
            usage=character_model.ControlUsage.TARGET,
        ),
        character_model.CharacterBinding(
            "pole",
            "owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.IK,
            usage=character_model.ControlUsage.POLE,
        ),
        character_model.CharacterBinding(
            "helper",
            "owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="custom.arm_helper",
            side=character_model.CharacterSide.LEFT,
            usage=character_model.ControlUsage.HELPER,
        ),
        character_model.CharacterBinding(
            "alias",
            "owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="custom.arm_alias",
            side=character_model.CharacterSide.LEFT,
        ),
        character_model.CharacterBinding(
            "right",
            "owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.arm",
            side=character_model.CharacterSide.RIGHT,
            mode=character_model.ControlMode.FK,
        ),
    )
    group = character_model.CharacterGroup(
        "group",
        "custom.arm_controls",
        "Arm Controls",
        members=("helper", "fk-a", "ik"),
    )
    chain = character_model.CharacterChain(
        "fk-chain",
        "awb.arm",
        "Left Arm FK",
        side=character_model.CharacterSide.LEFT,
        mode=character_model.ControlMode.FK,
        members=("fk-b", "fk-a"),
    )
    mapping = character_model.KinematicMapping(
        "arm-map",
        "awb.arm",
        side=character_model.CharacterSide.LEFT,
        fk_chain_id="fk-chain",
        ik_target_binding_id="ik",
        pole_binding_id="pole",
        extras=("helper", "alias"),
    )
    definition = character_model.AWBCharacter(
        "character",
        "Hero",
        1,
        (owner,),
        bindings,
        groups=(group,),
        chains=(chain,),
        opposites=(character_model.OppositePair("fk-a", "right"),),
        kinematics=(mapping,),
    )
    targets = {
        "fk-a": _resolved("FKA", 10),
        "fk-b": _resolved("FKB", 20),
        "ik": _resolved("IK", 30),
        "pole": _resolved("Pole", 40),
        "helper": _resolved("Helper", 50),
        "alias": _resolved("FKA Alias", 10),
        "right": _resolved("Right", 60),
    }
    view = SimpleNamespace(
        definition=definition,
        resolved_bindings=tuple((binding_id, targets[binding_id]) for binding_id in targets),
        issues=(),
        source_stamp=("stamp", 1),
    )
    return definition, targets, view


def test_semantic_index_is_call_scoped_and_preserves_explicit_graphs():
    definition, _targets, _view = _query_fixture()
    index = character_query.build_semantic_index(definition)

    assert tuple(binding.binding_id for binding in index.bindings_by_semantic["awb.arm"]) == (
        "fk-a",
        "fk-b",
        "ik",
        "pole",
        "right",
    )
    assert index.groups_by_id["group"].members == ("helper", "fk-a", "ik")
    assert index.chains_by_id["fk-chain"].members == ("fk-b", "fk-a")
    assert index.kinematics_by_id["arm-map"].fk_chain_id == "fk-chain"
    assert index.opposites_by_binding == {"fk-a": "right", "right": "fk-a"}


def test_pure_semantic_group_chain_opposite_and_kinematic_queries():
    definition, _targets, _view = _query_fixture()

    left_fk = character_query.members_for_semantic(
        definition,
        "awb.arm",
        side=character_model.CharacterSide.LEFT,
        mode=character_model.ControlMode.FK,
    )
    assert tuple(binding.binding_id for binding in left_fk) == ("fk-a", "fk-b")
    assert tuple(binding.binding_id for binding in character_query.group_members(definition, "group")) == (
        "helper",
        "fk-a",
        "ik",
    )
    assert tuple(
        binding.binding_id for binding in character_query.ordered_chain(definition, "fk-chain")
    ) == ("fk-b", "fk-a")
    assert character_query.opposite_for_binding(definition, "fk-a").binding_id == "right"
    assert character_query.opposite_for_binding(definition, "ik") is None
    assert character_query.kinematic_mapping(definition, "arm-map").semantic_key == "awb.arm"
    assert character_query.kinematic_mapping(definition, "missing") is None


def test_selection_targets_preserves_semantic_order_filters_helpers_and_dedups_alias_targets():
    _definition, _targets, view = _query_fixture()

    expansion = character_query.selection_targets(view, mapping_id="arm-map")
    assert expansion.binding_ids == ("fk-b", "fk-a", "ik", "pole", "alias")
    assert tuple(target.control.object_name for target in expansion.targets) == (
        "FKB",
        "FKA",
        "IK",
        "Pole",
    )
    assert expansion.issues == ()
    assert expansion.source_stamp == ("stamp", 1)

    exact = character_query.selection_targets(view, binding_id="fk-a")
    assert exact.binding_ids == ("fk-a",)
    assert tuple(target.control.object_name for target in exact.targets) == ("FKA",)
    assert exact.issues == ()

    with_helpers = character_query.selection_targets(
        view,
        mapping_id="arm-map",
        include_helpers=True,
    )
    assert with_helpers.binding_ids == ("fk-b", "fk-a", "ik", "pole", "helper", "alias")
    assert tuple(target.control.object_name for target in with_helpers.targets) == (
        "FKB",
        "FKA",
        "IK",
        "Pole",
        "Helper",
    )


def test_selection_targets_reports_missing_and_conflicting_scopes_without_mutation():
    _definition, _targets, view = _query_fixture()
    before = (view.definition, view.resolved_bindings, view.issues, view.source_stamp)

    missing = character_query.selection_targets(view, chain_id="missing")
    assert missing.binding_ids == ()
    assert {issue.code for issue in missing.issues} == {"MISSING_CHAIN"}

    conflicting = character_query.selection_targets(
        view,
        semantic_key="awb.arm",
        mapping_id="arm-map",
    )
    assert conflicting.binding_ids == ()
    assert {issue.code for issue in conflicting.issues} == {"MULTIPLE_SELECTION_SCOPES"}
    assert (view.definition, view.resolved_bindings, view.issues, view.source_stamp) == before


def test_characters_for_context_handles_multiple_characters_unassigned_and_active(monkeypatch):
    target_a = _resolved("A", 101)
    target_b = _resolved("B", 202)
    unassigned = _resolved("U", 303)
    definition_a = character_model.AWBCharacter("char-a", "A", 0, (), ())
    definition_b = character_model.AWBCharacter("char-b", "B", 0, (), ())
    views = {
        "char-a": SimpleNamespace(
            definition=definition_a,
            resolved_bindings=(("a", target_a),),
            issues=(),
            source_stamp=("a",),
        ),
        "char-b": SimpleNamespace(
            definition=definition_b,
            resolved_bindings=(("b", target_b),),
            issues=(),
            source_stamp=("b",),
        ),
    }
    metadata = ModuleType(f"{PACKAGE_NAME}.character_metadata")
    metadata.character_ids = lambda _scene: ("char-a", "char-b")
    metadata.resolve_character = lambda _scene, character_id: views[character_id]
    monkeypatch.setitem(sys.modules, f"{PACKAGE_NAME}.character_metadata", metadata)

    context = semantic_adapter.ControlContext(
        mode="OBJECT",
        controls=(target_a, target_b, unassigned),
        active=target_b,
    )
    summary = character_query.characters_for_context(object(), context)

    assert summary.character_ids == ("char-a", "char-b")
    assert summary.active_character_id == "char-b"
    assert summary.unassigned == (unassigned,)
    assert summary.issues == ()


def test_characters_for_context_reports_ambiguous_runtime_membership(monkeypatch):
    shared = _resolved("Shared", 404)
    definition_a = character_model.AWBCharacter("char-a", "A", 0, (), ())
    definition_b = character_model.AWBCharacter("char-b", "B", 0, (), ())
    views = {
        "char-a": SimpleNamespace(
            definition=definition_a,
            resolved_bindings=(("a", shared),),
            issues=(),
            source_stamp=("a",),
        ),
        "char-b": SimpleNamespace(
            definition=definition_b,
            resolved_bindings=(("b", shared),),
            issues=(),
            source_stamp=("b",),
        ),
    }
    metadata = ModuleType(f"{PACKAGE_NAME}.character_metadata")
    metadata.character_ids = lambda _scene: ("char-a", "char-b")
    metadata.resolve_character = lambda _scene, character_id: views[character_id]
    monkeypatch.setitem(sys.modules, f"{PACKAGE_NAME}.character_metadata", metadata)

    context = semantic_adapter.ControlContext("OBJECT", (shared,), shared)
    summary = character_query.characters_for_context(object(), context)

    assert summary.character_ids == ("char-a", "char-b")
    assert summary.active_character_id is None
    assert {issue.code for issue in summary.issues} == {
        "AMBIGUOUS_CONTEXT_MEMBERSHIP",
        "AMBIGUOUS_ACTIVE_CHARACTER",
    }
