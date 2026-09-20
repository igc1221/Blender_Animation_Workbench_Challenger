import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_character_model_tests"

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
character_model = _load_module("character_model")


def _character(*, owners=(), bindings=()):
    return character_model.AWBCharacter(
        character_id="character-a",
        label="Hero",
        revision=1,
        owners=tuple(owners),
        bindings=tuple(bindings),
    )


def _codes(definition):
    return {issue.code for issue in character_model.validate_character(definition)}


def test_valid_object_membership_has_no_issues():
    owner = character_model.CharacterOwner("owner-a", "Cube")
    binding = character_model.CharacterBinding(
        "binding-a",
        "owner-a",
        semantic_model.AWBControlKind.OBJECT,
    )

    assert character_model.validate_character(
        _character(owners=(owner,), bindings=(binding,))
    ) == ()


def test_object_binding_requires_existing_owner_row():
    binding = character_model.CharacterBinding(
        "binding-a",
        "missing-owner",
        semantic_model.AWBControlKind.OBJECT,
    )

    assert "MISSING_BINDING_OWNER" in _codes(_character(bindings=(binding,)))


def test_object_binding_rejects_bone_locator_fields():
    owner = character_model.CharacterOwner("owner-a", "Cube")
    binding = character_model.CharacterBinding(
        "binding-a",
        "owner-a",
        semantic_model.AWBControlKind.OBJECT,
        bone_id="not-valid-for-object",
        bone_name_hint="Bone",
    )

    assert "OBJECT_BINDING_HAS_BONE_LOCATOR" in _codes(
        _character(owners=(owner,), bindings=(binding,))
    )


def test_duplicate_owner_and_binding_ids_are_rejected():
    owners = (
        character_model.CharacterOwner("owner-a", "A"),
        character_model.CharacterOwner("owner-a", "B"),
    )
    bindings = (
        character_model.CharacterBinding(
            "binding-a",
            "owner-a",
            semantic_model.AWBControlKind.OBJECT,
        ),
        character_model.CharacterBinding(
            "binding-a",
            "owner-a",
            semantic_model.AWBControlKind.OBJECT,
        ),
    )

    codes = _codes(_character(owners=owners, bindings=bindings))
    assert "DUPLICATE_OWNER_ID" in codes
    assert "DUPLICATE_BINDING_ID" in codes


def test_bone_binding_contract_requires_token_without_changing_phase2_identity():
    owner = character_model.CharacterOwner("owner-rig", "Rig")
    missing = character_model.CharacterBinding(
        "binding-hand",
        "owner-rig",
        semantic_model.AWBControlKind.BONE,
        bone_name_hint="Hand.L",
    )
    valid = character_model.CharacterBinding(
        "binding-hand",
        "owner-rig",
        semantic_model.AWBControlKind.BONE,
        bone_id="bone-token",
        bone_name_hint="Hand.L",
    )

    assert "BONE_BINDING_MISSING_TOKEN" in _codes(
        _character(owners=(owner,), bindings=(missing,))
    )
    assert character_model.validate_character(
        _character(owners=(owner,), bindings=(valid,))
    ) == ()

    phase2_control = semantic_model.AWBControl.bone("Rig", "Hand.L")
    assert phase2_control.identity_key == (
        semantic_model.AWBControlKind.BONE,
        "Rig",
        "Hand.L",
    )


def test_semantic_key_accepts_canonical_and_unknown_custom_namespace_without_other_bucket():
    assert character_model.semantic_key_error("awb.hand", allow_empty=False) is None
    assert character_model.semantic_key_error("custom.tail", allow_empty=False) is None
    assert character_model.semantic_key_error("studio.hero.cape", allow_empty=False) is None
    assert character_model.semantic_key_error("robot.secondary_arm", allow_empty=False) is None
    assert (
        character_model.semantic_key_error("awb.not_a_core_key", allow_empty=False)
        == "UNKNOWN_AWB_SEMANTIC_KEY"
    )
    assert (
        character_model.semantic_key_error("Custom.Tail", allow_empty=False)
        == "INVALID_SEMANTIC_KEY"
    )
    assert character_model.semantic_key_error("", allow_empty=True) is None


def test_group_cycle_and_dangling_members_are_rejected():
    owner = character_model.CharacterOwner("owner-a", "Rig")
    binding = character_model.CharacterBinding(
        "binding-a",
        "owner-a",
        semantic_model.AWBControlKind.OBJECT,
    )
    groups = (
        character_model.CharacterGroup(
            "group-a",
            "custom.face",
            "Face",
            parent_group_id="group-b",
            members=("binding-a", "missing-binding"),
        ),
        character_model.CharacterGroup(
            "group-b",
            "custom.head_controls",
            "Head Controls",
            parent_group_id="group-a",
        ),
    )
    definition = _character(owners=(owner,), bindings=(binding,))
    definition = character_model.AWBCharacter(
        character_id=definition.character_id,
        label=definition.label,
        revision=definition.revision,
        owners=definition.owners,
        bindings=definition.bindings,
        groups=groups,
    )

    codes = _codes(definition)
    assert "GROUP_PARENT_CYCLE" in codes
    assert "DANGLING_GROUP_MEMBER" in codes


def test_chain_order_is_explicit_variable_length_and_duplicate_or_dangling_members_fail():
    owner = character_model.CharacterOwner("owner-a", "Rig")
    bindings = tuple(
        character_model.CharacterBinding(
            f"binding-{index}",
            "owner-a",
            semantic_model.AWBControlKind.OBJECT,
        )
        for index in range(4)
    )
    valid_chain = character_model.CharacterChain(
        "chain-spine",
        "awb.spine",
        "Spine",
        side=character_model.CharacterSide.CENTER,
        members=tuple(binding.binding_id for binding in bindings),
    )
    valid = character_model.AWBCharacter(
        "character-a",
        "Hero",
        1,
        (owner,),
        bindings,
        chains=(valid_chain,),
    )
    assert character_model.validate_character(valid) == ()
    assert valid.chains[0].members == (
        "binding-0",
        "binding-1",
        "binding-2",
        "binding-3",
    )

    invalid_chain = character_model.CharacterChain(
        "chain-tail",
        "custom.tail",
        "Tail",
        members=("binding-0", "binding-0", "missing"),
    )
    invalid = character_model.AWBCharacter(
        "character-a",
        "Hero",
        1,
        (owner,),
        bindings,
        chains=(invalid_chain,),
    )
    codes = _codes(invalid)
    assert "DUPLICATE_CHAIN_MEMBER" in codes
    assert "DANGLING_CHAIN_MEMBER" in codes


def test_explicit_opposite_requires_left_right_bindings_and_unique_participation():
    owner = character_model.CharacterOwner("owner-a", "Rig")
    left = character_model.CharacterBinding(
        "left-hand",
        "owner-a",
        semantic_model.AWBControlKind.OBJECT,
        semantic_key="awb.hand",
        side=character_model.CharacterSide.LEFT,
    )
    right = character_model.CharacterBinding(
        "right-hand",
        "owner-a",
        semantic_model.AWBControlKind.OBJECT,
        semantic_key="awb.hand",
        side=character_model.CharacterSide.RIGHT,
    )
    valid = character_model.AWBCharacter(
        "character-a",
        "Hero",
        1,
        (owner,),
        (left, right),
        opposites=(character_model.OppositePair("left-hand", "right-hand"),),
    )
    assert character_model.validate_character(valid) == ()

    wrong = character_model.AWBCharacter(
        "character-a",
        "Hero",
        1,
        (owner,),
        (left, right),
        opposites=(
            character_model.OppositePair("right-hand", "left-hand"),
            character_model.OppositePair("left-hand", "right-hand"),
        ),
    )
    codes = _codes(wrong)
    assert "OPPOSITE_LEFT_SIDE_MISMATCH" in codes
    assert "OPPOSITE_RIGHT_SIDE_MISMATCH" in codes
    assert "MULTIPLE_OPPOSITE_RELATIONS" in codes


def test_generic_kinematic_mapping_supports_arm_leg_and_custom_wing_without_fixed_kind():
    owner = character_model.CharacterOwner("owner-a", "Rig")
    bindings = tuple(
        character_model.CharacterBinding(
            binding_id,
            "owner-a",
            semantic_model.AWBControlKind.OBJECT,
        )
        for binding_id in ("fk-a", "fk-b", "ik-target", "pole", "toe", "alias")
    )
    arm_fk = character_model.CharacterChain(
        "chain-arm-fk",
        "awb.arm",
        "Left Arm FK",
        side=character_model.CharacterSide.LEFT,
        mode=character_model.ControlMode.FK,
        members=("fk-a", "fk-b"),
    )
    leg_fk = character_model.CharacterChain(
        "chain-leg-fk",
        "awb.leg",
        "Right Leg FK",
        side=character_model.CharacterSide.RIGHT,
        mode=character_model.ControlMode.FK,
        members=("fk-a", "fk-b", "toe"),
    )
    mappings = (
        character_model.KinematicMapping(
            "map-arm-left",
            "awb.arm",
            side=character_model.CharacterSide.LEFT,
            fk_chain_id="chain-arm-fk",
            ik_target_binding_id="ik-target",
            pole_binding_id="pole",
            extras=("alias",),
        ),
        character_model.KinematicMapping(
            "map-leg-right",
            "awb.leg",
            side=character_model.CharacterSide.RIGHT,
            fk_chain_id="chain-leg-fk",
        ),
        character_model.KinematicMapping(
            "map-wing-left",
            "custom.wing",
            side=character_model.CharacterSide.LEFT,
            ik_target_binding_id="ik-target",
        ),
    )
    definition = character_model.AWBCharacter(
        "character-a",
        "Hero",
        1,
        (owner,),
        bindings,
        chains=(arm_fk, leg_fk),
        kinematics=mappings,
    )

    assert character_model.validate_character(definition) == ()
    assert definition.kinematics[2].semantic_key == "custom.wing"


def test_kinematic_mapping_rejects_empty_or_dangling_references_without_solver_assumptions():
    owner = character_model.CharacterOwner("owner-a", "Rig")
    binding = character_model.CharacterBinding(
        "binding-a",
        "owner-a",
        semantic_model.AWBControlKind.OBJECT,
    )
    empty = character_model.KinematicMapping("empty", "awb.arm")
    broken = character_model.KinematicMapping(
        "broken",
        "custom.wing",
        fk_chain_id="missing-fk",
        ik_target_binding_id="missing-target",
        pole_binding_id="missing-pole",
        reference_chain_id="missing-reference",
        extras=("binding-a", "binding-a", "missing-extra"),
    )
    definition = character_model.AWBCharacter(
        "character-a",
        "Hero",
        1,
        (owner,),
        (binding,),
        kinematics=(empty, broken),
    )

    codes = _codes(definition)
    assert "EMPTY_KINEMATIC_MAPPING" in codes
    assert "DANGLING_KINEMATIC_FK_CHAIN" in codes
    assert "DANGLING_KINEMATIC_IK_TARGET" in codes
    assert "DANGLING_KINEMATIC_POLE" in codes
    assert "DANGLING_KINEMATIC_REFERENCE_CHAIN" in codes
    assert "DUPLICATE_KINEMATIC_EXTRA" in codes
    assert "DANGLING_KINEMATIC_EXTRA" in codes
