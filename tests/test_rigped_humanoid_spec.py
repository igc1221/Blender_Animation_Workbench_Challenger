import sys
from dataclasses import FrozenInstanceError, replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_humanoid_spec_tests"

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
rigped_contract = _load_module("rigped_contract")
humanoid = _load_module("rigped_humanoid_spec")


def _by_role(spec=None):
    spec = spec or humanoid.RigpedHumanoidSpec()
    return {bone.role: bone for bone in spec.resolved_bones()}


def test_default_humanoid_spec_validates_and_is_frozen():
    spec = humanoid.RigpedHumanoidSpec()
    spec.validate()
    assert len(spec.resolved_bones()) >= 60
    assert len(spec.resolved_kinematics()) == 4
    with pytest.raises(FrozenInstanceError):
        spec.character_label = "changed"


def test_default_humanoid_contains_product_minimum_authored_roles():
    by_role = _by_role()
    required = {
        "ROOT",
        "COM",
        "PELVIS",
        "SPINE",
        "NECK",
        "HEAD",
        "CLAVICLE.L",
        "CLAVICLE.R",
        "UPPER_ARM.L",
        "UPPER_ARM.R",
        "FOREARM.L",
        "FOREARM.R",
        "HAND.L",
        "HAND.R",
        "THIGH.L",
        "THIGH.R",
        "CALF.L",
        "CALF.R",
        "FOOT.L",
        "FOOT.R",
        "TOE.L",
        "TOE.R",
    }
    assert required <= set(by_role)
    assert all(by_role[role].layer == humanoid.HumanoidLayer.AUTHORED for role in required)


def test_humanoid_physically_separates_authored_mechanism_deform_and_export():
    by_role = _by_role()
    assert by_role["UPPER_ARM.L"].layer == humanoid.HumanoidLayer.AUTHORED
    assert by_role["MCH_UPPER_ARM.L"].layer == humanoid.HumanoidLayer.MECHANISM
    assert by_role["DEF_UPPER_ARM.L"].layer == humanoid.HumanoidLayer.DEFORM
    assert by_role["EXPORT_ROOT"].layer == humanoid.HumanoidLayer.EXPORT
    assert len(
        {
            by_role["UPPER_ARM.L"].name,
            by_role["MCH_UPPER_ARM.L"].name,
            by_role["DEF_UPPER_ARM.L"].name,
        }
    ) == 3


def test_humanoid_display_scale_must_be_finite_and_positive():
    for value in (0.0, -1.0, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="display_scale"):
            humanoid.RigpedHumanoidSpec(display_scale=value).validate()


def test_humanoid_all_awb_semantic_keys_are_canonical():
    spec = humanoid.RigpedHumanoidSpec()
    semantic_keys = [bone.semantic_key for bone in spec.resolved_bones()]
    semantic_keys += [group.semantic_key for group in spec.resolved_groups()]
    semantic_keys += [chain.semantic_key for chain in spec.resolved_chains()]
    semantic_keys += [mapping.semantic_key for mapping in spec.resolved_kinematics()]
    assert all(
        character_model.semantic_key_error(key, allow_empty=False) is None
        for key in semantic_keys
    )


def test_humanoid_has_four_nonstretch_two_bone_ik_mappings():
    spec = humanoid.RigpedHumanoidSpec()
    assert {mapping.key for mapping in spec.resolved_kinematics()} == {
        "ARM.L",
        "ARM.R",
        "LEG.L",
        "LEG.R",
    }
    ik = [constraint for constraint in spec.resolved_constraints() if constraint.kind == "IK"]
    assert len(ik) == 4
    assert all(constraint.chain_count == 2 for constraint in ik)
    assert all(constraint.use_stretch is False for constraint in ik)
    assert all(constraint.influence == 0.0 for constraint in ik)


def test_humanoid_fk_authority_does_not_full_transform_override_ik_result_chain():
    spec = humanoid.RigpedHumanoidSpec()
    mechanism_drives = [
        constraint
        for constraint in spec.resolved_constraints()
        if constraint.owner_role.startswith("MCH_")
        and constraint.target_role.split(".")[0]
        in {"UPPER_ARM", "FOREARM", "HAND", "THIGH", "CALF", "FOOT", "TOE"}
    ]
    assert mechanism_drives
    assert all(constraint.kind == "COPY_ROTATION" for constraint in mechanism_drives)
    assert all(constraint.target_space == "LOCAL" for constraint in mechanism_drives)
    assert all(constraint.owner_space == "LOCAL" for constraint in mechanism_drives)


def test_humanoid_toe_uses_mechanism_result_so_foot_ik_can_carry_it():
    spec = humanoid.RigpedHumanoidSpec()
    by_role = _by_role(spec)
    assert by_role["MCH_TOE.L"].parent_role == "MCH_FOOT.L"
    assert by_role["MCH_TOE.R"].parent_role == "MCH_FOOT.R"
    toe_deform_targets = {
        constraint.owner_role: constraint.target_role
        for constraint in spec.resolved_constraints()
        if constraint.owner_role in {"DEF_TOE.L", "DEF_TOE.R"}
    }
    assert toe_deform_targets == {
        "DEF_TOE.L": "MCH_TOE.L",
        "DEF_TOE.R": "MCH_TOE.R",
    }


def test_humanoid_i15_contact_hold_carriers_are_unparented_mechanism_targets():
    spec = humanoid.RigpedHumanoidSpec()
    by_role = _by_role(spec)
    expected = {
        "IK_HAND.L": "MCH_CONTACT_HAND.L",
        "IK_HAND.R": "MCH_CONTACT_HAND.R",
        "IK_FOOT.L": "MCH_CONTACT_FOOT.L",
        "IK_FOOT.R": "MCH_CONTACT_FOOT.R",
    }
    for hold_role in expected.values():
        hold = by_role[hold_role]
        assert hold.layer == humanoid.HumanoidLayer.MECHANISM
        assert hold.usage == "MECHANISM"
        assert hold.semantic_key == "awb.contact"
        assert hold.parent_role is None
        assert hold.deform is False

    hold_constraints = {
        constraint.owner_role: constraint
        for constraint in spec.resolved_constraints()
        if constraint.kind == "COPY_LOCATION"
        and constraint.target_role in set(expected.values())
    }
    assert set(hold_constraints) == set(expected)
    for owner_role, hold_role in expected.items():
        constraint = hold_constraints[owner_role]
        assert constraint.target_role == hold_role
        assert constraint.influence == 0.0
        assert constraint.target_space == "WORLD"
        assert constraint.owner_space == "WORLD"

    extras_by_key = {
        mapping.key: set(mapping.extra_roles)
        for mapping in spec.resolved_kinematics()
    }
    assert "MCH_CONTACT_HAND.L" in extras_by_key["ARM.L"]
    assert "MCH_CONTACT_HAND.R" in extras_by_key["ARM.R"]
    assert "MCH_CONTACT_FOOT.L" in extras_by_key["LEG.L"]
    assert "MCH_CONTACT_FOOT.R" in extras_by_key["LEG.R"]


def test_humanoid_opposites_cover_authored_left_right_animation_controls():
    spec = humanoid.RigpedHumanoidSpec()
    pairs = {(pair.left_role, pair.right_role) for pair in spec.resolved_opposites()}
    by_role = _by_role(spec)
    for stem in (
        "CLAVICLE",
        "UPPER_ARM",
        "FOREARM",
        "HAND",
        "IK_HAND",
        "POLE_ELBOW",
        "THIGH",
        "CALF",
        "FOOT",
        "TOE",
        "IK_FOOT",
        "POLE_KNEE",
    ):
        assert (f"{stem}.L", f"{stem}.R") in pairs
    assert all(
        by_role[left].layer == humanoid.HumanoidLayer.AUTHORED
        and by_role[right].layer == humanoid.HumanoidLayer.AUTHORED
        for left, right in pairs
    )


def test_humanoid_internal_layers_are_not_primary_animation_controls():
    spec = humanoid.RigpedHumanoidSpec()
    for bone in spec.resolved_bones():
        if bone.layer == humanoid.HumanoidLayer.MECHANISM:
            assert bone.usage == "MECHANISM"
        if bone.layer == humanoid.HumanoidLayer.DEFORM:
            assert bone.usage == "DEFORM"
        if bone.layer == humanoid.HumanoidLayer.EXPORT:
            assert bone.usage == "REFERENCE"


def test_humanoid_invalid_mirror_semantics_fail_closed():
    spec = humanoid.RigpedHumanoidSpec()
    bones = list(spec.resolved_bones())
    index = next(index for index, bone in enumerate(bones) if bone.role == "HAND.R")
    bones[index] = replace(bones[index], semantic_key="awb.foot")
    broken = replace(spec, bones=tuple(bones))
    with pytest.raises(ValueError, match="inconsistent semantics"):
        broken.validate()


def test_humanoid_invalid_parent_cycle_fail_closed():
    spec = humanoid.RigpedHumanoidSpec()
    bones = list(spec.resolved_bones())
    index = next(index for index, bone in enumerate(bones) if bone.role == "ROOT")
    bones[index] = replace(bones[index], parent_role="COM")
    broken = replace(spec, bones=tuple(bones))
    with pytest.raises(ValueError, match="parent cycle"):
        broken.validate()
