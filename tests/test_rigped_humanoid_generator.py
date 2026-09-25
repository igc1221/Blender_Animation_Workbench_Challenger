import sys
from dataclasses import FrozenInstanceError, fields
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_humanoid_generator_tests"

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
generator = _load_module("rigped_humanoid_generator")
create_policy = _load_module("rigped_create_policy")


def _resolved_signature(spec):
    return (
        spec.profile_id,
        spec.character_label,
        spec.collection_name,
        spec.armature_object_name,
        spec.armature_data_name,
        spec.world_location,
        spec.resolved_bones(),
        spec.resolved_constraints(),
        spec.resolved_groups(),
        spec.resolved_chains(),
        spec.resolved_opposites(),
        spec.resolved_kinematics(),
    )


def test_b1_parameters_are_frozen_and_validate_current_canonical_topology_only():
    parameters = generator.RigpedHumanoidParameters()
    parameters.validate()
    assert parameters.spine_segments == 2
    assert parameters.neck_segments == 1
    assert parameters.toe_segments == 1
    with pytest.raises(FrozenInstanceError):
        parameters.spine_segments = 3


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("spine_segments", 1, "exactly 2 spine"),
        ("spine_segments", 3, "exactly 2 spine"),
        ("neck_segments", 0, "exactly 1 neck"),
        ("neck_segments", 2, "exactly 1 neck"),
        ("toe_segments", 0, "exactly 1 toe"),
        ("toe_segments", 2, "exactly 1 toe"),
    ),
)
def test_b1_unsupported_structural_counts_fail_before_spec_generation(
    field_name,
    value,
    message,
):
    values = {field.name: field.default for field in fields(generator.RigpedHumanoidParameters)}
    values[field_name] = value
    with pytest.raises(ValueError, match=message):
        generator.generate_rigped_humanoid_spec(generator.RigpedHumanoidParameters(**values))


def test_b1_default_generation_is_deterministic_and_field_equivalent_to_canonical():
    canonical = humanoid.RigpedHumanoidSpec()
    first = generator.generate_rigped_humanoid_spec()
    second = generator.generate_rigped_humanoid_spec(generator.RigpedHumanoidParameters())

    assert first == second
    assert _resolved_signature(first) == _resolved_signature(canonical)
    assert first.bones == humanoid.DEFAULT_HUMANOID_BONES
    assert first.constraints == humanoid.DEFAULT_HUMANOID_CONSTRAINTS
    assert first.groups == humanoid.DEFAULT_HUMANOID_GROUPS
    assert first.chains == humanoid.DEFAULT_HUMANOID_CHAINS
    assert first.opposites == humanoid.DEFAULT_HUMANOID_OPPOSITES
    assert first.kinematics == humanoid.DEFAULT_HUMANOID_KINEMATICS
    first.validate()


def test_b1_default_generation_preserves_order_parenting_layers_and_semantics():
    generated = generator.generate_rigped_humanoid_spec()
    canonical = humanoid.RigpedHumanoidSpec()

    assert tuple(bone.role for bone in generated.resolved_bones()) == tuple(
        bone.role for bone in canonical.resolved_bones()
    )
    assert tuple(bone.name for bone in generated.resolved_bones()) == tuple(
        bone.name for bone in canonical.resolved_bones()
    )

    generated_bones = generated.resolved_bones()
    canonical_bones = canonical.resolved_bones()
    assert tuple(
        (
            bone.role,
            bone.name,
            bone.semantic_key,
            bone.side,
            bone.mode,
            bone.usage,
            bone.layer,
            bone.parent_role,
            bone.connected,
            bone.deform,
            bone.head,
            bone.tail,
        )
        for bone in generated_bones
    ) == tuple(
        (
            bone.role,
            bone.name,
            bone.semantic_key,
            bone.side,
            bone.mode,
            bone.usage,
            bone.layer,
            bone.parent_role,
            bone.connected,
            bone.deform,
            bone.head,
            bone.tail,
        )
        for bone in canonical_bones
    )


def test_b1_keeps_two_bone_nonstretch_limbs_one_toe_and_one_finger_per_side():
    generated = generator.generate_rigped_humanoid_spec()
    roles = tuple(bone.role for bone in generated.resolved_bones())

    ik_constraints = tuple(
        constraint
        for constraint in generated.resolved_constraints()
        if constraint.kind == "IK"
    )
    assert len(ik_constraints) == 4
    assert all(constraint.chain_count == 2 for constraint in ik_constraints)
    assert all(constraint.use_stretch is False for constraint in ik_constraints)

    assert roles.count("TOE.L") == 1
    assert roles.count("TOE.R") == 1
    assert roles.count("FINGER.L") == 1
    assert roles.count("FINGER.R") == 1
    assert roles.count("DEF_FINGER.L") == 1
    assert roles.count("DEF_FINGER.R") == 1
    assert sum(
        bone.semantic_key == "awb.finger"
        for bone in generated.resolved_bones()
    ) == 4
    assert any(
        group.semantic_key == "awb.hand" and "FINGER.L" in group.member_roles
        for group in generated.resolved_groups()
    )
    assert any(
        group.semantic_key == "awb.hand" and "FINGER.R" in group.member_roles
        for group in generated.resolved_groups()
    )


def test_b1_generated_spec_remains_compatible_with_create_fit_scaling_boundary():
    generated = generator.generate_rigped_humanoid_spec()
    source_height = create_policy.humanoid_spec_height(generated)
    fitted = create_policy.fitted_humanoid_spec(
        source_height * 1.25,
        (1.0, 2.0, 3.0),
        base=generated,
    )

    assert fitted.world_location == (1.0, 2.0, 3.0)
    assert create_policy.humanoid_spec_height(fitted) == pytest.approx(source_height * 1.25)
    assert fitted.resolved_constraints() == generated.resolved_constraints()
    assert fitted.resolved_groups() == generated.resolved_groups()
    assert fitted.resolved_chains() == generated.resolved_chains()
    assert fitted.resolved_opposites() == generated.resolved_opposites()
    assert fitted.resolved_kinematics() == generated.resolved_kinematics()


def test_b1_generator_module_stays_pure_data_and_runtime_independent():
    source = (PACKAGE_PATH / "rigped_humanoid_generator.py").read_text(encoding="utf-8")
    forbidden_imports = (
        "import bpy",
        "from bpy",
        "rigped_transform",
        "rigped_fit_runtime",
        "rigped_humanoid_builder",
        "phase4_contact_authoring",
    )
    assert all(token not in source for token in forbidden_imports)
