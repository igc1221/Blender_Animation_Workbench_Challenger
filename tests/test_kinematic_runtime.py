import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_kinematic_runtime_tests"

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
kinematic_runtime = _load_module("kinematic_runtime")


class FakeNative:
    def __init__(self, name: str, pointer: int):
        self.name = name
        self._pointer = pointer

    def as_pointer(self):
        return self._pointer


class FakePoseBone(FakeNative):
    def __init__(self, name: str, pointer: int):
        super().__init__(name, pointer)
        self.constraints = []


class FakeArmature(FakeNative):
    def __init__(self, name: str, pointer: int, bones):
        super().__init__(name, pointer)
        self.pose = SimpleNamespace(bones=list(bones))


class FakeConstraint:
    def __init__(
        self,
        *,
        target=None,
        subtarget="",
        pole_target=None,
        pole_subtarget="",
        use_rotation=False,
    ):
        self.type = "IK"
        self.target = target
        self.subtarget = subtarget
        self.pole_target = pole_target
        self.pole_subtarget = pole_subtarget
        self.use_rotation = use_rotation
        self.use_tail = True


def _resolved_bone(owner, bone):
    return semantic_adapter.ResolvedControl(
        semantic_model.AWBControl.bone(owner.name, bone.name),
        owner,
        bone,
        data_path_prefix=f'pose.bones["{bone.name}"]',
    )


def _resolved_object(obj):
    return semantic_adapter.ResolvedControl(
        semantic_model.AWBControl.object(obj.name),
        obj,
        obj,
        data_path_prefix="",
    )


def _fixture(*, authored_pole=True, actual_pole=True, second_constraint=False, outside=False):
    upper = FakePoseBone("Upper", 101)
    lower = FakePoseBone("Lower", 102)
    outside_bone = FakePoseBone("Outside", 103)
    rig = FakeArmature("Rig", 100, (upper, lower, outside_bone))
    ik = FakeNative("IK", 201)
    pole = FakeNative("Pole", 202)
    wrong_pole = FakeNative("WrongPole", 203)

    constraint = FakeConstraint(
        target=ik,
        pole_target=pole if actual_pole else None,
    )
    (outside_bone if outside else lower).constraints.append(constraint)
    if second_constraint:
        upper.constraints.append(FakeConstraint(target=ik, pole_target=pole if actual_pole else None))

    bindings = (
        character_model.CharacterBinding(
            "upper",
            "rig-owner",
            semantic_model.AWBControlKind.BONE,
            bone_id="upper-token",
            bone_name_hint="Upper",
            semantic_key="awb.arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.FK,
        ),
        character_model.CharacterBinding(
            "lower",
            "rig-owner",
            semantic_model.AWBControlKind.BONE,
            bone_id="lower-token",
            bone_name_hint="Lower",
            semantic_key="awb.arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.FK,
        ),
        character_model.CharacterBinding(
            "ik",
            "ik-owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.IK,
            usage=character_model.ControlUsage.TARGET,
        ),
        character_model.CharacterBinding(
            "pole",
            "pole-owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.IK,
            usage=character_model.ControlUsage.POLE,
        ),
        character_model.CharacterBinding(
            "wrong-pole",
            "wrong-pole-owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.IK,
            usage=character_model.ControlUsage.POLE,
        ),
    )
    fk_chain = character_model.CharacterChain(
        "fk-chain",
        "awb.arm",
        "FK",
        side=character_model.CharacterSide.LEFT,
        mode=character_model.ControlMode.FK,
        members=("upper", "lower"),
    )
    reference_chain = character_model.CharacterChain(
        "reference-chain",
        "awb.arm",
        "Reference",
        side=character_model.CharacterSide.LEFT,
        mode=character_model.ControlMode.NEUTRAL,
        members=("upper", "lower"),
    )
    mapping = character_model.KinematicMapping(
        "arm-map",
        "awb.arm",
        side=character_model.CharacterSide.LEFT,
        fk_chain_id="fk-chain",
        ik_target_binding_id="ik",
        pole_binding_id="pole" if authored_pole else None,
        reference_chain_id="reference-chain",
    )
    definition = character_model.AWBCharacter(
        "character",
        "Hero",
        3,
        (
            character_model.CharacterOwner("rig-owner", "Rig"),
            character_model.CharacterOwner("ik-owner", "IK"),
            character_model.CharacterOwner("pole-owner", "Pole"),
            character_model.CharacterOwner("wrong-pole-owner", "WrongPole"),
        ),
        bindings,
        chains=(fk_chain, reference_chain),
        kinematics=(mapping,),
    )
    resolved = {
        "upper": _resolved_bone(rig, upper),
        "lower": _resolved_bone(rig, lower),
        "ik": _resolved_object(ik),
        "pole": _resolved_object(pole),
        "wrong-pole": _resolved_object(wrong_pole),
    }
    view = SimpleNamespace(
        definition=definition,
        resolved_bindings=tuple(resolved.items()),
        issues=(),
        source_stamp=("character", 3),
    )
    return view, rig, upper, lower, outside_bone, constraint, ik, pole, wrong_pole


def test_native_ik_resolution_uses_reference_chain_and_exact_target_pole_identity():
    view, _rig, _upper, lower, _outside, constraint, _ik, _pole, _wrong = _fixture()
    before = (view.definition, view.resolved_bindings, view.source_stamp)

    result = kinematic_runtime.resolve_native_ik_capability(view, "arm-map")

    assert result.issues == ()
    capability = result.capability
    assert capability is not None
    assert capability.character_id == "character"
    assert capability.mapping_id == "arm-map"
    assert capability.driven_chain_id == "reference-chain"
    assert capability.driven_binding_ids == ("upper", "lower")
    assert capability.solver_owner.target is lower
    assert capability.constraint is constraint
    assert capability.ik_target.target.name == "IK"
    assert capability.pole_target.target.name == "Pole"
    assert capability.source_stamp == ("character", 3)
    assert (view.definition, view.resolved_bindings, view.source_stamp) == before


def test_native_ik_resolution_reports_missing_mapping_without_mutation():
    view, *_rest = _fixture()
    before = (view.definition, view.resolved_bindings, view.source_stamp)
    result = kinematic_runtime.resolve_native_ik_capability(view, "missing")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == ["MISSING_KINEMATIC_MAPPING"]
    assert (view.definition, view.resolved_bindings, view.source_stamp) == before


def test_native_ik_resolution_rejects_unresolved_chain_binding():
    view, *_rest = _fixture()
    view.resolved_bindings = tuple(
        item for item in view.resolved_bindings if item[0] != "upper"
    )
    result = kinematic_runtime.resolve_native_ik_capability(view, "arm-map")
    assert result.capability is None
    assert {issue.code for issue in result.issues} == {"UNRESOLVED_KINEMATIC_BINDING"}


def test_native_ik_resolution_rejects_multiple_armature_owners():
    view, _rig, _upper, _lower, _outside, _constraint, _ik, _pole, _wrong = _fixture()
    other_bone = FakePoseBone("OtherLower", 302)
    other_rig = FakeArmature("OtherRig", 300, (other_bone,))
    replacement = _resolved_bone(other_rig, other_bone)
    view.resolved_bindings = tuple(
        (binding_id, replacement if binding_id == "lower" else resolved)
        for binding_id, resolved in view.resolved_bindings
    )
    result = kinematic_runtime.resolve_native_ik_capability(view, "arm-map")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == ["NATIVE_IK_MULTIPLE_ARMATURE_OWNERS"]


def test_native_ik_resolution_requires_authored_pole_when_constraint_uses_one():
    view, *_rest = _fixture(authored_pole=False, actual_pole=True)
    result = kinematic_runtime.resolve_native_ik_capability(view, "arm-map")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == ["NATIVE_IK_POLE_NOT_AUTHORED"]


def test_native_ik_resolution_rejects_pole_mismatch():
    view, _rig, _upper, _lower, _outside, constraint, _ik, _pole, wrong_pole = _fixture()
    constraint.pole_target = wrong_pole
    result = kinematic_runtime.resolve_native_ik_capability(view, "arm-map")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == ["NATIVE_IK_POLE_MISMATCH"]


def test_native_ik_resolution_rejects_constraint_owner_outside_authored_chain():
    view, *_rest = _fixture(outside=True)
    result = kinematic_runtime.resolve_native_ik_capability(view, "arm-map")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == ["NATIVE_IK_OWNER_OUTSIDE_CHAIN"]


def test_native_ik_resolution_rejects_ambiguous_exact_constraints():
    view, *_rest = _fixture(second_constraint=True)
    result = kinematic_runtime.resolve_native_ik_capability(view, "arm-map")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == ["AMBIGUOUS_NATIVE_IK"]


def test_native_ik_resolution_rejects_rotation_target_until_offset_adapter_exists():
    view, _rig, _upper, _lower, _outside, constraint, *_rest = _fixture()
    constraint.use_rotation = True
    result = kinematic_runtime.resolve_native_ik_capability(view, "arm-map")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == ["NATIVE_IK_ROTATION_TARGET_UNSUPPORTED"]


def test_two_bone_pole_solver_uses_existing_bend_plane():
    result = kinematic_runtime.solve_two_bone_pole_position(
        root=(0.0, 0.0, 0.0),
        joint=(1.0, 1.0, 0.0),
        end=(2.0, 0.0, 0.0),
        distance=2.0,
    )
    assert result.ok is True
    assert result.issue_code is None
    assert result.position == (1.0, 3.0, 0.0)


def test_two_bone_pole_solver_fails_closed_for_straight_chain():
    result = kinematic_runtime.solve_two_bone_pole_position(
        root=(0.0, 0.0, 0.0),
        joint=(1.0, 0.0, 0.0),
        end=(2.0, 0.0, 0.0),
        distance=2.0,
    )
    assert result.ok is False
    assert result.position is None
    assert result.issue_code == "SINGULAR_TWO_BONE_POLE"


def test_two_bone_pole_solver_rejects_zero_distance():
    result = kinematic_runtime.solve_two_bone_pole_position(
        root=(0.0, 0.0, 0.0),
        joint=(1.0, 1.0, 0.0),
        end=(2.0, 0.0, 0.0),
        distance=0.0,
    )
    assert result.position is None
    assert result.issue_code == "SINGULAR_TWO_BONE_POLE"
