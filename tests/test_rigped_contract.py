import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_contract_tests"

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


class FakeID:
    def __init__(self, name: str, pointer: int, *, object_type: str = "EMPTY"):
        self.name = name
        self._pointer = pointer
        self.type = object_type
        self.parent = None
        self._props = {}

    def as_pointer(self):
        return self._pointer

    def get(self, key, default=None):
        return self._props.get(key, default)

    def __setitem__(self, key, value):
        self._props[key] = value


class FakeBoneData:
    def __init__(self, matrix_local, head, tail, length):
        self.matrix_local = matrix_local
        self.head_local = head
        self.tail_local = tail
        self.length = length


class FakePoseBone(FakeID):
    def __init__(self, name: str, pointer: int, bone):
        super().__init__(name, pointer, object_type="BONE")
        self.bone = bone
        self.constraints = []


class FakeConstraint:
    def __init__(self, *, target=None, pole_target=None):
        self.type = "IK"
        self.target = target
        self.subtarget = ""
        self.pole_target = pole_target
        self.pole_subtarget = ""
        self.chain_count = 2
        self.use_tail = True
        self.use_stretch = False
        self.use_rotation = False
        self.influence = 1.0


def _resolved_object(obj):
    return semantic_adapter.ResolvedControl(
        semantic_model.AWBControl.object(obj.name),
        obj,
        obj,
        data_path_prefix="",
    )


def _resolved_bone(owner, bone):
    return semantic_adapter.ResolvedControl(
        semantic_model.AWBControl.bone(owner.name, bone.name),
        owner,
        bone,
        data_path_prefix=f'pose.bones["{bone.name}"]',
    )


def _view(*, with_descriptor=True, stale_descriptor=False, unknown_binding=False):
    root = FakeID("Root Visible Label", 100)
    rig = FakeID("Armature Visible Label", 200, object_type="ARMATURE")
    target = FakeID("IK Visible Label", 300)
    pole = FakeID("Pole Visible Label", 400)

    upper_data = FakeBoneData(
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
    )
    lower_data = FakeBoneData(
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 2.0),
        1.0,
    )
    upper = FakePoseBone("Upper Visible Label", 201, upper_data)
    lower = FakePoseBone("Lower Visible Label", 202, lower_data)
    lower.parent = upper
    lower.constraints.append(FakeConstraint(target=target, pole_target=pole))

    bindings = [
        character_model.CharacterBinding(
            "root",
            "root-owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.root",
            side=character_model.CharacterSide.CENTER,
            usage=character_model.ControlUsage.PRIMARY,
        ),
        character_model.CharacterBinding(
            "upper",
            "rig-owner",
            semantic_model.AWBControlKind.BONE,
            bone_id="upper-token",
            semantic_key="awb.upper_arm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.FK,
            usage=character_model.ControlUsage.PRIMARY,
        ),
        character_model.CharacterBinding(
            "lower",
            "rig-owner",
            semantic_model.AWBControlKind.BONE,
            bone_id="lower-token",
            semantic_key="awb.forearm",
            side=character_model.CharacterSide.LEFT,
            mode=character_model.ControlMode.FK,
            usage=character_model.ControlUsage.PRIMARY,
        ),
        character_model.CharacterBinding(
            "ik",
            "ik-owner",
            semantic_model.AWBControlKind.OBJECT,
            semantic_key="awb.hand",
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
    ]
    if unknown_binding:
        bindings.append(
            character_model.CharacterBinding(
                "custom",
                "custom-owner",
                semantic_model.AWBControlKind.OBJECT,
                semantic_key="custom.unknown",
                usage=character_model.ControlUsage.PRIMARY,
            )
        )

    definition = character_model.AWBCharacter(
        "character-id",
        "Visible Character Label",
        7,
        (
            character_model.CharacterOwner("root-owner", "Root Hint"),
            character_model.CharacterOwner("rig-owner", "Rig Hint"),
            character_model.CharacterOwner("ik-owner", "IK Hint"),
            character_model.CharacterOwner("pole-owner", "Pole Hint"),
            *(
                (character_model.CharacterOwner("custom-owner", "Custom Hint"),)
                if unknown_binding
                else ()
            ),
        ),
        tuple(bindings),
        chains=(
            character_model.CharacterChain(
                "fk-chain",
                "awb.arm",
                "Visible Chain Label",
                side=character_model.CharacterSide.LEFT,
                mode=character_model.ControlMode.FK,
                members=("upper", "lower"),
            ),
        ),
        kinematics=(
            character_model.KinematicMapping(
                "arm-map",
                "awb.arm",
                side=character_model.CharacterSide.LEFT,
                fk_chain_id="fk-chain",
                ik_target_binding_id="ik",
                pole_binding_id="pole",
            ),
        ),
    )
    resolved = {
        "root": _resolved_object(root),
        "upper": _resolved_bone(rig, upper),
        "lower": _resolved_bone(rig, lower),
        "ik": _resolved_object(target),
        "pole": _resolved_object(pole),
    }
    if unknown_binding:
        custom = FakeID("Custom Visible Label", 500)
        resolved["custom"] = _resolved_object(custom)

    view = SimpleNamespace(
        definition=definition,
        resolved_bindings=tuple(resolved.items()),
        issues=(),
        source_stamp=("source", 7),
    )
    if with_descriptor:
        signature = rigped_contract.compute_setup_signature(view)
        root[rigped_contract.RIGPED_SETUP_PROPERTY] = {
            "schema_version": rigped_contract.RIGPED_SETUP_SCHEMA_VERSION,
            "character_id": definition.character_id,
            "profile_id": rigped_contract.RIGPED_PROFILE_HUMANOID_V1,
            "revision": 1,
            "signature": "stale" if stale_descriptor else signature,
            "lifecycle": rigped_contract.RigpedLifecycle.FITTED_UNBOUND.value,
        }
    return view, root, rig, upper, lower, target, pole


def _resolved_target(view, *selected_binding_ids: str, active_binding_id: str | None = None):
    result = rigped_contract.resolve_rigped_view(
        view,
        selected_binding_ids=tuple(selected_binding_ids),
        active_binding_id=active_binding_id,
    )
    assert result.target is not None
    assert not result.issues
    return result.target


def test_i0_capability_overlay_is_conservative_and_composable():
    view, *_ = _view()
    by_id = {binding.binding_id: binding for binding in view.definition.bindings}

    assert rigped_contract.capabilities_for_binding(by_id["root"]) == (
        rigped_contract.RigpedCapability.ANIMATOR_SELECTABLE,
        rigped_contract.RigpedCapability.KEY_POSITION,
        rigped_contract.RigpedCapability.KEY_ROTATION,
        rigped_contract.RigpedCapability.DIRECT_MOVE,
        rigped_contract.RigpedCapability.DIRECT_ROTATE,
    )
    assert rigped_contract.RigpedCapability.KINEMATIC_MOVE in rigped_contract.capabilities_for_binding(
        by_id["upper"]
    )
    assert rigped_contract.capabilities_for_binding(by_id["ik"]) == (
        rigped_contract.RigpedCapability.ANIMATOR_SELECTABLE,
        rigped_contract.RigpedCapability.KEY_POSITION,
        rigped_contract.RigpedCapability.DIRECT_MOVE,
        rigped_contract.RigpedCapability.IK_EFFECTOR,
    )
    assert rigped_contract.RigpedCapability.POLE_TARGET in rigped_contract.capabilities_for_binding(
        by_id["pole"]
    )


def test_i16_transform_resolver_routes_direct_and_semantic_single_control():
    view, *_ = _view()

    root = _resolved_target(view, "root", active_binding_id="root")
    assert rigped_contract.resolve_rigped_transform(
        root, rigped_contract.RigpedTransformGesture.MOVE
    ).route == rigped_contract.RigpedTransformRoute.NATIVE
    assert rigped_contract.resolve_rigped_transform(
        root, rigped_contract.RigpedTransformGesture.ROTATE
    ).route == rigped_contract.RigpedTransformRoute.NATIVE

    upper = _resolved_target(view, "upper", active_binding_id="upper")
    assert rigped_contract.resolve_rigped_transform(
        upper, rigped_contract.RigpedTransformGesture.MOVE
    ).route == rigped_contract.RigpedTransformRoute.SEMANTIC_KINEMATIC
    assert rigped_contract.resolve_rigped_transform(
        upper, rigped_contract.RigpedTransformGesture.ROTATE
    ).route == rigped_contract.RigpedTransformRoute.NATIVE

    ik = _resolved_target(view, "ik", active_binding_id="ik")
    assert rigped_contract.resolve_rigped_transform(
        ik, rigped_contract.RigpedTransformGesture.MOVE
    ).route == rigped_contract.RigpedTransformRoute.NATIVE


def test_i16_transform_resolver_routes_contact_move_to_awb_boundary():
    view, *_ = _view()
    target = _resolved_target(view, "upper", active_binding_id="upper")
    upper = next(control for control in target.controls if control.binding_id == "upper")
    contact = rigped_contract.RigpedControlContract(
        binding_id=upper.binding_id,
        semantic_key="awb.hand",
        side=upper.side,
        authored_mode=upper.authored_mode,
        usage=upper.usage,
        capabilities=upper.capabilities + (rigped_contract.RigpedCapability.CONTACT_OWNER,),
        target=upper.target,
    )
    contact_target = rigped_contract.RigpedTarget(
        character_id=target.character_id,
        descriptor=target.descriptor,
        controls=(contact,),
        selected_binding_ids=target.selected_binding_ids,
        active_binding_id=target.active_binding_id,
        source_stamp=target.source_stamp,
        selector_character_id=target.selector_character_id,
        selector_was_stale=target.selector_was_stale,
    )

    decision = rigped_contract.resolve_rigped_transform(
        contact_target,
        rigped_contract.RigpedTransformGesture.MOVE,
    )
    assert decision.ok
    assert decision.route == rigped_contract.RigpedTransformRoute.SEMANTIC_CONTACT


def test_i16_transform_resolver_fails_closed_for_scale_multi_and_missing_active():
    view, *_ = _view()

    root = _resolved_target(view, "root", active_binding_id="root")
    scale = rigped_contract.resolve_rigped_transform(
        root,
        rigped_contract.RigpedTransformGesture.SCALE,
    )
    assert not scale.ok
    assert scale.route == rigped_contract.RigpedTransformRoute.REFUSE
    assert [issue.code for issue in scale.issues] == ["I16_SCALE_UNSUPPORTED"]

    multi = _resolved_target(view, "root", "upper", active_binding_id="root")
    multi_move = rigped_contract.resolve_rigped_transform(
        multi,
        rigped_contract.RigpedTransformGesture.MOVE,
    )
    assert not multi_move.ok
    assert [issue.code for issue in multi_move.issues] == ["I16_MULTI_SELECTION_UNPROVEN"]

    missing_active = _resolved_target(view, "root", active_binding_id=None)
    no_active_move = rigped_contract.resolve_rigped_transform(
        missing_active,
        rigped_contract.RigpedTransformGesture.MOVE,
    )
    assert not no_active_move.ok
    assert [issue.code for issue in no_active_move.issues] == ["I16_ACTIVE_CONTROL_REQUIRED"]


def test_setup_signature_ignores_visible_names_but_detects_rest_change():
    view, root, rig, upper, _lower, target, pole = _view(with_descriptor=False)
    signature = rigped_contract.compute_setup_signature(view)

    view.definition = character_model.AWBCharacter(
        view.definition.character_id,
        "Renamed Character Label",
        view.definition.revision,
        tuple(
            character_model.CharacterOwner(owner.owner_id, "renamed hint")
            for owner in view.definition.owners
        ),
        view.definition.bindings,
        groups=view.definition.groups,
        chains=tuple(
            character_model.CharacterChain(
                chain.chain_id,
                chain.semantic_key,
                "Renamed Chain Label",
                side=chain.side,
                mode=chain.mode,
                members=chain.members,
            )
            for chain in view.definition.chains
        ),
        opposites=view.definition.opposites,
        kinematics=view.definition.kinematics,
    )
    root.name = "Root Renamed"
    rig.name = "Rig Renamed"
    upper.name = "Upper Renamed"
    target.name = "Target Renamed"
    pole.name = "Pole Renamed"
    assert rigped_contract.compute_setup_signature(view) == signature

    upper.bone.tail_local = (0.0, 0.0, 1.25)
    assert rigped_contract.compute_setup_signature(view) != signature


def test_setup_signature_detects_constraint_space_change():
    view, _root, _rig, _upper, lower, _target, _pole = _view(with_descriptor=False)
    signature = rigped_contract.compute_setup_signature(view)
    lower.constraints[0].owner_space = "LOCAL"
    assert rigped_contract.compute_setup_signature(view) != signature


def test_setup_signature_ignores_animatable_constraint_influence():
    view, _root, _rig, _upper, lower, _target, _pole = _view(with_descriptor=False)
    signature = rigped_contract.compute_setup_signature(view)
    lower.constraints[0].influence = 0.0
    assert rigped_contract.compute_setup_signature(view) == signature


def test_missing_setup_descriptor_fails_closed_without_mutation():
    view, root, *_ = _view(with_descriptor=False)
    before = (view.definition, view.resolved_bindings, view.source_stamp, dict(root._props))

    result = rigped_contract.resolve_rigped_view(view, selected_binding_ids=("root",))

    assert result.target is None
    assert [issue.code for issue in result.issues] == ["MISSING_RIGPED_SETUP_DESCRIPTOR"]
    assert (view.definition, view.resolved_bindings, view.source_stamp, dict(root._props)) == before


def test_stale_setup_signature_fails_closed_without_repair():
    view, root, *_ = _view(stale_descriptor=True)
    before_props = dict(root._props)

    result = rigped_contract.resolve_rigped_view(view, selected_binding_ids=("root",))

    assert result.target is None
    assert [issue.code for issue in result.issues] == ["STALE_RIGPED_SETUP_DESCRIPTOR"]
    assert root._props == before_props


def test_resolved_rigped_exposes_descriptor_capabilities_and_stale_selector_sync():
    view, *_ = _view()
    before = (view.definition, view.resolved_bindings, view.source_stamp)

    result = rigped_contract.resolve_rigped_view(
        view,
        selected_binding_ids=("upper", "lower"),
        active_binding_id="lower",
        selector_character_id="old-selector-character",
    )

    assert result.issues == ()
    assert result.selector_sync_character_id == "character-id"
    target = result.target
    assert target is not None
    assert target.character_id == "character-id"
    assert target.descriptor.revision == 1
    assert target.selected_binding_ids == ("upper", "lower")
    assert target.active_binding_id == "lower"
    assert target.selector_was_stale is True
    assert target.selector_character_id == "old-selector-character"
    assert {contract.binding_id for contract in target.controls} == {
        "root",
        "upper",
        "lower",
        "ik",
        "pole",
    }
    assert (view.definition, view.resolved_bindings, view.source_stamp) == before


def test_unknown_generated_primary_role_fails_closed_instead_of_raw_transform_fallback():
    view, *_ = _view(unknown_binding=True)
    result = rigped_contract.resolve_rigped_view(view, selected_binding_ids=("custom",))
    assert result.target is None
    assert [issue.code for issue in result.issues] == ["UNSUPPORTED_RIGPED_ROLE_CAPABILITY"]


def _summary(character_ids=(), *, unassigned=()):
    return character_query.CharacterSelectionSummary(
        character_ids=tuple(character_ids),
        active_character_id=(character_ids[0] if len(character_ids) == 1 else None),
        unassigned=tuple(unassigned),
        issues=(),
    )


def test_native_selection_target_resolution_does_not_use_selector_state():
    character_id, issues = rigped_contract.select_character_id_for_context(
        _summary(("native-character",)),
        has_native_controls=True,
    )
    assert issues == ()
    assert character_id == "native-character"


def test_mixed_or_ambiguous_native_selection_fails_closed():
    foreign = SimpleNamespace()
    character_id, issues = rigped_contract.select_character_id_for_context(
        _summary(("character",), unassigned=(foreign,)),
        has_native_controls=True,
    )
    assert character_id is None
    assert [issue.code for issue in issues] == ["MIXED_RIGPED_AND_UNASSIGNED_SELECTION"]

    character_id, issues = rigped_contract.select_character_id_for_context(
        _summary(("character-a", "character-b")),
        has_native_controls=True,
    )
    assert character_id is None
    assert [issue.code for issue in issues] == ["AMBIGUOUS_RIGPED_TARGET"]


def test_no_native_selection_does_not_fall_back_to_selector_or_scene_character():
    character_id, issues = rigped_contract.select_character_id_for_context(
        _summary(("some-character",)),
        has_native_controls=False,
    )
    assert character_id is None
    assert [issue.code for issue in issues] == ["NO_NATIVE_RIGPED_SELECTION"]
