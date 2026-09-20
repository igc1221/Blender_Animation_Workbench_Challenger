import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_character_metadata_tests"

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


class FakeObject:
    _next_pointer = 100

    def __init__(
        self,
        name: str,
        *,
        linked: bool = False,
        editable: bool = True,
        object_type: str = "MESH",
        data=None,
    ):
        self.name = name
        self.library = object() if linked else None
        self.is_editable = editable
        self.type = object_type
        self.data = data
        self.pose = None
        self._pointer = FakeObject._next_pointer
        FakeObject._next_pointer += 1

    def as_pointer(self):
        return self._pointer


class FakeBone(dict):
    def __init__(self, name: str):
        super().__init__()
        self.name = name


class FakeBoneCollection(list):
    def get(self, name: str):
        return next((bone for bone in self if bone.name == name), None)


class FakeArmatureData:
    def __init__(self, bones=(), *, linked: bool = False, users: int = 1):
        self.library = object() if linked else None
        self.bones = FakeBoneCollection(bones)
        self.users = users


class FakePoseBone:
    def __init__(self, owner, bone: FakeBone):
        self.id_data = owner
        self.bone = bone
        self.name = bone.name
        self.select = True

    def get(self, _key, default=None):
        return default

    def path_from_id(self):
        return f'pose.bones["{self.name}"]'


class FakePoseBones(dict):
    pass


def make_fake_rig(name="Rig", *, linked_object=False, linked_data=False, bone_names=("Hand.L",)):
    bones = [FakeBone(item) for item in bone_names]
    data = FakeArmatureData(bones, linked=linked_data)
    rig = FakeObject(name, linked=linked_object, object_type="ARMATURE", data=data)
    pose_bones = FakePoseBones({bone.name: FakePoseBone(rig, bone) for bone in bones})
    rig.pose = SimpleNamespace(bones=pose_bones)
    return rig, pose_bones


class FakeCollection(list):
    def __init__(self, factory):
        super().__init__()
        self.factory = factory

    def add(self):
        value = self.factory()
        self.append(value)
        return value

    def remove(self, index):
        del self[index]


class FakeOwnerRow:
    def __init__(self):
        self.owner_id = ""
        self.object_ref = None
        self.label_hint = ""


class FakeBindingRefRow:
    def __init__(self):
        self.binding_id = ""


class FakeBindingRow:
    def __init__(self):
        self.binding_id = ""
        self.owner_id = ""
        self.kind = semantic_model.AWBControlKind.OBJECT.value
        self.bone_id = ""
        self.bone_name_hint = ""
        self.semantic_key = ""
        self.side = character_model.CharacterSide.NONE.value
        self.mode = character_model.ControlMode.NEUTRAL.value
        self.usage = character_model.ControlUsage.PRIMARY.value


class FakeGroupRow:
    def __init__(self):
        self.group_id = ""
        self.semantic_key = ""
        self.label = ""
        self.side = character_model.CharacterSide.NONE.value
        self.parent_group_id = ""
        self.members = FakeCollection(FakeBindingRefRow)


class FakeChainRow:
    def __init__(self):
        self.chain_id = ""
        self.semantic_key = ""
        self.label = ""
        self.side = character_model.CharacterSide.NONE.value
        self.mode = character_model.ControlMode.NEUTRAL.value
        self.members = FakeCollection(FakeBindingRefRow)


class FakeOppositeRow:
    def __init__(self):
        self.left_binding_id = ""
        self.right_binding_id = ""


class FakeKinematicRow:
    def __init__(self):
        self.mapping_id = ""
        self.semantic_key = ""
        self.side = character_model.CharacterSide.NONE.value
        self.fk_chain_id = ""
        self.ik_target_binding_id = ""
        self.pole_binding_id = ""
        self.reference_chain_id = ""
        self.extras = FakeCollection(FakeBindingRefRow)


class FakeCharacterRow:
    def __init__(self):
        self.character_id = ""
        self.label = ""
        self.revision = 0
        self.owners = FakeCollection(FakeOwnerRow)
        self.bindings = FakeCollection(FakeBindingRow)
        self.groups = FakeCollection(FakeGroupRow)
        self.chains = FakeCollection(FakeChainRow)
        self.opposites = FakeCollection(FakeOppositeRow)
        self.kinematics = FakeCollection(FakeKinematicRow)


class FakeStore:
    def __init__(self, schema_version=1):
        self.schema_version = schema_version
        self.characters = FakeCollection(FakeCharacterRow)


class FakeScene:
    def __init__(self, objects=(), schema_version=1):
        self.objects = {obj.name: obj for obj in objects}
        self.awb_characters = FakeStore(schema_version=schema_version)


# Install the minimum bpy API needed to import the persistence module. These
# unit tests validate storage/preflight behavior only; actual RNA persistence is
# covered by the Blender runtime verifier.
bpy = ModuleType("bpy")
bpy.types = SimpleNamespace(
    PropertyGroup=object,
    Object=FakeObject,
    PoseBone=FakePoseBone,
    Scene=FakeScene,
)
bpy.utils = SimpleNamespace(register_class=lambda _cls: None, unregister_class=lambda _cls: None)
bpy.data = SimpleNamespace(scenes=[])
props = ModuleType("bpy.props")
for name in ("CollectionProperty", "IntProperty", "PointerProperty", "StringProperty"):
    setattr(props, name, lambda **_kwargs: None)
bpy.props = props
sys.modules["bpy"] = bpy
sys.modules["bpy.props"] = props

character_metadata = _load_module("character_metadata")


def test_empty_store_read_has_no_write():
    scene = FakeScene()
    before = len(scene.awb_characters.characters)

    assert character_metadata.read_character(scene, "missing") is None
    assert len(scene.awb_characters.characters) == before


def test_create_object_character_roundtrips_pure_snapshot():
    first = FakeObject("A")
    second = FakeObject("B")
    scene = FakeScene((first, second))

    character_id = character_metadata.create_character(scene, "Hero", (first, second))
    definition = character_metadata.read_character(scene, character_id)

    assert definition is not None
    assert definition.character_id == character_id
    assert definition.label == "Hero"
    assert definition.revision == 1
    assert len(definition.owners) == 2
    assert len(definition.bindings) == 2
    assert {owner.label_hint for owner in definition.owners} == {"A", "B"}
    assert all(
        binding.kind == semantic_model.AWBControlKind.OBJECT
        for binding in definition.bindings
    )


def test_object_target_cannot_belong_to_two_characters_in_same_scene():
    obj = FakeObject("A")
    scene = FakeScene((obj,))
    character_metadata.create_character(scene, "First", (obj,))
    before = len(scene.awb_characters.characters)

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.create_character(scene, "Second", (obj,))

    assert len(scene.awb_characters.characters) == before


def test_linked_object_rejects_without_partial_mutation():
    linked = FakeObject("Linked", linked=True)
    scene = FakeScene((linked,))

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.create_character(scene, "Nope", (linked,))

    assert scene.awb_characters.characters == []


def test_duplicate_object_target_is_deduplicated_before_write():
    obj = FakeObject("A")
    scene = FakeScene((obj,))

    character_id = character_metadata.create_character(scene, "Hero", (obj, obj))
    definition = character_metadata.read_character(scene, character_id)

    assert definition is not None
    assert len(definition.owners) == 1
    assert len(definition.bindings) == 1


def test_v2_bind_object_adds_explicit_owner_and_neutral_binding():
    first = FakeObject("A")
    second = FakeObject("B")
    scene = FakeScene((first, second), schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", (first,))
    before = character_metadata.character_stamp(scene, character_id)

    binding_id = character_metadata.bind_object(scene, character_id, second, expected_stamp=before)
    definition = character_metadata.read_character(scene, character_id)

    assert definition is not None
    assert definition.revision == 2
    assert len(definition.owners) == 2
    assert len(definition.bindings) == 2
    binding = next(item for item in definition.bindings if item.binding_id == binding_id)
    owner = next(item for item in definition.owners if item.owner_id == binding.owner_id)
    assert owner.label_hint == "B"
    assert binding.kind == semantic_model.AWBControlKind.OBJECT
    assert binding.semantic_key == ""
    assert binding.side == character_model.CharacterSide.NONE
    assert binding.mode == character_model.ControlMode.NEUTRAL
    assert binding.usage == character_model.ControlUsage.PRIMARY


def test_v2_bind_object_rejects_cross_character_and_read_only_without_partial_mutation():
    first = FakeObject("A")
    second = FakeObject("B")
    read_only = FakeObject("ReadOnly", editable=False)
    scene = FakeScene((first, second, read_only), schema_version=2)
    first_id = character_metadata.create_character(scene, "First", (first,))
    second_id = character_metadata.create_character(scene, "Second", (second,))
    before = character_metadata.character_stamp(scene, first_id)

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.bind_object(scene, first_id, second, expected_stamp=before)
    assert character_metadata.character_stamp(scene, first_id) == before

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.bind_object(scene, first_id, read_only, expected_stamp=before)
    assert character_metadata.character_stamp(scene, first_id) == before
    assert character_metadata.read_character(scene, second_id) is not None


def test_v2_unbind_last_object_binding_removes_orphan_owner_and_allows_reassignment():
    first = FakeObject("A")
    second = FakeObject("B")
    scene = FakeScene((first, second), schema_version=2)
    first_id = character_metadata.create_character(scene, "First", (first,))
    binding_id = character_metadata.bind_object(scene, first_id, second)

    character_metadata.remove_character_binding(scene, first_id, binding_id)
    first_definition = character_metadata.read_character(scene, first_id)
    assert first_definition is not None
    assert [owner.label_hint for owner in first_definition.owners] == ["A"]

    second_id = character_metadata.create_character(scene, "Second", (second,))
    second_definition = character_metadata.read_character(scene, second_id)
    assert second_definition is not None
    assert [owner.label_hint for owner in second_definition.owners] == ["B"]


def test_v2_unbind_referenced_binding_rejects_before_partial_mutation():
    first = FakeObject("A")
    second = FakeObject("B")
    scene = FakeScene((first, second), schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", (first, second))
    definition = character_metadata.read_character(scene, character_id)
    assert definition is not None
    by_label = {
        owner.label_hint: next(
            binding.binding_id
            for binding in definition.bindings
            if binding.owner_id == owner.owner_id
        )
        for owner in definition.owners
    }
    character_metadata.add_character_group(
        scene,
        character_id,
        semantic_key="custom.test",
        label="Test",
        members=(by_label["B"],),
    )
    before = character_metadata.character_stamp(scene, character_id)

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.remove_character_binding(scene, character_id, by_label["B"])

    assert character_metadata.character_stamp(scene, character_id) == before


def test_unknown_schema_is_preserved_and_write_rejected():
    obj = FakeObject("A")
    scene = FakeScene((obj,), schema_version=99)
    marker = scene.awb_characters.characters.add()
    marker.character_id = "future-data"

    with pytest.raises(character_metadata.UnsupportedCharacterSchema):
        character_metadata.read_character(scene, "future-data")
    with pytest.raises(character_metadata.UnsupportedCharacterSchema):
        character_metadata.create_character(scene, "Hero", (obj,))

    assert len(scene.awb_characters.characters) == 1
    assert scene.awb_characters.characters[0].character_id == "future-data"


def test_stale_remove_stamp_rejects_without_mutation():
    obj = FakeObject("A")
    scene = FakeScene((obj,))
    character_id = character_metadata.create_character(scene, "Hero", (obj,))
    stale = character_metadata.character_stamp(scene, character_id)
    scene.awb_characters.characters[0].revision += 1

    with pytest.raises(character_metadata.StaleCharacterDefinition):
        character_metadata.remove_character(scene, character_id, expected_stamp=stale)

    assert len(scene.awb_characters.characters) == 1


def test_remove_with_current_stamp_removes_metadata_only():
    obj = FakeObject("A")
    scene = FakeScene((obj,))
    character_id = character_metadata.create_character(scene, "Hero", (obj,))
    stamp = character_metadata.character_stamp(scene, character_id)

    character_metadata.remove_character(scene, character_id, expected_stamp=stamp)

    assert scene.awb_characters.characters == []
    assert scene.objects["A"] is obj


def test_bind_pose_bone_issues_token_and_adds_bone_binding():
    rig, pose_bones = make_fake_rig()
    scene = FakeScene((rig,))
    character_id = character_metadata.create_character(scene, "Hero", (rig,))
    before = character_metadata.character_stamp(scene, character_id)

    binding_id = character_metadata.bind_pose_bone(
        scene,
        character_id,
        pose_bones["Hand.L"],
        expected_stamp=before,
    )
    definition = character_metadata.read_character(scene, character_id)

    assert definition is not None
    bone_bindings = [item for item in definition.bindings if item.kind == semantic_model.AWBControlKind.BONE]
    assert len(bone_bindings) == 1
    assert bone_bindings[0].binding_id == binding_id
    assert bone_bindings[0].bone_name_hint == "Hand.L"
    assert bone_bindings[0].bone_id == rig.data.bones[0][character_metadata.BONE_TOKEN_PROPERTY]
    assert definition.revision == 2


def test_bone_binding_requires_existing_character_owner_object():
    rig, pose_bones = make_fake_rig()
    other = FakeObject("Other")
    scene = FakeScene((rig, other))
    character_id = character_metadata.create_character(scene, "Hero", (other,))

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.bind_pose_bone(scene, character_id, pose_bones["Hand.L"])

    assert character_metadata.BONE_TOKEN_PROPERTY not in rig.data.bones[0]


def test_linked_armature_data_rejects_before_token_write():
    rig, pose_bones = make_fake_rig(linked_data=True)
    scene = FakeScene((rig,))
    character_id = character_metadata.create_character(scene, "Hero", (rig,))

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.bind_pose_bone(scene, character_id, pose_bones["Hand.L"])

    assert character_metadata.BONE_TOKEN_PROPERTY not in rig.data.bones[0]


def test_duplicate_bone_token_is_ambiguous_and_repair_reissues_selected_target():
    rig, pose_bones = make_fake_rig(bone_names=("Hand.L", "HandCopy.L"))
    scene = FakeScene((rig,))
    character_id = character_metadata.create_character(scene, "Hero", (rig,))
    binding_id = character_metadata.bind_pose_bone(scene, character_id, pose_bones["Hand.L"])
    original_token = rig.data.bones[0][character_metadata.BONE_TOKEN_PROPERTY]
    rig.data.bones[1][character_metadata.BONE_TOKEN_PROPERTY] = original_token

    assert len(character_metadata.bone_token_matches(rig, original_token)) == 2
    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.rebind_pose_bone(
            scene,
            character_id,
            binding_id,
            pose_bones["HandCopy.L"],
        )

    repaired_token = character_metadata.rebind_pose_bone(
        scene,
        character_id,
        binding_id,
        pose_bones["HandCopy.L"],
        repair_duplicate_token=True,
    )
    assert repaired_token != original_token
    assert len(character_metadata.bone_token_matches(rig, original_token)) == 1
    assert len(character_metadata.bone_token_matches(rig, repaired_token)) == 1
    definition = character_metadata.read_character(scene, character_id)
    assert definition is not None
    rebound = next(item for item in definition.bindings if item.binding_id == binding_id)
    assert rebound.bone_id == repaired_token
    assert rebound.bone_name_hint == "HandCopy.L"


def test_duplicate_token_repair_rejects_if_same_character_other_binding_uses_token():
    rig, pose_bones = make_fake_rig(bone_names=("Hand.L", "HandCopy.L"))
    scene = FakeScene((rig,))
    character_id = character_metadata.create_character(scene, "Hero", (rig,))
    original_binding_id = character_metadata.bind_pose_bone(
        scene,
        character_id,
        pose_bones["Hand.L"],
    )
    row = scene.awb_characters.characters[0]
    original_binding = next(
        binding for binding in row.bindings if binding.binding_id == original_binding_id
    )
    original_token = original_binding.bone_id
    rig.data.bones[1][character_metadata.BONE_TOKEN_PROPERTY] = original_token

    missing = row.bindings.add()
    missing.binding_id = "missing-binding"
    missing.owner_id = original_binding.owner_id
    missing.kind = semantic_model.AWBControlKind.BONE.value
    missing.bone_id = "missing-token"
    missing.bone_name_hint = "Missing"
    before_revision = row.revision

    with pytest.raises(
        character_metadata.CharacterMetadataError,
        match="would retarget another Character binding",
    ):
        character_metadata.rebind_pose_bone(
            scene,
            character_id,
            missing.binding_id,
            pose_bones["Hand.L"],
            repair_duplicate_token=True,
        )

    assert row.revision == before_revision
    assert missing.bone_id == "missing-token"
    assert rig.data.bones[0][character_metadata.BONE_TOKEN_PROPERTY] == original_token
    assert rig.data.bones[1][character_metadata.BONE_TOKEN_PROPERTY] == original_token


def test_duplicate_token_repair_rejects_cross_scene_reference(monkeypatch):
    rig, pose_bones = make_fake_rig(bone_names=("Hand.L", "HandCopy.L"))
    scene_a = FakeScene((rig,))
    scene_b = FakeScene((rig,))
    monkeypatch.setattr(bpy.data, "scenes", [scene_a, scene_b])

    character_a = character_metadata.create_character(scene_a, "HeroA", (rig,))
    character_b = character_metadata.create_character(scene_b, "HeroB", (rig,))
    binding_a = character_metadata.bind_pose_bone(
        scene_a,
        character_a,
        pose_bones["Hand.L"],
    )
    binding_b = character_metadata.bind_pose_bone(
        scene_b,
        character_b,
        pose_bones["Hand.L"],
    )
    original_token = rig.data.bones[0][character_metadata.BONE_TOKEN_PROPERTY]
    rig.data.bones[1][character_metadata.BONE_TOKEN_PROPERTY] = original_token
    before_a = character_metadata.character_stamp(scene_a, character_a)
    before_b = character_metadata.character_stamp(scene_b, character_b)

    with pytest.raises(
        character_metadata.CharacterMetadataError,
        match="would retarget another Character binding",
    ):
        character_metadata.rebind_pose_bone(
            scene_a,
            character_a,
            binding_a,
            pose_bones["HandCopy.L"],
            repair_duplicate_token=True,
        )

    assert character_metadata.character_stamp(scene_a, character_a) == before_a
    assert character_metadata.character_stamp(scene_b, character_b) == before_b
    assert rig.data.bones[0][character_metadata.BONE_TOKEN_PROPERTY] == original_token
    assert rig.data.bones[1][character_metadata.BONE_TOKEN_PROPERTY] == original_token
    assert character_metadata.read_character(scene_b, character_b) is not None
    assert any(
        binding.binding_id == binding_b and binding.bone_id == original_token
        for binding in character_metadata.read_character(scene_b, character_b).bindings
    )


def test_shared_armature_data_is_disambiguated_by_character_owner_object():
    rig_a, pose_a = make_fake_rig(name="RigA")
    shared_data = rig_a.data
    rig_b = FakeObject("RigB", object_type="ARMATURE", data=shared_data)
    pose_bones_b = FakePoseBones(
        {bone.name: FakePoseBone(rig_b, bone) for bone in shared_data.bones}
    )
    rig_b.pose = SimpleNamespace(bones=pose_bones_b)
    scene = FakeScene((rig_a, rig_b))
    character_id = character_metadata.create_character(scene, "Hero", (rig_a, rig_b))

    first_binding = character_metadata.bind_pose_bone(scene, character_id, pose_a["Hand.L"])
    second_binding = character_metadata.bind_pose_bone(scene, character_id, pose_bones_b["Hand.L"])
    definition = character_metadata.read_character(scene, character_id)

    assert definition is not None
    by_id = {item.binding_id: item for item in definition.bindings}
    assert by_id[first_binding].bone_id == by_id[second_binding].bone_id
    assert by_id[first_binding].owner_id != by_id[second_binding].owner_id


def test_remove_bone_binding_preserves_native_bone_token():
    rig, pose_bones = make_fake_rig()
    scene = FakeScene((rig,))
    character_id = character_metadata.create_character(scene, "Hero", (rig,))
    binding_id = character_metadata.bind_pose_bone(scene, character_id, pose_bones["Hand.L"])
    token = rig.data.bones[0][character_metadata.BONE_TOKEN_PROPERTY]

    character_metadata.remove_character_binding(scene, character_id, binding_id)
    definition = character_metadata.read_character(scene, character_id)

    assert definition is not None
    assert all(item.binding_id != binding_id for item in definition.bindings)
    assert rig.data.bones[0][character_metadata.BONE_TOKEN_PROPERTY] == token


def test_explicit_v1_to_v2_migration_preserves_membership_and_uses_neutral_defaults():
    first = FakeObject("A")
    second = FakeObject("B")
    scene = FakeScene((first, second), schema_version=1)
    character_id = character_metadata.create_character(scene, "Hero", (first, second))
    before = character_metadata.read_character(scene, character_id)
    assert before is not None
    assert before.revision == 1
    assert scene.awb_characters.schema_version == 1

    migrated = character_metadata.migrate_store_v1_to_v2(scene)
    after = character_metadata.read_character(scene, character_id)

    assert migrated == (character_id,)
    assert scene.awb_characters.schema_version == 2
    assert after is not None
    assert after.revision == 2
    assert [binding.binding_id for binding in after.bindings] == [
        binding.binding_id for binding in before.bindings
    ]
    assert all(binding.semantic_key == "" for binding in after.bindings)
    assert all(binding.side == character_model.CharacterSide.NONE for binding in after.bindings)
    assert all(binding.mode == character_model.ControlMode.NEUTRAL for binding in after.bindings)
    assert all(binding.usage == character_model.ControlUsage.PRIMARY for binding in after.bindings)
    assert after.groups == ()
    assert after.chains == ()
    assert after.opposites == ()


def test_v1_migration_refuses_preexisting_semantic_data_instead_of_guessing():
    obj = FakeObject("A")
    scene = FakeScene((obj,), schema_version=1)
    character_id = character_metadata.create_character(scene, "Hero", (obj,))
    scene.awb_characters.characters[0].bindings[0].semantic_key = "awb.hand"

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.migrate_store_v1_to_v2(scene)

    assert scene.awb_characters.schema_version == 1
    assert character_metadata.read_character(scene, character_id) is not None


def test_v2_binding_semantics_preserve_unknown_custom_namespace_and_reject_unknown_awb_key():
    obj = FakeObject("A")
    scene = FakeScene((obj,), schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", (obj,))
    binding_id = scene.awb_characters.characters[0].bindings[0].binding_id

    character_metadata.set_binding_semantics(
        scene,
        character_id,
        binding_id,
        semantic_key="studio.hero.cape",
        side=character_model.CharacterSide.LEFT,
        mode=character_model.ControlMode.SHARED,
        usage=character_model.ControlUsage.HELPER,
    )
    definition = character_metadata.read_character(scene, character_id)
    assert definition is not None
    binding = definition.bindings[0]
    assert binding.semantic_key == "studio.hero.cape"
    assert binding.side == character_model.CharacterSide.LEFT
    assert binding.mode == character_model.ControlMode.SHARED
    assert binding.usage == character_model.ControlUsage.HELPER

    stamp = character_metadata.character_stamp(scene, character_id)
    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.set_binding_semantics(
            scene,
            character_id,
            binding_id,
            semantic_key="awb.not_defined",
            expected_stamp=stamp,
        )
    assert character_metadata.character_stamp(scene, character_id) == stamp


def test_v2_group_chain_and_opposite_roundtrip_preserves_explicit_membership_and_order():
    left = FakeObject("Left")
    center = FakeObject("Center")
    right = FakeObject("Right")
    scene = FakeScene((left, center, right), schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", (left, center, right))
    definition = character_metadata.read_character(scene, character_id)
    assert definition is not None
    by_label = {
        owner.label_hint: next(
            binding for binding in definition.bindings if binding.owner_id == owner.owner_id
        )
        for owner in definition.owners
    }

    character_metadata.set_binding_semantics(
        scene,
        character_id,
        by_label["Left"].binding_id,
        semantic_key="awb.hand",
        side="LEFT",
    )
    character_metadata.set_binding_semantics(
        scene,
        character_id,
        by_label["Center"].binding_id,
        semantic_key="custom.tail",
        side="CENTER",
    )
    character_metadata.set_binding_semantics(
        scene,
        character_id,
        by_label["Right"].binding_id,
        semantic_key="awb.hand",
        side="RIGHT",
    )

    group_id = character_metadata.add_character_group(
        scene,
        character_id,
        semantic_key="custom.hand_controls",
        label="Hands",
        members=(by_label["Right"].binding_id, by_label["Left"].binding_id),
    )
    chain_order = (
        by_label["Left"].binding_id,
        by_label["Center"].binding_id,
        by_label["Right"].binding_id,
    )
    chain_id = character_metadata.add_character_chain(
        scene,
        character_id,
        semantic_key="custom.tail",
        label="Custom Ordered Chain",
        members=chain_order,
    )
    character_metadata.add_opposite_pair(
        scene,
        character_id,
        by_label["Left"].binding_id,
        by_label["Right"].binding_id,
    )

    result = character_metadata.read_character(scene, character_id)
    assert result is not None
    assert result.groups[0].group_id == group_id
    assert result.groups[0].members == (
        by_label["Right"].binding_id,
        by_label["Left"].binding_id,
    )
    assert result.chains[0].chain_id == chain_id
    assert result.chains[0].members == chain_order
    assert result.chains[0].semantic_key == "custom.tail"
    assert result.opposites == (
        character_model.OppositePair(
            by_label["Left"].binding_id,
            by_label["Right"].binding_id,
        ),
    )


def test_v2_invalid_chain_is_rejected_without_partial_rna_mutation():
    obj = FakeObject("A")
    scene = FakeScene((obj,), schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", (obj,))
    binding_id = scene.awb_characters.characters[0].bindings[0].binding_id
    before = character_metadata.character_stamp(scene, character_id)

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.add_character_chain(
            scene,
            character_id,
            semantic_key="custom.tail",
            label="Broken",
            members=(binding_id, binding_id, "missing"),
            expected_stamp=before,
        )

    assert character_metadata.character_stamp(scene, character_id) == before
    assert scene.awb_characters.characters[0].chains == []


def test_v2_group_edit_and_remove_preflight_preserve_graph_integrity():
    objects = tuple(FakeObject(name) for name in ("A", "B", "C"))
    scene = FakeScene(objects, schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", objects)
    definition = character_metadata.read_character(scene, character_id)
    assert definition is not None
    by_label = {
        owner.label_hint: next(
            binding for binding in definition.bindings if binding.owner_id == owner.owner_id
        )
        for owner in definition.owners
    }
    parent_id = character_metadata.add_character_group(
        scene,
        character_id,
        semantic_key="custom.controls",
        label="Parent",
        members=(by_label["A"].binding_id, by_label["B"].binding_id),
    )
    child_id = character_metadata.add_character_group(
        scene,
        character_id,
        semantic_key="custom.controls.child",
        label="Child",
        members=(by_label["C"].binding_id,),
        parent_group_id=parent_id,
    )

    character_metadata.set_character_group(
        scene,
        character_id,
        parent_id,
        semantic_key="custom.controls.edited",
        label="Edited Parent",
        side="LEFT",
        members=(by_label["C"].binding_id, by_label["A"].binding_id),
    )
    edited = character_metadata.read_character(scene, character_id)
    assert edited is not None
    parent = next(group for group in edited.groups if group.group_id == parent_id)
    assert parent.label == "Edited Parent"
    assert parent.semantic_key == "custom.controls.edited"
    assert parent.side == character_model.CharacterSide.LEFT
    assert parent.members == (by_label["C"].binding_id, by_label["A"].binding_id)

    before = character_metadata.character_stamp(scene, character_id)
    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.remove_character_group(
            scene,
            character_id,
            parent_id,
            expected_stamp=before,
        )
    assert character_metadata.character_stamp(scene, character_id) == before

    character_metadata.remove_character_group(scene, character_id, child_id)
    character_metadata.remove_character_group(scene, character_id, parent_id)
    result = character_metadata.read_character(scene, character_id)
    assert result is not None
    assert result.groups == ()


def test_v2_chain_edit_order_is_authoritative_and_referenced_remove_fails_closed():
    objects = tuple(FakeObject(name) for name in ("A", "B", "C"))
    scene = FakeScene(objects, schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", objects)
    definition = character_metadata.read_character(scene, character_id)
    assert definition is not None
    by_label = {
        owner.label_hint: next(
            binding for binding in definition.bindings if binding.owner_id == owner.owner_id
        )
        for owner in definition.owners
    }
    chain_id = character_metadata.add_character_chain(
        scene,
        character_id,
        semantic_key="custom.chain",
        label="Initial",
        members=(
            by_label["A"].binding_id,
            by_label["B"].binding_id,
            by_label["C"].binding_id,
        ),
    )
    authored_order = (
        by_label["C"].binding_id,
        by_label["A"].binding_id,
        by_label["B"].binding_id,
    )
    character_metadata.set_character_chain(
        scene,
        character_id,
        chain_id,
        semantic_key="custom.chain.edited",
        label="Authored Order",
        side="RIGHT",
        mode="FK",
        members=authored_order,
    )
    edited = character_metadata.read_character(scene, character_id)
    assert edited is not None
    chain = edited.chains[0]
    assert chain.members == authored_order
    assert chain.label == "Authored Order"
    assert chain.side == character_model.CharacterSide.RIGHT
    assert chain.mode == character_model.ControlMode.FK

    character_metadata.add_kinematic_mapping(
        scene,
        character_id,
        semantic_key="custom.chain.edited",
        side="RIGHT",
        fk_chain_id=chain_id,
    )
    before = character_metadata.character_stamp(scene, character_id)
    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.remove_character_chain(
            scene,
            character_id,
            chain_id,
            expected_stamp=before,
        )
    assert character_metadata.character_stamp(scene, character_id) == before


def test_v2_group_and_chain_edit_reject_missing_member_without_partial_mutation():
    obj = FakeObject("A")
    scene = FakeScene((obj,), schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", (obj,))
    binding_id = scene.awb_characters.characters[0].bindings[0].binding_id
    group_id = character_metadata.add_character_group(
        scene,
        character_id,
        semantic_key="custom.group",
        label="Group",
        members=(binding_id,),
    )
    chain_id = character_metadata.add_character_chain(
        scene,
        character_id,
        semantic_key="custom.chain",
        label="Chain",
        members=(binding_id,),
    )
    before = character_metadata.character_stamp(scene, character_id)

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.set_character_group(
            scene,
            character_id,
            group_id,
            semantic_key="custom.group",
            label="Broken",
            members=("missing",),
            expected_stamp=before,
        )
    assert character_metadata.character_stamp(scene, character_id) == before

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.set_character_chain(
            scene,
            character_id,
            chain_id,
            semantic_key="custom.chain",
            label="Broken",
            members=("missing",),
            expected_stamp=before,
        )
    assert character_metadata.character_stamp(scene, character_id) == before


def test_v2_opposite_pair_remove_roundtrip_is_explicit_and_symmetric():
    left = FakeObject("Left")
    right = FakeObject("Right")
    scene = FakeScene((left, right), schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", (left, right))
    definition = character_metadata.read_character(scene, character_id)
    assert definition is not None
    by_label = {
        owner.label_hint: next(
            binding for binding in definition.bindings if binding.owner_id == owner.owner_id
        )
        for owner in definition.owners
    }
    character_metadata.set_binding_semantics(
        scene,
        character_id,
        by_label["Left"].binding_id,
        semantic_key="awb.hand",
        side="LEFT",
    )
    character_metadata.set_binding_semantics(
        scene,
        character_id,
        by_label["Right"].binding_id,
        semantic_key="awb.hand",
        side="RIGHT",
    )
    character_metadata.add_opposite_pair(
        scene,
        character_id,
        by_label["Left"].binding_id,
        by_label["Right"].binding_id,
    )
    result = character_metadata.read_character(scene, character_id)
    assert result is not None
    assert result.opposites == (
        character_model.OppositePair(
            by_label["Left"].binding_id,
            by_label["Right"].binding_id,
        ),
    )

    character_metadata.remove_opposite_pair(
        scene,
        character_id,
        by_label["Right"].binding_id,
        by_label["Left"].binding_id,
    )
    result = character_metadata.read_character(scene, character_id)
    assert result is not None
    assert result.opposites == ()


def test_v2_kinematic_mapping_edit_preflights_then_remove_roundtrips():
    objects = tuple(FakeObject(name) for name in ("FK0", "FK1", "IK", "Pole", "Ref", "Extra"))
    scene = FakeScene(objects, schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", objects)
    definition = character_metadata.read_character(scene, character_id)
    assert definition is not None
    by_label = {
        owner.label_hint: next(
            binding for binding in definition.bindings if binding.owner_id == owner.owner_id
        )
        for owner in definition.owners
    }
    fk_chain = character_metadata.add_character_chain(
        scene,
        character_id,
        semantic_key="custom.wing",
        label="Wing FK",
        side="LEFT",
        mode="FK",
        members=(by_label["FK0"].binding_id, by_label["FK1"].binding_id),
    )
    ref_chain = character_metadata.add_character_chain(
        scene,
        character_id,
        semantic_key="custom.wing",
        label="Wing Ref",
        side="LEFT",
        members=(by_label["Ref"].binding_id,),
    )
    mapping_id = character_metadata.add_kinematic_mapping(
        scene,
        character_id,
        semantic_key="custom.wing",
        side="LEFT",
        fk_chain_id=fk_chain,
    )

    character_metadata.set_kinematic_mapping(
        scene,
        character_id,
        mapping_id,
        semantic_key="custom.wing.edited",
        side="RIGHT",
        fk_chain_id=fk_chain,
        ik_target_binding_id=by_label["IK"].binding_id,
        pole_binding_id=by_label["Pole"].binding_id,
        reference_chain_id=ref_chain,
        extras=(by_label["Extra"].binding_id,),
    )
    result = character_metadata.read_character(scene, character_id)
    assert result is not None
    mapping = result.kinematics[0]
    assert mapping.mapping_id == mapping_id
    assert mapping.semantic_key == "custom.wing.edited"
    assert mapping.side == character_model.CharacterSide.RIGHT
    assert mapping.fk_chain_id == fk_chain
    assert mapping.ik_target_binding_id == by_label["IK"].binding_id
    assert mapping.pole_binding_id == by_label["Pole"].binding_id
    assert mapping.reference_chain_id == ref_chain
    assert mapping.extras == (by_label["Extra"].binding_id,)

    before = character_metadata.character_stamp(scene, character_id)
    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.set_kinematic_mapping(
            scene,
            character_id,
            mapping_id,
            semantic_key="custom.wing.edited",
            fk_chain_id="missing-chain",
            expected_stamp=before,
        )
    assert character_metadata.character_stamp(scene, character_id) == before

    character_metadata.remove_kinematic_mapping(scene, character_id, mapping_id)
    result = character_metadata.read_character(scene, character_id)
    assert result is not None
    assert result.kinematics == ()


def test_v2_clone_character_explicit_owner_map_preserves_full_logical_graph_without_source_mutation():
    source_left = FakeObject("SourceLeft")
    source_right = FakeObject("SourceRight")
    target_left = FakeObject("TargetLeft")
    target_right = FakeObject("TargetRight")
    scene = FakeScene((source_left, source_right, target_left, target_right), schema_version=2)
    source_id = character_metadata.create_character(scene, "Source", (source_left, source_right))
    source = character_metadata.read_character(scene, source_id)
    assert source is not None
    source_owner_by_label = {owner.label_hint: owner for owner in source.owners}
    source_binding_by_owner = {
        binding.owner_id: binding for binding in source.bindings
    }
    left_binding = source_binding_by_owner[source_owner_by_label["SourceLeft"].owner_id]
    right_binding = source_binding_by_owner[source_owner_by_label["SourceRight"].owner_id]
    character_metadata.set_binding_semantics(
        scene,
        source_id,
        left_binding.binding_id,
        semantic_key="awb.hand",
        side="LEFT",
    )
    character_metadata.set_binding_semantics(
        scene,
        source_id,
        right_binding.binding_id,
        semantic_key="awb.hand",
        side="RIGHT",
    )
    group_id = character_metadata.add_character_group(
        scene,
        source_id,
        semantic_key="custom.hands",
        label="Hands",
        members=(left_binding.binding_id, right_binding.binding_id),
    )
    chain_id = character_metadata.add_character_chain(
        scene,
        source_id,
        semantic_key="custom.arm",
        label="Arm",
        side="LEFT",
        mode="FK",
        members=(left_binding.binding_id, right_binding.binding_id),
    )
    character_metadata.add_opposite_pair(
        scene,
        source_id,
        left_binding.binding_id,
        right_binding.binding_id,
    )
    character_metadata.add_kinematic_mapping(
        scene,
        source_id,
        semantic_key="custom.arm",
        side="LEFT",
        fk_chain_id=chain_id,
        ik_target_binding_id=right_binding.binding_id,
        extras=(left_binding.binding_id,),
    )
    source_before = character_metadata.read_character(scene, source_id)
    source_stamp = character_metadata.character_stamp(scene, source_id)
    assert source_before is not None

    target_id = character_metadata.clone_character(
        scene,
        source_id,
        {
            source_owner_by_label["SourceLeft"].owner_id: target_left,
            source_owner_by_label["SourceRight"].owner_id: target_right,
        },
        label="Clone",
        expected_stamp=source_stamp,
    )

    assert character_metadata.character_stamp(scene, source_id) == source_stamp
    assert character_metadata.read_character(scene, source_id) == source_before
    clone = character_metadata.read_character(scene, target_id)
    assert clone is not None
    assert clone.character_id != source_id
    assert clone.label == "Clone"
    assert clone.revision == 1
    assert [owner.label_hint for owner in clone.owners] == ["TargetLeft", "TargetRight"]
    assert {binding.binding_id for binding in clone.bindings}.isdisjoint(
        {binding.binding_id for binding in source_before.bindings}
    )
    assert [binding.semantic_key for binding in clone.bindings] == [
        binding.semantic_key for binding in source_before.bindings
    ]
    assert [binding.side for binding in clone.bindings] == [
        binding.side for binding in source_before.bindings
    ]
    assert len(clone.groups) == 1
    assert clone.groups[0].group_id != group_id
    assert len(clone.chains) == 1
    assert clone.chains[0].chain_id != chain_id
    assert len(clone.opposites) == 1
    assert len(clone.kinematics) == 1
    assert clone.kinematics[0].fk_chain_id == clone.chains[0].chain_id
    assert clone.kinematics[0].ik_target_binding_id in {
        binding.binding_id for binding in clone.bindings
    }
    assert clone.kinematics[0].extras[0] in {
        binding.binding_id for binding in clone.bindings
    }


def test_v2_clone_character_owner_map_preflight_rejects_incomplete_map_without_partial_character():
    source = FakeObject("Source")
    target = FakeObject("Target")
    scene = FakeScene((source, target), schema_version=2)
    source_id = character_metadata.create_character(scene, "Source", (source,))
    before_count = len(scene.awb_characters.characters)
    before_stamp = character_metadata.character_stamp(scene, source_id)

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.clone_character(
            scene,
            source_id,
            {},
            expected_stamp=before_stamp,
        )

    assert len(scene.awb_characters.characters) == before_count
    assert character_metadata.character_stamp(scene, source_id) == before_stamp


def test_v2_generic_kinematic_mapping_roundtrip_preserves_custom_semantic_and_optional_fields():
    objects = tuple(FakeObject(name) for name in ("FK0", "FK1", "IK", "Pole", "Ref", "Extra"))
    scene = FakeScene(objects, schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", objects)
    definition = character_metadata.read_character(scene, character_id)
    assert definition is not None
    by_label = {
        owner.label_hint: next(
            binding for binding in definition.bindings if binding.owner_id == owner.owner_id
        )
        for owner in definition.owners
    }

    fk_chain = character_metadata.add_character_chain(
        scene,
        character_id,
        semantic_key="custom.wing",
        label="Wing FK",
        side="LEFT",
        mode="FK",
        members=(by_label["FK0"].binding_id, by_label["FK1"].binding_id),
    )
    reference_chain = character_metadata.add_character_chain(
        scene,
        character_id,
        semantic_key="custom.wing",
        label="Wing Reference",
        side="LEFT",
        members=(by_label["Ref"].binding_id,),
    )
    mapping_id = character_metadata.add_kinematic_mapping(
        scene,
        character_id,
        semantic_key="custom.wing",
        side="LEFT",
        fk_chain_id=fk_chain,
        ik_target_binding_id=by_label["IK"].binding_id,
        pole_binding_id=by_label["Pole"].binding_id,
        reference_chain_id=reference_chain,
        extras=(by_label["Extra"].binding_id,),
    )

    result = character_metadata.read_character(scene, character_id)
    assert result is not None
    assert result.kinematics == (
        character_model.KinematicMapping(
            mapping_id=mapping_id,
            semantic_key="custom.wing",
            side=character_model.CharacterSide.LEFT,
            fk_chain_id=fk_chain,
            ik_target_binding_id=by_label["IK"].binding_id,
            pole_binding_id=by_label["Pole"].binding_id,
            reference_chain_id=reference_chain,
            extras=(by_label["Extra"].binding_id,),
        ),
    )


def test_v2_invalid_kinematic_mapping_is_rejected_without_partial_rna_mutation():
    obj = FakeObject("A")
    scene = FakeScene((obj,), schema_version=2)
    character_id = character_metadata.create_character(scene, "Hero", (obj,))
    before = character_metadata.character_stamp(scene, character_id)

    with pytest.raises(character_metadata.CharacterMetadataError):
        character_metadata.add_kinematic_mapping(
            scene,
            character_id,
            semantic_key="awb.arm",
            fk_chain_id="missing-chain",
            expected_stamp=before,
        )

    assert character_metadata.character_stamp(scene, character_id) == before
    assert scene.awb_characters.characters[0].kinematics == []


def test_shared_armature_rejects_new_bone_token_write_without_mutation():
    rig, pose_bones = make_fake_rig("SharedRig")
    rig.data.users = 2
    bone = pose_bones["Hand.L"].bone

    with pytest.raises(character_metadata.CharacterMetadataError, match="single-user"):
        character_metadata.issue_bone_token(pose_bones["Hand.L"])

    assert character_metadata.BONE_TOKEN_PROPERTY not in bone


def test_shared_armature_can_reuse_existing_bone_token_without_mutation():
    rig, pose_bones = make_fake_rig("SharedRigExisting")
    rig.data.users = 2
    bone = pose_bones["Hand.L"].bone
    bone[character_metadata.BONE_TOKEN_PROPERTY] = "stable-token"

    assert character_metadata.issue_bone_token(pose_bones["Hand.L"]) == "stable-token"
    assert bone[character_metadata.BONE_TOKEN_PROPERTY] == "stable-token"


def test_lifecycle_diagnostics_report_shared_linked_and_unknown_schema_without_mutation():
    rig, pose_bones = make_fake_rig("LifecycleRig")
    scene = FakeScene((rig,), schema_version=2)
    character_id = character_metadata.create_character(scene, "Lifecycle", (rig,))
    pose_bones["Hand.L"].bone[character_metadata.BONE_TOKEN_PROPERTY] = "stable-token"
    character_metadata.bind_pose_bone(scene, character_id, pose_bones["Hand.L"])
    rig.data.users = 2

    shared_codes = {issue.code for issue in character_metadata.character_lifecycle_issues(scene)}
    assert "SHARED_ARMATURE_DATA_TOKEN_WRITE_BLOCKED" in shared_codes

    rig.library = object()
    linked_codes = {issue.code for issue in character_metadata.character_lifecycle_issues(scene)}
    assert "LINKED_OWNER_READ_ONLY" in linked_codes

    scene.awb_characters.schema_version = 99
    unknown = character_metadata.character_lifecycle_issues(scene)
    assert [issue.code for issue in unknown] == ["UNSUPPORTED_CHARACTER_SCHEMA"]


def test_read_only_scene_rejects_character_creation_before_store_mutation():
    obj = FakeObject("ReadOnlyTarget")
    scene = FakeScene((obj,), schema_version=2)
    scene.library = object()
    before = len(scene.awb_characters.characters)

    with pytest.raises(character_metadata.CharacterMetadataError, match="Scene"):
        character_metadata.create_character(scene, "Blocked", (obj,))

    assert len(scene.awb_characters.characters) == before
    assert "READ_ONLY_CHARACTER_SCENE" in {
        issue.code for issue in character_metadata.character_lifecycle_issues(scene)
    }


def test_read_only_scene_rejects_v1_migration_without_partial_mutation():
    obj = FakeObject("MigrationTarget")
    scene = FakeScene((obj,), schema_version=1)
    character_id = character_metadata.create_character(scene, "Legacy", (obj,))
    before = character_metadata.character_stamp(scene, character_id)
    scene.is_editable = False

    with pytest.raises(character_metadata.CharacterMetadataError, match="Scene"):
        character_metadata.migrate_store_v1_to_v2(scene)

    assert scene.awb_characters.schema_version == 1
    assert character_metadata.character_stamp(scene, character_id) == before


def test_detached_owner_pointer_is_diagnosed_as_missing_scene_target():
    obj = FakeObject("DetachedTarget")
    scene = FakeScene((obj,), schema_version=2)
    character_metadata.create_character(scene, "Detached", (obj,))
    del scene.objects[obj.name]

    codes = {issue.code for issue in character_metadata.character_lifecycle_issues(scene)}

    assert "MISSING_OBJECT_TARGET" in codes


def test_same_name_impostor_does_not_satisfy_scene_membership_identity():
    original = FakeObject("Rig")
    scene = FakeScene((original,), schema_version=2)
    character_metadata.create_character(scene, "Hero", (original,))
    impostor = FakeObject("Rig")
    scene.objects[original.name] = impostor

    codes = {issue.code for issue in character_metadata.character_lifecycle_issues(scene)}

    assert "MISSING_OBJECT_TARGET" in codes
    assert not character_metadata._scene_contains_object_identity(scene, original)
    assert character_metadata._scene_contains_object_identity(scene, impostor)
