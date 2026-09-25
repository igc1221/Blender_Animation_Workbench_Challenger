import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

PACKAGE_PATH = (
    Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
)
PACKAGE_NAME = "baw_semantic_adapter_tests"

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


class FakeRNA:
    def __init__(self, name: str, pointer: int, properties: dict | None = None):
        self.name = name
        self._pointer = pointer
        self._properties = properties or {}

    def as_pointer(self) -> int:
        return self._pointer

    def get(self, key: str, default=None):
        return self._properties.get(key, default)


class FakeObject(FakeRNA):
    def __init__(
        self,
        name: str,
        pointer: int,
        *,
        object_type: str = "MESH",
        selected: bool = True,
        properties: dict | None = None,
    ):
        super().__init__(name, pointer, properties)
        self.type = object_type
        self.selected = selected

    def select_get(self) -> bool:
        return self.selected


class FakePoseBone(FakeRNA):
    def __init__(
        self,
        name: str,
        pointer: int,
        *,
        selected: bool = True,
        properties: dict | None = None,
        owner=None,
    ):
        super().__init__(name, pointer, properties)
        self.select = selected
        self.id_data = owner
        self.bone = FakeRNA(f"{name}.bone", pointer + 10_000)

    def path_from_id(self) -> str:
        return f'pose.bones["{self.name}"]'


def _context(
    *,
    mode: str,
    active_object=None,
    selected_objects=None,
    active_pose_bone=None,
    selected_pose_bones=None,
):
    return SimpleNamespace(
        mode=mode,
        active_object=active_object,
        selected_objects=selected_objects,
        active_pose_bone=active_pose_bone,
        selected_pose_bones=selected_pose_bones,
    )


def _resolved_signature(resolved):
    return (
        resolved.control.identity_key,
        resolved.owner_object.as_pointer(),
        resolved.target.as_pointer(),
        resolved.data_path_prefix,
    )


def test_runtime_control_key_uses_owner_and_target_runtime_pointers():
    owner = FakeObject("Rig", 101, object_type="ARMATURE")
    bone = FakePoseBone("Hand.L", 202)
    resolved = semantic_adapter.ResolvedControl(
        control=semantic_model.AWBControl.bone("Rig", "Hand.L"),
        owner_object=owner,
        target=bone,
        data_path_prefix=bone.path_from_id(),
    )

    assert semantic_adapter.runtime_control_key(resolved) == (101, 202)


def test_resolve_control_target_decodes_object_and_posebone_without_selection(monkeypatch):
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(Object=FakeObject, PoseBone=FakePoseBone)
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    obj = FakeObject("Cube", 301, selected=False)
    rig = FakeObject("Rig", 302, object_type="ARMATURE", selected=False)
    bone = FakePoseBone("Hand.L", 303, selected=False, owner=rig)

    object_resolved = semantic_adapter.resolve_control_target(obj)
    bone_resolved = semantic_adapter.resolve_control_target(bone)

    assert object_resolved is not None
    assert object_resolved.control.identity_key == (
        semantic_model.AWBControlKind.OBJECT,
        "Cube",
        None,
    )
    assert bone_resolved is not None
    assert bone_resolved.owner_object is rig
    assert bone_resolved.target is bone
    assert bone_resolved.data_path_prefix == bone.path_from_id()
    assert semantic_adapter.resolve_control_target(FakeRNA("Other", 304)) is None


def test_object_control_context_matches_existing_resolvers_and_order():
    active = FakeObject("Zulu", 101)
    other = FakeObject("Alpha", 102)
    context = _context(
        mode="OBJECT",
        active_object=active,
        selected_objects=[other, active],
    )

    expected = semantic_adapter.keying_controls_for_context(context)
    trackbar = semantic_adapter.trackbar_controls_for_context(context)
    control_context = semantic_adapter.control_context_for_context(context)

    assert [item.control.object_name for item in expected] == ["Zulu", "Alpha"]
    assert [_resolved_signature(item) for item in control_context.controls] == [
        _resolved_signature(item) for item in expected
    ]
    assert [_resolved_signature(item) for item in trackbar] == [
        _resolved_signature(item) for item in expected
    ]
    assert control_context.active is not None
    assert semantic_adapter.runtime_control_key(control_context.active) == (101, 101)


def test_object_active_but_deselected_is_not_added_to_control_context():
    active = FakeObject("Active", 101, selected=False)
    selected = FakeObject("Selected", 102)
    context = _context(
        mode="OBJECT",
        active_object=active,
        selected_objects=[selected],
    )

    control_context = semantic_adapter.control_context_for_context(context)

    assert [item.control.object_name for item in control_context.controls] == ["Selected"]
    assert control_context.active is None


def test_empty_object_context_stays_empty():
    active = FakeObject("Active", 101, selected=False)
    context = _context(mode="OBJECT", active_object=active, selected_objects=[])

    control_context = semantic_adapter.control_context_for_context(context)

    assert control_context.controls == ()
    assert control_context.active is None


def test_pose_control_context_matches_existing_resolvers_and_selected_order():
    rig = FakeObject("Rig", 101, object_type="ARMATURE")
    first = FakePoseBone("Bone.B", 201, owner=rig)
    second = FakePoseBone("Bone.A", 202, owner=rig)
    context = _context(
        mode="POSE",
        active_object=rig,
        active_pose_bone=second,
        selected_pose_bones=[first, second],
    )

    expected = semantic_adapter.keying_controls_for_context(context)
    trackbar = semantic_adapter.trackbar_controls_for_context(context)
    control_context = semantic_adapter.control_context_for_context(context)

    assert [item.control.bone_name for item in expected] == ["Bone.B", "Bone.A"]
    assert [_resolved_signature(item) for item in control_context.controls] == [
        _resolved_signature(item) for item in expected
    ]
    assert [_resolved_signature(item) for item in trackbar] == [
        _resolved_signature(item) for item in expected
    ]
    assert control_context.active is not None
    assert semantic_adapter.runtime_control_key(control_context.active) == (101, 202)


def test_multi_armature_pose_selection_keeps_each_bone_native_owner():
    rig_a = FakeObject("RigA", 110, object_type="ARMATURE")
    rig_b = FakeObject("RigB", 120, object_type="ARMATURE")
    bone_b = FakePoseBone("Root", 220, owner=rig_b)
    context = _context(
        mode="POSE",
        active_object=rig_a,
        active_pose_bone=None,
        selected_pose_bones=[bone_b],
    )

    controls = semantic_adapter.keying_controls_for_context(context)
    control_context = semantic_adapter.control_context_for_context(context)

    assert len(controls) == 1
    assert controls[0].owner_object is rig_b
    assert controls[0].target is bone_b
    assert controls[0].control.object_name == "RigB"
    assert semantic_adapter.runtime_control_key(controls[0]) == (120, 220)
    assert control_context.controls == tuple(controls)
    assert control_context.active is None


def test_pose_active_but_deselected_is_not_added_to_control_context():
    rig = FakeObject("Rig", 101, object_type="ARMATURE")
    active = FakePoseBone("Active", 201, selected=False, owner=rig)
    selected = FakePoseBone("Selected", 202, owner=rig)
    context = _context(
        mode="POSE",
        active_object=rig,
        active_pose_bone=active,
        selected_pose_bones=[selected],
    )

    control_context = semantic_adapter.control_context_for_context(context)

    assert [item.control.bone_name for item in control_context.controls] == ["Selected"]
    assert control_context.active is None


def test_empty_pose_and_unsupported_mode_contexts_stay_empty():
    rig = FakeObject("Rig", 101, object_type="ARMATURE")
    pose_context = _context(
        mode="POSE",
        active_object=rig,
        active_pose_bone=None,
        selected_pose_bones=[],
    )
    unsupported_context = _context(
        mode="EDIT_ARMATURE",
        active_object=rig,
        selected_objects=[rig],
    )

    pose_result = semantic_adapter.control_context_for_context(pose_context)
    unsupported_result = semantic_adapter.control_context_for_context(unsupported_context)

    assert pose_result.controls == ()
    assert pose_result.active is None
    assert unsupported_result.controls == ()
    assert unsupported_result.active is None


def test_role_metadata_does_not_change_runtime_or_descriptor_identity():
    plain = FakeObject("Cube", 101)
    role = FakeObject("Cube", 101, properties={"awb_role": "ROOT"})

    plain_resolved = semantic_adapter.keying_controls_for_context(
        _context(mode="OBJECT", active_object=plain, selected_objects=[plain])
    )[0]
    role_resolved = semantic_adapter.keying_controls_for_context(
        _context(mode="OBJECT", active_object=role, selected_objects=[role])
    )[0]

    assert plain_resolved.control.identity_key == role_resolved.control.identity_key
    assert semantic_adapter.runtime_control_key(plain_resolved) == semantic_adapter.runtime_control_key(
        role_resolved
    )
    assert role_resolved.control.display_name == "Root"


class CountingFCurves:
    def __init__(self, curves):
        self._curves = tuple(curves)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self._curves)


class FakeFCurve:
    def __init__(self, data_path: str, array_index: int):
        self.data_path = data_path
        self.array_index = array_index


def _install_fake_anim_utils(monkeypatch, channelbags_by_animation_data):
    package = ModuleType("bpy_extras")
    module = ModuleType("bpy_extras.anim_utils")

    def get_channelbag(animation_data):
        return channelbags_by_animation_data.get(id(animation_data))

    module.animdata_get_channelbag_for_assigned_slot = get_channelbag
    package.anim_utils = module
    monkeypatch.setitem(sys.modules, "bpy_extras", package)
    monkeypatch.setitem(sys.modules, "bpy_extras.anim_utils", module)


def test_assigned_channelbag_is_lookup_only_and_handles_missing_action(monkeypatch):
    no_animation = FakeObject("NoAnimation", 301)
    no_animation.animation_data = None

    no_action = FakeObject("NoAction", 302)
    no_action.animation_data = SimpleNamespace(action=None)

    with_action = FakeObject("Animated", 303)
    with_action.animation_data = SimpleNamespace(action=object())
    channelbag = SimpleNamespace(fcurves=())
    _install_fake_anim_utils(
        monkeypatch,
        {id(with_action.animation_data): channelbag},
    )

    assert semantic_adapter.assigned_channelbag(no_animation) is None
    assert semantic_adapter.assigned_channelbag(no_action) is None
    assert semantic_adapter.assigned_channelbag(with_action) is channelbag


def test_channel_binding_token_tracks_runtime_owner_action_slot_and_bag(monkeypatch):
    owner = FakeObject("Animated", 310)
    action = FakeRNA("Action", 311)
    slot = FakeRNA("Slot", 312)
    channelbag = FakeRNA("ChannelBag", 313)
    channelbag.fcurves = ()
    owner.animation_data = SimpleNamespace(action=action, action_slot=slot)
    _install_fake_anim_utils(monkeypatch, {id(owner.animation_data): channelbag})

    assert semantic_adapter.channel_binding_token(owner) == (310, 311, 312, 313)

    owner.animation_data.action_slot = FakeRNA("OtherSlot", 314)
    other_bag = FakeRNA("OtherBag", 315)
    other_bag.fcurves = ()
    _install_fake_anim_utils(monkeypatch, {id(owner.animation_data): other_bag})

    assert semantic_adapter.channel_binding_token(owner) == (310, 311, 314, 315)


def test_group_animation_owners_preserves_control_order_and_unique_prefixes():
    rig = FakeObject("Rig", 401, object_type="ARMATURE")
    first = FakePoseBone("Bone.B", 501)
    second = FakePoseBone("Bone.A", 502)
    first_resolved = semantic_adapter._bone_control(rig, first)
    duplicate = semantic_adapter._bone_control(rig, first)
    second_resolved = semantic_adapter._bone_control(rig, second)

    groups = semantic_adapter.group_animation_owners(
        (first_resolved, duplicate, second_resolved)
    )

    assert len(groups) == 1
    assert groups[0].owner_object is rig
    assert [item.control.bone_name for item in groups[0].controls] == [
        "Bone.B",
        "Bone.B",
        "Bone.A",
    ]
    assert groups[0].prefixes == (
        first.path_from_id(),
        second.path_from_id(),
    )


def test_control_property_and_rotation_helpers_preserve_blender_paths():
    obj = FakeObject("Cube", 601)
    obj.rotation_mode = "QUATERNION"
    object_resolved = semantic_adapter._object_control(obj)

    rig = FakeObject("Rig", 602, object_type="ARMATURE")
    bone = FakePoseBone('Hand.\"L\\Ctrl', 603)
    bone.rotation_mode = "AXIS_ANGLE"
    bone_resolved = semantic_adapter._bone_control(rig, bone)

    assert semantic_adapter.control_property_path(object_resolved, "location") == "location"
    assert semantic_adapter.control_property_path(
        bone_resolved,
        "location",
    ) == f"{bone.path_from_id()}.location"
    assert semantic_adapter.rotation_property(obj) == "rotation_quaternion"
    assert semantic_adapter.rotation_property(bone) == "rotation_axis_angle"
    assert semantic_adapter.rotation_property(SimpleNamespace(rotation_mode="XYZ")) == (
        "rotation_euler"
    )


def test_matches_control_path_keeps_object_and_pose_scopes_separate():
    bone_prefix = 'pose.bones["Hand.L"]'

    assert semantic_adapter.matches_control_path("location", "")
    assert not semantic_adapter.matches_control_path(
        f"{bone_prefix}.location",
        "",
    )
    assert semantic_adapter.matches_control_path(
        f"{bone_prefix}.location",
        bone_prefix,
    )
    assert semantic_adapter.matches_control_path(
        f'{bone_prefix}["stretch"]',
        bone_prefix,
    )
    assert not semantic_adapter.matches_control_path(
        'pose.bones["Hand.R"].location',
        bone_prefix,
    )


def test_channels_for_controls_scans_shared_owner_once_and_routes_families(monkeypatch):
    rig = FakeObject("Rig", 701, object_type="ARMATURE")
    rig.animation_data = SimpleNamespace(action=object())
    left = FakePoseBone("Hand.L", 702)
    right = FakePoseBone("Hand.R", 703)
    left_resolved = semantic_adapter._bone_control(rig, left)
    right_resolved = semantic_adapter._bone_control(rig, right)

    curves = CountingFCurves(
        [
            FakeFCurve(f"{left.path_from_id()}.location", 0),
            FakeFCurve(f"{left.path_from_id()}.location", 1),
            FakeFCurve(f"{right.path_from_id()}.rotation_quaternion", 0),
            FakeFCurve("location", 2),
        ]
    )
    channelbag = SimpleNamespace(fcurves=curves)
    _install_fake_anim_utils(monkeypatch, {id(rig.animation_data): channelbag})

    channels = semantic_adapter.channels_for_controls((left_resolved, right_resolved))

    assert curves.iterations == 1
    assert [
        (channel.resolved.control.bone_name, channel.family, channel.array_index)
        for channel in channels
    ] == [
        ("Hand.L", semantic_model.AWBChannel.POSITION, 0),
        ("Hand.L", semantic_model.AWBChannel.POSITION, 1),
        ("Hand.R", semantic_model.AWBChannel.ROTATION, 0),
    ]


def test_channels_for_property_returns_exact_path_in_axis_order(monkeypatch):
    rig = FakeObject("Rig", 801, object_type="ARMATURE")
    rig.animation_data = SimpleNamespace(action=object())
    bone = FakePoseBone("Hand.L", 802)
    resolved = semantic_adapter._bone_control(rig, bone)
    prefix = bone.path_from_id()
    channelbag = SimpleNamespace(
        fcurves=CountingFCurves(
            [
                FakeFCurve(f"{prefix}.location", 2),
                FakeFCurve(f"{prefix}.rotation_euler", 0),
                FakeFCurve(f"{prefix}.location", 0),
                FakeFCurve(f"{prefix}.location", 1),
                FakeFCurve(f'{prefix}["location_like"]', 0),
            ]
        )
    )
    _install_fake_anim_utils(monkeypatch, {id(rig.animation_data): channelbag})

    channels = semantic_adapter.channels_for_property(resolved, "location")

    assert [channel.array_index for channel in channels] == [0, 1, 2]
    assert all(channel.data_path == f"{prefix}.location" for channel in channels)
