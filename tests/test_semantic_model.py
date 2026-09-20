import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "semantic_model.py"
)
SPEC = spec_from_file_location("baw_semantic_model", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
semantic_model = module_from_spec(SPEC)
sys.modules[SPEC.name] = semantic_model
SPEC.loader.exec_module(semantic_model)


def test_object_and_bone_control_ids_are_stable():
    object_control = semantic_model.AWBControl.object("Cube")
    bone_control = semantic_model.AWBControl.bone("Rig", "Hand.L")

    assert object_control.control_id == "object:Cube"
    assert object_control.display_name == "Cube"
    assert bone_control.control_id == "bone:Rig:Hand.L"
    assert bone_control.display_name == "Hand.L"


def test_semantic_name_overrides_display_name_without_changing_identity():
    control = semantic_model.AWBControl(
        kind=semantic_model.AWBControlKind.BONE,
        object_name="Rig",
        bone_name="CTRL_hand.L",
        semantic_name="Hand.L",
    )

    assert control.control_id == "bone:Rig:CTRL_hand.L"
    assert control.identity_key == (
        semantic_model.AWBControlKind.BONE,
        "Rig",
        "CTRL_hand.L",
    )
    assert control.display_name == "Hand.L"


def test_identity_key_ignores_role_and_display_metadata():
    base = semantic_model.AWBControl.bone("Rig", "CTRL_L_Hand")
    decorated = semantic_model.AWBControl(
        kind=semantic_model.AWBControlKind.BONE,
        object_name="Rig",
        bone_name="CTRL_L_Hand",
        semantic_name="Hand.L",
        role=semantic_model.AWBControlRole.HAND_L,
    )

    assert base.identity_key == decorated.identity_key
    assert base != decorated


def test_identity_key_avoids_legacy_control_id_delimiter_collision():
    first = semantic_model.AWBControl.bone("Rig:Part", "Hand")
    second = semantic_model.AWBControl.bone("Rig", "Part:Hand")

    assert first.control_id == second.control_id
    assert first.identity_key != second.identity_key


def test_control_role_provides_canonical_character_name():
    control = semantic_model.AWBControl.bone(
        "Rig",
        "CTRL_L_Hand",
        role=semantic_model.AWBControlRole.HAND_L,
    )

    assert control.control_id == "bone:Rig:CTRL_L_Hand"
    assert control.role == semantic_model.AWBControlRole.HAND_L
    assert control.display_name == "Hand.L"
    assert semantic_model.parse_control_role("HAND_L") == semantic_model.AWBControlRole.HAND_L
    assert semantic_model.parse_control_role("hand.l") == semantic_model.AWBControlRole.HAND_L
    assert semantic_model.parse_control_role("unknown") is None


def test_transform_channel_maps_object_and_pose_paths():
    cases = {
        "location": semantic_model.AWBChannel.POSITION,
        'pose.bones["Hand.L"].location': semantic_model.AWBChannel.POSITION,
        "rotation_euler": semantic_model.AWBChannel.ROTATION,
        'pose.bones["Hand.L"].rotation_quaternion': semantic_model.AWBChannel.ROTATION,
        "scale": semantic_model.AWBChannel.SCALE,
        'pose.bones["Hand.L"].scale': semantic_model.AWBChannel.SCALE,
        '["custom_prop"]': semantic_model.AWBChannel.OTHER,
    }

    for data_path, expected in cases.items():
        assert semantic_model.transform_channel(data_path) == expected


def test_awb_key_exposes_legacy_channel_names_for_trackbar_adapter():
    control = semantic_model.AWBControl.object("Cube")
    key = semantic_model.AWBKey(
        frame=21.0,
        control=control,
        channels=frozenset(
            {
                semantic_model.AWBChannel.POSITION,
                semantic_model.AWBChannel.ROTATION,
                semantic_model.AWBChannel.SCALE,
            }
        ),
        selected=True,
    )

    assert key.frame == 21.0
    assert key.selected is True
    assert key.key_type == semantic_model.AWBKeyType.TRANSFORM
    assert key.interpolation is None
    assert key.channel_names == frozenset({"POSITION", "ROTATION", "SCALE"})


def test_key_type_is_derived_from_semantic_channels():
    transform_channels = frozenset(
        {
            semantic_model.AWBChannel.POSITION,
            semantic_model.AWBChannel.OTHER,
        }
    )
    other_channels = frozenset({semantic_model.AWBChannel.OTHER})

    assert (
        semantic_model.key_type_for_channels(transform_channels)
        == semantic_model.AWBKeyType.TRANSFORM
    )
    assert semantic_model.key_type_for_channels(other_channels) == semantic_model.AWBKeyType.OTHER
