from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AWBControlKind(StrEnum):
    OBJECT = "OBJECT"
    BONE = "BONE"


class AWBControlRole(StrEnum):
    ROOT = "Root"
    COM = "COM"
    PELVIS = "Pelvis"
    SPINE = "Spine"
    NECK = "Neck"
    HEAD = "Head"
    CLAVICLE_L = "Clavicle.L"
    CLAVICLE_R = "Clavicle.R"
    UPPER_ARM_L = "UpperArm.L"
    UPPER_ARM_R = "UpperArm.R"
    FOREARM_L = "Forearm.L"
    FOREARM_R = "Forearm.R"
    HAND_L = "Hand.L"
    HAND_R = "Hand.R"
    THIGH_L = "Thigh.L"
    THIGH_R = "Thigh.R"
    CALF_L = "Calf.L"
    CALF_R = "Calf.R"
    FOOT_L = "Foot.L"
    FOOT_R = "Foot.R"
    TOE_L = "Toe.L"
    TOE_R = "Toe.R"


class AWBChannel(StrEnum):
    POSITION = "POSITION"
    ROTATION = "ROTATION"
    SCALE = "SCALE"
    OTHER = "OTHER"


class AWBKeyType(StrEnum):
    TRANSFORM = "TRANSFORM"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class AWBControl:
    """Stable AWB-facing identity for one animation control.

    Blender object/bone names are currently the source identity. A later rig
    adapter can replace ``semantic_name`` with character roles such as
    Pelvis/COM/Hand.L without changing Track Bar consumers.
    """

    kind: AWBControlKind
    object_name: str
    bone_name: str | None = None
    semantic_name: str | None = None
    role: AWBControlRole | None = None

    @classmethod
    def object(
        cls,
        object_name: str,
        *,
        role: AWBControlRole | None = None,
    ) -> AWBControl:
        return cls(kind=AWBControlKind.OBJECT, object_name=object_name, role=role)

    @classmethod
    def bone(
        cls,
        object_name: str,
        bone_name: str,
        *,
        role: AWBControlRole | None = None,
    ) -> AWBControl:
        return cls(
            kind=AWBControlKind.BONE,
            object_name=object_name,
            bone_name=bone_name,
            role=role,
        )

    @property
    def identity_key(self) -> tuple[AWBControlKind, str, str | None]:
        """Return the metadata-independent AWB control identity tuple."""
        return (self.kind, self.object_name, self.bone_name)

    @property
    def control_id(self) -> str:
        if self.kind == AWBControlKind.BONE and self.bone_name is not None:
            return f"bone:{self.object_name}:{self.bone_name}"
        return f"object:{self.object_name}"

    @property
    def display_name(self) -> str:
        if self.semantic_name:
            return self.semantic_name
        if self.role is not None:
            return self.role.value
        return self.bone_name or self.object_name


def parse_control_role(value: object) -> AWBControlRole | None:
    if isinstance(value, AWBControlRole):
        return value
    if not isinstance(value, str):
        return None

    normalized = value.strip().casefold()
    if not normalized:
        return None

    for role in AWBControlRole:
        if normalized in {role.name.casefold(), role.value.casefold()}:
            return role
    return None


@dataclass(frozen=True, slots=True)
class AWBKey:
    """One semantic key cell consumed by AWB animation UI."""

    frame: float
    control: AWBControl
    channels: frozenset[AWBChannel]
    key_type: AWBKeyType = AWBKeyType.TRANSFORM
    interpolation: str | None = None
    selected: bool = False

    @property
    def channel_names(self) -> frozenset[str]:
        return frozenset(channel.value for channel in self.channels)


@dataclass(frozen=True, slots=True)
class AWBFrameAggregate:
    """Frame-level union that preserves its per-control semantic contributors."""

    frame: float
    contributors: tuple[AWBKey, ...]
    channels: frozenset[AWBChannel]
    selected: bool = False

    @property
    def channel_names(self) -> frozenset[str]:
        return frozenset(channel.value for channel in self.channels)


def key_type_for_channels(channels: frozenset[AWBChannel]) -> AWBKeyType:
    transform_channels = {AWBChannel.POSITION, AWBChannel.ROTATION, AWBChannel.SCALE}
    if channels.intersection(transform_channels):
        return AWBKeyType.TRANSFORM
    return AWBKeyType.OTHER


def transform_channel(data_path: str) -> AWBChannel:
    if data_path == "location" or data_path.endswith(".location"):
        return AWBChannel.POSITION
    if data_path in {"rotation_euler", "rotation_quaternion", "rotation_axis_angle"}:
        return AWBChannel.ROTATION
    if data_path.endswith((".rotation_euler", ".rotation_quaternion", ".rotation_axis_angle")):
        return AWBChannel.ROTATION
    if data_path == "scale" or data_path.endswith(".scale"):
        return AWBChannel.SCALE
    return AWBChannel.OTHER
