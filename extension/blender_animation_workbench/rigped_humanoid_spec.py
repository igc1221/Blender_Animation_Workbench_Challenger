from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from .character_model import semantic_key_error
from .rigped_contract import RIGPED_PROFILE_HUMANOID_V1


class HumanoidLayer(StrEnum):
    AUTHORED = "AUTHORED"
    MECHANISM = "MECHANISM"
    DEFORM = "DEFORM"
    EXPORT = "EXPORT"


@dataclass(frozen=True, slots=True)
class HumanoidBoneSpec:
    role: str
    name: str
    semantic_key: str
    side: str
    mode: str
    usage: str
    layer: HumanoidLayer
    parent_role: str | None
    connected: bool
    deform: bool
    head: tuple[float, float, float]
    tail: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class HumanoidConstraintSpec:
    owner_role: str
    kind: str
    target_role: str
    pole_role: str | None = None
    chain_count: int = 0
    influence: float = 1.0
    use_tail: bool = False
    use_stretch: bool = False
    use_rotation: bool = False
    target_space: str = "WORLD"
    owner_space: str = "WORLD"
    mix_mode: str = "REPLACE"


@dataclass(frozen=True, slots=True)
class HumanoidGroupSpec:
    semantic_key: str
    label: str
    member_roles: tuple[str, ...]
    side: str = "NONE"


@dataclass(frozen=True, slots=True)
class HumanoidChainSpec:
    key: str
    semantic_key: str
    label: str
    member_roles: tuple[str, ...]
    side: str = "NONE"
    mode: str = "NEUTRAL"


@dataclass(frozen=True, slots=True)
class HumanoidOppositeSpec:
    left_role: str
    right_role: str


@dataclass(frozen=True, slots=True)
class HumanoidKinematicSpec:
    key: str
    semantic_key: str
    side: str
    fk_chain_key: str
    target_role: str
    pole_role: str
    reference_chain_key: str
    extra_roles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RigpedHumanoidSpec:
    profile_id: str = RIGPED_PROFILE_HUMANOID_V1
    character_label: str = "AWB Rigped"
    collection_name: str = "AWB Rigped"
    armature_object_name: str = "AWB_Rigped"
    armature_data_name: str = "AWB_Rigped_Data"
    world_location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    display_scale: float = 1.0
    bones: tuple[HumanoidBoneSpec, ...] = ()
    constraints: tuple[HumanoidConstraintSpec, ...] = ()
    groups: tuple[HumanoidGroupSpec, ...] = ()
    chains: tuple[HumanoidChainSpec, ...] = ()
    opposites: tuple[HumanoidOppositeSpec, ...] = ()
    kinematics: tuple[HumanoidKinematicSpec, ...] = ()

    def resolved_bones(self) -> tuple[HumanoidBoneSpec, ...]:
        return self.bones or DEFAULT_HUMANOID_BONES

    def resolved_constraints(self) -> tuple[HumanoidConstraintSpec, ...]:
        return self.constraints or DEFAULT_HUMANOID_CONSTRAINTS

    def resolved_groups(self) -> tuple[HumanoidGroupSpec, ...]:
        return self.groups or DEFAULT_HUMANOID_GROUPS

    def resolved_chains(self) -> tuple[HumanoidChainSpec, ...]:
        return self.chains or DEFAULT_HUMANOID_CHAINS

    def resolved_opposites(self) -> tuple[HumanoidOppositeSpec, ...]:
        return self.opposites or DEFAULT_HUMANOID_OPPOSITES

    def resolved_kinematics(self) -> tuple[HumanoidKinematicSpec, ...]:
        return self.kinematics or DEFAULT_HUMANOID_KINEMATICS

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("Humanoid profile_id must not be empty.")
        if not isfinite(float(self.display_scale)) or float(self.display_scale) <= 0.0:
            raise ValueError("Humanoid display_scale must be finite and positive.")
        bones = self.resolved_bones()
        if not bones:
            raise ValueError("Humanoid topology is empty.")
        by_role = {bone.role: bone for bone in bones}
        if len(by_role) != len(bones):
            raise ValueError("Humanoid topology contains duplicate roles.")
        names = tuple(bone.name for bone in bones)
        if len(set(names)) != len(names):
            raise ValueError("Humanoid topology contains duplicate bone names.")

        for bone in bones:
            if semantic_key_error(bone.semantic_key, allow_empty=False) is not None:
                raise ValueError(
                    f"Generated role {bone.role!r} uses non-canonical semantic key {bone.semantic_key!r}."
                )
            if bone.parent_role is not None and bone.parent_role not in by_role:
                raise ValueError(
                    f"Generated role {bone.role!r} references missing parent {bone.parent_role!r}."
                )
            if bone.connected and bone.parent_role is None:
                raise ValueError(f"Connected role {bone.role!r} requires a parent.")
            if bone.layer is not HumanoidLayer.DEFORM and bone.deform:
                raise ValueError(f"Only DEFORM roles may have deform=True: {bone.role!r}.")
            if bone.layer is HumanoidLayer.DEFORM and not bone.deform:
                raise ValueError(f"Every DEFORM role must have deform=True: {bone.role!r}.")
            if bone.usage == "PRIMARY" and bone.layer is not HumanoidLayer.AUTHORED:
                raise ValueError(f"Only AUTHORED roles may use PRIMARY: {bone.role!r}.")
            if bone.layer is HumanoidLayer.MECHANISM and bone.usage != "MECHANISM":
                raise ValueError(f"Mechanism role must use MECHANISM usage: {bone.role!r}.")
            if bone.layer is HumanoidLayer.DEFORM and bone.usage != "DEFORM":
                raise ValueError(f"Deform role must use DEFORM usage: {bone.role!r}.")
            if bone.layer is HumanoidLayer.EXPORT and bone.usage != "REFERENCE":
                raise ValueError(f"Export role must use REFERENCE usage: {bone.role!r}.")

        _validate_parent_cycles(by_role)
        _validate_required_authored_roles(by_role)
        _validate_symmetry(by_role)
        _validate_relationships(self, by_role)


def _validate_parent_cycles(by_role: dict[str, HumanoidBoneSpec]) -> None:
    for start in by_role:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise ValueError(f"Humanoid parent cycle detected at {start!r}.")
            seen.add(current)
            current = by_role[current].parent_role


def _validate_required_authored_roles(by_role: dict[str, HumanoidBoneSpec]) -> None:
    required = {
        "ROOT",
        "COM",
        "PELVIS",
        "SPINE",
        "SPINE2",
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
        "IK_HAND.L",
        "IK_HAND.R",
        "POLE_ELBOW.L",
        "POLE_ELBOW.R",
        "IK_FOOT.L",
        "IK_FOOT.R",
        "POLE_KNEE.L",
        "POLE_KNEE.R",
        "EXPORT_ROOT",
    }
    missing = sorted(required - set(by_role))
    if missing:
        raise ValueError(f"Humanoid minimum is missing required roles: {missing!r}.")

    for role in required - {"EXPORT_ROOT"}:
        bone = by_role[role]
        if bone.layer is not HumanoidLayer.AUTHORED:
            raise ValueError(f"Required animator role {role!r} is not AUTHORED.")
    if by_role["EXPORT_ROOT"].layer is not HumanoidLayer.EXPORT:
        raise ValueError("EXPORT_ROOT must use the EXPORT responsibility layer.")


def _validate_symmetry(by_role: dict[str, HumanoidBoneSpec]) -> None:
    mirrored_stems = (
        "CLAVICLE",
        "UPPER_ARM",
        "FOREARM",
        "HAND",
        "FINGER",
        "IK_HAND",
        "POLE_ELBOW",
        "THIGH",
        "CALF",
        "FOOT",
        "TOE",
        "IK_FOOT",
        "POLE_KNEE",
        "MCH_UPPER_ARM",
        "MCH_FOREARM",
        "MCH_HAND",
        "MCH_THIGH",
        "MCH_CALF",
        "MCH_FOOT",
        "MCH_TOE",
        "DEF_CLAVICLE",
        "DEF_UPPER_ARM",
        "DEF_FOREARM",
        "DEF_HAND",
        "DEF_FINGER",
        "DEF_THIGH",
        "DEF_CALF",
        "DEF_FOOT",
        "DEF_TOE",
    )
    for stem in mirrored_stems:
        left = by_role.get(f"{stem}.L")
        right = by_role.get(f"{stem}.R")
        if left is None or right is None:
            raise ValueError(f"Mirrored role pair is incomplete: {stem!r}.")
        if left.side != "LEFT" or right.side != "RIGHT":
            raise ValueError(f"Mirrored role pair has incorrect side metadata: {stem!r}.")
        if left.semantic_key != right.semantic_key or left.usage != right.usage:
            raise ValueError(f"Mirrored role pair has inconsistent semantics: {stem!r}.")


def _validate_relationships(
    spec: RigpedHumanoidSpec,
    by_role: dict[str, HumanoidBoneSpec],
) -> None:
    chain_by_key = {chain.key: chain for chain in spec.resolved_chains()}
    if len(chain_by_key) != len(spec.resolved_chains()):
        raise ValueError("Humanoid semantic plan contains duplicate chain keys.")
    for chain in spec.resolved_chains():
        if semantic_key_error(chain.semantic_key, allow_empty=False) is not None:
            raise ValueError(f"Chain {chain.key!r} has invalid semantic key.")
        if not chain.member_roles:
            raise ValueError(f"Chain {chain.key!r} is empty.")
        if any(role not in by_role for role in chain.member_roles):
            raise ValueError(f"Chain {chain.key!r} references missing roles.")

    for group in spec.resolved_groups():
        if semantic_key_error(group.semantic_key, allow_empty=False) is not None:
            raise ValueError(f"Group {group.label!r} has invalid semantic key.")
        if not group.member_roles or any(role not in by_role for role in group.member_roles):
            raise ValueError(f"Group {group.label!r} references invalid roles.")

    opposite_roles: set[str] = set()
    for pair in spec.resolved_opposites():
        if pair.left_role not in by_role or pair.right_role not in by_role:
            raise ValueError("Opposite pair references a missing role.")
        if pair.left_role in opposite_roles or pair.right_role in opposite_roles:
            raise ValueError("A generated role participates in multiple opposite pairs.")
        opposite_roles.update((pair.left_role, pair.right_role))
        if by_role[pair.left_role].side != "LEFT" or by_role[pair.right_role].side != "RIGHT":
            raise ValueError("Opposite pair side metadata is invalid.")
        if (
            by_role[pair.left_role].layer is not HumanoidLayer.AUTHORED
            or by_role[pair.right_role].layer is not HumanoidLayer.AUTHORED
        ):
            raise ValueError("Opposite pairs are limited to authored animator-facing roles.")

    mapping_keys: set[str] = set()
    for mapping in spec.resolved_kinematics():
        if mapping.key in mapping_keys:
            raise ValueError(f"Duplicate kinematic mapping key: {mapping.key!r}.")
        mapping_keys.add(mapping.key)
        if semantic_key_error(mapping.semantic_key, allow_empty=False) is not None:
            raise ValueError(f"Kinematic mapping {mapping.key!r} has invalid semantic key.")
        if mapping.fk_chain_key not in chain_by_key or mapping.reference_chain_key not in chain_by_key:
            raise ValueError(f"Kinematic mapping {mapping.key!r} references a missing chain.")
        for role in (mapping.target_role, mapping.pole_role, *mapping.extra_roles):
            if role not in by_role:
                raise ValueError(f"Kinematic mapping {mapping.key!r} references missing role {role!r}.")

    for constraint in spec.resolved_constraints():
        if constraint.owner_role not in by_role or constraint.target_role not in by_role:
            raise ValueError("Constraint plan references a missing owner/target role.")
        if constraint.pole_role is not None and constraint.pole_role not in by_role:
            raise ValueError("Constraint plan references a missing pole role.")
        if constraint.kind == "IK" and (constraint.chain_count != 2 or constraint.use_stretch):
            raise ValueError("Initial generated limb IK must be a non-stretch two-bone chain.")


def _bone(
    role: str,
    name: str,
    semantic_key: str,
    side: str,
    mode: str,
    usage: str,
    layer: HumanoidLayer,
    parent_role: str | None,
    connected: bool,
    deform: bool,
    head: tuple[float, float, float],
    tail: tuple[float, float, float],
) -> HumanoidBoneSpec:
    return HumanoidBoneSpec(
        role,
        name,
        semantic_key,
        side,
        mode,
        usage,
        layer,
        parent_role,
        connected,
        deform,
        head,
        tail,
    )


def _side_suffix(side: str) -> str:
    return "L" if side == "LEFT" else "R"


def _sx(side: str, value: float) -> float:
    return value if side == "LEFT" else -value


def _authored_limb_bones(side: str) -> tuple[HumanoidBoneSpec, ...]:
    suffix = _side_suffix(side)
    return (
        _bone(
            f"CLAVICLE.{suffix}",
            f"Clavicle.{suffix}",
            "awb.clavicle",
            side,
            "NEUTRAL",
            "PRIMARY",
            HumanoidLayer.AUTHORED,
            "SPINE2",
            False,
            False,
            (_sx(side, 0.05), 0.0, 1.58),
            (_sx(side, 0.22), 0.0, 1.58),
        ),
        _bone(
            f"UPPER_ARM.{suffix}",
            f"UpperArm.{suffix}",
            "awb.upper_arm",
            side,
            "FK",
            "PRIMARY",
            HumanoidLayer.AUTHORED,
            f"CLAVICLE.{suffix}",
            False,
            False,
            (_sx(side, 0.22), 0.0, 1.58),
            (_sx(side, 0.52), 0.0, 1.34),
        ),
        _bone(
            f"FOREARM.{suffix}",
            f"ForeArm.{suffix}",
            "awb.forearm",
            side,
            "FK",
            "PRIMARY",
            HumanoidLayer.AUTHORED,
            f"UPPER_ARM.{suffix}",
            True,
            False,
            (_sx(side, 0.52), 0.0, 1.34),
            (_sx(side, 0.78), 0.0, 1.12),
        ),
        _bone(
            f"HAND.{suffix}",
            f"Hand.{suffix}",
            "awb.hand",
            side,
            "FK",
            "PRIMARY",
            HumanoidLayer.AUTHORED,
            f"FOREARM.{suffix}",
            True,
            False,
            (_sx(side, 0.78), 0.0, 1.12),
            (_sx(side, 0.90), 0.0, 1.06),
        ),
        _bone(
            f"FINGER.{suffix}",
            f"Finger.{suffix}",
            "awb.finger",
            side,
            "FK",
            "PRIMARY",
            HumanoidLayer.AUTHORED,
            f"HAND.{suffix}",
            True,
            False,
            (_sx(side, 0.90), 0.0, 1.06),
            (_sx(side, 1.00), 0.0, 1.01),
        ),
        _bone(
            f"IK_HAND.{suffix}",
            f"IK_Hand.{suffix}",
            "awb.hand",
            side,
            "IK",
            "TARGET",
            HumanoidLayer.AUTHORED,
            "ROOT",
            False,
            False,
            (_sx(side, 0.78), 0.0, 1.12),
            (_sx(side, 0.90), 0.0, 1.06),
        ),
        _bone(
            f"POLE_ELBOW.{suffix}",
            f"IK_Elbow.{suffix}",
            "awb.arm",
            side,
            "IK",
            "POLE",
            HumanoidLayer.AUTHORED,
            "ROOT",
            False,
            False,
            (_sx(side, 0.52), -0.42, 1.34),
            (_sx(side, 0.52), -0.52, 1.34),
        ),
    )


def _authored_leg_bones(side: str) -> tuple[HumanoidBoneSpec, ...]:
    suffix = _side_suffix(side)
    return (
        _bone(
            f"THIGH.{suffix}",
            f"Thigh.{suffix}",
            "awb.thigh",
            side,
            "FK",
            "PRIMARY",
            HumanoidLayer.AUTHORED,
            "PELVIS",
            False,
            False,
            (_sx(side, 0.12), 0.0, 1.02),
            (_sx(side, 0.12), 0.0, 0.58),
        ),
        _bone(
            f"CALF.{suffix}",
            f"Calf.{suffix}",
            "awb.calf",
            side,
            "FK",
            "PRIMARY",
            HumanoidLayer.AUTHORED,
            f"THIGH.{suffix}",
            True,
            False,
            (_sx(side, 0.12), 0.0, 0.58),
            (_sx(side, 0.12), 0.0, 0.14),
        ),
        _bone(
            f"FOOT.{suffix}",
            f"Foot.{suffix}",
            "awb.foot",
            side,
            "FK",
            "PRIMARY",
            HumanoidLayer.AUTHORED,
            f"CALF.{suffix}",
            True,
            False,
            (_sx(side, 0.12), 0.0, 0.14),
            (_sx(side, 0.12), -0.20, 0.08),
        ),
        _bone(
            f"TOE.{suffix}",
            f"Toe.{suffix}",
            "awb.toe",
            side,
            "NEUTRAL",
            "PRIMARY",
            HumanoidLayer.AUTHORED,
            f"FOOT.{suffix}",
            True,
            False,
            (_sx(side, 0.12), -0.20, 0.08),
            (_sx(side, 0.12), -0.36, 0.08),
        ),
        _bone(
            f"IK_FOOT.{suffix}",
            f"IK_Foot.{suffix}",
            "awb.foot",
            side,
            "IK",
            "TARGET",
            HumanoidLayer.AUTHORED,
            "ROOT",
            False,
            False,
            (_sx(side, 0.12), -0.20, 0.08),
            (_sx(side, 0.12), -0.36, 0.08),
        ),
        _bone(
            f"POLE_KNEE.{suffix}",
            f"IK_Knee.{suffix}",
            "awb.leg",
            side,
            "IK",
            "POLE",
            HumanoidLayer.AUTHORED,
            "ROOT",
            False,
            False,
            (_sx(side, 0.12), -0.42, 0.58),
            (_sx(side, 0.12), -0.52, 0.58),
        ),
    )


def _mechanism_limb_bones(side: str) -> tuple[HumanoidBoneSpec, ...]:
    suffix = _side_suffix(side)
    return (
        _bone(
            f"MCH_UPPER_ARM.{suffix}",
            f"MCH_UpperArm.{suffix}",
            "awb.upper_arm",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            f"CLAVICLE.{suffix}",
            False,
            False,
            (_sx(side, 0.22), 0.0, 1.58),
            (_sx(side, 0.52), 0.0, 1.34),
        ),
        _bone(
            f"MCH_FOREARM.{suffix}",
            f"MCH_ForeArm.{suffix}",
            "awb.forearm",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            f"MCH_UPPER_ARM.{suffix}",
            True,
            False,
            (_sx(side, 0.52), 0.0, 1.34),
            (_sx(side, 0.78), 0.0, 1.12),
        ),
        _bone(
            f"MCH_HAND.{suffix}",
            f"MCH_Hand.{suffix}",
            "awb.hand",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            f"MCH_FOREARM.{suffix}",
            True,
            False,
            (_sx(side, 0.78), 0.0, 1.12),
            (_sx(side, 0.90), 0.0, 1.06),
        ),
        _bone(
            f"MCH_THIGH.{suffix}",
            f"MCH_Thigh.{suffix}",
            "awb.thigh",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            "PELVIS",
            False,
            False,
            (_sx(side, 0.12), 0.0, 1.02),
            (_sx(side, 0.12), 0.0, 0.58),
        ),
        _bone(
            f"MCH_CALF.{suffix}",
            f"MCH_Calf.{suffix}",
            "awb.calf",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            f"MCH_THIGH.{suffix}",
            True,
            False,
            (_sx(side, 0.12), 0.0, 0.58),
            (_sx(side, 0.12), 0.0, 0.14),
        ),
        _bone(
            f"MCH_FOOT.{suffix}",
            f"MCH_Foot.{suffix}",
            "awb.foot",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            f"MCH_CALF.{suffix}",
            True,
            False,
            (_sx(side, 0.12), 0.0, 0.14),
            (_sx(side, 0.12), -0.20, 0.08),
        ),
        _bone(
            f"MCH_TOE.{suffix}",
            f"MCH_Toe.{suffix}",
            "awb.toe",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            f"MCH_FOOT.{suffix}",
            True,
            False,
            (_sx(side, 0.12), -0.20, 0.08),
            (_sx(side, 0.12), -0.36, 0.08),
        ),
        # I15 point-hold carriers are intentionally unparented mechanism bones.
        # Their pose-space position therefore stays independent of authored Root
        # motion and can act as the replay authority for a world-style plant.
        _bone(
            f"MCH_CONTACT_HAND.{suffix}",
            f"MCH_ContactHand.{suffix}",
            "awb.contact",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            None,
            False,
            False,
            (_sx(side, 0.78), 0.0, 1.12),
            (_sx(side, 0.90), 0.0, 1.06),
        ),
        _bone(
            f"MCH_CONTACT_FOOT.{suffix}",
            f"MCH_ContactFoot.{suffix}",
            "awb.contact",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            None,
            False,
            False,
            (_sx(side, 0.12), -0.20, 0.08),
            (_sx(side, 0.12), -0.36, 0.08),
        ),
        # I19 keeps the authored point anchor separate from the I15 base/origin
        # hold carrier. Both are unparented so a World-space point can remain
        # replay-stable while the authored IK target rotates around it.
        _bone(
            f"MCH_CONTACT_POINT_HAND.{suffix}",
            f"MCH_ContactPointHand.{suffix}",
            "awb.contact_point",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            None,
            False,
            False,
            (_sx(side, 0.78), 0.0, 1.12),
            (_sx(side, 0.90), 0.0, 1.06),
        ),
        _bone(
            f"MCH_CONTACT_POINT_FOOT.{suffix}",
            f"MCH_ContactPointFoot.{suffix}",
            "awb.contact_point",
            side,
            "NEUTRAL",
            "MECHANISM",
            HumanoidLayer.MECHANISM,
            None,
            False,
            False,
            (_sx(side, 0.12), -0.20, 0.08),
            (_sx(side, 0.12), -0.36, 0.08),
        ),
    )


def _deform_bones(side: str) -> tuple[HumanoidBoneSpec, ...]:
    suffix = _side_suffix(side)
    return (
        _bone(
            f"DEF_CLAVICLE.{suffix}",
            f"DEF_Clavicle.{suffix}",
            "awb.clavicle",
            side,
            "NEUTRAL",
            "DEFORM",
            HumanoidLayer.DEFORM,
            "DEF_SPINE2",
            False,
            True,
            (_sx(side, 0.05), 0.0, 1.58),
            (_sx(side, 0.22), 0.0, 1.58),
        ),
        _bone(
            f"DEF_UPPER_ARM.{suffix}",
            f"DEF_UpperArm.{suffix}",
            "awb.upper_arm",
            side,
            "NEUTRAL",
            "DEFORM",
            HumanoidLayer.DEFORM,
            f"DEF_CLAVICLE.{suffix}",
            False,
            True,
            (_sx(side, 0.22), 0.0, 1.58),
            (_sx(side, 0.52), 0.0, 1.34),
        ),
        _bone(
            f"DEF_FOREARM.{suffix}",
            f"DEF_ForeArm.{suffix}",
            "awb.forearm",
            side,
            "NEUTRAL",
            "DEFORM",
            HumanoidLayer.DEFORM,
            f"DEF_UPPER_ARM.{suffix}",
            True,
            True,
            (_sx(side, 0.52), 0.0, 1.34),
            (_sx(side, 0.78), 0.0, 1.12),
        ),
        _bone(
            f"DEF_HAND.{suffix}",
            f"DEF_Hand.{suffix}",
            "awb.hand",
            side,
            "NEUTRAL",
            "DEFORM",
            HumanoidLayer.DEFORM,
            f"DEF_FOREARM.{suffix}",
            True,
            True,
            (_sx(side, 0.78), 0.0, 1.12),
            (_sx(side, 0.90), 0.0, 1.06),
        ),
        _bone(
            f"DEF_FINGER.{suffix}",
            f"DEF_Finger.{suffix}",
            "awb.finger",
            side,
            "NEUTRAL",
            "DEFORM",
            HumanoidLayer.DEFORM,
            f"DEF_HAND.{suffix}",
            True,
            True,
            (_sx(side, 0.90), 0.0, 1.06),
            (_sx(side, 1.00), 0.0, 1.01),
        ),
        _bone(
            f"DEF_THIGH.{suffix}",
            f"DEF_Thigh.{suffix}",
            "awb.thigh",
            side,
            "NEUTRAL",
            "DEFORM",
            HumanoidLayer.DEFORM,
            "DEF_PELVIS",
            False,
            True,
            (_sx(side, 0.12), 0.0, 1.02),
            (_sx(side, 0.12), 0.0, 0.58),
        ),
        _bone(
            f"DEF_CALF.{suffix}",
            f"DEF_Calf.{suffix}",
            "awb.calf",
            side,
            "NEUTRAL",
            "DEFORM",
            HumanoidLayer.DEFORM,
            f"DEF_THIGH.{suffix}",
            True,
            True,
            (_sx(side, 0.12), 0.0, 0.58),
            (_sx(side, 0.12), 0.0, 0.14),
        ),
        _bone(
            f"DEF_FOOT.{suffix}",
            f"DEF_Foot.{suffix}",
            "awb.foot",
            side,
            "NEUTRAL",
            "DEFORM",
            HumanoidLayer.DEFORM,
            f"DEF_CALF.{suffix}",
            True,
            True,
            (_sx(side, 0.12), 0.0, 0.14),
            (_sx(side, 0.12), -0.20, 0.08),
        ),
        _bone(
            f"DEF_TOE.{suffix}",
            f"DEF_Toe.{suffix}",
            "awb.toe",
            side,
            "NEUTRAL",
            "DEFORM",
            HumanoidLayer.DEFORM,
            f"DEF_FOOT.{suffix}",
            True,
            True,
            (_sx(side, 0.12), -0.20, 0.08),
            (_sx(side, 0.12), -0.36, 0.08),
        ),
    )


DEFAULT_HUMANOID_BONES: tuple[HumanoidBoneSpec, ...] = (
    _bone("ROOT", "Root", "awb.root", "CENTER", "NEUTRAL", "PRIMARY", HumanoidLayer.AUTHORED, None, False, False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.2)),
    _bone("COM", "COM", "awb.com", "CENTER", "NEUTRAL", "PRIMARY", HumanoidLayer.AUTHORED, "ROOT", False, False, (0.0, 0.0, 1.035), (0.0, 0.0, 1.115)),
    _bone("PELVIS", "Pelvis", "awb.pelvis", "CENTER", "NEUTRAL", "PRIMARY", HumanoidLayer.AUTHORED, "COM", False, False, (0.0, 0.0, 1.0), (0.0, 0.0, 1.15)),
    _bone("SPINE", "Spine", "awb.spine", "CENTER", "NEUTRAL", "PRIMARY", HumanoidLayer.AUTHORED, "COM", False, False, (0.0, 0.0, 1.15), (0.0, 0.0, 1.365)),
    _bone("SPINE2", "Spine2", "awb.spine", "CENTER", "NEUTRAL", "PRIMARY", HumanoidLayer.AUTHORED, "SPINE", True, False, (0.0, 0.0, 1.365), (0.0, 0.0, 1.58)),
    _bone("NECK", "Neck", "awb.neck", "CENTER", "NEUTRAL", "PRIMARY", HumanoidLayer.AUTHORED, "SPINE2", True, False, (0.0, 0.0, 1.58), (0.0, 0.0, 1.72)),
    _bone("HEAD", "Head", "awb.head", "CENTER", "NEUTRAL", "PRIMARY", HumanoidLayer.AUTHORED, "NECK", True, False, (0.0, 0.0, 1.72), (0.0, 0.0, 1.98)),
    *_authored_limb_bones("LEFT"),
    *_authored_limb_bones("RIGHT"),
    *_authored_leg_bones("LEFT"),
    *_authored_leg_bones("RIGHT"),
    *_mechanism_limb_bones("LEFT"),
    *_mechanism_limb_bones("RIGHT"),
    _bone("EXPORT_ROOT", "EXP_Root", "awb.root", "CENTER", "NEUTRAL", "REFERENCE", HumanoidLayer.EXPORT, None, False, False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.2)),
    _bone("DEF_PELVIS", "DEF_Pelvis", "awb.pelvis", "CENTER", "NEUTRAL", "DEFORM", HumanoidLayer.DEFORM, "EXPORT_ROOT", False, True, (0.0, 0.0, 1.0), (0.0, 0.0, 1.15)),
    _bone("DEF_SPINE", "DEF_Spine", "awb.spine", "CENTER", "NEUTRAL", "DEFORM", HumanoidLayer.DEFORM, "DEF_PELVIS", True, True, (0.0, 0.0, 1.15), (0.0, 0.0, 1.365)),
    _bone("DEF_SPINE2", "DEF_Spine2", "awb.spine", "CENTER", "NEUTRAL", "DEFORM", HumanoidLayer.DEFORM, "DEF_SPINE", True, True, (0.0, 0.0, 1.365), (0.0, 0.0, 1.58)),
    _bone("DEF_NECK", "DEF_Neck", "awb.neck", "CENTER", "NEUTRAL", "DEFORM", HumanoidLayer.DEFORM, "DEF_SPINE2", True, True, (0.0, 0.0, 1.58), (0.0, 0.0, 1.72)),
    _bone("DEF_HEAD", "DEF_Head", "awb.head", "CENTER", "NEUTRAL", "DEFORM", HumanoidLayer.DEFORM, "DEF_NECK", True, True, (0.0, 0.0, 1.72), (0.0, 0.0, 1.98)),
    *_deform_bones("LEFT"),
    *_deform_bones("RIGHT"),
)


def _copy(owner: str, target: str, *, influence: float = 1.0) -> HumanoidConstraintSpec:
    return HumanoidConstraintSpec(owner, "COPY_TRANSFORMS", target, influence=influence)


def _copy_rotation_local(
    owner: str,
    target: str,
    *,
    influence: float = 1.0,
) -> HumanoidConstraintSpec:
    return HumanoidConstraintSpec(
        owner,
        "COPY_ROTATION",
        target,
        influence=influence,
        target_space="LOCAL",
        owner_space="LOCAL",
    )


def _limb_constraints(side: str) -> tuple[HumanoidConstraintSpec, ...]:
    suffix = _side_suffix(side)
    return (
        _copy_rotation_local(f"MCH_UPPER_ARM.{suffix}", f"UPPER_ARM.{suffix}"),
        _copy_rotation_local(f"MCH_FOREARM.{suffix}", f"FOREARM.{suffix}"),
        _copy_rotation_local(f"MCH_HAND.{suffix}", f"HAND.{suffix}"),
        HumanoidConstraintSpec(
            f"MCH_FOREARM.{suffix}",
            "IK",
            f"IK_HAND.{suffix}",
            pole_role=f"POLE_ELBOW.{suffix}",
            chain_count=2,
            influence=0.0,
            use_tail=True,
            use_stretch=False,
            use_rotation=False,
        ),
        HumanoidConstraintSpec(
            f"MCH_HAND.{suffix}",
            "COPY_ROTATION",
            f"IK_HAND.{suffix}",
            influence=0.0,
        ),
        HumanoidConstraintSpec(
            f"IK_HAND.{suffix}",
            "COPY_LOCATION",
            f"MCH_CONTACT_HAND.{suffix}",
            influence=0.0,
            target_space="WORLD",
            owner_space="WORLD",
        ),
        HumanoidConstraintSpec(
            f"IK_HAND.{suffix}",
            "PIVOT",
            f"MCH_CONTACT_POINT_HAND.{suffix}",
            influence=0.0,
        ),
        _copy_rotation_local(f"MCH_THIGH.{suffix}", f"THIGH.{suffix}"),
        _copy_rotation_local(f"MCH_CALF.{suffix}", f"CALF.{suffix}"),
        _copy_rotation_local(f"MCH_FOOT.{suffix}", f"FOOT.{suffix}"),
        _copy_rotation_local(f"MCH_TOE.{suffix}", f"TOE.{suffix}"),
        HumanoidConstraintSpec(
            f"MCH_CALF.{suffix}",
            "IK",
            f"IK_FOOT.{suffix}",
            pole_role=f"POLE_KNEE.{suffix}",
            chain_count=2,
            influence=0.0,
            use_tail=True,
            use_stretch=False,
            use_rotation=False,
        ),
        HumanoidConstraintSpec(
            f"MCH_FOOT.{suffix}",
            "COPY_ROTATION",
            f"IK_FOOT.{suffix}",
            influence=0.0,
        ),
        HumanoidConstraintSpec(
            f"IK_FOOT.{suffix}",
            "COPY_LOCATION",
            f"MCH_CONTACT_FOOT.{suffix}",
            influence=0.0,
            target_space="WORLD",
            owner_space="WORLD",
        ),
        HumanoidConstraintSpec(
            f"IK_FOOT.{suffix}",
            "PIVOT",
            f"MCH_CONTACT_POINT_FOOT.{suffix}",
            influence=0.0,
        ),
    )


def _deform_constraints(side: str) -> tuple[HumanoidConstraintSpec, ...]:
    suffix = _side_suffix(side)
    return (
        _copy(f"DEF_CLAVICLE.{suffix}", f"CLAVICLE.{suffix}"),
        _copy(f"DEF_UPPER_ARM.{suffix}", f"MCH_UPPER_ARM.{suffix}"),
        _copy(f"DEF_FOREARM.{suffix}", f"MCH_FOREARM.{suffix}"),
        _copy(f"DEF_HAND.{suffix}", f"MCH_HAND.{suffix}"),
        _copy(f"DEF_FINGER.{suffix}", f"FINGER.{suffix}"),
        _copy(f"DEF_THIGH.{suffix}", f"MCH_THIGH.{suffix}"),
        _copy(f"DEF_CALF.{suffix}", f"MCH_CALF.{suffix}"),
        _copy(f"DEF_FOOT.{suffix}", f"MCH_FOOT.{suffix}"),
        _copy(f"DEF_TOE.{suffix}", f"MCH_TOE.{suffix}"),
    )


DEFAULT_HUMANOID_CONSTRAINTS: tuple[HumanoidConstraintSpec, ...] = (
    *_limb_constraints("LEFT"),
    *_limb_constraints("RIGHT"),
    _copy("EXPORT_ROOT", "ROOT"),
    _copy("DEF_PELVIS", "PELVIS"),
    _copy("DEF_SPINE", "SPINE"),
    _copy("DEF_SPINE2", "SPINE2"),
    _copy("DEF_NECK", "NECK"),
    _copy("DEF_HEAD", "HEAD"),
    *_deform_constraints("LEFT"),
    *_deform_constraints("RIGHT"),
)


DEFAULT_HUMANOID_GROUPS: tuple[HumanoidGroupSpec, ...] = (
    HumanoidGroupSpec("awb.root", "Root Motion Source Map", ("ROOT", "EXPORT_ROOT")),
    HumanoidGroupSpec("awb.com", "Body Center", ("COM", "PELVIS", "SPINE", "SPINE2", "NECK", "HEAD"), "CENTER"),
    HumanoidGroupSpec("awb.arm", "Left Arm", ("CLAVICLE.L", "UPPER_ARM.L", "FOREARM.L", "HAND.L", "FINGER.L", "IK_HAND.L", "POLE_ELBOW.L"), "LEFT"),
    HumanoidGroupSpec("awb.arm", "Right Arm", ("CLAVICLE.R", "UPPER_ARM.R", "FOREARM.R", "HAND.R", "FINGER.R", "IK_HAND.R", "POLE_ELBOW.R"), "RIGHT"),
    HumanoidGroupSpec("awb.leg", "Left Leg", ("THIGH.L", "CALF.L", "FOOT.L", "TOE.L", "IK_FOOT.L", "POLE_KNEE.L"), "LEFT"),
    HumanoidGroupSpec("awb.leg", "Right Leg", ("THIGH.R", "CALF.R", "FOOT.R", "TOE.R", "IK_FOOT.R", "POLE_KNEE.R"), "RIGHT"),
    HumanoidGroupSpec("awb.hand", "Left Hand Terminal Orientation", ("HAND.L", "FINGER.L", "IK_HAND.L"), "LEFT"),
    HumanoidGroupSpec("awb.hand", "Right Hand Terminal Orientation", ("HAND.R", "FINGER.R", "IK_HAND.R"), "RIGHT"),
    HumanoidGroupSpec("awb.foot", "Left Foot Terminal Orientation", ("FOOT.L", "IK_FOOT.L"), "LEFT"),
    HumanoidGroupSpec("awb.foot", "Right Foot Terminal Orientation", ("FOOT.R", "IK_FOOT.R"), "RIGHT"),
)


DEFAULT_HUMANOID_CHAINS: tuple[HumanoidChainSpec, ...] = (
    HumanoidChainSpec("SPINE", "awb.spine", "Spine", ("SPINE", "SPINE2", "NECK", "HEAD"), "CENTER", "NEUTRAL"),
    HumanoidChainSpec("ARM_FK.L", "awb.arm", "Left Arm FK", ("UPPER_ARM.L", "FOREARM.L"), "LEFT", "FK"),
    HumanoidChainSpec("ARM_FK.R", "awb.arm", "Right Arm FK", ("UPPER_ARM.R", "FOREARM.R"), "RIGHT", "FK"),
    HumanoidChainSpec("ARM_RESULT.L", "awb.arm", "Left Arm Result", ("MCH_UPPER_ARM.L", "MCH_FOREARM.L"), "LEFT", "NEUTRAL"),
    HumanoidChainSpec("ARM_RESULT.R", "awb.arm", "Right Arm Result", ("MCH_UPPER_ARM.R", "MCH_FOREARM.R"), "RIGHT", "NEUTRAL"),
    HumanoidChainSpec("LEG_FK.L", "awb.leg", "Left Leg FK", ("THIGH.L", "CALF.L"), "LEFT", "FK"),
    HumanoidChainSpec("LEG_FK.R", "awb.leg", "Right Leg FK", ("THIGH.R", "CALF.R"), "RIGHT", "FK"),
    HumanoidChainSpec("LEG_RESULT.L", "awb.leg", "Left Leg Result", ("MCH_THIGH.L", "MCH_CALF.L"), "LEFT", "NEUTRAL"),
    HumanoidChainSpec("LEG_RESULT.R", "awb.leg", "Right Leg Result", ("MCH_THIGH.R", "MCH_CALF.R"), "RIGHT", "NEUTRAL"),
)


def _opposites_for_stems(stems: tuple[str, ...]) -> tuple[HumanoidOppositeSpec, ...]:
    return tuple(HumanoidOppositeSpec(f"{stem}.L", f"{stem}.R") for stem in stems)


DEFAULT_HUMANOID_OPPOSITES: tuple[HumanoidOppositeSpec, ...] = _opposites_for_stems(
    (
        "CLAVICLE",
        "UPPER_ARM",
        "FOREARM",
        "HAND",
        "FINGER",
        "IK_HAND",
        "POLE_ELBOW",
        "THIGH",
        "CALF",
        "FOOT",
        "TOE",
        "IK_FOOT",
        "POLE_KNEE",
    )
)


DEFAULT_HUMANOID_KINEMATICS: tuple[HumanoidKinematicSpec, ...] = (
    HumanoidKinematicSpec(
        "ARM.L",
        "awb.arm",
        "LEFT",
        "ARM_FK.L",
        "IK_HAND.L",
        "POLE_ELBOW.L",
        "ARM_RESULT.L",
        ("MCH_HAND.L", "MCH_CONTACT_HAND.L", "MCH_CONTACT_POINT_HAND.L"),
    ),
    HumanoidKinematicSpec(
        "ARM.R",
        "awb.arm",
        "RIGHT",
        "ARM_FK.R",
        "IK_HAND.R",
        "POLE_ELBOW.R",
        "ARM_RESULT.R",
        ("MCH_HAND.R", "MCH_CONTACT_HAND.R", "MCH_CONTACT_POINT_HAND.R"),
    ),
    HumanoidKinematicSpec(
        "LEG.L",
        "awb.leg",
        "LEFT",
        "LEG_FK.L",
        "IK_FOOT.L",
        "POLE_KNEE.L",
        "LEG_RESULT.L",
        ("MCH_FOOT.L", "MCH_TOE.L", "MCH_CONTACT_FOOT.L", "MCH_CONTACT_POINT_FOOT.L"),
    ),
    HumanoidKinematicSpec(
        "LEG.R",
        "awb.leg",
        "RIGHT",
        "LEG_FK.R",
        "IK_FOOT.R",
        "POLE_KNEE.R",
        "LEG_RESULT.R",
        ("MCH_FOOT.R", "MCH_TOE.R", "MCH_CONTACT_FOOT.R", "MCH_CONTACT_POINT_FOOT.R"),
    ),
)
