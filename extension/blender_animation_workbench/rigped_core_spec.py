from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .rigped_contract import RIGPED_PROFILE_HUMANOID_V1


class GeneratedRole(StrEnum):
    ROOT = "ROOT"
    COM = "COM"
    PELVIS = "PELVIS"
    FK_UPPER = "FK_UPPER"
    FK_FORE = "FK_FORE"
    FK_HAND = "FK_HAND"
    IK_EFFECTOR = "IK_EFFECTOR"
    IK_POLE = "IK_POLE"
    MCH_UPPER = "MCH_UPPER"
    MCH_FORE = "MCH_FORE"
    MCH_HAND = "MCH_HAND"
    EXPORT_ROOT = "EXPORT_ROOT"
    DEF_PELVIS = "DEF_PELVIS"
    DEF_UPPER = "DEF_UPPER"
    DEF_FORE = "DEF_FORE"
    DEF_HAND = "DEF_HAND"


class GeneratedLayer(StrEnum):
    AUTHORED = "AUTHORED"
    MECHANISM = "MECHANISM"
    DEFORM = "DEFORM"
    EXPORT = "EXPORT"


@dataclass(frozen=True, slots=True)
class GeneratedBoneSpec:
    role: GeneratedRole
    name: str
    semantic_key: str
    side: str
    mode: str
    usage: str
    layer: GeneratedLayer
    parent_role: GeneratedRole | None
    connected: bool
    deform: bool
    head: tuple[float, float, float]
    tail: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class RigpedCoreSpec:
    profile_id: str = RIGPED_PROFILE_HUMANOID_V1
    character_label: str = "AWB Rigped Core"
    collection_name: str = "AWB Rigped Core"
    armature_object_name: str = "AWB_Rigped_Core"
    armature_data_name: str = "AWB_Rigped_Core_Data"
    topology: tuple[GeneratedBoneSpec, ...] = ()

    def resolved_topology(self) -> tuple[GeneratedBoneSpec, ...]:
        return self.topology or DEFAULT_CORE_TOPOLOGY

    def validate(self) -> None:
        topology = self.resolved_topology()
        if not topology:
            raise ValueError("Rigped core topology is empty.")
        if not self.profile_id:
            raise ValueError("Rigped profile_id must not be empty.")
        if not self.armature_object_name or not self.armature_data_name or not self.collection_name:
            raise ValueError("Generated datablock names must not be empty.")

        roles = [entry.role for entry in topology]
        names = [entry.name for entry in topology]
        if len(roles) != len(set(roles)):
            raise ValueError("Rigped core topology contains duplicate generated roles.")
        if len(names) != len(set(names)):
            raise ValueError("Rigped core topology contains duplicate bone names.")

        required = set(GeneratedRole)
        if set(roles) != required:
            missing = sorted(role.value for role in required - set(roles))
            extra = sorted(role.value for role in set(roles) - required)
            raise ValueError(
                f"Rigped core role cardinality mismatch; missing={missing}, extra={extra}."
            )

        by_role = {entry.role: entry for entry in topology}
        for entry in topology:
            if entry.parent_role is not None and entry.parent_role not in by_role:
                raise ValueError(
                    f"Missing parent role {entry.parent_role.value} for {entry.role.value}."
                )
            if entry.connected and entry.parent_role is None:
                raise ValueError(f"Connected bone {entry.name!r} requires a parent.")
            if entry.layer != GeneratedLayer.DEFORM and entry.deform:
                raise ValueError(
                    f"Only DEFORM layer bones may have use_deform enabled: {entry.name!r}."
                )
            if entry.usage == "PRIMARY" and entry.layer != GeneratedLayer.AUTHORED:
                raise ValueError(
                    f"Only AUTHORED layer rows may use PRIMARY: {entry.name!r}."
                )

        expected_parents = {
            GeneratedRole.COM: GeneratedRole.ROOT,
            GeneratedRole.PELVIS: GeneratedRole.COM,
            GeneratedRole.FK_UPPER: GeneratedRole.PELVIS,
            GeneratedRole.FK_FORE: GeneratedRole.FK_UPPER,
            GeneratedRole.FK_HAND: GeneratedRole.FK_FORE,
            GeneratedRole.IK_EFFECTOR: GeneratedRole.ROOT,
            GeneratedRole.IK_POLE: GeneratedRole.ROOT,
            GeneratedRole.MCH_UPPER: GeneratedRole.PELVIS,
            GeneratedRole.MCH_FORE: GeneratedRole.MCH_UPPER,
            GeneratedRole.MCH_HAND: GeneratedRole.MCH_FORE,
            GeneratedRole.DEF_PELVIS: GeneratedRole.EXPORT_ROOT,
            GeneratedRole.DEF_UPPER: GeneratedRole.DEF_PELVIS,
            GeneratedRole.DEF_FORE: GeneratedRole.DEF_UPPER,
            GeneratedRole.DEF_HAND: GeneratedRole.DEF_FORE,
        }
        for role, parent in expected_parents.items():
            if by_role[role].parent_role != parent:
                raise ValueError(f"Generated role {role.value} must parent to {parent.value}.")

        if by_role[GeneratedRole.ROOT].parent_role is not None:
            raise ValueError("Authored Root must remain top-level.")
        if by_role[GeneratedRole.EXPORT_ROOT].parent_role is not None:
            raise ValueError("Export root seed must remain top-level.")

        connected_roles = (
            GeneratedRole.FK_FORE,
            GeneratedRole.FK_HAND,
            GeneratedRole.MCH_FORE,
            GeneratedRole.MCH_HAND,
            GeneratedRole.DEF_FORE,
            GeneratedRole.DEF_HAND,
        )
        for role in connected_roles:
            if not by_role[role].connected:
                raise ValueError(f"Generated chain role {role.value} must be connected.")

        if by_role[GeneratedRole.IK_EFFECTOR].usage != "TARGET":
            raise ValueError("IK effector must use TARGET usage.")
        if by_role[GeneratedRole.IK_POLE].usage != "POLE":
            raise ValueError("IK pole must use POLE usage.")

        self._validate_parent_cycles(by_role)

    @staticmethod
    def _validate_parent_cycles(by_role: dict[GeneratedRole, GeneratedBoneSpec]) -> None:
        for start in by_role:
            seen: set[GeneratedRole] = set()
            current: GeneratedRole | None = start
            while current is not None:
                if current in seen:
                    raise ValueError(f"Generated parent cycle detected at {start.value}.")
                seen.add(current)
                current = by_role[current].parent_role


def _bone(
    role: GeneratedRole,
    name: str,
    semantic_key: str,
    side: str,
    mode: str,
    usage: str,
    layer: GeneratedLayer,
    parent_role: GeneratedRole | None,
    connected: bool,
    deform: bool,
    head: tuple[float, float, float],
    tail: tuple[float, float, float],
) -> GeneratedBoneSpec:
    return GeneratedBoneSpec(
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


DEFAULT_CORE_TOPOLOGY: tuple[GeneratedBoneSpec, ...] = (
    _bone(
        GeneratedRole.ROOT,
        "Root",
        "awb.root",
        "CENTER",
        "NEUTRAL",
        "PRIMARY",
        GeneratedLayer.AUTHORED,
        None,
        False,
        False,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.2),
    ),
    _bone(
        GeneratedRole.COM,
        "COM",
        "awb.com",
        "CENTER",
        "NEUTRAL",
        "PRIMARY",
        GeneratedLayer.AUTHORED,
        GeneratedRole.ROOT,
        False,
        False,
        (0.0, 0.0, 0.9),
        (0.0, 0.0, 1.1),
    ),
    _bone(
        GeneratedRole.PELVIS,
        "Pelvis",
        "awb.pelvis",
        "CENTER",
        "NEUTRAL",
        "PRIMARY",
        GeneratedLayer.AUTHORED,
        GeneratedRole.COM,
        False,
        False,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.18),
    ),
    _bone(
        GeneratedRole.FK_UPPER,
        "UpperArm.L",
        "awb.upper_arm",
        "LEFT",
        "FK",
        "PRIMARY",
        GeneratedLayer.AUTHORED,
        GeneratedRole.PELVIS,
        False,
        False,
        (0.12, 0.0, 1.42),
        (0.42, 0.0, 1.22),
    ),
    _bone(
        GeneratedRole.FK_FORE,
        "ForeArm.L",
        "awb.forearm",
        "LEFT",
        "FK",
        "PRIMARY",
        GeneratedLayer.AUTHORED,
        GeneratedRole.FK_UPPER,
        True,
        False,
        (0.42, 0.0, 1.22),
        (0.72, 0.0, 1.02),
    ),
    _bone(
        GeneratedRole.FK_HAND,
        "Hand.L",
        "awb.hand",
        "LEFT",
        "FK",
        "PRIMARY",
        GeneratedLayer.AUTHORED,
        GeneratedRole.FK_FORE,
        True,
        False,
        (0.72, 0.0, 1.02),
        (0.84, 0.0, 0.96),
    ),
    _bone(
        GeneratedRole.IK_EFFECTOR,
        "IK_Hand.L",
        "awb.hand",
        "LEFT",
        "IK",
        "TARGET",
        GeneratedLayer.AUTHORED,
        GeneratedRole.ROOT,
        False,
        False,
        (0.72, 0.0, 1.02),
        (0.84, 0.0, 0.96),
    ),
    _bone(
        GeneratedRole.IK_POLE,
        "IK_Elbow.L",
        "awb.arm",
        "LEFT",
        "IK",
        "POLE",
        GeneratedLayer.AUTHORED,
        GeneratedRole.ROOT,
        False,
        False,
        (0.42, -0.45, 1.22),
        (0.42, -0.55, 1.22),
    ),
    _bone(
        GeneratedRole.MCH_UPPER,
        "MCH_UpperArm.L",
        "awb.upper_arm",
        "LEFT",
        "NEUTRAL",
        "MECHANISM",
        GeneratedLayer.MECHANISM,
        GeneratedRole.PELVIS,
        False,
        False,
        (0.12, 0.0, 1.42),
        (0.42, 0.0, 1.22),
    ),
    _bone(
        GeneratedRole.MCH_FORE,
        "MCH_ForeArm.L",
        "awb.forearm",
        "LEFT",
        "NEUTRAL",
        "MECHANISM",
        GeneratedLayer.MECHANISM,
        GeneratedRole.MCH_UPPER,
        True,
        False,
        (0.42, 0.0, 1.22),
        (0.72, 0.0, 1.02),
    ),
    _bone(
        GeneratedRole.MCH_HAND,
        "MCH_Hand.L",
        "awb.hand",
        "LEFT",
        "NEUTRAL",
        "MECHANISM",
        GeneratedLayer.MECHANISM,
        GeneratedRole.MCH_FORE,
        True,
        False,
        (0.72, 0.0, 1.02),
        (0.84, 0.0, 0.96),
    ),
    _bone(
        GeneratedRole.EXPORT_ROOT,
        "EXP_Root",
        "awb.root",
        "CENTER",
        "NEUTRAL",
        "REFERENCE",
        GeneratedLayer.EXPORT,
        None,
        False,
        False,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.2),
    ),
    _bone(
        GeneratedRole.DEF_PELVIS,
        "DEF_Pelvis",
        "awb.pelvis",
        "CENTER",
        "NEUTRAL",
        "DEFORM",
        GeneratedLayer.DEFORM,
        GeneratedRole.EXPORT_ROOT,
        False,
        True,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.18),
    ),
    _bone(
        GeneratedRole.DEF_UPPER,
        "DEF_UpperArm.L",
        "awb.upper_arm",
        "LEFT",
        "NEUTRAL",
        "DEFORM",
        GeneratedLayer.DEFORM,
        GeneratedRole.DEF_PELVIS,
        False,
        True,
        (0.12, 0.0, 1.42),
        (0.42, 0.0, 1.22),
    ),
    _bone(
        GeneratedRole.DEF_FORE,
        "DEF_ForeArm.L",
        "awb.forearm",
        "LEFT",
        "NEUTRAL",
        "DEFORM",
        GeneratedLayer.DEFORM,
        GeneratedRole.DEF_UPPER,
        True,
        True,
        (0.42, 0.0, 1.22),
        (0.72, 0.0, 1.02),
    ),
    _bone(
        GeneratedRole.DEF_HAND,
        "DEF_Hand.L",
        "awb.hand",
        "LEFT",
        "NEUTRAL",
        "DEFORM",
        GeneratedLayer.DEFORM,
        GeneratedRole.DEF_FORE,
        True,
        True,
        (0.72, 0.0, 1.02),
        (0.84, 0.0, 0.96),
    ),
)
