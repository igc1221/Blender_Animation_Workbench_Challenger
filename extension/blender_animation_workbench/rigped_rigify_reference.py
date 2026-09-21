from __future__ import annotations

from dataclasses import replace

from .rigped_humanoid_spec import HumanoidBoneSpec, RigpedHumanoidSpec

# Reference proportions sampled from Blender 5.2.1 Rigify Basic Human metarig.
# Rigify is reference material only: no Rigify bones, constraints, controls,
# widgets, rig types, or generated-rig data are present in the returned AWB spec.
_REFERENCE_GEOMETRY: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "ROOT": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.20)),
    "COM": ((0.0, 0.0124, 1.0320), (0.0, 0.0124, 1.1120)),
    "PELVIS": ((0.0, 0.0314, 0.9983), (0.0, -0.0066, 1.1457)),
    "SPINE": ((0.0, 0.0172, 1.1573), (0.0, 0.0004, 1.2929)),
    "SPINE2": ((0.0, 0.0004, 1.2929), (0.0, 0.0114, 1.6582)),
    "NECK": ((0.0, 0.0114, 1.6582), (0.0, -0.0130, 1.7197)),
    "HEAD": ((0.0, -0.0130, 1.7197), (0.0, -0.0247, 1.9796)),
    "CLAVICLE.L": ((0.0183, -0.0684, 1.6051), (0.1694, 0.0205, 1.6050)),
    "UPPER_ARM.L": ((0.1953, 0.0267, 1.5846), (0.4424, 0.0885, 1.4491)),
    "FOREARM.L": ((0.4424, 0.0885, 1.4491), (0.6594, 0.0492, 1.3061)),
    "HAND.L": ((0.6594, 0.0492, 1.3061), (0.7234, 0.0412, 1.2585)),
    "THIGH.L": ((0.0980, 0.0124, 1.0720), (0.0980, -0.0286, 0.5372)),
    "CALF.L": ((0.0980, -0.0286, 0.5372), (0.0980, 0.0162, 0.0852)),
    "FOOT.L": ((0.0980, 0.0162, 0.0852), (0.0980, -0.0934, 0.0167)),
    "TOE.L": ((0.0980, -0.0934, 0.0167), (0.0980, -0.1606, 0.0167)),
}


def _mirror(point: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-float(point[0]), float(point[1]), float(point[2]))


for _stem in ("CLAVICLE", "UPPER_ARM", "FOREARM", "HAND", "THIGH", "CALF", "FOOT", "TOE"):
    _left = _REFERENCE_GEOMETRY[f"{_stem}.L"]
    _REFERENCE_GEOMETRY[f"{_stem}.R"] = (_mirror(_left[0]), _mirror(_left[1]))


_ROLE_SOURCE: dict[str, str] = {
    "IK_HAND.L": "HAND.L",
    "IK_HAND.R": "HAND.R",
    "IK_FOOT.L": "TOE.L",
    "IK_FOOT.R": "TOE.R",
    "MCH_UPPER_ARM.L": "UPPER_ARM.L",
    "MCH_UPPER_ARM.R": "UPPER_ARM.R",
    "MCH_FOREARM.L": "FOREARM.L",
    "MCH_FOREARM.R": "FOREARM.R",
    "MCH_HAND.L": "HAND.L",
    "MCH_HAND.R": "HAND.R",
    "MCH_THIGH.L": "THIGH.L",
    "MCH_THIGH.R": "THIGH.R",
    "MCH_CALF.L": "CALF.L",
    "MCH_CALF.R": "CALF.R",
    "MCH_FOOT.L": "FOOT.L",
    "MCH_FOOT.R": "FOOT.R",
    "MCH_TOE.L": "TOE.L",
    "MCH_TOE.R": "TOE.R",
    "MCH_CONTACT_HAND.L": "HAND.L",
    "MCH_CONTACT_HAND.R": "HAND.R",
    "MCH_CONTACT_POINT_HAND.L": "HAND.L",
    "MCH_CONTACT_POINT_HAND.R": "HAND.R",
    "MCH_CONTACT_FOOT.L": "TOE.L",
    "MCH_CONTACT_FOOT.R": "TOE.R",
    "MCH_CONTACT_POINT_FOOT.L": "TOE.L",
    "MCH_CONTACT_POINT_FOOT.R": "TOE.R",
    "EXPORT_ROOT": "ROOT",
    "DEF_PELVIS": "PELVIS",
    "DEF_SPINE": "SPINE",
    "DEF_SPINE2": "SPINE2",
    "DEF_NECK": "NECK",
    "DEF_HEAD": "HEAD",
    "DEF_CLAVICLE.L": "CLAVICLE.L",
    "DEF_CLAVICLE.R": "CLAVICLE.R",
    "DEF_UPPER_ARM.L": "UPPER_ARM.L",
    "DEF_UPPER_ARM.R": "UPPER_ARM.R",
    "DEF_FOREARM.L": "FOREARM.L",
    "DEF_FOREARM.R": "FOREARM.R",
    "DEF_HAND.L": "HAND.L",
    "DEF_HAND.R": "HAND.R",
    "DEF_THIGH.L": "THIGH.L",
    "DEF_THIGH.R": "THIGH.R",
    "DEF_CALF.L": "CALF.L",
    "DEF_CALF.R": "CALF.R",
    "DEF_FOOT.L": "FOOT.L",
    "DEF_FOOT.R": "FOOT.R",
    "DEF_TOE.L": "TOE.L",
    "DEF_TOE.R": "TOE.R",
}


def _finger_geometry(
    role: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    side = role.rsplit(".", 1)[1]
    hand_head, hand_tail = _REFERENCE_GEOMETRY[f"HAND.{side}"]
    direction = tuple(float(hand_tail[i] - hand_head[i]) for i in range(3))
    finger_tail = tuple(float(hand_tail[i] + (direction[i] * 0.75)) for i in range(3))
    return hand_tail, finger_tail


def _pole_geometry(
    role: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if role.startswith("POLE_ELBOW."):
        side = role.rsplit(".", 1)[1]
        joint = _REFERENCE_GEOMETRY[f"UPPER_ARM.{side}"][1]
        head = (joint[0], joint[1] - 0.42, joint[2])
        return head, (head[0], head[1] - 0.10, head[2])
    side = role.rsplit(".", 1)[1]
    joint = _REFERENCE_GEOMETRY[f"THIGH.{side}"][1]
    head = (joint[0], joint[1] - 0.42, joint[2])
    return head, (head[0], head[1] - 0.10, head[2])


def _geometry_for_role(
    role: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if role.startswith(("POLE_ELBOW.", "POLE_KNEE.")):
        return _pole_geometry(role)
    if role.startswith("DEF_FINGER."):
        return _finger_geometry(role.removeprefix("DEF_"))
    if role.startswith("FINGER."):
        return _finger_geometry(role)
    source = _ROLE_SOURCE.get(role, role)
    return _REFERENCE_GEOMETRY.get(source)


def rigify_reference_humanoid_spec(
    base: RigpedHumanoidSpec | None = None,
) -> RigpedHumanoidSpec:
    """Return the normal AWB topology with Rigify-referenced human proportions."""

    source = base or RigpedHumanoidSpec()
    source.validate()

    remapped: list[HumanoidBoneSpec] = []
    for bone in source.resolved_bones():
        geometry = _geometry_for_role(bone.role)
        if geometry is None:
            remapped.append(bone)
            continue
        remapped.append(replace(bone, head=geometry[0], tail=geometry[1]))

    result = replace(source, bones=tuple(remapped))
    result.validate()
    return result
