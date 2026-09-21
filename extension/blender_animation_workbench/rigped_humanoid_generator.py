from __future__ import annotations

from dataclasses import dataclass

from .rigped_humanoid_spec import (
    DEFAULT_HUMANOID_BONES,
    DEFAULT_HUMANOID_CHAINS,
    DEFAULT_HUMANOID_CONSTRAINTS,
    DEFAULT_HUMANOID_GROUPS,
    DEFAULT_HUMANOID_KINEMATICS,
    DEFAULT_HUMANOID_OPPOSITES,
    RigpedHumanoidSpec,
)


@dataclass(frozen=True, slots=True)
class RigpedHumanoidParameters:
    """Pure-data structural parameters for generated Rigped humanoids.

    B1 deliberately supports only the current canonical topology. Later B-series
    stages may widen the accepted values, but callers already have one stable
    parameter object and one parameter-to-spec generation boundary.
    """

    spine_segments: int = 2
    neck_segments: int = 1
    toe_segments: int = 1
    finger_segments: int = 1

    def validate(self) -> None:
        if self.spine_segments != 2:
            raise ValueError("B1 supports exactly 2 spine segments.")
        if self.neck_segments != 1:
            raise ValueError("B1 supports exactly 1 neck segment.")
        if self.toe_segments != 1:
            raise ValueError("B1 supports exactly 1 toe segment per side.")
        if self.finger_segments != 1:
            raise ValueError("B1 supports exactly 1 finger segment per side.")


def generate_rigped_humanoid_spec(
    parameters: RigpedHumanoidParameters | None = None,
) -> RigpedHumanoidSpec:
    """Materialize one deterministic RigpedHumanoidSpec from pure-data parameters.

    RigpedHumanoidSpec remains the only object consumed by the existing builder.
    This function performs no Blender/runtime mutation and intentionally exposes
    one default finger link per hand and no variable limb-IK topology in B1.
    """

    resolved_parameters = parameters or RigpedHumanoidParameters()
    resolved_parameters.validate()

    spec = RigpedHumanoidSpec(
        bones=DEFAULT_HUMANOID_BONES,
        constraints=DEFAULT_HUMANOID_CONSTRAINTS,
        groups=DEFAULT_HUMANOID_GROUPS,
        chains=DEFAULT_HUMANOID_CHAINS,
        opposites=DEFAULT_HUMANOID_OPPOSITES,
        kinematics=DEFAULT_HUMANOID_KINEMATICS,
    )
    spec.validate()
    return spec
