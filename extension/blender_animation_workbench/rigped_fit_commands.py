from __future__ import annotations

import math
from dataclasses import replace

from .rigped_fit_state import (
    FitDraft,
    FitOperation,
    FitPartValue,
    Vec3,
    derive_rest_parts,
    replace_draft_value,
    rotate_vector,
)


class FitCommandError(RuntimeError):
    pass


def _vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (float(a[0] + b[0]), float(a[1] + b[1]), float(a[2] + b[2]))


def _vec_length(value: Vec3) -> float:
    return math.sqrt(sum(float(component) * float(component) for component in value))


def _quat_conjugate(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (float(value[0]), -float(value[1]), -float(value[2]), -float(value[3]))


def fit_move_supported(draft: FitDraft, part_id: str) -> bool:
    definition = next(
        (part for part in draft.document.parts if part.part_id == str(part_id)),
        None,
    )
    return bool(
        definition is not None
        and FitOperation.MOVE in definition.allowed_operations
    )


def move_fit_part_rig_local(
    draft: FitDraft,
    part_id: str,
    delta_rig: Vec3,
) -> FitDraft:
    """Return a preview draft with one deterministic Figure Move applied.

    F3 starts with COM Move only. COM is an unconnected semantic frame whose
    descendants are the body branches, so changing only COM's attachment offset
    translates the whole Figure hierarchy while keeping every child-local value
    unchanged. Unsupported parts fail closed until their explicit Figure command
    semantics are migrated from the legacy EditBone path.
    """

    part_id = str(part_id)
    delta = tuple(float(value) for value in delta_rig)
    if len(delta) != 3 or not all(math.isfinite(value) for value in delta):
        raise FitCommandError("FIT_F3_MOVE_DELTA_INVALID")

    definition = next(
        (part for part in draft.document.parts if part.part_id == part_id),
        None,
    )
    if definition is None:
        raise FitCommandError("FIT_F3_PART_MISSING")
    if not fit_move_supported(draft, part_id):
        raise FitCommandError("FIT_F3_MOVE_UNSUPPORTED_PART")

    record = next((item for item in draft.values if item.part_id == part_id), None)
    if record is None:
        raise FitCommandError("FIT_F3_VALUE_MISSING")
    if _vec_length(delta) <= 1e-12:
        return draft

    local_delta: Vec3 = delta
    if definition.parent_part_id is not None:
        derived = {part.part_id: part for part in derive_rest_parts(draft)}
        parent = derived.get(definition.parent_part_id)
        if parent is None:
            raise FitCommandError("FIT_F3_PARENT_MISSING")
        local_delta = rotate_vector(_quat_conjugate(parent.orientation), delta)

    value: FitPartValue = replace(
        record.value,
        attachment_offset=_vec_add(record.value.attachment_offset, local_delta),
    )
    return replace_draft_value(draft, part_id, value)
