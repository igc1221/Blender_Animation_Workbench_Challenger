from __future__ import annotations

import math
from dataclasses import replace

from .rigped_fit_state import (
    DerivedRestPart,
    FitDraft,
    FitOperation,
    FitPartDefinition,
    FitPartKind,
    FitPartValue,
    FitPartValueRecord,
    FitStateError,
    Quat,
    Vec3,
    derive_rest_parts,
    normalize_quaternion,
    rotate_vector,
    validate_fit_draft,
)


class FitCommandError(RuntimeError):
    pass


_ALL_OPS = (FitOperation.MOVE, FitOperation.ROTATE, FitOperation.SCALE)
_ROTATE_SCALE = (FitOperation.ROTATE, FitOperation.SCALE)


def fit_operations_for_role(
    semantic_key: str,
    *,
    parent_semantic_key: str | None = None,
) -> tuple[FitOperation, ...]:
    """Coarse Figure capabilities for the migrated F3 transform matrix."""

    key = str(semantic_key)
    if key in {
        "awb.com",
        "awb.pelvis",
        "awb.upper_arm",
        "awb.forearm",
        "awb.hand",
        "awb.thigh",
        "awb.calf",
        "awb.foot",
        "awb.neck",
        "awb.toe",
    }:
        return _ALL_OPS
    if key == "awb.spine" and parent_semantic_key == "awb.spine":
        return _ALL_OPS
    if key in {"awb.spine", "awb.head", "awb.clavicle", "awb.finger"}:
        return _ROTATE_SCALE
    return ()


def _vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (float(a[0] + b[0]), float(a[1] + b[1]), float(a[2] + b[2]))


def _vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (float(a[0] - b[0]), float(a[1] - b[1]), float(a[2] - b[2]))


def _vec_scale(value: Vec3, scale: float) -> Vec3:
    return (
        float(value[0] * scale),
        float(value[1] * scale),
        float(value[2] * scale),
    )


def _vec_dot(a: Vec3, b: Vec3) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _vec_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        float(a[1] * b[2] - a[2] * b[1]),
        float(a[2] * b[0] - a[0] * b[2]),
        float(a[0] * b[1] - a[1] * b[0]),
    )


def _vec_length(value: Vec3) -> float:
    return math.sqrt(sum(float(component) * float(component) for component in value))


def _vec_normalized(value: Vec3) -> Vec3:
    length = _vec_length(value)
    if length <= 1e-12:
        raise FitCommandError("FIT_F3_VECTOR_DEGENERATE")
    return _vec_scale(value, 1.0 / length)


def _quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_conjugate(value: Quat) -> Quat:
    return (float(value[0]), -float(value[1]), -float(value[2]), -float(value[3]))


def _rotation_between(source: Vec3, target: Vec3) -> Quat:
    a = _vec_normalized(source)
    b = _vec_normalized(target)
    dot = max(-1.0, min(1.0, _vec_dot(a, b)))
    if dot >= 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if dot <= -1.0 + 1e-9:
        candidate = (1.0, 0.0, 0.0) if abs(a[0]) < 0.8 else (0.0, 0.0, 1.0)
        axis = _vec_normalized(_vec_cross(a, candidate))
        return (0.0, axis[0], axis[1], axis[2])
    cross = _vec_cross(a, b)
    return normalize_quaternion((1.0 + dot, cross[0], cross[1], cross[2]))


def _ordered_definitions(draft: FitDraft) -> tuple[FitPartDefinition, ...]:
    pending = list(draft.document.parts)
    ordered: list[FitPartDefinition] = []
    resolved: set[str] = set()
    while pending:
        progressed = False
        for definition in tuple(pending):
            parent_id = definition.parent_part_id
            if parent_id is None or parent_id in resolved:
                ordered.append(definition)
                resolved.add(definition.part_id)
                pending.remove(definition)
                progressed = True
        if not progressed:
            raise FitCommandError("FIT_F3_TOPOLOGY_ORDER_INVALID")
    return tuple(ordered)


def _definition_map(draft: FitDraft) -> dict[str, FitPartDefinition]:
    return {part.part_id: part for part in draft.document.parts}


def _value_map(draft: FitDraft) -> dict[str, FitPartValue]:
    return {record.part_id: record.value for record in draft.values}


def _derived_map(draft: FitDraft) -> dict[str, DerivedRestPart]:
    return {part.part_id: part for part in derive_rest_parts(draft)}


def _children_map(draft: FitDraft) -> dict[str, tuple[str, ...]]:
    children: dict[str, list[str]] = {}
    for part in draft.document.parts:
        if part.parent_part_id is not None:
            children.setdefault(part.parent_part_id, []).append(part.part_id)
    return {key: tuple(value) for key, value in children.items()}


def _subtree_ids(draft: FitDraft, part_id: str) -> tuple[str, ...]:
    children = _children_map(draft)
    ordered: list[str] = []
    stack = [str(part_id)]
    while stack:
        current = stack.pop()
        ordered.append(current)
        stack.extend(reversed(children.get(current, ())))
    return tuple(ordered)


def _descendant_ids(draft: FitDraft, part_id: str) -> tuple[str, ...]:
    return _subtree_ids(draft, part_id)[1:]


def _parent_key(draft: FitDraft, definition: FitPartDefinition) -> str | None:
    if definition.parent_part_id is None:
        return None
    parent = _definition_map(draft).get(definition.parent_part_id)
    return None if parent is None else parent.semantic_key


def _role(draft: FitDraft, part_id: str) -> str:
    definition = _definition_map(draft).get(str(part_id))
    if definition is None:
        raise FitCommandError("FIT_F3_PART_MISSING")
    if (
        definition.semantic_key == "awb.spine"
        and _parent_key(draft, definition) == "awb.spine"
    ):
        return "spine2"
    return definition.semantic_key.removeprefix("awb.")


def _operation_supported(
    draft: FitDraft,
    part_id: str,
    operation: FitOperation,
) -> bool:
    definition = _definition_map(draft).get(str(part_id))
    return bool(definition is not None and operation in definition.allowed_operations)


def fit_move_supported(draft: FitDraft, part_id: str) -> bool:
    return _operation_supported(draft, part_id, FitOperation.MOVE)


def fit_rotate_supported(draft: FitDraft, part_id: str) -> bool:
    return _operation_supported(draft, part_id, FitOperation.ROTATE)


def fit_scale_supported(draft: FitDraft, part_id: str) -> bool:
    return _operation_supported(draft, part_id, FitOperation.SCALE)


def _absolute_pose_targets(
    draft: FitDraft,
    *,
    part_ids: tuple[str, ...],
    pivot: Vec3 | None = None,
    rotation: Quat | None = None,
    translation: Vec3 | None = None,
) -> dict[str, tuple[Vec3, Quat]]:
    derived = _derived_map(draft)
    out: dict[str, tuple[Vec3, Quat]] = {}
    for part_id in part_ids:
        rest = derived[part_id]
        head = rest.head
        orientation = rest.orientation
        if rotation is not None:
            origin = pivot if pivot is not None else (0.0, 0.0, 0.0)
            head = _vec_add(origin, rotate_vector(rotation, _vec_sub(head, origin)))
            orientation = normalize_quaternion(_quat_mul(rotation, orientation))
        if translation is not None:
            head = _vec_add(head, translation)
        out[part_id] = (head, orientation)
    return out


def _reencode_absolute_pose(
    draft: FitDraft,
    targets: dict[str, tuple[Vec3, Quat]],
    *,
    value_overrides: dict[str, FitPartValue] | None = None,
) -> FitDraft:
    if not targets and not value_overrides:
        return draft
    values = _value_map(draft)
    overrides = value_overrides or {}
    absolute: dict[str, tuple[Vec3, Quat]] = {}

    for definition in _ordered_definitions(draft):
        part_id = definition.part_id
        base_value = overrides.get(part_id, values[part_id])
        target = targets.get(part_id)

        if target is not None:
            head, orientation = target
            orientation = normalize_quaternion(orientation)
            if definition.parent_part_id is None:
                attachment = head
                local_orientation = orientation
            else:
                parent_head, parent_orientation = absolute[definition.parent_part_id]
                parent_inverse = _quat_conjugate(parent_orientation)
                attachment = rotate_vector(
                    parent_inverse,
                    _vec_sub(head, parent_head),
                )
                local_orientation = normalize_quaternion(
                    _quat_mul(parent_inverse, orientation)
                )
            base_value = replace(
                base_value,
                attachment_offset=attachment,
                local_orientation=local_orientation,
            )
            values[part_id] = base_value
            absolute[part_id] = (head, orientation)
            continue

        # Untargeted parts retain their authored parent-local canonical values
        # exactly. Only derive the absolute pose needed by later descendants.
        values[part_id] = base_value
        local_orientation = normalize_quaternion(base_value.local_orientation)
        if definition.parent_part_id is None:
            head = tuple(float(item) for item in base_value.attachment_offset)
            orientation = local_orientation
        else:
            parent_head, parent_orientation = absolute[definition.parent_part_id]
            head = _vec_add(
                parent_head,
                rotate_vector(parent_orientation, base_value.attachment_offset),
            )
            orientation = normalize_quaternion(
                _quat_mul(parent_orientation, local_orientation)
            )
        absolute[part_id] = (head, orientation)

    result = FitDraft(
        document=draft.document,
        values=tuple(
            FitPartValueRecord(part_id=record.part_id, value=values[record.part_id])
            for record in draft.values
        ),
    )
    try:
        validate_fit_draft(result)
    except FitStateError as exc:
        raise FitCommandError(str(exc)) from exc
    return result


def _replace_values(
    draft: FitDraft,
    replacements: dict[str, FitPartValue],
) -> FitDraft:
    if not replacements:
        return draft
    result = FitDraft(
        document=draft.document,
        values=tuple(
            FitPartValueRecord(
                part_id=record.part_id,
                value=replacements.get(record.part_id, record.value),
            )
            for record in draft.values
        ),
    )
    try:
        validate_fit_draft(result)
    except FitStateError as exc:
        raise FitCommandError(str(exc)) from exc
    return result


def _two_bone_move(
    draft: FitDraft,
    *,
    upper_id: str,
    fore_id: str,
    target_delta: Vec3,
    distal_root_id: str,
) -> FitDraft:
    derived = _derived_map(draft)
    upper = derived[upper_id]
    fore = derived[fore_id]
    shoulder = upper.head
    elbow = fore.head
    wrist = fore.tail
    upper_vec = _vec_sub(elbow, shoulder)
    fore_vec = _vec_sub(wrist, elbow)
    upper_length = _vec_length(upper_vec)
    fore_length = _vec_length(fore_vec)
    target_vec = _vec_sub(_vec_add(wrist, target_delta), shoulder)
    target_distance = _vec_length(target_vec)
    if min(upper_length, fore_length, target_distance) <= 1e-9:
        raise FitCommandError("FIT_F3_TWO_BONE_DEGENERATE")

    target_dir = _vec_normalized(target_vec)
    reach_margin = max(1e-5, (upper_length + fore_length) * 0.001)
    min_reach = abs(upper_length - fore_length) + reach_margin
    max_reach = max(min_reach, upper_length + fore_length - reach_margin)
    solved_distance = min(max(target_distance, min_reach), max_reach)
    new_wrist = _vec_add(shoulder, _vec_scale(target_dir, solved_distance))

    along = (
        upper_length * upper_length
        - fore_length * fore_length
        + solved_distance * solved_distance
    ) / (2.0 * solved_distance)
    bend_height = max(0.0, upper_length * upper_length - along * along) ** 0.5

    baseline_target = _vec_sub(wrist, shoulder)
    baseline_distance = _vec_length(baseline_target)
    if baseline_distance <= 1e-9:
        raise FitCommandError("FIT_F3_TWO_BONE_DEGENERATE")
    baseline_dir = _vec_normalized(baseline_target)
    baseline_along = (
        upper_length * upper_length
        - fore_length * fore_length
        + baseline_distance * baseline_distance
    ) / (2.0 * baseline_distance)
    baseline_perp = _vec_sub(
        elbow,
        _vec_add(shoulder, _vec_scale(baseline_dir, baseline_along)),
    )

    # Match the legacy gesture-start bend preference without depending on
    # EditBone/IK-pole runtime state. Authored legs are intentionally straight,
    # so their documented knee pole points toward rig-local -Y. Arms normally
    # carry a meaningful authored bend; if one becomes effectively straight,
    # use the upper segment's authored local Z as the stable seed, matching the
    # legacy no-pole fallback.
    chain_length = upper_length + fore_length
    bend_epsilon = max(1e-7, chain_length * 1e-4)
    if _vec_length(baseline_perp) <= bend_epsilon:
        fore_role = _role(draft, fore_id)
        if fore_role == "calf":
            straight_seed = (0.0, -1.0, 0.0)
        elif fore_role == "forearm":
            straight_seed = rotate_vector(upper.orientation, (0.0, 0.0, 1.0))
        else:
            straight_seed = (1.0, 0.0, 0.0)
        baseline_perp = _vec_sub(
            straight_seed,
            _vec_scale(
                baseline_dir,
                _vec_dot(straight_seed, baseline_dir),
            ),
        )
    if _vec_length(baseline_perp) <= bend_epsilon:
        fallback = (1.0, 0.0, 0.0)
        if abs(_vec_dot(fallback, baseline_dir)) > 0.9:
            fallback = (0.0, 0.0, 1.0)
        baseline_perp = _vec_sub(
            fallback,
            _vec_scale(
                baseline_dir,
                _vec_dot(fallback, baseline_dir),
            ),
        )
    if _vec_length(baseline_perp) <= bend_epsilon:
        raise FitCommandError("FIT_F3_BEND_PLANE_UNRESOLVED")

    bend_plane_normal = _vec_cross(baseline_dir, baseline_perp)
    if _vec_length(bend_plane_normal) <= bend_epsilon:
        raise FitCommandError("FIT_F3_BEND_PLANE_UNRESOLVED")
    bend_plane_normal = _vec_normalized(bend_plane_normal)
    bend_dir = _vec_cross(bend_plane_normal, target_dir)
    if _vec_length(bend_dir) <= 1e-7:
        bend_dir = _vec_sub(
            baseline_perp,
            _vec_scale(target_dir, _vec_dot(baseline_perp, target_dir)),
        )
    if _vec_length(bend_dir) <= 1e-7:
        raise FitCommandError("FIT_F3_BEND_PLANE_UNRESOLVED")
    bend_dir = _vec_normalized(bend_dir)
    new_elbow = _vec_add(
        _vec_add(shoulder, _vec_scale(target_dir, along)),
        _vec_scale(bend_dir, bend_height),
    )

    upper_rotation = _rotation_between(upper_vec, _vec_sub(new_elbow, shoulder))
    fore_rotation = _rotation_between(fore_vec, _vec_sub(new_wrist, new_elbow))
    targets: dict[str, tuple[Vec3, Quat]] = {
        upper_id: (
            shoulder,
            normalize_quaternion(_quat_mul(upper_rotation, upper.orientation)),
        ),
        fore_id: (
            new_elbow,
            normalize_quaternion(_quat_mul(fore_rotation, fore.orientation)),
        ),
    }

    for descendant_id in _subtree_ids(draft, distal_root_id):
        if descendant_id == fore_id:
            continue
        rest = derived[descendant_id]
        targets[descendant_id] = (
            _vec_add(
                new_wrist,
                rotate_vector(fore_rotation, _vec_sub(rest.head, wrist)),
            ),
            normalize_quaternion(_quat_mul(fore_rotation, rest.orientation)),
        )
    return _reencode_absolute_pose(draft, targets)


def move_fit_part_rig_local(
    draft: FitDraft,
    part_id: str,
    delta_rig: Vec3,
) -> FitDraft:
    part_id = str(part_id)
    delta = tuple(float(value) for value in delta_rig)
    if len(delta) != 3 or not all(math.isfinite(value) for value in delta):
        raise FitCommandError("FIT_F3_MOVE_DELTA_INVALID")
    if not fit_move_supported(draft, part_id):
        raise FitCommandError("FIT_F3_MOVE_UNSUPPORTED_PART")
    if _vec_length(delta) <= 1e-12:
        return draft

    role = _role(draft, part_id)
    definitions = _definition_map(draft)
    values = _value_map(draft)
    derived = _derived_map(draft)
    definition = definitions[part_id]
    rest = derived[part_id]

    if role == "com":
        local_delta = delta
        if definition.parent_part_id is not None:
            parent = derived[definition.parent_part_id]
            local_delta = rotate_vector(_quat_conjugate(parent.orientation), delta)
        return _replace_values(
            draft,
            {
                part_id: replace(
                    values[part_id],
                    attachment_offset=_vec_add(
                        values[part_id].attachment_offset,
                        local_delta,
                    ),
                )
            },
        )

    if role == "pelvis":
        return draft

    if role in {"neck", "toe"}:
        targets = _absolute_pose_targets(
            draft,
            part_ids=_subtree_ids(draft, part_id),
            translation=delta,
        )
        return _reencode_absolute_pose(
            draft,
            targets,
            value_overrides={
                part_id: replace(
                    values[part_id],
                    connected_override=False,
                )
            },
        )

    if role == "spine2":
        baseline = _vec_sub(rest.tail, rest.head)
        desired = _vec_sub(_vec_add(rest.tail, delta), rest.head)
        rotation = _rotation_between(baseline, desired)
        targets = _absolute_pose_targets(
            draft,
            part_ids=_subtree_ids(draft, part_id),
            pivot=rest.head,
            rotation=rotation,
        )
        return _reencode_absolute_pose(draft, targets)

    if role in {"upper_arm", "thigh"}:
        baseline = _vec_sub(rest.tail, rest.head)
        desired = _vec_sub(_vec_add(rest.tail, delta), rest.head)
        rotation = _rotation_between(baseline, desired)
        new_tail = _vec_add(
            rest.head,
            _vec_scale(_vec_normalized(desired), _vec_length(baseline)),
        )
        endpoint_delta = _vec_sub(new_tail, rest.tail)
        targets = {
            part_id: (
                rest.head,
                normalize_quaternion(_quat_mul(rotation, rest.orientation)),
            )
        }
        for descendant_id in _descendant_ids(draft, part_id):
            descendant = derived[descendant_id]
            targets[descendant_id] = (
                _vec_add(descendant.head, endpoint_delta),
                descendant.orientation,
            )
        return _reencode_absolute_pose(draft, targets)

    if role in {"forearm", "calf"}:
        if definition.parent_part_id is None:
            raise FitCommandError("FIT_F3_PARENT_MISSING")
        return _two_bone_move(
            draft,
            upper_id=definition.parent_part_id,
            fore_id=part_id,
            target_delta=delta,
            distal_root_id=part_id,
        )

    if role in {"hand", "foot"}:
        if definition.parent_part_id is None:
            raise FitCommandError("FIT_F3_PARENT_MISSING")
        fore_definition = definitions[definition.parent_part_id]
        if fore_definition.parent_part_id is None:
            raise FitCommandError("FIT_F3_PARENT_MISSING")
        return _two_bone_move(
            draft,
            upper_id=fore_definition.parent_part_id,
            fore_id=definition.parent_part_id,
            target_delta=delta,
            distal_root_id=part_id,
        )

    raise FitCommandError("FIT_F3_MOVE_UNSUPPORTED_PART")


def rotate_fit_part_rig_local(
    draft: FitDraft,
    part_id: str,
    delta_quaternion_rig: Quat,
) -> FitDraft:
    part_id = str(part_id)
    if not fit_rotate_supported(draft, part_id):
        raise FitCommandError("FIT_F3_ROTATE_UNSUPPORTED_PART")
    raw_delta = tuple(float(value) for value in delta_quaternion_rig)
    if len(raw_delta) != 4 or not all(math.isfinite(value) for value in raw_delta):
        raise FitCommandError("FIT_F3_ROTATE_DELTA_INVALID")
    try:
        delta = normalize_quaternion(raw_delta)
    except FitStateError as exc:
        raise FitCommandError("FIT_F3_ROTATE_DELTA_INVALID") from exc
    if _vec_length((delta[1], delta[2], delta[3])) <= 1e-12:
        return draft

    role = _role(draft, part_id)
    derived = _derived_map(draft)
    rest = derived[part_id]
    subtree = _subtree_ids(draft, part_id)

    if role == "com":
        # Preserve the frozen F3b-1 authority contract: only COM canonical
        # fields change; descendants follow through ordinary derivation.
        definition = _definition_map(draft)[part_id]
        values = _value_map(draft)
        pivot = _vec_scale(_vec_add(rest.head, rest.tail), 0.5)
        new_head = _vec_add(
            pivot,
            rotate_vector(delta, _vec_sub(rest.head, pivot)),
        )
        new_absolute_orientation = normalize_quaternion(
            _quat_mul(delta, rest.orientation)
        )
        if definition.parent_part_id is None:
            new_attachment = new_head
            new_local_orientation = new_absolute_orientation
        else:
            parent = derived[definition.parent_part_id]
            parent_inverse = _quat_conjugate(parent.orientation)
            new_attachment = rotate_vector(
                parent_inverse,
                _vec_sub(new_head, parent.head),
            )
            new_local_orientation = normalize_quaternion(
                _quat_mul(parent_inverse, new_absolute_orientation)
            )
        return _replace_values(
            draft,
            {
                part_id: replace(
                    values[part_id],
                    attachment_offset=new_attachment,
                    local_orientation=new_local_orientation,
                )
            },
        )

    if role == "pelvis":
        pivot = _vec_scale(_vec_add(rest.head, rest.tail), 0.5)
        targets = {
            part_id: (
                _vec_add(pivot, rotate_vector(delta, _vec_sub(rest.head, pivot))),
                normalize_quaternion(_quat_mul(delta, rest.orientation)),
            )
        }
        for descendant_id in _descendant_ids(draft, part_id):
            descendant = derived[descendant_id]
            targets[descendant_id] = (descendant.head, descendant.orientation)
        return _reencode_absolute_pose(draft, targets)

    targets = _absolute_pose_targets(
        draft,
        part_ids=subtree,
        pivot=rest.head,
        rotation=delta,
    )
    return _reencode_absolute_pose(draft, targets)


def scale_fit_part_local(
    draft: FitDraft,
    part_id: str,
    axes: str,
    factor: float,
) -> FitDraft:
    part_id = str(part_id)
    if not fit_scale_supported(draft, part_id):
        raise FitCommandError("FIT_F3_SCALE_UNSUPPORTED_PART")
    axis_set = set(str(axes).upper())
    if not axis_set or not axis_set.issubset({"X", "Y", "Z"}):
        raise FitCommandError("FIT_F3_SCALE_AXIS_INVALID")
    scale = float(factor)
    if not math.isfinite(scale) or scale <= 0.0:
        raise FitCommandError("FIT_F3_SCALE_FACTOR_INVALID")
    if abs(scale - 1.0) <= 1e-12:
        return draft

    result = draft
    for axis in ("X", "Y", "Z"):
        if axis in axis_set:
            result = _scale_one_axis(result, part_id, axis, scale)
    return result


def _scale_one_axis(
    draft: FitDraft,
    part_id: str,
    axis: str,
    factor: float,
) -> FitDraft:
    definitions = _definition_map(draft)
    values = _value_map(draft)
    derived = _derived_map(draft)
    definition = definitions[part_id]
    value = values[part_id]
    rest = derived[part_id]
    role = _role(draft, part_id)

    if axis == "X":
        result = _replace_values(
            draft,
            {part_id: replace(value, width=max(1e-6, float(value.width) * factor))},
        )
        if role != "pelvis":
            return result

        result_derived = _derived_map(result)
        pelvis = result_derived[part_id]
        center = _vec_scale(_vec_add(pelvis.head, pelvis.tail), 0.5)
        x_axis = rotate_vector(pelvis.orientation, (1.0, 0.0, 0.0))
        targets: dict[str, tuple[Vec3, Quat]] = {}
        for child_id in _children_map(result).get(part_id, ()):
            child_definition = definitions[child_id]
            if child_definition.semantic_key != "awb.thigh":
                continue
            child = result_derived[child_id]
            offset = _vec_dot(_vec_sub(child.head, center), x_axis)
            lateral_delta = _vec_scale(x_axis, offset * (factor - 1.0))
            targets.update(
                _absolute_pose_targets(
                    result,
                    part_ids=_subtree_ids(result, child_id),
                    translation=lateral_delta,
                )
            )
        return _reencode_absolute_pose(result, targets)

    if axis == "Z":
        return _replace_values(
            draft,
            {
                part_id: replace(
                    value,
                    depth=max(1e-6, float(value.depth) * factor),
                )
            },
        )

    if definition.kind is FitPartKind.FRAME:
        old_effective_length = float(definition.carrier_length) * float(
            value.frame_length_scale
        )
        new_value = replace(
            value,
            frame_length_scale=max(
                0.02,
                float(value.frame_length_scale) * factor,
            ),
        )
        new_effective_length = float(definition.carrier_length) * float(
            new_value.frame_length_scale
        )
    else:
        if value.length is None:
            raise FitCommandError("FIT_F3_SCALE_LENGTH_MISSING")
        old_effective_length = float(value.length)
        new_value = replace(
            value,
            length=max(1e-6, float(value.length) * factor),
        )
        new_effective_length = float(new_value.length)

    if role == "pelvis":
        return _replace_values(draft, {part_id: new_value})

    direction = rotate_vector(rest.orientation, (0.0, 1.0, 0.0))
    tail_delta = _vec_scale(
        direction,
        new_effective_length - old_effective_length,
    )
    if definition.kind is FitPartKind.FRAME:
        half_delta = _vec_scale(tail_delta, 0.5)
        targets = {
            part_id: (
                _vec_sub(rest.head, half_delta),
                rest.orientation,
            )
        }
        targets.update(
            _absolute_pose_targets(
                draft,
                part_ids=_descendant_ids(draft, part_id),
                translation=half_delta,
            )
        )
        return _reencode_absolute_pose(
            draft,
            targets,
            value_overrides={part_id: new_value},
        )

    targets = _absolute_pose_targets(
        draft,
        part_ids=_descendant_ids(draft, part_id),
        translation=tail_delta,
    )
    return _reencode_absolute_pose(
        draft,
        targets,
        value_overrides={part_id: new_value},
    )
