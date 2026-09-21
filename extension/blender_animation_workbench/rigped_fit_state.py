from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z)


class FitOperation(StrEnum):
    MOVE = "MOVE"
    ROTATE = "ROTATE"
    SCALE = "SCALE"


class FitPartKind(StrEnum):
    BONE = "BONE"
    FRAME = "FRAME"


FIT_BONE_FRAME_CONVENTION = "BLENDER_REST_FRAME_Y_ALONG_BONE_XZ_ENCODE_ROLL_V1"
FIT_REST_ROUNDTRIP_TOLERANCE = 5e-6
FIT_STRUCTURE_FIELDS = ("head", "tail", "orientation")
FIT_APPEARANCE_FIELDS = ("width", "depth")


@dataclass(frozen=True, slots=True)
class FitSessionBaseline:
    character_id: str
    profile_id: str
    schema_version: int
    topology_fingerprint: str
    setup_revision: int
    setup_signature: str
    native_rest_digest: str
    appearance_digest: str
    owner_data_identity: tuple[str, ...]
    animation_footprint_digest: str


@dataclass(frozen=True, slots=True)
class FitPartDefinition:
    part_id: str
    binding_id: str
    semantic_key: str
    side: str
    parent_part_id: str | None
    connected: bool
    kind: FitPartKind = FitPartKind.BONE
    carrier_length: float | None = None
    allowed_operations: tuple[FitOperation, ...] = ()
    owned_derived_ids: tuple[str, ...] = ()
    name_hint: str = ""


@dataclass(frozen=True, slots=True)
class FitPartValue:
    attachment_offset: Vec3
    local_orientation: Quat
    length: float | None
    width: float
    depth: float


@dataclass(frozen=True, slots=True)
class FitPartValueRecord:
    part_id: str
    value: FitPartValue


@dataclass(frozen=True, slots=True)
class FitDocument:
    baseline: FitSessionBaseline
    parts: tuple[FitPartDefinition, ...]
    baseline_values: tuple[FitPartValueRecord, ...]


@dataclass(frozen=True, slots=True)
class FitDraft:
    document: FitDocument
    values: tuple[FitPartValueRecord, ...]


@dataclass(frozen=True, slots=True)
class FitRestPartSnapshot:
    """Plain committed-rest snapshot for the Blender-independent F1 model.

    Positions are Rigped-local. orientation is Blender's normalized rest-bone
    frame: local +Y runs head-to-tail and local X/Z encode roll as in
    Bone.matrix_local. width/depth map to native B-Bone X/Z display dimensions.

    FRAME is for semantic frames such as COM. Its native carrier-bone length is
    frozen into the part definition and is not an editable draft degree of freedom.
    Object/world transform support is deliberately owned by the later runtime adapter.
    """

    binding_id: str
    semantic_key: str
    side: str
    parent_binding_id: str | None
    connected: bool
    head: Vec3
    tail: Vec3
    orientation: Quat
    width: float
    depth: float
    kind: FitPartKind = FitPartKind.BONE
    allowed_operations: tuple[FitOperation, ...] = ()
    owned_derived_ids: tuple[str, ...] = ()
    name_hint: str = ""


@dataclass(frozen=True, slots=True)
class DerivedRestPart:
    part_id: str
    parent_part_id: str | None
    connected: bool
    head: Vec3
    tail: Vec3
    orientation: Quat
    width: float
    depth: float


@dataclass(frozen=True, slots=True)
class RestMutation:
    part_id: str
    before: DerivedRestPart
    derived: DerivedRestPart
    changed_fields: tuple[str, ...]
    structure_changed: bool
    appearance_changed: bool


@dataclass(frozen=True, slots=True)
class RestMutationPlan:
    topology_fingerprint: str
    mutations: tuple[RestMutation, ...]
    structure_changed: bool
    appearance_changed: bool
    candidate_rest_digest: str
    candidate_appearance_digest: str


class FitStateError(ValueError):
    pass


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_scale(v: Vec3, scale: float) -> Vec3:
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def _v_length(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _v_close(a: Vec3, b: Vec3, *, tolerance: float = 1e-6) -> bool:
    return _v_length(_v_sub(a, b)) <= tolerance


def _q_norm(q: Quat) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in q))


def normalize_quaternion(q: Quat) -> Quat:
    if not _finite(q):
        raise FitStateError("FIT_ORIENTATION_NONFINITE")
    norm = _q_norm(q)
    if norm <= 1e-12:
        raise FitStateError("FIT_ORIENTATION_ZERO")
    return tuple(float(value) / norm for value in q)  # type: ignore[return-value]


def _q_conjugate(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def _q_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rotate_vector(q: Quat, vector: Vec3) -> Vec3:
    qn = normalize_quaternion(q)
    rotated = _q_mul(_q_mul(qn, (0.0, *vector)), _q_conjugate(qn))
    return (rotated[1], rotated[2], rotated[3])


def _canonical_quaternion(q: Quat) -> Quat:
    qn = normalize_quaternion(q)
    for value in qn:
        if abs(value) <= 1e-12:
            continue
        if value < 0.0:
            return tuple(-item for item in qn)  # type: ignore[return-value]
        break
    return qn


def _quaternion_close(a: Quat, b: Quat, *, tolerance: float = 1e-6) -> bool:
    qa = normalize_quaternion(a)
    qb = normalize_quaternion(b)
    dot = abs(sum(x * y for x, y in zip(qa, qb, strict=True)))
    return abs(1.0 - min(1.0, dot)) <= tolerance


def _definition_payload(part: FitPartDefinition) -> tuple:
    return (
        part.part_id,
        part.binding_id,
        part.semantic_key,
        part.side,
        part.parent_part_id,
        bool(part.connected),
        str(part.kind),
        None if part.carrier_length is None else round(float(part.carrier_length), 9),
        tuple(str(operation) for operation in part.allowed_operations),
        tuple(part.owned_derived_ids),
    )


def topology_fingerprint(parts: tuple[FitPartDefinition, ...]) -> str:
    payload = tuple(sorted((_definition_payload(part) for part in parts), key=repr))
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _topological_parts(parts: tuple[FitPartDefinition, ...]) -> tuple[FitPartDefinition, ...]:
    by_id = {part.part_id: part for part in parts}
    ordered: list[FitPartDefinition] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(part_id: str) -> None:
        if part_id in visited:
            return
        if part_id in visiting:
            raise FitStateError("FIT_TOPOLOGY_CYCLE")
        part = by_id.get(part_id)
        if part is None:
            raise FitStateError("FIT_PART_MISSING")
        visiting.add(part_id)
        if part.parent_part_id is not None:
            if part.parent_part_id not in by_id:
                raise FitStateError("FIT_PARENT_MISSING")
            visit(part.parent_part_id)
        visiting.remove(part_id)
        visited.add(part_id)
        ordered.append(part)

    for part in parts:
        visit(part.part_id)
    return tuple(ordered)


def _value_map(records: tuple[FitPartValueRecord, ...]) -> dict[str, FitPartValue]:
    values: dict[str, FitPartValue] = {}
    for record in records:
        if record.part_id in values:
            raise FitStateError("FIT_VALUE_DUPLICATE")
        values[record.part_id] = record.value
    return values


def validate_fit_draft(draft: FitDraft) -> None:
    document = draft.document
    parts = document.parts
    if not parts:
        raise FitStateError("FIT_DOCUMENT_EMPTY")

    part_ids = [part.part_id for part in parts]
    if len(part_ids) != len(set(part_ids)):
        raise FitStateError("FIT_PART_ID_DUPLICATE")
    binding_ids = [part.binding_id for part in parts]
    if len(binding_ids) != len(set(binding_ids)):
        raise FitStateError("FIT_BINDING_ID_DUPLICATE")

    for part in parts:
        if not part.part_id or not part.binding_id:
            raise FitStateError("FIT_STABLE_ID_MISSING")
        if part.part_id != part.binding_id:
            raise FitStateError("FIT_PART_ID_MUST_USE_BINDING_ID")
        if part.connected and part.parent_part_id is None:
            raise FitStateError("FIT_CONNECTED_ROOT_INVALID")
        if part.kind is FitPartKind.FRAME:
            if part.carrier_length is None or not math.isfinite(float(part.carrier_length)):
                raise FitStateError("FIT_FRAME_CARRIER_LENGTH_INVALID")
            if float(part.carrier_length) <= 0.0:
                raise FitStateError("FIT_FRAME_CARRIER_LENGTH_INVALID")
        elif part.carrier_length is not None:
            raise FitStateError("FIT_BONE_CARRIER_LENGTH_UNEXPECTED")

    ordered = _topological_parts(parts)
    values = _value_map(draft.values)
    if set(values) != set(part_ids):
        raise FitStateError("FIT_VALUE_SET_MISMATCH")

    for part in ordered:
        value = values[part.part_id]
        numeric_values = (*value.attachment_offset, *value.local_orientation, value.width, value.depth)
        if value.length is not None:
            numeric_values = (*numeric_values, value.length)
        if not _finite(numeric_values):
            raise FitStateError("FIT_VALUE_NONFINITE")
        if part.kind is FitPartKind.FRAME:
            if value.length is not None:
                raise FitStateError("FIT_FRAME_LENGTH_MUST_BE_DERIVED")
        else:
            if value.length is None or value.length <= 0.0:
                raise FitStateError("FIT_LENGTH_NONPOSITIVE")
        if value.width <= 0.0 or value.depth <= 0.0:
            raise FitStateError("FIT_APPEARANCE_NONPOSITIVE")
        if abs(_q_norm(value.local_orientation) - 1.0) > 1e-6:
            raise FitStateError("FIT_ORIENTATION_NOT_NORMALIZED")

    derived = derive_rest_parts(draft, _skip_validation=True)
    derived_map = {part.part_id: part for part in derived}
    for part in ordered:
        if not part.connected or part.parent_part_id is None:
            continue
        child = derived_map[part.part_id]
        parent = derived_map[part.parent_part_id]
        if not _v_close(
            child.head,
            parent.tail,
            tolerance=FIT_REST_ROUNDTRIP_TOLERANCE,
        ):
            raise FitStateError("FIT_CONNECTED_ATTACHMENT_MISMATCH")

    if topology_fingerprint(parts) != document.baseline.topology_fingerprint:
        raise FitStateError("FIT_TOPOLOGY_FINGERPRINT_CHANGED")


def derive_rest_parts(
    draft: FitDraft,
    *,
    _skip_validation: bool = False,
) -> tuple[DerivedRestPart, ...]:
    if not _skip_validation:
        validate_fit_draft(draft)
    parts = _topological_parts(draft.document.parts)
    values = _value_map(draft.values)
    derived: dict[str, DerivedRestPart] = {}
    ordered: list[DerivedRestPart] = []

    for part in parts:
        value = values[part.part_id]
        local_orientation = normalize_quaternion(value.local_orientation)
        if part.parent_part_id is None:
            head = tuple(float(item) for item in value.attachment_offset)
            orientation = local_orientation
        else:
            parent = derived[part.parent_part_id]
            head = _v_add(parent.head, rotate_vector(parent.orientation, value.attachment_offset))
            orientation = normalize_quaternion(_q_mul(parent.orientation, local_orientation))
        effective_length = (
            float(part.carrier_length)
            if part.kind is FitPartKind.FRAME
            else float(value.length)
        )
        tail = _v_add(head, rotate_vector(orientation, (0.0, effective_length, 0.0)))
        item = DerivedRestPart(
            part_id=part.part_id,
            parent_part_id=part.parent_part_id,
            connected=part.connected,
            head=head,
            tail=tail,
            orientation=orientation,
            width=float(value.width),
            depth=float(value.depth),
        )
        derived[part.part_id] = item
        ordered.append(item)

    return tuple(ordered)


def _digest_rest(parts: tuple[DerivedRestPart, ...]) -> str:
    payload = tuple(
        (
            part.part_id,
            tuple(round(float(value), 9) for value in part.head),
            tuple(round(float(value), 9) for value in part.tail),
            tuple(round(float(value), 9) for value in _canonical_quaternion(part.orientation)),
            bool(part.connected),
        )
        for part in parts
    )
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _digest_appearance(parts: tuple[DerivedRestPart, ...]) -> str:
    payload = tuple(
        (
            part.part_id,
            round(float(part.width), 9),
            round(float(part.depth), 9),
        )
        for part in parts
    )
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def extract_fit_document(
    *,
    character_id: str,
    profile_id: str,
    schema_version: int,
    setup_revision: int,
    setup_signature: str,
    owner_data_identity: tuple[str, ...],
    animation_footprint_digest: str,
    snapshots: tuple[FitRestPartSnapshot, ...],
) -> tuple[FitDocument, FitDraft]:
    if not snapshots:
        raise FitStateError("FIT_SNAPSHOT_EMPTY")

    by_id: dict[str, FitRestPartSnapshot] = {}
    for snapshot in snapshots:
        part_id = str(snapshot.binding_id)
        if not part_id:
            raise FitStateError("FIT_STABLE_ID_MISSING")
        if part_id in by_id:
            raise FitStateError("FIT_BINDING_ID_DUPLICATE")
        by_id[part_id] = snapshot

    definitions = tuple(
        FitPartDefinition(
            part_id=str(snapshot.binding_id),
            binding_id=str(snapshot.binding_id),
            semantic_key=str(snapshot.semantic_key),
            side=str(snapshot.side),
            parent_part_id=(str(snapshot.parent_binding_id) if snapshot.parent_binding_id else None),
            connected=bool(snapshot.connected),
            kind=snapshot.kind,
            carrier_length=(
                _v_length(_v_sub(snapshot.tail, snapshot.head))
                if snapshot.kind is FitPartKind.FRAME
                else None
            ),
            allowed_operations=tuple(snapshot.allowed_operations),
            owned_derived_ids=tuple(str(value) for value in snapshot.owned_derived_ids),
            name_hint=str(snapshot.name_hint),
        )
        for snapshot in snapshots
    )
    ordered_definitions = _topological_parts(definitions)

    abs_orientation = {
        part_id: normalize_quaternion(snapshot.orientation)
        for part_id, snapshot in by_id.items()
    }
    values: list[FitPartValueRecord] = []
    for part in ordered_definitions:
        snapshot = by_id[part.part_id]
        head = tuple(float(value) for value in snapshot.head)
        tail = tuple(float(value) for value in snapshot.tail)
        if not _finite((*head, *tail, snapshot.width, snapshot.depth)):
            raise FitStateError("FIT_SNAPSHOT_NONFINITE")
        length = _v_length(_v_sub(tail, head))
        if length <= 0.0:
            raise FitStateError("FIT_LENGTH_NONPOSITIVE")

        orientation = abs_orientation[part.part_id]
        if part.parent_part_id is None:
            attachment_offset = head
            local_orientation = orientation
        else:
            parent_snapshot = by_id.get(part.parent_part_id)
            if parent_snapshot is None:
                raise FitStateError("FIT_PARENT_MISSING")
            parent_orientation = abs_orientation[part.parent_part_id]
            parent_head = tuple(float(value) for value in parent_snapshot.head)
            attachment_offset = rotate_vector(
                _q_conjugate(parent_orientation),
                _v_sub(head, parent_head),
            )
            local_orientation = normalize_quaternion(
                _q_mul(_q_conjugate(parent_orientation), orientation)
            )

        values.append(
            FitPartValueRecord(
                part_id=part.part_id,
                value=FitPartValue(
                    attachment_offset=attachment_offset,
                    local_orientation=local_orientation,
                    length=(None if part.kind is FitPartKind.FRAME else length),
                    width=float(snapshot.width),
                    depth=float(snapshot.depth),
                ),
            )
        )

    provisional_baseline = FitSessionBaseline(
        character_id=str(character_id),
        profile_id=str(profile_id),
        schema_version=int(schema_version),
        topology_fingerprint=topology_fingerprint(ordered_definitions),
        setup_revision=int(setup_revision),
        setup_signature=str(setup_signature),
        native_rest_digest="",
        appearance_digest="",
        owner_data_identity=tuple(str(value) for value in owner_data_identity),
        animation_footprint_digest=str(animation_footprint_digest),
    )
    provisional_document = FitDocument(
        baseline=provisional_baseline,
        parts=ordered_definitions,
        baseline_values=tuple(values),
    )
    provisional_draft = FitDraft(provisional_document, tuple(values))
    validate_fit_draft(provisional_draft)
    derived = derive_rest_parts(provisional_draft)

    for part in derived:
        source = by_id[part.part_id]
        if not _v_close(
            part.head,
            tuple(float(value) for value in source.head),
            tolerance=FIT_REST_ROUNDTRIP_TOLERANCE,
        ):
            raise FitStateError("FIT_ROUNDTRIP_HEAD_MISMATCH")
        if not _v_close(
            part.tail,
            tuple(float(value) for value in source.tail),
            tolerance=FIT_REST_ROUNDTRIP_TOLERANCE,
        ):
            raise FitStateError("FIT_ROUNDTRIP_TAIL_MISMATCH")
        if not _quaternion_close(part.orientation, source.orientation):
            raise FitStateError("FIT_ROUNDTRIP_ORIENTATION_MISMATCH")

    baseline = replace(
        provisional_baseline,
        native_rest_digest=_digest_rest(derived),
        appearance_digest=_digest_appearance(derived),
    )
    document = replace(provisional_document, baseline=baseline)
    draft = FitDraft(document, tuple(values))
    validate_fit_draft(draft)
    return document, draft


def replace_draft_value(draft: FitDraft, part_id: str, value: FitPartValue) -> FitDraft:
    records = list(draft.values)
    for index, record in enumerate(records):
        if record.part_id == part_id:
            records[index] = FitPartValueRecord(part_id, value)
            updated = FitDraft(draft.document, tuple(records))
            validate_fit_draft(updated)
            return updated
    raise FitStateError("FIT_PART_MISSING")


def build_rest_mutation_plan(draft: FitDraft) -> RestMutationPlan:
    validate_fit_draft(draft)
    baseline_draft = FitDraft(draft.document, draft.document.baseline_values)
    baseline_rest = {part.part_id: part for part in derive_rest_parts(baseline_draft)}
    candidate_rest_tuple = derive_rest_parts(draft)
    candidate_rest = {part.part_id: part for part in candidate_rest_tuple}

    mutations: list[RestMutation] = []
    structure_changed = False
    appearance_changed = False
    for definition in _topological_parts(draft.document.parts):
        before = baseline_rest[definition.part_id]
        after = candidate_rest[definition.part_id]
        changed_fields: list[str] = []
        if not _v_close(before.head, after.head):
            changed_fields.append("head")
        if not _v_close(before.tail, after.tail):
            changed_fields.append("tail")
        if not _quaternion_close(before.orientation, after.orientation):
            changed_fields.append("orientation")
        if abs(float(before.width) - float(after.width)) > 1e-9:
            changed_fields.append("width")
        if abs(float(before.depth) - float(after.depth)) > 1e-9:
            changed_fields.append("depth")
        structure = any(field in FIT_STRUCTURE_FIELDS for field in changed_fields)
        appearance = any(field in FIT_APPEARANCE_FIELDS for field in changed_fields)
        if structure or appearance:
            mutations.append(
                RestMutation(
                    part_id=definition.part_id,
                    before=before,
                    derived=after,
                    changed_fields=tuple(changed_fields),
                    structure_changed=structure,
                    appearance_changed=appearance,
                )
            )
        structure_changed = structure_changed or structure
        appearance_changed = appearance_changed or appearance

    return RestMutationPlan(
        topology_fingerprint=draft.document.baseline.topology_fingerprint,
        mutations=tuple(mutations),
        structure_changed=structure_changed,
        appearance_changed=appearance_changed,
        candidate_rest_digest=_digest_rest(candidate_rest_tuple),
        candidate_appearance_digest=_digest_appearance(candidate_rest_tuple),
    )
