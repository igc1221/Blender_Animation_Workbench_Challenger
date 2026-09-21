from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from mathutils import Vector

from .character_metadata import resolve_character
from .rigped_fit_commands import FitCommandError, fit_move_supported, move_fit_part_rig_local
from .rigped_fit_policy import FitSessionToken
from .rigped_fit_state import (
    FitDocument,
    FitDraft,
    FitPartKind,
    FitRestPartSnapshot,
    derive_rest_parts,
    extract_fit_document,
    rotate_vector,
)


@dataclass(frozen=True, slots=True)
class FitGeometryPart:
    part_id: str
    name_hint: str
    semantic_key: str
    side: str
    vertices: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class FitBodyGeometrySnapshot:
    character_id: str
    revision: int
    rig_pointer: int
    parts: tuple[FitGeometryPart, ...]


@dataclass(slots=True)
class FitSemanticSession:
    character_id: str
    token: FitSessionToken
    rig_object: Any
    document: FitDocument
    draft: FitDraft
    matrix_signature: tuple[float, ...]
    geometry: FitBodyGeometrySnapshot
    selected_part_ids: set[str] = field(default_factory=set)
    active_part_id: str | None = None
    revision: int = 0


_SESSIONS: dict[int, FitSemanticSession] = {}


class FitSemanticSessionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FitSemanticMoveReceipt:
    part_id: str
    revision_before: int
    revision_after: int
    world_delta: tuple[float, float, float]
    rig_delta: tuple[float, float, float]
    changed: bool


def _safe_pointer(value) -> int | None:
    if value is None:
        return None
    pointer = getattr(value, "as_pointer", None)
    if not callable(pointer):
        return None
    try:
        result = int(pointer())
    except ReferenceError:
        return None
    return result or None


def _window_key(context) -> int | None:
    return _safe_pointer(getattr(context, "window", None))


def _matrix_signature(matrix) -> tuple[float, ...]:
    return tuple(round(float(matrix[row][column]), 9) for row in range(4) for column in range(4))


def _animation_digest(token: FitSessionToken) -> str:
    return hashlib.sha256(repr(token.animation_signature).encode("utf-8")).hexdigest()


def _primary_rest_snapshots(scene, character_id: str):
    view = resolve_character(scene, character_id)
    resolved_map = dict(view.resolved_bindings)
    primary: list[tuple[Any, Any, Any]] = []
    for binding in view.definition.bindings:
        if str(binding.usage.value) != "PRIMARY":
            continue
        resolved = resolved_map.get(binding.binding_id)
        if resolved is None:
            continue
        target = getattr(resolved, "target", None)
        bone = getattr(target, "bone", None)
        if bone is None:
            continue
        primary.append((binding, resolved, bone))

    if not primary:
        raise FitSemanticSessionError("FIT_F2_NO_PRIMARY_PARTS")

    bone_to_binding = {bone.name: binding.binding_id for binding, _resolved, bone in primary}
    snapshots = []
    for binding, _resolved, bone in primary:
        parent_id = bone_to_binding.get(bone.parent.name) if bone.parent is not None else None
        orientation = bone.matrix_local.to_quaternion().normalized()
        snapshots.append(
            FitRestPartSnapshot(
                binding_id=str(binding.binding_id),
                semantic_key=str(binding.semantic_key),
                side=str(binding.side),
                parent_binding_id=parent_id,
                connected=bool(bone.use_connect),
                head=tuple(float(value) for value in bone.head_local),
                tail=tuple(float(value) for value in bone.tail_local),
                orientation=tuple(float(value) for value in orientation),
                width=float(bone.bbone_x),
                depth=float(bone.bbone_z),
                kind=(
                    FitPartKind.FRAME
                    if str(binding.semantic_key) == "awb.com"
                    else FitPartKind.BONE
                ),
                name_hint=str(bone.name),
            )
        )
    return view, tuple(primary), tuple(snapshots)


def _build_geometry(
    character_id: str,
    revision: int,
    rig,
    document: FitDocument,
    draft: FitDraft,
) -> FitBodyGeometrySnapshot:
    definition_by_id = {part.part_id: part for part in document.parts}
    derived = derive_rest_parts(draft)
    world = rig.matrix_world
    parts: list[FitGeometryPart] = []
    for rest in derived:
        definition = definition_by_id[rest.part_id]
        x_axis = Vector(rotate_vector(rest.orientation, (1.0, 0.0, 0.0))).normalized()
        z_axis = Vector(rotate_vector(rest.orientation, (0.0, 0.0, 1.0))).normalized()
        head = Vector(rest.head)
        tail = Vector(rest.tail)
        half_x = max(float(rest.width), 1e-5)
        half_z = max(float(rest.depth), 1e-5)
        vertices = []
        for origin in (head, tail):
            vertices.extend(
                (
                    origin - (x_axis * half_x) - (z_axis * half_z),
                    origin + (x_axis * half_x) - (z_axis * half_z),
                    origin + (x_axis * half_x) + (z_axis * half_z),
                    origin - (x_axis * half_x) + (z_axis * half_z),
                )
            )
        parts.append(
            FitGeometryPart(
                part_id=rest.part_id,
                name_hint=definition.name_hint,
                semantic_key=definition.semantic_key,
                side=definition.side,
                vertices=tuple(
                    tuple(float(value) for value in (world @ vertex))
                    for vertex in vertices
                ),
            )
        )
    return FitBodyGeometrySnapshot(
        character_id=character_id,
        revision=revision,
        rig_pointer=int(rig.as_pointer()),
        parts=tuple(parts),
    )


def begin_fit_semantic_session(
    context,
    *,
    character_id: str,
    token: FitSessionToken,
    rig,
) -> FitSemanticSession:
    key = _window_key(context)
    if key is None:
        raise FitSemanticSessionError("FIT_F2_WINDOW_UNAVAILABLE")
    if key in _SESSIONS:
        raise FitSemanticSessionError("FIT_F2_ALREADY_ACTIVE")
    if str(getattr(context, "mode", "")) != "OBJECT":
        raise FitSemanticSessionError("FIT_F2_REQUIRES_OBJECT_MODE")
    if rig is None or getattr(rig, "type", None) != "ARMATURE":
        raise FitSemanticSessionError("FIT_F2_RIG_INVALID")

    _view, primary, snapshots = _primary_rest_snapshots(context.scene, character_id)
    owner = primary[0][1].owner_object
    document, draft = extract_fit_document(
        character_id=character_id,
        profile_id=token.profile_id,
        schema_version=1,
        setup_revision=token.setup_revision,
        setup_signature=token.setup_signature,
        owner_data_identity=(
            str(int(owner.as_pointer())),
            str(int(owner.data.as_pointer())),
        ),
        animation_footprint_digest=_animation_digest(token),
        snapshots=snapshots,
    )
    geometry = _build_geometry(character_id, 0, rig, document, draft)
    session = FitSemanticSession(
        character_id=character_id,
        token=token,
        rig_object=rig,
        document=document,
        draft=draft,
        matrix_signature=_matrix_signature(rig.matrix_world),
        geometry=geometry,
    )
    _SESSIONS[key] = session
    return session


def fit_semantic_session(context) -> FitSemanticSession | None:
    key = _window_key(context)
    if key is None:
        return None
    session = _SESSIONS.get(key)
    if session is None:
        return None
    rig = session.rig_object
    if rig is None or _safe_pointer(rig) != session.geometry.rig_pointer:
        return None
    return session


def end_fit_semantic_session(context) -> FitSemanticSession | None:
    key = _window_key(context)
    return _SESSIONS.pop(key, None) if key is not None else None


def validate_fit_semantic_session(context, session: FitSemanticSession) -> tuple[str, ...]:
    issues: list[str] = []
    if str(getattr(context, "mode", "")) != "OBJECT":
        issues.append("FIT_F2_MODE_CHANGED")
    if _matrix_signature(session.rig_object.matrix_world) != session.matrix_signature:
        issues.append("FIT_F2_OBJECT_MATRIX_CHANGED")
    return tuple(issues)


def fit_geometry_snapshot(context) -> FitBodyGeometrySnapshot | None:
    session = fit_semantic_session(context)
    if session is None or validate_fit_semantic_session(context, session):
        return None
    return session.geometry


def fit_geometry_snapshot_for_rig(context, rig) -> FitBodyGeometrySnapshot | None:
    session = fit_semantic_session(context)
    if session is None or session.rig_object is not rig:
        return None
    if validate_fit_semantic_session(context, session):
        return None
    return session.geometry


def fit_geometry_part_by_name(context, rig, name: str) -> FitGeometryPart | None:
    snapshot = fit_geometry_snapshot_for_rig(context, rig)
    if snapshot is None:
        return None
    return next((part for part in snapshot.parts if part.name_hint == name), None)


def fit_selected_part_ids(context) -> tuple[str, ...]:
    session = fit_semantic_session(context)
    if session is None:
        return ()
    return tuple(
        part.part_id
        for part in session.geometry.parts
        if part.part_id in session.selected_part_ids
    )


def fit_active_part_id(context) -> str | None:
    session = fit_semantic_session(context)
    return session.active_part_id if session is not None else None


def apply_fit_part_selection(context, part_ids, action: str) -> bool:
    session = fit_semantic_session(context)
    if session is None:
        return False

    valid_ids = {part.part_id for part in session.geometry.parts}
    ordered = tuple(part_id for part_id in part_ids if part_id in valid_ids)
    action = str(action).upper()
    if action == "SET":
        session.selected_part_ids = set(ordered)
    elif action == "ADD":
        session.selected_part_ids.update(ordered)
    elif action == "REMOVE":
        session.selected_part_ids.difference_update(ordered)
    else:
        raise ValueError(f"Unsupported Fit selection action: {action}")

    if action != "REMOVE" and ordered:
        session.active_part_id = ordered[-1]
    elif session.active_part_id not in session.selected_part_ids:
        session.active_part_id = next(iter(session.selected_part_ids), None)

    rig = session.rig_object
    for obj in tuple(getattr(context, "selected_objects", ()) or ()):
        if obj is not rig:
            obj.select_set(False)
    rig.select_set(True)
    context.view_layer.objects.active = rig

    if context.area is not None:
        context.area.tag_redraw()
    return True


def _active_part_definition(session: FitSemanticSession):
    part_id = session.active_part_id
    if part_id is None or part_id not in session.selected_part_ids:
        return None
    return next((part for part in session.document.parts if part.part_id == part_id), None)


def fit_figure_move_available(context) -> bool:
    session = fit_semantic_session(context)
    if session is None or validate_fit_semantic_session(context, session):
        return False
    if len(session.selected_part_ids) != 1:
        return False
    definition = _active_part_definition(session)
    return bool(
        definition is not None
        and fit_move_supported(session.draft, definition.part_id)
    )


def fit_active_part_world_pivot_axes(
    context,
    *,
    orientation_mode: str = "LOCAL",
):
    session = fit_semantic_session(context)
    if session is None or validate_fit_semantic_session(context, session):
        return None, None
    definition = _active_part_definition(session)
    if definition is None:
        return None, None

    rest = next(
        (part for part in derive_rest_parts(session.draft) if part.part_id == definition.part_id),
        None,
    )
    if rest is None:
        return None, None

    if definition.name_hint.upper() in {"COM", "PELVIS"}:
        pivot_local = (Vector(rest.head) + Vector(rest.tail)) * 0.5
    else:
        pivot_local = Vector(rest.head)
    pivot_world = Vector(session.rig_object.matrix_world @ pivot_local)

    mode = str(orientation_mode).upper()
    if mode == "WORLD":
        axes = {
            "X": Vector((1.0, 0.0, 0.0)),
            "Y": Vector((0.0, 1.0, 0.0)),
            "Z": Vector((0.0, 0.0, 1.0)),
        }
    elif mode == "LOCAL":
        world3 = session.rig_object.matrix_world.to_3x3()
        axes = {}
        for name, basis in (
            ("X", (1.0, 0.0, 0.0)),
            ("Y", (0.0, 1.0, 0.0)),
            ("Z", (0.0, 0.0, 1.0)),
        ):
            axis = Vector(world3 @ Vector(rotate_vector(rest.orientation, basis)))
            if axis.length <= 1e-12:
                return None, None
            axes[name] = axis.normalized()
    else:
        return None, None
    return pivot_world, axes


def _publish_fit_draft(context, session: FitSemanticSession, draft: FitDraft) -> bool:
    if draft.document is not session.document:
        raise FitSemanticSessionError("FIT_F3_DRAFT_DOCUMENT_MISMATCH")
    if draft == session.draft:
        return False
    session.draft = draft
    session.revision += 1
    session.geometry = _build_geometry(
        session.character_id,
        session.revision,
        session.rig_object,
        session.document,
        session.draft,
    )
    if context.area is not None:
        context.area.tag_redraw()
    return True


def apply_fit_move_preview(
    context,
    *,
    baseline_draft: FitDraft,
    part_id: str,
    world_delta,
) -> FitSemanticMoveReceipt:
    session = fit_semantic_session(context)
    if session is None:
        raise FitSemanticSessionError("FIT_F3_SESSION_MISSING")
    issues = validate_fit_semantic_session(context, session)
    if issues:
        raise FitSemanticSessionError(issues[0])
    if baseline_draft.document is not session.document:
        raise FitSemanticSessionError("FIT_F3_DRAFT_DOCUMENT_MISMATCH")

    world_vector = Vector(world_delta)
    world3 = session.rig_object.matrix_world.to_3x3()
    if abs(float(world3.determinant())) <= 1e-12:
        raise FitSemanticSessionError("FIT_F3_OBJECT_MATRIX_SINGULAR")
    rig_vector = Vector(world3.inverted() @ world_vector)
    try:
        candidate = move_fit_part_rig_local(
            baseline_draft,
            str(part_id),
            tuple(float(value) for value in rig_vector),
        )
    except FitCommandError as exc:
        raise FitSemanticSessionError(str(exc)) from exc

    revision_before = int(session.revision)
    changed = _publish_fit_draft(context, session, candidate)
    return FitSemanticMoveReceipt(
        part_id=str(part_id),
        revision_before=revision_before,
        revision_after=int(session.revision),
        world_delta=tuple(float(value) for value in world_vector),
        rig_delta=tuple(float(value) for value in rig_vector),
        changed=changed,
    )


def restore_fit_draft_preview(context, baseline_draft: FitDraft) -> bool:
    session = fit_semantic_session(context)
    if session is None:
        return False
    if baseline_draft.document is not session.document:
        raise FitSemanticSessionError("FIT_F3_DRAFT_DOCUMENT_MISMATCH")
    return _publish_fit_draft(context, session, baseline_draft)
