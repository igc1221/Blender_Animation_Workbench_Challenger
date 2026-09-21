from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from mathutils import Matrix, Vector

from .character_metadata import resolve_character
from .rigped_fit_commands import FitCommandError, fit_move_supported, move_fit_part_rig_local
from .rigped_fit_policy import FitSessionToken
from .rigped_fit_state import (
    FitDocument,
    FitDraft,
    FitOperation,
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
    preview_serial: int
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
    matrix_world_frozen: tuple[tuple[float, float, float, float], ...]
    geometry: FitBodyGeometrySnapshot
    selected_part_ids: set[str] = field(default_factory=set)
    active_part_id: str | None = None
    revision: int = 0
    preview_serial: int = 0
    active_move_gesture: FitMoveGestureBaseline | None = None


_SESSIONS: dict[int, FitSemanticSession] = {}


class FitSemanticSessionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FitMoveGestureBaseline:
    part_id: str
    draft: FitDraft
    geometry: FitBodyGeometrySnapshot
    revision: int
    preview_serial: int
    matrix_signature: tuple[float, ...]
    selected_part_ids: tuple[str, ...]
    active_part_id: str


@dataclass(frozen=True, slots=True)
class FitSemanticMoveReceipt:
    part_id: str
    revision_before: int
    revision_after: int
    preview_serial: int
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


def _matrix_snapshot(matrix) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(float(matrix[row][column]) for column in range(4))
        for row in range(4)
    )


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
                allowed_operations=(
                    (FitOperation.MOVE,)
                    if str(binding.semantic_key) == "awb.com"
                    else ()
                ),
                name_hint=str(bone.name),
            )
        )
    return view, tuple(primary), tuple(snapshots)


def _build_geometry(
    character_id: str,
    revision: int,
    preview_serial: int,
    rig_pointer: int,
    world,
    document: FitDocument,
    draft: FitDraft,
) -> FitBodyGeometrySnapshot:
    definition_by_id = {part.part_id: part for part in document.parts}
    derived = derive_rest_parts(draft)
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
        revision=int(revision),
        preview_serial=int(preview_serial),
        rig_pointer=int(rig_pointer),
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
    matrix_signature = _matrix_signature(rig.matrix_world)
    matrix_world_frozen = _matrix_snapshot(rig.matrix_world)
    geometry = _build_geometry(
        character_id,
        0,
        0,
        int(rig.as_pointer()),
        Matrix(matrix_world_frozen),
        document,
        draft,
    )
    session = FitSemanticSession(
        character_id=character_id,
        token=token,
        rig_object=rig,
        document=document,
        draft=draft,
        matrix_signature=matrix_signature,
        matrix_world_frozen=matrix_world_frozen,
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

    if definition.semantic_key == "awb.com":
        pivot_local = (Vector(rest.head) + Vector(rest.tail)) * 0.5
    else:
        pivot_local = Vector(rest.head)
    frozen_world = Matrix(session.matrix_world_frozen)
    pivot_world = Vector(frozen_world @ pivot_local)

    mode = str(orientation_mode).upper()
    if mode == "WORLD":
        axes = {
            "X": Vector((1.0, 0.0, 0.0)),
            "Y": Vector((0.0, 1.0, 0.0)),
            "Z": Vector((0.0, 0.0, 1.0)),
        }
    elif mode == "LOCAL":
        world3 = frozen_world.to_3x3()
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


def _frozen_world3(session: FitSemanticSession):
    world3 = Matrix(session.matrix_world_frozen).to_3x3()
    determinant = float(world3.determinant())
    if determinant <= 1e-12:
        raise FitSemanticSessionError("FIT_F3_OBJECT_MATRIX_UNSUPPORTED")
    scales = tuple(Vector(world3.col[index]).length for index in range(3))
    maximum = max(scales)
    minimum = min(scales)
    if minimum <= 1e-12 or (maximum - minimum) > max(1e-6, maximum * 1e-6):
        raise FitSemanticSessionError("FIT_F3_OBJECT_MATRIX_UNSUPPORTED")
    return world3


def _ordered_selected_part_ids(session: FitSemanticSession) -> tuple[str, ...]:
    return tuple(
        part.part_id
        for part in session.geometry.parts
        if part.part_id in session.selected_part_ids
    )


def _rebuild_preview_geometry(context, session: FitSemanticSession, draft: FitDraft) -> bool:
    if draft.document is not session.document:
        raise FitSemanticSessionError("FIT_F3_DRAFT_DOCUMENT_MISMATCH")
    if draft == session.draft:
        return False
    session.draft = draft
    session.preview_serial += 1
    session.geometry = _build_geometry(
        session.character_id,
        session.revision,
        session.preview_serial,
        session.geometry.rig_pointer,
        Matrix(session.matrix_world_frozen),
        session.document,
        draft,
    )
    if context.area is not None:
        context.area.tag_redraw()
    return True


def begin_fit_move_gesture(context, *, part_id: str) -> FitMoveGestureBaseline:
    session = fit_semantic_session(context)
    if session is None:
        raise FitSemanticSessionError("FIT_F3_SESSION_MISSING")
    issues = validate_fit_semantic_session(context, session)
    if issues:
        raise FitSemanticSessionError(issues[0])
    if session.active_move_gesture is not None:
        raise FitSemanticSessionError("FIT_F3_GESTURE_ALREADY_ACTIVE")
    part_id = str(part_id)
    selected_part_ids = _ordered_selected_part_ids(session)
    if selected_part_ids != (part_id,) or session.active_part_id != part_id:
        raise FitSemanticSessionError("FIT_F3_MOVE_SELECTION_INVALID")
    if not fit_move_supported(session.draft, part_id):
        raise FitSemanticSessionError("FIT_F3_MOVE_UNSUPPORTED_PART")
    _frozen_world3(session)
    gesture = FitMoveGestureBaseline(
        part_id=part_id,
        draft=session.draft,
        geometry=session.geometry,
        revision=int(session.revision),
        preview_serial=int(session.preview_serial),
        matrix_signature=session.matrix_signature,
        selected_part_ids=selected_part_ids,
        active_part_id=part_id,
    )
    session.active_move_gesture = gesture
    return gesture


def _require_move_gesture(
    context,
    gesture: FitMoveGestureBaseline,
) -> FitSemanticSession:
    session = fit_semantic_session(context)
    if session is None:
        raise FitSemanticSessionError("FIT_F3_SESSION_MISSING")
    if session.active_move_gesture is not gesture:
        raise FitSemanticSessionError("FIT_F3_GESTURE_STALE")
    issues = validate_fit_semantic_session(context, session)
    if issues:
        raise FitSemanticSessionError(issues[0])
    if session.matrix_signature != gesture.matrix_signature:
        raise FitSemanticSessionError("FIT_F3_OBJECT_MATRIX_CHANGED")
    if _ordered_selected_part_ids(session) != gesture.selected_part_ids:
        raise FitSemanticSessionError("FIT_F3_MOVE_SELECTION_CHANGED")
    if session.active_part_id != gesture.active_part_id:
        raise FitSemanticSessionError("FIT_F3_MOVE_SELECTION_CHANGED")
    return session


def apply_fit_move_preview(
    context,
    *,
    gesture: FitMoveGestureBaseline,
    world_delta,
) -> FitSemanticMoveReceipt:
    session = _require_move_gesture(context, gesture)
    world_vector = Vector(world_delta)
    if len(world_vector) != 3:
        raise FitSemanticSessionError("FIT_F3_MOVE_DELTA_INVALID")
    world3 = _frozen_world3(session)
    rig_vector = Vector(world3.inverted() @ world_vector)
    try:
        candidate = move_fit_part_rig_local(
            gesture.draft,
            gesture.part_id,
            tuple(float(value) for value in rig_vector),
        )
    except FitCommandError as exc:
        raise FitSemanticSessionError(str(exc)) from exc

    changed = _rebuild_preview_geometry(context, session, candidate)
    return FitSemanticMoveReceipt(
        part_id=gesture.part_id,
        revision_before=gesture.revision,
        revision_after=int(session.revision),
        preview_serial=int(session.preview_serial),
        world_delta=tuple(float(value) for value in world_vector),
        rig_delta=tuple(float(value) for value in rig_vector),
        changed=changed,
    )


def commit_fit_move_gesture(
    context,
    gesture: FitMoveGestureBaseline,
) -> bool:
    session = _require_move_gesture(context, gesture)
    changed = session.draft != gesture.draft
    if changed:
        session.revision = gesture.revision + 1
        session.preview_serial += 1
        session.geometry = _build_geometry(
            session.character_id,
            session.revision,
            session.preview_serial,
            session.geometry.rig_pointer,
            Matrix(session.matrix_world_frozen),
            session.document,
            session.draft,
        )
    else:
        session.draft = gesture.draft
        session.geometry = gesture.geometry
        session.revision = gesture.revision
        session.preview_serial = gesture.preview_serial
    session.active_move_gesture = None
    if context.area is not None:
        context.area.tag_redraw()
    return changed


def cancel_fit_move_gesture(
    context,
    gesture: FitMoveGestureBaseline,
) -> bool:
    session = fit_semantic_session(context)
    if session is None or session.active_move_gesture is not gesture:
        return False
    session.draft = gesture.draft
    session.geometry = gesture.geometry
    session.revision = gesture.revision
    session.preview_serial = gesture.preview_serial
    session.active_move_gesture = None
    if context.area is not None:
        context.area.tag_redraw()
    return True
