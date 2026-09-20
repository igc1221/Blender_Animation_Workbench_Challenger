from __future__ import annotations

from dataclasses import dataclass
from math import atan2, exp
from typing import ClassVar

import bpy
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty, StringProperty
from bpy_extras.view3d_utils import (
    location_3d_to_region_2d,
    region_2d_to_origin_3d,
    region_2d_to_vector_3d,
)
from mathutils import Matrix, Quaternion, Vector
from mathutils.geometry import intersect_line_line, intersect_line_plane

from .gizmo_preferences import (
    GIZMO_HIGHLIGHT_COLOR,
    MAX_LINEAR_ROTATION_RADIANS_PER_PIXEL,
    arcball_world_step,
    clear_rotation_angle,
    gizmo_zoom_scale,
    linear_roll_screen_tangent,
    projected_axis_screen_direction,
    projection_world_per_pixel,
    show_rotation_angle,
    snapped_rotation_angle,
)
from .rigped_create_fit_ui import (
    capture_fit_structural_state,
    fit_orientation_mode,
    fit_transform_mode,
    fit_ui_state,
    record_fit_structural_change,
    redo_fit_structural_change,
    restore_fit_structural_state,
    undo_fit_structural_change,
)
from .rigped_limb_math import preferred_two_bone_bend


def _active_edit_bone(context):
    obj = getattr(context, "active_object", None)
    if obj is None or getattr(obj, "type", None) != "ARMATURE":
        return None
    edit_bones = getattr(obj.data, "edit_bones", None)
    if edit_bones is None:
        return None
    bone = getattr(edit_bones, "active", None)
    if bone is None or not bool(getattr(bone, "select", False)):
        return None
    return bone


def fit_scale_available(context) -> bool:
    return (
        getattr(context, "mode", "") == "EDIT_ARMATURE"
        and fit_ui_state(context) is not None
        and _active_edit_bone(context) is not None
    )


@dataclass(frozen=True)
class _FitBoneSnapshot:
    name: str
    head: Vector
    tail: Vector
    roll: float
    z_axis_local: Vector
    use_connect: bool


@dataclass
class _FitTransformSession:
    rig: object
    active_name: str
    pivot_world: Vector
    bones: tuple[_FitBoneSnapshot, ...]
    parent: _FitBoneSnapshot | None = None
    grandparent: _FitBoneSnapshot | None = None


def _fit_center_pivot_name(name: str) -> bool:
    return str(name).upper() in {"COM", "PELVIS"}


def _fit_pivot_local(bone) -> Vector:
    if _fit_center_pivot_name(str(bone.name)):
        return (Vector(bone.head) + Vector(bone.tail)) * 0.5
    return Vector(bone.head)


def _fit_chain_bones(active_bone) -> tuple[object, ...]:
    """Return the Fit transform closure for one active EditBone."""

    # Biped-style Pelvis is an isolated ball joint in both Fit and Animate.
    # Its rotation must not become a generic FK rotation of the leg branches.
    if str(active_bone.name).upper() == "PELVIS":
        return (active_bone,)

    ordered: list[object] = []
    pending = [active_bone]
    while pending:
        bone = pending.pop(0)
        ordered.append(bone)
        pending.extend(tuple(getattr(bone, "children", ())))
    return tuple(ordered)


def _begin_fit_transform_for_bone(rig, active) -> _FitTransformSession:
    snapshots = tuple(
        _FitBoneSnapshot(
            name=str(bone.name),
            head=Vector(bone.head),
            tail=Vector(bone.tail),
            roll=float(bone.roll),
            z_axis_local=Vector(bone.matrix.to_3x3() @ Vector((0.0, 0.0, 1.0))),
            use_connect=bool(bone.use_connect),
        )
        for bone in _fit_chain_bones(active)
    )
    parent = getattr(active, "parent", None)
    parent_snapshot = None
    if parent is not None:
        parent_snapshot = _FitBoneSnapshot(
            name=str(parent.name),
            head=Vector(parent.head),
            tail=Vector(parent.tail),
            roll=float(parent.roll),
            z_axis_local=Vector(parent.matrix.to_3x3() @ Vector((0.0, 0.0, 1.0))),
            use_connect=bool(parent.use_connect),
        )
    grandparent_snapshot = None
    if str(active.name) in {
        "Hand.L",
        "Hand.R",
        "HAND.L",
        "HAND.R",
        "Foot.L",
        "Foot.R",
        "FOOT.L",
        "FOOT.R",
    } and parent is not None:
        grandparent = getattr(parent, "parent", None)
        if grandparent is not None:
            grandparent_snapshot = _FitBoneSnapshot(
                name=str(grandparent.name),
                head=Vector(grandparent.head),
                tail=Vector(grandparent.tail),
                roll=float(grandparent.roll),
                z_axis_local=Vector(grandparent.matrix.to_3x3() @ Vector((0.0, 0.0, 1.0))),
                use_connect=bool(grandparent.use_connect),
            )
    return _FitTransformSession(
        rig=rig,
        active_name=str(active.name),
        pivot_world=Vector(rig.matrix_world @ _fit_pivot_local(active)),
        bones=snapshots,
        parent=parent_snapshot,
        grandparent=grandparent_snapshot,
    )


def _begin_fit_transform(context) -> _FitTransformSession | None:
    rig = getattr(context, "active_object", None)
    active = _active_edit_bone(context)
    if rig is None or active is None:
        return None
    return _begin_fit_transform_for_bone(rig, active)


def _begin_fit_transform_sessions(context) -> tuple[_FitTransformSession, ...]:
    """Capture non-overlapping selected Fit roots, active branch first."""

    rig = getattr(context, "active_object", None)
    active = _active_edit_bone(context)
    if rig is None or active is None:
        return ()
    selected = tuple(bone for bone in rig.data.edit_bones if bool(getattr(bone, "select", False)))
    if not selected:
        return (_begin_fit_transform_for_bone(rig, active),)
    selected_names = {str(bone.name) for bone in selected}

    def has_selected_ancestor(bone) -> bool:
        parent = getattr(bone, "parent", None)
        while parent is not None:
            if str(parent.name) in selected_names:
                return True
            parent = getattr(parent, "parent", None)
        return False

    roots = [bone for bone in selected if not has_selected_ancestor(bone)]
    if not roots:
        roots = [active]

    def contains_active(root) -> bool:
        return any(str(bone.name) == str(active.name) for bone in _fit_chain_bones(root))

    roots.sort(key=lambda bone: (0 if contains_active(bone) else 1, str(bone.name)))
    return tuple(_begin_fit_transform_for_bone(rig, bone) for bone in roots)


def _opposite_bone_name(name: str) -> str | None:
    if name.endswith(".L"):
        return f"{name[:-2]}.R"
    if name.endswith(".R"):
        return f"{name[:-2]}.L"
    return None


def _mirror_fit_world_vector(rig, vector: Vector, *, axial: bool = False) -> Vector:
    """Mirror a polar/axial world vector across the Rigped local X symmetry plane."""

    world3 = rig.matrix_world.to_3x3()
    local = Vector(rig.matrix_world.inverted_safe().to_3x3() @ Vector(vector))
    local.x = -local.x
    if axial:
        # Rotational axes are pseudovectors: a reflection contributes one
        # additional sign flip beyond the ordinary mirrored direction.
        local.negate()
    return Vector(world3 @ local)


def _mapped_fit_move_delta(
    active_session: _FitTransformSession,
    target_session: _FitTransformSession,
    world_delta: Vector,
) -> Vector:
    opposite = _opposite_bone_name(active_session.active_name)
    if opposite is not None and target_session.active_name == opposite:
        return _mirror_fit_world_vector(target_session.rig, world_delta, axial=False)
    return Vector(world_delta)


def _mapped_fit_rotation_axis(
    active_session: _FitTransformSession,
    target_session: _FitTransformSession,
    axis_world: Vector,
) -> Vector:
    opposite = _opposite_bone_name(active_session.active_name)
    if opposite is not None and target_session.active_name == opposite:
        mirrored = _mirror_fit_world_vector(target_session.rig, axis_world, axial=True)
        if mirrored.length > 1e-9:
            mirrored.normalize()
        return mirrored
    return Vector(axis_world)


def _apply_fit_move_sessions(
    active_session: _FitTransformSession,
    sessions: tuple[_FitTransformSession, ...],
    world_delta: Vector,
) -> None:
    for target_session in sessions:
        _apply_fit_move_world(
            target_session,
            _mapped_fit_move_delta(active_session, target_session, world_delta),
        )


def _apply_fit_rotation_sessions(
    active_session: _FitTransformSession,
    sessions: tuple[_FitTransformSession, ...],
    axis_world: Vector,
    angle: float,
) -> None:
    for target_session in sessions:
        mapped_axis = _mapped_fit_rotation_axis(active_session, target_session, axis_world)
        if mapped_axis.length <= 1e-9:
            continue
        _apply_fit_rotation_world(target_session, mapped_axis, angle)


def _restore_fit_transform_sessions(sessions: tuple[_FitTransformSession, ...]) -> None:
    for session in reversed(sessions):
        _restore_fit_transform(session)


def _restore_fit_transform(session: _FitTransformSession) -> None:
    edit_bones = session.rig.data.edit_bones
    ordered: list[_FitBoneSnapshot] = []
    seen: set[str] = set()
    for snapshot in (session.grandparent, session.parent, *session.bones):
        if snapshot is None or snapshot.name in seen:
            continue
        seen.add(snapshot.name)
        ordered.append(snapshot)

    # Disconnect first so restoring one connected endpoint cannot push another
    # snapshot out of place. Reconnect only after every baseline transform is
    # back; this is also what makes Toe's temporary Fit disconnection cancel-safe.
    for snapshot in ordered:
        bone = edit_bones.get(snapshot.name)
        if bone is not None:
            bone.use_connect = False
    for snapshot in ordered:
        bone = edit_bones.get(snapshot.name)
        if bone is None:
            continue
        bone.head = snapshot.head
        bone.tail = snapshot.tail
        bone.roll = snapshot.roll
    for snapshot in ordered:
        bone = edit_bones.get(snapshot.name)
        if bone is not None:
            bone.use_connect = snapshot.use_connect


def _fit_preferred_bend(
    session: _FitTransformSession,
    edit_bones,
    shoulder: Vector,
    baseline_perp: Vector,
    baseline_dir: Vector,
    first_length: float,
    second_length: float,
    primary_z: Vector,
    *,
    is_arm: bool,
) -> Vector | None:
    """Use the same gesture-start elbow/knee preference as Animate mode."""

    straight_seed: Vector
    if is_arm:
        suffix = ".L" if session.active_name.upper().endswith(".L") else ".R"
        pole = edit_bones.get(f"IK_Elbow{suffix}")
        if pole is not None:
            straight_seed = Vector(pole.head) - Vector(shoulder)
            straight_seed -= baseline_dir * float(straight_seed.dot(baseline_dir))
        else:
            straight_seed = Vector(primary_z)
    else:
        straight_seed = Vector((0.0, -1.0, 0.0))

    fallback = Vector((1.0, 0.0, 0.0))
    if abs(float(fallback.dot(baseline_dir))) > 0.9:
        fallback = Vector((0.0, 0.0, 1.0))
    return preferred_two_bone_bend(
        baseline_perp,
        baseline_dir,
        first_length,
        second_length,
        is_arm=is_arm,
        straight_seed=straight_seed,
        fallback=fallback,
    )


def _apply_fit_move_world(session: _FitTransformSession, world_delta: Vector) -> None:
    # Pelvis W is intentionally visual-only. Do not translate it and do not
    # reinterpret Move as Rotate; Fit Pelvis structure is edited with E/R.
    if session.active_name.upper() == "PELVIS":
        return
    inv_world3 = session.rig.matrix_world.inverted_safe().to_3x3()
    local_delta = Vector(inv_world3 @ Vector(world_delta))
    edit_bones = session.rig.data.edit_bones

    # Spine2 keeps its connection and length, but Move remains useful as a
    # rotation-compensated placement gesture. Interpret the dragged tail as a
    # direction target: the Spine2 head stays on the lower Spine endpoint, the
    # link rotates at fixed length, and every upper-body descendant follows the
    # same rigid rotation. This avoids an independent translation gap while
    # preserving a visible/useful Move gizmo for torso fitting.
    if session.active_name in {"Spine2", "SPINE2"} and session.bones:
        active_snapshot = session.bones[0]
        active = edit_bones.get(active_snapshot.name)
        if active is None:
            return
        pivot = Vector(active_snapshot.head)
        baseline = Vector(active_snapshot.tail) - pivot
        desired = Vector(active_snapshot.tail) + local_delta - pivot
        if baseline.length <= 1e-9 or desired.length <= 1e-9:
            return
        rotation = baseline.normalized().rotation_difference(desired.normalized())
        for snapshot in session.bones:
            bone = edit_bones.get(snapshot.name)
            if bone is None:
                continue
            bone.head = pivot + rotation @ (Vector(snapshot.head) - pivot)
            bone.tail = pivot + rotation @ (Vector(snapshot.tail) - pivot)
            target_z = Vector(rotation @ snapshot.z_axis_local)
            if target_z.length > 1e-9:
                bone.align_roll(target_z.normalized())
        return

    # Fit Neck Move is a rigid placement edit, not a connected-endpoint stretch.
    # Keep Neck parented to Spine2, but disconnect its base so translating Neck
    # cannot pull Spine2.tail and silently change torso length. Head and any
    # future Neck descendants follow as one rigid chain and keep their lengths.
    if session.active_name in {"Neck", "NECK"} and session.bones:
        for snapshot in session.bones:
            bone = edit_bones.get(snapshot.name)
            if bone is not None:
                bone.use_connect = False
        for snapshot in session.bones:
            bone = edit_bones.get(snapshot.name)
            if bone is None:
                continue
            bone.head = Vector(snapshot.head) + local_delta
            bone.tail = Vector(snapshot.tail) + local_delta
            bone.roll = snapshot.roll
        for snapshot in session.bones:
            bone = edit_bones.get(snapshot.name)
            if bone is None:
                continue
            bone.use_connect = False if snapshot.name == session.active_name else snapshot.use_connect
        return

    # Biped Figure-mode parity for the toe base: position Toe relative to Foot
    # without dragging Foot's tail with the connected EditBone endpoint. Keep
    # Foot as the parent so hierarchy inheritance remains intact, but make the
    # Toe base unconnected once it is structurally repositioned. Toe length and
    # any descendant toe-link lengths remain unchanged under the rigid move.
    if session.active_name in {"Toe.L", "Toe.R", "TOE.L", "TOE.R"} and session.bones:
        for snapshot in session.bones:
            bone = edit_bones.get(snapshot.name)
            if bone is not None:
                bone.use_connect = False
        for snapshot in session.bones:
            bone = edit_bones.get(snapshot.name)
            if bone is None:
                continue
            bone.head = Vector(snapshot.head) + local_delta
            bone.tail = Vector(snapshot.tail) + local_delta
            bone.roll = snapshot.roll
        for snapshot in session.bones:
            bone = edit_bones.get(snapshot.name)
            if bone is None:
                continue
            bone.use_connect = False if snapshot.name == session.active_name else snapshot.use_connect
        return

    # Biped-style Fit middle-limb move: the drag is interpreted as a new end
    # target, not as translation or local-Y length scaling. Solve the native
    # UpperArm+ForeArm or Thigh+Calf pair as a fixed-length two-bone chain,
    # preserving the existing bend side, then carry distal descendants rigidly.
    # This makes local-Y Move extend/fold through rotation instead of stretching.
    if session.active_name in {
        "ForeArm.L",
        "ForeArm.R",
        "FOREARM.L",
        "FOREARM.R",
        "Calf.L",
        "Calf.R",
        "CALF.L",
        "CALF.R",
    } and session.bones and session.parent is not None:
        active_snapshot = session.bones[0]
        parent_snapshot = session.parent
        parent = edit_bones.get(parent_snapshot.name)
        active = edit_bones.get(active_snapshot.name)
        if parent is None or active is None:
            return

        shoulder = Vector(parent_snapshot.head)
        elbow = Vector(active_snapshot.head)
        wrist = Vector(active_snapshot.tail)
        upper_vec = elbow - shoulder
        fore_vec = wrist - elbow
        upper_length = float(upper_vec.length)
        fore_length = float(fore_vec.length)
        target_vec = wrist + local_delta - shoulder
        target_distance = float(target_vec.length)
        if upper_length <= 1e-9 or fore_length <= 1e-9 or target_distance <= 1e-9:
            return

        target_dir = target_vec.normalized()
        # Keep a tiny amount of bend at full extension. A mathematically exact
        # straight two-bone chain has no unique elbow side, so even sub-pixel
        # target noise can make the solver appear to tremble there.
        reach_margin = max(1e-5, (upper_length + fore_length) * 0.001)
        min_reach = abs(upper_length - fore_length) + reach_margin
        max_reach = max(min_reach, upper_length + fore_length - reach_margin)
        solved_distance = min(max(target_distance, min_reach), max_reach)
        new_wrist = shoulder + target_dir * solved_distance

        along = (
            upper_length * upper_length
            - fore_length * fore_length
            + solved_distance * solved_distance
        ) / (2.0 * solved_distance)
        bend_height = max(0.0, upper_length * upper_length - along * along) ** 0.5

        baseline_target = wrist - shoulder
        baseline_distance = float(baseline_target.length)
        if baseline_distance <= 1e-9:
            return
        baseline_dir = baseline_target.normalized()
        baseline_along = (
            upper_length * upper_length
            - fore_length * fore_length
            + baseline_distance * baseline_distance
        ) / (2.0 * baseline_distance)
        baseline_perp = elbow - (shoulder + baseline_dir * baseline_along)
        is_arm = session.active_name in {"ForeArm.L", "ForeArm.R", "FOREARM.L", "FOREARM.R"}
        resolved_bend = _fit_preferred_bend(
            session,
            edit_bones,
            shoulder,
            baseline_perp,
            baseline_dir,
            upper_length,
            fore_length,
            parent_snapshot.z_axis_local,
            is_arm=is_arm,
        )
        if resolved_bend is None:
            return
        baseline_perp = resolved_bend

        # Freeze the bend plane at mouse-down. Within that plane the elbow/knee
        # direction is derived continuously from the moving end-effector. Do not
        # force it back to the original pole sign per frame: that sign correction
        # is discontinuous when the target passes behind/above the body and was
        # the source of the visible one-frame joint flip.
        bend_plane_normal = baseline_dir.cross(baseline_perp)
        if bend_plane_normal.length <= 1e-7:
            return
        bend_plane_normal.normalize()
        bend_dir = bend_plane_normal.cross(target_dir)
        if bend_dir.length <= 1e-7:
            bend_dir = Vector(baseline_perp)
            bend_dir -= target_dir * bend_dir.dot(target_dir)
        if bend_dir.length <= 1e-7:
            return
        bend_dir.normalize()
        new_elbow = shoulder + target_dir * along + bend_dir * bend_height

        new_upper_vec = new_elbow - shoulder
        new_fore_vec = new_wrist - new_elbow
        if new_upper_vec.length <= 1e-9 or new_fore_vec.length <= 1e-9:
            return
        upper_rotation = upper_vec.normalized().rotation_difference(new_upper_vec.normalized())
        fore_rotation = fore_vec.normalized().rotation_difference(new_fore_vec.normalized())

        parent.head = parent_snapshot.head
        parent.tail = new_elbow
        parent_z = Vector(upper_rotation @ parent_snapshot.z_axis_local)
        if parent_z.length > 1e-9:
            parent.align_roll(parent_z.normalized())

        for snapshot in session.bones:
            bone = edit_bones.get(snapshot.name)
            if bone is None:
                continue
            bone.head = new_elbow + fore_rotation @ (Vector(snapshot.head) - elbow)
            bone.tail = new_elbow + fore_rotation @ (Vector(snapshot.tail) - elbow)
            target_z = Vector(fore_rotation @ snapshot.z_axis_local)
            if target_z.length > 1e-9:
                bone.align_roll(target_z.normalized())
        return

    # Biped-style Fit distal-limb move: treat Hand/Foot head as the end-effector
    # of the fixed-length two-bone chain above it. Moving Hand or Foot therefore
    # bends/extends the limb through rotation while preserving every segment
    # length; the distal bone and its descendants follow rigidly from the joint.
    if (
        session.active_name in {
            "Hand.L",
            "Hand.R",
            "HAND.L",
            "HAND.R",
            "Foot.L",
            "Foot.R",
            "FOOT.L",
            "FOOT.R",
        }
        and session.bones
        and session.parent is not None
        and session.grandparent is not None
    ):
        hand_snapshot = session.bones[0]
        fore_snapshot = session.parent
        upper_snapshot = session.grandparent
        upper = edit_bones.get(upper_snapshot.name)
        fore = edit_bones.get(fore_snapshot.name)
        hand = edit_bones.get(hand_snapshot.name)
        if upper is None or fore is None or hand is None:
            return

        shoulder = Vector(upper_snapshot.head)
        elbow = Vector(fore_snapshot.head)
        wrist = Vector(fore_snapshot.tail)
        upper_vec = elbow - shoulder
        fore_vec = wrist - elbow
        upper_length = float(upper_vec.length)
        fore_length = float(fore_vec.length)
        target_vec = wrist + local_delta - shoulder
        target_distance = float(target_vec.length)
        if upper_length <= 1e-9 or fore_length <= 1e-9 or target_distance <= 1e-9:
            return

        target_dir = target_vec.normalized()
        reach_margin = max(1e-5, (upper_length + fore_length) * 0.001)
        min_reach = abs(upper_length - fore_length) + reach_margin
        max_reach = max(min_reach, upper_length + fore_length - reach_margin)
        solved_distance = min(max(target_distance, min_reach), max_reach)
        new_wrist = shoulder + target_dir * solved_distance

        along = (
            upper_length * upper_length
            - fore_length * fore_length
            + solved_distance * solved_distance
        ) / (2.0 * solved_distance)
        bend_height = max(0.0, upper_length * upper_length - along * along) ** 0.5

        baseline_target = wrist - shoulder
        baseline_distance = float(baseline_target.length)
        if baseline_distance <= 1e-9:
            return
        baseline_dir = baseline_target.normalized()
        baseline_along = (
            upper_length * upper_length
            - fore_length * fore_length
            + baseline_distance * baseline_distance
        ) / (2.0 * baseline_distance)
        baseline_perp = elbow - (shoulder + baseline_dir * baseline_along)
        is_arm = session.active_name in {"Hand.L", "Hand.R", "HAND.L", "HAND.R"}
        resolved_bend = _fit_preferred_bend(
            session,
            edit_bones,
            shoulder,
            baseline_perp,
            baseline_dir,
            upper_length,
            fore_length,
            upper_snapshot.z_axis_local,
            is_arm=is_arm,
        )
        if resolved_bend is None:
            return
        baseline_perp = resolved_bend

        bend_plane_normal = baseline_dir.cross(baseline_perp)
        if bend_plane_normal.length <= 1e-7:
            return
        bend_plane_normal.normalize()
        bend_dir = bend_plane_normal.cross(target_dir)
        if bend_dir.length <= 1e-7:
            bend_dir = Vector(baseline_perp)
            bend_dir -= target_dir * bend_dir.dot(target_dir)
        if bend_dir.length <= 1e-7:
            return
        bend_dir.normalize()
        new_elbow = shoulder + target_dir * along + bend_dir * bend_height

        new_upper_vec = new_elbow - shoulder
        new_fore_vec = new_wrist - new_elbow
        if new_upper_vec.length <= 1e-9 or new_fore_vec.length <= 1e-9:
            return
        upper_rotation = upper_vec.normalized().rotation_difference(new_upper_vec.normalized())
        fore_rotation = fore_vec.normalized().rotation_difference(new_fore_vec.normalized())

        upper.head = upper_snapshot.head
        upper.tail = new_elbow
        upper_z = Vector(upper_rotation @ upper_snapshot.z_axis_local)
        if upper_z.length > 1e-9:
            upper.align_roll(upper_z.normalized())

        fore.head = new_elbow
        fore.tail = new_wrist
        fore_z = Vector(fore_rotation @ fore_snapshot.z_axis_local)
        if fore_z.length > 1e-9:
            fore.align_roll(fore_z.normalized())

        for snapshot in session.bones:
            bone = edit_bones.get(snapshot.name)
            if bone is None:
                continue
            bone.head = new_wrist + fore_rotation @ (Vector(snapshot.head) - wrist)
            bone.tail = new_wrist + fore_rotation @ (Vector(snapshot.tail) - wrist)
            target_z = Vector(fore_rotation @ snapshot.z_axis_local)
            if target_z.length > 1e-9:
                bone.align_roll(target_z.normalized())
        return

    # Biped Figure-mode parity for the proximal limb segment: moving UpperArm or
    # Thigh is an IK-from-root gesture, not raw EditBone translation. Keep the
    # shoulder/hip joint fixed, keep the active segment length fixed, move the
    # elbow/knee toward the dragged target, and carry descendants rigidly with
    # the resulting joint delta.
    if session.active_name in {
        "UpperArm.L",
        "UpperArm.R",
        "UPPERARM.L",
        "UPPERARM.R",
        "Thigh.L",
        "Thigh.R",
        "THIGH.L",
        "THIGH.R",
    } and session.bones:
        active_snapshot = session.bones[0]
        active = edit_bones.get(active_snapshot.name)
        if active is None:
            return
        baseline = Vector(active_snapshot.tail) - Vector(active_snapshot.head)
        length = float(baseline.length)
        desired = Vector(active_snapshot.tail) + local_delta - Vector(active_snapshot.head)
        if length <= 1e-9 or desired.length <= 1e-9:
            return
        new_tail = Vector(active_snapshot.head) + desired.normalized() * length
        elbow_delta = new_tail - Vector(active_snapshot.tail)
        active.head = active_snapshot.head
        active.tail = new_tail
        active.roll = active_snapshot.roll
        for snapshot in session.bones[1:]:
            bone = edit_bones.get(snapshot.name)
            if bone is None:
                continue
            bone.head = snapshot.head + elbow_delta
            bone.tail = snapshot.tail + elbow_delta
            bone.roll = snapshot.roll
        return

    for snapshot in session.bones:
        bone = edit_bones.get(snapshot.name)
        if bone is None:
            continue
        bone.head = snapshot.head + local_delta
        bone.tail = snapshot.tail + local_delta
        bone.roll = snapshot.roll


def _apply_fit_rotation_world(
    session: _FitTransformSession,
    axis_world: Vector,
    angle: float,
) -> None:
    axis = Vector(axis_world)
    if axis.length <= 1e-9:
        return
    axis.normalize()
    rotation = Matrix.Rotation(float(angle), 3, axis)
    rig_world = session.rig.matrix_world.copy()
    rig_world3 = rig_world.to_3x3()
    inv_world = rig_world.inverted_safe()
    inv_world3 = inv_world.to_3x3()
    pivot = Vector(session.pivot_world)
    edit_bones = session.rig.data.edit_bones
    for snapshot in session.bones:
        bone = edit_bones.get(snapshot.name)
        if bone is None:
            continue
        head_world = Vector(rig_world @ snapshot.head)
        tail_world = Vector(rig_world @ snapshot.tail)
        bone.head = Vector(inv_world @ (pivot + rotation @ (head_world - pivot)))
        bone.tail = Vector(inv_world @ (pivot + rotation @ (tail_world - pivot)))
        target_z_world = Vector(rotation @ (rig_world3 @ snapshot.z_axis_local))
        target_z_local = Vector(inv_world3 @ target_z_world)
        if target_z_local.length > 1e-9:
            bone.align_roll(target_z_local.normalized())


def _apply_fit_scale_local(context, active_name: str, axis_name: str, factor: float) -> bool:
    rig = getattr(context, "active_object", None)
    if rig is None or getattr(rig, "type", None) != "ARMATURE":
        return False
    bone = rig.data.edit_bones.get(active_name)
    if bone is None:
        return False
    scale = max(0.02, float(factor))
    axes = set(axis_name)
    if not axes or not axes.issubset({"X", "Y", "Z"}):
        return False

    # Fit mode is structural FK. Scaling the active bone along its local Y axis
    # changes only that bone's length; descendants inherit the endpoint motion as
    # a rigid translation and keep their own length, thickness, roll, and scale.
    descendants = tuple(_fit_chain_bones(bone))[1:]
    descendant_baseline = tuple(
        (
            str(child.name),
            Vector(child.head),
            Vector(child.tail),
            float(child.roll),
        )
        for child in descendants
    )

    if "X" in axes:
        bone.bbone_x = max(1e-6, float(bone.bbone_x) * scale)

        # Pelvis width is structural hip spacing, not only display thickness.
        # Keep each leg segment rigid and move the two hip sockets laterally
        # around the Pelvis center along the Pelvis local-X axis. Thigh remains
        # parented (but intentionally unconnected) to Pelvis, while the entire
        # Thigh->Calf->Foot->Toe chain follows its socket as one translation.
        if active_name in {"Pelvis", "PELVIS"}:
            pelvis_x = Vector(bone.matrix.to_3x3().col[0])
            if pelvis_x.length > 1e-9:
                pelvis_x.normalize()
                center = (Vector(bone.head) + Vector(bone.tail)) * 0.5
                edit_bones = rig.data.edit_bones
                for thigh_name in (
                    "Thigh.L",
                    "Thigh.R",
                    "THIGH.L",
                    "THIGH.R",
                    "MCH_Thigh.L",
                    "MCH_Thigh.R",
                    "MCH_THIGH.L",
                    "MCH_THIGH.R",
                ):
                    thigh = edit_bones.get(thigh_name)
                    if thigh is None:
                        continue
                    offset = float((Vector(thigh.head) - center).dot(pelvis_x))
                    lateral_delta = pelvis_x * (offset * (scale - 1.0))
                    if lateral_delta.length <= 1e-12:
                        continue
                    chain = tuple(_fit_chain_bones(thigh))
                    baseline = tuple(
                        (
                            str(item.name),
                            Vector(item.head),
                            Vector(item.tail),
                            float(item.roll),
                            bool(item.use_connect),
                        )
                        for item in chain
                    )
                    for name, _head, _tail, _roll, _connected in baseline:
                        item = edit_bones.get(name)
                        if item is not None:
                            item.use_connect = False
                    for name, head, tail, roll, _connected in baseline:
                        item = edit_bones.get(name)
                        if item is None:
                            continue
                        item.head = head + lateral_delta
                        item.tail = tail + lateral_delta
                        item.roll = roll
                    for name, _head, _tail, _roll, connected in baseline:
                        item = edit_bones.get(name)
                        if item is not None:
                            item.use_connect = connected
    if "Y" in axes:
        head = Vector(bone.head)
        old_tail = Vector(bone.tail)
        direction = old_tail - head
        if direction.length <= 1e-9:
            return False
        bone.tail = head + direction * scale
        tail_delta = Vector(bone.tail) - old_tail
        # Pelvis length/shape is independent from torso height. Its Spine and
        # leg sockets are intentionally unconnected children, so changing
        # Pelvis local-Y must not push the entire upper body or legs. Torso
        # length is authored from the root-most Spine link instead.
        propagate_y_to_descendants = active_name not in {"Pelvis", "PELVIS"}
        if tail_delta.length > 1e-12 and propagate_y_to_descendants:
            edit_bones = rig.data.edit_bones
            for name, child_head, child_tail, child_roll in descendant_baseline:
                child = edit_bones.get(name)
                if child is None:
                    continue
                # Translate every descendant joint explicitly, parent-first. In
                # Edit Mode Blender can otherwise leave a connected child's head
                # at the previous joint until that child is touched, which would
                # collapse its own length when only the parent tail moves.
                child.head = child_head + tail_delta
                child.tail = child_tail + tail_delta
                child.roll = child_roll
    if "Z" in axes:
        bone.bbone_z = max(1e-6, float(bone.bbone_z) * scale)
    return True


def _fit_orientation_axes(context) -> dict[str, Vector] | None:
    rig = getattr(context, "active_object", None)
    bone = _active_edit_bone(context)
    if rig is None or bone is None:
        return None
    orientation = fit_orientation_mode(context)
    if orientation == "LOCAL":
        basis = (rig.matrix_world @ bone.matrix).to_3x3()
    else:
        basis = Matrix.Identity(3)
    result: dict[str, Vector] = {}
    for index, axis_name in enumerate(("X", "Y", "Z")):
        axis = Vector(basis.col[index])
        if axis.length <= 1e-9:
            return None
        result[axis_name] = axis.normalized()
    return result


def _fit_view_axis(context) -> Vector | None:
    region_data = getattr(context, "region_data", None)
    if region_data is None:
        return None
    axis = Vector(region_data.view_matrix.inverted_safe().to_3x3() @ Vector((0.0, 0.0, 1.0)))
    if axis.length <= 1e-9:
        return None
    return axis.normalized()


def _axis_point(
    context,
    pivot: Vector,
    axis: Vector,
    mouse_x: float,
    mouse_y: float,
) -> Vector | None:
    if context.region is None or context.region_data is None:
        return None
    mouse = Vector((float(mouse_x), float(mouse_y)))
    ray_origin = region_2d_to_origin_3d(context.region, context.region_data, mouse)
    ray_direction = region_2d_to_vector_3d(context.region, context.region_data, mouse)
    if ray_direction.length <= 1e-9:
        return None
    ray_direction.normalize()
    intersections = intersect_line_line(
        pivot - axis * 100000.0,
        pivot + axis * 100000.0,
        ray_origin,
        ray_origin + ray_direction * 100000.0,
    )
    if intersections is None:
        return None
    return Vector(intersections[0])


def _plane_point(
    context,
    pivot: Vector,
    normal: Vector,
    mouse_x: float,
    mouse_y: float,
) -> Vector | None:
    if context.region is None or context.region_data is None:
        return None
    mouse = Vector((float(mouse_x), float(mouse_y)))
    ray_origin = region_2d_to_origin_3d(context.region, context.region_data, mouse)
    ray_direction = region_2d_to_vector_3d(context.region, context.region_data, mouse)
    if ray_direction.length <= 1e-9:
        return None
    ray_direction.normalize()
    point = intersect_line_plane(
        ray_origin,
        ray_origin + ray_direction * 100000.0,
        pivot,
        normal,
        False,
    )
    return Vector(point) if point is not None else None


def _rotation_vector(
    context,
    pivot: Vector,
    axis: Vector,
    mouse_x: float,
    mouse_y: float,
) -> Vector | None:
    if context.region is None or context.region_data is None:
        return None
    mouse = Vector((float(mouse_x), float(mouse_y)))
    ray_origin = region_2d_to_origin_3d(context.region, context.region_data, mouse)
    ray_direction = region_2d_to_vector_3d(context.region, context.region_data, mouse)
    if ray_direction.length <= 1e-9:
        return None
    ray_direction.normalize()
    point = intersect_line_plane(
        ray_origin,
        ray_origin + ray_direction * 100000.0,
        pivot,
        axis,
        False,
    )
    if point is None:
        return None
    radial = Vector(point) - pivot
    radial -= axis * radial.dot(axis)
    if radial.length <= 1e-7:
        return None
    return radial.normalized()


def fit_transform_available(context) -> bool:
    # Keep the W/E gizmo shells visible for every selected Fit control. Pelvis W
    # is display-only (no structural mutation); Pelvis posing is authored by E
    # Rotate and R Scale rather than a synthetic Move->Rotate gesture.
    return fit_scale_available(context) and fit_transform_mode(context) in {"MOVE", "ROTATE"}


class BAW_OT_rigped_fit_commit_transform(bpy.types.Operator):
    """Commit one Fit structural gesture into the Fit-local history."""

    bl_idname = "baw.rigped_fit_commit_transform"
    bl_label = "Commit Rigped Fit Transform"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    mode: EnumProperty(
        items=(
            ("MOVE", "Move", "Commit a Fit FK move"),
            ("ROTATE", "Rotate", "Commit a Fit FK rotation"),
            ("SCALE", "Scale", "Commit a Fit structural scale"),
        ),
        default="MOVE",
    )
    active_bone: StringProperty(default="")
    world_delta: FloatVectorProperty(size=3, default=(0.0, 0.0, 0.0), subtype="TRANSLATION")
    axis_world: FloatVectorProperty(size=3, default=(0.0, 0.0, 1.0), subtype="DIRECTION")
    angle: FloatProperty(default=0.0, subtype="ANGLE")
    scale_axis: EnumProperty(
        items=(
            ("X", "X", "Scale local X"),
            ("Y", "Y", "Scale local Y"),
            ("Z", "Z", "Scale local Z"),
            ("XY", "XY", "Scale local X and Y"),
            ("XZ", "XZ", "Scale local X and Z"),
            ("YZ", "YZ", "Scale local Y and Z"),
            ("XYZ", "XYZ", "Scale all local dimensions"),
        ),
        default="Y",
    )
    scale_factor: FloatProperty(default=1.0, min=0.02)

    @classmethod
    def poll(cls, context):
        return fit_scale_available(context)

    def execute(self, context):
        active = _active_edit_bone(context)
        if active is None or (self.active_bone and active.name != self.active_bone):
            return {"CANCELLED"}
        if self.mode == "MOVE" and str(active.name).upper() == "PELVIS":
            return {"CANCELLED"}
        before = capture_fit_structural_state(context)
        if before is None:
            return {"CANCELLED"}
        sessions = _begin_fit_transform_sessions(context)
        if not sessions:
            return {"CANCELLED"}
        active_session = next(
            (session for session in sessions if session.active_name == str(active.name)),
            sessions[0],
        )
        applied = True
        if self.mode == "MOVE":
            delta = Vector(self.world_delta)
            if delta.length <= 1e-9:
                return {"CANCELLED"}
            for session in sessions:
                _apply_fit_move_world(
                    session,
                    _mapped_fit_move_delta(active_session, session, delta),
                )
        elif self.mode == "ROTATE":
            axis = Vector(self.axis_world)
            if axis.length <= 1e-9 or abs(float(self.angle)) <= 1e-9:
                return {"CANCELLED"}
            axis.normalize()
            for session in sessions:
                mapped_axis = _mapped_fit_rotation_axis(active_session, session, axis)
                if mapped_axis.length <= 1e-9:
                    continue
                _apply_fit_rotation_world(session, mapped_axis, float(self.angle))
        else:
            if abs(float(self.scale_factor) - 1.0) <= 1e-9:
                return {"CANCELLED"}
            # Scale remains an active-bone structural operation for now; the
            # multi-selection contract introduced here is Move/Rotate only.
            applied = _apply_fit_scale_local(
                context,
                active_session.active_name,
                self.scale_axis,
                float(self.scale_factor),
            )
        if not applied or not record_fit_structural_change(context, before):
            restore_fit_structural_state(context, before)
            return {"CANCELLED"}
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


class BAW_OT_rigped_fit_history_undo(bpy.types.Operator):
    bl_idname = "baw.rigped_fit_history_undo"
    bl_label = "Undo Fit Transform"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return fit_ui_state(context) is not None and getattr(context, "mode", "") == "EDIT_ARMATURE"

    def execute(self, context):
        undo_fit_structural_change(context)
        return {"FINISHED"}


class BAW_OT_rigped_fit_history_redo(bpy.types.Operator):
    bl_idname = "baw.rigped_fit_history_redo"
    bl_label = "Redo Fit Transform"
    bl_options: ClassVar[set[str]] = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return fit_ui_state(context) is not None and getattr(context, "mode", "") == "EDIT_ARMATURE"

    def execute(self, context):
        redo_fit_structural_change(context)
        return {"FINISHED"}


class BAW_OT_rigped_fit_transform_axis(bpy.types.Operator):
    """Fit-only FK-style EditBone transform preserving active-only selection."""

    bl_idname = "baw.rigped_fit_transform_axis"
    bl_label = "Transform Rigped Fit Bone Chain"
    bl_description = "Move or rotate the active Fit bone and all native descendants as one FK chain"
    # This operator owns only the live preview. The release path restores that
    # preview and delegates the final edit to BAW_OT_rigped_fit_commit_transform,
    # which is the single native Blender undo boundary for the gesture.
    bl_options: ClassVar[set[str]] = {"REGISTER", "BLOCKING"}

    mode: EnumProperty(
        items=(
            ("MOVE", "Move", "Move the active Fit bone and native descendants"),
            ("ROTATE", "Rotate", "Rotate the active Fit bone and native descendants"),
        ),
        default="MOVE",
    )
    axis: EnumProperty(
        items=(
            ("X", "X", "Use X axis"),
            ("Y", "Y", "Use Y axis"),
            ("Z", "Z", "Use Z axis"),
            ("XY", "XY", "Move in the XY plane"),
            ("XZ", "XZ", "Move in the XZ plane"),
            ("YZ", "YZ", "Move in the YZ plane"),
            ("FREE", "Free", "Free move or virtual-trackball rotate"),
            ("VIEW", "View", "Rotate around the current view axis"),
        ),
        default="X",
    )

    _session: _FitTransformSession | None = None
    _sessions: tuple[_FitTransformSession, ...] = ()
    _axis = Vector((1.0, 0.0, 0.0))
    _move_plane_normal = Vector((0.0, 0.0, 1.0))
    _pivot = Vector((0.0, 0.0, 0.0))
    _start_axis_point = Vector((0.0, 0.0, 0.0))
    _move_screen_axis = Vector((1.0, 0.0))
    _move_start_screen_projection = 0.0
    _move_world_per_pixel = 0.0
    _move_use_screen_axis = False
    _start_plane_point = Vector((0.0, 0.0, 0.0))
    _start_rotation_vector = Vector((1.0, 0.0, 0.0))
    _previous_rotation_vector = Vector((1.0, 0.0, 0.0))
    _previous_rotation_mouse = Vector((0.0, 0.0))
    _rotate_use_screen_tangent = False
    _rotate_screen_tangent = Vector((1.0, 0.0))
    _rotate_reference_radius = 0.0
    _previous_free_mouse = Vector((0.0, 0.0))
    _free_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    _current_move_delta = Vector((0.0, 0.0, 0.0))
    _raw_angle = 0.0
    _current_angle = 0.0

    trackball_radius_px: FloatProperty(
        name="",
        description="",
        default=64.0,
        min=8.0,
    )

    @classmethod
    def poll(cls, context):
        return fit_transform_available(context)

    def invoke(self, context, event):
        if self.mode == "ROTATE" and self.axis == "FREE":
            from .viewport_keymap import prioritize_awb_selection_over_free_rotate

            if prioritize_awb_selection_over_free_rotate(context, event):
                if context.area is not None:
                    context.area.tag_redraw()
                return {"FINISHED"}

        sessions = _begin_fit_transform_sessions(context)
        active = _active_edit_bone(context)
        if not sessions or active is None:
            return {"CANCELLED"}
        session = next(
            (item for item in sessions if item.active_name == str(active.name)),
            sessions[0],
        )
        axes = _fit_orientation_axes(context)
        if axes is None:
            return {"CANCELLED"}
        self._session = session
        self._sessions = sessions
        self._pivot = Vector(session.pivot_world)
        self._current_move_delta = Vector((0.0, 0.0, 0.0))
        self._raw_angle = 0.0
        self._current_angle = 0.0
        self._free_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        self._move_use_screen_axis = False
        self._move_world_per_pixel = 0.0
        self._rotate_use_screen_tangent = False
        self._rotate_reference_radius = 0.0
        if self.mode == "MOVE":
            if self.axis in {"X", "Y", "Z"}:
                axis = axes.get(self.axis)
                if axis is None:
                    self._session = None
                    return {"CANCELLED"}
                self._axis = Vector(axis)

                # Fit structural gestures are sensitive to tiny oscillations in
                # the ray/axis closest-point input. Freeze the projected screen
                # direction and pixel-to-world scale at mouse-down for limbs and
                # the central body chain so 1-2 px drags stay smooth/monotonic.
                if context.region is not None and context.region_data is not None:
                    pivot_2d = location_3d_to_region_2d(context.region, context.region_data, self._pivot)
                    active_snapshot = session.bones[0] if session.bones else None
                    if pivot_2d is not None and active_snapshot is not None:
                        local_length = Vector(active_snapshot.tail) - Vector(active_snapshot.head)
                        world_length = float(
                            Vector(session.rig.matrix_world.to_3x3() @ local_length).length
                        )
                        reference_world = max(0.05, world_length)
                        sample_2d = location_3d_to_region_2d(
                            context.region,
                            context.region_data,
                            self._pivot + self._axis * reference_world,
                        )
                        if sample_2d is not None:
                            screen_delta = Vector(sample_2d) - Vector(pivot_2d)
                            if screen_delta.length >= 3.0:
                                self._move_screen_axis = screen_delta.normalized()
                                mouse = Vector((event.mouse_region_x, event.mouse_region_y))
                                self._move_start_screen_projection = float(
                                    mouse.dot(self._move_screen_axis)
                                )
                                world_per_pixel = projection_world_per_pixel(
                                    context, self._pivot
                                )
                                if world_per_pixel is not None:
                                    self._move_world_per_pixel = float(world_per_pixel)
                                    self._move_use_screen_axis = True

                if not self._move_use_screen_axis:
                    start = _axis_point(
                        context,
                        self._pivot,
                        self._axis,
                        event.mouse_region_x,
                        event.mouse_region_y,
                    )
                    if start is None:
                        self._session = None
                        return {"CANCELLED"}
                    self._start_axis_point = Vector(start)
            else:
                if self.axis == "FREE":
                    normal = _fit_view_axis(context)
                else:
                    plane_axes = {
                        "XY": ("X", "Y"),
                        "XZ": ("X", "Z"),
                        "YZ": ("Y", "Z"),
                    }
                    pair = plane_axes.get(self.axis)
                    normal = None if pair is None else Vector(axes[pair[0]]).cross(axes[pair[1]])
                if normal is None or normal.length <= 1e-9:
                    self._session = None
                    return {"CANCELLED"}
                self._move_plane_normal = Vector(normal).normalized()
                start = _plane_point(
                    context,
                    self._pivot,
                    self._move_plane_normal,
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
                if start is None:
                    self._session = None
                    return {"CANCELLED"}
                self._start_plane_point = Vector(start)
        else:
            axis = (
                _fit_view_axis(context)
                if self.axis in {"VIEW", "FREE"}
                else axes.get(self.axis)
            )
            if axis is None:
                self._session = None
                return {"CANCELLED"}
            self._axis = Vector(axis)
            if self.axis == "FREE":
                region_data = getattr(context, "region_data", None)
                if region_data is None:
                    self._session = None
                    return {"CANCELLED"}
                view_basis = region_data.view_matrix.inverted().to_3x3()
                start = Vector(view_basis.col[0]).normalized()
                self._previous_free_mouse = Vector(
                    (float(event.mouse_region_x), float(event.mouse_region_y))
                )
            else:
                start = _rotation_vector(
                    context,
                    self._pivot,
                    self._axis,
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
            if start is None:
                self._session = None
                return {"CANCELLED"}
            self._start_rotation_vector = Vector(start)
            self._previous_rotation_vector = Vector(start)
            self._previous_rotation_mouse = Vector(
                (float(event.mouse_region_x), float(event.mouse_region_y))
            )
            if self.axis != "FREE":
                tangent = linear_roll_screen_tangent(
                    context,
                    self._pivot,
                    self._axis,
                    self._previous_rotation_mouse,
                    self._start_rotation_vector,
                )
                if tangent is None:
                    self._session = None
                    return {"CANCELLED"}
                self._rotate_screen_tangent = Vector(tangent)
                self._rotate_use_screen_tangent = True

            # Edge-on Fit rotation rings can make repeated mouse-ray/plane
            # intersections noisy for 1-2 px drags. Use the same calibrated
            # screen-space tangent path for the complete animator-facing body,
            # so left/right arms and legs do not have different rotation feel.
            stabilized_fit_rotate = session.active_name in {
                "COM",
                "Com",
                "Pelvis",
                "PELVIS",
                "Spine",
                "SPINE",
                "Spine2",
                "SPINE2",
                "Neck",
                "NECK",
                "Head",
                "HEAD",
                "Clavicle.L",
                "Clavicle.R",
                "CLAVICLE.L",
                "CLAVICLE.R",
                "UpperArm.L",
                "UpperArm.R",
                "UPPERARM.L",
                "UPPERARM.R",
                "ForeArm.L",
                "ForeArm.R",
                "FOREARM.L",
                "FOREARM.R",
                "Hand.L",
                "Hand.R",
                "HAND.L",
                "HAND.R",
                "Thigh.L",
                "Thigh.R",
                "THIGH.L",
                "THIGH.R",
                "Calf.L",
                "Calf.R",
                "CALF.L",
                "CALF.R",
                "Foot.L",
                "Foot.R",
                "FOOT.L",
                "FOOT.R",
                "Toe.L",
                "Toe.R",
                "TOE.L",
                "TOE.R",
            }
            if (
                self.axis != "FREE"
                and stabilized_fit_rotate
                and context.region is not None
                and context.region_data is not None
            ):
                pivot_2d = location_3d_to_region_2d(
                    context.region,
                    context.region_data,
                    self._pivot,
                )
                active_snapshot = session.bones[0] if session.bones else None
                if pivot_2d is not None and active_snapshot is not None:
                    local_length = Vector(active_snapshot.tail) - Vector(active_snapshot.head)
                    world_length = float(
                        Vector(session.rig.matrix_world.to_3x3() @ local_length).length
                    )
                    reference_radius = max(0.05, world_length)
                    reference_2d = location_3d_to_region_2d(
                        context.region,
                        context.region_data,
                        self._pivot + Vector(start) * reference_radius,
                    )
                    mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
                    if reference_2d is not None:
                        projected_radius = float((Vector(reference_2d) - Vector(pivot_2d)).length)
                        mouse_radius = float((mouse - Vector(pivot_2d)).length)
                        if projected_radius >= 1.0 and mouse_radius >= 6.0:
                            reference_radius *= mouse_radius / projected_radius
                            probe_angle = 0.05
                            probe_start = location_3d_to_region_2d(
                                context.region,
                                context.region_data,
                                self._pivot + Vector(start) * reference_radius,
                            )
                            probe_vector = Matrix.Rotation(
                                probe_angle,
                                3,
                                self._axis,
                            ) @ Vector(start)
                            probe_end = location_3d_to_region_2d(
                                context.region,
                                context.region_data,
                                self._pivot + probe_vector * reference_radius,
                            )
                            if probe_start is not None and probe_end is not None:
                                tangent_pixels = Vector(probe_end) - Vector(probe_start)
                                if tangent_pixels.length >= 0.5:
                                    self._rotate_reference_radius = reference_radius
                                    self._rotate_use_screen_tangent = True
        if self.mode == "ROTATE" and self.axis != "FREE":
            show_rotation_angle(context, 0.0, self._pivot)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        session = self._session
        if session is None:
            return {"CANCELLED"}
        if event.type == "MOUSEMOVE":
            if self.mode == "MOVE":
                if self.axis in {"X", "Y", "Z"}:
                    if self._move_use_screen_axis:
                        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
                        projection = float(mouse.dot(self._move_screen_axis))
                        distance = (
                            projection - self._move_start_screen_projection
                        ) * self._move_world_per_pixel
                        self._current_move_delta = self._axis * distance
                        _apply_fit_move_sessions(
                            session,
                            self._sessions or (session,),
                            self._current_move_delta,
                        )
                    else:
                        current = _axis_point(
                            context,
                            self._pivot,
                            self._axis,
                            event.mouse_region_x,
                            event.mouse_region_y,
                        )
                        if current is not None:
                            distance = float((Vector(current) - self._start_axis_point).dot(self._axis))
                            self._current_move_delta = self._axis * distance
                            _apply_fit_move_sessions(
                                session,
                                self._sessions or (session,),
                                self._current_move_delta,
                            )
                else:
                    current = _plane_point(
                        context,
                        self._pivot,
                        self._move_plane_normal,
                        event.mouse_region_x,
                        event.mouse_region_y,
                    )
                    if current is not None:
                        self._current_move_delta = Vector(current) - self._start_plane_point
                        _apply_fit_move_sessions(
                            session,
                            self._sessions or (session,),
                            self._current_move_delta,
                        )
            else:
                mouse = Vector(
                    (float(event.mouse_region_x), float(event.mouse_region_y))
                )
                if self.axis == "FREE":
                    rotation_step = arcball_world_step(
                        context,
                        self._pivot,
                        self._previous_free_mouse,
                        mouse,
                        float(self.trackball_radius_px),
                    )
                    if abs(float(rotation_step.angle)) > 1e-12:
                        self._free_rotation = (
                            rotation_step @ self._free_rotation
                        ).normalized()
                        axis_world = Vector(self._free_rotation.axis)
                        angle = float(self._free_rotation.angle)
                        if axis_world.length > 1e-9 and angle > 1e-12:
                            axis_world.normalize()
                            self._axis = axis_world
                            self._current_angle = angle
                            _apply_fit_rotation_sessions(
                                session,
                                self._sessions or (session,),
                                self._axis,
                                self._current_angle,
                            )
                    self._previous_free_mouse = Vector(mouse)
                elif self._rotate_use_screen_tangent:
                    mouse_delta = mouse - self._previous_rotation_mouse
                    raw_step = (
                        float(mouse_delta.dot(self._rotate_screen_tangent))
                        * MAX_LINEAR_ROTATION_RADIANS_PER_PIXEL
                    )
                    self._raw_angle += raw_step
                    self._current_angle = snapped_rotation_angle(
                        context,
                        event,
                        self._raw_angle,
                    )
                    self._previous_rotation_mouse = mouse
                    _apply_fit_rotation_sessions(
                        session,
                        self._sessions or (session,),
                        self._axis,
                        self._current_angle,
                    )
                    show_rotation_angle(context, self._current_angle, self._pivot)
                else:
                    current = _rotation_vector(
                        context,
                        self._pivot,
                        self._axis,
                        event.mouse_region_x,
                        event.mouse_region_y,
                    )
                    if current is not None:
                        previous = self._previous_rotation_vector
                        raw_step = atan2(
                            float(self._axis.dot(previous.cross(current))),
                            float(previous.dot(current)),
                        )
                        pixel_step = float((mouse - self._previous_rotation_mouse).length)
                        # Rotation-plane ray intersections can explode when the
                        # plane is nearly edge-on to the view. Bound each angular
                        # increment by the actual mouse travel so a 1-2 px motion
                        # cannot suddenly become a large rotation or +/-pi flip.
                        max_step = min(0.35, max(0.006, pixel_step * 0.025))
                        if abs(raw_step) > max_step * 4.0:
                            # Treat an angle wildly disproportionate to mouse travel
                            # as a ray/plane singularity, not intentional input.
                            step = 0.0
                        else:
                            step = max(-max_step, min(max_step, raw_step))
                        self._raw_angle += step
                        self._current_angle = snapped_rotation_angle(
                            context,
                            event,
                            self._raw_angle,
                        )
                        self._previous_rotation_vector = Vector(current)
                        self._previous_rotation_mouse = mouse
                        _apply_fit_rotation_sessions(
                            session,
                            self._sessions or (session,),
                            self._axis,
                            self._current_angle,
                        )
                        show_rotation_angle(context, self._current_angle, self._pivot)
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            active_name = session.active_name
            move_delta = Vector(self._current_move_delta)
            axis_world = Vector(self._axis)
            angle = float(self._current_angle)
            # Remove every live preview branch before the local-history commit
            # runs, so the recorded pre-state is exactly the Fit gesture baseline.
            _restore_fit_transform_sessions(self._sessions or (session,))
            self._session = None
            self._sessions = ()
            if self.mode == "MOVE":
                if move_delta.length <= 1e-9:
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}
                result = bpy.ops.baw.rigped_fit_commit_transform(
                    mode="MOVE",
                    active_bone=active_name,
                    world_delta=tuple(move_delta),
                )
            else:
                if abs(angle) <= 1e-9:
                    clear_rotation_angle(context)
                    if context.area is not None:
                        context.area.tag_redraw()
                    return {"CANCELLED"}
                result = bpy.ops.baw.rigped_fit_commit_transform(
                    mode="ROTATE",
                    active_bone=active_name,
                    axis_world=tuple(axis_world),
                    angle=angle,
                )
            clear_rotation_angle(context)
            if context.area is not None:
                context.area.tag_redraw()
            return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}

        if event.type in {"ESC", "RIGHTMOUSE"} or (
            event.type == "Z" and bool(getattr(event, "ctrl", False))
        ):
            _restore_fit_transform_sessions(self._sessions or (session,))
            self._session = None
            self._sessions = ()
            clear_rotation_angle(context)
            if context.area is not None:
                context.area.tag_redraw()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_OT_rigped_fit_scale_axis(bpy.types.Operator):
    """Preview one Fit local scale axis and commit it to Fit-local history."""

    bl_idname = "baw.rigped_fit_scale_axis"
    bl_label = "Scale Rigped Fit Bone"
    bl_options: ClassVar[set[str]] = {"REGISTER", "BLOCKING"}

    axis: EnumProperty(
        items=(
            ("X", "X", "Scale local B-Bone width"),
            ("Y", "Y", "Scale local bone length"),
            ("Z", "Z", "Scale local B-Bone depth"),
            ("XY", "XY", "Scale local width and length"),
            ("XZ", "XZ", "Scale local width and depth"),
            ("YZ", "YZ", "Scale local length and depth"),
            ("XYZ", "XYZ", "Scale all local dimensions"),
        ),
        default="Y",
    )

    _before = None
    _active_name = ""
    _axis = Vector((1.0, 0.0, 0.0))
    _pivot = Vector((0.0, 0.0, 0.0))
    _start_axis_point = Vector((0.0, 0.0, 0.0))
    _screen_pivot = Vector((0.0, 0.0))
    _screen_axis = Vector((1.0, 0.0))
    _start_screen_projection = 0.0
    _screen_reference = 48.0
    _start_radius = 1.0
    _reference_length = 1.0
    _radial_mode = False
    _factor = 1.0

    @classmethod
    def poll(cls, context):
        return fit_scale_available(context) and fit_transform_mode(context) == "SCALE"

    def invoke(self, context, event):
        active = _active_edit_bone(context)
        rig = getattr(context, "active_object", None)
        before = capture_fit_structural_state(context)
        axes = _fit_orientation_axes(context)
        if active is None or rig is None or before is None or axes is None:
            return {"CANCELLED"}
        pivot = Vector(rig.matrix_world @ _fit_pivot_local(active))
        radial_mode = len(self.axis) > 1
        axis = axes.get(self.axis) if not radial_mode else None
        if not radial_mode and axis is None:
            return {"CANCELLED"}
        if context.region is None or context.region_data is None:
            return {"CANCELLED"}
        pivot_2d = location_3d_to_region_2d(context.region, context.region_data, pivot)
        if pivot_2d is None:
            return {"CANCELLED"}
        mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
        self._screen_pivot = Vector(pivot_2d)
        screen_delta = mouse - self._screen_pivot
        if radial_mode:
            # Lock one outward screen direction at mouse-down. The sign stays
            # fixed for the whole gesture, so crossing the origin never makes
            # a continued shrink drag start growing again.
            if abs(screen_delta.x) >= abs(screen_delta.y):
                self._screen_axis = Vector((1.0 if screen_delta.x >= 0.0 else -1.0, 0.0))
            else:
                self._screen_axis = Vector((0.0, 1.0 if screen_delta.y >= 0.0 else -1.0))
            self._start_screen_projection = float(screen_delta.dot(self._screen_axis))
        else:
            self._axis = Vector(axis)
            screen_axis = projected_axis_screen_direction(
                context, pivot, self._axis
            )
            if screen_axis is None:
                return {"CANCELLED"}
            self._screen_axis = Vector(screen_axis)
            self._start_screen_projection = float(
                (mouse - self._screen_pivot).dot(self._screen_axis)
            )
        world_length = Vector(rig.matrix_world.to_3x3() @ (active.tail - active.head)).length
        self._before = before
        self._active_name = str(active.name)
        self._pivot = pivot
        self._reference_length = max(float(world_length), 0.05)
        self._radial_mode = radial_mode
        self._factor = 1.0
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        before = self._before
        if before is None:
            return {"CANCELLED"}
        if event.type == "MOUSEMOVE":
            mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
            projection = float((mouse - self._screen_pivot).dot(self._screen_axis))
            distance = projection - self._start_screen_projection
            self._factor = max(0.02, exp(distance / 260.0))
            if not restore_fit_structural_state(context, before):
                self._before = None
                return {"CANCELLED"}
            _apply_fit_scale_local(context, self._active_name, self.axis, self._factor)
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            factor = float(self._factor)
            active_name = self._active_name
            if not restore_fit_structural_state(context, before):
                self._before = None
                return {"CANCELLED"}
            self._before = None
            if abs(factor - 1.0) <= 1e-9:
                return {"CANCELLED"}
            result = bpy.ops.baw.rigped_fit_commit_transform(
                mode="SCALE",
                active_bone=active_name,
                scale_axis=self.axis,
                scale_factor=factor,
            )
            return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}

        if event.type in {"ESC", "RIGHTMOUSE"} or (
            event.type == "Z" and bool(getattr(event, "ctrl", False))
        ):
            restore_fit_structural_state(context, before)
            self._before = None
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


class BAW_GGT_rigped_fit_transform(bpy.types.GizmoGroup):
    bl_idname = "BAW_GGT_rigped_fit_transform"
    bl_label = "AWB Rigped Fit FK Transform"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    # Do not use SHOW_MODAL_ALL here: Move and Rotate share this group, so that
    # option makes rotation rings appear while a Move handle is being dragged.
    # Individual visible handles opt into modal drawing instead.
    bl_options: ClassVar[set[str]] = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context):
        return fit_transform_available(context)

    def setup(self, _context):
        self._zoom_reference_distance = None
        self.move_gizmos = {}
        self.move_hit_gizmos = {}
        self.move_plane_gizmos = {}
        self.move_plane_hit_gizmos = {}
        self.rotate_gizmos = {}
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        for axis_name in ("X", "Y", "Z"):
            # Keep the visible native-style arrow completely display-only.
            # Blender's built-in arrow gizmo applies its own modal `offset`
            # whenever target_set_operator() is used; that made the highlighted
            # arrow slide away from the Fit pivot while our operator separately
            # moved the rig. A second invisible hit gizmo owns interaction so the
            # visible arrow can stay rigidly anchored to the bone pivot.
            move = self.gizmos.new("GIZMO_GT_arrow_3d")
            move.draw_style = "NORMAL"
            move.draw_options = {"STEM"}
            move.color = colors[axis_name]
            move.alpha = 0.9
            move.color_highlight = colors[axis_name]
            move.alpha_highlight = 0.9
            move.line_width = 1.0
            move.scale_basis = 1.0
            move.hide_select = True
            move.use_draw_modal = True
            self.move_gizmos[axis_name] = move

            hit = self.gizmos.new("GIZMO_GT_arrow_3d")
            hit_props = hit.target_set_operator(BAW_OT_rigped_fit_transform_axis.bl_idname)
            hit_props.mode = "MOVE"
            hit_props.axis = axis_name
            hit.draw_style = "NORMAL"
            hit.draw_options = {"STEM"}
            hit.color = colors[axis_name]
            hit.alpha = 0.001
            # The interaction proxy is invisible normally, but on hover it
            # draws the same native arrow in white so the picked axis is
            # immediately obvious. It is deliberately not drawn during modal
            # drag; the anchored display-only arrow remains the visual authority.
            hit.color_highlight = (1.0, 1.0, 1.0)
            hit.alpha_highlight = 1.0
            hit.line_width = 1.0
            hit.scale_basis = 1.0
            hit.select_bias = 2.0
            hit.use_draw_modal = False
            self.move_hit_gizmos[axis_name] = hit

            rotate = self.gizmos.new("GIZMO_GT_dial_3d")
            rotate_props = rotate.target_set_operator(BAW_OT_rigped_fit_transform_axis.bl_idname)
            rotate_props.mode = "ROTATE"
            rotate_props.axis = axis_name
            rotate.color = colors[axis_name]
            rotate.alpha = 0.65
            rotate.color_highlight = GIZMO_HIGHLIGHT_COLOR
            rotate.alpha_highlight = 1.0
            rotate.line_width = 2.0
            rotate.scale_basis = 1.0
            rotate.draw_options = {"CLIP"}
            rotate.use_draw_modal = True
            self.rotate_gizmos[axis_name] = rotate

        # Blender-style two-axis plane handles. Their color follows the axis
        # excluded by the plane (XY=Z/blue, XZ=Y/green, YZ=X/red), matching the
        # ordinary Move gizmo visual language.
        plane_colors = {
            "XY": colors["Z"],
            "XZ": colors["Y"],
            "YZ": colors["X"],
        }
        for plane_name in ("XY", "XZ", "YZ"):
            plane = self.gizmos.new("GIZMO_GT_primitive_3d")
            plane.draw_style = "PLANE"
            plane.draw_inner = True
            plane.color = plane_colors[plane_name]
            plane.alpha = 0.5
            plane.color_highlight = plane_colors[plane_name]
            plane.alpha_highlight = 0.5
            plane.line_width = 1.0
            plane.scale_basis = 0.11
            plane.hide_select = True
            plane.use_draw_offset_scale = True
            plane.use_draw_modal = True
            self.move_plane_gizmos[plane_name] = plane

            plane_hit = self.gizmos.new("GIZMO_GT_primitive_3d")
            plane_hit_props = plane_hit.target_set_operator(BAW_OT_rigped_fit_transform_axis.bl_idname)
            plane_hit_props.mode = "MOVE"
            plane_hit_props.axis = plane_name
            plane_hit.draw_style = "PLANE"
            plane_hit.draw_inner = True
            plane_hit.color = plane_colors[plane_name]
            plane_hit.alpha = 0.001
            plane_hit.color_highlight = (1.0, 1.0, 1.0)
            plane_hit.alpha_highlight = 0.95
            plane_hit.line_width = 1.0
            plane_hit.scale_basis = 0.14
            plane_hit.select_bias = 2.0
            plane_hit.use_draw_offset_scale = True
            plane_hit.use_draw_modal = False
            self.move_plane_hit_gizmos[plane_name] = plane_hit

        move_center = self.gizmos.new("GIZMO_GT_dial_3d")
        move_center.color = (0.82, 0.82, 0.82)
        move_center.alpha = 0.9
        move_center.color_highlight = (0.82, 0.82, 0.82)
        move_center.alpha_highlight = 0.9
        move_center.line_width = 1.5
        move_center.scale_basis = 0.18
        move_center.hide_select = True
        move_center.use_draw_modal = True
        self.move_center_gizmo = move_center

        move_center_hit = self.gizmos.new("GIZMO_GT_dial_3d")
        move_center_props = move_center_hit.target_set_operator(BAW_OT_rigped_fit_transform_axis.bl_idname)
        move_center_props.mode = "MOVE"
        move_center_props.axis = "FREE"
        move_center_hit.color = (0.82, 0.82, 0.82)
        move_center_hit.alpha = 0.001
        move_center_hit.color_highlight = (1.0, 1.0, 1.0)
        move_center_hit.alpha_highlight = 1.0
        move_center_hit.line_width = 1.5
        move_center_hit.scale_basis = 0.20
        move_center_hit.select_bias = 2.0
        move_center_hit.use_draw_modal = False
        self.move_center_hit_gizmo = move_center_hit

        view_rotate = self.gizmos.new("GIZMO_GT_dial_3d")
        view_props = view_rotate.target_set_operator(BAW_OT_rigped_fit_transform_axis.bl_idname)
        view_props.mode = "ROTATE"
        view_props.axis = "VIEW"
        view_rotate.color = (0.85, 0.85, 0.85)
        view_rotate.alpha = 0.45
        view_rotate.color_highlight = GIZMO_HIGHLIGHT_COLOR
        view_rotate.alpha_highlight = 1.0
        view_rotate.line_width = 1.5
        view_rotate.scale_basis = 1.15
        view_rotate.use_draw_modal = True
        self.rotate_gizmos["VIEW"] = view_rotate

    def draw_prepare(self, context):
        session_mode = fit_transform_mode(context)
        pivot = None
        rig = getattr(context, "active_object", None)
        bone = _active_edit_bone(context)
        axes = _fit_orientation_axes(context)
        if rig is not None and bone is not None:
            pivot = Vector(rig.matrix_world @ _fit_pivot_local(bone))
        if pivot is None or axes is None:
            for gizmo in (
                *self.move_gizmos.values(),
                *self.move_hit_gizmos.values(),
                *self.move_plane_gizmos.values(),
                *self.move_plane_hit_gizmos.values(),
                self.move_center_gizmo,
                self.move_center_hit_gizmo,
                *self.rotate_gizmos.values(),
            ):
                gizmo.hide = True
            return

        gizmo_scale, self._zoom_reference_distance = gizmo_zoom_scale(
            context,
            self._zoom_reference_distance,
        )
        self.move_center_gizmo.scale_basis = 0.18 * gizmo_scale
        self.move_center_hit_gizmo.scale_basis = 0.20 * gizmo_scale
        for axis_name, gizmo in self.move_gizmos.items():
            hidden = session_mode != "MOVE"
            gizmo.hide = hidden
            hit = self.move_hit_gizmos[axis_name]
            hit.hide = hidden
            gizmo.scale_basis = gizmo_scale
            hit.scale_basis = gizmo_scale
            if hidden:
                continue
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(axes[axis_name])
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            base_color = {
                "X": (0.86, 0.16, 0.12),
                "Y": (0.20, 0.72, 0.18),
                "Z": (0.16, 0.38, 0.92),
            }[axis_name]
            gizmo.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else base_color
            )
            gizmo.matrix_basis = matrix
            hit.matrix_basis = matrix

        move_plane_axes = {
            "XY": ("X", "Y"),
            "XZ": ("X", "Z"),
            "YZ": ("Y", "Z"),
        }
        for plane_name, gizmo in self.move_plane_gizmos.items():
            hidden = session_mode != "MOVE"
            gizmo.hide = hidden
            hit = self.move_plane_hit_gizmos[plane_name]
            hit.hide = hidden
            gizmo.scale_basis = 0.11 * gizmo_scale
            hit.scale_basis = 0.14 * gizmo_scale
            if hidden:
                continue
            first_name, second_name = move_plane_axes[plane_name]
            first = Vector(axes[first_name])
            second = Vector(axes[second_name])
            normal = first.cross(second).normalized()
            matrix = Matrix((first, second, normal)).transposed().to_4x4()
            # Place each plane handle in its own axis quadrant instead of
            # stacking all three on the pivot. Use a small world-space offset
            # derived from the active bone length so XY/XZ/YZ separate like
            # Blender's native Move gizmo while staying close to the center.
            world_length = Vector(
                rig.matrix_world.to_3x3() @ (bone.tail - bone.head)
            ).length
            plane_offset = max(0.025, min(0.07, float(world_length) * 0.14))
            matrix.translation = pivot + (first + second) * plane_offset
            offset_matrix = Matrix.Identity(4)
            plane_base_color = {
                "XY": (0.16, 0.38, 0.92),
                "XZ": (0.20, 0.72, 0.18),
                "YZ": (0.86, 0.16, 0.12),
            }[plane_name]
            gizmo.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else plane_base_color
            )
            gizmo.matrix_basis = matrix
            gizmo.matrix_offset = offset_matrix
            hit.matrix_basis = matrix
            hit.matrix_offset = offset_matrix

        view_axis = _fit_view_axis(context)
        center_hidden = session_mode != "MOVE" or view_axis is None
        self.move_center_gizmo.hide = center_hidden
        self.move_center_hit_gizmo.hide = center_hidden
        if not center_hidden:
            center_rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(view_axis)
            center_matrix = center_rotation.to_matrix().to_4x4()
            center_matrix.translation = pivot
            self.move_center_gizmo.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(self.move_center_hit_gizmo, "is_highlight", False))
                else (0.82, 0.82, 0.82)
            )
            self.move_center_gizmo.matrix_basis = center_matrix
            self.move_center_hit_gizmo.matrix_basis = center_matrix

        for axis_name, gizmo in self.rotate_gizmos.items():
            gizmo.hide = session_mode != "ROTATE"
            gizmo.scale_basis = (1.15 if axis_name == "VIEW" else 1.0) * gizmo_scale
            if gizmo.hide:
                continue
            axis = view_axis if axis_name == "VIEW" else axes[axis_name]
            if axis is None:
                gizmo.hide = True
                continue
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(axis)
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            gizmo.matrix_basis = matrix


class BAW_GGT_rigped_fit_scale(bpy.types.GizmoGroup):
    """Fit-only native-style local structural Scale gizmo.

    Axis handles edit one local dimension, the three small plane handles edit
    two local dimensions together, and the white annulus scales X/Y/Z together.
    All gestures stay inside the Fit-local history; no PoseBone scale is authored.
    """

    bl_idname = "BAW_GGT_rigped_fit_scale"
    bl_label = "AWB Rigped Fit Local Scale"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    # Keep modal drawing opt-in per visible handle. SHOW_MODAL_ALL makes every
    # Scale handle participate in another handle's drag and also mirrors the
    # old Move cross-contamination problem.
    bl_options: ClassVar[set[str]] = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context):
        return fit_scale_available(context) and fit_transform_mode(context) == "SCALE"

    def setup(self, _context):
        self._zoom_reference_distance = None
        self.axis_gizmos = {}
        self.axis_hit_gizmos = {}
        self.plane_gizmos = {}
        self.plane_hit_gizmos = {}
        colors = {
            "X": (0.86, 0.16, 0.12),
            "Y": (0.20, 0.72, 0.18),
            "Z": (0.16, 0.38, 0.92),
        }
        for axis_name in ("X", "Y", "Z"):
            # Keep the standard box-ended X/Y/Z Scale handles visible. The
            # separate operator-bound proxy remains invisible so modal drag
            # cannot pull the displayed gizmo away from the Fit pivot.
            gizmo = self.gizmos.new("GIZMO_GT_arrow_3d")
            gizmo.draw_style = "BOX"
            gizmo.draw_options = {"STEM"}
            gizmo.color = colors[axis_name]
            gizmo.alpha = 0.8
            gizmo.color_highlight = colors[axis_name]
            gizmo.alpha_highlight = 0.8
            gizmo.line_width = 1.0
            gizmo.scale_basis = 1.0
            gizmo.hide_select = True
            gizmo.use_draw_modal = True
            self.axis_gizmos[axis_name] = gizmo

            hit = self.gizmos.new("GIZMO_GT_arrow_3d")
            props = hit.target_set_operator(BAW_OT_rigped_fit_scale_axis.bl_idname)
            props.axis = axis_name
            hit.draw_style = "BOX"
            # Interaction belongs to the box head only. Keeping the hit-proxy
            # STEM selectable lets near-camera axes (especially local Z) steal
            # clicks from the uniform center handle even when the user is
            # clearly aiming at the pivot.
            hit.draw_options = set()
            hit.color = colors[axis_name]
            hit.alpha = 0.001
            hit.color_highlight = colors[axis_name]
            hit.alpha_highlight = 0.001
            hit.line_width = 1.0
            hit.scale_basis = 1.0
            hit.select_bias = 2.0
            hit.use_draw_modal = False
            self.axis_hit_gizmos[axis_name] = hit

        # Scale intentionally exposes only the three local X/Y/Z axis handles
        # plus uniform XYZ. Two-axis XY/XZ/YZ plane squares are omitted from
        # both display and hit-testing to keep the Fit Scale gizmo uncluttered.

        # Uniform XYZ Scale is presented as one clean circle. The previous
        # primitive ANNULUS rendered inner/outer borders and looked like two
        # overlapping white rings, which made the Scale gizmo unnecessarily
        # ambiguous next to the three axis handles.
        uniform = self.gizmos.new("GIZMO_GT_dial_3d")
        uniform.color = (0.88, 0.88, 0.88)
        uniform.alpha = 0.55
        uniform.color_highlight = (0.88, 0.88, 0.88)
        uniform.alpha_highlight = 0.55
        uniform.line_width = 1.5
        uniform.scale_basis = 0.72
        uniform.hide_select = True
        uniform.use_draw_modal = True
        self.uniform_gizmo = uniform

        uniform_hit = self.gizmos.new("GIZMO_GT_dial_3d")
        uniform_props = uniform_hit.target_set_operator(BAW_OT_rigped_fit_scale_axis.bl_idname)
        uniform_props.axis = "XYZ"
        # Hover/click follows the visible ring itself. Axis interaction is now
        # limited to the X/Y/Z box heads, so this circular proxy can safely own
        # the ring circumference without stealing axis-handle input.
        uniform_hit.color = (0.88, 0.88, 0.88)
        uniform_hit.alpha = 0.001
        uniform_hit.color_highlight = (0.88, 0.88, 0.88)
        uniform_hit.alpha_highlight = 0.001
        uniform_hit.line_width = 1.5
        uniform_hit.scale_basis = 0.72
        uniform_hit.select_bias = 2.0
        uniform_hit.use_draw_modal = False
        self.uniform_hit_gizmo = uniform_hit

    def draw_prepare(self, context):
        bone = _active_edit_bone(context)
        rig = getattr(context, "active_object", None)
        if bone is None or rig is None:
            for gizmo in (
                *self.axis_gizmos.values(),
                *self.axis_hit_gizmos.values(),
                *self.plane_gizmos.values(),
                *self.plane_hit_gizmos.values(),
                self.uniform_gizmo,
                self.uniform_hit_gizmo,
            ):
                gizmo.hide = True
            return

        gizmo_scale, self._zoom_reference_distance = gizmo_zoom_scale(
            context,
            self._zoom_reference_distance,
        )
        self.uniform_gizmo.scale_basis = 0.72 * gizmo_scale
        self.uniform_hit_gizmo.scale_basis = 0.72 * gizmo_scale
        bone_basis = (rig.matrix_world @ bone.matrix).to_3x3()
        pivot = Vector(rig.matrix_world @ _fit_pivot_local(bone))
        local_axes = {
            "X": Vector(bone_basis @ Vector((1.0, 0.0, 0.0))).normalized(),
            "Y": Vector(bone_basis @ Vector((0.0, 1.0, 0.0))).normalized(),
            "Z": Vector(bone_basis @ Vector((0.0, 0.0, 1.0))).normalized(),
        }
        for axis_name, gizmo in self.axis_gizmos.items():
            hit = self.axis_hit_gizmos[axis_name]
            gizmo.scale_basis = gizmo_scale
            hit.scale_basis = gizmo_scale
            gizmo.hide = False
            hit.hide = False
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(local_axes[axis_name])
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            base_color = {
                "X": (0.86, 0.16, 0.12),
                "Y": (0.20, 0.72, 0.18),
                "Z": (0.16, 0.38, 0.92),
            }[axis_name]
            gizmo.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else base_color
            )
            gizmo.matrix_basis = matrix
            hit.matrix_basis = matrix

        plane_axes = {
            "XY": ("X", "Y"),
            "XZ": ("X", "Z"),
            "YZ": ("Y", "Z"),
        }
        world_length = Vector(rig.matrix_world.to_3x3() @ (bone.tail - bone.head)).length
        offset = max(0.035, min(0.12, float(world_length) * 0.18))
        for plane_name, gizmo in self.plane_gizmos.items():
            hit = self.plane_hit_gizmos[plane_name]
            gizmo.scale_basis = 0.11 * gizmo_scale
            hit.scale_basis = 0.14 * gizmo_scale
            first_name, second_name = plane_axes[plane_name]
            first = local_axes[first_name]
            second = local_axes[second_name]
            normal = first.cross(second).normalized()
            matrix = Matrix((first, second, normal)).transposed().to_4x4()
            matrix.translation = pivot + (first + second) * offset
            base_color = {
                "XY": (0.72, 0.62, 0.12),
                "XZ": (0.66, 0.26, 0.62),
                "YZ": (0.16, 0.62, 0.62),
            }[plane_name]
            gizmo.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(hit, "is_highlight", False))
                else base_color
            )
            gizmo.matrix_basis = matrix
            hit.matrix_basis = matrix
            gizmo.hide = False
            hit.hide = False

        view_axis = _fit_view_axis(context)
        if view_axis is None:
            self.uniform_gizmo.hide = True
            self.uniform_hit_gizmo.hide = True
        else:
            rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(view_axis)
            matrix = rotation.to_matrix().to_4x4()
            matrix.translation = pivot
            self.uniform_gizmo.color = (
                (1.0, 1.0, 1.0)
                if bool(getattr(self.uniform_hit_gizmo, "is_highlight", False))
                else (0.88, 0.88, 0.88)
            )
            self.uniform_gizmo.matrix_basis = matrix
            self.uniform_hit_gizmo.matrix_basis = matrix
            self.uniform_gizmo.hide = False
            self.uniform_hit_gizmo.hide = False
