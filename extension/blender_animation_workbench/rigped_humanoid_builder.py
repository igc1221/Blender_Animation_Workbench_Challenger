from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import bpy
from mathutils import Matrix, Vector

from .character_metadata import (
    add_character_chain,
    add_character_group,
    add_kinematic_mapping,
    add_opposite_pair,
    bind_pose_bone,
    character_ids,
    create_character,
    read_character,
    remove_character,
    remove_character_binding,
    resolve_character,
    set_binding_semantics,
)
from .phase4_contact_model import (
    AWB_CONTACT_EXTERNAL_CONSTRAINT,
    AWB_CONTACT_EXTERNAL_PLACEHOLDER_PROPERTY,
    AWB_CONTACT_PIVOT_CONSTRAINT,
    AWB_CONTACT_STATE_PROPERTY,
    ContactStateValue,
)
from .phase4_verification import (
    NoopStageHook,
    OperationStage,
    RollbackIncompleteError,
    StageHook,
)
from .rigped_contract import (
    RIGPED_SETUP_PROPERTY,
    RIGPED_SETUP_SCHEMA_VERSION,
    RigpedCapability,
    RigpedLifecycle,
    compute_setup_signature,
    control_contracts_for_view,
    read_setup_descriptor,
)
from .rigped_humanoid_spec import (
    HumanoidBoneSpec,
    HumanoidConstraintSpec,
    HumanoidLayer,
    RigpedHumanoidSpec,
)


class RigpedHumanoidBuildError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RigpedHumanoidBuildResult:
    character_id: str
    collection: Any
    armature_object: Any
    armature_data: Any
    external_contact_placeholder: Any
    setup_revision: int
    setup_signature: str
    binding_ids_by_role: tuple[tuple[str, str], ...]
    chain_ids_by_key: tuple[tuple[str, str], ...]
    mapping_ids_by_key: tuple[tuple[str, str], ...]


@dataclass(slots=True)
class _BuildJournal:
    collection: Any | None = None
    armature_object: Any | None = None
    armature_data: Any | None = None
    external_contact_placeholder: Any | None = None
    character_id: str | None = None
    previous_active: Any | None = None
    previous_selected: tuple[Any, ...] = ()
    residue: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _SemanticBuild:
    character_id: str
    binding_ids_by_role: tuple[tuple[str, str], ...]
    chain_ids_by_key: tuple[tuple[str, str], ...]
    mapping_ids_by_key: tuple[tuple[str, str], ...]


def _pointer(value) -> int | None:
    if value is None:
        return None
    try:
        pointer = getattr(value, "as_pointer", None)
        if not callable(pointer):
            return None
        result = int(pointer())
    except ReferenceError:
        return None
    return result or None


def _identity_alive(value, candidates) -> bool:
    pointer = _pointer(value)
    return pointer is not None and any(_pointer(candidate) == pointer for candidate in candidates)


def _object_alive(obj) -> bool:
    return _identity_alive(obj, bpy.data.objects)


def _armature_alive(data) -> bool:
    return _identity_alive(data, bpy.data.armatures)


def _collection_alive(collection) -> bool:
    return _identity_alive(collection, bpy.data.collections)


def _scene_contains_object(scene, obj) -> bool:
    pointer = _pointer(obj)
    return pointer is not None and any(_pointer(candidate) == pointer for candidate in scene.objects)


def _preflight(scene, spec: RigpedHumanoidSpec) -> None:
    spec.validate()
    if scene is None:
        raise RigpedHumanoidBuildError("A target Scene is required.")
    if getattr(scene, "library", None) is not None or getattr(scene, "is_editable", True) is False:
        raise RigpedHumanoidBuildError("Humanoid generation requires a local writable Scene.")
    if bpy.context.scene is not scene:
        raise RigpedHumanoidBuildError("Humanoid generation requires the target Scene to be active.")
    if str(getattr(bpy.context, "mode", "OBJECT")) != "OBJECT":
        raise RigpedHumanoidBuildError("Humanoid generation currently requires Object Mode.")


def _snapshot_context(journal: _BuildJournal) -> None:
    journal.previous_active = bpy.context.view_layer.objects.active
    journal.previous_selected = tuple(bpy.context.selected_objects)


def _restore_context(scene, journal: _BuildJournal) -> None:
    if str(getattr(bpy.context, "mode", "OBJECT")) != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
    for obj in scene.objects:
        try:
            obj.select_set(False)
        except ReferenceError:
            continue
    for obj in journal.previous_selected:
        if _object_alive(obj) and _scene_contains_object(scene, obj):
            obj.select_set(True)
    active = journal.previous_active
    bpy.context.view_layer.objects.active = (
        active if _object_alive(active) and _scene_contains_object(scene, active) else None
    )


def _create_container(scene, spec: RigpedHumanoidSpec, journal: _BuildJournal):
    collection = bpy.data.collections.new(spec.collection_name)
    scene.collection.children.link(collection)
    journal.collection = collection

    armature_data = bpy.data.armatures.new(spec.armature_data_name)
    armature_data.display_type = "BBONE"
    journal.armature_data = armature_data
    armature_object = bpy.data.objects.new(spec.armature_object_name, armature_data)
    armature_object.location = spec.world_location
    armature_object.show_in_front = True
    collection.objects.link(armature_object)
    journal.armature_object = armature_object

    placeholder = bpy.data.objects.new(f"{spec.armature_object_name}.ContactSpace", None)
    placeholder.hide_viewport = True
    placeholder.hide_render = True
    placeholder[AWB_CONTACT_EXTERNAL_PLACEHOLDER_PROPERTY] = True
    collection.objects.link(placeholder)
    journal.external_contact_placeholder = placeholder
    return armature_object


def _align_generated_bone_roll(bone) -> None:
    """Give generated bones one deterministic anatomical local-axis convention.

    Blender fixes local Y to head->tail and derives X/Z from EditBone.roll.
    Generated Rigped bones mostly lie in the character X/Z plane, so keeping
    local Z pointed toward character +Y gives authored/MCH/DEF counterparts a
    stable shared basis. Pole controls that run almost parallel to +Y use +Z as
    the fallback roll reference to avoid a degenerate projection.
    """

    direction = Vector(bone.tail) - Vector(bone.head)
    if direction.length <= 1e-9:
        return
    direction.normalize()
    reference = Vector((0.0, 1.0, 0.0))
    if abs(float(direction.dot(reference))) >= 0.999:
        reference = Vector((0.0, 0.0, 1.0))
    bone.align_roll(reference)


def _create_bones(
    armature_object,
    bones: tuple[HumanoidBoneSpec, ...],
    *,
    display_scale: float = 1.0,
) -> None:
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    armature_object.select_set(True)
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        created: dict[str, Any] = {}
        for entry in bones:
            bone = armature_object.data.edit_bones.new(entry.name)
            bone.head = entry.head
            bone.tail = entry.tail
            _align_generated_bone_roll(bone)
            bone.use_deform = bool(entry.deform)
            created[entry.role] = bone
        for entry in bones:
            if entry.parent_role is None:
                continue
            bone = created[entry.role]
            bone.parent = created[entry.parent_role]
            bone.use_connect = bool(entry.connected)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    # Blender initializes B-Bone display width/depth to fixed absolute values.
    # Scale those defaults with the same factor used for generated rest geometry
    # so creation height changes overall size without changing apparent body build.
    display_scale = max(1e-6, float(display_scale))
    for data_bone in armature_object.data.bones:
        data_bone.bbone_x = max(1e-6, float(data_bone.bbone_x) * display_scale)
        data_bone.bbone_z = max(1e-6, float(data_bone.bbone_z) * display_scale)

    # Biped-style default display sizing. Pelvis ends at each Thigh center, so
    # each thigh is visually overlapped by about half its width without moving
    # either hip socket. COM starts as a square/cube-like box centered in Pelvis.
    pelvis = armature_object.data.bones.get("Pelvis")
    com = armature_object.data.bones.get("COM")
    thigh_l = armature_object.data.bones.get("Thigh.L")
    thigh_r = armature_object.data.bones.get("Thigh.R")
    if pelvis is not None and thigh_l is not None and thigh_r is not None:
        hip_half_span = 0.5 * float((thigh_l.head_local - thigh_r.head_local).length)
        pelvis.bbone_x = max(1e-6, hip_half_span)
    if com is not None:
        com_half_size = max(1e-6, 0.5 * float(com.length))
        com.bbone_x = com_half_size
        com.bbone_z = com_half_size

    # Biped-style pelvis hierarchy: pelvis is a pose joint, not the whole-body
    # rotational root. Spine is parented directly to COM in the humanoid spec;
    # upper-leg links keep pelvis-relative hip position but break pelvis rotation
    # inheritance. This encodes the hierarchy natively without compensation keys.
    by_role = {entry.role: entry for entry in bones}
    for entry in bones:
        if entry.parent_role is None:
            continue
        parent = by_role.get(entry.parent_role)
        if parent is None or parent.semantic_key != "awb.pelvis":
            continue
        if entry.semantic_key not in {"awb.spine", "awb.thigh"}:
            continue
        data_bone = armature_object.data.bones.get(entry.name)
        if data_bone is not None:
            data_bone.use_inherit_rotation = False


def _setup_bone_collections(
    armature_object,
    bones: tuple[HumanoidBoneSpec, ...],
) -> None:
    collections = {
        layer: armature_object.data.collections.new(layer.value.title())
        for layer in HumanoidLayer
    }
    for layer, collection in collections.items():
        collection.is_visible = layer is HumanoidLayer.AUTHORED

    # Internal IK targets/poles must be hidden from the instant the rig is
    # created. Previously they were first assigned to the visible Authored
    # collection and only migrated into a hidden collection when Fit/Animate
    # was entered, which exposed the helper boxes immediately after Create.
    helper_collection = armature_object.data.collections.new("AWB Internal IK")
    helper_collection.is_visible = False
    helper_prefixes = ("IK_HAND.", "IK_FOOT.", "POLE_ELBOW.", "POLE_KNEE.")

    by_name = armature_object.data.bones
    for entry in bones:
        bone = by_name.get(entry.name)
        if bone is None:
            raise RigpedHumanoidBuildError(
                f"Generated bone {entry.name!r} is missing before collection assignment."
            )
        if entry.role.startswith(helper_prefixes):
            helper_collection.assign(bone)
            bone.hide = True
            bone.hide_select = True
        else:
            collections[entry.layer].assign(bone)


def _constraint_name(spec: HumanoidConstraintSpec) -> str:
    target = spec.target_role.replace(".", "_")
    if spec.kind == "IK":
        return f"AWB_IK_{target}"
    if spec.kind == "COPY_ROTATION":
        return f"AWB_ORIENT_{target}"
    if spec.kind == "PIVOT":
        return AWB_CONTACT_PIVOT_CONSTRAINT
    return f"AWB_COPY_{target}"


# Generated Rigped Animate keeps Biped-like freedom while refusing obviously
# non-human solver states. Elbow/knee are revolute joints rather than full
# 180-degree hinges; leaving a few degrees before full fold also avoids native
# two-bone singularity/candy-wrapper behavior near pi.
_RIGPED_HINGE_MAX_RADIANS = math.radians(155.0)
_RIGPED_UPPER_ARM_SWING_RADIANS = math.radians(170.0)
_RIGPED_UPPER_ARM_TWIST_RADIANS = math.radians(130.0)
_RIGPED_THIGH_SWING_RADIANS = math.radians(165.0)
_RIGPED_THIGH_TWIST_RADIANS = math.radians(110.0)


def _default_generated_rigped_hinge_branch(owner_role: str) -> int:
    return -1 if str(owner_role) == "MCH_FOREARM.R" else 1


def configure_generated_rigped_ik_hinge_branch(
    owner,
    branch_sign: int,
    *,
    owner_role: str | None = None,
) -> bool:
    """Keep one hidden solver joint on the requested two-bone bend branch.

    The branch is runtime/authoring state, not a left/right anatomical constant:
    a valid FK pose may cross the generated bone's local hinge zero. Only the
    hidden MCH solver joint is limited; public ForeArm/Calf controls stay free of
    native IK constraints and IK-axis restrictions.
    """

    role = str(owner_role or "")
    if not role:
        role = {
            "MCH_ForeArm.L": "MCH_FOREARM.L",
            "MCH_ForeArm.R": "MCH_FOREARM.R",
            "MCH_Calf.L": "MCH_CALF.L",
            "MCH_Calf.R": "MCH_CALF.R",
        }.get(str(getattr(owner, "name", "")), "")
    if role not in {"MCH_FOREARM.L", "MCH_FOREARM.R", "MCH_CALF.L", "MCH_CALF.R"}:
        return False
    sign = 1 if int(branch_sign) >= 0 else -1
    before = (
        bool(owner.lock_ik_x),
        bool(owner.lock_ik_y),
        bool(owner.lock_ik_z),
        bool(owner.use_ik_limit_x),
        bool(owner.use_ik_limit_z),
        float(owner.ik_min_x),
        float(owner.ik_max_x),
        float(owner.ik_min_z),
        float(owner.ik_max_z),
    )

    owner.lock_ik_x = False
    owner.lock_ik_y = False
    owner.lock_ik_z = False
    owner.use_ik_limit_x = False
    owner.use_ik_limit_y = False
    owner.use_ik_limit_z = False

    # Hidden elbow/knee solver joints expose only the same two meaningful DOFs
    # as their public Rigped controls: one anatomical hinge plus limited local-Y
    # axial roll. Leaving the third axis free lets Blender's native IK satisfy a
    # compound target by leaking large off-hinge swing into MCH_ForeArm/Calf;
    # that invalid rotation is then copied back onto the public control during
    # Sliding. Keep Y available for authored long-axis roll and lock only the
    # truly non-anatomical solver axis.
    if role in {"MCH_FOREARM.L", "MCH_FOREARM.R"}:
        owner.lock_ik_x = True
        owner.use_ik_limit_z = True
        if sign > 0:
            owner.ik_min_z = 0.0
            owner.ik_max_z = _RIGPED_HINGE_MAX_RADIANS
        else:
            owner.ik_min_z = -_RIGPED_HINGE_MAX_RADIANS
            owner.ik_max_z = 0.0
    else:
        owner.lock_ik_z = True
        owner.use_ik_limit_x = True
        if sign > 0:
            owner.ik_min_x = 0.0
            owner.ik_max_x = _RIGPED_HINGE_MAX_RADIANS
        else:
            owner.ik_min_x = -_RIGPED_HINGE_MAX_RADIANS
            owner.ik_max_x = 0.0

    after = (
        bool(owner.lock_ik_x),
        bool(owner.lock_ik_y),
        bool(owner.lock_ik_z),
        bool(owner.use_ik_limit_x),
        bool(owner.use_ik_limit_z),
        float(owner.ik_min_x),
        float(owner.ik_max_x),
        float(owner.ik_min_z),
        float(owner.ik_max_z),
    )
    return before != after


def _configure_generated_rigped_upper_ik_envelope(owner, owner_role: str) -> bool:
    """Apply a broad Biped-like spherical envelope to hidden shoulder/hip joints."""

    role = str(owner_role)
    if role not in {
        "MCH_UPPER_ARM.L",
        "MCH_UPPER_ARM.R",
        "MCH_THIGH.L",
        "MCH_THIGH.R",
    }:
        return False
    if role.startswith("MCH_UPPER_ARM"):
        swing = _RIGPED_UPPER_ARM_SWING_RADIANS
        twist = _RIGPED_UPPER_ARM_TWIST_RADIANS
    else:
        swing = _RIGPED_THIGH_SWING_RADIANS
        twist = _RIGPED_THIGH_TWIST_RADIANS

    before = (
        bool(owner.lock_ik_x),
        bool(owner.lock_ik_y),
        bool(owner.lock_ik_z),
        bool(owner.use_ik_limit_x),
        bool(owner.use_ik_limit_y),
        bool(owner.use_ik_limit_z),
        float(owner.ik_min_x),
        float(owner.ik_max_x),
        float(owner.ik_min_y),
        float(owner.ik_max_y),
        float(owner.ik_min_z),
        float(owner.ik_max_z),
    )
    owner.lock_ik_x = False
    owner.lock_ik_y = False
    owner.lock_ik_z = False
    owner.use_ik_limit_x = True
    owner.use_ik_limit_y = True
    owner.use_ik_limit_z = True
    # Blender bones use local Y along the link. Keep broad X/Z swing and a
    # slightly tighter axial-Y twist envelope; this is deliberately permissive
    # like Biped, but prevents solver inversion through a full half-turn.
    owner.ik_min_x = -swing
    owner.ik_max_x = swing
    owner.ik_min_y = -twist
    owner.ik_max_y = twist
    owner.ik_min_z = -swing
    owner.ik_max_z = swing
    after = (
        bool(owner.lock_ik_x),
        bool(owner.lock_ik_y),
        bool(owner.lock_ik_z),
        bool(owner.use_ik_limit_x),
        bool(owner.use_ik_limit_y),
        bool(owner.use_ik_limit_z),
        float(owner.ik_min_x),
        float(owner.ik_max_x),
        float(owner.ik_min_y),
        float(owner.ik_max_y),
        float(owner.ik_min_z),
        float(owner.ik_max_z),
    )
    return before != after


def ensure_generated_rigped_ik_hinge_limits(armature_object) -> bool:
    """Upgrade generated Rigpeds to the hidden anatomical IK envelope."""

    pose = getattr(getattr(armature_object, "pose", None), "bones", None)
    if pose is None:
        return False
    changed = False
    for role, bone_name in (
        ("MCH_UPPER_ARM.L", "MCH_UpperArm.L"),
        ("MCH_UPPER_ARM.R", "MCH_UpperArm.R"),
        ("MCH_THIGH.L", "MCH_Thigh.L"),
        ("MCH_THIGH.R", "MCH_Thigh.R"),
    ):
        owner = pose.get(bone_name)
        if owner is not None:
            changed = _configure_generated_rigped_upper_ik_envelope(owner, role) or changed
    for role, bone_name in (
        ("MCH_FOREARM.L", "MCH_ForeArm.L"),
        ("MCH_FOREARM.R", "MCH_ForeArm.R"),
        ("MCH_CALF.L", "MCH_Calf.L"),
        ("MCH_CALF.R", "MCH_Calf.R"),
    ):
        owner = pose.get(bone_name)
        if owner is not None:
            changed = configure_generated_rigped_ik_hinge_branch(
                owner,
                _default_generated_rigped_hinge_branch(role),
                owner_role=role,
            ) or changed
    return changed


def _setup_constraints(
    armature_object,
    bones: tuple[HumanoidBoneSpec, ...],
    constraints: tuple[HumanoidConstraintSpec, ...],
    external_contact_placeholder,
) -> None:
    role_to_name = {bone.role: bone.name for bone in bones}
    pose = armature_object.pose.bones
    for spec in constraints:
        owner = pose[role_to_name[spec.owner_role]]
        constraint = owner.constraints.new(type=spec.kind)
        constraint.name = _constraint_name(spec)
        constraint.target = armature_object
        constraint.subtarget = role_to_name[spec.target_role]
        constraint.influence = float(spec.influence)

        if hasattr(constraint, "target_space"):
            constraint.target_space = spec.target_space
        if hasattr(constraint, "owner_space"):
            constraint.owner_space = spec.owner_space
        if hasattr(constraint, "mix_mode"):
            constraint.mix_mode = spec.mix_mode

        if spec.kind == "PIVOT":
            constraint.rotation_range = "ALWAYS_ACTIVE"
            constraint.offset = (0.0, 0.0, 0.0)

        if spec.kind == "IK":
            owner[AWB_CONTACT_STATE_PROPERTY] = float(ContactStateValue.UNINITIALIZED)
            if spec.pole_role is not None:
                constraint.pole_target = armature_object
                constraint.pole_subtarget = role_to_name[spec.pole_role]
            constraint.chain_count = int(spec.chain_count)
            constraint.use_tail = bool(spec.use_tail)
            constraint.use_stretch = bool(spec.use_stretch)
            constraint.use_rotation = bool(spec.use_rotation)
            configure_generated_rigped_ik_hinge_branch(
                owner,
                _default_generated_rigped_hinge_branch(spec.owner_role),
                owner_role=spec.owner_role,
            )

    # Upper joints participate in the same two-bone solve even though the IK
    # constraint itself lives on the lower MCH joint. Apply their broad
    # anatomical envelope only after the whole generated pose chain exists.
    ensure_generated_rigped_ik_hinge_limits(armature_object)

    # I18 adds a dormant parent-like external Object space to every generated
    # Contact hold carrier. A Rigped-owned hidden placeholder keeps the
    # structural target class stable before/after user binding while influence
    # stays zero. The user Object remains entirely external/user-owned.
    for entry in bones:
        if entry.semantic_key != "awb.contact" or entry.usage != "MECHANISM":
            continue
        owner = pose[entry.name]
        external = owner.constraints.new(type="CHILD_OF")
        external.name = AWB_CONTACT_EXTERNAL_CONSTRAINT
        external.influence = 0.0
        external.target = external_contact_placeholder
        external.target_space = "WORLD"
        external.owner_space = "WORLD"
        external.inverse_matrix = Matrix.Identity(4)


def _bootstrap_character(scene, armature_object, spec, journal) -> tuple[str, str]:
    character_id = create_character(scene, spec.character_label, (armature_object,))
    journal.character_id = character_id
    initial = read_character(scene, character_id)
    if initial is None:
        raise RigpedHumanoidBuildError("Generated Character could not be read after creation.")
    object_bindings = tuple(
        binding.binding_id
        for binding in initial.bindings
        if str(binding.kind.value) == "OBJECT"
    )
    if len(object_bindings) != 1:
        raise RigpedHumanoidBuildError("Generated Character requires exactly one bootstrap Object binding.")
    return character_id, object_bindings[0]


def _semantic_build(
    scene,
    armature_object,
    spec: RigpedHumanoidSpec,
    journal: _BuildJournal,
) -> _SemanticBuild:
    character_id, bootstrap_binding = _bootstrap_character(scene, armature_object, spec, journal)
    binding_by_role: dict[str, str] = {}

    for entry in spec.resolved_bones():
        pose_bone = armature_object.pose.bones.get(entry.name)
        if pose_bone is None:
            raise RigpedHumanoidBuildError(f"Generated PoseBone {entry.name!r} is missing.")
        binding_id = bind_pose_bone(scene, character_id, pose_bone)
        set_binding_semantics(
            scene,
            character_id,
            binding_id,
            semantic_key=entry.semantic_key,
            side=entry.side,
            mode=entry.mode,
            usage=entry.usage,
        )
        binding_by_role[entry.role] = binding_id

    remove_character_binding(scene, character_id, bootstrap_binding)

    for group in spec.resolved_groups():
        add_character_group(
            scene,
            character_id,
            semantic_key=group.semantic_key,
            label=group.label,
            members=tuple(binding_by_role[role] for role in group.member_roles),
            side=group.side,
        )

    chain_by_key: dict[str, str] = {}
    for chain in spec.resolved_chains():
        chain_by_key[chain.key] = add_character_chain(
            scene,
            character_id,
            semantic_key=chain.semantic_key,
            label=chain.label,
            members=tuple(binding_by_role[role] for role in chain.member_roles),
            side=chain.side,
            mode=chain.mode,
        )

    for pair in spec.resolved_opposites():
        add_opposite_pair(
            scene,
            character_id,
            binding_by_role[pair.left_role],
            binding_by_role[pair.right_role],
        )

    mapping_by_key: dict[str, str] = {}
    for mapping in spec.resolved_kinematics():
        mapping_by_key[mapping.key] = add_kinematic_mapping(
            scene,
            character_id,
            semantic_key=mapping.semantic_key,
            side=mapping.side,
            fk_chain_id=chain_by_key[mapping.fk_chain_key],
            ik_target_binding_id=binding_by_role[mapping.target_role],
            pole_binding_id=binding_by_role[mapping.pole_role],
            reference_chain_id=chain_by_key[mapping.reference_chain_key],
            extras=tuple(binding_by_role[role] for role in mapping.extra_roles),
        )

    return _SemanticBuild(
        character_id,
        tuple(binding_by_role.items()),
        tuple(chain_by_key.items()),
        tuple(mapping_by_key.items()),
    )


def _apply_pose_transform_locks(scene, character_id: str) -> None:
    """Enforce I0 direct-transform capabilities at the native PoseBone layer.

    Before semantic W/E/R routing ships, unsupported connected-limb Move/Scale
    must fail closed instead of writing raw pose channels. Edit/Fit mode is not
    affected by these PoseBone locks.
    """

    view = resolve_character(scene, character_id)
    contracts, issues = control_contracts_for_view(view)
    if issues:
        raise RigpedHumanoidBuildError(
            f"Cannot apply Rigped pose transform locks: {issues!r}"
        )
    for contract in contracts:
        target = contract.target.target
        if not isinstance(target, bpy.types.PoseBone):
            continue
        capabilities = set(contract.capabilities)
        allow_move = RigpedCapability.DIRECT_MOVE in capabilities
        allow_rotate = RigpedCapability.DIRECT_ROTATE in capabilities
        target.lock_location = (not allow_move, not allow_move, not allow_move)
        target.lock_scale = (True, True, True)
        target.lock_rotation = (not allow_rotate, not allow_rotate, not allow_rotate)
        if hasattr(target, "lock_rotation_w"):
            target.lock_rotation_w = not allow_rotate


def _publish_descriptor(scene, build: _SemanticBuild, armature_object, spec: RigpedHumanoidSpec) -> str:
    view = resolve_character(scene, build.character_id)
    signature = compute_setup_signature(view)
    armature_object[RIGPED_SETUP_PROPERTY] = {
        "schema_version": RIGPED_SETUP_SCHEMA_VERSION,
        "character_id": build.character_id,
        "profile_id": spec.profile_id,
        "revision": 1,
        "signature": signature,
        "lifecycle": RigpedLifecycle.FITTED_UNBOUND.value,
    }
    descriptor, issues = read_setup_descriptor(resolve_character(scene, build.character_id))
    if descriptor is None or issues:
        raise RigpedHumanoidBuildError(f"Published Humanoid descriptor is invalid: {issues!r}")
    return signature


def _rollback(scene, journal: _BuildJournal, hook: StageHook) -> None:
    rollback_ordinal = 0

    def attempt(label: str, callback) -> None:
        nonlocal rollback_ordinal
        rollback_ordinal += 1
        try:
            hook.enter(
                OperationStage.ROLLBACK_WRITE,
                ordinal=rollback_ordinal,
                operation="rigped_humanoid_build",
                detail=label,
            )
            callback()
        except Exception as exc:  # noqa: BLE001 - rollback must continue best-effort
            journal.residue.append(f"{label}: {type(exc).__name__}: {exc}")

    if journal.armature_object is not None and _object_alive(journal.armature_object):
        attempt(
            "descriptor",
            lambda: (
                journal.armature_object.pop(RIGPED_SETUP_PROPERTY, None)
                if RIGPED_SETUP_PROPERTY in journal.armature_object
                else None
            ),
        )
    if journal.character_id is not None:
        attempt(
            "character",
            lambda: (
                remove_character(scene, journal.character_id)
                if journal.character_id in character_ids(scene)
                else None
            ),
        )
    if journal.external_contact_placeholder is not None:
        attempt(
            "external_contact_placeholder",
            lambda: (
                bpy.data.objects.remove(journal.external_contact_placeholder, do_unlink=True)
                if _object_alive(journal.external_contact_placeholder)
                else None
            ),
        )
    if journal.armature_object is not None:
        attempt(
            "armature_object",
            lambda: (
                bpy.data.objects.remove(journal.armature_object, do_unlink=True)
                if _object_alive(journal.armature_object)
                else None
            ),
        )
    if journal.armature_data is not None:
        attempt(
            "armature_data",
            lambda: (
                bpy.data.armatures.remove(journal.armature_data)
                if _armature_alive(journal.armature_data)
                else None
            ),
        )
    if journal.collection is not None:
        attempt(
            "collection",
            lambda: (
                bpy.data.collections.remove(journal.collection)
                if _collection_alive(journal.collection)
                else None
            ),
        )

    try:
        _restore_context(scene, journal)
    except Exception as exc:  # noqa: BLE001 - rollback verification must report residue
        journal.residue.append(f"context: {type(exc).__name__}: {exc}")

    hook.enter(
        OperationStage.ROLLBACK_VERIFY,
        operation="rigped_humanoid_build",
        detail="verify operation-owned residue",
    )
    if journal.character_id is not None:
        try:
            if journal.character_id in character_ids(scene):
                journal.residue.append("character row still exists")
        except Exception as exc:  # noqa: BLE001 - rollback verification must report residue
            journal.residue.append(f"character verification failed: {type(exc).__name__}: {exc}")
    if _object_alive(journal.external_contact_placeholder):
        journal.residue.append("external Contact placeholder still exists")
    if _object_alive(journal.armature_object):
        journal.residue.append("armature object still exists")
    if _armature_alive(journal.armature_data):
        journal.residue.append("armature data still exists")
    if _collection_alive(journal.collection):
        journal.residue.append("collection still exists")

    if journal.residue:
        raise RollbackIncompleteError("; ".join(journal.residue))


def build_generated_rigped_humanoid(
    scene,
    spec: RigpedHumanoidSpec | None = None,
    *,
    hook: StageHook | None = None,
) -> RigpedHumanoidBuildResult:
    spec = spec or RigpedHumanoidSpec()
    hook = hook or NoopStageHook()
    journal = _BuildJournal()
    _snapshot_context(journal)

    try:
        hook.enter(OperationStage.PREFLIGHT, operation="rigped_humanoid_build")
        _preflight(scene, spec)
        hook.enter(OperationStage.PLAN, operation="rigped_humanoid_build")

        armature_object = _create_container(scene, spec, journal)
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=1,
            operation="rigped_humanoid_build",
            detail="create humanoid Armature container",
        )

        _create_bones(
            armature_object,
            spec.resolved_bones(),
            display_scale=spec.display_scale,
        )
        _setup_bone_collections(armature_object, spec.resolved_bones())
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=2,
            operation="rigped_humanoid_build",
            detail="create humanoid bone topology and responsibility collections",
        )

        _setup_constraints(
            armature_object,
            spec.resolved_bones(),
            spec.resolved_constraints(),
            journal.external_contact_placeholder,
        )
        bpy.context.view_layer.update()
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=3,
            operation="rigped_humanoid_build",
            detail="create humanoid native constraints",
        )

        semantic = _semantic_build(scene, armature_object, spec, journal)
        _apply_pose_transform_locks(scene, semantic.character_id)
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=4,
            operation="rigped_humanoid_build",
            detail="publish humanoid Character semantic graph and capability transform locks",
        )

        signature = _publish_descriptor(scene, semantic, armature_object, spec)
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=5,
            operation="rigped_humanoid_build",
            detail="publish humanoid setup descriptor",
        )
        hook.enter(OperationStage.VERIFY, operation="rigped_humanoid_build")

        resolved = resolve_character(scene, semantic.character_id)
        descriptor, issues = read_setup_descriptor(resolved)
        if resolved.issues:
            raise RigpedHumanoidBuildError(f"Humanoid Character resolution failed: {resolved.issues!r}")
        if descriptor is None or issues:
            raise RigpedHumanoidBuildError(f"Humanoid descriptor verification failed: {issues!r}")
        if descriptor.revision != 1 or descriptor.signature != signature:
            raise RigpedHumanoidBuildError("Humanoid descriptor verification mismatch.")

        hook.enter(OperationStage.COMMIT, operation="rigped_humanoid_build")
        _restore_context(scene, journal)
        return RigpedHumanoidBuildResult(
            character_id=semantic.character_id,
            collection=journal.collection,
            armature_object=armature_object,
            armature_data=armature_object.data,
            external_contact_placeholder=journal.external_contact_placeholder,
            setup_revision=1,
            setup_signature=signature,
            binding_ids_by_role=semantic.binding_ids_by_role,
            chain_ids_by_key=semantic.chain_ids_by_key,
            mapping_ids_by_key=semantic.mapping_ids_by_key,
        )
    except Exception as exc:
        if isinstance(exc, RollbackIncompleteError):
            raise
        try:
            _rollback(scene, journal, hook)
        except RollbackIncompleteError as rollback_exc:
            raise rollback_exc from exc
        raise


def discard_generated_rigped_humanoid(scene, result: RigpedHumanoidBuildResult) -> None:
    """Remove one freshly created generated Rigped and only its owned data.

    This is intentionally narrow: it exists for the modal Create session before
    the user confirms placement/height. It must not be used as a general rig
    delete operator once foreign data may have been attached.
    """

    residue: list[str] = []

    def attempt(label: str, callback) -> None:
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 - cleanup must continue best-effort
            residue.append(f"{label}: {type(exc).__name__}: {exc}")

    if _object_alive(result.armature_object):
        attempt(
            "descriptor",
            lambda: (
                result.armature_object.pop(RIGPED_SETUP_PROPERTY, None)
                if RIGPED_SETUP_PROPERTY in result.armature_object
                else None
            ),
        )
    attempt(
        "character",
        lambda: (
            remove_character(scene, result.character_id)
            if result.character_id in character_ids(scene)
            else None
        ),
    )
    attempt(
        "external_contact_placeholder",
        lambda: (
            bpy.data.objects.remove(result.external_contact_placeholder, do_unlink=True)
            if _object_alive(result.external_contact_placeholder)
            else None
        ),
    )
    attempt(
        "armature_object",
        lambda: (
            bpy.data.objects.remove(result.armature_object, do_unlink=True)
            if _object_alive(result.armature_object)
            else None
        ),
    )
    attempt(
        "armature_data",
        lambda: (
            bpy.data.armatures.remove(result.armature_data)
            if _armature_alive(result.armature_data)
            else None
        ),
    )
    attempt(
        "collection",
        lambda: (
            bpy.data.collections.remove(result.collection)
            if _collection_alive(result.collection)
            else None
        ),
    )

    try:
        if result.character_id in character_ids(scene):
            residue.append("character row still exists")
    except Exception as exc:  # noqa: BLE001 - verification must report residue
        residue.append(f"character verification failed: {type(exc).__name__}: {exc}")
    if _object_alive(result.external_contact_placeholder):
        residue.append("external Contact placeholder still exists")
    if _object_alive(result.armature_object):
        residue.append("armature object still exists")
    if _armature_alive(result.armature_data):
        residue.append("armature data still exists")
    if _collection_alive(result.collection):
        residue.append("collection still exists")
    if residue:
        raise RollbackIncompleteError("; ".join(residue))
