from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bpy

from .character_metadata import (
    add_character_chain,
    add_character_group,
    add_kinematic_mapping,
    bind_pose_bone,
    character_ids,
    create_character,
    read_character,
    remove_character,
    remove_character_binding,
    resolve_character,
    set_binding_semantics,
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
    RigpedLifecycle,
    compute_setup_signature,
    read_setup_descriptor,
)
from .rigped_core_spec import GeneratedBoneSpec, GeneratedRole, RigpedCoreSpec


class RigpedCoreBuildError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RigpedCoreBuildResult:
    character_id: str
    armature_object: Any
    armature_data: Any
    setup_revision: int
    setup_signature: str
    mapping_id: str
    binding_ids_by_role: tuple[tuple[GeneratedRole, str], ...]


@dataclass(slots=True)
class _BuildJournal:
    collection: Any | None = None
    armature_object: Any | None = None
    armature_data: Any | None = None
    character_id: str | None = None
    descriptor_published: bool = False
    previous_active: Any | None = None
    previous_selected: tuple[Any, ...] = ()
    residue: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _SemanticBuild:
    character_id: str
    binding_ids_by_role: tuple[tuple[GeneratedRole, str], ...]
    mapping_id: str


def _safe_pointer(value) -> int | None:
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


def _object_alive(obj) -> bool:
    target = _safe_pointer(obj)
    return target is not None and any(_safe_pointer(candidate) == target for candidate in bpy.data.objects)


def _armature_alive(data) -> bool:
    target = _safe_pointer(data)
    return target is not None and any(_safe_pointer(candidate) == target for candidate in bpy.data.armatures)


def _collection_alive(collection) -> bool:
    target = _safe_pointer(collection)
    return target is not None and any(_safe_pointer(candidate) == target for candidate in bpy.data.collections)


def _scene_contains_object(scene, obj) -> bool:
    pointer = _safe_pointer(obj)
    return pointer is not None and any(_safe_pointer(candidate) == pointer for candidate in scene.objects)


def _preflight(scene, spec: RigpedCoreSpec) -> None:
    spec.validate()
    if scene is None:
        raise RigpedCoreBuildError("A target Scene is required.")
    if getattr(scene, "library", None) is not None or getattr(scene, "is_editable", True) is False:
        raise RigpedCoreBuildError("Rigped core generation requires a local writable Scene.")
    if bpy.context.scene is not scene:
        raise RigpedCoreBuildError("Rigped core generation requires the target Scene to be active.")
    if str(getattr(bpy.context, "mode", "OBJECT")) != "OBJECT":
        raise RigpedCoreBuildError("Rigped core generation currently requires Object Mode.")


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


def _create_armature(scene, spec: RigpedCoreSpec, journal: _BuildJournal) -> Any:
    collection = bpy.data.collections.new(spec.collection_name)
    scene.collection.children.link(collection)
    journal.collection = collection

    armature_data = bpy.data.armatures.new(spec.armature_data_name)
    journal.armature_data = armature_data
    armature_object = bpy.data.objects.new(spec.armature_object_name, armature_data)
    collection.objects.link(armature_object)
    journal.armature_object = armature_object
    return armature_object


def _create_bones(armature_object, topology: tuple[GeneratedBoneSpec, ...]) -> None:
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    armature_object.select_set(True)
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        created: dict[GeneratedRole, Any] = {}
        for entry in topology:
            bone = armature_object.data.edit_bones.new(entry.name)
            bone.head = entry.head
            bone.tail = entry.tail
            bone.use_deform = bool(entry.deform)
            created[entry.role] = bone
        for entry in topology:
            if entry.parent_role is None:
                continue
            bone = created[entry.role]
            bone.parent = created[entry.parent_role]
            bone.use_connect = bool(entry.connected)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def _copy_constraint(owner, armature_object, target_bone: str, *, influence: float = 1.0):
    constraint = owner.constraints.new(type="COPY_TRANSFORMS")
    constraint.target = armature_object
    constraint.subtarget = target_bone
    constraint.target_space = "WORLD"
    constraint.owner_space = "WORLD"
    constraint.influence = float(influence)
    return constraint


def _setup_native_constraints(armature_object) -> None:
    pose = armature_object.pose.bones

    for owner_name, target_name in (
        ("MCH_UpperArm.L", "UpperArm.L"),
        ("MCH_ForeArm.L", "ForeArm.L"),
        ("MCH_Hand.L", "Hand.L"),
    ):
        constraint = _copy_constraint(pose[owner_name], armature_object, target_name)
        constraint.name = f"AWB_FK_{owner_name}"

    ik = pose["MCH_ForeArm.L"].constraints.new(type="IK")
    ik.name = "AWB_IK_Arm.L"
    ik.target = armature_object
    ik.subtarget = "IK_Hand.L"
    ik.pole_target = armature_object
    ik.pole_subtarget = "IK_Elbow.L"
    ik.chain_count = 2
    ik.use_tail = True
    ik.use_stretch = False
    ik.use_rotation = False
    ik.influence = 0.0

    terminal = pose["MCH_Hand.L"].constraints.new(type="COPY_ROTATION")
    terminal.name = "AWB_IK_HandOrientation.L"
    terminal.target = armature_object
    terminal.subtarget = "IK_Hand.L"
    terminal.target_space = "WORLD"
    terminal.owner_space = "WORLD"
    terminal.mix_mode = "REPLACE"
    terminal.influence = 0.0

    export_root = _copy_constraint(pose["EXP_Root"], armature_object, "Root")
    export_root.name = "AWB_ExportRoot_From_Root"

    for owner_name, target_name in (
        ("DEF_Pelvis", "Pelvis"),
        ("DEF_UpperArm.L", "MCH_UpperArm.L"),
        ("DEF_ForeArm.L", "MCH_ForeArm.L"),
        ("DEF_Hand.L", "MCH_Hand.L"),
    ):
        constraint = _copy_constraint(pose[owner_name], armature_object, target_name)
        constraint.name = f"AWB_Deform_{owner_name}"


def _semantic_build(
    scene,
    armature_object,
    topology: tuple[GeneratedBoneSpec, ...],
    spec: RigpedCoreSpec,
    journal: _BuildJournal,
) -> _SemanticBuild:
    character_id = create_character(scene, spec.character_label, (armature_object,))
    journal.character_id = character_id
    initial = read_character(scene, character_id)
    if initial is None:
        raise RigpedCoreBuildError("Generated Character could not be read after creation.")
    object_binding_ids = tuple(
        binding.binding_id
        for binding in initial.bindings
        if str(binding.kind.value) == "OBJECT"
    )
    if len(object_binding_ids) != 1:
        raise RigpedCoreBuildError("Generated Character requires exactly one bootstrap Object binding.")

    binding_by_role: dict[GeneratedRole, str] = {}
    for entry in topology:
        pose_bone = armature_object.pose.bones.get(entry.name)
        if pose_bone is None:
            raise RigpedCoreBuildError(f"Generated PoseBone {entry.name!r} is missing.")
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

    remove_character_binding(scene, character_id, object_binding_ids[0])

    def members(*roles: GeneratedRole) -> tuple[str, ...]:
        return tuple(binding_by_role[role] for role in roles)

    # Responsibility layers are encoded by binding usage/layer ownership, not
    # by inventing non-canonical Character semantic keys. Keep only groups
    # whose meaning maps to existing canonical semantic roles.
    add_character_group(
        scene,
        character_id,
        semantic_key="awb.root",
        label="Root Motion Source Map",
        members=members(GeneratedRole.ROOT, GeneratedRole.EXPORT_ROOT),
    )
    add_character_group(
        scene,
        character_id,
        semantic_key="awb.hand",
        label="Left Arm Terminal Orientation",
        members=members(
            GeneratedRole.FK_HAND,
            GeneratedRole.IK_EFFECTOR,
            GeneratedRole.MCH_HAND,
        ),
        side="LEFT",
    )

    fk_chain_id = add_character_chain(
        scene,
        character_id,
        semantic_key="awb.arm",
        label="Left Arm FK",
        members=members(GeneratedRole.FK_UPPER, GeneratedRole.FK_FORE),
        side="LEFT",
        mode="FK",
    )
    reference_chain_id = add_character_chain(
        scene,
        character_id,
        semantic_key="awb.arm",
        label="Left Arm Result",
        members=members(GeneratedRole.MCH_UPPER, GeneratedRole.MCH_FORE),
        side="LEFT",
        mode="NEUTRAL",
    )
    mapping_id = add_kinematic_mapping(
        scene,
        character_id,
        semantic_key="awb.arm",
        side="LEFT",
        fk_chain_id=fk_chain_id,
        ik_target_binding_id=binding_by_role[GeneratedRole.IK_EFFECTOR],
        pole_binding_id=binding_by_role[GeneratedRole.IK_POLE],
        reference_chain_id=reference_chain_id,
        extras=(binding_by_role[GeneratedRole.MCH_HAND],),
    )
    return _SemanticBuild(
        character_id,
        tuple((role, binding_by_role[role]) for role in GeneratedRole),
        mapping_id,
    )


def _publish_descriptor(
    scene,
    build: _SemanticBuild,
    armature_object,
    *,
    profile_id: str,
) -> str:
    view = resolve_character(scene, build.character_id)
    signature = compute_setup_signature(view)
    armature_object[RIGPED_SETUP_PROPERTY] = {
        "schema_version": RIGPED_SETUP_SCHEMA_VERSION,
        "character_id": build.character_id,
        "profile_id": profile_id,
        "revision": 1,
        "signature": signature,
        "lifecycle": RigpedLifecycle.FITTED_UNBOUND.value,
    }
    descriptor, issues = read_setup_descriptor(resolve_character(scene, build.character_id))
    if descriptor is None or issues:
        raise RigpedCoreBuildError(f"Published Rigped setup descriptor is invalid: {issues!r}")
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
                operation="rigped_core_build",
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
        operation="rigped_core_build",
        detail="verify operation-owned residue",
    )
    if journal.character_id is not None:
        try:
            if journal.character_id in character_ids(scene):
                journal.residue.append("character row still exists")
        except Exception as exc:  # noqa: BLE001 - rollback verification must report residue
            journal.residue.append(f"character verification failed: {type(exc).__name__}: {exc}")
    if _object_alive(journal.armature_object):
        journal.residue.append("armature object still exists")
    if _armature_alive(journal.armature_data):
        journal.residue.append("armature data still exists")
    if _collection_alive(journal.collection):
        journal.residue.append("collection still exists")

    if journal.residue:
        raise RollbackIncompleteError("; ".join(journal.residue))


def build_generated_rigped_core(
    scene,
    spec: RigpedCoreSpec | None = None,
    *,
    hook: StageHook | None = None,
) -> RigpedCoreBuildResult:
    spec = spec or RigpedCoreSpec()
    hook = hook or NoopStageHook()
    journal = _BuildJournal()
    _snapshot_context(journal)

    try:
        hook.enter(OperationStage.PREFLIGHT, operation="rigped_core_build")
        _preflight(scene, spec)
        topology = spec.resolved_topology()
        hook.enter(OperationStage.PLAN, operation="rigped_core_build")

        armature_object = _create_armature(scene, spec, journal)
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=1,
            operation="rigped_core_build",
            detail="create armature container",
        )

        _create_bones(armature_object, topology)
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=2,
            operation="rigped_core_build",
            detail="create generated bones",
        )

        _setup_native_constraints(armature_object)
        bpy.context.view_layer.update()
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=3,
            operation="rigped_core_build",
            detail="create generated constraints",
        )

        semantic = _semantic_build(scene, armature_object, topology, spec, journal)
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=4,
            operation="rigped_core_build",
            detail="publish Character semantic ownership",
        )

        signature = _publish_descriptor(
            scene,
            semantic,
            armature_object,
            profile_id=spec.profile_id,
        )
        journal.descriptor_published = True
        hook.enter(
            OperationStage.APPLY_WRITE,
            ordinal=5,
            operation="rigped_core_build",
            detail="publish setup descriptor",
        )
        hook.enter(OperationStage.VERIFY, operation="rigped_core_build")

        descriptor, issues = read_setup_descriptor(resolve_character(scene, semantic.character_id))
        if descriptor is None or issues:
            raise RigpedCoreBuildError(f"Rigped core verification failed: {issues!r}")
        if descriptor.signature != signature or descriptor.revision != 1:
            raise RigpedCoreBuildError("Rigped core descriptor verification mismatch.")

        hook.enter(OperationStage.COMMIT, operation="rigped_core_build")
        _restore_context(scene, journal)
        return RigpedCoreBuildResult(
            character_id=semantic.character_id,
            armature_object=armature_object,
            armature_data=armature_object.data,
            setup_revision=1,
            setup_signature=signature,
            mapping_id=semantic.mapping_id,
            binding_ids_by_role=semantic.binding_ids_by_role,
        )
    except Exception as exc:
        if isinstance(exc, RollbackIncompleteError):
            raise
        try:
            _rollback(scene, journal, hook)
        except RollbackIncompleteError as rollback_exc:
            raise rollback_exc from exc
        raise
