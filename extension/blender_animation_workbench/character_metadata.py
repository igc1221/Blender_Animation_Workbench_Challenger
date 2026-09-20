from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import uuid4

import bpy
from bpy.props import CollectionProperty, IntProperty, PointerProperty, StringProperty

from .character_model import (
    AWBCharacter,
    CharacterBinding,
    CharacterChain,
    CharacterGroup,
    CharacterIssue,
    CharacterIssueSeverity,
    CharacterOwner,
    CharacterSide,
    ControlMode,
    ControlUsage,
    KinematicMapping,
    OppositePair,
    has_character_errors,
    validate_character,
)
from .semantic_model import AWBControlKind

SCHEMA_VERSION_V1 = 1
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION_V1, SCHEMA_VERSION})
STORE_PROPERTY = "awb_characters"
BONE_TOKEN_PROPERTY = "awb_character_bone_id"

_RESOLVED_CHARACTER_CACHE_KEY: tuple | None = None
_RESOLVED_CHARACTER_CACHE_VALUE = None


def _invalidate_resolved_character_cache(*_args) -> None:
    """Drop operation-local resolved RNA references after Blender identity changes."""

    global _RESOLVED_CHARACTER_CACHE_KEY, _RESOLVED_CHARACTER_CACHE_VALUE
    _RESOLVED_CHARACTER_CACHE_KEY = None
    _RESOLVED_CHARACTER_CACHE_VALUE = None


def _cached_resolved_character_is_live(value) -> bool:
    """Reject cached ResolvedControls whose Blender RNA was replaced by Undo/Redo."""

    try:
        for _binding_id, resolved in value.resolved_bindings:
            if int(resolved.owner_object.as_pointer()) == 0:
                return False
            if int(resolved.target.as_pointer()) == 0:
                return False
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    return True


class UnsupportedCharacterSchema(RuntimeError):
    pass


class CharacterNotFound(RuntimeError):
    pass


class StaleCharacterDefinition(RuntimeError):
    pass


class CharacterMetadataError(RuntimeError):
    pass


class BAW_PG_character_owner(bpy.types.PropertyGroup):
    owner_id: StringProperty(name="Owner ID")
    object_ref: PointerProperty(name="Object", type=bpy.types.Object)
    label_hint: StringProperty(name="Object Label Hint")


class BAW_PG_character_binding_ref(bpy.types.PropertyGroup):
    binding_id: StringProperty(name="Binding ID")


class BAW_PG_character_binding(bpy.types.PropertyGroup):
    binding_id: StringProperty(name="Binding ID")
    owner_id: StringProperty(name="Owner ID")
    kind: StringProperty(name="Control Kind", default=AWBControlKind.OBJECT.value)
    bone_id: StringProperty(name="Bone ID")
    bone_name_hint: StringProperty(name="Bone Name Hint")
    semantic_key: StringProperty(name="Semantic Key")
    side: StringProperty(name="Side", default=CharacterSide.NONE.value)
    mode: StringProperty(name="Control Mode", default=ControlMode.NEUTRAL.value)
    usage: StringProperty(name="Control Usage", default=ControlUsage.PRIMARY.value)


class BAW_PG_character_group(bpy.types.PropertyGroup):
    group_id: StringProperty(name="Group ID")
    semantic_key: StringProperty(name="Semantic Key")
    label: StringProperty(name="Label")
    side: StringProperty(name="Side", default=CharacterSide.NONE.value)
    parent_group_id: StringProperty(name="Parent Group ID")
    members: CollectionProperty(type=BAW_PG_character_binding_ref)


class BAW_PG_character_chain(bpy.types.PropertyGroup):
    chain_id: StringProperty(name="Chain ID")
    semantic_key: StringProperty(name="Semantic Key")
    label: StringProperty(name="Label")
    side: StringProperty(name="Side", default=CharacterSide.NONE.value)
    mode: StringProperty(name="Control Mode", default=ControlMode.NEUTRAL.value)
    members: CollectionProperty(type=BAW_PG_character_binding_ref)


class BAW_PG_character_opposite(bpy.types.PropertyGroup):
    left_binding_id: StringProperty(name="Left Binding ID")
    right_binding_id: StringProperty(name="Right Binding ID")


class BAW_PG_character_kinematic(bpy.types.PropertyGroup):
    mapping_id: StringProperty(name="Mapping ID")
    semantic_key: StringProperty(name="Semantic Key")
    side: StringProperty(name="Side", default=CharacterSide.NONE.value)
    fk_chain_id: StringProperty(name="FK Chain ID")
    ik_target_binding_id: StringProperty(name="IK Target Binding ID")
    pole_binding_id: StringProperty(name="Pole Binding ID")
    reference_chain_id: StringProperty(name="Reference Chain ID")
    extras: CollectionProperty(type=BAW_PG_character_binding_ref)


class BAW_PG_character(bpy.types.PropertyGroup):
    character_id: StringProperty(name="Character ID")
    label: StringProperty(name="Label")
    revision: IntProperty(name="Revision", default=0, min=0)
    owners: CollectionProperty(type=BAW_PG_character_owner)
    bindings: CollectionProperty(type=BAW_PG_character_binding)
    groups: CollectionProperty(type=BAW_PG_character_group)
    chains: CollectionProperty(type=BAW_PG_character_chain)
    opposites: CollectionProperty(type=BAW_PG_character_opposite)
    kinematics: CollectionProperty(type=BAW_PG_character_kinematic)


class BAW_PG_character_store(bpy.types.PropertyGroup):
    schema_version: IntProperty(name="Schema Version", default=SCHEMA_VERSION, min=1)
    characters: CollectionProperty(type=BAW_PG_character)


_METADATA_CLASSES = (
    BAW_PG_character_owner,
    BAW_PG_character_binding_ref,
    BAW_PG_character_binding,
    BAW_PG_character_group,
    BAW_PG_character_chain,
    BAW_PG_character_opposite,
    BAW_PG_character_kinematic,
    BAW_PG_character,
    BAW_PG_character_store,
)


@dataclass(frozen=True, slots=True)
class ResolvedCharacter:
    definition: AWBCharacter
    resolved_bindings: tuple[tuple[str, object], ...]
    issues: tuple[CharacterIssue, ...]
    source_stamp: tuple


def _store(scene):
    store = getattr(scene, STORE_PROPERTY, None)
    if store is None:
        raise CharacterMetadataError("AWB Character metadata is not registered on this Scene.")
    return store


def _require_supported_store(scene):
    store = _store(scene)
    version = int(getattr(store, "schema_version", SCHEMA_VERSION) or SCHEMA_VERSION)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(item) for item in sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise UnsupportedCharacterSchema(
            f"Unsupported AWB Character schema {version}; supported versions: {supported}."
        )
    return store


def _require_writable_character_scene(scene) -> None:
    if getattr(scene, "library", None) is not None or getattr(scene, "is_editable", True) is False:
        raise CharacterMetadataError(
            "The Scene storing AWB Character metadata is linked or read-only and cannot be modified."
        )


def _require_v2_store(scene):
    _require_writable_character_scene(scene)
    store = _require_supported_store(scene)
    version = int(store.schema_version)
    if version != SCHEMA_VERSION:
        raise UnsupportedCharacterSchema(
            f"Character semantic graph requires schema {SCHEMA_VERSION}; current store is {version}."
        )
    return store


def _row_by_character_id(store, character_id: str):
    for index, row in enumerate(store.characters):
        if str(row.character_id) == character_id:
            return index, row
    return None, None


def _decode_kind(value: str) -> AWBControlKind:
    try:
        return AWBControlKind(str(value))
    except ValueError as exc:
        raise CharacterMetadataError(f"Unsupported stored control kind: {value!r}") from exc


def _decode_enum(enum_type, value: str, label: str):
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise CharacterMetadataError(f"Unsupported stored {label}: {value!r}") from exc


def _decode_character(row) -> AWBCharacter:
    owners = tuple(
        CharacterOwner(
            owner_id=str(owner.owner_id),
            label_hint=str(owner.label_hint),
        )
        for owner in row.owners
    )
    bindings = tuple(
        CharacterBinding(
            binding_id=str(binding.binding_id),
            owner_id=str(binding.owner_id),
            kind=_decode_kind(binding.kind),
            bone_id=(str(binding.bone_id) or None),
            bone_name_hint=(str(binding.bone_name_hint) or None),
            semantic_key=str(getattr(binding, "semantic_key", "")),
            side=_decode_enum(
                CharacterSide,
                getattr(binding, "side", CharacterSide.NONE.value),
                "Character side",
            ),
            mode=_decode_enum(
                ControlMode,
                getattr(binding, "mode", ControlMode.NEUTRAL.value),
                "control mode",
            ),
            usage=_decode_enum(
                ControlUsage,
                getattr(binding, "usage", ControlUsage.PRIMARY.value),
                "control usage",
            ),
        )
        for binding in row.bindings
    )
    groups = tuple(
        CharacterGroup(
            group_id=str(group.group_id),
            semantic_key=str(group.semantic_key),
            label=str(group.label),
            side=_decode_enum(CharacterSide, group.side, "Character side"),
            parent_group_id=(str(group.parent_group_id) or None),
            members=tuple(str(member.binding_id) for member in group.members),
        )
        for group in getattr(row, "groups", ())
    )
    chains = tuple(
        CharacterChain(
            chain_id=str(chain.chain_id),
            semantic_key=str(chain.semantic_key),
            label=str(chain.label),
            side=_decode_enum(CharacterSide, chain.side, "Character side"),
            mode=_decode_enum(ControlMode, chain.mode, "control mode"),
            members=tuple(str(member.binding_id) for member in chain.members),
        )
        for chain in getattr(row, "chains", ())
    )
    opposites = tuple(
        OppositePair(
            left_binding_id=str(pair.left_binding_id),
            right_binding_id=str(pair.right_binding_id),
        )
        for pair in getattr(row, "opposites", ())
    )
    kinematics = tuple(
        KinematicMapping(
            mapping_id=str(mapping.mapping_id),
            semantic_key=str(mapping.semantic_key),
            side=_decode_enum(CharacterSide, mapping.side, "Character side"),
            fk_chain_id=(str(mapping.fk_chain_id) or None),
            ik_target_binding_id=(str(mapping.ik_target_binding_id) or None),
            pole_binding_id=(str(mapping.pole_binding_id) or None),
            reference_chain_id=(str(mapping.reference_chain_id) or None),
            extras=tuple(str(member.binding_id) for member in mapping.extras),
        )
        for mapping in getattr(row, "kinematics", ())
    )
    return AWBCharacter(
        character_id=str(row.character_id),
        label=str(row.label),
        revision=int(row.revision),
        owners=owners,
        bindings=bindings,
        groups=groups,
        chains=chains,
        opposites=opposites,
        kinematics=kinematics,
    )


def read_character(scene, character_id: str) -> AWBCharacter | None:
    """Read one Character without creating or mutating metadata."""

    store = _require_supported_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        return None
    definition = _decode_character(row)
    issues = validate_character(definition)
    if has_character_errors(issues):
        codes = ", ".join(issue.code for issue in issues)
        raise CharacterMetadataError(f"Invalid Character metadata: {codes}")
    return definition


def character_ids(scene) -> tuple[str, ...]:
    """Return live Character IDs in storage order without resolving native targets.

    Generated Rigped metadata whose owner Object was deleted from the Scene is
    treated as runtime-stale immediately. Keeping the RNA row until a safe load
    cleanup preserves Blender Undo while preventing stale Rigped rows from
    entering UI/runtime resolution hot paths.
    """

    store = _require_supported_store(scene)
    return tuple(
        str(row.character_id)
        for row in store.characters
        if not _row_is_stale_generated_rigped(scene, row)
    )


def character_lifecycle_issues(scene) -> tuple[CharacterIssue, ...]:
    """Diagnose Scene/owner persistence and editability without mutating Character metadata."""

    store = _store(scene)
    version = int(getattr(store, "schema_version", SCHEMA_VERSION) or SCHEMA_VERSION)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        return (
            CharacterIssue(
                code="UNSUPPORTED_CHARACTER_SCHEMA",
                severity=CharacterIssueSeverity.ERROR,
                detail=f"Unsupported AWB Character schema {version}.",
            ),
        )

    issues: list[CharacterIssue] = []
    if getattr(scene, "library", None) is not None or getattr(scene, "is_editable", True) is False:
        issues.append(
            CharacterIssue(
                code="READ_ONLY_CHARACTER_SCENE",
                severity=CharacterIssueSeverity.WARNING,
                detail="The Scene storing AWB Character metadata is linked or read-only.",
            )
        )
    if version == SCHEMA_VERSION_V1:
        issues.append(
            CharacterIssue(
                code="LEGACY_CHARACTER_SCHEMA",
                severity=CharacterIssueSeverity.WARNING,
                detail="AWB Character metadata uses schema v1 and requires explicit migration for v2 authoring.",
            )
        )

    for row in store.characters:
        if _row_is_stale_generated_rigped(scene, row):
            continue
        character_id = str(row.character_id)
        try:
            definition = _decode_character(row)
        except CharacterMetadataError as exc:
            issues.append(
                CharacterIssue(
                    code="CHARACTER_DECODE_ERROR",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=character_id or None,
                    detail=str(exc),
                )
            )
            continue

        graph_issues = validate_character(definition)
        issues.extend(graph_issues)
        bone_owner_ids = {
            binding.owner_id
            for binding in definition.bindings
            if binding.kind == AWBControlKind.BONE
        }
        for owner in row.owners:
            owner_id = str(owner.owner_id)
            obj = owner.object_ref
            if obj is None:
                continue
            if getattr(obj, "is_missing", False):
                issues.append(
                    CharacterIssue(
                        code="MISSING_LIBRARY_OWNER",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=character_id,
                        record_id=owner_id,
                        detail=f"Character owner {obj.name!r} is missing from its source library.",
                    )
                )
            elif getattr(obj, "library", None) is not None:
                issues.append(
                    CharacterIssue(
                        code="LINKED_OWNER_READ_ONLY",
                        severity=CharacterIssueSeverity.WARNING,
                        character_id=character_id,
                        record_id=owner_id,
                        detail=f"Character owner {obj.name!r} is linked and read-only.",
                    )
                )
            else:
                override = getattr(obj, "override_library", None)
                if (
                    override is not None
                    and (
                        getattr(override, "is_system_override", False)
                        or getattr(obj, "is_editable", True) is False
                    )
                ):
                    issues.append(
                        CharacterIssue(
                            code="NON_EDITABLE_OVERRIDE_OWNER",
                            severity=CharacterIssueSeverity.WARNING,
                            character_id=character_id,
                            record_id=owner_id,
                            detail=f"Character owner override {obj.name!r} is not user-editable.",
                        )
                    )
                elif getattr(obj, "is_editable", True) is False:
                    issues.append(
                        CharacterIssue(
                            code="READ_ONLY_OWNER",
                            severity=CharacterIssueSeverity.WARNING,
                            character_id=character_id,
                            record_id=owner_id,
                            detail=f"Character owner {obj.name!r} is read-only.",
                        )
                    )

            armature = getattr(obj, "data", None) if getattr(obj, "type", None) == "ARMATURE" else None
            if armature is None:
                continue
            if getattr(armature, "is_missing", False):
                issues.append(
                    CharacterIssue(
                        code="MISSING_LIBRARY_ARMATURE_DATA",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=character_id,
                        record_id=owner_id,
                        detail=f"Armature data for {obj.name!r} is missing from its source library.",
                    )
                )
            elif getattr(armature, "library", None) is not None:
                issues.append(
                    CharacterIssue(
                        code="LINKED_ARMATURE_DATA_READ_ONLY",
                        severity=CharacterIssueSeverity.WARNING,
                        character_id=character_id,
                        record_id=owner_id,
                        detail=f"Armature data for {obj.name!r} is linked and cannot store local AWB Bone tokens.",
                    )
                )
            else:
                armature_override = getattr(armature, "override_library", None)
                if (
                    armature_override is not None
                    and (
                        getattr(armature_override, "is_system_override", False)
                        or getattr(armature, "is_editable", True) is False
                    )
                ):
                    issues.append(
                        CharacterIssue(
                            code="NON_EDITABLE_OVERRIDE_ARMATURE_DATA",
                            severity=CharacterIssueSeverity.WARNING,
                            character_id=character_id,
                            record_id=owner_id,
                            detail=f"Armature-data override for {obj.name!r} is not user-editable.",
                        )
                    )
            users = int(getattr(armature, "users", 1) or 1)
            if owner_id in bone_owner_ids and users > 1:
                issues.append(
                    CharacterIssue(
                        code="SHARED_ARMATURE_DATA_TOKEN_WRITE_BLOCKED",
                        severity=CharacterIssueSeverity.WARNING,
                        character_id=character_id,
                        record_id=owner_id,
                        detail=(
                            f"Armature data for {obj.name!r} is shared by {users} Objects; "
                            "AWB Bone-token writes require single-user Armature data."
                        ),
                    )
                )

        if not has_character_errors(graph_issues):
            try:
                issues.extend(resolve_character(scene, character_id).issues)
            except CharacterMetadataError as exc:
                issues.append(
                    CharacterIssue(
                        code="CHARACTER_RESOLUTION_ERROR",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=character_id,
                        detail=str(exc),
                    )
                )

    unique: list[CharacterIssue] = []
    seen: set[tuple] = set()
    for issue in issues:
        key = (issue.code, issue.character_id, issue.record_id, issue.detail)
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return tuple(unique)


def character_stamp(scene, character_id: str) -> tuple:
    """Return an opaque operation-local stamp for stale edit detection."""

    store = _require_supported_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)

    owner_signature = tuple(
        (
            str(owner.owner_id),
            int(owner.object_ref.as_pointer()) if owner.object_ref is not None else 0,
            str(owner.label_hint),
        )
        for owner in row.owners
    )
    binding_signature = tuple(
        (
            str(binding.binding_id),
            str(binding.owner_id),
            str(binding.kind),
            str(binding.bone_id),
            str(binding.bone_name_hint),
            str(getattr(binding, "semantic_key", "")),
            str(getattr(binding, "side", CharacterSide.NONE.value)),
            str(getattr(binding, "mode", ControlMode.NEUTRAL.value)),
            str(getattr(binding, "usage", ControlUsage.PRIMARY.value)),
        )
        for binding in row.bindings
    )
    group_signature = tuple(
        (
            str(group.group_id),
            str(group.semantic_key),
            str(group.label),
            str(group.side),
            str(group.parent_group_id),
            tuple(str(member.binding_id) for member in group.members),
        )
        for group in getattr(row, "groups", ())
    )
    chain_signature = tuple(
        (
            str(chain.chain_id),
            str(chain.semantic_key),
            str(chain.label),
            str(chain.side),
            str(chain.mode),
            tuple(str(member.binding_id) for member in chain.members),
        )
        for chain in getattr(row, "chains", ())
    )
    opposite_signature = tuple(
        (str(pair.left_binding_id), str(pair.right_binding_id))
        for pair in getattr(row, "opposites", ())
    )
    kinematic_signature = tuple(
        (
            str(mapping.mapping_id),
            str(mapping.semantic_key),
            str(mapping.side),
            str(mapping.fk_chain_id),
            str(mapping.ik_target_binding_id),
            str(mapping.pole_binding_id),
            str(mapping.reference_chain_id),
            tuple(str(member.binding_id) for member in mapping.extras),
        )
        for mapping in getattr(row, "kinematics", ())
    )
    return (
        int(store.schema_version),
        str(row.character_id),
        int(row.revision),
        owner_signature,
        binding_signature,
        group_signature,
        chain_signature,
        opposite_signature,
        kinematic_signature,
    )


def _object_pointer_key(obj) -> int:
    pointer = getattr(obj, "as_pointer", None)
    if callable(pointer):
        value = int(pointer())
        if value:
            return value
    return id(obj)


def _scene_contains_object_identity(scene, obj) -> bool:
    """Return whether the exact Object identity is still a member of the Scene."""

    if not hasattr(scene, "objects"):
        return True
    objects = scene.objects
    candidates = objects.values() if hasattr(objects, "values") else objects
    pointer = _object_pointer_key(obj)
    return any(
        candidate is obj or _object_pointer_key(candidate) == pointer
        for candidate in candidates
    )


def _resolved_character_native_signature(scene, row) -> tuple:
    """Capture native structure that can invalidate cached binding resolution."""

    objects = getattr(scene, "objects", ())
    candidates = objects.values() if hasattr(objects, "values") else objects
    scene_objects = tuple(sorted(_object_pointer_key(obj) for obj in candidates))
    owners = []
    for owner in row.owners:
        obj = owner.object_ref
        if obj is None:
            owners.append((str(owner.owner_id), 0, 0, 0, 0, ()))
            continue
        data = getattr(obj, "data", None)
        bones = getattr(data, "bones", ()) if data is not None else ()
        bone_signature = tuple(
            (
                str(getattr(bone, "name", "")),
                _bone_token(bone) or "",
            )
            for bone in bones
        )
        owners.append(
            (
                str(owner.owner_id),
                _object_pointer_key(obj),
                int(getattr(obj, "session_uid", 0) or 0),
                0 if data is None else _object_pointer_key(data),
                0 if data is None else int(getattr(data, "session_uid", 0) or 0),
                bone_signature,
            )
        )
    return (
        _object_pointer_key(scene),
        int(getattr(scene, "session_uid", 0) or 0),
        scene_objects,
        tuple(owners),
    )


_GENERATED_RIGPED_CHARACTER_LABEL = "AWB Rigped"
_GENERATED_RIGPED_OWNER_PREFIX = "AWB_Rigped"


def _row_is_generated_rigped(row) -> bool:
    if str(getattr(row, "label", "")) == _GENERATED_RIGPED_CHARACTER_LABEL:
        return True
    return any(
        str(getattr(owner, "label_hint", "")).startswith(_GENERATED_RIGPED_OWNER_PREFIX)
        for owner in getattr(row, "owners", ())
    )


def _row_is_stale_generated_rigped(scene, row) -> bool:
    """Return True only for generated Rigped rows whose owner left this Scene."""

    if not _row_is_generated_rigped(row):
        return False
    for owner in getattr(row, "owners", ()):
        obj = getattr(owner, "object_ref", None)
        if obj is not None and _scene_contains_object_identity(scene, obj):
            return False
    return True


def prune_stale_generated_rigped_metadata(scene) -> tuple[str, ...]:
    """Physically remove stale generated Rigped rows at a safe lifecycle point.

    Runtime queries already hide stale rows immediately so ordinary Object
    deletion remains Undo-safe. This cleanup is intended for load/startup, when
    there is no useful deletion Undo transaction to preserve.
    """

    _require_writable_character_scene(scene)
    store = _require_supported_store(scene)
    stale = [
        (index, str(row.character_id))
        for index, row in enumerate(store.characters)
        if _row_is_stale_generated_rigped(scene, row)
    ]
    for index, _character_id in reversed(stale):
        store.characters.remove(index)
    return tuple(character_id for _index, character_id in stale)


def _preflight_object_targets(scene, object_targets) -> tuple:
    _require_writable_character_scene(scene)
    targets = []
    seen: set[int] = set()
    for obj in object_targets:
        if obj is None or not isinstance(obj, bpy.types.Object):
            raise CharacterMetadataError("Character Object targets must be Blender Objects.")
        if getattr(obj, "library", None) is not None or getattr(obj, "is_editable", True) is False:
            raise CharacterMetadataError(
                f"Object {obj.name!r} is linked or read-only and cannot enter writable Character metadata scope."
            )
        if not _scene_contains_object_identity(scene, obj):
            raise CharacterMetadataError(
                f"Object {obj.name!r} does not belong to the target Scene."
            )
        pointer = _object_pointer_key(obj)
        if pointer in seen:
            continue
        seen.add(pointer)
        targets.append(obj)

    if not targets:
        raise CharacterMetadataError("At least one selected local Object is required.")

    store = _require_supported_store(scene)
    existing: dict[int, str] = {}
    for character in store.characters:
        for owner in character.owners:
            obj = owner.object_ref
            if obj is None:
                continue
            existing[_object_pointer_key(obj)] = str(character.character_id)

    for obj in targets:
        owner_character = existing.get(_object_pointer_key(obj))
        if owner_character is not None:
            raise CharacterMetadataError(
                f"Object {obj.name!r} already belongs to Character {owner_character}."
            )

    return tuple(targets)


def create_character(scene, label: str, object_targets) -> str:
    """Create one Character from explicit local Object targets in the current supported store."""

    targets = _preflight_object_targets(scene, object_targets)
    store = _require_supported_store(scene)

    character_id = str(uuid4())
    while any(str(row.character_id) == character_id for row in store.characters):
        character_id = str(uuid4())

    owner_defs = tuple(
        CharacterOwner(owner_id=str(uuid4()), label_hint=str(obj.name)) for obj in targets
    )
    binding_defs = tuple(
        CharacterBinding(
            binding_id=str(uuid4()),
            owner_id=owner.owner_id,
            kind=AWBControlKind.OBJECT,
        )
        for owner in owner_defs
    )
    candidate = AWBCharacter(
        character_id=character_id,
        label=label.strip() or "Character",
        revision=1,
        owners=owner_defs,
        bindings=binding_defs,
    )
    issues = validate_character(candidate)
    if has_character_errors(issues):
        raise CharacterMetadataError(
            "Character preflight validation failed: "
            + ", ".join(issue.code for issue in issues)
        )

    # Mutate only after the complete candidate and native-target preflight pass.
    row = store.characters.add()
    row.character_id = candidate.character_id
    row.label = candidate.label
    row.revision = candidate.revision
    for owner_def, obj in zip(candidate.owners, targets, strict=True):
        owner = row.owners.add()
        owner.owner_id = owner_def.owner_id
        owner.object_ref = obj
        owner.label_hint = owner_def.label_hint
    for binding_def in candidate.bindings:
        binding = row.bindings.add()
        binding.binding_id = binding_def.binding_id
        binding.owner_id = binding_def.owner_id
        binding.kind = binding_def.kind.value
        binding.bone_id = ""
        binding.bone_name_hint = ""
        binding.semantic_key = ""
        binding.side = CharacterSide.NONE.value
        binding.mode = ControlMode.NEUTRAL.value
        binding.usage = ControlUsage.PRIMARY.value

    return character_id


def clone_character(
    scene,
    source_character_id: str,
    owner_map,
    *,
    label: str | None = None,
    expected_stamp: tuple | None = None,
) -> str:
    """Clone one schema-v2 Character semantic graph through an explicit source-owner -> target-Object map."""

    store = _require_v2_store(scene)
    source = read_character(scene, source_character_id)
    if source is None:
        raise CharacterNotFound(source_character_id)
    if expected_stamp is not None and character_stamp(scene, source_character_id) != expected_stamp:
        raise StaleCharacterDefinition(source_character_id)

    try:
        mapping = {str(owner_id): obj for owner_id, obj in dict(owner_map).items()}
    except Exception as exc:
        raise CharacterMetadataError("Clone owner_map must be a mapping of source owner IDs to target Objects.") from exc
    source_owner_ids = {owner.owner_id for owner in source.owners}
    if set(mapping) != source_owner_ids:
        missing = sorted(source_owner_ids - set(mapping))
        extra = sorted(set(mapping) - source_owner_ids)
        raise CharacterMetadataError(
            f"Clone owner_map must cover every source owner exactly; missing={missing}, extra={extra}."
        )

    ordered_targets = tuple(mapping[owner.owner_id] for owner in source.owners)
    targets = _preflight_object_targets(scene, ordered_targets)
    target_by_source_owner = {
        owner.owner_id: target
        for owner, target in zip(source.owners, targets, strict=True)
    }

    for binding in source.bindings:
        if binding.kind != AWBControlKind.BONE:
            continue
        target_owner = target_by_source_owner[binding.owner_id]
        _require_local_writable_armature(target_owner)
        token = binding.bone_id or ""
        matches = bone_token_matches(target_owner, token)
        if len(matches) != 1:
            raise CharacterMetadataError(
                f"Clone target Armature {target_owner.name!r} must resolve Bone token {token!r} exactly once."
            )

    existing_character_ids = {str(row.character_id) for row in store.characters}
    character_id = str(uuid4())
    while character_id in existing_character_ids:
        character_id = str(uuid4())

    owner_id_map = {owner.owner_id: str(uuid4()) for owner in source.owners}
    binding_id_map = {binding.binding_id: str(uuid4()) for binding in source.bindings}
    group_id_map = {group.group_id: str(uuid4()) for group in source.groups}
    chain_id_map = {chain.chain_id: str(uuid4()) for chain in source.chains}

    owners = tuple(
        CharacterOwner(
            owner_id=owner_id_map[owner.owner_id],
            label_hint=str(target_by_source_owner[owner.owner_id].name),
        )
        for owner in source.owners
    )
    bindings = tuple(
        CharacterBinding(
            binding_id=binding_id_map[binding.binding_id],
            owner_id=owner_id_map[binding.owner_id],
            kind=binding.kind,
            bone_id=(binding.bone_id if binding.kind == AWBControlKind.BONE else None),
            bone_name_hint=(
                binding.bone_name_hint if binding.kind == AWBControlKind.BONE else None
            ),
            semantic_key=binding.semantic_key,
            side=binding.side,
            mode=binding.mode,
            usage=binding.usage,
        )
        for binding in source.bindings
    )
    groups = tuple(
        CharacterGroup(
            group_id=group_id_map[group.group_id],
            semantic_key=group.semantic_key,
            label=group.label,
            side=group.side,
            parent_group_id=(
                group_id_map[group.parent_group_id]
                if group.parent_group_id is not None
                else None
            ),
            members=tuple(binding_id_map[item] for item in group.members),
        )
        for group in source.groups
    )
    chains = tuple(
        CharacterChain(
            chain_id=chain_id_map[chain.chain_id],
            semantic_key=chain.semantic_key,
            label=chain.label,
            side=chain.side,
            mode=chain.mode,
            members=tuple(binding_id_map[item] for item in chain.members),
        )
        for chain in source.chains
    )
    opposites = tuple(
        OppositePair(
            binding_id_map[pair.left_binding_id],
            binding_id_map[pair.right_binding_id],
        )
        for pair in source.opposites
    )
    kinematics = tuple(
        KinematicMapping(
            mapping_id=str(uuid4()),
            semantic_key=mapping_def.semantic_key,
            side=mapping_def.side,
            fk_chain_id=(
                chain_id_map[mapping_def.fk_chain_id]
                if mapping_def.fk_chain_id is not None
                else None
            ),
            ik_target_binding_id=(
                binding_id_map[mapping_def.ik_target_binding_id]
                if mapping_def.ik_target_binding_id is not None
                else None
            ),
            pole_binding_id=(
                binding_id_map[mapping_def.pole_binding_id]
                if mapping_def.pole_binding_id is not None
                else None
            ),
            reference_chain_id=(
                chain_id_map[mapping_def.reference_chain_id]
                if mapping_def.reference_chain_id is not None
                else None
            ),
            extras=tuple(binding_id_map[item] for item in mapping_def.extras),
        )
        for mapping_def in source.kinematics
    )
    candidate = AWBCharacter(
        character_id=character_id,
        label=(label.strip() if label is not None else f"{source.label} Copy") or "Character Copy",
        revision=1,
        owners=owners,
        bindings=bindings,
        groups=groups,
        chains=chains,
        opposites=opposites,
        kinematics=kinematics,
    )
    _raise_if_invalid(candidate)

    row = store.characters.add()
    row.character_id = candidate.character_id
    row.label = candidate.label
    row.revision = candidate.revision
    target_by_new_owner_id = {
        owner_id_map[source_owner.owner_id]: target_by_source_owner[source_owner.owner_id]
        for source_owner in source.owners
    }
    for owner in candidate.owners:
        target = row.owners.add()
        target.owner_id = owner.owner_id
        target.object_ref = target_by_new_owner_id[owner.owner_id]
        target.label_hint = owner.label_hint
    for binding in candidate.bindings:
        target = row.bindings.add()
        target.binding_id = binding.binding_id
        target.owner_id = binding.owner_id
        target.kind = binding.kind.value
        target.bone_id = binding.bone_id or ""
        target.bone_name_hint = binding.bone_name_hint or ""
        target.semantic_key = binding.semantic_key
        target.side = binding.side.value
        target.mode = binding.mode.value
        target.usage = binding.usage.value
    for group in candidate.groups:
        target = row.groups.add()
        target.group_id = group.group_id
        target.semantic_key = group.semantic_key
        target.label = group.label
        target.side = group.side.value
        target.parent_group_id = group.parent_group_id or ""
        for binding_id in group.members:
            member = target.members.add()
            member.binding_id = binding_id
    for chain in candidate.chains:
        target = row.chains.add()
        target.chain_id = chain.chain_id
        target.semantic_key = chain.semantic_key
        target.label = chain.label
        target.side = chain.side.value
        target.mode = chain.mode.value
        for binding_id in chain.members:
            member = target.members.add()
            member.binding_id = binding_id
    for pair in candidate.opposites:
        target = row.opposites.add()
        target.left_binding_id = pair.left_binding_id
        target.right_binding_id = pair.right_binding_id
    for mapping_def in candidate.kinematics:
        target = row.kinematics.add()
        target.mapping_id = mapping_def.mapping_id
        target.semantic_key = mapping_def.semantic_key
        target.side = mapping_def.side.value
        target.fk_chain_id = mapping_def.fk_chain_id or ""
        target.ik_target_binding_id = mapping_def.ik_target_binding_id or ""
        target.pole_binding_id = mapping_def.pole_binding_id or ""
        target.reference_chain_id = mapping_def.reference_chain_id or ""
        for binding_id in mapping_def.extras:
            member = target.extras.add()
            member.binding_id = binding_id

    return character_id


def bind_object(
    scene,
    character_id: str,
    obj,
    *,
    expected_stamp: tuple | None = None,
) -> str:
    """Explicitly bind one local Object to an existing schema-v2 Character."""

    if obj is None or not isinstance(obj, bpy.types.Object):
        raise CharacterMetadataError("Character Object target must be a Blender Object.")
    if getattr(obj, "library", None) is not None or getattr(obj, "is_editable", True) is False:
        raise CharacterMetadataError(
            f"Object {obj.name!r} is linked or read-only and cannot enter writable Character metadata scope."
        )
    if not _scene_contains_object_identity(scene, obj):
        raise CharacterMetadataError(f"Object {obj.name!r} does not belong to the target Scene.")

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)

    pointer = _object_pointer_key(obj)
    owner_row = None
    for character in store.characters:
        for owner in character.owners:
            owner_obj = owner.object_ref
            if owner_obj is None or _object_pointer_key(owner_obj) != pointer:
                continue
            if str(character.character_id) != character_id:
                raise CharacterMetadataError(
                    f"Object {obj.name!r} already belongs to Character {character.character_id}."
                )
            owner_row = owner

    if owner_row is not None:
        for binding in row.bindings:
            if (
                str(binding.owner_id) == str(owner_row.owner_id)
                and str(binding.kind) == AWBControlKind.OBJECT.value
            ):
                raise CharacterMetadataError(
                    f"Object {obj.name!r} is already bound to Character {character_id}."
                )

    definition = _decode_character(row)
    owner_id = str(owner_row.owner_id) if owner_row is not None else str(uuid4())
    owner_def = CharacterOwner(owner_id=owner_id, label_hint=str(obj.name))
    binding_def = CharacterBinding(
        binding_id=str(uuid4()),
        owner_id=owner_id,
        kind=AWBControlKind.OBJECT,
    )
    owners = definition.owners if owner_row is not None else definition.owners + (owner_def,)
    candidate = replace(
        definition,
        revision=definition.revision + 1,
        owners=owners,
        bindings=definition.bindings + (binding_def,),
    )
    _raise_if_invalid(candidate)

    if owner_row is None:
        owner_row = row.owners.add()
        owner_row.owner_id = owner_def.owner_id
        owner_row.object_ref = obj
        owner_row.label_hint = owner_def.label_hint

    binding = row.bindings.add()
    binding.binding_id = binding_def.binding_id
    binding.owner_id = binding_def.owner_id
    binding.kind = binding_def.kind.value
    binding.bone_id = ""
    binding.bone_name_hint = ""
    binding.semantic_key = ""
    binding.side = CharacterSide.NONE.value
    binding.mode = ControlMode.NEUTRAL.value
    binding.usage = ControlUsage.PRIMARY.value
    row.revision = int(row.revision) + 1
    return binding_def.binding_id


def _pose_bone_owner(pose_bone):
    owner = getattr(pose_bone, "id_data", None)
    if owner is None or not isinstance(owner, bpy.types.Object) or getattr(owner, "type", None) != "ARMATURE":
        raise CharacterMetadataError("PoseBone target must belong to an Armature Object.")
    return owner


def _require_local_writable_armature(owner) -> None:
    if getattr(owner, "library", None) is not None or getattr(owner, "is_editable", True) is False:
        raise CharacterMetadataError(
            f"Armature Object {owner.name!r} is linked or read-only and cannot enter writable Character metadata scope."
        )
    armature = getattr(owner, "data", None)
    if (
        armature is None
        or getattr(armature, "library", None) is not None
        or getattr(armature, "is_editable", True) is False
    ):
        raise CharacterMetadataError(
            f"Armature data for {owner.name!r} is linked or read-only and cannot store AWB Bone tokens."
        )


def _require_single_user_armature_token_write(owner) -> None:
    armature = getattr(owner, "data", None)
    users = int(getattr(armature, "users", 1) or 1) if armature is not None else 1
    if users > 1:
        raise CharacterMetadataError(
            f"Armature data for {owner.name!r} is shared by {users} Objects; "
            "make the Armature data single-user before changing AWB Bone tokens."
        )


def _bone_data_for_pose_bone(pose_bone):
    bone = getattr(pose_bone, "bone", None)
    if bone is None:
        raise CharacterMetadataError("PoseBone target has no Bone data locator.")
    return bone


def _bone_token(bone) -> str | None:
    value = bone.get(BONE_TOKEN_PROPERTY) if hasattr(bone, "get") else None
    value = str(value).strip() if value is not None else ""
    return value or None


def bone_token_matches(owner, bone_id: str) -> tuple:
    """Return Bone-data matches for one owner Armature without mutating anything."""

    token = str(bone_id).strip()
    if not token:
        return ()
    armature = getattr(owner, "data", None)
    bones = getattr(armature, "bones", ()) if armature is not None else ()
    return tuple(bone for bone in bones if _bone_token(bone) == token)


def issue_bone_token(pose_bone, *, force_new: bool = False) -> str:
    """Issue a local Bone token; duplicate repair may explicitly force a new token."""

    owner = _pose_bone_owner(pose_bone)
    _require_local_writable_armature(owner)
    bone = _bone_data_for_pose_bone(pose_bone)
    existing = _bone_token(bone)
    if existing and not force_new:
        return existing

    _require_single_user_armature_token_write(owner)
    token = str(uuid4())
    existing_tokens = {
        value
        for candidate in getattr(owner.data, "bones", ())
        if (value := _bone_token(candidate)) is not None and candidate is not bone
    }
    while token in existing_tokens:
        token = str(uuid4())
    bone[BONE_TOKEN_PROPERTY] = token
    return token


def _bone_token_reference_conflicts(
    owner,
    bone_id: str,
    *,
    excluded_scene,
    excluded_character_id: str,
    excluded_binding_id: str,
) -> tuple[tuple[object, object, object], ...]:
    """Find other Character bindings that rely on one native owner/token identity.

    Reissuing a duplicate Bone token changes which native Bone an unchanged
    token resolves to. The repair therefore must fail closed whenever another
    binding, including one in another Scene-local Character store, still relies
    on that owner/token pair.
    """

    token = str(bone_id).strip()
    if not token:
        return ()

    scenes = list(getattr(getattr(bpy, "data", None), "scenes", ()) or ())
    excluded_scene_pointer = _object_pointer_key(excluded_scene)
    if not any(_object_pointer_key(scene) == excluded_scene_pointer for scene in scenes):
        scenes.append(excluded_scene)

    owner_pointer = _object_pointer_key(owner)
    conflicts = []
    for candidate_scene in scenes:
        store = getattr(candidate_scene, STORE_PROPERTY, None)
        if store is None:
            continue
        candidate_scene_pointer = _object_pointer_key(candidate_scene)
        for character_row in getattr(store, "characters", ()):
            owners_by_id = {
                str(owner_row.owner_id): owner_row.object_ref
                for owner_row in getattr(character_row, "owners", ())
                if owner_row.object_ref is not None
            }
            for binding_row in getattr(character_row, "bindings", ()):
                if str(binding_row.kind) != AWBControlKind.BONE.value:
                    continue
                if str(binding_row.bone_id).strip() != token:
                    continue
                binding_owner = owners_by_id.get(str(binding_row.owner_id))
                if binding_owner is None or _object_pointer_key(binding_owner) != owner_pointer:
                    continue
                is_excluded = (
                    candidate_scene_pointer == excluded_scene_pointer
                    and str(character_row.character_id) == str(excluded_character_id)
                    and str(binding_row.binding_id) == str(excluded_binding_id)
                )
                if not is_excluded:
                    conflicts.append((candidate_scene, character_row, binding_row))
    return tuple(conflicts)


def _owner_row_for_object(character_row, obj):
    pointer = _object_pointer_key(obj)
    for owner in character_row.owners:
        if owner.object_ref is not None and _object_pointer_key(owner.object_ref) == pointer:
            return owner
    return None


def _binding_row_by_id(character_row, binding_id: str):
    for index, binding in enumerate(character_row.bindings):
        if str(binding.binding_id) == binding_id:
            return index, binding
    return None, None


def bind_pose_bone(
    scene,
    character_id: str,
    pose_bone,
    *,
    expected_stamp: tuple | None = None,
) -> str:
    """Bind one explicit local PoseBone to an existing Character owner Object."""

    _require_writable_character_scene(scene)
    store = _require_supported_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)

    owner = _pose_bone_owner(pose_bone)
    _require_local_writable_armature(owner)
    owner_row = _owner_row_for_object(row, owner)
    if owner_row is None:
        raise CharacterMetadataError(
            f"Armature Object {owner.name!r} must already be a Character owner before binding PoseBones."
        )

    bone = _bone_data_for_pose_bone(pose_bone)
    token = _bone_token(bone)
    if token:
        matches = bone_token_matches(owner, token)
        if len(matches) > 1:
            raise CharacterMetadataError(
                f"Bone token {token!r} is ambiguous on {owner.name!r}; repair the duplicate token first."
            )
        for binding in row.bindings:
            if (
                str(binding.owner_id) == str(owner_row.owner_id)
                and str(binding.kind) == AWBControlKind.BONE.value
                and str(binding.bone_id) == token
            ):
                raise CharacterMetadataError(
                    f"PoseBone {pose_bone.name!r} is already bound to Character {character_id}."
                )
    else:
        token = issue_bone_token(pose_bone)

    binding_id = str(uuid4())
    existing_ids = {str(binding.binding_id) for binding in row.bindings}
    while binding_id in existing_ids:
        binding_id = str(uuid4())

    binding = row.bindings.add()
    binding.binding_id = binding_id
    binding.owner_id = str(owner_row.owner_id)
    binding.kind = AWBControlKind.BONE.value
    binding.bone_id = token
    binding.bone_name_hint = str(pose_bone.name)
    binding.semantic_key = ""
    binding.side = CharacterSide.NONE.value
    binding.mode = ControlMode.NEUTRAL.value
    binding.usage = ControlUsage.PRIMARY.value
    row.revision = int(row.revision) + 1
    return binding_id


def rebind_pose_bone(
    scene,
    character_id: str,
    binding_id: str,
    pose_bone,
    *,
    expected_stamp: tuple | None = None,
    repair_duplicate_token: bool = False,
) -> str:
    """Retarget one Bone binding; duplicate repair can reissue only the chosen Bone token."""

    _require_writable_character_scene(scene)
    store = _require_supported_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    _binding_index, binding = _binding_row_by_id(row, binding_id)
    if binding is None:
        raise CharacterMetadataError(f"Character binding {binding_id!r} was not found.")
    if str(binding.kind) != AWBControlKind.BONE.value:
        raise CharacterMetadataError("Only Bone bindings can be rebound to a PoseBone.")

    owner = _pose_bone_owner(pose_bone)
    _require_local_writable_armature(owner)
    owner_row = _owner_row_for_object(row, owner)
    if owner_row is None:
        raise CharacterMetadataError(
            f"Armature Object {owner.name!r} must already be a Character owner before rebinding PoseBones."
        )

    bone = _bone_data_for_pose_bone(pose_bone)
    token = _bone_token(bone)
    matches = bone_token_matches(owner, token) if token else ()
    if repair_duplicate_token and token and len(matches) > 1:
        conflicts = _bone_token_reference_conflicts(
            owner,
            token,
            excluded_scene=scene,
            excluded_character_id=character_id,
            excluded_binding_id=binding_id,
        )
        if conflicts:
            raise CharacterMetadataError(
                "Duplicate Bone token repair would retarget another Character binding; "
                "resolve the other token references before repairing this binding."
            )
        token = issue_bone_token(pose_bone, force_new=True)
        matches = bone_token_matches(owner, token)
    elif not token:
        token = issue_bone_token(pose_bone)
        matches = bone_token_matches(owner, token)
    if len(matches) != 1:
        raise CharacterMetadataError(
            f"PoseBone {pose_bone.name!r} does not resolve to one unique Bone token."
        )

    for other in row.bindings:
        if other is binding:
            continue
        if (
            str(other.owner_id) == str(owner_row.owner_id)
            and str(other.kind) == AWBControlKind.BONE.value
            and str(other.bone_id) == token
        ):
            raise CharacterMetadataError(
                f"PoseBone {pose_bone.name!r} is already bound by another Character binding."
            )

    binding.owner_id = str(owner_row.owner_id)
    binding.bone_id = token
    binding.bone_name_hint = str(pose_bone.name)
    row.revision = int(row.revision) + 1
    return token


def remove_character_binding(
    scene,
    character_id: str,
    binding_id: str,
    *,
    expected_stamp: tuple | None = None,
) -> None:
    """Remove one binding after graph preflight; drop its owner only when no bindings still use it."""

    _require_writable_character_scene(scene)
    store = _require_supported_store(scene)
    _character_index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    binding_index, binding_row = _binding_row_by_id(row, binding_id)
    if binding_index is None or binding_row is None:
        raise CharacterMetadataError(f"Character binding {binding_id!r} was not found.")

    definition = _decode_character(row)
    bindings = tuple(binding for binding in definition.bindings if binding.binding_id != binding_id)
    owner_id = str(binding_row.owner_id)
    owner_still_used = any(binding.owner_id == owner_id for binding in bindings)
    owners = (
        definition.owners
        if owner_still_used
        else tuple(owner for owner in definition.owners if owner.owner_id != owner_id)
    )
    candidate = replace(
        definition,
        revision=definition.revision + 1,
        owners=owners,
        bindings=bindings,
    )
    _raise_if_invalid(candidate)

    row.bindings.remove(binding_index)
    if not owner_still_used:
        for owner_index, owner in enumerate(row.owners):
            if str(owner.owner_id) == owner_id:
                row.owners.remove(owner_index)
                break
    row.revision = int(row.revision) + 1


def _raise_if_invalid(definition: AWBCharacter) -> None:
    issues = validate_character(definition)
    if has_character_errors(issues):
        codes = ", ".join(issue.code for issue in issues)
        raise CharacterMetadataError(f"Invalid Character semantic graph: {codes}")


def _coerce_enum(enum_type, value, label: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise CharacterMetadataError(f"Unsupported {label}: {value!r}") from exc


def migrate_store_v1_to_v2(scene) -> tuple[str, ...]:
    """Explicitly migrate one Scene store from locator-only v1 to semantic-graph v2."""

    _require_writable_character_scene(scene)
    store = _store(scene)
    version = int(getattr(store, "schema_version", SCHEMA_VERSION_V1) or SCHEMA_VERSION_V1)
    if version == SCHEMA_VERSION:
        return ()
    if version != SCHEMA_VERSION_V1:
        raise UnsupportedCharacterSchema(
            f"Cannot migrate AWB Character schema {version}; expected {SCHEMA_VERSION_V1}."
        )

    migrated_ids: list[str] = []
    for row in store.characters:
        if (
            getattr(row, "groups", ())
            or getattr(row, "chains", ())
            or getattr(row, "opposites", ())
            or getattr(row, "kinematics", ())
        ):
            raise CharacterMetadataError(
                "Schema-v1 Character contains semantic graph rows; refusing destructive migration."
            )
        for binding in row.bindings:
            if str(getattr(binding, "semantic_key", "")):
                raise CharacterMetadataError(
                    "Schema-v1 binding contains semantic_key data; refusing implicit reinterpretation."
                )
            if str(getattr(binding, "side", CharacterSide.NONE.value)) != CharacterSide.NONE.value:
                raise CharacterMetadataError(
                    "Schema-v1 binding contains side data; refusing implicit reinterpretation."
                )
            if str(getattr(binding, "mode", ControlMode.NEUTRAL.value)) != ControlMode.NEUTRAL.value:
                raise CharacterMetadataError(
                    "Schema-v1 binding contains mode data; refusing implicit reinterpretation."
                )
            if str(getattr(binding, "usage", ControlUsage.PRIMARY.value)) != ControlUsage.PRIMARY.value:
                raise CharacterMetadataError(
                    "Schema-v1 binding contains usage data; refusing implicit reinterpretation."
                )
        _raise_if_invalid(_decode_character(row))
        migrated_ids.append(str(row.character_id))

    for row in store.characters:
        for binding in row.bindings:
            binding.semantic_key = ""
            binding.side = CharacterSide.NONE.value
            binding.mode = ControlMode.NEUTRAL.value
            binding.usage = ControlUsage.PRIMARY.value
        row.revision = int(row.revision) + 1
    store.schema_version = SCHEMA_VERSION
    return tuple(migrated_ids)


def set_binding_semantics(
    scene,
    character_id: str,
    binding_id: str,
    *,
    semantic_key: str,
    side: CharacterSide | str = CharacterSide.NONE,
    mode: ControlMode | str = ControlMode.NEUTRAL,
    usage: ControlUsage | str = ControlUsage.PRIMARY,
    expected_stamp: tuple | None = None,
) -> None:
    """Assign semantic metadata to one existing Object/Bone binding."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    binding_index, binding_row = _binding_row_by_id(row, binding_id)
    if binding_row is None or binding_index is None:
        raise CharacterMetadataError(f"Character binding {binding_id!r} was not found.")

    side_value = _coerce_enum(CharacterSide, side, "Character side")
    mode_value = _coerce_enum(ControlMode, mode, "control mode")
    usage_value = _coerce_enum(ControlUsage, usage, "control usage")
    definition = _decode_character(row)
    bindings = list(definition.bindings)
    bindings[binding_index] = replace(
        bindings[binding_index],
        semantic_key=str(semantic_key),
        side=side_value,
        mode=mode_value,
        usage=usage_value,
    )
    _raise_if_invalid(replace(definition, revision=definition.revision + 1, bindings=tuple(bindings)))

    binding_row.semantic_key = str(semantic_key)
    binding_row.side = side_value.value
    binding_row.mode = mode_value.value
    binding_row.usage = usage_value.value
    row.revision = int(row.revision) + 1


def add_character_group(
    scene,
    character_id: str,
    *,
    semantic_key: str,
    label: str,
    members: tuple[str, ...],
    side: CharacterSide | str = CharacterSide.NONE,
    parent_group_id: str | None = None,
    expected_stamp: tuple | None = None,
) -> str:
    """Add one unordered semantic group after full pure-model preflight."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    side_value = _coerce_enum(CharacterSide, side, "Character side")
    group = CharacterGroup(
        group_id=str(uuid4()),
        semantic_key=str(semantic_key),
        label=str(label),
        side=side_value,
        parent_group_id=parent_group_id or None,
        members=tuple(str(item) for item in members),
    )
    definition = _decode_character(row)
    _raise_if_invalid(replace(definition, revision=definition.revision + 1, groups=definition.groups + (group,)))

    target = row.groups.add()
    target.group_id = group.group_id
    target.semantic_key = group.semantic_key
    target.label = group.label
    target.side = group.side.value
    target.parent_group_id = group.parent_group_id or ""
    for binding_id in group.members:
        member = target.members.add()
        member.binding_id = binding_id
    row.revision = int(row.revision) + 1
    return group.group_id


def add_character_chain(
    scene,
    character_id: str,
    *,
    semantic_key: str,
    label: str,
    members: tuple[str, ...],
    side: CharacterSide | str = CharacterSide.NONE,
    mode: ControlMode | str = ControlMode.NEUTRAL,
    expected_stamp: tuple | None = None,
) -> str:
    """Add one ordered variable-length semantic chain."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    side_value = _coerce_enum(CharacterSide, side, "Character side")
    mode_value = _coerce_enum(ControlMode, mode, "control mode")
    chain = CharacterChain(
        chain_id=str(uuid4()),
        semantic_key=str(semantic_key),
        label=str(label),
        side=side_value,
        mode=mode_value,
        members=tuple(str(item) for item in members),
    )
    definition = _decode_character(row)
    _raise_if_invalid(replace(definition, revision=definition.revision + 1, chains=definition.chains + (chain,)))

    target = row.chains.add()
    target.chain_id = chain.chain_id
    target.semantic_key = chain.semantic_key
    target.label = chain.label
    target.side = chain.side.value
    target.mode = chain.mode.value
    for binding_id in chain.members:
        member = target.members.add()
        member.binding_id = binding_id
    row.revision = int(row.revision) + 1
    return chain.chain_id


def set_character_group(
    scene,
    character_id: str,
    group_id: str,
    *,
    semantic_key: str,
    label: str,
    members: tuple[str, ...],
    side: CharacterSide | str = CharacterSide.NONE,
    parent_group_id: str | None = None,
    expected_stamp: tuple | None = None,
) -> None:
    """Replace one authored group definition after full semantic-graph preflight."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    side_value = _coerce_enum(CharacterSide, side, "Character side")
    definition = _decode_character(row)
    group_index = next(
        (index for index, group in enumerate(definition.groups) if group.group_id == group_id),
        None,
    )
    if group_index is None:
        raise CharacterMetadataError(f"Character group {group_id!r} was not found.")
    groups = list(definition.groups)
    groups[group_index] = replace(
        groups[group_index],
        semantic_key=str(semantic_key),
        label=str(label),
        side=side_value,
        parent_group_id=parent_group_id or None,
        members=tuple(str(item) for item in members),
    )
    _raise_if_invalid(replace(definition, revision=definition.revision + 1, groups=tuple(groups)))

    target = next((group for group in row.groups if str(group.group_id) == group_id), None)
    if target is None:
        raise CharacterMetadataError(f"Character group {group_id!r} was not found in RNA storage.")
    target.semantic_key = str(semantic_key)
    target.label = str(label)
    target.side = side_value.value
    target.parent_group_id = parent_group_id or ""
    target.members.clear()
    for binding_id in members:
        member = target.members.add()
        member.binding_id = str(binding_id)
    row.revision = int(row.revision) + 1


def remove_character_group(
    scene,
    character_id: str,
    group_id: str,
    *,
    expected_stamp: tuple | None = None,
) -> None:
    """Remove one group only when no remaining semantic record depends on it."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    definition = _decode_character(row)
    groups = tuple(group for group in definition.groups if group.group_id != group_id)
    if len(groups) == len(definition.groups):
        raise CharacterMetadataError(f"Character group {group_id!r} was not found.")
    _raise_if_invalid(replace(definition, revision=definition.revision + 1, groups=groups))

    for index, group in enumerate(row.groups):
        if str(group.group_id) == group_id:
            row.groups.remove(index)
            row.revision = int(row.revision) + 1
            return
    raise CharacterMetadataError(f"Character group {group_id!r} was not found in RNA storage.")


def set_character_chain(
    scene,
    character_id: str,
    chain_id: str,
    *,
    semantic_key: str,
    label: str,
    members: tuple[str, ...],
    side: CharacterSide | str = CharacterSide.NONE,
    mode: ControlMode | str = ControlMode.NEUTRAL,
    expected_stamp: tuple | None = None,
) -> None:
    """Replace one ordered chain definition; member tuple order is authoritative."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    side_value = _coerce_enum(CharacterSide, side, "Character side")
    mode_value = _coerce_enum(ControlMode, mode, "control mode")
    definition = _decode_character(row)
    chain_index = next(
        (index for index, chain in enumerate(definition.chains) if chain.chain_id == chain_id),
        None,
    )
    if chain_index is None:
        raise CharacterMetadataError(f"Character chain {chain_id!r} was not found.")
    chains = list(definition.chains)
    chains[chain_index] = replace(
        chains[chain_index],
        semantic_key=str(semantic_key),
        label=str(label),
        side=side_value,
        mode=mode_value,
        members=tuple(str(item) for item in members),
    )
    _raise_if_invalid(replace(definition, revision=definition.revision + 1, chains=tuple(chains)))

    target = next((chain for chain in row.chains if str(chain.chain_id) == chain_id), None)
    if target is None:
        raise CharacterMetadataError(f"Character chain {chain_id!r} was not found in RNA storage.")
    target.semantic_key = str(semantic_key)
    target.label = str(label)
    target.side = side_value.value
    target.mode = mode_value.value
    target.members.clear()
    for binding_id in members:
        member = target.members.add()
        member.binding_id = str(binding_id)
    row.revision = int(row.revision) + 1


def remove_character_chain(
    scene,
    character_id: str,
    chain_id: str,
    *,
    expected_stamp: tuple | None = None,
) -> None:
    """Remove one chain only when no kinematic mapping or graph record still references it."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    definition = _decode_character(row)
    chains = tuple(chain for chain in definition.chains if chain.chain_id != chain_id)
    if len(chains) == len(definition.chains):
        raise CharacterMetadataError(f"Character chain {chain_id!r} was not found.")
    _raise_if_invalid(replace(definition, revision=definition.revision + 1, chains=chains))

    for index, chain in enumerate(row.chains):
        if str(chain.chain_id) == chain_id:
            row.chains.remove(index)
            row.revision = int(row.revision) + 1
            return
    raise CharacterMetadataError(f"Character chain {chain_id!r} was not found in RNA storage.")


def add_opposite_pair(
    scene,
    character_id: str,
    left_binding_id: str,
    right_binding_id: str,
    *,
    expected_stamp: tuple | None = None,
) -> None:
    """Add one explicit symmetric opposite relation without transform assumptions."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    pair = OppositePair(str(left_binding_id), str(right_binding_id))
    definition = _decode_character(row)
    _raise_if_invalid(
        replace(definition, revision=definition.revision + 1, opposites=definition.opposites + (pair,))
    )

    target = row.opposites.add()
    target.left_binding_id = pair.left_binding_id
    target.right_binding_id = pair.right_binding_id
    row.revision = int(row.revision) + 1


def add_kinematic_mapping(
    scene,
    character_id: str,
    *,
    semantic_key: str,
    side: CharacterSide | str = CharacterSide.NONE,
    fk_chain_id: str | None = None,
    ik_target_binding_id: str | None = None,
    pole_binding_id: str | None = None,
    reference_chain_id: str | None = None,
    extras: tuple[str, ...] = (),
    expected_stamp: tuple | None = None,
) -> str:
    """Add one generic authored FK/IK relationship record with no solver behavior."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)

    side_value = _coerce_enum(CharacterSide, side, "Character side")
    mapping = KinematicMapping(
        mapping_id=str(uuid4()),
        semantic_key=str(semantic_key),
        side=side_value,
        fk_chain_id=(str(fk_chain_id) if fk_chain_id else None),
        ik_target_binding_id=(str(ik_target_binding_id) if ik_target_binding_id else None),
        pole_binding_id=(str(pole_binding_id) if pole_binding_id else None),
        reference_chain_id=(str(reference_chain_id) if reference_chain_id else None),
        extras=tuple(str(item) for item in extras),
    )
    definition = _decode_character(row)
    _raise_if_invalid(
        replace(
            definition,
            revision=definition.revision + 1,
            kinematics=definition.kinematics + (mapping,),
        )
    )

    target = row.kinematics.add()
    target.mapping_id = mapping.mapping_id
    target.semantic_key = mapping.semantic_key
    target.side = mapping.side.value
    target.fk_chain_id = mapping.fk_chain_id or ""
    target.ik_target_binding_id = mapping.ik_target_binding_id or ""
    target.pole_binding_id = mapping.pole_binding_id or ""
    target.reference_chain_id = mapping.reference_chain_id or ""
    for binding_id in mapping.extras:
        member = target.extras.add()
        member.binding_id = binding_id
    row.revision = int(row.revision) + 1
    return mapping.mapping_id


def remove_opposite_pair(
    scene,
    character_id: str,
    left_binding_id: str,
    right_binding_id: str,
    *,
    expected_stamp: tuple | None = None,
) -> None:
    """Remove one explicit opposite relation after whole-graph preflight."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)

    definition = _decode_character(row)
    wanted = {str(left_binding_id), str(right_binding_id)}
    pair_index = next(
        (
            index
            for index, pair in enumerate(definition.opposites)
            if {pair.left_binding_id, pair.right_binding_id} == wanted
        ),
        None,
    )
    if pair_index is None:
        raise CharacterMetadataError("Character opposite relation was not found.")
    opposites = list(definition.opposites)
    opposites.pop(pair_index)
    _raise_if_invalid(
        replace(definition, revision=definition.revision + 1, opposites=tuple(opposites))
    )

    for index, pair in enumerate(row.opposites):
        if {str(pair.left_binding_id), str(pair.right_binding_id)} == wanted:
            row.opposites.remove(index)
            row.revision = int(row.revision) + 1
            return
    raise CharacterMetadataError("Character opposite relation was not found in RNA storage.")


def set_kinematic_mapping(
    scene,
    character_id: str,
    mapping_id: str,
    *,
    semantic_key: str,
    side: CharacterSide | str = CharacterSide.NONE,
    fk_chain_id: str | None = None,
    ik_target_binding_id: str | None = None,
    pole_binding_id: str | None = None,
    reference_chain_id: str | None = None,
    extras: tuple[str, ...] = (),
    expected_stamp: tuple | None = None,
) -> None:
    """Replace one generic kinematic relationship after full semantic-graph preflight."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)

    side_value = _coerce_enum(CharacterSide, side, "Character side")
    definition = _decode_character(row)
    mapping_index = next(
        (
            index
            for index, mapping in enumerate(definition.kinematics)
            if mapping.mapping_id == mapping_id
        ),
        None,
    )
    if mapping_index is None:
        raise CharacterMetadataError(f"Kinematic mapping {mapping_id!r} was not found.")
    mappings = list(definition.kinematics)
    mappings[mapping_index] = KinematicMapping(
        mapping_id=str(mapping_id),
        semantic_key=str(semantic_key),
        side=side_value,
        fk_chain_id=(str(fk_chain_id) if fk_chain_id else None),
        ik_target_binding_id=(str(ik_target_binding_id) if ik_target_binding_id else None),
        pole_binding_id=(str(pole_binding_id) if pole_binding_id else None),
        reference_chain_id=(str(reference_chain_id) if reference_chain_id else None),
        extras=tuple(str(item) for item in extras),
    )
    _raise_if_invalid(
        replace(definition, revision=definition.revision + 1, kinematics=tuple(mappings))
    )

    target = next(
        (mapping for mapping in row.kinematics if str(mapping.mapping_id) == mapping_id),
        None,
    )
    if target is None:
        raise CharacterMetadataError(f"Kinematic mapping {mapping_id!r} was not found in RNA storage.")
    mapping = mappings[mapping_index]
    target.semantic_key = mapping.semantic_key
    target.side = mapping.side.value
    target.fk_chain_id = mapping.fk_chain_id or ""
    target.ik_target_binding_id = mapping.ik_target_binding_id or ""
    target.pole_binding_id = mapping.pole_binding_id or ""
    target.reference_chain_id = mapping.reference_chain_id or ""
    target.extras.clear()
    for binding_id in mapping.extras:
        member = target.extras.add()
        member.binding_id = binding_id
    row.revision = int(row.revision) + 1


def remove_kinematic_mapping(
    scene,
    character_id: str,
    mapping_id: str,
    *,
    expected_stamp: tuple | None = None,
) -> None:
    """Remove one kinematic relationship without touching native rig or animation data."""

    store = _require_v2_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)

    definition = _decode_character(row)
    mappings = tuple(
        mapping for mapping in definition.kinematics if mapping.mapping_id != mapping_id
    )
    if len(mappings) == len(definition.kinematics):
        raise CharacterMetadataError(f"Kinematic mapping {mapping_id!r} was not found.")
    _raise_if_invalid(
        replace(definition, revision=definition.revision + 1, kinematics=mappings)
    )

    for index, mapping in enumerate(row.kinematics):
        if str(mapping.mapping_id) == mapping_id:
            row.kinematics.remove(index)
            row.revision = int(row.revision) + 1
            return
    raise CharacterMetadataError(f"Kinematic mapping {mapping_id!r} was not found in RNA storage.")


def resolve_character(scene, character_id: str) -> ResolvedCharacter:
    """Resolve current native targets using the existing Phase 2 adapter.

    Native binding resolution is structural: pose values and animation evaluation
    do not change which generated Object/PoseBone a Character binding names. Cache
    the last resolved Character behind both the metadata stamp and a native
    Object/Armature/Bone-token signature so high-frequency Contact/Track Bar paths
    do not repeatedly perform the same per-binding bone-token searches.
    """

    global _RESOLVED_CHARACTER_CACHE_KEY, _RESOLVED_CHARACTER_CACHE_VALUE

    store = _require_supported_store(scene)
    _index, row = _row_by_character_id(store, character_id)
    if row is None:
        raise CharacterNotFound(character_id)
    source_stamp = character_stamp(scene, character_id)
    cache_key = (
        str(character_id),
        source_stamp,
        _resolved_character_native_signature(scene, row),
    )
    if cache_key == _RESOLVED_CHARACTER_CACHE_KEY and _RESOLVED_CHARACTER_CACHE_VALUE is not None:
        if _cached_resolved_character_is_live(_RESOLVED_CHARACTER_CACHE_VALUE):
            return _RESOLVED_CHARACTER_CACHE_VALUE
        _invalidate_resolved_character_cache()

    definition = read_character(scene, character_id)
    if definition is None:
        raise CharacterNotFound(character_id)

    owner_rows = {str(owner.owner_id): owner for owner in row.owners}

    from .semantic_adapter import resolve_control_target

    resolved: list[tuple[str, object]] = []
    issues: list[CharacterIssue] = []
    for binding in definition.bindings:
        owner = owner_rows.get(binding.owner_id)
        owner_object = owner.object_ref if owner is not None else None
        owner_in_scene = owner_object is not None and _scene_contains_object_identity(
            scene,
            owner_object,
        )
        if not owner_in_scene:
            detail = (
                "The stored Character owner Object no longer exists."
                if owner_object is None
                else "The stored Character owner Object is no longer part of the target Scene."
            )
            issues.append(
                CharacterIssue(
                    code="MISSING_OBJECT_TARGET",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id,
                    record_id=binding.binding_id,
                    detail=detail,
                )
            )
            continue

        target = owner_object
        if binding.kind == AWBControlKind.BONE:
            matches = bone_token_matches(owner_object, binding.bone_id or "")
            if len(matches) == 0:
                issues.append(
                    CharacterIssue(
                        code="MISSING_BONE_TOKEN_TARGET",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id,
                        record_id=binding.binding_id,
                        detail=(
                            f"Bone token {binding.bone_id!r} no longer exists on "
                            f"Armature Object {owner_object.name!r}."
                        ),
                    )
                )
                continue
            if len(matches) > 1:
                issues.append(
                    CharacterIssue(
                        code="AMBIGUOUS_BONE_TOKEN_TARGET",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id,
                        record_id=binding.binding_id,
                        detail=(
                            f"Bone token {binding.bone_id!r} matches {len(matches)} Bones on "
                            f"Armature Object {owner_object.name!r}."
                        ),
                    )
                )
                continue
            pose = getattr(owner_object, "pose", None)
            pose_bones = getattr(pose, "bones", None) if pose is not None else None
            target = pose_bones.get(matches[0].name) if pose_bones is not None else None
            if target is None:
                issues.append(
                    CharacterIssue(
                        code="MISSING_POSE_BONE_TARGET",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id,
                        record_id=binding.binding_id,
                        detail="Resolved Bone data has no PoseBone on the stored owner Object.",
                    )
                )
                continue

        decoded = resolve_control_target(target)
        if decoded is None:
            issues.append(
                CharacterIssue(
                    code=(
                        "UNRESOLVED_BONE_TARGET"
                        if binding.kind == AWBControlKind.BONE
                        else "UNRESOLVED_OBJECT_TARGET"
                    ),
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id,
                    record_id=binding.binding_id,
                    detail="Phase 2 semantic adapter could not resolve the stored native target.",
                )
            )
            continue
        resolved.append((binding.binding_id, decoded))

    result = ResolvedCharacter(
        definition=definition,
        resolved_bindings=tuple(resolved),
        issues=tuple(issues),
        source_stamp=source_stamp,
    )
    _RESOLVED_CHARACTER_CACHE_KEY = cache_key
    _RESOLVED_CHARACTER_CACHE_VALUE = result
    return result


def remove_character(scene, character_id: str, expected_stamp: tuple | None = None) -> None:
    """Remove only the Scene metadata row; Blender Objects/animation remain untouched."""

    _require_writable_character_scene(scene)
    store = _require_supported_store(scene)
    index, _row = _row_by_character_id(store, character_id)
    if index is None:
        raise CharacterNotFound(character_id)
    if expected_stamp is not None and character_stamp(scene, character_id) != expected_stamp:
        raise StaleCharacterDefinition(character_id)
    store.characters.remove(index)


def _register_resolved_character_cache_handlers() -> None:
    try:
        from bpy.app.handlers import persistent

        persistent(_invalidate_resolved_character_cache)
    except (ImportError, AttributeError):
        pass
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is None:
        return
    for name in ("undo_post", "redo_post", "load_pre", "load_post"):
        rows = getattr(handlers, name, None)
        if rows is not None and _invalidate_resolved_character_cache not in rows:
            rows.append(_invalidate_resolved_character_cache)


def _unregister_resolved_character_cache_handlers() -> None:
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is None:
        return
    for name in ("undo_post", "redo_post", "load_pre", "load_post"):
        rows = getattr(handlers, name, None)
        if rows is not None and _invalidate_resolved_character_cache in rows:
            rows.remove(_invalidate_resolved_character_cache)
    _invalidate_resolved_character_cache()


def register() -> None:
    for cls in _METADATA_CLASSES:
        bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Scene, STORE_PROPERTY):
        setattr(
            bpy.types.Scene,
            STORE_PROPERTY,
            PointerProperty(type=BAW_PG_character_store),
        )
    _register_resolved_character_cache_handlers()


def unregister() -> None:
    _unregister_resolved_character_cache_handlers()
    if hasattr(bpy.types.Scene, STORE_PROPERTY):
        delattr(bpy.types.Scene, STORE_PROPERTY)
    for cls in reversed(_METADATA_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
