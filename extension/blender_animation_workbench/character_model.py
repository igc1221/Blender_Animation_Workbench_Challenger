from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from re import compile as compile_pattern

from .semantic_model import AWBControlKind

AWB_CANONICAL_SEMANTIC_KEYS = frozenset(
    {
        "awb.root",
        "awb.com",
        "awb.pelvis",
        "awb.spine",
        "awb.neck",
        "awb.head",
        "awb.clavicle",
        "awb.arm",
        "awb.upper_arm",
        "awb.forearm",
        "awb.hand",
        "awb.finger",
        "awb.leg",
        "awb.thigh",
        "awb.calf",
        "awb.foot",
        "awb.toe",
        "awb.contact",
        "awb.contact_point",
    }
)
_SEMANTIC_KEY_RE = compile_pattern(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9][a-z0-9_]*)+$")


class CharacterSide(StrEnum):
    NONE = "NONE"
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class ControlMode(StrEnum):
    NEUTRAL = "NEUTRAL"
    FK = "FK"
    IK = "IK"
    SHARED = "SHARED"


class ControlUsage(StrEnum):
    PRIMARY = "PRIMARY"
    TARGET = "TARGET"
    POLE = "POLE"
    TWIST = "TWIST"
    HELPER = "HELPER"
    DEFORM = "DEFORM"
    MECHANISM = "MECHANISM"
    SPACE = "SPACE"
    REFERENCE = "REFERENCE"


class CharacterIssueSeverity(StrEnum):
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class CharacterOwner:
    """Pure descriptor for one Blender Object owner used by Character bindings."""

    owner_id: str
    label_hint: str = ""


@dataclass(frozen=True, slots=True)
class CharacterBinding:
    """Persistent Character-local mapping row, independent of animation identity."""

    binding_id: str
    owner_id: str
    kind: AWBControlKind
    bone_id: str | None = None
    bone_name_hint: str | None = None
    semantic_key: str = ""
    side: CharacterSide = CharacterSide.NONE
    mode: ControlMode = ControlMode.NEUTRAL
    usage: ControlUsage = ControlUsage.PRIMARY


@dataclass(frozen=True, slots=True)
class CharacterGroup:
    """Unordered semantic grouping. Parent nesting is explicit and acyclic."""

    group_id: str
    semantic_key: str
    label: str
    side: CharacterSide = CharacterSide.NONE
    parent_group_id: str | None = None
    members: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CharacterChain:
    """Explicit ordered semantic chain; member tuple order is authoritative."""

    chain_id: str
    semantic_key: str
    label: str
    side: CharacterSide = CharacterSide.NONE
    mode: ControlMode = ControlMode.NEUTRAL
    members: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OppositePair:
    """Explicit symmetric left/right binding relation with no mirror-space meaning."""

    left_binding_id: str
    right_binding_id: str


@dataclass(frozen=True, slots=True)
class KinematicMapping:
    """Generic authored FK/IK relationship metadata with no solver behavior."""

    mapping_id: str
    semantic_key: str
    side: CharacterSide = CharacterSide.NONE
    fk_chain_id: str | None = None
    ik_target_binding_id: str | None = None
    pole_binding_id: str | None = None
    reference_chain_id: str | None = None
    extras: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AWBCharacter:
    """Immutable logical snapshot of one Scene-local AWB Character definition.

    Schema v2 is the current authoring model. Locator-only schema v1 remains a
    persistence compatibility state that must be explicitly migrated before
    semantic graph authoring.
    """

    character_id: str
    label: str
    revision: int
    owners: tuple[CharacterOwner, ...]
    bindings: tuple[CharacterBinding, ...]
    groups: tuple[CharacterGroup, ...] = ()
    chains: tuple[CharacterChain, ...] = ()
    opposites: tuple[OppositePair, ...] = ()
    kinematics: tuple[KinematicMapping, ...] = ()


@dataclass(frozen=True, slots=True)
class CharacterIssue:
    code: str
    severity: CharacterIssueSeverity
    character_id: str | None = None
    record_id: str | None = None
    detail: str = ""


def _nonempty(value: str) -> bool:
    return bool(value and value.strip())


def semantic_key_error(value: str, *, allow_empty: bool) -> str | None:
    """Return a stable validation code for one semantic key, or None when valid."""

    if value == "":
        return None if allow_empty else "EMPTY_SEMANTIC_KEY"
    if value != value.strip() or not _SEMANTIC_KEY_RE.fullmatch(value):
        return "INVALID_SEMANTIC_KEY"
    if value.startswith("awb.") and value not in AWB_CANONICAL_SEMANTIC_KEYS:
        return "UNKNOWN_AWB_SEMANTIC_KEY"
    return None


def _semantic_issue(
    value: str,
    *,
    allow_empty: bool,
    character_id: str,
    record_id: str | None,
) -> CharacterIssue | None:
    code = semantic_key_error(value, allow_empty=allow_empty)
    if code is None:
        return None
    return CharacterIssue(
        code=code,
        severity=CharacterIssueSeverity.ERROR,
        character_id=character_id or None,
        record_id=record_id,
        detail=f"Invalid Character semantic key: {value!r}",
    )


def validate_character(definition: AWBCharacter) -> tuple[CharacterIssue, ...]:
    """Validate Character schema structure without resolving Blender RNA."""

    issues: list[CharacterIssue] = []

    if not _nonempty(definition.character_id):
        issues.append(
            CharacterIssue(
                code="EMPTY_CHARACTER_ID",
                severity=CharacterIssueSeverity.ERROR,
                detail="Character ID must not be empty.",
            )
        )
    if definition.revision < 0:
        issues.append(
            CharacterIssue(
                code="INVALID_CHARACTER_REVISION",
                severity=CharacterIssueSeverity.ERROR,
                character_id=definition.character_id or None,
                detail="Character revision must be non-negative.",
            )
        )

    owner_ids: set[str] = set()
    for owner in definition.owners:
        if not _nonempty(owner.owner_id):
            issues.append(
                CharacterIssue(
                    code="EMPTY_OWNER_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    detail="Owner ID must not be empty.",
                )
            )
            continue
        if owner.owner_id in owner_ids:
            issues.append(
                CharacterIssue(
                    code="DUPLICATE_OWNER_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=owner.owner_id,
                    detail="Owner IDs must be unique within a Character.",
                )
            )
        owner_ids.add(owner.owner_id)

    binding_ids: set[str] = set()
    for binding in definition.bindings:
        if not _nonempty(binding.binding_id):
            issues.append(
                CharacterIssue(
                    code="EMPTY_BINDING_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    detail="Binding ID must not be empty.",
                )
            )
        elif binding.binding_id in binding_ids:
            issues.append(
                CharacterIssue(
                    code="DUPLICATE_BINDING_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=binding.binding_id,
                    detail="Binding IDs must be unique within a Character.",
                )
            )
        binding_ids.add(binding.binding_id)

        semantic_issue = _semantic_issue(
            binding.semantic_key,
            allow_empty=True,
            character_id=definition.character_id,
            record_id=binding.binding_id or None,
        )
        if semantic_issue is not None:
            issues.append(semantic_issue)

        if not _nonempty(binding.owner_id) or binding.owner_id not in owner_ids:
            issues.append(
                CharacterIssue(
                    code="MISSING_BINDING_OWNER",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=binding.binding_id or None,
                    detail="Every binding must reference an existing Character owner row.",
                )
            )

        if binding.kind == AWBControlKind.OBJECT:
            if binding.bone_id is not None or binding.bone_name_hint is not None:
                issues.append(
                    CharacterIssue(
                        code="OBJECT_BINDING_HAS_BONE_LOCATOR",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=binding.binding_id or None,
                        detail="Object bindings cannot contain Bone locator fields.",
                    )
                )
        elif binding.kind == AWBControlKind.BONE:
            if not _nonempty(binding.bone_id or ""):
                issues.append(
                    CharacterIssue(
                        code="BONE_BINDING_MISSING_TOKEN",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=binding.binding_id or None,
                        detail="Bone bindings require a non-empty Bone token.",
                    )
                )
        else:
            issues.append(
                CharacterIssue(
                    code="UNSUPPORTED_CONTROL_KIND",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=binding.binding_id or None,
                    detail=f"Unsupported control kind: {binding.kind!r}",
                )
            )

    group_ids: set[str] = set()
    group_by_id: dict[str, CharacterGroup] = {}
    for group in definition.groups:
        if not _nonempty(group.group_id):
            issues.append(
                CharacterIssue(
                    code="EMPTY_GROUP_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    detail="Group ID must not be empty.",
                )
            )
        elif group.group_id in group_ids:
            issues.append(
                CharacterIssue(
                    code="DUPLICATE_GROUP_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=group.group_id,
                    detail="Group IDs must be unique within a Character.",
                )
            )
        group_ids.add(group.group_id)
        group_by_id[group.group_id] = group

        semantic_issue = _semantic_issue(
            group.semantic_key,
            allow_empty=False,
            character_id=definition.character_id,
            record_id=group.group_id or None,
        )
        if semantic_issue is not None:
            issues.append(semantic_issue)

        seen_members: set[str] = set()
        for binding_id in group.members:
            if binding_id in seen_members:
                issues.append(
                    CharacterIssue(
                        code="DUPLICATE_GROUP_MEMBER",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=group.group_id or None,
                        detail=f"Group contains duplicate binding {binding_id!r}.",
                    )
                )
            seen_members.add(binding_id)
            if binding_id not in binding_ids:
                issues.append(
                    CharacterIssue(
                        code="DANGLING_GROUP_MEMBER",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=group.group_id or None,
                        detail=f"Group references missing binding {binding_id!r}.",
                    )
                )

    for group in definition.groups:
        parent_id = group.parent_group_id
        if parent_id is not None and parent_id not in group_ids:
            issues.append(
                CharacterIssue(
                    code="DANGLING_PARENT_GROUP",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=group.group_id or None,
                    detail=f"Parent group {parent_id!r} does not exist.",
                )
            )

    for start in group_ids:
        seen_path: set[str] = set()
        current = start
        while current in group_by_id:
            if current in seen_path:
                issues.append(
                    CharacterIssue(
                        code="GROUP_PARENT_CYCLE",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=start,
                        detail="Nested Character groups must be acyclic.",
                    )
                )
                break
            seen_path.add(current)
            parent = group_by_id[current].parent_group_id
            if parent is None:
                break
            current = parent

    chain_ids: set[str] = set()
    for chain in definition.chains:
        if not _nonempty(chain.chain_id):
            issues.append(
                CharacterIssue(
                    code="EMPTY_CHAIN_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    detail="Chain ID must not be empty.",
                )
            )
        elif chain.chain_id in chain_ids:
            issues.append(
                CharacterIssue(
                    code="DUPLICATE_CHAIN_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=chain.chain_id,
                    detail="Chain IDs must be unique within a Character.",
                )
            )
        chain_ids.add(chain.chain_id)

        semantic_issue = _semantic_issue(
            chain.semantic_key,
            allow_empty=False,
            character_id=definition.character_id,
            record_id=chain.chain_id or None,
        )
        if semantic_issue is not None:
            issues.append(semantic_issue)
        if not chain.members:
            issues.append(
                CharacterIssue(
                    code="EMPTY_CHAIN",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=chain.chain_id or None,
                    detail="Character chains require at least one binding member.",
                )
            )

        seen_members: set[str] = set()
        for binding_id in chain.members:
            if binding_id in seen_members:
                issues.append(
                    CharacterIssue(
                        code="DUPLICATE_CHAIN_MEMBER",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=chain.chain_id or None,
                        detail=f"Chain contains duplicate binding {binding_id!r}.",
                    )
                )
            seen_members.add(binding_id)
            if binding_id not in binding_ids:
                issues.append(
                    CharacterIssue(
                        code="DANGLING_CHAIN_MEMBER",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=chain.chain_id or None,
                        detail=f"Chain references missing binding {binding_id!r}.",
                    )
                )

    opposite_bindings: set[str] = set()
    opposite_pairs: set[tuple[str, str]] = set()
    binding_by_id = {binding.binding_id: binding for binding in definition.bindings}
    for pair in definition.opposites:
        relation = (pair.left_binding_id, pair.right_binding_id)
        if relation in opposite_pairs:
            issues.append(
                CharacterIssue(
                    code="DUPLICATE_OPPOSITE_PAIR",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    detail="Opposite relations must be unique within a Character.",
                )
            )
        opposite_pairs.add(relation)
        if pair.left_binding_id == pair.right_binding_id:
            issues.append(
                CharacterIssue(
                    code="SELF_OPPOSITE_BINDING",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=pair.left_binding_id or None,
                    detail="A binding cannot be opposite to itself.",
                )
            )
        for binding_id in relation:
            if binding_id not in binding_ids:
                issues.append(
                    CharacterIssue(
                        code="DANGLING_OPPOSITE_BINDING",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=binding_id or None,
                        detail=f"Opposite relation references missing binding {binding_id!r}.",
                    )
                )
            elif binding_id in opposite_bindings:
                issues.append(
                    CharacterIssue(
                        code="MULTIPLE_OPPOSITE_RELATIONS",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=binding_id,
                        detail="A binding may participate in only one opposite pair.",
                    )
                )
            opposite_bindings.add(binding_id)

        left = binding_by_id.get(pair.left_binding_id)
        right = binding_by_id.get(pair.right_binding_id)
        if left is not None and left.side != CharacterSide.LEFT:
            issues.append(
                CharacterIssue(
                    code="OPPOSITE_LEFT_SIDE_MISMATCH",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=pair.left_binding_id,
                    detail="OppositePair.left_binding_id must reference a LEFT binding.",
                )
            )
        if right is not None and right.side != CharacterSide.RIGHT:
            issues.append(
                CharacterIssue(
                    code="OPPOSITE_RIGHT_SIDE_MISMATCH",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=pair.right_binding_id,
                    detail="OppositePair.right_binding_id must reference a RIGHT binding.",
                )
            )

    mapping_ids: set[str] = set()
    for mapping in definition.kinematics:
        if not _nonempty(mapping.mapping_id):
            issues.append(
                CharacterIssue(
                    code="EMPTY_KINEMATIC_MAPPING_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    detail="Kinematic mapping ID must not be empty.",
                )
            )
        elif mapping.mapping_id in mapping_ids:
            issues.append(
                CharacterIssue(
                    code="DUPLICATE_KINEMATIC_MAPPING_ID",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=mapping.mapping_id,
                    detail="Kinematic mapping IDs must be unique within a Character.",
                )
            )
        mapping_ids.add(mapping.mapping_id)

        semantic_issue = _semantic_issue(
            mapping.semantic_key,
            allow_empty=False,
            character_id=definition.character_id,
            record_id=mapping.mapping_id or None,
        )
        if semantic_issue is not None:
            issues.append(semantic_issue)

        if not any(
            (
                mapping.fk_chain_id,
                mapping.ik_target_binding_id,
                mapping.pole_binding_id,
                mapping.reference_chain_id,
                mapping.extras,
            )
        ):
            issues.append(
                CharacterIssue(
                    code="EMPTY_KINEMATIC_MAPPING",
                    severity=CharacterIssueSeverity.ERROR,
                    character_id=definition.character_id or None,
                    record_id=mapping.mapping_id or None,
                    detail="Kinematic mapping must reference at least one authored rig element.",
                )
            )

        for chain_id, code, label in (
            (mapping.fk_chain_id, "DANGLING_KINEMATIC_FK_CHAIN", "FK chain"),
            (
                mapping.reference_chain_id,
                "DANGLING_KINEMATIC_REFERENCE_CHAIN",
                "reference chain",
            ),
        ):
            if chain_id is not None and chain_id not in chain_ids:
                issues.append(
                    CharacterIssue(
                        code=code,
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=mapping.mapping_id or None,
                        detail=f"Kinematic mapping references missing {label} {chain_id!r}.",
                    )
                )

        for binding_id, code, label in (
            (
                mapping.ik_target_binding_id,
                "DANGLING_KINEMATIC_IK_TARGET",
                "IK target",
            ),
            (mapping.pole_binding_id, "DANGLING_KINEMATIC_POLE", "pole"),
        ):
            if binding_id is not None and binding_id not in binding_ids:
                issues.append(
                    CharacterIssue(
                        code=code,
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=mapping.mapping_id or None,
                        detail=f"Kinematic mapping references missing {label} {binding_id!r}.",
                    )
                )

        seen_extras: set[str] = set()
        for binding_id in mapping.extras:
            if binding_id in seen_extras:
                issues.append(
                    CharacterIssue(
                        code="DUPLICATE_KINEMATIC_EXTRA",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=mapping.mapping_id or None,
                        detail=f"Kinematic extras contain duplicate binding {binding_id!r}.",
                    )
                )
            seen_extras.add(binding_id)
            if binding_id not in binding_ids:
                issues.append(
                    CharacterIssue(
                        code="DANGLING_KINEMATIC_EXTRA",
                        severity=CharacterIssueSeverity.ERROR,
                        character_id=definition.character_id or None,
                        record_id=mapping.mapping_id or None,
                        detail=f"Kinematic extras reference missing binding {binding_id!r}.",
                    )
                )

    return tuple(issues)


def has_character_errors(issues: tuple[CharacterIssue, ...]) -> bool:
    return any(issue.severity == CharacterIssueSeverity.ERROR for issue in issues)
