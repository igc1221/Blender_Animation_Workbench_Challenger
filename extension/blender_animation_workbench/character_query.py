from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .character_model import (
    AWBCharacter,
    CharacterBinding,
    CharacterChain,
    CharacterGroup,
    CharacterIssue,
    CharacterIssueSeverity,
    CharacterSide,
    ControlMode,
    ControlUsage,
    KinematicMapping,
)
from .semantic_adapter import ControlContext, ResolvedControl, runtime_control_key

if TYPE_CHECKING:
    from .character_metadata import ResolvedCharacter


@dataclass(frozen=True, slots=True)
class CharacterSemanticIndex:
    """Call-scoped pure lookup index for one immutable Character definition."""

    bindings_by_id: dict[str, CharacterBinding]
    bindings_by_semantic: dict[str, tuple[CharacterBinding, ...]]
    groups_by_id: dict[str, CharacterGroup]
    chains_by_id: dict[str, CharacterChain]
    kinematics_by_id: dict[str, KinematicMapping]
    opposites_by_binding: dict[str, str]


@dataclass(frozen=True, slots=True)
class CharacterSelectionSummary:
    """Read-only relationship between the current native selection and Characters."""

    character_ids: tuple[str, ...]
    active_character_id: str | None
    unassigned: tuple[ResolvedControl, ...]
    issues: tuple[CharacterIssue, ...]


@dataclass(frozen=True, slots=True)
class SelectionExpansion:
    """Read-only semantic expansion ready for a later native-selection command."""

    character_id: str
    binding_ids: tuple[str, ...]
    targets: tuple[ResolvedControl, ...]
    issues: tuple[CharacterIssue, ...]
    source_stamp: tuple


def build_semantic_index(definition: AWBCharacter) -> CharacterSemanticIndex:
    """Build a fresh non-persistent graph index without Blender/RNA references."""

    semantic: dict[str, list[CharacterBinding]] = {}
    for binding in definition.bindings:
        semantic.setdefault(binding.semantic_key, []).append(binding)

    opposites: dict[str, str] = {}
    for pair in definition.opposites:
        opposites[pair.left_binding_id] = pair.right_binding_id
        opposites[pair.right_binding_id] = pair.left_binding_id
    return CharacterSemanticIndex(
        bindings_by_id={binding.binding_id: binding for binding in definition.bindings},
        bindings_by_semantic={key: tuple(items) for key, items in semantic.items()},
        groups_by_id={group.group_id: group for group in definition.groups},
        chains_by_id={chain.chain_id: chain for chain in definition.chains},
        kinematics_by_id={mapping.mapping_id: mapping for mapping in definition.kinematics},
        opposites_by_binding=opposites,
    )


def members_for_semantic(
    definition: AWBCharacter,
    semantic_key: str,
    *,
    side: CharacterSide | None = None,
    mode: ControlMode | None = None,
    usages: frozenset[ControlUsage] | None = None,
) -> tuple[CharacterBinding, ...]:
    """Return authored bindings matching one semantic key and optional compositional filters."""

    index = build_semantic_index(definition)
    result: list[CharacterBinding] = []
    for binding in index.bindings_by_semantic.get(semantic_key, ()):
        if side is not None and binding.side != side:
            continue
        if mode is not None and binding.mode != mode:
            continue
        if usages is not None and binding.usage not in usages:
            continue
        result.append(binding)
    return tuple(result)


def group_members(definition: AWBCharacter, group_id: str) -> tuple[CharacterBinding, ...]:
    """Return direct members of an unordered group; missing groups return an empty tuple."""

    index = build_semantic_index(definition)
    group = index.groups_by_id.get(group_id)
    if group is None:
        return ()
    return tuple(
        index.bindings_by_id[binding_id]
        for binding_id in group.members
        if binding_id in index.bindings_by_id
    )


def ordered_chain(definition: AWBCharacter, chain_id: str) -> tuple[CharacterBinding, ...]:
    """Resolve one explicit Character chain while preserving its authored member order."""

    index = build_semantic_index(definition)
    chain = index.chains_by_id.get(chain_id)
    if chain is None:
        return ()
    return tuple(
        index.bindings_by_id[binding_id]
        for binding_id in chain.members
        if binding_id in index.bindings_by_id
    )


def opposite_for_binding(definition: AWBCharacter, binding_id: str) -> CharacterBinding | None:
    """Return the explicitly authored opposite binding, never a name-derived guess."""

    index = build_semantic_index(definition)
    opposite_id = index.opposites_by_binding.get(binding_id)
    return index.bindings_by_id.get(opposite_id) if opposite_id is not None else None


def kinematic_mapping(definition: AWBCharacter, mapping_id: str) -> KinematicMapping | None:
    """Return one authored kinematic relationship by stable Character-local mapping ID."""

    return build_semantic_index(definition).kinematics_by_id.get(mapping_id)


def _query_issue(
    definition: AWBCharacter,
    code: str,
    detail: str,
    *,
    record_id: str | None = None,
) -> CharacterIssue:
    return CharacterIssue(
        code=code,
        severity=CharacterIssueSeverity.ERROR,
        character_id=definition.character_id or None,
        record_id=record_id,
        detail=detail,
    )


def _append_unique(items: list[str], values) -> None:
    seen = set(items)
    for value in values:
        if value not in seen:
            items.append(value)
            seen.add(value)


def _kinematic_binding_ids(
    definition: AWBCharacter,
    index: CharacterSemanticIndex,
    mapping: KinematicMapping,
) -> tuple[tuple[str, ...], tuple[CharacterIssue, ...]]:
    binding_ids: list[str] = []
    issues: list[CharacterIssue] = []

    for chain_id, code, label in (
        (mapping.fk_chain_id, "MISSING_SELECTION_FK_CHAIN", "FK chain"),
        (mapping.reference_chain_id, "MISSING_SELECTION_REFERENCE_CHAIN", "reference chain"),
    ):
        if chain_id is None:
            continue
        chain = index.chains_by_id.get(chain_id)
        if chain is None:
            issues.append(
                _query_issue(
                    definition,
                    code,
                    f"Kinematic mapping references missing {label} {chain_id!r}.",
                    record_id=mapping.mapping_id,
                )
            )
            continue
        if chain_id == mapping.fk_chain_id:
            _append_unique(binding_ids, chain.members)

    for binding_id, code, label in (
        (mapping.ik_target_binding_id, "MISSING_SELECTION_IK_TARGET", "IK target"),
        (mapping.pole_binding_id, "MISSING_SELECTION_POLE", "pole"),
    ):
        if binding_id is None:
            continue
        if binding_id not in index.bindings_by_id:
            issues.append(
                _query_issue(
                    definition,
                    code,
                    f"Kinematic mapping references missing {label} {binding_id!r}.",
                    record_id=mapping.mapping_id,
                )
            )
            continue
        _append_unique(binding_ids, (binding_id,))

    if mapping.reference_chain_id is not None:
        reference = index.chains_by_id.get(mapping.reference_chain_id)
        if reference is not None:
            _append_unique(binding_ids, reference.members)

    for binding_id in mapping.extras:
        if binding_id not in index.bindings_by_id:
            issues.append(
                _query_issue(
                    definition,
                    "MISSING_SELECTION_EXTRA",
                    f"Kinematic mapping references missing extra binding {binding_id!r}.",
                    record_id=mapping.mapping_id,
                )
            )
            continue
        _append_unique(binding_ids, (binding_id,))

    return tuple(binding_ids), tuple(issues)


def selection_targets(
    view: ResolvedCharacter,
    *,
    binding_id: str | None = None,
    semantic_key: str | None = None,
    group_id: str | None = None,
    chain_id: str | None = None,
    mapping_id: str | None = None,
    side: CharacterSide | None = None,
    mode: ControlMode | None = None,
    usages: frozenset[ControlUsage] | None = None,
    include_helpers: bool = False,
) -> SelectionExpansion:
    """Expand one explicit Character scope to resolved native targets without selecting anything."""

    definition = view.definition
    index = build_semantic_index(definition)
    issues: list[CharacterIssue] = []
    scopes = tuple(
        name
        for name, value in (
            ("binding_id", binding_id),
            ("semantic_key", semantic_key),
            ("group_id", group_id),
            ("chain_id", chain_id),
            ("mapping_id", mapping_id),
        )
        if value is not None
    )
    if len(scopes) > 1:
        issues.append(
            _query_issue(
                definition,
                "MULTIPLE_SELECTION_SCOPES",
                f"Selection expansion accepts one explicit scope, got {', '.join(scopes)}.",
            )
        )
        return SelectionExpansion(definition.character_id, (), (), tuple(issues), view.source_stamp)

    binding_ids: tuple[str, ...]
    if binding_id is not None:
        binding_ids = (binding_id,)
    elif semantic_key is not None:
        members = members_for_semantic(
            definition,
            semantic_key,
            side=side,
            mode=mode,
            usages=usages,
        )
        binding_ids = tuple(binding.binding_id for binding in members)
        if not binding_ids:
            issues.append(
                _query_issue(
                    definition,
                    "MISSING_SEMANTIC_MEMBERS",
                    f"No Character bindings match semantic key {semantic_key!r} and the requested filters.",
                )
            )
    elif group_id is not None:
        group = index.groups_by_id.get(group_id)
        if group is None:
            binding_ids = ()
            issues.append(
                _query_issue(
                    definition,
                    "MISSING_GROUP",
                    f"Character group {group_id!r} does not exist.",
                    record_id=group_id,
                )
            )
        else:
            binding_ids = group.members
    elif chain_id is not None:
        chain = index.chains_by_id.get(chain_id)
        if chain is None:
            binding_ids = ()
            issues.append(
                _query_issue(
                    definition,
                    "MISSING_CHAIN",
                    f"Character chain {chain_id!r} does not exist.",
                    record_id=chain_id,
                )
            )
        else:
            binding_ids = chain.members
    elif mapping_id is not None:
        mapping = index.kinematics_by_id.get(mapping_id)
        if mapping is None:
            binding_ids = ()
            issues.append(
                _query_issue(
                    definition,
                    "MISSING_KINEMATIC_MAPPING",
                    f"Kinematic mapping {mapping_id!r} does not exist.",
                    record_id=mapping_id,
                )
            )
        else:
            binding_ids, mapping_issues = _kinematic_binding_ids(definition, index, mapping)
            issues.extend(mapping_issues)
    else:
        binding_ids = tuple(binding.binding_id for binding in definition.bindings)

    helper_usages = frozenset(
        {ControlUsage.HELPER, ControlUsage.TWIST, ControlUsage.MECHANISM}
    )
    filtered_ids: list[str] = []
    for candidate_binding_id in binding_ids:
        binding = index.bindings_by_id.get(candidate_binding_id)
        if binding is None:
            issues.append(
                _query_issue(
                    definition,
                    "MISSING_SELECTION_BINDING",
                    f"Selection scope references missing binding {candidate_binding_id!r}.",
                    record_id=candidate_binding_id,
                )
            )
            continue
        if side is not None and binding.side != side:
            continue
        if mode is not None and binding.mode != mode:
            continue
        if usages is not None and binding.usage not in usages:
            continue
        if not include_helpers and binding.usage in helper_usages:
            continue
        if candidate_binding_id not in filtered_ids:
            filtered_ids.append(candidate_binding_id)

    selected_ids = set(filtered_ids)
    issues.extend(
        issue
        for issue in view.issues
        if issue.record_id is None or issue.record_id in selected_ids
    )

    resolved_by_binding = {binding_id: target for binding_id, target in view.resolved_bindings}
    targets: list[ResolvedControl] = []
    target_keys: set[tuple[int, int]] = set()
    for candidate_binding_id in filtered_ids:
        target = resolved_by_binding.get(candidate_binding_id)
        if target is None:
            if not any(issue.record_id == candidate_binding_id for issue in issues):
                issues.append(
                    _query_issue(
                        definition,
                        "MISSING_SELECTION_TARGET",
                        f"Binding {candidate_binding_id!r} has no resolved native target.",
                        record_id=candidate_binding_id,
                    )
                )
            continue
        runtime_key = runtime_control_key(target)
        if runtime_key in target_keys:
            continue
        target_keys.add(runtime_key)
        targets.append(target)

    return SelectionExpansion(
        character_id=definition.character_id,
        binding_ids=tuple(filtered_ids),
        targets=tuple(targets),
        issues=tuple(issues),
        source_stamp=view.source_stamp,
    )


def characters_for_context(scene, control_context: ControlContext) -> CharacterSelectionSummary:
    """Describe which registered Characters own the currently selected native controls."""

    from .character_metadata import character_ids, resolve_character

    resolved_views = tuple(resolve_character(scene, character_id) for character_id in character_ids(scene))
    memberships: dict[tuple[int, int], list[str]] = {}
    issues: list[CharacterIssue] = []

    for view in resolved_views:
        issues.extend(view.issues)
        seen_for_character: set[tuple[int, int]] = set()
        for _binding_id, target in view.resolved_bindings:
            key = runtime_control_key(target)
            if key in seen_for_character:
                continue
            seen_for_character.add(key)
            memberships.setdefault(key, []).append(view.definition.character_id)

    selected_keys = {runtime_control_key(control) for control in control_context.controls}
    character_order = [view.definition.character_id for view in resolved_views]
    selected_character_ids = tuple(
        character_id
        for character_id in character_order
        if any(character_id in memberships.get(key, ()) for key in selected_keys)
    )

    unassigned: list[ResolvedControl] = []
    ambiguous_keys: set[tuple[int, int]] = set()
    for control in control_context.controls:
        key = runtime_control_key(control)
        owners = memberships.get(key, ())
        if not owners:
            unassigned.append(control)
        elif len(owners) > 1 and key not in ambiguous_keys:
            ambiguous_keys.add(key)
            issues.append(
                CharacterIssue(
                    code="AMBIGUOUS_CONTEXT_MEMBERSHIP",
                    severity=CharacterIssueSeverity.ERROR,
                    detail=(
                        "One selected native control resolves to multiple Characters: "
                        + ", ".join(owners)
                    ),
                )
            )

    active_character_id = None
    if control_context.active is not None:
        owners = memberships.get(runtime_control_key(control_context.active), ())
        if len(owners) == 1:
            active_character_id = owners[0]
        elif len(owners) > 1:
            issues.append(
                CharacterIssue(
                    code="AMBIGUOUS_ACTIVE_CHARACTER",
                    severity=CharacterIssueSeverity.ERROR,
                    detail="The active selected control belongs to multiple Characters.",
                )
            )

    return CharacterSelectionSummary(
        character_ids=selected_character_ids,
        active_character_id=active_character_id,
        unassigned=tuple(unassigned),
        issues=tuple(issues),
    )
