from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .character_metadata import resolve_character
from .phase4_representation_snap import resolve_limb_representation_capability
from .rigped_contract import RigpedCapability, RigpedControlContract, resolve_rigped_target
from .semantic_adapter import ControlContext, runtime_control_key


class OperationDomainKind(StrEnum):
    CONTACT_LIMB = "CONTACT_LIMB"
    DIRECT = "DIRECT"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class OperationDomainIssue:
    code: str
    detail: str
    binding_id: str | None = None
    runtime_key: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class OperationBindingDomain:
    selected_control_runtime_key: tuple[int, int]
    selected_binding_id: str | None
    kind: OperationDomainKind
    mapping_id: str | None = None
    domain_binding_ids: tuple[str, ...] = ()
    rejection_code: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class OperationDomainSnapshot:
    character_id: str
    setup_revision: int
    setup_signature: str
    character_source_stamp: tuple
    frame: int
    subframe: float
    selected_control_runtime_keys: tuple[tuple[int, int], ...]
    selected_binding_ids: tuple[str, ...]
    active_control_runtime_key: tuple[int, int] | None
    active_binding_id: str | None
    binding_domains: tuple[OperationBindingDomain, ...]
    contact_mapping_ids: tuple[str, ...]
    supported_direct_binding_ids: tuple[str, ...]
    rejected_control_runtime_keys: tuple[tuple[int, int], ...]
    selector_character_id: str | None
    selector_was_stale: bool


@dataclass(frozen=True, slots=True)
class OperationDomainResolution:
    snapshot: OperationDomainSnapshot | None
    issues: tuple[OperationDomainIssue, ...]

    @property
    def ok(self) -> bool:
        return self.snapshot is not None and not self.issues


def limb_domain_binding_ids(mapping: Any, capability: Any) -> tuple[str, ...]:
    """Return the animator-selectable binding footprint owned by one generated limb."""

    ids = [
        *tuple(str(binding_id) for binding_id in capability.fk_binding_ids),
        str(capability.authored_terminal_binding_id),
    ]
    if mapping.ik_target_binding_id:
        ids.append(str(mapping.ik_target_binding_id))
    if mapping.pole_binding_id:
        ids.append(str(mapping.pole_binding_id))
    return tuple(dict.fromkeys(binding_id for binding_id in ids if binding_id))


def _candidate_limb_domain_binding_ids(view: Any, mapping: Any) -> tuple[str, ...]:
    """Return topology-level animator binding candidates without requiring a valid solver."""

    chain = next(
        (
            item
            for item in getattr(view.definition, "chains", ())
            if item.chain_id == mapping.fk_chain_id
        ),
        None,
    )
    binding_by_id = {
        str(binding.binding_id): binding
        for binding in getattr(view.definition, "bindings", ())
    }
    ids: list[str] = []
    if chain is not None:
        ids.extend(str(binding_id) for binding_id in chain.members)

    ik_target_id = str(mapping.ik_target_binding_id or "")
    if ik_target_id:
        ids.append(ik_target_id)
    if mapping.pole_binding_id:
        ids.append(str(mapping.pole_binding_id))

    ik_binding = binding_by_id.get(ik_target_id)
    if ik_binding is not None:
        for binding in getattr(view.definition, "bindings", ()):
            usage = str(getattr(binding.usage, "value", binding.usage))
            kind = str(getattr(binding.kind, "value", binding.kind))
            if (
                binding.semantic_key == ik_binding.semantic_key
                and binding.side == ik_binding.side
                and usage == "PRIMARY"
                and kind == "BONE"
            ):
                ids.append(str(binding.binding_id))

    return tuple(dict.fromkeys(binding_id for binding_id in ids if binding_id))


def _supports_direct_domain(contract: RigpedControlContract) -> bool:
    capabilities = frozenset(contract.capabilities)
    return (
        RigpedCapability.ANIMATOR_SELECTABLE in capabilities
        and (
            RigpedCapability.DIRECT_MOVE in capabilities
            or RigpedCapability.DIRECT_ROTATE in capabilities
        )
    )


def _frame_identity(scene: Any) -> tuple[int, float]:
    return (
        int(getattr(scene, "frame_current", 0)),
        float(getattr(scene, "frame_subframe", 0.0)),
    )


def resolve_operation_domain(
    scene: Any,
    control_context: ControlContext,
    *,
    selector_character_id: str | None = None,
) -> OperationDomainResolution:
    """Freeze one read-only operation-domain classification from current native selection.

    The raw native selection identity is retained separately from Rigped binding IDs
    so a selected control cannot silently disappear during Character/Rigped resolution.
    This function never plans keys, cycles Contact state, runs a solver, or mutates
    Blender selection/animation data.
    """

    raw_selected_keys = tuple(runtime_control_key(control) for control in control_context.controls)
    raw_active_key = (
        runtime_control_key(control_context.active)
        if control_context.active is not None
        else None
    )

    rigped = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    target = rigped.target
    if target is None:
        return OperationDomainResolution(
            None,
            tuple(
                OperationDomainIssue(
                    code=str(issue.code),
                    detail=str(issue.detail),
                    binding_id=getattr(issue, "record_id", None),
                )
                for issue in rigped.issues
            ),
        )

    view = resolve_character(scene, target.character_id)
    contracts_by_runtime_key = {
        runtime_control_key(contract.target): contract
        for contract in target.controls
    }

    limb_owners_by_binding: dict[
        str,
        list[tuple[Any, Any | None, tuple[str, ...], tuple[str, ...]]],
    ] = {}
    for mapping in view.definition.kinematics:
        representation = resolve_limb_representation_capability(view, str(mapping.mapping_id))
        capability = representation.capability
        candidate_ids = _candidate_limb_domain_binding_ids(view, mapping)
        domain_ids = (
            limb_domain_binding_ids(mapping, capability)
            if capability is not None
            else candidate_ids
        )
        if capability is not None:
            candidate_ids = tuple(dict.fromkeys((*candidate_ids, *domain_ids)))
        issue_codes = tuple(str(issue.code) for issue in representation.issues)
        for binding_id in candidate_ids:
            limb_owners_by_binding.setdefault(binding_id, []).append(
                (mapping, capability, domain_ids, issue_codes)
            )

    domains: list[OperationBindingDomain] = []
    issues: list[OperationDomainIssue] = []
    contact_mapping_ids: list[str] = []
    direct_binding_ids: list[str] = []

    for selected in control_context.controls:
        selected_key = runtime_control_key(selected)
        contract = contracts_by_runtime_key.get(selected_key)
        if contract is None:
            issue = OperationDomainIssue(
                code="E1_SELECTED_CONTROL_OUTSIDE_RIGPED_TARGET",
                detail=(
                    "A native selected AWB control disappeared during Rigped binding "
                    "resolution and cannot be silently ignored."
                ),
                runtime_key=selected_key,
            )
            issues.append(issue)
            domains.append(
                OperationBindingDomain(
                    selected_control_runtime_key=selected_key,
                    selected_binding_id=None,
                    kind=OperationDomainKind.REJECTED,
                    rejection_code=issue.code,
                    detail=issue.detail,
                )
            )
            continue

        binding_id = str(contract.binding_id)
        owners = tuple(limb_owners_by_binding.get(binding_id, ()))
        if len(owners) > 1:
            issue = OperationDomainIssue(
                code="E1_AMBIGUOUS_LIMB_DOMAIN",
                detail=(
                    "Selected Rigped binding belongs to multiple generated limb "
                    "domain candidates."
                ),
                binding_id=binding_id,
                runtime_key=selected_key,
            )
            issues.append(issue)
            domains.append(
                OperationBindingDomain(
                    selected_control_runtime_key=selected_key,
                    selected_binding_id=binding_id,
                    kind=OperationDomainKind.REJECTED,
                    rejection_code=issue.code,
                    detail=issue.detail,
                )
            )
            continue

        if len(owners) == 1:
            mapping, capability, domain_ids, issue_codes = owners[0]
            mapping_id = str(mapping.mapping_id)
            if capability is None:
                detail = (
                    "Selected binding belongs to a generated limb mapping whose "
                    "representation capability failed closed"
                    + (f": {issue_codes!r}." if issue_codes else ".")
                )
                issue = OperationDomainIssue(
                    code="E1_UNSUPPORTED_LIMB_CAPABILITY",
                    detail=detail,
                    binding_id=binding_id,
                    runtime_key=selected_key,
                )
                issues.append(issue)
                domains.append(
                    OperationBindingDomain(
                        selected_control_runtime_key=selected_key,
                        selected_binding_id=binding_id,
                        kind=OperationDomainKind.REJECTED,
                        mapping_id=mapping_id,
                        domain_binding_ids=domain_ids,
                        rejection_code=issue.code,
                        detail=issue.detail,
                    )
                )
                continue

            if mapping_id not in contact_mapping_ids:
                contact_mapping_ids.append(mapping_id)
            domains.append(
                OperationBindingDomain(
                    selected_control_runtime_key=selected_key,
                    selected_binding_id=binding_id,
                    kind=OperationDomainKind.CONTACT_LIMB,
                    mapping_id=mapping_id,
                    domain_binding_ids=domain_ids,
                )
            )
            continue

        if _supports_direct_domain(contract):
            direct_binding_ids.append(binding_id)
            domains.append(
                OperationBindingDomain(
                    selected_control_runtime_key=selected_key,
                    selected_binding_id=binding_id,
                    kind=OperationDomainKind.DIRECT,
                    domain_binding_ids=(binding_id,),
                )
            )
            continue

        issue = OperationDomainIssue(
            code="E1_UNSUPPORTED_OPERATION_DOMAIN",
            detail=(
                "Selected Rigped binding is neither an unambiguous generated limb "
                "domain member nor an explicitly supported direct control."
            ),
            binding_id=binding_id,
            runtime_key=selected_key,
        )
        issues.append(issue)
        domains.append(
            OperationBindingDomain(
                selected_control_runtime_key=selected_key,
                selected_binding_id=binding_id,
                kind=OperationDomainKind.REJECTED,
                rejection_code=issue.code,
                detail=issue.detail,
            )
        )

    classified_binding_ids = tuple(
        domain.selected_binding_id
        for domain in domains
        if domain.selected_binding_id is not None
    )
    if set(classified_binding_ids) != set(target.selected_binding_ids):
        missing = tuple(
            binding_id
            for binding_id in target.selected_binding_ids
            if binding_id not in set(classified_binding_ids)
        )
        issue = OperationDomainIssue(
            code="E1_SELECTION_BINDING_MISMATCH",
            detail=(
                "Frozen Rigped selection and raw native selected-control identity "
                f"disagree; unmatched bindings={missing!r}."
            ),
        )
        issues.append(issue)

    frame, subframe = _frame_identity(scene)
    snapshot = OperationDomainSnapshot(
        character_id=target.character_id,
        setup_revision=int(target.descriptor.revision),
        setup_signature=str(target.descriptor.signature),
        character_source_stamp=tuple(target.source_stamp),
        frame=frame,
        subframe=subframe,
        selected_control_runtime_keys=raw_selected_keys,
        selected_binding_ids=tuple(target.selected_binding_ids),
        active_control_runtime_key=raw_active_key,
        active_binding_id=target.active_binding_id,
        binding_domains=tuple(domains),
        contact_mapping_ids=tuple(contact_mapping_ids),
        supported_direct_binding_ids=tuple(dict.fromkeys(direct_binding_ids)),
        rejected_control_runtime_keys=tuple(
            domain.selected_control_runtime_key
            for domain in domains
            if domain.kind is OperationDomainKind.REJECTED
        ),
        selector_character_id=target.selector_character_id,
        selector_was_stale=bool(target.selector_was_stale),
    )
    return OperationDomainResolution(snapshot, tuple(issues))
