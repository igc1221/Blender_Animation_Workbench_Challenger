from __future__ import annotations

from collections import defaultdict
from typing import Any

from .character_metadata import resolve_character
from .kinematic_runtime import resolve_native_ik_capability
from .phase4_operation_plan import (
    AllocationIntent,
    ChannelFamily,
    ChannelRowKey,
    DependencyFootprint,
    FreshnessResult,
    KinematicDependency,
    LiveFreshnessFacts,
    MissingAllocation,
    OperationPlan,
    OperationType,
    PlanBuildResult,
    PlannedChannel,
    PlannedOwnerGroup,
    PlanWriteFootprint,
    ReadFootprint,
    RotationRepresentation,
    channel_sort_key,
    diagnostic,
    validate_freshness,
    validate_plan_structure,
)
from .rigped_animation_baseline import RIGPED_KEY_STATE_PROPERTY, RigpedKeyState
from .rigped_contract import RigpedCapability, resolve_rigped_target
from .semantic_adapter import (
    assigned_channelbag,
    channel_binding_token,
    control_property_path,
    rotation_property,
    runtime_control_key,
)

_EULER_MODES = frozenset({"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"})


def _runtime_pointer(value: Any) -> int | None:
    pointer = getattr(value, "as_pointer", None)
    if not callable(pointer):
        return None
    try:
        result = int(pointer())
    except ReferenceError:
        return None
    return result or None


def _owner_is_read_only(owner: Any) -> bool:
    return bool(
        getattr(owner, "library", None) is not None
        or getattr(owner, "is_editable", True) is False
    )


def _find_fcurve(channelbag: Any, data_path: str, array_index: int) -> Any | None:
    if channelbag is None:
        return None
    curves = getattr(channelbag, "fcurves", None)
    if curves is None:
        return None
    finder = getattr(curves, "find", None)
    if callable(finder):
        try:
            found = finder(data_path, index=array_index)
        except TypeError:
            found = None
        if found is not None:
            return found
    return next(
        (
            curve
            for curve in curves
            if str(getattr(curve, "data_path", "")) == data_path
            and int(getattr(curve, "array_index", -1)) == array_index
        ),
        None,
    )


def _base_allocation_intent(owner: Any, channelbag: Any) -> AllocationIntent:
    animation_data = getattr(owner, "animation_data", None)
    if animation_data is None:
        return AllocationIntent.NEED_ANIMATION_DATA_ACTION_SLOT_BAG
    if getattr(animation_data, "action", None) is None:
        return AllocationIntent.NEED_ACTION_AND_ASSIGNED_SLOT
    if channelbag is None:
        return AllocationIntent.NEED_ASSIGNED_BAG
    return AllocationIntent.NEED_FCURVE


def _rotation_descriptor(target: Any) -> tuple[str, RotationRepresentation, tuple[int, ...], Any, str]:
    mode = str(getattr(target, "rotation_mode", ""))
    property_name = rotation_property(target)
    if property_name == "rotation_quaternion" and mode == "QUATERNION":
        return property_name, RotationRepresentation.QUATERNION, (0, 1, 2, 3), target.rotation_quaternion, mode
    if property_name == "rotation_euler" and mode in _EULER_MODES:
        return property_name, RotationRepresentation.EULER, (0, 1, 2), target.rotation_euler, mode
    if property_name == "rotation_axis_angle" and mode == "AXIS_ANGLE":
        return property_name, RotationRepresentation.AXIS_ANGLE, (0, 1, 2, 3), target.rotation_axis_angle, mode
    raise ValueError(f"Unsupported Rigped rotation representation: mode={mode!r}, property={property_name!r}")


def _contracts_by_id(target) -> dict[str, Any]:
    return {contract.binding_id: contract for contract in target.controls}


def _selected_contracts(target, *, active_only: bool) -> tuple[Any, ...]:
    selected_ids = target.selected_binding_ids
    if active_only:
        selected_ids = (target.active_binding_id,) if target.active_binding_id is not None else ()
    by_id = _contracts_by_id(target)
    return tuple(by_id[binding_id] for binding_id in selected_ids if binding_id in by_id)


def _selected_runtime_keys(target, selected_binding_ids: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    by_id = _contracts_by_id(target)
    return tuple(
        runtime_control_key(by_id[binding_id].target)
        for binding_id in selected_binding_ids
        if binding_id in by_id
    )


def _active_runtime_key(target) -> tuple[int, int] | None:
    if target.active_binding_id is None:
        return None
    contract = _contracts_by_id(target).get(target.active_binding_id)
    return runtime_control_key(contract.target) if contract is not None else None


def _native_selection_runtime_keys(control_context) -> tuple[tuple[int, int], ...]:
    return tuple(
        runtime_control_key(control)
        for control in getattr(control_context, "controls", ())
    )


def _native_active_runtime_key(control_context) -> tuple[int, int] | None:
    active = getattr(control_context, "active", None)
    return runtime_control_key(active) if active is not None else None


def _explicit_direct_subset_contracts(
    target,
    control_context,
    binding_ids: tuple[str, ...],
    operation_domain,
) -> tuple[tuple[Any, ...] | None, tuple]:
    if operation_domain is None:
        return None, (
            diagnostic(
                "E2_OPERATION_DOMAIN_REQUIRED",
                "Explicit direct-subset planning requires one frozen E1 operation-domain resolution.",
            ),
        )

    snapshot = getattr(operation_domain, "snapshot", None)
    if not bool(getattr(operation_domain, "ok", False)) or snapshot is None:
        return None, (
            diagnostic(
                "E2_OPERATION_DOMAIN_INVALID",
                "Explicit direct-subset planning requires an issue-free frozen E1 operation domain.",
            ),
        )

    if not binding_ids:
        return None, (
            diagnostic(
                "E2_EMPTY_DIRECT_SUBSET",
                "Explicit direct-subset planning requires at least one direct binding.",
            ),
        )

    if len(set(binding_ids)) != len(binding_ids):
        return None, (
            diagnostic(
                "E2_DUPLICATE_DIRECT_SUBSET_BINDING",
                f"Explicit direct subset contains duplicate binding IDs: {binding_ids!r}",
            ),
        )

    identity_mismatches: list[str] = []
    if str(getattr(snapshot, "character_id", "")) != str(target.character_id):
        identity_mismatches.append("character")
    if int(getattr(snapshot, "setup_revision", -1)) != int(target.descriptor.revision):
        identity_mismatches.append("setup_revision")
    if str(getattr(snapshot, "setup_signature", "")) != str(target.descriptor.signature):
        identity_mismatches.append("setup_signature")
    if tuple(getattr(snapshot, "character_source_stamp", ())) != tuple(target.source_stamp):
        identity_mismatches.append("character_source_stamp")
    if tuple(getattr(snapshot, "selected_binding_ids", ())) != tuple(target.selected_binding_ids):
        identity_mismatches.append("selected_binding_ids")
    if getattr(snapshot, "active_binding_id", None) != target.active_binding_id:
        identity_mismatches.append("active_binding_id")
    if tuple(getattr(snapshot, "selected_control_runtime_keys", ())) != _native_selection_runtime_keys(control_context):
        identity_mismatches.append("native_selection_runtime_keys")
    if getattr(snapshot, "active_control_runtime_key", None) != _native_active_runtime_key(control_context):
        identity_mismatches.append("native_active_runtime_key")
    if identity_mismatches:
        return None, (
            diagnostic(
                "E2_STALE_OPERATION_DOMAIN",
                "Frozen E1 operation-domain identity no longer matches current planning context: "
                f"{tuple(identity_mismatches)!r}",
            ),
        )

    frozen_selected = tuple(str(binding_id) for binding_id in snapshot.selected_binding_ids)
    requested = tuple(str(binding_id) for binding_id in binding_ids)
    outside_selection = tuple(
        binding_id for binding_id in requested if binding_id not in frozen_selected
    )
    if outside_selection:
        return None, (
            diagnostic(
                "E2_DIRECT_SUBSET_OUTSIDE_SELECTION",
                "Explicit direct subset contains bindings outside the frozen native selection: "
                f"{outside_selection!r}",
            ),
        )

    contracts_by_id: dict[str, list[Any]] = defaultdict(list)
    for contract in target.controls:
        contracts_by_id[str(contract.binding_id)].append(contract)

    domains_by_id: dict[str, list[Any]] = defaultdict(list)
    for item in getattr(snapshot, "binding_domains", ()):
        binding_id = getattr(item, "selected_binding_id", None)
        if binding_id is not None:
            domains_by_id[str(binding_id)].append(item)

    contracts: list[Any] = []
    supported_direct = tuple(str(binding_id) for binding_id in snapshot.supported_direct_binding_ids)
    for binding_id in requested:
        current_contracts = contracts_by_id.get(binding_id, ())
        if len(current_contracts) != 1:
            return None, (
                diagnostic(
                    "E2_DIRECT_SUBSET_UNRESOLVED",
                    "Explicit direct subset binding does not resolve exactly once in the current Rigped: "
                    f"{binding_id!r}",
                ),
            )

        domains = domains_by_id.get(binding_id, ())
        if len(domains) != 1:
            return None, (
                diagnostic(
                    "E2_DIRECT_SUBSET_DOMAIN_UNRESOLVED",
                    "Frozen operation domain does not classify the requested binding exactly once: "
                    f"{binding_id!r}",
                ),
            )

        domain = domains[0]
        raw_kind = getattr(domain, "kind", "")
        kind = str(getattr(raw_kind, "value", raw_kind))
        if kind != "DIRECT" or binding_id not in supported_direct:
            return None, (
                diagnostic(
                    "E2_DIRECT_SUBSET_NOT_DIRECT",
                    "Explicit direct subset contains a Contact-owned, rejected, or unsupported binding: "
                    f"{binding_id!r}",
                ),
            )

        contract = current_contracts[0]
        if runtime_control_key(contract.target) != getattr(
            domain,
            "selected_control_runtime_key",
            None,
        ):
            return None, (
                diagnostic(
                    "E2_DIRECT_SUBSET_REBOUND",
                    "Requested direct binding now resolves to a different native control identity: "
                    f"{binding_id!r}",
                ),
            )
        contracts.append(contract)

    return tuple(contracts), ()


def _build_channel(
    contract,
    family: ChannelFamily,
    property_name: str,
    array_index: int,
    value: float,
    *,
    rotation_mode: str | None = None,
    rotation_representation: RotationRepresentation | None = None,
) -> PlannedChannel:
    resolved = contract.target
    owner = resolved.owner_object
    owner_key = _runtime_pointer(owner)
    if owner_key is None:
        raise ValueError("Resolved animation owner has no live runtime identity.")
    binding_token = channel_binding_token(owner)
    channelbag = assigned_channelbag(owner)
    data_path = control_property_path(resolved, property_name)
    fcurve = _find_fcurve(channelbag, data_path, array_index)
    if fcurve is not None:
        fcurve_token = _runtime_pointer(fcurve)
        if fcurve_token is None:
            raise ValueError("Existing FCurve has no live runtime identity.")
        allocation = AllocationIntent.EXISTING_FCURVE
    else:
        fcurve_token = None
        allocation = _base_allocation_intent(owner, channelbag)
    return PlannedChannel(
        row_key=ChannelRowKey(contract.binding_id, family, data_path, array_index),
        semantic_key=contract.semantic_key,
        owner_runtime_key=owner_key,
        control_runtime_key=runtime_control_key(resolved),
        owner_binding_token=binding_token,
        rotation_mode=rotation_mode,
        rotation_representation=rotation_representation,
        existing_fcurve_token=fcurve_token,
        allocation_intent=allocation,
        target_value=float(value),
    )


def _build_semantic_state_channel(contract, value: float) -> PlannedChannel:
    resolved = contract.target
    owner = resolved.owner_object
    owner_key = _runtime_pointer(owner)
    if owner_key is None:
        raise ValueError("Resolved animation owner has no live runtime identity.")
    binding_token = channel_binding_token(owner)
    channelbag = assigned_channelbag(owner)
    prefix = str(resolved.data_path_prefix or "")
    data_path = (
        f'{prefix}["{RIGPED_KEY_STATE_PROPERTY}"]'
        if prefix
        else f'["{RIGPED_KEY_STATE_PROPERTY}"]'
    )
    fcurve = _find_fcurve(channelbag, data_path, 0)
    if fcurve is not None:
        fcurve_token = _runtime_pointer(fcurve)
        if fcurve_token is None:
            raise ValueError("Existing semantic-state FCurve has no live runtime identity.")
        allocation = AllocationIntent.EXISTING_FCURVE
    else:
        fcurve_token = None
        allocation = _base_allocation_intent(owner, channelbag)
    return PlannedChannel(
        row_key=ChannelRowKey(
            contract.binding_id,
            ChannelFamily.SEMANTIC_STATE,
            data_path,
            0,
        ),
        semantic_key=contract.semantic_key,
        owner_runtime_key=owner_key,
        control_runtime_key=runtime_control_key(resolved),
        owner_binding_token=binding_token,
        rotation_mode=None,
        rotation_representation=None,
        existing_fcurve_token=fcurve_token,
        allocation_intent=allocation,
        target_value=float(value),
    )

def build_direct_key_plan(
    scene,
    control_context,
    *,
    operation_id: str,
    requested_families: tuple[ChannelFamily, ...] = (
        ChannelFamily.POSITION,
        ChannelFamily.ROTATION,
    ),
    selector_character_id: str | None = None,
    active_only: bool = False,
    include_rigped_free_marker: bool = False,
    binding_ids: tuple[str, ...] | None = None,
    operation_domain=None,
) -> PlanBuildResult:
    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return PlanBuildResult(
            None,
            tuple(
                diagnostic(issue.code, issue.detail)
                for issue in resolution.issues
            ),
        )
    target = resolution.target
    if binding_ids is not None and active_only:
        return PlanBuildResult(
            None,
            (
                diagnostic(
                    "E2_DIRECT_SUBSET_ACTIVE_ONLY_CONFLICT",
                    "Explicit direct-subset planning cannot also derive its target from active_only.",
                ),
            ),
        )

    if binding_ids is None:
        contracts = _selected_contracts(target, active_only=active_only)
    else:
        contracts, subset_issues = _explicit_direct_subset_contracts(
            target,
            control_context,
            tuple(binding_ids),
            operation_domain,
        )
        if contracts is None:
            return PlanBuildResult(None, tuple(subset_issues))

    if not contracts:
        return PlanBuildResult(None, (diagnostic("I3_NO_SELECTED_CONTROLS", "No selected authored Rigped controls are available for planning."),))
    internal_selected = tuple(
        contract.binding_id
        for contract in contracts
        if RigpedCapability.INTERNAL in contract.capabilities
    )
    if internal_selected:
        return PlanBuildResult(
            None,
            (
                diagnostic(
                    "I3_INTERNAL_CONTROL_SELECTED",
                    f"Mechanism/deform/internal Rigped bindings are not direct animation targets: {internal_selected!r}",
                ),
            ),
        )

    rows: list[PlannedChannel] = []
    for contract in contracts:
        owner = contract.target.owner_object
        if _owner_is_read_only(owner):
            return PlanBuildResult(
                None,
                (
                    diagnostic(
                        "I3_READ_ONLY_ANIMATION_OWNER",
                        f"Selected Rigped binding {contract.binding_id!r} has a linked/read-only animation owner.",
                    ),
                ),
            )
        capabilities = set(contract.capabilities)
        if (
            ChannelFamily.POSITION in requested_families
            and RigpedCapability.KEY_POSITION in capabilities
        ):
            for index, value in enumerate(contract.target.target.location):
                rows.append(
                    _build_channel(
                        contract,
                        ChannelFamily.POSITION,
                        "location",
                        index,
                        value,
                    )
                )
        if (
            ChannelFamily.ROTATION in requested_families
            and RigpedCapability.KEY_ROTATION in capabilities
        ):
            try:
                property_name, representation, indices, values, mode = _rotation_descriptor(
                    contract.target.target
                )
            except ValueError as exc:
                return PlanBuildResult(
                    None,
                    (diagnostic("I3_UNSUPPORTED_ROTATION_REPRESENTATION", str(exc)),),
                )
            for index in indices:
                rows.append(
                    _build_channel(
                        contract,
                        ChannelFamily.ROTATION,
                        property_name,
                        index,
                        values[index],
                        rotation_mode=mode,
                        rotation_representation=representation,
                    )
                )
        if include_rigped_free_marker:
            rows.append(
                _build_semantic_state_channel(
                    contract,
                    float(RigpedKeyState.FREE),
                )
            )

    if not rows:
        return PlanBuildResult(
            None,
            (
                diagnostic(
                    "I3_NO_KEYABLE_REQUESTED_CHANNELS",
                    "Selected Rigped controls expose none of the requested key channel families.",
                ),
            ),
        )

    rows.sort(key=channel_sort_key)
    grouped: dict[int, list[PlannedChannel]] = defaultdict(list)
    for channel in rows:
        grouped[channel.owner_runtime_key].append(channel)
    owner_groups = tuple(
        PlannedOwnerGroup(
            owner_runtime_key=owner_key,
            owner_binding_token=channels[0].owner_binding_token,
            channels=tuple(channels),
        )
        for owner_key, channels in sorted(grouped.items())
    )

    missing_allocations: list[MissingAllocation] = []
    for group in owner_groups:
        by_intent: dict[AllocationIntent, list[ChannelRowKey]] = defaultdict(list)
        for channel in group.channels:
            if channel.allocation_intent is not AllocationIntent.EXISTING_FCURVE:
                by_intent[channel.allocation_intent].append(channel.row_key)
        for intent, row_keys in sorted(by_intent.items(), key=lambda item: item[0].value):
            missing_allocations.append(
                MissingAllocation(
                    group.owner_runtime_key,
                    group.owner_binding_token,
                    intent,
                    tuple(sorted(row_keys)),
                )
            )

    selected_binding_ids = tuple(target.selected_binding_ids)
    selection_runtime_keys = _selected_runtime_keys(target, selected_binding_ids)
    owner_binding_tokens = tuple(group.owner_binding_token for group in owner_groups)
    control_runtime_keys = tuple(runtime_control_key(contract.target) for contract in contracts)
    native_selection_runtime_keys = (
        tuple(operation_domain.snapshot.selected_control_runtime_keys)
        if binding_ids is not None
        else ()
    )
    native_active_runtime_key = (
        operation_domain.snapshot.active_control_runtime_key
        if binding_ids is not None
        else None
    )
    read = ReadFootprint(
        character_source_stamp=target.source_stamp,
        setup_revision=target.descriptor.revision,
        setup_signature=target.descriptor.signature,
        selected_binding_ids=selected_binding_ids,
        active_binding_id=target.active_binding_id,
        selection_runtime_keys=selection_runtime_keys,
        active_runtime_key=_active_runtime_key(target),
        owner_binding_tokens=owner_binding_tokens,
        control_runtime_keys=control_runtime_keys,
        native_selection_runtime_keys=native_selection_runtime_keys,
        native_active_runtime_key=native_active_runtime_key,
    )
    write = PlanWriteFootprint(
        row_keys=tuple(channel.row_key for channel in rows),
        missing_allocations=tuple(missing_allocations),
    )
    plan = OperationPlan(
        operation_id=operation_id,
        operation_type=OperationType.DIRECT_KEY,
        character_id=target.character_id,
        setup_revision=target.descriptor.revision,
        setup_signature=target.descriptor.signature,
        character_source_stamp=target.source_stamp,
        frame=int(scene.frame_current),
        subframe=float(getattr(scene, "frame_subframe", 0.0)),
        selected_binding_ids=selected_binding_ids,
        active_binding_id=target.active_binding_id,
        owner_groups=owner_groups,
        read_footprint=read,
        dependency_footprint=DependencyFootprint(
            semantic_binding_ids=tuple(contract.binding_id for contract in contracts),
            owner_binding_tokens=owner_binding_tokens,
        ),
        write_footprint=write,
    )
    structure_issues = validate_plan_structure(plan)
    return PlanBuildResult(None, structure_issues) if structure_issues else PlanBuildResult(plan)


def build_all_key_plan(
    scene,
    control_context,
    *,
    operation_id: str,
    selector_character_id: str | None = None,
) -> PlanBuildResult:
    """Plan a complete direct-control P/R pose anchor for the current Rigped.

    Native selection identifies the current Character and remains part of the
    freshness contract, but it does not limit the authored controls included in
    the write footprint. Contact/kinematic hidden dependency closure is added by
    the owning later feature when those states become public; this I8 slice is
    deliberately the direct authored-control closure only.
    """

    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return PlanBuildResult(
            None,
            tuple(diagnostic(issue.code, issue.detail) for issue in resolution.issues),
        )
    target = resolution.target
    authored_contracts = tuple(
        contract
        for contract in target.controls
        if RigpedCapability.INTERNAL not in contract.capabilities
        and RigpedCapability.ANIMATOR_SELECTABLE in contract.capabilities
    )
    if not authored_contracts:
        return PlanBuildResult(
            None,
            (
                diagnostic(
                    "I8_NO_AUTHORED_CONTROLS",
                    "Current Rigped exposes no authored animation controls for All Key.",
                ),
            ),
        )

    rows: list[PlannedChannel] = []
    for contract in authored_contracts:
        owner = contract.target.owner_object
        if _owner_is_read_only(owner):
            return PlanBuildResult(
                None,
                (
                    diagnostic(
                        "I8_READ_ONLY_ANIMATION_OWNER",
                        f"Rigped binding {contract.binding_id!r} has a linked/read-only animation owner.",
                    ),
                ),
            )
        capabilities = set(contract.capabilities)
        if RigpedCapability.KEY_POSITION in capabilities:
            for index, value in enumerate(contract.target.target.location):
                rows.append(
                    _build_channel(
                        contract,
                        ChannelFamily.POSITION,
                        "location",
                        index,
                        value,
                    )
                )
        if RigpedCapability.KEY_ROTATION in capabilities:
            try:
                property_name, representation, indices, values, mode = _rotation_descriptor(
                    contract.target.target
                )
            except ValueError as exc:
                return PlanBuildResult(
                    None,
                    (
                        diagnostic(
                            "I8_UNSUPPORTED_AUTHORED_ROTATION",
                            f"All Key requires complete replayable rotation closure: {exc}",
                        ),
                    ),
                )
            for index in indices:
                rows.append(
                    _build_channel(
                        contract,
                        ChannelFamily.ROTATION,
                        property_name,
                        index,
                        values[index],
                        rotation_mode=mode,
                        rotation_representation=representation,
                    )
                )

    if not rows:
        return PlanBuildResult(
            None,
            (
                diagnostic(
                    "I8_EMPTY_DIRECT_CLOSURE",
                    "Current Rigped authored controls expose no supported Position/Rotation closure.",
                ),
            ),
        )

    rows.sort(key=channel_sort_key)
    grouped: dict[int, list[PlannedChannel]] = defaultdict(list)
    for channel in rows:
        grouped[channel.owner_runtime_key].append(channel)
    owner_groups = tuple(
        PlannedOwnerGroup(
            owner_runtime_key=owner_key,
            owner_binding_token=channels[0].owner_binding_token,
            channels=tuple(channels),
        )
        for owner_key, channels in sorted(grouped.items())
    )

    missing_allocations: list[MissingAllocation] = []
    for group in owner_groups:
        by_intent: dict[AllocationIntent, list[ChannelRowKey]] = defaultdict(list)
        for channel in group.channels:
            if channel.allocation_intent is not AllocationIntent.EXISTING_FCURVE:
                by_intent[channel.allocation_intent].append(channel.row_key)
        for intent, row_keys in sorted(by_intent.items(), key=lambda item: item[0].value):
            missing_allocations.append(
                MissingAllocation(
                    group.owner_runtime_key,
                    group.owner_binding_token,
                    intent,
                    tuple(sorted(row_keys)),
                )
            )

    selected_binding_ids = tuple(target.selected_binding_ids)
    selection_runtime_keys = _selected_runtime_keys(target, selected_binding_ids)
    owner_binding_tokens = tuple(group.owner_binding_token for group in owner_groups)
    authored_binding_ids = tuple(contract.binding_id for contract in authored_contracts)
    read = ReadFootprint(
        character_source_stamp=target.source_stamp,
        setup_revision=target.descriptor.revision,
        setup_signature=target.descriptor.signature,
        selected_binding_ids=selected_binding_ids,
        active_binding_id=target.active_binding_id,
        selection_runtime_keys=selection_runtime_keys,
        active_runtime_key=_active_runtime_key(target),
        owner_binding_tokens=owner_binding_tokens,
        control_runtime_keys=tuple(
            runtime_control_key(contract.target) for contract in authored_contracts
        ),
    )
    write = PlanWriteFootprint(
        row_keys=tuple(channel.row_key for channel in rows),
        missing_allocations=tuple(missing_allocations),
    )
    plan = OperationPlan(
        operation_id=operation_id,
        operation_type=OperationType.ALL_KEY,
        character_id=target.character_id,
        setup_revision=target.descriptor.revision,
        setup_signature=target.descriptor.signature,
        character_source_stamp=target.source_stamp,
        frame=int(scene.frame_current),
        subframe=float(getattr(scene, "frame_subframe", 0.0)),
        selected_binding_ids=selected_binding_ids,
        active_binding_id=target.active_binding_id,
        owner_groups=owner_groups,
        read_footprint=read,
        dependency_footprint=DependencyFootprint(
            semantic_binding_ids=authored_binding_ids,
            owner_binding_tokens=owner_binding_tokens,
        ),
        write_footprint=write,
    )
    structure_issues = validate_plan_structure(plan)
    return PlanBuildResult(None, structure_issues) if structure_issues else PlanBuildResult(plan)


def _binding_id_for_runtime_key(target, key: tuple[int, int]) -> str | None:
    return next(
        (
            contract.binding_id
            for contract in target.controls
            if runtime_control_key(contract.target) == key
        ),
        None,
    )


def build_kinematic_dependency_plan(
    scene,
    control_context,
    *,
    operation_id: str,
    mapping_id: str,
    selector_character_id: str | None = None,
) -> PlanBuildResult:
    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return PlanBuildResult(
            None,
            tuple(diagnostic(issue.code, issue.detail) for issue in resolution.issues),
        )
    target = resolution.target
    view = resolve_character(scene, target.character_id)
    capability_resolution = resolve_native_ik_capability(view, mapping_id)
    if capability_resolution.capability is None:
        detail = "; ".join(issue.detail for issue in capability_resolution.issues) or mapping_id
        return PlanBuildResult(
            None,
            (diagnostic("I3_KINEMATIC_CAPABILITY_UNRESOLVED", detail),),
        )
    capability = capability_resolution.capability
    solver_key = runtime_control_key(capability.solver_owner)
    target_key = runtime_control_key(capability.ik_target)
    pole_key = runtime_control_key(capability.pole_target) if capability.pole_target is not None else None
    solver_binding = _binding_id_for_runtime_key(target, solver_key)
    ik_binding = _binding_id_for_runtime_key(target, target_key)
    pole_binding = _binding_id_for_runtime_key(target, pole_key) if pole_key is not None else None
    constraint_token = _runtime_pointer(capability.constraint)
    if solver_binding is None or ik_binding is None or constraint_token is None:
        return PlanBuildResult(
            None,
            (
                diagnostic(
                    "I3_KINEMATIC_RUNTIME_IDENTITY_INCOMPLETE",
                    "Resolved kinematic capability cannot be mapped completely to the current Rigped contract.",
                ),
            ),
        )
    dependency = KinematicDependency(
        mapping_id=mapping_id,
        driven_binding_ids=tuple(capability.driven_binding_ids),
        ik_target_binding_id=ik_binding,
        pole_binding_id=pole_binding,
        solver_owner_binding_id=solver_binding,
        constraint_runtime_token=constraint_token,
        solver_owner_runtime_key=solver_key,
        ik_target_runtime_key=target_key,
        pole_runtime_key=pole_key,
        chain_count=int(getattr(capability.constraint, "chain_count", 0)),
        use_tail=bool(getattr(capability.constraint, "use_tail", False)),
        use_stretch=bool(getattr(capability.constraint, "use_stretch", False)),
        use_rotation=bool(getattr(capability.constraint, "use_rotation", False)),
    )
    selected_runtime_keys = _selected_runtime_keys(target, target.selected_binding_ids)
    semantic_binding_ids = tuple(
        dict.fromkeys(
            (*dependency.driven_binding_ids, solver_binding, ik_binding, *((pole_binding,) if pole_binding else ()))
        )
    )
    by_id = _contracts_by_id(target)
    read = ReadFootprint(
        character_source_stamp=target.source_stamp,
        setup_revision=target.descriptor.revision,
        setup_signature=target.descriptor.signature,
        selected_binding_ids=target.selected_binding_ids,
        active_binding_id=target.active_binding_id,
        selection_runtime_keys=selected_runtime_keys,
        active_runtime_key=_active_runtime_key(target),
        owner_binding_tokens=(),
        control_runtime_keys=tuple(
            runtime_control_key(by_id[binding_id].target)
            for binding_id in semantic_binding_ids
            if binding_id in by_id
        ),
    )
    plan = OperationPlan(
        operation_id=operation_id,
        operation_type=OperationType.KINEMATIC_MOVE,
        character_id=target.character_id,
        setup_revision=target.descriptor.revision,
        setup_signature=target.descriptor.signature,
        character_source_stamp=target.source_stamp,
        frame=int(scene.frame_current),
        subframe=float(getattr(scene, "frame_subframe", 0.0)),
        selected_binding_ids=target.selected_binding_ids,
        active_binding_id=target.active_binding_id,
        owner_groups=(),
        read_footprint=read,
        dependency_footprint=DependencyFootprint(
            semantic_binding_ids=semantic_binding_ids,
            owner_binding_tokens=(),
            kinematic=dependency,
        ),
        write_footprint=PlanWriteFootprint((), ()),
        kinematic_dependency=dependency,
    )
    return PlanBuildResult(plan)


def collect_live_freshness_facts(
    scene,
    control_context,
    plan: OperationPlan,
    *,
    selector_character_id: str | None = None,
) -> tuple[LiveFreshnessFacts | None, tuple]:
    resolution = resolve_rigped_target(
        scene,
        control_context,
        selector_character_id=selector_character_id,
    )
    if resolution.target is None:
        return None, tuple(diagnostic(issue.code, issue.detail) for issue in resolution.issues)
    target = resolution.target
    if target.character_id != plan.character_id:
        return None, (
            diagnostic("I3_CHARACTER_TARGET_CHANGED", "Native selection now resolves to a different Character."),
        )
    by_id = _contracts_by_id(target)
    selected_runtime = _selected_runtime_keys(target, target.selected_binding_ids)
    active_runtime = _active_runtime_key(target)

    owner_tokens: list[tuple[int | None, int | None, int | None, int | None]] = []
    for planned_group in plan.owner_groups:
        contract = next(
            (
                candidate
                for candidate in target.controls
                if _runtime_pointer(candidate.target.owner_object) == planned_group.owner_runtime_key
            ),
            None,
        )
        if contract is None:
            owner_tokens.append((None, None, None, None))
        else:
            owner_tokens.append(channel_binding_token(contract.target.owner_object))

    control_runtime_keys = tuple(
        runtime_control_key(by_id[binding_id].target)
        for binding_id in plan.dependency_footprint.semantic_binding_ids
        if binding_id in by_id
    )
    fcurve_tokens: list[tuple[ChannelRowKey, int | None]] = []
    for group in plan.owner_groups:
        for channel in group.channels:
            if channel.allocation_intent is not AllocationIntent.EXISTING_FCURVE:
                continue
            contract = by_id.get(channel.row_key.binding_id)
            token = None
            if contract is not None:
                bag = assigned_channelbag(contract.target.owner_object)
                curve = _find_fcurve(
                    bag,
                    channel.row_key.data_path,
                    channel.row_key.array_index,
                )
                token = _runtime_pointer(curve)
            fcurve_tokens.append((channel.row_key, token))

    constraint_token = None
    solver_key = None
    ik_key = None
    pole_key = None
    if plan.kinematic_dependency is not None:
        view = resolve_character(scene, target.character_id)
        capability_resolution = resolve_native_ik_capability(
            view,
            plan.kinematic_dependency.mapping_id,
        )
        capability = capability_resolution.capability
        if capability is not None:
            constraint_token = _runtime_pointer(capability.constraint)
            solver_key = runtime_control_key(capability.solver_owner)
            ik_key = runtime_control_key(capability.ik_target)
            pole_key = runtime_control_key(capability.pole_target) if capability.pole_target is not None else None

    return (
        LiveFreshnessFacts(
            character_source_stamp=target.source_stamp,
            setup_revision=target.descriptor.revision,
            setup_signature=target.descriptor.signature,
            selected_binding_ids=target.selected_binding_ids,
            active_binding_id=target.active_binding_id,
            selection_runtime_keys=selected_runtime,
            active_runtime_key=active_runtime,
            owner_binding_tokens=tuple(owner_tokens),
            control_runtime_keys=control_runtime_keys,
            existing_fcurve_tokens=tuple(fcurve_tokens),
            native_selection_runtime_keys=_native_selection_runtime_keys(control_context),
            native_active_runtime_key=_native_active_runtime_key(control_context),
            kinematic_constraint_token=constraint_token,
            kinematic_solver_owner_key=solver_key,
            kinematic_target_key=ik_key,
            kinematic_pole_key=pole_key,
        ),
        (),
    )


def validate_plan_fresh(
    scene,
    control_context,
    plan: OperationPlan,
    *,
    selector_character_id: str | None = None,
) -> FreshnessResult:
    facts, issues = collect_live_freshness_facts(
        scene,
        control_context,
        plan,
        selector_character_id=selector_character_id,
    )
    if facts is None:
        return FreshnessResult(False, tuple(issues))
    return validate_freshness(plan, facts)
