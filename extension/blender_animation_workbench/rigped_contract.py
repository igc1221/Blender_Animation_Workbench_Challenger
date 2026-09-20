from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from json import dumps
from typing import TYPE_CHECKING, Any

from .character_model import CharacterBinding, CharacterSide, ControlMode, ControlUsage
from .character_query import CharacterSelectionSummary
from .semantic_adapter import ControlContext, ResolvedControl, runtime_control_key

if TYPE_CHECKING:
    from .character_metadata import ResolvedCharacter

RIGPED_SETUP_PROPERTY = "awb_rigped_setup"
RIGPED_SETUP_SCHEMA_VERSION = 1
RIGPED_PROFILE_HUMANOID_V1 = "awb.rigped.humanoid.v1"


class RigpedLifecycle(StrEnum):
    FITTED_UNBOUND = "FITTED_UNBOUND"
    BOUND = "BOUND"


class RigpedCapability(StrEnum):
    ANIMATOR_SELECTABLE = "ANIMATOR_SELECTABLE"
    KEY_POSITION = "KEY_POSITION"
    KEY_ROTATION = "KEY_ROTATION"
    DIRECT_MOVE = "DIRECT_MOVE"
    DIRECT_ROTATE = "DIRECT_ROTATE"
    KINEMATIC_MOVE = "KINEMATIC_MOVE"
    CONTACT_OWNER = "CONTACT_OWNER"
    IK_EFFECTOR = "IK_EFFECTOR"
    POLE_TARGET = "POLE_TARGET"
    FIT_STRUCTURAL = "FIT_STRUCTURAL"
    INTERNAL = "INTERNAL"


class RigpedIssueSeverity(StrEnum):
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class RigpedIssue:
    code: str
    severity: RigpedIssueSeverity
    character_id: str | None = None
    record_id: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class RigpedSetupDescriptor:
    schema_version: int
    character_id: str
    profile_id: str
    revision: int
    signature: str
    lifecycle: RigpedLifecycle


@dataclass(frozen=True, slots=True)
class RigpedControlContract:
    binding_id: str
    semantic_key: str
    side: CharacterSide
    authored_mode: ControlMode
    usage: ControlUsage
    capabilities: tuple[RigpedCapability, ...]
    target: ResolvedControl


@dataclass(frozen=True, slots=True)
class RigpedTarget:
    """Operation-local, read-only Rigped resolution from fresh native selection."""

    character_id: str
    descriptor: RigpedSetupDescriptor
    controls: tuple[RigpedControlContract, ...]
    selected_binding_ids: tuple[str, ...]
    active_binding_id: str | None
    source_stamp: tuple
    selector_character_id: str | None
    selector_was_stale: bool


@dataclass(frozen=True, slots=True)
class RigpedResolution:
    target: RigpedTarget | None
    issues: tuple[RigpedIssue, ...]
    selector_sync_character_id: str | None = None


class RigpedTransformGesture(StrEnum):
    MOVE = "MOVE"
    ROTATE = "ROTATE"
    SCALE = "SCALE"


class RigpedTransformRoute(StrEnum):
    NATIVE = "NATIVE"
    SEMANTIC_KINEMATIC = "SEMANTIC_KINEMATIC"
    SEMANTIC_CONTACT = "SEMANTIC_CONTACT"
    REFUSE = "REFUSE"


@dataclass(frozen=True, slots=True)
class RigpedTransformDecision:
    gesture: RigpedTransformGesture
    route: RigpedTransformRoute
    selected_binding_ids: tuple[str, ...]
    active_binding_id: str | None
    issues: tuple[RigpedIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return self.route != RigpedTransformRoute.REFUSE and not self.issues


_INTERNAL_USAGES = frozenset(
    {
        ControlUsage.TWIST,
        ControlUsage.HELPER,
        ControlUsage.DEFORM,
        ControlUsage.MECHANISM,
        ControlUsage.SPACE,
        ControlUsage.REFERENCE,
    }
)
_POSE_ROTATION_SEMANTICS = frozenset(
    {
        "awb.pelvis",
        "awb.spine",
        "awb.neck",
        "awb.head",
        "awb.clavicle",
        "awb.toe",
    }
)
_LIMB_SEMANTICS = frozenset(
    {
        "awb.arm",
        "awb.upper_arm",
        "awb.forearm",
        "awb.leg",
        "awb.thigh",
        "awb.calf",
    }
)
_TERMINAL_CONTACT_SEMANTICS = frozenset({"awb.hand", "awb.foot"})


def _issue(
    code: str,
    detail: str,
    *,
    severity: RigpedIssueSeverity = RigpedIssueSeverity.ERROR,
    character_id: str | None = None,
    record_id: str | None = None,
) -> RigpedIssue:
    return RigpedIssue(code, severity, character_id, record_id, detail)


def _has_errors(issues: tuple[RigpedIssue, ...] | list[RigpedIssue]) -> bool:
    return any(issue.severity == RigpedIssueSeverity.ERROR for issue in issues)


def _native_key(value: Any) -> tuple[str, int]:
    if value is None:
        return ("NONE", 0)
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        result = int(pointer())
        if result:
            return ("RNA", result)
    return ("PY", id(value))


def _float_token(value: Any) -> str:
    return float(value).hex()


def _numeric_sequence(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    if items and not isinstance(items[0], (int, float)):
        flattened: list[str] = []
        try:
            for row in items:
                flattened.extend(_float_token(component) for component in row)
        except (TypeError, ValueError):
            return None
        return tuple(flattened)
    try:
        return tuple(_float_token(component) for component in items)
    except (TypeError, ValueError):
        return None


def _resolved_maps(view: ResolvedCharacter):
    by_binding = {
        str(binding_id): resolved
        for binding_id, resolved in view.resolved_bindings
    }
    by_target = {
        _native_key(resolved.target): str(binding_id)
        for binding_id, resolved in view.resolved_bindings
    }
    object_targets = {
        _native_key(resolved.target): str(binding_id)
        for binding_id, resolved in view.resolved_bindings
        if resolved.target is resolved.owner_object
    }
    bone_targets = {
        (_native_key(resolved.owner_object), str(getattr(resolved.target, "name", ""))): str(binding_id)
        for binding_id, resolved in view.resolved_bindings
        if resolved.target is not resolved.owner_object
    }
    return by_binding, by_target, object_targets, bone_targets


def _constraint_target_binding(
    constraint: Any,
    *,
    object_targets: dict[tuple[str, int], str],
    bone_targets: dict[tuple[tuple[str, int], str], str],
    target_attr: str,
    subtarget_attr: str,
) -> str | None:
    target = getattr(constraint, target_attr, None)
    subtarget = str(getattr(constraint, subtarget_attr, "") or "")
    if target is None:
        return None
    if subtarget:
        return bone_targets.get((_native_key(target), subtarget), "EXTERNAL_BONE")
    return object_targets.get(_native_key(target), "EXTERNAL_OBJECT")


def _constraint_signature(
    resolved: ResolvedControl,
    *,
    object_targets: dict[tuple[str, int], str],
    bone_targets: dict[tuple[tuple[str, int], str], str],
) -> tuple[tuple, ...]:
    rows: list[tuple] = []
    for constraint in getattr(resolved.target, "constraints", ()) or ():
        row = (
            str(getattr(constraint, "type", "")),
            _constraint_target_binding(
                constraint,
                object_targets=object_targets,
                bone_targets=bone_targets,
                target_attr="target",
                subtarget_attr="subtarget",
            ),
            _constraint_target_binding(
                constraint,
                object_targets=object_targets,
                bone_targets=bone_targets,
                target_attr="pole_target",
                subtarget_attr="pole_subtarget",
            ),
            int(getattr(constraint, "chain_count", 0) or 0),
            bool(getattr(constraint, "use_tail", False)),
            bool(getattr(constraint, "use_stretch", False)),
            bool(getattr(constraint, "use_rotation", False)),
            str(getattr(constraint, "target_space", "")),
            str(getattr(constraint, "owner_space", "")),
            str(getattr(constraint, "mix_mode", "")),
            bool(getattr(constraint, "use_x", False)),
            bool(getattr(constraint, "use_y", False)),
            bool(getattr(constraint, "use_z", False)),
            bool(getattr(constraint, "invert_x", False)),
            bool(getattr(constraint, "invert_y", False)),
            bool(getattr(constraint, "invert_z", False)),
            str(getattr(constraint, "rotation_range", "")),
            _numeric_sequence(getattr(constraint, "offset", None)),
        )
        rows.append(row)
    return tuple(sorted(rows, key=repr))


def _native_structure_row(
    binding: CharacterBinding,
    resolved: ResolvedControl | None,
    *,
    by_target: dict[tuple[str, int], str],
    object_targets: dict[tuple[str, int], str],
    bone_targets: dict[tuple[tuple[str, int], str], str],
) -> tuple:
    if resolved is None:
        return (binding.binding_id, "UNRESOLVED")

    target = resolved.target
    parent = getattr(target, "parent", None)
    parent_binding_id = by_target.get(_native_key(parent)) if parent is not None else None
    bone = getattr(target, "bone", None)
    rest_matrix = _numeric_sequence(getattr(bone, "matrix_local", None)) if bone is not None else None
    head = _numeric_sequence(getattr(bone, "head_local", None)) if bone is not None else None
    tail = _numeric_sequence(getattr(bone, "tail_local", None)) if bone is not None else None
    bone_length = (
        _float_token(getattr(bone, "length", 0.0))
        if bone is not None and hasattr(bone, "length")
        else None
    )
    return (
        binding.binding_id,
        str(getattr(resolved.control.kind, "value", resolved.control.kind)),
        str(getattr(resolved.owner_object, "type", "")),
        parent_binding_id,
        rest_matrix,
        head,
        tail,
        bone_length,
        _constraint_signature(
            resolved,
            object_targets=object_targets,
            bone_targets=bone_targets,
        ),
    )


def compute_setup_signature(view: ResolvedCharacter) -> str:
    """Return a deterministic structural signature without using visible names.

    The signature intentionally excludes animation, current pose transforms,
    selection, UI/display state and label/name hints. It is safe to recompute
    during read-only preflight and survives ordinary Object/Bone renaming.
    """

    definition = view.definition
    by_binding, by_target, object_targets, bone_targets = _resolved_maps(view)
    payload = {
        "schema": RIGPED_SETUP_SCHEMA_VERSION,
        "character_id": definition.character_id,
        "bindings": tuple(
            (
                binding.binding_id,
                binding.owner_id,
                str(binding.kind.value),
                binding.bone_id,
                binding.semantic_key,
                binding.side.value,
                binding.mode.value,
                binding.usage.value,
            )
            for binding in definition.bindings
        ),
        "groups": tuple(
            (
                group.group_id,
                group.semantic_key,
                group.side.value,
                group.parent_group_id,
                tuple(group.members),
            )
            for group in definition.groups
        ),
        "chains": tuple(
            (
                chain.chain_id,
                chain.semantic_key,
                chain.side.value,
                chain.mode.value,
                tuple(chain.members),
            )
            for chain in definition.chains
        ),
        "opposites": tuple(
            (pair.left_binding_id, pair.right_binding_id)
            for pair in definition.opposites
        ),
        "kinematics": tuple(
            (
                mapping.mapping_id,
                mapping.semantic_key,
                mapping.side.value,
                mapping.fk_chain_id,
                mapping.ik_target_binding_id,
                mapping.pole_binding_id,
                mapping.reference_chain_id,
                tuple(mapping.extras),
            )
            for mapping in definition.kinematics
        ),
        "native": tuple(
            _native_structure_row(
                binding,
                by_binding.get(binding.binding_id),
                by_target=by_target,
                object_targets=object_targets,
                bone_targets=bone_targets,
            )
            for binding in definition.bindings
        ),
    }
    encoded = dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _mapping_value(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(key, default)
    getter = getattr(raw, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _root_binding(view: ResolvedCharacter) -> tuple[CharacterBinding | None, tuple[RigpedIssue, ...]]:
    matches = tuple(
        binding
        for binding in view.definition.bindings
        if binding.semantic_key == "awb.root" and binding.usage == ControlUsage.PRIMARY
    )
    character_id = view.definition.character_id
    if not matches:
        return None, (
            _issue(
                "MISSING_RIGPED_ROOT_BINDING",
                "Generated Rigped contract requires exactly one primary awb.root binding.",
                character_id=character_id,
            ),
        )
    if len(matches) > 1:
        return None, (
            _issue(
                "AMBIGUOUS_RIGPED_ROOT_BINDING",
                "Generated Rigped contract has multiple primary awb.root bindings.",
                character_id=character_id,
            ),
        )
    return matches[0], ()


def read_setup_descriptor(view: ResolvedCharacter) -> tuple[RigpedSetupDescriptor | None, tuple[RigpedIssue, ...]]:
    """Read and validate the persistent generated Rigped descriptor without repair."""

    root, issues = _root_binding(view)
    if root is None:
        return None, issues
    resolved = dict(view.resolved_bindings).get(root.binding_id)
    if resolved is None:
        return None, (
            _issue(
                "UNRESOLVED_RIGPED_ROOT",
                "The primary Rigped Root binding has no current native target.",
                character_id=view.definition.character_id,
                record_id=root.binding_id,
            ),
        )

    carrier = resolved.owner_object
    getter = getattr(carrier, "get", None)
    raw = getter(RIGPED_SETUP_PROPERTY) if callable(getter) else None
    if raw is None:
        return None, (
            _issue(
                "MISSING_RIGPED_SETUP_DESCRIPTOR",
                "The resolved Rigped owner has no persistent setup/rest descriptor.",
                character_id=view.definition.character_id,
                record_id=root.binding_id,
            ),
        )

    try:
        schema_version = int(_mapping_value(raw, "schema_version", 0) or 0)
        character_id = str(_mapping_value(raw, "character_id", "") or "")
        profile_id = str(_mapping_value(raw, "profile_id", "") or "")
        revision = int(_mapping_value(raw, "revision", 0) or 0)
        signature = str(_mapping_value(raw, "signature", "") or "")
        lifecycle = RigpedLifecycle(str(_mapping_value(raw, "lifecycle", "") or ""))
    except (TypeError, ValueError) as exc:
        return None, (
            _issue(
                "INVALID_RIGPED_SETUP_DESCRIPTOR",
                f"Rigped setup/rest descriptor cannot be decoded: {exc}",
                character_id=view.definition.character_id,
                record_id=root.binding_id,
            ),
        )

    validation: list[RigpedIssue] = []
    if schema_version != RIGPED_SETUP_SCHEMA_VERSION:
        validation.append(
            _issue(
                "UNSUPPORTED_RIGPED_SETUP_SCHEMA",
                f"Rigped setup schema {schema_version} is unsupported; expected {RIGPED_SETUP_SCHEMA_VERSION}.",
                character_id=view.definition.character_id,
            )
        )
    if character_id != view.definition.character_id:
        validation.append(
            _issue(
                "RIGPED_SETUP_CHARACTER_MISMATCH",
                "Rigped setup descriptor belongs to a different Character identity.",
                character_id=view.definition.character_id,
            )
        )
    if not profile_id:
        validation.append(
            _issue(
                "MISSING_RIGPED_PROFILE",
                "Rigped setup descriptor has no generator/profile identity.",
                character_id=view.definition.character_id,
            )
        )
    if revision < 1:
        validation.append(
            _issue(
                "INVALID_RIGPED_SETUP_REVISION",
                "Rigped setup/rest revision must be at least 1.",
                character_id=view.definition.character_id,
            )
        )
    expected_signature = compute_setup_signature(view)
    if not signature or signature != expected_signature:
        validation.append(
            _issue(
                "STALE_RIGPED_SETUP_DESCRIPTOR",
                "Persisted Rigped setup signature does not match current semantic/rest structure.",
                character_id=view.definition.character_id,
            )
        )
    if validation:
        return None, tuple(validation)

    return (
        RigpedSetupDescriptor(
            schema_version=schema_version,
            character_id=character_id,
            profile_id=profile_id,
            revision=revision,
            signature=signature,
            lifecycle=lifecycle,
        ),
        (),
    )


def capabilities_for_binding(binding: CharacterBinding) -> tuple[RigpedCapability, ...]:
    """Return the conservative I0 capability overlay for one generated binding."""

    if binding.usage in _INTERNAL_USAGES:
        return (RigpedCapability.INTERNAL,)
    if binding.usage == ControlUsage.POLE:
        return (
            RigpedCapability.ANIMATOR_SELECTABLE,
            RigpedCapability.KEY_POSITION,
            RigpedCapability.DIRECT_MOVE,
            RigpedCapability.POLE_TARGET,
        )
    if binding.usage == ControlUsage.TARGET:
        return (
            RigpedCapability.ANIMATOR_SELECTABLE,
            RigpedCapability.KEY_POSITION,
            RigpedCapability.DIRECT_MOVE,
            RigpedCapability.IK_EFFECTOR,
        )

    semantic_key = binding.semantic_key
    if semantic_key in {"awb.root", "awb.com"}:
        return (
            RigpedCapability.ANIMATOR_SELECTABLE,
            RigpedCapability.KEY_POSITION,
            RigpedCapability.KEY_ROTATION,
            RigpedCapability.DIRECT_MOVE,
            RigpedCapability.DIRECT_ROTATE,
        )
    if semantic_key in _POSE_ROTATION_SEMANTICS:
        return (
            RigpedCapability.ANIMATOR_SELECTABLE,
            RigpedCapability.KEY_ROTATION,
            RigpedCapability.DIRECT_ROTATE,
        )
    if semantic_key in _LIMB_SEMANTICS:
        return (
            RigpedCapability.ANIMATOR_SELECTABLE,
            RigpedCapability.KEY_ROTATION,
            RigpedCapability.DIRECT_ROTATE,
            RigpedCapability.KINEMATIC_MOVE,
        )
    if semantic_key in _TERMINAL_CONTACT_SEMANTICS:
        return (
            RigpedCapability.ANIMATOR_SELECTABLE,
            RigpedCapability.KEY_ROTATION,
            RigpedCapability.DIRECT_ROTATE,
            RigpedCapability.KINEMATIC_MOVE,
            RigpedCapability.CONTACT_OWNER,
        )
    return ()


def _transform_refusal(
    target: RigpedTarget,
    gesture: RigpedTransformGesture,
    code: str,
    detail: str,
    *,
    record_id: str | None = None,
) -> RigpedTransformDecision:
    return RigpedTransformDecision(
        gesture=gesture,
        route=RigpedTransformRoute.REFUSE,
        selected_binding_ids=target.selected_binding_ids,
        active_binding_id=target.active_binding_id,
        issues=(
            _issue(
                code,
                detail,
                character_id=target.character_id,
                record_id=record_id,
            ),
        ),
    )


def resolve_rigped_transform(
    target: RigpedTarget,
    gesture: RigpedTransformGesture,
) -> RigpedTransformDecision:
    """Resolve one I16 animator transform gesture without mutating Blender state.

    Blender-native direct transforms support normal Pose Mode multi-selection:
    the gizmo is anchored to the native active control while Blender applies the
    transform to the selected controls. AWB semantic/kinematic transforms remain
    single-control only so mixed or ambiguous multi-control solves still fail
    closed instead of falling back to raw transform channels.
    """

    if not target.selected_binding_ids:
        return _transform_refusal(
            target,
            gesture,
            "I16_NO_SELECTED_CONTROL",
            "Semantic transform requires a current native Rigped control selection.",
        )

    selected_contracts: list[RigpedControlContract] = []
    for binding_id in target.selected_binding_ids:
        contract = next(
            (control for control in target.controls if control.binding_id == binding_id),
            None,
        )
        if contract is None:
            return _transform_refusal(
                target,
                gesture,
                "I16_SELECTED_CONTROL_OUTSIDE_CONTRACT",
                "Selected Rigped binding has no current generated control contract.",
                record_id=binding_id,
            )
        capabilities = frozenset(contract.capabilities)
        if RigpedCapability.ANIMATOR_SELECTABLE not in capabilities:
            return _transform_refusal(
                target,
                gesture,
                "I16_CONTROL_NOT_ANIMATOR_SELECTABLE",
                "Internal or non-animator Rigped controls cannot own an animator transform gesture.",
                record_id=binding_id,
            )
        selected_contracts.append(contract)

    active_binding_id = target.active_binding_id
    if active_binding_id is None or active_binding_id not in target.selected_binding_ids:
        return _transform_refusal(
            target,
            gesture,
            "I16_ACTIVE_CONTROL_REQUIRED",
            "Rigped transform requires one of the selected controls to be the native active control.",
            record_id=active_binding_id,
        )

    if len(selected_contracts) > 1:
        if gesture == RigpedTransformGesture.MOVE and all(
            RigpedCapability.DIRECT_MOVE in frozenset(contract.capabilities)
            for contract in selected_contracts
        ):
            return RigpedTransformDecision(
                gesture=gesture,
                route=RigpedTransformRoute.NATIVE,
                selected_binding_ids=target.selected_binding_ids,
                active_binding_id=active_binding_id,
            )
        if gesture == RigpedTransformGesture.ROTATE and all(
            RigpedCapability.DIRECT_ROTATE in frozenset(contract.capabilities)
            for contract in selected_contracts
        ):
            return RigpedTransformDecision(
                gesture=gesture,
                route=RigpedTransformRoute.NATIVE,
                selected_binding_ids=target.selected_binding_ids,
                active_binding_id=active_binding_id,
            )
        return _transform_refusal(
            target,
            gesture,
            "I16_MULTI_SELECTION_UNPROVEN",
            "Multi-selection is supported only when every selected Rigped control owns the requested Blender-native direct transform.",
        )

    contract = selected_contracts[0]
    binding_id = contract.binding_id
    if active_binding_id != binding_id:
        return _transform_refusal(
            target,
            gesture,
            "I16_ACTIVE_CONTROL_REQUIRED",
            "Semantic transform requires the selected Rigped control to be the native active control.",
            record_id=binding_id,
        )

    capabilities = frozenset(contract.capabilities)

    if gesture == RigpedTransformGesture.MOVE:
        if RigpedCapability.DIRECT_MOVE in capabilities:
            route = RigpedTransformRoute.NATIVE
        elif RigpedCapability.KINEMATIC_MOVE in capabilities:
            route = (
                RigpedTransformRoute.SEMANTIC_CONTACT
                if RigpedCapability.CONTACT_OWNER in capabilities
                else RigpedTransformRoute.SEMANTIC_KINEMATIC
            )
        else:
            return _transform_refusal(
                target,
                gesture,
                "I16_MOVE_UNSUPPORTED",
                "Selected Rigped control does not authorize direct or semantic Move.",
                record_id=binding_id,
            )
    elif gesture == RigpedTransformGesture.ROTATE:
        if RigpedCapability.DIRECT_ROTATE not in capabilities:
            return _transform_refusal(
                target,
                gesture,
                "I16_ROTATE_UNSUPPORTED",
                "Selected Rigped control does not authorize direct Rotate.",
                record_id=binding_id,
            )
        route = RigpedTransformRoute.NATIVE
    else:
        return _transform_refusal(
            target,
            gesture,
            "I16_SCALE_UNSUPPORTED",
            "Generated Rigped animation controls do not authorize animated Scale in the current I0 contract.",
            record_id=binding_id,
        )

    return RigpedTransformDecision(
        gesture=gesture,
        route=route,
        selected_binding_ids=target.selected_binding_ids,
        active_binding_id=target.active_binding_id,
    )


def control_contracts_for_view(
    view: ResolvedCharacter,
) -> tuple[tuple[RigpedControlContract, ...], tuple[RigpedIssue, ...]]:
    resolved_by_binding = dict(view.resolved_bindings)
    contracts: list[RigpedControlContract] = []
    issues: list[RigpedIssue] = []
    for binding in view.definition.bindings:
        resolved = resolved_by_binding.get(binding.binding_id)
        if resolved is None:
            issues.append(
                _issue(
                    "UNRESOLVED_RIGPED_BINDING",
                    "Rigped binding has no resolved current native target.",
                    character_id=view.definition.character_id,
                    record_id=binding.binding_id,
                )
            )
            continue
        capabilities = capabilities_for_binding(binding)
        if not capabilities:
            issues.append(
                _issue(
                    "UNSUPPORTED_RIGPED_ROLE_CAPABILITY",
                    f"No I0 generated capability contract exists for semantic role {binding.semantic_key!r} / usage {binding.usage.value}.",
                    character_id=view.definition.character_id,
                    record_id=binding.binding_id,
                )
            )
            continue
        contracts.append(
            RigpedControlContract(
                binding_id=binding.binding_id,
                semantic_key=binding.semantic_key,
                side=binding.side,
                authored_mode=binding.mode,
                usage=binding.usage,
                capabilities=capabilities,
                target=resolved,
            )
        )
    return tuple(contracts), tuple(issues)


def select_character_id_for_context(
    summary: CharacterSelectionSummary,
    *,
    has_native_controls: bool,
) -> tuple[str | None, tuple[RigpedIssue, ...]]:
    """Select one Character only from native selection membership, never selector UI state."""

    if not has_native_controls:
        return None, (
            _issue(
                "NO_NATIVE_RIGPED_SELECTION",
                "Rigped operation target requires a current native Object/Pose control selection.",
            ),
        )
    if summary.unassigned:
        code = (
            "MIXED_RIGPED_AND_UNASSIGNED_SELECTION"
            if summary.character_ids
            else "UNASSIGNED_NATIVE_SELECTION"
        )
        return None, (
            _issue(
                code,
                "Current native selection contains controls outside one unambiguous Rigped Character.",
            ),
        )
    if not summary.character_ids:
        return None, (
            _issue(
                "NO_RIGPED_CHARACTER_FOR_SELECTION",
                "Current native selection does not resolve to a Character.",
            ),
        )
    if len(summary.character_ids) != 1:
        return None, (
            _issue(
                "AMBIGUOUS_RIGPED_TARGET",
                "Current native selection spans multiple Characters; one Rigped target is required.",
            ),
        )
    return summary.character_ids[0], ()


def _selected_binding_ids(view: ResolvedCharacter, control_context: ControlContext) -> tuple[str, ...]:
    selected_keys = {runtime_control_key(control) for control in control_context.controls}
    return tuple(
        binding_id
        for binding_id, resolved in view.resolved_bindings
        if runtime_control_key(resolved) in selected_keys
    )


def _active_binding_id(view: ResolvedCharacter, control_context: ControlContext) -> str | None:
    active = control_context.active
    if active is None:
        return None
    key = runtime_control_key(active)
    return next(
        (
            binding_id
            for binding_id, resolved in view.resolved_bindings
            if runtime_control_key(resolved) == key
        ),
        None,
    )


def resolve_rigped_view(
    view: ResolvedCharacter,
    *,
    selected_binding_ids: tuple[str, ...] = (),
    active_binding_id: str | None = None,
    selector_character_id: str | None = None,
) -> RigpedResolution:
    """Validate one already-resolved Character as a generated Rigped, read-only."""

    issues: list[RigpedIssue] = []
    for issue in view.issues:
        issues.append(
            _issue(
                issue.code,
                issue.detail,
                character_id=issue.character_id or view.definition.character_id,
                record_id=issue.record_id,
            )
        )
    descriptor, descriptor_issues = read_setup_descriptor(view)
    issues.extend(descriptor_issues)
    contracts, contract_issues = control_contracts_for_view(view)
    issues.extend(contract_issues)
    if descriptor is None or _has_errors(issues):
        return RigpedResolution(
            None,
            tuple(issues),
            selector_sync_character_id=view.definition.character_id,
        )

    selected_known = {contract.binding_id for contract in contracts}
    if any(binding_id not in selected_known for binding_id in selected_binding_ids):
        issues.append(
            _issue(
                "SELECTED_BINDING_OUTSIDE_RIGPED_CONTRACT",
                "A selected Character binding is outside the current generated Rigped capability contract.",
                character_id=view.definition.character_id,
            )
        )
    if active_binding_id is not None and active_binding_id not in selected_known:
        issues.append(
            _issue(
                "ACTIVE_BINDING_OUTSIDE_RIGPED_CONTRACT",
                "The active Character binding is outside the current generated Rigped capability contract.",
                character_id=view.definition.character_id,
                record_id=active_binding_id,
            )
        )
    if _has_errors(issues):
        return RigpedResolution(
            None,
            tuple(issues),
            selector_sync_character_id=view.definition.character_id,
        )

    target = RigpedTarget(
        character_id=view.definition.character_id,
        descriptor=descriptor,
        controls=contracts,
        selected_binding_ids=selected_binding_ids,
        active_binding_id=active_binding_id,
        source_stamp=view.source_stamp,
        selector_character_id=selector_character_id,
        selector_was_stale=(
            selector_character_id is not None
            and selector_character_id != view.definition.character_id
        ),
    )
    return RigpedResolution(
        target,
        tuple(issues),
        selector_sync_character_id=view.definition.character_id,
    )


def resolve_rigped_target(
    scene,
    control_context: ControlContext,
    *,
    selector_character_id: str | None = None,
) -> RigpedResolution:
    """Resolve the current native selection to one validated generated Rigped.

    `selector_character_id` is synchronization/UI intent only. It never selects
    the mutation target and therefore cannot override current native selection.
    """

    from .character_metadata import resolve_character
    from .character_query import characters_for_context

    summary = characters_for_context(scene, control_context)
    character_id, selection_issues = select_character_id_for_context(
        summary,
        has_native_controls=bool(control_context.controls),
    )
    if character_id is None:
        return RigpedResolution(None, selection_issues, selector_sync_character_id=None)

    view = resolve_character(scene, character_id)
    result = resolve_rigped_view(
        view,
        selected_binding_ids=_selected_binding_ids(view, control_context),
        active_binding_id=_active_binding_id(view, control_context),
        selector_character_id=selector_character_id,
    )
    if selection_issues:
        return RigpedResolution(
            result.target,
            selection_issues + result.issues,
            selector_sync_character_id=result.selector_sync_character_id,
        )
    return result
