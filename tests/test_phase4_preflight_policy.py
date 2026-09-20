import sys
from enum import StrEnum
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import bpy

if not hasattr(bpy.types, "Operator"):
    bpy.types.Operator = object

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_phase4_preflight_policy_tests"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package


def _load(name: str):
    qualified = f"{PACKAGE_NAME}.{name}"
    spec = spec_from_file_location(qualified, PACKAGE_PATH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


verification = _load("phase4_verification")
operation_plan = _load("phase4_operation_plan")


class RigpedCapability(StrEnum):
    ANIMATOR_SELECTABLE = "ANIMATOR_SELECTABLE"
    INTERNAL = "INTERNAL"
    KEY_POSITION = "KEY_POSITION"
    KEY_ROTATION = "KEY_ROTATION"


_CURRENT_RESOLUTION = None


rigped_contract = ModuleType(f"{PACKAGE_NAME}.rigped_contract")
rigped_contract.RigpedCapability = RigpedCapability
rigped_contract.RIGPED_SETUP_PROPERTY = "awb_rigped_setup"
rigped_contract.resolve_rigped_target = lambda *_args, **_kwargs: _CURRENT_RESOLUTION
sys.modules[f"{PACKAGE_NAME}.rigped_contract"] = rigped_contract

character_metadata = ModuleType(f"{PACKAGE_NAME}.character_metadata")
character_metadata.BONE_TOKEN_PROPERTY = "awb_bone_token"
character_metadata.read_character = lambda *_args, **_kwargs: None
character_metadata.resolve_character = lambda *_args, **_kwargs: None
sys.modules[f"{PACKAGE_NAME}.character_metadata"] = character_metadata

kinematic_runtime = ModuleType(f"{PACKAGE_NAME}.kinematic_runtime")
kinematic_runtime.resolve_native_ik_capability = lambda *_args, **_kwargs: None
sys.modules[f"{PACKAGE_NAME}.kinematic_runtime"] = kinematic_runtime

semantic_adapter = ModuleType(f"{PACKAGE_NAME}.semantic_adapter")
semantic_adapter.ResolvedControl = object
semantic_adapter.assigned_channelbag = lambda owner: owner.bag
semantic_adapter.channel_binding_token = lambda owner: owner.binding_token
semantic_adapter.control_context_for_context = lambda *_args, **_kwargs: None
semantic_adapter.control_property_path = (
    lambda resolved, prop: f"{resolved.data_path_prefix}.{prop}"
    if resolved.data_path_prefix
    else prop
)
semantic_adapter.rotation_property = lambda target: (
    "rotation_quaternion"
    if target.rotation_mode == "QUATERNION"
    else "rotation_euler"
)
semantic_adapter.runtime_control_key = lambda resolved: resolved.runtime_key
sys.modules[f"{PACKAGE_NAME}.semantic_adapter"] = semantic_adapter

preflight = _load("phase4_preflight")


class FakeOwner:
    def __init__(self, pointer: int):
        self.pointer = pointer
        self.library = None
        self.is_editable = True
        self.animation_data = None
        self.bag = None
        self.binding_token = (pointer, None, None, None)

    def as_pointer(self):
        return self.pointer


class FakeTarget:
    def __init__(self, *, location=(0.0, 0.0, 0.0), rotation_mode="QUATERNION"):
        self.location = location
        self.rotation_mode = rotation_mode
        self.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        self.rotation_euler = (0.0, 0.0, 0.0)
        self.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)


class FakeResolved:
    def __init__(self, owner, target, runtime_key, prefix):
        self.owner_object = owner
        self.target = target
        self.runtime_key = runtime_key
        self.data_path_prefix = prefix


class FakeContract:
    def __init__(self, binding_id, semantic_key, capabilities, resolved):
        self.binding_id = binding_id
        self.semantic_key = semantic_key
        self.capabilities = tuple(capabilities)
        self.target = resolved


def _resolution(contracts, *, selected, active):
    descriptor = SimpleNamespace(revision=1, signature="sig")
    target = SimpleNamespace(
        character_id="char",
        descriptor=descriptor,
        controls=tuple(contracts),
        selected_binding_ids=tuple(selected),
        active_binding_id=active,
        source_stamp=("stamp", 1),
    )
    return SimpleNamespace(target=target, issues=())


def _scene():
    return SimpleNamespace(frame_current=12, frame_subframe=0.25)


def _control_context(*controls, active=None):
    return SimpleNamespace(
        controls=tuple(controls),
        active=active if active is not None else (controls[0] if controls else None),
    )


def _operation_domain(target, control_context, *, kinds):
    contracts_by_id = {
        str(contract.binding_id): contract
        for contract in target.controls
    }
    binding_domains = tuple(
        SimpleNamespace(
            selected_control_runtime_key=contracts_by_id[binding_id].target.runtime_key,
            selected_binding_id=binding_id,
            kind=kind,
        )
        for binding_id, kind in kinds.items()
    )
    direct_ids = tuple(
        binding_id
        for binding_id, kind in kinds.items()
        if kind == "DIRECT"
    )
    snapshot = SimpleNamespace(
        character_id=target.character_id,
        setup_revision=target.descriptor.revision,
        setup_signature=target.descriptor.signature,
        character_source_stamp=target.source_stamp,
        selected_control_runtime_keys=tuple(
            control.runtime_key for control in control_context.controls
        ),
        selected_binding_ids=target.selected_binding_ids,
        active_control_runtime_key=(
            control_context.active.runtime_key
            if control_context.active is not None
            else None
        ),
        active_binding_id=target.active_binding_id,
        binding_domains=binding_domains,
        supported_direct_binding_ids=direct_ids,
    )
    return SimpleNamespace(ok=True, snapshot=snapshot, issues=())


def test_pr_request_keeps_rotation_only_fk_control():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    fk = FakeContract(
        "fk",
        "awb.upper_arm",
        (RigpedCapability.KEY_ROTATION,),
        FakeResolved(owner, FakeTarget(), (1, 20), 'pose.bones["UpperArm.L"]'),
    )
    _CURRENT_RESOLUTION = _resolution((root, fk), selected=("root", "fk"), active="fk")

    result = preflight.build_direct_key_plan(
        _scene(),
        object(),
        operation_id="op",
        requested_families=(
            operation_plan.ChannelFamily.POSITION,
            operation_plan.ChannelFamily.ROTATION,
        ),
    )

    assert result.ok, result.diagnostics
    rows = tuple(channel for group in result.plan.owner_groups for channel in group.channels)
    fk_rows = tuple(row for row in rows if row.row_key.binding_id == "fk")
    assert fk_rows
    assert {row.row_key.family for row in fk_rows} == {operation_plan.ChannelFamily.ROTATION}
    assert len(fk_rows) == 4


def test_selected_control_without_requested_family_remains_in_selection_freshness():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_ROTATION,),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    position_only = FakeContract(
        "pole",
        "awb.arm",
        (RigpedCapability.KEY_POSITION,),
        FakeResolved(owner, FakeTarget(), (1, 30), 'pose.bones["IK_Elbow.L"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root, position_only),
        selected=("root", "pole"),
        active="root",
    )

    result = preflight.build_direct_key_plan(
        _scene(),
        object(),
        operation_id="op-selection",
        requested_families=(operation_plan.ChannelFamily.ROTATION,),
    )

    assert result.ok, result.diagnostics
    plan = result.plan
    assert plan.selected_binding_ids == ("root", "pole")
    assert plan.read_footprint.selection_runtime_keys == ((1, 10), (1, 30))
    assert plan.dependency_footprint.semantic_binding_ids == ("root", "pole")
    rows = tuple(channel for group in plan.owner_groups for channel in group.channels)
    assert {row.row_key.binding_id for row in rows} == {"root"}


def test_internal_selected_control_fails_closed():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    internal = FakeContract(
        "mch",
        "awb.arm.result",
        (RigpedCapability.INTERNAL,),
        FakeResolved(owner, FakeTarget(), (1, 50), 'pose.bones["MCH_UpperArm.L"]'),
    )
    _CURRENT_RESOLUTION = _resolution((internal,), selected=("mch",), active="mch")

    result = preflight.build_direct_key_plan(
        _scene(),
        object(),
        operation_id="op-internal",
    )

    assert result.plan is None
    assert [issue.code for issue in result.diagnostics] == ["I3_INTERNAL_CONTROL_SELECTED"]


def test_active_only_freezes_full_native_selection_but_targets_only_active_control():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    fk = FakeContract(
        "fk",
        "awb.upper_arm",
        (RigpedCapability.KEY_ROTATION,),
        FakeResolved(owner, FakeTarget(), (1, 20), 'pose.bones["UpperArm.L"]'),
    )
    _CURRENT_RESOLUTION = _resolution((root, fk), selected=("root", "fk"), active="root")

    result = preflight.build_direct_key_plan(
        _scene(),
        object(),
        operation_id="op-active",
        active_only=True,
    )

    assert result.ok, result.diagnostics
    plan = result.plan
    assert plan.selected_binding_ids == ("root", "fk")
    assert plan.read_footprint.selection_runtime_keys == ((1, 10), (1, 20))
    assert plan.dependency_footprint.semantic_binding_ids == ("root",)
    assert {row.row_key.binding_id for group in plan.owner_groups for row in group.channels} == {"root"}


def test_all_key_targets_whole_authored_rigped_not_only_native_selection():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (
            RigpedCapability.ANIMATOR_SELECTABLE,
            RigpedCapability.KEY_POSITION,
            RigpedCapability.KEY_ROTATION,
        ),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    fk = FakeContract(
        "fk",
        "awb.upper_arm",
        (RigpedCapability.ANIMATOR_SELECTABLE, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 20), 'pose.bones["UpperArm.L"]'),
    )
    pole = FakeContract(
        "pole",
        "awb.arm",
        (RigpedCapability.ANIMATOR_SELECTABLE, RigpedCapability.KEY_POSITION),
        FakeResolved(owner, FakeTarget(), (1, 30), 'pose.bones["IK_Elbow.L"]'),
    )
    internal = FakeContract(
        "mch",
        "awb.upper_arm",
        (RigpedCapability.INTERNAL,),
        FakeResolved(owner, FakeTarget(), (1, 40), 'pose.bones["MCH_UpperArm.L"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root, fk, pole, internal),
        selected=("root",),
        active="root",
    )

    result = preflight.build_all_key_plan(
        _scene(),
        object(),
        operation_id="all-key",
    )

    assert result.ok, result.diagnostics
    plan = result.plan
    assert plan.operation_type is operation_plan.OperationType.ALL_KEY
    assert plan.selected_binding_ids == ("root",)
    assert plan.read_footprint.selection_runtime_keys == ((1, 10),)
    assert plan.dependency_footprint.semantic_binding_ids == ("root", "fk", "pole")
    rows = tuple(channel for group in plan.owner_groups for channel in group.channels)
    assert {row.row_key.binding_id for row in rows} == {"root", "fk", "pole"}
    assert {row.row_key.family for row in rows if row.row_key.binding_id == "root"} == {
        operation_plan.ChannelFamily.POSITION,
        operation_plan.ChannelFamily.ROTATION,
    }
    assert {row.row_key.family for row in rows if row.row_key.binding_id == "fk"} == {
        operation_plan.ChannelFamily.ROTATION,
    }
    assert {row.row_key.family for row in rows if row.row_key.binding_id == "pole"} == {
        operation_plan.ChannelFamily.POSITION,
    }
    assert all(
        row.allocation_intent
        is operation_plan.AllocationIntent.NEED_ANIMATION_DATA_ACTION_SLOT_BAG
        for row in rows
    )


def test_all_key_fails_whole_plan_when_authored_rotation_is_not_replayable():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (
            RigpedCapability.ANIMATOR_SELECTABLE,
            RigpedCapability.KEY_POSITION,
            RigpedCapability.KEY_ROTATION,
        ),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    bad_rotation = FakeContract(
        "bad",
        "awb.upper_arm",
        (RigpedCapability.ANIMATOR_SELECTABLE, RigpedCapability.KEY_ROTATION),
        FakeResolved(
            owner,
            FakeTarget(rotation_mode="SWING_TWIST"),
            (1, 99),
            'pose.bones["Unsupported"]',
        ),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root, bad_rotation),
        selected=("root",),
        active="root",
    )

    result = preflight.build_all_key_plan(
        _scene(),
        object(),
        operation_id="all-key-bad-rotation",
    )

    assert result.plan is None
    assert [issue.code for issue in result.diagnostics] == ["I8_UNSUPPORTED_AUTHORED_ROTATION"]


def test_all_key_channel_model_has_no_scale_family():
    values = {family.value for family in operation_plan.ChannelFamily}
    assert {"POSITION", "ROTATION"} <= values
    assert "SCALE" not in values


def test_e2_explicit_direct_subset_keeps_full_selection_provenance():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    contact = FakeContract(
        "fk",
        "awb.upper_arm",
        (RigpedCapability.KEY_ROTATION,),
        FakeResolved(owner, FakeTarget(), (1, 20), 'pose.bones["UpperArm.L"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root, contact),
        selected=("root", "fk"),
        active="root",
    )
    target = _CURRENT_RESOLUTION.target
    context = _control_context(root.target, contact.target, active=root.target)
    domain = _operation_domain(
        target,
        context,
        kinds={"root": "DIRECT", "fk": "CONTACT_LIMB"},
    )

    result = preflight.build_direct_key_plan(
        _scene(),
        context,
        operation_id="e2-mixed",
        binding_ids=("root",),
        operation_domain=domain,
    )

    assert result.ok, result.diagnostics
    plan = result.plan
    assert plan.selected_binding_ids == ("root", "fk")
    assert plan.read_footprint.selection_runtime_keys == ((1, 10), (1, 20))
    assert plan.read_footprint.native_selection_runtime_keys == ((1, 10), (1, 20))
    assert plan.read_footprint.native_active_runtime_key == (1, 10)
    assert plan.dependency_footprint.semantic_binding_ids == ("root",)
    rows = tuple(
        channel
        for group in plan.owner_groups
        for channel in group.channels
    )
    assert {row.row_key.binding_id for row in rows} == {"root"}
    assert {
        row_key.binding_id
        for missing in plan.write_footprint.missing_allocations
        for row_key in missing.row_keys
    } == {"root"}


def test_e2_contact_subset_fails_before_row_or_allocation_planning(monkeypatch):
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    contact = FakeContract(
        "fk",
        "awb.upper_arm",
        (RigpedCapability.KEY_ROTATION,),
        FakeResolved(owner, FakeTarget(), (1, 20), 'pose.bones["UpperArm.L"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root, contact),
        selected=("root", "fk"),
        active="root",
    )
    target = _CURRENT_RESOLUTION.target
    context = _control_context(root.target, contact.target, active=root.target)
    domain = _operation_domain(
        target,
        context,
        kinds={"root": "DIRECT", "fk": "CONTACT_LIMB"},
    )

    monkeypatch.setattr(
        preflight,
        "_build_channel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("row construction must not run")
        ),
    )

    result = preflight.build_direct_key_plan(
        _scene(),
        context,
        operation_id="e2-contact-reject",
        binding_ids=("fk",),
        operation_domain=domain,
    )

    assert result.plan is None
    assert [issue.code for issue in result.diagnostics] == [
        "E2_DIRECT_SUBSET_NOT_DIRECT"
    ]


def test_e2_subset_outside_frozen_selection_fails_before_rows(monkeypatch):
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    other = FakeContract(
        "other",
        "awb.com",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 30), 'pose.bones["COM"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root, other),
        selected=("root",),
        active="root",
    )
    target = _CURRENT_RESOLUTION.target
    context = _control_context(root.target, active=root.target)
    domain = _operation_domain(target, context, kinds={"root": "DIRECT"})

    monkeypatch.setattr(
        preflight,
        "_build_channel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("row construction must not run")
        ),
    )

    result = preflight.build_direct_key_plan(
        _scene(),
        context,
        operation_id="e2-outside-selection",
        binding_ids=("other",),
        operation_domain=domain,
    )

    assert result.plan is None
    assert [issue.code for issue in result.diagnostics] == [
        "E2_DIRECT_SUBSET_OUTSIDE_SELECTION"
    ]


def test_e2_stale_raw_native_selection_fails_before_rows(monkeypatch):
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    contact = FakeContract(
        "fk",
        "awb.upper_arm",
        (RigpedCapability.KEY_ROTATION,),
        FakeResolved(owner, FakeTarget(), (1, 20), 'pose.bones["UpperArm.L"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root, contact),
        selected=("root", "fk"),
        active="root",
    )
    target = _CURRENT_RESOLUTION.target
    frozen_context = _control_context(root.target, contact.target, active=root.target)
    domain = _operation_domain(
        target,
        frozen_context,
        kinds={"root": "DIRECT", "fk": "CONTACT_LIMB"},
    )
    changed_native = FakeResolved(
        owner,
        FakeTarget(),
        (1, 99),
        'pose.bones["DifferentNativeSelection"]',
    )
    current_context = _control_context(
        root.target,
        changed_native,
        active=root.target,
    )

    monkeypatch.setattr(
        preflight,
        "_build_channel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("row construction must not run")
        ),
    )

    result = preflight.build_direct_key_plan(
        _scene(),
        current_context,
        operation_id="e2-stale-native-selection",
        binding_ids=("root",),
        operation_domain=domain,
    )

    assert result.plan is None
    assert [issue.code for issue in result.diagnostics] == [
        "E2_STALE_OPERATION_DOMAIN"
    ]


def test_e2_rebound_binding_id_fails_before_rows(monkeypatch):
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    old_root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (old_root,),
        selected=("root",),
        active="root",
    )
    frozen_target = _CURRENT_RESOLUTION.target
    frozen_context = _control_context(old_root.target, active=old_root.target)
    domain = _operation_domain(
        frozen_target,
        frozen_context,
        kinds={"root": "DIRECT"},
    )

    rebound_root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 77), 'pose.bones["RootRebound"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (rebound_root,),
        selected=("root",),
        active="root",
    )

    monkeypatch.setattr(
        preflight,
        "_build_channel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("row construction must not run")
        ),
    )

    result = preflight.build_direct_key_plan(
        _scene(),
        frozen_context,
        operation_id="e2-rebound",
        binding_ids=("root",),
        operation_domain=domain,
    )

    assert result.plan is None
    assert [issue.code for issue in result.diagnostics] == [
        "E2_DIRECT_SUBSET_REBOUND"
    ]


def test_e2_native_selection_identity_participates_in_freshness():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root,),
        selected=("root",),
        active="root",
    )
    target = _CURRENT_RESOLUTION.target
    frozen_context = _control_context(root.target, active=root.target)
    domain = _operation_domain(
        target,
        frozen_context,
        kinds={"root": "DIRECT"},
    )
    built = preflight.build_direct_key_plan(
        _scene(),
        frozen_context,
        operation_id="e2-freshness",
        binding_ids=("root",),
        operation_domain=domain,
    )
    assert built.ok, built.diagnostics

    changed_native = FakeResolved(
        owner,
        FakeTarget(),
        (1, 88),
        'pose.bones["ChangedNative"]',
    )
    changed_context = _control_context(changed_native, active=changed_native)

    freshness = preflight.validate_plan_fresh(
        _scene(),
        changed_context,
        built.plan,
    )

    assert freshness.ok is False
    assert "E2_NATIVE_SELECTION_RUNTIME_CHANGED" in {
        issue.code for issue in freshness.diagnostics
    }
    assert "E2_NATIVE_ACTIVE_RUNTIME_CHANGED" in {
        issue.code for issue in freshness.diagnostics
    }


def test_e2_native_active_none_to_control_fails_freshness():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root,),
        selected=("root",),
        active=None,
    )
    target = _CURRENT_RESOLUTION.target
    frozen_context = SimpleNamespace(controls=(root.target,), active=None)
    domain = _operation_domain(
        target,
        frozen_context,
        kinds={"root": "DIRECT"},
    )
    built = preflight.build_direct_key_plan(
        _scene(),
        frozen_context,
        operation_id="e2-native-active-none",
        binding_ids=("root",),
        operation_domain=domain,
    )
    assert built.ok, built.diagnostics
    assert built.plan.read_footprint.native_active_runtime_key is None

    changed_context = _control_context(root.target, active=root.target)
    freshness = preflight.validate_plan_fresh(
        _scene(),
        changed_context,
        built.plan,
    )

    assert freshness.ok is False
    assert "E2_NATIVE_ACTIVE_RUNTIME_CHANGED" in {
        issue.code for issue in freshness.diagnostics
    }


def test_e2_explicit_subset_rejects_active_only_second_authority():
    global _CURRENT_RESOLUTION
    owner = FakeOwner(1)
    root = FakeContract(
        "root",
        "awb.root",
        (RigpedCapability.KEY_POSITION, RigpedCapability.KEY_ROTATION),
        FakeResolved(owner, FakeTarget(), (1, 10), 'pose.bones["Root"]'),
    )
    _CURRENT_RESOLUTION = _resolution(
        (root,),
        selected=("root",),
        active="root",
    )
    context = _control_context(root.target, active=root.target)
    domain = _operation_domain(
        _CURRENT_RESOLUTION.target,
        context,
        kinds={"root": "DIRECT"},
    )

    result = preflight.build_direct_key_plan(
        _scene(),
        context,
        operation_id="e2-active-only-conflict",
        active_only=True,
        binding_ids=("root",),
        operation_domain=domain,
    )

    assert result.plan is None
    assert [issue.code for issue in result.diagnostics] == [
        "E2_DIRECT_SUBSET_ACTIVE_ONLY_CONFLICT"
    ]
