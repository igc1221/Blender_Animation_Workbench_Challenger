import sys
from dataclasses import FrozenInstanceError
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_phase4_operation_plan_tests"

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
plan_model = _load("phase4_operation_plan")


def _channel(
    binding: str,
    family,
    path: str,
    index: int,
    *,
    owner: int = 1,
    control: int = 10,
    intent=None,
):
    intent = intent or plan_model.AllocationIntent.EXISTING_FCURVE
    rotation = family is plan_model.ChannelFamily.ROTATION
    return plan_model.PlannedChannel(
        row_key=plan_model.ChannelRowKey(binding, family, path, index),
        semantic_key=f"awb.{binding}",
        owner_runtime_key=owner,
        control_runtime_key=(owner, control),
        owner_binding_token=(owner, 2, 3, 4),
        rotation_mode="QUATERNION" if rotation else None,
        rotation_representation=(
            plan_model.RotationRepresentation.QUATERNION if rotation else None
        ),
        existing_fcurve_token=99 if intent is plan_model.AllocationIntent.EXISTING_FCURVE else None,
        allocation_intent=intent,
        target_value=0.0,
    )


def _plan(*channels):
    groups = ()
    if channels:
        groups = (
            plan_model.PlannedOwnerGroup(
                owner_runtime_key=1,
                owner_binding_token=(1, 2, 3, 4),
                channels=tuple(channels),
            ),
        )
    rows = tuple(channel.row_key for channel in channels)
    read = plan_model.ReadFootprint(
        character_source_stamp=("stamp",),
        setup_revision=1,
        setup_signature="sig",
        selected_binding_ids=("root",),
        active_binding_id="root",
        selection_runtime_keys=((1, 10),),
        active_runtime_key=(1, 10),
        owner_binding_tokens=((1, 2, 3, 4),) if channels else (),
        control_runtime_keys=((1, 10),) if channels else (),
    )
    return plan_model.OperationPlan(
        operation_id="op",
        operation_type=plan_model.OperationType.DIRECT_KEY,
        character_id="character",
        setup_revision=1,
        setup_signature="sig",
        character_source_stamp=("stamp",),
        frame=12,
        subframe=0.25,
        selected_binding_ids=("root",),
        active_binding_id="root",
        owner_groups=groups,
        read_footprint=read,
        dependency_footprint=plan_model.DependencyFootprint(("root",), read.owner_binding_tokens),
        write_footprint=plan_model.PlanWriteFootprint(rows, ()),
    )


def test_plan_records_are_frozen():
    channel = _channel("root", plan_model.ChannelFamily.POSITION, "location", 0)
    with pytest.raises(FrozenInstanceError):
        channel.target_value = 1.0


def test_channel_family_excludes_scale_by_contract():
    values = {family.value for family in plan_model.ChannelFamily}
    assert {"POSITION", "ROTATION"} <= values
    assert "SCALE" not in values


def test_structure_accepts_position_and_quaternion_rows():
    plan = _plan(
        _channel("root", plan_model.ChannelFamily.POSITION, "location", 0),
        _channel("root", plan_model.ChannelFamily.ROTATION, "rotation_quaternion", 3),
    )
    assert plan_model.validate_plan_structure(plan) == ()


def test_structure_rejects_duplicate_channel_rows():
    row = _channel("root", plan_model.ChannelFamily.POSITION, "location", 0)
    plan = _plan(row, row)
    codes = {issue.code for issue in plan_model.validate_plan_structure(plan)}
    assert "I3_DUPLICATE_CHANNEL_ROW" in codes


def test_structure_rejects_existing_fcurve_without_token():
    row = _channel("root", plan_model.ChannelFamily.POSITION, "location", 0)
    row = plan_model.PlannedChannel(
        row.row_key,
        row.semantic_key,
        row.owner_runtime_key,
        row.control_runtime_key,
        row.owner_binding_token,
        row.rotation_mode,
        row.rotation_representation,
        None,
        plan_model.AllocationIntent.EXISTING_FCURVE,
        row.target_value,
    )
    codes = {issue.code for issue in plan_model.validate_plan_structure(_plan(row))}
    assert "I3_EXISTING_FCURVE_TOKEN_MISSING" in codes


def test_freshness_rejects_selection_and_owner_binding_changes():
    plan = _plan(_channel("root", plan_model.ChannelFamily.POSITION, "location", 0))
    live = plan_model.LiveFreshnessFacts(
        character_source_stamp=("stamp",),
        setup_revision=1,
        setup_signature="sig",
        selected_binding_ids=("com",),
        active_binding_id="root",
        selection_runtime_keys=((1, 10),),
        active_runtime_key=(1, 10),
        owner_binding_tokens=((1, 9, 9, 9),),
        control_runtime_keys=((1, 10),),
        existing_fcurve_tokens=((plan.write_footprint.row_keys[0], 99),),
    )
    result = plan_model.validate_freshness(plan, live)
    codes = {issue.code for issue in result.diagnostics}
    assert result.ok is False
    assert "I3_SELECTION_CHANGED" in codes
    assert "I3_OWNER_BINDING_CHANGED" in codes


def test_freshness_rejects_existing_fcurve_identity_change():
    plan = _plan(_channel("root", plan_model.ChannelFamily.POSITION, "location", 0))
    live = plan_model.LiveFreshnessFacts(
        character_source_stamp=("stamp",),
        setup_revision=1,
        setup_signature="sig",
        selected_binding_ids=("root",),
        active_binding_id="root",
        selection_runtime_keys=((1, 10),),
        active_runtime_key=(1, 10),
        owner_binding_tokens=((1, 2, 3, 4),),
        control_runtime_keys=((1, 10),),
        existing_fcurve_tokens=((plan.write_footprint.row_keys[0], 100),),
    )
    result = plan_model.validate_freshness(plan, live)
    assert result.ok is False
    assert [issue.code for issue in result.diagnostics] == ["I3_FCURVE_BINDING_CHANGED"]


def test_kinematic_freshness_checks_all_runtime_participants():
    dependency = plan_model.KinematicDependency(
        mapping_id="map",
        driven_binding_ids=("upper", "lower"),
        ik_target_binding_id="ik",
        pole_binding_id="pole",
        solver_owner_binding_id="lower",
        constraint_runtime_token=77,
        solver_owner_runtime_key=(1, 20),
        ik_target_runtime_key=(1, 30),
        pole_runtime_key=(1, 40),
        chain_count=2,
        use_tail=True,
        use_stretch=False,
        use_rotation=False,
    )
    base = _plan()
    plan = plan_model.OperationPlan(
        operation_id="kin",
        operation_type=plan_model.OperationType.KINEMATIC_MOVE,
        character_id=base.character_id,
        setup_revision=base.setup_revision,
        setup_signature=base.setup_signature,
        character_source_stamp=base.character_source_stamp,
        frame=base.frame,
        subframe=base.subframe,
        selected_binding_ids=base.selected_binding_ids,
        active_binding_id=base.active_binding_id,
        owner_groups=(),
        read_footprint=base.read_footprint,
        dependency_footprint=plan_model.DependencyFootprint(("upper", "lower", "ik", "pole"), (), dependency),
        write_footprint=plan_model.PlanWriteFootprint((), ()),
        kinematic_dependency=dependency,
    )
    live = plan_model.LiveFreshnessFacts(
        character_source_stamp=("stamp",),
        setup_revision=1,
        setup_signature="sig",
        selected_binding_ids=("root",),
        active_binding_id="root",
        selection_runtime_keys=((1, 10),),
        active_runtime_key=(1, 10),
        owner_binding_tokens=((1, 2, 3, 4),),
        control_runtime_keys=((1, 10),),
        existing_fcurve_tokens=(),
        kinematic_constraint_token=78,
        kinematic_solver_owner_key=(1, 20),
        kinematic_target_key=(1, 30),
        kinematic_pole_key=(1, 40),
    )
    result = plan_model.validate_freshness(plan, live)
    assert result.ok is False
    assert "I3_KINEMATIC_CONSTRAINT_CHANGED" in {issue.code for issue in result.diagnostics}
