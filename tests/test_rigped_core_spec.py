import sys
from dataclasses import FrozenInstanceError, replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_core_spec_tests"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package


def _load_module(name: str):
    qualified_name = f"{PACKAGE_NAME}.{name}"
    spec = spec_from_file_location(qualified_name, PACKAGE_PATH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


semantic_model = _load_module("semantic_model")
semantic_adapter = _load_module("semantic_adapter")
character_model = _load_module("character_model")
character_query = _load_module("character_query")
rigped_contract = _load_module("rigped_contract")
rigped_core_spec = _load_module("rigped_core_spec")


def test_default_core_spec_is_valid_and_frozen():
    spec = rigped_core_spec.RigpedCoreSpec()
    spec.validate()
    assert len(spec.resolved_topology()) == len(rigped_core_spec.GeneratedRole)
    with pytest.raises(FrozenInstanceError):
        spec.profile_id = "changed"


def test_core_physically_separates_authored_mechanism_deform_and_export_layers():
    topology = rigped_core_spec.RigpedCoreSpec().resolved_topology()
    by_role = {entry.role: entry for entry in topology}

    assert by_role[rigped_core_spec.GeneratedRole.FK_UPPER].layer == rigped_core_spec.GeneratedLayer.AUTHORED
    assert by_role[rigped_core_spec.GeneratedRole.MCH_UPPER].layer == rigped_core_spec.GeneratedLayer.MECHANISM
    assert by_role[rigped_core_spec.GeneratedRole.DEF_UPPER].layer == rigped_core_spec.GeneratedLayer.DEFORM
    assert by_role[rigped_core_spec.GeneratedRole.EXPORT_ROOT].layer == rigped_core_spec.GeneratedLayer.EXPORT
    assert len(
        {
            by_role[rigped_core_spec.GeneratedRole.FK_UPPER].name,
            by_role[rigped_core_spec.GeneratedRole.MCH_UPPER].name,
            by_role[rigped_core_spec.GeneratedRole.DEF_UPPER].name,
        }
    ) == 3


def test_all_generated_semantic_keys_are_character_canonical():
    topology = rigped_core_spec.RigpedCoreSpec().resolved_topology()
    assert all(
        character_model.semantic_key_error(entry.semantic_key, allow_empty=False) is None
        for entry in topology
    )


def test_only_deform_layer_bones_are_deforming():
    topology = rigped_core_spec.RigpedCoreSpec().resolved_topology()
    assert {entry.layer for entry in topology if entry.deform} == {
        rigped_core_spec.GeneratedLayer.DEFORM
    }
    assert all(entry.deform for entry in topology if entry.layer == rigped_core_spec.GeneratedLayer.DEFORM)


def test_root_com_pelvis_and_limb_authored_roles_are_not_collapsed():
    topology = rigped_core_spec.RigpedCoreSpec().resolved_topology()
    authored = {
        entry.role
        for entry in topology
        if entry.layer == rigped_core_spec.GeneratedLayer.AUTHORED
    }
    assert {
        rigped_core_spec.GeneratedRole.ROOT,
        rigped_core_spec.GeneratedRole.COM,
        rigped_core_spec.GeneratedRole.PELVIS,
        rigped_core_spec.GeneratedRole.FK_UPPER,
        rigped_core_spec.GeneratedRole.FK_FORE,
        rigped_core_spec.GeneratedRole.FK_HAND,
        rigped_core_spec.GeneratedRole.IK_EFFECTOR,
        rigped_core_spec.GeneratedRole.IK_POLE,
    } <= authored


def test_invalid_duplicate_name_is_rejected():
    topology = list(rigped_core_spec.RigpedCoreSpec().resolved_topology())
    topology[1] = replace(topology[1], name=topology[0].name)
    spec = rigped_core_spec.RigpedCoreSpec(topology=tuple(topology))
    with pytest.raises(ValueError, match="duplicate bone names"):
        spec.validate()


def test_invalid_layer_deform_ownership_is_rejected():
    topology = list(rigped_core_spec.RigpedCoreSpec().resolved_topology())
    role = rigped_core_spec.GeneratedRole.FK_UPPER
    index = next(index for index, entry in enumerate(topology) if entry.role == role)
    topology[index] = replace(topology[index], deform=True)
    spec = rigped_core_spec.RigpedCoreSpec(topology=tuple(topology))
    with pytest.raises(ValueError, match="Only DEFORM"):
        spec.validate()


def test_invalid_parent_cycle_is_rejected():
    topology = list(rigped_core_spec.RigpedCoreSpec().resolved_topology())
    root_index = next(
        index
        for index, entry in enumerate(topology)
        if entry.role == rigped_core_spec.GeneratedRole.ROOT
    )
    topology[root_index] = replace(
        topology[root_index],
        parent_role=rigped_core_spec.GeneratedRole.COM,
    )
    spec = rigped_core_spec.RigpedCoreSpec(topology=tuple(topology))
    with pytest.raises(ValueError):
        spec.validate()
