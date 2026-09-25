from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_rigify_reference_tests"

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


_load_module("semantic_model")
_load_module("semantic_adapter")
_load_module("character_model")
_load_module("character_query")
_load_module("rigped_contract")
humanoid = _load_module("rigped_humanoid_spec")
reference = _load_module("rigped_rigify_reference")


def test_rigify_reference_keeps_exact_awb_topology_and_contract_data():
    base = humanoid.RigpedHumanoidSpec()
    result = reference.rigify_reference_humanoid_spec(base)

    assert tuple(b.role for b in result.resolved_bones()) == tuple(
        b.role for b in base.resolved_bones()
    )
    assert tuple(b.name for b in result.resolved_bones()) == tuple(
        b.name for b in base.resolved_bones()
    )
    assert result.resolved_constraints() == base.resolved_constraints()
    assert result.resolved_groups() == base.resolved_groups()
    assert result.resolved_chains() == base.resolved_chains()
    assert result.resolved_opposites() == base.resolved_opposites()
    assert result.resolved_kinematics() == base.resolved_kinematics()


def test_rigify_reference_imports_only_reference_proportions_not_rigify_bones():
    result = reference.rigify_reference_humanoid_spec()
    roles = {bone.role for bone in result.resolved_bones()}
    names = {bone.name for bone in result.resolved_bones()}

    assert "UPPER_ARM.L" in roles
    assert "THIGH.L" in roles
    assert "breast.L" not in roles
    assert "heel.02.L" not in roles
    assert "spine.003" not in names
    assert not any("ORG-" in name or "MCH-" in name or "DEF-" in name for name in names)


def test_rigify_reference_updates_awb_human_proportions_and_mirrors_sides():
    result = reference.rigify_reference_humanoid_spec()
    by_role = {bone.role: bone for bone in result.resolved_bones()}

    assert by_role["UPPER_ARM.L"].head == (0.1953, 0.0267, 1.5846)
    assert by_role["FOREARM.L"].tail == (0.6594, 0.0492, 1.3061)
    assert by_role["THIGH.L"].tail == (0.0980, -0.0286, 0.5372)
    assert by_role["FOOT.L"].tail == (0.0980, -0.0934, 0.0167)

    left = by_role["UPPER_ARM.L"]
    right = by_role["UPPER_ARM.R"]
    assert right.head == (-left.head[0], left.head[1], left.head[2])
    assert right.tail == (-left.tail[0], left.tail[1], left.tail[2])


def test_rigify_reference_reuses_awb_geometry_for_awb_only_solver_layers():
    result = reference.rigify_reference_humanoid_spec()
    by_role = {bone.role: bone for bone in result.resolved_bones()}

    for target, source in (
        ("IK_HAND.L", "HAND.L"),
        ("MCH_UPPER_ARM.L", "UPPER_ARM.L"),
        ("DEF_UPPER_ARM.L", "UPPER_ARM.L"),
        ("MCH_CONTACT_HAND.L", "HAND.L"),
        ("IK_FOOT.L", "TOE.L"),
        ("MCH_CONTACT_FOOT.L", "TOE.L"),
        ("DEF_FOOT.L", "FOOT.L"),
    ):
        assert by_role[target].head == by_role[source].head
        assert by_role[target].tail == by_role[source].tail


def test_rigify_reference_module_is_pure_data_source():
    with open(reference.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "import bpy" not in source
    assert "rigify_generate" not in source
    assert "armature_basic_human_metarig_add" not in source
