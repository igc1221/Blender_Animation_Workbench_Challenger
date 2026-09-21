import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_create_policy_tests"

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
humanoid = _load_module("rigped_humanoid_spec")
create_policy = _load_module("rigped_create_policy")


def test_vertical_drag_scales_from_initial_mesh_height_and_clamps_small_values():
    assert create_policy.height_from_vertical_drag(2.0, 0.0) == pytest.approx(2.0)
    assert create_policy.height_from_vertical_drag(2.0, 200.0) == pytest.approx(3.0)
    assert create_policy.height_from_vertical_drag(2.0, -1000.0) == pytest.approx(0.25)


def test_free_mouse_sizing_is_smooth_persistent_and_symmetric_by_doubling_distance():
    assert create_policy.height_from_free_mouse(1.8, 0.0) == pytest.approx(1.8)
    assert create_policy.height_from_free_mouse(1.8, 360.0) == pytest.approx(3.6)
    assert create_policy.height_from_free_mouse(1.8, -360.0) == pytest.approx(0.9)
    assert create_policy.height_from_free_mouse(1.8, -5000.0) == pytest.approx(0.25)


def test_fitted_spec_uniformly_scales_generated_topology_before_build():
    base = humanoid.RigpedHumanoidSpec()
    source_height = create_policy.humanoid_spec_height(base)
    target_height = source_height * 1.5
    fitted = create_policy.fitted_humanoid_spec(target_height, (1.0, 2.0, 3.0), base=base)
    assert fitted.world_location == (1.0, 2.0, 3.0)
    assert create_policy.humanoid_spec_height(fitted) == pytest.approx(target_height)
    assert fitted.display_scale == pytest.approx(1.5)

    original = {bone.role: bone for bone in base.resolved_bones()}
    scaled = {bone.role: bone for bone in fitted.resolved_bones()}
    for role in ("HEAD", "UPPER_ARM.L", "IK_FOOT.R", "MCH_TOE.L", "DEF_HAND.R"):
        assert scaled[role].head == pytest.approx(tuple(value * 1.5 for value in original[role].head))
        assert scaled[role].tail == pytest.approx(tuple(value * 1.5 for value in original[role].tail))


def test_fitted_spec_accumulates_display_scale_when_refitting_an_already_scaled_source():
    base = humanoid.RigpedHumanoidSpec(display_scale=0.75)
    source_height = create_policy.humanoid_spec_height(base)
    fitted = create_policy.fitted_humanoid_spec(source_height * 2.0, (0.0, 0.0, 0.0), base=base)
    assert fitted.display_scale == pytest.approx(1.5)


def test_fitted_spec_keeps_semantic_graph_contract_and_responsibility_layers():
    fitted = create_policy.fitted_humanoid_spec(1.8, (0.0, 0.0, 0.0))
    fitted.validate()
    assert fitted.resolved_constraints() == humanoid.DEFAULT_HUMANOID_CONSTRAINTS
    assert fitted.resolved_groups() == humanoid.DEFAULT_HUMANOID_GROUPS
    assert fitted.resolved_chains() == humanoid.DEFAULT_HUMANOID_CHAINS
    assert fitted.resolved_opposites() == humanoid.DEFAULT_HUMANOID_OPPOSITES
    assert fitted.resolved_kinematics() == humanoid.DEFAULT_HUMANOID_KINEMATICS
