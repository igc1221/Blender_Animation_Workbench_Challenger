import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_fit_policy_tests"

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
fit = _load_module("rigped_fit_policy")


def _facts(**overrides):
    values = {
        "descriptor_valid": True,
        "lifecycle": rigped_contract.RigpedLifecycle.FITTED_UNBOUND,
        "has_animation": False,
        "writable": True,
        "character_coherent": True,
    }
    values.update(overrides)
    return fit.FitEntryFacts(**values)


def test_fitted_unbound_unanimated_writable_character_may_enter_fit():
    decision = fit.evaluate_fit_entry(_facts())
    assert decision.allowed
    assert decision.issue_codes == ()


def test_bound_character_cannot_reenter_ordinary_fit():
    decision = fit.evaluate_fit_entry(
        _facts(lifecycle=rigped_contract.RigpedLifecycle.BOUND)
    )
    assert not decision.allowed
    assert "FIT_AFTER_BIND_BLOCKED" in decision.issue_codes


def test_existing_animation_may_reenter_fit_when_runtime_freezes_keys():
    decision = fit.evaluate_fit_entry(_facts(has_animation=True))
    assert decision.allowed
    assert decision.issue_codes == ()


def test_invalid_or_readonly_setup_fails_closed():
    decision = fit.evaluate_fit_entry(
        _facts(descriptor_valid=False, writable=False, character_coherent=False)
    )
    assert not decision.allowed
    assert decision.issue_codes == (
        "FIT_CHARACTER_INCOHERENT",
        "FIT_DESCRIPTOR_INVALID",
        "FIT_TARGET_READ_ONLY",
    )


def test_noop_fit_commit_does_not_bump_revision():
    decision = fit.decide_fit_commit(
        previous_revision=4,
        previous_signature="same",
        candidate_signature="same",
    )
    assert not decision.structural_change
    assert decision.next_revision == 4


def test_structural_fit_commit_bumps_revision_exactly_once():
    decision = fit.decide_fit_commit(
        previous_revision=4,
        previous_signature="before",
        candidate_signature="after",
    )
    assert decision.structural_change
    assert decision.next_revision == 5


def test_fit_session_freshness_detects_graph_and_animation_binding_changes():
    session = fit.FitSessionToken(
        character_id="char",
        profile_id="profile",
        setup_revision=2,
        setup_signature="sig",
        source_stamp=("source", 2),
        owner_binding_tokens=((1, 2, 3, 4),),
        animation_signature=(),
    )
    issues = fit.validate_fit_session_fresh(
        session,
        character_id="char",
        setup_revision=2,
        stored_signature="sig",
        source_stamp=("source", 3),
        owner_binding_tokens=((1, 9, 8, 7),),
    )
    assert issues == (
        "FIT_SESSION_CHARACTER_GRAPH_CHANGED",
        "FIT_SESSION_ANIMATION_BINDING_CHANGED",
    )
