from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_fit_commands_tests"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package

fit_spec = spec_from_file_location(
    f"{PACKAGE_NAME}.rigped_fit_state",
    PACKAGE_PATH / "rigped_fit_state.py",
)
assert fit_spec is not None and fit_spec.loader is not None
fit = module_from_spec(fit_spec)
sys.modules[fit_spec.name] = fit
fit_spec.loader.exec_module(fit)

commands_spec = spec_from_file_location(
    f"{PACKAGE_NAME}.rigped_fit_commands",
    PACKAGE_PATH / "rigped_fit_commands.py",
)
assert commands_spec is not None and commands_spec.loader is not None
commands = module_from_spec(commands_spec)
sys.modules[commands_spec.name] = commands
commands_spec.loader.exec_module(commands)

FitCommandError = commands.FitCommandError
fit_move_supported = commands.fit_move_supported
move_fit_part_rig_local = commands.move_fit_part_rig_local
FitPartKind = fit.FitPartKind
FitRestPartSnapshot = fit.FitRestPartSnapshot
derive_rest_parts = fit.derive_rest_parts
extract_fit_document = fit.extract_fit_document


def _draft():
    _document, draft = extract_fit_document(
        character_id="char",
        profile_id="rigped",
        schema_version=1,
        setup_revision=1,
        setup_signature="setup",
        owner_data_identity=("obj", "data"),
        animation_footprint_digest="anim",
        snapshots=(
            FitRestPartSnapshot(
                binding_id="root",
                semantic_key="awb.root",
                side="CENTER",
                parent_binding_id=None,
                connected=False,
                head=(0.0, 0.0, 0.0),
                tail=(0.0, 1.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                width=0.1,
                depth=0.1,
                name_hint="Root",
            ),
            FitRestPartSnapshot(
                binding_id="com",
                semantic_key="awb.com",
                side="CENTER",
                parent_binding_id="root",
                connected=False,
                head=(0.0, 0.2, 0.0),
                tail=(0.0, 0.7, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                width=0.2,
                depth=0.2,
                kind=FitPartKind.FRAME,
                name_hint="COM",
            ),
            FitRestPartSnapshot(
                binding_id="pelvis",
                semantic_key="awb.pelvis",
                side="CENTER",
                parent_binding_id="com",
                connected=False,
                head=(0.0, 0.2, 0.0),
                tail=(0.0, 0.8, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                width=0.2,
                depth=0.15,
                name_hint="Pelvis",
            ),
        ),
    )
    return draft


def _rest_map(draft):
    return {part.part_id: part for part in derive_rest_parts(draft)}


def test_com_move_translates_com_subtree_only():
    draft = _draft()
    before = _rest_map(draft)
    delta = (0.3, -0.2, 0.5)

    moved = move_fit_part_rig_local(draft, "com", delta)
    after = _rest_map(moved)

    assert after["root"].head == pytest.approx(before["root"].head)
    assert after["root"].tail == pytest.approx(before["root"].tail)
    assert after["com"].head == pytest.approx(
        tuple(before["com"].head[index] + delta[index] for index in range(3))
    )
    assert after["pelvis"].head == pytest.approx(
        tuple(before["pelvis"].head[index] + delta[index] for index in range(3))
    )
    assert after["com"].orientation == pytest.approx(before["com"].orientation)
    assert after["pelvis"].orientation == pytest.approx(before["pelvis"].orientation)


def test_com_move_is_the_only_first_slice_move_capability():
    draft = _draft()

    assert fit_move_supported(draft, "com")
    assert not fit_move_supported(draft, "pelvis")

    with pytest.raises(FitCommandError, match="FIT_F3_MOVE_UNSUPPORTED_PART"):
        move_fit_part_rig_local(draft, "pelvis", (0.1, 0.0, 0.0))


def test_zero_delta_is_exact_noop():
    draft = _draft()

    assert move_fit_part_rig_local(draft, "com", (0.0, 0.0, 0.0)) is draft
