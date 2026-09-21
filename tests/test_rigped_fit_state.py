import dataclasses
import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_rigped_fit_state_tests"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package

spec = spec_from_file_location(
    f"{PACKAGE_NAME}.rigped_fit_state",
    PACKAGE_PATH / "rigped_fit_state.py",
)
assert spec is not None and spec.loader is not None
fit = module_from_spec(spec)
sys.modules[spec.name] = fit
spec.loader.exec_module(fit)


def _qz(degrees: float):
    half = math.radians(degrees) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _snapshot(
    binding_id,
    *,
    parent=None,
    connected=False,
    head=(0.0, 0.0, 0.0),
    tail=(0.0, 1.0, 0.0),
    orientation=(1.0, 0.0, 0.0, 0.0),
    width=0.1,
    depth=0.1,
    kind=None,
):
    return fit.FitRestPartSnapshot(
        binding_id=binding_id,
        semantic_key=f"awb.{binding_id}",
        side="CENTER",
        parent_binding_id=parent,
        connected=connected,
        head=head,
        tail=tail,
        orientation=orientation,
        width=width,
        depth=depth,
        kind=(fit.FitPartKind.BONE if kind is None else kind),
        allowed_operations=(
            fit.FitOperation.MOVE,
            fit.FitOperation.ROTATE,
            fit.FitOperation.SCALE,
        ),
        name_hint=binding_id,
    )


def _document(*snapshots):
    return fit.extract_fit_document(
        character_id="char",
        profile_id="profile",
        schema_version=1,
        setup_revision=3,
        setup_signature="sig",
        owner_data_identity=("owner:1", "data:2"),
        animation_footprint_digest="anim",
        snapshots=tuple(snapshots),
    )


def test_fit_state_round_trip_preserves_connected_and_disconnected_rest_geometry():
    root = _snapshot(
        "root-binding",
        head=(0.5, -0.25, 1.0),
        tail=(0.5, 1.75, 1.0),
        width=0.2,
        depth=0.15,
    )
    child = _snapshot(
        "child-binding",
        parent="root-binding",
        connected=True,
        head=root.tail,
        tail=(-0.5, 1.75, 1.0),
        orientation=_qz(90.0),
        width=0.12,
        depth=0.1,
    )
    offset = _snapshot(
        "offset-binding",
        parent="root-binding",
        connected=False,
        head=(1.5, 0.75, 1.0),
        tail=(1.5, 1.75, 1.0),
        width=0.08,
        depth=0.07,
    )

    document, draft = _document(offset, child, root)
    derived = {part.part_id: part for part in fit.derive_rest_parts(draft)}

    assert tuple(part.part_id for part in document.parts) == (
        "root-binding",
        "offset-binding",
        "child-binding",
    )
    for source in (root, child, offset):
        part = derived[source.binding_id]
        assert part.head == pytest.approx(source.head)
        assert part.tail == pytest.approx(source.tail)
        assert abs(sum(a * b for a, b in zip(part.orientation, source.orientation, strict=True))) == pytest.approx(1.0)

    assert document.baseline.native_rest_digest
    assert document.baseline.appearance_digest


def test_canonical_fit_values_do_not_store_head_tail_or_roll_as_independent_authority():
    fields = {field.name for field in dataclasses.fields(fit.FitPartValue)}
    assert fields == {
        "attachment_offset",
        "local_orientation",
        "length",
        "width",
        "depth",
    }


def test_fit_part_id_is_existing_binding_identity_not_name_hint():
    snapshot = _snapshot("stable-binding")
    document, _draft = _document(snapshot)
    part = document.parts[0]
    assert part.part_id == "stable-binding"
    assert part.binding_id == "stable-binding"
    assert part.name_hint == "stable-binding"


def test_width_depth_only_edit_is_appearance_change_without_structure_revision_change():
    snapshot = _snapshot("root-binding")
    _document_value, draft = _document(snapshot)
    original = draft.values[0].value
    changed = fit.replace_draft_value(
        draft,
        "root-binding",
        dataclasses.replace(original, width=original.width * 1.5),
    )
    plan = fit.build_rest_mutation_plan(changed)

    assert plan.appearance_changed
    assert not plan.structure_changed
    assert len(plan.mutations) == 1
    assert plan.mutations[0].appearance_changed
    assert not plan.mutations[0].structure_changed
    assert plan.mutations[0].changed_fields == ("width",)
    assert plan.mutations[0].before.width == pytest.approx(original.width)
    assert plan.mutations[0].derived.width == pytest.approx(original.width * 1.5)
    assert plan.candidate_rest_digest == draft.document.baseline.native_rest_digest
    assert plan.candidate_appearance_digest != draft.document.baseline.appearance_digest


def test_length_edit_is_structural_and_moves_derived_tail_without_storing_tail():
    snapshot = _snapshot("root-binding")
    _document_value, draft = _document(snapshot)
    original = draft.values[0].value
    changed = fit.replace_draft_value(
        draft,
        "root-binding",
        dataclasses.replace(original, length=2.0),
    )
    plan = fit.build_rest_mutation_plan(changed)

    assert plan.structure_changed
    assert not plan.appearance_changed
    assert plan.mutations[0].changed_fields == ("tail",)
    assert plan.mutations[0].derived.tail == pytest.approx((0.0, 2.0, 0.0))


def test_validation_rejects_non_normalized_orientation_and_nonpositive_dimensions():
    snapshot = _snapshot("root-binding")
    _document_value, draft = _document(snapshot)
    original = draft.values[0].value

    with pytest.raises(fit.FitStateError, match="FIT_ORIENTATION_NOT_NORMALIZED"):
        fit.replace_draft_value(
            draft,
            "root-binding",
            dataclasses.replace(original, local_orientation=(2.0, 0.0, 0.0, 0.0)),
        )

    for field in ("length", "width", "depth"):
        with pytest.raises(fit.FitStateError):
            fit.replace_draft_value(
                draft,
                "root-binding",
                dataclasses.replace(original, **{field: 0.0}),
            )


def test_validation_rejects_topology_cycle_and_connected_attachment_mismatch():
    root = _snapshot("root-binding")
    child = _snapshot(
        "child-binding",
        parent="root-binding",
        connected=True,
        head=(0.0, 2.0, 0.0),
        tail=(0.0, 3.0, 0.0),
    )
    with pytest.raises(fit.FitStateError, match="FIT_CONNECTED_ATTACHMENT_MISMATCH"):
        _document(root, child)

    definition_a = fit.FitPartDefinition(
        "a", "a", "awb.a", "CENTER", "b", False
    )
    definition_b = fit.FitPartDefinition(
        "b", "b", "awb.b", "CENTER", "a", False
    )
    baseline = fit.FitSessionBaseline(
        "char", "profile", 1, fit.topology_fingerprint((definition_a, definition_b)),
        1, "sig", "rest", "appearance", (), "anim"
    )
    value = fit.FitPartValue((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 1.0, 0.1, 0.1)
    document = fit.FitDocument(
        baseline,
        (definition_a, definition_b),
        (fit.FitPartValueRecord("a", value), fit.FitPartValueRecord("b", value)),
    )
    with pytest.raises(fit.FitStateError, match="FIT_TOPOLOGY_CYCLE"):
        fit.validate_fit_draft(
            fit.FitDraft(document, document.baseline_values)
        )



def test_com_frame_separates_semantic_frame_from_native_carrier_length():
    com = _snapshot(
        "com-binding",
        head=(0.0, 0.0, 1.0),
        tail=(0.0, 0.25, 1.0),
        width=0.3,
        depth=0.3,
        kind=fit.FitPartKind.FRAME,
    )
    document, draft = _document(com)
    definition = document.parts[0]
    value = draft.values[0].value
    derived = fit.derive_rest_parts(draft)[0]

    assert definition.kind is fit.FitPartKind.FRAME
    assert definition.carrier_length == pytest.approx(0.25)
    assert value.length is None
    assert derived.head == pytest.approx(com.head)
    assert derived.tail == pytest.approx(com.tail)

    moved = fit.replace_draft_value(
        draft,
        "com-binding",
        dataclasses.replace(value, attachment_offset=(0.5, 0.0, 1.0)),
    )
    plan = fit.build_rest_mutation_plan(moved)
    assert plan.structure_changed
    assert not plan.appearance_changed
    assert plan.mutations[0].changed_fields == ("head", "tail")


def test_fit_snapshot_pins_blender_rest_frame_and_bbone_appearance_contract():
    assert fit.FIT_BONE_FRAME_CONVENTION == "BLENDER_REST_FRAME_Y_ALONG_BONE_XZ_ENCODE_ROLL_V1"
    assert fit.FIT_STRUCTURE_FIELDS == ("head", "tail", "orientation")
    assert fit.FIT_APPEARANCE_FIELDS == ("width", "depth")
    fields = {field.name for field in dataclasses.fields(fit.FitRestPartSnapshot)}
    assert {"head", "tail", "orientation", "width", "depth", "connected", "parent_binding_id"} <= fields


def test_fit_state_module_has_no_blender_runtime_dependency():
    source = (PACKAGE_PATH / "rigped_fit_state.py").read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "mathutils" not in source
    assert "EditBone" not in source
