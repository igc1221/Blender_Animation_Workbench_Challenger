import inspect
import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PACKAGE_PATH = Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "baw_phase4_representation_snap_tests"

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
kinematic_runtime = _load_module("kinematic_runtime")
representation_snap = _load_module("phase4_representation_snap")


class FakeNative:
    def __init__(self, name: str, pointer: int):
        self.name = name
        self._pointer = pointer

    def as_pointer(self):
        return self._pointer


class FakePoseBone(FakeNative):
    def __init__(self, name: str, pointer: int):
        super().__init__(name, pointer)
        self.constraints = []


class FakeArmature(FakeNative):
    def __init__(self, name: str, pointer: int, bones):
        super().__init__(name, pointer)
        self.pose = SimpleNamespace(bones=list(bones))


class FakeConstraint:
    _next_pointer = 1000

    def __init__(
        self,
        kind: str,
        *,
        target=None,
        subtarget="",
        pole_target=None,
        pole_subtarget="",
        influence=1.0,
    ):
        self.type = kind
        self._pointer = FakeConstraint._next_pointer
        FakeConstraint._next_pointer += 1
        self.target = target
        self.subtarget = subtarget
        self.pole_target = pole_target
        self.pole_subtarget = pole_subtarget
        self.influence = influence
        self.use_rotation = False
        self.use_tail = True
        self.use_stretch = False
        self.chain_count = 2

    def as_pointer(self):
        return self._pointer


def _resolved_bone(owner, bone):
    return semantic_adapter.ResolvedControl(
        semantic_model.AWBControl.bone(owner.name, bone.name),
        owner,
        bone,
        data_path_prefix=f'pose.bones["{bone.name}"]',
    )


def _binding(
    binding_id: str,
    bone_name: str,
    semantic_key: str,
    *,
    mode=character_model.ControlMode.NEUTRAL,
    usage=character_model.ControlUsage.PRIMARY,
):
    return character_model.CharacterBinding(
        binding_id,
        "rig-owner",
        semantic_model.AWBControlKind.BONE,
        bone_id=f"token-{binding_id}",
        bone_name_hint=bone_name,
        semantic_key=semantic_key,
        side=character_model.CharacterSide.LEFT,
        mode=mode,
        usage=usage,
    )


def _fixture(*, duplicate_result_terminal=False, duplicate_authored_terminal=False, remove_terminal_fk=False):
    fk_upper = FakePoseBone("FK Upper Renamed", 101)
    fk_lower = FakePoseBone("FK Lower Renamed", 102)
    authored_terminal = FakePoseBone("Authored Terminal Renamed", 103)
    result_upper = FakePoseBone("Result Upper Renamed", 111)
    result_lower = FakePoseBone("Result Lower Renamed", 112)
    result_terminal = FakePoseBone("Result Terminal Renamed", 113)
    ik_target = FakePoseBone("IK Target Renamed", 121)
    pole = FakePoseBone("Pole Renamed", 122)
    extra_terminal = FakePoseBone("Extra Result Terminal", 123)
    extra_authored = FakePoseBone("Extra Authored Terminal", 124)

    bones = [
        fk_upper,
        fk_lower,
        authored_terminal,
        result_upper,
        result_lower,
        result_terminal,
        ik_target,
        pole,
        extra_terminal,
        extra_authored,
    ]
    rig = FakeArmature("Rig", 100, bones)

    result_upper.constraints.append(
        FakeConstraint("COPY_ROTATION", target=rig, subtarget=fk_upper.name)
    )
    result_lower.constraints.append(
        FakeConstraint("COPY_ROTATION", target=rig, subtarget=fk_lower.name)
    )
    result_lower.constraints.append(
        FakeConstraint(
            "IK",
            target=rig,
            subtarget=ik_target.name,
            pole_target=rig,
            pole_subtarget=pole.name,
            influence=0.0,
        )
    )
    if not remove_terminal_fk:
        result_terminal.constraints.append(
            FakeConstraint("COPY_ROTATION", target=rig, subtarget=authored_terminal.name)
        )
    result_terminal.constraints.append(
        FakeConstraint("COPY_ROTATION", target=rig, subtarget=ik_target.name, influence=0.0)
    )

    bindings = [
        _binding("fk-upper", fk_upper.name, "awb.arm", mode=character_model.ControlMode.FK),
        _binding("fk-lower", fk_lower.name, "awb.arm", mode=character_model.ControlMode.FK),
        _binding("authored-terminal", authored_terminal.name, "awb.hand", mode=character_model.ControlMode.FK),
        _binding(
            "result-upper",
            result_upper.name,
            "awb.arm",
            usage=character_model.ControlUsage.MECHANISM,
        ),
        _binding(
            "result-lower",
            result_lower.name,
            "awb.arm",
            usage=character_model.ControlUsage.MECHANISM,
        ),
        _binding(
            "result-terminal",
            result_terminal.name,
            "awb.hand",
            usage=character_model.ControlUsage.MECHANISM,
        ),
        _binding(
            "ik-target",
            ik_target.name,
            "awb.hand",
            mode=character_model.ControlMode.IK,
            usage=character_model.ControlUsage.TARGET,
        ),
        _binding(
            "pole",
            pole.name,
            "awb.arm",
            mode=character_model.ControlMode.IK,
            usage=character_model.ControlUsage.POLE,
        ),
    ]
    resolved = {
        "fk-upper": _resolved_bone(rig, fk_upper),
        "fk-lower": _resolved_bone(rig, fk_lower),
        "authored-terminal": _resolved_bone(rig, authored_terminal),
        "result-upper": _resolved_bone(rig, result_upper),
        "result-lower": _resolved_bone(rig, result_lower),
        "result-terminal": _resolved_bone(rig, result_terminal),
        "ik-target": _resolved_bone(rig, ik_target),
        "pole": _resolved_bone(rig, pole),
    }
    extras = ["result-terminal"]

    if duplicate_result_terminal:
        bindings.append(
            _binding(
                "result-terminal-extra",
                extra_terminal.name,
                "awb.hand",
                usage=character_model.ControlUsage.MECHANISM,
            )
        )
        resolved["result-terminal-extra"] = _resolved_bone(rig, extra_terminal)
        extras.append("result-terminal-extra")

    if duplicate_authored_terminal:
        bindings.append(
            _binding(
                "authored-terminal-extra",
                extra_authored.name,
                "awb.hand",
                mode=character_model.ControlMode.FK,
            )
        )
        resolved["authored-terminal-extra"] = _resolved_bone(rig, extra_authored)

    fk_chain = character_model.CharacterChain(
        "fk-chain",
        "awb.arm",
        "FK",
        side=character_model.CharacterSide.LEFT,
        mode=character_model.ControlMode.FK,
        members=("fk-upper", "fk-lower"),
    )
    result_chain = character_model.CharacterChain(
        "result-chain",
        "awb.arm",
        "Result",
        side=character_model.CharacterSide.LEFT,
        mode=character_model.ControlMode.NEUTRAL,
        members=("result-upper", "result-lower"),
    )
    mapping = character_model.KinematicMapping(
        "arm-map",
        "awb.arm",
        side=character_model.CharacterSide.LEFT,
        fk_chain_id="fk-chain",
        ik_target_binding_id="ik-target",
        pole_binding_id="pole",
        reference_chain_id="result-chain",
        extras=tuple(extras),
    )
    definition = character_model.AWBCharacter(
        "character",
        "Hero",
        3,
        (character_model.CharacterOwner("rig-owner", "Rig"),),
        tuple(bindings),
        chains=(fk_chain, result_chain),
        kinematics=(mapping,),
    )
    view = SimpleNamespace(
        definition=definition,
        resolved_bindings=tuple(resolved.items()),
        issues=(),
        source_stamp=("character", 3),
    )
    return view, rig, result_upper, result_lower, result_terminal, authored_terminal, ik_target


def test_resolves_generated_terminal_correspondence_without_visible_name_pattern():
    view, _rig, result_upper, result_lower, result_terminal, authored_terminal, ik_target = _fixture()

    result = representation_snap.resolve_limb_representation_capability(view, "arm-map")

    assert result.issues == ()
    capability = result.capability
    assert capability is not None
    assert capability.fk_binding_ids == ("fk-upper", "fk-lower")
    assert capability.result_binding_ids == ("result-upper", "result-lower")
    assert capability.authored_terminal_binding_id == "authored-terminal"
    assert capability.result_terminal_binding_id == "result-terminal"
    assert capability.result_controls[0].target is result_upper
    assert capability.result_controls[1].target is result_lower
    assert capability.result_terminal.target is result_terminal
    assert capability.authored_terminal.target is authored_terminal
    assert capability.native_ik.ik_target.target is ik_target
    assert len(capability.fk_copy_constraints) == 2
    assert capability.terminal_fk_constraint.target is _rig
    assert capability.terminal_ik_constraint.target is _rig


def test_rejects_multiple_result_terminals_with_same_semantic_side():
    view, *_rest = _fixture(duplicate_result_terminal=True)
    result = representation_snap.resolve_limb_representation_capability(view, "arm-map")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == [
        "AMBIGUOUS_RESULT_TERMINAL_CORRESPONDENCE"
    ]


def test_rejects_multiple_authored_primary_terminals_with_same_semantic_side():
    view, *_rest = _fixture(duplicate_authored_terminal=True)
    result = representation_snap.resolve_limb_representation_capability(view, "arm-map")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == [
        "AMBIGUOUS_AUTHORED_TERMINAL_CORRESPONDENCE"
    ]


def test_rejects_missing_exact_terminal_fk_copy_constraint():
    view, *_rest = _fixture(remove_terminal_fk=True)
    result = representation_snap.resolve_limb_representation_capability(view, "arm-map")
    assert result.capability is None
    assert [issue.code for issue in result.issues] == [
        "MISSING_REPRESENTATION_COPY_ROTATION"
    ]


def test_snap_payload_freezes_control_and_constraint_runtime_identity():
    view, *_rest = _fixture()
    result = representation_snap.resolve_limb_representation_capability(view, "arm-map")
    capability = result.capability
    assert capability is not None

    payload = representation_snap.build_representation_snap_payload(
        capability,
        representation_snap.SnapDirection.FK_TO_IK,
    )

    assert payload.fk_control_runtime_keys == tuple(
        semantic_adapter.runtime_control_key(control)
        for control in capability.fk_controls
    )
    assert payload.result_control_runtime_keys == tuple(
        semantic_adapter.runtime_control_key(control)
        for control in capability.result_controls
    )
    assert payload.authored_terminal_runtime_key == semantic_adapter.runtime_control_key(
        capability.authored_terminal
    )
    assert payload.result_terminal_runtime_key == semantic_adapter.runtime_control_key(
        capability.result_terminal
    )
    assert representation_snap.representation_payload_matches(payload, capability)

    capability.terminal_ik_constraint._pointer += 10000
    assert not representation_snap.representation_payload_matches(payload, capability)


def test_snap_preflight_rejects_unsupported_stretch_and_chain_count():
    view, *_rest = _fixture()
    result = representation_snap.resolve_limb_representation_capability(view, "arm-map")
    capability = result.capability
    assert capability is not None

    capability.native_ik.constraint.use_stretch = True
    issue = representation_snap._capability_preflight_issue(
        capability,
        representation_snap.SnapDirection.FK_TO_IK,
    )
    assert issue is not None
    assert issue[0] == "I12_UNSUPPORTED_IK_STRETCH"

    capability.native_ik.constraint.use_stretch = False
    capability.native_ik.constraint.chain_count = 3
    issue = representation_snap._capability_preflight_issue(
        capability,
        representation_snap.SnapDirection.FK_TO_IK,
    )
    assert issue is not None
    assert issue[0] == "I12_UNSUPPORTED_CHAIN_COUNT"



def test_generated_rigped_transient_hinge_branch_uses_builder_authority() -> None:
    source = inspect.getsource(representation_snap._configure_generated_rigped_hinge_branch)
    assert "configure_generated_rigped_ik_hinge_branch" in source
    assert "math.radians(179.0)" not in source


class _BranchQuaternion:
    def __init__(self, angle: float):
        self.w = math.cos(angle * 0.5)
        self.x = math.sin(angle * 0.5)
        self.y = 0.0
        self.z = 0.0

    def normalized(self):
        return self


@pytest.mark.parametrize(
    ("solver_name", "angle", "expected"),
    (
        ("MCH_Calf.L", math.radians(40.0), 1),
        ("MCH_Calf.R", math.radians(-40.0), -1),
        ("MCH_Calf.L", math.radians(0.1), 1),
        ("MCH_ForeArm.L", math.radians(40.0), 1),
        ("MCH_ForeArm.R", math.radians(-40.0), -1),
        ("MCH_ForeArm.L", math.radians(0.1), -1),
    ),
)
def test_generated_rigped_hinge_branch_tracks_authored_fk_and_uses_mapping_fallback(
    solver_name: str,
    angle: float,
    expected: int,
) -> None:
    authored_fk = SimpleNamespace(
        matrix_basis=SimpleNamespace(to_quaternion=lambda: _BranchQuaternion(angle)),
        # A conflicting evaluated matrix proves branch selection reads the FK
        # authored basis instead of the dormant IK/result representation.
        matrix=SimpleNamespace(to_quaternion=lambda: _BranchQuaternion(-angle)),
    )
    capability = SimpleNamespace(
        native_ik=SimpleNamespace(
            solver_owner=SimpleNamespace(target=SimpleNamespace(name=solver_name))
        ),
        fk_controls=(None, SimpleNamespace(target=authored_fk)),
    )

    assert representation_snap._generated_rigped_hinge_branch(capability) == expected


class _DiagnosticQuaternion(tuple):
    def __new__(cls, values):
        return super().__new__(cls, values)

    def normalized(self):
        return self


class _DiagnosticMatrix:
    def __init__(self, position, rotation=(1.0, 0.0, 0.0, 0.0)):
        self._position = position
        self._rotation = _DiagnosticQuaternion(rotation)

    def to_translation(self):
        return self._position

    def to_quaternion(self):
        return self._rotation


def test_fk_to_ik_residual_gate_remeasures_after_one_bounded_depsgraph_update() -> None:
    source = inspect.getsource(representation_snap.execute_representation_snap)
    retry = source.index("payload.direction is SnapDirection.FK_TO_IK")
    remeasure = source.index("residuals = _residuals_for_expected(", retry)
    final_gate = source.index(
        "if not _residuals_within_tolerance(residuals, scale=residual_scale):",
        remeasure + 1,
    )
    assert "bpy.context.view_layer.update()" in source[retry:final_gate]
    assert "Keep the frozen tolerance unchanged; only re-measure once." in source
    assert "_position_tolerance(residual_scale)" in source


def test_fk_to_ik_residual_trace_is_bounded_and_contains_snap_comparison(monkeypatch) -> None:
    captured = []
    trace_module = ModuleType(f"{PACKAGE_NAME}.debug_trace")
    trace_module.trace_event = lambda channel, event, **data: captured.append(
        (channel, event, data)
    )
    monkeypatch.setitem(sys.modules, trace_module.__name__, trace_module)

    expected_matrix = _DiagnosticMatrix((1.0, 2.0, 3.0))
    expected_terminal = _DiagnosticMatrix((4.0, 5.0, 6.0))
    ik_target_matrix = _DiagnosticMatrix((4.0, 5.0, 6.0))
    pole_matrix = _DiagnosticMatrix((7.0, 8.0, 9.0))
    representation_snap._trace_fk_to_ik_residual_failure(
        plan=SimpleNamespace(operation_id="i12-test-operation"),
        payload=SimpleNamespace(mapping_id="LEG.R"),
        context=object(),
        hinge_branch=-1,
        expected_result=(expected_matrix, expected_matrix, expected_matrix),
        expected_terminal=expected_terminal,
        ik_target=SimpleNamespace(target=SimpleNamespace(matrix=ik_target_matrix)),
        pole_target=SimpleNamespace(target=SimpleNamespace(matrix=pole_matrix)),
        pole_angle=0.25,
        residuals=representation_snap.SnapResiduals(0.0002, 0.0003, 0.0004, 0.0005),
        scale=5.0,
        position_tolerance=0.000016,
    )

    assert len(captured) == 1
    channel, event, data = captured[0]
    assert (channel, event) == ("CONTACT", "FK_TO_IK_SNAP_RESIDUAL_FAILURE")
    assert data["operation_id"] == "i12-test-operation"
    assert data["mapping_id"] == "LEG.R"
    assert data["selected_hinge_branch"] == -1
    assert len(data["expected_result"]) == 2
    assert data["expected_terminal"]["position"] == (4.0, 5.0, 6.0)
    assert data["post_write_ik_target"]["position"] == (4.0, 5.0, 6.0)
    assert data["post_write_pole_target"]["position"] == (7.0, 8.0, 9.0)
    assert data["residuals"] == {
        "chain_position": 0.0002,
        "chain_rotation": 0.0003,
        "terminal_position": 0.0004,
        "terminal_rotation": 0.0005,
    }


def test_generated_rigped_canonical_pole_preserves_solved_gauge() -> None:
    source = inspect.getsource(representation_snap._canonical_generated_rigped_pole)
    assert "return tuple(float(value) for value in pole_world_position), float(pole_angle)" in source
    assert "mirrored" not in source
    assert "+ math.pi" not in source