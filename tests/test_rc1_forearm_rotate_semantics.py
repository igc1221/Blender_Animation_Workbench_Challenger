from __future__ import annotations

import ast
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension" / "blender_animation_workbench"
TRANSFORM_PATH = EXT / "rigped_transform.py"
SOURCE = TRANSFORM_PATH.read_text(encoding="utf-8")
BUILDER_SOURCE = (EXT / "rigped_humanoid_builder.py").read_text(encoding="utf-8")
FIT_COMMIT_SOURCE = (EXT / "rigped_fit_commit.py").read_text(encoding="utf-8")
SNAP_SOURCE = (EXT / "phase4_representation_snap.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _class(name: str) -> ast.ClassDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    cls = _class(class_name)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"missing {class_name}.{method_name}")


def _source(node: ast.AST) -> str:
    segment = ast.get_source_segment(SOURCE, node)
    assert segment is not None
    return segment


def _evaluate_function(name: str, namespace: dict[str, object]) -> object:
    scope = dict(namespace)
    module = ast.Module(body=[_function(name)], type_ignores=[])
    exec(compile(module, str(TRANSFORM_PATH), "exec"), scope)  # noqa: S102
    return scope[name]


class _FakeVector:
    def __init__(self, values: object) -> None:
        self.values = [float(value) for value in values]  # type: ignore[arg-type]

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]

    @property
    def length(self) -> float:
        return math.sqrt(sum(value * value for value in self.values))

    def normalize(self) -> None:
        magnitude = self.length
        if magnitude > 0.0:
            self.values = [value / magnitude for value in self.values]

    def dot(self, other: object) -> float:
        return sum(a * b for a, b in zip(self, other, strict=True))  # type: ignore[arg-type]

    def __add__(self, other: object) -> _FakeVector:
        return _FakeVector(
            a + b for a, b in zip(self, other, strict=True)  # type: ignore[arg-type]
        )

    def __sub__(self, other: object) -> _FakeVector:
        return _FakeVector(
            a - b for a, b in zip(self, other, strict=True)  # type: ignore[arg-type]
        )

    def __mul__(self, scalar: float) -> _FakeVector:
        return _FakeVector(value * float(scalar) for value in self.values)

    __rmul__ = __mul__


class _FakeQuaternion:
    def __init__(self, values: object, angle: float | None = None) -> None:
        if angle is None:
            self.values = [float(value) for value in values]  # type: ignore[arg-type]
        else:
            axis = _FakeVector(values)
            axis.normalize()
            half_angle = float(angle) * 0.5
            self.values = [
                math.cos(half_angle),
                *(component * math.sin(half_angle) for component in axis),
            ]

    def __iter__(self):
        return iter(self.values)

    @classmethod
    def identity(cls) -> _FakeQuaternion:
        return cls((1.0, 0.0, 0.0, 0.0))

    @classmethod
    def from_axis_angle(
        cls,
        axis: tuple[float, float, float],
        angle: float,
    ) -> _FakeQuaternion:
        return cls(axis, angle)

    def normalized(self) -> _FakeQuaternion:
        magnitude = math.sqrt(sum(value * value for value in self.values))
        if magnitude <= 1e-15:
            return _FakeQuaternion.identity()
        return _FakeQuaternion(value / magnitude for value in self.values)

    def conjugated(self) -> _FakeQuaternion:
        return _FakeQuaternion(
            (self.values[0], -self.values[1], -self.values[2], -self.values[3])
        )

    def __matmul__(self, other: object) -> object:
        if isinstance(other, _FakeVector):
            pure = _FakeQuaternion((0.0, *other.values))
            rotated = self @ pure @ self.conjugated()
            assert isinstance(rotated, _FakeQuaternion)
            return _FakeVector(rotated.values[1:])
        assert isinstance(other, _FakeQuaternion)
        w1, x1, y1, z1 = self.values
        w2, x2, y2, z2 = other.values
        return _FakeQuaternion(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        )


class _FakeMatrix3:
    def __init__(self, rotation: _FakeQuaternion | None = None) -> None:
        self.rotation = rotation or _FakeQuaternion.identity()

    def to_3x3(self) -> _FakeMatrix3:
        return self

    def normalized(self) -> _FakeMatrix3:
        return self

    def to_quaternion(self) -> _FakeQuaternion:
        return self.rotation

    def __matmul__(self, other: _FakeMatrix3) -> _FakeMatrix3:
        result = self.rotation @ other.rotation
        assert isinstance(result, _FakeQuaternion)
        return _FakeMatrix3(result)


class _FakePoseBone:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRotateState:
    def __init__(self, pose_bone: _FakePoseBone, hinge_state: tuple[float, ...]) -> None:
        self.control = type("Control", (), {"target": pose_bone})()
        self.hinge_state = hinge_state


class _FakeSlidingSession:
    def __init__(self, start_rotation: _FakeQuaternion) -> None:
        self.root_world = (0.0, 0.0, 0.0)
        self.joint_world = (0.0, 1.0, 0.0)
        self.end_world = (0.0, 0.0, 1.0)
        owner = type("Owner", (), {"matrix_world": _FakeMatrix3()})()
        self.second_control = type("Control", (), {"owner_object": owner})()
        self.second_start_matrix = _FakeMatrix3(start_rotation)


class _FakeProjection:
    def __init__(
        self,
        swivel_angle: float,
        roll_angle: float,
        residual_radians: float,
        conditioning: float,
    ) -> None:
        self.swivel_angle = swivel_angle
        self.roll_angle = roll_angle
        self.residual_radians = residual_radians
        self.conditioning = conditioning


def _projection_test_namespace() -> dict[str, object]:
    fake_bpy = type(
        "Bpy",
        (),
        {"types": type("Types", (), {"PoseBone": _FakePoseBone})},
    )()
    namespace = {
        "Vector": _FakeVector,
        "Quaternion": _FakeQuaternion,
        "FkTwoBoneMoveSession": object,
        "DirectRotateControlState": object,
        "SlidingGlobalRotateProjection": _FakeProjection,
        "bpy": fake_bpy,
        "_RIGPED_LOCAL_AXES": {"Y": _FakeVector((0.0, 1.0, 0.0))},
        "_RIGPED_HINGE_JOINT_LIMITS": {"Calf.L": (0.0, 0.0, 0.0, 1.0)},
        "_clamp_scalar": lambda value, minimum, maximum: max(
            minimum,
            min(maximum, value),
        ),
        "sqrt": math.sqrt,
        "atan2": math.atan2,
        "pi": math.pi,
        "radians": math.radians,
    }
    namespace["_quaternion_rotation_vector_components"] = _evaluate_function(
        "_quaternion_rotation_vector_components",
        {"sqrt": math.sqrt, "atan2": math.atan2},
    )
    namespace["_damped_two_axis_rotation_step"] = _evaluate_function(
        "_damped_two_axis_rotation_step",
        {},
    )
    return namespace


def test_rc1_lower_limb_axis_contract_is_x_hinge_y_roll_z_swivel() -> None:
    axes = _source(_function("direct_transform_axes"))
    assert "ForeArm" not in axes
    assert "Calf" not in axes
    assert "return _orientation_axes_for_control(context, selected.active_control)" in axes

    assert '"ForeArm.L": ("X",' in SOURCE
    assert '"ForeArm.R": ("X",' in SOURCE
    assert '"Calf.L": ("X",' in SOURCE
    assert '"Calf.R": ("X",' in SOURCE

    session = _source(_function("_lower_limb_special_z_session"))
    assert "local_basis.col[2]" in session
    assert "ContactKeyType.FREE" in session
    assert "ContactKeyType.SLIDING" in session
    assert '"ForeArm.L"' in session
    assert '"Calf.L"' in session

    assert "owner.lock_ik_z = True" in BUILDER_SOURCE
    assert "owner.use_ik_limit_x = True" in BUILDER_SOURCE
    assert 'return -1 if str(owner_role) in {"MCH_FOREARM.L", "MCH_FOREARM.R"} else 1' in BUILDER_SOURCE
    assert "axis_index = 0" in SNAP_SOURCE
    assert "fallback = -1" in SNAP_SOURCE


def test_rc1_lower_limb_swivel_free_follows_terminal_sliding_pins_terminal() -> None:
    solved = _source(_function("_apply_solved_two_bone_fk_pose"))
    assert "terminal_follows_second: bool = False" in solved
    assert "session.second_start_matrix.inverted_safe()" in solved
    assert "@ session.terminal_start_matrix" in solved
    assert "terminal_matrix = second_matrix @ terminal_relative" in solved
    assert "terminal_matrix.translation = desired_end" in solved

    swivel = _source(_function("_apply_lower_limb_special_z_rotation"))
    assert "terminal_follows_second: bool" in swivel
    assert "terminal_follows_second=terminal_follows_second" in swivel

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert "terminal_follows_second=not bool(self._sliding_syncs)" in preview


def test_rc1_sliding_lower_long_roll_is_shared_by_forearm_and_calf() -> None:
    replay = _source(_function("_sync_generated_sliding_lower_roll_from_public_pose"))
    assert '"MCH_ForeArm.L"' in replay
    assert '"MCH_Calf.L"' in replay
    assert '"ForeArm.L"' in replay
    assert '"Calf.L"' in replay

    live = _source(_function("_apply_sliding_lower_long_roll"))
    assert "terminal world transform fixed" in live
    assert '_RIGPED_LOCAL_AXES["Y"]' in live

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert '{"ForeArm.L", "ForeArm.R", "Calf.L", "Calf.R"}' in preview
    assert "_apply_sliding_lower_long_roll(" in preview


def test_rc1_sliding_rotate_terminal_policy_is_axis_specific() -> None:
    source = _source(_function("_apply_direct_rotate_sliding_syncs"))
    assert "pin_terminal: bool = True" in source
    assert "if session.terminal_selected or not pin_terminal" in source
    assert "else session.start_ik_state" in source

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert 'self._orientation == "LOCAL"' in preview
    assert 'self.axis == "X"' in preview
    assert "pin_terminal=not sliding_lower_local_x" in preview


def test_rc1_forbidden_hinge_axis_bounds_to_zero_instead_of_preserving_requested_angle() -> None:
    source = _source(_function("_bounded_hinge_direct_rotate_angle"))
    assert "effective = False" in source
    assert "effective = True" in source
    assert "if not effective:" in source
    assert "return 0.0" in source


def test_rc1_forbidden_hinge_axis_skips_sliding_fk_to_ik_sync_preview() -> None:
    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    guard = preview.index("sliding_global_projection = bool(")
    early_return = preview.index(
        "if hinge_states and not generic_states and not hinge_axis_effective:"
    )
    sliding_sync = preview.index(
        "_apply_direct_rotate_sliding_syncs(",
        early_return,
    )
    assert guard < early_return < sliding_sync
    assert (
        "hinge_axis_effective = sliding_global_projection or sliding_lower_local_x or any("
        in preview
    )
    assert "self._current_angle = 0.0" in preview[early_return:sliding_sync]


def test_rc1_sliding_lower_local_x_is_effective_without_removing_forbidden_axis_guard() -> None:
    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    supported = preview.index("sliding_lower_local_x = (")
    guard = preview.index("if hinge_states and not generic_states and not hinge_axis_effective:")
    early_return = preview.index("self._current_angle = 0.0", guard)
    sliding_sync = preview.index("_apply_direct_rotate_sliding_syncs(", early_return)
    assert supported < guard < early_return < sliding_sync
    assert 'self._orientation == "LOCAL"' in preview[supported:guard]
    assert 'self.axis == "X"' in preview[supported:guard]
    assert "sliding_lower_single or sliding_lower_pair" in preview[supported:guard]
    assert "pin_terminal=not sliding_lower_local_x" in preview[sliding_sync:]


def test_rc1_local_z_sign_is_captured_once_from_frozen_axis_alignment() -> None:
    invoke = _source(_method("BAW_OT_rigped_direct_rotate_axis", "invoke"))
    assert "self._single_lower_link_z_axis_sign = (" in invoke
    assert "_lower_limb_special_z_axis_sign(" in invoke
    assert "lower_local_z = (" in invoke
    assert "axis_sign = _lower_limb_special_z_axis_sign(" in invoke
    assert 'active_name.endswith(".L")' not in invoke
    assert 'lower_name.endswith(".R")' not in invoke
    assert 'lower_name in {"ForeArm.L", "ForeArm.R"}' not in invoke
    sign_helper = _source(_function("_lower_limb_special_z_axis_sign"))
    assert "session.end_world" in sign_helper
    assert "session.root_world" in sign_helper
    assert "input_sign * semantic_sign" in sign_helper
    assert "max(range(3), key=lambda index: abs(" in sign_helper

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert "self._current_angle * self._single_lower_link_z_axis_sign" in preview
    assert "terminal_follows_second=not bool(self._sliding_syncs)" in preview


def test_rc1_local_z_axis_sign_preserves_hemisphere_and_degenerate_tie_break() -> None:
    sign = _evaluate_function(
        "_lower_limb_special_z_axis_sign",
        {"sqrt": math.sqrt, "Vector": object},
    )

    class Session:
        root_world = (0.0, 0.0, 0.0)
        end_world = (0.0, 0.0, 2.0)

    session = Session()
    assert sign(session, (0.0, 0.0, 1.0)) == 1.0
    assert sign(session, (0.0, 0.0, -1.0)) == -1.0

    # Perpendicular axes use a repeatable dominant-component tie-break, with
    # the same result for tiny perturbations inside the degeneracy threshold.
    assert sign(session, (1.0, 0.0, 1e-6)) == 1.0
    assert sign(session, (1.0, 0.0, -1e-6)) == 1.0
    assert sign(session, (-1.0, 0.0, 1e-6)) == -1.0

    session.end_world = (0.0, 0.0, 0.0)
    assert sign(session, (0.0, 0.0, 1.0)) == 1.0


def test_rc1_sliding_global_rotate_projects_to_swivel_and_axial_roll() -> None:
    projector = _source(_function("_project_global_rotation_to_sliding_lower_dofs"))
    assert "swivel_axis = end - root" in projector
    assert "roll_generator = post_swivel_rotation @ local_y" in projector
    assert "_damped_two_axis_rotation_step(" in projector
    assert "swivel_bound = pi - radians(0.25)" in projector
    assert "roll_minimum" in projector
    assert "roll_maximum" in projector
    assert "largest_step > 0.35" in projector
    assert "_quaternion_rotation_vector_components(" in projector
    assert "start_radial" not in projector
    assert "desired_radial" not in projector
    assert "atan2(" not in projector

    preview = _source(_method("BAW_OT_rigped_direct_rotate_axis", "_apply_preview"))
    assert "if hinge_states and not sliding_global_projection:" in preview
    assert "_apply_solved_two_bone_fk_pose(" in preview
    assert '_RIGPED_LOCAL_AXES["Y"]' in preview
    assert "sliding_global_projection_diagnostics=projection_diagnostics" in _source(
        _method("BAW_OT_rigped_direct_rotate_axis", "modal")
    )
    assert "_sliding_global_projected_dofs: tuple[tuple[str, float, float], ...]" in SOURCE


def test_rc1_rotation_vector_is_continuous_across_pi() -> None:
    rotation_vector = _evaluate_function(
        "_quaternion_rotation_vector_components",
        {"atan2": math.atan2, "sqrt": math.sqrt},
    )
    assert callable(rotation_vector)

    def axis_quaternion(angle: float) -> tuple[float, float, float, float]:
        return (math.cos(angle * 0.5), math.sin(angle * 0.5), 0.0, 0.0)

    before = rotation_vector(axis_quaternion(math.pi - 0.01))
    after = rotation_vector(axis_quaternion(math.pi + 0.01))
    assert isinstance(before, tuple) and isinstance(after, tuple)
    assert math.isclose(before[0], math.pi - 0.01, abs_tol=1e-9)
    assert math.isclose(after[0], math.pi + 0.01, abs_tol=1e-9)
    assert math.isclose(after[0] - before[0], 0.02, abs_tol=1e-9)


def test_rc1_damped_projection_is_proportional_and_bounded_near_singularity() -> None:
    solve_step = _evaluate_function("_damped_two_axis_rotation_step", {})
    assert callable(solve_step)
    small = solve_step(0.1, 0.0, 0.0, 0.05)
    doubled = solve_step(0.2, 0.0, 0.0, 0.05)
    assert math.isclose(doubled[0], small[0] * 2.0, rel_tol=1e-10)
    assert math.isclose(doubled[1], small[1] * 2.0, abs_tol=1e-10)

    near_singular = solve_step(0.8378, 0.8378, 0.999999, 0.3)
    assert all(math.isfinite(value) for value in near_singular)
    assert max(abs(value) for value in near_singular) < 0.5


def test_rc1_sliding_global_projection_is_continuous_for_small_axes_and_collinear_limbs() -> None:
    projection = _evaluate_function(
        "_project_global_rotation_to_sliding_lower_dofs",
        _projection_test_namespace(),
    )

    def make_session(start_roll: float = 0.0) -> tuple[object, object]:
        pose_bone = _FakePoseBone("Calf.L")
        session = _FakeSlidingSession(
            start_rotation=_FakeQuaternion.from_axis_angle(
                (1.0, 0.0, 0.0),
                math.pi / 2.0 - 1e-5,
            )
        )
        state = _FakeRotateState(pose_bone, (0.0, start_roll))
        return session, state

    session, state = make_session()
    samples = {}
    for axis_name, axis in (
        ("X+", (1.0, 0.0, 0.0)),
        ("X-", (-1.0, 0.0, 0.0)),
        ("Y+", (0.0, 1.0, 0.0)),
        ("Y-", (0.0, -1.0, 0.0)),
    ):
        result = projection(session, state, axis, 0.04)
        values = (result.swivel_angle, result.roll_angle, result.residual_radians)
        assert all(math.isfinite(value) for value in values), axis_name
        assert abs(result.swivel_angle) < 0.1, axis_name
        assert abs(result.roll_angle) < 0.1, axis_name
        samples[axis_name] = result

    for positive, negative in (("X+", "X-"), ("Y+", "Y-")):
        for field in ("swivel_angle", "roll_angle"):
            assert abs(
                getattr(samples[positive], field) - getattr(samples[negative], field)
            ) < 0.2

    # The frozen sign/branch is stable for a nearly collinear swivel and roll
    # generator pair; unreachable input remains in the residual.
    assert all(0.0 <= result.residual_radians <= math.pi for result in samples.values())
    assert all(0.0 <= result.conditioning < 0.01 for result in samples.values())


def test_rc1_sliding_global_projection_obeys_existing_roll_limit() -> None:
    projection = _evaluate_function(
        "_project_global_rotation_to_sliding_lower_dofs",
        _projection_test_namespace(),
    )
    session = _FakeSlidingSession(start_rotation=_FakeQuaternion.identity())
    state = _FakeRotateState(_FakePoseBone("Calf.L"), (0.0, 0.9))
    result = projection(session, state, (0.0, 1.0, 0.0), 0.5)
    assert result.roll_angle <= 0.100001
    assert result.roll_angle >= -1.900001
    assert result.residual_radians > 0.3


def test_rc1_builder_forearm_roll_comes_from_chain_geometry_with_safe_fallback() -> None:
    assert 'str(entry.semantic_key) == "awb.forearm"' in BUILDER_SOURCE
    assert "bend_normal = upper.cross(lower)" in BUILDER_SOURCE
    assert "bend_normal.length > 1e-5" in BUILDER_SOURCE
    assert "local_x = -bend_normal.normalized()" in BUILDER_SOURCE
    assert "local_x = local_x - direction * float(local_x.dot(direction))" in BUILDER_SOURCE
    assert "local_z = local_x.cross(direction)" in BUILDER_SOURCE
    assert "bone.align_roll(local_z)" in BUILDER_SOURCE
    assert "bone.align_roll(reference)" in BUILDER_SOURCE


def test_rc1_fit_commit_canonicalizes_forearm_targets_as_column_basis() -> None:
    assert "def _canonical_forearm_target_orientation(" in FIT_COMMIT_SOURCE
    assert "if bend_normal.length <= 1e-5:" in FIT_COMMIT_SOURCE
    assert "return forearm_target.orientation" in FIT_COMMIT_SOURCE
    assert "local_x = -bend_normal.normalized()" in FIT_COMMIT_SOURCE
    assert "local_x = local_x - local_y * float(local_x.dot(local_y))" in FIT_COMMIT_SOURCE
    assert "local_z = local_x.cross(local_y)" in FIT_COMMIT_SOURCE
    assert "(local_x.x, local_y.x, local_z.x)" in FIT_COMMIT_SOURCE
    assert "(local_x.y, local_y.y, local_z.y)" in FIT_COMMIT_SOURCE
    assert "(local_x.z, local_y.z, local_z.z)" in FIT_COMMIT_SOURCE
    assert 'if semantic_key == "awb.forearm":' in FIT_COMMIT_SOURCE
    assert "forearm_target_names.add(str(bone.name))" in FIT_COMMIT_SOURCE
    assert "FIT_F4_FOREARM_FRAME_PARENT_MISSING" in FIT_COMMIT_SOURCE
