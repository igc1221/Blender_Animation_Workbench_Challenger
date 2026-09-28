from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "extension" / "blender_animation_workbench"
REPLAY_PATH = PACKAGE_ROOT / "debug_replay.py"


def _load_pure_modules():
    package_name = "_awb_rc2_w3_pure"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules[package_name] = package

    modules = {}
    for module_name in ("phase4_contact_model", "debug_replay_payload"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            PACKAGE_ROOT / f"{module_name}.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)
        modules[module_name] = module
    return modules["phase4_contact_model"], modules["debug_replay_payload"]


CONTACT_MODEL, PAYLOAD = _load_pure_modules()


def _pose_state(
    *,
    object_name: str = "AWB_Rigped",
    bone_name: str = "ForeArm.L",
    location=(1, 2, 3),
    rotation_mode: str = "QUATERNION",
    rotation_property: str = "rotation_quaternion",
    rotation=(1, 0, 0, 0),
):
    return {
        "object": object_name,
        "bone": bone_name,
        "location": location,
        "rotation_mode": rotation_mode,
        "rotation_property": rotation_property,
        "rotation": rotation,
    }


def _sliding_state(*, mapping_id: str = "map-arm-l"):
    return {
        "mapping_id": mapping_id,
        "solver_object": "AWB_Rigped",
        "solver_bone": "MCH_ForeArm.L",
        "pole_angle": 0,
        "ik_influence": 1,
        "ik_mute": False,
        "terminal_ik_influence": 1,
        "terminal_ik_mute": False,
        "feedback_mutes": [False, True, False],
        "hinge_settings": [True, False, True, False, True, False, 0, 1, 2, 3, 4, 5],
        "contact_state": 2,
    }


def _result_payload(*, pose_states=None, sliding=None):
    return {
        "schema": PAYLOAD.DIRECT_ROTATE_RECORDED_RESULT_SCHEMA,
        "pose_states": [_pose_state()] if pose_states is None else pose_states,
        "sliding": [] if sliding is None else sliding,
    }


def test_w3_pure_modules_import_with_bpy_explicitly_blocked() -> None:
    script = f"""
import builtins
import importlib.util
import sys
import types
from pathlib import Path

root = Path({str(PACKAGE_ROOT)!r})
real_import = builtins.__import__


def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "bpy" or name.startswith("bpy."):
        raise AssertionError("pure replay payload import touched bpy")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = blocked_import
package_name = "_awb_rc2_w3_subprocess"
package = types.ModuleType(package_name)
package.__path__ = [str(root)]
sys.modules[package_name] = package
for module_name in ("phase4_contact_model", "debug_replay_payload"):
    spec = importlib.util.spec_from_file_location(
        f"{{package_name}}.{{module_name}}",
        root / f"{{module_name}}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_w3_contact_normalizers_preserve_sequence_and_diagnostic_contract() -> None:
    assert PAYLOAD.recorded_contact_type(
        "SLIDING",
        field_name="contact_type",
    ) is CONTACT_MODEL.ContactKeyType.SLIDING

    with pytest.raises(
        RuntimeError,
        match=r"contact_type='BROKEN' is not a valid Contact type\.$",
    ):
        PAYLOAD.recorded_contact_type("BROKEN", field_name="contact_type")

    list_pairs = {
        "mapping_contact_types": [["a", "FREE"], ["b", "SLIDING"]],
    }
    dict_pairs = {
        "mapping_contact_types": {"a": "FREE", "b": "SLIDING"},
    }
    expected = (
        ("a", CONTACT_MODEL.ContactKeyType.FREE),
        ("b", CONTACT_MODEL.ContactKeyType.SLIDING),
    )
    assert PAYLOAD.recorded_mapping_contact_types(list_pairs) == expected
    assert PAYLOAD.recorded_mapping_contact_types(dict_pairs) == expected

    with pytest.raises(RuntimeError, match="missing/duplicate mapping id 'a'"):
        PAYLOAD.recorded_mapping_contact_types(
            {"mapping_contact_types": [("a", "FREE"), ("a", "SLIDING")]}
        )
    with pytest.raises(RuntimeError, match="must contain exact"):
        PAYLOAD.recorded_mapping_contact_types(
            {"mapping_contact_types": [("a", "FREE", "extra")]}
        )

    assert PAYLOAD.recorded_direct_binding_ids(
        {"direct_binding_ids": ["body:Root", "body:COM"]}
    ) == ("body:Root", "body:COM")
    with pytest.raises(TypeError, match="must be a sequence, not a string"):
        PAYLOAD.recorded_direct_binding_ids({"direct_binding_ids": "body:Root"})
    with pytest.raises(RuntimeError, match="empty or duplicate binding id"):
        PAYLOAD.recorded_direct_binding_ids(
            {"direct_binding_ids": ["body:Root", "body:Root"]}
        )


def test_w3_pose_payload_normalizes_with_legacy_value_semantics() -> None:
    row = PAYLOAD.validate_recorded_pose_state_payload(
        _pose_state(
            location=["1", 2, 3.0],
            rotation=["1", 0, 0.0, 0],
        )
    )
    assert row["key"] == ("AWB_Rigped", "ForeArm.L")
    assert row["location"] == (1.0, 2.0, 3.0)
    assert row["rotation"] == (1.0, 0.0, 0.0, 0.0)

    euler = PAYLOAD.validate_recorded_pose_state_payload(
        _pose_state(
            rotation_mode="XYZ",
            rotation_property="rotation_euler",
            rotation=[0, 1, 2],
        )
    )
    assert euler["rotation"] == (0.0, 1.0, 2.0)


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (lambda row: row.pop("rotation"), RuntimeError, "pose state is incomplete"),
        (lambda row: row.__setitem__("bone", ""), RuntimeError, "empty pose identity"),
        (
            lambda row: row.__setitem__("location", [1, 2]),
            RuntimeError,
            "invalid location",
        ),
        (
            lambda row: row.__setitem__("rotation_property", "rotation_euler"),
            RuntimeError,
            "inconsistent rotation representation",
        ),
        (
            lambda row: row.__setitem__("rotation", [1, 0, 0]),
            RuntimeError,
            "invalid rotation",
        ),
    ],
)
def test_w3_pose_payload_malformed_failures_are_pure_and_exact_class(
    mutation,
    error_type,
    message,
) -> None:
    row = _pose_state()
    mutation(row)
    with pytest.raises(error_type, match=message):
        PAYLOAD.validate_recorded_pose_state_payload(row)


def test_w3_direct_rotate_structure_rejects_duplicate_and_missing_coverage() -> None:
    duplicate_pose = _result_payload(
        pose_states=[_pose_state(), _pose_state()],
    )
    with pytest.raises(RuntimeError, match="duplicate pose identities"):
        PAYLOAD.validate_direct_rotate_recorded_result_payload(duplicate_pose)

    duplicate_mapping = _result_payload(
        sliding=[_sliding_state(mapping_id="m"), _sliding_state(mapping_id="m")],
    )
    with pytest.raises(RuntimeError, match="duplicate Sliding mappings"):
        PAYLOAD.validate_direct_rotate_recorded_result_payload(duplicate_mapping)

    incomplete = _sliding_state()
    incomplete.pop("terminal_ik_influence")
    with pytest.raises(RuntimeError, match="Sliding state is incomplete"):
        PAYLOAD.validate_direct_rotate_recorded_result_payload(
            _result_payload(sliding=[incomplete])
        )


def test_w3_sliding_payload_preserves_feedback_hinge_numeric_and_mute_errors() -> None:
    bad_feedback = _sliding_state()
    bad_feedback["feedback_mutes"] = [False, 0, False]
    with pytest.raises(RuntimeError, match="feedback coverage mismatch"):
        PAYLOAD.validate_direct_rotate_recorded_result_payload(
            _result_payload(sliding=[bad_feedback])
        )

    bad_hinge = _sliding_state()
    bad_hinge["hinge_settings"] = [True] * 11
    with pytest.raises(RuntimeError, match="hinge settings are incomplete"):
        PAYLOAD.validate_direct_rotate_recorded_result_payload(
            _result_payload(sliding=[bad_hinge])
        )

    bad_numeric = _sliding_state()
    bad_numeric["pole_angle"] = "not-a-number"
    with pytest.raises(RuntimeError, match="Sliding numeric state is invalid"):
        PAYLOAD.validate_direct_rotate_recorded_result_payload(
            _result_payload(sliding=[bad_numeric])
        )

    bad_mute = _sliding_state()
    bad_mute["ik_mute"] = 0
    with pytest.raises(TypeError, match="Sliding mute state is invalid"):
        PAYLOAD.validate_direct_rotate_recorded_result_payload(
            _result_payload(sliding=[bad_mute])
        )


def test_w3_normalized_rows_are_the_single_live_binding_input() -> None:
    pose_rows, sliding_rows = PAYLOAD.validate_direct_rotate_recorded_result_payload(
        _result_payload(sliding=[_sliding_state()])
    )
    assert pose_rows[0]["object"] == "AWB_Rigped"
    assert pose_rows[0]["bone"] == "ForeArm.L"
    assert sliding_rows[0]["mapping_id"] == "map-arm-l"
    assert sliding_rows[0]["pole_angle"] == 0.0
    assert sliding_rows[0]["ik_influence"] == 1.0
    assert sliding_rows[0]["terminal_ik_influence"] == 1.0
    assert sliding_rows[0]["contact_state"] == 2.0
    assert sliding_rows[0]["hinge_settings"][6:] == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

    replay_source = REPLAY_PATH.read_text(encoding="utf-8")
    assert "validate_direct_rotate_recorded_result_payload(payload)" in replay_source
    assert "_bind_recorded_pose_state(row) for row in primitive_pose_rows" in replay_source
    assert "for row in primitive_sliding_rows:" in replay_source
