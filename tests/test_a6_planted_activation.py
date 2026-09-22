from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTACT_UI_PATH = ROOT / "extension" / "blender_animation_workbench" / "phase4_contact_ui.py"
CONTACT_PATH = ROOT / "extension" / "blender_animation_workbench" / "phase4_contact_authoring.py"
PIVOT_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_ik_pivot_overlay.py"
TRANSFORM_PATH = ROOT / "extension" / "blender_animation_workbench" / "rigped_transform.py"
DRAWING_PATH = ROOT / "extension" / "blender_animation_workbench" / "trackbar_drawing.py"


def _source(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _function(path: Path, name: str) -> str:
    source, tree = _source(path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"missing function {name} in {path}")


def _class(path: Path, name: str) -> str:
    source, tree = _source(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"missing class {name} in {path}")


def test_a6_public_contact_cycle_defaults_to_free_sliding_planted() -> None:
    enabled = _function(CONTACT_UI_PATH, "contact_enabled_types")
    assert "ContactKeyType.FREE" in enabled
    assert "ContactKeyType.SLIDING" in enabled
    assert "ContactKeyType.PLANTED" in enabled
    assert "enabled.append(ContactKeyType.PLANTED)" in enabled
    assert "return default_cycle" in enabled
    assert "return tuple(enabled) or default_cycle" in enabled

    command = _function(CONTACT_PATH, "execute_contact_command")
    signature = command.split(") -> ContactAuthoringResult:", 1)[0]
    assert "ContactKeyType.FREE" in signature
    assert "ContactKeyType.SLIDING" in signature
    assert "ContactKeyType.PLANTED" in signature


def test_a6_planted_preference_is_exposed_and_default_on() -> None:
    preferences = _class(CONTACT_UI_PATH, "BAW_AP_preferences")
    assert 'contact_planted_enabled: BoolProperty(' in preferences
    planted = preferences.split('contact_planted_enabled: BoolProperty(', 1)[1].split(")", 1)[0]
    assert 'description="Include Planted in the Contact cycle"' in planted
    assert "default=True" in planted

    draw = _function(CONTACT_UI_PATH, "draw")
    assert 'row.prop(self, "contact_planted_enabled", toggle=True)' in draw


def test_a6_authoritative_red_pivot_includes_sliding_and_planted() -> None:
    source = PIVOT_PATH.read_text(encoding="utf-8")
    assert "int(ContactStateValue.SLIDING)" in source
    assert "int(ContactStateValue.PLANTED)" in source
    assert "_authoritative_contact_state_at_frame(" in source


def test_a6_planted_track_bar_color_remains_light_blue() -> None:
    source = DRAWING_PATH.read_text(encoding="utf-8")
    assert "ContactKeyType.PLANTED: (0.36, 0.72, 0.98, 1.0)" in source


def test_a6_planted_replay_skips_public_fk_joint_reclamp() -> None:
    clamp = _function(TRANSFORM_PATH, "apply_rigped_joint_limits_to_rig")
    assert "ContactKeyType.SLIDING" in clamp
    assert "ContactKeyType.PLANTED" in clamp
    assert "contact_type in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}" in clamp


def test_a6_planted_replay_reuses_straight_limb_stall_seed_without_new_authority() -> None:
    replay = _function(TRANSFORM_PATH, "_sync_rigped_sliding_replay_display")
    assert "contact_type is ContactKeyType.PLANTED" in replay
    assert "_seed_stalled_sliding_native_ik(" in replay
    assert "contact_type not in {ContactKeyType.SLIDING, ContactKeyType.PLANTED}" in replay


def test_a6_planted_move_remains_fail_closed() -> None:
    source = TRANSFORM_PATH.read_text(encoding="utf-8")
    assert "Planted semantic Move is fail-closed; release or slide the Contact first." in source
