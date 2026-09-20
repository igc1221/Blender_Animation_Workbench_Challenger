import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = (
    Path(__file__).parents[1]
    / "extension"
    / "blender_animation_workbench"
    / "ui_language.py"
)

spec = spec_from_file_location("baw_ui_language_tests", MODULE_PATH)
assert spec is not None and spec.loader is not None
ui_language = module_from_spec(spec)
sys.modules[spec.name] = ui_language
spec.loader.exec_module(ui_language)


def test_english_is_the_default_language():
    assert ui_language.DEFAULT_LANGUAGE == "EN"
    assert ui_language.text("rigped.new") == "New Rigped"
    assert ui_language.text("tooltip.rigped.reset_fit") == "Restore the state from before this Fit began"


def test_korean_pack_contains_matching_ui_and_tooltip_text():
    assert ui_language.text("rigped.new", language="KO") == "새 Rigped"
    assert ui_language.text("rigped.select", language="KO") == "선택"
    assert ui_language.text("tooltip.rigped.new", language="KO") == "새 Rigped를 생성합니다"
    assert ui_language.text("tooltip.rigped.reset_fit", language="KO") == "이번 Fit 시작 전 상태로 되돌립니다"


def test_context_language_switches_text_without_changing_the_default():
    context = SimpleNamespace(scene=SimpleNamespace(baw_ui_language="KO"))
    assert ui_language.text("rigped.animate", context) == "애니메이션"
    assert ui_language.text("rigped.animate") == "Animate"


def test_unknown_language_and_missing_translation_fail_back_to_english_or_key():
    assert ui_language.text("rigped.fit", language="XX") == "Fit"
    assert ui_language.text("missing.key", language="KO") == "missing.key"


def test_formatted_language_pack_text_uses_selected_language():
    assert ui_language.text("rigped.create_preview.height", language="EN", height=1.8) == "Rigped Height  1.800"
    assert ui_language.text("rigped.create_preview.height", language="KO", height=1.8) == "Rigped 높이  1.800"
