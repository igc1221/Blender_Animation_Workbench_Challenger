from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
PACKAGE_DIR = ROOT / "extension" / "blender_animation_workbench"
PACKAGE_NAME = "blender_animation_workbench_binding_test"


class _Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _Key:
    def __init__(self, frame: float, value: float):
        self.co = _Vector(frame, value)
        self.handle_left = _Vector(frame - 0.25, value)
        self.handle_right = _Vector(frame + 0.25, value)
        self.handle_left_type = "FREE"
        self.handle_right_type = "FREE"
        self.interpolation = "BEZIER"
        self.easing = "AUTO"
        self.amplitude = 0.0
        self.back = 0.0
        self.period = 0.0
        self.type = "KEYFRAME"
        self.select_control_point = False
        self.select_left_handle = False
        self.select_right_handle = False


class _Points(list):
    def remove(self, value, *, fast=False):
        super().remove(value)

    def insert(self, frame, value, *, options=None, keyframe_type="KEYFRAME"):
        key = _Key(frame, value)
        key.type = keyframe_type
        self.append(key)
        return key


class _FCurve:
    def __init__(self, data_path: str, *keys: _Key, array_index: int = 0):
        self.data_path = data_path
        self.array_index = array_index
        self.keyframe_points = _Points(keys)
        self.update_count = 0

    def update(self):
        self.update_count += 1


class _Bag:
    _next_pointer = 1000

    def __init__(self, *fcurves: _FCurve):
        self.fcurves = list(fcurves)
        self._pointer = _Bag._next_pointer
        _Bag._next_pointer += 1

    def as_pointer(self):
        return self._pointer


class _Owner:
    _next_pointer = 2000

    def __init__(self, bag: _Bag, token):
        self.bag = bag
        self.binding_token = token
        self._pointer = _Owner._next_pointer
        _Owner._next_pointer += 1
        self.update_count = 0

    def as_pointer(self):
        return self._pointer

    def update_tag(self, *, refresh):
        self.update_count += 1


class _Context:
    def __init__(self, semantic=None):
        self.control_context = SimpleNamespace(semantic=semantic)
        self.resolved_controls = []
        self.scene = SimpleNamespace(frame_current=0, frame_set=lambda _frame: None)


class _Contributor:
    def __init__(self, owner: _Owner, frame: float, data_path="location", array_index=0):
        self.frame = float(frame)
        self.channel = SimpleNamespace(
            resolved=SimpleNamespace(owner_object=owner),
            data_path=data_path,
            array_index=array_index,
        )


def _load_module():
    package = ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[PACKAGE_NAME] = package

    key_edit_path = PACKAGE_DIR / "trackbar_key_edit.py"
    key_edit_name = f"{PACKAGE_NAME}.trackbar_key_edit"
    key_edit_spec = spec_from_file_location(key_edit_name, key_edit_path)
    assert key_edit_spec is not None and key_edit_spec.loader is not None
    key_edit = module_from_spec(key_edit_spec)
    sys.modules[key_edit_name] = key_edit
    key_edit_spec.loader.exec_module(key_edit)

    semantic_adapter = ModuleType(f"{PACKAGE_NAME}.semantic_adapter")
    semantic_adapter.active_control_for_context = lambda _context: None
    semantic_adapter.assigned_channelbag = lambda owner: owner.bag
    semantic_adapter.channel_binding_token = lambda owner: owner.binding_token
    semantic_adapter.control_context_for_context = lambda context: context.control_context
    semantic_adapter.group_animation_owners = lambda _controls: ()
    semantic_adapter.matches_control_path = lambda _path, _prefix: True
    semantic_adapter.trackbar_controls_for_context = lambda context: context.resolved_controls
    sys.modules[semantic_adapter.__name__] = semantic_adapter

    semantic_model = ModuleType(f"{PACKAGE_NAME}.semantic_model")
    semantic_model.AWBControl = object
    sys.modules[semantic_model.__name__] = semantic_model

    semantic_query = ModuleType(f"{PACKAGE_NAME}.semantic_query")
    semantic_query.query_keys = lambda control_context: control_context.semantic
    semantic_query.contributors_for_frames = lambda semantic, frames: tuple(
        contributor
        for contributor in semantic.contributors
        if any(abs(contributor.frame - float(frame)) <= 1e-4 for frame in frames)
    )
    semantic_query.resolve_contributor_key = lambda _contributor, epsilon=1e-4: None
    semantic_query.semantic_keys_for_context = lambda _context: ()
    sys.modules[semantic_query.__name__] = semantic_query

    model_path = PACKAGE_DIR / "trackbar_model.py"
    model_name = f"{PACKAGE_NAME}.trackbar_model"
    model_spec = spec_from_file_location(model_name, model_path)
    assert model_spec is not None and model_spec.loader is not None
    model = module_from_spec(model_spec)
    sys.modules[model_name] = model
    model_spec.loader.exec_module(model)
    return model


def _curve_state(curve: _FCurve):
    return [
        (
            round(float(key.co.x), 4),
            float(key.co.y),
            bool(key.select_control_point),
            str(key.interpolation),
        )
        for key in sorted(curve.keyframe_points, key=lambda item: item.co.x)
    ]


def test_shared_channelbag_uses_one_preview_journal_and_cancel_restores_collision():
    model = _load_module()
    source = _Key(0.0, 1.0)
    source.select_control_point = True
    destination = _Key(10.0, 9.0)
    curve = _FCurve("location", source, destination)
    shared_bag = _Bag(curve)
    owner_a = _Owner(shared_bag, (1, 2, 3, 4))
    owner_b = _Owner(shared_bag, (1, 2, 3, 4))
    semantic = SimpleNamespace(
        contributors=(
            _Contributor(owner_a, 0.0),
            _Contributor(owner_b, 0.0),
        )
    )
    context = _Context(semantic)
    model.clear_key_selection_for_context = lambda _context: False
    model.select_key_frame_for_context = lambda *_args, **_kwargs: True

    transaction = model.snapshot_multi_key_preview_for_context(
        context,
        [0.0],
        mode="MOVE",
    )
    assert transaction is not None
    assert len(transaction.items) == 1
    target, _owner_transaction = transaction.items[0]
    assert len(target.owner_bindings) == 2

    assert model.update_multi_key_preview_for_context(context, transaction, 10)
    assert _curve_state(curve) == [(10.0, 1.0, True, "BEZIER")]

    assert model.restore_multi_key_preview_for_context(context, transaction)
    assert _curve_state(curve) == [
        (0.0, 1.0, True, "BEZIER"),
        (10.0, 9.0, False, "BEZIER"),
    ]


def test_shared_binding_survives_fresh_channelbag_python_wrappers():
    model = _load_module()
    source = _Key(0.0, 2.0)
    source.select_control_point = True
    destination = _Key(10.0, 8.0)
    curve = _FCurve("location", source, destination)
    shared_bag = _Bag(curve)
    owner_a = _Owner(shared_bag, (101, 102, 103, 104))
    owner_b = _Owner(shared_bag, (101, 102, 103, 104))
    semantic = SimpleNamespace(
        contributors=(
            _Contributor(owner_a, 0.0),
            _Contributor(owner_b, 0.0),
        )
    )
    context = _Context(semantic)

    def fresh_channelbag(owner):
        return SimpleNamespace(fcurves=owner.bag.fcurves)

    assert fresh_channelbag(owner_a) is not fresh_channelbag(owner_a)
    model.assigned_channelbag = fresh_channelbag
    model.clear_key_selection_for_context = lambda _context: False
    model.select_key_frame_for_context = lambda *_args, **_kwargs: True

    transaction = model.snapshot_multi_key_preview_for_context(
        context,
        [0.0],
        mode="MOVE",
    )
    assert transaction is not None
    assert len(transaction.items) == 1

    assert model.update_multi_key_preview_for_context(context, transaction, 10)
    assert _curve_state(curve) == [(10.0, 2.0, True, "BEZIER")]

    assert model.restore_multi_key_preview_for_context(context, transaction)
    assert _curve_state(curve) == [
        (0.0, 2.0, True, "BEZIER"),
        (10.0, 8.0, False, "BEZIER"),
    ]


def test_active_snapshot_binding_change_is_complete_noop_on_new_channelbag():
    model = _load_module()
    curve_a = _FCurve("location", _Key(0.0, 1.0))
    bag_a = _Bag(curve_a)
    owner = _Owner(bag_a, (10, 11, 12, 13))
    context = _Context(SimpleNamespace(contributors=()))
    model._target_channelbags_for_context = lambda _context: [(owner, None, bag_a)]

    snapshots = model.snapshot_active_key_state_for_context(context)
    curve_b = _FCurve("location", _Key(0.0, 90.0), _Key(50.0, 99.0))
    curve_b.keyframe_points[1].select_control_point = True
    bag_b = _Bag(curve_b)
    owner.bag = bag_b
    owner.binding_token = (20, 21, 22, 23)
    before = _curve_state(curve_b)

    assert not model.restore_active_key_state_for_context(context, snapshots)
    assert _curve_state(curve_b) == before


def test_key_property_target_rejects_reassigned_binding_before_any_write():
    model = _load_module()
    curve_a = _FCurve("location", _Key(0.0, 1.0))
    bag_a = _Bag(curve_a)
    owner = _Owner(bag_a, (30, 31, 32, 33))
    target = model.KeyPropertyTarget(
        owner_object=owner,
        control_id="object",
        data_path_prefix=None,
        data_path="location",
        array_index=0,
        frame=0.0,
        binding_token=owner.binding_token,
    )

    curve_b = _FCurve("location", _Key(0.0, 90.0), _Key(50.0, 99.0))
    bag_b = _Bag(curve_b)
    owner.bag = bag_b
    owner.binding_token = (40, 41, 42, 43)
    context = _Context(SimpleNamespace(contributors=()))
    context.resolved_controls = [
        SimpleNamespace(
            owner_object=owner,
            control=SimpleNamespace(control_id="object"),
            data_path_prefix=None,
        )
    ]
    before = _curve_state(curve_b)

    with pytest.raises(RuntimeError, match="no longer resolves"):
        model.apply_key_properties_for_context(
            context,
            (target,),
            interpolation="CONSTANT",
        )
    assert _curve_state(curve_b) == before


def test_preview_batch_preflights_all_bindings_before_mutating_any_curve():
    model = _load_module()
    curve_a = _FCurve("location", _Key(0.0, 1.0))
    curve_b = _FCurve("location", _Key(0.0, 2.0))
    owner_a = _Owner(_Bag(curve_a), (50, 51, 52, 53))
    owner_b = _Owner(_Bag(curve_b), (60, 61, 62, 63))
    semantic = SimpleNamespace(
        contributors=(
            _Contributor(owner_a, 0.0),
            _Contributor(owner_b, 0.0),
        )
    )
    context = _Context(semantic)
    selection_calls = []
    model.clear_key_selection_for_context = lambda _context: selection_calls.append("clear")
    model.select_key_frame_for_context = (
        lambda *_args, **_kwargs: selection_calls.append("select")
    )

    transaction = model.snapshot_multi_key_preview_for_context(
        context,
        [0.0],
        mode="MOVE",
    )
    assert transaction is not None
    assert len(transaction.items) == 2

    owner_b.bag = _Bag(_FCurve("location", _Key(0.0, 200.0)))
    owner_b.binding_token = (70, 71, 72, 73)
    before_a = _curve_state(curve_a)

    assert not model.update_multi_key_preview_for_context(context, transaction, 5)
    assert _curve_state(curve_a) == before_a
    assert selection_calls == []

    assert not model.restore_multi_key_preview_for_context(context, transaction)
    assert selection_calls == []


def test_range_restore_binding_mismatch_does_not_touch_current_selection():
    model = _load_module()
    curve = _FCurve("location", _Key(0.0, 1.0), _Key(10.0, 2.0))
    owner = _Owner(_Bag(curve), (80, 81, 82, 83))
    target = model.TrackbarOwnerEditTarget(
        owner_object=owner,
        binding_token=owner.binding_token,
        curve_keys=(("location", 0),),
        owner_bindings=((owner, owner.binding_token),),
        binding_key=owner.binding_token[1:],
    )
    transaction = model.ContextSelectionRangeScalePreviewTransaction(
        (0.0, 10.0),
        pivot_frame=0,
        source_handle_frame=10,
        items=((target, None),),
    )
    selection_calls = []
    model.clear_key_selection_for_context = lambda _context: selection_calls.append("clear")
    model.select_key_frame_for_context = (
        lambda *_args, **_kwargs: selection_calls.append("select")
    )

    owner.bag = _Bag(_FCurve("location", _Key(0.0, 100.0)))
    owner.binding_token = (90, 91, 92, 93)

    assert not model.restore_selection_range_scale_preview_for_context(
        _Context(SimpleNamespace(contributors=())),
        transaction,
    )
    assert selection_calls == []
