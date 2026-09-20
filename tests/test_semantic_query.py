import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

PACKAGE_PATH = (
    Path(__file__).parents[1] / "extension" / "blender_animation_workbench"
)
PACKAGE_NAME = "baw_semantic_query_tests"

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
semantic_query = _load_module("semantic_query")
trackbar_key_edit = _load_module("trackbar_key_edit")
trackbar_model = _load_module("trackbar_model")


class FakeRNA:
    def __init__(self, name: str, pointer: int):
        self.name = name
        self._pointer = pointer

    def as_pointer(self) -> int:
        return self._pointer

    def get(self, _key: str, default=None):
        return default


class FakeObject(FakeRNA):
    def __init__(self, name: str, pointer: int, *, selected: bool = True):
        super().__init__(name, pointer)
        self.type = "MESH"
        self.selected = selected
        self.animation_data = None

    def select_get(self) -> bool:
        return self.selected


class FakePoint:
    def __init__(
        self,
        frame: float,
        *,
        selected: bool = False,
        interpolation: str = "BEZIER",
    ):
        self.co = SimpleNamespace(x=float(frame))
        self.select_control_point = selected
        self.interpolation = interpolation


class FakeFCurve:
    def __init__(self, data_path: str, array_index: int, points):
        self.data_path = data_path
        self.array_index = array_index
        self.keyframe_points = tuple(points)


def _install_fake_anim_utils(monkeypatch, channelbags_by_animation_data):
    package = ModuleType("bpy_extras")
    module = ModuleType("bpy_extras.anim_utils")

    def get_channelbag(animation_data):
        return channelbags_by_animation_data.get(id(animation_data))

    module.animdata_get_channelbag_for_assigned_slot = get_channelbag
    package.anim_utils = module
    monkeypatch.setitem(sys.modules, "bpy_extras", package)
    monkeypatch.setitem(sys.modules, "bpy_extras.anim_utils", module)


def _attach_channelbag(obj, curves):
    obj.animation_data = SimpleNamespace(action=object())
    return SimpleNamespace(fcurves=tuple(curves))


def _object_context(obj):
    return SimpleNamespace(
        mode="OBJECT",
        active_object=obj,
        selected_objects=[obj] if obj.selected else [],
        active_pose_bone=None,
        selected_pose_bones=[],
    )


def test_query_preserves_exact_subframes_selected_or_and_mixed_interpolation(monkeypatch):
    obj = FakeObject("Cube", 101)
    channelbag = _attach_channelbag(
        obj,
        [
            FakeFCurve(
                "location",
                0,
                [
                    FakePoint(10.0, interpolation="BEZIER"),
                    FakePoint(10.25, interpolation="LINEAR"),
                ],
            ),
            FakeFCurve(
                "location",
                1,
                [FakePoint(10.0, selected=True, interpolation="LINEAR")],
            ),
            FakeFCurve(
                "rotation_euler",
                2,
                [FakePoint(10.0, interpolation="BEZIER")],
            ),
            FakeFCurve(
                "scale",
                0,
                [FakePoint(10.5, interpolation="CONSTANT")],
            ),
            FakeFCurve(
                '["custom"]',
                0,
                [FakePoint(10.0, interpolation="BEZIER")],
            ),
        ],
    )
    _install_fake_anim_utils(monkeypatch, {id(obj.animation_data): channelbag})

    result = semantic_query.query_keys(
        semantic_adapter.control_context_for_context(_object_context(obj))
    )

    assert [key.frame for key in result.keys] == [10.0, 10.25, 10.5]
    first, second, third = result.keys
    assert first.channels == frozenset(
        {
            semantic_model.AWBChannel.POSITION,
            semantic_model.AWBChannel.ROTATION,
            semantic_model.AWBChannel.OTHER,
        }
    )
    assert first.selected is True
    assert first.interpolation == "MIXED"
    assert second.channels == frozenset({semantic_model.AWBChannel.POSITION})
    assert second.interpolation == "LINEAR"
    assert third.channels == frozenset({semantic_model.AWBChannel.SCALE})
    assert third.interpolation == "CONSTANT"
    assert len(result.contributors) == 6


def test_frame_aggregate_keeps_multi_control_contributors(monkeypatch):
    first = FakeObject("A", 201)
    second = FakeObject("B", 202)
    first_bag = _attach_channelbag(
        first,
        [FakeFCurve("location", 0, [FakePoint(20.0, selected=True)])],
    )
    second_bag = _attach_channelbag(
        second,
        [FakeFCurve("rotation_euler", 0, [FakePoint(20.0)])],
    )
    _install_fake_anim_utils(
        monkeypatch,
        {
            id(first.animation_data): first_bag,
            id(second.animation_data): second_bag,
        },
    )
    controls = (
        semantic_adapter._object_control(first),
        semantic_adapter._object_control(second),
    )
    query = semantic_query.query_keys(
        semantic_adapter.ControlContext(mode="OBJECT", controls=controls, active=controls[0])
    )

    aggregates = semantic_query.frame_aggregates(query)

    assert len(query.keys) == 2
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate.frame == 20.0
    assert [key.control.object_name for key in aggregate.contributors] == ["A", "B"]
    assert aggregate.channels == frozenset(
        {semantic_model.AWBChannel.POSITION, semantic_model.AWBChannel.ROTATION}
    )
    assert aggregate.selected is True


def test_contributor_expansion_filters_control_family_and_exact_time(monkeypatch):
    first = FakeObject("A", 301)
    second = FakeObject("B", 302)
    first_bag = _attach_channelbag(
        first,
        [
            FakeFCurve("location", 0, [FakePoint(20.0), FakePoint(20.25)]),
            FakeFCurve("scale", 0, [FakePoint(20.0)]),
        ],
    )
    second_bag = _attach_channelbag(
        second,
        [FakeFCurve("location", 0, [FakePoint(20.0)])],
    )
    _install_fake_anim_utils(
        monkeypatch,
        {
            id(first.animation_data): first_bag,
            id(second.animation_data): second_bag,
        },
    )
    first_resolved = semantic_adapter._object_control(first)
    second_resolved = semantic_adapter._object_control(second)
    query = semantic_query.query_keys(
        semantic_adapter.ControlContext(
            mode="OBJECT",
            controls=(first_resolved, second_resolved),
            active=first_resolved,
        )
    )

    first_key = frozenset({semantic_adapter.runtime_control_key(first_resolved)})
    position = frozenset({semantic_model.AWBChannel.POSITION})
    matched = semantic_query.contributors_for_frames(
        query,
        [20.0],
        control_keys=first_key,
        families=position,
    )

    assert len(matched) == 1
    assert matched[0].channel.resolved.control.object_name == "A"
    assert matched[0].channel.family == semantic_model.AWBChannel.POSITION
    assert matched[0].frame == 20.0
    assert semantic_query.contributors_for_frames(
        query,
        [20.0],
        control_keys=frozenset(),
    ) == ()
    assert semantic_query.contributors_for_frames(
        query,
        [20.0],
        families=frozenset(),
    ) == ()
    assert semantic_query.contributors_for_frames(
        query,
        [20.25005],
        control_keys=first_key,
        families=position,
    )[0].frame == 20.25


def test_context_compatibility_facade_keeps_active_deselected_excluded(monkeypatch):
    active = FakeObject("Active", 401, selected=False)
    selected = FakeObject("Selected", 402)
    selected_bag = _attach_channelbag(
        selected,
        [FakeFCurve("location", 0, [FakePoint(7.0)])],
    )
    _install_fake_anim_utils(
        monkeypatch,
        {id(selected.animation_data): selected_bag},
    )
    context = SimpleNamespace(
        mode="OBJECT",
        active_object=active,
        selected_objects=[selected],
        active_pose_bone=None,
        selected_pose_bones=[],
    )

    keys = semantic_query.semantic_keys_for_context(context)

    assert len(keys) == 1
    assert keys[0].control.object_name == "Selected"
    assert keys[0].frame == 7.0


def test_trackbar_edit_target_rejects_changed_action_slot_binding(monkeypatch):
    obj = FakeObject("Cube", 501)
    action = FakeRNA("Action", 502)
    slot = FakeRNA("Slot", 503)
    curve = FakeFCurve("location", 0, [FakePoint(10.0)])
    channelbag = FakeRNA("ChannelBag", 504)
    channelbag.fcurves = (curve,)
    obj.animation_data = SimpleNamespace(action=action, action_slot=slot)
    mapping = {id(obj.animation_data): channelbag}
    _install_fake_anim_utils(monkeypatch, mapping)

    targets = trackbar_model._edit_targets_for_frames(_object_context(obj), [10.0])

    assert len(targets) == 1
    target = targets[0]
    assert target.curve_keys == (("location", 0),)
    assert trackbar_model._fcurves_for_edit_target(target) == (curve,)

    obj.animation_data.action_slot = FakeRNA("OtherSlot", 505)

    assert trackbar_model._fcurves_for_edit_target(target) is None
