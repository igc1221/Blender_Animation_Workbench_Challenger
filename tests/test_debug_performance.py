from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from scripts import awb_debug_performance as performance
from tests.test_debug_native_escalation_schemas import SchemaValidator

_SHA_A = "a" * 64
_SHA_B = "b" * 64

_SCHEMA_DIR = Path(__file__).parents[1] / "docs" / "DEBUG" / "schemas"
_COMPARISON_SCHEMA_NAME = "AWB_PERFORMANCE_COMPARISON_V1.schema.json"
_RECEIPT_SCHEMA_NAME = "AWB_PERFORMANCE_RECEIPT_V1.schema.json"
_SCHEMA_MAP_KEYWORDS = ("properties", "$defs")
_SCHEMA_LIST_KEYWORDS = ("allOf", "anyOf", "oneOf")
_SCHEMA_VALUE_KEYWORDS = ("enum", "const", "default", "examples")


def _comparison_schema() -> dict:
    return json.loads((_SCHEMA_DIR / _COMPARISON_SCHEMA_NAME).read_text(encoding="utf-8"))


def _comparison_validator() -> SchemaValidator:
    path = _SCHEMA_DIR / _COMPARISON_SCHEMA_NAME
    return SchemaValidator(json.loads(path.read_text(encoding="utf-8")), path)


def _schema_keywords(node, found: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            if key in _SCHEMA_VALUE_KEYWORDS:
                continue
            if key in _SCHEMA_MAP_KEYWORDS:
                for subschema in value.values():
                    _schema_keywords(subschema, found)
            elif key in _SCHEMA_LIST_KEYWORDS:
                for subschema in value:
                    _schema_keywords(subschema, found)
            else:
                _schema_keywords(value, found)
    elif isinstance(node, list):
        for item in node:
            _schema_keywords(item, found)


def _mutate(base: dict, **changes) -> dict:
    clone = copy.deepcopy(base)
    clone.update(changes)
    return clone


def _replace_metric(comparison: dict, name: str, replacement: dict) -> dict:
    clone = copy.deepcopy(comparison)
    clone["metric_comparisons"][name] = replacement
    return clone


def _metrics(*, wall: float = 100.0, event_count: int = 10, cpu: float = 80.0):
    return {
        "wall_duration_ms": performance.measured(wall, "ms", "harness/run.json"),
        "process_cpu_duration_ms": performance.measured(cpu, "ms", "harness/run.json"),
        "event_count": performance.measured(event_count, "count", "harness/events.json"),
        "sample_count": performance.measured(9, "count", "harness/samples.json"),
        "dropped_sample_count": performance.measured(1, "count", "harness/samples.json"),
        "trace_bytes": performance.measured(512, "bytes", "artifacts/trace.jsonl"),
        "raw_input_bytes": performance.measured(128, "bytes", "artifacts/raw-input.jsonl"),
        "stdout_bytes": performance.measured(64, "bytes", "artifacts/stdout.log"),
    }


def _heartbeat(offset: float = 0.0):
    return {
        "status": "MEASURED",
        "sample_count": 10,
        "method": "EXPLICIT_HEARTBEAT_SAMPLE_SUMMARY",
        "evidence_ref": "harness/heartbeat.json",
        "interval_ms": {
            "min_ms": 9.0 + offset,
            "mean_ms": 10.0 + offset,
            "p50_ms": 10.0 + offset,
            "p95_ms": 12.0 + offset,
            "max_ms": 13.0 + offset,
        },
        "jitter_ms": {
            "min_ms": 0.0,
            "mean_ms": 0.5 + offset,
            "p50_ms": 0.4 + offset,
            "p95_ms": 1.0 + offset,
            "max_ms": 1.5 + offset,
        },
    }


def _duration_summary(scope: str = "CALLBACK", name: str = "draw", offset: float = 0.0):
    method = "EXPLICIT_CALLBACK_SAMPLES" if scope == "CALLBACK" else "EXPLICIT_LIFECYCLE_SAMPLES"
    return {
        "scope": scope,
        "name": name,
        "sample_count": 10,
        "total_ms": 45.0 + offset,
        "min_ms": 1.0 + offset,
        "mean_ms": 4.5 + offset,
        "p50_ms": 4.0 + offset,
        "p95_ms": 8.0 + offset,
        "max_ms": 10.0 + offset,
        "method": method,
        "evidence_ref": "harness/durations.json",
    }


def _receipt(
    role: str,
    *,
    sequence_index: int | None = None,
    flags: dict[str, bool] | None = None,
    allowed: list[str] | None = None,
    workload_class: str = "UI_INTERACTION",
    blender_identity: dict | None = None,
    run_config: dict | None = None,
    metrics: dict | None = None,
    heartbeat: dict | None = None,
    duration_summaries: list | None = None,
):
    if flags is None:
        flags = {
            "depsgraph_trace": False,
            "debug_trace": role == "INSTRUMENTED",
            "profile_callbacks": role == "INSTRUMENTED",
        }
    return performance.build_receipt(
        role=role,
        pair_id="PAIR-01",
        sequence_index=(
            sequence_index if sequence_index is not None else (10 if role == "CONTROL" else 11)
        ),
        blender_identity=blender_identity
        or {
            "binary_sha256": _SHA_A,
            "version": "5.2.1",
            "build_hash": "build-5.2.1-test",
        },
        scene_workload_identity={
            "scene_id": "scene-fixture-01",
            "workload_id": "viewport-draw-100-events",
            "workload_class": workload_class,
        },
        extension_source_identity={"extension_id": "awb", "source_sha256": _SHA_B},
        run_config=run_config or {"scene_seed": 42, "event_batch": 100},
        active_debug_flags=flags,
        allowed_instrumentation_toggles=(
            ["debug_trace", "profile_callbacks"] if allowed is None else allowed
        ),
        metrics=_metrics() if metrics is None else metrics,
        heartbeat=heartbeat if heartbeat is not None else _heartbeat(),
        duration_summaries=(
            duration_summaries if duration_summaries is not None else [_duration_summary()]
        ),
    )


def test_receipts_are_copied_and_comparison_reports_signed_overhead_and_zero_semantics():
    control_flags = {"depsgraph_trace": False, "debug_trace": False, "profile_callbacks": False}
    control_metrics = _metrics(event_count=0)
    control = _receipt("CONTROL", flags=control_flags, metrics=control_metrics)
    control_flags["depsgraph_trace"] = True
    control_metrics["wall_duration_ms"]["value"] = 9999
    instrumented = _receipt(
        "INSTRUMENTED",
        metrics=_metrics(wall=125.0, event_count=2, cpu=90.0),
        heartbeat=_heartbeat(1.0),
        duration_summaries=[_duration_summary(offset=5.0)],
    )

    result = performance.compare_receipts(control, instrumented)

    assert result["valid"] is True
    assert result["threshold_policy"] == "NONE_CONFIGURED"
    assert result["metric_comparisons"]["wall_duration_ms"] == {
        "status": "COMPUTED",
        "unit": "ms",
        "control_value": 100.0,
        "instrumented_value": 125.0,
        "absolute_overhead": 25.0,
        "percent_overhead": 25.0,
        "reason": None,
    }
    zero_baseline = result["metric_comparisons"]["event_count"]
    assert zero_baseline["absolute_overhead"] == 2
    assert zero_baseline["percent_overhead"] is None
    assert zero_baseline["reason"] == "ZERO_CONTROL_BASELINE"
    assert result["metric_comparisons"]["heartbeat.interval.mean_ms"]["absolute_overhead"] == 1.0
    callback_mean = result["metric_comparisons"]["duration:CALLBACK:draw:mean_ms"]
    assert callback_mean["absolute_overhead"] == 5.0
    assert control["active_debug_flags"]["depsgraph_trace"] is False
    assert control["metrics"]["wall_duration_ms"]["value"] == 100.0


def test_unavailable_is_not_zero_and_is_reported_without_percentages():
    metrics = _metrics()
    metrics["process_cpu_duration_ms"] = performance.unavailable("PROCESS_CPU_NOT_COLLECTED", "ms")
    control = _receipt("CONTROL", metrics=metrics)
    instrumented_metrics = _metrics()
    instrumented_metrics["process_cpu_duration_ms"] = performance.measured(
        90, "ms", "harness/run.json"
    )
    instrumented = _receipt("INSTRUMENTED", metrics=instrumented_metrics)

    comparison = performance.compare_receipts(control, instrumented)
    cpu = comparison["metric_comparisons"]["process_cpu_duration_ms"]

    assert comparison["valid"] is True
    assert cpu["status"] == "UNAVAILABLE"
    assert cpu["control_value"] is None
    assert cpu["instrumented_value"] == 90
    assert cpu["absolute_overhead"] is None
    assert cpu["percent_overhead"] is None
    assert cpu["reason"] == "METRIC_UNAVAILABLE"


@pytest.mark.parametrize(
    ("control_kwargs", "instrumented_kwargs", "issue_code"),
    [
        ({}, {"run_config": {"scene_seed": 99, "event_batch": 100}}, "PAIR_IDENTITY_MISMATCH"),
        (
                {},
                {
                    "blender_identity": {
                        "binary_sha256": "c" * 64,
                        "version": "5.2.1",
                    "build_hash": "build-5.2.1-test",
                }
            },
            "PAIR_IDENTITY_MISMATCH",
        ),
        ({"sequence_index": 1}, {"sequence_index": 3}, "RUNS_NOT_ADJACENT"),
            (
                {
                    "flags": {
                        "depsgraph_trace": False,
                        "debug_trace": False,
                        "profile_callbacks": False,
                    },
                    "allowed": [],
                },
                {
                    "flags": {
                        "depsgraph_trace": False,
                        "debug_trace": True,
                        "profile_callbacks": True,
                    },
                    "allowed": [],
                },
            "DEBUG_FLAG_MISMATCH",
        ),
    ],
)
def test_pair_rejects_config_identity_order_and_unallowed_toggle_mismatches(
    control_kwargs, instrumented_kwargs, issue_code
):
    control = _receipt("CONTROL", **control_kwargs)
    instrumented = _receipt("INSTRUMENTED", **instrumented_kwargs)

    result = performance.compare_receipts(control, instrumented)

    assert result["valid"] is False
    assert any(issue["code"] == issue_code for issue in result["issues"])
    assert result["metric_comparisons"]["wall_duration_ms"]["status"] == "INVALID_PAIR"
    assert result["metric_comparisons"]["wall_duration_ms"]["absolute_overhead"] is None


def test_explicitly_allowed_instrumentation_toggles_can_differ():
    control = _receipt(
        "CONTROL",
        flags={"depsgraph_trace": False, "debug_trace": False, "profile_callbacks": False},
        allowed=["debug_trace", "profile_callbacks"],
    )
    instrumented = _receipt(
        "INSTRUMENTED",
        flags={"depsgraph_trace": False, "debug_trace": True, "profile_callbacks": True},
        allowed=["debug_trace", "profile_callbacks"],
    )

    assert performance.compare_receipts(control, instrumented)["valid"] is True


def test_depsgraph_trace_is_rejected_except_for_declared_depsgraph_workload():
    kwargs = {
        "flags": {"depsgraph_trace": True, "debug_trace": False, "profile_callbacks": False},
        "allowed": ["debug_trace", "profile_callbacks", "depsgraph_trace"],
    }
    with pytest.raises(ValueError, match="DEPSGRAPH_TRACE_NOT_ALLOWED"):
        _receipt("CONTROL", **kwargs)

    receipt = _receipt("CONTROL", workload_class=performance.DEPSGRAPH_PERFORMANCE, **kwargs)
    assert performance.validate_receipt(receipt) == []


def test_duration_summary_requires_explicit_bounded_evidence_and_portable_reference():
    missing_evidence = _duration_summary()
    missing_evidence.pop("evidence_ref")
    with pytest.raises(ValueError, match="INVALID_EVIDENCE_REF"):
        _receipt("CONTROL", duration_summaries=[missing_evidence])

    absolute_evidence = _duration_summary()
    absolute_evidence["evidence_ref"] = "C:/private/callbacks.json"
    with pytest.raises(ValueError, match="NON_PORTABLE_EVIDENCE_REF"):
        _receipt("CONTROL", duration_summaries=[absolute_evidence])

    too_many = [_duration_summary(name=f"callback_{index}") for index in range(33)]
    with pytest.raises(ValueError, match="TOO_MANY_DURATION_SUMMARIES"):
        _receipt("CONTROL", duration_summaries=too_many)


def test_receipt_byte_limit_bounds_arbitrary_run_configuration():
    with pytest.raises(ValueError, match="RECEIPT_TOO_LARGE"):
        _receipt("CONTROL", run_config={"payload": "x" * (300 * 1024)})


def test_nonfinite_or_missing_metric_evidence_is_rejected():
    metrics = _metrics()
    metrics["wall_duration_ms"]["value"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON data"):
        _receipt("CONTROL", metrics=metrics)

    metrics = _metrics()
    metrics["trace_bytes"]["evidence_ref"] = "../../outside/trace.jsonl"
    with pytest.raises(ValueError, match="NON_PORTABLE_EVIDENCE_REF"):
        _receipt("CONTROL", metrics=metrics)


def test_event_and_byte_metrics_are_integral():
    metrics = _metrics()
    metrics["event_count"] = performance.measured(1.5, "count", "harness/events.json")

    with pytest.raises(ValueError, match="INVALID_METRIC_VALUE"):
        _receipt("CONTROL", metrics=metrics)


def test_performance_schema_documents_are_valid_json():
    schema_dir = _SCHEMA_DIR
    receipt_schema = json.loads(
        (schema_dir / _RECEIPT_SCHEMA_NAME).read_text(encoding="utf-8")
    )
    comparison_schema = json.loads(
        (schema_dir / _COMPARISON_SCHEMA_NAME).read_text(encoding="utf-8")
    )

    assert receipt_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert comparison_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    evidence_pattern = re.compile(receipt_schema["$defs"]["evidenceRef"]["pattern"])
    assert evidence_pattern.search("native_artifacts/dump.dmp")
    assert not evidence_pattern.search("../private/dump.dmp")
    assert not evidence_pattern.search("C:/private/dump.dmp")
    assert not evidence_pattern.search("native_artifacts/dump.dmp\n")


def test_heartbeat_stats_require_explicit_samples_and_consistent_summary():
    heartbeat = _heartbeat()
    heartbeat["method"] = "TIMESTAMP_GAP_ESTIMATE"
    with pytest.raises(ValueError, match="INVALID_HEARTBEAT_METHOD"):
        _receipt("CONTROL", heartbeat=heartbeat)

    heartbeat = _heartbeat()
    heartbeat["jitter_ms"]["p95_ms"] = 0.1
    with pytest.raises(ValueError, match="INCONSISTENT_SUMMARY_STATS"):
        _receipt("CONTROL", heartbeat=heartbeat)


def test_unavailable_heartbeat_is_explicitly_recorded():
    receipt = _receipt("CONTROL", heartbeat={"status": "UNAVAILABLE", "reason": "NO_WATCHDOG"})

    assert performance.validate_receipt(receipt) == []
    comparison = performance.compare_receipts(
        receipt,
        _receipt("INSTRUMENTED", heartbeat={"status": "UNAVAILABLE", "reason": "NO_WATCHDOG"}),
    )
    assert comparison["metric_comparisons"]["heartbeat.interval.mean_ms"]["status"] == "UNAVAILABLE"
    heartbeat_mean = comparison["metric_comparisons"]["heartbeat.interval.mean_ms"]
    assert heartbeat_mean["percent_overhead"] is None


def test_comparator_does_not_mutate_receipts():
    control = _receipt("CONTROL")
    instrumented = _receipt("INSTRUMENTED")
    before = copy.deepcopy((control, instrumented))

    performance.compare_receipts(control, instrumented)

    assert (control, instrumented) == before


def test_invalid_receipts_still_produce_json_safe_comparison_reports():
    control = _receipt("CONTROL")
    instrumented = _receipt("INSTRUMENTED")
    control["metrics"]["wall_duration_ms"]["unit"] = ["ms"]

    comparison = performance.compare_receipts(control, instrumented)

    assert comparison["valid"] is False
    assert any(issue["code"] == "INVALID_METRIC_UNIT" for issue in comparison["issues"])
    json.dumps(comparison, allow_nan=False)


_COMPUTED_WITH_PERCENT = {
    "status": "COMPUTED",
    "unit": "ms",
    "control_value": 100.0,
    "instrumented_value": 125.0,
    "absolute_overhead": 25.0,
    "percent_overhead": 25.0,
    "reason": None,
}


def _normal_percent_comparison() -> dict:
    return performance.compare_receipts(
        _receipt("CONTROL"),
        _receipt(
            "INSTRUMENTED",
            metrics=_metrics(wall=125.0, cpu=100.0),
            heartbeat=_heartbeat(1.0),
            duration_summaries=[_duration_summary(offset=5.0)],
        ),
    )


def test_performance_schemas_only_use_validator_supported_keywords():
    for name in (_RECEIPT_SCHEMA_NAME, _COMPARISON_SCHEMA_NAME):
        found: set[str] = set()
        _schema_keywords(json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8")), found)
        unsupported = found - SchemaValidator.SUPPORTED_KEYWORDS
        assert not unsupported, f"{name} uses unsupported keywords: {sorted(unsupported)}"


def test_schema_validator_enforces_property_names():
    string_name = SchemaValidator(
        {"type": "object", "propertyNames": {"type": "string", "minLength": 1}},
        _SCHEMA_DIR / _COMPARISON_SCHEMA_NAME,
    )
    restricted = SchemaValidator(
        {"type": "object", "propertyNames": {"enum": ["wall_duration_ms"]}},
        _SCHEMA_DIR / _COMPARISON_SCHEMA_NAME,
    )

    assert string_name.is_valid({"wall_duration_ms": 1})
    assert not string_name.is_valid({"": 1})
    assert restricted.is_valid({"wall_duration_ms": 1})
    assert not restricted.is_valid({"event_count": 1})


def test_comparison_schema_rejects_empty_metric_name():
    comparison = _normal_percent_comparison()
    renamed = copy.deepcopy(comparison)
    renamed["metric_comparisons"][""] = renamed["metric_comparisons"].pop("wall_duration_ms")

    assert _comparison_validator().is_valid(comparison)
    assert not _comparison_validator().is_valid(renamed)


def test_producer_normal_percent_overhead_satisfies_comparison_schema():
    comparison = _normal_percent_comparison()
    wall = comparison["metric_comparisons"]["wall_duration_ms"]

    assert comparison["valid"] is True
    assert comparison["comparison_status"] == "VALID"
    assert wall == _COMPUTED_WITH_PERCENT
    assert _comparison_validator().errors(comparison) == []


def test_producer_zero_control_baseline_is_computed_without_percent_overhead():
    comparison = performance.compare_receipts(
        _receipt("CONTROL", metrics=_metrics(event_count=0)),
        _receipt("INSTRUMENTED", metrics=_metrics(event_count=3)),
    )
    event_count = comparison["metric_comparisons"]["event_count"]

    assert event_count == {
        "status": "COMPUTED",
        "unit": "count",
        "control_value": 0,
        "instrumented_value": 3,
        "absolute_overhead": 3,
        "percent_overhead": None,
        "reason": "ZERO_CONTROL_BASELINE",
    }
    assert _comparison_validator().errors(comparison) == []


def test_producer_percent_overflow_keeps_absolute_overhead_and_nulls_percent():
    comparison = performance.compare_receipts(
        _receipt("CONTROL", metrics=_metrics(wall=5e-324)),
        _receipt("INSTRUMENTED", metrics=_metrics(wall=1e308)),
    )
    wall = comparison["metric_comparisons"]["wall_duration_ms"]

    assert wall["status"] == "COMPUTED"
    assert wall["control_value"] == 5e-324
    assert wall["instrumented_value"] == 1e308
    assert wall["absolute_overhead"] == 1e308
    assert wall["percent_overhead"] is None
    assert wall["reason"] == "PERCENT_OVERFLOW"
    assert _comparison_validator().errors(comparison) == []
    json.dumps(comparison, allow_nan=False)


def test_producer_unavailable_metric_never_reports_overhead_values():
    control_metrics = _metrics()
    control_metrics["process_cpu_duration_ms"] = performance.unavailable(
        "PROCESS_CPU_NOT_COLLECTED", "ms"
    )
    comparison = performance.compare_receipts(
        _receipt("CONTROL", metrics=control_metrics),
        _receipt("INSTRUMENTED", metrics=_metrics(cpu=90.0)),
    )
    cpu = comparison["metric_comparisons"]["process_cpu_duration_ms"]

    assert cpu["status"] == "UNAVAILABLE"
    assert cpu["control_value"] is None
    assert cpu["instrumented_value"] == 90.0
    assert cpu["absolute_overhead"] is None
    assert cpu["percent_overhead"] is None
    assert cpu["reason"] == "METRIC_UNAVAILABLE"
    assert _comparison_validator().errors(comparison) == []


def test_producer_identity_invalid_pair_never_reports_overhead_values():
    comparison = performance.compare_receipts(
        _receipt("CONTROL"),
        _receipt("INSTRUMENTED", run_config={"scene_seed": 99, "event_batch": 100}),
    )

    assert comparison["valid"] is False
    assert comparison["comparison_status"] == "INVALID"
    assert any(issue["code"] == "PAIR_IDENTITY_MISMATCH" for issue in comparison["issues"])
    assert comparison["metric_comparisons"]
    for name, metric in comparison["metric_comparisons"].items():
        assert metric["status"] == "INVALID_PAIR", name
        assert metric["absolute_overhead"] is None, name
        assert metric["percent_overhead"] is None, name
        assert metric["reason"] == "PAIR_VALIDATION_FAILED", name
    assert _comparison_validator().errors(comparison) == []


def test_comparison_schema_reason_vocabulary_matches_producer_reasons():
    reason_schema = _comparison_schema()["$defs"]["metricComparison"]["properties"]["reason"]
    assert set(reason_schema["enum"]) == {
        None,
        "ZERO_CONTROL_BASELINE",
        "PERCENT_OVERFLOW",
        "METRIC_UNAVAILABLE",
        "PAIR_VALIDATION_FAILED",
    }


@pytest.mark.parametrize(
    "replacement",
    [
        _mutate(_COMPUTED_WITH_PERCENT, reason="ZERO_CONTROL_BASELINE"),
        _mutate(_COMPUTED_WITH_PERCENT, reason="PERCENT_OVERFLOW"),
        _mutate(_COMPUTED_WITH_PERCENT, percent_overhead=None),
        _mutate(_COMPUTED_WITH_PERCENT, percent_overhead=None, reason="METRIC_UNAVAILABLE"),
        _mutate(
            _COMPUTED_WITH_PERCENT,
            percent_overhead=None,
            reason="PERCENT_DENOMINATOR_NEAR_ZERO",
        ),
        _mutate(_COMPUTED_WITH_PERCENT, absolute_overhead=None),
        _mutate(_COMPUTED_WITH_PERCENT, control_value=None),
        _mutate(_COMPUTED_WITH_PERCENT, instrumented_value=None),
        {
            "status": "UNAVAILABLE",
            "unit": "ms",
            "control_value": 100.0,
            "instrumented_value": 125.0,
            "absolute_overhead": 25.0,
            "percent_overhead": None,
            "reason": "METRIC_UNAVAILABLE",
        },
        {
            "status": "UNAVAILABLE",
            "unit": "ms",
            "control_value": 100.0,
            "instrumented_value": 125.0,
            "absolute_overhead": None,
            "percent_overhead": 25.0,
            "reason": "METRIC_UNAVAILABLE",
        },
        {
            "status": "INVALID_PAIR",
            "unit": "ms",
            "control_value": 100.0,
            "instrumented_value": 125.0,
            "absolute_overhead": 25.0,
            "percent_overhead": None,
            "reason": "PAIR_VALIDATION_FAILED",
        },
        _mutate(_COMPUTED_WITH_PERCENT, status="APPROXIMATE"),
        {key: value for key, value in _COMPUTED_WITH_PERCENT.items() if key != "percent_overhead"},
    ],
)
def test_comparison_schema_rejects_inconsistent_metric_states(replacement):
    comparison = _replace_metric(_normal_percent_comparison(), "wall_duration_ms", replacement)

    errors = _comparison_validator().errors(comparison)

    assert errors
    assert any("wall_duration_ms" in error for error in errors)


@pytest.mark.parametrize(
    "replacement",
    [
        _mutate(_COMPUTED_WITH_PERCENT, percent_overhead=None, reason="ZERO_CONTROL_BASELINE"),
        _mutate(_COMPUTED_WITH_PERCENT, percent_overhead=None, reason="PERCENT_OVERFLOW"),
        _mutate(_COMPUTED_WITH_PERCENT, percent_overhead=-40.0),
        _mutate(_COMPUTED_WITH_PERCENT, absolute_overhead=-25.0),
    ],
)
def test_comparison_schema_accepts_computed_states_without_percentage(replacement):
    comparison = _replace_metric(_normal_percent_comparison(), "wall_duration_ms", replacement)

    assert _comparison_validator().errors(comparison) == []
