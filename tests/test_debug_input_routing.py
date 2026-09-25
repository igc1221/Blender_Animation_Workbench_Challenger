from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "awb_debug_input_routing.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("awb_debug_input_routing_tested", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected(**changes):
    value = {
        "gesture_id": "gesture-1",
        "event": {"type": "F13", "value": "PRESS"},
        "binding": {
            "owner": "AWB",
            "binding_id": "awb.transform.invoke",
            "operator_id": "awb.transform",
        },
    }
    value.update(changes)
    return value


def _raw(**changes):
    value = {
        "schema": "awb-raw-input-trace/v1",
        "record_kind": "RAW_INPUT",
        "raw_input_seq": 17,
        "origin": "LIVE_USER",
        "event": {"type": "F13", "value": "PRESS"},
        "keymap_probe": {
            "seen": True,
            "bounded": True,
            "authority": "keyconfigs.user",
            "complete": True,
            "truncated": False,
            "cross_keymap_ordering": "NOT_REQUIRED",
            "binding": {
                "binding_id": "awb.transform.invoke",
                "owner": "AWB",
                "status": "ENABLED",
                "enabled": True,
                "eligible": True,
            },
            "competing_matches": [],
        },
    }
    value.update(changes)
    return value


def _ingress(**changes):
    value = {
        "schema": "awb-raw-input-trace/v1",
        "record_kind": "OPERATOR_INGRESS",
        "raw_input_seq": 18,
        "correlated_raw_input_seq": 17,
        "origin": "LIVE_USER",
        "binding_id": "awb.transform.invoke",
        "operator_id": "awb.transform",
    }
    value.update(changes)
    return value


def _modal_records(*, route=None, terminal=True):
    records = [
        {
            "schema": "awb-raw-input-trace/v1",
            "record_kind": "MODAL_OWNER_BEGIN",
            "owner_id": "modal-1",
            "active": True,
        },
        _raw(
            modal_entry=True,
            modal_owner_id="modal-1",
            keymap_probe={},
        ),
    ]
    if route is not None:
        records.append(
            {
                "schema": "awb-raw-input-trace/v1",
                "record_kind": "MODAL_ROUTE",
                "raw_input_seq": 18,
                "correlated_raw_input_seq": 17,
                "owner_id": "modal-1",
                **route,
            }
        )
    if terminal:
        records.append(
            {
                "schema": "awb-raw-input-trace/v1",
                "record_kind": "MODAL_OWNER_TERMINAL",
                "owner_id": "modal-1",
                "status": "FINISHED",
            }
        )
    return records


def test_no_raw_input_evidence_makes_no_os_or_device_loss_claim():
    module = _load_module()

    result = module.analyze_input_routing([], _expected())

    assert result["classification"] == "NO_RAW_INPUT_EVIDENCE"
    assert "does not establish whether the operating system or input device delivered it" in result[
        "message"
    ]


def test_keymap_conflict_requires_explicit_awb_binding_evidence():
    module = _load_module()
    raw = _raw(
        keymap_probe={
            "seen": True,
            "bounded": True,
            "authority": "keyconfigs.user",
            "complete": True,
            "binding": {
                "binding_id": "awb.transform.invoke",
                "owner": "AWB",
                "status": "MISSING",
            },
        }
    )

    result = module.analyze_input_routing([raw], _expected())

    assert result["classification"] == "KEYMAP_CONFLICT"
    assert result["reason"] == "EXPECTED_AWB_BINDING_MISSING"


def test_missing_or_incomplete_keymap_evidence_does_not_prove_conflict():
    module = _load_module()
    raw = _raw(
        keymap_probe={
            "seen": True,
            "bounded": True,
            "authority": "keyconfigs.user",
            "complete": False,
            "binding": {
                "binding_id": "awb.transform.invoke",
                "owner": "AWB",
                "status": "MISSING",
            },
        }
    )

    result = module.analyze_input_routing([raw], _expected())

    assert result["classification"] == "UNCLASSIFIED"

    raw["keymap_probe"]["complete"] = True
    raw["keymap_probe"]["binding"]["owner"] = "OTHER"
    result = module.analyze_input_routing([raw], _expected())
    assert result["classification"] == "UNCLASSIFIED"


def test_explicitly_disabled_awb_binding_is_a_keymap_conflict():
    module = _load_module()
    raw = _raw()
    raw["keymap_probe"]["binding"].update(status="DISABLED", enabled=False)

    result = module.analyze_input_routing([raw], _expected())

    assert result["classification"] == "KEYMAP_CONFLICT"
    assert result["reason"] == "EXPECTED_AWB_BINDING_DISABLED"


def test_unresolved_cross_keymap_order_is_ambiguous_not_a_winner_inference():
    module = _load_module()
    raw = _raw()
    raw["keymap_probe"].update(
        cross_keymap_ordering="UNRESOLVED",
        competing_matches=[
            {"binding_id": "awb.transform.invoke", "keymap": "3D View"},
            {"binding_id": "wm.native", "keymap": "Window"},
        ],
    )

    result = module.analyze_input_routing([raw], _expected())

    assert result["classification"] == "KEYMAP_STATE_AMBIGUOUS"


def test_competing_matches_need_an_authoritative_winner():
    module = _load_module()
    raw = _raw()
    raw["keymap_probe"]["competing_matches"] = [
        {"binding_id": "awb.transform.invoke", "keymap": "3D View"},
        {"binding_id": "wm.native", "keymap": "Window"},
    ]

    assert module.analyze_input_routing([raw], _expected())["classification"] == (
        "KEYMAP_STATE_AMBIGUOUS"
    )

    raw["keymap_probe"].update(winner_authoritative=True, winner_binding_id="wm.native")
    shadowed = module.analyze_input_routing([raw], _expected())
    assert shadowed["classification"] == "UNCLASSIFIED"
    assert shadowed["reason"] == "KEYMAP_SHADOWED_BY_EXTERNAL_WINNER"


def test_operator_not_started_requires_enabled_eligible_awb_binding_and_no_ingress():
    module = _load_module()

    result = module.analyze_input_routing([_raw()], _expected())

    assert result["classification"] == "OPERATOR_NOT_STARTED"


def test_correlated_operator_ingress_is_routed():
    module = _load_module()

    result = module.analyze_input_routing([_raw(), _ingress()], _expected())

    assert result["classification"] == "ROUTED"
    assert result["reason"] == "CORRELATED_OPERATOR_INGRESS"

    wrong_target = _ingress(operator_id="wm.native")
    result = module.analyze_input_routing([_raw(), wrong_target], _expected())
    assert result["classification"] == "UNCLASSIFIED"
    assert result["reason"] == "INGRESS_TARGET_MISMATCH"

    result = module.analyze_input_routing([_ingress(), _raw()], _expected())
    assert result["classification"] == "UNCLASSIFIED"
    assert result["reason"] == "INGRESS_ORDER_INVALID"


def test_replay_input_cannot_satisfy_live_user_expectation():
    module = _load_module()
    replay = _raw(origin="REPLAY", replay_execution_id="replay-9")

    result = module.analyze_input_routing([replay], _expected())

    assert result["classification"] == "NO_RAW_INPUT_EVIDENCE"

    replay_expected = _expected(origin="REPLAY", replay_execution_id="replay-9")
    assert module.analyze_input_routing([replay], replay_expected)["classification"] == (
        "OPERATOR_NOT_STARTED"
    )
    wrong_execution = _raw(origin="REPLAY", replay_execution_id="replay-other")
    assert module.analyze_input_routing([wrong_execution], replay_expected)["classification"] == (
        "NO_RAW_INPUT_EVIDENCE"
    )


def test_modal_routing_failure_requires_active_owner_entry_and_terminal():
    module = _load_module()
    expected = _expected(modal_owner_id="modal-1")

    result = module.analyze_input_routing(_modal_records(), expected)
    assert result["classification"] == "MODAL_ROUTING_FAILURE"

    incomplete = module.analyze_input_routing(_modal_records(terminal=False), expected)
    assert incomplete["classification"] == "UNCLASSIFIED"

    owner_missing = _modal_records()
    owner_missing.pop(0)
    assert module.analyze_input_routing(owner_missing, expected)["classification"] == (
        "UNCLASSIFIED"
    )


def test_valid_modal_route_is_routed_and_invalid_modal_route_fails():
    module = _load_module()
    expected = _expected(modal_owner_id="modal-1")

    routed = module.analyze_input_routing(_modal_records(route={"valid": True}), expected)
    failed = module.analyze_input_routing(_modal_records(route={"valid": False}), expected)

    assert routed["classification"] == "ROUTED"
    assert failed["classification"] == "MODAL_ROUTING_FAILURE"


def test_modal_raw_owner_id_field_is_accepted():
    module = _load_module()
    expected = _expected(modal_owner_id="modal-1")
    records = _modal_records(route={"valid": True})
    raw = records[1]
    raw["owner_id"] = raw.pop("modal_owner_id")

    result = module.analyze_input_routing(records, expected)

    assert result["classification"] == "ROUTED"


def test_dropped_evidence_prevents_absence_based_operator_classification():
    module = _load_module()
    raw = _raw(dropped=1)

    result = module.analyze_input_routing([raw], _expected())

    assert result["classification"] == "UNCLASSIFIED"
    assert result["reason"] == "INGRESS_EVIDENCE_INCOMPLETE"


def test_duplicate_matching_events_require_explicit_sequence():
    module = _load_module()

    result = module.analyze_input_routing([_raw(), _raw(raw_input_seq=18)], _expected())

    assert result["classification"] == "UNCLASSIFIED"
    assert result["reason"] == "GESTURE_CORRELATION_AMBIGUOUS"

    non_monotonic = module.analyze_input_routing(
        [
            _raw(raw_input_seq=18),
            _raw(raw_input_seq=17, event={"type": "F12", "value": "PRESS"}),
        ],
        _expected(),
    )
    assert non_monotonic["classification"] == "UNCLASSIFIED"
    assert non_monotonic["reason"] == "RAW_INPUT_SEQUENCE_INVALID"


def test_read_raw_jsonl_skips_interaction_timeline_envelopes(tmp_path):
    module = _load_module()
    raw = _raw()
    timeline = tmp_path / "timeline.jsonl"
    interaction_envelope = {
        "schema": "awb-debug-incident-timeline/v1",
        "incident_id": "INC-BB8-TEST",
        "record": {"schema": "awb-interaction-trace/v1", "event": "IGNORED"},
    }
    raw_envelope = {
        "schema": "awb-debug-raw-input-timeline/v1",
        "incident_id": "INC-BB8-TEST",
        "record": raw,
    }
    timeline.write_text(
        json.dumps(interaction_envelope) + "\n" + json.dumps(raw_envelope) + "\n",
        encoding="utf-8",
    )

    assert module._read_raw_jsonl(timeline) == [raw]


def test_oversized_raw_stream_fails_closed():
    module = _load_module()

    result = module.analyze_input_routing([{}] * (module.MAX_RECORDS + 1), _expected())

    assert result["classification"] == "UNCLASSIFIED"
    assert result["reason"] == "RECORD_LIMIT_EXCEEDED"
