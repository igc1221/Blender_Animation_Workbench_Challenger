from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from scripts import awb_native_incident_evidence as native_evidence_module
from scripts import awb_supervisor_heartbeat as raw_heartbeat_module

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "DEBUG" / "schemas"

# Naming note: HEARTBEAT_SCHEMA below is the *analyzed* BB-10 heartbeat
# (awb-debug-heartbeat/v1), which is classified and evidence-bearing. The
# *raw canonical* supervisor heartbeat record (awb-supervisor-heartbeat/v1)
# is a different, much smaller contract tracked by RAW_SUPERVISOR_HEARTBEAT_SCHEMA.
HEARTBEAT_SCHEMA = "AWB_DEBUG_HEARTBEAT_V1.schema.json"
SUPERVISOR_SCHEMA = "AWB_DEBUG_SUPERVISOR_RESULT_V1.schema.json"
ESCALATION_SCHEMA = "AWB_DEBUG_NATIVE_ESCALATION_V1.schema.json"
BB10_SCHEMAS = (HEARTBEAT_SCHEMA, SUPERVISOR_SCHEMA, ESCALATION_SCHEMA)
RAW_SUPERVISOR_HEARTBEAT_SCHEMA = "AWB_SUPERVISOR_HEARTBEAT_V1.schema.json"

INCIDENT_ID = "INC-20260926-101500-0a1b2c3d"
SHA_A = "a" * 64
SHA_B = "b" * 64

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional cross-check engine
    jsonschema = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(one, other) for one, other in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right


def _type_matches(value: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "string":
        return isinstance(value, str)
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    raise AssertionError(f"unsupported JSON Schema type keyword: {name!r}")


class SchemaValidator:
    """Stdlib-only validator for the JSON Schema subset used by the AWB schemas."""

    SUPPORTED_KEYWORDS = frozenset(
        {
            "$defs",
            "$id",
            "$ref",
            "$schema",
            "additionalProperties",
            "allOf",
            "anyOf",
            "const",
            "default",
            "deprecated",
            "description",
            "else",
            "enum",
            "examples",
            "exclusiveMaximum",
            "exclusiveMinimum",
            "format",
            "if",
            "items",
            "maxItems",
            "maxLength",
            "maximum",
            "minItems",
            "minLength",
            "minimum",
            "not",
            "oneOf",
            "pattern",
            "properties",
            "propertyNames",
            "required",
            "then",
            "title",
            "type",
            "uniqueItems",
        }
    )

    def __init__(self, root: Any, root_path: Path) -> None:
        self._root = root
        self._root_path = root_path.resolve()
        self._root_dir = self._root_path.parent
        self._documents: dict[Path, Any] = {self._root_path: root}

    @classmethod
    def from_file(cls, name: str) -> SchemaValidator:
        path = SCHEMA_DIR / name
        return cls(_read_json(path), path)

    def _load(self, path: Path) -> Any:
        key = path.resolve()
        if key not in self._documents:
            self._documents[key] = _read_json(key)
        return self._documents[key]

    def errors(self, payload: Any) -> list[str]:
        found: list[str] = []
        self._check(self._root, self._root, self._root_dir, payload, found, "$")
        return found

    def is_valid(self, payload: Any) -> bool:
        return not self.errors(payload)

    def _probe(self, schema: Any, document: Any, doc_dir: Path, value: Any) -> bool:
        probe: list[str] = []
        self._check(schema, document, doc_dir, value, probe, "$probe")
        return not probe

    def _resolve(self, ref: str, document: Any, doc_dir: Path) -> tuple[Any, Any, Path]:
        file_part, _, fragment = ref.partition("#")
        if file_part:
            path = (doc_dir / file_part).resolve()
            target_document = self._load(path)
            target_dir = path.parent
        else:
            target_document = document
            target_dir = doc_dir
        target = target_document
        for token in [part for part in fragment.split("/") if part]:
            target = target[token.replace("~1", "/").replace("~0", "~")]
        return target, target_document, target_dir

    def _check(
        self,
        schema: Any,
        document: Any,
        doc_dir: Path,
        value: Any,
        errors: list[str],
        path: str,
    ) -> None:
        if schema is True:
            return
        if schema is False:
            errors.append(f"{path}: schema forbids any value")
            return
        if not isinstance(schema, dict):
            raise TypeError(f"{path}: schema must be an object or boolean")
        if "$ref" in schema:
            target, target_document, target_dir = self._resolve(schema["$ref"], document, doc_dir)
            self._check(target, target_document, target_dir, value, errors, path)
        declared = schema.get("type")
        if declared is not None:
            names = declared if isinstance(declared, list) else [declared]
            if not any(_type_matches(value, name) for name in names):
                errors.append(f"{path}: expected type {declared!r}, got {type(value).__name__}")
                return
        if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
            errors.append(f"{path}: value is not one of {schema['enum']!r}")
        if "const" in schema and not _json_equal(value, schema["const"]):
            errors.append(f"{path}: value must equal {schema['const']!r}")
        if isinstance(value, str):
            self._check_string(schema, value, errors, path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self._check_number(schema, value, errors, path)
        if isinstance(value, list):
            self._check_array(schema, document, doc_dir, value, errors, path)
        if isinstance(value, dict):
            self._check_object(schema, document, doc_dir, value, errors, path)
        for index, subschema in enumerate(schema.get("allOf", [])):
            self._check(subschema, document, doc_dir, value, errors, f"{path}/allOf[{index}]")
        for keyword in ("anyOf", "oneOf"):
            branches = schema.get(keyword)
            if branches is not None and not any(
                self._probe(branch, document, doc_dir, value) for branch in branches
            ):
                errors.append(f"{path}: no {keyword} branch matched")
        if "not" in schema and not self._probe(schema["not"], document, doc_dir, value):
            errors.append(f"{path}: value matched a forbidden schema")
        if "if" in schema:
            branch = "then" if self._probe(schema["if"], document, doc_dir, value) else "else"
            if branch in schema:
                self._check(schema[branch], document, doc_dir, value, errors, f"{path}/{branch}")

    def _check_string(self, schema: dict, value: str, errors: list[str], path: str) -> None:
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")

    def _check_number(self, schema: dict, value: float, errors: list[str], path: str) -> None:
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: not above exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: not below exclusiveMaximum {schema['exclusiveMaximum']}")

    def _check_array(
        self,
        schema: dict,
        document: Any,
        doc_dir: Path,
        value: list,
        errors: list[str],
        path: str,
    ) -> None:
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems {schema['maxItems']}")
        if schema.get("uniqueItems") and len(
            {json.dumps(item, sort_keys=True) for item in value}
        ) != len(value):
            errors.append(f"{path}: items are not unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                self._check(item_schema, document, doc_dir, item, errors, f"{path}[{index}]")

    def _check_object(
        self,
        schema: dict,
        document: Any,
        doc_dir: Path,
        value: dict,
        errors: list[str],
        path: str,
    ) -> None:
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        properties = schema.get("properties", {})
        for name, subschema in properties.items():
            if name in value:
                self._check(subschema, document, doc_dir, value[name], errors, f"{path}.{name}")
        property_names = schema.get("propertyNames")
        if property_names is not None:
            for name in value:
                self._check(property_names, document, doc_dir, name, errors, f"{path} key {name!r}")
        additional = schema.get("additionalProperties", True)
        unknown = [name for name in value if name not in properties]
        if additional is False:
            for name in unknown:
                errors.append(f"{path}: unexpected property {name!r}")
        elif isinstance(additional, dict):
            for name in unknown:
                self._check(additional, document, doc_dir, value[name], errors, f"{path}.{name}")


def _external_is_valid(schema_name: str, payload: Any) -> bool | None:
    if jsonschema is None:
        return None
    schema = _read_json(SCHEMA_DIR / schema_name)
    validator_cls = jsonschema.validators.validator_for(schema)
    base_uri = SCHEMA_DIR.resolve().as_uri() + "/"
    try:
        resolver = jsonschema.RefResolver(base_uri=base_uri, referrer=schema)
        validator = validator_cls(schema, resolver=resolver)
        return bool(validator.is_valid(payload))
    except Exception:  # noqa: BLE001 -- optional external validator must not gate stdlib checks.
        return None


def _mutate(payload: dict, **changes: Any) -> dict:
    clone = copy.deepcopy(payload)
    clone.update(changes)
    return clone


def _mutate_nested(payload: dict, key: str, **changes: Any) -> dict:
    clone = copy.deepcopy(payload)
    clone[key] = _mutate(clone[key], **changes)
    return clone


def _drop_nested(payload: dict, key: str, *names: str) -> dict:
    clone = copy.deepcopy(payload)
    for name in names:
        clone[key].pop(name, None)
    return clone


def _set_path(payload: dict, path: tuple[str, ...], value: Any) -> dict:
    clone = copy.deepcopy(payload)
    target = clone
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = copy.deepcopy(value)
    return clone


def _artifact(index: int, **overrides: Any) -> dict:
    artifact = {
        "name": f"artifact-{index}.dmp",
        "kind": "DUMP",
        "status": "PRESENT",
        "source_path": "debug/incidents/dump/artifact.dmp",
        "size_bytes": 4096,
        "sha256": SHA_A,
        "hash_algorithm": "sha256",
        "captured_bytes": 4096,
        "bounded": True,
        "truncated": False,
        "collected_utc": "2026-09-26T10:16:00.000000Z",
        "missing_reason": None,
    }
    artifact.update(overrides)
    return artifact


def _missing_artifact(**overrides) -> dict:
    values = {
        "name": "blender_runtime.log",
        "kind": "RUNTIME_LOG",
        "status": "MISSING",
        "source_path": None,
        "size_bytes": None,
        "sha256": None,
        "hash_algorithm": None,
        "captured_bytes": 0,
        "truncated": False,
        "collected_utc": None,
        "missing_reason": "runtime log was never created",
    }
    values.update(overrides)
    return _artifact(1, **values)


def _heartbeat_alive_payload() -> dict:
    return {
        "schema": "awb-debug-heartbeat/v1",
        "heartbeat_id": "HB-0001",
        "incident_id": INCIDENT_ID,
        "observed_utc": "2026-09-26T10:15:00.000000Z",
        "monitor_role": "EXTERNAL_SUPERVISOR",
        "sequence": 7,
        "interval_seconds": 1.0,
        "stale_threshold_seconds": 5.0,
        "classification": "ALIVE_RESPONSIVE",
        "classification_reason": None,
        "confidence": "PROVEN",
        "last_heartbeat_utc": "2026-09-26T10:14:59.500000Z",
        "age_seconds": 0.5,
        "process": {
            "pid": 1234,
            "process_start_token": "start-abc",
            "image_path": "C:/tools/blender/blender.exe",
            "alive": True,
            "exit_code": None,
            "exit_kind": "RUNNING",
        },
        "writer": {
            "source": "AWB_PY_TIMER",
            "pid": 1234,
            "thread_id": 9911,
            "probe_kind": "HEARTBEAT_WRITE",
        },
        "evidence": {
            "status": "AVAILABLE",
            "reason": None,
            "sample_count": 1,
            "dropped_samples": 0,
            "truncated": False,
            "max_bytes": 65536,
        },
    }


def _heartbeat_exited_payload() -> dict:
    payload = _heartbeat_alive_payload()
    payload["classification"] = "PROCESS_EXITED"
    payload["classification_reason"] = "process handle signalled"
    payload["age_seconds"] = 12.5
    payload["last_heartbeat_utc"] = "2026-09-26T10:14:47.500000Z"
    payload["process"] = {
        "pid": 1234,
        "process_start_token": "start-abc",
        "image_path": "C:/tools/blender/blender.exe",
        "alive": False,
        "exit_code": -1073741819,
        "exit_kind": "WINDOWS_EXCEPTION_CODE",
    }
    payload["writer"] = {
        "source": "EXTERNAL_SUPERVISOR",
        "pid": 4242,
        "thread_id": None,
        "probe_kind": "PROCESS_POLL",
    }
    payload["evidence"] = {
        "status": "PARTIAL",
        "reason": "no in-process heartbeat after exit",
        "sample_count": 1,
        "dropped_samples": 3,
        "truncated": False,
        "max_bytes": 65536,
    }
    return payload


def _heartbeat_unavailable_payload() -> dict:
    payload = _heartbeat_alive_payload()
    payload["classification"] = "UNAVAILABLE"
    payload["classification_reason"] = "no external supervisor attached"
    payload["confidence"] = "UNAVAILABLE"
    payload["incident_id"] = None
    payload["last_heartbeat_utc"] = None
    payload["age_seconds"] = None
    payload["process"] = {
        "pid": None,
        "process_start_token": None,
        "image_path": None,
        "alive": None,
        "exit_code": None,
        "exit_kind": "UNAVAILABLE",
    }
    payload["writer"] = {
        "source": "UNAVAILABLE",
        "pid": None,
        "thread_id": None,
        "probe_kind": "UNAVAILABLE",
    }
    payload["evidence"] = {
        "status": "UNAVAILABLE",
        "reason": "process polling is not supported in this runtime",
        "sample_count": 0,
        "dropped_samples": 0,
        "truncated": False,
        "max_bytes": 0,
    }
    return payload


def _raw_heartbeat_payload(**overrides: Any) -> dict:
    payload = {
        "schema": "awb-supervisor-heartbeat/v1",
        "incident_id": "incident-7",
        "pid": 1234,
        "sequence": 7,
        "utc": "2026-09-26T10:15:00.123Z",
    }
    payload.update(overrides)
    return payload


def _raw_heartbeat_optional_payload() -> dict:
    return _raw_heartbeat_payload(
        awb_session_id="session-3",
        trace_id="trace-9",
    )


def _termination(**overrides: Any) -> dict:
    decision = {
        "initiated": False,
        "initiated_by": "USER",
        "reason_code": "UNCLASSIFIED",
        "reason": None,
        "requested_utc": None,
        "grace_period_seconds": None,
        "forced": False,
        "capture_attempted_before_termination": False,
        "capture_completed": False,
    }
    decision.update(overrides)
    return decision


def _supervisor_native_crash_payload() -> dict:
    return {
        "schema": "awb-debug-supervisor-result/v1",
        "supervision_id": "SUP-0001",
        "supervisor_session_id": "sup-session-1",
        "incident_id": INCIDENT_ID,
        "incident_id_status": "BOUND",
        "started_utc": "2026-09-26T10:14:00.000000Z",
        "completed_utc": "2026-09-26T10:16:00.000000Z",
        "process": {
            "pid": 1234,
            "process_start_token": "start-abc",
            "image_path": "C:/tools/blender/blender.exe",
            "command_line": None,
            "architecture": "X64",
        },
        "lifecycle": {
            "status": "ABNORMAL",
            "exit_code": -1073741819,
            "exit_kind": "WINDOWS_EXCEPTION_CODE",
            "exception_code": "0xC0000005",
            "duration_seconds": 42.5,
            "watchdog_recovered": False,
        },
        "termination": {
            "supervisor_initiated_termination": _termination(),
        },
        "heartbeat": {
            "status": "STALE",
            "heartbeat_id": "HB-0001",
            "last_heartbeat_utc": "2026-09-26T10:15:47.500000Z",
            "age_seconds": 12.5,
            "beats_observed": 42,
        },
        "artifacts": [_artifact(0), _missing_artifact()],
        "evidence": {
            "status": "PARTIAL",
            "reason": "runtime log was still open when the process died",
            "preserved_preceding_causal_evidence": True,
            "incident_manifest_path": f"debug/incidents/{INCIDENT_ID}/manifest.json",
            "incident_manifest_sha256": SHA_A,
            "runtime_log_path": None,
            "runtime_log_sha256": None,
        },
        "hypothesis": "AWB_NATIVE_LAYER",
        "escalation_required": True,
    }


def _supervisor_watchdog_termination_payload() -> dict:
    return {
        "schema": "awb-debug-supervisor-result/v1",
        "supervision_id": "SUP-0002",
        "supervisor_session_id": "sup-session-2",
        "incident_id": INCIDENT_ID,
        "incident_id_status": "BOUND",
        "started_utc": "2026-09-26T11:00:00.000000Z",
        "completed_utc": "2026-09-26T11:01:30.000000Z",
        "process": {
            "pid": 4321,
            "process_start_token": "start-def",
            "image_path": "C:/tools/blender/blender.exe",
            "command_line": "blender.exe --background",
            "architecture": "X64",
        },
        "lifecycle": {
            "status": "FAILED",
            "exit_code": -1,
            "exit_kind": "EXIT_CODE",
            "exception_code": None,
            "duration_seconds": 90.0,
            "watchdog_recovered": False,
        },
        "termination": {
            "supervisor_initiated_termination": _termination(
                initiated=True,
                initiated_by="SUPERVISOR",
                reason_code="HEARTBEAT_TIMEOUT",
                reason="no heartbeat for 30s while the main thread was blocked",
                requested_utc="2026-09-26T11:01:20.000000Z",
                grace_period_seconds=5.0,
                forced=False,
                capture_attempted_before_termination=True,
                capture_completed=True,
            ),
        },
        "heartbeat": {
            "status": "MISSING",
            "heartbeat_id": "HB-0002",
            "last_heartbeat_utc": None,
            "age_seconds": None,
            "beats_observed": 0,
        },
        "artifacts": [
            _artifact(
                2,
                name="heartbeat.json",
                kind="HEARTBEAT",
                size_bytes=1024,
                captured_bytes=1024,
                sha256=SHA_B,
            )
        ],
        "evidence": {
            "status": "UNAVAILABLE",
            "reason": "the incident bundle could not be captured from a blocked main thread",
            "preserved_preceding_causal_evidence": False,
            "incident_manifest_path": None,
            "incident_manifest_sha256": None,
            "runtime_log_path": None,
            "runtime_log_sha256": None,
        },
        "hypothesis": "UNDETERMINED",
        "escalation_required": True,
    }


def _supervisor_unavailable_payload() -> dict:
    return {
        "schema": "awb-debug-supervisor-result/v1",
        "supervision_id": "SUP-0003",
        "supervisor_session_id": "sup-session-3",
        "incident_id": None,
        "incident_id_status": "UNAVAILABLE",
        "started_utc": "2026-09-26T12:00:00.000000Z",
        "completed_utc": None,
        "process": {
            "pid": None,
            "process_start_token": None,
            "image_path": None,
            "command_line": None,
            "architecture": "UNAVAILABLE",
        },
        "lifecycle": {
            "status": "UNAVAILABLE",
            "exit_code": None,
            "exit_kind": "UNAVAILABLE",
            "exception_code": None,
            "duration_seconds": None,
            "watchdog_recovered": False,
        },
        "termination": {
            "supervisor_initiated_termination": _termination(
                initiated_by="UNAVAILABLE",
                reason_code="UNAVAILABLE",
            ),
        },
        "heartbeat": {
            "status": "UNAVAILABLE",
            "heartbeat_id": None,
            "last_heartbeat_utc": None,
            "age_seconds": None,
            "beats_observed": 0,
        },
        "artifacts": [_missing_artifact()],
        "evidence": {
            "status": "UNAVAILABLE",
            "reason": "no supervisor process attached to this Blender instance",
            "preserved_preceding_causal_evidence": False,
            "incident_manifest_path": None,
            "incident_manifest_sha256": None,
            "runtime_log_path": None,
            "runtime_log_sha256": None,
        },
        "hypothesis": "UNAVAILABLE",
        "escalation_required": True,
    }


def _escalation_analyzed_payload() -> dict:
    return {
        "schema": "awb-debug-native-escalation/v1",
        "escalation_id": "ESC-0001",
        "incident_id": INCIDENT_ID,
        "incident_id_status": "BOUND",
        "created_utc": "2026-09-26T10:20:00.000000Z",
        "trigger": {
            "kind": "WINDOWS_EXCEPTION",
            "detail": "0xC0000005 access violation after the last semantic operation",
            "detected_utc": "2026-09-26T10:16:02.000000Z",
            "exit_code": -1073741819,
        },
        "preconditions": {
            "python_evidence_status": "EXHAUSTED",
            "semantic_evidence_status": "EXHAUSTED",
            "last_cause_event": {
                "trace_id": "trace-1",
                "span_id": "span-9",
                "parent_span_id": "span-1",
                "operation_id": "op-77",
                "checkpoint_id": f"{INCIDENT_ID}:after-operation",
                "checkpoint_seq": 12,
                "boundary": "AFTER_OPERATION",
            },
            "preceding_causal_evidence_preserved": True,
            "incident_manifest_path": f"debug/incidents/{INCIDENT_ID}/manifest.json",
            "incident_manifest_sha256": SHA_A,
            "supervisor_result": _supervisor_native_crash_payload(),
        },
        "route": {
            "attempted": True,
            "route": "WINDBG_LIVE",
            "reason": "the cheap AWB layers cannot explain the access violation",
            "requires_reproduction": False,
            "next_higher_layer": "WINDBG_TTD",
        },
        "tooling": {
            "name": "WINDBG",
            "version": "1.2506.12001",
            "path": "C:/tools/windbg/x64/windbg.exe",
            "status": "AVAILABLE",
            "error": None,
        },
        "dump": {
            "status": "CAPTURED",
            "kind": "FULL",
            "full_memory": True,
            "path": "debug/incidents/dump/full.dmp",
            "size_bytes": 8123456789,
            "sha256": SHA_B,
            "hash_algorithm": "sha256",
            "reason": None,
        },
        "analysis": {
            "status": "ANALYZED",
            "analyzed_utc": "2026-09-26T10:40:00.000000Z",
            "exception_code": "0xC0000005",
            "faulting_module": "blender.exe",
            "faulting_symbol": "MEM_ADDRESS!memcpy_sse2_unaligned_erms",
            "root_cause_summary": "native write through a freed rig buffer during evaluation",
            "notes": None,
        },
        "verdict": {
            "classification": "BLENDER_CORE",
            "responsibility": "PROVEN",
            "summary": "faulting frame is Blender core evaluation, not AWB Python",
        },
        "supervisor_initiated_termination": _termination(
            initiated=True,
            initiated_by="SUPERVISOR",
            reason_code="WATCHDOG_RECOVERY",
            reason="watchdog ended the unresponsive process after the dump window",
            requested_utc="2026-09-26T10:16:10.000000Z",
            grace_period_seconds=2.0,
            forced=True,
            capture_attempted_before_termination=True,
            capture_completed=True,
        ),
        "artifacts": [
            _artifact(
                3,
                name="full.dmp",
                kind="DUMP",
                size_bytes=8123456789,
                captured_bytes=8123456789,
                sha256=SHA_B,
            ),
            _missing_artifact(),
        ],
        "evidence": {
            "status": "AVAILABLE",
            "reason": None,
        },
        "recommended_next_actions": [
            "file the access violation upstream with the captured dump",
        ],
    }


def _escalation_pending_payload() -> dict:
    return {
        "schema": "awb-debug-native-escalation/v1",
        "escalation_id": "ESC-0002",
        "incident_id": None,
        "incident_id_status": "PENDING",
        "created_utc": "2026-09-26T13:00:00.000000Z",
        "trigger": {
            "kind": "HANG",
            "detail": None,
            "detected_utc": "2026-09-26T12:59:00.000000Z",
            "exit_code": None,
        },
        "preconditions": {
            "python_evidence_status": "INCONCLUSIVE",
            "semantic_evidence_status": "UNAVAILABLE",
            "last_cause_event": {
                "trace_id": None,
                "span_id": None,
                "parent_span_id": None,
                "operation_id": None,
                "checkpoint_id": None,
                "checkpoint_seq": None,
                "boundary": None,
            },
            "preceding_causal_evidence_preserved": False,
            "incident_manifest_path": None,
            "incident_manifest_sha256": None,
            "supervisor_result": None,
        },
        "route": {
            "attempted": False,
            "route": "PROC_DUMP",
            "reason": "waiting for the supervisor capture window to open",
            "requires_reproduction": False,
            "next_higher_layer": "WINDBG_TTD",
        },
        "tooling": {
            "name": "PROCDUMP",
            "version": None,
            "path": None,
            "status": "NOT_ATTEMPTED",
            "error": None,
        },
        "dump": {
            "status": "PENDING",
            "kind": "UNAVAILABLE",
            "full_memory": False,
            "path": None,
            "size_bytes": None,
            "sha256": None,
            "hash_algorithm": None,
            "reason": "capture has not started",
        },
        "analysis": {
            "status": "PENDING",
            "analyzed_utc": None,
            "exception_code": None,
            "faulting_module": None,
            "faulting_symbol": None,
            "root_cause_summary": None,
            "notes": None,
        },
        "verdict": {
            "classification": "UNDETERMINED",
            "responsibility": "UNDETERMINED",
            "summary": None,
        },
        "supervisor_initiated_termination": None,
        "artifacts": [],
        "evidence": {
            "status": "PARTIAL",
            "reason": "only the last durable causal trace survives so far",
        },
        "recommended_next_actions": [
            "re-run the supervised session and capture a triage dump",
        ],
    }


def _escalation_unavailable_payload() -> dict:
    return {
        "schema": "awb-debug-native-escalation/v1",
        "escalation_id": "ESC-0003",
        "incident_id": None,
        "incident_id_status": "UNAVAILABLE",
        "created_utc": "2026-09-26T14:00:00.000000Z",
        "trigger": {
            "kind": "UNAVAILABLE",
            "detail": None,
            "detected_utc": None,
            "exit_code": None,
        },
        "preconditions": {
            "python_evidence_status": "UNAVAILABLE",
            "semantic_evidence_status": "UNAVAILABLE",
            "last_cause_event": {
                "trace_id": None,
                "span_id": None,
                "parent_span_id": None,
                "operation_id": None,
                "checkpoint_id": None,
                "checkpoint_seq": None,
                "boundary": None,
            },
            "preceding_causal_evidence_preserved": False,
            "incident_manifest_path": None,
            "incident_manifest_sha256": None,
            "supervisor_result": None,
        },
        "route": {
            "attempted": False,
            "route": "NONE",
            "reason": "no native escalation tooling is installed",
            "requires_reproduction": False,
            "next_higher_layer": "UNAVAILABLE",
        },
        "tooling": {
            "name": "UNAVAILABLE",
            "version": None,
            "path": None,
            "status": "UNAVAILABLE",
            "error": "windbg is not installed on this machine",
        },
        "dump": {
            "status": "UNAVAILABLE",
            "kind": "UNAVAILABLE",
            "full_memory": False,
            "path": None,
            "size_bytes": None,
            "sha256": None,
            "hash_algorithm": None,
            "reason": "no dump capture tool available",
        },
        "analysis": {
            "status": "NOT_ATTEMPTED",
            "analyzed_utc": None,
            "exception_code": None,
            "faulting_module": None,
            "faulting_symbol": None,
            "root_cause_summary": None,
            "notes": None,
        },
        "verdict": {
            "classification": "UNAVAILABLE",
            "responsibility": "UNAVAILABLE",
            "summary": None,
        },
        "supervisor_initiated_termination": None,
        "artifacts": [],
        "evidence": {
            "status": "UNAVAILABLE",
            "reason": "escalation cannot run without tooling or an incident bundle",
        },
        "recommended_next_actions": [
            "install WinDbg or procDump before the next native incident",
        ],
    }


def _validation_cases() -> list[tuple[str, Any, bool, str]]:
    cases: list[tuple[str, Any, bool, str]] = []

    def add(schema: str, case_id: str, payload: Any, expected: bool) -> None:
        cases.append((schema, payload, expected, case_id))

    add(HEARTBEAT_SCHEMA, "heartbeat-responsive", _heartbeat_alive_payload(), True)
    add(HEARTBEAT_SCHEMA, "heartbeat-process-exited", _heartbeat_exited_payload(), True)
    add(HEARTBEAT_SCHEMA, "heartbeat-unavailable", _heartbeat_unavailable_payload(), True)
    unavailable = _heartbeat_unavailable_payload()
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-missing-beat-is-not-proven-crash",
        _mutate(unavailable, classification="HEARTBEAT_MISSING", confidence="PROVEN"),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-responsive-without-freshness",
        _mutate(_heartbeat_alive_payload(), last_heartbeat_utc=None, age_seconds=None),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-responsive-without-evidence",
        _mutate_nested(_heartbeat_alive_payload(), "evidence", status="UNAVAILABLE"),
        False,
    )
    exited = _heartbeat_exited_payload()
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-exited-but-still-alive",
        _mutate_nested(exited, "process", alive=True),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-exited-with-unknown-exit-kind",
        _mutate_nested(exited, "process", exit_kind="UNKNOWN"),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-exited-missing-exit-evidence",
        _mutate(exited, last_heartbeat_utc=None, age_seconds=None),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-rejects-crash-vocabulary",
        _mutate(_heartbeat_alive_payload(), classification="CRASHED"),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-rejects-unknown-property",
        _mutate(_heartbeat_alive_payload(), crash_proven=True),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-rejects-bad-incident-id",
        _mutate(_heartbeat_alive_payload(), incident_id="INC-2026"),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-rejects-negative-pid",
        _mutate_nested(_heartbeat_alive_payload(), "process", pid=-1),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-rejects-boolean-sequence",
        _mutate(_heartbeat_alive_payload(), sequence=True),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-rejects-missing-evidence-bound",
        _drop_nested(_heartbeat_alive_payload(), "evidence", "max_bytes"),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-rejects-zero-stale-threshold",
        _mutate(_heartbeat_alive_payload(), stale_threshold_seconds=0),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-unavailable-without-reason",
        _mutate(_heartbeat_unavailable_payload(), classification_reason=None),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-rejects-unknown-writer",
        _mutate_nested(_heartbeat_alive_payload(), "writer", supervisor="taskmgr"),
        False,
    )
    add(
        HEARTBEAT_SCHEMA,
        "heartbeat-rejects-unknown-monitor-role",
        _mutate(_heartbeat_alive_payload(), monitor_role="GOD_MODE"),
        False,
    )

    add(SUPERVISOR_SCHEMA, "supervisor-native-crash", _supervisor_native_crash_payload(), True)
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-watchdog-termination",
        _supervisor_watchdog_termination_payload(),
        True,
    )
    add(SUPERVISOR_SCHEMA, "supervisor-unavailable", _supervisor_unavailable_payload(), True)
    watchdog = _supervisor_watchdog_termination_payload()
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-initiated-by-user-is-rejected",
        _mutate_nested(
            watchdog,
            "termination",
            supervisor_initiated_termination=_termination(
                initiated=True,
                initiated_by="USER",
                reason_code="HEARTBEAT_TIMEOUT",
                reason="operator asked for it",
                requested_utc="2026-09-26T11:01:20.000000Z",
                grace_period_seconds=0.0,
                capture_attempted_before_termination=True,
                capture_completed=False,
            ),
        ),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-initiated-with-unclassified-reason",
        _mutate_nested(
            watchdog,
            "termination",
            supervisor_initiated_termination=_termination(
                initiated=True,
                initiated_by="SUPERVISOR",
                reason_code="UNCLASSIFIED",
                reason="something happened",
                requested_utc="2026-09-26T11:01:20.000000Z",
                grace_period_seconds=0.0,
                capture_attempted_before_termination=True,
                capture_completed=False,
            ),
        ),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "tool-chain-initiated-with-unclassified-reason",
        _mutate_nested(
            watchdog,
            "termination",
            supervisor_initiated_termination=_termination(
                initiated=True,
                initiated_by="TOOL_CHAIN",
                reason_code="UNCLASSIFIED",
                reason="forced termination reported without a more specific reason",
                requested_utc="2026-09-26T11:01:20.000000Z",
                grace_period_seconds=0.0,
                forced=True,
                capture_attempted_before_termination=False,
                capture_completed=False,
            ),
        ),
        True,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-initiated-without-request-time",
        _mutate_nested(
            watchdog,
            "termination",
            supervisor_initiated_termination=_termination(
                initiated=True,
                initiated_by="SUPERVISOR",
                reason_code="HEARTBEAT_TIMEOUT",
                reason="no heartbeat",
                requested_utc=None,
                grace_period_seconds=None,
                capture_attempted_before_termination=True,
                capture_completed=False,
            ),
        ),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-not-initiated-by-supervisor-is-rejected",
        _mutate_nested(
            _supervisor_native_crash_payload(),
            "termination",
            supervisor_initiated_termination=_termination(
                initiated=False,
                initiated_by="SUPERVISOR",
            ),
        ),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-capture-completed-without-attempt",
        _mutate_nested(
            watchdog,
            "termination",
            supervisor_initiated_termination=_termination(
                initiated=True,
                initiated_by="SUPERVISOR",
                reason_code="RESPONSIVENESS_TIMEOUT",
                reason="window pump stopped",
                requested_utc="2026-09-26T11:01:20.000000Z",
                grace_period_seconds=1.0,
                capture_attempted_before_termination=False,
                capture_completed=True,
            ),
        ),
        False,
    )
    crash = _supervisor_native_crash_payload()
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-present-artifact-without-hash",
        _mutate(crash, artifacts=[_artifact(0, sha256=None), _missing_artifact()]),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-missing-artifact-without-reason",
        _mutate(crash, artifacts=[_artifact(0), _missing_artifact(missing_reason=None)]),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-missing-artifact-with-hash",
        _mutate(crash, artifacts=[_artifact(0), _missing_artifact(sha256=SHA_B, size_bytes=10)]),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-artifact-with-empty-name",
        _mutate(crash, artifacts=[_artifact(0, name=""), _missing_artifact()]),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-artifact-hash-must-be-lowercase-sha256",
        _mutate(crash, artifacts=[_artifact(0, sha256=SHA_A.upper()), _missing_artifact()]),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-artifact-inventory-is-bounded",
        _mutate(crash, artifacts=[_missing_artifact() for _ in range(65)]),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-bound-requires-incident-id",
        _mutate(crash, incident_id=None),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-pending-cannot-carry-incident-id",
        _mutate(crash, incident_id_status="PENDING"),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-running-cannot-report-exception-exit",
        _mutate_nested(crash, "lifecycle", status="RUNNING"),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-exception-code-format",
        _mutate_nested(crash, "lifecycle", exception_code="C0000005"),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-native-hypothesis-requires-escalation",
        _mutate(crash, escalation_required=False),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-unavailable-evidence-cannot-claim-preserved",
        _mutate(
            crash,
            evidence={
                "status": "UNAVAILABLE",
                "reason": "nothing survived",
                "preserved_preceding_causal_evidence": True,
                "incident_manifest_path": f"debug/incidents/{INCIDENT_ID}/manifest.json",
                "incident_manifest_sha256": SHA_A,
                "runtime_log_path": None,
                "runtime_log_sha256": None,
            },
        ),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-preserved-evidence-requires-hashed-manifest",
        _mutate_nested(crash, "evidence", incident_manifest_sha256=None),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-missing-heartbeat-cannot-carry-age",
        _mutate_nested(
            crash,
            "heartbeat",
            status="MISSING",
            last_heartbeat_utc=None,
        ),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-watchdog-recovered-must-be-boolean",
        _mutate_nested(crash, "lifecycle", watchdog_recovered="false"),
        False,
    )
    add(
        SUPERVISOR_SCHEMA,
        "supervisor-rejects-unknown-property",
        _mutate(crash, hang_detected=True),
        False,
    )

    add(ESCALATION_SCHEMA, "escalation-analyzed", _escalation_analyzed_payload(), True)
    add(ESCALATION_SCHEMA, "escalation-pending", _escalation_pending_payload(), True)
    add(ESCALATION_SCHEMA, "escalation-unavailable", _escalation_unavailable_payload(), True)
    analyzed = _escalation_analyzed_payload()
    add(
        ESCALATION_SCHEMA,
        "escalation-proven-verdict-requires-classification",
        _mutate_nested(analyzed, "verdict", classification="UNDETERMINED"),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-proven-verdict-requires-analysis",
        _mutate(
            analyzed,
            analysis={
                "status": "PENDING",
                "analyzed_utc": None,
                "exception_code": None,
                "faulting_module": None,
                "faulting_symbol": None,
                "root_cause_summary": None,
                "notes": None,
            },
        ),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-captured-dump-requires-hash-and-size",
        _mutate_nested(analyzed, "dump", sha256=None, hash_algorithm=None),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-not-captured-dump-cannot-have-path",
        _mutate_nested(analyzed, "dump", status="NOT_CAPTURED", path="debug/dump/full.dmp"),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-not-captured-dump-requires-reason",
        _mutate_nested(analyzed, "dump", status="UNAVAILABLE", reason=None, kind="UNAVAILABLE"),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-ttd-requires-reproduction",
        _mutate_nested(
            analyzed,
            "route",
            route="WINDBG_TTD",
            requires_reproduction=False,
        ),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-none-route-cannot-be-attempted",
        _mutate_nested(analyzed, "route", route="NONE"),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-unavailable-tooling-requires-error",
        _mutate_nested(analyzed, "tooling", status="UNAVAILABLE", error=None),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-available-tooling-requires-version-and-path",
        _mutate_nested(analyzed, "tooling", version=None, path=None),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-analyzed-requires-root-cause",
        _mutate_nested(analyzed, "analysis", root_cause_summary=None),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-unavailable-analysis-cannot-claim-summary",
        _mutate_nested(analyzed, "analysis", status="NOT_ATTEMPTED"),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-exception-code-format",
        _mutate_nested(analyzed, "analysis", exception_code="C0000005"),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-preserved-evidence-requires-hashed-manifest",
        _mutate_nested(analyzed, "preconditions", incident_manifest_sha256=None),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-preserved-evidence-requires-manifest-path",
        _mutate_nested(analyzed, "preconditions", incident_manifest_path=None),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-preserved-evidence-needs-python-evidence",
        _mutate_nested(analyzed, "preconditions", python_evidence_status="UNAVAILABLE"),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-unavailable-semantic-evidence-drops-causal-ids",
        _set_path(
            _set_path(
                analyzed,
                ("preconditions", "semantic_evidence_status"),
                "UNAVAILABLE",
            ),
            ("preconditions", "last_cause_event", "trace_id"),
            "trace-1",
        ),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-exhausted-python-requires-semantic-evidence",
        _set_path(
            _set_path(
                analyzed,
                ("preconditions", "semantic_evidence_status"),
                "UNAVAILABLE",
            ),
            ("preconditions", "last_cause_event"),
            {
                "trace_id": None,
                "span_id": None,
                "parent_span_id": None,
                "operation_id": None,
                "checkpoint_id": None,
                "checkpoint_seq": None,
                "boundary": None,
            },
        ),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-pending-cannot-carry-incident-id",
        _mutate(analyzed, incident_id_status="PENDING"),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-windows-exception-trigger-requires-detail",
        _mutate_nested(analyzed, "trigger", detail=None),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-supervisor-termination-must-be-complete",
        _mutate(analyzed, supervisor_initiated_termination={"initiated": True}),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-embedded-supervisor-result-must-validate",
        _mutate_nested(
            analyzed,
            "preconditions",
            supervisor_result={"schema": "awb-debug-supervisor-result/v1"},
        ),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-artifact-inventory-is-bounded",
        _mutate(analyzed, artifacts=[_missing_artifact() for _ in range(65)]),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-next-actions-are-bounded",
        _mutate(analyzed, recommended_next_actions=["step"] * 17),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-rejects-unknown-property",
        _mutate(analyzed, resolved=True),
        False,
    )
    add(
        ESCALATION_SCHEMA,
        "escalation-unavailable-evidence-requires-reason",
        _mutate_nested(analyzed, "evidence", status="UNAVAILABLE", reason=None),
        False,
    )
    return cases


VALIDATION_CASES = _validation_cases()
VALIDATION_PARAMS = [
    pytest.param(schema, payload, expected, id=case_id)
    for schema, payload, expected, case_id in VALIDATION_CASES
]


def _raw_heartbeat_validation_cases() -> list[tuple[str, Any, bool, str]]:
    cases: list[tuple[str, Any, bool, str]] = []

    def add(payload: Any, case_id: str, expected: bool) -> None:
        cases.append((RAW_SUPERVISOR_HEARTBEAT_SCHEMA, payload, expected, case_id))

    add(_raw_heartbeat_payload(), "raw-heartbeat-minimal", True)
    add(_raw_heartbeat_optional_payload(), "raw-heartbeat-with-optional-identity", True)
    add(_raw_heartbeat_payload(sequence=0), "raw-heartbeat-first-sequence-is-zero", True)
    add(
        _raw_heartbeat_payload(utc="2026-09-26T10:15:00+00:00"),
        "raw-heartbeat-explicit-zero-offset",
        True,
    )
    add(_raw_heartbeat_payload(utc=""), "raw-heartbeat-rejects-empty-utc", False)
    add(
        _raw_heartbeat_payload(utc="2026-09-26T10:15:00"),
        "raw-heartbeat-rejects-naive-utc",
        False,
    )
    add(
        _raw_heartbeat_payload(utc="2026-09-26T10:15:00+09:00"),
        "raw-heartbeat-rejects-non-utc-offset",
        False,
    )
    add(_raw_heartbeat_payload(incident_id=""), "raw-heartbeat-rejects-blank-incident", False)
    add(
        _raw_heartbeat_payload(incident_id="   "),
        "raw-heartbeat-rejects-whitespace-incident",
        False,
    )
    add(_raw_heartbeat_payload(pid=0), "raw-heartbeat-rejects-zero-pid", False)
    add(_raw_heartbeat_payload(pid=-4), "raw-heartbeat-rejects-negative-pid", False)
    add(_raw_heartbeat_payload(pid=True), "raw-heartbeat-rejects-boolean-pid", False)
    add(_raw_heartbeat_payload(pid="1234"), "raw-heartbeat-rejects-string-pid", False)
    add(_raw_heartbeat_payload(sequence=-1), "raw-heartbeat-rejects-negative-sequence", False)
    add(_raw_heartbeat_payload(sequence=True), "raw-heartbeat-rejects-boolean-sequence", False)
    add(
        _raw_heartbeat_payload(awb_session_id=None),
        "raw-heartbeat-optional-identity-is-omitted-not-null",
        False,
    )
    add(
        _raw_heartbeat_payload(trace_id=" "),
        "raw-heartbeat-rejects-blank-trace-id",
        False,
    )
    add(
        _mutate(_raw_heartbeat_payload(), schema="awb-debug-heartbeat/v1"),
        "raw-heartbeat-rejects-analyzed-schema-tag",
        False,
    )
    for name in ("incident_id", "pid", "sequence", "utc"):
        payload = _raw_heartbeat_payload()
        payload.pop(name)
        add(payload, f"raw-heartbeat-requires-{name.replace('_', '-')}", False)
    for name, value in (
        ("classification", "ALIVE_RESPONSIVE"),
        ("confidence", "PROVEN"),
        ("age_seconds", 0.5),
        ("last_heartbeat_utc", "2026-09-26T10:14:59.500Z"),
        ("stale", True),
        ("sequence_advanced", True),
        ("elapsed_since_advance", 1.25),
        ("interval_seconds", 1.0),
        ("stale_threshold_seconds", 5.0),
        ("monitor_role", "EXTERNAL_SUPERVISOR"),
        ("process_start_token", "start-abc"),
        ("monotonic_seconds", 12.5),
    ):
        add(
            _mutate(_raw_heartbeat_payload(), **{name: value}),
            f"raw-heartbeat-carries-no-{name.replace('_', '-')}",
            False,
        )
    return cases


RAW_HEARTBEAT_CASES = _raw_heartbeat_validation_cases()
RAW_HEARTBEAT_PARAMS = [
    pytest.param(schema, payload, expected, id=case_id)
    for schema, payload, expected, case_id in RAW_HEARTBEAT_CASES
]


_SCHEMA_MAP_KEYWORDS = ("properties", "$defs")
_SCHEMA_LIST_KEYWORDS = ("allOf", "anyOf", "oneOf")
_SCHEMA_VALUE_KEYWORDS = ("enum", "const", "default", "examples")


def _walk_keywords(node: Any, found: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            if key in _SCHEMA_VALUE_KEYWORDS:
                continue
            if key in _SCHEMA_MAP_KEYWORDS:
                for subschema in value.values():
                    _walk_keywords(subschema, found)
            elif key in _SCHEMA_LIST_KEYWORDS:
                for subschema in value:
                    _walk_keywords(subschema, found)
            else:
                _walk_keywords(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_keywords(item, found)


def test_bb10_schemas_parse_and_declare_v1_contracts():
    expected_ids = {
        HEARTBEAT_SCHEMA: "awb-debug-heartbeat/v1",
        SUPERVISOR_SCHEMA: "awb-debug-supervisor-result/v1",
        ESCALATION_SCHEMA: "awb-debug-native-escalation/v1",
    }
    for name, schema_id in expected_ids.items():
        payload = _read_json(SCHEMA_DIR / name)
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["$id"] == schema_id
        assert payload["$id"].endswith("/v1")
        assert payload["type"] == "object"
        assert payload["additionalProperties"] is False
        assert payload["title"].endswith("v1")
        assert "rigped" not in json.dumps(payload, ensure_ascii=False).lower()


def test_bb10_schemas_only_use_supported_keywords():
    for name in BB10_SCHEMAS:
        found: set[str] = set()
        _walk_keywords(_read_json(SCHEMA_DIR / name), found)
        unsupported = found - SchemaValidator.SUPPORTED_KEYWORDS
        assert not unsupported, f"{name} uses unsupported keywords: {sorted(unsupported)}"


def test_bb10_escalation_reuses_supervisor_definitions():
    escalation = _read_json(SCHEMA_DIR / ESCALATION_SCHEMA)
    refs: set[str] = set()

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(SUPERVISOR_SCHEMA):
                refs.add(ref)
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(escalation)
    assert refs == {
        SUPERVISOR_SCHEMA,
        f"{SUPERVISOR_SCHEMA}#/$defs/termination_decision",
        f"{SUPERVISOR_SCHEMA}#/$defs/artifact",
    }
    supervisor = _read_json(SCHEMA_DIR / SUPERVISOR_SCHEMA)
    assert "termination_decision" in supervisor["$defs"]
    assert "artifact" in supervisor["$defs"]


def test_bb10_artifact_inventory_is_explicitly_bounded():
    supervisor = _read_json(SCHEMA_DIR / SUPERVISOR_SCHEMA)
    escalation = _read_json(SCHEMA_DIR / ESCALATION_SCHEMA)
    for schema in (supervisor, escalation):
        artifacts = schema["properties"]["artifacts"]
        assert artifacts["type"] == "array"
        assert artifacts["maxItems"] == 64
    item = supervisor["$defs"]["artifact"]
    for name in (
        "size_bytes",
        "sha256",
        "hash_algorithm",
        "source_path",
        "collected_utc",
        "missing_reason",
        "status",
    ):
        assert name in item["required"]
        assert name in item["properties"]
    assert item["properties"]["sha256"]["pattern"] == "^[0-9a-f]{64}$"
    assert item["properties"]["hash_algorithm"]["enum"] == ["sha256", None]
    assert item["properties"]["size_bytes"]["type"] == ["integer", "null"]
    assert escalation["properties"]["recommended_next_actions"]["maxItems"] == 16


def test_bb10_supervisor_initiated_termination_is_explicitly_encoded():
    supervisor = _read_json(SCHEMA_DIR / SUPERVISOR_SCHEMA)
    termination = supervisor["$defs"]["termination_decision"]
    assert supervisor["properties"]["termination"]["required"] == [
        "supervisor_initiated_termination"
    ]
    for name in (
        "initiated",
        "initiated_by",
        "reason_code",
        "requested_utc",
        "grace_period_seconds",
        "forced",
        "capture_attempted_before_termination",
        "capture_completed",
    ):
        assert name in termination["required"]
    escalation = _read_json(SCHEMA_DIR / ESCALATION_SCHEMA)
    assert "supervisor_initiated_termination" in escalation["required"]
    escalation_branches = escalation["properties"]["supervisor_initiated_termination"]["anyOf"]
    assert escalation_branches[0] == {"type": "null"}
    assert escalation_branches[1]["$ref"].endswith("#/$defs/termination_decision")


@pytest.mark.parametrize("termination_kind", [None, "KILL_ON_JOB_CLOSE"])
def test_native_escalation_writer_output_validates_against_checked_in_schema(
    termination_kind: str | None,
):
    temp_root = ROOT / "tests" / ".awb_schema_writer_tmp"
    temp_root.mkdir(exist_ok=True)
    test_dir = temp_root / uuid4().hex
    test_dir.mkdir()
    bundle = test_dir / "bundle"
    bundle.mkdir()
    try:
        incident_id = "INC-20260926-120000-a1b2c3d4"
        manifest_bytes = (
            json.dumps(
                {
                    "schema": native_evidence_module.MANIFEST_SCHEMA,
                    "incident_id": incident_id,
                    "capture_status": "COMPLETE",
                    "artifacts": {},
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        (bundle / "manifest.json").write_bytes(manifest_bytes)
        source = test_dir / "private" / "native dump.dmp"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"native dump")
        native_evidence_module.copy_native_artifact(bundle, source)

        if termination_kind is None:
            native_evidence_module.write_native_escalation(
                bundle,
                incident_id=incident_id,
            )
        else:
            native_evidence_module.write_native_escalation(
                bundle,
                incident_id=incident_id,
                termination_kind=termination_kind,
            )
        payload = json.loads(
            (bundle / "native_escalation.json").read_text(encoding="utf-8")
        )

        validator = SchemaValidator.from_file(ESCALATION_SCHEMA)
        errors = validator.errors(payload)
        assert not errors, "\n".join(errors)
        assert payload["preconditions"]["incident_manifest_sha256"] == hashlib.sha256(
            manifest_bytes
        ).hexdigest()
        if termination_kind is None:
            assert payload["supervisor_initiated_termination"] is None
        else:
            assert payload["supervisor_initiated_termination"]["initiated"] is True
            assert payload["supervisor_initiated_termination"]["reason_code"] == "UNCLASSIFIED"
        assert payload["verdict"]["responsibility"] == "UNDETERMINED"
        assert (bundle / "manifest.json").read_bytes() == manifest_bytes
        serialized = json.dumps(payload)
        assert str(source) not in serialized
        assert str(source.parent) not in serialized
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        if temp_root.exists() and not any(temp_root.iterdir()):
            temp_root.rmdir()


def test_bb10_incident_identity_is_nullable_but_bound_explicitly():
    for name in (SUPERVISOR_SCHEMA, ESCALATION_SCHEMA):
        schema = _read_json(SCHEMA_DIR / name)
        assert schema["properties"]["incident_id"]["type"] == ["string", "null"]
        assert (
            schema["properties"]["incident_id"]["pattern"]
            == "^INC-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$"
        )
        assert schema["properties"]["incident_id_status"]["enum"] == [
            "BOUND",
            "PENDING",
            "UNAVAILABLE",
        ]
    heartbeat = _read_json(SCHEMA_DIR / HEARTBEAT_SCHEMA)
    assert heartbeat["properties"]["incident_id"]["type"] == ["string", "null"]


def test_bb10_heartbeat_vocabulary_is_conservative():
    heartbeat = _read_json(SCHEMA_DIR / HEARTBEAT_SCHEMA)
    classification = heartbeat["properties"]["classification"]["enum"]
    assert "CRASHED" not in classification
    assert "HUNG" not in classification
    assert set(classification) == {
        "ALIVE_RESPONSIVE",
        "ALIVE_UNRESPONSIVE",
        "ALIVE_STALLED",
        "HEARTBEAT_MISSING",
        "PROCESS_EXITED",
        "PROCESS_ABSENT",
        "UNAVAILABLE",
    }
    assert heartbeat["properties"]["confidence"]["enum"] == [
        "PROVEN",
        "SUSPECTED",
        "UNDETERMINED",
        "UNAVAILABLE",
    ]


def test_bb10_representative_payloads_match_schema_expectations():
    assert len(VALIDATION_CASES) >= 50
    validators = {name: SchemaValidator.from_file(name) for name in BB10_SCHEMAS}
    for schema_name, payload, expected, case_id in VALIDATION_CASES:
        errors = validators[schema_name].errors(payload)
        assert bool(errors) is not expected, (
            f"{case_id}: {schema_name} expected valid={expected} but reported {errors}"
        )


@pytest.mark.parametrize("schema_name,payload,expected", VALIDATION_PARAMS)
def test_bb10_payloads_agree_with_optional_jsonschema_engine(
    schema_name: str,
    payload: Any,
    expected: bool,
):
    external = _external_is_valid(schema_name, payload)
    if external is None:
        pytest.skip("jsonschema is not installed; stdlib validator is the only engine")
    assert external is expected


def test_stdlib_validator_rejects_and_accepts_representative_shapes():
    validator = SchemaValidator.from_file(HEARTBEAT_SCHEMA)
    assert validator.is_valid(_heartbeat_alive_payload())
    assert not validator.is_valid({})
    assert not validator.is_valid(_mutate(_heartbeat_alive_payload(), sequence="seven"))
    inline = SchemaValidator(
        {
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": False,
        },
        SCHEMA_DIR / HEARTBEAT_SCHEMA,
    )
    assert inline.is_valid({"a": 1})
    assert not inline.is_valid({"a": 1, "b": 2})
    assert not inline.is_valid({})
    assert not inline.is_valid({"a": True})
    enum_null = SchemaValidator({"enum": ["sha256", None]}, SCHEMA_DIR / HEARTBEAT_SCHEMA)
    assert enum_null.is_valid(None)
    assert enum_null.is_valid("sha256")
    assert not enum_null.is_valid(0)
    assert not enum_null.is_valid(False)


def test_stdlib_validator_resolves_local_and_cross_file_refs():
    validator = SchemaValidator.from_file(ESCALATION_SCHEMA)
    assert validator.is_valid(_escalation_analyzed_payload())
    assert not validator.is_valid(_mutate(_escalation_analyzed_payload(), escalation_id=""))
    with_termination = _escalation_analyzed_payload()
    assert validator.is_valid(with_termination)
    broken_termination = _mutate(
        with_termination,
        supervisor_initiated_termination={"initiated": True, "initiated_by": "SUPERVISOR"},
    )
    assert not validator.is_valid(broken_termination)
    assert any(
        "supervisor_initiated_termination" in error
        for error in validator.errors(broken_termination)
    )
    missing_artifact = _mutate(
        with_termination,
        artifacts=[_artifact(0, status="MISSING", sha256=SHA_A)],
    )
    assert not validator.is_valid(missing_artifact)


def test_bb10_schemas_are_independent_of_product_capture_code():
    capture = (ROOT / "scripts" / "capture_awb_incident.py").read_text(encoding="utf-8")
    assert "awb-debug-heartbeat/v1" not in capture
    assert "awb-debug-supervisor-result/v1" not in capture
    assert "awb-debug-native-escalation/v1" not in capture
    for name in BB10_SCHEMAS:
        assert (SCHEMA_DIR / name).is_file()


RAW_HEARTBEAT_REQUIRED = ("schema", "incident_id", "pid", "sequence", "utc")
RAW_HEARTBEAT_OPTIONAL = ("awb_session_id", "trace_id")
RAW_HEARTBEAT_PROPERTIES = RAW_HEARTBEAT_REQUIRED + RAW_HEARTBEAT_OPTIONAL

# Names the analyzed awb-debug-heartbeat/v1 vocabulary uses but the raw
# canonical record must never carry, and child/supervisor monotonic verdicts
# that must not leak into the child's own file.
RAW_HEARTBEAT_FORBIDDEN_NAMES = (
    "classification",
    "classification_reason",
    "confidence",
    "heartbeat_id",
    "observed_utc",
    "monitor_role",
    "interval_seconds",
    "stale_threshold_seconds",
    "last_heartbeat_utc",
    "age_seconds",
    "process",
    "writer",
    "evidence",
    "stale",
    "sequence_advanced",
    "sequence_regressed",
    "elapsed_since_advance",
    "monotonic_seconds",
    "monotonic",
    "uptime_seconds",
    "elapsed_seconds",
    "exited",
    "exit_code",
    "alive",
)


def test_raw_supervisor_heartbeat_schema_is_a_closed_minimal_record():
    schema = _read_json(SCHEMA_DIR / RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "awb-supervisor-heartbeat/v1"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["title"].endswith("v1")
    assert schema["properties"]["schema"]["const"] == "awb-supervisor-heartbeat/v1"
    assert tuple(schema["required"]) == RAW_HEARTBEAT_REQUIRED
    assert set(schema["properties"]) == set(RAW_HEARTBEAT_PROPERTIES)
    for name in RAW_HEARTBEAT_REQUIRED:
        assert name in schema["properties"]
    for name in RAW_HEARTBEAT_OPTIONAL:
        assert name in schema["properties"]
        assert name not in schema["required"]
        assert schema["properties"][name]["type"] == "string"
        assert schema["properties"][name]["minLength"] == 1
    serialized = json.dumps(schema, ensure_ascii=False)
    assert "rigped" not in serialized.lower()
    # No classification and no child monotonic or staleness verdict may appear,
    # not even inside a description.
    for name in RAW_HEARTBEAT_FORBIDDEN_NAMES:
        assert f'"{name}"' not in serialized, f"raw record must not mention {name!r}"


def test_raw_supervisor_heartbeat_schema_field_bounds_match_runtime_validation():
    schema = _read_json(SCHEMA_DIR / RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    properties = schema["properties"]
    # Heartbeat.__post_init__ requires a positive integer pid and rejects bool.
    assert properties["pid"]["type"] == "integer"
    assert properties["pid"]["exclusiveMinimum"] == 0
    # Heartbeat.__post_init__ accepts sequence 0 and rejects negatives.
    assert properties["sequence"]["type"] == "integer"
    assert properties["sequence"]["minimum"] == 0
    # _validate_utc requires a non-empty ISO-8601 string with a UTC timezone.
    assert properties["utc"]["type"] == "string"
    assert properties["utc"]["minLength"] == 1
    assert properties["utc"]["pattern"] == "(Z|[+-]00:00)$"
    # incident_id and the optional identities must be non-blank, like the code.
    assert properties["incident_id"]["minLength"] == 1
    assert properties["incident_id"]["pattern"] == "\\S"
    for name in RAW_HEARTBEAT_OPTIONAL:
        assert properties[name]["pattern"] == "\\S"


def test_raw_supervisor_heartbeat_schema_only_uses_supported_keywords():
    found: set[str] = set()
    _walk_keywords(_read_json(SCHEMA_DIR / RAW_SUPERVISOR_HEARTBEAT_SCHEMA), found)
    unsupported = found - SchemaValidator.SUPPORTED_KEYWORDS
    assert not unsupported, f"unsupported keywords: {sorted(unsupported)}"


def test_raw_supervisor_heartbeat_schema_stays_distinct_from_analyzed_heartbeat():
    raw = _read_json(SCHEMA_DIR / RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    analyzed = _read_json(SCHEMA_DIR / HEARTBEAT_SCHEMA)
    assert raw["$id"] != analyzed["$id"]
    # Only the generic identity carriers are shared; the raw record has no
    # heartbeat_id / observed_utc / freshness vocabulary of the analyzed one.
    assert set(raw["properties"]) & set(analyzed["properties"]) == {
        "schema",
        "incident_id",
        "sequence",
    }
    for name in ("classification", "confidence", "monitor_role", "evidence"):
        assert name in analyzed["properties"]
        assert name not in raw["properties"]
    raw_validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    analyzed_validator = SchemaValidator.from_file(HEARTBEAT_SCHEMA)
    raw_payload = _raw_heartbeat_payload()
    assert raw_validator.is_valid(raw_payload)
    assert not analyzed_validator.is_valid(raw_payload)
    analyzed_payload = _heartbeat_alive_payload()
    assert analyzed_validator.is_valid(analyzed_payload)
    assert not raw_validator.is_valid(analyzed_payload)


def test_raw_supervisor_heartbeat_shape_cases_match_schema_expectations():
    assert len(RAW_HEARTBEAT_CASES) >= 25
    validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    for schema_name, payload, expected, case_id in RAW_HEARTBEAT_CASES:
        assert schema_name == RAW_SUPERVISOR_HEARTBEAT_SCHEMA
        errors = validator.errors(payload)
        assert bool(errors) is not expected, (
            f"{case_id}: expected valid={expected} but reported {errors}"
        )


@pytest.mark.parametrize("schema_name,payload,expected", RAW_HEARTBEAT_PARAMS)
def test_raw_supervisor_heartbeat_shape_cases_agree_with_optional_jsonschema_engine(
    schema_name: str,
    payload: Any,
    expected: bool,
):
    external = _external_is_valid(schema_name, payload)
    if external is None:
        pytest.skip("jsonschema is not installed; stdlib validator is the only engine")
    assert external is expected


def test_raw_supervisor_heartbeat_optional_identity_is_omitted_not_null():
    validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    bare = _raw_heartbeat_payload()
    assert set(bare) == set(RAW_HEARTBEAT_REQUIRED)
    assert validator.is_valid(bare)
    # The canonical record omits absent optional identity; explicit null is
    # not part of the serialized contract and must fail closed.
    assert not validator.is_valid(_mutate(bare, awb_session_id=None))
    assert not validator.is_valid(_mutate(bare, trace_id=None))
    assert not validator.is_valid(_mutate(bare, awb_session_id=None, trace_id=None))
    assert validator.is_valid(_raw_heartbeat_optional_payload())


def test_raw_supervisor_heartbeat_schema_is_not_confused_with_analysis_vocabulary():
    capture = (ROOT / "scripts" / "capture_awb_incident.py").read_text(encoding="utf-8")
    assert "awb-supervisor-heartbeat/v1" not in capture
    assert (SCHEMA_DIR / RAW_SUPERVISOR_HEARTBEAT_SCHEMA).is_file()


def test_raw_supervisor_heartbeat_schema_id_matches_runtime_constant():
    schema = _read_json(SCHEMA_DIR / RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    assert raw_heartbeat_module.HEARTBEAT_SCHEMA == "awb-supervisor-heartbeat/v1"
    assert schema["$id"] == raw_heartbeat_module.HEARTBEAT_SCHEMA
    assert schema["properties"]["schema"]["const"] == raw_heartbeat_module.HEARTBEAT_SCHEMA


def test_raw_supervisor_heartbeat_schema_field_set_matches_runtime_dataclass():
    by_name = {field.name: field for field in dataclasses.fields(raw_heartbeat_module.Heartbeat)}
    fields = tuple(by_name)
    assert fields == (
        "incident_id",
        "pid",
        "sequence",
        "utc",
        "awb_session_id",
        "trace_id",
    )
    schema = _read_json(SCHEMA_DIR / RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    assert tuple(schema["required"]) == ("schema", *fields[:4])
    assert set(schema["properties"]) == {"schema", *fields}
    # Dataclass requiredness must match schema required/optional split.
    for name in fields[:4]:
        assert by_name[name].default is dataclasses.MISSING
        assert name in schema["required"]
    for name in fields[4:]:
        assert by_name[name].default is None
        assert name not in schema["required"]


@pytest.mark.parametrize(
    "utc",
    [
        "2026-09-26T10:15:00Z",
        "2026-09-26T10:15:00.123Z",
        "2026-09-26T10:15:00.123456Z",
        "2026-09-26T10:15:00+00:00",
        "2026-09-26T10:15:00-00:00",
    ],
)
@pytest.mark.parametrize(
    "optional",
    [
        {},
        {"awb_session_id": "session-3"},
        {"trace_id": "trace-9"},
        {"awb_session_id": "session-3", "trace_id": "trace-9"},
    ],
)
@pytest.mark.parametrize("sequence", [0, 1, 99])
def test_runtime_heartbeat_records_satisfy_raw_schema(utc: str, optional: dict, sequence: int):
    validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    record = raw_heartbeat_module.Heartbeat("incident-7", 1234, sequence, utc, **optional)
    payload = record.to_dict()

    assert validator.is_valid(payload), validator.errors(payload)
    assert raw_heartbeat_module.Heartbeat.from_dict(payload) == record


def test_make_heartbeat_output_satisfies_raw_schema():
    validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    record = raw_heartbeat_module.make_heartbeat(
        incident_id="incident-7",
        pid=1234,
        sequence=0,
        now_utc=datetime(2026, 9, 26, tzinfo=UTC),
    )
    payload = record.to_dict()

    assert payload["schema"] == "awb-supervisor-heartbeat/v1"
    assert payload["utc"].endswith("Z")
    assert set(payload) == set(RAW_HEARTBEAT_REQUIRED)
    assert validator.is_valid(payload), validator.errors(payload)


RAW_RUNTIME_REJECTED_CASES = (
    ("pid-not-positive", {"pid": 0}),
    ("pid-negative", {"pid": -4}),
    ("pid-boolean", {"pid": True}),
    ("pid-not-integer", {"pid": "1234"}),
    ("sequence-negative", {"sequence": -1}),
    ("sequence-boolean", {"sequence": True}),
    ("blank-incident-id", {"incident_id": "   "}),
    ("empty-incident-id", {"incident_id": ""}),
    ("naive-utc", {"utc": "2026-09-26T10:15:00"}),
    ("non-utc-utc", {"utc": "2026-09-26T10:15:00+09:00"}),
    ("blank-session-id", {"awb_session_id": " "}),
    ("blank-trace-id", {"trace_id": ""}),
    ("wrong-schema-tag", {"schema": "awb-debug-heartbeat/v1"}),
    ("unknown-field", {"classification": "ALIVE_STALLED"}),
    ("analyzed-nested-object", {"process": {"pid": 1234}}),
)


@pytest.mark.parametrize(
    "case_id,changes",
    RAW_RUNTIME_REJECTED_CASES,
    ids=[case_id for case_id, _changes in RAW_RUNTIME_REJECTED_CASES],
)
def test_runtime_and_raw_schema_reject_the_same_invalid_records(case_id: str, changes: dict):
    payload = _mutate(_raw_heartbeat_payload(), **changes)
    validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)

    with pytest.raises(
        (raw_heartbeat_module.HeartbeatError, raw_heartbeat_module.HeartbeatCorruptError)
    ):
        raw_heartbeat_module.Heartbeat.from_dict(payload)
    assert not validator.is_valid(payload), f"{case_id} should fail the raw schema"


@pytest.mark.parametrize("name", ("incident_id", "pid", "sequence", "utc"))
def test_runtime_and_raw_schema_require_the_same_fields(name: str):
    payload = _raw_heartbeat_payload()
    payload.pop(name)
    validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)

    with pytest.raises(raw_heartbeat_module.HeartbeatCorruptError):
        raw_heartbeat_module.Heartbeat.from_dict(payload)
    assert not validator.is_valid(payload)


def test_raw_schema_rejects_non_object_root_like_the_runtime():
    validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    for payload in ([], "awb-supervisor-heartbeat/v1", 1, None):
        with pytest.raises(raw_heartbeat_module.HeartbeatCorruptError):
            raw_heartbeat_module.Heartbeat.from_dict(payload)
        assert not validator.is_valid(payload)


def test_absent_optional_identity_is_omitted_and_explicit_null_fails_closed():
    record = raw_heartbeat_module.Heartbeat(
        "incident-7", 1234, 7, "2026-09-26T10:15:00.123Z"
    )
    payload = record.to_dict()
    validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)

    assert "awb_session_id" not in payload
    assert "trace_id" not in payload
    assert validator.is_valid(payload)
    # Known, deliberate strictness: the reader tolerates an explicit null and
    # normalizes it away, but the canonical record never carries null, so the
    # schema fails closed instead of widening the serialized contract.
    tolerant = _mutate(_raw_heartbeat_payload(), awb_session_id=None, trace_id=None)
    assert raw_heartbeat_module.Heartbeat.from_dict(tolerant) == record
    assert not validator.is_valid(tolerant)


def test_raw_schema_does_not_widen_the_runtime_allowed_field_set():
    validator = SchemaValidator.from_file(RAW_SUPERVISOR_HEARTBEAT_SCHEMA)
    runtime_names = {
        field.name for field in dataclasses.fields(raw_heartbeat_module.Heartbeat)
    } | {"schema"}
    schema_names = set(_read_json(SCHEMA_DIR / RAW_SUPERVISOR_HEARTBEAT_SCHEMA)["properties"])
    assert schema_names == runtime_names
    for name in sorted(RAW_HEARTBEAT_FORBIDDEN_NAMES):
        assert name not in schema_names
        assert not validator.is_valid(_mutate(_raw_heartbeat_payload(), **{name: None}))
