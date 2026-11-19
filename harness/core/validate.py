"""Minimal JSON-schema-subset validator. Pure, stdlib-only, so it can live
in the core: both the runner (envelope/data validation) and gate adapters
use it without depending outward.

Supports exactly these keywords — anything else in a schema is a spec
error, never silently ignored: type, properties, required, items,
anyOf, enum. Types are string/integer/number/boolean/object/array/null,
plus union types in two forms:

  - a JSON-Schema-style type list:  {"type": ["string", "null"]}
  - anyOf:                          {"anyOf": [{"type": "string"},
                                               {"type": "null"}]}

A union means the value must match at least one member. Nullability is
just a union with "null": a key listed in `required` must be *present*,
but its value may be None when the union includes "null". `items` is
the element schema for arrays, checked recursively.

Validation fails closed throughout: an unknown type name is a problem
in validate_schema AND a mismatch in validate_data (a misspelled type
must not silently disable enforcement), a malformed union or items
spec is a problem, and a schema keyword outside the supported set is
rejected by validate_schema rather than ignored.

validate_data returns a list of problem strings; empty means valid.
validate_schema checks a schema itself; build-time definition
validation rejects a schema when it returns problems, raising
DefinitionError there instead of failing at runtime.
"""

from __future__ import annotations

KNOWN_TYPES = frozenset(
    {"string", "integer", "number", "boolean", "object", "array", "null"})

SUPPORTED_KEYWORDS = frozenset(
    {"type", "properties", "required", "items", "anyOf", "enum"})


def _check_type(value, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "null":
        return value is None
    return False  # unknown type: fail closed, never silently pass


def _union_problems(type_spec, path: str) -> list[str]:
    """Problems with a type *list* (union), or [] if it is well-formed.
    Both forms are strict: a single type name must be a known type,
    and every union member must be one too."""
    if not isinstance(type_spec, list):
        return []
    if not type_spec:
        return [f"{path}: type union must not be empty"]
    problems = []
    for member in type_spec:
        if not isinstance(member, str) or member not in KNOWN_TYPES:
            problems.append(
                f"{path}: unknown union member type {member!r} "
                f"(known: {sorted(KNOWN_TYPES)})")
    return problems


def validate_schema(schema, path: str = "$") -> list[str]:
    """Check a schema for spec errors, recursively: unsupported
    keywords, unknown type names (single or in unions), malformed
    unions, malformed items/properties/required/enum. Returns problem
    strings; empty means the schema is within the supported subset
    and well-formed. Build-time callers (app/builder) turn problems
    into DefinitionError."""
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object"]
    problems: list[str] = []
    for keyword in schema:
        if keyword not in SUPPORTED_KEYWORDS:
            problems.append(
                f"{path}: unsupported schema keyword '{keyword}' "
                f"(supported: {sorted(SUPPORTED_KEYWORDS)})")
    if "type" in schema:
        type_spec = schema["type"]
        if isinstance(type_spec, list):
            problems.extend(_union_problems(type_spec, path))
        elif isinstance(type_spec, str):
            if type_spec not in KNOWN_TYPES:
                problems.append(
                    f"{path}: unknown type '{type_spec}' "
                    f"(known: {sorted(KNOWN_TYPES)})")
        else:
            problems.append(
                f"{path}: type must be a string or a list of strings")
    if "anyOf" in schema:
        options = schema["anyOf"]
        if not isinstance(options, list) or not options:
            problems.append(f"{path}: anyOf must be a non-empty list")
        else:
            for i, option in enumerate(options):
                problems.extend(
                    validate_schema(option, f"{path}.anyOf[{i}]"))
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict):
            problems.append(f"{path}: properties must be an object")
        else:
            for key, subschema in properties.items():
                problems.extend(
                    validate_schema(subschema, f"{path}.properties.{key}"))
    if "items" in schema:
        items = schema["items"]
        if not isinstance(items, dict):
            problems.append(f"{path}: items must be a schema object")
        else:
            problems.extend(validate_schema(items, f"{path}.items"))
    if "required" in schema:
        required = schema["required"]
        if (not isinstance(required, list)
                or not all(isinstance(k, str) for k in required)):
            problems.append(f"{path}: required must be a list of strings")
    if "enum" in schema:
        if not isinstance(schema["enum"], list) or not schema["enum"]:
            problems.append(f"{path}: enum must be a non-empty list")
    return problems


def validate_data(data, schema: dict, path: str = "$") -> list[str]:
    problems: list[str] = []
    if not isinstance(schema, dict):
        return problems
    type_spec = schema.get("type")
    if isinstance(type_spec, list):
        union_problems = _union_problems(type_spec, path)
        if union_problems:
            return union_problems  # malformed spec: fail closed
        if not any(_check_type(data, member) for member in type_spec):
            problems.append(
                f"{path}: expected one of {type_spec}, "
                f"got {type(data).__name__}")
            return problems  # no deeper checks after a type mismatch
    elif type_spec and not _check_type(data, type_spec):
        problems.append(
            f"{path}: expected {type_spec}, got {type(data).__name__}")
        return problems  # no deeper checks after a type mismatch
    if "anyOf" in schema:
        options = schema["anyOf"]
        if not isinstance(options, list) or not options:
            return [f"{path}: anyOf must be a non-empty list"]
        if not any(validate_data(data, option, path) == []
                   for option in options):
            problems.append(f"{path}: does not match any anyOf option")
            return problems
    if isinstance(data, dict):
        for required_key in schema.get("required", []):
            if required_key not in data:
                problems.append(f"{path}: missing required key '{required_key}'")
        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                problems.extend(validate_data(data[key], subschema, f"{path}.{key}"))
    if "items" in schema:
        items_schema = schema["items"]
        if not isinstance(items_schema, dict):
            problems.append(f"{path}: items must be a schema object")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                problems.extend(
                    validate_data(item, items_schema, f"{path}[{i}]"))
    if "enum" in schema and data not in schema["enum"]:
        problems.append(f"{path}: {data!r} not one of {schema['enum']}")
    return problems
