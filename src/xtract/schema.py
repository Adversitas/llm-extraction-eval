"""Output schema, tolerant JSON parsing and validation."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Entities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persons: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    misc: list[str] = Field(default_factory=list)


SCHEMA_HINT = json.dumps(
    {"persons": ["..."], "organizations": ["..."], "locations": ["..."], "misc": ["..."]}
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json_block(text: str) -> str | None:
    """Return the first balanced {...} block (inside a code fence if there is one)."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse(text: str) -> tuple[Entities | None, str | None]:
    """Parse model output into ``Entities``. Returns (value, error message)."""
    block = extract_json_block(text)
    if block is None:
        return None, "no JSON object found"
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    try:
        return Entities.model_validate(data), None
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        return None, f"schema error at {loc}: {first['msg']}"
