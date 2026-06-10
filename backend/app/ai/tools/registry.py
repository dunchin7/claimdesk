"""Agent tool registry (Week 11).

Each tool is a (schema, function) pair:
- Pydantic input model — the LLM emits arguments validated against this
- async function — receives the validated input, returns a Pydantic output
- Pydantic output model — serialized to JSON for the next LLM turn

The registry surfaces tools to the model in OpenAI's `tools=[...]` format
(LiteLLM normalizes across providers). The agent loop looks up the
function by name when the model emits a tool_call.

Design rules:
- Tools are async (DB / network). Sync tools wrap their work in
  `asyncio.to_thread`.
- Tools NEVER raise to the caller. They return an error-shaped output
  (`status="error", message="..."`) so the LLM can read the error in
  its next turn and decide what to do.
- Side-effecting tools (`create_rma`, `escalate_to_human`) take an
  idempotency_key so re-runs from a checkpoint don't double-fire.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class ToolSpec:
    """One tool's metadata + handler."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[BaseModel]]


# Module-level registry. Populated by `register_tool()` on import of the
# individual tool modules. The agent loop reads `_TOOLS` to build the
# OpenAI tools schema and to dispatch on tool name.
_TOOLS: dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> None:
    """Register a tool. Idempotent on `name`."""
    _TOOLS[spec.name] = spec


def get_tool(name: str) -> ToolSpec | None:
    return _TOOLS.get(name)


def all_tools() -> dict[str, ToolSpec]:
    return dict(_TOOLS)


def to_openai_tools_schema(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Render the registry as the `tools=[...]` array OpenAI/LiteLLM expects.

    Pass `names` to filter — useful when the agent should only see a subset
    of tools for a given phase (e.g., disable RMA creation in a dry-run).
    """
    tools = _TOOLS if names is None else {n: _TOOLS[n] for n in names if n in _TOOLS}
    out: list[dict[str, Any]] = []
    for spec in tools.values():
        schema = spec.input_model.model_json_schema()
        # Pydantic 2 emits "$defs" alongside the top-level keys; OpenAI
        # accepts them either way.
        out.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": schema,
            },
        })
    return out


async def dispatch_tool(name: str, args_json: str) -> tuple[bool, str]:
    """Run a tool by name with JSON-encoded args.

    Returns (ok, serialized_output_json). `ok=False` means the tool's
    output model has `status=="error"` OR validation failed before
    dispatch. The agent loop feeds `serialized_output_json` back to the
    LLM regardless — the model is the recovery mechanism.
    """
    spec = get_tool(name)
    if spec is None:
        return False, json.dumps({
            "status": "error",
            "message": f"Unknown tool {name!r}. Known: {sorted(_TOOLS)}",
        })
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        return False, json.dumps({
            "status": "error",
            "message": f"Tool arguments are not valid JSON: {e}",
        })
    try:
        validated = spec.input_model.model_validate(args)
    except Exception as e:  # noqa: BLE001 — Pydantic ValidationError or anything
        return False, json.dumps({
            "status": "error",
            "message": f"Tool arguments failed validation: {e}",
        })
    try:
        result = await spec.handler(validated)
    except Exception as e:  # noqa: BLE001
        return False, json.dumps({
            "status": "error",
            "message": f"Tool execution failed: {type(e).__name__}: {e}",
        })
    payload = result.model_dump(mode="json")
    is_err = isinstance(payload, dict) and payload.get("status") == "error"
    return (not is_err), json.dumps(payload, default=str)
