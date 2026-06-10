"""Agent tools package.

Importing this module side-effect-registers all tools with the registry.
"""

from app.ai.tools import actions, retrieval, shopify, vision_tool  # noqa: F401
from app.ai.tools.registry import (  # noqa: F401
    ToolSpec,
    all_tools,
    dispatch_tool,
    get_tool,
    register_tool,
    to_openai_tools_schema,
)
