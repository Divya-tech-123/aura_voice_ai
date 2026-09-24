"""Tool Registry for AURA.

Enables central registration, discovery, schema introspection, and safe execution
of tools available to the AURA Brain.
"""

from typing import Dict, List, Optional, Any
import logging

from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.weather import WeatherTool
from app.tools.search import SearchTool
from app.tools.files import FileTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry storing and providing tools to AURA subsystems."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a new tool.

        Args:
            tool: An instance implementing BaseTool.

        Raises:
            TypeError: If tool is not an instance of BaseTool.
            ValueError: If tool does not have a valid name.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool instance, got {type(tool).__name__}.")

        name = (tool.name or "").strip().lower()
        if not name:
            raise ValueError("Tool must have a non-empty 'name'.")

        self._tools[name] = tool
        logger.debug(f"Registered tool: '{name}'")

    def get(self, name: str) -> Optional[BaseTool]:
        """Look up a tool by its registered name.

        Args:
            name: The tool name (case-insensitive).

        Returns:
            The BaseTool instance, or None if not found.
        """
        clean_name = (name or "").strip().lower()
        return self._tools.get(clean_name)

    def list_tools(self) -> List[str]:
        """List the names of all registered tools.

        Returns:
            List of registered tool names sorted alphabetically.
        """
        return sorted(list(self._tools.keys()))

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Retrieve schemas for all registered tools, formatted for LLM prompts.

        Returns:
            List of dictionary schemas with 'name', 'description', and 'parameters'.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute a registered tool safely by name.

        Args:
            tool_name: The name of the tool to execute.
            **kwargs: Arguments to pass to the tool.

        Returns:
            Structured result dictionary with 'success', 'result' or 'data', and 'error'.
        """
        clean_name = (tool_name or "").strip().lower()
        tool = self.get(clean_name)

        if not tool:
            available = ", ".join(self.list_tools()) or "none"
            return {
                "success": False,
                "result": None,
                "error": f"Tool '{tool_name}' not found. Available tools: [{available}].",
            }

        return tool.run(**kwargs)

    def __contains__(self, name: str) -> bool:
        return (name or "").strip().lower() in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def create_default_registry() -> ToolRegistry:
    """Factory creating a standard ToolRegistry populated with AURA's core tools.

    Returns:
        ToolRegistry instance with calculator, weather, search, and files registered.
    """
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(SearchTool())
    registry.register(FileTool())
    return registry
