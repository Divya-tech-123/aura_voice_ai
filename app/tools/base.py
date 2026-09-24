"""Base tool interface for AURA.

Defines a clean, unified contract for executable tools with input validation,
structured execution results, and safe exception handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class defining the contract for all AURA executable tools."""

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool's core logic with validated inputs.

        Args:
            **kwargs: Tool-specific arguments defined in self.parameters.

        Returns:
            Dict containing at minimum:
                - 'success': bool indicating outcome
                - 'result' or 'data': The primary payload (or None on failure)
                - 'error': Error description string if success is False, else None.
        """
        pass

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Safely execute the tool, wrapping any unexpected exception into a structured failure.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            Structured result dictionary with 'success', 'result', and 'error'.
        """
        try:
            return self.execute(**kwargs)
        except Exception as exc:
            logger.exception(f"Unhandled error executing tool '{self.name}': {exc}")
            return {
                "success": False,
                "result": None,
                "error": f"Tool '{self.name}' failed: {str(exc)}",
            }

    def to_schema(self) -> Dict[str, Any]:
        """Export tool definition and input schema for LLM tool selection prompts.

        Returns:
            Dictionary describing the tool name, purpose, and required arguments.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
