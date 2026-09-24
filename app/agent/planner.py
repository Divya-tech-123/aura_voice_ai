"""AURA Agent Goal Planner.

Provides a rule-based planning engine that decomposes user goals
into an ordered sequence of discrete execution steps and actions.
"""

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AURA.Agent.Planner")


class Planner:
    """Decomposes high-level user requests into structured, executable plans."""

    # Regex pattern to identify arithmetic expressions (e.g. '25 * 48', '10 + 5')
    MATH_PATTERN = re.compile(r"\b\d+\s*[\+\-\*\/\%x\^]\s*\d+\b", re.IGNORECASE)

    MATH_KEYWORDS = {
        "calculate", "compute", "calculator", "multiply", "divide",
        "plus", "minus", "sum", "math", "arithmetic", "sqrt", "equation"
    }

    WEATHER_KEYWORDS = {
        "weather", "temperature", "forecast", "climate", "rain",
        "humidity", "sunny", "wind", "celsius", "fahrenheit"
    }

    SEARCH_KEYWORDS = {
        "search", "lookup", "find out", "google",
        "browse", "latest news", "internet", "web search"
    }

    FILE_KEYWORDS = {
        "read file", "list files", "write file", "save file",
        "filesystem", "directory", "folder"
    }

    DOCUMENT_KEYWORDS = {
        "according to my document", "according to my documents",
        "according to the document", "according to the documents",
        "according to our document", "according to our documents",
        "according to my notes", "according to the notes",
        "according to documents", "according to notes",
        "in my document", "in my documents",
        "in the document", "in the documents",
        "in my notes", "in the notes",
        "from my document", "from my documents",
        "from the document", "from the documents",
        "from my notes", "from the notes",
        "my notes document", "my document", "my documents",
        "my notes", "our documents", "our notes",
        "document says", "document say", "documents say",
        "notes say", "notes says",
        "retrieve from document", "retrieve from documents",
    }

    DOCUMENT_PATTERNS = [
        re.compile(r"\baccording\s+to\s+(?:my|the|our)?\s*(?:ai\s+)?(?:documents?|notes|docs?|pdf)\b", re.IGNORECASE),
        re.compile(r"\b(?:what\s+does|what\s+do)\s+(?:my|the|our)?\s*(?:ai\s+)?(?:documents?|notes|docs?|pdf)\s+say\b", re.IGNORECASE),
        re.compile(r"\b(?:in|from|based\s+on)\s+(?:my|the|our)\s+(?:ai\s+)?(?:documents?|notes|docs?|pdf)\b", re.IGNORECASE),
        re.compile(r"\b(?:my|our)\s+(?:ai\s+)?(?:notes|documents?|docs?|pdf)\b", re.IGNORECASE),
        re.compile(r"\b(?:search|check|consult|look\s+up\s+in|find(?:\s+information)?\s+in)\s+(?:my|the|our)?\s*(?:documents?|notes|docs?)\b", re.IGNORECASE),
        re.compile(r"\b(?:find|retrieve)\s+.*?\b(?:in|from)\s+(?:my|the|our)?\s*(?:documents?|notes|docs?)\b", re.IGNORECASE),
        re.compile(r"\b(?:document|documents|notes)\s+say\b", re.IGNORECASE),
        re.compile(r"\brag\b", re.IGNORECASE),
    ]

    def is_document_query(self, goal: str) -> bool:
        """Determine if a user query requires document retrieval (RAG).

        Args:
            goal: User prompt or instruction.

        Returns:
            True if the query references documents, notes, or knowledge retrieval; False otherwise.
        """
        clean = (goal or "").strip().lower()
        if not clean:
            return False
        if any(kw in clean for kw in self.DOCUMENT_KEYWORDS):
            return True
        return any(bool(pattern.search(clean)) for pattern in self.DOCUMENT_PATTERNS)

    def create_plan(
        self,
        goal: str,
        context: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze user goal and conversation context to generate an ordered list of planned actions.

        Args:
            goal: User prompt or instruction.
            context: Optional conversation context or history to aid planning.

        Returns:
            List of step dictionaries containing 'step', 'action', and 'reason'.
        """
        clean_goal = (goal or "").strip().lower()
        plan: List[Dict[str, Any]] = []
        step_number = 1

        if not clean_goal:
            return [{
                "step": 1,
                "action": "respond",
                "reason": "Explain result to user"
            }]

        # Check for document retrieval needs (RAG)
        has_retrieve = self.is_document_query(clean_goal)
        if has_retrieve:
            plan.append({
                "step": step_number,
                "action": "retrieve",
                "reason": "Retrieve relevant context from documents"
            })
            step_number += 1

        # Check for mathematical computation needs
        has_math = any(kw in clean_goal for kw in self.MATH_KEYWORDS) or bool(self.MATH_PATTERN.search(clean_goal))
        if has_math:
            plan.append({
                "step": step_number,
                "action": "calculator",
                "reason": "Need mathematical computation"
            })
            step_number += 1

        # Check for weather query needs
        has_weather = any(kw in clean_goal for kw in self.WEATHER_KEYWORDS)
        if has_weather:
            plan.append({
                "step": step_number,
                "action": "weather",
                "reason": "Retrieve weather information"
            })
            step_number += 1

        # Check for web search needs
        has_search = any(kw in clean_goal for kw in self.SEARCH_KEYWORDS)
        if has_search and not has_weather and not has_retrieve:  # Avoid duplicate triggers
            plan.append({
                "step": step_number,
                "action": "search",
                "reason": "Search external web sources for information"
            })
            step_number += 1

        # Check for file system needs
        has_files = any(kw in clean_goal for kw in self.FILE_KEYWORDS)
        if has_files and not has_retrieve:
            plan.append({
                "step": step_number,
                "action": "files",
                "reason": "Access local file system or document storage"
            })
            step_number += 1

        # Terminal action: explain or respond to the user
        plan.append({
            "step": step_number,
            "action": "respond",
            "reason": "Explain result to user"
        })

        logger.info(f"Generated plan with {len(plan)} steps for goal: '{goal}'")
        return plan
