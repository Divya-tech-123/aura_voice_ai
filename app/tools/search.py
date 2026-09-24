"""Web search tool for AURA.

Executes web search queries to fetch real-time facts and references using
structured provider adapters without arbitrary or fragile web scraping.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import requests

from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Search Provider Interface & Implementations
# ==============================================================================

class BaseSearchProvider(ABC):
    """Abstract interface for web search providers."""

    @abstractmethod
    def search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """Execute a search query and return structured results.

        Args:
            query: The search query text.
            num_results: Maximum number of search results to return.

        Returns:
            Dict containing 'success' (bool), 'query' (str), 'results' (list), and 'error' (str/None).
        """
        pass


class SerpAPISearchProvider(BaseSearchProvider):
    """Production search provider using the SerpAPI Google Search engine."""

    BASE_URL = "https://serpapi.com/search.json"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0) -> None:
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY", "").strip()
        self.timeout = timeout

    def search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        clean_query = (query or "").strip()
        if not clean_query:
            return {
                "success": False,
                "query": "",
                "results": [],
                "error": "Search query cannot be empty.",
            }

        if not self.api_key:
            return {
                "success": False,
                "query": clean_query,
                "results": [],
                "error": (
                    "SerpAPI key is not configured. "
                    "Please set SERPAPI_API_KEY in your .env file."
                ),
            }

        params = {
            "q": clean_query,
            "api_key": self.api_key,
            "num": max(1, min(num_results, 10)),
            "engine": "google",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "query": clean_query,
                "results": [],
                "error": f"Search provider timed out after {self.timeout}s.",
            }
        except requests.exceptions.RequestException as exc:
            return {
                "success": False,
                "query": clean_query,
                "results": [],
                "error": f"Failed to connect to search provider: {exc}",
            }

        if response.status_code == 401:
            return {
                "success": False,
                "query": clean_query,
                "results": [],
                "error": "Authentication failed: Invalid SerpAPI key (HTTP 401).",
            }
        elif response.status_code == 429:
            return {
                "success": False,
                "query": clean_query,
                "results": [],
                "error": "Search rate limit or monthly quota exceeded (HTTP 429).",
            }
        elif response.status_code >= 400:
            return {
                "success": False,
                "query": clean_query,
                "results": [],
                "error": f"Search provider returned HTTP error {response.status_code}.",
            }

        try:
            data = response.json()
            organic_results = data.get("organic_results", [])
            formatted = []

            for item in organic_results[:num_results]:
                formatted.append({
                    "title": item.get("title", "No title"),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                })

            return {
                "success": True,
                "query": clean_query,
                "results": formatted,
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "query": clean_query,
                "results": [],
                "error": f"Failed to parse search response: {exc}",
            }


class MockSearchProvider(BaseSearchProvider):
    """Offline deterministic search provider for testing and offline development."""

    def __init__(
        self,
        canned_results: Optional[Dict[str, List[Dict[str, str]]]] = None,
        simulated_error: Optional[str] = None,
    ) -> None:
        self.canned_results = canned_results or {}
        self.simulated_error = simulated_error

    def search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        if self.simulated_error:
            return {
                "success": False,
                "query": query,
                "results": [],
                "error": self.simulated_error,
            }

        clean = (query or "").strip()
        if not clean:
            return {
                "success": False,
                "query": "",
                "results": [],
                "error": "Search query cannot be empty.",
            }

        # Check canned matches
        for key, res in self.canned_results.items():
            if key.lower() in clean.lower():
                return {
                    "success": True,
                    "query": clean,
                    "results": res[:num_results],
                    "error": None,
                }

        # Generic realistic mock results
        return {
            "success": True,
            "query": clean,
            "results": [
                {
                    "title": f"Overview of {clean}",
                    "snippet": f"Comprehensive information and reference documentation regarding {clean}.",
                    "url": f"https://example.org/search?q={clean.replace(' ', '+')}",
                },
                {
                    "title": f"{clean} — Latest Updates and Insights",
                    "snippet": f"Recent developments, facts, and analysis related to {clean}.",
                    "url": f"https://example.org/news/{clean.replace(' ', '-')}",
                },
            ][:num_results],
            "error": None,
        }


def get_search_provider(provider_name: Optional[str] = None, **kwargs: Any) -> BaseSearchProvider:
    """Factory to instantiate the appropriate search provider."""
    name = (provider_name or os.getenv("SEARCH_PROVIDER", "")).strip().lower()
    if name == "mock":
        return MockSearchProvider(**kwargs)
    return SerpAPISearchProvider(**kwargs)


# ==============================================================================
# 2. Tool Implementation & Top-Level Helper
# ==============================================================================

def search_web(
    query: str,
    num_results: int = 3,
    api_key: Optional[str] = None,
    provider: Optional[BaseSearchProvider] = None,
) -> Dict[str, Any]:
    """Execute a web search for the query.

    Args:
        query: Search query string.
        num_results: Maximum results to return.
        api_key: Optional API key overriding environment configuration.
        provider: Optional custom BaseSearchProvider instance.

    Returns:
        Dict with 'success' (bool), 'query' (str), 'results' (list), and 'error' (str/None).
    """
    if provider:
        active_provider = provider
    elif api_key:
        active_provider = SerpAPISearchProvider(api_key=api_key)
    else:
        active_provider = get_search_provider()

    return active_provider.search(query=query, num_results=num_results)


class SearchTool(BaseTool):
    """AURA executable tool for searching the web."""

    name: str = "search"
    description: str = "Searches the web for factual information, current news, and technical references."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or keyword phrase.",
            },
            "num_results": {
                "type": "integer",
                "description": "Maximum number of search results to return (default: 3).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, provider: Optional[BaseSearchProvider] = None) -> None:
        self.provider = provider

    def execute(self, query: str = "", num_results: int = 3, **kwargs: Any) -> Dict[str, Any]:
        """Execute the search tool."""
        return search_web(query=query, num_results=num_results, provider=self.provider)
