"""Unit and integration tests for AURA Phase 5 — Tools.

Covers:
- Safe Calculator tool (valid, invalid, and unsafe expressions)
- Weather tool & providers (configuration handling, errors, mock provider)
- Search tool & providers (valid search, provider failures, missing config)
- Safe Files tool (sandboxed read, write, metadata, path traversal security)
- Tool Registry (registration, lookup, unknown tool, schema export)
- LLM Brain Tool Calling & Memory Preservation (multi-turn tool loop)
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.tools.base import BaseTool
from app.tools.calculator import calculate, CalculatorTool
from app.tools.weather import (
    get_weather,
    WeatherTool,
    OpenWeatherProvider,
    MockWeatherProvider,
)
from app.tools.search import (
    search_web,
    SearchTool,
    SerpAPISearchProvider,
    MockSearchProvider,
)
from app.tools.files import SafeFileManager, FileTool
from app.tools.registry import ToolRegistry, create_default_registry
from app.brain.llm import LLMBrain, MockLLMProvider
from app.memory.conversation import ConversationMemory


# ==============================================================================
# 1. Calculator Tool Tests
# ==============================================================================

class TestCalculatorTool:
    """Tests for safe mathematical evaluation."""

    def test_valid_arithmetic_expressions(self):
        """Verify standard arithmetic operations evaluate accurately."""
        res_mul = calculate("25 * 48")
        assert res_mul["success"] is True
        assert res_mul["result"] == 1200
        assert res_mul["error"] is None

        res_div = calculate("100 / 4")
        assert res_div["success"] is True
        assert res_div["result"] == 25

        res_sub = calculate("50 - 18.5")
        assert res_sub["success"] is True
        assert res_sub["result"] == 31.5

        res_pow = calculate("2 ** 8")
        assert res_pow["success"] is True
        assert res_pow["result"] == 256

        res_mod = calculate("17 % 5")
        assert res_mod["success"] is True
        assert res_mod["result"] == 2

    def test_math_functions_and_constants(self):
        """Verify whitelisted mathematical functions and constants."""
        res_sqrt = calculate("sqrt(144)")
        assert res_sqrt["success"] is True
        assert res_sqrt["result"] == 12

        res_sin = calculate("sin(0)")
        assert res_sin["success"] is True
        assert res_sin["result"] == 0

        res_abs = calculate("abs(-42)")
        assert res_abs["success"] is True
        assert res_abs["result"] == 42

        res_round = calculate("round(3.14159, 2)")
        assert res_round["success"] is True
        assert res_round["result"] == 3.14

        res_const = calculate("pi * 2")
        assert res_const["success"] is True
        assert abs(res_const["result"] - 6.283185) < 1e-4

    def test_invalid_syntax_and_division_by_zero(self):
        """Verify invalid expressions and division by zero return structured errors."""
        res_zero = calculate("100 / 0")
        assert res_zero["success"] is False
        assert res_zero["result"] is None
        assert "Division by zero" in res_zero["error"]

        res_syntax = calculate("25 + * 3")
        assert res_syntax["success"] is False
        assert res_syntax["result"] is None
        assert "Invalid mathematical syntax" in res_syntax["error"]

        res_empty = calculate("")
        assert res_empty["success"] is False
        assert "empty" in res_empty["error"].lower()

    def test_unsafe_expressions_rejected(self):
        """Verify dangerous and non-arithmetic expressions are blocked."""
        # Unrestricted eval / os calls
        res_import = calculate("__import__('os').system('echo pwned')")
        assert res_import["success"] is False
        assert res_import["result"] is None

        # Arbitrary builtins
        res_open = calculate("open('secret.txt')")
        assert res_open["success"] is False
        assert res_open["result"] is None

        # Class attribute access
        res_attr = calculate("().__class__.__base__")
        assert res_attr["success"] is False
        assert res_attr["result"] is None

        # Unapproved functions
        res_unapproved = calculate("eval('2 + 2')")
        assert res_unapproved["success"] is False
        assert res_unapproved["result"] is None

    def test_calculator_tool_wrapper(self):
        """Verify CalculatorTool BaseTool class execution."""
        tool = CalculatorTool()
        assert tool.name == "calculator"
        schema = tool.to_schema()
        assert "expression" in schema["parameters"]["properties"]

        res = tool.run(expression="10 + 20")
        assert res["success"] is True
        assert res["result"] == 30


# ==============================================================================
# 2. Weather Tool Tests
# ==============================================================================

class TestWeatherTool:
    """Tests for weather tool and provider interfaces."""

    def test_missing_api_key_configuration_handling(self, monkeypatch):
        """Verify missing weather API key produces a clear configuration error."""
        monkeypatch.delenv("WEATHER_API_KEY", raising=False)
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)

        provider = OpenWeatherProvider(api_key="")
        res = provider.get_weather("London")
        assert res["success"] is False
        assert res["data"] is None
        assert "Weather API key is not configured" in res["error"]

    def test_mock_weather_provider(self):
        """Verify MockWeatherProvider returns structured data for valid city."""
        mock = MockWeatherProvider()
        res = mock.get_weather("London")
        assert res["success"] is True
        assert res["city"] == "London"
        assert res["data"]["temperature"] == 16.5
        assert res["data"]["condition"] == "Light Rain"
        assert res["error"] is None

    def test_mock_weather_empty_city(self):
        """Verify empty city name returns structured error."""
        mock = MockWeatherProvider()
        res = mock.get_weather("")
        assert res["success"] is False
        assert "City name cannot be empty" in res["error"]

    def test_weather_provider_error_handling(self):
        """Verify simulated provider error is handled cleanly."""
        mock = MockWeatherProvider(simulated_error="Network timeout connecting to station.")
        res = mock.get_weather("Paris")
        assert res["success"] is False
        assert res["data"] is None
        assert "Network timeout" in res["error"]

    @patch("requests.get")
    def test_openweather_api_http_error(self, mock_get):
        """Verify OpenWeatherProvider handles HTTP errors cleanly."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        provider = OpenWeatherProvider(api_key="fake_key")
        res = provider.get_weather("NonexistentAtlantisCity")
        assert res["success"] is False
        assert "was not found" in res["error"]

    def test_weather_tool_wrapper(self):
        """Verify WeatherTool class wrapper works with a provider."""
        mock = MockWeatherProvider()
        tool = WeatherTool(provider=mock)
        assert tool.name == "weather"
        res = tool.run(city="Tokyo")
        assert res["success"] is True
        assert res["city"] == "Tokyo"
        assert res["data"]["condition"] == "Clear Sky"


# ==============================================================================
# 3. Search Tool Tests
# ==============================================================================

class TestSearchTool:
    """Tests for search tool and provider interfaces."""

    def test_missing_serpapi_key_configuration(self, monkeypatch):
        """Verify missing SerpAPI key produces a clear configuration error."""
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        provider = SerpAPISearchProvider(api_key="")
        res = provider.search("python programming")
        assert res["success"] is False
        assert res["results"] == []
        assert "SerpAPI key is not configured" in res["error"]

    def test_mock_search_provider_valid_query(self):
        """Verify MockSearchProvider returns structured results."""
        mock = MockSearchProvider()
        res = mock.search("quantum computing", num_results=2)
        assert res["success"] is True
        assert res["query"] == "quantum computing"
        assert len(res["results"]) == 2
        assert "title" in res["results"][0]
        assert "snippet" in res["results"][0]
        assert "url" in res["results"][0]
        assert res["error"] is None

    def test_search_provider_failure(self):
        """Verify provider failure returns structured error."""
        mock = MockSearchProvider(simulated_error="SerpAPI upstream connection failed.")
        res = mock.search("python")
        assert res["success"] is False
        assert res["results"] == []
        assert "upstream connection failed" in res["error"]

    def test_search_empty_query(self):
        """Verify empty query string returns structured error."""
        mock = MockSearchProvider()
        res = mock.search("")
        assert res["success"] is False
        assert "cannot be empty" in res["error"]

    def test_search_tool_wrapper(self):
        """Verify SearchTool class wrapper execution."""
        mock = MockSearchProvider()
        tool = SearchTool(provider=mock)
        assert tool.name == "search"
        res = tool.run(query="artificial intelligence", num_results=1)
        assert res["success"] is True
        assert len(res["results"]) == 1


# ==============================================================================
# 4. Safe Files Tool Tests & Security Sandbox
# ==============================================================================

class TestSafeFileManager:
    """Tests for sandboxed filesystem manager and path traversal defense."""

    @pytest.fixture
    def sandbox(self, tmp_path):
        """Create a temporary sandbox directory for test isolation."""
        sandbox_dir = tmp_path / "safe_sandbox"
        sandbox_dir.mkdir()
        # Seed test file
        test_file = sandbox_dir / "sample.txt"
        test_file.write_text("Hello from inside the safe sandbox!", encoding="utf-8")
        return SafeFileManager(base_dir=sandbox_dir)

    def test_valid_file_access(self, sandbox):
        """Verify reading valid files and listing contents inside the sandbox."""
        list_res = sandbox.list_files()
        assert list_res["success"] is True
        assert list_res["data"]["count"] >= 1
        names = [item["name"] for item in list_res["data"]["items"]]
        assert "sample.txt" in names

        read_res = sandbox.read_file("sample.txt")
        assert read_res["success"] is True
        assert "Hello from inside" in read_res["data"]["content"]

    def test_safe_write_and_metadata(self, sandbox):
        """Verify writing files and querying metadata within sandbox."""
        write_res = sandbox.write_file("notes.txt", "Important reminder.")
        assert write_res["success"] is True

        meta_res = sandbox.get_metadata("notes.txt")
        assert meta_res["success"] is True
        assert meta_res["data"]["is_file"] is True
        assert meta_res["data"]["size_bytes"] > 0

    def test_nonexistent_file(self, sandbox):
        """Verify reading a nonexistent file returns structured error without crashing."""
        res = sandbox.read_file("missing_file.txt")
        assert res["success"] is False
        assert "File does not exist" in res["error"]

    def test_path_traversal_attacks_blocked(self, sandbox):
        """Verify attempts to traverse outside sandbox are blocked."""
        traversal_attempts = [
            "../../secret.txt",
            "../outside.txt",
            "subdir/../../outside.txt",
            "..\\..\\windows\\system32",
        ]

        for attempt in traversal_attempts:
            # Direct resolver should raise PermissionError
            with pytest.raises(PermissionError):
                sandbox.resolve_safe_path(attempt)

            # Public methods should return structured error safely
            read_res = sandbox.read_file(attempt)
            assert read_res["success"] is False
            assert "Security restriction" in read_res["error"]

    def test_absolute_path_outside_allowed_directory(self, sandbox):
        """Verify absolute paths pointing outside sandbox are blocked."""
        external_abs = str(Path(sandbox.base_dir).parent / "outside_root.txt")
        with pytest.raises(PermissionError):
            sandbox.resolve_safe_path(external_abs)

        res = sandbox.read_file(external_abs)
        assert res["success"] is False
        assert "Security restriction" in res["error"]

    def test_file_tool_wrapper(self, sandbox):
        """Verify FileTool dispatches actions correctly."""
        tool = FileTool(manager=sandbox)
        assert tool.name == "files"

        list_res = tool.run(action="list_files")
        assert list_res["success"] is True

        read_res = tool.run(action="read_file", path="sample.txt")
        assert read_res["success"] is True
        assert "sandbox" in read_res["data"]["content"]


# ==============================================================================
# 5. Tool Registry Tests
# ==============================================================================

class TestToolRegistry:
    """Tests for central ToolRegistry discovery and lookup."""

    def test_registration_and_lookup(self):
        """Verify tools can be registered and retrieved by name."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)

        assert "calculator" in registry
        retrieved = registry.get("calculator")
        assert retrieved is calc
        assert registry.get("CALCULATOR") is calc  # case-insensitive

    def test_list_tools_and_schemas(self):
        """Verify listing registered tools and exporting schemas."""
        registry = create_default_registry()
        tools = registry.list_tools()
        assert "calculator" in tools
        assert "weather" in tools
        assert "search" in tools
        assert "files" in tools

        schemas = registry.get_schemas()
        assert len(schemas) == 4
        schema_names = [s["name"] for s in schemas]
        assert "calculator" in schema_names
        assert "weather" in schema_names

    def test_unknown_tool_handling(self):
        """Verify executing an unknown tool returns a structured error."""
        registry = create_default_registry()
        res = registry.execute("nonexistent_tool_abc", param="value")
        assert res["success"] is False
        assert "not found" in res["error"].lower()

    def test_invalid_tool_registration(self):
        """Verify TypeError when registering non-BaseTool object."""
        registry = ToolRegistry()
        with pytest.raises(TypeError):
            registry.register("not_a_tool")  # type: ignore


# ==============================================================================
# 6. LLM Tool Calling & Memory Integration Tests
# ==============================================================================

class TestLLMToolCallingAndMemory:
    """Integration tests for LLM Tool Decision -> Tool Execution -> Memory -> Final Response."""

    def test_end_to_end_calculator_tool_calling(self):
        """Verify complete pipeline: User asks math -> LLM calls calculator -> Tool returns 1200 -> LLM responds."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        brain = LLMBrain(
            provider=MockLLMProvider(),
            tool_registry=registry,
        )
        memory = ConversationMemory(max_messages=10)

        # 1. User asks calculation
        query = "What is 25 * 48?"
        memory.add_user_message(query)

        # 2. Brain orchestrates tool call and synthesizes response
        response = brain.ask(user_message=query, context=memory)

        # 3. Verify final answer reflects tool output
        assert "1200" in response

        # 4. Verify conversation history preserved complete turn sequence
        history = memory.get_history()
        roles = [turn["role"] for turn in history]
        # Expect: user -> tool_call -> tool
        assert "user" in roles
        assert "tool_call" in roles
        assert "tool" in roles

        # Inspect tool call details
        tool_call_turn = next(t for t in history if t["role"] == "tool_call")
        assert tool_call_turn["name"] == "calculator"
        assert "25 * 48" in tool_call_turn["parameters"]["expression"]

        # Inspect tool result details
        tool_result_turn = next(t for t in history if t["role"] == "tool")
        assert tool_result_turn["name"] == "calculator"
        assert tool_result_turn["result"]["result"] == 1200

    def test_conversational_turn_without_tool(self):
        """Verify ordinary conversational query does not invoke a tool."""
        registry = create_default_registry()
        brain = LLMBrain(provider=MockLLMProvider(), tool_registry=registry)
        memory = ConversationMemory()

        query = "Hello Aura, how are you?"
        memory.add_user_message(query)

        response = brain.ask(user_message=query, context=memory)
        assert "AURA" in response

        # History should NOT contain any tool_call or tool messages
        history = memory.get_history()
        roles = [t["role"] for t in history]
        assert "tool_call" not in roles
        assert "tool" not in roles

    def test_weather_tool_calling_flow(self):
        """Verify weather query triggers weather tool and incorporates condition in response."""
        registry = ToolRegistry()
        mock_weather = MockWeatherProvider()
        registry.register(WeatherTool(provider=mock_weather))

        brain = LLMBrain(provider=MockLLMProvider(), tool_registry=registry)
        memory = ConversationMemory()

        query = "What is the weather in London?"
        memory.add_user_message(query)

        response = brain.ask(user_message=query, context=memory)
        assert "London" in response
        assert "16.5" in response or "Rain" in response

    def test_sandboxed_file_tool_calling_flow(self, tmp_path):
        """Verify file operation tool calling flow through LLM Brain."""
        test_dir = tmp_path / "aura_data"
        test_dir.mkdir()
        (test_dir / "notes.txt").write_text("Secret recipe is chocolate.", encoding="utf-8")

        file_mgr = SafeFileManager(base_dir=test_dir)
        registry = ToolRegistry()
        registry.register(FileTool(manager=file_mgr))

        brain = LLMBrain(provider=MockLLMProvider(), tool_registry=registry)
        memory = ConversationMemory()

        query = "read file notes.txt"
        memory.add_user_message(query)

        response = brain.ask(user_message=query, context=memory)
        assert "notes.txt" in response
        assert "chocolate" in response
