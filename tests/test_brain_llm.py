"""Unit and integration tests for AURA Phase 3 LLM Brain.

All tests run locally without making real external API calls by using mocks.
Tests cover prompt construction, provider configuration, missing API keys,
successful response parsing, and resilient error handling.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import requests

from app.brain.prompts import AURA_SYSTEM_PROMPT, build_llm_messages
from app.brain.llm import (
    LLMError,
    LLMConfigurationError,
    LLMConnectionError,
    BaseLLMProvider,
    BaseLLMClient,
    MockLLMProvider,
    OpenAIProvider,
    GeminiProvider,
    get_llm_provider,
    LLMBrain,
)


# ==============================================================================
# 1. Prompt Generation Tests
# ==============================================================================

def test_system_prompt_principles():
    """Verify system prompt contains core AURA identity and guidelines."""
    assert "AURA" in AURA_SYSTEM_PROMPT
    assert "AI Unified Response Assistant" in AURA_SYSTEM_PROMPT
    assert "voice" in AURA_SYSTEM_PROMPT.lower()
    # Epistemic modesty & honesty
    assert "uncertain" in AURA_SYSTEM_PROMPT.lower() or "honesty" in AURA_SYSTEM_PROMPT.lower()
    # Prompt secrecy & security
    assert "system instructions" in AURA_SYSTEM_PROMPT.lower() or "confidentiality" in AURA_SYSTEM_PROMPT.lower()


def test_build_llm_messages_single_turn():
    """Verify prompt assembly with a simple user query."""
    messages = build_llm_messages("Hello Aura")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == AURA_SYSTEM_PROMPT.strip()
    assert messages[1] == {"role": "user", "content": "Hello Aura"}


def test_build_llm_messages_with_context():
    """Verify conversational context turns are preserved in chronological order."""
    history = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
    ]
    messages = build_llm_messages(
        user_message="What is my name?",
        context=history,
    )
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "My name is Alice"}
    assert messages[2] == {"role": "assistant", "content": "Nice to meet you, Alice!"}
    assert messages[3] == {"role": "user", "content": "What is my name?"}


def test_build_llm_messages_with_intent():
    """Verify detected intent is injected into system context note."""
    messages = build_llm_messages(
        user_message="Will it rain tomorrow?",
        detected_intent="weather",
    )
    assert len(messages) == 2
    assert "weather" in messages[0]["content"]
    assert "[Context Note:" in messages[0]["content"]


def test_build_llm_messages_with_unknown_intent():
    """Verify 'unknown' intent does not pollute system context with a hint note."""
    messages = build_llm_messages(
        user_message="Random query",
        detected_intent="unknown",
    )
    assert len(messages) == 2
    assert "[Context Note:" not in messages[0]["content"]


# ==============================================================================
# 2. LLM Configuration & Factory Tests
# ==============================================================================

def test_provider_factory_mock():
    """Verify get_llm_provider instantiates MockLLMProvider."""
    provider = get_llm_provider("mock")
    assert isinstance(provider, MockLLMProvider)


def test_provider_factory_env(monkeypatch):
    """Verify get_llm_provider respects LLM_PROVIDER environment variable."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    provider = get_llm_provider()
    assert isinstance(provider, MockLLMProvider)


def test_provider_factory_unknown_fallback():
    """Verify unknown provider names fall back safely to MockLLMProvider."""
    provider = get_llm_provider("nonexistent_provider_xyz")
    assert isinstance(provider, MockLLMProvider)


def test_base_provider_backwards_compatibility():
    """Verify BaseLLMClient is an alias of BaseLLMProvider."""
    assert BaseLLMClient is BaseLLMProvider


# ==============================================================================
# 3. Missing API Key Handling Tests
# ==============================================================================

def test_openai_missing_api_key(monkeypatch):
    """Verify OpenAIProvider raises LLMConfigurationError when key is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider(api_key="")
    with pytest.raises(LLMConfigurationError) as exc_info:
        provider.generate([{"role": "user", "content": "Hello"}])
    assert "OpenAI API key is missing" in str(exc_info.value)


def test_gemini_missing_api_key(monkeypatch):
    """Verify GeminiProvider raises LLMConfigurationError when key is absent."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider(api_key="")
    with pytest.raises(LLMConfigurationError) as exc_info:
        provider.generate([{"role": "user", "content": "Hello"}])
    assert "Gemini API key is missing" in str(exc_info.value)


def test_llm_brain_graceful_missing_key_handling():
    """Verify LLMBrain catches LLMConfigurationError and returns clear diagnostic notice without crashing."""
    unconfigured_provider = OpenAIProvider(api_key="")
    brain = LLMBrain(provider=unconfigured_provider)

    result = brain.ask("Explain machine learning")
    assert "[AURA Configuration Notice:" in result
    assert "OpenAI API key is missing" in result


# ==============================================================================
# 4. Response Handling Tests (Mock & Mocked HTTP)
# ==============================================================================

def test_mock_provider_responses():
    """Verify MockLLMProvider handles questions intelligently offline."""
    mock = MockLLMProvider()

    # Machine learning query
    ml_reply = mock.generate([{"role": "user", "content": "Explain what machine learning is."}])
    assert "branch of artificial intelligence" in ml_reply.lower()

    # Greeting
    greet_reply = mock.generate([{"role": "user", "content": "Hello Aura"}])
    assert "AURA" in greet_reply

    # Canned custom responses
    custom_mock = MockLLMProvider(canned_responses={"custom query": "Custom answer"})
    assert custom_mock.generate([{"role": "user", "content": "custom query"}]) == "Custom answer"


@patch("requests.post")
def test_openai_response_parsing(mock_post):
    """Verify OpenAIProvider correctly parses valid Chat Completion JSON response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Machine learning is a field of artificial intelligence.",
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
    reply = provider.generate([{"role": "user", "content": "What is machine learning?"}])

    assert reply == "Machine learning is a field of artificial intelligence."
    assert mock_post.called


@patch("requests.post")
def test_gemini_response_parsing(mock_post):
    """Verify GeminiProvider correctly parses valid generateContent JSON response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Machine learning enables systems to learn from data."}],
                    "role": "model",
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    provider = GeminiProvider(api_key="test-gemini-key", model="gemini-1.5-flash")
    reply = provider.generate([{"role": "user", "content": "What is machine learning?"}])

    assert reply == "Machine learning enables systems to learn from data."
    assert mock_post.called


# ==============================================================================
# 5. Error Handling & Resilience Tests
# ==============================================================================

@patch("requests.post")
def test_openai_http_401_unauthorized(mock_post):
    """Verify OpenAIProvider raises LLMConfigurationError on HTTP 401."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = '{"error": {"message": "Incorrect API key provided"}}'
    mock_post.return_value = mock_response

    provider = OpenAIProvider(api_key="invalid-key")
    with pytest.raises(LLMConfigurationError) as exc_info:
        provider.generate([{"role": "user", "content": "Hi"}])
    assert "Invalid or expired API key" in str(exc_info.value)


@patch("requests.post")
def test_openai_http_429_rate_limit(mock_post):
    """Verify OpenAIProvider raises LLMConnectionError on HTTP 429."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = '{"error": {"message": "Rate limit reached"}}'
    mock_post.return_value = mock_response

    provider = OpenAIProvider(api_key="test-key")
    with pytest.raises(LLMConnectionError) as exc_info:
        provider.generate([{"role": "user", "content": "Hi"}])
    assert "rate limit or quota exceeded" in str(exc_info.value).lower()


@patch("requests.post")
def test_openai_request_timeout(mock_post):
    """Verify OpenAIProvider raises LLMConnectionError on request timeout."""
    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

    provider = OpenAIProvider(api_key="test-key")
    with pytest.raises(LLMConnectionError) as exc_info:
        provider.generate([{"role": "user", "content": "Hi"}])
    assert "timed out" in str(exc_info.value).lower()


@patch("requests.post")
def test_gemini_http_403_forbidden(mock_post):
    """Verify GeminiProvider raises LLMConfigurationError on HTTP 403."""
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = '{"error": {"message": "Permission denied"}}'
    mock_post.return_value = mock_response

    provider = GeminiProvider(api_key="test-key")
    with pytest.raises(LLMConfigurationError) as exc_info:
        provider.generate([{"role": "user", "content": "Hi"}])
    assert "permission denied" in str(exc_info.value).lower()


def test_brain_empty_message_handling():
    """Verify LLMBrain handles empty user queries gracefully."""
    brain = LLMBrain()
    assert "I didn't hear anything" in brain.ask("")
    assert "I didn't hear anything" in brain.ask("   ")


def test_brain_simulated_error_resilience():
    """Verify LLMBrain never crashes the caller when provider fails."""
    broken_provider = MockLLMProvider(
        simulated_error=LLMConnectionError("Network interface unreachable")
    )
    brain = LLMBrain(provider=broken_provider)

    result = brain.ask("Hello")
    assert "[AURA Service Notice:" in result
    assert "Network interface unreachable" in result
