"""Large Language Model (LLM) brain and provider abstraction for AURA.

Provides a pluggable, decoupled interface across different LLM backends
(Mock, OpenAI, Google Gemini) with tool calling orchestration, robust error handling,
and context assembly.
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import requests

from app.brain.prompts import AURA_SYSTEM_PROMPT, build_llm_messages
from app.tools.registry import ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Custom Exceptions
# ==============================================================================

class LLMError(Exception):
    """Base exception for all LLM-related errors in AURA."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when an LLM provider is misconfigured (e.g. missing API keys)."""
    pass


class LLMConnectionError(LLMError):
    """Raised when an external LLM API request fails, times out, or cannot be reached."""
    pass


# ==============================================================================
# 2. Tool Decision Parser
# ==============================================================================

def extract_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """Inspect text for a tool decision JSON structure.

    Args:
        response_text: Generated string from the LLM.

    Returns:
        Dictionary with 'tool' and 'parameters' if found, else None.
    """
    clean = response_text.strip()

    # 1. Attempt direct JSON parsing
    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "tool" in data:
            return data
    except Exception:
        pass

    # 2. Attempt extracting from markdown code block ```json { ... } ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "tool" in data:
                return data
        except Exception:
            pass

    # 3. Search for raw JSON pattern containing "tool":
    match = re.search(r"(\{\s*\"tool\"\s*:\s*\"[^\"]+\".*?\})", clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "tool" in data:
                return data
        except Exception:
            pass

    return None


# ==============================================================================
# 3. Abstract Provider Interface
# ==============================================================================

class BaseLLMProvider(ABC):
    """Abstract base class defining the contract for LLM backends."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a natural language text response given a list of chat messages.

        Args:
            messages: List of message dicts with 'role' ('system', 'user', 'assistant') and 'content'.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Optional limit on the number of generated tokens.

        Returns:
            The generated response string.

        Raises:
            LLMError: If generation fails.
        """
        pass

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Backward-compatible alias for generate()."""
        return self.generate(messages, temperature=temperature, max_tokens=max_tokens)


# Backward compatibility alias for Phase 1 code
BaseLLMClient = BaseLLMProvider


# ==============================================================================
# 4. Concrete Providers
# ==============================================================================

class MockLLMProvider(BaseLLMProvider):
    """Offline, deterministic LLM provider for testing and zero-setup local execution.

    Generates conversational responses and tool call decisions without requiring
    external API keys or network access.
    """

    def __init__(
        self,
        canned_responses: Optional[Dict[str, str]] = None,
        default_response: Optional[str] = None,
        simulated_error: Optional[Exception] = None,
    ) -> None:
        self.canned_responses = canned_responses or {}
        self.default_response = default_response
        self.simulated_error = simulated_error

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self.simulated_error:
            raise self.simulated_error

        # Separate system instructions, conversation history, and current message
        user_messages: List[str] = []
        assistant_messages: List[str] = []
        system_context = ""
        tool_results: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_context = content
            elif role == "user":
                if content.startswith("[Tool Result ("):
                    # Parse tool result turn
                    match = re.search(r"\[Tool Result \(([^)]+)\)\]:\s*(.*)", content, re.DOTALL)
                    if match:
                        t_name = match.group(1).lower()
                        try:
                            t_data = json.loads(match.group(2).strip())
                        except Exception:
                            t_data = {}
                        tool_results.append({"tool": t_name, "data": t_data})
                else:
                    user_messages.append(content)
            elif role == "assistant":
                assistant_messages.append(content)

        # ----------------------------------------------------------------------
        # 1. Synthesis Turn: Active ONLY if the latest message is a Tool Result
        # ----------------------------------------------------------------------
        is_synthesis_turn = bool(messages and str(messages[-1].get("content", "")).startswith("[Tool Result ("))

        if is_synthesis_turn and tool_results:
            latest = tool_results[-1]
            t_name = latest["tool"]
            t_data = latest["data"]

            if not t_data.get("success", False):
                err = t_data.get("error", "The tool operation encountered an error.")
                return f"I ran into an issue while using the {t_name} tool: {err}"

            if t_name == "calculator":
                val = t_data.get("result")
                # Look for the original math question in user messages
                orig = user_messages[-1] if user_messages else ""
                clean_orig = re.sub(r"^(?:what is|calculate|compute)\s+", "", orig, flags=re.IGNORECASE).strip("? .")
                if re.search(r"\band\s+explain\b", clean_orig, flags=re.IGNORECASE):
                    expr_part = re.sub(r"\s+and\s+explain.*$", "", clean_orig, flags=re.IGNORECASE).strip("? .")
                    return f"{expr_part} equals {val}. The calculation of {expr_part} evaluates to {val}."
                if clean_orig:
                    return f"{clean_orig} is {val}."
                return f"The calculation result is {val}."

            elif t_name == "weather":
                city = t_data.get("city", "the requested city")
                w_info = t_data.get("data", {})
                cond = w_info.get("condition", "clear")
                temp = w_info.get("temperature", 20)
                return f"The current weather in {city} is {cond} with a temperature of {temp}°C."

            elif t_name == "search":
                results = t_data.get("results", [])
                if results:
                    top = results[0]
                    return f"According to web search, {top.get('title')}: {top.get('snippet')}"
                return "I completed the search, but no relevant results were found."

            elif t_name == "files":
                f_data = t_data.get("data", {})
                if "content" in f_data:
                    return f"Contents of {f_data.get('path')}:\n{f_data.get('content')}"
                elif "items" in f_data:
                    names = [i["name"] for i in f_data.get("items", [])]
                    return f"Found {f_data.get('count')} files in data directory: {', '.join(names)}."
                elif "size_bytes" in f_data:
                    return f"File '{f_data.get('path')}' size is {f_data.get('size_bytes')} bytes."
                elif "bytes_written" in f_data:
                    return f"Successfully wrote {f_data.get('bytes_written')} bytes to {f_data.get('path')}."
                return "The requested file operation completed successfully."

        user_message = user_messages[-1] if user_messages else ""
        prior_user_messages = user_messages[:-1] if len(user_messages) > 1 else []
        clean_user = user_message.strip().lower()

        # Check explicit canned responses first
        for pattern, reply in self.canned_responses.items():
            if pattern.lower() in clean_user:
                return reply

        # ----------------------------------------------------------------------
        # 2. Tool Decision: Does the query require a tool?
        # ----------------------------------------------------------------------
        tools_enabled = "[Available Tools]" in system_context

        if tools_enabled:
            # Calculator trigger
            normalized_math = re.sub(r"(\d+)\s*[×x]\s*(\d+)", r"\1 * \2", clean_user)
            normalized_math = re.sub(r"(\d+)\s+times\s+(\d+)", r"\1 * \2", normalized_math)
            math_match = re.search(r"\b(?:what is|what's|calculate|compute)\s+([0-9\.\s\+\-\*\/\(\)\%\^]+|sqrt\([0-9\.]+\))\b", normalized_math)
            arith_match = re.search(r"(\d+\s*(?:[\+\-\*\/]|\*\*)\s*\d+)", normalized_math)
            sqrt_match = re.search(r"(sqrt\([0-9\.]+\))", normalized_math)

            if math_match:
                expr = math_match.group(1).strip()
                return json.dumps({"tool": "calculator", "parameters": {"expression": expr}})
            elif arith_match:
                expr = arith_match.group(1).strip()
                return json.dumps({"tool": "calculator", "parameters": {"expression": expr}})
            elif sqrt_match:
                expr = sqrt_match.group(1).strip()
                return json.dumps({"tool": "calculator", "parameters": {"expression": expr}})

            # Weather trigger
            weather_match = re.search(r"(?:weather\s+(?:in|for|at)|how(?:'s|\s+is)\s+the\s+weather\s+in)\s+([A-Za-z\s]+)", clean_user)
            if weather_match:
                city = weather_match.group(1).strip("?. ")
                return json.dumps({"tool": "weather", "parameters": {"city": city}})

            # Search trigger
            search_match = re.search(r"(?:search(?:\s+for|\s+web|\s+online)?|look\s+up|find\s+info\s+on)\s+(.+)", clean_user)
            if search_match and not any(f in clean_user for f in ("file", "files")):
                query = search_match.group(1).strip("?. ")
                return json.dumps({"tool": "search", "parameters": {"query": query}})

            # Files trigger
            if any(phrase in clean_user for phrase in ("list files", "show files", "view files", "dir data")):
                return json.dumps({"tool": "files", "parameters": {"action": "list_files", "path": ""}})

            read_match = re.search(r"read\s+file\s+([\w\.\/\-]+)", clean_user)
            if read_match:
                path = read_match.group(1).strip()
                return json.dumps({"tool": "files", "parameters": {"action": "read_file", "path": path}})

            meta_match = re.search(r"(?:file\s+metadata|metadata\s+for)\s+([\w\.\/\-]+)", clean_user)
            if meta_match:
                path = meta_match.group(1).strip()
                return json.dumps({"tool": "files", "parameters": {"action": "get_metadata", "path": path}})

        # ----------------------------------------------------------------------
        # 3. RAG Context Grounding Turn
        # ----------------------------------------------------------------------
        if "[Retrieved Context from Documents]" in system_context:
            excerpt_match = re.search(
                r"--- Excerpt 1 \[Source:\s*([^|\]]+).*?\] ---\s*(.*?)(?=\n---|\n\[Context|$)",
                system_context,
                re.DOTALL,
            )
            if excerpt_match:
                src_info = excerpt_match.group(1).strip()
                excerpt_text = excerpt_match.group(2).strip()

                filename = src_info.split()[0] if src_info else "document"
                page_part = ""
                p_match = re.search(r"page\s*(\d+)", src_info)
                c_match = re.search(r"chunk\s*([\w\-]+)", src_info)
                if p_match:
                    page_part = f"\n- page {p_match.group(1)}"
                elif c_match:
                    page_part = f"\n- chunk {c_match.group(1)}"

                # Extract first sentence as retrieved fact
                sentences = re.split(r"(?<=[.!?])\s+", excerpt_text)
                retrieved_fact = " ".join(sentences[:2]).strip() if sentences else excerpt_text

                return (
                    f"According to the reference document, {retrieved_fact}\n\n"
                    f"Based on my analysis, this retrieved context provides the direct foundation "
                    f"for addressing '{user_message.strip()}'.\n\n"
                    f"Source:\n- {filename}{page_part}"
                )
            else:
                return "The available documents do not provide enough information to answer this question."

        # Document query with no retrieved context or retrieval yield
        doc_query_indicators = (
            "according to my documents", "according to the documents", "according to documents",
            "according to my notes", "according to the notes", "according to notes",
            "in my documents", "in my notes", "from my documents", "from my notes",
            "document say", "documents say", "notes say", "my notes document",
        )
        if any(ind in clean_user for ind in doc_query_indicators):
            return "The available documents do not provide enough information to answer this question."

        # ----------------------------------------------------------------------
        # 3. Dynamic Context Awareness across Multi-Turn Conversation
        # ----------------------------------------------------------------------
        name_pattern = re.compile(r"\b(?:my name is|i am|call me|name's)\s+([A-Za-z]+)\b", re.IGNORECASE)

        detected_name: Optional[str] = None
        for past_msg in user_messages:
            match = name_pattern.search(past_msg)
            if match:
                detected_name = match.group(1).capitalize()

        # Query asking about user's name
        if any(q in clean_user for q in ("what is my name", "what's my name", "who am i", "do you know my name", "remember my name")):
            if detected_name:
                return f"Your name is {detected_name}."
            return "You haven't told me your name yet! What should I call you?"

        # User is introducing themselves in this turn
        intro_match = name_pattern.search(user_message)
        if intro_match and any(phrase in clean_user for phrase in ("my name is", "call me", "i am")):
            new_name = intro_match.group(1).capitalize()
            return f"Nice to meet you, {new_name}! How can I assist you today?"

        # Query asking what was previously said
        if any(q in clean_user for q in ("what did i just say", "what was my last question", "what was my previous message")):
            if prior_user_messages:
                return f"Earlier you said: '{prior_user_messages[-1]}'."
            return "This is the start of our conversation."

        # Query asking what the assistant previously said
        if any(q in clean_user for q in ("what did you just say", "what was your last response")):
            if assistant_messages:
                return f"Earlier I mentioned: '{assistant_messages[-1]}'."
            return "I haven't said anything yet."

        # User preferences / facts memory (e.g. "My favorite programming language is Python")
        fav_stmt_pattern = re.compile(r"\bmy favorite\s+([\w\s]+?)\s+is\s+([^.!?]+)", re.IGNORECASE)
        fav_query_pattern = re.compile(r"\b(?:what is|what's|do you know)\s+my favorite\s+([\w\s]+?)\??$", re.IGNORECASE)

        # Check if user is asking about a previously stated favorite
        fav_query_match = fav_query_pattern.search(clean_user)
        if fav_query_match:
            target_topic = fav_query_match.group(1).strip().lower()
            for past_msg in reversed(user_messages):
                fs_match = fav_stmt_pattern.search(past_msg)
                if fs_match:
                    topic = fs_match.group(1).strip().lower()
                    val = fs_match.group(2).strip()
                    if topic == target_topic or target_topic in topic or topic in target_topic:
                        return f"Your favorite {topic} is {val}."
            return f"You haven't mentioned your favorite {target_topic} yet."

        # User is stating a preference in the current turn
        fav_current = fav_stmt_pattern.search(user_message)
        if fav_current:
            topic = fav_current.group(1).strip()
            val = fav_current.group(2).strip()
            return f"I've noted that your favorite {topic} is {val}."

        # Domain knowledge simulations for typical voice assistant prompts
        if "machine learning" in clean_user:
            return (
                "Machine learning is a branch of artificial intelligence that enables computers "
                "to learn patterns directly from data and make predictions or decisions without "
                "being explicitly programmed."
            )
        elif "artificial intelligence" in clean_user or clean_user == "what is ai":
            return (
                "Artificial intelligence is the field of computer science dedicated to creating "
                "systems capable of performing tasks that typically require human intelligence, "
                "such as visual perception, decision-making, and natural language understanding."
            )
        elif any(g in clean_user for g in ("hello", "hi aura", "hey aura", "good morning", "good evening")):
            if detected_name:
                return f"Hello {detected_name}! I am AURA, your AI voice assistant. How can I assist you today?"
            return "Hello! I am AURA, your AI voice assistant. How can I assist you today?"
        elif any(b in clean_user for b in ("goodbye", "bye", "see you", "exit", "quit")):
            if detected_name:
                return f"Goodbye {detected_name}! Have a great day and feel free to reach out anytime."
            return "Goodbye! Have a great day and feel free to reach out anytime."
        elif "thank" in clean_user:
            return "You are very welcome! Let me know if there is anything else I can do for you."
        elif "calculate" in clean_user or any(op in clean_user for op in ("times", "plus", "divided", "minus")):
            return "I can perform arithmetic calculations for you quickly."
        elif re.search(r"\btime\b", clean_user):
            return "I can help with the time! Checking your current system clock."
        elif "weather" in clean_user:
            return "I can check the weather forecast for your area whenever you need."

        if self.default_response:
            return self.default_response

        return (
            f"I understand your query regarding '{user_message}'. "
            "As AURA, I am here to assist you with clear, accurate information."
        )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Chat Completions API provider (e.g. gpt-4o-mini, gpt-4o).

    Uses direct HTTPS REST requests via python-requests for lightweight execution.
    """

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        self.timeout = float(os.getenv("LLM_TIMEOUT", str(timeout)))

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.api_key:
            raise LLMConfigurationError(
                "OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            raise LLMConnectionError(
                f"Request to OpenAI API timed out after {self.timeout} seconds."
            )
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(f"Failed to connect to OpenAI API: {exc}")

        if response.status_code == 401:
            raise LLMConfigurationError(
                "OpenAI authentication failed: Invalid or expired API key (HTTP 401)."
            )
        elif response.status_code == 429:
            raise LLMConnectionError(
                "OpenAI rate limit or quota exceeded (HTTP 429). Check your billing and quota limits."
            )
        elif response.status_code >= 400:
            raise LLMConnectionError(
                f"OpenAI API returned error status HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected response format from OpenAI API: {exc}")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider (e.g. gemini-1.5-flash).

    Uses direct HTTPS REST requests via python-requests to Google's Generative Language endpoint.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model = model or os.getenv("LLM_MODEL", "gemini-1.5-flash").strip()
        self.timeout = float(os.getenv("LLM_TIMEOUT", str(timeout)))

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.api_key:
            raise LLMConfigurationError(
                "Gemini API key is missing. Please set GEMINI_API_KEY in your .env file."
            )

        endpoint = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        system_instruction_text = ""
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue

            if role == "system":
                system_instruction_text += (content + "\n")
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }

        if system_instruction_text.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction_text.strip()}]
            }

        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            raise LLMConnectionError(
                f"Request to Google Gemini API timed out after {self.timeout} seconds."
            )
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(f"Failed to connect to Google Gemini API: {exc}")

        if response.status_code == 400 and "API_KEY_INVALID" in response.text:
            raise LLMConfigurationError(
                "Gemini authentication failed: Invalid API key (HTTP 400)."
            )
        elif response.status_code == 403:
            raise LLMConfigurationError(
                "Gemini API permission denied (HTTP 403). Check API key permissions and project activation."
            )
        elif response.status_code == 429:
            raise LLMConnectionError(
                "Gemini rate limit exceeded (HTTP 429). Please slow down requests."
            )
        elif response.status_code >= 400:
            raise LLMConnectionError(
                f"Gemini API returned error status HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected response format from Gemini API: {exc}")


# ==============================================================================
# 5. Provider Factory
# ==============================================================================

def get_llm_provider(provider_name: Optional[str] = None, **kwargs: Any) -> BaseLLMProvider:
    """Instantiate and return the appropriate LLM provider.

    Args:
        provider_name: Optional name ('mock', 'openai', 'gemini').
        **kwargs: Additional parameters passed to the provider constructor.

    Returns:
        An instance conforming to BaseLLMProvider.
    """
    resolved_name = (provider_name or os.getenv("LLM_PROVIDER", "mock")).lower().strip()

    if resolved_name == "mock":
        return MockLLMProvider(**kwargs)
    elif resolved_name in ("openai", "chatgpt"):
        return OpenAIProvider(**kwargs)
    elif resolved_name in ("gemini", "google"):
        return GeminiProvider(**kwargs)
    else:
        logger.warning(
            f"Unrecognized LLM_PROVIDER '{resolved_name}'. Defaulting to 'mock' provider."
        )
        return MockLLMProvider(**kwargs)


# ==============================================================================
# 6. LLM Brain Orchestrator
# ==============================================================================

class LLMBrain:
    """Coordinates prompt construction, conversation context, tool execution, and the active LLM provider.

    Serves as the high-level cognitive engine of AURA:
    1. Evaluates user input against system prompts and available tool definitions.
    2. Identifies when external tools are needed and dispatches calls safely via ToolRegistry.
    3. Incorporates structured tool results back into context for final synthesis.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_context_messages: Optional[int] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tools: bool = True,
        retriever: Optional[Any] = None,
        enable_rag: bool = True,
        rag_top_k: int = 3,
        rag_min_score: float = 0.05,
    ) -> None:
        """Initialize the LLMBrain.

        Args:
            provider: Concrete BaseLLMProvider instance. Defaults to get_llm_provider().
            system_prompt: Core system prompt string. Defaults to AURA_SYSTEM_PROMPT.
            temperature: Sampling temperature. Defaults to LLM_TEMPERATURE env var or 0.7.
            max_tokens: Maximum token length. Defaults to LLM_MAX_TOKENS env var or None.
            max_context_messages: Limit on number of recent turns passed to LLM context.
            tool_registry: Central registry of executable tools. Defaults to create_default_registry().
            enable_tools: Whether tool calling is activated for this brain.
            retriever: Optional LocalRetriever instance for grounding answers in local documents.
            enable_rag: Whether RAG retrieval is enabled.
            rag_top_k: Number of relevant document chunks to retrieve per query.
            rag_min_score: Minimum similarity score threshold for retrieved chunks.
        """
        self.provider = provider or get_llm_provider()
        self.system_prompt = system_prompt or AURA_SYSTEM_PROMPT
        self.enable_tools = enable_tools
        self.tool_registry = tool_registry or (create_default_registry() if enable_tools else None)
        self.retriever = retriever
        self.enable_rag = enable_rag
        self.rag_top_k = rag_top_k
        self.rag_min_score = rag_min_score

        temp_env = os.getenv("LLM_TEMPERATURE", "0.7")
        try:
            self.temperature = temperature if temperature is not None else float(temp_env)
        except ValueError:
            self.temperature = 0.7

        tokens_env = os.getenv("LLM_MAX_TOKENS", "").strip()
        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif tokens_env.isdigit():
            self.max_tokens = int(tokens_env)
        else:
            self.max_tokens = None

        ctx_env = os.getenv("LLM_MAX_CONTEXT_MESSAGES", "").strip()
        if max_context_messages is not None:
            self.max_context_messages = max_context_messages
        elif ctx_env.isdigit():
            self.max_context_messages = int(ctx_env)
        else:
            self.max_context_messages = None

    def ask(
        self,
        user_message: str,
        context: Optional[Any] = None,
        detected_intent: Optional[str] = None,
        max_context_messages: Optional[int] = None,
        use_rag: Optional[bool] = None,
        rag_top_k: Optional[int] = None,
        rag_min_score: Optional[float] = None,
        rag_context: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[Dict[str, Any]] = None,
        observations: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Process a user query through the prompt template, tool orchestration loop, and LLM provider.

        Args:
            user_message: The raw user message text.
            context: Optional list of previous conversation turns, or a ConversationMemory instance.
            detected_intent: Optional intent tag identified by the classifier.
            max_context_messages: Optional limit overriding instance context limit.
            use_rag: Optional boolean to override default RAG retrieval behavior.
            rag_top_k: Optional integer overriding default top-k retrieval count.
            rag_min_score: Optional float overriding default minimum similarity threshold.
            rag_context: Optional pre-retrieved list of document chunk dictionaries.
            tool_results: Optional dictionary of executed tool outputs to include in synthesis.
            observations: Optional list of agent observations to provide context.

        Returns:
            The final natural language response string.
        """
        clean_input = (user_message or "").strip()
        if not clean_input:
            return "I didn't hear anything. How can I help you?"

        limit = max_context_messages or self.max_context_messages

        # Resolve context turns (supports List[Dict] or ConversationMemory instance)
        context_turns: Optional[List[Dict[str, Any]]] = None
        if context is not None:
            if hasattr(context, "get_context"):
                context_turns = context.get_context(max_messages=limit)
            elif isinstance(context, list):
                if limit and limit > 0 and len(context) > limit:
                    context_turns = context[-limit:]
                else:
                    context_turns = list(context)

        # Retrieve tool schemas if tools are enabled
        tool_schemas = self.tool_registry.get_schemas() if (self.tool_registry and self.enable_tools) else None

        # Resolve RAG context: use pre-retrieved chunks if provided, else retrieve if enabled
        rag_chunks: Optional[List[Dict[str, Any]]] = None
        if rag_context is not None:
            rag_chunks = rag_context
        else:
            should_use_rag = self.enable_rag if use_rag is None else use_rag
            if should_use_rag and self.retriever is not None and clean_input:
                k = rag_top_k or self.rag_top_k
                threshold = rag_min_score if rag_min_score is not None else self.rag_min_score
                try:
                    rag_chunks = self.retriever.retrieve(clean_input, top_k=k, min_score=threshold)
                except Exception as r_err:
                    logger.warning(f"RAG retrieval failed: {r_err}")
                    rag_chunks = None

        # If pre-computed tool results were provided (e.g. from AuraAgent execution):
        active_tool_results = {
            k: v for k, v in (tool_results or {}).items()
            if k != "retrieve"
        }
        if active_tool_results:
            synthesis_context = list(context_turns) if context_turns else []
            synthesis_context.append({"role": "user", "content": clean_input})
            for t_name, t_data in active_tool_results.items():
                synthesis_context.append({
                    "role": "tool_call",
                    "name": t_name,
                    "content": json.dumps({"tool": t_name, "parameters": {}}),
                })
                synthesis_context.append({
                    "role": "tool",
                    "name": t_name,
                    "content": json.dumps(t_data) if isinstance(t_data, dict) else str(t_data),
                })

            synthesis_messages = build_llm_messages(
                user_message="",
                context=synthesis_context,
                detected_intent=detected_intent,
                system_prompt=self.system_prompt,
                tool_schemas=tool_schemas,
                rag_chunks=rag_chunks,
            )
            try:
                return self.provider.generate(
                    messages=synthesis_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ).strip()
            except LLMConfigurationError as err:
                logger.error(f"LLM Configuration Error: {err}")
                return f"[AURA Configuration Notice: {err}]"
            except LLMError as err:
                logger.error(f"LLM Error during synthesis: {err}")
                return f"[AURA Offline Notice: {err}]"

        # Construct initial message sequence
        messages = build_llm_messages(
            user_message=clean_input,
            context=context_turns,
            detected_intent=detected_intent,
            system_prompt=self.system_prompt,
            tool_schemas=tool_schemas,
            rag_chunks=rag_chunks,
        )

        try:
            # 1. Primary LLM Generation (produces either direct response or tool decision)
            initial_response = self.provider.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ).strip()

            # 2. Check if a tool decision was emitted
            tool_decision = extract_tool_call(initial_response)

            if tool_decision and self.tool_registry and self.enable_tools:
                tool_name = tool_decision.get("tool", "").strip().lower()
                parameters = tool_decision.get("parameters", {})
                logger.info(f"[TOOL] {tool_name} selected")

                # Record tool call in conversation memory if provided
                if hasattr(context, "add_tool_call"):
                    context.add_tool_call(tool_name, parameters)

                # Execute the tool safely via registry
                tool_result = self.tool_registry.execute(tool_name, **parameters)
                logger.info(f"[TOOL] {tool_name} completed")

                # Record tool result in conversation memory if provided
                if hasattr(context, "add_tool_result"):
                    context.add_tool_result(tool_name, tool_result)

                # Assemble synthesis context with the tool interaction
                if hasattr(context, "get_context"):
                    synthesis_context = context.get_context(max_messages=limit)
                elif isinstance(context, list):
                    synthesis_context = list(context) + [
                        {"role": "tool_call", "name": tool_name, "content": json.dumps(tool_decision)},
                        {"role": "tool", "name": tool_name, "content": json.dumps(tool_result)},
                    ]
                else:
                    synthesis_context = [
                        {"role": "user", "content": clean_input},
                        {"role": "tool_call", "name": tool_name, "content": json.dumps(tool_decision)},
                        {"role": "tool", "name": tool_name, "content": json.dumps(tool_result)},
                    ]

                # Re-query the LLM with tool results to synthesize natural language response
                logger.info("[LLM] Generating final response")
                synthesis_messages = build_llm_messages(
                    user_message="",
                    context=synthesis_context,
                    detected_intent=detected_intent,
                    system_prompt=self.system_prompt,
                    tool_schemas=tool_schemas,
                    rag_chunks=rag_chunks,
                )

                final_response = self.provider.generate(
                    messages=synthesis_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ).strip()

                return final_response

            # Direct natural response (no tool needed)
            logger.info("[LLM] Generating final response")
            return initial_response

        except LLMConfigurationError as err:
            logger.error(f"LLM Configuration Error: {err}")
            return f"[AURA Configuration Notice: {err}]"

        except LLMConnectionError as err:
            logger.error(f"LLM Connection Error: {err}")
            return f"[AURA Service Notice: Unable to reach language model. {err}]"

        except Exception as err:
            logger.exception(f"Unexpected error during LLM generation: {err}")
            return f"[AURA Notice: An unexpected error occurred while communicating with the brain: {err}]"

    def ask_with_rag(
        self,
        user_message: str,
        context: Optional[Any] = None,
        detected_intent: Optional[str] = None,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Query the LLM Brain with explicit RAG retrieval and return the answer along with source metadata.

        Args:
            user_message: The raw user query text.
            context: Optional conversation context or memory buffer.
            detected_intent: Optional detected query intent.
            top_k: Maximum number of relevant chunks to retrieve.
            min_score: Minimum similarity score threshold.

        Returns:
            Dict containing 'answer', 'chunks', and 'sources'.
        """
        clean_input = (user_message or "").strip()
        rag_chunks: List[Dict[str, Any]] = []

        if self.retriever is not None and clean_input:
            try:
                rag_chunks = self.retriever.retrieve(clean_input, top_k=top_k, min_score=min_score)
            except Exception as exc:
                logger.warning(f"Error during RAG retrieval: {exc}")
                rag_chunks = []

        answer = self.ask(
            user_message=clean_input,
            context=context,
            detected_intent=detected_intent,
            use_rag=True,
            rag_top_k=top_k,
            rag_min_score=min_score,
        )

        sources = []
        for c in rag_chunks:
            meta = c.get("metadata", {})
            sources.append({
                "filename": meta.get("filename", "unknown"),
                "page": meta.get("page"),
                "chunk_id": c.get("id") or meta.get("chunk_id"),
                "score": c.get("score"),
            })

        return {
            "answer": answer,
            "chunks": rag_chunks,
            "sources": sources,
        }

