"""Comprehensive Integration Tests for AURA Phase 8 — Full System Integration.

Covers all 10 required scenarios from the Phase 8 specification:
1. Text → LLM
2. Text → calculator
3. Text → RAG → LLM
4. Text → memory → LLM
5. Tool failure handling
6. RAG failure handling
7. LLM failure handling
8. Voice pipeline using mocks
9. Text mode
10. Exit behavior

All tests run deterministically with zero dependencies on real microphones, speakers,
or external paid network APIs.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from app.orchestrator import AURAOrchestrator, RouteType, OrchestratorResult
from app.brain.llm import LLMBrain, MockLLMProvider, LLMConnectionError
from app.brain.intent import IntentClassifier
from app.memory.conversation import ConversationMemory
from app.tools.registry import create_default_registry, ToolRegistry
from app.tools.calculator import CalculatorTool
from app.rag.retriever import LocalRetriever, LocalVectorStore
from app.rag.embeddings import LocalTFIDFEmbeddingModel
from app.speech.stt import MockSpeechToText, MicrophoneError
from app.speech.tts import MockTextToSpeech, AudioDeviceError


@pytest.fixture
def mock_retriever(tmp_path):
    """Fixture providing an isolated, populated LocalRetriever for tests."""
    storage_file = str(tmp_path / "test_store.json")
    model = LocalTFIDFEmbeddingModel(dimension=64)
    store = LocalVectorStore(storage_path=storage_file)
    retriever = LocalRetriever(embedding_model=model, vector_store=store, storage_path=storage_file)

    # Index sample document chunk
    retriever.add_documents([
        {
            "chunk_id": "test_ai_notes_1",
            "text": "Overfitting occurs when a machine learning model memorizes training noise rather than generalizable patterns.",
            "metadata": {"filename": "AI_Notes.txt", "page": 1},
        }
    ])
    return retriever


@pytest.fixture
def orchestrator(mock_retriever):
    """Fixture providing a fully initialized test AURAOrchestrator with mock providers."""
    mock_stt = MockSpeechToText(responses=["What is machine learning?"])
    mock_tts = MockTextToSpeech()
    registry = create_default_registry()
    memory = ConversationMemory(max_messages=10)
    classifier = IntentClassifier()
    brain = LLMBrain(
        provider=MockLLMProvider(),
        tool_registry=registry,
        retriever=mock_retriever,
        enable_tools=True,
        enable_rag=True,
    )

    return AURAOrchestrator(
        classifier=classifier,
        brain=brain,
        memory=memory,
        retriever=mock_retriever,
        tool_registry=registry,
        stt=mock_stt,
        tts=mock_tts,
        enable_rag=True,
    )


# ==============================================================================
# 1. Text → LLM
# ==============================================================================
def test_text_to_llm(orchestrator):
    """CASE 1: Verify direct concept query routes to LLM without invoking RAG or Tools."""
    result = orchestrator.process_query("Explain artificial intelligence.")
    assert result.route in {RouteType.LLM, RouteType.CONVERSATIONAL}
    assert result.sources == []  # Unnecessary RAG avoided
    assert "artificial intelligence" in result.response.lower()
    # Ensure turn was preserved in conversation memory
    assert len(orchestrator.memory) == 2
    assert orchestrator.memory.history[0]["content"] == "Explain artificial intelligence."


# ==============================================================================
# 2. Text → Calculator
# ==============================================================================
def test_text_to_calculator(orchestrator):
    """CASE 2: Verify arithmetic query triggers calculator tool, validates, executes, and synthesizes."""
    # Test with standard multiplication asterisk
    result = orchestrator.process_query("What is 125 * 32?")
    assert result.route == RouteType.CALCULATOR
    assert "4000" in result.response

    # Test with unicode multiplication cross '×'
    result2 = orchestrator.process_query("What is 25 × 40?")
    assert result2.route == RouteType.CALCULATOR
    assert "1000" in result2.response


# ==============================================================================
# 3. Text → RAG → LLM
# ==============================================================================
def test_text_to_rag_llm(orchestrator):
    """CASE 3: Verify document question triggers RAG retrieval, grounds answer, and provides source."""
    query = "What does my AI notes say about overfitting?"
    result = orchestrator.process_query(query)
    assert result.route == RouteType.RAG
    assert len(result.sources) > 0
    assert result.sources[0]["filename"] == "AI_Notes.txt"
    assert "overfitting" in result.response.lower()
    assert "AI_Notes.txt" in result.response


# ==============================================================================
# 4. Text → Memory → LLM
# ==============================================================================
def test_text_to_memory_llm(orchestrator):
    """CASE 4: Verify multi-turn conversational context retention across turns."""
    # Turn 1: State preference
    res1 = orchestrator.process_query("My favorite programming language is Python.")
    assert "python" in res1.response.lower()
    assert len(orchestrator.memory) == 2

    # Turn 2: Recall preference
    res2 = orchestrator.process_query("What is my favorite programming language?")
    assert "python" in res2.response.lower()
    assert len(orchestrator.memory) == 4


# ==============================================================================
# 5. Tool Failure Handling
# ==============================================================================
def test_tool_failure_handling(orchestrator):
    """CASE 5: Verify tool failure (e.g. division by zero or bad expression) does not crash the orchestrator."""
    result = orchestrator.process_query("What is 100 / 0?")
    assert result.route == RouteType.CALCULATOR
    # Should explain the issue politely rather than raising an uncaught ZeroDivisionError
    assert "issue" in result.response.lower() or "zero" in result.response.lower()
    # Memory should remain coherent
    assert len(orchestrator.memory) >= 2


# ==============================================================================
# 6. RAG Failure Handling
# ==============================================================================
def test_rag_failure_handling(orchestrator):
    """CASE 6: Verify RAG retrieval failure degrades gracefully to direct LLM response without crashing."""
    # Force the retriever to raise an exception
    orchestrator.retriever.retrieve = MagicMock(side_effect=RuntimeError("Corrupt index"))

    result = orchestrator.process_query("According to my notes, what is machine learning?")
    # Must return a valid response without raising uncaught exception
    assert result.response
    assert isinstance(result.response, str)


# ==============================================================================
# 7. LLM Failure Handling
# ==============================================================================
def test_llm_failure_handling(orchestrator):
    """CASE 7: Verify LLM provider connection/auth failure returns error message safely without crashing."""
    orchestrator.brain.provider = MockLLMProvider(
        simulated_error=LLMConnectionError("Remote API timed out after 30 seconds.")
    )

    result = orchestrator.process_query("Explain neural networks.")
    assert "problem" in result.response.lower() or "unable" in result.response.lower() or "service" in result.response.lower()


# ==============================================================================
# 8. Voice Pipeline Using Mocks
# ==============================================================================
def test_voice_pipeline_using_mocks(orchestrator):
    """CASE 8: Verify voice turn executes STT -> Brain -> TTS pipeline using mock audio hardware."""
    mock_stt = MockSpeechToText(responses=["What is machine learning?"])
    mock_tts = MockTextToSpeech()
    orchestrator.stt = mock_stt
    orchestrator.tts = mock_tts

    user_text, response, is_exit = orchestrator.run_voice_turn()
    assert user_text == "What is machine learning?"
    assert "machine learning" in response.lower()
    assert is_exit is False
    assert len(mock_tts.spoken_texts) == 1
    assert mock_tts.spoken_texts[0] == response


# ==============================================================================
# 9. Text Mode
# ==============================================================================
def test_text_mode(orchestrator):
    """CASE 9: Verify text mode execution turn functions without audio hardware."""
    reply = orchestrator.run_text_turn("Hello AURA")
    assert "aura" in reply.lower() or "hello" in reply.lower() or "help" in reply.lower()


# ==============================================================================
# 10. Exit Behavior
# ==============================================================================
def test_exit_behavior(orchestrator):
    """CASE 10: Verify explicit exit phrases and goodbye triggers terminate the session cleanly."""
    exit_queries = ["goodbye", "exit", "quit", "bye"]
    for word in exit_queries:
        res = orchestrator.process_query(word)
        assert res.is_exit is True
        assert res.route == RouteType.EXIT
        assert "goodbye" in res.response.lower()
