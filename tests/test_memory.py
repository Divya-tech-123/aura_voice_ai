"""Unit and integration tests for AURA Phase 4 Memory.

Tests cover:
- Adding user and assistant messages
- Preserving message roles and chronological ordering
- Retrieving history and context generation
- Conversation trimming with sliding window limits
- Empty memory behavior and clearing memory
- Context limits and multi-turn LLM context integration (offline)
- Long-term Vector Memory interface and local similarity retrieval
"""

import pytest
from app.memory.conversation import ConversationMemory
from app.memory.vector_memory import VectorMemory, BaseVectorMemory
from app.brain.llm import LLMBrain, MockLLMProvider
from app.brain.prompts import build_llm_messages


# ==============================================================================
# 1. Short-Term Conversation Memory Tests
# ==============================================================================

def test_add_user_message():
    """Verify user message is stored with role 'user' and text content."""
    memory = ConversationMemory()
    assert memory.is_empty()
    assert len(memory) == 0

    memory.add_user_message("My name is Rahul")
    assert len(memory) == 1
    assert not memory.is_empty()

    history = memory.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "My name is Rahul"


def test_add_assistant_message():
    """Verify assistant message is stored with role 'assistant' and text content."""
    memory = ConversationMemory()
    memory.add_assistant_message("Nice to meet you, Rahul!")

    history = memory.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "assistant"
    assert history[0]["content"] == "Nice to meet you, Rahul!"


def test_retrieving_history():
    """Verify retrieving history returns a list of turns and supports limits."""
    memory = ConversationMemory(max_messages=10)
    memory.add_user_message("Turn 1")
    memory.add_assistant_message("Reply 1")
    memory.add_user_message("Turn 2")
    memory.add_assistant_message("Reply 2")

    # Full history
    full_history = memory.get_history()
    assert len(full_history) == 4
    assert [m["content"] for m in full_history] == ["Turn 1", "Reply 1", "Turn 2", "Reply 2"]

    # Backward-compatible alias get_messages()
    alias_history = memory.get_messages()
    assert alias_history == full_history

    # History with limit parameter
    recent_two = memory.get_history(limit=2)
    assert len(recent_two) == 2
    assert recent_two[0]["content"] == "Turn 2"
    assert recent_two[1]["content"] == "Reply 2"


def test_message_ordering():
    """Verify messages remain in strict chronological order."""
    memory = ConversationMemory(max_messages=10)
    sequence = [
        ("user", "What is Python?"),
        ("assistant", "Python is a programming language."),
        ("user", "Who created it?"),
        ("assistant", "Guido van Rossum created Python."),
        ("user", "In what year?"),
    ]

    for role, content in sequence:
        if role == "user":
            memory.add_user_message(content)
        else:
            memory.add_assistant_message(content)

    history = memory.get_history()
    assert len(history) == 5

    for i, (expected_role, expected_content) in enumerate(sequence):
        assert history[i]["role"] == expected_role
        assert history[i]["content"] == expected_content


def test_conversation_trimming():
    """Verify memory trims oldest turns when exceeding max_messages using sliding window."""
    max_messages = 4
    memory = ConversationMemory(max_messages=max_messages)

    # Insert 6 messages
    for i in range(6):
        if i % 2 == 0:
            memory.add_user_message(f"User message {i}")
        else:
            memory.add_assistant_message(f"Assistant response {i}")

    # Should retain exactly max_messages = 4 (messages 2, 3, 4, 5)
    history = memory.get_history()
    assert len(history) == 4
    assert history[0]["content"] == "User message 2"
    assert history[1]["content"] == "Assistant response 3"
    assert history[2]["content"] == "User message 4"
    assert history[3]["content"] == "Assistant response 5"


def test_conversation_trimming_via_max_turns():
    """Verify backward compatibility with max_turns parameter."""
    memory = ConversationMemory(max_turns=2)  # max_turns=2 -> max_messages=4

    for i in range(6):
        memory.add_user_message(f"Msg {i}")

    assert len(memory.get_history()) == 4
    assert memory.get_history()[0]["content"] == "Msg 2"


def test_clearing_memory():
    """Verify clear() and clear_history() completely reset the memory buffer."""
    memory = ConversationMemory()
    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi")
    assert len(memory) == 2

    memory.clear()
    assert len(memory) == 0
    assert memory.is_empty()
    assert memory.get_history() == []

    # Test clear_history alias
    memory.add_user_message("Testing again")
    assert len(memory) == 1
    memory.clear_history()
    assert len(memory) == 0


def test_context_generation():
    """Verify get_context and format_context_text generate proper context outputs."""
    memory = ConversationMemory(max_messages=10)
    memory.add_user_message("My name is Rahul")
    memory.add_assistant_message("Nice to meet you, Rahul!")

    # Context turns
    context = memory.get_context()
    assert len(context) == 2
    assert context[0]["role"] == "user"
    assert context[1]["role"] == "assistant"

    # Context text formatting
    context_text = memory.format_context_text()
    assert "User: My name is Rahul" in context_text
    assert "Assistant: Nice to meet you, Rahul!" in context_text


def test_empty_memory():
    """Verify behavior of completely empty memory."""
    memory = ConversationMemory()
    assert memory.is_empty() is True
    assert len(memory) == 0
    assert memory.get_history() == []
    assert memory.get_messages() == []
    assert memory.get_context() == []
    assert memory.format_context_text() == ""


def test_maximum_history_size():
    """Verify configurable maximum history limits are enforced."""
    small_memory = ConversationMemory(max_messages=2)
    small_memory.add_user_message("One")
    small_memory.add_assistant_message("Two")
    small_memory.add_user_message("Three")

    assert len(small_memory.get_history()) == 2
    assert small_memory.get_history()[0]["content"] == "Two"
    assert small_memory.get_history()[1]["content"] == "Three"


# ==============================================================================
# 2. LLM Context Integration Tests (Multi-turn, dynamic without hardcoding)
# ==============================================================================

def test_llm_multi_turn_context_integration():
    """Verify that conversation history enables the LLM to recall context across turns."""
    provider = MockLLMProvider()
    brain = LLMBrain(provider=provider)
    memory = ConversationMemory(max_messages=10)

    # Turn 1: User introduces themselves
    turn1_input = "My name is Rahul"
    memory.add_user_message(turn1_input)
    turn1_reply = brain.ask(turn1_input, context=memory)
    memory.add_assistant_message(turn1_reply)
    assert "Rahul" in turn1_reply

    # Turn 2: User asks for their name based on conversation history
    turn2_input = "What is my name?"
    memory.add_user_message(turn2_input)
    turn2_reply = brain.ask(turn2_input, context=memory)
    memory.add_assistant_message(turn2_reply)

    # Verify that the LLM answered using conversation history context
    assert "Rahul" in turn2_reply
    assert "Your name is Rahul" in turn2_reply


def test_llm_context_is_not_hardcoded():
    """Verify that the context mechanism is dynamic and works for any user name."""
    provider = MockLLMProvider()
    brain = LLMBrain(provider=provider)
    memory = ConversationMemory(max_messages=10)

    # User introduces as Priya
    memory.add_user_message("Hello, my name is Priya.")
    reply1 = brain.ask("Hello, my name is Priya.", context=memory)
    memory.add_assistant_message(reply1)

    # Ask for name
    memory.add_user_message("What is my name?")
    reply2 = brain.ask("What is my name?", context=memory)
    memory.add_assistant_message(reply2)

    assert "Priya" in reply2
    assert "Rahul" not in reply2


def test_llm_recall_previous_statement():
    """Verify recalling previous user statements from conversation history."""
    provider = MockLLMProvider()
    brain = LLMBrain(provider=provider)
    memory = ConversationMemory(max_messages=10)

    memory.add_user_message("The secret word is pineapple")
    reply1 = brain.ask("The secret word is pineapple", context=memory)
    memory.add_assistant_message(reply1)

    memory.add_user_message("What did I just say?")
    reply2 = brain.ask("What did I just say?", context=memory)

    assert "The secret word is pineapple" in reply2


def test_llm_brain_respects_context_limit():
    """Verify LLMBrain limits the number of context messages supplied to prompt."""
    provider = MockLLMProvider()
    brain = LLMBrain(provider=provider, max_context_messages=2)

    # Create memory with 6 messages
    memory = ConversationMemory(max_messages=10)
    for i in range(3):
        memory.add_user_message(f"User query {i}")
        memory.add_assistant_message(f"Assistant reply {i}")

    # Ask with max_context_messages=2
    messages = build_llm_messages(
        user_message="Current question",
        context=memory.get_context(max_messages=2),
    )
    # [System, Context Msg 1, Context Msg 2, Current User] = 4 messages total
    assert len(messages) == 4
    assert messages[1]["content"] == "User query 2"
    assert messages[2]["content"] == "Assistant reply 2"
    assert messages[3]["content"] == "Current question"


# ==============================================================================
# 3. Long-Term Vector Memory Abstraction Tests
# ==============================================================================

def test_vector_memory_interface_and_local_store():
    """Verify VectorMemory implements BaseVectorMemory with store and search."""
    vector_mem = VectorMemory()
    assert isinstance(vector_mem, BaseVectorMemory)
    assert vector_mem.count() == 0

    # Store long-term knowledge documents
    doc1_id = vector_mem.store("User lives in Hyderabad and works as a software engineer.", {"category": "profile"})
    doc2_id = vector_mem.store("User prefers concise voice responses without bullet points.", {"category": "preference"})
    doc3_id = vector_mem.store("Python was created by Guido van Rossum in 1991.", {"category": "fact"})

    assert vector_mem.count() == 3
    assert doc1_id == "mem_1"
    assert doc2_id == "mem_2"

    # Search semantically relevant documents
    results = vector_mem.search("Where does the user live?", top_k=2)
    assert len(results) > 0
    assert "Hyderabad" in results[0]["text"]
    assert results[0]["score"] > 0.0
    assert results[0]["metadata"]["category"] == "profile"


def test_vector_memory_clear_and_empty():
    """Verify VectorMemory empty search and clear behavior."""
    vector_mem = VectorMemory()
    # Search empty memory
    assert vector_mem.search("anything") == []

    vector_mem.store("Test fact")
    assert vector_mem.count() == 1

    vector_mem.clear()
    assert vector_mem.count() == 0
    assert vector_mem.get_all() == []

