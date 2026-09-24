"""Tests for AURA Phase 10: Agent Module.

Verifies:
- Step 24:
  - AgentState initialization, mutation, serialization, and reset
  - Planner generating structured, valid rule-based plans
  - AuraAgent initialization and lifecycle
  - execute_plan() and action stubs raising NotImplementedError
- Step 25:
  - Agent can access memory
  - Agent can access LLM
  - User goal is stored in AgentState
  - Conversation history is passed to the agent
  - Plan is stored
  - LLM response is stored as final_response
  - Conversation is saved to memory
  - Mocks for LLM and memory
"""

from unittest.mock import MagicMock
import pytest

from app.agent.state import AgentState, AgentStatus
from app.agent.planner import Planner
from app.agent.agent import AuraAgent, PlanningInfo
from app.agent.actions import (
    execute_action,
    ActionExecutor,
    execute_calculator,
    execute_weather,
    execute_search,
    execute_files,
    execute_respond,
)
from app.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
)
from app.memory.conversation import ConversationMemory
from app.brain.llm import LLMBrain


class TestAgentState:
    """Test suite for AgentState dataclass and lifecycle status."""

    def test_default_initialization(self) -> None:
        state = AgentState()
        assert state.user_goal == ""
        assert state.goal == ""
        assert state.current_step == 0
        assert state.plan == []
        assert state.completed_actions == []
        assert state.observations == []
        assert state.retrieved_context == []
        assert state.tool_results == {}
        assert state.final_response is None
        assert state.status == AgentStatus.IDLE
        assert state.steps_count == 0

    def test_custom_initialization(self) -> None:
        state = AgentState(
            user_goal="Calculate 10 + 5",
            current_step=1,
            plan=[{"step": 1, "action": "calculator", "reason": "compute"}],
            status="planning",
        )
        assert state.user_goal == "Calculate 10 + 5"
        assert state.goal == "Calculate 10 + 5"
        assert state.current_step == 1
        assert len(state.plan) == 1
        assert state.steps_count == 1
        assert state.status == AgentStatus.PLANNING

    def test_invalid_status_fallback(self) -> None:
        state = AgentState(status="invalid_status_xyz")  # type: ignore
        assert state.status == AgentStatus.IDLE

    def test_add_observation_and_action(self) -> None:
        state = AgentState()
        state.add_observation("Observed 1200 result")
        state.add_completed_action({"step": 1, "action": "calculator", "success": True})

        assert state.observations == ["Observed 1200 result"]
        assert len(state.completed_actions) == 1
        assert state.completed_actions[0]["action"] == "calculator"

    def test_to_dict_serialization(self) -> None:
        state = AgentState(user_goal="Check weather", status=AgentStatus.EXECUTING)
        d = state.to_dict()
        assert isinstance(d, dict)
        assert d["user_goal"] == "Check weather"
        assert d["status"] == "executing"
        assert "plan" in d

    def test_dict_subscript_access(self) -> None:
        state = AgentState(user_goal="Test goal", status=AgentStatus.PLANNING)
        assert state["user_goal"] == "Test goal"
        assert state["goal"] == "Test goal"
        assert state["status"] == "planning"
        assert state.get("goal") == "Test goal"
        assert state.get("non_existent", "default_val") == "default_val"
        assert "goal" in state
        assert "user_goal" in state
        assert "plan" in state
        with pytest.raises(KeyError):
            _ = state["non_existent_field"]

    def test_reset(self) -> None:
        state = AgentState(
            user_goal="Some goal",
            current_step=2,
            plan=[{"step": 1}],
            status=AgentStatus.COMPLETED,
        )
        state.reset()
        assert state.user_goal == ""
        assert state.current_step == 0
        assert state.plan == []
        assert state.status == AgentStatus.IDLE


class TestPlanner:
    """Test suite for the rule-based Planner."""

    @pytest.fixture
    def planner(self) -> Planner:
        return Planner()

    def test_math_calculation_plan(self, planner: Planner) -> None:
        """Verify the exact example from the requirements."""
        goal = "Calculate 25 * 48 and explain."
        plan = planner.create_plan(goal)

        expected = [
            {"step": 1, "action": "calculator", "reason": "Need mathematical computation"},
            {"step": 2, "action": "respond", "reason": "Explain result to user"},
        ]
        assert plan == expected

    def test_weather_query_plan(self, planner: Planner) -> None:
        goal = "What is the weather in Tokyo tomorrow?"
        plan = planner.create_plan(goal)

        assert len(plan) == 2
        assert plan[0]["step"] == 1
        assert plan[0]["action"] == "weather"
        assert plan[1]["step"] == 2
        assert plan[1]["action"] == "respond"

    def test_search_query_plan(self, planner: Planner) -> None:
        goal = "Search the web for latest quantum computing developments"
        plan = planner.create_plan(goal)

        assert len(plan) == 2
        assert plan[0]["step"] == 1
        assert plan[0]["action"] == "search"
        assert plan[1]["step"] == 2
        assert plan[1]["action"] == "respond"

    def test_file_query_plan(self, planner: Planner) -> None:
        goal = "Read file data/notes.txt and list files"
        plan = planner.create_plan(goal)

        assert len(plan) == 2
        assert plan[0]["step"] == 1
        assert plan[0]["action"] == "files"
        assert plan[1]["step"] == 2
        assert plan[1]["action"] == "respond"

    def test_conversational_fallback_plan(self, planner: Planner) -> None:
        goal = "Hello AURA, how are you today?"
        plan = planner.create_plan(goal)

        assert len(plan) == 1
        assert plan[0]["step"] == 1
        assert plan[0]["action"] == "respond"
        assert plan[0]["reason"] == "Explain result to user"

    def test_empty_goal_plan(self, planner: Planner) -> None:
        plan = planner.create_plan("")
        assert len(plan) == 1
        assert plan[0]["action"] == "respond"

    def test_create_plan_with_context(self, planner: Planner) -> None:
        context = [{"role": "user", "content": "Let's do some math"}]
        plan = planner.create_plan("Calculate 12 * 12", context=context)
        assert len(plan) == 2
        assert plan[0]["action"] == "calculator"


class TestAuraAgent:
    """Test suite for AuraAgent coordination."""

    def test_agent_initialization(self) -> None:
        agent = AuraAgent()
        assert agent.planner is not None
        assert agent.state is not None
        assert agent.memory is not None
        assert agent.llm is not None
        assert agent.state.status == AgentStatus.IDLE
        assert agent.state.user_goal == ""
        assert agent.state.plan == []

    def test_create_plan_updates_state(self) -> None:
        agent = AuraAgent()
        plan = agent.create_plan("Calculate 100 / 4 and show work")

        assert agent.state.user_goal == "Calculate 100 / 4 and show work"
        assert agent.state.status == AgentStatus.PLANNING
        assert agent.state.plan == plan
        assert len(plan) >= 2
        assert plan[0]["action"] == "calculator"

    def test_process_goal_returns_agent_state(self) -> None:
        agent = AuraAgent()
        goal = "Calculate 25 * 48 and explain."
        result = agent.process_goal(goal)

        assert isinstance(result, AgentState)
        assert result["goal"] == goal
        assert result.goal == goal
        assert "plan" in result
        assert len(result["plan"]) == 2
        assert result.plan[0]["action"] == "calculator"
        assert result.plan[1]["action"] == "respond"
        assert result.status == AgentStatus.COMPLETED
        assert result["status"] == "completed"
        assert result.steps_count == 2
        assert result.final_response is not None

    def test_display_plan_formatting(self) -> None:
        agent = AuraAgent()
        agent.state.user_goal = "Calculate 5 + 5"
        plan = [{"step": 1, "action": "calculator", "reason": "Math"}]
        display_str = agent.display_plan(plan)

        assert "AURA AGENT PLAN" in display_str
        assert "Goal: Calculate 5 + 5" in display_str
        assert "1. [CALCULATOR] Math" in display_str

    def test_execute_plan_raises_not_implemented(self) -> None:
        agent = AuraAgent()
        agent.create_plan("What is the weather in Tokyo tomorrow?")
        with pytest.raises(NotImplementedError) as exc_info:
            agent.execute_plan()
        assert "Tool execution is disabled" in str(exc_info.value)


class TestAgentMemoryAndLLM:
    """Specific tests for Step 25: Memory and LLM integration."""

    def test_agent_can_access_memory(self) -> None:
        custom_memory = ConversationMemory(max_messages=6)
        agent = AuraAgent(memory=custom_memory)
        assert agent.memory is custom_memory
        assert agent.memory.max_messages == 6

    def test_agent_can_access_llm(self) -> None:
        mock_llm = MagicMock(spec=LLMBrain)
        agent = AuraAgent(llm=mock_llm)
        assert agent.llm is mock_llm

    def test_user_goal_is_stored_in_agent_state(self) -> None:
        agent = AuraAgent()
        goal_text = "Summarize the history of computing."
        state = agent.process_goal(goal_text)

        assert state.user_goal == goal_text
        assert state.goal == goal_text

    def test_conversation_history_is_passed_to_agent(self) -> None:
        memory = ConversationMemory()
        memory.add_user_message("My favorite language is Python.")
        memory.add_assistant_message("Python is a fantastic language.")

        mock_llm = MagicMock()
        mock_llm.ask.return_value = "You previously stated your favorite language is Python."

        agent = AuraAgent(llm=mock_llm, memory=memory)
        state = agent.process_goal("What is my favorite language?")

        # Retrieved context in state should have the 2 past messages
        assert len(state.retrieved_context) == 2
        assert state.retrieved_context[0]["content"] == "My favorite language is Python."
        assert state.retrieved_context[1]["content"] == "Python is a fantastic language."

        # Verify LLM was called with the context
        mock_llm.ask.assert_called_once()
        call_kwargs = mock_llm.ask.call_args
        assert call_kwargs.kwargs["user_message"] == "What is my favorite language?"
        assert call_kwargs.kwargs["context"] == state.retrieved_context

    def test_plan_is_stored_in_agent_state(self) -> None:
        agent = AuraAgent()
        state = agent.process_goal("Search the web for python tutorials")

        assert len(state.plan) > 0
        assert any(step["action"] == "search" for step in state.plan)
        assert state.plan[-1]["action"] == "respond"

    def test_llm_response_is_stored_as_final_response(self) -> None:
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Machine learning is a subset of AI that learns from data."

        agent = AuraAgent(llm=mock_llm)
        state = agent.process_goal("Explain what machine learning is.")

        assert state.final_response == "Machine learning is a subset of AI that learns from data."
        assert state.status == AgentStatus.COMPLETED

    def test_conversation_is_saved_to_memory(self) -> None:
        memory = ConversationMemory()
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Quantum computing uses qubits."

        agent = AuraAgent(llm=mock_llm, memory=memory)
        assert len(memory) == 0

        goal = "Explain quantum computing briefly."
        state = agent.process_goal(goal)

        # Memory should now contain 2 turns (user goal and assistant response)
        assert len(memory) == 2
        messages = memory.get_history()
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == goal
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Quantum computing uses qubits."

    def test_multi_turn_continuity_with_default_components(self) -> None:
        agent = AuraAgent()
        state1 = agent.process_goal("Hello AURA, remember that I like green tea.")
        assert state1.status == AgentStatus.COMPLETED
        assert len(agent.memory) == 2

        state2 = agent.process_goal("What beverage do I like?")
        assert state2.status == AgentStatus.COMPLETED
        assert len(state2.retrieved_context) == 2
        assert len(agent.memory) == 4


class TestActionsAndPrompts:
    """Test suite verifying action stubs and decoupled prompts."""

    def test_execute_action_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            execute_action("calculator", expression="25 * 48")
        assert "Tool execution is disabled" in str(exc_info.value)

    def test_action_executor_raises_not_implemented(self) -> None:
        executor = ActionExecutor()
        with pytest.raises(NotImplementedError):
            executor.execute({"step": 1, "action": "weather"})

    def test_individual_action_stubs_raise_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            execute_calculator()
        with pytest.raises(NotImplementedError):
            execute_weather()
        with pytest.raises(NotImplementedError):
            execute_search()
        with pytest.raises(NotImplementedError):
            execute_files()
        with pytest.raises(NotImplementedError):
            execute_respond()

    def test_prompts_integrity(self) -> None:
        assert isinstance(AGENT_SYSTEM_PROMPT, str) and len(AGENT_SYSTEM_PROMPT) > 50
        assert isinstance(PLANNER_SYSTEM_PROMPT, str) and len(PLANNER_SYSTEM_PROMPT) > 50
        assert isinstance(EXECUTOR_SYSTEM_PROMPT, str) and len(EXECUTOR_SYSTEM_PROMPT) > 50
        assert isinstance(REFLECTOR_SYSTEM_PROMPT, str) and len(REFLECTOR_SYSTEM_PROMPT) > 50
        assert "AURA Agent" in AGENT_SYSTEM_PROMPT


class TestRAGAwareAgent:
    """Test suite for Phase 10 Step 27: RAG-Aware Agent."""

    @pytest.fixture
    def planner(self) -> Planner:
        return Planner()

    def test_planner_recognizes_document_question_ai_notes(self, planner: Planner) -> None:
        """Verify planner identifies query referencing notes document and schedules retrieve -> respond."""
        goal = "What does my AI notes document say about supervised learning?"
        assert planner.is_document_query(goal) is True

        plan = planner.create_plan(goal)
        assert len(plan) == 2
        assert plan[0]["step"] == 1
        assert plan[0]["action"] == "retrieve"
        assert plan[1]["step"] == 2
        assert plan[1]["action"] == "respond"

    def test_planner_recognizes_document_question_according_to(self, planner: Planner) -> None:
        """Verify planner identifies 'according to my documents' query and schedules retrieve -> respond."""
        goal = "What is overfitting according to my documents?"
        assert planner.is_document_query(goal) is True

        plan = planner.create_plan(goal)
        assert len(plan) == 2
        assert plan[0]["step"] == 1
        assert plan[0]["action"] == "retrieve"
        assert plan[1]["step"] == 2
        assert plan[1]["action"] == "respond"

    def test_planner_recognizes_normal_question(self, planner: Planner) -> None:
        """Verify planner does not schedule retrieve for general knowledge questions like 'What is Python?'."""
        goal = "What is Python?"
        assert planner.is_document_query(goal) is False

        plan = planner.create_plan(goal)
        assert len(plan) == 1
        assert plan[0]["step"] == 1
        assert plan[0]["action"] == "respond"

    def test_planner_recognizes_conversational_question(self, planner: Planner) -> None:
        """Verify conversational questions do not trigger retrieval."""
        goal = "Tell me a joke about programming"
        assert planner.is_document_query(goal) is False

        plan = planner.create_plan(goal)
        assert len(plan) == 1
        assert plan[0]["action"] == "respond"

    def test_agent_executes_retrieval(self) -> None:
        """Verify agent invokes retriever with user query when retrieve action is planned."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {
                "id": "chunk_1",
                "text": "Overfitting happens when a model learns noise in training data.",
                "score": 0.92,
                "metadata": {"filename": "AI_Notes.txt", "page": 2},
            }
        ]
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Overfitting is memorizing noise."

        agent = AuraAgent(llm=mock_llm, retriever=mock_retriever)
        goal = "What is overfitting according to my documents?"
        state = agent.process_goal(goal)

        mock_retriever.retrieve.assert_called_once_with(goal)
        assert state.status == AgentStatus.COMPLETED

    def test_retrieved_context_is_stored_in_agent_state(self) -> None:
        """Verify retrieved document chunks are stored in AgentState.retrieved_context."""
        sample_chunks = [
            {
                "id": "c_ai_1",
                "text": "Supervised learning algorithms map inputs to outputs using labeled training sets.",
                "score": 0.89,
                "metadata": {"filename": "AI_Notes.txt", "page": 1},
            }
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Supervised learning uses labeled training sets."

        agent = AuraAgent(llm=mock_llm, retriever=mock_retriever)
        state = agent.process_goal("What does my AI notes document say about supervised learning?")

        assert state.retrieved_context == sample_chunks
        assert len(state.retrieved_context) == 1
        assert state.retrieved_context[0]["id"] == "c_ai_1"
        assert "Supervised learning" in state.retrieved_context[0]["text"]

    def test_source_metadata_is_preserved(self) -> None:
        """Verify filename, page number, and similarity score metadata are preserved accurately."""
        sample_chunks = [
            {
                "id": "chunk_42",
                "text": "Regularization helps reduce overfitting by penalizing large weights.",
                "score": 0.8765,
                "metadata": {
                    "filename": "deep_learning.pdf",
                    "page": 7,
                    "source": "/path/to/deep_learning.pdf",
                },
            }
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Regularization reduces overfitting."

        agent = AuraAgent(llm=mock_llm, retriever=mock_retriever)
        state = agent.process_goal("What is overfitting according to my documents?")

        # Check preserved in retrieved_context
        first_chunk = state.retrieved_context[0]
        assert first_chunk["metadata"]["filename"] == "deep_learning.pdf"
        assert first_chunk["metadata"]["page"] == 7
        assert first_chunk["score"] == 0.8765

        # Check preserved in tool_results
        assert "retrieve" in state.tool_results
        sources = state.tool_results["retrieve"]["sources"]
        assert len(sources) == 1
        assert sources[0]["filename"] == "deep_learning.pdf"
        assert sources[0]["page"] == 7
        assert sources[0]["score"] == 0.8765
        assert sources[0]["chunk_id"] == "chunk_42"

    def test_llm_receives_retrieved_context(self) -> None:
        """Verify LLM is supplied user goal, retrieved context, and conversation history."""
        sample_chunks = [
            {
                "id": "c1",
                "text": "Overfitting happens when a model learns noise.",
                "score": 0.95,
                "metadata": {"filename": "notes.txt", "page": 1},
            }
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Grounded response."

        memory = ConversationMemory()
        memory.add_user_message("Hello AURA")
        memory.add_assistant_message("Hello! How can I help?")

        agent = AuraAgent(llm=mock_llm, retriever=mock_retriever, memory=memory)
        goal = "What is overfitting according to my documents?"
        state = agent.process_goal(goal)

        mock_llm.ask.assert_called_once()
        call_kwargs = mock_llm.ask.call_args.kwargs
        assert call_kwargs["user_message"] == goal
        assert call_kwargs["rag_context"] == sample_chunks
        assert len(call_kwargs["context"]) == 2

    def test_agent_produces_final_response_from_rag_context(self) -> None:
        """Verify agent produces grounded answer citing source document and page using MockLLMProvider."""
        sample_chunks = [
            {
                "id": "ai_chunk_1",
                "text": "Supervised learning trains a model using labeled inputs and outputs.",
                "score": 0.91,
                "metadata": {"filename": "AI_Notes.txt", "page": 3},
            }
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks

        # Use actual LLMBrain with MockLLMProvider
        agent = AuraAgent(llm=LLMBrain(enable_tools=False), retriever=mock_retriever)
        state = agent.process_goal("What does my AI notes document say about supervised learning?")

        assert state.final_response is not None
        assert "Supervised learning trains a model" in state.final_response
        assert "AI_Notes.txt" in state.final_response
        assert "page 3" in state.final_response
        assert state.status == AgentStatus.COMPLETED

    def test_retrieval_failure_handled_safely(self) -> None:
        """Verify retriever exception is handled gracefully without crashing the agent."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("Vector database disk read error")

        mock_llm = MagicMock()
        mock_llm.ask.return_value = "The available documents do not provide enough information."

        agent = AuraAgent(llm=mock_llm, retriever=mock_retriever)
        state = agent.process_goal("What is overfitting according to my documents?")

        assert state.status == AgentStatus.COMPLETED
        assert state.retrieved_context == []
        assert any("Retrieval failed" in obs for obs in state.observations)
        assert state.final_response is not None

    def test_no_context_questions_do_not_unnecessarily_call_rag(self) -> None:
        """Verify general queries like 'What is Python?' never query the RAG retriever."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Python is a high-level programming language."

        agent = AuraAgent(llm=mock_llm, retriever=mock_retriever)
        state = agent.process_goal("What is Python?")

        mock_retriever.retrieve.assert_not_called()
        assert state.final_response == "Python is a high-level programming language."
        assert state.status == AgentStatus.COMPLETED

    def test_agent_state_fields_updated_correctly(self) -> None:
        """Verify all 8 AgentState fields are properly updated during RAG lifecycle."""
        sample_chunks = [
            {
                "id": "ch_1",
                "text": "Overfitting happens when a model learns noise.",
                "score": 0.88,
                "metadata": {"filename": "notes.md", "page": 1},
            }
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Overfitting is learning noise."

        agent = AuraAgent(llm=mock_llm, retriever=mock_retriever)
        goal = "What is overfitting according to my documents?"
        state = agent.process_goal(goal)

        # 1. user_goal
        assert state.user_goal == goal
        # 2. plan
        assert len(state.plan) == 2
        assert state.plan[0]["action"] == "retrieve"
        assert state.plan[1]["action"] == "respond"
        # 3. current_step
        assert state.current_step == 2
        # 4. retrieved_context
        assert state.retrieved_context == sample_chunks
        # 5. observations
        assert len(state.observations) >= 2
        assert any("Retrieved" in obs for obs in state.observations)
        # 6. tool_results
        assert "retrieve" in state.tool_results
        assert state.tool_results["retrieve"]["count"] == 1
        # 7. final_response
        assert state.final_response == "Overfitting is learning noise."
        # 8. status
        assert state.status == AgentStatus.COMPLETED

    def test_insufficient_context_handled_cleanly(self) -> None:
        """Verify LLM clearly states available documents do not provide enough information when retrieval returns no chunks."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        agent = AuraAgent(llm=LLMBrain(enable_tools=False), retriever=mock_retriever)
        state = agent.process_goal("What is overfitting according to my documents?")

        assert state.retrieved_context == []
        assert "available documents do not provide enough information" in state.final_response.lower()


class TestMultiStepPlanningAndObservation:
    """Test suite for Phase 10 Step 28: Multi-Step Planning & Observation."""

    def test_two_step_calculator_response_flow(self) -> None:
        """Verify agent generates and executes 1. calculator -> 2. respond flow."""
        agent = AuraAgent()
        goal = "Calculate 25 * 40 and explain the answer."
        state = agent.process_goal(goal)

        # Plan inspection
        assert len(state.plan) == 2
        assert state.plan[0]["action"] == "calculator"
        assert state.plan[1]["action"] == "respond"

        # Execution inspection
        assert state.status == AgentStatus.COMPLETED
        assert len(state.completed_actions) == 2
        assert state.completed_actions[0]["action"] == "calculator"
        assert state.completed_actions[0]["success"] is True
        assert state.completed_actions[0]["result"] == 1000
        assert state.completed_actions[1]["action"] == "respond"
        assert state.completed_actions[1]["success"] is True

        # Tool result preserved
        assert "calculator" in state.tool_results
        assert state.tool_results["calculator"]["result"] == 1000

        # Observations
        assert len(state.observations) >= 2
        assert any("1000" in obs for obs in state.observations)

        # Final response uses actual result
        assert state.final_response is not None
        assert "1000" in state.final_response

    def test_retrieve_response_flow(self) -> None:
        """Verify agent generates and executes 1. retrieve -> 2. respond flow."""
        mock_retriever = MagicMock()
        sample_chunks = [
            {
                "id": "c_sl_1",
                "text": "Supervised learning algorithms map inputs to outputs based on labeled pairs.",
                "score": 0.94,
                "metadata": {"filename": "ml_guide.txt", "page": 1},
            }
        ]
        mock_retriever.retrieve.return_value = sample_chunks

        agent = AuraAgent(llm=LLMBrain(enable_tools=False), retriever=mock_retriever)
        goal = "Find information about supervised learning in my documents and explain it."
        state = agent.process_goal(goal)

        # Plan inspection
        assert len(state.plan) == 2
        assert state.plan[0]["action"] == "retrieve"
        assert state.plan[1]["action"] == "respond"

        # Execution inspection
        assert state.status == AgentStatus.COMPLETED
        mock_retriever.retrieve.assert_called_once_with(goal)
        assert state.retrieved_context == sample_chunks
        assert "retrieve" in state.tool_results
        assert state.tool_results["retrieve"]["count"] == 1

        # Observations & completed actions
        assert len(state.observations) >= 2
        assert any("Retrieved 1" in obs for obs in state.observations)
        assert len(state.completed_actions) == 2
        assert state.completed_actions[0]["action"] == "retrieve"
        assert state.completed_actions[1]["action"] == "respond"

        # Final response is grounded
        assert state.final_response is not None
        assert "Supervised learning" in state.final_response

    def test_multiple_observations(self) -> None:
        """Verify sequential actions record distinct observations in order without losing any."""
        agent = AuraAgent()
        goal = "Calculate 100 / 4 and explain the answer."
        state = agent.process_goal(goal)

        assert len(state.observations) >= 2
        # First observation records computation
        assert "25" in state.observations[0]
        # Second observation records response synthesis
        assert "response" in state.observations[1].lower()

    def test_previous_tool_result_available_to_later_step(self) -> None:
        """Verify the output of an earlier action is available to and used by a subsequent action."""
        agent = AuraAgent()
        # Custom plan chaining two calculator steps then respond
        custom_plan = [
            {"step": 1, "action": "calculator", "expression": "25 * 40", "reason": "Initial computation"},
            {"step": 2, "action": "calculator", "expression": "{previous} + 50", "reason": "Chained addition"},
            {"step": 3, "action": "respond", "reason": "Final synthesis"},
        ]
        agent.state.plan = custom_plan
        agent.state.user_goal = "Compute 25 * 40 and then add 50"
        state = agent.execute_plan()

        assert state.status == AgentStatus.COMPLETED
        assert len(state.completed_actions) == 3

        # Step 1 produced 1000
        assert state.completed_actions[0]["result"] == 1000
        # Step 2 used Step 1's 1000 to compute 1050
        assert state.completed_actions[1]["result"] == 1050

        # Observations recorded for both steps
        assert any("1000" in obs for obs in state.observations)
        assert any("1050" in obs for obs in state.observations)

        # Final response receives the final result
        assert state.final_response is not None
        assert "1050" in state.final_response

    def test_intermediate_action_failure(self) -> None:
        """Verify failure in an intermediate action halts execution, stores error, and sets FAILED status."""
        agent = AuraAgent()
        custom_plan = [
            {"step": 1, "action": "calculator", "expression": "10 / 0", "reason": "Divide by zero"},
            {"step": 2, "action": "respond", "reason": "Should not execute"},
        ]
        agent.state.plan = custom_plan
        agent.state.user_goal = "Calculate 10 / 0 and explain."
        state = agent.execute_plan()

        # Agent must record failure, set status FAILED, and stop
        assert state.status == AgentStatus.FAILED
        assert state.error is not None
        assert "zero" in state.error.lower()

        # Step 1 recorded as failure
        assert len(state.completed_actions) == 1
        assert state.completed_actions[0]["action"] == "calculator"
        assert state.completed_actions[0]["success"] is False
        assert "zero" in state.completed_actions[0]["error"].lower()

        # Step 2 was never executed
        assert not any(act.get("action") == "respond" for act in state.completed_actions)

        # Observations record the failure
        assert any("failed" in obs.lower() for obs in state.observations)

    def test_maximum_step_protection(self) -> None:
        """Verify agent halts execution when the number of actions exceeds the max_steps limit."""
        # Agent configured with a strict limit of 2 steps
        agent = AuraAgent(max_steps=2)
        long_plan = [
            {"step": 1, "action": "calculator", "expression": "1 + 1", "reason": "step 1"},
            {"step": 2, "action": "calculator", "expression": "2 + 2", "reason": "step 2"},
            {"step": 3, "action": "calculator", "expression": "3 + 3", "reason": "step 3"},
            {"step": 4, "action": "respond", "reason": "step 4"},
        ]
        agent.state.plan = long_plan
        agent.state.user_goal = "Perform multiple steps"
        state = agent.execute_plan()

        # Execution halted due to exceeding max_steps limit
        assert state.status == AgentStatus.FAILED
        assert state.error is not None
        assert "maximum step limit" in state.error.lower()
        assert any("maximum step limit" in obs.lower() for obs in state.observations)
        # Did not execute all 4 steps
        assert len(state.completed_actions) < 4

    def test_final_response_receives_all_relevant_context(self) -> None:
        """Verify respond action passes user goal, conversation context, rag_context, tool_results, and observations to LLM."""
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Everything was passed accurately."

        mock_retriever = MagicMock()
        retrieved_chunks = [{"id": "c1", "text": "Document text", "metadata": {"filename": "doc.txt"}}]
        mock_retriever.retrieve.return_value = retrieved_chunks

        memory = ConversationMemory()
        memory.add_user_message("Prior query")
        memory.add_assistant_message("Prior reply")

        agent = AuraAgent(
            llm=mock_llm,
            memory=memory,
            retriever=mock_retriever,
        )

        custom_plan = [
            {"step": 1, "action": "retrieve", "query": "Document search", "reason": "Find docs"},
            {"step": 2, "action": "calculator", "expression": "50 * 2", "reason": "Compute"},
            {"step": 3, "action": "respond", "reason": "Explain"},
        ]
        agent.state.plan = custom_plan
        agent.state.user_goal = "Search docs, compute 50 * 2, and explain."
        state = agent.execute_plan()

        assert state.status == AgentStatus.COMPLETED
        mock_llm.ask.assert_called_once()
        call_kwargs = mock_llm.ask.call_args.kwargs

        # Verify all relevant context was provided to LLM
        assert call_kwargs["user_message"] == "Search docs, compute 50 * 2, and explain."
        assert len(call_kwargs["context"]) == 2
        assert call_kwargs["rag_context"] == retrieved_chunks
        assert "calculator" in call_kwargs["tool_results"]
        assert call_kwargs["tool_results"]["calculator"]["result"] == 100
        assert "retrieve" in call_kwargs["tool_results"]
        assert len(call_kwargs["observations"]) >= 2

    def test_agent_state_correctly_records_completed_actions(self) -> None:
        """Verify AgentState tracks every completed action in order with outcome metadata."""
        agent = AuraAgent()
        goal = "Calculate 15 * 6 and explain."
        state = agent.process_goal(goal)

        assert len(state.completed_actions) == 2
        act1 = state.completed_actions[0]
        assert act1["step"] == 1
        assert act1["action"] == "calculator"
        assert act1["success"] is True
        assert act1["result"] == 90

        act2 = state.completed_actions[1]
        assert act2["step"] == 2
        assert act2["action"] == "respond"
        assert act2["success"] is True
        assert "response" in act2


