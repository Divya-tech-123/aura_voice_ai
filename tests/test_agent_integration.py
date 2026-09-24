"""End-to-End Integration Tests for AURA Phase 10 Agent (Step 30).

Validates the full coordination of the autonomous AURA Agent:
- Planner
- AgentState
- Conversation Memory
- LLM
- Tools
- RAG
- Observations
- Safety limits (max steps, invalid actions)
- Error handling & tool failures
- Filesystem sandbox security (path traversal defense)

All tests run deterministically and safely with zero dependency on real external APIs.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path

from app.agent.agent import AuraAgent
from app.agent.state import AgentState, AgentStatus
from app.agent.planner import Planner
from app.memory.conversation import ConversationMemory
from app.brain.llm import LLMBrain, MockLLMProvider
from app.tools.registry import ToolRegistry, create_default_registry
from app.tools.files import SafeFileManager, FileTool
from app.tools.calculator import CalculatorTool


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def sample_rag_chunks() -> List[Dict[str, Any]]:
    """Fixture providing realistic document chunks with source metadata."""
    return [
        {
            "id": "chunk_sl_001",
            "text": "Supervised learning algorithms learn a mapping from input features to output targets using labeled training datasets.",
            "score": 0.945,
            "metadata": {
                "filename": "AI_Notes.txt",
                "page": 3,
                "author": "AURA Research",
                "section": "Fundamentals",
            },
        },
        {
            "id": "chunk_of_002",
            "text": "Overfitting happens when a statistical model describes random error or noise instead of the underlying relationship.",
            "score": 0.912,
            "metadata": {
                "filename": "Machine_Learning_Guide.pdf",
                "page": 12,
                "author": "Dr. AI",
                "section": "Model Diagnostics",
            },
        },
    ]


@pytest.fixture
def mock_retriever(sample_rag_chunks: List[Dict[str, Any]]) -> MagicMock:
    """Fixture providing a mock RAG retriever."""
    retriever = MagicMock()
    retriever.retrieve.return_value = sample_rag_chunks
    return retriever


# ==============================================================================
# 2. Test Case — Normal Conversation
# ==============================================================================

class TestNormalConversation:
    """Validates basic conversational flow without tools or RAG."""

    def test_normal_conversation_end_to_end(self) -> None:
        """Verify:
        Agent initializes
        -> plan is created (respond)
        -> LLM is called
        -> final response is generated
        -> state becomes completed
        """
        agent = AuraAgent()
        assert agent.state.status == AgentStatus.IDLE

        input_query = "What is artificial intelligence?"
        state = agent.process_goal(input_query)

        # 1. Agent State verification
        assert isinstance(state, AgentState)
        assert state.user_goal == input_query
        assert state.goal == input_query
        assert state.status == AgentStatus.COMPLETED

        # 2. Plan verification
        assert len(state.plan) == 1
        assert state.plan[0]["action"] == "respond"
        assert "explain" in state.plan[0]["reason"].lower()

        # 3. LLM call & final response verification
        assert state.final_response is not None
        assert len(state.final_response) > 0
        assert "artificial intelligence" in state.final_response.lower()

        # 4. Completed actions and observations recorded
        assert len(state.completed_actions) == 1
        assert state.completed_actions[0]["action"] == "respond"
        assert state.completed_actions[0]["success"] is True
        assert len(state.observations) >= 1

        # 5. Conversation memory recorded
        assert len(agent.memory) == 2
        history = agent.memory.get_history()
        assert history[0]["role"] == "user"
        assert history[0]["content"] == input_query
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == state.final_response

    def test_normal_conversation_mock_llm(self) -> None:
        """Verify agent coordinates cleanly when LLM provider is explicitly mocked."""
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "AI is the simulation of human intelligence by computer systems."

        agent = AuraAgent(llm=mock_llm)
        state = agent.process_goal("What is artificial intelligence?")

        assert state.status == AgentStatus.COMPLETED
        assert state.final_response == "AI is the simulation of human intelligence by computer systems."
        mock_llm.ask.assert_called_once()
        call_kwargs = mock_llm.ask.call_args.kwargs
        assert call_kwargs["user_message"] == "What is artificial intelligence?"


# ==============================================================================
# 3. Test Case — Calculator Agent
# ==============================================================================

class TestCalculatorAgent:
    """Validates mathematical calculation workflow and LLM context availability."""

    def test_calculator_agent_flow(self) -> None:
        """Input: 'Calculate 125 * 32 and explain the answer.'
        Verify:
        planner -> calculator -> observation -> respond -> final answer
        Verify the actual calculator result (4000) is available to the LLM.
        """
        agent = AuraAgent()
        input_query = "Calculate 125 * 32 and explain the answer."
        state = agent.process_goal(input_query)

        # 1. Planner verified
        assert len(state.plan) == 2
        assert state.plan[0]["step"] == 1
        assert state.plan[0]["action"] == "calculator"
        assert state.plan[1]["step"] == 2
        assert state.plan[1]["action"] == "respond"

        # 2. Calculator step executed and result produced
        assert "calculator" in state.tool_results
        calc_result = state.tool_results["calculator"]
        assert calc_result["success"] is True
        assert calc_result["result"] == 4000

        # 3. Observation recorded
        assert len(state.observations) >= 2
        assert any("4000" in obs for obs in state.observations)

        # 4. Final answer generated and completed
        assert state.status == AgentStatus.COMPLETED
        assert state.final_response is not None
        assert "4000" in state.final_response

        # 5. Completed actions record in order
        assert len(state.completed_actions) == 2
        assert state.completed_actions[0]["action"] == "calculator"
        assert state.completed_actions[0]["result"] == 4000
        assert state.completed_actions[1]["action"] == "respond"
        assert state.completed_actions[1]["success"] is True

    def test_calculator_result_passed_to_llm(self) -> None:
        """Explicitly verify that the calculator's numeric output is supplied to the LLM."""
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "The product of 125 and 32 is exactly 4000."

        agent = AuraAgent(llm=mock_llm)
        input_query = "Calculate 125 * 32 and explain the answer."
        state = agent.process_goal(input_query)

        assert state.status == AgentStatus.COMPLETED
        mock_llm.ask.assert_called_once()
        call_kwargs = mock_llm.ask.call_args.kwargs

        # Verify tool_results was passed with 4000
        assert "tool_results" in call_kwargs
        assert "calculator" in call_kwargs["tool_results"]
        assert call_kwargs["tool_results"]["calculator"]["result"] == 4000
        # Verify observations containing 4000 were passed
        assert "observations" in call_kwargs
        assert any("4000" in obs for obs in call_kwargs["observations"])


# ==============================================================================
# 4. Test Case — RAG Agent
# ==============================================================================

class TestRAGAgent:
    """Validates document retrieval, metadata preservation, and grounded synthesis."""

    def test_rag_agent_flow(self, mock_retriever: MagicMock, sample_rag_chunks: List[Dict[str, Any]]) -> None:
        """Input: 'What is supervised learning according to my documents?'
        Verify:
        planner -> retrieve -> retrieved context -> observation -> respond -> final answer
        Verify source metadata is preserved.
        """
        agent = AuraAgent(retriever=mock_retriever)
        input_query = "What is supervised learning according to my documents?"
        state = agent.process_goal(input_query)

        # 1. Planner verified: retrieve -> respond
        assert len(state.plan) == 2
        assert state.plan[0]["action"] == "retrieve"
        assert state.plan[1]["action"] == "respond"

        # 2. Retriever was called with user query
        mock_retriever.retrieve.assert_called_once_with(input_query)

        # 3. Retrieved context populated in AgentState
        assert state.retrieved_context == sample_rag_chunks

        # 4. Source metadata preserved
        assert "retrieve" in state.tool_results
        retrieve_res = state.tool_results["retrieve"]
        assert retrieve_res["success"] is True
        assert retrieve_res["count"] == len(sample_rag_chunks)
        assert len(retrieve_res["sources"]) == 2

        src0 = retrieve_res["sources"][0]
        assert src0["filename"] == "AI_Notes.txt"
        assert src0["page"] == 3
        assert src0["score"] == 0.945
        assert src0["chunk_id"] == "chunk_sl_001"

        src1 = retrieve_res["sources"][1]
        assert src1["filename"] == "Machine_Learning_Guide.pdf"
        assert src1["page"] == 12
        assert src1["score"] == 0.912
        assert src1["chunk_id"] == "chunk_of_002"

        # 5. Observation recorded
        assert any(f"Retrieved {len(sample_rag_chunks)}" in obs for obs in state.observations)

        # 6. Final response is grounded and citations present
        assert state.status == AgentStatus.COMPLETED
        assert state.final_response is not None
        assert "Supervised learning" in state.final_response
        assert "AI_Notes.txt" in state.final_response
        assert "page 3" in state.final_response

    def test_rag_source_metadata_passed_to_llm(
        self, mock_retriever: MagicMock, sample_rag_chunks: List[Dict[str, Any]]
    ) -> None:
        """Verify mock LLM receives complete chunk metadata through rag_context."""
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "Supervised learning uses labeled pairs (AI_Notes.txt, page 3)."

        agent = AuraAgent(llm=mock_llm, retriever=mock_retriever)
        goal = "What is supervised learning according to my documents?"
        state = agent.process_goal(goal)

        assert state.status == AgentStatus.COMPLETED
        mock_llm.ask.assert_called_once()
        call_kwargs = mock_llm.ask.call_args.kwargs
        assert call_kwargs["rag_context"] == sample_rag_chunks
        assert call_kwargs["rag_context"][0]["metadata"]["filename"] == "AI_Notes.txt"
        assert call_kwargs["rag_context"][0]["metadata"]["page"] == 3


# ==============================================================================
# 5. Test Case — Memory
# ==============================================================================

class TestMemoryIntegration:
    """Validates multi-turn context retention across agent interactions."""

    def test_memory_two_interactions(self) -> None:
        """Perform two interactions:
        1. 'My favorite programming language is Python.'
        2. 'What is my favorite programming language?'
        Verify conversation memory provides the previous context.
        """
        agent = AuraAgent()

        # Turn 1
        turn1_query = "My favorite programming language is Python."
        state1 = agent.process_goal(turn1_query)
        assert state1.status == AgentStatus.COMPLETED
        assert state1.final_response is not None
        assert len(agent.memory) == 2

        # Turn 2
        turn2_query = "What is my favorite programming language?"
        state2 = agent.process_goal(turn2_query)
        assert state2.status == AgentStatus.COMPLETED
        assert state2.final_response is not None
        # Verify the answer recalls Python from memory
        assert "python" in state2.final_response.lower()

        # Verify full 4-turn memory history
        assert len(agent.memory) == 4
        history = agent.memory.get_history()
        assert history[0]["role"] == "user"
        assert history[0]["content"] == turn1_query
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == turn2_query
        assert history[3]["role"] == "assistant"

    def test_memory_context_passed_to_llm(self) -> None:
        """Verify conversation history is passed into LLM call kwargs on turn 2."""
        mock_llm = MagicMock()
        mock_llm.ask.side_effect = [
            "Noted that your favorite language is Python.",
            "Your favorite programming language is Python.",
        ]

        agent = AuraAgent(llm=mock_llm)
        agent.process_goal("My favorite programming language is Python.")
        agent.process_goal("What is my favorite programming language?")

        assert mock_llm.ask.call_count == 2
        second_call_kwargs = mock_llm.ask.call_args_list[1].kwargs

        # Second call must include conversation context with turn 1
        assert "context" in second_call_kwargs
        context = second_call_kwargs["context"]
        assert len(context) == 2
        assert context[0]["content"] == "My favorite programming language is Python."
        assert context[1]["content"] == "Noted that your favorite language is Python."


# ==============================================================================
# 6. Test Case — Multi-Step Task
# ==============================================================================

class TestMultiStepTask:
    """Validates execution of tasks requiring two or more sequential actions."""

    def test_multi_step_rag_task(self, mock_retriever: MagicMock, sample_rag_chunks: List[Dict[str, Any]]) -> None:
        """Goal: 'Find information about overfitting in my documents and explain it.'
        Verify:
        Goal -> Plan -> Retrieve -> Observation -> Respond -> Final Answer
        """
        agent = AuraAgent(retriever=mock_retriever)
        goal = "Find information about overfitting in my documents and explain it."
        state = agent.process_goal(goal)

        # 1. Goal verified
        assert state.user_goal == goal

        # 2. Plan verified (at least 2 actions)
        assert len(state.plan) >= 2
        assert state.plan[0]["action"] == "retrieve"
        assert state.plan[-1]["action"] == "respond"

        # 3. Retrieve verified
        mock_retriever.retrieve.assert_called_once_with(goal)
        assert len(state.retrieved_context) > 0

        # 4. Observations verified
        assert len(state.observations) >= 2
        assert any("Retrieved" in obs for obs in state.observations)
        assert any("Synthesized" in obs for obs in state.observations)

        # 5. Final Answer verified
        assert state.status == AgentStatus.COMPLETED
        assert state.final_response is not None
        assert "overfitting" in state.final_response.lower()

        # 6. Sequential execution integrity
        assert len(state.completed_actions) == 2
        assert state.completed_actions[0]["action"] == "retrieve"
        assert state.completed_actions[1]["action"] == "respond"

    def test_multi_step_custom_chained_actions(self) -> None:
        """Verify custom multi-step pipeline passes intermediate results across actions."""
        agent = AuraAgent()

        # Register custom pipeline actions
        execution_order: List[str] = []

        def step1_action(step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
            execution_order.append("fetch_data")
            return {"success": True, "result": [10, 20, 30]}

        def step2_action(step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
            execution_order.append("aggregate")
            prev = state.tool_results.get("fetch_data", {}).get("result", [])
            return {"success": True, "result": sum(prev)}

        agent.register_action_handler("fetch_data", step1_action)
        agent.register_action_handler("aggregate", step2_action)

        custom_plan = [
            {"step": 1, "action": "fetch_data", "reason": "Fetch data points"},
            {"step": 2, "action": "aggregate", "reason": "Sum data points"},
            {"step": 3, "action": "respond", "reason": "Deliver final response"},
        ]

        agent.state.plan = custom_plan
        agent.state.user_goal = "Fetch points and compute total sum."
        state = agent.execute_plan()

        assert state.status == AgentStatus.COMPLETED
        assert execution_order == ["fetch_data", "aggregate"]
        assert state.tool_results["fetch_data"]["result"] == [10, 20, 30]
        assert state.tool_results["aggregate"]["result"] == 60
        assert len(state.completed_actions) == 3
        assert len(state.observations) >= 3


# ==============================================================================
# 7. Test Case — Tool Failure
# ==============================================================================

class TestToolFailure:
    """Validates safe termination and error recording upon tool failure."""

    def test_mock_tool_failure(self) -> None:
        """Verify:
        - error is captured
        - AgentState records the failure (status == FAILED)
        - Agent does not pretend the tool succeeded
        - task terminates safely
        """
        agent = AuraAgent()

        # Register a failing tool action
        def failing_action(step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
            return {"success": False, "error": "External calculation engine unreachable."}

        agent.register_action_handler("calc_engine", failing_action)

        plan = [
            {"step": 1, "action": "calc_engine", "reason": "Execute remote compute"},
            {"step": 2, "action": "respond", "reason": "Should never be reached"},
        ]
        agent.state.plan = plan
        agent.state.user_goal = "Execute critical calculation."
        state = agent.execute_plan()

        # 1. Error is captured
        assert state.error is not None
        assert "unreachable" in state.error

        # 2. AgentState records failure
        assert state.status == AgentStatus.FAILED

        # 3. Agent does NOT pretend tool succeeded
        assert len(state.completed_actions) == 1
        assert state.completed_actions[0]["action"] == "calc_engine"
        assert state.completed_actions[0]["success"] is False
        assert not any(act.get("action") == "respond" for act in state.completed_actions)

        # 4. Task terminates safely with user-facing message
        assert state.final_response is not None
        assert "error" in state.final_response.lower()
        assert any("failed" in obs.lower() for obs in state.observations)

    def test_tool_exception_crash_prevention(self) -> None:
        """Verify tool raising unexpected runtime exception is handled safely."""
        agent = AuraAgent()

        def exploding_tool(step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
            raise ZeroDivisionError("division by zero in low-level math kernel")

        agent.register_action_handler("kernel_compute", exploding_tool)

        agent.state.plan = [
            {"step": 1, "action": "kernel_compute", "reason": "Perform kernel math"},
            {"step": 2, "action": "respond", "reason": "Synthesize"},
        ]
        agent.state.user_goal = "Run kernel computation"
        state = agent.execute_plan()

        assert state.status == AgentStatus.FAILED
        assert "division by zero" in state.error.lower()
        assert state.completed_actions[0]["success"] is False


# ==============================================================================
# 8. Test Case — Maximum Steps
# ==============================================================================

class TestMaximumSteps:
    """Validates step bounds protection preventing runaway execution loops."""

    def test_plan_exceeding_max_steps_halts_immediately(self) -> None:
        """Verify:
        - Agent stops
        - no infinite loop occurs
        - termination reason is recorded
        """
        max_limit = 3
        agent = AuraAgent(max_steps=max_limit)

        # Plan with 5 steps (exceeding limit of 3)
        oversized_plan = [
            {"step": 1, "action": "calculator", "expression": "1 + 1", "reason": "Step 1"},
            {"step": 2, "action": "calculator", "expression": "2 + 2", "reason": "Step 2"},
            {"step": 3, "action": "calculator", "expression": "3 + 3", "reason": "Step 3"},
            {"step": 4, "action": "calculator", "expression": "4 + 4", "reason": "Step 4"},
            {"step": 5, "action": "respond", "reason": "Step 5"},
        ]
        agent.state.plan = oversized_plan
        agent.state.user_goal = "Run 5-step task"
        state = agent.execute_plan()

        # 1. Agent stops and sets FAILED
        assert state.status == AgentStatus.FAILED

        # 2. No infinite loop occurred; zero steps executed because guard prevented start
        assert len(state.completed_actions) == 0

        # 3. Termination reason is recorded
        assert state.error is not None
        assert "maximum step limit" in state.error.lower()
        assert f"{max_limit}" in state.error
        assert any("maximum step limit" in obs.lower() for obs in state.observations)
        assert state.final_response is not None
        assert "maximum step limit" in state.final_response.lower()

    def test_runtime_step_limit_enforcement(self) -> None:
        """Verify step limit enforces stopping during execution loop."""
        agent = AuraAgent(max_steps=2)
        # 3 steps plan
        agent.state.plan = [
            {"step": 1, "action": "calculator", "expression": "10 * 2", "reason": "Step 1"},
            {"step": 2, "action": "calculator", "expression": "20 * 2", "reason": "Step 2"},
            {"step": 3, "action": "respond", "reason": "Step 3"},
        ]
        agent.state.user_goal = "Run two calcs and respond"
        state = agent.execute_plan()

        assert state.status == AgentStatus.FAILED
        assert state.error is not None
        assert "maximum step limit" in state.error.lower()


# ==============================================================================
# 9. Test Case — Invalid Tool
# ==============================================================================

class TestInvalidTool:
    """Validates defense against unknown tool execution and arbitrary code execution."""

    def test_plan_with_invalid_tool_rejected(self) -> None:
        """Request an unknown tool in plan.
        Verify:
        - tool is rejected
        - no arbitrary execution occurs
        - failure is recorded
        """
        agent = AuraAgent()
        invalid_plan = [
            {
                "step": 1,
                "action": "unauthorized_shell_exec",
                "parameters": {"command": "rm -rf /"},
                "reason": "Malicious attempt",
            },
            {"step": 2, "action": "respond", "reason": "Explain"},
        ]
        agent.state.plan = invalid_plan
        agent.state.user_goal = "Execute unauthorized action"
        state = agent.execute_plan()

        # 1. Tool is rejected
        assert state.status == AgentStatus.FAILED
        assert state.error is not None
        assert "unsupported action" in state.error.lower() or "unknown" in state.error.lower()
        assert "unauthorized_shell_exec" in state.error

        # 2. No arbitrary execution occurs; subsequent steps halted
        assert len(state.completed_actions) == 1
        assert state.completed_actions[0]["action"] == "unauthorized_shell_exec"
        assert state.completed_actions[0]["success"] is False

        # 3. Failure recorded in observations and response
        assert any("unauthorized_shell_exec" in obs for obs in state.observations)
        assert state.final_response is not None
        assert "unsupported action" in state.final_response.lower()

    def test_tool_registry_rejects_unregistered_tool(self) -> None:
        """Verify ToolRegistry rejects unregistered tool requests safely."""
        registry = create_default_registry()
        res = registry.execute("unknown_system_tool", cmd="whoami")

        assert res["success"] is False
        assert "not found" in res["error"].lower()
        assert "available tools" in res["error"].lower()


# ==============================================================================
# 10. Test Case — Path Traversal
# ==============================================================================

class TestPathTraversal:
    """Validates filesystem sandbox boundaries and path traversal rejection."""

    @pytest.fixture
    def safe_dir(self, tmp_path: Path) -> Path:
        """Fixture creating a dedicated sandbox directory."""
        sandbox = tmp_path / "sandbox_data"
        sandbox.mkdir(parents=True, exist_ok=True)
        # Create a legitimate safe file
        (sandbox / "valid_note.txt").write_text("Legitimate sandbox data", encoding="utf-8")
        # Create an outside sensitive file
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("CONFIDENTIAL_SYSTEM_KEY", encoding="utf-8")
        return sandbox

    def test_path_traversal_resolve_safe_path(self, safe_dir: Path) -> None:
        """Verify resolve_safe_path rejects malicious relative and absolute paths."""
        manager = SafeFileManager(base_dir=safe_dir)

        malicious_paths = [
            "../../secret.txt",
            "..\\..\\secret.txt",
            "../../../secret.txt",
            "../outside.txt",
            "subfolder/../../secret.txt",
            "./../secret.txt",
            "..\\..\\windows\\system32",
        ]

        for mal_path in malicious_paths:
            with pytest.raises(PermissionError) as exc_info:
                manager.resolve_safe_path(mal_path)
            assert "outside sandbox" in str(exc_info.value).lower() or "blocked" in str(exc_info.value).lower()

    def test_path_traversal_read_and_write(self, safe_dir: Path) -> None:
        """Verify read_file and write_file safely return failure for path traversal attempts."""
        manager = SafeFileManager(base_dir=safe_dir)

        # Attempt to read outside file
        read_res = manager.read_file("../../secret.txt")
        assert read_res["success"] is False
        assert "blocked" in read_res["error"].lower() or "outside sandbox" in read_res["error"].lower()
        assert read_res["data"] is None

        # Attempt to write outside file
        write_res = manager.write_file("../../malicious.txt", "corrupted content")
        assert write_res["success"] is False
        assert "blocked" in write_res["error"].lower() or "outside sandbox" in write_res["error"].lower()

        # Legitimate file operations succeed
        valid_read = manager.read_file("valid_note.txt")
        assert valid_read["success"] is True
        assert valid_read["data"]["content"] == "Legitimate sandbox data"

    def test_file_tool_interface_rejects_traversal(self, safe_dir: Path) -> None:
        """Verify FileTool interface returns error dictionary when path traversal is attempted."""
        manager = SafeFileManager(base_dir=safe_dir)
        tool = FileTool(manager=manager)

        res = tool.execute(action="read_file", path="../../secret.txt")
        assert res["success"] is False
        assert "blocked" in res["error"].lower() or "outside sandbox" in res["error"].lower()
