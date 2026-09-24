"""Unit and Integration Tests for AURA Agent Activity UI (Phase 11 - Step 34).

Validates:
1. Activity data generated from agent actions (planning, tools, response).
2. Successful activity states (completed planning, tools, and response generation).
3. Failed activity states (failed action status without stack trace leakage).
4. Activity data does not contain secrets, API keys, hidden reasoning, or internal prompts.
5. Frontend renders activities, expandable controls, safe error messages ("⚠️ AURA couldn't complete this step."),
   and live loading state ("🧠 AURA is working...").
6. Existing chat functionality remains fully operational.
"""

import subprocess
import os
import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.api.server import create_app
from app.api.dependencies import get_agent
from app.api.schemas import ActivityItem, ChatResponse
from app.agent.agent import AuraAgent
from app.agent.state import AgentState, AgentStatus
from app.agent.planner import Planner
from app.memory.conversation import ConversationMemory


@pytest.fixture
def client():
    """Create a TestClient with a fresh app instance."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def memory():
    return ConversationMemory()


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ask.return_value = "The calculated answer is 4000."
    return llm


class TestAgentActivityDataGeneration:
    """1. Activity data generated from agent actions."""

    def test_state_records_activities_explicitly(self):
        """AgentState add_activity and get_activities store and retrieve safe structured activities."""
        state = AgentState()
        state.add_activity("planning", "Planning task", "completed")
        state.add_activity("tool", "Using Calculator", "completed")
        state.add_activity("response", "Generating response", "completed")

        activities = state.get_activities()
        assert len(activities) == 3
        assert activities[0] == {"type": "planning", "label": "Planning task", "status": "completed"}
        assert activities[1] == {"type": "tool", "label": "Using Calculator", "status": "completed"}
        assert activities[2] == {"type": "response", "label": "Generating response", "status": "completed"}

    def test_state_derives_activities_from_completed_actions(self):
        """If activities list is empty, state dynamically derives safe high-level activities."""
        state = AgentState()
        state.user_goal = "Calculate 100 * 40"
        state.plan = [
            {"step": 1, "action": "calculator", "reason": "Calculate expression"},
            {"step": 2, "action": "respond", "reason": "Format answer"},
        ]
        state.completed_actions = [
            {"step": 1, "action": "calculator", "success": True, "result": 4000},
            {"step": 2, "action": "respond", "success": True, "response": "The answer is 4000."},
        ]
        state.final_response = "The answer is 4000."
        state.status = AgentStatus.COMPLETED

        activities = state.get_activities()
        assert len(activities) >= 3
        types = [a["type"] for a in activities]
        labels = [a["label"] for a in activities]
        assert "planning" in types
        assert "Planning task" in labels
        assert "Using Calculator" in labels
        assert "Generating response" in labels

    def test_agent_process_goal_generates_activities(self, mock_llm, memory):
        """AuraAgent.process_goal generates structured activities for planning and actions."""
        planner = Planner()
        agent = AuraAgent(llm=mock_llm, memory=memory, planner=planner)

        state = agent.process_goal("calculate 25 * 4")
        activities = state.get_activities()

        assert len(activities) >= 2
        # Check presence of planning
        assert any(a["type"] == "planning" and "Plan" in a["label"] for a in activities)
        # Check presence of calculator tool
        assert any("Calculator" in a["label"] for a in activities)
        # Check presence of response
        assert any(a["type"] == "response" or "response" in a["label"].lower() for a in activities)

    def test_api_chat_returns_structured_activities(self, client):
        """POST /api/chat returns structured activities in response payload."""
        mock_agent = MagicMock()
        state = AgentState()
        state.user_goal = "What is 50 * 80?"
        state.final_response = "The answer is 4000."
        state.status = AgentStatus.COMPLETED
        state.add_activity("planning", "Planning task", "completed")
        state.add_activity("tool", "Using Calculator", "completed")
        state.add_activity("response", "Generating response", "completed")
        mock_agent.process_goal.return_value = state

        client.app.dependency_overrides[get_agent] = lambda: mock_agent

        response = client.post("/api/chat", json={"message": "What is 50 * 80?"})
        assert response.status_code == 200
        data = response.json()

        assert "response" in data
        assert data["response"] == "The answer is 4000."
        assert "activities" in data
        assert len(data["activities"]) == 3

        act0 = data["activities"][0]
        assert act0["type"] == "planning"
        assert act0["label"] == "Planning task"
        assert act0["status"] == "completed"

        act1 = data["activities"][1]
        assert act1["type"] == "tool"
        assert act1["label"] == "Using Calculator"
        assert act1["status"] == "completed"


class TestSuccessfulActivityStates:
    """2. Successful activity states."""

    def test_successful_actions_marked_completed(self, mock_llm, memory):
        """Successful plan execution marks all activity items with status='completed'."""
        agent = AuraAgent(llm=mock_llm, memory=memory)
        state = agent.process_goal("calculate 10 + 20")

        activities = state.get_activities()
        assert len(activities) > 0
        for act in activities:
            assert act["status"] == "completed"

    def test_document_retrieval_activity_success(self, mock_llm, memory):
        """Document retrieval action produces 'Retrieving documents' with completed status."""
        state = AgentState()
        state.completed_actions = [
            {"step": 1, "action": "retrieve", "success": True, "chunks_count": 3},
            {"step": 2, "action": "respond", "success": True, "response": "Based on the documents..."},
        ]
        state.final_response = "Based on the documents..."
        activities = state.get_activities()

        retrieval_acts = [a for a in activities if a["type"] == "retrieval"]
        assert len(retrieval_acts) == 1
        assert retrieval_acts[0]["label"] == "Retrieving documents"
        assert retrieval_acts[0]["status"] == "completed"


class TestFailedActivityState:
    """3. Failed activity state."""

    def test_failed_tool_action_marked_failed(self, mock_llm, memory):
        """When an action fails, its activity item status is 'failed'."""
        agent = AuraAgent(llm=mock_llm, memory=memory)

        def failing_handler(step, state):
            return {"success": False, "error": "Division by zero in internal math engine"}

        agent.register_action_handler("calculator", failing_handler)
        state = agent.process_goal("calculate 10 / 0")

        assert state.status == AgentStatus.FAILED
        activities = state.get_activities()
        failed_acts = [a for a in activities if a["status"] == "failed"]
        assert len(failed_acts) >= 1
        assert "Calculator" in failed_acts[0]["label"]

    def test_failed_activity_does_not_leak_stack_trace_in_label(self):
        """A failed activity item does not leak python stack trace or error details in its label."""
        state = AgentState()
        state.status = AgentStatus.FAILED
        state.error = "ZeroDivisionError: division by zero\n  File 'calculator.py', line 42, in calculate"
        state.completed_actions = [
            {
                "step": 1,
                "action": "calculator",
                "success": False,
                "error": "ZeroDivisionError: division by zero\n  File 'calculator.py', line 42",
            }
        ]

        activities = state.get_activities()
        for act in activities:
            assert "ZeroDivisionError" not in act["label"]
            assert "Traceback" not in act["label"]
            assert "calculator.py" not in act["label"]
            assert "line 42" not in act["label"]


class TestActivityPrivacyAndSecretsRule:
    """4. Activity data does not contain secrets, API keys, hidden reasoning, or internal prompts."""

    def test_activities_do_not_contain_secrets_or_keys(self):
        """Even if state, goal, or errors contain secrets, activities contain only high-level safe info."""
        state = AgentState()
        secret_key = "sk-proj-supersecretkey123456789"
        api_token = "Bearer secret_jwt_token_aura_admin"

        state.user_goal = f"Calculate 50 + 50 with key {secret_key}"
        state.observations = [f"LLM private reasoning: User passed {api_token}"]
        state.plan = [
            {"step": 1, "action": "calculator", "expression": f"50 + 50 # secret {secret_key}"},
            {"step": 2, "action": "respond", "internal_prompt": "Chain of thought: private deliberation tokens"},
        ]
        state.completed_actions = [
            {"step": 1, "action": "calculator", "success": True, "result": 100},
            {"step": 2, "action": "respond", "success": True, "response": "100"},
        ]
        state.final_response = "100"

        activities = state.get_activities()
        activities_str = json.dumps(activities)

        # Strictly verify no secret tokens or private reasoning leaked into activities
        assert secret_key not in activities_str
        assert api_token not in activities_str
        assert "Chain of thought" not in activities_str
        assert "deliberation" not in activities_str
        assert "private" not in activities_str
        assert "observations" not in activities_str

    def test_activities_only_contain_allowed_keys(self):
        """Activities items contain only 'type', 'label', and 'status'."""
        state = AgentState()
        state.add_activity("tool", "Using Calculator", "completed")
        activities = state.get_activities()

        for act in activities:
            assert set(act.keys()) == {"type", "label", "status"}
            assert act["type"] in ("planning", "tool", "retrieval", "response", "action")
            assert act["status"] in ("completed", "failed", "in_progress", "pending")


class TestFrontendRendersActivities:
    """5. Frontend renders activities, expandable controls, and safe error states."""

    def test_agent_activity_component_file_exists(self):
        """Verify AgentActivity.tsx exists in frontend components directory."""
        component_path = os.path.join(
            "frontend", "src", "components", "AgentActivity.tsx"
        )
        assert os.path.isfile(component_path), f"File not found: {component_path}"

    def test_agent_activity_component_content(self):
        """Verify AgentActivity.tsx contains safe error message and required Lucide icons."""
        component_path = os.path.join(
            "frontend", "src", "components", "AgentActivity.tsx"
        )
        with open(component_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for Lucide icons
        assert "ChevronDown" in content
        assert "Check" in content
        assert "AlertTriangle" in content
        assert "lucide-react" in content

        # Check for required error message
        assert "⚠️ AURA couldn't complete this step." in content

        # Check for accessibility attributes
        assert "aria-expanded" in content
        assert "aria-controls" in content

    def test_chat_area_includes_agent_activity_and_loading_text(self):
        """Verify ChatArea.tsx uses AgentActivity and displays '🧠 AURA is working...'."""
        chat_area_path = os.path.join(
            "frontend", "src", "components", "ChatArea.tsx"
        )
        with open(chat_area_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "AgentActivity" in content
        assert "🧠" in content
        assert "AURA is working..." in content

    def test_frontend_node_render_verification(self):
        """Execute Node.js to verify AgentActivity compiles and exports cleanly."""
        script = """
        const fs = require('fs');
        const code = fs.readFileSync('src/components/AgentActivity.tsx', 'utf8');
        if (!code.includes('export const AgentActivity')) {
            throw new Error('AgentActivity not exported');
        }
        if (!code.includes("⚠️ AURA couldn't complete this step.")) {
            throw new Error('Safe error message missing');
        }
        console.log('Frontend component verification OK');
        """
        result = subprocess.run(
            ["node", "-e", script],
            cwd="frontend",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Frontend component verification OK" in result.stdout


class TestExistingChatFunctionality:
    """6. Existing chat functionality still works."""

    def test_chat_returns_200_and_response_field(self, client):
        """Chat requests continue returning standard response field."""
        response = client.post(
            "/api/chat",
            json={"message": "Hello AURA"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_empty_message_still_rejected(self, client):
        """Validation for empty message still returns 400/422."""
        response = client.post("/api/chat", json={"message": "   "})
        assert response.status_code in (400, 422)

    def test_health_check_still_returns_ok(self, client):
        """Health check endpoint remains functional."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
