"""Unit and Integration Tests for AURA REST API Layer (Phase 11 - Step 32).

Validates:
- Valid chat request handling
- Empty and whitespace-only message rejection
- Request payload size limit enforcement
- Successful AuraAgent response formatting
- Agent error and exception handling (graceful 500 without stack trace leaks)
- Health check endpoint
- CORS headers and preflight handling
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.api.server import create_app
from app.api.dependencies import get_agent
from app.agent.state import AgentState, AgentStatus


@pytest.fixture
def client():
    """Create a TestClient with a fresh app instance."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    # Clear overrides after each test
    app.dependency_overrides.clear()


@pytest.fixture
def mock_agent():
    """Create a mock AuraAgent."""
    agent = MagicMock()
    state = AgentState()
    state.user_goal = "Test query"
    state.final_response = "Artificial intelligence is the simulation of human intelligence by machines."
    state.status = AgentStatus.COMPLETED
    agent.process_goal.return_value = state
    return agent


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_check_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "AURA"


class TestChatEndpointValidations:
    """Tests for input validation and constraints on POST /api/chat."""

    def test_empty_message_rejected(self, client):
        """Empty string should be rejected with 400 or 422."""
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code in (400, 422)

    def test_whitespace_only_message_rejected(self, client):
        """Whitespace-only string should be rejected."""
        response = client.post("/api/chat", json={"message": "    \n\t  "})
        assert response.status_code in (400, 422)

    def test_missing_message_field_rejected(self, client):
        """Payload without 'message' field should be rejected."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_message_exceeding_max_length_rejected(self, client):
        """Messages larger than 4096 characters should be rejected."""
        huge_message = "A" * 5000
        response = client.post("/api/chat", json={"message": huge_message})
        assert response.status_code == 422

    def test_request_body_exceeding_size_limit_rejected(self, client):
        """Payloads exceeding 64KB should trigger request size limit."""
        huge_payload = {"message": "A" * (70 * 1024)}
        response = client.post("/api/chat", json=huge_payload)
        assert response.status_code in (413, 422)


class TestChatEndpointAgentIntegration:
    """Tests for POST /api/chat agent execution and responses."""

    def test_successful_agent_response(self, client, mock_agent):
        """Chat request returns agent's response when agent succeeds."""
        client.app.dependency_overrides[get_agent] = lambda: mock_agent

        response = client.post(
            "/api/chat",
            json={"message": "What is artificial intelligence?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "simulation of human intelligence" in data["response"]
        mock_agent.process_goal.assert_called_once_with("What is artificial intelligence?")

    def test_agent_exception_handled_gracefully(self, client):
        """When agent raises an unexpected exception, return safe 500 without stack trace."""
        failing_agent = MagicMock()
        failing_agent.process_goal.side_effect = RuntimeError("Critical internal database crash with SECRET_KEY=12345")
        client.app.dependency_overrides[get_agent] = lambda: failing_agent

        response = client.post(
            "/api/chat",
            json={"message": "Trigger unexpected agent crash"},
        )
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        # Ensure sensitive information and stack traces are NOT leaked
        assert "SECRET_KEY" not in data["detail"]
        assert "Traceback" not in data["detail"]
        assert "RuntimeError" not in data["detail"]

    def test_agent_failure_state_handled(self, client):
        """When agent status is FAILED and final_response is empty, return 500."""
        failing_agent = MagicMock()
        failed_state = AgentState()
        failed_state.status = AgentStatus.FAILED
        failed_state.error = "Agent could not complete plan."
        failed_state.final_response = None
        failing_agent.process_goal.return_value = failed_state
        client.app.dependency_overrides[get_agent] = lambda: failing_agent

        response = client.post(
            "/api/chat",
            json={"message": "Test agent failure"},
        )
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_real_agent_chat_smoke(self, client):
        """End-to-end smoke test using the default AuraAgent with mock LLM."""
        response = client.post(
            "/api/chat",
            json={"message": "Hello AURA, can you hear me?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0


class TestCORSConfiguration:
    """Tests for development CORS headers."""

    def test_cors_preflight_allowed_origin(self, client):
        """OPTIONS preflight from allowed origin returns CORS headers."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        response = client.options("/api/chat", headers=headers)
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert "POST" in response.headers.get("access-control-allow-methods", "")

    def test_cors_headers_on_chat_response(self, client, mock_agent):
        """Regular POST request includes allowed origin."""
        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        headers = {"Origin": "http://localhost:5173"}
        response = client.post(
            "/api/chat",
            json={"message": "Hello CORS"},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
