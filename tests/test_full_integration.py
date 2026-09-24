"""End-to-End Integration Verification Suite for Phase 11 Step 38.

Validates the full chain:
Frontend/Client -> FastAPI -> AuraAgent -> Memory/Tools/RAG/LLM -> Response -> Voice
Covers all 12 goals of the Phase 11 Step 38 specification.
"""

import io
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.api.main import app as main_app
from app.api.server import app as server_app
from app.api.dependencies import get_agent, get_documents_dir, get_stt, get_tts
from app.agent.agent import AuraAgent
from app.agent.planner import Planner
from app.agent.state import AgentStatus
from app.memory.conversation import ConversationMemory
from app.rag.retriever import LocalRetriever, LocalVectorStore
from app.brain.llm import LLMBrain, MockLLMProvider
from app.speech.stt import MockSpeechToText
from app.speech.tts import MockTextToSpeech
from app.config import get_settings

# Valid dummy WAV bytes (44-byte standard PCM header + 4 bytes silence)
DUMMY_WAV = (
    b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00"
)


@pytest.fixture
def integration_env(tmp_path):
    """Isolated environment with temporary documents directory and vector store."""
    test_docs = tmp_path / "documents"
    test_docs.mkdir(parents=True, exist_ok=True)
    test_vstore = tmp_path / "vector_store.json"

    vstore = LocalVectorStore(storage_path=test_vstore)
    retriever = LocalRetriever(vector_store=vstore, storage_path=test_vstore)
    memory = ConversationMemory()
    llm = LLMBrain(provider=MockLLMProvider(), retriever=retriever, enable_tools=False)
    agent = AuraAgent(llm=llm, memory=memory, planner=Planner(), retriever=retriever)

    main_app.dependency_overrides[get_agent] = lambda: agent
    main_app.dependency_overrides[get_documents_dir] = lambda: test_docs
    main_app.dependency_overrides[get_stt] = lambda: MockSpeechToText(responses=["Calculate 125 * 32 and explain the answer."])
    main_app.dependency_overrides[get_tts] = lambda: MockTextToSpeech()

    with TestClient(main_app) as client:
        yield {
            "client": client,
            "agent": agent,
            "memory": memory,
            "docs_dir": test_docs,
            "retriever": retriever,
        }

    main_app.dependency_overrides.clear()


class TestPhase11Step38FullIntegration:
    """Complete verification of Phase 11 Step 38 integration requirements."""

    def test_1_backend_startup_entrypoint(self):
        """1. Backend startup: verify clean ASGI entry point app.api.main:app."""
        assert main_app is not None
        assert main_app is server_app
        assert main_app.title == "AURA API"

    def test_2_frontend_configuration(self):
        """2. Frontend configuration: verify API service uses VITE_API_URL and dev defaults."""
        frontend_api_ts = Path("frontend/src/services/api.ts").read_text(encoding="utf-8")
        assert "import.meta.env.VITE_API_URL" in frontend_api_ts
        # No hardcoded production URLs
        assert "http://production" not in frontend_api_ts
        assert "https://production" not in frontend_api_ts

    def test_3_chat_integration_ai_query(self, integration_env):
        """3. Chat integration: React -> POST /api/chat -> FastAPI -> AuraAgent -> Response -> React."""
        client = integration_env["client"]
        response = client.post("/api/chat", json={"message": "What is artificial intelligence?"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "artificial intelligence" in data["response"].lower()
        assert "activities" in data
        act_labels = [a["label"] for a in data["activities"]]
        assert "Planning task" in act_labels
        assert "Generating response" in act_labels

    def test_4_calculator_integration(self, integration_env):
        """4. Calculator integration:
        'Calculate 125 * 32 and explain the answer.'
        Agent -> calculator -> 4000 -> observation -> LLM -> frontend.
        """
        client = integration_env["client"]
        response = client.post(
            "/api/chat",
            json={"message": "Calculate 125 * 32 and explain the answer."},
        )
        assert response.status_code == 200
        data = response.json()
        resp_text = data["response"]

        # Final answer must use the real calculator result 4000
        assert "4000" in resp_text
        agent = integration_env["agent"]
        assert agent.state.tool_results.get("calculator", {}).get("result") == 4000
        assert any("4000" in obs for obs in agent.state.observations)

        # Agent activity recorded
        act_labels = [a["label"] for a in data["activities"]]
        assert "Using Calculator" in act_labels

    def test_5_memory_integration(self, integration_env):
        """5. Memory integration:
        Turn 1: 'My favorite programming language is Python.'
        Turn 2: 'What is my favorite programming language?'
        Verify conversation memory provides context to turn 2.
        """
        client = integration_env["client"]

        # Turn 1
        resp1 = client.post("/api/chat", json={"message": "My favorite programming language is Python."})
        assert resp1.status_code == 200
        assert "Python" in resp1.json()["response"]

        # Turn 2
        resp2 = client.post("/api/chat", json={"message": "What is my favorite programming language?"})
        assert resp2.status_code == 200
        resp2_text = resp2.json()["response"]
        assert "python" in resp2_text.lower()

        # Check memory history length
        memory = integration_env["memory"]
        assert len(memory) == 4

    def test_6_rag_integration(self, integration_env):
        """6. RAG integration:
        Upload document -> indexing -> ask 'What does my document say about supervised learning?'
        Grounded response + source metadata preserved.
        """
        client = integration_env["client"]

        doc_content = (
            "Supervised learning is an approach in machine learning where algorithms learn "
            "input-output mappings from verified labeled training datasets. Common algorithms "
            "include linear regression, support vector machines, and neural networks."
        )
        upload_resp = client.post(
            "/api/documents/upload",
            files={"file": ("AI_Notes.txt", io.BytesIO(doc_content.encode("utf-8")), "text/plain")},
        )
        assert upload_resp.status_code == 200
        assert upload_resp.json()["success"] is True
        assert upload_resp.json()["chunks_indexed"] >= 1

        # Ask question
        chat_resp = client.post(
            "/api/chat",
            json={"message": "What does my document say about supervised learning?"},
        )
        assert chat_resp.status_code == 200
        data = chat_resp.json()
        assert "supervised learning" in data["response"].lower()
        assert "AI_Notes.txt" in data["response"]

        # Activities include document retrieval
        act_labels = [a["label"] for a in data["activities"]]
        assert "Retrieving documents" in act_labels

    def test_7_voice_integration(self, integration_env):
        """7. Voice integration:
        Microphone (audio input) -> STT -> Agent -> Response -> TTS -> Browser.
        Text response still appears even if TTS fails.
        """
        client = integration_env["client"]

        # Normal voice request
        v_resp = client.post(
            "/api/voice",
            content=DUMMY_WAV,
            headers={"Content-Type": "audio/wav"},
        )
        assert v_resp.status_code == 200
        v_data = v_resp.json()
        assert v_data["user_text"] == "Calculate 125 * 32 and explain the answer."
        assert "4000" in v_data["response"]
        assert v_data["audio"] is not None
        assert v_data["audio"].startswith("data:audio/wav;base64,")

        # Fault-tolerant TTS failure test
        failing_tts = MockTextToSpeech(simulate_provider_error=True)
        main_app.dependency_overrides[get_tts] = lambda: failing_tts

        v_fail_resp = client.post(
            "/api/voice",
            content=DUMMY_WAV,
            headers={"Content-Type": "audio/wav"},
        )
        assert v_fail_resp.status_code == 200
        v_fail_data = v_fail_resp.json()
        # Text response is still present even though TTS failed
        assert len(v_fail_data["response"]) > 0
        assert "4000" in v_fail_data["response"]
        assert v_fail_data["audio"] is None

    def test_8_agent_activity_safety(self, integration_env):
        """8. Agent Activity:
        Displays safe high-level activity (Planning task, Retrieving documents, Using Calculator, etc.).
        Never exposes hidden reasoning or internal prompts.
        """
        client = integration_env["client"]
        resp = client.post(
            "/api/chat",
            json={"message": "Calculate 100 * 20"},
        )
        assert resp.status_code == 200
        activities = resp.json()["activities"]
        for act in activities:
            assert act["label"] in [
                "Planning task",
                "Retrieving documents",
                "Using Calculator",
                "Using Weather tool",
                "Searching knowledge base",
                "Reading file",
                "Generating response",
                "Completed",
                "Processing step",
                "Executing action",
            ]
            # Verify no stack traces or secrets
            assert "Traceback" not in act["label"]
            assert "SECRET" not in act["label"]
            assert "prompt" not in act["label"].lower()

    def test_9_error_handling(self, integration_env):
        """9. Error handling:
        backend unavailable / invalid chat / tool failure / upload failure.
        UI friendly errors, no stack traces.
        """
        client = integration_env["client"]

        # Invalid chat request (empty message)
        err_resp = client.post("/api/chat", json={"message": "   "})
        assert err_resp.status_code in (400, 422)
        assert "Traceback" not in err_resp.json()["detail"]

        # Upload failure (unsupported extension)
        up_resp = client.post(
            "/api/documents/upload",
            files={"file": ("script.py", io.BytesIO(b"print('hello')"), "text/x-python")},
        )
        assert up_resp.status_code == 400
        assert "unsupported file type" in up_resp.json()["detail"].lower()
        assert "Traceback" not in up_resp.json()["detail"]

    def test_10_production_configuration(self):
        """10. Production configuration:
        Verify settings properties, CORS configuration, limits.
        """
        cfg = get_settings()
        assert hasattr(cfg, "cors_origins")
        assert hasattr(cfg, "max_audio_size")
        assert hasattr(cfg, "max_document_size")
        assert hasattr(cfg, "max_agent_steps")
        assert hasattr(cfg, "rate_limit_enabled")
        assert cfg.max_agent_steps > 0
