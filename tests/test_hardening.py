"""Comprehensive Tests for Production Backend & API Hardening (Phase 11 - Step 37).

Tests:
1. Centralized Configuration (Settings, defaults, production validation, wildcard CORS rejection)
2. Request Validation (Valid requests, missing fields, empty message, oversized message, oversized payload)
3. Safe Error Handling (No stack traces, no internal paths, no API keys, structured error format)
4. Health Endpoint (GET /api/health, status ok, version 0.11.0, no secret leakage)
5. CORS Configuration (Development origins, strict production origins, no wildcard in prod)
6. Upload Security Review (Path traversal, unsafe characters, disguised executables, oversized uploads)
7. Rate Limiting Architecture (Sliding window limits, HTTP 429, Retry-After header, exemptions, toggle)
8. Logging Redaction (Masking of API keys, tokens, passwords, and base64 audio data)
"""

import io
import os
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings, set_settings
from app.api.server import create_app, get_cors_origins
from app.api.dependencies import get_agent, get_documents_dir, get_stt, get_tts
from app.agent.state import AgentState, AgentStatus
from app.api.rate_limiter import InMemoryRateLimiter, RateLimitMiddleware
from app.api.logging_config import redact_sensitive_text, SensitiveDataFilter
from app.api.errors import sanitize_error_detail
from app.speech.stt import MockSpeechToText
from app.speech.tts import MockTextToSpeech


# Standard dummy WAV bytes for voice testing
DUMMY_WAV = (
    b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00"
)


@pytest.fixture
def mock_agent():
    """Mock AuraAgent for fast API testing."""
    agent = MagicMock()
    state = AgentState()
    state.user_goal = "Test query"
    state.final_response = "AURA test response"
    state.status = AgentStatus.COMPLETED
    agent.process_goal.return_value = state
    return agent


@pytest.fixture
def test_client(mock_agent, tmp_path):
    """TestClient with isolated settings and mocked external services."""
    test_docs = tmp_path / "documents"
    test_docs.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        app_env="testing",
        api_host="127.0.0.1",
        api_port=8000,
        cors_origins=["http://localhost:5173", "http://localhost:3000"],
        llm_provider="mock",
        max_content_length=64 * 1024,
        max_audio_size=10 * 1024 * 1024,
        max_document_size=10 * 1024 * 1024,
        max_agent_steps=10,
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=100,
        documents_dir=test_docs,
    )

    app = create_app(settings=settings)
    app.dependency_overrides[get_agent] = lambda: mock_agent
    app.dependency_overrides[get_documents_dir] = lambda: test_docs
    app.dependency_overrides[get_stt] = lambda: MockSpeechToText(responses=["Test speech"])
    app.dependency_overrides[get_tts] = lambda: MockTextToSpeech()

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


# ==============================================================================
# 1. Configuration Tests
# ==============================================================================

class TestCentralizedConfiguration:
    """Verify centralized environment variable loading and security constraints."""

    def test_default_settings_are_valid(self):
        """Settings initializes with safe development defaults."""
        cfg = Settings()
        assert cfg.app_env in ("development", "production", "testing")
        assert cfg.api_port > 0
        assert cfg.max_content_length == 64 * 1024
        assert cfg.max_document_size == 10 * 1024 * 1024
        assert cfg.max_agent_steps == 10
        assert cfg.rate_limit_enabled is True
        assert len(cfg.cors_origins) > 0

    def test_production_cors_wildcard_rejected(self, monkeypatch):
        """In production mode, wildcard '*' in CORS origins is strictly stripped."""
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("CORS_ORIGINS", "*,https://aura.ai")

        cfg = Settings()
        assert "*" not in cfg.cors_origins
        assert "https://aura.ai" in cfg.cors_origins

    def test_settings_custom_overrides(self):
        """Custom settings override defaults correctly."""
        cfg = Settings(
            app_env="production",
            api_host="0.0.0.0",
            api_port=9000,
            cors_origins=["https://aura.example.com"],
            max_agent_steps=5,
            rate_limit_requests_per_minute=50,
        )
        assert cfg.is_production is True
        assert cfg.api_host == "0.0.0.0"
        assert cfg.api_port == 9000
        assert cfg.cors_origins == ["https://aura.example.com"]
        assert cfg.max_agent_steps == 5
        assert cfg.rate_limit_requests_per_minute == 50


# ==============================================================================
# 2. Health Endpoint Tests
# ==============================================================================

class TestHealthEndpointHardening:
    """Verify GET /api/health meets Step 37 requirements."""

    def test_health_returns_status_and_version(self, test_client):
        """GET /api/health returns status 'ok' and version '0.11.0' without leaking secrets."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.11.0"
        assert "api_key" not in data
        assert "secret" not in data
        assert "password" not in data
        assert "env" not in data


# ==============================================================================
# 3. Request Validation Tests
# ==============================================================================

class TestRequestValidation:
    """Verify strict Pydantic model validation on API endpoints."""

    def test_valid_chat_request(self, test_client):
        """Valid chat payload succeeds with 200 OK."""
        response = test_client.post("/api/chat", json={"message": "Hello AURA"})
        assert response.status_code == 200
        assert "response" in response.json()

    def test_empty_message_rejected(self, test_client):
        """Empty string message rejected with 400 or 422."""
        response = test_client.post("/api/chat", json={"message": ""})
        assert response.status_code in (400, 422)
        assert "error" in response.json() or "detail" in response.json()

    def test_whitespace_message_rejected(self, test_client):
        """Whitespace-only message rejected."""
        response = test_client.post("/api/chat", json={"message": "   \n\t  "})
        assert response.status_code in (400, 422)

    def test_missing_message_field_rejected(self, test_client):
        """Missing required message field returns 422."""
        response = test_client.post("/api/chat", json={})
        assert response.status_code == 422
        data = response.json()
        assert "error" in data or "detail" in data

    def test_oversized_message_rejected(self, test_client):
        """Message exceeding 4096 characters rejected with 422."""
        response = test_client.post("/api/chat", json={"message": "X" * 5000})
        assert response.status_code == 422

    def test_oversized_request_payload_rejected(self, test_client):
        """Payload exceeding 64KB rejected with 413 or 422."""
        response = test_client.post("/api/chat", json={"message": "Y" * (70 * 1024)})
        assert response.status_code in (413, 422)


# ==============================================================================
# 4. Safe Error Handling Tests
# ==============================================================================

class TestSafeErrorHandling:
    """Verify errors never leak stack traces, internal paths, or API keys."""

    def test_uncaught_exception_returns_safe_500(self, test_client):
        """Internal server error returns generic message without stack trace or secrets."""
        failing_agent = MagicMock()
        failing_agent.process_goal.side_effect = RuntimeError(
            "Fatal failure in C:\\Users\\Admin\\aura\\secret.py with API_KEY=sk-test1234567890abcdef"
        )
        test_client.app.dependency_overrides[get_agent] = lambda: failing_agent

        response = test_client.post("/api/chat", json={"message": "Trigger failure"})
        assert response.status_code == 500
        data = response.json()

        # Both 'error' and 'detail' are safe
        assert data.get("error") == "Unable to process the request."
        # Verify no secrets or sensitive data leaked
        resp_str = response.text
        assert "Traceback" not in resp_str
        assert "sk-test" not in resp_str
        assert "secret.py" not in resp_str
        assert "C:\\Users" not in resp_str

    def test_sanitize_error_detail_redacts_paths_and_keys(self):
        """sanitize_error_detail cleans paths, tracebacks, and API keys."""
        dirty = (
            "Error occurred at C:\\Users\\Harshitha\\app.py with sk-abcdef1234567890123456 "
            "Traceback (most recent call last): line 42"
        )
        cleaned = sanitize_error_detail(dirty)
        assert "C:\\Users" not in cleaned
        assert "sk-abcdef" not in cleaned
        assert "Traceback" not in cleaned
        assert "[internal_path]" in cleaned or "***REDACTED***" in cleaned


# ==============================================================================
# 5. CORS Configuration Tests
# ==============================================================================

class TestCORSConfigurationHardening:
    """Verify CORS handles development and production environments correctly."""

    def test_cors_preflight_allowed(self, test_client):
        """Preflight OPTIONS request from allowed origin returns CORS headers."""
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        response = test_client.options("/api/chat", headers=headers)
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_production_no_wildcard(self):
        """Production configuration strictly disallows wildcard '*'."""
        prod_settings = Settings(
            app_env="production",
            cors_origins=["*", "https://aura.mydomain.com"],
        )
        origins = get_cors_origins(prod_settings)
        assert "*" not in origins
        assert "https://aura.mydomain.com" in origins


# ==============================================================================
# 6. Upload Security Review Tests
# ==============================================================================

class TestUploadSecurityHardening:
    """Verify document upload security, sanitization, and size limits."""

    def test_path_traversal_upload_rejected(self, test_client):
        """Filenames containing path traversal patterns are rejected with HTTP 400."""
        traversal_names = [
            "../../etc/passwd.txt",
            "..\\..\\boot.ini.txt",
            "subdir/notes.txt",
            "subdir\\notes.txt",
        ]
        for name in traversal_names:
            response = test_client.post(
                "/api/documents/upload",
                files={"file": (name, io.BytesIO(b"Valid content"), "text/plain")},
            )
            assert response.status_code == 400
            data = response.json()
            assert "traversal" in str(data).lower() or "invalid filename" in str(data).lower()

    def test_disguised_executable_rejected(self, test_client):
        """Filenames with disguised executable extensions are rejected."""
        disguised_names = [
            "exploit.exe.txt",
            "script.sh.txt",
            "backdoor.py.txt",
            "payload.bat.txt",
        ]
        for name in disguised_names:
            response = test_client.post(
                "/api/documents/upload",
                files={"file": (name, io.BytesIO(b"malicious script"), "text/plain")},
            )
            assert response.status_code == 400
            data = response.json()
            assert "executable" in str(data).lower() or "invalid filename" in str(data).lower()

    def test_unsupported_extension_rejected(self, test_client):
        """Non-document extensions (.sh, .exe, .py, .zip) are rejected with 400."""
        response = test_client.post(
            "/api/documents/upload",
            files={"file": ("malware.exe", io.BytesIO(b"\x4d\x5a"), "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "unsupported file type" in str(response.json()).lower()

    def test_oversized_upload_rejected(self, test_client):
        """Files exceeding size limit are rejected with 413."""
        with patch("app.api.routes.MAX_DOCUMENT_SIZE", 300):
            response = test_client.post(
                "/api/documents/upload",
                files={"file": ("big.txt", io.BytesIO(b"Z" * 500), "text/plain")},
            )
            assert response.status_code == 413


# ==============================================================================
# 7. Rate Limiting Architecture Tests
# ==============================================================================

class TestRateLimiterArchitecture:
    """Verify sliding-window in-memory rate limiter."""

    def test_rate_limiter_allows_under_limit(self):
        """Requests under limit are allowed with accurate remaining counts."""
        limiter = InMemoryRateLimiter(requests_per_minute=5, window_seconds=60.0)
        allowed, remaining, retry_after = limiter.check("192.168.1.1")
        assert allowed is True
        assert remaining == 4
        assert retry_after == 0.0

    def test_rate_limiter_blocks_over_limit(self):
        """Requests exceeding limit are blocked with retry_after > 0."""
        limiter = InMemoryRateLimiter(requests_per_minute=3, window_seconds=60.0)
        for _ in range(3):
            allowed, _, _ = limiter.check("10.0.0.1")
            assert allowed is True

        # 4th request must be blocked
        allowed, remaining, retry_after = limiter.check("10.0.0.1")
        assert allowed is False
        assert remaining == 0
        assert retry_after > 0.0

    def test_rate_limit_middleware_returns_429(self, mock_agent, tmp_path):
        """Client hitting rate limit receives HTTP 429 and Retry-After header."""
        test_docs = tmp_path / "documents"
        test_docs.mkdir(parents=True, exist_ok=True)

        # Create rate limiter with very small limit
        limiter = InMemoryRateLimiter(requests_per_minute=2, window_seconds=60.0)
        settings = Settings(
            app_env="testing",
            rate_limit_enabled=True,
            documents_dir=test_docs,
        )
        app = create_app(settings=settings)
        app.dependency_overrides[get_agent] = lambda: mock_agent
        app.dependency_overrides[get_documents_dir] = lambda: test_docs

        # Re-attach limiter with small limit to middleware
        for middleware in app.user_middleware:
            if middleware.cls == RateLimitMiddleware:
                middleware.kwargs["limiter"] = limiter

        with TestClient(app) as client:
            r1 = client.post("/api/chat", json={"message": "Query 1"})
            r2 = client.post("/api/chat", json={"message": "Query 2"})
            r3 = client.post("/api/chat", json={"message": "Query 3"})

            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r3.status_code == 429
            assert "Retry-After" in r3.headers
            assert "rate limit exceeded" in str(r3.json()).lower()

            # Health endpoint is exempt from rate limit
            r_health = client.get("/api/health")
            assert r_health.status_code == 200


# ==============================================================================
# 8. Logging Redaction Tests
# ==============================================================================

class TestLoggingRedaction:
    """Verify secrets and sensitive data are redacted from logs."""

    def test_redact_sensitive_text_masks_keys_and_tokens(self):
        """API keys, passwords, and tokens are masked with ***REDACTED***."""
        text = (
            "Connecting to OpenAI with api_key=sk-1234567890abcdef1234567890 "
            "and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz "
            "and password='SuperSecretPassword123'"
        )
        redacted = redact_sensitive_text(text)
        assert "sk-1234567890" not in redacted
        assert "SuperSecretPassword123" not in redacted
        assert "***REDACTED***" in redacted

    def test_redact_base64_audio(self):
        """Large base64 audio data URI is replaced with redacted placeholder."""
        audio_log = "Processing audio payload data:audio/wav;base64,UklGRiQAAABXQVZlZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
        redacted = redact_sensitive_text(audio_log)
        assert "[REDACTED_AUDIO]" in redacted
        assert "UklGRiQAAABXQVZl" not in redacted

    def test_sensitive_data_filter(self):
        """SensitiveDataFilter modifies LogRecord messages to redact secrets."""
        filt = SensitiveDataFilter()
        record = logging.LogRecord(
            name="AURA.Test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="User authenticated with api_key=my_secret_token_12345",
            args=(),
            exc_info=None,
        )
        filt.filter(record)
        assert "my_secret_token_12345" not in record.msg
        assert "***REDACTED***" in record.msg
