"""Tests for AURA Voice API Subsystem (Phase 11 - Step 33).

Validates:
- Valid voice audio requests (binary, multipart, JSON base64)
- Successful voice pipeline (STT -> Agent -> TTS)
- Empty audio rejection (400 Bad Request)
- Unsupported audio format rejection (415 Unsupported Media Type)
- Payload size limit enforcement for audio (413 Payload Too Large)
- STT failure handling without leaking stack traces
- Agent failure handling without leaking internal error details
- Graceful TTS failure recovery (text response preserved, audio is None)
- Pure mocked tests with zero hardware or physical microphone dependencies
"""

import io
import base64
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.api.server import create_app
from app.api.dependencies import get_agent, get_stt, get_tts
from app.agent.state import AgentState, AgentStatus
from app.speech.stt import (
    MockSpeechToText,
    SpeechRecognitionError,
    NoSpeechDetectedError,
    AudioDecodeError,
    SpeechServiceUnavailableError,
    convert_audio_to_wav,
    validate_and_inspect_wav,
)
from app.speech.tts import MockTextToSpeech, TTSProviderError, AudioDeviceError


# Minimal valid 44-byte standard PCM WAV header + 4 bytes audio silence
DUMMY_WAV_BYTES = (
    b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00"
)


@pytest.fixture
def client():
    """Create a TestClient with a fresh app instance."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_agent():
    """Create a mock AuraAgent returning a valid response."""
    agent = MagicMock()
    state = AgentState()
    state.user_goal = "What is machine learning?"
    state.final_response = "Machine learning is a subset of artificial intelligence."
    state.status = AgentStatus.COMPLETED
    agent.process_goal.return_value = state
    return agent


@pytest.fixture
def mock_stt():
    """Create a mock STT provider."""
    return MockSpeechToText(responses=["What is machine learning?"])


@pytest.fixture
def mock_tts():
    """Create a mock TTS provider."""
    return MockTextToSpeech()


class TestVoicePipelineSuccess:
    """Tests for successful end-to-end voice pipeline execution."""

    def test_voice_binary_wav_success(self, client, mock_agent, mock_stt, mock_tts):
        """Verify binary audio payload is transcribed, processed by agent, and synthesized."""
        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: mock_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        response = client.post(
            "/api/voice",
            content=DUMMY_WAV_BYTES,
            headers={"Content-Type": "audio/wav"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_text"] == "What is machine learning?"
        assert data["response"] == "Machine learning is a subset of artificial intelligence."
        assert data["audio"] is not None
        assert data["audio"].startswith("data:audio/wav;base64,")
        assert data["audio_format"] == "audio/wav"

        mock_agent.process_goal.assert_called_once_with("What is machine learning?")
        assert len(mock_tts.spoken_texts) == 1
        assert "machine learning" in mock_tts.spoken_texts[0].lower()

    def test_voice_multipart_form_data_success(self, client, mock_agent, mock_stt, mock_tts):
        """Verify multipart/form-data upload succeeds."""
        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: mock_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        files = {"audio": ("sample.wav", io.BytesIO(DUMMY_WAV_BYTES), "audio/wav")}
        response = client.post("/api/voice", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["user_text"] == "What is machine learning?"
        assert "subset of artificial intelligence" in data["response"]
        assert data["audio"] is not None

    def test_voice_json_base64_success(self, client, mock_agent, mock_stt, mock_tts):
        """Verify base64 audio in JSON payload succeeds."""
        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: mock_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        b64_str = base64.b64encode(DUMMY_WAV_BYTES).decode("utf-8")
        payload = {"audio": f"data:audio/wav;base64,{b64_str}", "format": "audio/wav"}

        response = client.post("/api/voice", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["user_text"] == "What is machine learning?"
        assert data["audio"] is not None

    def test_voice_binary_webm_opus_success(self, client, mock_agent, mock_stt, mock_tts):
        """Verify WebM/Opus audio (as produced by browser MediaRecorder) is accepted and processed."""
        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: mock_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        # Sample WebM bytes
        dummy_webm = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08"
        response = client.post(
            "/api/voice",
            content=dummy_webm,
            headers={"Content-Type": "audio/webm;codecs=opus"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_text"] == "What is machine learning?"
        assert data["response"] == "Machine learning is a subset of artificial intelligence."
        assert data["audio"] is not None



class TestVoiceInputValidations:
    """Tests for input validation, security constraints, and error handling."""

    def test_empty_audio_rejected(self, client, mock_agent, mock_stt, mock_tts):
        """Empty audio payload should be rejected with 400 Bad Request."""
        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: mock_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        response = client.post(
            "/api/voice",
            content=b"",
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code == 400
        assert "empty or missing" in response.json()["detail"].lower()

    def test_unsupported_audio_format_rejected(self, client, mock_agent, mock_stt, mock_tts):
        """Unsupported content types (e.g. text/plain, image/png) should be rejected with 415."""
        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: mock_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        response = client.post(
            "/api/voice",
            content=b"some text pretending to be audio",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code in (415, 400)
        assert "unsupported audio format" in response.json()["detail"].lower()

    def test_audio_exceeding_size_limit_rejected(self, client):
        """Audio payload exceeding 10 MB should be rejected with 413."""
        huge_audio = b"\x00" * (11 * 1024 * 1024)  # 11 MB
        response = client.post(
            "/api/voice",
            content=huge_audio,
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code in (413, 422)


class TestVoiceFailureHandling:
    """Tests for subsystem failures and fault-tolerant degradation."""

    def test_stt_recognition_failure_handled_gracefully(self, client, mock_agent, mock_tts):
        """When STT cannot recognize speech, return 400 without leaking stack traces."""
        failing_stt = MockSpeechToText(simulate_recognition_error=True)
        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: failing_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        response = client.post(
            "/api/voice",
            content=DUMMY_WAV_BYTES,
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "traceback" not in data["detail"].lower()
        assert "understand speech" in data["detail"].lower()

    def test_stt_no_speech_detected_handled_gracefully(self, client, mock_agent, mock_tts):
        """When no speech is detected in audio, return 400."""
        failing_stt = MockSpeechToText(simulate_no_speech=True)
        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: failing_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        response = client.post(
            "/api/voice",
            content=DUMMY_WAV_BYTES,
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code == 400

    def test_agent_failure_handled_gracefully(self, client, mock_stt, mock_tts):
        """When agent encounters internal error, return 500 without stack trace leaks."""
        failing_agent = MagicMock()
        failing_agent.process_goal.side_effect = RuntimeError("Internal DB connection error with API_KEY=xyz")

        client.app.dependency_overrides[get_agent] = lambda: failing_agent
        client.app.dependency_overrides[get_stt] = lambda: mock_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        response = client.post(
            "/api/voice",
            content=DUMMY_WAV_BYTES,
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "API_KEY" not in data["detail"]
        assert "traceback" not in data["detail"].lower()

    def test_tts_failure_preserves_text_response(self, client, mock_agent, mock_stt):
        """When TTS fails, endpoint returns 200 with text response and audio set to None."""
        failing_tts = MockTextToSpeech(simulate_provider_error=True)

        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: mock_stt
        client.app.dependency_overrides[get_tts] = lambda: failing_tts

        response = client.post(
            "/api/voice",
            content=DUMMY_WAV_BYTES,
            headers={"Content-Type": "audio/wav"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_text"] == "What is machine learning?"
        assert data["response"] == "Machine learning is a subset of artificial intelligence."
        # Audio is gracefully None when TTS fails
        assert data["audio"] is None

    def test_audio_decode_error_handled_gracefully(self, client, mock_agent, mock_tts):
        """When STT raises AudioDecodeError (e.g. corrupt WAV), return 400 Bad Request with decode explanation."""
        failing_stt = MagicMock()
        failing_stt.transcribe_audio_data.side_effect = AudioDecodeError("Could not parse WAV structure: file does not start with RIFF id")

        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: failing_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        response = client.post(
            "/api/voice",
            content=b"not a valid wav file at all",
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "could not be decoded" in data["detail"].lower()

    def test_speech_service_unavailable_handled_gracefully(self, client, mock_agent, mock_tts):
        """When STT raises SpeechServiceUnavailableError (e.g. external API timeout), return 502 Bad Gateway."""
        failing_stt = MagicMock()
        failing_stt.transcribe_audio_data.side_effect = SpeechServiceUnavailableError("recognition connection reset by peer")

        client.app.dependency_overrides[get_agent] = lambda: mock_agent
        client.app.dependency_overrides[get_stt] = lambda: failing_stt
        client.app.dependency_overrides[get_tts] = lambda: mock_tts

        response = client.post(
            "/api/voice",
            content=DUMMY_WAV_BYTES,
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code == 502
        data = response.json()
        assert "detail" in data
        assert "speech recognition service error" in data["detail"].lower()


class TestAudioUtilities:
    """Unit tests for audio validation and transcoding functions."""

    def test_validate_and_inspect_wav_valid(self):
        """Valid dummy WAV is recognized and inspected correctly."""
        info = validate_and_inspect_wav(DUMMY_WAV_BYTES)
        assert info["channels"] == 1
        assert info["sample_rate"] == 44100
        assert info["sample_width"] == 2
        assert info["n_frames"] == 2

    def test_validate_and_inspect_wav_corrupt(self):
        """Corrupt audio payload raises AudioDecodeError."""
        with pytest.raises(AudioDecodeError, match="too small|RIFF"):
            validate_and_inspect_wav(b"RIFFjunkdata")

    def test_convert_audio_to_wav_passes_wav(self):
        """Native WAV payload passes through without transcoding."""
        wav_out, mime = convert_audio_to_wav(DUMMY_WAV_BYTES, "audio/wav")
        assert wav_out == DUMMY_WAV_BYTES
        assert mime == "audio/wav"

    def test_convert_audio_to_wav_empty_raises(self):
        """Empty payload raises NoSpeechDetectedError."""
        with pytest.raises(NoSpeechDetectedError):
            convert_audio_to_wav(b"")

