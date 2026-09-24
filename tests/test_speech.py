"""Unit and integration tests for AURA Phase 7 Voice Subsystem.

Tests STT (Speech-to-Text), TTS (Text-to-Speech), hardware error handling,
audio device degradation, safe exit mechanics, and the end-to-end voice loop.
All tests use mock hardware and synthetic audio providers—no real microphone
or external network API is required.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from app.speech.stt import (
    BaseSpeechToText,
    MockSpeechToText,
    SpeechRecognitionSTT,
    get_stt_provider,
    speech_to_text,
    SpeechError,
    MicrophoneError,
    NoSpeechDetectedError,
    SpeechRecognitionError,
)
from app.speech.tts import (
    BaseTextToSpeech,
    MockTextToSpeech,
    Pyttsx3TextToSpeech,
    get_tts_provider,
    speak,
    TTSError,
    AudioDeviceError,
    TTSProviderError,
)
from app.speech.voice_loop import run_voice_loop, is_exit_command
from app.brain.intent import IntentClassifier
from app.brain.llm import LLMBrain
from app.memory.conversation import ConversationMemory


# ==============================================================================
# 1. Speech-to-Text (STT) Unit Tests
# ==============================================================================

class TestMockSpeechToText:
    """Tests for MockSpeechToText provider behavior and error simulation."""

    def test_stt_successful_transcription(self):
        """Verify mock STT returns configured transcriptions and records history."""
        provider = MockSpeechToText(responses=["What is artificial intelligence?"])
        result = provider.listen_and_transcribe()
        assert result == "What is artificial intelligence?"
        assert provider.transcription_history == ["What is artificial intelligence?"]

    def test_stt_multiple_responses_sequence(self):
        """Verify sequential utterances are popped and returned in order."""
        provider = MockSpeechToText(responses=["Hello", "How are you?", "Goodbye"])
        assert provider.listen_and_transcribe() == "Hello"
        assert provider.listen_and_transcribe() == "How are you?"
        assert provider.listen_and_transcribe() == "Goodbye"
        assert len(provider.transcription_history) == 3

    def test_stt_no_speech_detected_error(self):
        """Verify NoSpeechDetectedError is raised when simulate_no_speech is set."""
        provider = MockSpeechToText(simulate_no_speech=True)
        with pytest.raises(NoSpeechDetectedError) as exc_info:
            provider.listen_and_transcribe()
        assert "No speech was detected" in str(exc_info.value)

    def test_stt_empty_queue_raises_no_speech(self):
        """Verify NoSpeechDetectedError is raised when the responses list is exhausted."""
        provider = MockSpeechToText(responses=[])
        with pytest.raises(NoSpeechDetectedError):
            provider.listen_and_transcribe()

    def test_stt_recognition_failure(self):
        """Verify SpeechRecognitionError is raised when speech is unintelligible."""
        provider = MockSpeechToText(simulate_recognition_error=True)
        with pytest.raises(SpeechRecognitionError) as exc_info:
            provider.listen_and_transcribe()
        assert "unintelligible" in str(exc_info.value).lower()

    def test_stt_microphone_error(self):
        """Verify MicrophoneError is raised when hardware/driver failure occurs."""
        provider = MockSpeechToText(simulate_microphone_error=True)
        with pytest.raises(MicrophoneError) as exc_info:
            provider.listen_and_transcribe()
        assert "microphone hardware" in str(exc_info.value).lower()


class TestSpeechRecognitionSTTWithMocks:
    """Tests for the production SpeechRecognitionSTT provider using mocked hardware."""

    def test_speech_recognition_success(self):
        """Verify SpeechRecognitionSTT properly invokes microphone and recognizer."""
        with patch("speech_recognition.Recognizer") as mock_rec_cls, \
             patch("speech_recognition.Microphone") as mock_mic_cls:

            mock_rec = MagicMock()
            mock_rec.listen.return_value = MagicMock()
            mock_rec.recognize_google.return_value = "What is deep learning?"
            mock_rec_cls.return_value = mock_rec

            mock_mic = MagicMock()
            mock_mic.__enter__.return_value = MagicMock()
            mock_mic.__exit__.return_value = None
            mock_mic_cls.return_value = mock_mic

            stt = SpeechRecognitionSTT(engine="google")
            result = stt.listen_and_transcribe()

            assert result == "What is deep learning?"
            mock_rec.adjust_for_ambient_noise.assert_called_once()
            mock_rec.listen.assert_called_once()
            mock_rec.recognize_google.assert_called_once()

    def test_speech_recognition_timeout_raises_no_speech(self):
        """Verify sr.WaitTimeoutError is mapped to NoSpeechDetectedError."""
        import speech_recognition as sr
        with patch("speech_recognition.Recognizer") as mock_rec_cls, \
             patch("speech_recognition.Microphone") as mock_mic_cls:

            mock_rec = MagicMock()
            mock_rec.listen.side_effect = sr.WaitTimeoutError("Listening timed out")
            mock_rec_cls.return_value = mock_rec

            mock_mic = MagicMock()
            mock_mic.__enter__.return_value = MagicMock()
            mock_mic.__exit__.return_value = None
            mock_mic_cls.return_value = mock_mic

            stt = SpeechRecognitionSTT()
            with pytest.raises(NoSpeechDetectedError):
                stt.listen_and_transcribe()

    def test_speech_recognition_unknown_value_raises_recognition_error(self):
        """Verify sr.UnknownValueError is mapped to SpeechRecognitionError."""
        import speech_recognition as sr
        with patch("speech_recognition.Recognizer") as mock_rec_cls, \
             patch("speech_recognition.Microphone") as mock_mic_cls:

            mock_rec = MagicMock()
            mock_rec.recognize_google.side_effect = sr.UnknownValueError()
            mock_rec_cls.return_value = mock_rec

            mock_mic = MagicMock()
            mock_mic.__enter__.return_value = MagicMock()
            mock_mic.__exit__.return_value = None
            mock_mic_cls.return_value = mock_mic

            stt = SpeechRecognitionSTT()
            with pytest.raises(SpeechRecognitionError):
                stt.listen_and_transcribe()

    def test_speech_recognition_mic_os_error_raises_microphone_error(self):
        """Verify hardware OSError is wrapped in MicrophoneError."""
        with patch("speech_recognition.Microphone") as mock_mic_cls:
            mock_mic_cls.side_effect = OSError("No default input device found")

            stt = SpeechRecognitionSTT()
            with pytest.raises(MicrophoneError) as exc_info:
                stt.listen_and_transcribe()
            assert "microphone" in str(exc_info.value).lower()


class TestSTTFactoryAndHelper:
    """Tests for get_stt_provider factory and speech_to_text top-level helper."""

    def test_get_stt_provider_mock(self):
        """Verify factory returns MockSpeechToText when requested."""
        provider = get_stt_provider("mock", responses=["Test speech"])
        assert isinstance(provider, MockSpeechToText)
        assert provider.listen_and_transcribe() == "Test speech"

    def test_speech_to_text_helper_success(self):
        """Verify speech_to_text helper returns string from provided provider."""
        mock_p = MockSpeechToText(responses=["Testing top level"])
        result = speech_to_text(provider=mock_p)
        assert result == "Testing top level"

    def test_speech_to_text_helper_silence_returns_none(self):
        """Verify speech_to_text helper handles silence gracefully by returning None."""
        mock_p = MockSpeechToText(simulate_no_speech=True)
        result = speech_to_text(provider=mock_p)
        assert result is None


# ==============================================================================
# 2. Text-to-Speech (TTS) Unit Tests
# ==============================================================================

class TestMockTextToSpeech:
    """Tests for MockTextToSpeech provider and audio playback recording."""

    def test_tts_valid_text(self):
        """Verify valid text is recorded in spoken_texts history."""
        tts = MockTextToSpeech()
        assert tts.speak("Hello from AURA.") is True
        assert tts.spoken_texts == ["Hello from AURA."]

    def test_tts_empty_text_returns_false(self):
        """Verify empty and whitespace strings return False and are not spoken."""
        tts = MockTextToSpeech()
        assert tts.speak("") is False
        assert tts.speak("   ") is False
        assert len(tts.spoken_texts) == 0

    def test_tts_provider_failure(self):
        """Verify TTSProviderError is raised when synthesis fails."""
        tts = MockTextToSpeech(simulate_provider_error=True)
        with pytest.raises(TTSProviderError):
            tts.speak("Should fail")

    def test_tts_device_error(self):
        """Verify AudioDeviceError is raised on simulated hardware missing."""
        tts = MockTextToSpeech(simulate_device_error=True)
        with pytest.raises(AudioDeviceError):
            tts.speak("Should fail with device error")

    def test_tts_stop_call(self):
        """Verify stop call executes cleanly."""
        tts = MockTextToSpeech()
        tts.stop()  # Should not raise


class TestPyttsx3WithMocks:
    """Tests for Pyttsx3TextToSpeech wrapper with mocked pyttsx3 engine."""

    def test_pyttsx3_success_calls_engine(self):
        """Verify pyttsx3 calls say and runAndWait with sanitized text."""
        with patch("pyttsx3.init") as mock_init:
            mock_engine = MagicMock()
            mock_init.return_value = mock_engine

            tts = Pyttsx3TextToSpeech(rate=175, volume=0.9)
            success = tts.speak("AURA is speaking.")

            assert success is True
            mock_engine.say.assert_called_once_with("AURA is speaking.")
            mock_engine.runAndWait.assert_called_once()
            mock_engine.setProperty.assert_any_call("rate", 175)
            mock_engine.setProperty.assert_any_call("volume", 0.9)

    def test_pyttsx3_empty_text_skips_engine(self):
        """Verify empty text does not invoke speech engine."""
        with patch("pyttsx3.init") as mock_init:
            mock_engine = MagicMock()
            mock_init.return_value = mock_engine

            tts = Pyttsx3TextToSpeech()
            result = tts.speak("   \n\t  ")

            assert result is False
            mock_engine.say.assert_not_called()
            mock_engine.runAndWait.assert_not_called()

    def test_pyttsx3_init_failure_raises_device_error(self):
        """Verify driver init failure raises AudioDeviceError."""
        with patch("pyttsx3.init", side_effect=Exception("COM library not initialized")):
            with pytest.raises(AudioDeviceError):
                Pyttsx3TextToSpeech()


class TestTTSFactoryAndHelper:
    """Tests for get_tts_provider factory and speak top-level helper."""

    def test_get_tts_provider_mock(self):
        """Verify factory returns MockTextToSpeech when requested."""
        provider = get_tts_provider("mock")
        assert isinstance(provider, MockTextToSpeech)

    def test_speak_helper_valid_text(self):
        """Verify speak helper speaks text successfully via provided mock."""
        mock_p = MockTextToSpeech()
        result = speak("Hello world", provider=mock_p)
        assert result is True
        assert mock_p.spoken_texts == ["Hello world"]

    def test_speak_helper_empty_text(self):
        """Verify speak helper returns False on empty string."""
        mock_p = MockTextToSpeech()
        assert speak("", provider=mock_p) is False
        assert len(mock_p.spoken_texts) == 0

    def test_speak_helper_graceful_failure_never_crashes(self):
        """Verify speak helper returns False instead of crashing on provider errors."""
        mock_p = MockTextToSpeech(simulate_device_error=True)
        result = speak("This will error", provider=mock_p)
        assert result is False  # Graceful recovery!


# ==============================================================================
# 3. Integration Tests (STT -> Brain -> Memory -> TTS)
# ==============================================================================

class TestVoiceIntegrationAndLoop:
    """End-to-end integration tests connecting Voice STT, Brain, Memory, and TTS."""

    @pytest.fixture
    def setup_pipeline(self):
        """Prepare core pipeline components for integration testing."""
        classifier = IntentClassifier()
        brain = LLMBrain(enable_tools=False, enable_rag=False)
        memory = ConversationMemory(max_messages=10)
        return classifier, brain, memory

    def test_voice_loop_single_turn_and_goodbye(self, setup_pipeline):
        """Test complete single dialogue turn: STT -> Brain -> Response -> TTS -> Safe Exit."""
        classifier, brain, memory = setup_pipeline

        # Scripted user conversation:
        # Turn 1: "Explain what machine learning is."
        # Turn 2: "Goodbye"
        stt = MockSpeechToText(responses=[
            "Explain what machine learning is.",
            "Goodbye",
        ])
        tts = MockTextToSpeech()

        run_voice_loop(
            classifier=classifier,
            brain=brain,
            memory=memory,
            stt=stt,
            tts=tts,
        )

        # Verify STT was invoked twice
        assert len(stt.transcription_history) == 2
        assert stt.transcription_history[0] == "Explain what machine learning is."
        assert stt.transcription_history[1] == "Goodbye"

        # Verify TTS spoke both the answer and the farewell
        assert len(tts.spoken_texts) == 2
        assert "machine learning" in tts.spoken_texts[0].lower()
        assert "goodbye" in tts.spoken_texts[1].lower()

        # Verify memory recorded user and assistant turns
        history = memory.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Explain what machine learning is."
        assert history[1]["role"] == "assistant"

    def test_voice_loop_multi_turn_with_memory_context(self, setup_pipeline):
        """Verify conversation context persists across multiple spoken voice turns."""
        classifier, brain, memory = setup_pipeline

        stt = MockSpeechToText(responses=[
            "Hello AURA",
            "What can you do?",
            "quit",
        ])
        tts = MockTextToSpeech()

        run_voice_loop(
            classifier=classifier,
            brain=brain,
            memory=memory,
            stt=stt,
            tts=tts,
        )

        assert len(stt.transcription_history) == 3
        # 2 answers + 1 farewell
        assert len(tts.spoken_texts) == 3
        assert "goodbye" in tts.spoken_texts[-1].lower()

        # Check memory has both conversational turns
        history = memory.get_history()
        assert len(history) == 4  # (user, assistant) x 2

    def test_voice_loop_handles_no_speech_and_recovers(self, setup_pipeline):
        """Verify that silence / timeout does not crash the loop and the assistant recovers."""
        classifier, brain, memory = setup_pipeline

        # Simulate:
        # Turn 1: None (silence / timeout)
        # Turn 2: "Hello"
        # Turn 3: "exit"
        stt = MockSpeechToText(responses=[
            None,
            "Hello",
            "exit",
        ])
        tts = MockTextToSpeech()

        run_voice_loop(
            classifier=classifier,
            brain=brain,
            memory=memory,
            stt=stt,
            tts=tts,
        )

        # Answered "Hello" and farewell on "exit"
        assert len(tts.spoken_texts) == 2
        assert "goodbye" in tts.spoken_texts[-1].lower()

    def test_voice_loop_handles_mic_error_cleanly(self, setup_pipeline):
        """Verify that a catastrophic microphone error terminates gracefully without crashing."""
        classifier, brain, memory = setup_pipeline

        stt = MockSpeechToText(simulate_microphone_error=True)
        tts = MockTextToSpeech()

        # Should log error and exit without raising an uncaught exception
        run_voice_loop(
            classifier=classifier,
            brain=brain,
            memory=memory,
            stt=stt,
            tts=tts,
        )

        assert len(tts.spoken_texts) == 0

    def test_is_exit_command_helper(self):
        """Verify the exit command recognition logic."""
        assert is_exit_command("goodbye") is True
        assert is_exit_command("Exit.") is True
        assert is_exit_command("quit!") is True
        assert is_exit_command("bye") is True
        assert is_exit_command("Goodbye AURA") is True
        assert is_exit_command("What is machine learning?") is False
        assert is_exit_command("Can you calculate 2 + 2?") is False
