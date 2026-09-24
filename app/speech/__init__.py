"""Speech subsystem for AURA.

Handles Speech-to-Text (STT) audio input processing and
Text-to-Speech (TTS) audio synthesis output.
"""

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
from app.speech.voice_loop import run_voice_loop

__all__ = [
    # STT
    "BaseSpeechToText",
    "MockSpeechToText",
    "SpeechRecognitionSTT",
    "get_stt_provider",
    "speech_to_text",
    "SpeechError",
    "MicrophoneError",
    "NoSpeechDetectedError",
    "SpeechRecognitionError",
    # TTS
    "BaseTextToSpeech",
    "MockTextToSpeech",
    "Pyttsx3TextToSpeech",
    "get_tts_provider",
    "speak",
    "TTSError",
    "AudioDeviceError",
    "TTSProviderError",
    # Voice Loop
    "run_voice_loop",
]
