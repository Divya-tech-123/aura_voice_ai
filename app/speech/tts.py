"""Text-to-Speech (TTS) interface and implementations for AURA.

Responsible for converting textual responses from AURA's cognitive Brain
into audible speech output via local or cloud synthesis engines.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Custom Exceptions
# ==============================================================================

class TTSError(Exception):
    """Base exception for all Text-to-Speech errors in AURA."""
    pass


class AudioDeviceError(TTSError):
    """Raised when audio output hardware / speakers are missing, disabled, or failed."""
    pass


class TTSProviderError(TTSError):
    """Raised when a TTS provider fails during speech synthesis or initialization."""
    pass


# ==============================================================================
# 2. Base Interface
# ==============================================================================

class BaseTextToSpeech(ABC):
    """Abstract base class / interface for TTS providers."""

    @abstractmethod
    def speak(self, text: str) -> bool:
        """Synthesize and play the provided text through audio speakers.

        Args:
            text: The text string to be spoken.

        Returns:
            True if text was synthesized and played successfully, False otherwise.

        Raises:
            AudioDeviceError: If output audio hardware is missing or inaccessible.
            TTSProviderError: If synthesis fails.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop any ongoing speech playback immediately."""
        pass

    @abstractmethod
    def synthesize_to_bytes(self, text: str) -> Optional[bytes]:
        """Synthesize text into playable audio bytes (WAV format).

        Args:
            text: The message string to be synthesized.

        Returns:
            WAV audio bytes if successful, None if text was empty.

        Raises:
            AudioDeviceError: If audio hardware is missing or inaccessible.
            TTSProviderError: If speech synthesis fails.
        """
        pass


# ==============================================================================
# 3. Mock TTS Provider (Offline / Testing / Headless)
# ==============================================================================

class MockTextToSpeech(BaseTextToSpeech):
    """Deterministic mock TTS provider for unit testing and CI pipelines."""

    def __init__(
        self,
        simulate_device_error: bool = False,
        simulate_provider_error: bool = False,
    ) -> None:
        """Initialize mock TTS provider with controllable test behaviors.

        Args:
            simulate_device_error: If True, raises AudioDeviceError on speak.
            simulate_provider_error: If True, raises TTSProviderError on speak.
        """
        self.simulate_device_error = simulate_device_error
        self.simulate_provider_error = simulate_provider_error
        self.spoken_texts: List[str] = []

    def speak(self, text: str) -> bool:
        """Record the spoken text in memory without playing actual sound."""
        if self.simulate_device_error:
            raise AudioDeviceError("Simulated audio output device error: Speaker not detected.")

        if self.simulate_provider_error:
            raise TTSProviderError("Simulated TTS provider error: Synthesis pipeline failed.")

        clean_text = (text or "").strip()
        if not clean_text:
            logger.debug("[Mock TTS] Received empty or whitespace-only text; skipping.")
            return False

        logger.info("[TTS] Speaking response")
        self.spoken_texts.append(clean_text)
        logger.debug(f"[Mock TTS] Spoke: \"{clean_text}\"")
        return True

    def synthesize_to_bytes(self, text: str) -> Optional[bytes]:
        """Synthesize text into playable audio bytes without hardware."""
        if self.simulate_device_error:
            raise AudioDeviceError("Simulated audio output device error: Speaker not detected.")

        if self.simulate_provider_error:
            raise TTSProviderError("Simulated TTS provider error: Synthesis pipeline failed.")

        clean_text = (text or "").strip()
        if not clean_text:
            logger.debug("[Mock TTS] Received empty or whitespace-only text; skipping synthesis.")
            return None

        self.spoken_texts.append(clean_text)
        logger.info("[TTS] Synthesizing speech to audio bytes")
        # Return minimal valid 44-byte standard PCM WAV header + 4 bytes silence
        return (
            b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00"
        )

    def stop(self) -> None:
        """Simulate stopping speech."""
        logger.debug("[Mock TTS] Speech stopped.")


# ==============================================================================
# 4. Production Local Pyttsx3 TTS Provider
# ==============================================================================

class Pyttsx3TextToSpeech(BaseTextToSpeech):
    """Offline, local Text-to-Speech provider using the `pyttsx3` library.

    Uses native system speech synthesizers (SAPI5 on Windows, NSSpeechSynthesizer on macOS,
    eSpeak on Linux) with zero external API calls or latency.
    """

    def __init__(
        self,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        voice_id: Optional[str] = None,
    ) -> None:
        """Initialize the pyttsx3 speech engine.

        Args:
            rate: Speech speed in words per minute (default 180).
            volume: Volume float between 0.0 and 1.0 (default 1.0).
            voice_id: Optional exact voice ID or name substring (e.g. 'David', 'Zira').
        """
        self.rate = rate if rate is not None else int(os.getenv("TTS_VOICE_RATE", "180"))
        try:
            self.volume = volume if volume is not None else float(os.getenv("TTS_VOLUME", "1.0"))
        except ValueError:
            self.volume = 1.0
        self.voice_id = voice_id or os.getenv("TTS_VOICE_ID", "").strip() or None

        self._engine = None
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize and configure the pyttsx3 engine safely."""
        try:
            import pyttsx3
        except ImportError as exc:
            raise TTSProviderError(
                "pyttsx3 is not installed. Install with `pip install pyttsx3`."
            ) from exc

        try:
            self._engine = pyttsx3.init()
        except Exception as exc:
            raise AudioDeviceError(
                f"Failed to initialize pyttsx3 audio driver / output device: {exc}"
            ) from exc

        # Configure voice rate
        try:
            self._engine.setProperty("rate", self.rate)
        except Exception as exc:
            logger.warning(f"Could not set TTS rate to {self.rate}: {exc}")

        # Configure volume
        try:
            self._engine.setProperty("volume", max(0.0, min(1.0, self.volume)))
        except Exception as exc:
            logger.warning(f"Could not set TTS volume to {self.volume}: {exc}")

        # Configure voice if specified
        if self.voice_id:
            self._select_voice(self.voice_id)

    def _select_voice(self, target_voice: str) -> None:
        """Select a voice matching the target ID or name substring."""
        if not self._engine:
            return

        try:
            voices = self._engine.getProperty("voices") or []
            target_lower = target_voice.lower()
            for v in voices:
                v_id = getattr(v, "id", "")
                v_name = getattr(v, "name", "")
                if target_lower in v_id.lower() or target_lower in v_name.lower():
                    self._engine.setProperty("voice", v_id)
                    logger.info(f"Selected TTS Voice: {v_name} ({v_id})")
                    return
            logger.warning(f"Voice matching '{target_voice}' not found among available system voices.")
        except Exception as exc:
            logger.warning(f"Error selecting TTS voice '{target_voice}': {exc}")

    def speak(self, text: str) -> bool:
        """Synthesize text and play audio through speaker."""
        clean_text = (text or "").strip()
        if not clean_text:
            return False

        if self._engine is None:
            self._init_engine()

        logger.info("[TTS] Speaking response")
        try:
            logger.debug(f"[pyttsx3] Synthesizing: \"{clean_text[:60]}...\"")
            self._engine.say(clean_text)
            self._engine.runAndWait()
            return True
        except Exception as exc:
            logger.error(f"Pyttsx3 TTS synthesis error: {exc}")
            # Try to recover engine state if an interruption occurred
            try:
                self._engine.endLoop()
            except Exception:
                pass
            raise TTSProviderError(f"TTS synthesis failed: {exc}") from exc

    def stop(self) -> None:
        """Stop current speech synthesis."""
        if self._engine:
            try:
                self._engine.stop()
            except Exception as exc:
                logger.debug(f"pyttsx3 stop call: {exc}")

    def synthesize_to_bytes(self, text: str) -> Optional[bytes]:
        """Synthesize text into playable WAV audio bytes using pyttsx3."""
        clean_text = (text or "").strip()
        if not clean_text:
            return None

        if self._engine is None:
            self._init_engine()

        import tempfile
        logger.info("[TTS] Synthesizing speech to audio bytes")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._engine.save_to_file(clean_text, tmp_path)
            self._engine.runAndWait()
            with open(tmp_path, "rb") as f:
                data = f.read()
            return data
        except Exception as exc:
            logger.error(f"Pyttsx3 TTS synthesis error: {exc}")
            raise TTSProviderError(f"TTS synthesis failed: {exc}") from exc
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


# ==============================================================================
# 5. Factory & Abstraction Interface
# ==============================================================================

def get_tts_provider(provider_type: Optional[str] = None, **kwargs) -> BaseTextToSpeech:
    """Factory to instantiate the configured TTS provider.

    Args:
        provider_type: Optional provider name ('pyttsx3', 'local', 'mock').
                       Defaults to TTS_PROVIDER or TTS_ENGINE environment variables.
        **kwargs: Additional parameters forwarded to provider constructor.

    Returns:
        Concrete BaseTextToSpeech instance.
    """
    raw_provider = provider_type or os.getenv("TTS_PROVIDER") or os.getenv("TTS_ENGINE") or "pyttsx3"
    provider_name = raw_provider.strip().lower()

    if provider_name == "mock":
        return MockTextToSpeech(**kwargs)

    return Pyttsx3TextToSpeech(**kwargs)


# Default singleton instance for top-level speak() calls
_default_tts_instance: Optional[BaseTextToSpeech] = None


def speak(text: str, provider: Optional[BaseTextToSpeech] = None) -> bool:
    """Top-level decoupled TTS abstraction for AURA.

    Synthesizes and speaks text using the active TTS provider.
    Fails gracefully if audio hardware or the provider encounters an error,
    ensuring that speech failures never crash the assistant.

    Args:
        text: The message string to be spoken.
        provider: Optional explicit BaseTextToSpeech provider instance.

    Returns:
        True if spoken successfully, False if skipped or failed.

    Example:
        ```python
        from app.speech.tts import speak

        speak("Hello, I am AURA.")
        ```
    """
    global _default_tts_instance
    clean_text = (text or "").strip()
    if not clean_text:
        return False

    active_provider = provider or _default_tts_instance
    if active_provider is None:
        try:
            active_provider = get_tts_provider()
            if provider is None:
                _default_tts_instance = active_provider
        except Exception as exc:
            logger.warning(f"Could not initialize default TTS provider: {exc}")
            return False

    try:
        return active_provider.speak(clean_text)
    except (AudioDeviceError, TTSProviderError, TTSError) as exc:
        logger.error(f"Failed to speak response: {exc}")
        return False
    except Exception as exc:
        logger.error(f"Unexpected error during speech playback: {exc}")
        return False


def synthesize_speech(text: str, provider: Optional[BaseTextToSpeech] = None) -> Optional[bytes]:
    """Top-level decoupled TTS synthesis abstraction for AURA.

    Synthesizes text into playable audio bytes using the active TTS provider.
    Fails gracefully returning None if speech synthesis encounters an error,
    ensuring that speech failures never crash the assistant.

    Args:
        text: The message string to be synthesized.
        provider: Optional explicit BaseTextToSpeech provider instance.

    Returns:
        Audio bytes if synthesized successfully, None if skipped or failed.
    """
    global _default_tts_instance
    clean_text = (text or "").strip()
    if not clean_text:
        return None

    active_provider = provider or _default_tts_instance
    if active_provider is None:
        try:
            active_provider = get_tts_provider()
            if provider is None:
                _default_tts_instance = active_provider
        except Exception as exc:
            logger.warning(f"Could not initialize default TTS provider: {exc}")
            return None

    try:
        return active_provider.synthesize_to_bytes(clean_text)
    except (AudioDeviceError, TTSProviderError, TTSError) as exc:
        logger.error(f"Failed to synthesize speech: {exc}")
        return None
    except Exception as exc:
        logger.error(f"Unexpected error during speech synthesis: {exc}")
        return None

