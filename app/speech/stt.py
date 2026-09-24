"""Speech-to-Text (STT) interface and implementations for AURA.

Provides an extensible, decoupled speech recognition subsystem capable of capturing
audio from the microphone, performing ambient noise cancellation, enforcing timeout
and recording limits, and transcribing speech to clean text.
"""

import os
import io
import math
import struct
import shutil
import subprocess
import wave
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Custom Exceptions
# ==============================================================================

class SpeechError(Exception):
    """Base exception for all speech subsystem errors in AURA."""
    pass


class MicrophoneError(SpeechError):
    """Raised when the microphone is unavailable, missing, or encounters hardware/driver errors."""
    pass


class NoSpeechDetectedError(SpeechError):
    """Raised when listening times out because no speech was detected."""
    pass


class SpeechRecognitionError(SpeechError):
    """Raised when audio was captured but could not be understood or transcribed."""
    pass


class AudioDecodeError(SpeechRecognitionError):
    """Raised when audio data cannot be decoded, is corrupted, or requires an unavailable transcoder."""
    pass


class SpeechServiceUnavailableError(SpeechRecognitionError):
    """Raised when the STT backend service is unreachable or returns a service-level error."""
    pass


# ==============================================================================
# 2. Audio Validation & Transcoding Utilities
# ==============================================================================

def validate_and_inspect_wav(wav_bytes: bytes) -> Dict[str, Any]:
    """Validate that byte payload conforms to standard PCM WAV format and inspect audio characteristics.

    Args:
        wav_bytes: Byte payload of candidate WAV file.

    Returns:
        Dict with keys: 'channels', 'sample_width', 'sample_rate', 'n_frames', 'duration', 'rms', 'max_amplitude'.

    Raises:
        AudioDecodeError: If header is invalid, corrupted, or unreadable as WAV.
        NoSpeechDetectedError: If audio has 0 duration or is completely empty.
    """
    if not wav_bytes or len(wav_bytes) < 44:
        raise AudioDecodeError(
            f"Audio payload too small to contain valid WAV header ({len(wav_bytes) if wav_bytes else 0} bytes)."
        )

    if not (wav_bytes.startswith(b"RIFF") and b"WAVE" in wav_bytes[8:16]):
        raise AudioDecodeError("Audio payload does not have valid RIFF/WAVE header descriptor.")

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
    except Exception as exc:
        raise AudioDecodeError(f"Could not parse WAV structure: {exc}") from exc

    if channels not in (1, 2):
        raise AudioDecodeError(f"Unsupported channel count: {channels}. Only mono or stereo WAV supported.")

    if sample_rate <= 0:
        raise AudioDecodeError(f"Invalid sample rate in WAV header: {sample_rate} Hz.")

    bytes_per_frame = channels * sample_width
    actual_frames = len(frames) // bytes_per_frame if bytes_per_frame > 0 else 0
    duration = actual_frames / float(sample_rate) if sample_rate > 0 else 0.0

    if actual_frames == 0:
        raise NoSpeechDetectedError("Audio contains zero audio frames.")

    # Compute peak amplitude and RMS energy for 16-bit PCM
    rms = 0.0
    max_amp = 0
    if sample_width == 2 and frames:
        total_samples = len(frames) // 2
        if total_samples > 0:
            sum_sq = 0
            # Sample up to first 32,000 samples to keep check fast
            step = max(1, total_samples // 16000)
            count = 0
            for i in range(0, len(frames) - 1, 2 * step):
                sample = struct.unpack_from("<h", frames, i)[0]
                abs_s = abs(sample)
                if abs_s > max_amp:
                    max_amp = abs_s
                sum_sq += sample * sample
                count += 1
            rms = math.sqrt(sum_sq / count) if count > 0 else 0.0

    logger.debug(
        f"[Audio Validation] WAV valid: {channels}ch, {sample_width * 8}bit, "
        f"{sample_rate}Hz, {n_frames} frames ({duration:.2f}s), max_amp={max_amp}, RMS={rms:.1f}"
    )

    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "n_frames": n_frames,
        "duration": duration,
        "rms": rms,
        "max_amplitude": max_amp,
    }


def convert_audio_to_wav(audio_bytes: bytes, content_type: Optional[str] = None) -> Tuple[bytes, str]:
    """Ensure audio data is valid WAV PCM, performing FFmpeg transcoding if required.

    Args:
        audio_bytes: Raw audio byte payload.
        content_type: Optional MIME type (e.g. 'audio/wav', 'audio/webm', 'audio/ogg').

    Returns:
        Tuple of (wav_bytes, resolved_content_type).

    Raises:
        AudioDecodeError: If audio is invalid or conversion fails/is unavailable.
        NoSpeechDetectedError: If audio payload is empty.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise NoSpeechDetectedError("Audio data is empty (0 bytes).")

    clean_content_type = (content_type or "").split(";")[0].strip().lower()

    # If it already has a RIFF/WAVE header, treat directly as WAV
    if audio_bytes.startswith(b"RIFF") and len(audio_bytes) >= 12 and b"WAVE" in audio_bytes[8:12]:
        logger.debug("[Audio Converter] Audio already in native RIFF/WAVE format. Skipping transcoding.")
        return audio_bytes, "audio/wav"

    # Non-WAV format requiring FFmpeg transcoding (e.g., WebM, OGG, MP3)
    ffmpeg_bin = shutil.which("ffmpeg") or os.getenv("FFMPEG_PATH")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    if not ffmpeg_bin:
        fmt_label = clean_content_type or "non-WAV"
        raise AudioDecodeError(
            f"Audio format '{fmt_label}' cannot be decoded without FFmpeg. "
            "FFmpeg is not installed on the system or not found in PATH. "
            "Please install FFmpeg or send audio in standard PCM WAV format ('audio/wav')."
        )

    logger.info(
        f"[Audio Converter] Transcoding {clean_content_type or 'compressed'} audio "
        f"({len(audio_bytes)} bytes) to 16kHz mono PCM WAV via FFmpeg..."
    )

    try:
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        process = subprocess.Popen(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-i", "pipe:0",
                "-ac", "1",
                "-ar", "16000",
                "-f", "wav",
                "-acodec", "pcm_s16le",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
        )
        wav_output, stderr_output = process.communicate(input=audio_bytes, timeout=10)

        if process.returncode != 0 or not wav_output:
            err_msg = stderr_output.decode("utf-8", errors="ignore").strip()
            logger.warning(f"[Audio Converter] FFmpeg transcoding failed (code {process.returncode}): {err_msg}")
            raise AudioDecodeError(f"FFmpeg audio transcoding failed: {err_msg or 'Unspecified error'}")

        logger.info(f"[Audio Converter] FFmpeg transcoding successful: output size = {len(wav_output)} bytes")
        return wav_output, "audio/wav"
    except subprocess.TimeoutExpired:
        process.kill()
        raise AudioDecodeError("FFmpeg audio transcoding timed out after 10 seconds.")
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"Unexpected error during audio transcoding: {exc}") from exc


# ==============================================================================
# 3. Base Interface
# ==============================================================================

class BaseSpeechToText(ABC):
    """Abstract base class / interface for STT providers."""

    @abstractmethod
    def listen_and_transcribe(
        self,
        timeout: Optional[int] = None,
        phrase_time_limit: Optional[int] = None,
    ) -> Optional[str]:
        """Capture audio from the microphone and return transcribed text.

        Args:
            timeout: Maximum seconds to wait for speech to begin before timing out.
            phrase_time_limit: Maximum seconds to record once speech has begun.

        Returns:
            Transcribed and cleaned text string if successful.

        Raises:
            MicrophoneError: If audio hardware is missing or inaccessible.
            NoSpeechDetectedError: If no speech starts within the timeout window.
            SpeechRecognitionError: If speech could not be transcribed or understood.
        """
        pass

    @abstractmethod
    def transcribe_audio_data(
        self,
        audio_data: bytes,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        """Capture audio from raw bytes and return transcribed text.

        Args:
            audio_data: Raw audio byte payload (WAV format, etc.).
            content_type: Optional MIME type (e.g. 'audio/wav', 'audio/webm').

        Returns:
            Transcribed and cleaned text string if successful.

        Raises:
            NoSpeechDetectedError: If no speech was found or audio is empty.
            SpeechRecognitionError: If speech could not be transcribed.
        """
        pass


# ==============================================================================
# 3. Mock STT Provider (Offline / Testing / Headless)
# ==============================================================================

class MockSpeechToText(BaseSpeechToText):
    """Deterministic mock STT provider for unit testing, CI pipelines, and offline environments."""

    def __init__(
        self,
        responses: Optional[List[Optional[str]]] = None,
        simulate_no_speech: bool = False,
        simulate_recognition_error: bool = False,
        simulate_microphone_error: bool = False,
    ) -> None:
        """Initialize mock STT provider with controllable test behaviors.

        Args:
            responses: Predefined sequence of transcribed utterances to return in order.
            simulate_no_speech: If True, raises NoSpeechDetectedError.
            simulate_recognition_error: If True, raises SpeechRecognitionError.
            simulate_microphone_error: If True, raises MicrophoneError.
        """
        self.responses = list(responses) if responses is not None else ["Hello AURA"]
        self.simulate_no_speech = simulate_no_speech
        self.simulate_recognition_error = simulate_recognition_error
        self.simulate_microphone_error = simulate_microphone_error
        self.transcription_history: List[str] = []

    def set_responses(self, responses: List[Optional[str]]) -> None:
        """Set or update the canned responses list."""
        self.responses = list(responses)

    def listen_and_transcribe(
        self,
        timeout: Optional[int] = None,
        phrase_time_limit: Optional[int] = None,
    ) -> Optional[str]:
        """Simulate audio capture and return next pre-configured transcription."""
        if self.simulate_microphone_error:
            raise MicrophoneError("Simulated microphone hardware or permission error.")

        if self.simulate_no_speech:
            raise NoSpeechDetectedError("Simulated timeout: No speech was detected.")

        if self.simulate_recognition_error:
            raise SpeechRecognitionError("Simulated recognition failure: Speech was unintelligible.")

        if not self.responses:
            raise NoSpeechDetectedError("No more scripted responses available.")

        response = self.responses.pop(0)
        if response is None:
            raise NoSpeechDetectedError("No speech detected.")

        clean_text = response.strip()
        self.transcription_history.append(clean_text)
        logger.info("[STT] Speech received")
        logger.info("[STT] Transcription completed")
        logger.debug(f"[Mock STT] Transcribed: '{clean_text}'")
        return clean_text

    def transcribe_audio_data(
        self,
        audio_data: bytes,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        """Simulate transcribing uploaded audio data."""
        if not audio_data or len(audio_data.strip()) == 0:
            raise NoSpeechDetectedError("No audio data provided.")

        if self.simulate_microphone_error:
            raise MicrophoneError("Simulated microphone hardware or permission error.")

        if self.simulate_no_speech:
            raise NoSpeechDetectedError("Simulated timeout: No speech was detected.")

        if self.simulate_recognition_error:
            raise SpeechRecognitionError("Simulated recognition failure: Speech was unintelligible.")

        if not self.responses:
            raise NoSpeechDetectedError("No more scripted responses available.")

        response = self.responses.pop(0)
        if response is None:
            raise NoSpeechDetectedError("No speech detected.")

        clean_text = response.strip()
        self.transcription_history.append(clean_text)
        logger.info("[STT] Speech received")
        logger.info("[STT] Transcription completed")
        logger.debug(f"[Mock STT] Transcribed: '{clean_text}'")
        return clean_text


# ==============================================================================
# 4. Production SpeechRecognition STT Provider
# ==============================================================================

class SpeechRecognitionSTT(BaseSpeechToText):
    """STT provider powered by the `SpeechRecognition` library and PyAudio.

    Supports dynamic ambient noise adjustment, configurable timeouts, phrase length caps,
    and multiple backend recognition engines (defaults to Google Web Speech API with no API key).
    """

    def __init__(
        self,
        engine: str = "google",
        device_index: Optional[int] = None,
        default_timeout: int = 5,
        default_phrase_limit: int = 10,
        energy_threshold: Optional[int] = None,
        dynamic_energy_threshold: bool = True,
    ) -> None:
        """Initialize the SpeechRecognition STT provider.

        Args:
            engine: Recognition engine to use ('google', 'whisper', etc.).
            device_index: Specific microphone hardware index, or None for system default.
            default_timeout: Seconds to wait for speech before raising NoSpeechDetectedError.
            default_phrase_limit: Max recording length per utterance in seconds.
            energy_threshold: Initial microphone energy threshold for sound detection.
            dynamic_energy_threshold: Whether to automatically adapt to ambient room noise.
        """
        self.engine = engine.lower()
        self.device_index = device_index
        self.default_timeout = default_timeout
        self.default_phrase_limit = default_phrase_limit
        self.energy_threshold = energy_threshold
        self.dynamic_energy_threshold = dynamic_energy_threshold

        try:
            import speech_recognition as sr
            self._sr = sr
        except ImportError as exc:
            raise MicrophoneError(
                "SpeechRecognition package is not installed. Install with `pip install SpeechRecognition`."
            ) from exc

        self.recognizer = self._sr.Recognizer()
        if energy_threshold is not None:
            self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold

    def _get_microphone(self):
        """Acquire and validate a Microphone instance."""
        try:
            return self._sr.Microphone(device_index=self.device_index)
        except Exception as exc:
            raise MicrophoneError(f"Failed to initialize microphone device: {exc}") from exc

    def listen_and_transcribe(
        self,
        timeout: Optional[int] = None,
        phrase_time_limit: Optional[int] = None,
    ) -> Optional[str]:
        """Listen to the microphone and transcribe speech to clean text."""
        effective_timeout = timeout if timeout is not None else self.default_timeout
        effective_limit = phrase_time_limit if phrase_time_limit is not None else self.default_phrase_limit

        sr = self._sr

        try:
            mic = self._get_microphone()
            with mic as source:
                logger.debug("Calibrating ambient noise baseline (0.5s)...")
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception as exc:
                    logger.warning(f"Ambient noise adjustment warning: {exc}")

                logger.info("Microphone listening... Speak now.")
                audio = self.recognizer.listen(
                    source,
                    timeout=effective_timeout,
                    phrase_time_limit=effective_limit,
                )
        except sr.WaitTimeoutError as exc:
            raise NoSpeechDetectedError("No speech detected within the listening timeout.") from exc
        except (OSError, IOError, AttributeError) as exc:
            raise MicrophoneError(f"Microphone hardware or audio driver failure: {exc}") from exc
        except Exception as exc:
            # Check if this was caused by PyAudio or audio driver error
            exc_str = str(exc).lower()
            if "pyaudio" in exc_str or "device" in exc_str or "input" in exc_str:
                raise MicrophoneError(f"Microphone input device error: {exc}") from exc
            raise

        logger.debug("Audio captured. Sending to speech recognizer...")

        # Transcribe audio using selected engine
        try:
            if self.engine in {"google", "local", "default"}:
                # Google Web Speech API (free, built-in, no API key needed)
                text = self.recognizer.recognize_google(audio)
            elif self.engine == "whisper":
                # Local whisper model if installed
                text = self.recognizer.recognize_whisper(audio)
            elif self.engine == "sphinx":
                text = self.recognizer.recognize_sphinx(audio)
            else:
                text = self.recognizer.recognize_google(audio)
        except sr.UnknownValueError as exc:
            raise SpeechRecognitionError("Speech was unclear or could not be understood.") from exc
        except sr.RequestError as exc:
            raise SpeechRecognitionError(f"STT recognition service error: {exc}") from exc
        except Exception as exc:
            raise SpeechRecognitionError(f"Speech transcription failed: {exc}") from exc

        clean_text = (text or "").strip()
        if not clean_text:
            raise NoSpeechDetectedError("Empty transcription returned.")

        logger.info("[STT] Speech received")
        logger.info("[STT] Transcription completed")
        logger.info(f"Transcribed speech: \"{clean_text}\"")
        return clean_text

    def transcribe_audio_data(
        self,
        audio_data: bytes,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        """Transcribe raw audio bytes (such as WAV or converted WebM) to clean text."""
        if not audio_data or len(audio_data) == 0:
            raise NoSpeechDetectedError("Empty audio data provided.")

        # 1. Convert to WAV if required (e.g. WebM/Opus via FFmpeg)
        wav_bytes, resolved_type = convert_audio_to_wav(audio_data, content_type=content_type)

        # 2. Validate WAV structure and extract audio info
        audio_info = validate_and_inspect_wav(wav_bytes)

        # 3. Read audio data with SpeechRecognition AudioFile
        sr = self._sr
        try:
            with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
                audio = self.recognizer.record(source)
        except Exception as exc:
            logger.warning(f"Could not read audio data with AudioFile: {exc}")
            raise AudioDecodeError(f"Audio file format invalid or unreadable: {exc}") from exc

        # 4. Transcribe using configured engine
        try:
            if self.engine in {"google", "local", "default"}:
                text = self.recognizer.recognize_google(audio)
            elif self.engine == "whisper":
                text = self.recognizer.recognize_whisper(audio)
            elif self.engine == "sphinx":
                text = self.recognizer.recognize_sphinx(audio)
            else:
                text = self.recognizer.recognize_google(audio)
        except sr.UnknownValueError as exc:
            logger.info("[STT] Recognizer completed: No speech could be recognized (UnknownValueError).")
            raise NoSpeechDetectedError("No speech detected in audio or speech was unintelligible.") from exc
        except sr.RequestError as exc:
            logger.error(f"[STT] Recognition service request failed: {exc}")
            raise SpeechServiceUnavailableError(f"STT recognition service error: {exc}") from exc
        except Exception as exc:
            logger.error(f"[STT] Unexpected transcription failure: {exc}")
            raise SpeechRecognitionError(f"Speech transcription failed: {exc}") from exc

        clean_text = (text or "").strip()
        if not clean_text:
            raise NoSpeechDetectedError("Empty transcription returned.")

        logger.info("[STT] Speech received")
        logger.info("[STT] Transcription completed")
        logger.info(f"Transcribed speech: \"{clean_text}\"")
        return clean_text


# ==============================================================================
# 5. Factory & Abstraction Interface
# ==============================================================================

def get_stt_provider(provider_type: Optional[str] = None, **kwargs) -> BaseSpeechToText:
    """Factory to instantiate the configured STT provider.

    Args:
        provider_type: Optional explicit provider name ('local', 'google', 'mock').
                       Defaults to STT_PROVIDER or STT_ENGINE environment variables.
        **kwargs: Additional parameters forwarded to provider constructor.

    Returns:
        Concrete BaseSpeechToText instance.
    """
    raw_provider = provider_type or os.getenv("STT_PROVIDER") or os.getenv("STT_ENGINE") or "local"
    provider_name = raw_provider.strip().lower()

    timeout = int(os.getenv("STT_TIMEOUT", "5"))
    phrase_limit = int(os.getenv("STT_PHRASE_TIME_LIMIT", "10"))
    energy = os.getenv("STT_ENERGY_THRESHOLD", "").strip()
    energy_threshold = int(energy) if energy.isdigit() else 300

    if provider_name == "mock":
        return MockSpeechToText(**kwargs)

    return SpeechRecognitionSTT(
        engine=provider_name,
        default_timeout=kwargs.get("default_timeout", timeout),
        default_phrase_limit=kwargs.get("default_phrase_limit", phrase_limit),
        energy_threshold=kwargs.get("energy_threshold", energy_threshold),
        **{k: v for k, v in kwargs.items() if k not in {"default_timeout", "default_phrase_limit", "energy_threshold"}},
    )


# Default singleton instance for top-level speech_to_text() calls
_default_stt_instance: Optional[BaseSpeechToText] = None


def speech_to_text(
    timeout: Optional[int] = None,
    phrase_time_limit: Optional[int] = None,
    provider: Optional[BaseSpeechToText] = None,
) -> Optional[str]:
    """Top-level decoupled STT abstraction for AURA.

    Captures speech from the active microphone and returns transcribed text.

    Args:
        timeout: Maximum seconds to wait for speech before timing out.
        phrase_time_limit: Maximum recording duration once speech has begun.
        provider: Optional explicit BaseSpeechToText provider instance.

    Returns:
        Clean transcribed text string, or None if no speech was detected.

    Example:
        ```python
        from app.speech.stt import speech_to_text

        text = speech_to_text()
        if text:
            print(f"Recognized: {text}")
        ```
    """
    global _default_stt_instance
    active_provider = provider or _default_stt_instance
    if active_provider is None:
        active_provider = get_stt_provider()
        if provider is None:
            _default_stt_instance = active_provider

    try:
        return active_provider.listen_and_transcribe(
            timeout=timeout,
            phrase_time_limit=phrase_time_limit,
        )
    except (NoSpeechDetectedError, SpeechRecognitionError) as exc:
        logger.debug(f"speech_to_text captured non-fatal condition: {exc}")
        return None


def transcribe_audio(
    audio_data: bytes,
    content_type: Optional[str] = None,
    provider: Optional[BaseSpeechToText] = None,
) -> Optional[str]:
    """Top-level decoupled audio transcription abstraction for AURA.

    Transcribes audio bytes into clean text using the active STT provider.

    Args:
        audio_data: Raw audio byte payload.
        content_type: Optional MIME type.
        provider: Optional explicit BaseSpeechToText provider instance.

    Returns:
        Clean transcribed text string, or None if transcription failed/was empty.
    """
    global _default_stt_instance
    active_provider = provider or _default_stt_instance
    if active_provider is None:
        active_provider = get_stt_provider()
        if provider is None:
            _default_stt_instance = active_provider

    try:
        return active_provider.transcribe_audio_data(
            audio_data=audio_data,
            content_type=content_type,
        )
    except (NoSpeechDetectedError, SpeechRecognitionError) as exc:
        logger.debug(f"transcribe_audio captured non-fatal condition: {exc}")
        return None

