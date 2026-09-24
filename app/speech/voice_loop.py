"""Voice interaction loop for AURA.

Coordinates the end-to-end multi-turn voice pipeline:
Microphone -> STT -> Intent Classification -> Conversation Memory ->
RAG / Tool Calling -> LLM Brain -> Conversation Memory -> TTS -> Speaker.
"""

import sys
import logging
from typing import Optional, List, Set, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.brain.intent import IntentClassifier

from app.brain.llm import LLMBrain
from app.memory.conversation import ConversationMemory
from app.speech.stt import (
    BaseSpeechToText,
    get_stt_provider,
    SpeechError,
    MicrophoneError,
    NoSpeechDetectedError,
    SpeechRecognitionError,
)
from app.speech.tts import (
    BaseTextToSpeech,
    get_tts_provider,
    TTSError,
)

logger = logging.getLogger(__name__)

DEFAULT_EXIT_PHRASES: Set[str] = {
    "goodbye",
    "exit",
    "quit",
    "bye",
    "stop",
    "shut down",
    "farewell",
}


def is_exit_command(text: str, detected_intent: Optional[str] = None, exit_phrases: Optional[Set[str]] = None) -> bool:
    """Check if the transcribed user text or intent corresponds to an exit request."""
    clean = text.lower().strip().strip(".!?,")
    phrases = exit_phrases or DEFAULT_EXIT_PHRASES

    if clean in phrases:
        return True

    # Check if phrase starts or ends with an exit word
    words = set(clean.split())
    if words & {"goodbye", "farewell"}:
        return True

    if detected_intent == "goodbye" and (words & {"bye", "goodbye", "exit", "quit", "stop"}):
        return True

    return False


def run_voice_loop(
    classifier: Any,
    brain: LLMBrain,
    memory: ConversationMemory,
    stt: Optional[BaseSpeechToText] = None,
    tts: Optional[BaseTextToSpeech] = None,
    exit_phrases: Optional[Set[str]] = None,
    max_turns: Optional[int] = None,
    user_display_prefix: str = "You: ",
    aura_display_prefix: str = "AURA: ",
    verbose_intent: bool = False,
) -> None:
    """Run an interactive voice conversation loop with AURA.

    Listens to microphone speech, transcribes it, runs intent classification and LLM reasoning
    (with conversation memory, tools, and RAG), prints the exchange to the terminal, and speaks
    the response through the speakers.

    Args:
        classifier: IntentClassifier instance for query understanding.
        brain: LLMBrain instance for cognitive reasoning and response generation.
        memory: ConversationMemory instance for multi-turn short-term dialogue context.
        stt: Optional concrete BaseSpeechToText provider. Defaults to get_stt_provider().
        tts: Optional concrete BaseTextToSpeech provider. Defaults to get_tts_provider().
        exit_phrases: Custom set of phrases that safely exit the loop.
        max_turns: Optional maximum number of conversational turns (useful for tests).
        user_display_prefix: Terminal label for the user (default: 'You: ').
        aura_display_prefix: Terminal label for AURA (default: 'AURA: ').
        verbose_intent: Whether to display detected intent and confidence in console.
    """
    stt_provider = stt or get_stt_provider()
    tts_provider = tts or get_tts_provider()
    phrases = exit_phrases or DEFAULT_EXIT_PHRASES

    print("\n========================================================")
    print("           AURA Phase 7: Interactive Voice Mode         ")
    print("========================================================")
    print("  * Speak into your microphone after the prompt appears.")
    print("  * Say 'goodbye', 'exit', or press Ctrl+C to stop.")
    print("========================================================\n")

    turns_completed = 0

    while True:
        if max_turns is not None and turns_completed >= max_turns:
            logger.info(f"Max voice turns reached ({max_turns}). Exiting loop.")
            break

        try:
            print("\n[Listening...] Speak into your microphone...")

            # 1. Listen & Transcribe via STT
            try:
                user_text = stt_provider.listen_and_transcribe()
            except NoSpeechDetectedError:
                print("[No speech detected. Listening again...]")
                continue
            except SpeechRecognitionError as exc:
                print(f"[Could not understand speech: {exc}. Please try again...]")
                continue
            except MicrophoneError as exc:
                print(f"\n[Microphone Error] {exc}")
                print("Audio input device is not responding. Exiting voice loop.")
                break

            if not user_text or not user_text.strip():
                print("[Empty input detected. Listening again...]")
                continue

            user_text = user_text.strip()

            # 2. Display Recognized Speech
            print(f"\n{user_display_prefix}{user_text}")

            # 3. Check for immediate explicit exit commands
            if is_exit_command(user_text, exit_phrases=phrases):
                farewell = "Goodbye! Have a wonderful day!"
                print(f"{aura_display_prefix}{farewell}\n")
                try:
                    tts_provider.speak(farewell)
                except Exception as tts_err:
                    logger.warning(f"TTS farewell error: {tts_err}")
                break

            # 4. Intent Classification
            intent_res = classifier.predict(user_text)
            intent = intent_res.get("intent", "general")
            confidence = intent_res.get("confidence", 0.0)

            if verbose_intent:
                print(f"       [Intent: {intent.upper()} ({confidence:.1%})]")

            # Check if intent + text together represent exit
            if is_exit_command(user_text, detected_intent=intent, exit_phrases=phrases):
                farewell = "Goodbye! Let me know if you need anything else."
                print(f"{aura_display_prefix}{farewell}\n")
                try:
                    tts_provider.speak(farewell)
                except Exception as tts_err:
                    logger.warning(f"TTS farewell error: {tts_err}")
                break

            # 5. Store user message in short-term conversation memory
            memory.add_user_message(user_text)

            # 6. Brain processes query (Tools, RAG context, and LLM reasoning)
            response = brain.ask(
                user_message=user_text,
                context=memory,
                detected_intent=intent,
            )

            # 7. Store assistant response in short-term conversation memory
            memory.add_assistant_message(response)

            # 8. Display response in terminal
            print(f"{aura_display_prefix}{response}")

            # 9. Speak response through audio speaker
            try:
                tts_provider.speak(response)
            except Exception as tts_err:
                logger.error(f"Text-to-Speech playback failed: {tts_err}")
                print("[Audio output error: Response could not be spoken.]")

            turns_completed += 1

        except (KeyboardInterrupt, EOFError):
            print("\n\n[Voice interaction stopped by user. Goodbye!]")
            try:
                tts_provider.speak("Goodbye!")
            except Exception:
                pass
            break
        except Exception as exc:
            logger.error(f"Unexpected error in voice loop: {exc}", exc_info=True)
            print(f"\n[An unexpected error occurred: {exc}]")
            # Don't crash the loop on transient errors
            continue

    print("\n--- AURA Voice Session Ended ---")
