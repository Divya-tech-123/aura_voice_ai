"""AURA Central Application Orchestrator.

Phase 8 — Full System Integration:
Coordinates Speech-to-Text (STT), Intent Classification, Short-Term Conversation Memory,
Long-Term / RAG Document Retrieval, Safe Tool Execution, LLM Brain Reasoning,
and Text-to-Speech (TTS) into a unified, resilient AI assistant pipeline.
"""

import os
import re
import sys
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple

from app.brain.intent import IntentClassifier
from app.brain.llm import LLMBrain, LLMError, get_llm_provider
from app.memory.conversation import ConversationMemory
from app.tools.registry import ToolRegistry, create_default_registry
from app.rag.retriever import LocalRetriever
from app.rag.loader import index_documents
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
    AudioDeviceError,
    TTSProviderError,
)

logger = logging.getLogger("AURA.Orchestrator")


class RouteType(str, Enum):
    """Enumeration of execution routes supported by AURA."""
    LLM = "llm"
    CALCULATOR = "calculator"
    WEATHER = "weather"
    SEARCH = "search"
    FILES = "files"
    RAG = "rag"
    EXIT = "exit"
    CONVERSATIONAL = "conversational"


@dataclass
class OrchestratorResult:
    """Structured result of an orchestrated user query turn."""
    user_text: str
    intent: str
    confidence: float
    route: RouteType
    response: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    is_exit: bool = False


DEFAULT_EXIT_PHRASES: Set[str] = {
    "goodbye",
    "exit",
    "quit",
    "bye",
    "stop",
    "shut down",
    "farewell",
}


def is_exit_request(text: str, detected_intent: Optional[str] = None, exit_phrases: Optional[Set[str]] = None) -> bool:
    """Determine if a user input text or detected intent indicates session termination."""
    clean = (text or "").lower().strip().strip(".!?,")
    phrases = exit_phrases or DEFAULT_EXIT_PHRASES

    if clean in phrases:
        return True

    words = set(clean.split())
    if words & {"goodbye", "farewell"}:
        return True

    if detected_intent == "goodbye" and (words & {"bye", "goodbye", "exit", "quit", "stop"}):
        return True

    return False


class AURAOrchestrator:
    """Central orchestration layer coordinating all AURA subsystems."""

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        brain: Optional[LLMBrain] = None,
        memory: Optional[ConversationMemory] = None,
        retriever: Optional[LocalRetriever] = None,
        tool_registry: Optional[ToolRegistry] = None,
        stt: Optional[BaseSpeechToText] = None,
        tts: Optional[BaseTextToSpeech] = None,
        enable_rag: bool = True,
        rag_top_k: int = 3,
        rag_min_score: float = 0.15,
        exit_phrases: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the orchestrator with all necessary subsystems.

        Subsystems can be explicitly injected or initialized with default configurations.
        """
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever
        self.tool_registry = tool_registry if tool_registry is not None else create_default_registry()
        self.memory = memory or ConversationMemory(max_messages=16)
        self.brain = brain or LLMBrain(
            tool_registry=self.tool_registry,
            retriever=self.retriever,
            enable_tools=True,
            enable_rag=enable_rag,
        )
        self.stt = stt
        self.tts = tts
        self.enable_rag = enable_rag
        self.rag_top_k = rag_top_k
        self.rag_min_score = rag_min_score
        self.exit_phrases = exit_phrases or DEFAULT_EXIT_PHRASES

        logger.info("AURA Orchestrator initialized successfully.")

    def determine_route(self, user_text: str, intent: str, confidence: float) -> RouteType:
        """Intelligently route the user request to the appropriate execution subsystem.

        Decision hierarchy:
        1. Exit verification: Goodbye/exit keywords or intent -> EXIT
        2. Arithmetic / Math pattern or calculator intent -> CALCULATOR
        3. Weather query pattern or weather intent -> WEATHER
        4. Web search pattern or search intent -> SEARCH
        5. File operations pattern -> FILES
        6. Document knowledge indicators (notes, docs, according to) -> RAG
        7. Default -> LLM (General reasoning, conversations, memory recall)
        """
        clean = (user_text or "").strip().lower()

        # 1. Exit check
        if is_exit_request(clean, detected_intent=intent, exit_phrases=self.exit_phrases):
            return RouteType.EXIT

        # 2. Tool routing: Calculator
        math_norm = re.sub(r"(\d+)\s*[×x]\s*(\d+)", r"\1 * \2", clean)
        math_norm = re.sub(r"(\d+)\s+times\s+(\d+)", r"\1 * \2", math_norm)
        has_arith = bool(re.search(r"(\d+\s*(?:[\+\-\*\/]|\*\*)\s*\d+)", math_norm))
        has_math_phrase = bool(re.search(r"\b(?:what is|what's|calculate|compute|solve)\s+([0-9\.\s\+\-\*\/\(\)\%\^]+|sqrt\([0-9\.]+\))\b", math_norm))
        if intent == "calculator" or has_arith or has_math_phrase:
            return RouteType.CALCULATOR

        # 3. Tool routing: Weather
        if intent == "weather" or re.search(r"(?:weather\s+(?:in|for|at)|how(?:'s|\s+is)\s+the\s+weather)", clean):
            return RouteType.WEATHER

        # 4. Tool routing: Web Search
        if intent == "search" or re.search(r"(?:search(?:\s+for|\s+web|\s+online)?|google\s+this|look\s+up|find\s+info\s+on)", clean):
            if not any(f in clean for f in ("file", "files", "note", "notes", "doc")):
                return RouteType.SEARCH

        # 5. Tool routing: Files
        if any(p in clean for p in ("list files", "show files", "read file", "file metadata", "view files")):
            return RouteType.FILES

        # 6. RAG Document Knowledge routing
        # Check if query references indexed documentation / notes or asks for document grounding
        rag_patterns = [
            r"\b(?:according to|in|from|per)\s+(?:my\s+)?(?:ai\s+)?(?:notes|document|documents|doc|docs|paper|pdf|text)\b",
            r"\b(?:what\s+does|what\s+do)\s+(?:my\s+)?(?:ai\s+)?(?:notes|document|documents|doc|docs)\s+say\b",
            r"\bmy\s+ai\s+notes\b",
            r"\baccording\s+to\s+my\b",
            r"\bcheck\s+(?:the\s+)?(?:document|documents|notes)\b",
        ]
        if self.enable_rag and self.retriever is not None:
            for pattern in rag_patterns:
                if re.search(pattern, clean):
                    return RouteType.RAG

        # 7. Conversational fast-paths (greetings/thanks/capabilities)
        if intent in {"greeting", "thanks", "capabilities"}:
            return RouteType.CONVERSATIONAL

        # 8. Standard LLM reasoning (using conversational memory, no unnecessary document retrieval)
        return RouteType.LLM

    def process_query(self, user_text: str) -> OrchestratorResult:
        """Process a user text query through the complete AURA pipeline.

        Pipeline:
        User Text -> Intent Classifier -> Intelligent Routing -> Memory (add user) ->
        Brain Processing (RAG / Tool / LLM) -> Memory (add assistant) -> OrchestratorResult.
        """
        clean_text = (user_text or "").strip()
        if not clean_text:
            return OrchestratorResult(
                user_text="",
                intent="unknown",
                confidence=0.0,
                route=RouteType.LLM,
                response="I didn't hear anything. How can I help you?",
            )

        # 1. Intent Classification
        try:
            intent_res = self.classifier.predict(clean_text)
            intent = intent_res.get("intent", "unknown")
            confidence = intent_res.get("confidence", 0.0)
        except Exception as exc:
            logger.warning(f"Intent classification failed: {exc}. Falling back to 'unknown'.")
            intent = "unknown"
            confidence = 0.0

        logger.info(f"[INTENT] intent={intent} confidence={confidence:.2f}")

        # 2. Exit Check
        if is_exit_request(clean_text, detected_intent=intent, exit_phrases=self.exit_phrases):
            farewell = "Goodbye! Have a wonderful day!"
            logger.info("[ORCHESTRATOR] Exit command recognized.")
            return OrchestratorResult(
                user_text=clean_text,
                intent="goodbye",
                confidence=1.0,
                route=RouteType.EXIT,
                response=farewell,
                is_exit=True,
            )

        # 3. Intelligent Routing
        route = self.determine_route(clean_text, intent, confidence)
        logger.info(f"[ROUTING] route={route.value}")

        # 4. Record user message in Short-Term Conversation Memory
        self.memory.add_user_message(clean_text)

        # 5. Cognitive Execution via Brain
        logger.info("[BRAIN] Processing request")
        sources: List[Dict[str, Any]] = []
        response = ""

        try:
            if route == RouteType.RAG and self.retriever is not None:
                # Question requires document knowledge -> fetch relevant chunks and ground response
                logger.info(f"[RAG] Retrieving documents for query: '{clean_text}'")
                try:
                    rag_result = self.brain.ask_with_rag(
                        user_message=clean_text,
                        context=self.memory,
                        detected_intent=intent,
                        top_k=self.rag_top_k,
                        min_score=self.rag_min_score,
                    )
                    response = rag_result.get("answer", "")
                    sources = rag_result.get("sources", [])
                    logger.info(f"[RAG] Document retrieval grounded with {len(sources)} source(s)")
                except Exception as rag_err:
                    logger.warning(f"RAG retrieval encountered an error: {rag_err}. Falling back to direct LLM.")
                    response = self.brain.ask(
                        user_message=clean_text,
                        context=self.memory,
                        detected_intent=intent,
                        use_rag=False,
                    )

            elif route in {RouteType.CALCULATOR, RouteType.WEATHER, RouteType.SEARCH, RouteType.FILES}:
                # Tool route -> disable RAG to prevent unnecessary document retrieval
                response = self.brain.ask(
                    user_message=clean_text,
                    context=self.memory,
                    detected_intent=intent,
                    use_rag=False,
                )

            else:
                # LLM reasoning / conversational / memory recall -> RAG not needed
                response = self.brain.ask(
                    user_message=clean_text,
                    context=self.memory,
                    detected_intent=intent,
                    use_rag=False,
                )

        except LLMError as llm_err:
            logger.error(f"LLM Brain error during execution: {llm_err}")
            response = f"I encountered a problem with my language engine: {llm_err}"
        except Exception as exc:
            logger.exception(f"Unexpected error in cognitive processing: {exc}")
            response = "I encountered an unexpected issue while processing your request."

        # 6. Record assistant response in Short-Term Conversation Memory
        self.memory.add_assistant_message(response)

        return OrchestratorResult(
            user_text=clean_text,
            intent=intent,
            confidence=confidence,
            route=route,
            response=response,
            sources=sources,
            is_exit=False,
        )

    def run_text_turn(self, user_input: str) -> str:
        """Run a single interaction turn using textual input and return the response."""
        result = self.process_query(user_input)
        return result.response

    def run_text_loop(self) -> None:
        """Run an interactive console text mode loop for development and headless environments."""
        provider_name = os.getenv("LLM_PROVIDER", "mock")
        print("\n========================================================")
        print("          AURA Phase 8: Interactive Text Mode           ")
        print("========================================================")
        print(f"  * Cognitive Brain Active (Provider: {provider_name})")
        print("  * Type your message and press Enter.")
        print("  * Type 'exit', 'quit', or 'goodbye' to stop.")
        print("========================================================\n")

        while True:
            try:
                user_input = input("User > ").strip()
                if not user_input:
                    continue

                result = self.process_query(user_input)
                print(f"AURA > {result.response}\n")

                if result.is_exit:
                    break

            except (KeyboardInterrupt, EOFError):
                print("\nExiting AURA. Goodbye!")
                break
            except Exception as exc:
                logger.error(f"Text loop error: {exc}")
                print(f"\n[Error: {exc}]\n")

    def run_voice_turn(self) -> Tuple[Optional[str], Optional[str], bool]:
        """Execute one complete voice turn:

        Listen -> Transcribe -> Display -> Process -> Display -> Speak -> Return status.

        Returns:
            Tuple of (user_text, assistant_response, is_exit)
        """
        stt_provider = self.stt or get_stt_provider()
        tts_provider = self.tts or get_tts_provider()

        print("\n[Listening...] Speak into your microphone...")

        try:
            user_text = stt_provider.listen_and_transcribe()
        except NoSpeechDetectedError:
            print("[No speech detected.]")
            return None, None, False
        except SpeechRecognitionError as exc:
            print(f"[Could not understand speech: {exc}]")
            return None, None, False
        except MicrophoneError as exc:
            logger.error(f"Microphone error: {exc}")
            print(f"\n[Microphone Error: {exc}]")
            raise

        if not user_text or not user_text.strip():
            return None, None, False

        clean_text = user_text.strip()
        print(f"\nYou > {clean_text}")

        # Process through orchestrator pipeline
        result = self.process_query(clean_text)

        # Display response
        print(f"AURA > {result.response}")

        # Speak response via TTS
        try:
            tts_provider.speak(result.response)
        except Exception as tts_err:
            logger.error(f"TTS playback error: {tts_err}")
            print(f"[Audio output notice: Response could not be played aloud: {tts_err}]")

        return clean_text, result.response, result.is_exit

    def run_voice_loop(self, max_turns: Optional[int] = None) -> None:
        """Run the interactive voice loop coordinating speech input and output."""
        tts_provider = self.tts or get_tts_provider()

        print("\n========================================================")
        print("          AURA Phase 8: Interactive Voice Mode          ")
        print("========================================================")
        print("  * Speak into your microphone after the prompt appears.")
        print("  * Say 'goodbye', 'exit', or press Ctrl+C to stop.")
        print("========================================================\n")

        turns_count = 0

        while True:
            if max_turns is not None and turns_count >= max_turns:
                logger.info(f"Max turns ({max_turns}) reached. Exiting voice loop.")
                break

            try:
                user_text, response, should_exit = self.run_voice_turn()
                if user_text is not None:
                    turns_count += 1

                if should_exit:
                    break

            except MicrophoneError as mic_err:
                logger.warning(f"Microphone unavailable: {mic_err}. Falling back to text mode.")
                print(f"\n[Microphone unavailable ({mic_err}). Switching to Text Mode.]\n")
                self.run_text_loop()
                break
            except (KeyboardInterrupt, EOFError):
                print("\n\n[Voice session stopped by user.]")
                try:
                    tts_provider.speak("Goodbye!")
                except Exception:
                    pass
                break
            except Exception as exc:
                logger.error(f"Unexpected error in voice loop: {exc}", exc_info=True)
                print(f"\n[An unexpected error occurred: {exc}]")
                continue

        print("\n--- AURA Voice Session Ended ---")
