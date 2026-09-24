"""AURA Phase 8: End-to-End Full System Integration Demonstration.

Demonstrates the five required integration cases:
CASE 1 — Normal LLM: Direct language reasoning without unnecessary tool/RAG retrieval.
CASE 2 — Calculator: Tool call decision, validation, safe execution, and natural language synthesis.
CASE 3 — RAG: Document knowledge retrieval, chunk scoring, LLM answer synthesis, and source attribution.
CASE 4 — Memory: Conversational continuity preserving user facts across multi-turn dialogue.
CASE 5 — Voice: End-to-end voice flow (STT -> AURA Brain -> LLM -> TTS) using mock speech hardware.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.orchestrator import AURAOrchestrator, RouteType
from app.rag.loader import index_documents
from app.rag.retriever import LocalRetriever
from app.speech.stt import MockSpeechToText
from app.speech.tts import MockTextToSpeech
from app.tools.registry import create_default_registry


def run_integration_demo(orchestrator: Optional[AURAOrchestrator] = None) -> None:
    """Execute the Phase 8 end-to-end full system demonstration."""
    print("\n" + "=" * 70)
    print("        AURA PHASE 8: FULL SYSTEM INTEGRATION DEMONSTRATION")
    print("=" * 70)

    # 1. Prepare RAG index if not already populated
    retriever = LocalRetriever()
    if len(retriever.vector_store) == 0:
        print("[Setup] Indexing 'documents/' directory for RAG grounding...")
        index_documents("documents")
        retriever.load_index()

    # 2. Setup mock audio providers for automated voice demonstration
    mock_stt = MockSpeechToText(responses=["What is machine learning?"])
    mock_tts = MockTextToSpeech()

    app = orchestrator or AURAOrchestrator(
        retriever=retriever,
        stt=mock_stt,
        tts=mock_tts,
    )

    # --------------------------------------------------------------------------
    # CASE 1 — Normal LLM
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CASE 1 — Normal LLM (Direct Concept Reasoning)")
    print("User Query: 'Explain artificial intelligence.'")
    print("Expected Pipeline: User -> LLM -> Response (No RAG / No Tool)")
    print("-" * 70)
    res1 = app.process_query("Explain artificial intelligence.")
    print(f"  [Intent Identified] : {res1.intent.upper()} (Confidence: {res1.confidence:.1%})")
    print(f"  [Route Selected]    : {res1.route.value.upper()}")
    print(f"  [RAG Invoked?]      : {'Yes' if res1.sources else 'No (Unnecessary retrieval avoided)'}")
    print(f"  [AURA Response]     :\n  {res1.response}")

    # --------------------------------------------------------------------------
    # CASE 2 — Calculator Tool
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CASE 2 — Calculator (Safe Tool Calling & Synthesis)")
    print("User Query: 'What is 125 * 32?'")
    print("Expected Pipeline: User -> Tool Request -> Tool Validation & Exec -> LLM -> Response")
    print("-" * 70)
    res2 = app.process_query("What is 125 * 32?")
    print(f"  [Intent Identified] : {res2.intent.upper()} (Confidence: {res2.confidence:.1%})")
    print(f"  [Route Selected]    : {res2.route.value.upper()}")
    print(f"  [AURA Response]     :\n  {res2.response}")

    # --------------------------------------------------------------------------
    # CASE 3 — RAG Document Knowledge
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CASE 3 — RAG (Document Knowledge Retrieval & Source Attribution)")
    print("User Query: 'What does my AI notes say about overfitting?'")
    print("Expected Pipeline: User -> Retriever -> Relevant Document -> LLM -> Answer + Source")
    print("-" * 70)
    res3 = app.process_query("What does my AI notes say about overfitting?")
    print(f"  [Intent Identified] : {res3.intent.upper()} (Confidence: {res3.confidence:.1%})")
    print(f"  [Route Selected]    : {res3.route.value.upper()}")
    print(f"  [Sources Retrieved] : {len(res3.sources)} document excerpt(s)")
    for idx, s in enumerate(res3.sources, 1):
        score = s.get("score", 0.0)
        print(f"    {idx}. File: {s.get('filename')} (Score: {score:.3f})")
    print(f"  [AURA Response]     :\n  {res3.response}")

    # --------------------------------------------------------------------------
    # CASE 4 — Conversation Memory
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CASE 4 — Memory (Multi-Turn Short-Term Context Retention)")
    print("Turn 4A: 'My favorite programming language is Python.'")
    print("Turn 4B: 'What is my favorite programming language?'")
    print("Expected Pipeline: User -> Conversation Memory -> LLM -> Grounded Recall")
    print("-" * 70)
    res4a = app.process_query("My favorite programming language is Python.")
    print(f"  User Turn 1 > 'My favorite programming language is Python.'")
    print(f"  AURA Turn 1 > {res4a.response}")

    res4b = app.process_query("What is my favorite programming language?")
    print(f"\n  User Turn 2 > 'What is my favorite programming language?'")
    print(f"  [Route Selected] : {res4b.route.value.upper()}")
    print(f"  [Memory Buffer]  : {len(app.memory)} messages in short-term buffer")
    print(f"  AURA Turn 2 > {res4b.response}")

    # --------------------------------------------------------------------------
    # CASE 5 — Voice Interaction Loop
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CASE 5 — Voice (Audio Speech Pipeline via Mock Hardware)")
    print("Spoken Query: 'What is machine learning?'")
    print("Expected Pipeline: Microphone -> STT -> AURA Brain -> LLM -> TTS -> Speaker")
    print("-" * 70)
    orig_stt = app.stt
    orig_tts = app.tts
    app.stt = mock_stt
    app.tts = mock_tts
    try:
        user_speech, spoken_resp, is_exit = app.run_voice_turn()
        print(f"  [STT Input]  : '{user_speech}'")
        print(f"  [TTS Output] : Spoke {len(mock_tts.spoken_texts)} utterance(s)")
        if mock_tts.spoken_texts:
            print(f"  [Audio Text] : \"{mock_tts.spoken_texts[-1]}\"")
    finally:
        app.stt = orig_stt
        app.tts = orig_tts

    print("\n" + "=" * 70)
    print("     PHASE 8 END-TO-END DEMONSTRATION SUCCESSFULLY COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_integration_demo()
