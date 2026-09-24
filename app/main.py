"""AURA — AI Unified Response Assistant.

Application entry point and subsystem coordinator.
Integrates Phase 7 Voice System (Microphone STT + Speaker TTS) with
LLM Brain, Intent Classification, Conversation Memory, Executable Tools, and Local RAG.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Ensure workspace root is in sys.path when running directly
workspace_root = str(Path(__file__).resolve().parent.parent)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Attempt to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.brain.intent import IntentClassifier
from app.brain.llm import LLMBrain
from app.memory.conversation import ConversationMemory
from app.tools.registry import create_default_registry, ToolRegistry
from app.rag.retriever import LocalRetriever
from app.rag.loader import index_documents
from app.speech.stt import get_stt_provider, MicrophoneError
from app.speech.tts import get_tts_provider
from app.speech.voice_loop import run_voice_loop
from app.orchestrator import AURAOrchestrator


def configure_logging() -> logging.Logger:
    """Set up structured application logging based on environment variables."""
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("AURA")


def check_subsystems() -> Dict[str, bool]:
    """Verify that all core subsystems can be loaded."""
    status = {}
    subsystems = [
        ("Speech STT", "app.speech.stt"),
        ("Speech TTS", "app.speech.tts"),
        ("Brain Intent", "app.brain.intent"),
        ("Brain LLM", "app.brain.llm"),
        ("Brain Prompts", "app.brain.prompts"),
        ("Memory Conversation", "app.memory.conversation"),
        ("Memory Vector", "app.memory.vector_memory"),
        ("Tools Base", "app.tools.base"),
        ("Tools Registry", "app.tools.registry"),
        ("Tools Calculator", "app.tools.calculator"),
        ("Tools Weather", "app.tools.weather"),
        ("Tools Search", "app.tools.search"),
        ("Tools Files", "app.tools.files"),
        ("RAG Loader", "app.rag.loader"),
        ("RAG Embeddings", "app.rag.embeddings"),
        ("RAG Retriever", "app.rag.retriever"),
        ("Orchestrator", "app.orchestrator"),
    ]

    for label, module_name in subsystems:
        try:
            __import__(module_name)
            status[label] = True
        except Exception as exc:
            logging.getLogger("AURA").error(f"Failed to import {label} ({module_name}): {exc}")
            status[label] = False

    return status


def print_banner() -> None:
    """Display the AURA startup banner."""
    banner = r"""
    ========================================================
       ___   __  __ ___    _      _   ___ 
      /   \ |  \/  | _ \  /_\    /_\ |_ _|
     | - - || |\/| |   / / _ \  / _ \ | | 
     |_|_|_||_|  |_|_|_\/_/ \_\/_/ \_\___|
      AI Unified Response Assistant - v0.8.0 (Phase 8: Full Integration)
    ========================================================
    """
    print(banner)


def run_pipeline_demo(
    classifier: IntentClassifier,
    brain: LLMBrain,
    memory: ConversationMemory,
    retriever: LocalRetriever,
) -> None:
    """Demonstrate end-to-end tool calling, conversation memory, and RAG document grounding."""
    demo_queries = [
        "What is 25 * 48?",
        "What is the weather in London?",
        "What is supervised learning?",
        "What is overfitting?",
        "Explain what machine learning is.",
        "What was my calculation question earlier?",
    ]

    print("\n========================================================")
    print("         AURA Phase 6: RAG & Tools Pipeline Demo         ")
    print("========================================================")

    for query in demo_queries:
        print(f"\nUser > \"{query}\"")

        # 1. Intent Classification
        intent_res = classifier.predict(query)
        intent = intent_res["intent"]
        confidence = intent_res["confidence"]
        print(f"       [Intent: {intent.upper()} ({confidence:.1%})]")

        # 2. Store user message in short-term memory
        memory.add_user_message(query)

        # 3. Check for RAG document context
        retrieved_chunks = retriever.retrieve(query, top_k=2, min_score=0.15)
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            score = top_chunk.get("score", 0.0)
            meta = top_chunk.get("metadata", {})
            src = meta.get("filename", "unknown")
            page = meta.get("page", 1)
            preview = top_chunk.get("text", "").replace("\n", " ")[:90]
            print(f"       [RAG Retrieval: Matched '{src}' (page {page}) | Score: {score:.3f}]")
            print(f"       [Top Chunk Preview: \"{preview}...\"]")

        # 4. Brain executes cognitive loop (Tools / RAG / Persona)
        response = brain.ask(
            user_message=query,
            context=memory,
            detected_intent=intent,
        )

        # 5. Store assistant response in short-term memory
        memory.add_assistant_message(response)

        # 6. Display response
        print(f"AURA > {response}")
        print(f"       [Memory Buffer Size: {len(memory)} messages]")

    print("\n--------------------------------------------------------")
    print("  Complete Multi-Turn Conversation Memory Transcript:")
    print("--------------------------------------------------------")
    for idx, turn in enumerate(memory.get_history(), 1):
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        if len(content) > 75:
            content = content[:72] + "..."
        print(f"  {idx:02d}. [{role:9}] {content}")
    print("========================================================\n")


def interactive_loop(classifier: IntentClassifier, brain: LLMBrain, memory: ConversationMemory) -> None:
    """Run an interactive console loop connecting user input to the LLM Brain with Tools & Memory."""
    provider_name = os.getenv("LLM_PROVIDER", "mock")
    print(f"AURA Brain, Tools, Memory & RAG active (Provider: {provider_name}). Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue

            # 1. Intent & Query Understanding
            intent_res = classifier.predict(user_input)
            intent = intent_res["intent"]
            confidence = intent_res["confidence"]

            # 2. Store user message in short-term memory
            memory.add_user_message(user_input)

            # 3. Send relevant history + current message to LLM Brain
            response = brain.ask(
                user_message=user_input,
                context=memory,
                detected_intent=intent,
            )

            # 4. Store assistant response in short-term memory
            memory.add_assistant_message(response)

            # 5. Display response
            print(f"AURA > {response}\n")

            if intent == "goodbye" and user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
                break

        except (KeyboardInterrupt, EOFError):
            print("\nExiting AURA.")
            break


def main() -> int:
    """Application entry point for AURA."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logger = configure_logging()
    print_banner()
    logger.info("Initializing AURA Phase 7 Voice engine...")

    app_env = os.getenv("APP_ENV", "development")
    provider_name = os.getenv("LLM_PROVIDER", "mock")
    logger.info(f"Environment: {app_env} | LLM Provider: {provider_name}")

    # Subsystem verification
    subsystem_status = check_subsystems()
    all_healthy = all(subsystem_status.values())

    print("\n--- Subsystem Readiness Check ---")
    for name, ready in subsystem_status.items():
        status_icon = "READY" if ready else "FAILED"
        print(f"  [{status_icon:6}] {name}")
    print("---------------------------------\n")

    if not all_healthy:
        logger.error("Some subsystems failed to load.")
        return 1

    # Initialize RAG Retriever & ensure sample documents are indexed
    retriever = LocalRetriever()
    if len(retriever.vector_store) == 0:
        logger.info("Vector index empty. Indexing 'documents/' directory...")
        index_documents("documents")
        retriever.load_index()

    # Initialize Components & Orchestrator
    classifier = IntentClassifier()
    registry = create_default_registry()
    brain = LLMBrain(tool_registry=registry, retriever=retriever)
    memory = ConversationMemory(max_messages=16)

    try:
        stt_provider = get_stt_provider()
    except Exception as exc:
        logger.warning(f"Could not initialize default STT provider: {exc}")
        stt_provider = None

    try:
        tts_provider = get_tts_provider()
    except Exception as exc:
        logger.warning(f"Could not initialize default TTS provider: {exc}")
        tts_provider = None

    orchestrator = AURAOrchestrator(
        classifier=classifier,
        brain=brain,
        memory=memory,
        retriever=retriever,
        tool_registry=registry,
        stt=stt_provider,
        tts=tts_provider,
    )

    # Check CLI options
    cli_args = [arg.lower() for arg in sys.argv[1:]]
    run_demo = "--demo" in cli_args or os.getenv("RUN_DEMO", "false").lower() == "true"
    run_server_mode = "--server" in cli_args or "--api" in cli_args or os.getenv("AURA_SERVER", "false").lower() == "true"
    force_text = "--text" in cli_args
    force_voice = "--voice" in cli_args

    if run_server_mode:
        from app.api.server import run_server
        logger.info("Starting AURA in API Server Mode...")
        run_server()
        return 0

    if run_demo:
        try:
            from demo_integration import run_integration_demo
            run_integration_demo(orchestrator)
        except ImportError:
            run_pipeline_demo(classifier, brain, memory, retriever)

    # Determine interactive mode: voice mode (default) vs text mode
    aura_mode = os.getenv("AURA_MODE", os.getenv("VOICE_MODE", "voice")).strip().lower()
    is_voice_mode = (force_voice or aura_mode in {"voice", "true", "1"}) and not force_text

    if sys.stdin.isatty():
        if is_voice_mode:
            try:
                orchestrator.run_voice_loop()
            except MicrophoneError as mic_err:
                logger.warning(f"Microphone unavailable ({mic_err}). Falling back to text mode.")
                print(f"\n[Notice: Microphone unavailable ({mic_err}). Switching to Text Mode.]\n")
                orchestrator.run_text_loop()
            except Exception as exc:
                logger.error(f"Failed to start voice loop: {exc}. Falling back to text mode.")
                print(f"\n[Notice: Voice loop could not start ({exc}). Switching to Text Mode.]\n")
                orchestrator.run_text_loop()
        else:
            orchestrator.run_text_loop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
