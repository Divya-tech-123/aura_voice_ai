"""Dependency injection providers for AURA FastAPI application.

Reuses the central AuraAgent instance across API requests without duplicating
subsystems (LLM, ConversationMemory, Planner, ToolRegistry, LocalRetriever).
"""

import os
import logging
from pathlib import Path
from typing import Optional
from app.config import get_settings
from app.agent.agent import AuraAgent
from app.speech.stt import BaseSpeechToText, get_stt_provider
from app.speech.tts import BaseTextToSpeech, get_tts_provider

logger = logging.getLogger("AURA.API.Dependencies")

_agent_instance: Optional[AuraAgent] = None
_stt_instance: Optional[BaseSpeechToText] = None
_tts_instance: Optional[BaseTextToSpeech] = None


def get_agent() -> AuraAgent:
    """Provide the shared AuraAgent instance for request processing.

    Lazy-initializes the existing AuraAgent on first call using centralized settings.
    Can be overridden in tests via app.dependency_overrides[get_agent].
    """
    global _agent_instance
    if _agent_instance is None:
        cfg = get_settings()
        logger.info(f"Initializing shared AuraAgent (max_steps={cfg.max_agent_steps}) for FastAPI API layer...")
        _agent_instance = AuraAgent(max_steps=cfg.max_agent_steps)
    return _agent_instance


def set_agent(agent: Optional[AuraAgent]) -> None:
    """Explicitly set or reset the shared AuraAgent instance (e.g. for testing)."""
    global _agent_instance
    _agent_instance = agent


def get_stt() -> BaseSpeechToText:
    """Provide the shared BaseSpeechToText instance for request processing.

    Can be overridden in tests via app.dependency_overrides[get_stt].
    """
    global _stt_instance
    if _stt_instance is None:
        logger.info("Initializing shared STT provider for FastAPI API layer...")
        _stt_instance = get_stt_provider()
    return _stt_instance


def set_stt(provider: Optional[BaseSpeechToText]) -> None:
    """Explicitly set or reset the shared STT provider (e.g. for testing)."""
    global _stt_instance
    _stt_instance = provider


def get_tts() -> BaseTextToSpeech:
    """Provide the shared BaseTextToSpeech instance for request processing.

    Can be overridden in tests via app.dependency_overrides[get_tts].
    """
    global _tts_instance
    if _tts_instance is None:
        logger.info("Initializing shared TTS provider for FastAPI API layer...")
        _tts_instance = get_tts_provider()
    return _tts_instance


def set_tts(provider: Optional[BaseTextToSpeech]) -> None:
    """Explicitly set or reset the shared TTS provider (e.g. for testing)."""
    global _tts_instance
    _tts_instance = provider


_documents_dir: Optional[Path] = None


def get_documents_dir() -> Path:
    """Provide the controlled documents directory for document uploads.

    Can be overridden in tests via app.dependency_overrides[get_documents_dir].
    """
    global _documents_dir
    if _documents_dir is not None:
        return _documents_dir
    path = get_settings().documents_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_documents_dir(doc_dir: Optional[Path]) -> None:
    """Explicitly set or reset the documents directory (e.g. for testing)."""
    global _documents_dir
    _documents_dir = doc_dir


