"""Foundation tests verifying all modules and packages are importable."""

import importlib
import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "app",
        "app.main",
        "app.speech",
        "app.speech.stt",
        "app.speech.tts",
        "app.brain",
        "app.brain.llm",
        "app.brain.intent",
        "app.brain.prompts",
        "app.memory",
        "app.memory.conversation",
        "app.memory.vector_memory",
        "app.tools",
        "app.tools.base",
        "app.tools.registry",
        "app.tools.calculator",
        "app.tools.weather",
        "app.tools.search",
        "app.tools.files",
        "app.rag",
        "app.rag.loader",
        "app.rag.embeddings",
        "app.rag.retriever",
        "app.api",
        "app.api.schemas",
        "app.api.dependencies",
        "app.api.routes",
        "app.api.server",
    ],
)
def test_module_imports(module_name: str):
    """Verify that all core modules import cleanly without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None
