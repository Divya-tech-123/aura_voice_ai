"""AURA API Package.

Provides FastAPI REST endpoints for the AURA Assistant.
"""

from app.api.server import create_app
from app.api.routes import router

__all__ = ["create_app", "router"]
