"""Vercel entrypoint re-exporting AURA FastAPI application."""
from api.index import app, create_app

__all__ = ["app", "create_app"]
