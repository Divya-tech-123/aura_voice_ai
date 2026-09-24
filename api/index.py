"""Vercel Serverless Function entrypoint for AURA FastAPI backend."""
import sys
from pathlib import Path

# Ensure project root directory is in sys.path for Python imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.api.server import app, create_app

# Export the FastAPI instance for Vercel ASGI serverless handler
__all__ = ["app", "create_app"]
