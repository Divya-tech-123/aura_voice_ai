"""AURA FastAPI Entrypoint inside app directory."""
import sys
from pathlib import Path
from fastapi import FastAPI

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.api.server import create_app

app: FastAPI = create_app()

__all__ = ["app", "create_app"]
