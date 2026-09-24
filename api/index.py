"""Vercel Serverless Function entrypoint (api/index.py)."""
import os
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import app

__all__ = ["app"]
