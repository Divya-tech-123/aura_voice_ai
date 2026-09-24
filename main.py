"""AURA FastAPI Entrypoint for Vercel and ASGI Servers."""
import os
import sys
from pathlib import Path
from fastapi import FastAPI

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.api.server import create_app

# Instantiate FastAPI application instance for Vercel & Uvicorn
app: FastAPI = create_app()

__all__ = ["app", "create_app"]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
