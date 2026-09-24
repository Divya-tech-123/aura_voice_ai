"""AURA API Application entrypoint.

Exposes the FastAPI application instance for ASGI servers like Uvicorn:
    uvicorn app.api.main:app --reload

Does not recreate or duplicate the FastAPI application instance.
Re-exports the production-hardened server instance configured in app.api.server.
"""

from app.api.server import app, create_app, run_server

__all__ = ["app", "create_app", "run_server"]

if __name__ == "__main__":
    run_server()
