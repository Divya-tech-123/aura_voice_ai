"""FastAPI server application factory and startup entrypoint.

Configures production-hardened CORS, security headers, rate limiting,
request size limits, structured audit logging, centralized error handling,
and routing for AURA.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure workspace root is in sys.path
workspace_root = str(Path(__file__).resolve().parent.parent.parent)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from app.config import get_settings, Settings
from app.api.routes import router
from app.api.errors import register_error_handlers
from app.api.logging_config import RequestLoggingMiddleware, SensitiveDataFilter
from app.api.rate_limiter import RateLimitMiddleware, get_rate_limiter

logger = logging.getLogger("AURA.API.Server")

# Apply sensitive data filter to root logging handlers
for handler in logging.getLogger().handlers:
    handler.addFilter(SensitiveDataFilter())


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request payload size limits and prevent resource exhaustion."""

    def __init__(self, app, settings: Optional[Settings] = None):
        super().__init__(app)
        self.settings = settings or get_settings()

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                is_voice_route = request.url.path.endswith("/voice")
                is_doc_upload_route = request.url.path.endswith("/documents/upload")

                if is_voice_route:
                    limit = self.settings.max_audio_size
                elif is_doc_upload_route:
                    limit = self.settings.max_document_size
                else:
                    limit = self.settings.max_content_length

                if length > limit:
                    logger.warning(
                        f"Rejected request exceeding size limit: {length} bytes for {request.url.path} (limit: {limit} bytes)"
                    )
                    return JSONResponse(
                        status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                        content={
                            "error": f"Request payload too large (max {limit} bytes).",
                            "detail": f"Request payload too large (max {limit} bytes).",
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


def get_cors_origins(settings: Optional[Settings] = None) -> List[str]:
    """Retrieve validated CORS origins from centralized settings.

    Never permits unrestricted wildcards ('*') in production.
    """
    cfg = settings or get_settings()
    origins = list(cfg.cors_origins)

    if cfg.is_production and "*" in origins:
        logger.warning("Wildcard CORS detected in production mode. Restricting to empty list for safety.")
        origins = [o for o in origins if o != "*"]

    return origins


# OpenAPI Tags Metadata for Swagger documentation
TAGS_METADATA = [
    {
        "name": "Health",
        "description": "API status, readiness, and version verification endpoints.",
    },
    {
        "name": "Chat",
        "description": "Autonomous conversational agent cognitive loop endpoints.",
    },
    {
        "name": "Voice",
        "description": "End-to-end voice pipeline endpoints (Speech-to-Text -> Agent -> Text-to-Speech).",
    },
    {
        "name": "Documents",
        "description": "Knowledge base document management and RAG indexing endpoints.",
    },
]


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the production-hardened FastAPI application instance."""
    cfg = settings or get_settings()

    app = FastAPI(
        title="AURA API",
        description="Production-ready REST API interface for the AURA AI Unified Response Assistant.",
        version="0.11.0",
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. Structured request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # 2. In-memory sliding window rate limiting
    rate_limiter = get_rate_limiter(requests_per_minute=cfg.rate_limit_requests_per_minute)
    app.add_middleware(
        RateLimitMiddleware,
        limiter=rate_limiter,
        enabled=cfg.rate_limit_enabled,
    )

    # 3. Request payload size limit enforcement
    app.add_middleware(RequestSizeLimitMiddleware, settings=cfg)

    # 4. CORS configuration (explicit origins only)
    allowed_origins = get_cors_origins(cfg)
    logger.info(f"Configuring CORS with origins: {allowed_origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
    )

    # 5. Centralized API error handling
    register_error_handlers(app)

    # 6. Mount API router
    app.include_router(router)

    return app


# Module-level application instance for ASGI servers (e.g. uvicorn app.api.server:app)
app = create_app()


def run_server():
    """Run the API server with uvicorn using centralized configuration."""
    import uvicorn

    cfg = get_settings()
    logger.info(f"Starting AURA API server on http://{cfg.api_host}:{cfg.api_port} (Env: {cfg.app_env})")
    uvicorn.run("app.api.server:app", host=cfg.api_host, port=cfg.api_port, reload=cfg.is_development)


if __name__ == "__main__":
    run_server()
