"""AURA FastAPI Entrypoint for Vercel and ASGI Servers."""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure workspace root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.api.server import (
    TAGS_METADATA,
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    get_rate_limiter,
    get_cors_origins,
    register_error_handlers,
    router,
    get_settings,
)

cfg = get_settings()

# Explicit FastAPI instance declaration for Vercel static AST parser
app = FastAPI(
    title="AURA API",
    description="Production-ready REST API interface for the AURA AI Unified Response Assistant.",
    version="0.11.0",
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 1. Structured logging middleware
app.add_middleware(RequestLoggingMiddleware)

# 2. Rate limiting middleware
rate_limiter = get_rate_limiter(requests_per_minute=cfg.rate_limit_requests_per_minute)
app.add_middleware(
    RateLimitMiddleware,
    limiter=rate_limiter,
    enabled=cfg.rate_limit_enabled,
)

# 3. Payload size limit middleware
app.add_middleware(RequestSizeLimitMiddleware, settings=cfg)

# 4. CORS configuration
allowed_origins = get_cors_origins(cfg)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# 5. Error handlers
register_error_handlers(app)

# 6. Include API routers
app.include_router(router)

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
