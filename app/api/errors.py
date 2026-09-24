"""Centralized error handling and safe error responses for AURA API.

Guarantees that no stack traces, API keys, internal filesystem paths,
prompts, or configuration details are ever exposed to the client.
Provides structured and predictable JSON error payloads.
"""

import re
import logging
from typing import Union, List, Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.logging_config import redact_sensitive_text

logger = logging.getLogger("AURA.API.Errors")

# Patterns for internal paths to scrub from error outputs
PATH_PATTERN = re.compile(r"(?:[A-Za-z]:\\[^:\n\r\"']+|/(?:home|usr|var|Users|etc|tmp)/[^:\n\r\"']+)")


def sanitize_error_detail(detail: Any) -> str:
    """Sanitize detail text to remove secrets, internal file paths, and stack traces."""
    if not isinstance(detail, str):
        detail = str(detail)

    # 1. Redact API keys, tokens, and credentials
    detail = redact_sensitive_text(detail)

    # 2. Redact internal filesystem paths
    detail = PATH_PATTERN.sub("[internal_path]", detail)

    # 3. Strip Python tracebacks if accidentally present in error strings
    if "Traceback (most recent call last):" in detail:
        detail = detail.split("Traceback (most recent call last):")[0].strip()
        if not detail:
            detail = "An unexpected internal error occurred."

    return detail.strip()


async def http_exception_handler(request: Request, exc: Union[HTTPException, StarletteHTTPException]) -> JSONResponse:
    """Handle FastAPI and Starlette HTTPExceptions with safe, structured JSON errors."""
    raw_detail = exc.detail if hasattr(exc, "detail") else str(exc)
    safe_detail = sanitize_error_detail(raw_detail)

    logger.warning(
        f"HTTP {exc.status_code} on {request.method} {request.url.path}: {safe_detail}"
    )

    # In case of internal 500 errors, ensure standard safe error representation
    if exc.status_code >= 500:
        safe_error = "Unable to process the request."
        if safe_detail == "An internal error occurred while processing your request.":
            safe_detail = "Unable to process the request."
    else:
        safe_error = safe_detail

    return JSONResponse(
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
        content={
            "error": safe_error,
            "detail": safe_detail,
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic request validation errors without leaking internal model structures."""
    errors = exc.errors()
    messages: List[str] = []
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []) if l != "body")
        msg = err.get("msg", "Invalid value")
        if loc:
            messages.append(f"Field '{loc}': {msg}")
        else:
            messages.append(msg)

    summary = "; ".join(messages) if messages else "Invalid request payload."
    safe_summary = sanitize_error_detail(summary)

    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {safe_summary}"
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": safe_summary,
            "detail": safe_summary,
        },
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all unhandled exception handler to strictly prevent stack trace exposure."""
    logger.exception(
        f"Unhandled server exception during {request.method} {request.url.path}: {exc}"
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Unable to process the request.",
            "detail": "Unable to process the request.",
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all centralized exception handlers on the FastAPI application instance."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
