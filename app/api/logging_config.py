"""Structured logging configuration and sensitive data redaction for AURA.

Enforces security best practices:
- Logs request lifecycle (received, completed with duration).
- Logs tool execution and RAG retrieval metadata safely.
- Intercepts and redacts API keys, bearer tokens, passwords, and raw audio payloads.
- Prevents logging of hidden chain-of-thought or sensitive internal details.
"""

import re
import time
import logging
from typing import Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("AURA.API.Logging")

# Regex patterns for sensitive information redaction
SENSITIVE_PATTERNS = [
    # API keys and secrets in query strings or key-value formats
    (re.compile(r"(?i)(api[-_]?key|secret[-_]?key|password|token|auth|authorization)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-.]{6,})['\"]?"), r"\1=***REDACTED***"),
    # Bearer tokens in headers or logs
    (re.compile(r"(?i)Bearer\s+[a-zA-Z0-9_\-.\~+/=]{8,}"), "Bearer ***REDACTED***"),
    # OpenAI key pattern (sk-...)
    (re.compile(r"\bsk-[a-zA-Z0-9T3BlbkFJ]{20,}\b"), "sk-***REDACTED***"),
    # Google API key pattern (AIza...)
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"), "AIza***REDACTED***"),
    # Base64 data URIs for audio or files
    (re.compile(r"data:audio/[a-zA-Z0-9.-]+;base64,[a-zA-Z0-9+/=]{30,}"), "data:audio/...;base64,[REDACTED_AUDIO]"),
]


def redact_sensitive_text(text: str) -> str:
    """Mask sensitive keys, tokens, and binary payloads from a string."""
    if not isinstance(text, str):
        return text
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class SensitiveDataFilter(logging.Filter):
    """Logging filter that redacts secrets, credentials, and audio data from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: redact_sensitive_text(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_sensitive_text(str(a)) if isinstance(a, str) else a for a in record.args)
        return True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to produce structured request audit logs without exposing secrets or bodies."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        # Log request received
        logger.info(f"Request received: {request.method} {request.url.path} from {client_ip}")

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration_ms:.2f}ms"
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            safe_error = redact_sensitive_text(str(exc))
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"duration={duration_ms:.2f}ms error={safe_error}"
            )
            raise


def log_tool_execution(tool_name: str, status: str, duration_ms: Optional[float] = None) -> None:
    """Log tool execution with safe sanitized metadata."""
    duration_str = f" in {duration_ms:.2f}ms" if duration_ms is not None else ""
    logger.info(f"Tool execution: tool={tool_name} status={status}{duration_str}")


def log_rag_retrieval(query_length: int, results_count: int, top_score: Optional[float] = None) -> None:
    """Log RAG retrieval summary safely without leaking document secrets or full queries."""
    score_str = f" top_score={top_score:.3f}" if top_score is not None else ""
    logger.info(
        f"RAG retrieval: query_length={query_length} results_count={results_count}{score_str}"
    )
