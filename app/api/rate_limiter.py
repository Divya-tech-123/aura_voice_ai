"""Rate limiting architecture for AURA API.

Provides an extensible abstraction for rate limiting and an efficient in-memory
sliding-window implementation suited for standalone and development environments
without external infrastructure dependencies (such as Redis).
"""

import time
import threading
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("AURA.API.RateLimiter")


class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    def check(self, key: str) -> Tuple[bool, int, float]:
        """Check whether a request for the given key is allowed.

        Args:
            key: Unique identifier for the client (e.g. client IP address).

        Returns:
            Tuple of (is_allowed: bool, remaining_requests: int, retry_after_seconds: float).
        """
        pass

    @abstractmethod
    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limit history for a specific key or all keys."""
        pass


class InMemoryRateLimiter(BaseRateLimiter):
    """Thread-safe sliding-window in-memory rate limiter.

    Maintains a list of timestamps for each key within a specified window.
    Automatically purges expired records to prevent unbounded memory growth.
    """

    def __init__(self, requests_per_minute: int = 100, window_seconds: float = 60.0) -> None:
        self.limit = requests_per_minute
        self.window = window_seconds
        self._records: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def _purge_expired(self, current_time: float) -> None:
        """Periodic cleanup of inactive clients to prevent memory leaks."""
        cutoff = current_time - self.window
        keys_to_remove = []
        for key, timestamps in self._records.items():
            valid_timestamps = [t for t in timestamps if t > cutoff]
            if valid_timestamps:
                self._records[key] = valid_timestamps
            else:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self._records.pop(key, None)

    def check(self, key: str) -> Tuple[bool, int, float]:
        """Check if request is within rate limits.

        Returns:
            (is_allowed, remaining, retry_after)
        """
        now = time.monotonic()
        with self._lock:
            # Perform background cleanup every 60 seconds
            if now - self._last_cleanup > 60.0:
                self._purge_expired(now)
                self._last_cleanup = now

            cutoff = now - self.window
            timestamps = self._records.setdefault(key, [])
            # Filter timestamps within active window
            active_timestamps = [t for t in timestamps if t > cutoff]
            self._records[key] = active_timestamps

            if len(active_timestamps) >= self.limit:
                oldest = active_timestamps[0]
                retry_after = max(0.0, (oldest + self.window) - now)
                return False, 0, retry_after

            # Allow request and record timestamp
            active_timestamps.append(now)
            remaining = self.limit - len(active_timestamps)
            return True, remaining, 0.0

    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limiter state for a specific key or all keys."""
        with self._lock:
            if key is not None:
                self._records.pop(key, None)
            else:
                self._records.clear()


# Default module-level rate limiter
_default_rate_limiter: Optional[BaseRateLimiter] = None


def get_rate_limiter(requests_per_minute: int = 100) -> BaseRateLimiter:
    """Get or create the global rate limiter instance."""
    global _default_rate_limiter
    if _default_rate_limiter is None:
        _default_rate_limiter = InMemoryRateLimiter(requests_per_minute=requests_per_minute)
    return _default_rate_limiter


def set_rate_limiter(rate_limiter: Optional[BaseRateLimiter]) -> None:
    """Explicitly set or reset the rate limiter instance (e.g. for testing)."""
    global _default_rate_limiter
    _default_rate_limiter = rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware applying rate limits per client IP."""

    # Paths exempt from rate limiting
    EXEMPT_PATHS = {"/api/health", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app, limiter: Optional[BaseRateLimiter] = None, enabled: bool = True):
        super().__init__(app)
        self.limiter = limiter or get_rate_limiter()
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # Skip exempted paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Determine client identifier (IP address)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "127.0.0.1"

        allowed, remaining, retry_after = self.limiter.check(client_ip)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for client {client_ip} on {request.url.path}. "
                f"Retry-After: {int(retry_after) + 1}s"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(int(retry_after) + 1)},
                content={
                    "error": "Rate limit exceeded. Please try again later.",
                    "detail": "Rate limit exceeded. Please try again later.",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(getattr(self.limiter, "limit", 100))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
