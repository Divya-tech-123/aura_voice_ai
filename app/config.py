"""Centralized configuration module for AURA.

Loads settings from environment variables and local .env files.
Provides strong typing and validation via Pydantic.
Ensures secrets and sensitive parameters are never hardcoded.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Set
from pydantic import BaseModel, Field, model_validator

# Load .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("AURA.Config")


class Settings(BaseModel):
    """Application configuration settings for AURA backend."""

    # 1. Environment & Deployment
    app_env: str = Field(
        default_factory=lambda: os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).lower().strip()
    )
    api_host: str = Field(
        default_factory=lambda: os.getenv("API_HOST", "127.0.0.1").strip()
    )
    api_port: int = Field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000"))
    )

    # 2. CORS & Frontend Origin
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            orig.strip()
            for orig in os.getenv(
                "CORS_ORIGINS",
                os.getenv(
                    "FRONTEND_ORIGIN",
                    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
                ),
            ).split(",")
            if orig.strip()
        ]
    )

    # 3. LLM Provider & Secrets
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "mock").lower().strip()
    )
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY"))
    )
    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    weather_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("WEATHER_API_KEY", os.getenv("OPENWEATHER_API_KEY"))
    )
    search_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("SEARCH_API_KEY", os.getenv("SERPAPI_API_KEY"))
    )

    # 4. Request & Upload Size Limits
    max_content_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONTENT_LENGTH", str(64 * 1024)))
    )
    max_audio_size: int = Field(
        default_factory=lambda: int(os.getenv("MAX_AUDIO_SIZE", str(10 * 1024 * 1024)))
    )
    max_document_size: int = Field(
        default_factory=lambda: int(os.getenv("MAX_DOCUMENT_SIZE", str(10 * 1024 * 1024)))
    )
    allowed_doc_extensions: Set[str] = Field(
        default_factory=lambda: {".pdf", ".docx", ".txt"}
    )
    documents_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("AURA_DOCUMENTS_DIR", "documents")).resolve()
    )

    # 5. Autonomous Agent Limits
    max_agent_steps: int = Field(
        default_factory=lambda: int(os.getenv("MAX_AGENT_STEPS", os.getenv("AGENT_MAX_STEPS", "10")))
    )

    # 6. Rate Limiting Configuration
    rate_limit_enabled: bool = Field(
        default_factory=lambda: os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    rate_limit_requests_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "100"))
    )

    # 7. Logging Configuration
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    @model_validator(mode="after")
    def validate_production_constraints(self):
        """Validate production constraints such as disallowing wildcards in CORS."""
        cleaned = [o.strip() for o in self.cors_origins if o.strip()]
        if self.app_env == "production":
            if "*" in cleaned:
                logger.warning(
                    "Wildcard CORS '*' is prohibited in production. Stripping wildcard origin."
                )
                cleaned = [o for o in cleaned if o != "*"]
            if not cleaned:
                logger.warning("No explicit CORS origins configured for production.")
        self.cors_origins = cleaned
        return self

    @property
    def is_production(self) -> bool:
        """Return True if running in production mode."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Return True if running in development mode."""
        return self.app_env == "development"

    @property
    def is_testing(self) -> bool:
        """Return True if running in test mode."""
        return self.app_env in ("testing", "test")


_cached_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Retrieve the singleton Settings instance.

    Args:
        reload: If True, re-instantiates Settings from current environment variables.
    """
    global _cached_settings
    if _cached_settings is None or reload:
        _cached_settings = Settings()
    return _cached_settings


def set_settings(settings: Optional[Settings]) -> None:
    """Explicitly set or reset the settings instance (e.g. for testing)."""
    global _cached_settings
    _cached_settings = settings
