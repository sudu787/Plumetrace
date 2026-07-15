"""Application configuration for PlumeTrace."""

import logging
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_config_logger = logging.getLogger("plumetrace.config")


class Settings(BaseSettings):
    """Environment-backed runtime settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    MQTT_BROKER_HOST: str = Field(default="localhost")
    MQTT_BROKER_PORT: int = Field(default=1883, ge=1, le=65535)
    MQTT_TOPIC: str = Field(default="city/airquality/+")
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./plumetrace.db")
    ALLOWED_ORIGINS: str = Field(
        default=(
            "http://localhost:3000,"
            "http://localhost:5173,"
            "http://localhost:8000,"
            "http://127.0.0.1:3000,"
            "http://127.0.0.1:5173,"
            "http://127.0.0.1:8000"
        )
    )
    MQTT_CLIENT_ID: str = Field(default="plumetrace-backend")
    MQTT_CONNECT_RETRIES: int = Field(default=10, ge=1)
    MQTT_RETRY_BACKOFF_SECONDS: float = Field(default=2.0, gt=0.0)
    LOG_LEVEL: str = Field(default="INFO")

    PLUMETRACE_API_KEY: str = Field(default="dev-insecure-key")
    MQTT_USERNAME: str | None = Field(default=None)
    MQTT_PASSWORD: str | None = Field(default=None)

    @property
    def allowed_origins(self) -> list[str]:
        """Return parsed CORS origins from a comma-separated environment value."""
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    s = Settings()
    if s.PLUMETRACE_API_KEY in ("dev-insecure-key", "test-api-key-123"):
        _config_logger.warning(
            "SECURITY: PLUMETRACE_API_KEY is set to a default insecure value. "
            "Set a strong key via environment variable before deploying."
        )
    return s


settings = get_settings()
