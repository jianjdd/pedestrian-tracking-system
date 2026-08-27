from __future__ import annotations

from pathlib import Path
from typing import Any, List

try:
    from pydantic import field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    HAS_PYDANTIC_SETTINGS = True
except ModuleNotFoundError:
    # Compatibility fallback for environments that only have pydantic installed.
    from pydantic.v1 import BaseSettings, validator

    HAS_PYDANTIC_SETTINGS = False
    SettingsConfigDict = dict


class Settings(BaseSettings):
    """Centralized runtime configuration powered by pydantic-settings."""

    if HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

    APP_NAME: str = "Pedestrian Tracking"
    API_PREFIX: str = "/api"
    DATABASE_URL: str = "mysql+pymysql://user:password@localhost/pedestrian_tracking"
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024

    if HAS_PYDANTIC_SETTINGS:
        @field_validator("BACKEND_CORS_ORIGINS", mode="before")
        @classmethod
        def parse_origins(cls, value: Any) -> List[str]:
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            if isinstance(value, list):
                return value
            return ["http://localhost:5173", "http://127.0.0.1:5173"]
    else:
        @validator("BACKEND_CORS_ORIGINS", pre=True)
        def parse_origins(cls, value: Any) -> List[str]:
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            if isinstance(value, list):
                return value
            return ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def BASE_DIR(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / "data"

    @property
    def UPLOADS_DIR(self) -> Path:
        return self.DATA_DIR / "uploads"

    @property
    def UPLOAD_WEIGHTS_DIR(self) -> Path:
        return self.UPLOADS_DIR / "weights"

    @property
    def UPLOAD_VIDEOS_DIR(self) -> Path:
        return self.UPLOADS_DIR / "videos"

    @property
    def LOGS_DIR(self) -> Path:
        return self.DATA_DIR / "logs"

    @property
    def WEIGHTS_DIR(self) -> Path:
        return self.BASE_DIR / "weights"

    # Backward-compatible aliases used by existing service code.
    @property
    def UPLOAD_FOLDER(self) -> str:
        return str(self.UPLOADS_DIR)

    @property
    def LOG_FOLDER(self) -> str:
        return str(self.LOGS_DIR)


settings = Settings()
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
