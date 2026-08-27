from __future__ import annotations

from typing import Generator

from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.camera_manager import CameraManager, get_camera_manager


def get_manager() -> CameraManager:
    return get_camera_manager()


def get_db_session() -> Generator[Session, None, None]:
    yield from get_db()
