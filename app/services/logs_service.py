from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.detection_log import DetectionLog
from app.schemas.logs import DetectionLogItem
from app.services.camera_manager import CameraManager


def save_log(manager: CameraManager, save_dir: Path) -> dict:
    return manager.save_log(str(save_dir))


def start_fast_analysis(manager: CameraManager, video_path: str, save_dir: Path) -> None:
    manager.run_fast_analysis(video_path, str(save_dir))


def list_detection_logs(db: Session, limit: int = 50, offset: int = 0) -> list[DetectionLogItem]:
    stmt = select(DetectionLog).order_by(DetectionLog.id.desc()).offset(offset).limit(limit)
    records = db.scalars(stmt).all()
    return [
        DetectionLogItem(
            id=row.id,
            timestamp=row.timestamp,
            camera_source=row.camera_source,
            count_a_to_b=row.count_a_to_b,
            count_b_to_a=row.count_b_to_a,
            total_count=row.total_count,
            current_object_count=row.current_object_count,
            details_json=row.details_json,
        )
        for row in records
    ]


def count_detection_logs(db: Session) -> int:
    stmt = select(func.count(DetectionLog.id))
    return int(db.scalar(stmt) or 0)
