from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db_session, get_manager
from app.core.config import settings
from app.schemas.common import api_success
from app.schemas.logs import AnalysisRequest
from app.services.camera_manager import CameraManager
from app.services import logs_service


logs_router = APIRouter(prefix="/logs", tags=["logs"])


@logs_router.post("/save")
def save_log(manager: CameraManager = Depends(get_manager)):
    result = logs_service.save_log(manager, settings.LOGS_DIR)
    if result.get("success"):
        return api_success(data=result, message="log saved")
    raise HTTPException(status_code=400, detail=result.get("message", "log save failed"))


@logs_router.get("/download/{filename}")
def download_log(filename: str):
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="invalid filename")

    file_path = settings.LOGS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(str(file_path), filename=safe_name, media_type="application/octet-stream")


@logs_router.get("/list")
def list_logs():
    files = os.listdir(settings.LOGS_DIR) if settings.LOGS_DIR.exists() else []
    return api_success(data={"files": sorted(files, reverse=True)})


@logs_router.get("/history")
def list_log_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db_session),
):
    items = logs_service.list_detection_logs(db=db, limit=limit, offset=offset)
    total = logs_service.count_detection_logs(db=db)
    return api_success(data={"items": [item.model_dump() for item in items], "total": total, "limit": limit, "offset": offset})


@logs_router.post("/analysis/start")
def start_analysis(data: AnalysisRequest, manager: CameraManager = Depends(get_manager)):
    if not data.video_path:
        raise HTTPException(status_code=400, detail="video path is required")
    logs_service.start_fast_analysis(manager, data.video_path, settings.LOGS_DIR)
    return api_success(message="analysis started")


@logs_router.get("/analysis/progress")
def analysis_progress(manager: CameraManager = Depends(get_manager)):
    return api_success(data=manager.get_analysis_progress())
