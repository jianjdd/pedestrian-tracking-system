from __future__ import annotations

from fastapi import APIRouter

from app.api.detection import detection_router
from app.api.logs import logs_router
from app.api.settings import settings_router
from app.api.tracking import tracking_router
from app.api.video import video_router


api_router = APIRouter()
api_router.include_router(video_router)
api_router.include_router(detection_router)
api_router.include_router(tracking_router)
api_router.include_router(settings_router)
api_router.include_router(logs_router)

