from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_manager
from app.schemas.common import api_success
from app.schemas.tracking import TrackingStatusResponse
from app.services.camera_manager import CameraManager


tracking_router = APIRouter(prefix="/tracking", tags=["tracking"])


@tracking_router.get("/status")
def tracking_status(manager: CameraManager = Depends(get_manager)):
    stats = manager.get_status()
    payload = TrackingStatusResponse(
        is_running=stats.get("is_running", False),
        is_paused=stats.get("is_paused", False),
        has_source=stats.get("has_source", False),
        infer_device=stats.get("infer_device"),
    )
    return api_success(
        data=payload.model_dump(),
    )
