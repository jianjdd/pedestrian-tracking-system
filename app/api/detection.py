from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_manager
from app.schemas.common import api_success
from app.schemas.detection import SetClassesRequest, StartCountingRequest
from app.services.camera_manager import CameraManager
from app.services import detection_service


detection_router = APIRouter(prefix="/detection", tags=["detection"])


@detection_router.post("/count/start")
def start_counting(data: StartCountingRequest = StartCountingRequest(), manager: CameraManager = Depends(get_manager)):
    detection_service.start_counting(manager, data.source)
    return api_success(message="started")


@detection_router.post("/count/pause")
def pause_counting(manager: CameraManager = Depends(get_manager)):
    return api_success(data=detection_service.pause_counting(manager), message="pause toggled")


@detection_router.post("/count/stop")
def stop_counting(manager: CameraManager = Depends(get_manager)):
    detection_service.stop_counting(manager)
    return api_success(message="stopped")


@detection_router.post("/count/reset")
def reset_count(manager: CameraManager = Depends(get_manager)):
    detection_service.reset_counter(manager)
    return api_success(message="counter reset")


@detection_router.get("/stats")
def get_stats(manager: CameraManager = Depends(get_manager)):
    return api_success(data=manager.get_stats())


@detection_router.get("/stats/detailed")
def get_detailed_stats(manager: CameraManager = Depends(get_manager)):
    return api_success(data=manager.get_detailed_stats())


@detection_router.get("/stats/history")
def get_stats_history(manager: CameraManager = Depends(get_manager)):
    return api_success(data=manager.get_stats_history())


@detection_router.get("/status")
def get_status(manager: CameraManager = Depends(get_manager)):
    return api_success(data=manager.get_status())


@detection_router.get("/classes")
def get_classes(manager: CameraManager = Depends(get_manager)):
    return api_success(data=manager.get_classes())


@detection_router.post("/classes")
def set_classes(data: SetClassesRequest, manager: CameraManager = Depends(get_manager)):
    manager.set_enabled_classes(data.enabled)
    return api_success(message="classes updated")
