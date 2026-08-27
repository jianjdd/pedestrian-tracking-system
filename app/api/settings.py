from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_manager
from app.schemas.common import api_success
from app.schemas.settings import PresetRequest, SetLineRequest, UpdateSettingsRequest
from app.services.camera_manager import CameraManager
from app.services import settings_service


settings_router = APIRouter(prefix="/settings", tags=["settings"])


@settings_router.get("")
def get_settings(manager: CameraManager = Depends(get_manager)):
    return api_success(data=manager.get_settings())


@settings_router.post("")
def update_settings(data: UpdateSettingsRequest, manager: CameraManager = Depends(get_manager)):
    settings_service.apply_settings(manager, data.model_dump(exclude_none=True))
    return api_success(message="settings updated")


@settings_router.post("/preset")
def load_preset(data: PresetRequest, manager: CameraManager = Depends(get_manager)):
    if settings_service.load_preset(manager, data.name):
        return api_success(data=manager.get_settings(), message="preset loaded")
    raise HTTPException(status_code=404, detail=f"unknown preset: {data.name}")


@settings_router.post("/save")
def save_config(manager: CameraManager = Depends(get_manager)):
    manager.save_config()
    return api_success(message="settings saved")


@settings_router.post("/line/set")
def set_line(data: SetLineRequest, manager: CameraManager = Depends(get_manager)):
    manager.set_line(data.point1, data.point2, reset_counts=data.reset_counts)
    return api_success(message="line set")


@settings_router.post("/line/clear")
def clear_line(manager: CameraManager = Depends(get_manager)):
    manager.clear_line()
    return api_success(message="line cleared")


@settings_router.get("/line")
def get_line(manager: CameraManager = Depends(get_manager)):
    return api_success(data={"line": manager.get_line_points()})
