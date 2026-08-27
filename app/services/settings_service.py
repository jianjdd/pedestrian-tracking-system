from __future__ import annotations

from app.services.camera_manager import CameraManager


def apply_settings(manager: CameraManager, payload: dict) -> None:
    manager.update_settings(payload)


def load_preset(manager: CameraManager, preset_name: str) -> bool:
    return manager.load_preset(preset_name)
