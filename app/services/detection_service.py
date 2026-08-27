from __future__ import annotations

from app.services.camera_manager import CameraManager


def start_counting(manager: CameraManager, source: str | None) -> None:
    manager.start_detection(source)


def pause_counting(manager: CameraManager) -> dict[str, bool]:
    manager.pause_detection()
    return {"is_paused": manager.is_paused}


def stop_counting(manager: CameraManager) -> None:
    manager.stop_detection()


def reset_counter(manager: CameraManager) -> None:
    manager.reset_counter()
