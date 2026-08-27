from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.core.config import settings
from app.services.camera_manager import CameraManager


def frame_stream(manager: CameraManager):
    """MJPEG 推流：检测线程产出新帧时即时推送，无新帧时定期重发以避免浏览器超时。"""
    last_sent = None
    last_send_time = 0.0
    resend_interval = 0.3  # 无新帧时每隔 0.3 秒重发一次，保活浏览器连接
    while True:
        frame_bytes = manager.get_frame()
        now = time.time()
        if frame_bytes:
            if frame_bytes is not last_sent or now - last_send_time >= resend_interval:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                last_sent = frame_bytes
                last_send_time = now
        time.sleep(0.01)


def save_upload(file: UploadFile, save_dir: Path) -> str:
    save_path = save_dir / file.filename
    content = file.file.read()
    with save_path.open("wb") as output:
        output.write(content)
    return str(save_path)


def list_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    
    # helper functon to scan a directory for pt files
    def scan_dir(directory: Path):
        if directory.is_dir():
            for file_name in os.listdir(directory):
                if file_name.endswith(".pt") and not file_name.startswith("osnet"):
                    model_path = directory / file_name
                    size_mb = round(model_path.stat().st_size / 1024 / 1024, 1)
                    models.append({"name": file_name, "path": str(model_path), "size_mb": size_mb})

    # Scan weights dir
    scan_dir(settings.WEIGHTS_DIR)
    
    # Scan default upload weights dir
    scan_dir(settings.UPLOAD_WEIGHTS_DIR)

    # Backward compatibility: keep scanning legacy uploads root.
    scan_dir(settings.UPLOADS_DIR)
    
    # remove duplicate paths if any
    unique_models = []
    seen_paths = set()
    for m in models:
        norm_path = os.path.normpath(m["path"])
        if norm_path not in seen_paths:
            seen_paths.add(norm_path)
            unique_models.append(m)

    return unique_models

def list_videos() -> list[dict[str, Any]]:
    videos = []
    if settings.UPLOAD_VIDEOS_DIR.is_dir():
        for file_name in os.listdir(settings.UPLOAD_VIDEOS_DIR):
            if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                video_path = settings.UPLOAD_VIDEOS_DIR / file_name
                size_mb = round(video_path.stat().st_size / 1024 / 1024, 1)
                videos.append({"name": file_name, "path": str(video_path), "size_mb": size_mb})

    # Backward compatibility: include videos from legacy uploads root.
    if settings.UPLOADS_DIR.is_dir():
        for file_name in os.listdir(settings.UPLOADS_DIR):
            if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                video_path = settings.UPLOADS_DIR / file_name
                size_mb = round(video_path.stat().st_size / 1024 / 1024, 1)
                videos.append({"name": file_name, "path": str(video_path), "size_mb": size_mb})

    unique_videos = []
    seen_paths = set()
    for v in videos:
        norm_path = os.path.normpath(v["path"])
        if norm_path not in seen_paths:
            seen_paths.add(norm_path)
            unique_videos.append(v)
    return unique_videos
