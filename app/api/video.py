from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import get_manager
from app.core.config import settings
from app.schemas.common import api_success
from app.schemas.video import ModelPathRequest, SourceSetRequest
from app.services.camera_manager import CameraManager
from app.services.video_service import frame_stream, list_models, list_videos


video_router = APIRouter(prefix="/video", tags=["video"])


@video_router.get("/mjpeg_feed")
def mjpeg_feed(manager: CameraManager = Depends(get_manager)):
    return StreamingResponse(frame_stream(manager), media_type="multipart/x-mixed-replace; boundary=frame")


@video_router.get("/video_feed")
def video_feed(manager: CameraManager = Depends(get_manager)):
    return StreamingResponse(frame_stream(manager), media_type="multipart/x-mixed-replace; boundary=frame")


@video_router.post("/model/load")
async def load_model(
    manager: CameraManager = Depends(get_manager),
    model: Optional[UploadFile] = File(None),
    path: Optional[str] = None,
):
    if model and model.filename:
        save_path = settings.UPLOAD_WEIGHTS_DIR / model.filename
        content = await model.read()
        with save_path.open("wb") as output:
            output.write(content)
        if manager.load_model(str(save_path)):
            return api_success(data=manager.get_model_info(), message="model loaded")
        raise HTTPException(status_code=400, detail="model load failed")

    if path:
        if manager.load_model(path):
            return api_success(data=manager.get_model_info(), message="model loaded")
        raise HTTPException(status_code=400, detail="model load failed")

    raise HTTPException(status_code=400, detail="model file or path is required")


@video_router.get("/model/list")
def get_model_list():
    return api_success(data={"models": list_models()})


@video_router.post("/model/load_local")
def load_model_local(data: ModelPathRequest, manager: CameraManager = Depends(get_manager)):
    if not os.path.exists(data.path):
        raise HTTPException(status_code=404, detail=f"model file not found: {data.path}")

    if manager.load_model(data.path):
        return api_success(data=manager.get_model_info(), message="model loaded")
    raise HTTPException(status_code=400, detail="model load failed")


@video_router.get("/model/info")
def model_info(manager: CameraManager = Depends(get_manager)):
    return api_success(data=manager.get_model_info())


@video_router.get("/sources")
def get_sources(manager: CameraManager = Depends(get_manager)):
    cameras = manager.detect_cameras(blocking=False)
    return api_success(data={
        "cameras": cameras,
        "scanning": manager.is_camera_scan_in_progress(),
        "videos": list_videos()
    })


@video_router.get("/videos")
def get_videos():
    return api_success(data={"videos": list_videos()})


@video_router.get("/cameras")
def detect_cameras(manager: CameraManager = Depends(get_manager)):
    cameras = manager.detect_cameras(blocking=False)
    return api_success(data={"cameras": cameras, "scanning": manager.is_camera_scan_in_progress()})


@video_router.post("/source/set")
def set_source(data: SourceSetRequest, manager: CameraManager = Depends(get_manager)):
    source = data.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if manager.set_video_source(source):
        return api_success(data={"source": str(source)}, message="source set")
    raise HTTPException(status_code=400, detail="unable to open source")


@video_router.post("/source/load")
async def upload_video(video: UploadFile = File(...), manager: CameraManager = Depends(get_manager)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    save_path = settings.UPLOAD_VIDEOS_DIR / video.filename
    content = await video.read()
    with save_path.open("wb") as output:
        output.write(content)

    if manager.set_video_source(str(save_path)):
        return api_success(data={"source": str(save_path), "filename": video.filename}, message="video uploaded")
    raise HTTPException(status_code=400, detail="unable to open video")
