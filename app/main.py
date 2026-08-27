from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.api_router import api_router
from app.core.config import settings
from app.schemas.common import api_error
from app.services.camera_manager import get_camera_manager


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pre-initialize singleton manager so cold-start failures surface early.
    manager = get_camera_manager()
    manager.load_config()
    manager.start_camera_scan_async(force=True)
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=api_error(message=str(exc.detail)))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=api_error(message="validation error", data=exc.errors()))


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content=api_error(message=f"internal server error: {exc}"))


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "healthy"}


app.include_router(api_router, prefix=settings.API_PREFIX)
