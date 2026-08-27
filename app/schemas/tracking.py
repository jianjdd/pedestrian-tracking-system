from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TrackingStatusResponse(BaseModel):
    is_running: bool
    is_paused: bool
    has_source: bool
    infer_device: Optional[str] = None

