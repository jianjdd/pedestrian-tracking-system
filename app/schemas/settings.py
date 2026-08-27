from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class UpdateSettingsRequest(BaseModel):
    yolo: Optional[Dict[str, Any]] = None
    tracking: Optional[Dict[str, Any]] = None
    display: Optional[Dict[str, Any]] = None


class PresetRequest(BaseModel):
    name: str = "standard"


class SetLineRequest(BaseModel):
    point1: List[int]
    point2: List[int]
    reset_counts: bool = True
