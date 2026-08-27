from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    video_path: str


class DetectionLogItem(BaseModel):
    id: int
    timestamp: datetime
    camera_source: Optional[str] = None
    count_a_to_b: int
    count_b_to_a: int
    total_count: int
    current_object_count: int
    details_json: Optional[str] = None


class DetectionLogList(BaseModel):
    items: List[DetectionLogItem]
    total: int
    limit: int
    offset: int
