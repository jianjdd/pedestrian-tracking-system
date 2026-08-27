from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class StartCountingRequest(BaseModel):
    source: Optional[str] = None


class SetClassesRequest(BaseModel):
    enabled: List[str] = Field(default_factory=list)
