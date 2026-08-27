from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base


class DetectionLog(Base):
    __tablename__ = "detection_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    camera_source: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    count_a_to_b: Mapped[int] = mapped_column(Integer, default=0)
    count_b_to_a: Mapped[int] = mapped_column(Integer, default=0)
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    current_object_count: Mapped[int] = mapped_column(Integer, default=0)
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
