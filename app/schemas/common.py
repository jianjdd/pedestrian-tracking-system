from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ResponseSchema(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None


def api_success(data: Any = None, message: str = "ok") -> dict:
    return {"status": "ok", "message": message, "data": data}


def api_error(message: str = "error", data: Any = None) -> dict:
    return {"status": "error", "message": message, "data": data}
