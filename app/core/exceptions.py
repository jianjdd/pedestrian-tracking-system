from __future__ import annotations

from fastapi import HTTPException


class ServiceError(HTTPException):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=detail)

