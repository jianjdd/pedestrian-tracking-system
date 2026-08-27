from __future__ import annotations

from typing import Union

from pydantic import BaseModel


class ModelPathRequest(BaseModel):
    path: str


class SourceSetRequest(BaseModel):
    source: Union[str, int]

