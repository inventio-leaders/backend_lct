from __future__ import annotations

import json
from datetime import datetime as Dt
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelOut(BaseModel):
    model_id: int
    training_date: Dt = Field(..., description="Дата первичного обучения")
    last_retrained: Dt = Field(..., description="Дата последнего переобучения")
    metrics: Dict[str, Any] = Field(default_factory=dict, description='Метрики модели (JSON)')
    name: str = Field(..., max_length=50)
    version: str = Field(..., max_length=20)
    file_path: str = Field(..., max_length=255)

    model_config = ConfigDict(from_attributes=True)

    @field_validator("metrics", mode="before")
    @classmethod
    def _parse_metrics(cls, v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return {}
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {"_raw": v}
        return {"_raw": v}
