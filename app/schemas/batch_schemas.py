from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

class TrainIn(BaseModel):
    narx: bool = True
    ae: bool = True
    window: int = 24
    train_from: Optional[datetime] = Field(None)
    train_to: Optional[datetime] = Field(None)
    ae_threshold_percentile: int = 99

class TaskOut(BaseModel):
    task_id: str
    status: str
    progress: str | None = None
    result: dict | None = None
    error: str | None = None

class ForecastRunIn(BaseModel):
    horizon_hours: int = 48

class AnomalyScanIn(BaseModel):
    from_dt: Optional[datetime] = None
    to_dt: Optional[datetime] = None
