from __future__ import annotations

from datetime import datetime as Dt
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from app.models.models import EventLabel


class EventLogOut(BaseModel):
    event_id: int
    datetime: Dt = Field(..., description="Время события")
    confirmed_at: Optional[Dt] = Field(None, description="Когда подтверждено оператором")
    anomaly_id: int = Field(..., ge=1, description="FK → anomalies.anomaly_id")
    label: EventLabel = Field(..., description="Метка события (enum)")
    operator_id: str = Field(..., max_length=50, description="Идентификатор оператора")

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)
