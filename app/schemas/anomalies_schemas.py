from __future__ import annotations

from datetime import datetime as Dt
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, condecimal

from app.models.models import SeverityLevel

Dec8_6_Pos = condecimal(max_digits=8, decimal_places=6, ge=0)


class AnomalyBase(BaseModel):
    datetime: Dt = Field(..., description="Момент фиксации аномалии")
    is_confirmed: bool = Field(..., description="Подтверждена оператором")
    operator_notes: Optional[str] = Field(None, description="Комментарий оператора")
    mse_error: Dec8_6_Pos = Field(..., description="Ошибка MSE для события")
    severity_level: SeverityLevel = Field(..., description="Уровень серьёзности")
    forecast_id: int = Field(..., ge=1, description="FK → forecasts.forecast_id")

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class AnomalyCreate(AnomalyBase):
    created_at: Optional[Dt] = Field(None, description="Когда создана запись")
    updated_at: Optional[Dt] = Field(None, description="Когда обновлена запись")


class AnomalyUpdate(BaseModel):
    datetime: Optional[Dt] = None
    is_confirmed: Optional[bool] = None
    operator_notes: Optional[str] = None
    mse_error: Optional[Dec8_6_Pos] = None
    severity_level: Optional[SeverityLevel] = None
    forecast_id: Optional[int] = Field(None, ge=1)
    created_at: Optional[Dt] = None
    updated_at: Optional[Dt] = None

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class AnomalyOut(AnomalyBase):
    anomaly_id: int
    created_at: Dt
    updated_at: Dt
