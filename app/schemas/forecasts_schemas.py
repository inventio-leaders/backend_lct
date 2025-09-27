from __future__ import annotations

from datetime import datetime as Dt
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, condecimal

Dec8_3_Pos = condecimal(max_digits=8, decimal_places=3, ge=0)
Dec4_3_Prob = condecimal(max_digits=4, decimal_places=3, ge=0, le=1)


class ForecastBase(BaseModel):
    datetime: Dt = Field(..., description="Момент, на который даётся прогноз")
    predicted_consumption_gvs: Dec8_3_Pos
    model_version: str = Field(..., max_length=20)
    confidence_score: Dec4_3_Prob

    processed_data_id: int = Field(..., ge=1)
    model_id: int = Field(..., ge=1)

    model_config = ConfigDict(from_attributes=True)


class ForecastCreate(ForecastBase):
    pass


class ForecastUpdate(BaseModel):
    datetime: Optional[Dt] = None
    predicted_consumption_gvs: Optional[Dec8_3_Pos] = None
    model_version: Optional[str] = Field(None, max_length=20)
    confidence_score: Optional[Dec4_3_Prob] = None

    processed_data_id: Optional[int] = Field(None, ge=1)
    model_id: Optional[int] = Field(None, ge=1)

    model_config = ConfigDict(from_attributes=True)


class ForecastOut(ForecastBase):
    forecast_id: int
