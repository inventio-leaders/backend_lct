from __future__ import annotations

from datetime import datetime as Dt
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, condecimal

from app.models.models import MeasurementType

Dec10_3 = condecimal(max_digits=10, decimal_places=3)


class RawDataBase(BaseModel):
    datetime: Dt = Field(..., description="Момент измерения")
    sensor_id: int = Field(..., ge=1, description="FK → sensors.sensor_id")
    value: Dec10_3 = Field(..., description="Значение измерения")
    measurement_type: MeasurementType = Field(..., description="Тип измерения")

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class RawDataCreate(RawDataBase):
    pass


class RawDataUpdate(BaseModel):
    datetime: Optional[Dt] = None
    sensor_id: Optional[int] = Field(None, ge=1)
    value: Optional[Dec10_3] = None
    measurement_type: Optional[MeasurementType] = None

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class RawDataOut(RawDataBase):
    record_id: int
