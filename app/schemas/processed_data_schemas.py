from __future__ import annotations

from datetime import datetime as Dt
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, condecimal

# DECIMAL(8,3) и DECIMAL(5,2)
Dec8_3 = condecimal(max_digits=8, decimal_places=3)
Dec5_2 = condecimal(max_digits=5, decimal_places=2)


class ProcessedDataBase(BaseModel):
    datetime: Dt = Field(..., description="Метка времени (вычисления на срезе)")
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Mon ... 6=Sun (или как у вас заведено)")
    is_weekend: bool

    consumption_gvs: Dec8_3
    consumption_hvs: Dec8_3
    delta_gvs_hvs: Dec8_3

    temp_gvs_supply: Dec5_2
    temp_gvs_return: Dec5_2
    temp_delta: Dec5_2

    model_config = ConfigDict(from_attributes=True)


class ProcessedDataCreate(ProcessedDataBase):
    pass


class ProcessedDataUpdate(BaseModel):
    datetime: Optional[Dt] = None
    hour: Optional[int] = Field(None, ge=0, le=23)
    day_of_week: Optional[int] = Field(None, ge=0, le=6)
    is_weekend: Optional[bool] = None

    consumption_gvs: Optional[Dec8_3] = None
    consumption_hvs: Optional[Dec8_3] = None
    delta_gvs_hvs: Optional[Dec8_3] = None

    temp_gvs_supply: Optional[Dec5_2] = None
    temp_gvs_return: Optional[Dec5_2] = None
    temp_delta: Optional[Dec5_2] = None

    model_config = ConfigDict(from_attributes=True)


class ProcessedDataOut(ProcessedDataBase):
    record_id: int
