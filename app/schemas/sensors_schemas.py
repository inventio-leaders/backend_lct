from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class SensorBase(BaseModel):
    name: str = Field(..., max_length=50, description='Тип ресурса: "ГВС", "ХВС" и т.п.')
    location: str = Field(..., max_length=100, description="Местоположение/узел")

    model_config = ConfigDict(from_attributes=True)


class SensorCreate(SensorBase):
    pass


class SensorUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=50)
    location: Optional[str] = Field(None, max_length=100)

    model_config = ConfigDict(from_attributes=True)


class SensorOut(SensorBase):
    sensor_id: int
