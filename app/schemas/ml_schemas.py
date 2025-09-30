from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class ForecastIn(BaseModel):
    processed_data_id: int
    model_id: Optional[int] = Field(None, description="ID записи в models для NARX-LSTM")
    rows: List[Dict[str, float]]

class ForecastOut(BaseModel):
    forecast_id: int
    predicted_consumption_gvs: float
    confidence_score: float
    model_version: str

class DetectIn(BaseModel):
    processed_data_id: int
    forecast_id: Optional[int] = Field(None, description="связываем аномалию с прогнозом, если есть")
    model_id: Optional[int] = Field(None, description="ID модели LSTM-AE")
    rows: List[Dict[str, float]]

class DetectOut(BaseModel):
    mse: float
    threshold: float
    is_anomaly: bool
    anomaly_id: Optional[int] = None
    severity_level: Optional[str] = None

class FeedbackIn(BaseModel):
    anomaly_id: Optional[int] = None
    forecast_id: Optional[int] = None
    label: Literal["Утечка", "Ложное", "Норма"]
    confirm: bool = True
    notes: Optional[str] = None

class FeedbackOut(BaseModel):
    anomaly_id: int
    event_id: int
    status: str
