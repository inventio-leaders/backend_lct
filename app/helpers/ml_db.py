from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Models, ProcessedData, Forecasts, Anomalies, EventLog, SeverityLevel, EventLabel

async def get_latest_model_by_name(db: AsyncSession, name: str) -> Optional[Models]:
    q = select(Models).where(Models.name == name).order_by(Models.last_retrained.desc()).limit(1)
    res = await db.execute(q)
    return res.scalar_one_or_none()

async def get_processed_by_id(db: AsyncSession, pid: int) -> Optional[ProcessedData]:
    res = await db.execute(select(ProcessedData).where(ProcessedData.record_id == pid))
    return res.scalar_one_or_none()

def _severity_from_mse(mse: float, thr: float) -> SeverityLevel:
    if mse <= thr * 1.2:
        return SeverityLevel.Низкий
    if mse <= thr * 2.0:
        return SeverityLevel.Средний
    return SeverityLevel.Высокий

async def create_forecast(
    db: AsyncSession,
    *,
    pd_row: ProcessedData,
    value: float,
    confidence: float,
    model: Models,
) -> Forecasts:
    obj = Forecasts(
        datetime=pd_row.datetime + timedelta(hours=1),
        predicted_consumption_gvs=Decimal(str(round(value, 3))),
        model_version=model.version,
        confidence_score=Decimal(str(round(confidence, 3))),
        processed_data_id=pd_row.record_id,
        model_id=model.model_id,
    )
    db.add(obj)
    await db.flush()
    return obj

async def create_anomaly(
    db: AsyncSession,
    *,
    pd_row: ProcessedData,
    mse_error: float,
    threshold: float,
    severity: SeverityLevel,
    forecast: Optional[Forecasts],
    is_confirmed: bool = False,
    operator_notes: Optional[str] = None,
) -> Anomalies:
    now = datetime.utcnow()
    obj = Anomalies(
        datetime=pd_row.datetime,
        is_confirmed=is_confirmed,
        operator_notes=operator_notes,
        created_at=now,
        updated_at=now,
        mse_error=Decimal(str(round(mse_error, 6))),
        severity_level=severity,
        forecast_id=forecast.forecast_id if forecast else None,
    )
    db.add(obj)
    await db.flush()
    return obj

async def add_event(
    db: AsyncSession,
    *,
    anomaly: Anomalies,
    label: EventLabel,
    notes: Optional[str],
) -> EventLog:
    ev = EventLog(
        anomaly_id=anomaly.anomaly_id,
        label=label,
        notes=notes,
        created_at=datetime.utcnow(),
    )
    db.add(ev)
    anomaly.updated_at = datetime.utcnow()
    await db.flush()
    return ev
