from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple

from sqlalchemy import select, func, update as sa_update, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Anomalies, Forecasts, SeverityLevel
from app.schemas.anomalies_schemas import AnomalyCreate, AnomalyUpdate


async def _ensure_fk_forecast(db: AsyncSession, forecast_id: Optional[int]) -> None:
    if forecast_id is None:
        return
    res = await db.execute(
        select(Forecasts.forecast_id).where(Forecasts.forecast_id == forecast_id)
    )
    if res.scalar_one_or_none() is None:
        raise ValueError("Forecast does not exist")


async def get_anomaly(db: AsyncSession, anomaly_id: int) -> Optional[Anomalies]:
    res = await db.execute(select(Anomalies).where(Anomalies.anomaly_id == anomaly_id))
    return res.scalar_one_or_none()


async def list_anomalies(
    db: AsyncSession,
    *,
    forecast_id: Optional[int] = None,
    severity_level: Optional[SeverityLevel] = None,
    is_confirmed: Optional[bool] = None,
    dt_from: Optional[datetime] = None,
    dt_to: Optional[datetime] = None,
    mse_from: Optional[float] = None,
    mse_to: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    order_desc: bool = True,
) -> Tuple[Sequence[Anomalies], int]:
    stmt = select(Anomalies)
    cnt = select(func.count(Anomalies.anomaly_id))

    if forecast_id is not None:
        stmt = stmt.where(Anomalies.forecast_id == forecast_id)
        cnt = cnt.where(Anomalies.forecast_id == forecast_id)

    if severity_level is not None:
        stmt = stmt.where(Anomalies.severity_level == severity_level)
        cnt = cnt.where(Anomalies.severity_level == severity_level)

    if is_confirmed is not None:
        stmt = stmt.where(Anomalies.is_confirmed == is_confirmed)
        cnt = cnt.where(Anomalies.is_confirmed == is_confirmed)

    if dt_from is not None:
        stmt = stmt.where(Anomalies.datetime >= dt_from)
        cnt = cnt.where(Anomalies.datetime >= dt_from)

    if dt_to is not None:
        stmt = stmt.where(Anomalies.datetime < dt_to)
        cnt = cnt.where(Anomalies.datetime < dt_to)

    if mse_from is not None:
        stmt = stmt.where(Anomalies.mse_error >= mse_from)
        cnt = cnt.where(Anomalies.mse_error >= mse_from)

    if mse_to is not None:
        stmt = stmt.where(Anomalies.mse_error <= mse_to)
        cnt = cnt.where(Anomalies.mse_error <= mse_to)

    stmt = stmt.order_by(
        Anomalies.datetime.desc() if order_desc else Anomalies.datetime.asc()
    ).limit(limit).offset(offset)

    items_res = await db.execute(stmt)
    cnt_res = await db.execute(cnt)

    items = items_res.scalars().all()
    total = cnt_res.scalar_one() or 0
    return items, total


async def create_anomaly(db: AsyncSession, payload: AnomalyCreate) -> Anomalies:
    await _ensure_fk_forecast(db, payload.forecast_id)
    now = datetime.utcnow()
    created_at = payload.created_at or now
    updated_at = payload.updated_at or now

    obj = Anomalies(
        datetime=payload.datetime,
        is_confirmed=payload.is_confirmed,
        operator_notes=payload.operator_notes,
        created_at=created_at,
        updated_at=updated_at,
        mse_error=payload.mse_error,
        severity_level=payload.severity_level,
        forecast_id=payload.forecast_id,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_anomaly(
    db: AsyncSession, anomaly_id: int, payload: AnomalyUpdate
) -> Optional[Anomalies]:
    data = payload.model_dump(exclude_unset=True, round_trip=True)

    await _ensure_fk_forecast(db, data.get("forecast_id"))

    data.setdefault("updated_at", datetime.utcnow())

    stmt = (
        sa_update(Anomalies)
        .where(Anomalies.anomaly_id == anomaly_id)
        .values(**data)
        .execution_options(synchronize_session="fetch")
    )
    res = await db.execute(stmt)
    if (res.rowcount or 0) == 0:
        return None

    await db.commit()
    return await get_anomaly(db, anomaly_id)


async def delete_anomaly(db: AsyncSession, anomaly_id: int) -> bool:
    res = await db.execute(sa_delete(Anomalies).where(Anomalies.anomaly_id == anomaly_id))
    await db.commit()
    return (res.rowcount or 0) > 0
