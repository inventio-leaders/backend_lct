from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple

from sqlalchemy import select, func, update as sa_update, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Forecasts, ProcessedData, Models
from app.schemas.forecasts_schemas import ForecastCreate, ForecastUpdate


async def get_forecast(db: AsyncSession, forecast_id: int) -> Optional[Forecasts]:
    res = await db.execute(select(Forecasts).where(Forecasts.forecast_id == forecast_id))
    return res.scalar_one_or_none()


async def list_forecasts(
    db: AsyncSession,
    *,
    processed_data_id: Optional[int] = None,
    model_id: Optional[int] = None,
    model_version: Optional[str] = None,
    dt_from: Optional[datetime] = None,
    dt_to: Optional[datetime] = None,
    conf_from: Optional[float] = None,
    conf_to: Optional[float] = None,
    pred_from: Optional[float] = None,
    pred_to: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    order_desc: bool = True,
) -> Tuple[Sequence[Forecasts], int]:
    stmt = select(Forecasts)
    cnt = select(func.count(Forecasts.forecast_id))

    if processed_data_id is not None:
        stmt = stmt.where(Forecasts.processed_data_id == processed_data_id)
        cnt = cnt.where(Forecasts.processed_data_id == processed_data_id)

    if model_id is not None:
        stmt = stmt.where(Forecasts.model_id == model_id)
        cnt = cnt.where(Forecasts.model_id == model_id)

    if model_version:
        stmt = stmt.where(Forecasts.model_version == model_version)
        cnt = cnt.where(Forecasts.model_version == model_version)

    if dt_from is not None:
        stmt = stmt.where(Forecasts.datetime >= dt_from)
        cnt = cnt.where(Forecasts.datetime >= dt_from)

    if dt_to is not None:
        stmt = stmt.where(Forecasts.datetime < dt_to)
        cnt = cnt.where(Forecasts.datetime < dt_to)

    if conf_from is not None:
        stmt = stmt.where(Forecasts.confidence_score >= conf_from)
        cnt = cnt.where(Forecasts.confidence_score >= conf_from)

    if conf_to is not None:
        stmt = stmt.where(Forecasts.confidence_score <= conf_to)
        cnt = cnt.where(Forecasts.confidence_score <= conf_to)

    if pred_from is not None:
        stmt = stmt.where(Forecasts.predicted_consumption_gvs >= pred_from)
        cnt = cnt.where(Forecasts.predicted_consumption_gvs >= pred_from)

    if pred_to is not None:
        stmt = stmt.where(Forecasts.predicted_consumption_gvs <= pred_to)
        cnt = cnt.where(Forecasts.predicted_consumption_gvs <= pred_to)

    stmt = stmt.order_by(
        Forecasts.datetime.desc() if order_desc else Forecasts.datetime.asc()
    ).limit(limit).offset(offset)

    items_res = await db.execute(stmt)
    cnt_res = await db.execute(cnt)

    items = items_res.scalars().all()
    total = cnt_res.scalar_one() or 0
    return items, total


async def _ensure_fk_exist(db: AsyncSession, processed_data_id: Optional[int], model_id: Optional[int]):
    if processed_data_id is not None:
        pd = await db.execute(select(ProcessedData.record_id).where(ProcessedData.record_id == processed_data_id))
        if pd.scalar_one_or_none() is None:
            raise ValueError("ProcessedData does not exist")
    if model_id is not None:
        m = await db.execute(select(Models.model_id).where(Models.model_id == model_id))
        if m.scalar_one_or_none() is None:
            raise ValueError("Model does not exist")


async def create_forecast(db: AsyncSession, payload: ForecastCreate) -> Forecasts:
    await _ensure_fk_exist(db, payload.processed_data_id, payload.model_id)

    obj = Forecasts(
        datetime=payload.datetime,
        predicted_consumption_gvs=payload.predicted_consumption_gvs,
        model_version=payload.model_version,
        confidence_score=payload.confidence_score,
        processed_data_id=payload.processed_data_id,
        model_id=payload.model_id,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_forecast(db: AsyncSession, forecast_id: int, payload: ForecastUpdate) -> Optional[Forecasts]:
    data = payload.model_dump(exclude_unset=True, round_trip=True)

    await _ensure_fk_exist(db, data.get("processed_data_id"), data.get("model_id"))

    stmt = (
        sa_update(Forecasts)
        .where(Forecasts.forecast_id == forecast_id)
        .values(**data)
        .execution_options(synchronize_session="fetch")
    )
    res = await db.execute(stmt)
    if (res.rowcount or 0) == 0:
        return None

    await db.commit()
    return await get_forecast(db, forecast_id)


async def delete_forecast(db: AsyncSession, forecast_id: int) -> bool:
    res = await db.execute(sa_delete(Forecasts).where(Forecasts.forecast_id == forecast_id))
    await db.commit()
    return (res.rowcount or 0) > 0
