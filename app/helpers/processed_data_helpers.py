from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple

from sqlalchemy import select, func, update as sa_update, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import ProcessedData
from app.schemas.processed_data_schemas import ProcessedDataCreate, ProcessedDataUpdate


async def get_processed_record(db: AsyncSession, record_id: int) -> Optional[ProcessedData]:
    res = await db.execute(select(ProcessedData).where(ProcessedData.record_id == record_id))
    return res.scalar_one_or_none()


async def list_processed_data(
    db: AsyncSession,
    *,
    dt_from: Optional[datetime] = None,
    dt_to: Optional[datetime] = None,
    hour: Optional[int] = None,
    day_of_week: Optional[int] = None,
    is_weekend: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
    order_desc: bool = True,
) -> Tuple[Sequence[ProcessedData], int]:
    stmt = select(ProcessedData)
    cnt = select(func.count(ProcessedData.record_id))

    if dt_from is not None:
        stmt = stmt.where(ProcessedData.datetime >= dt_from)
        cnt = cnt.where(ProcessedData.datetime >= dt_from)

    if dt_to is not None:
        stmt = stmt.where(ProcessedData.datetime < dt_to)
        cnt = cnt.where(ProcessedData.datetime < dt_to)

    if hour is not None:
        stmt = stmt.where(ProcessedData.hour == hour)
        cnt = cnt.where(ProcessedData.hour == hour)

    if day_of_week is not None:
        stmt = stmt.where(ProcessedData.day_of_week == day_of_week)
        cnt = cnt.where(ProcessedData.day_of_week == day_of_week)

    if is_weekend is not None:
        stmt = stmt.where(ProcessedData.is_weekend == is_weekend)
        cnt = cnt.where(ProcessedData.is_weekend == is_weekend)

    stmt = stmt.order_by(
        ProcessedData.datetime.desc() if order_desc else ProcessedData.datetime.asc()
    ).limit(limit).offset(offset)

    items_res = await db.execute(stmt)
    cnt_res = await db.execute(cnt)

    items = items_res.scalars().all()
    total = cnt_res.scalar_one() or 0
    return items, total


async def create_processed_record(db: AsyncSession, payload: ProcessedDataCreate) -> ProcessedData:
    obj = ProcessedData(
        datetime=payload.datetime,
        hour=payload.hour,
        day_of_week=payload.day_of_week,
        is_weekend=payload.is_weekend,
        consumption_gvs=payload.consumption_gvs,
        consumption_hvs=payload.consumption_hvs,
        delta_gvs_hvs=payload.delta_gvs_hvs,
        temp_gvs_supply=payload.temp_gvs_supply,
        temp_gvs_return=payload.temp_gvs_return,
        temp_delta=payload.temp_delta,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_processed_record(
    db: AsyncSession, record_id: int, payload: ProcessedDataUpdate
) -> Optional[ProcessedData]:
    data = payload.model_dump(exclude_unset=True, round_trip=True)

    stmt = (
        sa_update(ProcessedData)
        .where(ProcessedData.record_id == record_id)
        .values(**data)
        .execution_options(synchronize_session="fetch")
    )
    res = await db.execute(stmt)
    if (res.rowcount or 0) == 0:
        return None

    await db.commit()
    return await get_processed_record(db, record_id)


async def delete_processed_record(db: AsyncSession, record_id: int) -> bool:
    res = await db.execute(sa_delete(ProcessedData).where(ProcessedData.record_id == record_id))
    await db.commit()
    return (res.rowcount or 0) > 0
