from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple

from sqlalchemy import select, func, update as sa_update, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import RawData, Sensors, MeasurementType
from app.schemas.raw_data_schemas import RawDataCreate, RawDataUpdate


async def get_raw_record(db: AsyncSession, record_id: int) -> Optional[RawData]:
    res = await db.execute(select(RawData).where(RawData.record_id == record_id))
    return res.scalar_one_or_none()


async def list_raw_data(
    db: AsyncSession,
    *,
    sensor_id: Optional[int] = None,
    measurement_type: Optional[MeasurementType] = None,  # 👈 Enum, не str
    dt_from: Optional[datetime] = None,
    dt_to: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
    order_desc: bool = True,
) -> Tuple[Sequence[RawData], int]:
    """
    Возвращает (items, total) с фильтрами по sensor_id, measurement_type и диапазону дат.
    """
    stmt = select(RawData)
    cnt = select(func.count(RawData.record_id))

    if sensor_id is not None:
        stmt = stmt.where(RawData.sensor_id == sensor_id)
        cnt = cnt.where(RawData.sensor_id == sensor_id)

    if measurement_type is not None:
        stmt = stmt.where(RawData.measurement_type == measurement_type)
        cnt = cnt.where(RawData.measurement_type == measurement_type)

    if dt_from is not None:
        stmt = stmt.where(RawData.datetime >= dt_from)
        cnt = cnt.where(RawData.datetime >= dt_from)

    if dt_to is not None:
        stmt = stmt.where(RawData.datetime < dt_to)
        cnt = cnt.where(RawData.datetime < dt_to)

    stmt = stmt.order_by(
        RawData.datetime.desc() if order_desc else RawData.datetime.asc()
    ).limit(limit).offset(offset)

    items_res = await db.execute(stmt)
    cnt_res = await db.execute(cnt)

    items = items_res.scalars().all()
    total = cnt_res.scalar_one() or 0
    return items, total


async def create_raw_record(db: AsyncSession, payload: RawDataCreate) -> RawData:
    # проверяем наличие сенсора
    exists = await db.execute(
        select(Sensors.sensor_id).where(Sensors.sensor_id == payload.sensor_id)
    )
    if exists.scalar_one_or_none() is None:
        raise ValueError("Sensor does not exist")

    obj = RawData(
        datetime=payload.datetime,
        sensor_id=payload.sensor_id,
        value=payload.value,
        measurement_type=payload.measurement_type,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_raw_record(db: AsyncSession, record_id: int, payload: RawDataUpdate):
    data = payload.model_dump(exclude_unset=True, round_trip=True)

    if "sensor_id" in data:
        exists = await db.execute(
            select(Sensors.sensor_id).where(Sensors.sensor_id == data["sensor_id"])
        )
        if exists.scalar_one_or_none() is None:
            raise ValueError("Sensor does not exist")

    stmt = (
        sa_update(RawData)
        .where(RawData.record_id == record_id)
        .values(**data)
        .execution_options(synchronize_session="fetch")
    )
    res = await db.execute(stmt)
    if (res.rowcount or 0) == 0:
        return None

    await db.commit()
    return await get_raw_record(db, record_id)


async def delete_raw_record(db: AsyncSession, record_id: int) -> bool:
    res = await db.execute(sa_delete(RawData).where(RawData.record_id == record_id))
    await db.commit()
    return (res.rowcount or 0) > 0
