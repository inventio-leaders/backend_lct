from __future__ import annotations

from typing import Iterable, Optional, Tuple, Sequence

from sqlalchemy import select, update as sa_update, delete as sa_delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Sensors
from app.schemas.sensors_schemas import SensorCreate, SensorUpdate

async def get_sensor(db: AsyncSession, sensor_id: int) -> Optional[Sensors]:
    stmt = select(Sensors).where(Sensors.sensor_id == sensor_id)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()


async def list_sensors(
    db: AsyncSession,
    *,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    order_desc: bool = False,
) -> Tuple[Sequence[Sensors], int]:
    """
    Возвращает (items, total). Поиск по частичному совпадению name/location.
    """
    where_clauses = []
    if q:
        like = f"%{q}%"
        where_clauses.append(
            (Sensors.name.ilike(like)) | (Sensors.location.ilike(like))
        )

    base = select(Sensors)
    if where_clauses:
        base = base.where(*where_clauses)

    order_by = Sensors.sensor_id.desc() if order_desc else Sensors.sensor_id.asc()
    items_stmt = base.order_by(order_by).limit(limit).offset(offset)
    total_stmt = select(func.count(Sensors.sensor_id))
    if where_clauses:
        total_stmt = total_stmt.where(*where_clauses)

    items_res = await db.execute(items_stmt)
    total_res = await db.execute(total_stmt)

    items = items_res.scalars().all()
    total = total_res.scalar_one() or 0
    return items, total


# --------- Create ---------
async def create_sensor(db: AsyncSession, payload: SensorCreate) -> Sensors:
    obj = Sensors(
        name=payload.name,
        location=payload.location,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


# --------- Update ---------
async def update_sensor(
    db: AsyncSession, sensor_id: int, payload: SensorUpdate
) -> Optional[Sensors]:
    data = payload.model_dump(exclude_unset=True)
    if not data:
        # ничего обновлять
        return await get_sensor(db, sensor_id)

    stmt = (
        sa_update(Sensors)
        .where(Sensors.sensor_id == sensor_id)
        .values(**data)
        .execution_options(synchronize_session="fetch")
    )
    res = await db.execute(stmt)
    if res.rowcount == 0:
        return None

    await db.commit()
    # перечитываем объект
    return await get_sensor(db, sensor_id)


# --------- Delete ---------
async def delete_sensor(db: AsyncSession, sensor_id: int) -> bool:
    stmt = sa_delete(Sensors).where(Sensors.sensor_id == sensor_id)
    res = await db.execute(stmt)
    await db.commit()
    return res.rowcount > 0
