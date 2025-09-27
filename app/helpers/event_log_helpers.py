from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import EventLog, EventLabel


async def list_event_logs(
    db: AsyncSession,
    *,
    anomaly_id: Optional[int] = None,
    label: Optional[EventLabel] = None,
    operator_id: Optional[str] = None,
    dt_from: Optional[datetime] = None,
    dt_to: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
    order_desc: bool = True,
) -> Tuple[Sequence[EventLog], int]:
    """
    Возвращает (items, total) со стандартными фильтрами.
    """
    stmt = select(EventLog)
    cnt = select(func.count(EventLog.event_id))

    if anomaly_id is not None:
        stmt = stmt.where(EventLog.anomaly_id == anomaly_id)
        cnt = cnt.where(EventLog.anomaly_id == anomaly_id)

    if label is not None:
        stmt = stmt.where(EventLog.label == label)   # сравнение по Enum
        cnt = cnt.where(EventLog.label == label)

    if operator_id:
        stmt = stmt.where(EventLog.operator_id == operator_id)
        cnt = cnt.where(EventLog.operator_id == operator_id)

    if dt_from is not None:
        stmt = stmt.where(EventLog.datetime >= dt_from)
        cnt = cnt.where(EventLog.datetime >= dt_from)

    if dt_to is not None:
        stmt = stmt.where(EventLog.datetime < dt_to)
        cnt = cnt.where(EventLog.datetime < dt_to)

    stmt = stmt.order_by(
        EventLog.datetime.desc() if order_desc else EventLog.datetime.asc()
    ).limit(limit).offset(offset)

    items_res = await db.execute(stmt)
    cnt_res = await db.execute(cnt)

    items = items_res.scalars().all()
    total = cnt_res.scalar_one() or 0
    return items, total
