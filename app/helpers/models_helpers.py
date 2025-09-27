from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Models  # ваша ORM-модель


async def get_model(db: AsyncSession, model_id: int) -> Optional[Models]:
    res = await db.execute(select(Models).where(Models.model_id == model_id))
    return res.scalar_one_or_none()


async def list_models(
    db: AsyncSession,
    *,
    q: Optional[str] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    trained_from: Optional[datetime] = None,
    trained_to: Optional[datetime] = None,
    retrained_from: Optional[datetime] = None,
    retrained_to: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
    order_desc: bool = True,
) -> Tuple[Sequence[Models], int]:
    stmt = select(Models)
    cnt = select(func.count(Models.model_id))

    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            (Models.name.ilike(like)) |
            (Models.version.ilike(like)) |
            (Models.file_path.ilike(like))
        )
        cnt = cnt.where(
            (Models.name.ilike(like)) |
            (Models.version.ilike(like)) |
            (Models.file_path.ilike(like))
        )

    if name:
        stmt = stmt.where(Models.name == name)
        cnt = cnt.where(Models.name == name)

    if version:
        stmt = stmt.where(Models.version == version)
        cnt = cnt.where(Models.version == version)

    if trained_from:
        stmt = stmt.where(Models.training_date >= trained_from)
        cnt = cnt.where(Models.training_date >= trained_from)
    if trained_to:
        stmt = stmt.where(Models.training_date < trained_to)
        cnt = cnt.where(Models.training_date < trained_to)

    if retrained_from:
        stmt = stmt.where(Models.last_retrained >= retrained_from)
        cnt = cnt.where(Models.last_retrained >= retrained_from)
    if retrained_to:
        stmt = stmt.where(Models.last_retrained < retrained_to)
        cnt = cnt.where(Models.last_retrained < retrained_to)

    order_col = Models.last_retrained.desc() if order_desc else Models.last_retrained.asc()
    stmt = stmt.order_by(order_col).limit(limit).offset(offset)

    items_res = await db.execute(stmt)
    cnt_res = await db.execute(cnt)

    items = items_res.scalars().all()
    total = cnt_res.scalar_one() or 0
    return items, total
