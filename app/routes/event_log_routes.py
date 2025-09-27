from __future__ import annotations

from datetime import datetime as Dt
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.routes.dependecies import current_user
from app.models.models import EventLabel
from app.schemas.event_log_schemas import EventLogOut
from app.helpers.event_log_helpers import list_event_logs

event_log_router = APIRouter(
    prefix="/event-log",
    tags=["event-log"],
    dependencies=[Depends(current_user)],
)


@event_log_router.get("/", response_model=List[EventLogOut])
async def api_list_event_logs(
    anomaly_id: Optional[int] = Query(None, ge=1),
    label: Optional[EventLabel] = Query(None, description="Метка события (enum)"),
    operator_id: Optional[str] = Query(None, max_length=50),
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order_desc: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    items, _ = await list_event_logs(
        db,
        anomaly_id=anomaly_id,
        label=label,
        operator_id=operator_id,
        dt_from=dt_from,
        dt_to=dt_to,
        limit=limit,
        offset=offset,
        order_desc=order_desc,
    )
    return items
