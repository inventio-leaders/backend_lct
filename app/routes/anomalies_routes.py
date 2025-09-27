from __future__ import annotations

from datetime import datetime as Dt
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, DataError

from app.database import get_db
from app.routes.dependecies import current_user
from app.models.models import SeverityLevel
from app.schemas.anomalies_schemas import (
    AnomalyOut,
    AnomalyCreate,
    AnomalyUpdate,
)
from app.helpers.anomalies_helpers import (
    list_anomalies,
    get_anomaly,
    create_anomaly,
    update_anomaly,
    delete_anomaly,
)

anomalies_router = APIRouter(
    prefix="/anomalies",
    tags=["anomalies"],
    dependencies=[Depends(current_user)],
)


@anomalies_router.get("/", response_model=List[AnomalyOut])
async def api_list_anomalies(
    forecast_id: Optional[int] = Query(None, ge=1),
    severity_level: Optional[SeverityLevel] = Query(None),
    is_confirmed: Optional[bool] = Query(None),
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    mse_from: Optional[Decimal] = Query(None, ge=0),
    mse_to: Optional[Decimal] = Query(None, ge=0),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order_desc: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    items, _ = await list_anomalies(
        db,
        forecast_id=forecast_id,
        severity_level=severity_level,
        is_confirmed=is_confirmed,
        dt_from=dt_from,
        dt_to=dt_to,
        mse_from=float(mse_from) if mse_from is not None else None,
        mse_to=float(mse_to) if mse_to is not None else None,
        limit=limit,
        offset=offset,
        order_desc=order_desc,
    )
    return items


@anomalies_router.get("/count", response_model=int)
async def api_count_anomalies(
    forecast_id: Optional[int] = Query(None, ge=1),
    severity_level: Optional[SeverityLevel] = Query(None),
    is_confirmed: Optional[bool] = Query(None),
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    mse_from: Optional[Decimal] = Query(None, ge=0),
    mse_to: Optional[Decimal] = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    _, total = await list_anomalies(
        db,
        forecast_id=forecast_id,
        severity_level=severity_level,
        is_confirmed=is_confirmed,
        dt_from=dt_from,
        dt_to=dt_to,
        mse_from=float(mse_from) if mse_from is not None else None,
        mse_to=float(mse_to) if mse_to is not None else None,
        limit=1,
        offset=0,
    )
    return total


@anomalies_router.get("/{anomaly_id}", response_model=AnomalyOut)
async def api_get_anomaly(
    anomaly_id: int,
    db: AsyncSession = Depends(get_db),
):
    obj = await get_anomaly(db, anomaly_id)
    if not obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")
    return obj


@anomalies_router.post("/", response_model=AnomalyOut, status_code=status.HTTP_201_CREATED)
async def api_create_anomaly(
    payload: AnomalyCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        obj = await create_anomaly(db, payload)
        return obj
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@anomalies_router.patch("/{anomaly_id}", response_model=AnomalyOut)
async def api_patch_anomaly(
    anomaly_id: int,
    payload: AnomalyUpdate,
    db: AsyncSession = Depends(get_db),
):
    try:
        obj = await update_anomaly(db, anomaly_id, payload)
        if not obj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")
        return obj
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@anomalies_router.delete("/{anomaly_id}", status_code=status.HTTP_204_NO_CONTENT)
async def api_delete_anomaly(
    anomaly_id: int,
    db: AsyncSession = Depends(get_db),
):
    ok = await delete_anomaly(db, anomaly_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")
    return None
