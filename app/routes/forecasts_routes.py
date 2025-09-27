from __future__ import annotations

from datetime import datetime as Dt
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, DataError

from app.database import get_db
from app.routes.dependecies import current_user
from app.schemas.forecasts_schemas import ForecastOut, ForecastCreate, ForecastUpdate
from app.helpers.forecasts_helpers import (
    list_forecasts,
    get_forecast,
    create_forecast,
    update_forecast,
    delete_forecast,
)

forecasts_router = APIRouter(
    prefix="/forecasts",
    tags=["forecasts"],
    dependencies=[Depends(current_user)],
)


@forecasts_router.get("/", response_model=List[ForecastOut])
async def api_list_forecasts(
    processed_data_id: Optional[int] = Query(None, ge=1),
    model_id: Optional[int] = Query(None, ge=1),
    model_version: Optional[str] = Query(None, max_length=20),
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    conf_from: Optional[Decimal] = Query(None, ge=0, le=1),
    conf_to: Optional[Decimal] = Query(None, ge=0, le=1),
    pred_from: Optional[Decimal] = Query(None, ge=0),
    pred_to: Optional[Decimal] = Query(None, ge=0),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order_desc: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    items, _ = await list_forecasts(
        db,
        processed_data_id=processed_data_id,
        model_id=model_id,
        model_version=model_version,
        dt_from=dt_from,
        dt_to=dt_to,
        conf_from=float(conf_from) if conf_from is not None else None,
        conf_to=float(conf_to) if conf_to is not None else None,
        pred_from=float(pred_from) if pred_from is not None else None,
        pred_to=float(pred_to) if pred_to is not None else None,
        limit=limit,
        offset=offset,
        order_desc=order_desc,
    )
    return items


@forecasts_router.get("/count", response_model=int)
async def api_count_forecasts(
    processed_data_id: Optional[int] = Query(None, ge=1),
    model_id: Optional[int] = Query(None, ge=1),
    model_version: Optional[str] = Query(None, max_length=20),
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    conf_from: Optional[Decimal] = Query(None, ge=0, le=1),
    conf_to: Optional[Decimal] = Query(None, ge=0, le=1),
    pred_from: Optional[Decimal] = Query(None, ge=0),
    pred_to: Optional[Decimal] = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    _, total = await list_forecasts(
        db,
        processed_data_id=processed_data_id,
        model_id=model_id,
        model_version=model_version,
        dt_from=dt_from,
        dt_to=dt_to,
        conf_from=float(conf_from) if conf_from is not None else None,
        conf_to=float(conf_to) if conf_to is not None else None,
        pred_from=float(pred_from) if pred_from is not None else None,
        pred_to=float(pred_to) if pred_to is not None else None,
        limit=1,
        offset=0,
    )
    return total


@forecasts_router.get("/{forecast_id}", response_model=ForecastOut)
async def api_get_forecast(
    forecast_id: int,
    db: AsyncSession = Depends(get_db),
):
    obj = await get_forecast(db, forecast_id)
    if not obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast not found")
    return obj


@forecasts_router.post("/", response_model=ForecastOut, status_code=status.HTTP_201_CREATED)
async def api_create_forecast(
    payload: ForecastCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        obj = await create_forecast(db, payload)
        return obj
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@forecasts_router.patch("/{forecast_id}", response_model=ForecastOut)
async def api_patch_forecast(
    forecast_id: int,
    payload: ForecastUpdate,
    db: AsyncSession = Depends(get_db),
):
    try:
        obj = await update_forecast(db, forecast_id, payload)
        if not obj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast not found")
        return obj
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@forecasts_router.delete("/{forecast_id}", status_code=status.HTTP_204_NO_CONTENT)
async def api_delete_forecast(
    forecast_id: int,
    db: AsyncSession = Depends(get_db),
):
    ok = await delete_forecast(db, forecast_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast not found")
    return None
