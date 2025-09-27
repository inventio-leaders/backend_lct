from __future__ import annotations

from datetime import datetime as Dt
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, DataError

from app.database import get_db
from app.routes.dependecies import current_user
from app.schemas.raw_data_schemas import RawDataOut, RawDataCreate, RawDataUpdate
from app.helpers.raw_data_helpers import (
    list_raw_data,
    get_raw_record,
    create_raw_record,
    update_raw_record,
    delete_raw_record,
)
from app.models.models import MeasurementType

raw_router = APIRouter(
    prefix="/raw-data",
    tags=["raw-data"],
    dependencies=[Depends(current_user)],
)


@raw_router.get("/", response_model=List[RawDataOut])
async def api_list_raw_data(
    sensor_id: Optional[int] = Query(None, ge=1),
    measurement_type: Optional[MeasurementType] = Query(
        None,
        description='Тип измерения: "Подача" | "Обратка" | "Потребление" | "T1" | "T2"',
    ),
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order_desc: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    items, _ = await list_raw_data(
        db,
        sensor_id=sensor_id,
        measurement_type=measurement_type,  # можно Enum напрямую
        dt_from=dt_from,
        dt_to=dt_to,
        limit=limit,
        offset=offset,
        order_desc=order_desc,
    )
    return items


@raw_router.get("/count", response_model=int)
async def api_count_raw_data(
    sensor_id: Optional[int] = Query(None, ge=1),
    measurement_type: Optional[MeasurementType] = Query(None),
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    _, total = await list_raw_data(
        db,
        sensor_id=sensor_id,
        measurement_type=measurement_type,
        dt_from=dt_from,
        dt_to=dt_to,
        limit=1,
        offset=0,
    )
    return total


@raw_router.get("/{record_id}", response_model=RawDataOut)
async def api_get_raw_record(
    record_id: int,
    db: AsyncSession = Depends(get_db),
):
    obj = await get_raw_record(db, record_id)
    if not obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
    return obj


@raw_router.post("/", response_model=RawDataOut, status_code=status.HTTP_201_CREATED)
async def api_create_raw_record(
    payload: RawDataCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        obj = await create_raw_record(db, payload)
        return obj
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@raw_router.patch("/{record_id}", response_model=RawDataOut)
async def api_patch_raw_record(
    record_id: int,
    payload: RawDataUpdate,
    db: AsyncSession = Depends(get_db),
):
    try:
        obj = await update_raw_record(db, record_id, payload)
        if not obj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
        return obj
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@raw_router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
async def api_delete_raw_record(
    record_id: int,
    db: AsyncSession = Depends(get_db),
):
    ok = await delete_raw_record(db, record_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
    return None
