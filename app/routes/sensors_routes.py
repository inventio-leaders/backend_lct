from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.helpers.sensors_helpers import list_sensors, get_sensor, create_sensor, update_sensor, delete_sensor
from app.routes.dependecies import current_user
from app.schemas.sensors_schemas import SensorOut, SensorCreate, SensorUpdate

sensor_router = APIRouter(prefix="/sensors", tags=["sensors"], dependencies=[Depends(current_user)])


@sensor_router.get("/", response_model=List[SensorOut])
async def api_list_sensors(
    q: Optional[str] = Query(None, description="Поиск по name/location, like"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    order_desc: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    items, _ = await list_sensors(db, q=q, limit=limit, offset=offset, order_desc=order_desc)
    return items


@sensor_router.get("/count", response_model=int)
async def api_count_sensors(
    q: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    _, total = await list_sensors(db, q=q, limit=1, offset=0)
    return total


@sensor_router.get("/{sensor_id}", response_model=SensorOut)
async def api_get_sensor(
    sensor_id: int,
    db: AsyncSession = Depends(get_db),
):
    obj = await get_sensor(db, sensor_id)
    if not obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sensor not found")
    return obj


@sensor_router.post("/", response_model=SensorOut, status_code=status.HTTP_201_CREATED)
async def api_create_sensor(
    payload: SensorCreate,
    db: AsyncSession = Depends(get_db),
):
    obj = await create_sensor(db, payload)
    return obj


@sensor_router.patch("/{sensor_id}", response_model=SensorOut)
async def api_patch_sensor(
    sensor_id: int,
    payload: SensorUpdate,
    db: AsyncSession = Depends(get_db),
):
    obj = await update_sensor(db, sensor_id, payload)
    if not obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sensor not found")
    return obj


@sensor_router.delete("/{sensor_id}", status_code=status.HTTP_204_NO_CONTENT)
async def api_delete_sensor(
    sensor_id: int,
    db: AsyncSession = Depends(get_db),
):
    ok = await delete_sensor(db, sensor_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sensor not found")
    return None
