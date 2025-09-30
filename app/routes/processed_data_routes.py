from __future__ import annotations

from datetime import datetime as Dt
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File
from fastapi.responses import StreamingResponse
from io import BytesIO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, DataError

from app.database import get_db
from app.routes.dependecies import current_user
from app.schemas.processed_data_schemas import (
    ProcessedDataOut,
    ProcessedDataCreate,
    ProcessedDataUpdate,
)
from app.helpers.processed_data_helpers import (
    list_processed_data,
    get_processed_record,
    create_processed_record,
    update_processed_record,
    delete_processed_record, import_processed_data_from_excels,
)

from app.helpers.anomalies_helpers import (
    compute_anomalies_from_processed,
    persist_anomalies,
    build_anomalies_xlsx_bytes,
)

processed_router = APIRouter(
    prefix="/processed-data",
    tags=["processed-data"],
    dependencies=[Depends(current_user)],
)


@processed_router.get("/", response_model=List[ProcessedDataOut])
async def api_list_processed_data(
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    hour: Optional[int] = Query(None, ge=0, le=23),
    day_of_week: Optional[int] = Query(None, ge=0, le=6),
    is_weekend: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order_desc: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    items, _ = await list_processed_data(
        db,
        dt_from=dt_from,
        dt_to=dt_to,
        hour=hour,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
        limit=limit,
        offset=offset,
        order_desc=order_desc,
    )
    return items


@processed_router.get("/count", response_model=int)
async def api_count_processed_data(
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    hour: Optional[int] = Query(None, ge=0, le=23),
    day_of_week: Optional[int] = Query(None, ge=0, le=6),
    is_weekend: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    _, total = await list_processed_data(
        db,
        dt_from=dt_from,
        dt_to=dt_to,
        hour=hour,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
        limit=1,
        offset=0,
    )
    return total

@processed_router.get("/export-xlsx")
async def api_export_processed_to_xlsx(
    dt_from: Optional[Dt] = Query(None),
    dt_to: Optional[Dt] = Query(None),
    threshold_pct: float = Query(10.0, ge=0, le=100, description="Порог отклонения в процентах"),
    save_to_db: bool = Query(True, description="Сохранять найденные аномалии в БД"),
    filename: str = Query("processed_anomalies.xlsx"),
    db: AsyncSession = Depends(get_db),
):
    """
    Экспортирует XLSX с «Все данные» и «Аномалии», рассчитанными по processed_data.
    При save_to_db=True — записывает новые аномалии в таблицу anomalies (без дублей по datetime).
    """
    try:
        df_all, df_anoms = await compute_anomalies_from_processed(
            db, dt_from=dt_from, dt_to=dt_to, threshold_pct=threshold_pct
        )

        inserted = skipped = 0
        if save_to_db and not df_anoms.empty:
            inserted, skipped = await persist_anomalies(db, df_anoms)

        xlsx_bytes = await build_anomalies_xlsx_bytes(df_all, df_anoms)

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
        return StreamingResponse(
            BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@processed_router.get("/{record_id}", response_model=ProcessedDataOut)
async def api_get_processed_record(
    record_id: int,
    db: AsyncSession = Depends(get_db),
):
    obj = await get_processed_record(db, record_id)
    if not obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
    return obj


@processed_router.post("/", response_model=ProcessedDataOut, status_code=status.HTTP_201_CREATED)
async def api_create_processed_record(
    payload: ProcessedDataCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        obj = await create_processed_record(db, payload)
        return obj
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@processed_router.patch("/{record_id}", response_model=ProcessedDataOut)
async def api_patch_processed_record(
    record_id: int,
    payload: ProcessedDataUpdate,
    db: AsyncSession = Depends(get_db),
):
    try:
        obj = await update_processed_record(db, record_id, payload)
        if not obj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
        return obj
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@processed_router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
async def api_delete_processed_record(
    record_id: int,
    db: AsyncSession = Depends(get_db),
):
    ok = await delete_processed_record(db, record_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
    return None

@processed_router.post("/import-excel", status_code=status.HTTP_201_CREATED)
async def api_import_processed_data_from_excels(
    files: List[UploadFile] = File(..., description="Один или несколько .xlsx файлов"),
    dedupe: bool = Query(True, description="Пропускать уже существующие по datetime записи"),
    db: AsyncSession = Depends(get_db),
):
    if not files:
        raise HTTPException(status_code=400, detail="Не переданы файлы")

    try:
        result = await import_processed_data_from_excels(db, files, dedupe=dedupe)
        return result
    except (ValueError, IntegrityError, DataError) as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
