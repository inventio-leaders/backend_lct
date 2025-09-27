from __future__ import annotations

from datetime import datetime as Dt
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.routes.dependecies import current_user
from app.schemas.models_schemas import ModelOut
from app.helpers.models_helpers import list_models, get_model

models_router = APIRouter(
    prefix="/models",
    tags=["models"],
    dependencies=[Depends(current_user)],
)


@models_router.get("/", response_model=List[ModelOut])
async def api_list_models(
    q: Optional[str] = Query(None, description="Поиск по name/version/file_path (ILIKE)"),
    name: Optional[str] = Query(None),
    version: Optional[str] = Query(None),
    trained_from: Optional[Dt] = Query(None),
    trained_to: Optional[Dt] = Query(None),
    retrained_from: Optional[Dt] = Query(None),
    retrained_to: Optional[Dt] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    order_desc: bool = Query(True, description="Сортировка по last_retrained"),
    db: AsyncSession = Depends(get_db),
):
    items, _ = await list_models(
        db,
        q=q,
        name=name,
        version=version,
        trained_from=trained_from,
        trained_to=trained_to,
        retrained_from=retrained_from,
        retrained_to=retrained_to,
        limit=limit,
        offset=offset,
        order_desc=order_desc,
    )
    return items


@models_router.get("/count", response_model=int)
async def api_count_models(
    q: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    version: Optional[str] = Query(None),
    trained_from: Optional[Dt] = Query(None),
    trained_to: Optional[Dt] = Query(None),
    retrained_from: Optional[Dt] = Query(None),
    retrained_to: Optional[Dt] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    _, total = await list_models(
        db,
        q=q,
        name=name,
        version=version,
        trained_from=trained_from,
        trained_to=trained_to,
        retrained_from=retrained_from,
        retrained_to=retrained_to,
        limit=1,
        offset=0,
    )
    return total


@models_router.get("/{model_id}", response_model=ModelOut)
async def api_get_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
):
    obj = await get_model(db, model_id)
    if not obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return obj
