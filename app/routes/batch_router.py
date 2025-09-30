from __future__ import annotations
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from app.database import async_session
from app.routes.dependecies import current_user
from app.schemas.batch_schemas import TrainIn, TaskOut, ForecastRunIn, AnomalyScanIn
from app.services.task_manager import TASKS
from app.services.training_jobs import train_job
from app.services.batch_jobs import forecast_job, anomaly_scan_job

batch_router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    dependencies=[Depends(current_user)],
)

def _db_factory():
    return async_session()

@batch_router.post("/train", response_model=TaskOut)
async def api_train(payload: TrainIn):
    tid = TASKS.create("train")
    asyncio.create_task(
        train_job(
            tid,
            _db_factory,
            train_narx=payload.narx,
            train_ae=payload.ae,
            window=payload.window,
            dt_from=payload.train_from,
            dt_to=payload.train_to,
            ae_threshold_percentile=payload.ae_threshold_percentile,
        )
    )
    t = TASKS.get(tid)
    return TaskOut(task_id=tid, status=t.status, progress=t.progress)

@batch_router.post("/forecast/run", response_model=TaskOut)
async def api_forecast_run(payload: ForecastRunIn):
    tid = TASKS.create("forecast_run")
    asyncio.create_task(forecast_job(tid, _db_factory, payload.horizon_hours))
    t = TASKS.get(tid)
    return TaskOut(task_id=tid, status=t.status, progress=t.progress)

@batch_router.post("/anomaly/scan", response_model=TaskOut)
async def api_anomaly_scan(payload: AnomalyScanIn):
    tid = TASKS.create("anomaly_scan")
    asyncio.create_task(anomaly_scan_job(tid, _db_factory, payload.from_dt, payload.to_dt))
    t = TASKS.get(tid)
    return TaskOut(task_id=tid, status=t.status, progress=t.progress)

@batch_router.get("/tasks/{task_id}", response_model=TaskOut)
async def api_task_status(task_id: str):
    t = TASKS.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return TaskOut(task_id=task_id, status=t.status, progress=t.progress, result=t.result, error=t.error)
