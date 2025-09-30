from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from traceback import format_exc

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.task_manager import TASKS
from app.services.ml_io import load_pickle, load_keras, MODEL_DIR
from app.helpers.ml_db import (
    get_latest_model_by_name,
    create_forecast,
    create_anomaly,
    _severity_from_mse,
)
from app.models.models import ProcessedData, Models


async def _fetch_processed_window(db: AsyncSession, last_hours: int) -> List[ProcessedData]:
    q = select(ProcessedData).order_by(ProcessedData.datetime.desc()).limit(last_hours)
    res = await db.execute(q)
    rows = list(reversed(res.scalars().all()))
    return rows

def _row_to_features(pd_row: ProcessedData) -> Dict[str, float]:
    return {
        "Consumption_GVS": float(pd_row.consumption_gvs),
        "Consumption_HVS": float(pd_row.consumption_hvs),
        "Temp_GVS_Supply": float(pd_row.temp_gvs_supply),
        "Temp_GVS_Return": float(pd_row.temp_gvs_return),
        "Delta_GVS_HVS": float(pd_row.delta_gvs_hvs),
        "Temp_Delta": float(pd_row.temp_delta),
        "Hour": int(pd_row.hour),
        "DayOfWeek": int(pd_row.day_of_week),
        "IsWeekend": int(pd_row.is_weekend),
    }

def _row_to_features_ae(pd_row: ProcessedData) -> Dict[str, float]:
    d = _row_to_features(pd_row)
    d.pop("Consumption_GVS", None)
    return d

async def _ensure_model_meta(
    db: AsyncSession,
    *,
    name: str,
    file_name: str,
    version: str = "disk-import",
    metrics: Optional[dict] = None,
) -> Models:
    """
    Если записи модели нет — создаём минимальную,
    указывая путь к уже существующему файлу весов.
    """
    m = await get_latest_model_by_name(db, name)
    if m:
        return m

    m = Models(
        training_date=datetime.utcnow(),
        last_retrained=datetime.utcnow(),
        metrics=json.dumps(metrics or {}),  # например {"rmse": ..., "mae": ...}
        name=name,
        version=version,
        file_path=str((MODEL_DIR / file_name).as_posix()),
    )
    db.add(m)
    await db.commit()
    return m


async def forecast_job(tid: str, db_factory, horizon_hours: int):
    """
    Итеративный прогноз на horizon_hours вперёд.
    Использует последние W часов из ProcessedData.
    Если отсутствует запись Models для NARX-LSTM — регистрирует автоматически.
    """
    info = TASKS.get(tid)
    info.set(status="RUNNING", progress="prepare")

    try:
        fi = load_pickle("feature_info.pkl")
        W = int(fi["window_size"])
        feats = fi["narx_features"]

        model = load_keras("narx_lstm_model.h5")     # внутри load_keras: compile=False
        scaler = load_pickle("scaler_narx.pkl")

        async with db_factory() as db:
            m = await _ensure_model_meta(
                db,
                name="NARX-LSTM",
                file_name="narx_lstm_model.h5",
                version="disk-import",
                metrics={},  # если нужно — можешь проставить фактические метрики
            )

            last_rows = await _fetch_processed_window(db, W)
            if len(last_rows) < W:
                info.set(status="FAILURE", error=f"Need {W} rows, got {len(last_rows)}")
                return

            # историческое окно -> масштабирование
            X_hist = np.array([[ _row_to_features(r)[f] for f in feats ] for r in last_rows], dtype=float)
            Xs = scaler.transform(X_hist)

            preds: List[float] = []
            curr = Xs.copy()

            # итеративный прогноз
            for h in range(horizon_hours):
                x = curr[-W:, :][None, :, :]  # (1, W, F)
                y_s = model.predict(x, verbose=0)  # (1,1) scaled

                # inverse только целевой
                pad = np.concatenate([y_s, np.zeros((1, len(feats) - 1))], axis=1)
                y = float(scaler.inverse_transform(pad)[0, 0])
                preds.append(round(y, 3))

                # формируем "следующую" строку фич
                next_features = _row_to_features(last_rows[-1])
                next_hour_before = next_features["Hour"]

                next_features["Consumption_GVS"] = y
                next_features["Hour"] = (next_features["Hour"] + 1) % 24
                if next_hour_before == 23:
                    next_features["DayOfWeek"] = (next_features["DayOfWeek"] + 1) % 7
                    next_features["IsWeekend"] = 1 if next_features["DayOfWeek"] >= 5 else 0

                new_row_scaled = scaler.transform(
                    np.array([[next_features[f] for f in feats]], dtype=float)
                )[0]
                curr = np.vstack([curr, new_row_scaled])

            base = last_rows[-1]
            created = 0
            for i, y in enumerate(preds, 1):
                fc = await create_forecast(db, pd_row=base, value=y, confidence=1.0, model=m)
                fc.datetime = base.datetime + timedelta(hours=i)
                created += 1

            await db.commit()

        info.set(status="SUCCESS", result={"created": created}, progress="done")

    except Exception as e:
        info.set(status="FAILURE", error=f"{e}\n{format_exc()}")
        return


async def anomaly_scan_job(tid: str, db_factory, dt_from: Optional[datetime], dt_to: Optional[datetime]):
    """
    Скан аномалий LSTM-AE по всему диапазону (скользящее окно W).
    Если записи модели в БД нет — используем дефолтный threshold и продолжаем.
    """

    def _to_naive_utc(dt):
        if dt is None: return None
        return dt if dt.tzinfo is None else dt.astimezone(timezone.utc).replace(tzinfo=None)

    dt_from = _to_naive_utc(dt_from)
    dt_to = _to_naive_utc(dt_to)

    info = TASKS.get(tid)
    info.set(status="RUNNING", progress="prepare")

    try:
        fi = load_pickle("feature_info.pkl")
        W = int(fi["window_size"])
        feats_ae = fi["ae_features"]

        model = load_keras("ae_model.h5")            # compile=False внутри load_keras
        scaler = load_pickle("scaler_ae.pkl")

        # 1) Пытаемся взять threshold из метаданных модели (если есть)
        thr = None
        async with db_factory() as db:
            meta = await get_latest_model_by_name(db, "LSTM-AE")
            if meta:
                try:
                    metrics = json.loads(meta.metrics) if isinstance(meta.metrics, str) else meta.metrics
                    thr = float(metrics.get("ae_threshold")) if metrics else None
                except Exception:
                    thr = None

        # 2) Фоллбек, если метаданных нет/повреждены
        if thr is None:
            thr = 0.02  # аккуратный дефолт

        # 3) Берём данные и сканим
        async with db_factory() as db:
            q = select(ProcessedData)
            if dt_from:
                q = q.where(ProcessedData.datetime >= dt_from)
            if dt_to:
                q = q.where(ProcessedData.datetime < dt_to)
            q = q.order_by(ProcessedData.datetime.asc())
            res = await db.execute(q)
            rows = res.scalars().all()

            if len(rows) < W:
                info.set(status="FAILURE", error=f"Need at least {W} rows")
                return

            created = 0
            buf: List[List[float]] = []

            for i, r in enumerate(rows):
                buf.append([_row_to_features_ae(r)[f] for f in feats_ae])
                if len(buf) < W:
                    continue

                X = np.array(buf[-W:], dtype=float)
                Xs = scaler.transform(X)
                x = Xs[None, :, :]
                x_rec = model.predict(x, verbose=0)
                mse = float(np.mean(np.square(x - x_rec)))

                if mse > thr:
                    severity = _severity_from_mse(mse, thr)
                    await create_anomaly(
                        db,
                        pd_row=rows[i],       # текущий конец окна
                        mse_error=mse,
                        threshold=thr,        # используется для вычисления severity
                        severity=severity,
                        forecast=None,
                        is_confirmed=False,
                        operator_notes=None,
                    )
                    created += 1

            await db.commit()

        info.set(status="SUCCESS", result={"anomalies_created": created}, progress="done")

    except Exception as e:
        info.set(status="FAILURE", error=f"{e}\n{format_exc()}")
        return

