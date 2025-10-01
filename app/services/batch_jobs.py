from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from traceback import format_exc

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

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
    """Если записи модели нет — создаём минимальную, с путём к файлу весов."""
    m = await get_latest_model_by_name(db, name)
    if m:
        return m

    m = Models(
        training_date=datetime.utcnow(),
        last_retrained=datetime.utcnow(),
        metrics=json.dumps(metrics or {}),
        name=name,
        version=version,
        file_path=str((MODEL_DIR / file_name).as_posix()),
    )
    db.add(m)
    await db.commit()
    return m



async def forecast_job(tid: str, db_factory, horizon_hours: int):
    """Итеративный прогноз на horizon_hours вперёд (использует последние W часов)."""
    info = TASKS.get(tid)
    info.set(status="RUNNING", progress="prepare")

    try:
        fi = load_pickle("feature_info.pkl")
        W = int(fi["window_size"])
        feats = fi["narx_features"]

        model = load_keras("narx_lstm_model.h5")
        scaler = load_pickle("scaler_narx.pkl")

        async with db_factory() as db:
            m = await _ensure_model_meta(
                db,
                name="NARX-LSTM",
                file_name="narx_lstm_model.h5",
                version="disk-import",
                metrics={},
            )

            last_rows = await _fetch_processed_window(db, W)
            if len(last_rows) < W:
                info.set(status="FAILURE", error=f"Need {W} rows, got {len(last_rows)}")
                return

            X_hist = np.array([[ _row_to_features(r)[f] for f in feats ] for r in last_rows], dtype=float)
            Xs = scaler.transform(X_hist)

            preds: List[float] = []
            curr = Xs.copy()

            for _ in range(horizon_hours):
                x = curr[-W:, :][None, :, :]  # (1, W, F)
                y_s = await run_in_threadpool(model.predict, x, verbose=0) # scaled (1,1)

                pad = np.concatenate([y_s, np.zeros((1, len(feats) - 1))], axis=1)
                y = float(scaler.inverse_transform(pad)[0, 0])
                y = max(0.0, y)
                preds.append(round(y, 3))

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
    Скан аномалий LSTM-AE по диапазону [dt_from, dt_to).
    Если в диапазоне меньше W записей, используем исторический префикс (до W-1 строк) из БД,
    чтобы сформировать окна нужной длины и посчитать аномалии для часов внутри диапазона.
    """

    def _to_naive_utc(dt):
        if dt is None:
            return None
        return dt if dt.tzinfo is None else dt.astimezone(timezone.utc).replace(tzinfo=None)

    dt_from = _to_naive_utc(dt_from)
    dt_to = _to_naive_utc(dt_to)

    info = TASKS.get(tid)
    info.set(status="RUNNING", progress="prepare")

    try:
        fi = load_pickle("feature_info.pkl")
        W = int(fi["window_size"])
        feats_ae = fi["ae_features"]

        model = load_keras("ae_model.h5")
        scaler = load_pickle("scaler_ae.pkl")

        thr = None
        async with db_factory() as db:
            meta = await get_latest_model_by_name(db, "LSTM-AE")
            if meta:
                try:
                    metrics = json.loads(meta.metrics) if isinstance(meta.metrics, str) else meta.metrics
                    thr = float(metrics.get("ae_threshold")) if metrics else None
                except Exception:
                    thr = None
        if thr is None:
            thr = 0.02

        async with db_factory() as db:
            q_target = select(ProcessedData)
            if dt_from:
                q_target = q_target.where(ProcessedData.datetime >= dt_from)
            if dt_to:
                q_target = q_target.where(ProcessedData.datetime < dt_to)
            q_target = q_target.order_by(ProcessedData.datetime.asc())
            res = await db.execute(q_target)
            target_rows: List[ProcessedData] = res.scalars().all()

            if not target_rows:
                q_all = select(ProcessedData).order_by(ProcessedData.datetime.asc())
                res_all = await db.execute(q_all)
                all_rows: List[ProcessedData] = res_all.scalars().all()
                if len(all_rows) < W:
                    info.set(status="FAILURE", error=f"Need at least {W} rows globally, found {len(all_rows)}")
                    return
                combined_rows = all_rows
                prefix_len = 0
                mark_from_dt = None
                mark_to_dt = None
            else:
                if len(target_rows) < W:
                    first_dt = target_rows[0].datetime
                    q_prefix = (
                        select(ProcessedData)
                        .where(ProcessedData.datetime < first_dt)
                        .order_by(ProcessedData.datetime.desc())
                        .limit(W - 1)
                    )
                    res_pref = await db.execute(q_prefix)
                    prefix_rows_desc = res_pref.scalars().all()
                    prefix_rows = list(reversed(prefix_rows_desc))
                else:
                    prefix_rows = []

                combined_rows = prefix_rows + target_rows
                prefix_len = len(prefix_rows)

                mark_from_dt = target_rows[0].datetime
                mark_to_dt = target_rows[-1].datetime if not dt_to else dt_to

                if len(combined_rows) < W:
                    q_more = (
                        select(ProcessedData)
                        .where(ProcessedData.datetime < (target_rows[0].datetime if target_rows else dt_to))
                        .order_by(ProcessedData.datetime.desc())
                        .limit(W - len(combined_rows))
                    )
                    res_more = await db.execute(q_more)
                    more_desc = res_more.scalars().all()
                    more_prefix = list(reversed(more_desc))
                    combined_rows = more_prefix + combined_rows
                    prefix_len += len(more_prefix)

                if len(combined_rows) < W:
                    from sqlalchemy import func
                    res_cnt = await db.execute(select(func.count(ProcessedData.record_id)))
                    total_cnt = int(res_cnt.scalar_one() or 0)
                    info.set(status="FAILURE", error=f"Need at least {W} rows globally, found {total_cnt}")
                    return

            X_all = np.array(
                [[_row_to_features_ae(r)[f] for f in feats_ae] for r in combined_rows],
                dtype=float,
            )
            Xs_all = scaler.transform(X_all)

            created = 0
            for i in range(len(combined_rows)):
                if i + 1 < W:
                    continue

                end_row = combined_rows[i]

                if mark_from_dt and end_row.datetime < mark_from_dt:
                    continue
                if mark_to_dt and end_row.datetime >= mark_to_dt:
                    continue

                window_Xs = Xs_all[i - W + 1 : i + 1]
                x = window_Xs[None, :, :]
                x_rec = await run_in_threadpool(model.predict, x, verbose=0)
                mse = float(np.mean(np.square(window_Xs - x_rec[0])))

                if mse > thr:
                    severity = _severity_from_mse(mse, thr)
                    await create_anomaly(
                        db,
                        pd_row=end_row,
                        mse_error=mse,
                        threshold=thr,
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

