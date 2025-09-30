from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from starlette.concurrency import run_in_threadpool

from app.services.task_manager import TASKS
from app.services.ml_io import save_pickle, save_keras, new_version, MODEL_DIR
from app.models.models import Models, ProcessedData


# ---------------------- time helpers ----------------------

def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Привести datetime к naive UTC, чтобы сравнивать с колонкой DateTime (без TZ)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt  # уже naive
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _fix_range(dt_from: Optional[datetime], dt_to: Optional[datetime]) -> tuple[Optional[datetime], Optional[datetime]]:
    """Нормализуем границы. Если from>=to — снимаем фильтр."""
    f = _to_naive_utc(dt_from)
    t = _to_naive_utc(dt_to)
    if f and t and f >= t:
        return None, None
    return f, t


# ---------------------- DB helpers ----------------------

async def processed_stats(db: AsyncSession):
    """Вернёт (count, min_datetime, max_datetime) по processed_data."""
    res = await db.execute(
        select(
            func.count(ProcessedData.record_id),
            func.min(ProcessedData.datetime),
            func.max(ProcessedData.datetime),
        )
    )
    cnt, dt_min, dt_max = res.fetchone()
    return int(cnt or 0), dt_min, dt_max


async def fetch_processed_df(db: AsyncSession, dt_from: Optional[datetime], dt_to: Optional[datetime]) -> pd.DataFrame:
    """
    Тянем processed_data; если фильтр дал 0 строк — фоллбек на весь диапазон.
    Даты приводим к naive UTC перед запросом.
    """
    dt_from, dt_to = _fix_range(dt_from, dt_to)

    base_q = select(ProcessedData).order_by(ProcessedData.datetime.asc())
    q = base_q
    if dt_from is not None:
        q = q.where(ProcessedData.datetime >= dt_from)
    if dt_to is not None:
        q = q.where(ProcessedData.datetime < dt_to)

    res = await db.execute(q)
    rows = res.scalars().all()

    # фоллбек без фильтра, если выборка пустая
    if not rows and (dt_from is not None or dt_to is not None):
        res_f = await db.execute(base_q)
        rows = res_f.scalars().all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "datetime": r.datetime,  # naive
        "Hour": r.hour,
        "DayOfWeek": r.day_of_week,
        "IsWeekend": r.is_weekend,
        "Consumption_GVS": float(r.consumption_gvs),
        "Consumption_HVS": float(r.consumption_hvs),
        "Delta_GVS_HVS": float(r.delta_gvs_hvs),
        "Temp_GVS_Supply": float(r.temp_gvs_supply),
        "Temp_GVS_Return": float(r.temp_gvs_return),
        "Temp_Delta": float(r.temp_delta),
        "processed_id": r.record_id,
    } for r in rows])
    return df


# ---------------------- ML helpers ----------------------

def _create_sequences(data: np.ndarray, target: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(target[i])
    return np.array(X), np.array(y)


# ---------------------- main job ----------------------

async def train_job(
    tid: str,
    db_factory,  # callable -> AsyncSession
    *,
    train_narx: bool,
    train_ae: bool,
    window: int = 24,
    dt_from: Optional[datetime],
    dt_to: Optional[datetime],
    ae_threshold_percentile: int = 99
):
    info = TASKS.get(tid)
    info.set(status="RUNNING", progress="fetch data")

    try:
        # 1) Загружаем данные
        async with db_factory() as db:
            df = await fetch_processed_df(db, dt_from, dt_to)

        if df.empty:
            # добавим полезную статистику, чтобы было видно, что в БД вообще есть
            async with db_factory() as db:
                cnt, dt_min, dt_max = await processed_stats(db)
            msg = "No processed_data in range"
            if cnt:
                msg += f" | available_count={cnt}, available_min={dt_min}, available_max={dt_max}"
            info.set(status="FAILURE", error=msg)
            return

        feature_columns = [
            'Consumption_GVS', 'Consumption_HVS', 'Temp_GVS_Supply', 'Temp_GVS_Return',
            'Delta_GVS_HVS', 'Temp_Delta', 'Hour', 'DayOfWeek', 'IsWeekend'
        ]
        ae_feature_columns = [
            'Consumption_HVS', 'Temp_GVS_Supply', 'Temp_GVS_Return',
            'Delta_GVS_HVS', 'Temp_Delta', 'Hour', 'DayOfWeek', 'IsWeekend'
        ]
        target_column = 'Consumption_GVS'

        # сохраняем мета-файл с фичами — он нужен и для инференса AE
        save_pickle({
            "narx_features": feature_columns,
            "ae_features": ae_feature_columns,
            "window_size": window,
            "target_column": target_column
        }, "feature_info.pkl")

        results: Dict[str, dict] = {}
        any_trained = False

        # 2) Обучение NARX-LSTM
        if train_narx:
            if len(df) <= window:
                results["narx"] = {"skipped": True, "reason": f"not enough rows: len={len(df)} <= window={window}"}
            else:
                info.set(progress="train NARX")
                scaler_narx = MinMaxScaler()
                data_narx = scaler_narx.fit_transform(df[feature_columns].values)
                Xn, yn = _create_sequences(data_narx, df[target_column].values, window)
                if len(Xn) == 0:
                    results["narx"] = {"skipped": True, "reason": f"no sequences for window={window}"}
                else:
                    s1 = int(0.8 * len(Xn)); s2 = int(0.9 * len(Xn))
                    # защита от слишком маленьких датасетов
                    s1 = max(1, min(s1, len(Xn)-2))
                    s2 = max(s1+1, min(s2, len(Xn)-1))

                    Xtr, Xval, Xte = Xn[:s1], Xn[s1:s2], Xn[s2:]
                    ytr, yval, yte = yn[:s1], yn[s1:s2], yn[s2:]

                    model_narx = Sequential([
                        LSTM(128, activation='relu', input_shape=(window, len(feature_columns)), return_sequences=True),
                        Dropout(0.2),
                        LSTM(64, activation='relu'),
                        Dropout(0.2),
                        Dense(32, activation='relu'),
                        Dense(1, activation='relu')
                    ])
                    # важно: без строковых метрик; метрики посчитаем руками
                    model_narx.compile(optimizer=Adam(1e-3), loss='mse')

                    # уводим тяжёлую работу в поток
                    await run_in_threadpool(
                        model_narx.fit,
                        Xtr, ytr,
                        validation_data=(Xval, yval), epochs=50, batch_size=32, verbose=0
                    )
                    # обратное масштабирование для оценки
                    def _inv(vec):
                        pad = np.concatenate([vec.reshape(-1, 1), np.zeros((len(vec), len(feature_columns) - 1))], axis=1)
                        return scaler_narx.inverse_transform(pad)[:, 0]

                    yte_inv = _inv(yte)
                    yhat_scaled = await run_in_threadpool(model_narx.predict, Xte, verbose=0)
                    yhat_inv = _inv(yhat_scaled.reshape(-1))
                    mae = float(mean_absolute_error(yte_inv, yhat_inv))
                    rmse = float(np.sqrt(mean_squared_error(yte_inv, yhat_inv)))

                    # сохраняем артефакты
                    save_keras(model_narx, "narx_lstm_model.h5")
                    save_pickle(scaler_narx, "scaler_narx.pkl")

                    version = new_version("NARX", max_len=20)
                    results["narx"] = {"version": version, "rmse": rmse, "mae": mae}
                    any_trained = True

                    async with db_factory() as db:
                        m = Models(
                            training_date=datetime.utcnow(),
                            last_retrained=datetime.utcnow(),
                            metrics=json.dumps({"rmse": rmse, "mae": mae}),
                            name="NARX-LSTM",
                            version=version,
                            file_path=str((MODEL_DIR / "narx_lstm_model.h5").as_posix()),
                        )
                        db.add(m)
                        await db.commit()

        # 3) Обучение LSTM-AE
        if train_ae:
            if len(df) <= window:
                results["ae"] = {"skipped": True, "reason": f"not enough rows: len={len(df)} <= window={window}"}
            else:
                info.set(progress="train AE")
                scaler_ae = MinMaxScaler()
                data_ae = scaler_ae.fit_transform(df[ae_feature_columns].values)
                Xa, Ya = _create_sequences(data_ae, data_ae, window)
                if len(Xa) == 0:
                    results["ae"] = {"skipped": True, "reason": f"no sequences for window={window}"}
                else:
                    model_ae = Sequential([
                        LSTM(64, activation='relu', return_sequences=True, input_shape=(window, len(ae_feature_columns))),
                        LSTM(32, activation='relu', return_sequences=False),
                        RepeatVector(window),
                        LSTM(32, activation='relu', return_sequences=True),
                        LSTM(64, activation='relu', return_sequences=True),
                        TimeDistributed(Dense(len(ae_feature_columns)))
                    ])
                    model_ae.compile(optimizer=Adam(1e-3), loss='mae')

                    s1 = int(0.8 * len(Xa)); s2 = int(0.9 * len(Xa))
                    s1 = max(1, min(s1, len(Xa)-1))
                    s2 = max(s1, min(s2, len(Xa)))  # может не быть валидации

                    Xtr = Xa[:s1]
                    Xval = Xa[s1:s2] if s2 > s1 else None
                    val_kwargs = {"validation_data": (Xval, Xval)} if (Xval is not None and len(Xval)) else {}

                    # обучение в тредпуле
                    await run_in_threadpool(
                        model_ae.fit,
                        Xtr, Xtr,
                        epochs=50, batch_size=32, verbose=0, **val_kwargs
                    )

                    # threshold по валидации (если нет — по трейну)
                    Xref = Xval if (Xval is not None and len(Xval)) else Xtr
                    Xref_rec = await run_in_threadpool(model_ae.predict, Xref, verbose=0)
                    mse_val = np.mean(np.square(Xref - Xref_rec), axis=(1, 2))
                    thr = float(np.percentile(mse_val, ae_threshold_percentile))

                    save_keras(model_ae, "ae_model.h5")
                    save_pickle(scaler_ae, "scaler_ae.pkl")

                    version = new_version("AE", max_len=20)
                    results["ae"] = {"version": version, "ae_threshold": thr}
                    any_trained = True

                    async with db_factory() as db:
                        m = Models(
                            training_date=datetime.utcnow(),
                            last_retrained=datetime.utcnow(),
                            metrics=json.dumps({"ae_threshold": thr}),
                            name="LSTM-AE",
                            version=version,
                            file_path=str((MODEL_DIR / "ae_model.h5").as_posix()),
                        )
                        db.add(m)
                        await db.commit()

        if any_trained:
            info.set(status="SUCCESS", result=results, progress="done")
        else:
            info.set(status="FAILURE", error=f"Nothing trained: {results}", progress="done")

    except Exception as e:
        info.set(status="FAILURE", error=f"{e}\n{traceback.format_exc()}")
        return
