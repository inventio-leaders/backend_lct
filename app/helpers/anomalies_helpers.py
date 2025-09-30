from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, insert, func, update as sa_update, delete as sa_delete

from app.models.models import Forecasts, ProcessedData, Anomalies, SeverityLevel
from app.schemas.anomalies_schemas import AnomalyCreate, AnomalyUpdate

from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

async def _ensure_fk_forecast(db: AsyncSession, forecast_id: Optional[int]) -> None:
    if forecast_id is None:
        return
    res = await db.execute(
        select(Forecasts.forecast_id).where(Forecasts.forecast_id == forecast_id)
    )
    if res.scalar_one_or_none() is None:
        raise ValueError("Forecast does not exist")


async def get_anomaly(db: AsyncSession, anomaly_id: int) -> Optional[Anomalies]:
    res = await db.execute(select(Anomalies).where(Anomalies.anomaly_id == anomaly_id))
    return res.scalar_one_or_none()


async def list_anomalies(
    db: AsyncSession,
    *,
    forecast_id: Optional[int] = None,
    severity_level: Optional[SeverityLevel] = None,
    is_confirmed: Optional[bool] = None,
    dt_from: Optional[datetime] = None,
    dt_to: Optional[datetime] = None,
    mse_from: Optional[float] = None,
    mse_to: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    order_desc: bool = True,
) -> Tuple[Sequence[Anomalies], int]:
    stmt = select(Anomalies)
    cnt = select(func.count(Anomalies.anomaly_id))

    if forecast_id is not None:
        stmt = stmt.where(Anomalies.forecast_id == forecast_id)
        cnt = cnt.where(Anomalies.forecast_id == forecast_id)

    if severity_level is not None:
        stmt = stmt.where(Anomalies.severity_level == severity_level)
        cnt = cnt.where(Anomalies.severity_level == severity_level)

    if is_confirmed is not None:
        stmt = stmt.where(Anomalies.is_confirmed == is_confirmed)
        cnt = cnt.where(Anomalies.is_confirmed == is_confirmed)

    if dt_from is not None:
        stmt = stmt.where(Anomalies.datetime >= dt_from)
        cnt = cnt.where(Anomalies.datetime >= dt_from)

    if dt_to is not None:
        stmt = stmt.where(Anomalies.datetime < dt_to)
        cnt = cnt.where(Anomalies.datetime < dt_to)

    if mse_from is not None:
        stmt = stmt.where(Anomalies.mse_error >= mse_from)
        cnt = cnt.where(Anomalies.mse_error >= mse_from)

    if mse_to is not None:
        stmt = stmt.where(Anomalies.mse_error <= mse_to)
        cnt = cnt.where(Anomalies.mse_error <= mse_to)

    stmt = stmt.order_by(
        Anomalies.datetime.desc() if order_desc else Anomalies.datetime.asc()
    ).limit(limit).offset(offset)

    items_res = await db.execute(stmt)
    cnt_res = await db.execute(cnt)

    items = items_res.scalars().all()
    total = cnt_res.scalar_one() or 0
    return items, total


async def create_anomaly(db: AsyncSession, payload: AnomalyCreate) -> Anomalies:
    await _ensure_fk_forecast(db, payload.forecast_id)
    now = datetime.utcnow()
    created_at = payload.created_at or now
    updated_at = payload.updated_at or now

    obj = Anomalies(
        datetime=payload.datetime,
        is_confirmed=payload.is_confirmed,
        operator_notes=payload.operator_notes,
        created_at=created_at,
        updated_at=updated_at,
        mse_error=payload.mse_error,
        severity_level=payload.severity_level,
        forecast_id=payload.forecast_id,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_anomaly(
    db: AsyncSession, anomaly_id: int, payload: AnomalyUpdate
) -> Optional[Anomalies]:
    data = payload.model_dump(exclude_unset=True, round_trip=True)

    await _ensure_fk_forecast(db, data.get("forecast_id"))

    data.setdefault("updated_at", datetime.utcnow())

    stmt = (
        sa_update(Anomalies)
        .where(Anomalies.anomaly_id == anomaly_id)
        .values(**data)
        .execution_options(synchronize_session="fetch")
    )
    res = await db.execute(stmt)
    if (res.rowcount or 0) == 0:
        return None

    await db.commit()
    return await get_anomaly(db, anomaly_id)


async def delete_anomaly(db: AsyncSession, anomaly_id: int) -> bool:
    res = await db.execute(sa_delete(Anomalies).where(Anomalies.anomaly_id == anomaly_id))
    await db.commit()
    return (res.rowcount or 0) > 0

def _q(val: float | Decimal, q: str) -> Decimal:
    d = Decimal(str(val))
    return d.quantize(Decimal(q), rounding=ROUND_HALF_UP)

def _severity_from_pct(pct: float) -> SeverityLevel:
    """
    Перевод процента отклонения в уровень серьёзности:
      <= 20%  -> Низкий
      20–40%  -> Средний
      > 40%   -> Высокий
    """
    if pct <= 20:
        return SeverityLevel.Низкий
    elif pct <= 40:
        return SeverityLevel.Средний
    return SeverityLevel.Высокий


async def compute_anomalies_from_processed(
    db: AsyncSession,
    *,
    dt_from: Optional[datetime] = None,
    dt_to: Optional[datetime] = None,
    threshold_pct: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает (df_all, df_anoms).
    df_all включает служебные вычисляемые колонки.
    """
    stmt = select(
        ProcessedData.datetime,
        ProcessedData.hour,
        ProcessedData.consumption_hvs,
        ProcessedData.consumption_gvs,
    )
    if dt_from is not None:
        stmt = stmt.where(ProcessedData.datetime >= dt_from)
    if dt_to is not None:
        stmt = stmt.where(ProcessedData.datetime < dt_to)

    stmt = stmt.order_by(ProcessedData.datetime.asc())
    res = await db.execute(stmt)
    rows = res.all()

    if not rows:
        # Пустой результат — пустые датафреймы
        df_all = pd.DataFrame(columns=[
            "datetime", "hour", "ХВС_ИТП", "ГВС_ОДПУ", "Разница", "Отклонение_%", "Аномалия"
        ])
        return df_all, df_all.copy()

    # Преобразуем в DataFrame
    df = pd.DataFrame(rows, columns=["datetime", "hour", "consumption_hvs", "consumption_gvs"])

    # Вычисления
    df["ХВС_ИТП"] = df["consumption_hvs"].astype(float)
    df["ГВС_ОДПУ"] = df["consumption_gvs"].astype(float)
    df["Разница"] = df["ХВС_ИТП"] - df["ГВС_ОДПУ"]

    denom = (df["ХВС_ИТП"] + df["ГВС_ОДПУ"]) / 2.0
    df["Отклонение_%"] = (df["Разница"].abs() / denom.replace(0, pd.NA)) * 100.0
    df["Отклонение_%"] = df["Отклонение_%"].fillna(0.0)

    df["Аномалия"] = df["Отклонение_%"] > float(threshold_pct)

    df_all = df[[
        "datetime", "hour", "ХВС_ИТП", "ГВС_ОДПУ", "Разница", "Отклонение_%", "Аномалия"
    ]].copy()

    df_anoms = df_all[df_all["Аномалия"]].copy()

    return df_all, df_anoms

async def persist_anomalies(db: AsyncSession, df_anoms: pd.DataFrame) -> Tuple[int, int]:
    if df_anoms.empty:
        return 0, 0

    try:
        dt_idx  = df_anoms.columns.get_loc("datetime")
        diff_idx = df_anoms.columns.get_loc("Разница")
        pct_idx  = df_anoms.columns.get_loc("Отклонение_%")
    except KeyError as e:
        raise ValueError(f"В df_anoms не найдена колонка: {e}")

    dts = df_anoms.iloc[:, dt_idx].tolist()

    res = await db.execute(select(Anomalies.datetime).where(Anomalies.datetime.in_(dts)))
    existing = {r[0] for r in res.all()}

    to_insert: List[Dict[str, Any]] = []
    now = datetime.utcnow()

    for row in df_anoms.itertuples(index=False, name=None):
        dt = row[dt_idx]
        if dt in existing:
            continue

        diff = float(row[diff_idx])
        pct = float(row[pct_idx])
        sev = _severity_from_pct(pct)

        rec = {
            "datetime": dt,
            "is_confirmed": False,
            "operator_notes": None,
            "created_at": now,
            "updated_at": now,
            "mse_error": _q(diff * diff, "0.000001"),
            "severity_level": sev,
            "forecast_id": None,
        }
        to_insert.append(rec)

    if not to_insert:
        return 0, len(df_anoms)

    await db.execute(insert(Anomalies).values(to_insert))
    await db.commit()
    return len(to_insert), len(df_anoms) - len(to_insert)


async def build_anomalies_xlsx_bytes(
    df_all: pd.DataFrame,
    df_anoms: pd.DataFrame,
    *,
    filename_all_sheet: str = "Все данные",
    filename_anom_sheet: str = "Аномалии",
) -> bytes:
    """
    Собирает XLSX (две вкладки) в память и возвращает bytes.
    """
    df_all_out = df_all.copy()
    df_all_out["Отклонение_%"] = df_all_out["Отклонение_%"].round(2)

    df_anoms_out = df_anoms.copy()
    if not df_anoms_out.empty:
        df_anoms_out["Отклонение_%"] = df_anoms_out["Отклонение_%"].round(2)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_all_out.to_excel(writer, index=False, sheet_name=filename_all_sheet)
        df_anoms_out.to_excel(writer, index=False, sheet_name=filename_anom_sheet)
    buf.seek(0)
    return buf.read()