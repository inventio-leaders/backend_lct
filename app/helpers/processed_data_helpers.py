from __future__ import annotations

from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Sequence, Tuple, List, Dict, Any

import pandas as pd
from sqlalchemy import select, func, update as sa_update, delete as sa_delete, insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import ProcessedData
from app.schemas.processed_data_schemas import ProcessedDataCreate, ProcessedDataUpdate

async def get_processed_record(db: AsyncSession, record_id: int) -> Optional[ProcessedData]:
    res = await db.execute(select(ProcessedData).where(ProcessedData.record_id == record_id))
    return res.scalar_one_or_none()


async def list_processed_data(
    db: AsyncSession,
    *,
    dt_from: Optional[datetime] = None,
    dt_to: Optional[datetime] = None,
    hour: Optional[int] = None,
    day_of_week: Optional[int] = None,
    is_weekend: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
    order_desc: bool = True,
) -> Tuple[Sequence[ProcessedData], int]:
    stmt = select(ProcessedData)
    cnt = select(func.count(ProcessedData.record_id))

    if dt_from is not None:
        stmt = stmt.where(ProcessedData.datetime >= dt_from)
        cnt = cnt.where(ProcessedData.datetime >= dt_from)

    if dt_to is not None:
        stmt = stmt.where(ProcessedData.datetime < dt_to)
        cnt = cnt.where(ProcessedData.datetime < dt_to)

    if hour is not None:
        stmt = stmt.where(ProcessedData.hour == hour)
        cnt = cnt.where(ProcessedData.hour == hour)

    if day_of_week is not None:
        stmt = stmt.where(ProcessedData.day_of_week == day_of_week)
        cnt = cnt.where(ProcessedData.day_of_week == day_of_week)

    if is_weekend is not None:
        stmt = stmt.where(ProcessedData.is_weekend == is_weekend)
        cnt = cnt.where(ProcessedData.is_weekend == is_weekend)

    stmt = stmt.order_by(
        ProcessedData.datetime.desc() if order_desc else ProcessedData.datetime.asc()
    ).limit(limit).offset(offset)

    items_res = await db.execute(stmt)
    cnt_res = await db.execute(cnt)

    items = items_res.scalars().all()
    total = cnt_res.scalar_one() or 0
    return items, total


async def create_processed_record(db: AsyncSession, payload: ProcessedDataCreate) -> ProcessedData:
    obj = ProcessedData(
        datetime=payload.datetime,
        hour=payload.hour,
        day_of_week=payload.day_of_week,
        is_weekend=payload.is_weekend,
        consumption_gvs=payload.consumption_gvs,
        consumption_hvs=payload.consumption_hvs,
        delta_gvs_hvs=payload.delta_gvs_hvs,
        temp_gvs_supply=payload.temp_gvs_supply,
        temp_gvs_return=payload.temp_gvs_return,
        temp_delta=payload.temp_delta,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_processed_record(
    db: AsyncSession, record_id: int, payload: ProcessedDataUpdate
) -> Optional[ProcessedData]:
    data = payload.model_dump(exclude_unset=True, round_trip=True)

    stmt = (
        sa_update(ProcessedData)
        .where(ProcessedData.record_id == record_id)
        .values(**data)
        .execution_options(synchronize_session="fetch")
    )
    res = await db.execute(stmt)
    if (res.rowcount or 0) == 0:
        return None

    await db.commit()
    return await get_processed_record(db, record_id)


async def delete_processed_record(db: AsyncSession, record_id: int) -> bool:
    res = await db.execute(sa_delete(ProcessedData).where(ProcessedData.record_id == record_id))
    await db.commit()
    return (res.rowcount or 0) > 0

GVS_REQUIRED = {
    "Дата",
    "Время суток, ч",
    "Подача, м3",
    "Обратка, м3",
    "Потребление за период, м3",
    "Т1 гвс, оС",
    "Т2 гвс, оС",
}

HVS_REQUIRED_MIN = {
    "Дата",
    "Время суток, ч",
    "Потребление за период, м3",
}

def _to_decimal(val: Any, q: str) -> Decimal:
    """
    Преобразовать в Decimal и квантизировать по шаблону q.
    Примеры q: '0.001', '0.01'.
    """
    if val is None or (isinstance(val, float) and (pd.isna(val))):
        val = 0
    d = Decimal(str(val))
    return d.quantize(Decimal(q), rounding=ROUND_HALF_UP)

def _parse_hour(hour_cell: Any) -> Optional[int]:
    """
    '0-1' -> 0;  '13-14' -> 13;  7 -> 7
    """
    if isinstance(hour_cell, str) and "-" in hour_cell:
        try:
            return int(hour_cell.split("-")[0].strip())
        except Exception:
            return None
    try:
        return int(hour_cell)
    except Exception:
        return None

def _normalize_gvs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = GVS_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Файл ГВС: отсутствуют колонки: {', '.join(sorted(missing))}")

    df["hour"] = df["Время суток, ч"].apply(_parse_hour)
    if df["hour"].isna().any():
        raise ValueError("Файл ГВС: не удалось распарсить 'Время суток, ч' в час (int)")

    df["date"] = pd.to_datetime(df["Дата"], dayfirst=True, errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Файл ГВС: не удалось распарсить 'Дата'")

    df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

    df = df.rename(
        columns={
            "Подача, м3": "gvs_supply_m3",
            "Обратка, м3": "gvs_return_m3",
            "Потребление за период, м3": "gvs_consumption_m3",
            "Т1 гвс, оС": "t1_supply",
            "Т2 гвс, оС": "t2_return",
        }
    )

    return df[
        ["datetime", "hour", "gvs_supply_m3", "gvs_return_m3", "gvs_consumption_m3", "t1_supply", "t2_return"]
    ]

def _normalize_hvs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = HVS_REQUIRED_MIN - set(df.columns)
    if missing:
        raise ValueError(f"Файл ХВС: отсутствуют колонки: {', '.join(sorted(missing))}")

    df["hour"] = df["Время суток, ч"].apply(_parse_hour)
    if df["hour"].isna().any():
        raise ValueError("Файл ХВС: не удалось распарсить 'Время суток, ч' в час (int)")

    df["date"] = pd.to_datetime(df["Дата"], dayfirst=True, errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Файл ХВС: не удалось распарсить 'Дата'")

    df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

    df = df.rename(
        columns={
            "Потребление за период, м3": "hvs_consumption_m3",
        }
    )

    return df[["datetime", "hour", "hvs_consumption_m3"]]

async def import_processed_data_from_excels(
    db: AsyncSession,
    files: List["UploadFile"],
    *,
    dedupe: bool = True,
) -> Dict[str, Any]:
    """
    Импортирует данные из одного или нескольких .xlsx.
    - Можно передать только ГВС, только ХВС или оба — строки объединятся по datetime+hour.
    - Если в каком-то наборе не хватает значений, числовые поля дополнительно заполняются нулями.
    - Dedupe=True: не вставлять записи с datetime, которые уже есть в БД.
    """
    gvs_frames: List[pd.DataFrame] = []
    hvs_frames: List[pd.DataFrame] = []

    for f in files:
        if not f.filename.lower().endswith(".xlsx"):
            raise ValueError(f"Поддерживаются только .xlsx (получен {f.filename})")

        content = await f.read()
        try:
            xls = pd.ExcelFile(content, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"Не удалось открыть {f.filename}: {e}")

        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)

            cols = set(df.columns)

            if GVS_REQUIRED.issubset(cols):
                gvs_frames.append(_normalize_gvs(df))
            elif HVS_REQUIRED_MIN.issubset(cols):
                hvs_frames.append(_normalize_hvs(df))
            else:
                continue

    if not gvs_frames and not hvs_frames:
        raise ValueError("Не найдены подходящие листы. Проверьте, что колонки соответствуют шаблонам ГВС/ХВС.")

    gvs = pd.concat(gvs_frames, ignore_index=True) if gvs_frames else pd.DataFrame(
        columns=["datetime", "hour", "gvs_supply_m3", "gvs_return_m3", "gvs_consumption_m3", "t1_supply", "t2_return"]
    )
    hvs = pd.concat(hvs_frames, ignore_index=True) if hvs_frames else pd.DataFrame(
        columns=["datetime", "hour", "hvs_consumption_m3"]
    )

    merged = pd.merge(
        gvs,
        hvs,
        on=["datetime", "hour"],
        how="outer",
        validate="one_to_one",
    ).sort_values(["datetime", "hour"])

    if merged.empty:
        return {"inserted": 0, "skipped_existing": 0, "total_rows": 0}

    merged["day_of_week"] = merged["datetime"].dt.weekday
    merged["is_weekend"] = merged["day_of_week"].isin([5, 6])

    for col in ["gvs_consumption_m3", "hvs_consumption_m3", "t1_supply", "t2_return"]:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0)

    merged["temp_delta"] = merged["t1_supply"] - merged["t2_return"]
    merged["delta_gvs_hvs"] = merged["gvs_consumption_m3"] - merged["hvs_consumption_m3"]

    merged = merged.rename(
        columns={
            "gvs_consumption_m3": "consumption_gvs",
            "hvs_consumption_m3": "consumption_hvs",
            "t1_supply": "temp_gvs_supply",
            "t2_return": "temp_gvs_return",
        }
    )

    records: List[Dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        dt_val: datetime = getattr(row, "datetime")
        hour_val: int = int(getattr(row, "hour"))

        rec = {
            "datetime": dt_val,
            "hour": hour_val,
            "day_of_week": int(getattr(row, "day_of_week")),
            "is_weekend": bool(getattr(row, "is_weekend")),
            "consumption_gvs": _to_decimal(getattr(row, "consumption_gvs"), "0.001"),
            "consumption_hvs": _to_decimal(getattr(row, "consumption_hvs"), "0.001"),
            "delta_gvs_hvs": _to_decimal(getattr(row, "delta_gvs_hvs"), "0.001"),
            "temp_gvs_supply": _to_decimal(getattr(row, "temp_gvs_supply"), "0.01"),
            "temp_gvs_return": _to_decimal(getattr(row, "temp_gvs_return"), "0.01"),
            "temp_delta": _to_decimal(getattr(row, "temp_delta"), "0.01"),
        }
        records.append(rec)

    total_rows = len(records)

    skipped_existing = 0
    if dedupe:
        CHUNK = 1000
        existing_set = set()
        for i in range(0, total_rows, CHUNK):
            chunk = records[i : i + CHUNK]
            dts = [r["datetime"] for r in chunk]
            res = await db.execute(
                select(ProcessedData.datetime).where(ProcessedData.datetime.in_(dts))
            )
            existing_set.update([row[0] for row in res.all()])
        if existing_set:
            new_records = [r for r in records if r["datetime"] not in existing_set]
            skipped_existing = total_rows - len(new_records)
            records = new_records

    if not records:
        return {"inserted": 0, "skipped_existing": skipped_existing, "total_rows": total_rows}

    CHUNK = 1000
    inserted = 0
    try:
        for i in range(0, len(records), CHUNK):
            chunk = records[i : i + CHUNK]
            await db.execute(insert(ProcessedData).values(chunk))
        await db.commit()
        inserted = len(records)
    except Exception:
        await db.rollback()
        raise

    return {
        "inserted": inserted,
        "skipped_existing": skipped_existing,
        "total_rows": total_rows,
    }