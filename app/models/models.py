from fastapi import Depends
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTable, SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
    DECIMAL,
    Enum as SAEnum,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from app.database import get_db, Base
import enum

class MeasurementType(enum.Enum):
    ПОДАЧА = "Подача"
    ОБРАТКА = "Обратка"
    ПОТРЕБЛЕНИЕ = "Потребление"
    T1 = "T1"
    T2 = "T2"


class SeverityLevel(enum.Enum):
    НИЗКИЙ = "Низкий"
    СРЕДНИЙ = "Средний"
    ВЫСОКИЙ = "Высокий"


class EventLabel(enum.Enum):
    УТЕЧКА = "Реальная утечка"
    ЛОЖНОЕ = "Ложное срабатывание"
    НОРМА = "Норма"


class User(SQLAlchemyBaseUserTable[int], Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role_id = Column(Integer)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)


class Sensors(Base):
    __tablename__ = "sensors"
    sensor_id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)  # "ГВС", "ХВС"
    location = Column(String(100), nullable=False)

    raw_data = relationship("RawData", back_populates="sensor")


class RawData(Base):
    __tablename__ = "raw_data"
    record_id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, nullable=False)
    sensor_id = Column(Integer, ForeignKey("sensors.sensor_id"), nullable=False)
    value = Column(DECIMAL(10, 3), nullable=False)

    measurement_type = Column(
        SAEnum(
            MeasurementType,
            name="measurement_type_enum",
            metadata=Base.metadata,
        ),
        nullable=False,
    )

    sensor = relationship("Sensors", back_populates="raw_data")


class ProcessedData(Base):
    __tablename__ = "processed_data"
    record_id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, nullable=False)
    hour = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    is_weekend = Column(Boolean, nullable=False)
    consumption_gvs = Column(DECIMAL(8, 3), nullable=False)
    consumption_hvs = Column(DECIMAL(8, 3), nullable=False)
    delta_gvs_hvs = Column(DECIMAL(8, 3), nullable=False)
    temp_gvs_supply = Column(DECIMAL(5, 2), nullable=False)
    temp_gvs_return = Column(DECIMAL(5, 2), nullable=False)
    temp_delta = Column(DECIMAL(5, 2), nullable=False)

    forecasts = relationship("Forecasts", back_populates="processed_data")


class Models(Base):
    __tablename__ = "models"
    model_id = Column(Integer, primary_key=True)
    training_date = Column(DateTime, nullable=False)
    last_retrained = Column(DateTime, nullable=False)
    metrics = Column(String, nullable=False)  # JSON: {"rmse": 0.05, "f1": 0.92}
    name = Column(String(50), nullable=False)  # "NARX-LSTM", "LSTM-AE"
    version = Column(String(20), nullable=False)
    file_path = Column(String(255), nullable=False)

    forecasts = relationship("Forecasts", back_populates="model")


class Forecasts(Base):
    __tablename__ = "forecasts"
    forecast_id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, nullable=False)
    predicted_consumption_gvs = Column(DECIMAL(8, 3), nullable=False)
    model_version = Column(String(20), nullable=False)
    confidence_score = Column(DECIMAL(4, 3), nullable=False)

    processed_data_id = Column(Integer, ForeignKey("processed_data.record_id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.model_id"), nullable=False)

    processed_data = relationship("ProcessedData", back_populates="forecasts")
    model = relationship("Models", back_populates="forecasts")

    anomaly = relationship("Anomalies", uselist=False, back_populates="forecast")


class Anomalies(Base):
    __tablename__ = "anomalies"
    anomaly_id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, nullable=False)
    is_confirmed = Column(Boolean, nullable=False)
    operator_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    mse_error = Column(DECIMAL(8, 6), nullable=False)

    severity_level = Column(
        SAEnum(
            SeverityLevel,
            name="severity_level_enum",
            metadata=Base.metadata,
        ),
        nullable=False,
    )

    forecast_id = Column(Integer, ForeignKey("forecasts.forecast_id"), nullable=False)

    forecast = relationship("Forecasts", back_populates="anomaly")
    events = relationship("EventLog", back_populates="anomaly")


class EventLog(Base):
    __tablename__ = "event_log"
    event_id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, nullable=False)
    confirmed_at = Column(DateTime, nullable=True)

    anomaly_id = Column(Integer, ForeignKey("anomalies.anomaly_id"), nullable=False)

    label = Column(
        SAEnum(
            EventLabel,
            name="event_label_enum",
            metadata=Base.metadata,
        ),
        nullable=False,
    )

    operator_id = Column(String(50), nullable=False)

    anomaly = relationship("Anomalies", back_populates="events")

async def get_user_db(session: AsyncSession = Depends(get_db)):
    yield SQLAlchemyUserDatabase(session, User)
