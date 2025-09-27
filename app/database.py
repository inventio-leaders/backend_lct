import os
from typing import Any, Generator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

load_dotenv()

DATABASE_URL: Any = os.getenv("DATABASE")
engine = create_async_engine(DATABASE_URL, echo=True, future=True, pool_pre_ping=True)
async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)
Base = declarative_base()


async def get_db() -> Generator:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()