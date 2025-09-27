from typing import Optional
from pydantic import EmailStr
from fastapi_users.schemas import BaseUser, BaseUserCreate


class UserRead(BaseUser[int]):
    id: int
    email: EmailStr
    username: str
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False

    class Config:
        orm_mode = True


class UserOut(BaseUser[int]):
    id: int
    email: EmailStr

    class Config:
        orm_mode = True


class UserCreate(BaseUserCreate):
    username: str
    email: EmailStr
    password: Optional[str] = None
    role_id: int
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False
    is_verified: Optional[bool] = False