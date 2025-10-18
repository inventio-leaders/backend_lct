from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.models import User
from app.routes.dependecies import current_user

notification_router = APIRouter(prefix="/notifications", tags=["notifications"])

@notification_router.get("/status")
async def get_notifications_status(
    user: User = Depends(current_user),
):
    """Вернёт текущее состояние уведомлений (on/off)."""
    return {"notifications_enabled": user.notifications_enabled}

@notification_router.post("/toggle")
async def toggle_notifications(
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Переключает уведомления."""
    user.notifications_enabled = not user.notifications_enabled
    await db.merge(user)
    await db.commit()
    return {
        "notifications_enabled": user.notifications_enabled,
        "message": "Уведомления включены" if user.notifications_enabled else "Уведомления выключены"
    }
