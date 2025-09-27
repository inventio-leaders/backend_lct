import uuid

from fastapi_users import FastAPIUsers

from app.auth.auth import auth_backend
from app.auth.manager import get_user_manager
from app.models.models import User

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)


async def get_enabled_backends():
    return [auth_backend]


current_user = fastapi_users.current_user(get_enabled_backends=get_enabled_backends)