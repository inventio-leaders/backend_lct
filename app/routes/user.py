
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from .dependecies import fastapi_users
from app.auth.auth import auth_backend
from app.helpers.helpers import to_start, to_shutdown, create_admin
from app.schemas.user_schemas import UserCreate, UserRead, UserOut

from app.logging_config import app_logger
from .raw_data_routes import raw_router
from .sensors_routes import sensor_router


@asynccontextmanager
async def lifespan_func(app: FastAPI):
    await to_start()
    # await create_admin()
    app_logger.info("База готова")
    yield
    await to_shutdown()
    app_logger.info("База очищена")

app = FastAPI(lifespan=lifespan_func)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        body = await request.body()
        body_str = body.decode("utf-8") if body else ""
    except Exception:
        body_str = "<не удалось прочитать body>"

    app_logger.info(f"REQUEST {request.method} {request.url} | body={body_str[:500]}")

    response = await call_next(request)

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    async def new_body_iterator():
        yield response_body

    response.body_iterator = new_body_iterator()

    app_logger.info(
        f"RESPONSE {request.method} {request.url} | "
        f"status={response.status_code} | body={response_body.decode('utf-8')[:500]}"
    )

    return response


@app.get("/ping")
async def ping():
    app_logger.info("Ping endpoint вызван")
    return {"status": "ok", "message": "Приложение поднялось!"}


app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_users_router(UserOut, UserCreate),
    tags=["me"],
)

app.include_router(sensor_router)
app.include_router(raw_router)
