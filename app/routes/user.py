import json
from typing import AsyncIterator

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from .anomalies_routes import anomalies_router
from .dependecies import fastapi_users
from app.auth.auth import auth_backend
from app.helpers.helpers import to_start, to_shutdown, create_admin
from app.schemas.user_schemas import UserCreate, UserRead, UserOut

from app.logging_config import app_logger
from .forecasts_routes import forecasts_router
from .models_routes import models_router
from .processed_data_routes import processed_router
from .raw_data_routes import raw_router
from .sensors_routes import sensor_router


@asynccontextmanager
async def lifespan_func(app: FastAPI):
    await to_start()
    await create_admin()
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

MAX_LOG_BYTES = 2000

def _is_textual(content_type: str) -> bool:
    if not content_type:
        return False
    ct = content_type.lower()
    return (
        ct.startswith("text/")
        or "application/json" in ct
        or "application/xml" in ct
        or "application/xhtml+xml" in ct
        or "application/javascript" in ct
        or "application/x-www-form-urlencoded" in ct
    )

def _safe_decode(body: bytes, content_type: str) -> str:
    """
    Пытаемся красиво декодировать текстовые типы, иначе показываем <N bytes binary data>.
    """
    if not _is_textual(content_type):
        return f"<{len(body)} bytes binary data>"
    if "application/json" in content_type.lower():
        try:
            obj = json.loads(body)
            txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            return txt
        except Exception:
            pass
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("utf-8", errors="replace")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        req_body = await request.body()
        req_ct = request.headers.get("content-type", "")
        req_text = _safe_decode(req_body[:MAX_LOG_BYTES], req_ct) if req_body else ""
    except Exception:
        req_text = "<не удалось прочитать body>"

    try:
        app_logger.info(
            "REQUEST %s %s | content-type=%s | body=%s",
            request.method, str(request.url), request.headers.get("content-type", ""),
            (req_text[:MAX_LOG_BYTES] if isinstance(req_text, str) else str(req_text)) ,
        )
    except Exception:
        pass

    response = await call_next(request)

    try:
        response_body: bytes = b""
        if getattr(response, "body_iterator", None) is not None:
            async for chunk in response.body_iterator:
                response_body += chunk
        else:
            response_body = getattr(response, "body", b"") or b""

        async def new_body_iterator() -> AsyncIterator[bytes]:
            yield response_body
        response.body_iterator = new_body_iterator()

        content_encoding = response.headers.get("content-encoding", "")
        resp_ct = response.headers.get("content-type", "")
        if content_encoding:
            resp_text = f"<{len(response_body)} bytes {content_encoding} encoded data>"
        else:
            resp_text = _safe_decode(response_body[:MAX_LOG_BYTES], resp_ct)

        app_logger.info(
            "RESPONSE %s %s | status=%s | content-type=%s | body=%s",
            request.method, str(request.url),
            response.status_code, resp_ct,
            (resp_text[:MAX_LOG_BYTES] if isinstance(resp_text, str) else str(resp_text)),
        )

        if "content-length" in response.headers:
            response.headers["content-length"] = str(len(response_body))

    except Exception:
        app_logger.exception("log_requests: ошибка при логировании ответа")

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
app.include_router(processed_router)
app.include_router(models_router)
app.include_router(forecasts_router)
app.include_router(anomalies_router)
