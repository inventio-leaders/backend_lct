from __future__ import annotations

import asyncio
import os, json, smtplib, ssl, threading, queue
from typing import Iterable, Set, List
from email.message import EmailMessage
from uuid import UUID

from sqlalchemy import select, cast, String

from app.database import async_session
from app.models.models import User
from app.services.task_manager import TaskManager, TaskInfo

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@example.com")
SMTP_TO = [a.strip() for a in os.getenv("SMTP_TO", "").split(",") if a.strip()]
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() not in {"0", "false", "no"}
SUBJECT_PREFIX = os.getenv("SMTP_SUBJECT_PREFIX", "[Tasks]")

class EmailNotifier:
    def __init__(self, to_addresses: Iterable[str] | None = None,
                 only_on_fields: Set[str] | None = None,
                 subject_prefix: str = SUBJECT_PREFIX):
        self.default_to = list(to_addresses) if to_addresses else (SMTP_TO or [])
        self.only_on_fields = set(only_on_fields or {"status"})
        self.subject_prefix = subject_prefix

        self._q: "queue.Queue[tuple[List[str], EmailMessage]]" = queue.Queue(maxsize=1000)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._started = False

    def start(self):
        if not self._started:
            self._worker.start()
            self._started = True

    def attach_to_task_manager(self, manager: "TaskManager"):
        self.start()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()

        def listener(tid: str, info: "TaskInfo", changed: set[str]):
            if self.only_on_fields and not (self.only_on_fields & changed):
                return
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    asyncio.create_task, self._handle_event_async(tid, info, changed)
                )
            else:
                to = info.recipients or self.default_to
                if not to:
                    return
                subject = f"{self.subject_prefix} {info.kind} · {tid[:8]} · {info.status}"
                body = self._render_body(tid, info, changed)
                msg = self._build_msg(subject, body)
                try:
                    self._q.put_nowait((to, msg))
                except queue.Full:
                    pass

        manager.subscribe(listener)

    async def _handle_event_async(self, tid: str, info: "TaskInfo", changed: set[str]):
        uid = getattr(info, "owner_user_id", None)
        if uid is not None:
            criterion = None
            try:
                if isinstance(uid, str) and uid.isdigit():
                    uid = int(uid)
            except Exception:
                pass
            try:
                if isinstance(uid, int):
                    criterion = (User.id == uid)
                else:
                    if isinstance(uid, str):
                        try:
                            uid_uuid = UUID(uid)
                            criterion = (User.id == uid_uuid)
                        except ValueError:
                            criterion = (cast(User.id, String) == uid)
                    else:
                        criterion = (cast(User.id, String) == cast(uid, String))
            except Exception:
                return

            async with async_session() as db:
                res = await db.execute(select(User).where(criterion))
                user = res.scalar_one_or_none()
                if user and not getattr(user, "notifications_enabled", True):
                    return  # у пользователя уведомления выключены — выходим

        to = info.recipients or self.default_to
        if not to:
            return

        subject = f"{self.subject_prefix} {info.kind} · {tid[:8]} · {info.status}"
        body = self._render_body(tid, info, changed)
        msg = self._build_msg(subject, body)
        try:
            self._q.put_nowait((to, msg))
        except queue.Full:
            pass

    def _render_body(self, tid, info, changed) -> str:
        def j(x):
            try:
                return json.dumps(x, ensure_ascii=False, indent=2)
            except Exception:
                return str(x)
        lines = [
            f"ID задачи:   {tid}",
            f"Название:      {info.kind}",
            f"Статус:    {info.status}",
        ]
        if info.progress:
            lines.append(f"Статус:  {info.progress}")
        if info.error:
            lines.append(f"Ошибки:     {info.error}")
        if changed:
            lines.append(f"Изменения:   {', '.join(sorted(changed))}")
        if info.result:
            lines.append("\nРезультат:")
            lines.append(j(info.result))
        if info.owner_user_id:
            lines.append(f"\nUser ID:  {info.owner_user_id}")
        return "\n".join(lines)

    def _build_msg(self, subject: str, body: str) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg.set_content(body)
        return msg

    def _worker_loop(self):
        while True:
            to, msg = self._q.get()
            try:
                self._send(to, msg)
            finally:
                self._q.task_done()

    def _send(self, to: List[str], msg: EmailMessage):
        if not SMTP_HOST or not to:
            return
        msg["To"] = ", ".join(to)
        if SMTP_USE_TLS:
            ctx = ssl.create_default_context()
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
                s.starttls(context=ctx)
                if SMTP_USER:
                    s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
                if SMTP_USER:
                    s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
