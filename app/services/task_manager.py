from __future__ import annotations
import uuid, threading
from typing import Any, Dict, Optional, Literal, Callable, List

TaskStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILURE"]

class TaskInfo:
    def __init__(
        self,
        kind: str,
        on_change: Optional[Callable[['TaskInfo', set[str]], None]] = None,
        *,
        owner_user_id: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.kind = kind
        self.status: TaskStatus = "PENDING"
        self.progress: str = ""
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None

        # --- новое, для нотификаций ---
        self.owner_user_id = owner_user_id          # id пользователя fastapi-users (строкой/UUID->str)
        self.recipients: List[str] = recipients or []  # конкретные email'ы
        self.meta: Dict[str, Any] = meta or {}         # любые доп. данные

        self._lock = threading.RLock()
        self._on_change = on_change
        self._id: Optional[str] = None

    def set(self, **kwargs):
        changed: set[str] = set()
        with self._lock:
            for k, v in kwargs.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
                    changed.add(k)
                else:
                    old = getattr(self, k)
                    if old != v:
                        setattr(self, k, v)
                        changed.add(k)
        if changed and self._on_change:
            try:
                self._on_change(self, changed)
            except Exception:
                pass

class TaskManager:
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.RLock()
        self._listeners: List[Callable[[str, TaskInfo, set[str]], None]] = []

    def subscribe(self, listener: Callable[[str, TaskInfo, set[str]], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def _notify(self, tid: str, info: TaskInfo, changed: set[str]) -> None:
        for fn in list(self._listeners):
            try:
                fn(tid, info, changed)
            except Exception:
                pass

    def create(
        self,
        kind: str,
        *,
        owner_user_id: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        tid = uuid.uuid4().hex

        def on_change(info: TaskInfo, changed: set[str]):
            self._notify(tid, info, changed)

        with self._lock:
            ti = TaskInfo(
                kind,
                on_change=on_change,
                owner_user_id=owner_user_id,
                recipients=recipients,
                meta=meta,
            )
            ti._id = tid
            self._tasks[tid] = ti
        return tid

    def get(self, tid: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(tid)

    def update(self, tid: str, **kwargs) -> Optional[TaskInfo]:
        info = self.get(tid)
        if not info:
            return None
        info.set(**kwargs)
        return info

TASKS = TaskManager()
