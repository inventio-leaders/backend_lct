from __future__ import annotations
import uuid, threading, time
from typing import Any, Dict, Optional, Literal

TaskStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILURE"]

class TaskInfo:
    def __init__(self, kind: str):
        self.kind = kind
        self.status: TaskStatus = "PENDING"
        self.progress: str = ""
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self._lock = threading.RLock()

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

class TaskManager:
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.RLock()

    def create(self, kind: str) -> str:
        tid = uuid.uuid4().hex
        with self._lock:
            self._tasks[tid] = TaskInfo(kind)
        return tid

    def get(self, tid: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(tid)

TASKS = TaskManager()
