# services/ml_io.py
from __future__ import annotations
from pathlib import Path
import os
import pickle
import re
import time
from keras.models import load_model, save_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "model")).resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def save_pickle(obj, name: str):
    with open(MODEL_DIR / name, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(name: str):
    p = MODEL_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Model artifact not found: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)

def save_keras(model, name: str):
    save_model(model, MODEL_DIR / name)

def load_keras(name: str):
    p = MODEL_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Keras model not found: {p}")
    return load_model(p, compile=False)

def new_version(prefix: str, max_len: int = 20) -> str:
    """
    Версия, которая гарантированно помещается в VARCHAR(max_len).
    Формат: <PREFIX>-<YYMMDDHHMMSS> (только [A-Za-z0-9-]).
    """
    safe_prefix = re.sub(r"[^A-Za-z0-9]", "", prefix) or "V"
    ts = time.strftime("%y%m%d%H%M%S")  # 12 символов
    ver = f"{safe_prefix}-{ts}"
    if len(ver) > max_len:
        keep = max_len - (1 + len(ts))
        ver = f"{safe_prefix[:max(1, keep)]}-{ts}"
    return ver
