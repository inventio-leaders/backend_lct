from __future__ import annotations

import os
from pathlib import Path
import pickle, json, time
from typing import Any, Dict, Tuple

from keras.models import load_model, save_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "model")).resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def save_pickle(obj: Any, name: str):
    with open(MODEL_DIR / name, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(name: str) -> Any:
    with open(MODEL_DIR / name, "rb") as f:
        return pickle.load(f)

def save_keras(model, name: str):
    save_model(model, MODEL_DIR / name)

def load_keras(name: str):
    p = MODEL_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Keras model not found: {p}")
    return load_model(p, compile=False)

def new_version(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y-%m-%dT%H-%M-%S')}"
