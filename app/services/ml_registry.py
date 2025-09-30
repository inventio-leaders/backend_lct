from __future__ import annotations
from pathlib import Path
import pickle
from threading import RLock
from typing import Any, Dict

from tensorflow.keras.models import load_model

MODEL_DIR = Path("model")

class _MLRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._cache: Dict[str, Any] = {}

    def _pkl(self, name: str):
        with open(MODEL_DIR / name, "rb") as f:
            return pickle.load(f)

    def feature_info(self) -> Dict[str, Any]:
        with self._lock:
            if "feature_info" not in self._cache:
                self._cache["feature_info"] = self._pkl("feature_info.pkl")
            return self._cache["feature_info"]

    def narx(self):
        with self._lock:
            if "narx_model" not in self._cache:
                self._cache["narx_model"] = load_model(MODEL_DIR / "narx_lstm_model.h5")
            if "scaler_narx" not in self._cache:
                self._cache["scaler_narx"] = self._pkl("scaler_narx.pkl")
            return self._cache["narx_model"], self._cache["scaler_narx"]

    def ae(self):
        with self._lock:
            if "ae_model" not in self._cache:
                self._cache["ae_model"] = load_model(MODEL_DIR / "ae_model.h5")
            if "scaler_ae" not in self._cache:
                self._cache["scaler_ae"] = self._pkl("scaler_ae.pkl")
            return self._cache["ae_model"], self._cache["scaler_ae"]

ML_REGISTRY = _MLRegistry()
