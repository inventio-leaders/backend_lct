# app/services/ml_infer.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from .ml_io import load_pickle, load_keras

def _last_window(mat: np.ndarray, W: int) -> np.ndarray:
    if mat.shape[0] < W:
        raise ValueError(f"Need at least {W} rows, got {mat.shape[0]}")
    return mat[-W:, :][None, :, :]

def narx_forecast(rows: List[Dict[str, float]]) -> Tuple[float, float]:
    fi = load_pickle("feature_info.pkl")
    feats = fi["narx_features"]; W = fi["window_size"]
    model = load_keras("narx_lstm_model.h5")
    scaler = load_pickle("scaler_narx.pkl")

    X = np.array([[float(r[f]) for f in feats] for r in rows], dtype=float)
    Xs = scaler.transform(X)
    x = _last_window(Xs, W)

    y_hat_s = model.predict(x, verbose=0)
    pad = np.concatenate([y_hat_s, np.zeros((1, len(feats)-1))], axis=1)
    y_hat = scaler.inverse_transform(pad)[0, 0]

    pad_win = np.concatenate([x[:, :, 0:1], np.zeros((1, W, len(feats)-1))], axis=2).reshape(W, len(feats))
    y_win = scaler.inverse_transform(pad_win)[:, 0]
    conf = float(np.clip(1.0 / (1.0 + np.std(y_win) + 1e-9), 0, 1))
    return float(round(y_hat, 3)), conf

def ae_score(rows: List[Dict[str, float]], threshold: float | None) -> Tuple[float, float, bool]:
    fi = load_pickle("feature_info.pkl")
    feats = fi["ae_features"]; W = fi["window_size"]
    model = load_keras("ae_model.h5")
    scaler = load_pickle("scaler_ae.pkl")

    X = np.array([[float(r[f]) for f in feats] for r in rows], dtype=float)
    Xs = scaler.transform(X)
    x = _last_window(Xs, W)

    x_rec = model.predict(x, verbose=0)
    mse = float(np.mean(np.square(x - x_rec)))
    thr = float(threshold if threshold is not None else 0.02)
    return float(mse), thr, bool(mse > thr)
