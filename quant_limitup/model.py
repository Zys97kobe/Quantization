from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .intraday_features import INTRADAY_FEATURE_COLUMNS, TAIL_CUTOFF


MIN_TAIL_TRAIN_DATES = 20
MIN_TAIL_TRAIN_ROWS = 1000


@dataclass
class LogisticLimitUpModel:
    feature_columns: list[str]
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = _feature_matrix(frame, self.feature_columns)
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        z = (x - mean) / std
        tail_indices = [i for i, col in enumerate(self.feature_columns) if col in INTRADAY_FEATURE_COLUMNS]
        if tail_indices:
            tail_columns = [self.feature_columns[i] for i in tail_indices]
            available = _tail_available_mask(frame, tail_columns).to_numpy()
            # Old JSON files remain loadable, but cannot certify the coverage
            # on which their minute coefficients were fitted.
            if not self.metadata.get("tail_features", {}).get("enabled", False):
                available[:] = False
            z[np.ix_(~available, tail_indices)] = 0.0
        logits = z @ np.asarray(self.weights) + self.bias
        return 1 / (1 + np.exp(-np.clip(logits, -35, 35)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "feature_columns": self.feature_columns,
                    "weights": self.weights,
                    "bias": self.bias,
                    "mean": self.mean,
                    "std": self.std,
                    "metadata": self.metadata,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "LogisticLimitUpModel":
        data = json.loads(path.read_text())
        return cls(**data)


def train_logistic(
    frame: pd.DataFrame,
    target: str = "target_limit_up_next",
    feature_columns: list[str] | None = None,
    epochs: int = 1200,
    lr: float = 0.08,
    l2: float = 0.02,
    tail_min_dates: int = MIN_TAIL_TRAIN_DATES,
    tail_min_rows: int = MIN_TAIL_TRAIN_ROWS,
) -> tuple[LogisticLimitUpModel, dict]:
    if tail_min_dates < 1 or tail_min_rows < 1:
        raise ValueError("Tail training coverage thresholds must be positive")
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    tail_columns = [col for col in feature_columns if col in INTRADAY_FEATURE_COLUMNS]
    daily_columns = [col for col in feature_columns if col not in tail_columns]
    frame = frame.copy()
    for col in daily_columns:
        if col not in frame.columns:
            frame[col] = 0.0
    frame = frame.dropna(subset=daily_columns + [target]).copy()
    tail_available = _tail_available_mask(frame, tail_columns)
    dates = pd.to_datetime(frame["date"], errors="coerce") if "date" in frame else pd.Series(pd.NaT, index=frame.index)
    tail_available &= dates.notna()
    available_rows = int(tail_available.sum())
    available_dates = int(dates[tail_available].dt.normalize().nunique())
    tail_enabled = bool(tail_columns and available_rows >= tail_min_rows and available_dates >= tail_min_dates)
    reason = (
        "enabled" if tail_enabled else
        "not_requested" if not tail_columns else
        "missing_coverage_metadata" if "tail_available" not in frame else
        "insufficient_complete_history"
    )
    tail_metadata = {
        "enabled": tail_enabled,
        "reason": reason,
        "available_rows": available_rows,
        "available_dates": available_dates,
        "min_rows": tail_min_rows,
        "min_dates": tail_min_dates,
        "cutoff": TAIL_CUTOFF,
        "missing_policy": "zero_standardized_contribution",
    }
    if not tail_enabled:
        feature_columns = daily_columns
    if not feature_columns:
        raise ValueError("No features remain after applying minute coverage requirements")
    x = _feature_matrix(frame, feature_columns)
    y = frame[target].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Training target has only one class; provide a wider date range.")

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    if tail_enabled:
        available = tail_available.to_numpy()
        tail_indices = [feature_columns.index(col) for col in tail_columns]
        tail_values = x[np.ix_(available, tail_indices)]
        # Estimate minute scaling only on complete sessions. Missing sessions
        # contribute no minute signal, while their daily features still train.
        mean[tail_indices] = tail_values.mean(axis=0)
        std[tail_indices] = tail_values.std(axis=0)
        x[np.ix_(~available, tail_indices)] = mean[tail_indices]
    std[std == 0] = 1.0
    z = (x - mean) / std
    weights = np.zeros(z.shape[1], dtype=float)
    bias = float(np.log(y.mean() / (1 - y.mean())))

    for _ in range(epochs):
        logits = z @ weights + bias
        pred = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
        error = pred - y
        weights -= lr * ((z.T @ error) / len(y) + l2 * weights)
        bias -= lr * float(error.mean())

    model = LogisticLimitUpModel(
        feature_columns=feature_columns,
        weights=weights.tolist(),
        bias=float(bias),
        mean=mean.tolist(),
        std=std.tolist(),
        metadata={"tail_features": tail_metadata},
    )
    metrics = classification_metrics(y, model.predict_proba(frame))
    metrics["tail_features"] = tail_metadata.copy()
    return model, metrics


def _tail_available_mask(frame: pd.DataFrame, tail_columns: list[str]) -> pd.Series:
    if not tail_columns or "tail_available" not in frame or any(col not in frame for col in tail_columns):
        return pd.Series(False, index=frame.index)
    available = pd.to_numeric(frame["tail_available"], errors="coerce").eq(1)
    finite = np.isfinite(frame[tail_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).all(axis=1)
    return available & finite


def _feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    work = frame.copy()
    for col in feature_columns:
        if col not in work.columns:
            work[col] = 0.0
    return work[feature_columns].to_numpy(dtype=float)


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    pred = (score >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(len(y), 1)
    top_decile = pd.DataFrame({"y": y, "score": score}).sort_values("score", ascending=False)
    top_n = max(1, len(top_decile) // 10)
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy_at_0_5": float(accuracy),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "top_decile_hit_rate": float(top_decile.head(top_n)["y"].mean()),
    }
