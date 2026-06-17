from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS


@dataclass
class LogisticLimitUpModel:
    feature_columns: list[str]
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_columns].to_numpy(dtype=float)
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        z = (x - mean) / std
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
) -> tuple[LogisticLimitUpModel, dict]:
    feature_columns = feature_columns or FEATURE_COLUMNS
    frame = frame.dropna(subset=feature_columns + [target]).copy()
    x = frame[feature_columns].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Training target has only one class; provide a wider date range.")

    mean = x.mean(axis=0)
    std = x.std(axis=0)
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
    )
    metrics = classification_metrics(y, model.predict_proba(frame))
    return model, metrics


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
