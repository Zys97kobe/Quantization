from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import rank_candidates
from .config import DEFAULT_PREDICTION_ACCURACY
from .model import LogisticLimitUpModel


def write_prediction_accuracy(
    frame: pd.DataFrame,
    model: LogisticLimitUpModel,
    summary: dict,
    learning: dict,
    path: Path = DEFAULT_PREDICTION_ACCURACY,
) -> dict | None:
    if not learning or learning.get("reason"):
        return None
    signal_date = learning.get("latest_labeled_signal_date")
    if not signal_date:
        return None

    day = frame[pd.to_datetime(frame["date"]) == pd.to_datetime(signal_date)].copy()
    if day.empty or "target_limit_up_next" not in day.columns:
        return None
    day = day.dropna(subset=["target_limit_up_next"])
    if day.empty:
        return None

    ranked = rank_candidates(frame, model, signal_date)
    labels = day[["symbol", "target_limit_up_next"]].copy()
    ranked = ranked.merge(labels, on="symbol", how="left")
    ranked = ranked.dropna(subset=["target_limit_up_next"])
    if ranked.empty:
        return None

    record = _build_accuracy_record(ranked, summary, learning, signal_date)
    _upsert_accuracy(path, record)
    return record


def _build_accuracy_record(
    ranked: pd.DataFrame,
    summary: dict,
    learning: dict,
    signal_date: str,
) -> dict:
    top1 = ranked.iloc[0]
    return {
        "date": summary.get("date"),
        "phase": summary.get("phase"),
        "signal_date": signal_date,
        "result_date": learning.get("latest_result_date"),
        "candidate_count": int(len(ranked)),
        "actual_limit_up_count": int(ranked["target_limit_up_next"].sum()),
        "market_actual_limit_up_count": learning.get("actual_limit_up_count"),
        "positive_rate": float(ranked["target_limit_up_next"].mean()),
        "top1_symbol": top1["symbol"],
        "top1_name": top1["name"],
        "top1_score": float(top1["score"]),
        "top1_hit": int(top1["target_limit_up_next"] == 1),
        "top3_hit_rate": _top_hit_rate(ranked, 3),
        "top5_hit_rate": _top_hit_rate(ranked, 5),
        "top10_hit_rate": _top_hit_rate(ranked, 10),
        "top20_hit_rate": _top_hit_rate(ranked, 20),
        "top3_symbols": _top_symbols(ranked, 3),
        "top5_symbols": _top_symbols(ranked, 5),
    }


def _top_hit_rate(ranked: pd.DataFrame, n: int) -> float:
    return float(ranked.head(n)["target_limit_up_next"].mean()) if not ranked.empty else 0.0


def _top_symbols(ranked: pd.DataFrame, n: int) -> str:
    return ",".join(ranked.head(n)["symbol"].astype(str).tolist())


def _upsert_accuracy(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([record])
    if not path.exists():
        new_row.to_csv(path, index=False)
        return

    existing = pd.read_csv(path)
    if existing.empty:
        new_row.to_csv(path, index=False)
        return

    key_cols = ["date", "phase", "signal_date"]
    missing = [col for col in key_cols if col not in existing.columns]
    if missing:
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        mask = pd.Series(True, index=existing.index)
        for col in key_cols:
            mask &= existing[col].astype(str) == str(record.get(col))
        combined = pd.concat([existing.loc[~mask], new_row], ignore_index=True, sort=False)
    combined.to_csv(path, index=False)
