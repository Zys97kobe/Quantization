from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .config import FEATURE_COLUMNS, TradingConfig
from .filters import filter_tradeable
from .model import train_logistic


class ProbabilityModel(Protocol):
    def predict_proba(self, frame: pd.DataFrame): ...


@dataclass
class SklearnProbabilityModel:
    estimator: object
    feature_columns: list[str]

    def predict_proba(self, frame: pd.DataFrame):
        work = _ensure_feature_columns(frame, self.feature_columns)
        values = work[self.feature_columns].to_numpy(dtype=float)
        proba = self.estimator.predict_proba(values)
        return proba[:, 1]


def compare_models(
    frame: pd.DataFrame,
    out_file: Path,
    lookback_dates: int = 60,
    train_days: int = 180,
    min_train_rows: int = 20_000,
    max_train_rows: int = 250_000,
) -> dict:
    data = frame.dropna(subset=["target_limit_up_next", "next_high", "next_close", "next_limit_up_price"]).copy()
    data["date"] = pd.to_datetime(data["date"])
    dates = sorted(data["date"].dropna().unique())
    eval_dates = dates[-lookback_dates:]
    models = ["logistic", "gbdt"]
    rows: dict[str, list[dict]] = {name: [] for name in models}
    skipped: dict[str, str] = {}

    for eval_date in eval_dates:
        train_start = pd.to_datetime(eval_date) - pd.Timedelta(days=train_days * 2)
        train = data[(data["date"] < eval_date) & (data["date"] >= train_start)].copy()
        test = data[data["date"] == eval_date].copy()
        if len(train) < min_train_rows or test.empty or train["target_limit_up_next"].nunique() < 2:
            continue
        train = _cap_train_rows(train, max_train_rows)

        logistic, _ = train_logistic(train, epochs=350)
        rows["logistic"].append(_evaluate_one_day("logistic", logistic, test, TradingConfig()))

        if "gbdt" not in skipped:
            try:
                gbdt = _train_gbdt(train)
            except Exception as exc:  # noqa: BLE001 - optional dependency / model failure.
                skipped["gbdt"] = str(exc)
            else:
                rows["gbdt"].append(_evaluate_one_day("gbdt", gbdt, test, TradingConfig()))

    summary = {
        "lookback_dates": lookback_dates,
        "train_days": train_days,
        "models": {
            name: _summarize_model(items, skipped.get(name))
            for name, items in rows.items()
        },
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


def _train_gbdt(train: pd.DataFrame) -> ProbabilityModel:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scikit-learn is not installed; install sklearn to enable GBDT comparison") from exc

    train = _ensure_feature_columns(train, FEATURE_COLUMNS)
    x = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train["target_limit_up_next"].to_numpy(dtype=int)
    model = HistGradientBoostingClassifier(
        max_iter=140,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=7,
    )
    model.fit(x, y)
    return SklearnProbabilityModel(model, FEATURE_COLUMNS)


def _cap_train_rows(train: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(train) <= max_rows:
        return train
    positives = train[train["target_limit_up_next"] == 1]
    negatives = train[train["target_limit_up_next"] == 0]
    neg_n = max(max_rows - len(positives), 0)
    negatives = negatives.sample(n=min(len(negatives), neg_n), random_state=7)
    return pd.concat([positives, negatives], ignore_index=True, sort=False).sort_values(["date", "symbol"])


def _evaluate_one_day(name: str, model: ProbabilityModel, test: pd.DataFrame, config: TradingConfig) -> dict:
    day = filter_tradeable(test.copy())
    if day.empty:
        return {"model": name, "date": str(pd.to_datetime(test["date"].iloc[0]).date()), "candidate_count": 0}
    day["score"] = model.predict_proba(day)
    ranked = day.sort_values("score", ascending=False).reset_index(drop=True)
    picks = _buyable_picks(ranked, config)
    return {
        "model": name,
        "date": str(pd.to_datetime(ranked["date"].iloc[0]).date()),
        "candidate_count": int(len(ranked)),
        "positive_rate": float(ranked["target_limit_up_next"].mean()),
        "actual_limit_up_count": int(ranked["target_limit_up_next"].sum()),
        "top1_hit": _top_hit(ranked, 1),
        "top3_hit_rate": _top_hit(ranked, 3),
        "top5_hit_rate": _top_hit(ranked, 5),
        "top10_hit_rate": _top_hit(ranked, 10),
        "buy_count": int(len(picks)),
        "buy_hit_rate": _top_hit(picks, len(picks)) if not picks.empty else 0.0,
        "buy_avg_next_return": _avg_next_return(picks),
    }


def _ensure_feature_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = frame.copy()
    for col in columns:
        if col not in work.columns:
            work[col] = 0.0
    return work


def _buyable_picks(ranked: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
    if ranked.empty:
        return ranked
    market_limit_hits = float(pd.to_numeric(ranked.get("market_limit_hits", pd.Series([999])), errors="coerce").iloc[0])
    max_positions = config.max_positions_per_day
    max_position_pct = config.max_position_pct
    min_score = config.min_score_to_buy
    if market_limit_hits < config.weak_market_limit_hits:
        max_positions = min(max_positions, config.weak_market_max_positions)
        max_position_pct = min(max_position_pct, config.weak_market_max_position_pct)
        min_score = max(min_score, config.weak_market_min_score)
    cash = config.initial_cash
    picks = []
    for row in ranked[ranked["score"] >= min_score].itertuples(index=False):
        if len(picks) >= max_positions:
            break
        buy_price = float(row.close) * (1 + config.buy_slippage_bps / 10_000)
        shares = int(min(cash, config.initial_cash * max_position_pct) / buy_price / 100) * 100
        if shares <= 0:
            continue
        cost = shares * buy_price
        if cost > cash:
            continue
        cash -= cost
        picks.append(row._asdict())
    return pd.DataFrame(picks)


def _top_hit(frame: pd.DataFrame, n: int) -> float:
    if frame.empty or n <= 0:
        return 0.0
    return float(frame.head(n)["target_limit_up_next"].mean())


def _avg_next_return(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    hit = frame["next_high"] >= frame["next_limit_up_price"] * 0.999
    raw_sell = frame["next_limit_up_price"].where(hit, frame["next_close"])
    buy = frame["close"]
    return float((raw_sell / buy - 1).mean())


def _summarize_model(rows: list[dict], skipped_reason: str | None = None) -> dict:
    if skipped_reason:
        return {"status": "skipped", "reason": skipped_reason}
    if not rows:
        return {"status": "no_evaluation_rows"}
    frame = pd.DataFrame(rows)
    metrics = [
        "candidate_count",
        "positive_rate",
        "actual_limit_up_count",
        "top1_hit",
        "top3_hit_rate",
        "top5_hit_rate",
        "top10_hit_rate",
        "buy_count",
        "buy_hit_rate",
        "buy_avg_next_return",
    ]
    return {
        "status": "ok",
        "evaluated_days": int(len(frame)),
        "averages": {col: float(pd.to_numeric(frame[col], errors="coerce").mean()) for col in metrics},
        "daily": rows,
    }
