from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import TradingConfig
from .model import LogisticLimitUpModel


def optimize_threshold(
    frame: pd.DataFrame,
    model: LogisticLimitUpModel,
    config: TradingConfig,
    out_file: Path | None = None,
) -> tuple[float, dict]:
    history = frame.dropna(subset=["next_high", "next_close", "next_limit_up_price"]).copy()
    if history.empty:
        return config.min_score_to_buy, {"reason": "no labeled rows"}

    scored = history.copy()
    scored["score"] = model.predict_proba(scored)
    quantiles = scored["score"].quantile([0.70, 0.80, 0.90, 0.95, 0.98]).tolist()
    factor_threshold = _load_factor_threshold(out_file)
    base_candidates = [config.min_score_to_buy, 0.001, 0.003, 0.005, 0.01, *quantiles]
    if factor_threshold is not None:
        base_candidates.append(factor_threshold)
    candidates = sorted(set(base_candidates))
    candidates = [max(0.0, min(float(x), 0.95)) for x in candidates if pd.notna(x)]

    best_threshold = config.min_score_to_buy
    best_summary: dict | None = None
    results = []
    for threshold in candidates:
        cfg = TradingConfig(
            initial_cash=config.initial_cash,
            max_positions_per_day=config.max_positions_per_day,
            max_position_pct=config.max_position_pct,
            min_score_to_buy=threshold,
            buy_slippage_bps=config.buy_slippage_bps,
            sell_slippage_bps=config.sell_slippage_bps,
            commission_bps=config.commission_bps,
            stamp_tax_bps=config.stamp_tax_bps,
            min_commission=config.min_commission,
        )
        trades, summary = run_backtest(history, model, cfg)
        score = summary["total_return"] + summary["max_drawdown"] * 0.8
        result = {"threshold": threshold, "objective": score, **summary}
        results.append(result)
        if best_summary is None or score > best_summary["objective"]:
            best_threshold = threshold
            best_summary = result

    payload = {"best_threshold": best_threshold, "results": results}
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload, indent=2, default=str))
    return best_threshold, best_summary or {}


def initialize_factor_params(
    frame: pd.DataFrame,
    lookback_days: int = 22,
    out_file: Path | None = None,
) -> dict:
    data = frame.dropna(subset=["target_limit_up_next"]).copy()
    if data.empty:
        params = {"reason": "no labeled rows", "lookback_days": lookback_days}
        if out_file:
            out_file.write_text(json.dumps(params, indent=2, ensure_ascii=False))
        return params
    data["date"] = pd.to_datetime(data["date"])
    cutoff = data["date"].max() - pd.Timedelta(days=lookback_days * 2)
    recent = data[data["date"] >= cutoff].copy()
    if recent.empty:
        recent = data.copy()

    factors = [
        "pct_chg",
        "intraday_ret",
        "close_to_high",
        "volume_ratio_5",
        "amount_ratio_5",
        "ret_3",
        "ret_5",
        "limit_gap",
        "market_limit_hits",
        "tail_ret_1430_1457",
        "tail_ret_1450_1457",
        "tail_volume_ratio",
        "tail_amount_ratio",
        "tail_volume_vs_5d",
        "tail_high_break",
        "tail_close_to_high",
        "tail_limit_gap",
        "tail_vwap_deviation",
        "tail_pullback",
        "tail_range",
    ]
    available = [col for col in factors if col in recent.columns]
    positives = recent[recent["target_limit_up_next"] == 1]
    negatives = recent[recent["target_limit_up_next"] == 0]

    factor_stats = []
    for col in available:
        pos_mean = float(positives[col].mean()) if not positives.empty else 0.0
        neg_mean = float(negatives[col].mean()) if not negatives.empty else 0.0
        pos_q25 = float(positives[col].quantile(0.25)) if not positives.empty else 0.0
        pos_q50 = float(positives[col].quantile(0.50)) if not positives.empty else 0.0
        pos_q75 = float(positives[col].quantile(0.75)) if not positives.empty else 0.0
        spread = pos_mean - neg_mean
        factor_stats.append(
            {
                "factor": col,
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "spread": spread,
                "positive_q25": pos_q25,
                "positive_q50": pos_q50,
                "positive_q75": pos_q75,
            }
        )

    rules = _build_initial_rules(factor_stats)
    score_threshold = _suggest_score_threshold(recent)
    params = {
        "lookback_days": lookback_days,
        "sample_start": str(recent["date"].min().date()),
        "sample_end": str(recent["date"].max().date()),
        "rows": int(len(recent)),
        "positive_rows": int(recent["target_limit_up_next"].sum()),
        "positive_rate": float(recent["target_limit_up_next"].mean()),
        "suggested_score_threshold": score_threshold,
        "initial_rules": rules,
        "factor_stats": sorted(factor_stats, key=lambda item: abs(item["spread"]), reverse=True),
    }
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(params, indent=2, ensure_ascii=False, default=str))
    return params


def _build_initial_rules(factor_stats: list[dict]) -> dict:
    by_name = {item["factor"]: item for item in factor_stats}

    def q50(name: str, default: float) -> float:
        return float(by_name.get(name, {}).get("positive_q50", default))

    def q25(name: str, default: float) -> float:
        return float(by_name.get(name, {}).get("positive_q25", default))

    return {
        "min_tail_ret_1430_1457": q25("tail_ret_1430_1457", 0.0),
        "min_tail_ret_1450_1457": q25("tail_ret_1450_1457", 0.0),
        "min_tail_volume_ratio": q25("tail_volume_ratio", 0.0),
        "min_tail_volume_vs_5d": q25("tail_volume_vs_5d", 1.0),
        "max_tail_limit_gap": q50("tail_limit_gap", 0.08),
        "min_volume_ratio_5": q25("volume_ratio_5", 1.0),
        "min_ret_3": q25("ret_3", 0.0),
    }


def _suggest_score_threshold(frame: pd.DataFrame) -> float:
    positives = frame[frame["target_limit_up_next"] == 1]
    if positives.empty:
        return 0.005
    # Start permissive: retain at least 75% of recent actual limit-up samples by raw factor proxy.
    proxy = (
        frame["ret_3"].fillna(0)
        + frame["volume_ratio_5"].fillna(1) * 0.01
        + frame.get("tail_ret_1430_1457", 0).fillna(0)
    )
    pos_proxy = proxy.loc[positives.index]
    percentile = float((proxy >= pos_proxy.quantile(0.25)).mean())
    return max(0.001, min(0.05, 1 - percentile))


def _load_factor_threshold(out_file: Path | None) -> float | None:
    if out_file is None:
        return None
    params_file = out_file.parent.parent / "models" / "factor_params.json"
    if not params_file.exists():
        return None
    try:
        data = json.loads(params_file.read_text())
        value = data.get("suggested_score_threshold")
        return float(value) if value is not None else None
    except (ValueError, OSError, json.JSONDecodeError):
        return None


def build_learning_report(frame: pd.DataFrame, model: LogisticLimitUpModel, current_date: str | None = None) -> dict:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    current = pd.to_datetime(current_date) if current_date else data["date"].max()
    today = data[data["date"] == current].copy()
    labeled = data.dropna(subset=["target_limit_up_next"]).copy()
    if labeled.empty:
        return {"date": str(current.date()), "reason": "no labeled rows"}

    scored = labeled.copy()
    scored["score"] = model.predict_proba(scored)
    latest_labeled_date = scored["date"].max()
    latest = scored[scored["date"] == latest_labeled_date].sort_values("score", ascending=False)
    actual = latest[latest["target_limit_up_next"] == 1][["symbol", "name", "score"]].head(50)

    report = {
        "date": str(current.date()),
        "latest_labeled_signal_date": str(latest_labeled_date.date()),
        "latest_result_date": _next_result_date(data, latest_labeled_date),
        "labeled_rows": int(len(labeled)),
        "positive_rows": int(labeled["target_limit_up_next"].sum()),
        "positive_rate": float(labeled["target_limit_up_next"].mean()),
        "actual_limit_up_count": int(latest["target_limit_up_next"].sum()),
        "top3_hit_rate": _top_hit_rate(latest, 3),
        "top5_hit_rate": _top_hit_rate(latest, 5),
        "top10_hit_rate": _top_hit_rate(latest, 10),
        "top20_hit_rate": _top_hit_rate(latest, 20),
        "actual_limit_ups": actual.to_dict("records"),
    }
    if not today.empty:
        today_scored = today.copy()
        today_scored["score"] = model.predict_proba(today_scored)
        report["today_candidate_count"] = int(len(today_scored))
        report["today_top_score"] = float(today_scored["score"].max())
    return report


def _top_hit_rate(frame: pd.DataFrame, n: int) -> float:
    if frame.empty:
        return 0.0
    return float(frame.head(n)["target_limit_up_next"].mean())


def _next_result_date(data: pd.DataFrame, signal_date: pd.Timestamp) -> str | None:
    later = data[data["date"] > signal_date]["date"]
    if later.empty:
        return None
    return str(later.min().date())
