from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import TradingConfig
from .filters import filter_tradeable
from .model import LogisticLimitUpModel, train_logistic
from .execution import prepare_execution_quotes


_OPTIMIZATION_VERSION = 5
_OOS_SETTINGS = {
    "validation_days": 60,
    "fold_days": 5,
    "train_days": 120,
    "min_train_days": 20,
    "min_train_rows": 200,
    "min_selection_days": 20,
    "min_holdout_days": 10,
    "epochs": 300,
    "freeze_days": 5,
    # Finite daily rolling windows use at most ten prior sessions. A provider's
    # fixed-length history window can change these at its advancing frontier;
    # differences beyond this bounded allowance still invalidate the cache.
    "cache_frontier_warmup_days": 10,
    # Rebuilding rolling statistics can change their last floating-point bits.
    # Quantization applies only to cache fingerprints, never to model inputs.
    "cache_float_decimal_places": 8,
}


def optimize_threshold(
    frame: pd.DataFrame,
    model: LogisticLimitUpModel,
    config: TradingConfig,
    out_file: Path | None = None,
    *,
    minute_bars: pd.DataFrame | None = None,
) -> tuple[float, dict]:
    """Select parameters using purged rolling predictions, then report an untouched holdout.

    The caller's model supplies the feature contract only: its in-sample scores
    must never influence historical parameter selection.
    """
    history = _optimization_history(frame)
    quotes = prepare_execution_quotes(minute_bars)
    # Sign only the outcome quotes belonging to each historical signal row;
    # newly appended sessions must not break the existing freeze period.
    history["execution_quotes"] = [
        json.dumps(quotes.get(row.symbol, row.target_end_date), sort_keys=True)
        for row in history.itertuples(index=False)
    ]
    signature = _optimization_signature(frame, model, config)
    if not quotes.rows:
        return _default_optimization(config, out_file, "missing_execution_history", signature, [])
    execution_dates = {date for (_, date), quote in quotes.rows.items()
                       if quote.get("morning_valid") and quote.get("force_valid")}
    if len(execution_dates) < _OOS_SETTINGS["min_selection_days"] + _OOS_SETTINGS["min_holdout_days"] + 1:
        return _default_optimization(config, out_file, "insufficient_execution_dates", signature, [])
    all_dates = sorted(pd.to_datetime(frame["date"], errors="coerce").dropna().unique())
    cached = _load_oos_cache(out_file, history, all_dates, signature)
    if cached is not None:
        return float(cached["threshold"]), cached

    scored, folds = _generate_oos_predictions(history, list(model.feature_columns))
    dates = sorted(scored["date"].unique()) if not scored.empty else []
    min_holdout = _OOS_SETTINGS["min_holdout_days"]
    if len(dates) < _OOS_SETTINGS["min_selection_days"] + min_holdout + 1:
        return _default_optimization(config, out_file, "insufficient_oos_dates", signature, folds)
    holdout_days = max(min_holdout, len(dates) // 3)
    holdout_start = pd.Timestamp(dates[-holdout_days])
    selection = scored[(scored["date"] < holdout_start) & (scored["target_end_date"] < holdout_start)].copy()
    holdout = scored[scored["date"] >= holdout_start].copy()
    if selection["date"].nunique() < _OOS_SETTINGS["min_selection_days"]:
        return _default_optimization(config, out_file, "insufficient_purged_selection_dates", signature, folds)
    if selection["target_limit_up_next"].value_counts().min() < 10 or selection["target_limit_up_next"].nunique() < 2:
        return _default_optimization(config, out_file, "insufficient_selection_classes", signature, folds)

    candidates = _threshold_candidates(selection, config)
    position_candidates = sorted(set([0.30, 0.34, 0.40, 0.50, config.max_position_pct]))
    results = []
    best_summary = None
    for threshold in candidates:
        for position_pct in position_candidates:
            cfg = replace(config, max_position_pct=position_pct, min_score_to_buy=threshold)
            _, summary = run_backtest(selection, model, cfg, scores=selection["score"], minute_bars=quotes)
            if not summary["execution_complete"]:
                return _default_optimization(config, out_file, "incomplete_selection_execution", signature, folds)
            objective = summary["total_return"] + summary["max_drawdown"] * 0.8
            result = {"threshold": threshold, "max_position_pct": position_pct, "objective": objective, **summary}
            results.append(result)
            if _prefer_candidate(result, best_summary, config):
                best_summary = result

    chosen_config = replace(config, min_score_to_buy=best_summary["threshold"], max_position_pct=best_summary["max_position_pct"])
    _, holdout_summary = run_backtest(holdout, model, chosen_config, scores=holdout["score"], minute_bars=quotes)
    _, default_holdout = run_backtest(holdout, model, config, scores=holdout["score"], minute_bars=quotes)
    # Quote completeness is a data requirement, not selection by holdout return.
    if not holdout_summary["execution_complete"] or not default_holdout["execution_complete"]:
        return _default_optimization(config, out_file, "incomplete_holdout_execution", signature, folds)
    result = {
        **best_summary,
        "status": "optimized_oos",
        "evaluation_method": "purged_rolling_oos",
        "selection": {**_sample_period(selection), **best_summary},
        "holdout": {**_sample_period(holdout), **holdout_summary},
        "default_holdout": default_holdout,
        "holdout_used_for_selection": False,
        "cached": False,
    }
    history_end = history["date"].max()
    payload = {
        "version": _OPTIMIZATION_VERSION,
        "evaluation_method": "purged_rolling_oos",
        "status": "optimized_oos",
        "signature": signature,
        "as_of_date": str(pd.Timestamp(all_dates[-1]).date()),
        "history_end": str(history_end.date()),
        "cache_validation": {
            "history_start": str(history["date"].min().date()),
            "history_end": str(history_end.date()),
            "date_digests": _history_date_digests(history),
            "frontier_warmup_days": _OOS_SETTINGS["cache_frontier_warmup_days"],
            "float_decimal_places": _OOS_SETTINGS["cache_float_decimal_places"],
        },
        "best_threshold": best_summary["threshold"],
        "best_max_position_pct": best_summary["max_position_pct"],
        "result": result,
        "folds": folds,
        "results": results,
    }
    _write_optimization(out_file, payload)
    return float(best_summary["threshold"]), result


def _optimization_history(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])
    inferred_end = work.groupby("symbol")["date"].shift(-1)
    if "target_end_date" in work:
        work["target_end_date"] = pd.to_datetime(work["target_end_date"], errors="coerce").dt.normalize().fillna(inferred_end)
    else:
        work["target_end_date"] = inferred_end
    if "market_data_eligible" in work:
        eligible = work["market_data_eligible"].astype(str).str.lower().isin({"true", "1", "1.0"})
        complete_dates = eligible.groupby(work["date"]).all()
        valid_dates = complete_dates[complete_dates].index
        work = work[work["date"].isin(valid_dates) & work["target_end_date"].isin(valid_dates)]
    work = work.dropna(subset=["target_limit_up_next", "next_high", "next_close", "next_limit_up_price", "target_end_date"])
    as_of = pd.to_datetime(frame["date"], errors="coerce").max()
    work = work[(work["target_end_date"] > work["date"]) & (work["target_end_date"] <= as_of)]
    return work.sort_values(["date", "symbol"]).reset_index(drop=True)


def _generate_oos_predictions(history: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    dates = sorted(history["date"].unique())
    validation_dates = dates[-_OOS_SETTINGS["validation_days"]:]
    parts, folds = [], []
    for start in range(0, len(validation_dates), _OOS_SETTINGS["fold_days"]):
        fold_dates = validation_dates[start:start + _OOS_SETTINGS["fold_days"]]
        fold_start = pd.Timestamp(fold_dates[0])
        train = history[(history["date"] < fold_start) & (history["target_end_date"] < fold_start)]
        train_dates = sorted(train["date"].unique())[-_OOS_SETTINGS["train_days"]:]
        train = train[train["date"].isin(train_dates)]
        info = {"prediction_start": str(fold_start.date()), "prediction_end": str(pd.Timestamp(fold_dates[-1]).date()), "train_rows": len(train)}
        if len(train_dates) < _OOS_SETTINGS["min_train_days"] or len(train) < _OOS_SETTINGS["min_train_rows"] or train["target_limit_up_next"].nunique() < 2:
            folds.append({**info, "status": "insufficient_training_data"})
            continue
        try:
            fold_model, metrics = train_logistic(train, feature_columns=feature_columns, epochs=_OOS_SETTINGS["epochs"])
        except ValueError as error:
            folds.append({**info, "status": "training_unavailable", "reason": str(error)})
            continue
        if metrics.get("rows", len(train)) < _OOS_SETTINGS["min_train_rows"]:
            folds.append({**info, "status": "insufficient_complete_training_rows"})
            continue
        test = history[history["date"].isin(fold_dates)].copy()
        test["score"] = fold_model.predict_proba(test)
        test = test[np.isfinite(test["score"]) & test["score"].between(0.0, 1.0)]
        parts.append(test)
        folds.append({**info, "status": "scored", "train_end": str(train["date"].max().date()), "train_target_end": str(train["target_end_date"].max().date()), "prediction_rows": len(test), "feature_columns": list(fold_model.feature_columns)})
    return (pd.concat(parts).sort_values(["date", "symbol"]) if parts else history.iloc[:0].copy()), folds


def _threshold_candidates(scored: pd.DataFrame, config: TradingConfig) -> list[float]:
    ranked = filter_tradeable(scored).sort_values(["date", "score"], ascending=[True, False])
    boundaries = ranked.groupby("date", sort=False).head(config.max_positions_per_day)["score"]
    # These quantiles describe scores near the actual daily trade boundary,
    # rather than the mostly irrelevant bottom of the market-wide ranking.
    values = boundaries.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).dropna().tolist()
    return sorted(set([float(config.min_score_to_buy), *map(float, values), float(np.nextafter(1.0, np.inf))]))


def _prefer_candidate(candidate: dict, best: dict | None, config: TradingConfig) -> bool:
    if best is None or candidate["objective"] > best["objective"] + 1e-12:
        return True
    if candidate["objective"] < best["objective"] - 1e-12:
        return False
    def tie_key(item: dict) -> tuple:
        default_threshold = np.isclose(item["threshold"], config.min_score_to_buy, rtol=0, atol=1e-12)
        default_position = np.isclose(item["max_position_pct"], config.max_position_pct, rtol=0, atol=1e-12)
        return (not (default_threshold and default_position), item["max_position_pct"], not default_threshold, -item["threshold"])
    return tie_key(candidate) < tie_key(best)


def _sample_period(frame: pd.DataFrame) -> dict:
    return {"start": str(frame["date"].min().date()), "end": str(frame["date"].max().date()), "days": int(frame["date"].nunique()), "rows": len(frame)}


def _optimization_signature(frame: pd.DataFrame, model: LogisticLimitUpModel, config: TradingConfig) -> str:
    contract = {"config": asdict(config), "feature_columns": list(model.feature_columns), "schema": sorted((str(col), str(dtype)) for col, dtype in frame.dtypes.items()), "settings": _OOS_SETTINGS}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def _history_digest(history: pd.DataFrame) -> str:
    normalized = history.sort_values(["date", "symbol"]).reset_index(drop=True).copy()
    floats = normalized.select_dtypes(include=["floating"]).columns
    rounded = normalized[floats].round(_OOS_SETTINGS["cache_float_decimal_places"])
    # Positive and negative zero have different binary hashes but represent the
    # same quantized value; normalize both before computing the fingerprint.
    normalized[floats] = rounded.mask(rounded == 0, 0.0)
    values = pd.util.hash_pandas_object(normalized, index=False)
    return hashlib.sha256(values.to_numpy().tobytes()).hexdigest()


def _history_date_digests(history: pd.DataFrame) -> dict[str, str]:
    return {str(pd.Timestamp(date).date()): _history_digest(rows) for date, rows in history.groupby("date", sort=True)}


def _cache_history_matches(payload: dict, history: pd.DataFrame, new_days: int) -> bool:
    """Check a frozen calibration without treating a rolling prefix as a revision.

    A moved frontier may drop at most one old session for each newly observed
    session. Only its ten-session feature warmup is exempt from comparison;
    selection and holdout always remain fully signed. All rows and all dates in
    the remaining overlap must match, including training observations there.
    """
    validation = payload["cache_validation"]
    expected = validation["date_digests"]
    old_dates = sorted(pd.Timestamp(date) for date in expected)
    if not old_dates or history.empty:
        return False
    old_start, old_end = old_dates[0], old_dates[-1]
    if old_start != pd.Timestamp(validation["history_start"]) or old_end != pd.Timestamp(validation["history_end"]):
        return False
    current_dates = sorted(history["date"].unique())
    current_start = pd.Timestamp(current_dates[0])
    if current_start < old_start:
        # Newly backfilled earlier observations can change the fitted models.
        return False
    compare_start = old_start
    if current_start > old_start:
        dropped_days = sum(date < current_start for date in old_dates)
        if dropped_days > new_days:
            return False
        warmup_days = _OOS_SETTINGS["cache_frontier_warmup_days"]
        if len(current_dates) <= warmup_days:
            return False
        compare_start = pd.Timestamp(current_dates[warmup_days])

    selection = payload["result"]["selection"]
    holdout = payload["result"]["holdout"]
    # Never silently exclude either evaluation period to make a cache match.
    if compare_start > pd.Timestamp(selection["start"]) or old_end < pd.Timestamp(holdout["end"]):
        return False
    expected_overlap = {date: digest for date, digest in expected.items() if pd.Timestamp(date) >= compare_start}
    current_overlap = history[history["date"].between(compare_start, old_end)]
    return _history_date_digests(current_overlap) == expected_overlap


def _load_oos_cache(out_file: Path | None, history: pd.DataFrame, all_dates: list, signature: str) -> dict | None:
    if out_file is None or not out_file.exists() or not all_dates:
        return None
    try:
        payload = json.loads(out_file.read_text())
        if payload.get("version") != _OPTIMIZATION_VERSION or payload.get("evaluation_method") != "purged_rolling_oos" or payload.get("status") != "optimized_oos" or payload.get("signature") != signature:
            return None
        as_of = pd.Timestamp(payload["as_of_date"])
        new_days = sum(pd.Timestamp(date) > as_of for date in all_dates)
        if as_of not in all_dates or new_days >= _OOS_SETTINGS["freeze_days"]:
            return None
        if not _cache_history_matches(payload, history, new_days):
            return None
        result = payload["result"]
        if result.get("status") != "optimized_oos" or not result.get("holdout") or result.get("holdout_used_for_selection") is not False:
            return None
        threshold = float(result["threshold"])
        position = float(result["max_position_pct"])
        if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(position) or not 0 < position <= 1:
            return None
        if result["holdout"].get("days", 0) < _OOS_SETTINGS["min_holdout_days"] or result.get("selection", {}).get("days", 0) < _OOS_SETTINGS["min_selection_days"]:
            return None
        return {**result, "cached": True, "calibrated_as_of_date": str(as_of.date())}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _default_optimization(config: TradingConfig, out_file: Path | None, reason: str, signature: str, folds: list[dict]) -> tuple[float, dict]:
    result = {"status": "insufficient_oos_data", "evaluation_method": "purged_rolling_oos", "reason": reason, "threshold": config.min_score_to_buy, "max_position_pct": config.max_position_pct, "cached": False, "execution_mode": "scheduled_minutes", "execution_complete": False}
    _write_optimization(out_file, {"version": _OPTIMIZATION_VERSION, "signature": signature, "status": result["status"], "evaluation_method": "purged_rolling_oos", "best_threshold": config.min_score_to_buy, "best_max_position_pct": config.max_position_pct, "result": result, "folds": folds, "results": []})
    return config.min_score_to_buy, result


def _write_optimization(out_file: Path | None, payload: dict) -> None:
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload, indent=2, default=str))


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


def build_learning_report(
    frame: pd.DataFrame,
    model: LogisticLimitUpModel,
    current_date: str | None = None,
    bought_symbols: set[str] | None = None,
    candidate_rank: pd.DataFrame | None = None,
) -> dict:
    data = filter_tradeable(frame.copy())
    data["date"] = pd.to_datetime(data["date"])
    current = pd.to_datetime(current_date) if current_date else data["date"].max()
    today = data[data["date"] == current].copy()
    labeled = data.dropna(subset=["target_limit_up_next"]).copy()
    if labeled.empty:
        return {"date": str(current.date()), "reason": "no labeled rows"}

    latest_labeled_date = labeled["date"].max()
    result_date = _next_result_date(data, latest_labeled_date)
    latest = _evaluation_rank(data, model, latest_labeled_date, candidate_rank)
    target_column = "target_limit_up_next"
    latest["candidate_rank"] = range(1, len(latest) + 1)
    actual = latest[latest[target_column] == 1][["symbol", "name", "score"]].head(50)
    bought_symbols = bought_symbols or set()
    top10 = latest.head(10).copy()
    extra_bought = latest[
        latest["symbol"].isin(bought_symbols) & ~latest["symbol"].isin(top10["symbol"])
    ].copy()
    evaluated = pd.concat([top10, extra_bought], ignore_index=True)
    evaluated["hit_limit"] = evaluated[target_column].map(
        lambda value: None if pd.isna(value) else bool(value)
    )
    evaluated["was_bought"] = evaluated["symbol"].isin(bought_symbols)
    evaluated["in_top10"] = evaluated["candidate_rank"] <= 10

    report = {
        "date": str(current.date()),
        "latest_labeled_signal_date": str(latest_labeled_date.date()),
        "latest_result_date": str(pd.to_datetime(result_date).date()) if result_date is not None else None,
        "evaluation_source": "daily",
        "labeled_rows": int(len(labeled)),
        "positive_rows": int(labeled["target_limit_up_next"].sum()),
        "positive_rate": float(labeled["target_limit_up_next"].mean()),
        "actual_limit_up_count": int(latest[target_column].sum()),
        "top3_hit_rate": _top_hit_rate(latest, 3, target_column),
        "top5_hit_rate": _top_hit_rate(latest, 5, target_column),
        "top10_hit_rate": _top_hit_rate(latest, 10, target_column),
        "top20_hit_rate": _top_hit_rate(latest, 20, target_column),
        "top10_evaluated_count": int(latest.head(10)[target_column].notna().sum()),
        "actual_limit_ups": actual.to_dict("records"),
        "evaluated_candidates": evaluated[
            ["candidate_rank", "symbol", "name", "score", "hit_limit", "was_bought", "in_top10"]
        ].to_dict("records"),
    }
    if not today.empty:
        today_scored = today.copy()
        today_scored["score"] = model.predict_proba(today_scored)
        report["today_candidate_count"] = int(len(today_scored))
        report["today_top_score"] = float(today_scored["score"].max())
    return report


def _evaluation_rank(
    data: pd.DataFrame,
    model: LogisticLimitUpModel,
    signal_date: pd.Timestamp,
    candidate_rank: pd.DataFrame | None,
) -> pd.DataFrame:
    day = data[data["date"] == signal_date].copy()
    if candidate_rank is not None and not candidate_rank.empty:
        snapshot = candidate_rank.copy()
        if "date" in snapshot.columns:
            snapshot = snapshot[pd.to_datetime(snapshot["date"]) == signal_date]
        if not snapshot.empty and {"symbol", "score"}.issubset(snapshot.columns):
            metadata = day.drop(columns=["name", "board"], errors="ignore")
            snapshot = snapshot.merge(metadata, on="symbol", how="inner", suffixes=("", "_feature"))
            return snapshot.sort_values("score", ascending=False).reset_index(drop=True)
    day["score"] = model.predict_proba(day)
    return day.sort_values("score", ascending=False).reset_index(drop=True)


def _top_hit_rate(frame: pd.DataFrame, n: int, target_column: str = "target_limit_up_next") -> float:
    if frame.empty:
        return 0.0
    evaluated = frame.head(n)[target_column].dropna()
    return float(evaluated.mean()) if not evaluated.empty else 0.0


def _next_result_date(data: pd.DataFrame, signal_date: pd.Timestamp) -> str | None:
    later = data[data["date"] > signal_date]["date"]
    if later.empty:
        return None
    return str(later.min().date())
