from __future__ import annotations

import numpy as np
import pandas as pd


INTRADAY_FEATURE_COLUMNS = [
    # Keep the historical names for model/CSV compatibility. Both return
    # windows now end at TAIL_CUTOFF, as do all volume and VWAP denominators.
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

TAIL_CUTOFF = "14:55"
TAIL_BAR_MINUTES = 5
TAIL_METADATA_COLUMNS = [
    "tail_available",
    "tail_coverage_ratio",
    "tail_observed_bars",
    "tail_expected_bars",
    "tail_duplicate_bars",
    "tail_unavailable_reason",
]


def _expected_times(bar_minutes: int = TAIL_BAR_MINUTES) -> list[str]:
    if bar_minutes <= 0 or 120 % bar_minutes or 115 % bar_minutes:
        raise ValueError("bar_minutes must divide both trading sessions ending at 14:55")
    morning = pd.date_range("2000-01-01 09:30", "2000-01-01 11:30", freq=f"{bar_minutes}min")[1:]
    afternoon = pd.date_range("2000-01-01 13:00", f"2000-01-01 {TAIL_CUTOFF}", freq=f"{bar_minutes}min")[1:]
    return list(morning.strftime("%H:%M")) + list(afternoon.strftime("%H:%M"))


def read_minute_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_minute_bars(df)


def normalize_minute_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "symbol", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Minute bars missing columns: {sorted(missing)}")
    df = frame.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    duplicate = df.duplicated(["symbol", "datetime"], keep=False)
    if "minute_duplicate" in df.columns:
        duplicate |= df["minute_duplicate"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["minute_duplicate"] = duplicate
    df["date"] = df["datetime"].dt.normalize()
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["datetime", "open", "high", "low", "close", "volume", "amount"])


def build_tail_features(
    minute_bars: pd.DataFrame,
    daily_features: pd.DataFrame,
    bar_minutes: int = TAIL_BAR_MINUTES,
) -> pd.DataFrame:
    expected_times = _expected_times(bar_minutes)
    if minute_bars.empty:
        return _empty_tail(daily_features, len(expected_times))
    minute = normalize_minute_bars(minute_bars)
    daily = daily_features[["date", "symbol", "limit_up_price"]].copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.drop_duplicates(["date", "symbol"])

    rows = []
    for (date, symbol), group in minute.groupby(["date", "symbol"], sort=False):
        # Bars are timestamped at their end. Ignore the lunch break and data
        # after the decision cutoff; neither belongs to the feature window.
        in_session = (
            ((group["time"] > "09:30") & (group["time"] <= "11:30"))
            | ((group["time"] > "13:00") & (group["time"] <= TAIL_CUTOFF))
        )
        group = group[in_session].sort_values("datetime")
        duplicate_bars = int(group["minute_duplicate"].sum())
        group = group.drop_duplicates("datetime", keep="last")
        on_grid = group["time"].isin(expected_times) & (group["datetime"] == group["datetime"].dt.floor("min"))
        observed = int(group.loc[on_grid, "time"].nunique())
        record = {
            "date": date,
            "symbol": symbol,
            "tail_available": 0,
            "tail_coverage_ratio": observed / len(expected_times),
            "tail_observed_bars": observed,
            "tail_expected_bars": len(expected_times),
            "tail_duplicate_bars": duplicate_bars,
            "tail_unavailable_reason": "",
        }
        values = group[["open", "high", "low", "close", "volume", "amount"]]
        valid_prices = (
            (group[["open", "high", "low", "close"]] > 0).all(axis=1)
            & (group["high"] >= group[["open", "close", "low"]].max(axis=1))
            & (group["low"] <= group[["open", "close", "high"]].min(axis=1))
        )
        if duplicate_bars:
            reason = "duplicate_bars"
        elif not on_grid.all():
            reason = "unexpected_timestamps"
        elif observed != len(expected_times):
            reason = "incomplete_session"
        elif not np.isfinite(values.to_numpy(dtype=float)).all() or not valid_prices.all():
            reason = "invalid_bars"
        elif (group[["volume", "amount"]] < 0).any().any() or group["volume"].sum() <= 0 or group["amount"].sum() <= 0:
            reason = "invalid_volume_or_amount"
        else:
            reason = ""
        record["tail_unavailable_reason"] = reason
        if reason:
            rows.append(record)
            continue

        tail = group[group["time"] >= "14:30"]
        tail_1450 = group[group["time"] >= "14:50"]
        day_volume = group["volume"].sum()
        day_amount = group["amount"].sum()
        day_high_before_tail = group[group["time"] < "14:30"]["high"].max()
        tail_open = float(tail.iloc[0]["open"])
        tail_close = float(tail.iloc[-1]["close"])
        tail_high = float(tail["high"].max())
        tail_low = float(tail["low"].min())
        tail_amount = float(tail["amount"].sum())
        tail_volume = float(tail["volume"].sum())
        vwap = float(day_amount / day_volume)
        ret_1450 = float(tail_1450.iloc[-1]["close"] / tail_1450.iloc[0]["open"] - 1)
        record.update(
            {
                "tail_available": 1,
                "tail_last_close": tail_close,
                "tail_ret_1430_1457": tail_close / tail_open - 1,
                "tail_ret_1450_1457": ret_1450,
                "tail_volume_ratio": tail_volume / day_volume if day_volume > 0 else 0.0,
                "tail_amount_ratio": tail_amount / day_amount if day_amount > 0 else 0.0,
                "tail_high_break": float(pd.notna(day_high_before_tail) and tail_high > day_high_before_tail),
                "tail_close_to_high": tail_close / tail_high - 1 if tail_high > 0 else 0.0,
                "tail_vwap_deviation": tail_close / vwap - 1 if vwap and vwap > 0 else 0.0,
                "tail_pullback": tail_close / tail_high - 1 if tail_high > 0 else 0.0,
                "tail_range": (tail_high - tail_low) / tail_open if tail_open > 0 else 0.0,
            }
        )
        rows.append(record)

    features = pd.DataFrame(rows)
    if features.empty:
        return _empty_tail(daily_features, len(expected_times))
    if "tail_last_close" not in features:
        features["tail_last_close"] = np.nan
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features = daily.merge(features, on=["date", "symbol"], how="left")
    features = _fill_metadata(features, len(expected_times))
    features["tail_limit_gap"] = features["limit_up_price"] / features["tail_last_close"] - 1
    invalid_limit = (features["tail_available"] == 1) & (
        ~np.isfinite(features["limit_up_price"]) | (features["limit_up_price"] <= 0)
    )
    features.loc[invalid_limit, "tail_available"] = 0
    features.loc[invalid_limit, "tail_unavailable_reason"] = "invalid_limit_price"
    for col in INTRADAY_FEATURE_COLUMNS:
        if col not in features:
            features[col] = np.nan
    valid = features["tail_available"] == 1
    features.loc[~valid, INTRADAY_FEATURE_COLUMNS] = np.nan

    volume_ma = features.loc[valid].sort_values(["symbol", "date"]).groupby("symbol")["tail_volume_ratio"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    features["tail_volume_vs_5d"] = features["tail_volume_ratio"] / volume_ma
    features.loc[valid, "tail_volume_vs_5d"] = (
        features.loc[valid, "tail_volume_vs_5d"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    )
    return features[["date", "symbol", *INTRADAY_FEATURE_COLUMNS, *TAIL_METADATA_COLUMNS]]


def merge_tail_features(daily_features: pd.DataFrame, minute_bars: pd.DataFrame | None) -> pd.DataFrame:
    out = daily_features.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    tail = _empty_tail(out) if minute_bars is None or minute_bars.empty else build_tail_features(minute_bars, out)
    out = out.drop(columns=[*INTRADAY_FEATURE_COLUMNS, *TAIL_METADATA_COLUMNS], errors="ignore")
    merged = out.merge(tail, on=["date", "symbol"], how="left")
    return _fill_metadata(merged, len(_expected_times()))


def _empty_tail(daily_features: pd.DataFrame, expected_bars: int | None = None) -> pd.DataFrame:
    out = daily_features[["date", "symbol"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for col in INTRADAY_FEATURE_COLUMNS:
        out[col] = np.nan
    return _fill_metadata(out, expected_bars or len(_expected_times()))


def _fill_metadata(out: pd.DataFrame, expected_bars: int) -> pd.DataFrame:
    defaults = {
        "tail_available": 0,
        "tail_coverage_ratio": 0.0,
        "tail_observed_bars": 0,
        "tail_expected_bars": expected_bars,
        "tail_duplicate_bars": 0,
        "tail_unavailable_reason": "no_minute_data",
    }
    for col, default in defaults.items():
        out[col] = out[col].fillna(default) if col in out else default
    return out
