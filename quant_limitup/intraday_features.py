from __future__ import annotations

import numpy as np
import pandas as pd


INTRADAY_FEATURE_COLUMNS = [
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


def read_minute_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_minute_bars(df)


def normalize_minute_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "symbol", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Minute bars missing columns: {sorted(missing)}")
    df = frame.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.normalize()
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "volume", "amount"])


def build_tail_features(minute_bars: pd.DataFrame, daily_features: pd.DataFrame) -> pd.DataFrame:
    if minute_bars.empty:
        return _empty_tail(daily_features)
    minute = normalize_minute_bars(minute_bars) if "time" not in minute_bars.columns else minute_bars.copy()
    daily = daily_features[["date", "symbol", "limit_up_price"]].copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    rows = []
    for (date, symbol), group in minute.groupby(["date", "symbol"], sort=False):
        group = group.sort_values("datetime")
        tail = group[(group["time"] >= "14:30") & (group["time"] <= "14:57")]
        tail_1450 = group[(group["time"] >= "14:50") & (group["time"] <= "14:57")]
        if tail.empty:
            continue
        day_volume = group["volume"].sum()
        day_amount = group["amount"].sum()
        day_high_before_tail = group[group["time"] < "14:30"]["high"].max()
        tail_open = float(tail.iloc[0]["open"])
        tail_close = float(tail.iloc[-1]["close"])
        tail_high = float(tail["high"].max())
        tail_low = float(tail["low"].min())
        tail_amount = float(tail["amount"].sum())
        tail_volume = float(tail["volume"].sum())
        vwap = float(day_amount / day_volume) if day_volume > 0 else np.nan
        ret_1450 = 0.0
        if not tail_1450.empty:
            ret_1450 = float(tail_1450.iloc[-1]["close"] / tail_1450.iloc[0]["open"] - 1)
        rows.append(
            {
                "date": date,
                "symbol": symbol,
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

    features = pd.DataFrame(rows)
    if features.empty:
        return _empty_tail(daily_features)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features = features.merge(daily, on=["date", "symbol"], how="left")
    features["tail_limit_gap"] = features["limit_up_price"] / features["tail_last_close"] - 1

    volume_ma = features.sort_values(["symbol", "date"]).groupby("symbol")["tail_volume_ratio"].transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    features["tail_volume_vs_5d"] = features["tail_volume_ratio"] / volume_ma
    features["tail_volume_vs_5d"] = features["tail_volume_vs_5d"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return features[["date", "symbol", *INTRADAY_FEATURE_COLUMNS]]


def merge_tail_features(daily_features: pd.DataFrame, minute_bars: pd.DataFrame | None) -> pd.DataFrame:
    out = daily_features.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if minute_bars is None or minute_bars.empty:
        for col in INTRADAY_FEATURE_COLUMNS:
            out[col] = 0.0
        return out
    tail = build_tail_features(minute_bars, out)
    merged = out.merge(tail, on=["date", "symbol"], how="left")
    for col in INTRADAY_FEATURE_COLUMNS:
        merged[col] = merged[col].fillna(0.0)
    return merged


def _empty_tail(daily_features: pd.DataFrame) -> pd.DataFrame:
    out = daily_features[["date", "symbol"]].copy()
    for col in INTRADAY_FEATURE_COLUMNS:
        out[col] = 0.0
    return out
