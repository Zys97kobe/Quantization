from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .data import limit_up_price
from .intraday_features import merge_tail_features


def build_feature_frame(prices: pd.DataFrame, minute_bars: pd.DataFrame | None = None) -> pd.DataFrame:
    df = prices.sort_values(["symbol", "date"]).copy()
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df = df.dropna(subset=["prev_close"]).copy()
    df["limit_up_price"] = [
        limit_up_price(prev, board, is_st)
        for prev, board, is_st in zip(df["prev_close"], df["board"], df["is_st"])
    ]
    df["limit_hit"] = (df["high"] >= df["limit_up_price"] * 0.999).astype(int)

    grp = df.groupby("symbol", group_keys=False)
    df["pct_chg"] = df["close"] / df["prev_close"] - 1
    df["intraday_ret"] = df["close"] / df["open"] - 1
    df["close_to_high"] = df["close"] / df["high"] - 1
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["prev_close"]
    df["volume_ma5"] = grp["volume"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    df["amount_ma5"] = grp["amount"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    df["volume_ratio_5"] = df["volume"] / df["volume_ma5"]
    df["amount_ratio_5"] = df["amount"] / df["amount_ma5"]
    df["ret_3"] = grp["close"].transform(lambda s: s / s.shift(3) - 1)
    df["ret_5"] = grp["close"].transform(lambda s: s / s.shift(5) - 1)
    df["ret_10"] = grp["close"].transform(lambda s: s / s.shift(10) - 1)
    df["volatility_5"] = grp["pct_chg"].transform(lambda s: s.rolling(5, min_periods=3).std())
    df["limit_gap"] = df["limit_up_price"] / df["close"] - 1
    df["mkt_cap_log"] = np.log1p(df["free_float_mkt_cap"])

    market = (
        df.groupby("date")
        .agg(market_ret=("pct_chg", "mean"), market_limit_hits=("limit_hit", "sum"))
        .reset_index()
    )
    df = df.merge(market, on="date", how="left")

    for board in ["main", "star", "chinext", "bse"]:
        df[f"board_{board}"] = (df["board"] == board).astype(int)

    df["next_open"] = grp["open"].shift(-1)
    df["next_high"] = grp["high"].shift(-1)
    df["next_close"] = grp["close"].shift(-1)
    df["next_limit_up_price"] = grp["limit_up_price"].shift(-1)
    has_next = df["next_limit_up_price"].notna()
    df["target_limit_up_next"] = pd.NA
    df.loc[has_next, "target_limit_up_next"] = (
        df.loc[has_next, "next_high"] >= df.loc[has_next, "next_limit_up_price"] * 0.999
    ).astype(int)
    df["target_close_limit_up_next"] = pd.NA
    df.loc[has_next, "target_close_limit_up_next"] = (
        df.loc[has_next, "next_close"] >= df.loc[has_next, "next_limit_up_price"] * 0.999
    ).astype(int)

    existing_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    needed = existing_features + [
        "date",
        "symbol",
        "name",
        "board",
        "open",
        "high",
        "low",
        "close",
        "limit_up_price",
        "next_open",
        "next_high",
        "next_close",
        "next_limit_up_price",
        "target_limit_up_next",
        "target_close_limit_up_next",
    ]
    out = df[needed].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=existing_features + ["date", "symbol", "close", "limit_up_price"]).copy()
    out = merge_tail_features(out, minute_bars)
    out = out.replace([np.inf, -np.inf], np.nan)
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].fillna(0.0)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)
