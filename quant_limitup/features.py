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
    df["close_limit_hit"] = (df["close"] >= df["limit_up_price"] * 0.999).astype(int)
    df["failed_limit_hit"] = ((df["limit_hit"] == 1) & (df["close_limit_hit"] == 0)).astype(int)

    grp = df.groupby("symbol", group_keys=False)
    df["pct_chg"] = df["close"] / df["prev_close"] - 1
    df["intraday_ret"] = df["close"] / df["open"] - 1
    df["close_to_high"] = df["close"] / df["high"] - 1
    df["range"] = (df["high"] - df["low"]) / df["prev_close"]
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

    df["limit_hit_prev_1"] = grp["limit_hit"].shift(1)
    df["close_limit_hit_prev_1"] = grp["close_limit_hit"].shift(1)
    df["limit_hits_5"] = grp["limit_hit"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    df["limit_hits_10"] = grp["limit_hit"].transform(lambda s: s.rolling(10, min_periods=1).sum())
    df["failed_limit_hits_5"] = grp["failed_limit_hit"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    df["consecutive_limit_hits"] = grp["limit_hit"].transform(_consecutive_hits)
    _add_curve_features(df, grp)

    market = (
        df.groupby("date")
        .agg(
            market_ret=("pct_chg", "mean"),
            market_limit_hits=("limit_hit", "sum"),
            market_failed_limit_hits=("failed_limit_hit", "sum"),
            market_symbol_count=("symbol", "nunique"),
        )
        .reset_index()
    )
    market["market_limit_hit_rate"] = market["market_limit_hits"] / market["market_symbol_count"].clip(lower=1)
    df = df.merge(market.drop(columns=["market_symbol_count"]), on="date", how="left")
    board_heat = (
        df.groupby(["date", "board"])
        .agg(board_limit_hits=("limit_hit", "sum"), board_symbol_count=("symbol", "nunique"))
        .reset_index()
    )
    board_heat["board_limit_hit_rate"] = board_heat["board_limit_hits"] / board_heat["board_symbol_count"].clip(lower=1)
    df = df.merge(
        board_heat.drop(columns=["board_symbol_count"]),
        on=["date", "board"],
        how="left",
    )

    for board in ["main", "star", "chinext", "bse"]:
        df[f"board_{board}"] = (df["board"] == board).astype(int)

    # merge() creates a new index, so rebuild the symbol groups before shifting
    # next-day outcomes. Reusing the pre-merge groupby can align another symbol's
    # values onto the current row.
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    grp = df.groupby("symbol", group_keys=False)
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
        "prev_close",
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


def _consecutive_hits(series: pd.Series) -> pd.Series:
    streak = []
    current = 0
    for value in series.fillna(0).astype(int):
        current = current + 1 if value == 1 else 0
        streak.append(current)
    return pd.Series(streak, index=series.index, dtype=float)


def _add_curve_features(df: pd.DataFrame, grp: pd.core.groupby.DataFrameGroupBy) -> None:
    for lag in range(1, 6):
        df[f"pct_chg_lag_{lag}"] = grp["pct_chg"].shift(lag)
        df[f"intraday_ret_lag_{lag}"] = grp["intraday_ret"].shift(lag)
        df[f"range_lag_{lag}"] = grp["range"].shift(lag)
        df[f"close_to_high_lag_{lag}"] = grp["close_to_high"].shift(lag)
        df[f"volume_ratio_lag_{lag}"] = grp["volume_ratio_5"].shift(lag)
        df[f"limit_gap_lag_{lag}"] = grp["limit_gap"].shift(lag)
        df[f"limit_hit_lag_{lag}"] = grp["limit_hit"].shift(lag)
        df[f"failed_limit_hit_lag_{lag}"] = grp["failed_limit_hit"].shift(lag)

    df["curve_slope_5"] = grp["close"].transform(lambda s: s / s.shift(5) - 1)
    df["curve_accel_5"] = grp["pct_chg"].transform(
        lambda s: s.rolling(3, min_periods=2).mean() - s.shift(3).rolling(2, min_periods=1).mean()
    )
    rolling_high_5 = grp["high"].transform(lambda s: s.rolling(5, min_periods=2).max())
    rolling_low_5 = grp["low"].transform(lambda s: s.rolling(5, min_periods=2).min())
    df["curve_max_drawdown_5"] = df["close"] / rolling_high_5 - 1
    df["curve_close_position_5"] = (df["close"] - rolling_low_5) / (rolling_high_5 - rolling_low_5)
    df["volume_trend_5"] = grp["volume"].transform(lambda s: s / s.shift(5) - 1)
    df["amount_trend_5"] = grp["amount"].transform(lambda s: s / s.shift(5) - 1)
    df["range_mean_5"] = grp["range"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    df["close_to_high_mean_5"] = grp["close_to_high"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    df["limit_gap_mean_5"] = grp["limit_gap"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    df["limit_gap_min_5"] = grp["limit_gap"].transform(lambda s: s.rolling(5, min_periods=2).min())
