from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "date",
    "symbol",
    "name",
    "board",
    "is_st",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "free_float_mkt_cap",
}


def read_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Price file is missing columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["is_st"] = df["is_st"].astype(int)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "free_float_mkt_cap",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def write_sample_prices(path: Path, symbols: int = 80, days: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    boards = ["main", "main", "main", "chinext", "star", "bse"]
    rows: list[dict] = []

    for idx in range(symbols):
        board = boards[idx % len(boards)]
        symbol = _sample_symbol(idx, board)
        name = f"Sample{idx + 1:03d}"
        is_st = 1 if idx % 37 == 0 else 0
        price = rng.uniform(5, 45)
        base_volume = rng.uniform(20_000_000, 250_000_000)
        free_float = rng.uniform(2e9, 40e9)
        momentum = 0.0

        for date in dates:
            theme_boost = 0.0
            if idx % 11 in {1, 2, 3} and date.day % 17 in {1, 2, 3}:
                theme_boost = rng.uniform(0.006, 0.025)
            momentum = 0.62 * momentum + rng.normal(0, 0.009) + theme_boost
            ret = np.clip(momentum + rng.normal(0, 0.017), -0.09, 0.12)
            open_gap = rng.normal(0, 0.012)
            open_price = max(1.0, price * (1 + open_gap))
            close = max(1.0, price * (1 + ret))
            high = max(open_price, close) * (1 + abs(rng.normal(0.012, 0.015)))
            low = min(open_price, close) * (1 - abs(rng.normal(0.012, 0.012)))
            limit_rate = limit_up_rate(board, is_st)
            high = min(high, price * (1 + limit_rate))
            close = min(close, price * (1 + limit_rate))
            if close > 160:
                close = close * rng.uniform(0.86, 0.98)
            volume = base_volume * (1 + abs(ret) * 8 + rng.uniform(-0.25, 0.35))
            volume = max(volume, base_volume * 0.2)
            amount = volume * (open_price + close) / 2
            turnover = np.clip(volume / (free_float / close), 0.003, 0.65)

            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "name": name,
                    "board": board,
                    "is_st": is_st,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": int(volume),
                    "amount": round(amount, 2),
                    "turnover": round(float(turnover), 6),
                    "free_float_mkt_cap": round(float(free_float), 2),
                }
            )
            price = close

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def limit_up_rate(board: str, is_st: int) -> float:
    if is_st:
        return 0.05
    if board in {"star", "chinext"}:
        return 0.20
    if board == "bse":
        return 0.30
    return 0.10


def limit_up_price(prev_close: float, board: str, is_st: int) -> float:
    return math.floor(prev_close * (1 + limit_up_rate(board, is_st)) * 100 + 0.5) / 100


def _sample_symbol(idx: int, board: str) -> str:
    if board == "star":
        return f"688{idx:03d}.SH"
    if board == "chinext":
        return f"300{idx:03d}.SZ"
    if board == "bse":
        return f"83{idx:04d}.BJ"
    if idx % 2:
        return f"60{idx:04d}.SH"
    return f"00{idx:04d}.SZ"
