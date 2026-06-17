from __future__ import annotations

import pandas as pd


def tradeable_mask(frame: pd.DataFrame) -> pd.Series:
    name = frame["name"].astype(str).str.upper()
    board = frame["board"].astype(str).str.lower() if "board" in frame.columns else pd.Series("", index=frame.index)
    symbol = frame["symbol"].astype(str)
    return (
        ~name.str.contains("ST", regex=False)
        & ~name.str.contains("退", regex=False)
        & (board != "bse")
        & ~symbol.str.endswith(".BJ")
    )


def filter_tradeable(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[tradeable_mask(frame)].copy()
