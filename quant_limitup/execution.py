"""Observed quotes and decisions shared by scheduled paper trading and replay."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TradingConfig


@dataclass
class ExecutionQuotes:
    rows: dict = field(default_factory=dict)

    def get(self, symbol, date) -> dict:
        return self.rows.get((str(symbol), pd.Timestamp(date).normalize()), {})


def prepare_execution_quotes(bars: pd.DataFrame | ExecutionQuotes | None) -> ExecutionQuotes:
    if isinstance(bars, ExecutionQuotes):
        return bars
    result = ExecutionQuotes()
    required = {"symbol", "datetime", "open", "high", "low", "close"}
    if bars is None or bars.empty or not required.issubset(bars.columns):
        return result
    work = bars.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work = work.dropna(subset=["symbol", "datetime"]).sort_values("datetime")
    work["date"] = work["datetime"].dt.normalize()
    work["time"] = work["datetime"].dt.strftime("%H:%M:%S")
    columns = ["open", "high", "low", "close"]
    work[columns] = work[columns].apply(pd.to_numeric, errors="coerce")
    values = work[columns]
    work["valid"] = np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1)
    work["valid"] &= work.high.ge(values.max(axis=1)) & work.low.le(values.min(axis=1))
    work["valid"] &= work.datetime.eq(work.datetime.dt.floor("min"))
    work["valid"] &= ~work.duplicated(["symbol", "datetime"], keep=False)
    if "minute_duplicate" in work:
        work["valid"] &= ~work.minute_duplicate.astype(str).str.lower().isin(["true", "1", "1.0"])
    if "volume" in work:
        volume = pd.to_numeric(work.volume, errors="coerce")
        work["valid"] &= volume.gt(0) & np.isfinite(volume)
    expected = set(pd.date_range("2000-01-01 09:35", "2000-01-01 10:30", freq="5min").strftime("%H:%M:%S"))
    for (symbol, date), group in work.groupby(["symbol", "date"], sort=False):
        morning = group[group.time.between("09:35:00", "10:30:00")]
        force = group[group.time == "14:30:00"]
        quote = {
            "morning_valid": bool(set(morning.time) == expected and len(morning) == 12 and morning.valid.all()),
            "force_valid": bool(len(force) == 1 and force.valid.all()),
        }
        if quote["morning_valid"]:
            quote.update(morning_open=float(morning.iloc[0].open), morning_high=float(morning.high.max()),
                         morning_close=float(morning.iloc[-1].close))
        if quote["force_valid"]:
            quote["force_close"] = float(force.iloc[-1].close)
        result.rows[(str(symbol), pd.Timestamp(date))] = quote
    return result


def scheduled_sell_decision(pos, row: pd.Series, quote: dict, mode: str, config: TradingConfig):
    limit = float(row["limit_up_price"])
    if mode == "force":
        if not quote.get("force_valid"):
            return None
        price = quote["force_close"]
        return price, price >= limit * 0.999, "force_sell"
    if not quote.get("morning_valid"):
        return None
    previous = float(row.get("prev_close", pos.buy_price))
    if not np.isfinite(previous) or previous <= 0:
        return None
    if quote["morning_open"] / previous - 1 <= config.morning_opening_defer_return:
        return None
    price = quote["morning_close"]
    # The task observes the morning at 10:30; an earlier touch is a signal,
    # never an entitlement to fill retrospectively at the earlier high.
    if quote["morning_high"] >= limit * 0.999:
        return price, price >= limit * 0.999, "morning_limit_hit"
    if price / quote["morning_high"] - 1 > config.morning_pullback_threshold:
        return None
    proceeds = pos.shares * price * (1 - config.sell_slippage_bps / 10_000)
    pnl = proceeds - max(proceeds * config.commission_bps / 10_000, config.min_commission)
    pnl -= proceeds * config.stamp_tax_bps / 10_000 + pos.cost
    if pnl / max(pos.cost, 1) < config.morning_force_defer_return:
        return None
    if pnl < -config.morning_pullback_min_loss:
        return price, price >= limit * 0.999, "morning_pullback"
    return None
