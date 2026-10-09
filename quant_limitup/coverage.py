"""Check daily market snapshots against an explicit, independently known pool.

Historical checks deliberately use the supplied pool on every date. A current
pool is only an approximation to the historical universe: new listings,
delistings, ST changes and suspensions can reduce historical coverage. Supply an
appropriate point-in-time pool when that distinction matters. Coverage measures
valid daily bars, not whether an order could actually have traded.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .filters import filter_tradeable


_BAR_COLUMNS = ("open", "high", "low", "close", "volume")
_HISTORY_COLUMNS = (
    "date",
    "market_data_eligible",
    "market_coverage_ratio",
    "market_expected_symbols",
    "market_covered_symbols",
    "market_coverage_reason",
    "market_sh_coverage_ratio",
    "market_sz_coverage_ratio",
)


def _validate_thresholds(total: float, exchange: float) -> None:
    for name, value in (("min_total_coverage", total), ("min_exchange_coverage", exchange)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")


def _expected_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if "symbol" not in pool.columns:
        return pd.DataFrame(columns=["symbol", "exchange"])
    expected = pool.copy()
    expected["symbol"] = expected["symbol"].fillna("").astype(str).str.strip().str.upper()
    expected = expected[expected["symbol"] != ""].drop_duplicates("symbol", keep="last")
    if "name" not in expected.columns:
        expected["name"] = ""
    expected = filter_tradeable(expected)
    if "is_st" in expected.columns:
        is_st = expected["is_st"].astype(str).str.lower().isin(["1", "1.0", "true"])
        expected = expected[~is_st].copy()
    expected["exchange"] = expected["symbol"].str.rsplit(".", n=1).str[-1]
    return expected[["symbol", "exchange"]]


def _price_rows(prices: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "symbol", *_BAR_COLUMNS]
    if "amount" in prices.columns:
        columns.append("amount")
    rows = prices.reindex(columns=columns).copy()
    rows["symbol"] = rows["symbol"].fillna("").astype(str).str.strip().str.upper()
    rows = rows[rows["symbol"].isin(expected["symbol"])].copy()
    # Daily bars use their stated calendar date; do not shift it through UTC.
    rows["date"] = pd.to_datetime(rows["date"].astype(str).str[:10], errors="coerce")
    rows = rows.dropna(subset=["date"]).drop_duplicates(["date", "symbol"], keep="last")
    numeric_columns = [*_BAR_COLUMNS, *(["amount"] if "amount" in rows.columns else [])]
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    ohlc = rows[["open", "high", "low", "close"]]
    valid = pd.Series(np.isfinite(ohlc).all(axis=1), index=rows.index) & ohlc.gt(0).all(axis=1)
    valid &= rows["high"].ge(ohlc.max(axis=1)) & rows["low"].le(ohlc.min(axis=1))
    for column in ("volume", "amount"):
        if column in rows.columns:
            valid &= np.isfinite(rows[column]) & rows[column].ge(0)
    rows["_valid"] = valid
    return rows


def _assess_snapshot(
    snapshot: pd.DataFrame,
    expected: pd.DataFrame,
    date: str | None,
    expected_date: str | None,
    min_total_coverage: float,
    min_exchange_coverage: float,
) -> dict:
    covered = set(snapshot.loc[snapshot["_valid"], "symbol"])
    expected_count = len(expected)
    covered_count = len(covered)
    ratio = covered_count / expected_count if expected_count else 0.0
    reasons: list[str] = []
    if not expected_count:
        reasons.append("empty_tradeable_pool")
    if date is None:
        reasons.append("missing_price_snapshot")
    elif expected_date is not None and date != expected_date:
        reasons.append("stale_price_snapshot" if date < expected_date else "future_price_snapshot")
    if not covered_count:
        reasons.append("no_valid_price_rows")
    if ratio < min_total_coverage:
        reasons.append("total_coverage_below_threshold")
    if not expected["exchange"].isin(["SH", "SZ"]).all():
        reasons.append("unsupported_pool_exchange")
    exchanges = {}
    for exchange, rows in expected.groupby("exchange", sort=True):
        count = len(rows)
        hits = len(set(rows["symbol"]) & covered)
        exchange_ratio = hits / count
        exchanges[str(exchange)] = {
            "expected_symbols": int(count),
            "covered_symbols": int(hits),
            "coverage_ratio": float(exchange_ratio),
            "eligible": bool(exchange_ratio >= min_exchange_coverage),
        }
        if exchange_ratio < min_exchange_coverage:
            reasons.append(f"{exchange}_coverage_below_threshold")
    return {
        "status": "blocked" if reasons else "ok",
        "eligible": not reasons,
        "reasons": reasons,
        "date": date,
        "expected_date": expected_date,
        "covered_symbols": int(covered_count),
        "expected_symbols": int(expected_count),
        "coverage_ratio": float(ratio),
        "exchanges": exchanges,
    }


def assess_price_coverage(
    prices: pd.DataFrame,
    pool: pd.DataFrame,
    expected_date: str | pd.Timestamp | None = None,
    min_total_coverage: float = 0.9,
    min_exchange_coverage: float = 0.85,
) -> dict:
    """Return a JSON-safe gate for the latest pool-related daily snapshot.

    Only unique valid bars in that snapshot count. The last supplied row wins
    when a symbol/date is duplicated, including when that last row is invalid.
    Zero volume is accepted (a suspension need not imply a provider outage).
    ``expected_date`` must come from the caller's trading calendar: an earlier
    or later snapshot fails the gate. If omitted, this function checks coverage
    of the latest stored snapshot without claiming it is current.
    """
    _validate_thresholds(min_total_coverage, min_exchange_coverage)
    target = None
    if expected_date is not None:
        timestamp = pd.Timestamp(expected_date)
        if pd.isna(timestamp):
            raise ValueError("expected_date must be a valid trading date")
        target = timestamp.strftime("%Y-%m-%d")
    expected = _expected_pool(pool)
    rows = _price_rows(prices, expected)
    latest = rows["date"].max()
    date = None if pd.isna(latest) else latest.strftime("%Y-%m-%d")
    snapshot = rows[rows["date"] == latest]
    return _assess_snapshot(snapshot, expected, date, target, min_total_coverage, min_exchange_coverage)


def historical_coverage_by_date(
    prices: pd.DataFrame,
    pool: pd.DataFrame,
    min_total_coverage: float = 0.9,
    min_exchange_coverage: float = 0.85,
) -> pd.DataFrame:
    """Return coverage flags to merge onto daily features by ``date``.

    The denominator is the explicitly supplied tradeable pool on every date;
    it is never inferred from successfully downloaded symbols. Entirely absent
    dates cannot be inferred from prices and are not synthesized here. A merge
    must treat missing eligibility as false. See the module note about using
    today's pool for historical IPOs, delistings and ST changes.
    """
    _validate_thresholds(min_total_coverage, min_exchange_coverage)
    expected = _expected_pool(pool)
    rows = _price_rows(prices, expected)
    records = []
    for date, snapshot in rows.groupby("date", sort=True):
        day = date.strftime("%Y-%m-%d")
        assessment = _assess_snapshot(
            snapshot, expected, day, day, min_total_coverage, min_exchange_coverage
        )
        records.append({
            "date": date,
            "market_data_eligible": assessment["eligible"],
            "market_coverage_ratio": assessment["coverage_ratio"],
            "market_expected_symbols": assessment["expected_symbols"],
            "market_covered_symbols": assessment["covered_symbols"],
            "market_coverage_reason": ";".join(assessment["reasons"]),
            "market_sh_coverage_ratio": assessment["exchanges"].get("SH", {}).get("coverage_ratio"),
            "market_sz_coverage_ratio": assessment["exchanges"].get("SZ", {}).get("coverage_ratio"),
        })
    result = pd.DataFrame(records, columns=_HISTORY_COLUMNS)
    result["date"] = pd.to_datetime(result["date"])
    result["market_data_eligible"] = result["market_data_eligible"].astype(bool)
    return result
