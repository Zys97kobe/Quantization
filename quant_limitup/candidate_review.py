from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import limit_up_price


def load_rank_history(path: Path, fallback_files: tuple[Path, ...] = ()) -> pd.DataFrame:
    frames = []
    covered_dates: set[str] = set()
    sources = (path, *fallback_files)
    for source in sources:
        if source.exists():
            frame = pd.read_csv(source)
            if not frame.empty and {"date", "symbol", "score"}.issubset(frame.columns):
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                if covered_dates:
                    frame = frame[~frame["date"].isin(covered_dates)]
                covered_dates.update(frame["date"].dropna().astype(str))
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True, sort=False)
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    history = history.dropna(subset=["date", "symbol", "score"])
    history = history.drop_duplicates(["date", "symbol"], keep="last")
    return _sort_history(history).reset_index(drop=True)


def upsert_rank_history(path: Path, rank: pd.DataFrame) -> None:
    if rank.empty or not {"date", "symbol", "score"}.issubset(rank.columns):
        return
    rank = rank.copy()
    if "display_rank" not in rank.columns:
        rank["display_rank"] = rank.groupby("date").cumcount() + 1
    existing = load_rank_history(path)
    dates = set(pd.to_datetime(rank["date"]).dt.strftime("%Y-%m-%d"))
    if not existing.empty:
        existing = existing[~existing["date"].isin(dates)]
    combined = pd.concat([existing, rank], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    combined = _sort_history(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def load_completed_signals(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if frame.empty or "signal_date" not in frame.columns:
        return set()
    return set(frame["signal_date"].astype(str))


def pending_candidate_pool(
    history: pd.DataFrame,
    completed_signals: set[str],
    trades: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for signal_date, ranked in history.groupby("date", sort=True):
        if str(signal_date) in completed_signals:
            continue
        ranked = _sort_snapshot(ranked)
        selected = ranked.head(10).copy()
        bought = _bought_symbols(trades, str(signal_date))
        if bought:
            selected = pd.concat(
                [selected, ranked[ranked["symbol"].astype(str).isin(bought)]],
                ignore_index=True,
            ).drop_duplicates("symbol")
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    pool = pd.concat(rows, ignore_index=True, sort=False).drop_duplicates("symbol")
    pool["is_st"] = pool.get("is_st", pool["name"].astype(str).str.contains("ST", case=False, regex=False)).astype(int)
    return pool


def build_pending_reviews(
    prices: pd.DataFrame,
    history: pd.DataFrame,
    completed_signals: set[str],
    trades: pd.DataFrame | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> list[dict]:
    if prices.empty or history.empty:
        return []
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    as_of = pd.to_datetime(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.now().normalize()
    prices = prices[prices["date"] < as_of]
    reviews = []
    for signal_text, ranked in history.groupby("date", sort=True):
        signal_text = str(signal_text)
        if signal_text in completed_signals:
            continue
        signal_date = pd.to_datetime(signal_text).normalize()
        later = prices.loc[prices["date"] > signal_date, "date"]
        if later.empty:
            continue
        result_date = later.min()
        ranked = _sort_snapshot(ranked).copy()
        ranked["candidate_rank"] = range(1, len(ranked) + 1)
        candidate_count = min(10, len(ranked))
        top10 = ranked.head(candidate_count).copy()
        bought = _bought_symbols(trades, signal_text)
        extras = ranked[
            ranked["symbol"].astype(str).isin(bought)
            & ~ranked["symbol"].astype(str).isin(set(top10["symbol"].astype(str)))
        ]
        evaluated = pd.concat([top10, extras], ignore_index=True)
        result = prices[prices["date"] == result_date].set_index("symbol")
        records = []
        for item in evaluated.itertuples(index=False):
            hit = None
            if item.symbol in result.index:
                outcome = result.loc[item.symbol]
                if isinstance(outcome, pd.DataFrame):
                    outcome = outcome.iloc[-1]
                raw_is_st = getattr(item, "is_st", None)
                is_st = (
                    int(raw_is_st)
                    if raw_is_st is not None and not pd.isna(raw_is_st)
                    else int(str(item.name).upper().startswith(("ST", "*ST")))
                )
                expected_limit = limit_up_price(float(item.close), item.board, is_st)
                hit = bool(float(outcome["high"]) >= expected_limit * 0.999)
            records.append({
                "candidate_rank": int(item.candidate_rank),
                "symbol": str(item.symbol),
                "name": str(item.name),
                "score": float(item.score),
                "hit_limit": hit,
                "was_bought": str(item.symbol) in bought,
                "in_top10": int(item.candidate_rank) <= candidate_count,
            })
        top_records = [item for item in records if item["in_top10"]]
        available = [item for item in top_records if item["hit_limit"] is not None]
        hit_rate = sum(bool(item["hit_limit"]) for item in available) / len(available) if available else 0.0
        reviews.append({
            "signal_date": signal_text,
            "result_date": result_date.strftime("%Y-%m-%d"),
            "top10_hit_rate": hit_rate,
            "top10_evaluated_count": len(available),
            "candidate_count": candidate_count,
            "evaluated_candidates": records,
        })
    return reviews


def mark_review_completed(path: Path, review: dict) -> None:
    candidate_count = int(review.get("candidate_count", 10))
    records = review.get("evaluated_candidates") or []
    hit_count = sum(
        bool(item.get("hit_limit"))
        for item in records
        if item.get("in_top10") and item.get("hit_limit") is not None
    )
    row = pd.DataFrame([{
        "signal_date": review["signal_date"],
        "result_date": review["result_date"],
        "hit_count": hit_count,
        "candidate_count": candidate_count,
        "top10_hit_rate": review["top10_hit_rate"],
        "hit_rate_pct": review["top10_hit_rate"] * 100,
        "top10_evaluated_count": review["top10_evaluated_count"],
    }])
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty and "signal_date" in existing.columns:
        existing = existing[existing["signal_date"].astype(str) != str(review["signal_date"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([existing, row], ignore_index=True, sort=False)
    columns = [
        "signal_date",
        "result_date",
        "hit_count",
        "candidate_count",
        "top10_evaluated_count",
        "top10_hit_rate",
        "hit_rate_pct",
    ]
    combined["hit_count"] = pd.to_numeric(combined["hit_count"], errors="coerce").astype("Int64")
    combined["candidate_count"] = pd.to_numeric(combined["candidate_count"], errors="coerce").astype("Int64")
    combined["top10_evaluated_count"] = pd.to_numeric(
        combined["top10_evaluated_count"], errors="coerce"
    ).astype("Int64")
    combined[columns].to_csv(path, index=False)


def _bought_symbols(trades: pd.DataFrame | None, signal_date: str) -> set[str]:
    if trades is None or trades.empty or not {"date", "action", "symbol"}.issubset(trades.columns):
        return set()
    buys = trades[(trades["date"].astype(str) == signal_date) & (trades["action"] == "BUY")]
    return set(buys["symbol"].astype(str))


def _sort_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    if "display_rank" in frame.columns and frame["display_rank"].notna().any():
        return frame.sort_values(["display_rank", "score"], ascending=[True, False], na_position="last")
    return frame.sort_values("score", ascending=False)


def _sort_history(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("date", sort=True):
        parts.append(_sort_snapshot(group))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else frame
