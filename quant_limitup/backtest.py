from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import TradingConfig
from .filters import filter_tradeable
from .model import LogisticLimitUpModel


def rank_candidates(frame: pd.DataFrame, model: LogisticLimitUpModel, date: str | None = None) -> pd.DataFrame:
    df = frame.copy()
    if date is None:
        date_value = df["date"].max()
    else:
        date_value = pd.to_datetime(date)
    day = df[df["date"] == date_value].copy()
    if day.empty:
        raise ValueError(f"No feature rows found for date {date_value.date()}")
    day["score"] = model.predict_proba(day)
    day["suggest_reason"] = day.apply(_reason, axis=1)
    day = filter_tradeable(day)
    cols = [
        "date",
        "symbol",
        "name",
        "board",
        "close",
        "score",
        "turnover",
        "ret_3",
        "ret_5",
        "volume_ratio_5",
        "market_ret",
        "market_limit_hits",
        "market_limit_hit_rate",
        "market_failed_limit_hits",
        "market_symbol_count",
        "market_data_eligible",
        "market_coverage_ratio",
        "market_expected_symbols",
        "market_covered_symbols",
        "market_coverage_reason",
        "market_sh_coverage_ratio",
        "market_sz_coverage_ratio",
        "board_limit_hits",
        "limit_hits_5",
        "failed_limit_hits_5",
        "consecutive_limit_hits",
        "suggest_reason",
    ]
    cols = [col for col in cols if col in day.columns]
    return day.sort_values("score", ascending=False)[cols].reset_index(drop=True)


def run_backtest(
    frame: pd.DataFrame,
    model: LogisticLimitUpModel,
    config: TradingConfig | None = None,
    *,
    scores: pd.Series | None = None,
    minute_bars=None,
) -> tuple[pd.DataFrame, dict]:
    """Replay sell-morning, sell-force and buy in order, without writing accounts.

    A missing minute archive enables an explicitly labelled daily-close proxy
    for research only. It is not an intraday execution backtest.
    """
    # Local import avoids the paper module's rank_candidates import cycle.
    from .paper import PaperAccount, _open_positions, _settle_positions, _mark_to_market
    from .execution import prepare_execution_quotes

    cfg = config or TradingConfig()
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    if scores is not None:
        if not frame.index.is_unique or not scores.index.is_unique or not frame.index.isin(scores.index).all():
            raise ValueError("Precomputed scores must have a unique index covering every feature row.")
        df["score"] = pd.to_numeric(scores.reindex(frame.index), errors="coerce")
        if df["score"].isna().any() or not df["score"].between(0.0, 1.0).all():
            raise ValueError("Precomputed scores must be finite probabilities between zero and one.")
    if "target_end_date" not in df:
        df["target_end_date"] = df.sort_values("date").groupby("symbol")["date"].shift(-1)
    df["target_end_date"] = pd.to_datetime(df["target_end_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    df = df.dropna(subset=["next_close", "next_limit_up_price", "target_end_date"])
    df = df[df.target_end_date > df.date].sort_values(["date", "symbol"])
    if scores is None and not df.empty:
        df["score"] = model.predict_proba(df)
    quotes = prepare_execution_quotes(minute_bars)
    intraday = minute_bars is not None
    signals = {date: day for date, day in df.groupby("date", sort=True)}
    # Outcomes provide quotes for final liquidation dates outside a selection
    # slice. They never provide scores or entry decisions for those dates.
    outcomes = df.copy()
    outcomes["prev_close"] = outcomes["close"]
    outcomes["date"] = outcomes["target_end_date"]
    outcomes["close"] = outcomes["next_close"]
    outcomes["limit_up_price"] = outcomes["next_limit_up_price"]
    # Prefer outcome rows: their prior close and limit are available even in
    # minimal externally scored fixtures that omit daily feature columns.
    price_columns = ["date", "symbol", "name", "board", "close", "prev_close", "limit_up_price"]
    observed = frame.assign(date=pd.to_datetime(frame["date"]).dt.normalize().astype("datetime64[ns]")).reindex(columns=price_columns)
    prices = pd.concat([observed, outcomes[price_columns]], ignore_index=True)
    prices = prices.drop_duplicates(["date", "symbol"], keep="last")
    days = {date: day for date, day in prices.groupby("date", sort=True)}
    dates = sorted(set(signals) | set(outcomes.date))
    account = PaperAccount(initial_cash=cfg.initial_cash, cash=cfg.initial_cash)
    trades, curve = [], []
    missing_quotes = set()
    for date in dates:
        day = days[date]
        held = {pos.symbol: pos for pos in account.positions}
        sales = []
        if intraday:
            for pos in account.positions:
                quote = quotes.get(pos.symbol, date)
                if not quote.get("morning_valid"):
                    missing_quotes.add((pos.symbol, date, "10:30"))
            _settle_positions(account, day, date, cfg, sales, quotes, "morning")
            for pos in account.positions:
                if not quotes.get(pos.symbol, date).get("force_valid"):
                    missing_quotes.add((pos.symbol, date, "14:30"))
            _settle_positions(account, day, date, cfg, sales, quotes, "force")
        else:
            _settle_positions(account, day, date, cfg, sales, None, "eod")
        for sale in sales:
            pos = held[sale["symbol"]]
            trades.append({
                **sale, "date": pd.Timestamp(pos.buy_date), "sell_date": date,
                "sell_time": "10:30" if sale["reason"].startswith("morning") else "14:30" if intraday else "15:00",
                "shares": pos.shares, "buy_price": pos.buy_price, "sell_price": sale["price"],
                "execution_source": "scheduled_minutes" if intraday else "daily_close_proxy",
            })
        if date in signals:
            rank = filter_tradeable(signals[date]).sort_values("score", ascending=False)
            _open_positions(account, rank, signals[date], date, cfg, [])
        curve.append({"date": date, "equity": _mark_to_market(account, day)})
    trades_df = pd.DataFrame(trades)
    summary = _summary(trades_df, pd.DataFrame(curve), cfg.initial_cash)
    summary.update({
        "execution_mode": "scheduled_minutes" if intraday else "daily_close_proxy",
        "execution_complete": bool(intraday and not missing_quotes and not account.positions),
        "missing_execution_quotes": len(missing_quotes), "open_positions": len(account.positions),
    })
    return trades_df, summary


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))


def _summary(trades: pd.DataFrame, curve: pd.DataFrame, initial_cash: float) -> dict:
    final_equity = float(curve["equity"].iloc[-1]) if not curve.empty else initial_cash
    if curve.empty:
        max_drawdown = 0.0
    else:
        running_max = curve["equity"].cummax().clip(lower=initial_cash)
        drawdown = curve["equity"] / running_max - 1
        max_drawdown = float(drawdown.min())
    if trades.empty:
        return {
            "trades": 0,
            "final_equity": final_equity,
            "total_return": final_equity / initial_cash - 1,
            "max_drawdown": max_drawdown,
            "win_rate": 0.0,
            "limit_hit_rate": 0.0,
            "avg_trade_return": 0.0,
        }
    return {
        "trades": int(len(trades)),
        "final_equity": final_equity,
        "total_return": final_equity / initial_cash - 1,
        "max_drawdown": max_drawdown,
        "win_rate": float((trades["pnl"] > 0).mean()),
        "limit_hit_rate": float(trades["hit_limit"].mean()),
        "avg_trade_return": float(trades["return_pct"].mean()),
    }


def _reason(row: pd.Series) -> str:
    reasons = []
    if row["volume_ratio_5"] >= 1.5:
        reasons.append("成交量放大")
    if row["ret_3"] >= 0.05:
        reasons.append("短线动量强")
    if row["turnover"] >= 0.08:
        reasons.append("换手活跃")
    if row["market_limit_hits"] >= 10:
        reasons.append("市场情绪强")
    if row.get("board_limit_hits", 0) >= 5:
        reasons.append("板块情绪强")
    if row.get("limit_hits_5", 0) >= 1:
        reasons.append("近5日触板记忆")
    if row.get("failed_limit_hits_5", 0) >= 2:
        reasons.append("近期炸板偏多")
    return "，".join(reasons) or "模型综合评分"
