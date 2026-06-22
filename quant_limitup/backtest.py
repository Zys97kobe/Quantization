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
        "market_limit_hits",
        "suggest_reason",
    ]
    return day.sort_values("score", ascending=False)[cols].reset_index(drop=True)


def run_backtest(
    frame: pd.DataFrame,
    model: LogisticLimitUpModel,
    config: TradingConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = config or TradingConfig()
    df = frame.dropna(subset=["next_high", "next_close", "next_limit_up_price"]).copy()
    df = df.sort_values(["date", "symbol"])
    df["score"] = model.predict_proba(df)
    cash = cfg.initial_cash
    equity_curve = []
    trades = []

    for date, day in df.groupby("date", sort=True):
        picks = (
            filter_tradeable(day[day["score"] >= cfg.min_score_to_buy])
            .sort_values("score", ascending=False)
        )
        if picks.empty:
            equity_curve.append({"date": date, "equity": cash})
            continue

        day_start_cash = cash
        available_cash = cash
        pending_proceeds = 0.0
        opened = 0
        for row in picks.itertuples(index=False):
            if opened >= cfg.max_positions_per_day:
                break
            capital_per_trade = min(available_cash, day_start_cash * cfg.max_position_pct)
            buy_price = row.close * (1 + cfg.buy_slippage_bps / 10_000)
            shares = int(capital_per_trade / buy_price / 100) * 100
            if shares <= 0:
                continue
            gross_buy = shares * buy_price
            buy_fee = max(gross_buy * cfg.commission_bps / 10_000, cfg.min_commission)
            cost = gross_buy + buy_fee
            if cost > available_cash:
                continue
            available_cash -= cost

            hit_limit = row.next_high >= row.next_limit_up_price * 0.999
            raw_sell_price = row.next_limit_up_price if hit_limit else row.next_close
            sell_price = raw_sell_price * (1 - cfg.sell_slippage_bps / 10_000)
            gross_sell = shares * sell_price
            sell_fee = max(gross_sell * cfg.commission_bps / 10_000, cfg.min_commission)
            stamp_tax = gross_sell * cfg.stamp_tax_bps / 10_000
            proceeds = gross_sell - sell_fee - stamp_tax
            pending_proceeds += proceeds
            pnl = gross_sell - sell_fee - stamp_tax - gross_buy - buy_fee
            opened += 1

            trades.append(
                {
                    "date": date,
                    "symbol": row.symbol,
                    "name": row.name,
                    "score": row.score,
                    "shares": shares,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "hit_limit": bool(hit_limit),
                    "pnl": pnl,
                    "return_pct": pnl / max(gross_buy + buy_fee, 1),
                    "cash_after": available_cash + pending_proceeds,
                }
            )
        cash = available_cash + pending_proceeds
        equity_curve.append({"date": date, "equity": cash})

    trades_df = pd.DataFrame(trades)
    curve = pd.DataFrame(equity_curve)
    summary = _summary(trades_df, curve, cfg.initial_cash)
    return trades_df, summary


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))


def _summary(trades: pd.DataFrame, curve: pd.DataFrame, initial_cash: float) -> dict:
    final_equity = float(curve["equity"].iloc[-1]) if not curve.empty else initial_cash
    if curve.empty:
        max_drawdown = 0.0
    else:
        running_max = curve["equity"].cummax()
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
    return "，".join(reasons) or "模型综合评分"
