from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import TradingConfig
from .model import train_logistic


def run_walk_forward(
    frame: pd.DataFrame,
    train_days: int = 30,
    min_train_rows: int = 200,
    top_ks: tuple[int, ...] = (3, 5, 10, 20),
    config: TradingConfig | None = None,
    epochs: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = config or TradingConfig()
    data = frame.dropna(subset=["target_limit_up_next", "next_high", "next_close", "next_limit_up_price"]).copy()
    data["date"] = pd.to_datetime(data["date"])
    dates = sorted(data["date"].unique())
    predictions = []
    trades = []
    cash = cfg.initial_cash
    equity_curve = []

    for test_date in dates:
        train_start = pd.Timestamp(test_date) - pd.Timedelta(days=train_days * 2)
        train = data[(data["date"] < test_date) & (data["date"] >= train_start)].copy()
        if len(train) < min_train_rows:
            train = data[data["date"] < test_date].tail(min_train_rows * 3).copy()
        if len(train) < min_train_rows or train["target_limit_up_next"].nunique() < 2:
            continue

        test = data[data["date"] == test_date].copy()
        if test.empty:
            continue
        try:
            model, _ = train_logistic(train, epochs=epochs)
        except ValueError:
            continue
        test["score"] = model.predict_proba(test)
        ranked = test.sort_values("score", ascending=False).reset_index(drop=True)

        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            predictions.append(
                {
                    "date": row.date,
                    "rank": rank,
                    "symbol": row.symbol,
                    "name": row.name,
                    "score": row.score,
                    "hit_limit": int(row.target_limit_up_next),
                    "next_close": row.next_close,
                    "next_limit_up_price": row.next_limit_up_price,
                }
            )

        picks = ranked.head(cfg.max_positions_per_day)
        if picks.empty:
            equity_curve.append({"date": test_date, "equity": cash})
            continue
        capital_per_trade = min(cash / len(picks), cfg.initial_cash * cfg.max_position_pct)
        for row in picks.itertuples(index=False):
            buy_price = row.close * (1 + cfg.buy_slippage_bps / 10_000)
            shares = int(capital_per_trade / buy_price / 100) * 100
            if shares <= 0:
                continue
            gross_buy = shares * buy_price
            buy_fee = max(gross_buy * cfg.commission_bps / 10_000, cfg.min_commission)
            if gross_buy + buy_fee > cash:
                continue
            cash -= gross_buy + buy_fee

            hit_limit = row.next_high >= row.next_limit_up_price * 0.999
            raw_sell_price = row.next_limit_up_price if hit_limit else row.next_close
            sell_price = raw_sell_price * (1 - cfg.sell_slippage_bps / 10_000)
            gross_sell = shares * sell_price
            sell_fee = max(gross_sell * cfg.commission_bps / 10_000, cfg.min_commission)
            stamp_tax = gross_sell * cfg.stamp_tax_bps / 10_000
            cash += gross_sell - sell_fee - stamp_tax
            pnl = gross_sell - sell_fee - stamp_tax - gross_buy - buy_fee
            trades.append(
                {
                    "date": test_date,
                    "symbol": row.symbol,
                    "name": row.name,
                    "score": row.score,
                    "shares": shares,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "hit_limit": bool(hit_limit),
                    "pnl": pnl,
                    "return_pct": pnl / max(gross_buy + buy_fee, 1),
                    "cash_after": cash,
                }
            )
        equity_curve.append({"date": test_date, "equity": cash})

    pred_df = pd.DataFrame(predictions)
    trades_df = pd.DataFrame(trades)
    summary = summarize_walk_forward(pred_df, trades_df, pd.DataFrame(equity_curve), cfg.initial_cash, top_ks)
    return pred_df, trades_df, summary


def summarize_walk_forward(
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_cash: float,
    top_ks: tuple[int, ...],
) -> dict:
    if predictions.empty:
        return {"evaluated_days": 0, "predictions": 0, "reason": "no walk-forward predictions"}
    summary = {
        "evaluated_days": int(predictions["date"].nunique()),
        "predictions": int(len(predictions)),
        "actual_hit_rate_all": float(predictions["hit_limit"].mean()),
    }
    for k in top_ks:
        top = predictions[predictions["rank"] <= k]
        summary[f"top{k}_hit_rate"] = float(top["hit_limit"].mean()) if not top.empty else 0.0
        summary[f"top{k}_days_with_hit"] = float(top.groupby("date")["hit_limit"].max().mean()) if not top.empty else 0.0

    final_equity = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else initial_cash
    if equity_curve.empty:
        max_drawdown = 0.0
    else:
        running_max = equity_curve["equity"].cummax()
        max_drawdown = float((equity_curve["equity"] / running_max - 1).min())
    summary.update(
        {
            "trades": int(len(trades)),
            "final_equity": final_equity,
            "total_return": final_equity / initial_cash - 1,
            "max_drawdown": max_drawdown,
            "trade_win_rate": float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0,
            "trade_limit_hit_rate": float(trades["hit_limit"].mean()) if not trades.empty else 0.0,
            "avg_trade_return": float(trades["return_pct"].mean()) if not trades.empty else 0.0,
        }
    )
    return summary


def write_walk_forward_report(
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    summary: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "walk_forward_predictions.csv", index=False)
    trades.to_csv(out_dir / "walk_forward_trades.csv", index=False)
    (out_dir / "walk_forward_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
