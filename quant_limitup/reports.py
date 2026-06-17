from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_dashboard(rank: pd.DataFrame, trades: pd.DataFrame, summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top_rank = rank.head(20).copy()
    latest_trades = trades.tail(30).copy() if not trades.empty else trades
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A Share Limit-Up Lab</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #17202a; background: #f7f8fa; }}
    header {{ background: #111827; color: white; padding: 22px 32px; }}
    main {{ padding: 24px 32px; max-width: 1280px; margin: 0 auto; }}
    h1 {{ margin: 0; font-size: 24px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    .metric b {{ display: block; font-size: 20px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf0f3; font-size: 13px; text-align: right; }}
    th:first-child, td:first-child, td:nth-child(2), th:nth-child(2), td:nth-child(3), th:nth-child(3) {{ text-align: left; }}
    th {{ background: #f1f5f9; color: #334155; }}
  </style>
</head>
<body>
  <header>
    <h1>A 股次日涨停预测模拟系统</h1>
  </header>
  <main>
    <section class="metrics">
      {_metric("交易次数", summary.get("trades", 0))}
      {_metric("总收益", _pct(summary.get("total_return", 0)))}
      {_metric("最大回撤", _pct(summary.get("max_drawdown", 0)))}
      {_metric("胜率", _pct(summary.get("win_rate", 0)))}
      {_metric("触及涨停率", _pct(summary.get("limit_hit_rate", 0)))}
      {_metric("单笔均值", _pct(summary.get("avg_trade_return", 0)))}
    </section>
    <h2>最新候选排名</h2>
    {top_rank.to_html(index=False, classes="rank", float_format=lambda x: f"{x:.4f}")}
    <h2>最近模拟交易</h2>
    {latest_trades.to_html(index=False, classes="trades", float_format=lambda x: f"{x:.4f}")}
  </main>
</body>
</html>"""
    path.write_text(html)


def _metric(label: str, value: object) -> str:
    return f'<div class="metric"><span>{label}</span><b>{value}</b></div>'


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"
