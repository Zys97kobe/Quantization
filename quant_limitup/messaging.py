from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests


def load_feishu_webhook(path: Path | None = None) -> str:
    value = os.environ.get("FEISHU_WEBHOOK", "").strip()
    if value:
        return value
    if path and path.exists():
        value = path.read_text().strip()
        if value:
            return value
    raise RuntimeError("Missing Feishu webhook. Set FEISHU_WEBHOOK or create config/feishu_webhook.txt.")


def send_feishu_daily(
    webhook: str,
    summary: dict,
    rank: pd.DataFrame,
    buys: pd.DataFrame | None = None,
    sells: pd.DataFrame | None = None,
    learning: dict | None = None,
) -> None:
    text = _daily_text(summary, rank, buys, sells, learning)
    payload = {"msg_type": "text", "content": {"text": text}}
    response = requests.post(webhook, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("code") not in (0, None):
        raise RuntimeError(f"Feishu webhook returned error: {data}")


def send_feishu_trades(webhook: str, date: str, action: str, trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    title = "买入通知" if action == "BUY" else "卖出通知"
    lines = [f"A股模拟盘{title} {date}", ""]
    lines.extend(_trade_lines(trades, action))
    payload = {"msg_type": "text", "content": {"text": "\n".join(lines)}}
    response = requests.post(webhook, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("code") not in (0, None):
        raise RuntimeError(f"Feishu webhook returned error: {data}")


def _daily_text(
    summary: dict,
    rank: pd.DataFrame,
    buys: pd.DataFrame | None,
    sells: pd.DataFrame | None,
    learning: dict | None,
) -> str:
    lines = [
        f"A股模拟盘日报 {summary['date']}",
        f"资金池余额: {summary['equity']:.2f}",
        f"现金: {summary['cash']:.2f}",
        f"当日收益: {summary['daily_pnl']:.2f} ({summary['daily_return'] * 100:.2f}%)",
        f"累计收益: {summary['total_return'] * 100:.2f}%",
        f"当前持仓: {summary['open_positions']}",
        f"今日新开仓: {summary['opened_positions']}",
        f"今日已结算: {summary['closed_trades']}",
        f"优化阈值: {summary.get('optimized_threshold', 0):.4f}",
        "",
        "今日买入:",
    ]
    lines.extend(_trade_lines(buys, "BUY"))
    lines.extend(
        [
            "",
            "今日卖出:",
        ]
    )
    lines.extend(_trade_lines(sells, "SELL"))
    if learning:
        lines.extend(
            [
                "",
                "策略学习:",
                f"最近有结果信号日: {learning.get('latest_labeled_signal_date')}",
                f"实际触及涨停数: {learning.get('actual_limit_up_count')}",
                f"Top3命中率: {learning.get('top3_hit_rate', 0) * 100:.2f}%",
                f"Top5命中率: {learning.get('top5_hit_rate', 0) * 100:.2f}%",
                f"Top10命中率: {learning.get('top10_hit_rate', 0) * 100:.2f}%",
                f"历史正例率: {learning.get('positive_rate', 0) * 100:.2f}%",
            ]
        )
    lines.extend([
        "",
        "候选前 8:",
    ])
    top = rank.head(8)
    if top.empty:
        lines.append("无候选")
    else:
        for item in top.itertuples(index=False):
            lines.append(
                f"{item.symbol} {item.name} 模型分数={item.score:.4f} "
                f"收盘价={item.close:.2f} {item.suggest_reason}"
            )
    lines.append("")
    lines.append("说明: 当前为虚拟交易记录，不代表真实交易建议。")
    return "\n".join(lines)


def _trade_lines(trades: pd.DataFrame | None, action: str) -> list[str]:
    if trades is None or trades.empty:
        return ["无"]
    lines = []
    for item in trades.itertuples(index=False):
        if action == "BUY":
            lines.append(
                f"{item.symbol} {item.name} {int(item.shares)}股 买入价={item.price:.4f} "
                f"成本={item.cost:.2f} score={item.score:.4f}"
            )
        else:
            lines.append(
                f"{item.symbol} {item.name} {int(item.shares)}股 卖出价={item.price:.4f} "
                f"收益={item.pnl:.2f} 收益率={item.return_pct * 100:.2f}% 涨停={bool(item.hit_limit)}"
            )
    return lines
