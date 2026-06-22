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
    positions: pd.DataFrame | None = None,
    learning: dict | None = None,
) -> None:
    text = _daily_text(summary, rank, buys, sells, positions, learning)
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


def send_feishu_candidate_review(webhook: str, review: dict) -> None:
    payload = {"msg_type": "text", "content": {"text": candidate_review_text(review)}}
    response = requests.post(webhook, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("code") not in (0, None):
        raise RuntimeError(f"Feishu webhook returned error: {data}")


def candidate_review_text(review: dict) -> str:
    candidate_count = int(review.get("candidate_count", 10))
    lines = [
        f"昨日候选日线复盘 - {review['result_date']}",
        f"信号日: {review['signal_date']}",
        f"Top{candidate_count}命中率: {review.get('top10_hit_rate', 0) * 100:.2f}%",
        f"Top{candidate_count}日线覆盖: {review.get('top10_evaluated_count', 0)}/{candidate_count}",
        "",
        f"Top{candidate_count}明细:",
    ]
    records = review.get("evaluated_candidates") or []
    for item in (row for row in records if row.get("in_top10")):
        hit = _hit_label(item.get("hit_limit"), "daily")
        bought = "｜已买入" if item.get("was_bought") else ""
        lines.append(
            f"{int(item['candidate_rank'])}. {item['symbol']} {item['name']} "
            f"分数={item['score']:.4f}｜{hit}{bought}"
        )
    extras = [row for row in records if not row.get("in_top10") and row.get("was_bought")]
    if extras:
        lines.append("昨日买入（Top10外）:")
        for item in extras:
            hit = _hit_label(item.get("hit_limit"), "daily")
            lines.append(
                f"排名{int(item['candidate_rank'])} {item['symbol']} {item['name']} "
                f"分数={item['score']:.4f}｜{hit}｜已买入"
            )
    lines.extend(["", "口径: 使用结果日日线最高价判断是否触及涨停。"])
    return "\n".join(lines)


def _daily_text(
    summary: dict,
    rank: pd.DataFrame,
    buys: pd.DataFrame | None,
    sells: pd.DataFrame | None,
    positions: pd.DataFrame | None,
    learning: dict | None,
) -> str:
    show_strategy = summary.get("phase") in {"buy", "full"}
    show_evaluation = (
        summary.get("phase") == "sell-morning"
        and learning
        and learning.get("evaluated_candidates")
        and str(learning.get("latest_result_date")) == str(summary.get("date"))
    )
    lines = [
        _report_title(summary),
        f"资金池余额: {summary['equity']:.2f}",
        f"现金: {summary['cash']:.2f}",
        f"当日收益: {summary['daily_pnl']:.2f} ({summary['daily_return'] * 100:.2f}%)",
        f"累计收益: {summary['total_return'] * 100:.2f}%",
        f"当前持仓: {summary['open_positions']}",
        f"今日新开仓: {summary['opened_positions']}",
        f"今日已结算: {summary['closed_trades']}",
    ]
    if show_strategy:
        lines.extend([
            f"优化阈值: {summary.get('optimized_threshold', 0):.4f}",
            f"优化单只仓位上限: {summary.get('optimized_max_position_pct', 0) * 100:.0f}%",
            "",
            "今日买入:",
        ])
        lines.extend(_trade_lines(buys, "BUY"))
    else:
        lines.extend(["", "今日卖出:"])
        lines.extend(_trade_lines(sells, "SELL"))
    lines.extend(["", "当前持仓明细:"])
    lines.extend(_position_lines(positions))
    if show_evaluation:
        lines.extend(
            [
                "",
                "昨日候选日线复盘:",
                f"Top10命中率: {learning.get('top10_hit_rate', 0) * 100:.2f}%",
            ]
        )
        lines.extend(_evaluation_lines(learning))
    if show_strategy:
        lines.extend(["", "候选前 10:"])
        top = rank.head(10)
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


def _report_title(summary: dict) -> str:
    prefix = {
        "sell-morning": "Sell Morining",
        "sell-force": "Sell Force",
        "buy": "Buy",
    }.get(summary.get("phase"), "A股模拟盘日报")
    return f"{prefix} - {summary['date']}"


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


def _position_lines(positions: pd.DataFrame | None) -> list[str]:
    if positions is None or positions.empty:
        return ["无"]
    return [
        f"{item.symbol} {item.name} {int(item.shares)}股 买入价={item.buy_price:.4f} "
        f"现价={item.current_price:.2f} 市值={item.market_value:.2f} "
        f"浮动盈亏={item.unrealized_pnl:.2f} 模型分数={item.score:.4f}"
        for item in positions.itertuples(index=False)
    ]


def _evaluation_lines(learning: dict) -> list[str]:
    records = learning.get("evaluated_candidates") or []
    if not records:
        return []
    result_date = learning.get("latest_result_date")
    signal_date = learning.get("latest_labeled_signal_date")
    lines = ["", f"Top10明细（信号日 {signal_date}，结果日 {result_date}）:"]
    top10 = [item for item in records if item.get("in_top10")]
    extras = [item for item in records if not item.get("in_top10") and item.get("was_bought")]
    for item in top10:
        hit = _hit_label(item.get("hit_limit"), learning.get("evaluation_source"))
        bought = "｜已买入" if item.get("was_bought") else ""
        lines.append(
            f"{int(item['candidate_rank'])}. {item['symbol']} {item['name']} "
            f"分数={item['score']:.4f}｜{hit}{bought}"
        )
    if extras:
        lines.append("昨日买入（Top10外）:")
        for item in extras:
            hit = _hit_label(item.get("hit_limit"), learning.get("evaluation_source"))
            lines.append(
                f"排名{int(item['candidate_rank'])} {item['symbol']} {item['name']} "
                f"分数={item['score']:.4f}｜{hit}｜已买入"
            )
    return lines


def _hit_label(value: bool | None, source: str | None = None) -> str:
    if value is None:
        return "无日线数据" if source == "daily" else "无分钟数据"
    return "涨停" if value else "未涨停"
