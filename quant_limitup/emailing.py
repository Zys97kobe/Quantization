from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from .config import EmailConfig
from .messaging import candidate_review_text


def load_email_config(path: Path) -> EmailConfig:
    if not path.exists():
        raise RuntimeError(f"Missing email config: {path}")
    data = json.loads(path.read_text())
    required = {"smtp_host", "smtp_port", "username", "password", "sender", "recipient"}
    missing = required - set(data)
    if missing:
        raise RuntimeError(f"Email config missing fields: {sorted(missing)}")
    return EmailConfig(**data)


def write_email_config_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "smtp_host": "smtp.qq.com",
        "smtp_port": 587,
        "username": "your_email@qq.com",
        "password": "your_smtp_authorization_code",
        "sender": "your_email@qq.com",
        "recipient": "recipient@example.com",
        "use_tls": True,
    }
    path.write_text(json.dumps(template, indent=2, ensure_ascii=False))


def send_daily_email(
    config: EmailConfig,
    summary: dict,
    rank: pd.DataFrame,
    buys: pd.DataFrame | None = None,
    sells: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    learning: dict | None = None,
) -> None:
    top = rank.head(10)[["symbol", "name", "score", "close", "suggest_reason"]].copy()
    subject = _report_title(summary)
    body = _build_body(summary, top, buys, sells, positions, learning)

    message = EmailMessage()
    message["From"] = config.sender
    message["To"] = config.recipient
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30) as smtp:
        if config.use_tls:
            smtp.starttls()
        smtp.login(config.username, config.password)
        smtp.send_message(message)


def send_trade_email(config: EmailConfig, date: str, action: str, trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    title = "买入通知" if action == "BUY" else "卖出通知"
    message = EmailMessage()
    message["From"] = config.sender
    message["To"] = config.recipient
    message["Subject"] = f"A股模拟盘{title} {date}"
    lines = [f"A股模拟盘{title} {date}", ""]
    lines.extend(_trade_lines(trades, action))
    message.set_content("\n".join(lines))
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30) as smtp:
        if config.use_tls:
            smtp.starttls()
        smtp.login(config.username, config.password)
        smtp.send_message(message)


def send_candidate_review_email(config: EmailConfig, review: dict) -> None:
    message = EmailMessage()
    message["From"] = config.sender
    message["To"] = config.recipient
    message["Subject"] = f"候选日线复盘 {review['signal_date']} → {review['result_date']}"
    message.set_content(candidate_review_text(review))
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30) as smtp:
        if config.use_tls:
            smtp.starttls()
        smtp.login(config.username, config.password)
        smtp.send_message(message)


def _build_body(
    summary: dict,
    top: pd.DataFrame,
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
        f"资金池余额/总权益: {summary['equity']:.2f}",
        f"现金: {summary['cash']:.2f}",
        f"当日收益: {summary['daily_pnl']:.2f} ({summary['daily_return'] * 100:.2f}%)",
        f"累计收益: {summary['total_return'] * 100:.2f}%",
        f"当前持仓数: {summary['open_positions']}",
        f"今日新开仓: {summary['opened_positions']}",
        f"今日已结算交易: {summary['closed_trades']}",
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
        lines.extend(["", "今日候选前10:"])
        if top.empty:
            lines.append("无候选。")
        else:
            for item in top.itertuples(index=False):
                lines.append(
                    f"{item.symbol} {item.name} 模型分数={item.score:.4f} "
                    f"收盘价={item.close:.2f} {item.suggest_reason}"
                )
    lines.extend(["", "说明: 当前为虚拟交易记录，不代表真实交易建议。"])
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
    signal_date = learning.get("latest_labeled_signal_date")
    result_date = learning.get("latest_result_date")
    lines = ["", f"Top10明细（信号日 {signal_date}，结果日 {result_date}）:"]
    for item in (row for row in records if row.get("in_top10")):
        hit = _hit_label(item.get("hit_limit"), learning.get("evaluation_source"))
        bought = "｜已买入" if item.get("was_bought") else ""
        lines.append(
            f"{int(item['candidate_rank'])}. {item['symbol']} {item['name']} "
            f"分数={item['score']:.4f}｜{hit}{bought}"
        )
    extras = [row for row in records if not row.get("in_top10") and row.get("was_bought")]
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
