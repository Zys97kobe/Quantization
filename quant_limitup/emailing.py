from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from .config import EmailConfig


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
    learning: dict | None = None,
) -> None:
    top = rank.head(8)[["symbol", "name", "score", "close", "suggest_reason"]].copy()
    subject = f"A股模拟盘日报 {summary['date']} 余额 {summary['equity']:.2f}"
    body = _build_body(summary, top, buys, sells, learning)

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


def _build_body(
    summary: dict,
    top: pd.DataFrame,
    buys: pd.DataFrame | None,
    sells: pd.DataFrame | None,
    learning: dict | None,
) -> str:
    lines = [
        f"日期: {summary['date']}",
        f"资金池余额/总权益: {summary['equity']:.2f}",
        f"现金: {summary['cash']:.2f}",
        f"当日收益: {summary['daily_pnl']:.2f} ({summary['daily_return'] * 100:.2f}%)",
        f"累计收益: {summary['total_return'] * 100:.2f}%",
        f"当前持仓数: {summary['open_positions']}",
        f"今日新开仓: {summary['opened_positions']}",
        f"今日已结算交易: {summary['closed_trades']}",
        "",
        "今日买入:",
    ]
    lines.extend(_trade_lines(buys, "BUY"))
    lines.extend(["", "今日卖出:"])
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
        "今日候选前列:",
    ])
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
