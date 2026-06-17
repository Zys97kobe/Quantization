from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .backtest import rank_candidates
from .config import DEFAULT_PAPER_DAILY, DEFAULT_PAPER_STATE, DEFAULT_PAPER_TRADES, TradingConfig
from .filters import filter_tradeable
from .model import LogisticLimitUpModel


@dataclass
class Position:
    symbol: str
    name: str
    board: str
    buy_date: str
    shares: int
    buy_price: float
    cost: float
    score: float


@dataclass
class PaperAccount:
    initial_cash: float = 10_000.0
    cash: float = 10_000.0
    positions: list[Position] = field(default_factory=list)
    last_run_date: str | None = None
    last_buy_date: str | None = None
    last_sell_date: str | None = None


def run_paper_day(
    frame: pd.DataFrame,
    model: LogisticLimitUpModel,
    config: TradingConfig,
    state_file: Path = DEFAULT_PAPER_STATE,
    trades_file: Path = DEFAULT_PAPER_TRADES,
    daily_file: Path = DEFAULT_PAPER_DAILY,
    date: str | None = None,
    settle: bool = True,
    open_new: bool = True,
    minute_bars: pd.DataFrame | None = None,
    sell_mode: str = "eod",
) -> tuple[PaperAccount, pd.DataFrame, dict]:
    account = load_account(state_file, config.initial_cash)
    work = frame.copy()
    current_date = pd.to_datetime(date) if date else work["date"].max()
    current_date_str = current_date.strftime("%Y-%m-%d")
    day = work[work["date"] == current_date].copy()
    if day.empty:
        raise RuntimeError(f"No rows available for paper date {current_date.date()}")

    trade_rows: list[dict] = []
    skipped = []
    if settle:
        if account.last_sell_date == current_date_str:
            skipped.append("sell_already_ran")
        else:
            _settle_positions(account, day, current_date, config, trade_rows, minute_bars, sell_mode)
            if sell_mode in {"eod", "force"}:
                account.last_sell_date = current_date_str

    rank = rank_candidates(work, model, current_date.strftime("%Y-%m-%d"))
    opened = 0
    if open_new:
        if account.last_buy_date == current_date_str:
            skipped.append("buy_already_ran")
        else:
            opened = _open_positions(account, rank, day, current_date, config, trade_rows)
            if opened > 0:
                account.last_buy_date = current_date_str

    equity = _mark_to_market(account, day)
    daily_pnl = equity - _previous_equity(daily_file, account.initial_cash)
    daily_return = daily_pnl / max(equity - daily_pnl, 1)
    account.last_run_date = current_date.strftime("%Y-%m-%d")
    save_account(account, state_file)
    append_csv(trades_file, trade_rows)

    summary = {
        "date": account.last_run_date,
        "initial_cash": account.initial_cash,
        "cash": account.cash,
        "equity": equity,
        "daily_pnl": daily_pnl,
        "daily_return": daily_return,
        "total_return": equity / account.initial_cash - 1,
        "open_positions": len(account.positions),
        "opened_positions": opened,
        "closed_trades": len([row for row in trade_rows if row["action"] == "SELL"]),
        "phase": _phase_name(settle, open_new, sell_mode),
    }
    if skipped:
        summary["skipped"] = ",".join(skipped)
    append_csv(daily_file, [summary])
    return account, rank, summary


def load_account(path: Path, initial_cash: float) -> PaperAccount:
    if not path.exists():
        return PaperAccount(initial_cash=initial_cash, cash=initial_cash)
    data = json.loads(path.read_text())
    positions = [Position(**item) for item in data.get("positions", [])]
    return PaperAccount(
        initial_cash=float(data.get("initial_cash", initial_cash)),
        cash=float(data.get("cash", initial_cash)),
        positions=positions,
        last_run_date=data.get("last_run_date"),
        last_buy_date=data.get("last_buy_date") or data.get("last_run_date"),
        last_sell_date=data.get("last_sell_date"),
    )


def save_account(account: PaperAccount, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(account)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True, sort=False)
    frame.to_csv(path, index=False)


def reset_account(path: Path = DEFAULT_PAPER_STATE, initial_cash: float = 10_000.0) -> PaperAccount:
    account = PaperAccount(initial_cash=initial_cash, cash=initial_cash)
    save_account(account, path)
    return account


def _settle_positions(
    account: PaperAccount,
    day: pd.DataFrame,
    current_date: pd.Timestamp,
    config: TradingConfig,
    trade_rows: list[dict],
    minute_bars: pd.DataFrame | None = None,
    sell_mode: str = "eod",
) -> None:
    remaining = []
    day_by_symbol = day.set_index("symbol")
    for pos in account.positions:
        buy_date = pd.to_datetime(pos.buy_date)
        if buy_date >= current_date or pos.symbol not in day_by_symbol.index:
            remaining.append(pos)
            continue
        row = day_by_symbol.loc[pos.symbol]
        decision = _sell_decision(pos, row, current_date, minute_bars, sell_mode)
        if decision is None:
            remaining.append(pos)
            continue
        raw_sell_price, hit_limit, reason = decision
        sell_price = raw_sell_price * (1 - config.sell_slippage_bps / 10_000)
        gross_sell = pos.shares * sell_price
        sell_fee = max(gross_sell * config.commission_bps / 10_000, config.min_commission)
        stamp_tax = gross_sell * config.stamp_tax_bps / 10_000
        proceeds = gross_sell - sell_fee - stamp_tax
        pnl = proceeds - pos.cost
        account.cash += proceeds
        trade_rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "action": "SELL",
                "symbol": pos.symbol,
                "name": pos.name,
                "shares": pos.shares,
                "price": sell_price,
                "score": pos.score,
                "cash_after": account.cash,
                "pnl": pnl,
                "return_pct": pnl / max(pos.cost, 1),
                "hit_limit": hit_limit,
                "reason": reason,
            }
        )
    account.positions = remaining


def _open_positions(
    account: PaperAccount,
    rank: pd.DataFrame,
    day: pd.DataFrame,
    current_date: pd.Timestamp,
    config: TradingConfig,
    trade_rows: list[dict],
) -> int:
    existing_symbols = {pos.symbol for pos in account.positions}
    candidates = filter_tradeable(rank[rank["score"] >= config.min_score_to_buy]).head(config.max_positions_per_day)
    if candidates.empty:
        return 0
    day_by_symbol = day.set_index("symbol")
    opened = 0
    cash_budget = account.cash
    for row in candidates.itertuples(index=False):
        if row.symbol in existing_symbols or row.symbol not in day_by_symbol.index:
            continue
        price_row = day_by_symbol.loc[row.symbol]
        capital = min(cash_budget / max(config.max_positions_per_day - opened, 1), account.initial_cash * config.max_position_pct)
        buy_price = float(price_row["close"]) * (1 + config.buy_slippage_bps / 10_000)
        shares = int(capital / buy_price / 100) * 100
        if shares <= 0:
            continue
        gross_buy = shares * buy_price
        buy_fee = max(gross_buy * config.commission_bps / 10_000, config.min_commission)
        cost = gross_buy + buy_fee
        if cost > account.cash:
            continue
        account.cash -= cost
        account.positions.append(
            Position(
                symbol=row.symbol,
                name=row.name,
                board=row.board,
                buy_date=current_date.strftime("%Y-%m-%d"),
                shares=shares,
                buy_price=buy_price,
                cost=cost,
                score=float(row.score),
            )
        )
        trade_rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "action": "BUY",
                "symbol": row.symbol,
                "name": row.name,
                "shares": shares,
                "price": buy_price,
                "cost": cost,
                "score": float(row.score),
                "cash_after": account.cash,
                "pnl": 0.0,
                "return_pct": 0.0,
                "hit_limit": False,
            }
        )
        opened += 1
    return opened


def _mark_to_market(account: PaperAccount, day: pd.DataFrame) -> float:
    day_by_symbol = day.set_index("symbol")
    value = account.cash
    for pos in account.positions:
        if pos.symbol in day_by_symbol.index:
            value += pos.shares * float(day_by_symbol.loc[pos.symbol]["close"])
        else:
            value += pos.cost
    return float(value)


def _previous_equity(daily_file: Path, initial_cash: float) -> float:
    if not daily_file.exists():
        return initial_cash
    daily = pd.read_csv(daily_file)
    if daily.empty:
        return initial_cash
    return float(daily.iloc[-1]["equity"])


def _sell_decision(
    pos: Position,
    row: pd.Series,
    current_date: pd.Timestamp,
    minute_bars: pd.DataFrame | None,
    sell_mode: str,
) -> tuple[float, bool, str] | None:
    limit_price = float(row["limit_up_price"])
    if sell_mode == "morning":
        bars = _position_minutes(pos.symbol, current_date, minute_bars)
        morning = bars[(bars["time"] >= "09:30") & (bars["time"] <= "10:30")] if not bars.empty else bars
        if morning.empty:
            return None
        if float(morning["high"].max()) >= limit_price * 0.999:
            return limit_price, True, "morning_limit_hit"
        running_high = morning["high"].cummax()
        pullback = morning["close"] / running_high - 1
        hit_pullback = morning[pullback <= -0.03]
        if not hit_pullback.empty:
            return float(hit_pullback.iloc[0]["close"]), False, "morning_pullback"
        return None

    if sell_mode == "force":
        bars = _position_minutes(pos.symbol, current_date, minute_bars)
        if not bars.empty:
            before_force = bars[bars["time"] <= "14:50"]
            price = float((before_force if not before_force.empty else bars).iloc[-1]["close"])
            hit_limit = float(bars["high"].max()) >= limit_price * 0.999
            return (limit_price if hit_limit else price), hit_limit, "force_sell"
        hit_limit = float(row["high"]) >= limit_price * 0.999
        return (limit_price if hit_limit else float(row["close"])), hit_limit, "force_sell_daily"

    hit_limit = float(row["high"]) >= limit_price * 0.999
    return (limit_price if hit_limit else float(row["close"])), hit_limit, "eod_settle"


def _position_minutes(symbol: str, current_date: pd.Timestamp, minute_bars: pd.DataFrame | None) -> pd.DataFrame:
    if minute_bars is None or minute_bars.empty:
        return pd.DataFrame()
    bars = minute_bars.copy()
    if "datetime" not in bars.columns:
        return pd.DataFrame()
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars["date"] = bars["datetime"].dt.normalize()
    bars["time"] = bars["datetime"].dt.strftime("%H:%M")
    return bars[(bars["symbol"] == symbol) & (bars["date"] == current_date.normalize())].sort_values("datetime")


def _phase_name(settle: bool, open_new: bool, sell_mode: str = "eod") -> str:
    if settle and open_new:
        return "full"
    if settle:
        return f"sell-{sell_mode}"
    if open_new:
        return "buy"
    return "none"
