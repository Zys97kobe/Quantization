from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import sleep

import pandas as pd

from .accuracy import write_prediction_accuracy
from .backtest import rank_candidates, run_backtest, write_summary
from .candidate_review import (
    build_pending_reviews,
    load_completed_signals,
    load_rank_history,
    mark_review_completed,
    pending_candidate_pool,
    upsert_rank_history,
)
from .config import (
    CONFIG_DIR,
    DEFAULT_PAPER_DAILY,
    DEFAULT_LEARNING_REPORT,
    DEFAULT_FACTOR_PARAMS,
    DEFAULT_PAPER_STATE,
    DEFAULT_PAPER_TRADES,
    DEFAULT_PREDICTION_ACCURACY,
    MINUTE_FILE,
    MODEL_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORT_DIR,
    TradingConfig,
    ensure_dirs,
)
from .data import read_prices, write_sample_prices
from .emailing import (
    load_email_config,
    send_candidate_review_email,
    send_daily_email,
    write_email_config_template,
)
from .features import build_feature_frame
from .intraday_features import normalize_minute_bars
from .messaging import load_feishu_webhook, send_feishu_candidate_review, send_feishu_daily
from .model import LogisticLimitUpModel, train_logistic
from .paper import _daily_trade_counts, _previous_equity, load_account, reset_account, run_paper_day
from .providers import (
    fetch_akshare_daily_prices,
    fetch_sina_candidate_daily_prices,
    fetch_sina_daily_prices,
    fetch_sina_minute_bars,
    fetch_tushare_daily_prices,
    sina_minute_market_is_current,
    update_sina_stock_pool,
)
from .reports import write_dashboard
from .strategy import build_learning_report, initialize_factor_params, optimize_threshold
from .walk_forward import run_walk_forward, write_walk_forward_report


DEFAULT_PRICE_FILE = RAW_DIR / "daily_prices.csv"
DEFAULT_FEATURE_FILE = PROCESSED_DIR / "features.csv"
DEFAULT_MODEL_FILE = MODEL_DIR / "limitup_logistic.json"
DEFAULT_TUSHARE_TOKEN_FILE = CONFIG_DIR / "tushare_token.txt"
DEFAULT_EMAIL_CONFIG_FILE = CONFIG_DIR / "email.json"
DEFAULT_FEISHU_WEBHOOK_FILE = CONFIG_DIR / "feishu_webhook.txt"
DEFAULT_SYMBOL_POOL_FILE = CONFIG_DIR / "stock_pool.csv"
DAILY_RANK_SNAPSHOT = REPORT_DIR / "today_rank_all.csv"
RANK_HISTORY_FILE = REPORT_DIR / "rank_history.csv"
CANDIDATE_REVIEW_HISTORY = DEFAULT_PAPER_TRADES.parent / "candidate_reviews.csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A-share next-day limit-up research MVP")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init-sample", help="Create synthetic sample daily data")
    init.add_argument("--symbols", type=int, default=80)
    init.add_argument("--days", type=int, default=260)

    build = sub.add_parser("build-dataset", help="Build feature and label dataset")
    build.add_argument("--prices", type=Path, default=DEFAULT_PRICE_FILE)
    build.add_argument("--out", type=Path, default=DEFAULT_FEATURE_FILE)

    train = sub.add_parser("train", help="Train baseline probability model")
    train.add_argument("--features", type=Path, default=DEFAULT_FEATURE_FILE)
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL_FILE)
    train.add_argument("--train-end", type=str, default=None)

    rank = sub.add_parser("rank", help="Rank candidates for one date")
    rank.add_argument("--features", type=Path, default=DEFAULT_FEATURE_FILE)
    rank.add_argument("--model", type=Path, default=DEFAULT_MODEL_FILE)
    rank.add_argument("--date", type=str, default=None)
    rank.add_argument("--out", type=Path, default=REPORT_DIR / "latest_rank.csv")

    bt = sub.add_parser("backtest", help="Run simple next-day paper-trading backtest")
    bt.add_argument("--features", type=Path, default=DEFAULT_FEATURE_FILE)
    bt.add_argument("--model", type=Path, default=DEFAULT_MODEL_FILE)

    fetch = sub.add_parser("fetch-akshare", help="Fetch real A-share daily prices through AkShare")
    fetch.add_argument("--start-date", type=str, required=True, help="YYYYMMDD, e.g. 20240101")
    fetch.add_argument("--end-date", type=str, default=None, help="YYYYMMDD, defaults to today")
    fetch.add_argument("--out", type=Path, default=DEFAULT_PRICE_FILE)
    fetch.add_argument("--adjust", type=str, default="qfq", choices=["", "qfq", "hfq"])
    fetch.add_argument("--limit-symbols", type=int, default=None, help="Debug only: fetch first N symbols")

    fetch_ts = sub.add_parser("fetch-tushare", help="Fetch real A-share daily prices through TuShare Pro")
    fetch_ts.add_argument("--start-date", type=str, required=True, help="YYYYMMDD, e.g. 20240101")
    fetch_ts.add_argument("--end-date", type=str, default=None, help="YYYYMMDD, defaults to today")
    fetch_ts.add_argument("--out", type=Path, default=DEFAULT_PRICE_FILE)
    fetch_ts.add_argument("--token", type=str, default=None, help="Defaults to TUSHARE_TOKEN env var")
    fetch_ts.add_argument("--token-file", type=Path, default=DEFAULT_TUSHARE_TOKEN_FILE)

    fetch_sina = sub.add_parser("fetch-sina", help="Fetch real A-share daily prices through Sina K-line API")
    fetch_sina.add_argument("--out", type=Path, default=DEFAULT_PRICE_FILE)
    fetch_sina.add_argument("--symbols", type=Path, default=DEFAULT_SYMBOL_POOL_FILE)
    fetch_sina.add_argument("--days", type=int, default=260)

    fetch_min = sub.add_parser("fetch-sina-minute", help="Fetch Sina intraday minute bars")
    fetch_min.add_argument("--out", type=Path, default=MINUTE_FILE)
    fetch_min.add_argument("--symbols", type=Path, default=DEFAULT_SYMBOL_POOL_FILE)
    fetch_min.add_argument("--scale", type=int, default=5)
    fetch_min.add_argument("--bars", type=int, default=80)
    fetch_min.add_argument("--max-symbols", type=int, default=None)

    pool = sub.add_parser("update-stock-pool", help="Update full A-share stock pool through Sina")
    pool.add_argument("--out", type=Path, default=DEFAULT_SYMBOL_POOL_FILE)

    real = sub.add_parser("run-real", help="Run pipeline from existing real daily_prices.csv")
    real.add_argument("--prices", type=Path, default=DEFAULT_PRICE_FILE)

    daily = sub.add_parser("daily", help="Fetch data, train, rank, backtest, and write dashboard")
    daily.add_argument("--provider", type=str, default="sina", choices=["akshare", "tushare", "sina", "csv"])
    daily.add_argument("--start-date", type=str, default="20240101")
    daily.add_argument("--end-date", type=str, default=None)
    daily.add_argument("--prices", type=Path, default=DEFAULT_PRICE_FILE)
    daily.add_argument("--limit-symbols", type=int, default=None)
    daily.add_argument("--token", type=str, default=None, help="TuShare token; defaults to TUSHARE_TOKEN")
    daily.add_argument("--token-file", type=Path, default=DEFAULT_TUSHARE_TOKEN_FILE)
    daily.add_argument("--paper", action="store_true", help="Run persistent paper trading account")
    daily.add_argument("--phase", type=str, default="full", choices=["full", "buy", "sell-morning", "sell-force"])
    daily.add_argument("--send-email", action="store_true", help="Send daily paper trading email")
    daily.add_argument("--email-config", type=Path, default=DEFAULT_EMAIL_CONFIG_FILE)
    daily.add_argument("--send-feishu", action="store_true", help="Send daily paper trading Feishu bot message")
    daily.add_argument("--feishu-webhook-file", type=Path, default=DEFAULT_FEISHU_WEBHOOK_FILE)
    daily.add_argument("--symbols", type=Path, default=DEFAULT_SYMBOL_POOL_FILE)
    daily.add_argument("--days", type=int, default=260)
    daily.add_argument("--refresh-stock-pool", action="store_true")
    daily.add_argument("--use-minute", action="store_true")
    daily.add_argument("--minute-mode", type=str, default="top", choices=["top", "all"])
    daily.add_argument("--minute-top-n", type=int, default=300)
    daily.add_argument("--minute-scale", type=int, default=5)
    daily.add_argument("--minute-bars", type=int, default=80)

    launchd = sub.add_parser("make-launchd", help="Write macOS launchd plist for daily run")
    launchd.add_argument("--time", type=str, default="14:45", help="HH:MM in local time")
    launchd.add_argument("--provider", type=str, default="sina", choices=["akshare", "tushare", "sina", "csv"])
    launchd.add_argument("--start-date", type=str, default="20240101")
    launchd.add_argument("--paper", action="store_true")
    launchd.add_argument("--phase", type=str, default="full", choices=["full", "buy", "sell-morning", "sell-force"])
    launchd.add_argument("--send-email", action="store_true")
    launchd.add_argument("--send-feishu", action="store_true")
    launchd.add_argument("--symbols", type=Path, default=DEFAULT_SYMBOL_POOL_FILE)
    launchd.add_argument("--refresh-stock-pool", action="store_true")
    launchd.add_argument("--use-minute", action="store_true")
    launchd.add_argument("--minute-mode", type=str, default="top", choices=["top", "all"])
    launchd.add_argument("--minute-top-n", type=int, default=300)
    launchd.add_argument("--out", type=Path, default=REPORT_DIR / "com.quant.limitup.daily.plist")

    paper = sub.add_parser("paper-daily", help="Run persistent paper trading from existing prices")
    paper.add_argument("--prices", type=Path, default=DEFAULT_PRICE_FILE)
    paper.add_argument("--minute-bars-file", type=Path, default=None)
    paper.add_argument("--phase", type=str, default="full", choices=["full", "buy", "sell-morning", "sell-force"])
    paper.add_argument("--send-email", action="store_true")
    paper.add_argument("--email-config", type=Path, default=DEFAULT_EMAIL_CONFIG_FILE)
    paper.add_argument("--send-feishu", action="store_true")
    paper.add_argument("--feishu-webhook-file", type=Path, default=DEFAULT_FEISHU_WEBHOOK_FILE)

    reset = sub.add_parser("reset-paper", help="Reset paper account state")
    reset.add_argument("--initial-cash", type=float, default=10_000.0)

    sub.add_parser("init-email-config", help="Write email config template")

    init_factors = sub.add_parser("init-factors", help="Initialize limit-up factor parameters from recent history")
    init_factors.add_argument("--prices", type=Path, default=DEFAULT_PRICE_FILE)
    init_factors.add_argument("--minute-bars-file", type=Path, default=None)
    init_factors.add_argument("--lookback-days", type=int, default=22)

    wf = sub.add_parser("walk-forward", help="Run rolling out-of-sample model evaluation")
    wf.add_argument("--prices", type=Path, default=DEFAULT_PRICE_FILE)
    wf.add_argument("--features", type=Path, default=DEFAULT_FEATURE_FILE)
    wf.add_argument("--minute-bars-file", type=Path, default=None)
    wf.add_argument("--train-days", type=int, default=30)
    wf.add_argument("--min-train-rows", type=int, default=200)
    wf.add_argument("--epochs", type=int, default=500)
    wf.add_argument("--build", action="store_true", help="Rebuild feature dataset before evaluation")
    wf.add_argument("--pretrain-final", action="store_true", help="Train final model on all labeled rows after evaluation")

    sub.add_parser("run-pipeline", help="Run build, train, rank, backtest, and dashboard")

    args = parser.parse_args(argv)
    ensure_dirs()

    if args.cmd == "init-sample":
        df = write_sample_prices(DEFAULT_PRICE_FILE, symbols=args.symbols, days=args.days)
        print(f"Wrote {len(df)} rows to {DEFAULT_PRICE_FILE}")
        return

    if args.cmd == "build-dataset":
        build_dataset(args.prices, args.out)
        return

    if args.cmd == "train":
        train_model(args.features, args.model, args.train_end)
        return

    if args.cmd == "rank":
        rank_to_file(args.features, args.model, args.out, args.date)
        return

    if args.cmd == "backtest":
        backtest_to_files(args.features, args.model)
        return

    if args.cmd == "fetch-akshare":
        df = fetch_akshare_daily_prices(
            args.out,
            start_date=args.start_date,
            end_date=args.end_date,
            adjust=args.adjust,
            limit_symbols=args.limit_symbols,
        )
        print(f"Wrote {len(df)} real price rows to {args.out}")
        return

    if args.cmd == "fetch-tushare":
        df = fetch_tushare_daily_prices(
            args.out,
            start_date=args.start_date,
            end_date=args.end_date,
            token=_resolve_token(args.token, args.token_file),
        )
        print(f"Wrote {len(df)} real price rows to {args.out}")
        return

    if args.cmd == "fetch-sina":
        df = fetch_sina_daily_prices(args.out, args.symbols, days=args.days)
        print(f"Wrote {len(df)} real price rows to {args.out}")
        return

    if args.cmd == "fetch-sina-minute":
        df = fetch_sina_minute_bars(
            args.out,
            args.symbols,
            scale=args.scale,
            bars=args.bars,
            max_symbols=args.max_symbols,
        )
        print(f"Wrote {len(df)} minute rows to {args.out}")
        return

    if args.cmd == "update-stock-pool":
        pool = update_sina_stock_pool(args.out)
        print(f"Wrote {len(pool)} A-share symbols to {args.out}")
        return

    if args.cmd == "run-real":
        run_from_prices(args.prices)
        return

    if args.cmd == "daily":
        no_position_sell_phase = False
        if args.provider == "akshare":
            fetch_akshare_daily_prices(
                args.prices,
                start_date=args.start_date,
                end_date=args.end_date,
                limit_symbols=args.limit_symbols,
            )
        elif args.provider == "tushare":
            fetch_tushare_daily_prices(
                args.prices,
                start_date=args.start_date,
                end_date=args.end_date,
                token=_resolve_token(args.token, args.token_file),
            )
        elif args.provider == "sina":
            if args.paper and args.phase == "sell-morning":
                run_pending_candidate_reviews(
                    send_email=args.send_email,
                    email_config_file=args.email_config,
                    send_feishu=args.send_feishu,
                    feishu_webhook_file=args.feishu_webhook_file,
                )
            if args.paper and args.use_minute and not sina_minute_market_is_current(
                args.symbols,
                scale=args.minute_scale,
                bars=min(args.minute_bars, 2),
            ):
                print(f"Skipping {args.phase}: no minute bars for {pd.Timestamp.now():%Y-%m-%d} (market closed)")
                return
            if args.refresh_stock_pool:
                safe_update_sina_stock_pool(args.symbols)
            run_with_retries("daily price fetch", fetch_sina_daily_prices, args.prices, args.symbols, days=args.days)
            if args.use_minute:
                minute_symbols = args.symbols
                max_symbols = None
                if args.phase in {"sell-morning", "sell-force"}:
                    if paper_account_has_positions():
                        minute_symbols = build_position_symbol_pool(
                            args.symbols,
                            CONFIG_DIR / "position_stock_pool.csv",
                        )
                    else:
                        no_position_sell_phase = True
                        print(f"Skipping minute fetch for {args.phase}: no open positions")
                elif args.minute_mode == "top":
                    minute_symbols = build_minute_symbol_pool(
                        args.prices,
                        args.symbols,
                        CONFIG_DIR / "minute_stock_pool.csv",
                        top_n=args.minute_top_n,
                    )
                if not no_position_sell_phase:
                    run_with_retries(
                        "minute bar fetch",
                        fetch_sina_minute_bars,
                        MINUTE_FILE,
                        minute_symbols,
                        scale=args.minute_scale,
                        bars=args.minute_bars,
                        max_symbols=max_symbols,
                    )
        if args.paper:
            if args.use_minute and args.phase in {"sell-morning", "sell-force"} and no_position_sell_phase:
                send_no_position_sell_report(
                    args.phase,
                    args.send_email,
                    args.email_config,
                    args.send_feishu,
                    args.feishu_webhook_file,
                )
                return
            if args.use_minute and not minute_data_is_current(MINUTE_FILE):
                print(f"Skipping {args.phase}: no minute bars for {pd.Timestamp.now():%Y-%m-%d} (market closed)")
                return
            run_paper_from_prices(
                args.prices,
                MINUTE_FILE if args.use_minute else None,
                args.send_email,
                args.email_config,
                args.send_feishu,
                args.feishu_webhook_file,
                args.phase,
            )
        else:
            run_from_prices(args.prices)
        return

    if args.cmd == "make-launchd":
        write_launchd_plist(
            args.out,
            args.time,
            args.provider,
            args.start_date,
            args.paper,
            args.send_email,
            args.send_feishu,
            args.symbols,
            args.refresh_stock_pool,
            args.use_minute,
            args.minute_mode,
            args.minute_top_n,
            args.phase,
        )
        print(f"Wrote launchd template to {args.out}")
        return

    if args.cmd == "paper-daily":
        run_paper_from_prices(
            args.prices,
            args.minute_bars_file,
            args.send_email,
            args.email_config,
            args.send_feishu,
            args.feishu_webhook_file,
            args.phase,
        )
        return

    if args.cmd == "reset-paper":
        reset_account(DEFAULT_PAPER_STATE, args.initial_cash)
        for path in [DEFAULT_PAPER_TRADES, DEFAULT_PAPER_DAILY]:
            if path.exists():
                path.unlink()
        print(f"Reset paper account to {args.initial_cash:.2f}")
        return

    if args.cmd == "init-email-config":
        write_email_config_template(DEFAULT_EMAIL_CONFIG_FILE)
        print(f"Wrote email config template to {DEFAULT_EMAIL_CONFIG_FILE}")
        return

    if args.cmd == "init-factors":
        features = build_dataset(args.prices, DEFAULT_FEATURE_FILE, args.minute_bars_file)
        params = initialize_factor_params(features, args.lookback_days, DEFAULT_FACTOR_PARAMS)
        print(json.dumps(params, ensure_ascii=False, indent=2, default=str))
        return

    if args.cmd == "walk-forward":
        if args.build or not args.features.exists():
            frame = build_dataset(args.prices, args.features, args.minute_bars_file)
        else:
            frame = pd.read_csv(args.features, parse_dates=["date"])
        preds, trades, summary = run_walk_forward(
            frame,
            train_days=args.train_days,
            min_train_rows=args.min_train_rows,
            epochs=args.epochs,
        )
        write_walk_forward_report(preds, trades, summary, REPORT_DIR)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        if args.pretrain_final:
            train_model(args.features, DEFAULT_MODEL_FILE, None)
        return

    if args.cmd == "run-pipeline":
        if not DEFAULT_PRICE_FILE.exists():
            write_sample_prices(DEFAULT_PRICE_FILE)
        run_from_prices(DEFAULT_PRICE_FILE)
        return


def build_dataset(price_file: Path, out_file: Path, minute_file: Path | None = None) -> pd.DataFrame:
    prices = read_prices(price_file)
    minute_bars = None
    if minute_file and minute_file.exists():
        minute_bars = normalize_minute_bars(pd.read_csv(minute_file))
    features = build_feature_frame(prices, minute_bars)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_file, index=False)
    print(f"Wrote {len(features)} feature rows to {out_file}")
    return features


def train_model(feature_file: Path, model_file: Path, train_end: str | None) -> LogisticLimitUpModel:
    frame = pd.read_csv(feature_file, parse_dates=["date"])
    train_frame = frame.dropna(subset=["target_limit_up_next"])
    train_frame = train_frame if train_end is None else train_frame[train_frame["date"] <= pd.to_datetime(train_end)]
    model, metrics = train_logistic(train_frame)
    model.save(model_file)
    metrics_file = model_file.with_suffix(".metrics.json")
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote model to {model_file}")
    print(json.dumps(metrics, indent=2))
    return model


def rank_to_file(feature_file: Path, model_file: Path, out_file: Path, date: str | None) -> pd.DataFrame:
    frame = pd.read_csv(feature_file, parse_dates=["date"])
    model = LogisticLimitUpModel.load(model_file)
    rank = rank_candidates(frame, model, date)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rank.to_csv(out_file, index=False)
    print(f"Wrote {len(rank)} ranked candidates to {out_file}")
    print(rank.head(10).to_string(index=False))
    return rank


def backtest_to_files(feature_file: Path, model_file: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(feature_file, parse_dates=["date"])
    model = LogisticLimitUpModel.load(model_file)
    trades, summary = run_backtest(frame, model)
    trades_file = REPORT_DIR / "backtest_trades.csv"
    summary_file = REPORT_DIR / "backtest_summary.json"
    trades.to_csv(trades_file, index=False)
    write_summary(summary, summary_file)
    print(f"Wrote trades to {trades_file}")
    print(json.dumps(summary, indent=2))
    return trades, summary


def run_from_prices(price_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    build_dataset(price_file, DEFAULT_FEATURE_FILE)
    train_model(DEFAULT_FEATURE_FILE, DEFAULT_MODEL_FILE, None)
    rank = rank_to_file(DEFAULT_FEATURE_FILE, DEFAULT_MODEL_FILE, REPORT_DIR / "latest_rank.csv", None)
    trades, summary = backtest_to_files(DEFAULT_FEATURE_FILE, DEFAULT_MODEL_FILE)
    write_dashboard(rank, trades, summary, REPORT_DIR / "dashboard.html")
    print(f"Wrote dashboard to {REPORT_DIR / 'dashboard.html'}")
    return rank, trades, summary


def build_minute_symbol_pool(
    price_file: Path,
    full_symbols_file: Path,
    out_file: Path,
    top_n: int = 300,
) -> Path:
    feature_file = PROCESSED_DIR / "features_daily_prefilter.csv"
    features = build_dataset(price_file, feature_file, None)
    train_frame = features.dropna(subset=["target_limit_up_next"])
    if train_frame["target_limit_up_next"].nunique() < 2:
        pool = pd.read_csv(full_symbols_file).head(top_n)
    else:
        model, _ = train_logistic(train_frame, epochs=300)
        rank = rank_candidates(features, model)
        top_symbols = set(rank.head(top_n)["symbol"].astype(str))
        account = load_account(DEFAULT_PAPER_STATE, 10_000.0)
        top_symbols.update(pos.symbol for pos in account.positions)
        pool = pd.read_csv(full_symbols_file)
        pool = pool[pool["symbol"].astype(str).isin(top_symbols)].copy()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out_file, index=False)
    print(f"Wrote {len(pool)} symbols for minute fetch to {out_file}")
    return out_file


def build_position_symbol_pool(full_symbols_file: Path, out_file: Path) -> Path:
    account = load_account(DEFAULT_PAPER_STATE, 10_000.0)
    symbols = {pos.symbol for pos in account.positions}
    pool = pd.read_csv(full_symbols_file)
    if symbols:
        pool = pool[pool["symbol"].astype(str).isin(symbols)].copy()
    else:
        pool = pool.head(0).copy()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out_file, index=False)
    print(f"Wrote {len(pool)} position symbols for minute fetch to {out_file}")
    return out_file


def safe_update_sina_stock_pool(path: Path, attempts: int = 3, retry_sleep_seconds: float = 2.0) -> pd.DataFrame | None:
    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return update_sina_stock_pool(path)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                print(f"Warning: stock pool refresh attempt {attempt}/{attempts} failed: {exc}; retrying")
                sleep(retry_sleep_seconds)
    if stock_pool_is_usable(path):
        print(f"Warning: stock pool refresh failed after {attempts} attempts, using existing {path}: {last_error}")
        return None
    raise RuntimeError(f"Stock pool refresh failed and no usable local pool exists: {last_error}") from last_error


def run_with_retries(label: str, func, *args, attempts: int = 3, retry_sleep_seconds: float = 2.0, **kwargs):
    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                print(f"Warning: {label} attempt {attempt}/{attempts} failed: {exc}; retrying")
                sleep(retry_sleep_seconds)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


def stock_pool_is_usable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        pool = pd.read_csv(path, usecols=["symbol", "name"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    return not pool.empty


def paper_account_has_positions() -> bool:
    account = load_account(DEFAULT_PAPER_STATE, 10_000.0)
    return bool(account.positions)


def send_no_position_sell_report(
    phase: str,
    send_email: bool,
    email_config_file: Path,
    send_feishu: bool,
    feishu_webhook_file: Path,
) -> dict:
    account = load_account(DEFAULT_PAPER_STATE, 10_000.0)
    current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    equity = float(account.cash)
    previous_equity = _previous_equity(DEFAULT_PAPER_DAILY, account.initial_cash, pd.to_datetime(current_date))
    daily_pnl = equity - previous_equity
    daily_return = daily_pnl / max(equity - daily_pnl, 1)
    opened_today, closed_today = _daily_trade_counts(DEFAULT_PAPER_TRADES, current_date)
    summary = {
        "date": current_date,
        "initial_cash": account.initial_cash,
        "cash": account.cash,
        "equity": equity,
        "daily_pnl": daily_pnl,
        "daily_return": daily_return,
        "total_return": equity / account.initial_cash - 1,
        "open_positions": 0,
        "opened_positions": opened_today,
        "closed_trades": closed_today,
        "phase": phase,
        "skipped": "no_positions",
        "note": f"{phase} 无持仓，本阶段无需卖出",
    }
    append_daily_summary_once(DEFAULT_PAPER_DAILY, summary)
    (REPORT_DIR / "paper_daily_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    empty_rank = pd.DataFrame()
    empty_trades = pd.DataFrame()
    empty_positions = pd.DataFrame()
    if send_email:
        email_config = load_email_config(email_config_file)
        send_daily_email(email_config, summary, empty_rank, empty_trades, empty_trades, empty_positions, None)
        print(f"Sent daily email to {email_config.recipient}")
    if send_feishu:
        webhook = load_feishu_webhook(feishu_webhook_file)
        send_feishu_daily(webhook, summary, empty_rank, empty_trades, empty_trades, empty_positions, None)
        print("Sent daily Feishu message")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def append_daily_summary_once(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([summary])
    if not path.exists():
        new_row.to_csv(path, index=False)
        return
    existing = pd.read_csv(path)
    if existing.empty:
        new_row.to_csv(path, index=False)
        return
    same = (
        (existing.get("date", pd.Series(dtype=str)).astype(str) == str(summary.get("date")))
        & (existing.get("phase", pd.Series(dtype=str)).astype(str) == str(summary.get("phase")))
        & (existing.get("skipped", pd.Series(dtype=str)).fillna("").astype(str) == str(summary.get("skipped", "")))
    )
    if same.any():
        return
    pd.concat([existing, new_row], ignore_index=True, sort=False).to_csv(path, index=False)


def run_paper_from_prices(
    price_file: Path,
    minute_file: Path | None,
    send_email: bool,
    email_config_file: Path,
    send_feishu: bool,
    feishu_webhook_file: Path,
    phase: str = "full",
) -> tuple[pd.DataFrame, dict]:
    build_dataset(price_file, DEFAULT_FEATURE_FILE, minute_file)
    model = train_model(DEFAULT_FEATURE_FILE, DEFAULT_MODEL_FILE, None)
    frame = pd.read_csv(DEFAULT_FEATURE_FILE, parse_dates=["date"])
    base_config = TradingConfig(initial_cash=10_000.0)
    initialize_factor_params(frame, 22, DEFAULT_FACTOR_PARAMS)
    threshold, best = optimize_threshold(frame, model, base_config, REPORT_DIR / "strategy_optimization.json")
    minute_bars = normalize_minute_bars(pd.read_csv(minute_file)) if minute_file and minute_file.exists() else None
    latest_labeled = frame.loc[frame["target_limit_up_next"].notna(), "date"].max()
    bought_symbols = load_buy_symbols(latest_labeled)
    candidate_rank = load_rank_snapshot(latest_labeled)
    learning = build_learning_report(
        frame,
        model,
        bought_symbols=bought_symbols,
        candidate_rank=candidate_rank,
    )
    DEFAULT_LEARNING_REPORT.write_text(json.dumps(learning, indent=2, ensure_ascii=False, default=str))
    config = TradingConfig(
        initial_cash=10_000.0,
        max_positions_per_day=base_config.max_positions_per_day,
        max_position_pct=float(best.get("max_position_pct", base_config.max_position_pct)),
        min_score_to_buy=threshold,
        buy_slippage_bps=base_config.buy_slippage_bps,
        sell_slippage_bps=base_config.sell_slippage_bps,
        commission_bps=base_config.commission_bps,
        stamp_tax_bps=base_config.stamp_tax_bps,
        min_commission=base_config.min_commission,
    )
    settle, open_new, sell_mode = _phase_flags(phase)
    account, rank, summary = run_paper_day(
        frame,
        model,
        config,
        settle=settle,
        open_new=open_new,
        minute_bars=minute_bars,
        sell_mode=sell_mode,
    )
    summary["optimized_threshold"] = threshold
    summary["optimized_max_position_pct"] = config.max_position_pct
    summary["optimization_total_return"] = best.get("total_return")
    summary["learning_top5_hit_rate"] = learning.get("top5_hit_rate")
    rank.to_csv(REPORT_DIR / "latest_rank.csv", index=False)
    display_rank = rank.head(10).copy()
    display_rank["display_rank"] = range(1, len(display_rank) + 1)
    if phase in {"buy", "full"}:
        display_rank.to_csv(DAILY_RANK_SNAPSHOT, index=False)
        upsert_rank_history(RANK_HISTORY_FILE, display_rank)
    accuracy = write_prediction_accuracy(frame, model, summary, learning, DEFAULT_PREDICTION_ACCURACY)
    if accuracy:
        summary["prediction_accuracy_file"] = str(DEFAULT_PREDICTION_ACCURACY)
    (REPORT_DIR / "paper_daily_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    buys, sells = load_trade_lists(summary["date"])
    positions = build_position_list(account, rank)
    notification_learning = None if phase == "sell-morning" else learning
    if send_email:
        email_config = load_email_config(email_config_file)
        send_daily_email(email_config, summary, display_rank, buys, sells, positions, notification_learning)
        print(f"Sent daily email to {email_config.recipient}")
    if send_feishu:
        webhook = load_feishu_webhook(feishu_webhook_file)
        send_feishu_daily(webhook, summary, display_rank, buys, sells, positions, notification_learning)
        print("Sent daily Feishu message")
    print(json.dumps(summary, indent=2, default=str))
    return rank, summary


def load_trade_lists(date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DEFAULT_PAPER_TRADES.exists():
        return pd.DataFrame(), pd.DataFrame()
    trades = pd.read_csv(DEFAULT_PAPER_TRADES)
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    day = trades[trades["date"].astype(str) == str(date)].copy()
    if "cost" not in day.columns:
        day["cost"] = day["shares"] * day["price"]
    buys = day[day["action"] == "BUY"].copy()
    sells = day[day["action"] == "SELL"].copy()
    return buys, sells


def load_buy_symbols(date: str | pd.Timestamp | None) -> set[str]:
    if date is None or pd.isna(date) or not DEFAULT_PAPER_TRADES.exists():
        return set()
    trades = pd.read_csv(DEFAULT_PAPER_TRADES)
    if trades.empty or not {"date", "action", "symbol"}.issubset(trades.columns):
        return set()
    target = pd.to_datetime(date).strftime("%Y-%m-%d")
    buys = trades[(trades["date"].astype(str) == target) & (trades["action"] == "BUY")]
    return set(buys["symbol"].astype(str))


def load_rank_snapshot(date: str | pd.Timestamp | None) -> pd.DataFrame | None:
    if date is None or pd.isna(date) or not DAILY_RANK_SNAPSHOT.exists():
        return None
    rank = pd.read_csv(DAILY_RANK_SNAPSHOT)
    if rank.empty or "date" not in rank.columns:
        return None
    target = pd.to_datetime(date).normalize()
    snapshot = rank[pd.to_datetime(rank["date"], errors="coerce").dt.normalize() == target].copy()
    return snapshot if not snapshot.empty else None


def run_pending_candidate_reviews(
    send_email: bool,
    email_config_file: Path,
    send_feishu: bool,
    feishu_webhook_file: Path,
) -> list[dict]:
    history = load_rank_history(
        RANK_HISTORY_FILE,
        fallback_files=(DAILY_RANK_SNAPSHOT, REPORT_DIR / "latest_rank.csv"),
    )
    completed = load_completed_signals(CANDIDATE_REVIEW_HISTORY)
    trades = pd.read_csv(DEFAULT_PAPER_TRADES) if DEFAULT_PAPER_TRADES.exists() else pd.DataFrame()
    candidates = pending_candidate_pool(history, completed, trades)
    if candidates.empty:
        return []
    prices = fetch_sina_candidate_daily_prices(candidates, days=5)
    reviews = build_pending_reviews(prices, history, completed, trades)
    if not reviews:
        return []
    email_config = load_email_config(email_config_file) if send_email else None
    webhook = load_feishu_webhook(feishu_webhook_file) if send_feishu else None
    for review in reviews:
        if email_config:
            send_candidate_review_email(email_config, review)
        if webhook:
            send_feishu_candidate_review(webhook, review)
        report_file = REPORT_DIR / f"candidate_review_{review['signal_date']}.json"
        report_file.write_text(json.dumps(review, indent=2, ensure_ascii=False, default=str))
        mark_review_completed(CANDIDATE_REVIEW_HISTORY, review)
        candidate_count = int(review.get("candidate_count", 10))
        print(
            f"Reviewed candidates {review['signal_date']} -> {review['result_date']}: "
            f"Top{candidate_count} hit rate {review['top10_hit_rate'] * 100:.2f}%"
        )
    return reviews


def minute_data_is_current(path: Path, expected_date: str | None = None) -> bool:
    if not path.exists():
        return False
    try:
        minute = pd.read_csv(path, usecols=["datetime"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    if minute.empty:
        return False
    dates = pd.to_datetime(minute["datetime"], errors="coerce").dropna().dt.normalize()
    if dates.empty:
        return False
    expected = pd.to_datetime(expected_date).normalize() if expected_date else pd.Timestamp.now().normalize()
    return bool((dates == expected).any())


def build_position_list(account, rank: pd.DataFrame) -> pd.DataFrame:
    if not account.positions:
        return pd.DataFrame()
    prices = rank.set_index("symbol")["close"] if not rank.empty else pd.Series(dtype=float)
    rows = []
    for pos in account.positions:
        current_price = float(prices.get(pos.symbol, pos.buy_price))
        market_value = pos.shares * current_price
        rows.append(
            {
                "symbol": pos.symbol,
                "name": pos.name,
                "shares": pos.shares,
                "buy_price": pos.buy_price,
                "cost": pos.cost,
                "score": pos.score,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": market_value - pos.cost,
            }
        )
    return pd.DataFrame(rows)


def write_launchd_plist(
    out_file: Path,
    run_time: str,
    provider: str,
    start_date: str,
    paper: bool = False,
    send_email: bool = False,
    send_feishu: bool = False,
    symbols_file: Path = DEFAULT_SYMBOL_POOL_FILE,
    refresh_stock_pool: bool = False,
    use_minute: bool = False,
    minute_mode: str = "top",
    minute_top_n: int = 300,
    phase: str = "full",
) -> None:
    hour, minute = _parse_hhmm(run_time)
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "reports"
    args = [
        sys.executable,
        "-m",
        "quant_limitup.cli",
        "daily",
        "--provider",
        provider,
        "--start-date",
        start_date,
        "--phase",
        phase,
    ]
    if provider == "tushare":
        args.extend(["--token-file", str(DEFAULT_TUSHARE_TOKEN_FILE)])
    if provider == "sina":
        args.extend(["--symbols", str(symbols_file)])
        if refresh_stock_pool:
            args.append("--refresh-stock-pool")
        if use_minute:
            args.append("--use-minute")
            args.extend(["--minute-mode", minute_mode, "--minute-top-n", str(minute_top_n)])
    if paper:
        args.append("--paper")
    if send_email:
        args.extend(["--send-email", "--email-config", str(DEFAULT_EMAIL_CONFIG_FILE)])
    if send_feishu:
        args.extend(["--send-feishu", "--feishu-webhook-file", str(DEFAULT_FEISHU_WEBHOOK_FILE)])
    plist_args = "\n".join(f"    <string>{arg}</string>" for arg in args)
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{_launchd_label(phase)}</string>
  <key>WorkingDirectory</key>
  <string>{root}</string>
  <key>ProgramArguments</key>
  <array>
{plist_args}
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>{hour}</integer>
    <key>Minute</key>
    <integer>{minute}</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>{log_dir / "daily.out.log"}</string>
  <key>StandardErrorPath</key>
  <string>{log_dir / "daily.err.log"}</string>
</dict>
</plist>
"""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content)


def _phase_flags(phase: str) -> tuple[bool, bool, str]:
    if phase == "buy":
        return False, True, "eod"
    if phase == "sell-morning":
        return True, False, "morning"
    if phase == "sell-force":
        return True, False, "force"
    return True, True, "eod"


def _launchd_label(phase: str) -> str:
    suffix = {
        "full": "daily",
        "buy": "buy",
        "sell-morning": "sell-morning",
        "sell-force": "sell-force",
    }[phase]
    return f"com.quant.limitup.{suffix}"


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("--time must use HH:MM")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("--time must use HH:MM in 24-hour time")
    return hour, minute


def _resolve_token(token: str | None, token_file: Path | None) -> str | None:
    if token:
        return token.strip()
    if token_file and token_file.exists():
        value = token_file.read_text().strip()
        return value or None
    return None


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
