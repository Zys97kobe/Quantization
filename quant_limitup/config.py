from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MINUTE_FILE = RAW_DIR / "minute_bars.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class TradingConfig:
    initial_cash: float = 10_000.0
    max_positions_per_day: int = 3
    max_position_pct: float = 0.34
    min_score_to_buy: float = 0.005
    buy_slippage_bps: float = 8.0
    sell_slippage_bps: float = 8.0
    commission_bps: float = 2.5
    stamp_tax_bps: float = 5.0
    min_commission: float = 5.0


@dataclass(frozen=True)
class EmailConfig:
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    sender: str
    recipient: str
    use_tls: bool = True


PAPER_DIR = DATA_DIR / "paper"
DEFAULT_PAPER_STATE = PAPER_DIR / "account.json"
DEFAULT_PAPER_TRADES = PAPER_DIR / "trades.csv"
DEFAULT_PAPER_DAILY = PAPER_DIR / "daily_returns.csv"
DEFAULT_PREDICTION_ACCURACY = PAPER_DIR / "prediction_accuracy.csv"
DEFAULT_LEARNING_REPORT = REPORT_DIR / "learning_report.json"
DEFAULT_FACTOR_PARAMS = MODEL_DIR / "factor_params.json"


FEATURE_COLUMNS = [
    "pct_chg",
    "intraday_ret",
    "close_to_high",
    "upper_shadow",
    "volume_ratio_5",
    "amount_ratio_5",
    "turnover",
    "ret_3",
    "ret_5",
    "ret_10",
    "volatility_5",
    "limit_gap",
    "mkt_cap_log",
    "market_ret",
    "market_limit_hits",
    "board_main",
    "board_star",
    "board_chinext",
    "board_bse",
    "is_st",
    "tail_ret_1430_1457",
    "tail_ret_1450_1457",
    "tail_volume_ratio",
    "tail_amount_ratio",
    "tail_volume_vs_5d",
    "tail_high_break",
    "tail_close_to_high",
    "tail_limit_gap",
    "tail_vwap_deviation",
    "tail_pullback",
    "tail_range",
]


def ensure_dirs() -> None:
    for path in [CONFIG_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, PAPER_DIR]:
        path.mkdir(parents=True, exist_ok=True)
