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
    weak_market_limit_hits: int = 80
    weak_market_max_positions: int = 1
    weak_market_max_position_pct: float = 0.20
    weak_market_min_score: float = 0.15
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
    "market_limit_hit_rate",
    "market_failed_limit_hits",
    "board_limit_hits",
    "board_limit_hit_rate",
    "board_main",
    "board_star",
    "board_chinext",
    "board_bse",
    "is_st",
    "close_limit_hit",
    "failed_limit_hit",
    "limit_hit_prev_1",
    "close_limit_hit_prev_1",
    "limit_hits_5",
    "limit_hits_10",
    "failed_limit_hits_5",
    "consecutive_limit_hits",
    "curve_slope_5",
    "curve_accel_5",
    "curve_max_drawdown_5",
    "curve_close_position_5",
    "volume_trend_5",
    "amount_trend_5",
    "range_mean_5",
    "close_to_high_mean_5",
    "limit_gap_mean_5",
    "limit_gap_min_5",
    "pct_chg_lag_1",
    "pct_chg_lag_2",
    "pct_chg_lag_3",
    "pct_chg_lag_4",
    "pct_chg_lag_5",
    "intraday_ret_lag_1",
    "intraday_ret_lag_2",
    "intraday_ret_lag_3",
    "intraday_ret_lag_4",
    "intraday_ret_lag_5",
    "range_lag_1",
    "range_lag_2",
    "range_lag_3",
    "range_lag_4",
    "range_lag_5",
    "close_to_high_lag_1",
    "close_to_high_lag_2",
    "close_to_high_lag_3",
    "close_to_high_lag_4",
    "close_to_high_lag_5",
    "volume_ratio_lag_1",
    "volume_ratio_lag_2",
    "volume_ratio_lag_3",
    "volume_ratio_lag_4",
    "volume_ratio_lag_5",
    "limit_gap_lag_1",
    "limit_gap_lag_2",
    "limit_gap_lag_3",
    "limit_gap_lag_4",
    "limit_gap_lag_5",
    "limit_hit_lag_1",
    "limit_hit_lag_2",
    "limit_hit_lag_3",
    "limit_hit_lag_4",
    "limit_hit_lag_5",
    "failed_limit_hit_lag_1",
    "failed_limit_hit_lag_2",
    "failed_limit_hit_lag_3",
    "failed_limit_hit_lag_4",
    "failed_limit_hit_lag_5",
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
