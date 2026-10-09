import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from quant_limitup import cli
from quant_limitup.config import TradingConfig
from quant_limitup.paper import PaperAccount, Position, load_account, save_account


TODAY = pd.Timestamp.now().normalize()
PREVIOUS = TODAY - pd.Timedelta(days=1)
HELD = ["000001.SZ", "600001.SH"]


def prices_for(symbols, *, history=True):
    rows = []
    for date in ([PREVIOUS, TODAY] if history else [TODAY]):
        for symbol in symbols:
            rows.append({
                "date": str(date.date()), "symbol": symbol, "name": "测试公司", "board": "main", "is_st": 0,
                "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
                "volume": 1000.0, "amount": 10000.0, "turnover": 1.0, "free_float_mkt_cap": 1e9,
            })
    return pd.DataFrame(rows)


class FullCoverageSettlementTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "daily.csv"
        self.pool = self.root / "pool.csv"
        self.state = self.root / "account.json"
        self.trades = self.root / "trades.csv"
        self.daily = self.root / "paper_daily.csv"
        self.reports = self.root / "reports"
        self.model = self.root / "model.json"
        self.raw.write_bytes(b"last accepted source snapshot\n")
        self.model.write_bytes(b"last trained model\n")
        pd.DataFrame({"symbol": [*HELD, "000002.SZ"], "name": ["测试公司"] * 3}).to_csv(self.pool, index=False)
        positions = [Position(symbol, "测试公司", "main", str(PREVIOUS.date()), 100, 10.0, 1000.0, 0.9) for symbol in HELD]
        save_account(PaperAccount(cash=8000.0, positions=positions), self.state)
        self.original_account = self.state.read_bytes()

    @contextlib.contextmanager
    def isolated(self, primary, fallback=None, *, phase="full"):
        def fetch_primary(path, symbols_file, days):
            self.assertNotEqual(path, self.raw)
            primary.to_csv(path, index=False)
            return primary

        def fetch_fallback(path, symbols_file, days):
            if fallback is None:
                raise RuntimeError("fixture source outage")
            fallback.to_csv(path, index=False)
            return fallback

        with contextlib.ExitStack() as stack:
            for name, value in {
                "DEFAULT_PAPER_STATE": self.state, "DEFAULT_PAPER_TRADES": self.trades,
                "DEFAULT_PAPER_DAILY": self.daily, "REPORT_DIR": self.reports,
                "DEFAULT_MODEL_FILE": self.model,
            }.items():
                stack.enter_context(patch.object(cli, name, value))
            stack.enter_context(patch.object(cli, "ensure_dirs"))
            stack.enter_context(patch.object(cli, "fetch_tencent_daily_prices", side_effect=fetch_primary))
            stack.enter_context(patch.object(cli, "fetch_sina_daily_prices", side_effect=fetch_fallback))
            for name in ("fetch_sina_minute_bars", "build_dataset", "train_model", "initialize_factor_params", "optimize_threshold", "safe_notify"):
                stack.enter_context(patch.object(cli, name, side_effect=AssertionError(f"unexpected {name}")))
            if phase == "buy":
                stack.enter_context(patch.object(cli, "load_account", side_effect=AssertionError("buy must not access account")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def run_daily(self, primary, *, phase="full", fallback=None):
        with self.isolated(primary, fallback, phase=phase):
            cli.main([
                "daily", "--provider", "sina", "--paper", "--phase", phase,
                "--prices", str(self.raw), "--symbols", str(self.pool), "--use-minute",
            ])
        self.assertEqual(self.raw.read_bytes(), b"last accepted source snapshot\n")
        self.assertEqual(self.model.read_bytes(), b"last trained model\n")
        return json.loads((self.reports / "paper_daily_summary.json").read_text())

    def test_full_rejected_market_still_settles_complete_held_quotes_without_new_buys(self):
        summary = self.run_daily(prices_for(HELD))
        account = load_account(self.state, 10000.0)
        self.assertFalse(account.positions)
        self.assertEqual(account.last_sell_date, str(TODAY.date()))
        self.assertEqual(summary["opened_positions"], 0)
        self.assertEqual(summary["closed_trades"], 2)
        self.assertEqual(summary["skipped"], "buy_data_coverage")
        self.assertEqual(summary["data_quality"]["settlement_price_provider"], "tencent")
        trades = pd.read_csv(self.trades)
        self.assertEqual(trades.action.tolist(), ["SELL", "SELL"])
        self.assertEqual(set(trades.symbol), set(HELD))

    def test_missing_held_quote_preserves_account_and_same_day_retry_can_settle(self):
        summary = self.run_daily(prices_for(HELD[:1]))
        self.assertEqual(summary["settlement_skipped"], "missing_valid_current_prices")
        self.assertEqual(summary["settlement_missing_symbols"], HELD[1:])
        self.assertEqual(self.state.read_bytes(), self.original_account)
        self.assertFalse(self.trades.exists())
        self.assertFalse(self.daily.exists())
        summary = self.run_daily(prices_for(HELD))
        self.assertEqual(summary["closed_trades"], 2)

    def test_buy_rejected_market_does_not_read_or_write_account(self):
        summary = self.run_daily(prices_for(HELD), phase="buy")
        self.assertEqual(summary["opened_positions"], 0)
        self.assertEqual(self.state.read_bytes(), self.original_account)
        self.assertFalse(self.trades.exists())
        self.assertFalse(self.daily.exists())

    def test_invalid_current_ohlc_does_not_settle_or_lock_day(self):
        prices = prices_for(HELD)
        prices.loc[(prices.symbol == HELD[1]) & (prices.date == str(TODAY.date())), "high"] = 1.0
        summary = self.run_daily(prices)
        self.assertEqual(summary["settlement_missing_symbols"], HELD[1:])
        self.assertEqual(self.state.read_bytes(), self.original_account)

    def test_rejected_sources_are_not_spliced_to_complete_held_quotes(self):
        summary = self.run_daily(prices_for(HELD[:1]), fallback=prices_for(HELD[1:]))
        self.assertEqual(summary["settlement_skipped"], "missing_valid_current_prices")
        self.assertEqual(self.state.read_bytes(), self.original_account)

    def test_missing_previous_close_leaves_account_untouched(self):
        summary = self.run_daily(prices_for(HELD, history=False))
        self.assertEqual(summary["settlement_skipped"], "missing_previous_close")
        self.assertEqual(self.state.read_bytes(), self.original_account)

    def test_end_of_day_uses_close_even_when_st_limit_was_touched(self):
        prices = prices_for(HELD)
        prices.loc[(prices.symbol == HELD[1]) & (prices.date == str(TODAY.date())), "is_st"] = 1
        summary = self.run_daily(prices)
        self.assertEqual(summary["closed_trades"], 2)
        trades = pd.read_csv(self.trades).set_index("symbol")
        self.assertAlmostEqual(
            trades.loc[HELD[1], "price"],
            10.0 * (1 - TradingConfig().sell_slippage_bps / 10000),
        )

    def test_direct_full_uses_same_settlement_guard(self):
        prices_for(HELD).to_csv(self.raw, index=False)
        with self.isolated(prices_for(HELD)):
            _, summary = cli.run_paper_from_prices(
                self.raw, None, False, self.root / "email.json", False, self.root / "webhook.txt",
                phase="full", symbols_file=self.pool, expected_date=str(TODAY.date()),
            )
        self.assertEqual(summary["closed_trades"], 2)
        self.assertEqual(summary["opened_positions"], 0)


if __name__ == "__main__":
    unittest.main()
