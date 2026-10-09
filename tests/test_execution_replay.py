import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_limitup.backtest import run_backtest
from quant_limitup import cli
from quant_limitup.config import TradingConfig
from quant_limitup.execution import ExecutionQuotes, prepare_execution_quotes
from quant_limitup.paper import PaperAccount, Position, _sell_decision, load_account, run_paper_day, save_account
from quant_limitup.strategy import optimize_threshold


BUY = pd.Timestamp("2026-09-21")
SELL = pd.Timestamp("2026-09-22")


class ConstantModel:
    feature_columns = ["ret_3"]

    def predict_proba(self, frame):
        return np.full(len(frame), 0.9)


def features(symbols=1, weak=False):
    rows = []
    for date in (BUY, SELL):
        for i in range(symbols):
            rows.append({
                "date": date, "symbol": f"60000{i}.SH", "name": f"公司{i}", "board": "main",
                "close": 10.0 if date == BUY else 9.8, "prev_close": 10.0,
                "open": 10.0, "high": 11.0, "low": 9.8, "limit_up_price": 11.0,
                "next_high": 11.0 if date == BUY else np.nan,
                "next_close": 9.8 if date == BUY else np.nan,
                "next_limit_up_price": 11.0 if date == BUY else np.nan,
                "target_end_date": SELL if date == BUY else pd.NaT,
                "target_limit_up_next": 1 if date == BUY else np.nan,
                "volume_ratio_5": 1.0, "ret_3": 0.0, "turnover": 0.01,
                "market_symbol_count": 5500, "market_limit_hits": 50 if weak else 200,
                "market_failed_limit_hits": 30 if weak else 20,
                "market_ret": -0.012 if weak else 0.01,
                "market_limit_hit_rate": 0.01 if weak else 0.04,
                "market_data_eligible": True,
            })
    return pd.DataFrame(rows)


def minutes(symbols=1, morning_hit=False):
    rows = []
    times = list(pd.date_range(SELL + pd.Timedelta(hours=9, minutes=35), SELL + pd.Timedelta(hours=10, minutes=30), freq="5min"))
    times += [SELL + pd.Timedelta(hours=14, minutes=30), SELL + pd.Timedelta(hours=14, minutes=50), SELL + pd.Timedelta(hours=15)]
    for i in range(symbols):
        for stamp in times:
            close = 10.2 if stamp.hour < 14 else 9.8
            high = 11.0 if morning_hit and stamp.strftime("%H:%M") == "09:45" else max(10.3, close)
            rows.append({"symbol": f"60000{i}.SH", "datetime": stamp, "open": 10.0,
                         "high": high, "low": min(close, 10.0), "close": close, "volume": 100.0})
    return pd.DataFrame(rows)


class ExecutionReplayTest(unittest.TestCase):
    def setUp(self):
        self.pos = Position("600000.SH", "公司0", "main", str(BUY.date()), 100, 10, 1005, 0.9)
        self.row = features().iloc[-1]

    def test_force_uses_1430_close_even_if_limit_was_hit_earlier_or_after_cutoff(self):
        bars = minutes(morning_hit=True)
        late = bars.datetime.dt.strftime("%H:%M") == "15:00"
        bars.loc[late, ["high", "close"]] = 20.0
        price, hit, reason = _sell_decision(self.pos, self.row, SELL, bars, "force", TradingConfig())
        self.assertEqual(price, 9.8)
        self.assertFalse(hit)
        self.assertEqual(reason, "force_sell")

    def test_morning_touch_is_executed_at_1030_not_historical_limit(self):
        price, hit, _ = _sell_decision(self.pos, self.row, SELL, minutes(morning_hit=True), "morning", TradingConfig())
        self.assertEqual(price, 10.2)
        self.assertFalse(hit)

    def test_morning_pullback_uses_current_quote_and_current_loss(self):
        bars = minutes()
        bars.loc[bars.datetime.dt.strftime("%H:%M") == "09:40", "low"] = 9.7
        bars.loc[bars.datetime.dt.strftime("%H:%M") == "09:40", "close"] = 9.7
        large = replace(self.pos, shares=1000, cost=10005)
        # The earlier dip recovers by the scheduled decision time.
        self.assertIsNone(_sell_decision(large, self.row, SELL, bars, "morning", TradingConfig()))
        bars.loc[bars.datetime.dt.strftime("%H:%M") == "10:30", ["low", "close"]] = 9.7
        decision = _sell_decision(large, self.row, SELL, bars, "morning", TradingConfig())
        self.assertEqual(decision[0], 9.7)

    def test_missing_invalid_or_duplicate_cutoff_never_uses_future_or_daily_prices(self):
        base = minutes()
        cutoff = base.datetime.dt.strftime("%H:%M") == "14:30"
        invalid = base.copy()
        invalid.loc[cutoff, "close"] = np.nan
        zero_volume = base.copy()
        zero_volume.loc[cutoff, "volume"] = 0
        late_fraction = base.copy()
        late_fraction.loc[cutoff, "datetime"] += pd.Timedelta(milliseconds=500)
        for bars in [base[~cutoff], invalid, zero_volume, late_fraction, pd.concat([base, base[cutoff]])]:
            with self.subTest(rows=len(bars)):
                self.assertIsNone(_sell_decision(self.pos, self.row, SELL, bars, "force", TradingConfig()))
        self.assertIsNone(_sell_decision(self.pos, self.row, SELL, None, "force", TradingConfig()))
        incomplete = base[base.datetime.dt.strftime("%H:%M") != "09:45"]
        self.assertFalse(prepare_execution_quotes(incomplete).get(self.pos.symbol, SELL)["morning_valid"])

    def test_missing_force_quote_leaves_account_retryable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = dict(state_file=root / "account.json", trades_file=root / "trades.csv", daily_file=root / "daily.csv")
            save_account(PaperAccount(cash=8995, positions=[self.pos]), args["state_file"])
            bars = minutes()
            partial = bars[bars.datetime.dt.strftime("%H:%M") != "14:30"]
            account, _, first = run_paper_day(features(), ConstantModel(), TradingConfig(),
                date=str(SELL.date()), minute_bars=partial, sell_mode="force", open_new=False, **args)
            self.assertEqual(first["closed_trades"], 0)
            self.assertIsNone(account.last_sell_date)
            account, _, second = run_paper_day(features(), ConstantModel(), TradingConfig(),
                date=str(SELL.date()), minute_bars=bars, sell_mode="force", open_new=False, **args)
            self.assertEqual(second["closed_trades"], 1)
            self.assertFalse(account.positions)

    def test_minute_replay_matches_paper_cash_trades_fees_and_weak_market_limits(self):
        for weak in (False, True):
            for hit in (False, True):
                with self.subTest(weak=weak, hit=hit), TemporaryDirectory() as directory:
                    frame, bars, cfg = features(4, weak), minutes(4, hit), TradingConfig()
                    trades, summary = run_backtest(frame, ConstantModel(), cfg, minute_bars=bars)
                    self.assertTrue(summary["execution_complete"])
                    self.assertEqual(len(trades), 1 if weak else 3)
                    root = Path(directory)
                    args = dict(state_file=root / "account.json", trades_file=root / "trades.csv", daily_file=root / "daily.csv")
                    run_paper_day(frame, ConstantModel(), cfg, date=str(BUY.date()), settle=False, **args)
                    for mode in ("morning", "force"):
                        run_paper_day(frame, ConstantModel(), cfg, date=str(SELL.date()),
                            minute_bars=bars, sell_mode=mode, open_new=False, **args)
                    account = load_account(args["state_file"], cfg.initial_cash)
                    self.assertAlmostEqual(summary["final_equity"], account.cash)
                    sales = pd.read_csv(args["trades_file"])
                    sales = sales[sales.action == "SELL"].reset_index(drop=True)
                    np.testing.assert_allclose(trades.sell_price, sales.price)
                    np.testing.assert_allclose(trades.pnl, sales.pnl)
                    self.assertEqual(trades.reason.tolist(), sales.reason.tolist())

    def test_daily_only_replay_is_explicit_close_proxy_and_never_sells_at_high(self):
        trades, summary = run_backtest(features(), ConstantModel())
        self.assertEqual(summary["execution_mode"], "daily_close_proxy")
        self.assertFalse(summary["execution_complete"])
        self.assertAlmostEqual(trades.iloc[0].sell_price, 9.8 * (1 - TradingConfig().sell_slippage_bps / 10000))

    def test_incomplete_intraday_replay_keeps_position_instead_of_daily_fallback(self):
        bars = minutes()
        bars = bars[bars.datetime.dt.strftime("%H:%M") != "14:30"]
        trades, summary = run_backtest(features(), ConstantModel(), minute_bars=bars)
        self.assertTrue(trades.empty)
        self.assertEqual(summary["open_positions"], 1)
        self.assertFalse(summary["execution_complete"])

    def test_buy_coverage_guard_is_shared_by_replay(self):
        frame = features()
        frame["market_data_eligible"] = False
        trades, summary = run_backtest(frame, ConstantModel(), minute_bars=minutes())
        self.assertTrue(trades.empty)
        self.assertEqual(summary["final_equity"], TradingConfig().initial_cash)

    def test_backtest_cli_requires_explicit_matching_minute_file(self):
        with patch.object(cli, "ensure_dirs"), patch.object(cli, "backtest_to_files") as replay:
            cli.main(["backtest"])
            self.assertIsNone(replay.call_args.args[2])
            cli.main(["backtest", "--minute-bars-file", "/tmp/replay-fixture.csv"])
            self.assertEqual(replay.call_args.args[2], Path("/tmp/replay-fixture.csv"))
        _, summary = run_backtest(features(), ConstantModel(), minute_bars=pd.DataFrame())
        self.assertEqual(summary["execution_mode"], "scheduled_minutes")
        self.assertFalse(summary["execution_complete"])
        self.assertEqual(summary["open_positions"], 1)

    def test_future_exit_proceeds_cannot_fund_an_earlier_buy(self):
        frame = features().iloc[[0, 0]].copy().reset_index(drop=True)
        frame.loc[1, "date"] = SELL
        frame.loc[1, "symbol"] = "600001.SH"
        frame["target_end_date"] = pd.Timestamp("2026-09-23")
        frame["next_close"] = 11.0
        cfg = TradingConfig(initial_cash=1000, max_position_pct=1, buy_slippage_bps=0, sell_slippage_bps=0,
                            commission_bps=0, min_commission=0, stamp_tax_bps=0)
        trades, _ = run_backtest(frame, ConstantModel(), cfg)
        self.assertEqual(trades.symbol.tolist(), ["600000.SH"])
        self.assertEqual(trades.iloc[0].sell_date, pd.Timestamp("2026-09-23"))

    def test_optimizer_does_not_calibrate_from_daily_proxy_or_old_cache(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "optimization.json"
            path.write_text(json.dumps({"version": 4, "best_threshold": 0.9}))
            with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=AssertionError("no execution history")):
                threshold, result = optimize_threshold(features(), ConstantModel(), TradingConfig(), path)
            self.assertEqual(threshold, TradingConfig().min_score_to_buy)
            self.assertEqual(result["reason"], "missing_execution_history")
            self.assertFalse(result["cached"])

    def test_optimizer_rejects_missing_candidate_quotes_even_if_other_symbols_have_history(self):
        dates = pd.bdate_range("2026-01-02", periods=81)
        rows = []
        quotes = ExecutionQuotes()
        for i, date in enumerate(dates):
            row = features().iloc[0].to_dict()
            row.update(date=date, target_end_date=dates[i + 1] if i < 80 else pd.NaT,
                       target_limit_up_next=i % 2 if i < 80 else np.nan)
            rows.append(row)
            # Having an archive for unrelated stocks cannot price our picks.
            quotes.rows[("600999.SH", date)] = {"morning_valid": True, "force_valid": True}
        frame = pd.DataFrame(rows)
        def scored(history, columns):
            recent = sorted(history.date.unique())[-60:]
            return history[history.date.isin(recent)].assign(score=0.9), []
        with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=scored):
            threshold, result = optimize_threshold(frame, ConstantModel(), TradingConfig(), minute_bars=quotes)
        self.assertEqual(threshold, TradingConfig().min_score_to_buy)
        self.assertEqual(result["reason"], "incomplete_selection_execution")


if __name__ == "__main__":
    unittest.main()
