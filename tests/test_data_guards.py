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
from quant_limitup.intraday_features import build_tail_features
from quant_limitup.minute_history import minute_history_path, read_minute_history, update_minute_history
from quant_limitup.model import LogisticLimitUpModel
from quant_limitup.paper import PaperAccount, Position, run_paper_day
from quant_limitup.providers import fetch_sina_minute_bars, fetch_sina_spot_prices, fetch_tencent_spot_prices


EXPECTED_DATE = "2026-09-22"


def _pool():
    return pd.DataFrame([
        {"symbol": "000001.SZ", "name": "深市公司", "board": "main", "is_st": 0},
        {"symbol": "600001.SH", "name": "沪市公司", "board": "main", "is_st": 0},
    ])


def _daily(symbols, date=EXPECTED_DATE, price=10.0):
    return pd.DataFrame([
        {
            "symbol": symbol, "date": date, "name": "测试公司", "board": "main", "is_st": 0,
            "open": price, "high": price + 1, "low": price - 1, "close": price,
            "volume": 1000.0, "amount": price * 1000, "turnover": 1.0, "free_float_mkt_cap": 1e9,
        }
        for symbol in symbols
    ])


def _minutes(date, close=10.0, volume=100.0):
    return pd.DataFrame([
        {
            "datetime": f"{date} 14:55:00", "symbol": "000001.SZ",
            "open": 10.0, "high": 11.0, "low": 9.0,
            "close": close, "volume": volume, "amount": close * volume,
        }
    ])


class BuyDataGuardTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prices_file = self.root / "daily_prices.csv"
        self.pool_file = self.root / "stock_pool.csv"
        self.reports = self.root / "reports"
        _pool().to_csv(self.pool_file, index=False)

    def test_partial_primary_uses_complete_fallback_without_splicing_sources(self):
        accepted = _daily(_pool().symbol, date="2026-09-21", price=9)
        accepted.to_csv(self.prices_file, index=False)
        primary = _daily(["000001.SZ"], price=30)
        fallback = pd.concat([
            _daily(_pool().symbol, date="2026-09-21", price=19),
            _daily(_pool().symbol, price=20),
        ], ignore_index=True)
        staged_paths = []

        def fetch_primary(path, symbols_file, days):
            staged_paths.append(path)
            self.assertEqual(symbols_file, self.pool_file)
            self.assertEqual(days, 60)
            primary.to_csv(path, index=False)
            return primary

        def fetch_fallback(path, symbols_file, days):
            staged_paths.append(path)
            # The primary's incomplete response has not replaced accepted data.
            pd.testing.assert_frame_equal(pd.read_csv(self.prices_file), accepted)
            fallback.to_csv(path, index=False)
            return fallback

        with patch.object(cli, "REPORT_DIR", self.reports), \
             patch.object(cli, "fetch_tencent_daily_prices", side_effect=fetch_primary) as first, \
             patch.object(cli, "fetch_sina_daily_prices", side_effect=fetch_fallback) as second:
            result = cli.fetch_validated_buy_prices(
                self.prices_file, self.pool_file, days=60, expected_date=EXPECTED_DATE
            )
        first.assert_called_once()
        second.assert_called_once()
        self.assertTrue(all(path != self.prices_file for path in staged_paths))
        pd.testing.assert_frame_equal(result, fallback)
        pd.testing.assert_frame_equal(pd.read_csv(self.prices_file), fallback)
        report = json.loads((self.reports / "data_quality.json").read_text())
        self.assertEqual(report["provider"], "sina")
        self.assertEqual([attempt["eligible"] for attempt in report["attempts"]], [False, True])

    def test_preclose_sina_spot_uses_fresh_dated_quotes_and_accepted_history(self):
        accepted = _daily(_pool().symbol, date="2026-09-21", price=9)
        accepted.to_csv(self.prices_file, index=False)
        spot = _daily(_pool().symbol, price=10)
        spot["free_float_mkt_cap"] = 0.0
        spot["quote_prev_close"] = 9.0
        with patch.object(cli, "REPORT_DIR", self.reports), \
             patch.object(cli, "fetch_sina_spot_prices", return_value=spot), \
             patch.object(cli, "fetch_tencent_daily_prices", side_effect=AssertionError("slow fallback")), \
             patch.object(cli, "fetch_sina_daily_prices", side_effect=AssertionError("slow fallback")):
            result = cli.fetch_validated_buy_prices(
                self.prices_file, self.pool_file, expected_date=EXPECTED_DATE, prefer_live=True
            )
        self.assertEqual(str(pd.to_datetime(result.date).max().date()), EXPECTED_DATE)
        self.assertEqual(len(result), 4)
        self.assertTrue((result[result.date == pd.Timestamp(EXPECTED_DATE)]["free_float_mkt_cap"] == 1e9).all())
        report = json.loads((self.reports / "data_quality.json").read_text())
        self.assertEqual(report["provider"], "sina_spot")
        self.assertTrue(report["eligible"])

    def test_sina_spot_rejects_stale_quote_timestamp(self):
        def line(code, date, time):
            fields = [""] * 32
            fields[0], fields[1], fields[2], fields[3] = "测试", "10", "9", "10"
            fields[4], fields[5], fields[8], fields[9] = "11", "9", "1000", "10000"
            fields[30], fields[31] = date, time
            return f'var hq_str_{code}="{",".join(fields)}";'

        response = type("Response", (), {"encoding": None, "text": "\n".join([
            line("sz000001", EXPECTED_DATE, "14:49:30"),
            line("sh600001", EXPECTED_DATE, "10:30:00"),
        ]), "raise_for_status": lambda self: None})()
        with patch("quant_limitup.providers.requests.get", return_value=response):
            result = fetch_sina_spot_prices(
                self.pool_file, expected_date=EXPECTED_DATE,
                observed_at=pd.Timestamp(f"{EXPECTED_DATE} 14:50:00"),
            )
        self.assertEqual(result.symbol.tolist(), ["000001.SZ"])

    def test_sina_close_accepts_settlement_timestamp_but_not_during_buy_window(self):
        fields = [""] * 32
        fields[0], fields[1], fields[2], fields[3] = "测试", "10", "9", "10"
        fields[4], fields[5], fields[8], fields[9] = "11", "9", "1000", "10000"
        fields[30], fields[31] = EXPECTED_DATE, "15:00:01"
        response = type("Response", (), {
            "encoding": None,
            "text": f'var hq_str_sh600001="{",".join(fields)}";',
            "raise_for_status": lambda self: None,
        })()
        with patch("quant_limitup.providers.requests.get", return_value=response):
            result = fetch_sina_spot_prices(
                self.pool_file, expected_date=EXPECTED_DATE,
                observed_at=pd.Timestamp(f"{EXPECTED_DATE} 16:30:00"),
                accept_closed_session=True,
            )
        self.assertEqual(result.symbol.tolist(), ["600001.SH"])
        with self.assertRaises(ValueError):
            fetch_sina_spot_prices(
                self.pool_file, expected_date=EXPECTED_DATE,
                observed_at=pd.Timestamp(f"{EXPECTED_DATE} 14:50:00"),
                accept_closed_session=True,
            )

    def test_tencent_spot_checks_time_and_infers_volume_units(self):
        pool = pd.concat([_pool(), pd.DataFrame([{
            "symbol": "688001.SH", "name": "科创公司", "board": "star", "is_st": 0,
        }])], ignore_index=True)
        pool.to_csv(self.pool_file, index=False)

        def line(code, volume, amount, stamp):
            fields = [""] * 38
            fields[3], fields[4], fields[5] = "10", "9", "10"
            fields[6], fields[30], fields[33], fields[34] = str(volume), stamp, "11", "9"
            fields[35] = f"10/{volume}/{amount}"
            return f'v_{code}="{"~".join(fields)}";'

        response = type("Response", (), {
            "encoding": None,
            "text": "\n".join([
                line("sz000001", 100, 100000, "20260922144930"),
                line("sh688001", 10000, 100000, "20260922144930"),
                line("sh600001", 100, 100000, "20260922143000"),
            ]),
            "raise_for_status": lambda self: None,
        })()
        with patch("quant_limitup.providers.requests.get", return_value=response):
            result = fetch_tencent_spot_prices(
                self.pool_file, expected_date=EXPECTED_DATE,
                observed_at=pd.Timestamp(f"{EXPECTED_DATE} 14:50:00"),
            )
        self.assertEqual(set(result.symbol), {"000001.SZ", "688001.SH"})
        self.assertEqual(result.set_index("symbol").volume.to_dict(), {
            "000001.SZ": 10000.0, "688001.SH": 10000.0,
        })

    def test_preclose_buy_uses_whole_tencent_spot_when_sina_is_partial(self):
        accepted = _daily(_pool().symbol, date="2026-09-21", price=9)
        accepted.to_csv(self.prices_file, index=False)
        partial = _daily(["000001.SZ"]).assign(quote_prev_close=9.0)
        complete = _daily(_pool().symbol).assign(quote_prev_close=9.0)
        with patch.object(cli, "REPORT_DIR", self.reports), \
             patch.object(cli, "fetch_sina_spot_prices", return_value=partial), \
             patch.object(cli, "fetch_tencent_spot_prices", return_value=complete):
            result = cli.fetch_validated_buy_prices(
                self.prices_file, self.pool_file, expected_date=EXPECTED_DATE, prefer_live=True
            )
        self.assertEqual(len(result), 4)
        report = json.loads((self.reports / "data_quality.json").read_text())
        self.assertEqual(report["provider"], "tencent_spot")
        self.assertEqual([item["eligible"] for item in report["attempts"]], [False, True])

    def test_preclose_buy_fails_promptly_when_live_coverage_is_low(self):
        accepted = _daily(_pool().symbol, date="2026-09-21", price=9)
        accepted.to_csv(self.prices_file, index=False)
        before = self.prices_file.read_bytes()
        with patch.object(cli, "REPORT_DIR", self.reports), \
             patch.object(cli, "fetch_sina_spot_prices", return_value=_daily(["000001.SZ"]).assign(quote_prev_close=9.0)), \
             patch.object(cli, "fetch_tencent_spot_prices", return_value=_daily(["000001.SZ"]).assign(quote_prev_close=9.0)), \
             patch.object(cli, "fetch_tencent_daily_prices", side_effect=AssertionError("slow fallback")), \
             patch.object(cli, "fetch_sina_daily_prices", side_effect=AssertionError("slow fallback")), \
             self.assertRaises(cli.BuyCoverageError) as raised:
            cli.fetch_validated_buy_prices(
                self.prices_file, self.pool_file, expected_date=EXPECTED_DATE, prefer_live=True
            )
        self.assertEqual(self.prices_file.read_bytes(), before)
        self.assertEqual(raised.exception.report["attempts"][0]["provider"], "sina_spot")
        self.assertIn("total_coverage_below_threshold", raised.exception.report["reasons"])

    def test_preclose_buy_rejects_quote_with_mismatched_history_close(self):
        accepted = _daily(_pool().symbol, date="2026-09-21", price=9)
        accepted.to_csv(self.prices_file, index=False)
        spot = _daily(_pool().symbol, price=10).assign(quote_prev_close=[9.0, 7.0])
        with patch.object(cli, "REPORT_DIR", self.reports), \
             patch.object(cli, "fetch_sina_spot_prices", return_value=spot), \
             patch.object(cli, "fetch_tencent_spot_prices", return_value=spot), \
             self.assertRaises(cli.BuyCoverageError) as raised:
            cli.fetch_validated_buy_prices(
                self.prices_file, self.pool_file, expected_date=EXPECTED_DATE, prefer_live=True
            )
        self.assertEqual(raised.exception.report["covered_symbols"], 1)

    def test_preclose_buy_does_not_train_on_unfinished_today_outcome(self):
        frame = pd.DataFrame([
            {"date": pd.Timestamp("2026-09-21"), "target_end_date": pd.Timestamp("2026-09-22"),
             "next_open": 11.0, "next_high": 12.0, "next_close": 11.5,
             "next_limit_up_price": 11.0, "target_limit_up_next": 1,
             "target_close_limit_up_next": 1},
            {"date": pd.Timestamp("2026-09-22"), "target_end_date": pd.Timestamp("2026-09-23"),
             "next_open": 10.0, "next_high": 10.0, "next_close": 10.0,
             "next_limit_up_price": 11.0, "target_limit_up_next": 0,
             "target_close_limit_up_next": 0},
            {"date": pd.Timestamp("2026-09-23"), "target_end_date": pd.NaT,
             "next_open": None, "next_high": None, "next_close": None,
             "next_limit_up_price": None, "target_limit_up_next": None,
             "target_close_limit_up_next": None},
        ])
        masked = cli.mask_unfinished_buy_outcomes(frame)
        self.assertEqual(masked.loc[0, "target_limit_up_next"], 1)
        self.assertTrue(pd.isna(masked.loc[1, "target_limit_up_next"]))
        self.assertTrue(pd.isna(masked.loc[1, "next_close"]))
        self.assertEqual(masked.loc[1, "date"], pd.Timestamp("2026-09-22"))

    def test_both_sources_fail_without_replacing_accepted_prices_model_or_account(self):
        protected = {
            self.prices_file: b"accepted daily history\n",
            self.root / "model.json": b'{"version": "accepted-model"}',
            self.root / "account.json": b'{"cash": 1234, "positions": []}',
            self.prices_file.with_suffix(".failures.csv"): b"previous failure report\n",
        }
        for path, content in protected.items():
            path.write_bytes(content)

        def partial_fetch(path, symbols_file, days):
            response = _daily(["000001.SZ"])
            response.to_csv(path, index=False)
            return response

        def failed_fetch(path, symbols_file, days):
            path.write_text("incomplete staged response")
            raise RuntimeError("fixture outage")

        with patch.object(cli, "REPORT_DIR", self.reports), \
             patch.object(cli, "fetch_tencent_daily_prices", side_effect=partial_fetch), \
             patch.object(cli, "fetch_sina_daily_prices", side_effect=failed_fetch), \
             self.assertRaises(cli.BuyCoverageError) as raised:
            cli.fetch_validated_buy_prices(self.prices_file, self.pool_file, expected_date=EXPECTED_DATE)
        self.assertEqual(raised.exception.report["status"], "blocked")
        self.assertEqual(len(raised.exception.report["attempts"]), 2)
        for path, content in protected.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertFalse(list(self.root.glob("daily-fetch-*")))

    def test_buy_phase_blocks_partial_data_before_feature_build_training_or_account_changes(self):
        _daily(["000001.SZ"]).to_csv(self.prices_file, index=False)
        original = self.prices_file.read_bytes()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(cli, "REPORT_DIR", self.reports))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            prohibited = {
                name: stack.enter_context(patch.object(cli, name, side_effect=AssertionError(f"unexpected {name}")))
                for name in (
                    "build_dataset", "build_feature_frame", "train_model", "initialize_factor_params",
                    "optimize_threshold", "run_paper_day", "safe_notify",
                )
            }
            rank, summary = cli.run_paper_from_prices(
                self.prices_file, None, False, self.root / "email.json", False, self.root / "webhook.txt",
                phase="buy", symbols_file=self.pool_file, expected_date=EXPECTED_DATE,
            )
        self.assertTrue(rank.empty)
        self.assertEqual(summary["skipped"], "buy_data_coverage")
        self.assertEqual(summary["opened_positions"], 0)
        self.assertEqual(summary["date"], EXPECTED_DATE)
        for mocked in prohibited.values():
            mocked.assert_not_called()
        self.assertEqual(self.prices_file.read_bytes(), original)
        self.assertEqual(json.loads((self.reports / "data_quality.json").read_text())["status"], "blocked")

    def test_explicit_ineligible_market_prevents_paper_buy_even_with_high_score(self):
        frame = _daily(["000001.SZ"])
        frame["date"] = pd.to_datetime(frame["date"])
        rank = frame.assign(
            score=0.99, market_data_eligible=False, market_coverage_ratio=0.5,
            market_coverage_reason="SH_coverage_below_threshold", market_symbol_count=5000,
            market_limit_hit_rate=0.05, market_ret=0.02,
        )
        with patch("quant_limitup.paper.rank_candidates", return_value=rank):
            account, _, summary = run_paper_day(
                frame, LogisticLimitUpModel([], [], -35.0, [], []), TradingConfig(),
                state_file=self.root / "account.json", trades_file=self.root / "trades.csv",
                daily_file=self.root / "daily.csv", date=EXPECTED_DATE, settle=False, open_new=True,
            )
        self.assertEqual(account.cash, 10000.0)
        self.assertFalse(account.positions)
        self.assertEqual(summary["opened_positions"], 0)
        self.assertEqual(summary["actual_max_positions"], 0)
        self.assertEqual(summary["market_regime"], "insufficient")
        self.assertFalse((self.root / "trades.csv").exists())

    def test_training_excludes_incomplete_signal_dates_and_label_end_dates(self):
        dates = pd.bdate_range("2026-09-14", periods=6)
        frame = pd.DataFrame({
            "date": dates,
            "target_end_date": list(dates[1:]) + [pd.NaT],
            "target_limit_up_next": [0, 1, 0, 1, 0, None],
            "market_data_eligible": [True, True, False, True, True, True],
        })
        selected = cli.eligible_training_rows(frame)
        self.assertEqual(selected.index.tolist(), [0, 3, 4])
        # A signal before the cutoff is still unavailable if its label has not
        # ended yet. The cutoff is an information boundary, not just row date.
        selected = cli.eligible_training_rows(frame, train_end=str(dates[4].date()))
        self.assertEqual(selected.index.tolist(), [0, 3])

    def test_full_phase_with_partial_current_prices_still_settles_without_new_buys(self):
        pd.concat([
            _daily(["000001.SZ"], date="2026-09-21"), _daily(["000001.SZ"]),
        ], ignore_index=True).to_csv(self.prices_file, index=False)
        account = PaperAccount(positions=[Position(
            symbol="000001.SZ", name="测试公司", board="main", buy_date="2026-09-21",
            shares=100, buy_price=10.0, cost=1005.0, score=0.8,
        )])
        with patch.object(cli, "REPORT_DIR", self.reports), \
             patch.object(cli, "load_account", return_value=account), \
             patch.object(cli, "train_model", side_effect=AssertionError("must not train")), \
             patch.object(cli, "run_paper_day", return_value=(None, None, {"closed_trades": 1})) as settle, \
             contextlib.redirect_stdout(io.StringIO()):
            _, summary = cli.run_paper_from_prices(
                self.prices_file, None, False, self.root / "email.json", False, self.root / "webhook.txt",
                phase="full", symbols_file=self.pool_file, expected_date=EXPECTED_DATE,
            )
        settle.assert_called_once()
        self.assertTrue(settle.call_args.kwargs["settle"])
        self.assertFalse(settle.call_args.kwargs["open_new"])
        self.assertEqual(summary["closed_trades"], 1)
        self.assertEqual(summary["opened_positions"], 0)


class MinuteHistoryGuardTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = self.root / "minute_bars.csv"

    def test_archive_accumulates_days_and_repeated_fetch_corrects_without_duplicating_volume(self):
        old = _minutes("2026-09-21", volume=100)
        old.to_csv(self.snapshot, index=False)
        today = _minutes(EXPECTED_DATE, volume=200)
        update_minute_history(self.snapshot, today)
        today.to_csv(self.snapshot, index=False)
        corrected = _minutes(EXPECTED_DATE, close=10.5, volume=250)
        update_minute_history(self.snapshot, corrected)
        corrected.to_csv(self.snapshot, index=False)
        archive = pd.read_csv(minute_history_path(self.snapshot))
        self.assertEqual(len(archive), 2)
        self.assertEqual(archive.volume.sum(), 350)
        history = read_minute_history(self.snapshot)
        self.assertEqual(len(history), 2)
        self.assertEqual(history.datetime.dt.strftime("%Y-%m-%d").tolist(), ["2026-09-21", EXPECTED_DATE])
        self.assertEqual(history.iloc[-1].close, 10.5)

    def test_read_combines_archive_and_snapshot_preferring_snapshot_overlap(self):
        pd.concat([_minutes("2026-09-21"), _minutes(EXPECTED_DATE, volume=200)]).to_csv(
            minute_history_path(self.snapshot), index=False
        )
        _minutes(EXPECTED_DATE, close=10.5, volume=250).to_csv(self.snapshot, index=False)
        history = read_minute_history(self.snapshot)
        self.assertEqual(len(history), 2)
        self.assertEqual(history.iloc[-1].volume, 250)
        self.snapshot.unlink()
        self.assertEqual(len(read_minute_history(self.snapshot)), 2)
        minute_history_path(self.snapshot).unlink()
        self.assertTrue(read_minute_history(self.snapshot).empty)

    def test_read_preserves_duplicates_within_snapshot_for_feature_rejection(self):
        duplicated = pd.concat([_minutes(EXPECTED_DATE), _minutes(EXPECTED_DATE)], ignore_index=True)
        _minutes("2026-09-21").to_csv(minute_history_path(self.snapshot), index=False)
        duplicated.to_csv(self.snapshot, index=False)
        history = read_minute_history(self.snapshot)
        self.assertEqual(len(history), 3)
        self.assertEqual(int(history.duplicated(["symbol", "datetime"]).sum()), 1)

    def test_archiving_duplicates_keeps_session_unavailable_after_snapshot_advances(self):
        times = list(pd.date_range(f"{EXPECTED_DATE} 09:35", f"{EXPECTED_DATE} 11:30", freq="5min"))
        times += list(pd.date_range(f"{EXPECTED_DATE} 13:05", f"{EXPECTED_DATE} 14:55", freq="5min"))
        clean = pd.concat([_minutes(EXPECTED_DATE)] * len(times), ignore_index=True)
        clean["datetime"] = times
        duplicate = pd.concat([clean, clean.iloc[[10]]], ignore_index=True)
        duplicate.to_csv(self.snapshot, index=False)
        update_minute_history(self.snapshot, duplicate)
        tomorrow = _minutes("2026-09-23")
        update_minute_history(self.snapshot, tomorrow)
        tomorrow.to_csv(self.snapshot, index=False)

        daily = pd.DataFrame({"date": [EXPECTED_DATE], "symbol": ["000001.SZ"], "limit_up_price": [11.0]})
        history = read_minute_history(self.snapshot)
        prior = history[history.datetime.dt.strftime("%Y-%m-%d") == EXPECTED_DATE]
        self.assertEqual(len(prior), 47)
        result = build_tail_features(prior, daily).iloc[0]
        self.assertEqual(result.tail_available, 0)
        self.assertEqual(result.tail_unavailable_reason, "duplicate_bars")

        # A genuinely corrected fetch repairs the session; merely rolling the
        # snapshot forward must not repair it or multiply volume.
        update_minute_history(self.snapshot, clean)
        history = read_minute_history(self.snapshot)
        prior = history[history.datetime.dt.strftime("%Y-%m-%d") == EXPECTED_DATE]
        result = build_tail_features(prior, daily).iloc[0]
        self.assertEqual(result.tail_available, 1)
        self.assertEqual(result.tail_duplicate_bars, 0)
        self.assertEqual(prior.volume.sum(), 4700)

    def test_minute_provider_automatically_archives_previous_and_current_sessions(self):
        pool_file = self.root / "pool.csv"
        _pool().head(1).to_csv(pool_file, index=False)
        prior, current = _minutes("2026-09-21"), _minutes(EXPECTED_DATE)
        with patch("quant_limitup.providers._fetch_one_sina_minute", side_effect=[prior, current]) as fetcher, \
             patch("quant_limitup.providers._fetch_one_tencent_minute", side_effect=AssertionError("unexpected fallback")):
            fetch_sina_minute_bars(self.snapshot, pool_file, workers=1, pause_seconds=0)
            fetch_sina_minute_bars(self.snapshot, pool_file, workers=1, pause_seconds=0)
        self.assertEqual(fetcher.call_count, 2)
        snapshot = pd.read_csv(self.snapshot)
        self.assertEqual(len(snapshot), 1)
        self.assertTrue(snapshot.iloc[0].datetime.startswith(EXPECTED_DATE))
        history = read_minute_history(self.snapshot)
        self.assertEqual(history.datetime.dt.strftime("%Y-%m-%d").tolist(), ["2026-09-21", EXPECTED_DATE])


if __name__ == "__main__":
    unittest.main()
