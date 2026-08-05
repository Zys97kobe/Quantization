import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant_limitup.accuracy import write_prediction_accuracy
from quant_limitup.backtest import rank_candidates, run_backtest
from quant_limitup.candidate_review import (
    build_pending_reviews,
    load_completed_signals,
    load_rank_history,
    mark_review_completed,
    upsert_rank_history,
)
from quant_limitup.data import limit_up_price, read_prices, write_sample_prices
from quant_limitup.features import build_feature_frame
from quant_limitup.model import train_logistic
from quant_limitup.paper import (
    PaperAccount,
    Position,
    _daily_trade_counts,
    _previous_equity,
    run_paper_day,
    save_account,
)
from quant_limitup.config import TradingConfig
from quant_limitup.strategy import build_learning_report, optimize_threshold
from quant_limitup.messaging import _daily_text
from quant_limitup.providers import fetch_sina_minute_bars, sina_minute_market_is_current
from quant_limitup.cli import append_daily_summary_once, main, minute_data_is_current, safe_notify, safe_update_sina_stock_pool

class PipelineTest(unittest.TestCase):
    def test_sample_pipeline_runs(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        price_file = Path(tmp.name) / "daily_prices.csv"
        write_sample_prices(price_file, symbols=24, days=90)
        prices = read_prices(price_file)
        features = build_feature_frame(prices)

        self.assertFalse(features.empty)
        self.assertTrue({
            "target_limit_up_next",
            "volume_ratio_5",
            "market_limit_hits",
            "curve_slope_5",
            "curve_accel_5",
            "pct_chg_lag_1",
            "pct_chg_lag_5",
            "limit_gap_lag_1",
            "limit_hit_lag_5",
        }.issubset(features.columns))
        latest_rows = features[features["date"] == features["date"].max()]
        self.assertTrue(latest_rows["target_limit_up_next"].isna().all())
        for _, symbol_rows in features.groupby("symbol"):
            ordered = symbol_rows.sort_values("date")
            expected = ordered["high"].shift(-1)
            pd.testing.assert_series_equal(
                ordered["next_high"].reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
            )

        model, metrics = train_logistic(features, epochs=200)
        self.assertEqual(metrics["rows"], int(features["target_limit_up_next"].notna().sum()))

        rank = rank_candidates(features, model)
        self.assertFalse(rank.empty)
        self.assertTrue(rank["score"].is_monotonic_decreasing)

        trades, summary = run_backtest(features, model)
        self.assertIn("total_return", summary)
        self.assertIsInstance(trades, pd.DataFrame)

        _, optimized = optimize_threshold(features, model, TradingConfig())
        self.assertIn(optimized["max_position_pct"], {0.30, 0.34, 0.40, 0.50})

    def test_prediction_accuracy_file_upserts(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        price_file = Path(tmp.name) / "daily_prices.csv"
        accuracy_file = Path(tmp.name) / "prediction_accuracy.csv"
        write_sample_prices(price_file, symbols=24, days=90)
        features = build_feature_frame(read_prices(price_file))
        model, _ = train_logistic(features, epochs=200)
        learning = build_learning_report(features, model)
        summary = {"date": learning["date"], "phase": "buy"}

        first = write_prediction_accuracy(features, model, summary, learning, accuracy_file)
        second = write_prediction_accuracy(features, model, summary, learning, accuracy_file)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        saved = pd.read_csv(accuracy_file)
        self.assertEqual(len(saved), 1)
        self.assertTrue({"top1_symbol", "top5_hit_rate", "signal_date"}.issubset(saved.columns))

    def test_morning_sell_uses_newer_minute_date(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        price_file = root / "daily_prices.csv"
        write_sample_prices(price_file, symbols=24, days=90)
        features = build_feature_frame(read_prices(price_file))
        model, _ = train_logistic(features, epochs=50)
        latest = features.iloc[-1]
        signal_date = pd.to_datetime(latest["date"])
        trade_date = signal_date + pd.offsets.BDay(1)
        expected_limit = limit_up_price(float(latest["close"]), latest.board, 0)
        state_file = root / "account.json"
        save_account(
            PaperAccount(
                initial_cash=10_000,
                cash=8_000,
                positions=[Position(latest.symbol, latest["name"], latest.board, signal_date.strftime("%Y-%m-%d"), 100, 20, 2_000, 0.5)],
            ),
            state_file,
        )
        minute_bars = pd.DataFrame(
            [{
                "datetime": f"{trade_date:%Y-%m-%d} 10:00:00",
                "symbol": latest.symbol,
                "open": expected_limit,
                "high": expected_limit,
                "low": expected_limit,
                "close": expected_limit,
            }]
        )

        account, _, summary = run_paper_day(
            features,
            model,
            TradingConfig(),
            state_file=state_file,
            trades_file=root / "trades.csv",
            daily_file=root / "daily.csv",
            settle=True,
            open_new=False,
            minute_bars=minute_bars,
            sell_mode="morning",
        )

        self.assertEqual(summary["date"], trade_date.strftime("%Y-%m-%d"))
        self.assertEqual(summary["closed_trades"], 1)
        self.assertFalse(account.positions)

    def test_morning_pullback_does_not_sell_profitable_position(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        price_file = root / "daily_prices.csv"
        write_sample_prices(price_file, symbols=24, days=90)
        features = build_feature_frame(read_prices(price_file))
        model, _ = train_logistic(features, epochs=50)
        latest = features.iloc[-1]
        signal_date = pd.to_datetime(latest["date"])
        trade_date = signal_date + pd.offsets.BDay(1)
        expected_limit = limit_up_price(float(latest["close"]), latest.board, 0)
        running_high = min(expected_limit * 0.8, 22.0)
        buy_price = running_high * 0.90
        profitable_pullback_close = running_high * 0.96
        state_file = root / "account.json"
        save_account(
            PaperAccount(
                initial_cash=10_000,
                cash=8_000,
                positions=[
                    Position(
                        latest.symbol,
                        latest["name"],
                        latest.board,
                        signal_date.strftime("%Y-%m-%d"),
                        100,
                        buy_price,
                        buy_price * 100,
                        0.5,
                    )
                ],
            ),
            state_file,
        )
        minute_bars = pd.DataFrame(
            [
                {
                    "datetime": f"{trade_date:%Y-%m-%d} 09:35:00",
                    "symbol": latest.symbol,
                    "open": running_high,
                    "high": running_high,
                    "low": running_high,
                    "close": running_high,
                },
                {
                    "datetime": f"{trade_date:%Y-%m-%d} 09:40:00",
                    "symbol": latest.symbol,
                    "open": profitable_pullback_close,
                    "high": profitable_pullback_close,
                    "low": profitable_pullback_close,
                    "close": profitable_pullback_close,
                },
            ]
        )

        account, _, summary = run_paper_day(
            features,
            model,
            TradingConfig(),
            state_file=state_file,
            trades_file=root / "trades.csv",
            daily_file=root / "daily.csv",
            settle=True,
            open_new=False,
            minute_bars=minute_bars,
            sell_mode="morning",
        )

        self.assertEqual(summary["date"], trade_date.strftime("%Y-%m-%d"))
        self.assertEqual(summary["closed_trades"], 0)
        self.assertEqual(len(account.positions), 1)

    def test_opening_drop_over_six_percent_defers_sale_until_force(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        price_file = root / "daily_prices.csv"
        write_sample_prices(price_file, symbols=24, days=90)
        features = build_feature_frame(read_prices(price_file))
        model, _ = train_logistic(features, epochs=50)
        latest = features.iloc[-1]
        signal_date = pd.to_datetime(latest["date"])
        trade_date = signal_date + pd.offsets.BDay(1)
        prev_close = float(latest["close"])
        state_file = root / "account.json"
        trades_file = root / "trades.csv"
        daily_file = root / "daily.csv"
        save_account(
            PaperAccount(
                initial_cash=10_000,
                cash=8_000,
                positions=[
                    Position(
                        latest.symbol,
                        latest["name"],
                        latest.board,
                        signal_date.strftime("%Y-%m-%d"),
                        100,
                        prev_close,
                        prev_close * 100,
                        0.5,
                    )
                ],
            ),
            state_file,
        )
        minute_bars = pd.DataFrame(
            [
                {
                    "datetime": f"{trade_date:%Y-%m-%d} 09:35:00",
                    "symbol": latest.symbol,
                    "open": prev_close * 0.93,
                    "high": prev_close * 0.93,
                    "low": prev_close * 0.92,
                    "close": prev_close * 0.92,
                },
                {
                    "datetime": f"{trade_date:%Y-%m-%d} 09:40:00",
                    "symbol": latest.symbol,
                    "open": prev_close * 0.92,
                    "high": prev_close * 0.92,
                    "low": prev_close * 0.88,
                    "close": prev_close * 0.88,
                },
                {
                    "datetime": f"{trade_date:%Y-%m-%d} 14:50:00",
                    "symbol": latest.symbol,
                    "open": prev_close * 0.90,
                    "high": prev_close * 0.91,
                    "low": prev_close * 0.89,
                    "close": prev_close * 0.90,
                },
            ]
        )

        account, _, morning = run_paper_day(
            features,
            model,
            TradingConfig(),
            state_file=state_file,
            trades_file=trades_file,
            daily_file=daily_file,
            settle=True,
            open_new=False,
            minute_bars=minute_bars,
            sell_mode="morning",
        )

        self.assertEqual(morning["closed_trades"], 0)
        self.assertEqual(len(account.positions), 1)

        account, _, force = run_paper_day(
            features,
            model,
            TradingConfig(),
            state_file=state_file,
            trades_file=trades_file,
            daily_file=daily_file,
            settle=True,
            open_new=False,
            minute_bars=minute_bars,
            sell_mode="force",
        )

        self.assertEqual(force["closed_trades"], 1)
        self.assertFalse(account.positions)
        trades = pd.read_csv(trades_file)
        self.assertEqual(trades.iloc[-1]["reason"], "force_sell")

    def test_morning_pullback_requires_loss_over_two_hundred(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        price_file = root / "daily_prices.csv"
        write_sample_prices(price_file, symbols=24, days=90)
        features = build_feature_frame(read_prices(price_file))
        model, _ = train_logistic(features, epochs=50)
        latest = features.iloc[-1]
        signal_date = pd.to_datetime(latest["date"])
        trade_date = signal_date + pd.offsets.BDay(1)
        state_file = root / "account.json"
        buy_price = 14.21136
        save_account(
            PaperAccount(
                initial_cash=10_000,
                cash=7_152.728,
                positions=[
                    Position(
                        latest.symbol,
                        latest["name"],
                        latest.board,
                        signal_date.strftime("%Y-%m-%d"),
                        200,
                        buy_price,
                        buy_price * 200 + 5,
                        0.5,
                    )
                ],
            ),
            state_file,
        )
        minute_bars = pd.DataFrame(
            [{
                "datetime": f"{trade_date:%Y-%m-%d} 09:35:00",
                "symbol": latest.symbol,
                "open": 14.69,
                "high": 14.90,
                "low": 14.10,
                "close": 14.21,
            }]
        )

        account, _, summary = run_paper_day(
            features,
            model,
            TradingConfig(),
            state_file=state_file,
            trades_file=root / "trades.csv",
            daily_file=root / "daily.csv",
            settle=True,
            open_new=False,
            minute_bars=minute_bars,
            sell_mode="morning",
        )

        self.assertEqual(summary["closed_trades"], 0)
        self.assertEqual(len(account.positions), 1)

    def test_large_pullback_loss_sells_unless_return_below_five_percent(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        price_file = root / "daily_prices.csv"
        write_sample_prices(price_file, symbols=24, days=90)
        features = build_feature_frame(read_prices(price_file))
        model, _ = train_logistic(features, epochs=50)
        latest = features.iloc[-1]
        signal_date = pd.to_datetime(latest["date"])
        trade_date = signal_date + pd.offsets.BDay(1)

        base_price = float(latest["close"])

        def run_case(close_ratio: float, suffix: str) -> tuple[PaperAccount, dict]:
            state_file = root / f"account_{suffix}.json"
            save_account(
                PaperAccount(
                    initial_cash=100_000,
                    cash=100_000 - (base_price * 1_000 + 5),
                    positions=[
                        Position(
                            latest.symbol,
                            latest["name"],
                            latest.board,
                            signal_date.strftime("%Y-%m-%d"),
                            1_000,
                            base_price,
                            base_price * 1_000 + 5,
                            0.5,
                        )
                    ],
                ),
                state_file,
            )
            minute_bars = pd.DataFrame(
                [{
                    "datetime": f"{trade_date:%Y-%m-%d} 09:35:00",
                    "symbol": latest.symbol,
                    "open": base_price,
                    "high": base_price * 1.01,
                    "low": base_price * close_ratio,
                    "close": base_price * close_ratio,
                }]
            )
            account, _, summary = run_paper_day(
                features,
                model,
                TradingConfig(),
                state_file=state_file,
                trades_file=root / f"trades_{suffix}.csv",
                daily_file=root / f"daily_{suffix}.csv",
                settle=True,
                open_new=False,
                minute_bars=minute_bars,
                sell_mode="morning",
            )
            return account, summary

        sold_account, sold_summary = run_case(0.97, "moderate")
        deferred_account, deferred_summary = run_case(0.94, "severe")

        self.assertEqual(sold_summary["closed_trades"], 1)
        self.assertFalse(sold_account.positions)
        self.assertEqual(deferred_summary["closed_trades"], 0)
        self.assertEqual(len(deferred_account.positions), 1)

    def test_buy_uses_newer_minute_date(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        price_file = root / "daily_prices.csv"
        write_sample_prices(price_file, symbols=24, days=90)
        features = build_feature_frame(read_prices(price_file))
        model, _ = train_logistic(features, epochs=50)
        signal_date = features["date"].max()
        trade_date = signal_date + pd.offsets.BDay(1)
        latest = features[features["date"] == signal_date].iloc[0]
        minute_bars = pd.DataFrame([{
            "datetime": f"{trade_date:%Y-%m-%d} 14:50:00",
            "symbol": latest.symbol,
            "open": latest.close,
            "high": latest.close,
            "low": latest.close,
            "close": latest.close,
        }])

        _, _, summary = run_paper_day(
            features,
            model,
            TradingConfig(min_score_to_buy=1.0),
            state_file=root / "account.json",
            trades_file=root / "trades.csv",
            daily_file=root / "daily.csv",
            settle=False,
            open_new=True,
            minute_bars=minute_bars,
        )

        self.assertEqual(summary["date"], trade_date.strftime("%Y-%m-%d"))
        self.assertEqual(summary["phase"], "buy")

    def test_daily_message_contains_position_details(self) -> None:
        positions = pd.DataFrame([{
            "symbol": "000001.SZ", "name": "平安银行", "shares": 100,
            "buy_price": 10.0, "current_price": 10.5, "market_value": 1050.0,
            "unrealized_pnl": 45.0, "score": 0.25,
        }])
        summary = {
            "date": "2026-06-18", "equity": 10050.0, "cash": 9000.0,
            "daily_pnl": 50.0, "daily_return": 0.005, "total_return": 0.005,
            "open_positions": 1, "opened_positions": 1, "closed_trades": 0,
            "phase": "buy",
        }
        rank = pd.DataFrame(columns=["symbol", "name", "score", "close", "suggest_reason"])

        text = _daily_text(summary, rank, None, None, positions, None)

        self.assertIn("当前持仓明细:", text)
        self.assertIn("候选前 10:", text)
        self.assertNotIn("策略学习:", text)
        self.assertNotIn("昨日候选日线复盘:", text)
        self.assertIn("000001.SZ 平安银行 100股", text)
        self.assertIn("浮动盈亏=45.00", text)

        learning = {
            "latest_labeled_signal_date": "2026-06-17",
            "latest_result_date": "2026-06-18",
            "actual_limit_up_count": 119,
            "top3_hit_rate": 1 / 3,
            "top5_hit_rate": 0.2,
            "top10_hit_rate": 0.1,
            "positive_rate": 0.02,
            "evaluation_source": "daily",
            "evaluated_candidates": [
                {
                    "candidate_rank": 1, "symbol": "000001.SZ", "name": "平安银行",
                    "score": 0.25, "hit_limit": True, "was_bought": True, "in_top10": True,
                },
                {
                    "candidate_rank": 12, "symbol": "000002.SZ", "name": "万科A",
                    "score": 0.10, "hit_limit": False, "was_bought": True, "in_top10": False,
                },
            ],
        }
        learning_text = _daily_text({**summary, "phase": "sell-morning"}, rank, None, None, positions, learning)
        self.assertNotIn("实际触及涨停数", learning_text)
        self.assertIn("昨日候选日线复盘", learning_text)
        self.assertIn("Top10明细（信号日 2026-06-17，结果日 2026-06-18）", learning_text)
        self.assertIn("000001.SZ 平安银行 分数=0.2500｜涨停｜已买入", learning_text)
        self.assertIn("昨日买入（Top10外）", learning_text)
        self.assertIn("排名12 000002.SZ 万科A", learning_text)

        sell_summary = {**summary, "phase": "sell-morning"}
        sell_text = _daily_text(sell_summary, rank, None, None, positions, {"top3_hit_rate": 1})
        self.assertTrue(sell_text.startswith("Sell Morining - 2026-06-18"))
        self.assertNotIn("策略学习:", sell_text)
        self.assertNotIn("候选前 10:", sell_text)

    def test_daily_totals_accumulate_across_phases(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        daily_file = root / "daily.csv"
        trades_file = root / "trades.csv"
        pd.DataFrame([
            {"date": "2026-06-17", "equity": 9979.11, "phase": "buy"},
            {"date": "2026-06-18", "equity": 10252.12, "phase": "sell-morning"},
        ]).to_csv(daily_file, index=False)
        pd.DataFrame([
            {"date": "2026-06-18", "action": "SELL"},
            {"date": "2026-06-18", "action": "SELL"},
            {"date": "2026-06-18", "action": "SELL"},
        ]).to_csv(trades_file, index=False)

        baseline = _previous_equity(daily_file, 10_000, pd.Timestamp("2026-06-18"))
        opened, closed = _daily_trade_counts(trades_file, "2026-06-18")

        self.assertEqual(baseline, 9979.11)
        self.assertEqual(opened, 0)
        self.assertEqual(closed, 3)

    def test_market_closed_when_minute_data_is_stale(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        minute_file = Path(tmp.name) / "minute.csv"
        pd.DataFrame([{"datetime": "2026-06-18 15:00:00"}]).to_csv(minute_file, index=False)
        self.assertTrue(minute_data_is_current(minute_file, "2026-06-18"))
        self.assertFalse(minute_data_is_current(minute_file, "2026-06-19"))

    def test_market_probe_requires_current_minute_data(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        symbols = Path(tmp.name) / "symbols.csv"
        pd.DataFrame([{
            "symbol": "000001.SZ", "name": "平安银行", "board": "main", "is_st": 0,
        }]).to_csv(symbols, index=False)
        stale = pd.DataFrame([{"datetime": "2026-06-18 15:00:00"}])
        current = pd.DataFrame([{"datetime": "2026-06-22 10:25:00"}])

        with patch("quant_limitup.providers._fetch_one_sina_minute", return_value=stale):
            self.assertFalse(sina_minute_market_is_current(symbols, "2026-06-22"))
        with patch("quant_limitup.providers._fetch_one_sina_minute", return_value=current):
            self.assertTrue(sina_minute_market_is_current(symbols, "2026-06-22"))

    def test_market_probe_uses_tencent_fallback(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        symbols = Path(tmp.name) / "symbols.csv"
        pd.DataFrame([{
            "symbol": "000001.SZ", "name": "平安银行", "board": "main", "is_st": 0,
        }]).to_csv(symbols, index=False)
        current = pd.DataFrame([{"datetime": "2026-06-22 10:25:00"}])

        with (
            patch("quant_limitup.providers._fetch_one_sina_minute", side_effect=RuntimeError("sina failed")),
            patch("quant_limitup.providers._fetch_one_tencent_minute", return_value=current),
        ):
            self.assertTrue(sina_minute_market_is_current(symbols, "2026-06-22"))

    def test_minute_fetch_uses_tencent_fallback_when_sina_fails(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        symbols = root / "symbols.csv"
        out = root / "minute.csv"
        pd.DataFrame([{
            "symbol": "000001.SZ", "name": "平安银行", "board": "main", "is_st": 0,
        }]).to_csv(symbols, index=False)
        fallback = pd.DataFrame([{
            "datetime": pd.Timestamp("2026-06-22 10:25:00"),
            "symbol": "000001.SZ",
            "open": 10.0,
            "high": 10.2,
            "low": 9.9,
            "close": 10.1,
            "volume": 1000.0,
            "amount": 10050.0,
        }])

        with (
            patch("quant_limitup.providers._fetch_one_sina_minute", side_effect=RuntimeError("sina failed")),
            patch("quant_limitup.providers._fetch_one_tencent_minute", return_value=fallback),
        ):
            result = fetch_sina_minute_bars(out, symbols, workers=1)

        self.assertEqual(len(result), 1)
        self.assertTrue(out.exists())

    def test_buy_phase_uses_tencent_daily_before_sina(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        symbols = root / "symbols.csv"
        prices = root / "daily.csv"
        pd.DataFrame([{"symbol": "000001.SZ", "name": "平安银行", "board": "main", "is_st": 0}]).to_csv(symbols, index=False)

        with (
            patch("quant_limitup.cli.fetch_tencent_daily_prices", return_value=pd.DataFrame()) as fetch_tencent,
            patch("quant_limitup.cli.fetch_sina_daily_prices") as fetch_sina,
            patch("quant_limitup.cli.run_from_prices") as run_from_prices,
        ):
            main([
                "daily",
                "--provider",
                "sina",
                "--phase",
                "buy",
                "--symbols",
                str(symbols),
                "--prices",
                str(prices),
            ])

        fetch_tencent.assert_called_once()
        fetch_sina.assert_not_called()
        run_from_prices.assert_called_once_with(prices)

    def test_closed_market_skips_before_fetch_and_model_prefilter(self) -> None:
        with (
            patch("quant_limitup.cli.sina_minute_market_is_current", return_value=False),
            patch("quant_limitup.cli.fetch_sina_daily_prices") as fetch_daily,
            patch("quant_limitup.cli.build_minute_symbol_pool") as build_pool,
            patch("quant_limitup.cli.run_paper_from_prices") as run_paper,
            patch("quant_limitup.cli.run_pending_candidate_reviews") as run_reviews,
        ):
            main(["daily", "--provider", "sina", "--paper", "--use-minute", "--phase", "sell-morning"])

        run_reviews.assert_called_once()
        fetch_daily.assert_not_called()
        build_pool.assert_not_called()
        run_paper.assert_not_called()

    def test_stock_pool_refresh_failure_falls_back_to_existing_pool(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        symbols = Path(tmp.name) / "symbols.csv"
        pd.DataFrame([{"symbol": "000001.SZ", "name": "平安银行"}]).to_csv(symbols, index=False)

        with patch("quant_limitup.cli.update_sina_stock_pool", side_effect=ConnectionError("reset")):
            self.assertIsNone(safe_update_sina_stock_pool(symbols, attempts=1))

    def test_no_position_sell_summary_is_idempotent(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        daily_file = Path(tmp.name) / "daily.csv"
        summary = {
            "date": "2026-06-24",
            "phase": "sell-morning",
            "skipped": "no_positions",
            "equity": 10000.0,
        }

        append_daily_summary_once(daily_file, summary)
        append_daily_summary_once(daily_file, summary)
        saved = pd.read_csv(daily_file)

        self.assertEqual(len(saved), 1)

    def test_notification_failure_does_not_raise(self) -> None:
        def fail_notify() -> None:
            raise RuntimeError("frequency limited")

        self.assertFalse(safe_notify("feishu", fail_notify))

    def test_failed_review_notification_remains_pending(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "candidate_reviews.csv"
        review = {
            "signal_date": "2026-07-21",
            "result_date": "2026-07-22",
            "candidate_count": 10,
            "top10_evaluated_count": 10,
            "top10_hit_rate": 0.4,
            "evaluated_candidates": [],
        }

        mark_review_completed(path, review, notification_sent=False)
        self.assertNotIn("2026-07-21", load_completed_signals(path))
        mark_review_completed(path, review, notification_sent=True)
        self.assertIn("2026-07-21", load_completed_signals(path))

    def test_reviews_follow_next_trading_day_across_market_closure(self) -> None:
        history = pd.DataFrame([
            {"date": "2026-06-17", "symbol": "002297.SZ", "name": "博云新材", "board": "main", "is_st": 0, "close": 10.0, "score": 0.4},
            {"date": "2026-06-18", "symbol": "300088.SZ", "name": "长信科技", "board": "chinext", "is_st": 0, "close": 10.0, "score": 0.3},
        ])
        prices_through_18 = pd.DataFrame([
            {"date": "2026-06-18", "symbol": "002297.SZ", "high": 11.0},
            {"date": "2026-06-18", "symbol": "300088.SZ", "high": 10.5},
        ])

        first = build_pending_reviews(prices_through_18, history, set(), as_of_date="2026-06-19")

        self.assertEqual([item["signal_date"] for item in first], ["2026-06-17"])
        self.assertEqual(first[0]["result_date"], "2026-06-18")
        self.assertTrue(first[0]["evaluated_candidates"][0]["hit_limit"])

        prices_through_22 = pd.concat([
            prices_through_18,
            pd.DataFrame([{"date": "2026-06-22", "symbol": "300088.SZ", "high": 12.0}]),
        ], ignore_index=True)
        not_yet = build_pending_reviews(
            prices_through_22,
            history,
            {"2026-06-17"},
            as_of_date="2026-06-22",
        )
        self.assertEqual(not_yet, [])
        second = build_pending_reviews(
            prices_through_22,
            history,
            {"2026-06-17"},
            as_of_date="2026-06-23",
        )

        self.assertEqual([item["signal_date"] for item in second], ["2026-06-18"])
        self.assertEqual(second[0]["result_date"], "2026-06-22")

    def test_rank_history_preserves_displayed_order(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "rank_history.csv"
        displayed = pd.DataFrame([
            {"date": "2026-06-17", "display_rank": 1, "symbol": "A", "name": "A", "score": 0.2},
            {"date": "2026-06-17", "display_rank": 2, "symbol": "B", "name": "B", "score": 0.9},
        ])

        upsert_rank_history(path, displayed)
        saved = load_rank_history(path)

        self.assertEqual(saved["symbol"].tolist(), ["A", "B"])



if __name__ == "__main__":
    unittest.main()
