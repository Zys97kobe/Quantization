import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_limitup.backtest import run_backtest
from quant_limitup.config import TradingConfig
from quant_limitup.data import read_prices, write_sample_prices
from quant_limitup.features import build_feature_frame
from quant_limitup.strategy import (
    _generate_oos_predictions,
    _optimization_history,
    _threshold_candidates,
    optimize_threshold as _optimize_threshold,
)
from quant_limitup.execution import ExecutionQuotes


def optimize_threshold(frame, model, config, out_file=None):
    # Isolate OOS selection/cache tests from the minute-availability gate.
    quotes = ExecutionQuotes()
    for row in frame.itertuples(index=False):
        date = getattr(row, "target_end_date", pd.NaT)
        if pd.isna(date) or pd.isna(row.next_close):
            continue
        quotes.rows[(row.symbol, pd.Timestamp(date))] = {
            "morning_valid": True, "force_valid": True, "morning_open": float(row.close),
            "morning_high": float(row.next_high), "morning_close": float(row.next_close),
            "force_close": float(row.next_close),
        }
    return _optimize_threshold(frame, model, config, out_file, minute_bars=quotes)


class HistoricalModelMustNotScore:
    feature_columns = ["signal"]

    def predict_proba(self, frame):
        raise AssertionError("The full-history model must not score optimization data")


class FoldModel:
    feature_columns = ["signal"]

    def predict_proba(self, frame):
        return frame["signal"].to_numpy()


def feature_fixture(days=100):
    dates = pd.bdate_range("2025-01-02", periods=days + 1)
    rows = []
    for day, date in enumerate(dates[:-1]):
        for symbol in range(10):
            hit = int(symbol >= 8)
            rows.append({
                "date": date,
                "symbol": f"600{symbol:03}.SH",
                "name": f"Company {symbol}",
                "board": "main",
                "close": 10.0,
                "next_high": 11.0 if hit else 10.1,
                "next_close": 10.5 if hit else 9.8,
                "next_limit_up_price": 11.0,
                "target_limit_up_next": hit,
                "target_end_date": dates[day + 1],
                "market_data_eligible": True,
                "signal": 0.9 - symbol * 0.08,
            })
    return pd.DataFrame(rows)


def fixed_oos(history, feature_columns):
    dates = sorted(history["date"].unique())[-60:]
    scored = history[history["date"].isin(dates)].copy()
    scored["score"] = scored["signal"]
    return scored, [{"status": "scored"}]


class StrategyOptimizationTest(unittest.TestCase):
    def test_fold_training_purges_every_unavailable_label_and_predicts_future_dates(self):
        frame = feature_fixture()
        history = _optimization_history(frame)
        # A label with an unusually delayed outcome must not enter earlier folds.
        history.loc[history.index[5], "target_end_date"] = frame["date"].max()
        fitted = []

        def train(training, **kwargs):
            fitted.append(training.copy())
            return FoldModel(), {"rows": len(training)}

        with patch("quant_limitup.strategy.train_logistic", side_effect=train):
            scored, folds = _generate_oos_predictions(history, ["signal"])

        successful = [fold for fold in folds if fold["status"] == "scored"]
        self.assertEqual(len(fitted), len(successful))
        self.assertGreater(len(successful), 1)
        for train, fold in zip(fitted, successful):
            start = pd.Timestamp(fold["prediction_start"])
            self.assertTrue((train["date"] < start).all())
            self.assertTrue((train["target_end_date"] < start).all())
            self.assertNotIn(history.index[5], train.index)
        self.assertLessEqual(scored["date"].nunique(), 60)
        self.assertTrue(scored.index.is_unique)

    def test_inferred_target_dates_and_incomplete_market_days_are_excluded(self):
        frame = feature_fixture(40).drop(columns="target_end_date")
        bad_date = frame["date"].unique()[20]
        frame.loc[frame["date"] == bad_date, "market_data_eligible"] = False
        history = _optimization_history(frame)
        self.assertNotIn(bad_date, history["date"].unique())
        self.assertNotIn(bad_date, history["target_end_date"].unique())
        self.assertNotIn(frame["date"].max(), history["date"].unique())
        self.assertTrue((history["target_end_date"] > history["date"]).all())

    def test_no_buy_is_available_and_holdout_cannot_choose_parameters(self):
        frame = feature_fixture()
        model = HistoricalModelMustNotScore()
        with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=fixed_oos):
            threshold, first = optimize_threshold(frame, model, TradingConfig())
            # Change only the latest holdout returns, keeping the selection past fixed.
            last_dates = sorted(frame["date"].unique())[-20:]
            changed = frame.copy()
            changed.loc[changed["date"].isin(last_dates), "next_high"] = 11.0
            changed.loc[changed["date"].isin(last_dates), "next_close"] = 11.0
            changed.loc[changed["date"].isin(last_dates), "target_limit_up_next"] = 1
            second_threshold, second = optimize_threshold(changed, model, TradingConfig())
        self.assertGreater(threshold, 1.0)
        self.assertEqual(first["trades"], 0)
        self.assertEqual(threshold, second_threshold)
        self.assertEqual(first["max_position_pct"], second["max_position_pct"])
        self.assertEqual(first["selection"], second["selection"])
        self.assertFalse(first["holdout_used_for_selection"])
        self.assertGreater(second["default_holdout"]["total_return"], first["default_holdout"]["total_return"])

    def test_tied_results_preserve_default_parameters(self):
        config = TradingConfig(initial_cash=100.0)
        with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=fixed_oos):
            threshold, result = optimize_threshold(feature_fixture(), HistoricalModelMustNotScore(), config)
        self.assertEqual(threshold, config.min_score_to_buy)
        self.assertEqual(result["max_position_pct"], config.max_position_pct)
        self.assertEqual(result["trades"], 0)

    def test_threshold_grid_targets_daily_leading_candidates(self):
        frame = feature_fixture(3)
        frame["score"] = frame["signal"]
        thresholds = _threshold_candidates(frame, TradingConfig())
        boundaries = [value for value in thresholds if value != TradingConfig().min_score_to_buy and value <= 1]
        self.assertGreaterEqual(min(boundaries), 0.74 - 1e-12)
        self.assertGreater(max(thresholds), 1.0)

    def test_cache_freezes_for_four_new_trading_days_and_expires_on_fifth(self):
        with TemporaryDirectory() as directory:
            report = Path(directory) / "optimization.json"
            with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=fixed_oos) as scorer:
                optimize_threshold(feature_fixture(100), HistoricalModelMustNotScore(), TradingConfig(), report)
                _, cached = optimize_threshold(feature_fixture(104), HistoricalModelMustNotScore(), TradingConfig(), report)
                self.assertEqual(scorer.call_count, 1)
                self.assertTrue(cached["cached"])
                _, refreshed = optimize_threshold(feature_fixture(105), HistoricalModelMustNotScore(), TradingConfig(), report)
                self.assertEqual(scorer.call_count, 2)
                self.assertFalse(refreshed["cached"])

    def test_fixed_length_window_and_frontier_warmup_keep_cache_until_fifth_day(self):
        def rolling_frame(advanced):
            frame = feature_fixture(260 + advanced)
            dates = sorted(frame["date"].unique())[-260:]
            frame = frame[frame["date"].isin(dates)].copy()
            # Rebuilding rolling features from a later raw-data frontier may
            # change the first few observations even without a data revision.
            frame.loc[frame["date"].isin(dates[:10]), "signal"] = 0.2 + advanced * 0.01
            return frame

        with TemporaryDirectory() as directory:
            report = Path(directory) / "optimization.json"
            with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=fixed_oos) as scorer:
                optimize_threshold(rolling_frame(0), HistoricalModelMustNotScore(), TradingConfig(), report)
                payload = json.loads(report.read_text())
                self.assertIn("date_digests", payload["cache_validation"])
                for advanced in (1, 4):
                    _, result = optimize_threshold(rolling_frame(advanced), HistoricalModelMustNotScore(), TradingConfig(), report)
                    self.assertTrue(result["cached"])
                    self.assertEqual(scorer.call_count, 1)
                _, refreshed = optimize_threshold(rolling_frame(5), HistoricalModelMustNotScore(), TradingConfig(), report)
                self.assertFalse(refreshed["cached"])
                self.assertEqual(scorer.call_count, 2)

    def test_real_feature_rebuild_reuses_rolling_cache_despite_floating_point_roundoff(self):
        def score_features(history, feature_columns):
            dates = sorted(history["date"].unique())[-60:]
            scored = history[history["date"].isin(dates)].copy()
            scored["score"] = (0.5 + scored["pct_chg"]).clip(0, 1)
            return scored, [{"status": "scored"}]

        with TemporaryDirectory() as directory:
            price_file = Path(directory) / "prices.csv"
            report = Path(directory) / "optimization.json"
            # Keep the generated labels deterministic and plentiful enough for
            # parameter selection while exercising the real feature builder.
            dates = pd.bdate_range("2025-01-02", periods=261)
            with patch("quant_limitup.data.pd.bdate_range", return_value=dates):
                write_sample_prices(price_file, symbols=80, days=261)
            prices = read_prices(price_file)
            first = build_feature_frame(prices[prices["date"].isin(dates[:-1])])
            rolled = build_feature_frame(prices[prices["date"].isin(dates[1:])])
            model = HistoricalModelMustNotScore()
            model.feature_columns = ["pct_chg"]
            with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=score_features) as scorer:
                _, calibrated = optimize_threshold(first, model, TradingConfig(), report)
                self.assertEqual(calibrated["status"], "optimized_oos")
                _, cached = optimize_threshold(rolled, model, TradingConfig(), report)
                self.assertTrue(cached["cached"])
                self.assertEqual(scorer.call_count, 1)
                payload = json.loads(report.read_text())
                self.assertEqual(payload["cache_validation"]["float_decimal_places"], 8)

                # A material revision in the stable historical overlap must
                # still invalidate the fingerprints after quantization.
                revised = rolled.copy()
                index = revised.index[revised["date"] == dates[-30]][0]
                revised.loc[index, "volatility_5"] += 1e-5
                _, refreshed = optimize_threshold(revised, model, TradingConfig(), report)
                self.assertFalse(refreshed["cached"])
                self.assertEqual(scorer.call_count, 2)

    def test_cache_normalizes_roundoff_and_signed_zero_without_changing_inputs(self):
        frame = feature_fixture().assign(zero_feature=0.0)
        noisy = frame.copy()
        noisy["signal"] = np.nextafter(noisy["signal"], np.inf)
        noisy["zero_feature"] = -1e-15
        original_noisy = noisy.copy(deep=True)
        with TemporaryDirectory() as directory:
            report = Path(directory) / "optimization.json"
            with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=fixed_oos) as scorer:
                optimize_threshold(frame, HistoricalModelMustNotScore(), TradingConfig(), report)
                _, cached = optimize_threshold(noisy, HistoricalModelMustNotScore(), TradingConfig(), report)
                self.assertTrue(cached["cached"])
                self.assertEqual(scorer.call_count, 1)
        pd.testing.assert_frame_equal(noisy, original_noisy)

    def test_rolling_cache_still_rejects_stable_training_and_evaluation_revisions_or_deletions(self):
        frame = feature_fixture(260)
        rolled = feature_fixture(261)
        dates = sorted(rolled["date"].unique())[-260:]
        rolled = rolled[rolled["date"].isin(dates)].copy()
        with TemporaryDirectory() as directory:
            report = Path(directory) / "optimization.json"
            with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=fixed_oos) as scorer:
                optimize_threshold(frame, HistoricalModelMustNotScore(), TradingConfig(), report)
                original_report = report.read_text()
                for offset in (20, 210, 245):
                    revised = rolled.copy()
                    index = revised.index[revised["date"] == dates[offset]][0]
                    revised.loc[index, "signal"] = 0.1
                    cases = [
                        revised,
                        rolled.drop(index=index),
                        rolled[rolled["date"] != dates[offset]],
                    ]
                    for changed in cases:
                        with self.subTest(offset=offset, rows=len(changed)):
                            report.write_text(original_report)
                            before = scorer.call_count
                            _, result = optimize_threshold(changed, HistoricalModelMustNotScore(), TradingConfig(), report)
                            self.assertFalse(result["cached"])
                            self.assertEqual(scorer.call_count, before + 1)

    def test_cache_cannot_drop_selection_dates_or_more_history_than_new_sessions(self):
        with TemporaryDirectory() as directory:
            report = Path(directory) / "optimization.json"
            with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=fixed_oos) as scorer:
                # With only 60 days, the advancing prefix belongs to selection;
                # there is no safe training-only frontier to exempt.
                optimize_threshold(feature_fixture(60), HistoricalModelMustNotScore(), TradingConfig(), report)
                rolled = feature_fixture(61)
                rolled = rolled[rolled["date"] > rolled["date"].min()]
                _, result = optimize_threshold(rolled, HistoricalModelMustNotScore(), TradingConfig(), report)
                self.assertFalse(result["cached"])
                self.assertEqual(scorer.call_count, 2)

                optimize_threshold(feature_fixture(100), HistoricalModelMustNotScore(), TradingConfig(), report)
                rolled = feature_fixture(101)
                dates = sorted(rolled["date"].unique())
                rolled = rolled[rolled["date"] >= dates[2]]
                _, result = optimize_threshold(rolled, HistoricalModelMustNotScore(), TradingConfig(), report)
                self.assertFalse(result["cached"])
                self.assertEqual(scorer.call_count, 4)

    def test_cache_rejects_legacy_report_revised_data_config_features_and_schema(self):
        frame = feature_fixture()
        config = TradingConfig()
        with TemporaryDirectory() as directory:
            report = Path(directory) / "optimization.json"
            report.write_text(json.dumps({"best_threshold": 0.001, "results": []}))
            with patch("quant_limitup.strategy._generate_oos_predictions", side_effect=fixed_oos) as scorer:
                optimize_threshold(frame, HistoricalModelMustNotScore(), config, report)
                original_report = report.read_text()
                previous_version = json.loads(original_report)
                previous_version["version"] -= 1
                report.write_text(json.dumps(previous_version))
                _, result = optimize_threshold(frame, HistoricalModelMustNotScore(), config, report)
                self.assertFalse(result["cached"])
                self.assertEqual(scorer.call_count, 2)
                revised = frame.copy()
                revised.loc[0, "next_close"] = 9.0
                cases = [
                    (revised, HistoricalModelMustNotScore(), config),
                    (frame, HistoricalModelMustNotScore(), replace(config, commission_bps=4.0)),
                    (frame.assign(new_feature=0.0), HistoricalModelMustNotScore(), config),
                ]
                changed_features = HistoricalModelMustNotScore()
                changed_features.feature_columns = ["signal", "close"]
                cases.append((frame, changed_features, config))
                for changed_frame, changed_model, changed_config in cases:
                    report.write_text(original_report)
                    before = scorer.call_count
                    _, result = optimize_threshold(changed_frame, changed_model, changed_config, report)
                    self.assertEqual(scorer.call_count, before + 1)
                    self.assertFalse(result["cached"])

    def test_insufficient_history_returns_explicit_default_without_scoring_full_model(self):
        config = TradingConfig()
        threshold, result = optimize_threshold(feature_fixture(12), HistoricalModelMustNotScore(), config)
        self.assertEqual(threshold, config.min_score_to_buy)
        self.assertEqual(result["max_position_pct"], config.max_position_pct)
        self.assertEqual(result["status"], "insufficient_oos_data")

    def test_precomputed_scores_are_index_aligned_and_initial_loss_counts_in_drawdown(self):
        frame = feature_fixture(1).iloc[:2].copy()
        frame.index = [9, 4]
        scores = pd.Series([0.8, 0.2], index=[4, 9])
        trades, summary = run_backtest(frame, HistoricalModelMustNotScore(), TradingConfig(max_positions_per_day=1), scores=scores)
        self.assertEqual(trades.iloc[0]["symbol"], frame.loc[4, "symbol"])
        self.assertLess(summary["max_drawdown"], 0.0)
        self.assertAlmostEqual(summary["max_drawdown"], summary["total_return"])
        for invalid in (pd.Series([0.8], index=[4]), pd.Series([np.nan, 0.2], index=[4, 9])):
            with self.assertRaises(ValueError):
                run_backtest(frame, HistoricalModelMustNotScore(), scores=invalid)


if __name__ == "__main__":
    unittest.main()
