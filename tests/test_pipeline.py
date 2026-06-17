import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from quant_limitup.accuracy import write_prediction_accuracy
from quant_limitup.backtest import rank_candidates, run_backtest
from quant_limitup.data import read_prices, write_sample_prices
from quant_limitup.features import build_feature_frame
from quant_limitup.model import train_logistic
from quant_limitup.strategy import build_learning_report

class PipelineTest(unittest.TestCase):
    def test_sample_pipeline_runs(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        price_file = Path(tmp.name) / "daily_prices.csv"
        write_sample_prices(price_file, symbols=24, days=90)
        prices = read_prices(price_file)
        features = build_feature_frame(prices)

        self.assertFalse(features.empty)
        self.assertTrue({"target_limit_up_next", "volume_ratio_5", "market_limit_hits"}.issubset(features.columns))

        model, metrics = train_logistic(features, epochs=200)
        self.assertEqual(metrics["rows"], int(features["target_limit_up_next"].notna().sum()))

        rank = rank_candidates(features, model)
        self.assertFalse(rank.empty)
        self.assertTrue(rank["score"].is_monotonic_decreasing)

        trades, summary = run_backtest(features, model)
        self.assertIn("total_return", summary)
        self.assertIsInstance(trades, pd.DataFrame)

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


if __name__ == "__main__":
    unittest.main()
