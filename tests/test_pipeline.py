import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from quant_limitup.backtest import rank_candidates, run_backtest
from quant_limitup.data import read_prices, write_sample_prices
from quant_limitup.features import build_feature_frame
from quant_limitup.model import train_logistic

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


if __name__ == "__main__":
    unittest.main()
