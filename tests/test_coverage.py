import json
import unittest

import pandas as pd

from quant_limitup.coverage import assess_price_coverage, historical_coverage_by_date


def _pool(symbols):
    return pd.DataFrame([
        {"symbol": symbol, "name": f"公司{index}", "board": "main", "is_st": 0}
        for index, symbol in enumerate(symbols)
    ])


def _bar(symbol, date="2026-09-21", **updates):
    row = {
        "date": date, "symbol": symbol, "open": 10.0, "high": 11.0,
        "low": 9.0, "close": 10.5, "volume": 100.0, "amount": 1050.0,
    }
    row.update(updates)
    return row


class PriceCoverageTest(unittest.TestCase):
    def test_missing_exchange_blocks_despite_good_total_coverage(self):
        symbols = [f"{index:06d}.SZ" for index in range(19)] + ["600001.SH"]
        result = assess_price_coverage(pd.DataFrame([_bar(s) for s in symbols[:-1]]), _pool(symbols))
        self.assertEqual(result["expected_symbols"], 20)
        self.assertEqual(result["covered_symbols"], 19)
        self.assertEqual(result["coverage_ratio"], 0.95)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("SH_coverage_below_threshold", result["reasons"])
        self.assertEqual(result["exchanges"]["SH"]["coverage_ratio"], 0.0)

    def test_intentional_single_exchange_pool_is_supported(self):
        pool = _pool(["000001.SZ", "000002.SZ"])
        result = assess_price_coverage(pd.DataFrame([_bar(s) for s in pool.symbol]), pool, "2026-09-21")
        self.assertTrue(result["eligible"])
        self.assertEqual(set(result["exchanges"]), {"SZ"})
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_last_duplicate_and_invalid_bars_do_not_inflate_coverage(self):
        pool = _pool(["000001.SZ", "000002.SZ", "600001.SH"])
        prices = pd.DataFrame([
            _bar("000001.SZ"), _bar("000001.SZ"),
            _bar("000002.SZ"), _bar("000002.SZ", volume=-1),
            _bar("600001.SH", close=float("inf")), _bar("600999.SH"),
        ])
        result = assess_price_coverage(prices, pool)
        self.assertEqual(result["expected_symbols"], 3)
        self.assertEqual(result["covered_symbols"], 1)
        self.assertFalse(result["eligible"])

    def test_validity_rejects_bad_ohlc_and_negative_amount_but_accepts_zero_volume(self):
        pool = _pool(["000001.SZ"])
        for updates in ({"low": 12}, {"high": 9}, {"open": 0}, {"close": None}, {"amount": -1}):
            with self.subTest(updates=updates):
                result = assess_price_coverage(pd.DataFrame([_bar("000001.SZ", **updates)]), pool)
                self.assertEqual(result["covered_symbols"], 0)
        result = assess_price_coverage(pd.DataFrame([_bar("000001.SZ", volume=0, amount=0)]), pool)
        self.assertTrue(result["eligible"])

    def test_stale_and_future_snapshot_are_blocked(self):
        pool = _pool(["000001.SZ"])
        prices = pd.DataFrame([_bar("000001.SZ")])
        stale = assess_price_coverage(prices, pool, "2026-09-22")
        future = assess_price_coverage(prices, pool, "2026-09-18")
        self.assertIn("stale_price_snapshot", stale["reasons"])
        self.assertIn("future_price_snapshot", future["reasons"])
        self.assertEqual(stale["date"], "2026-09-21")
        self.assertEqual(stale["expected_date"], "2026-09-22")

    def test_stale_symbol_rows_cannot_fill_current_date_gaps(self):
        pool = _pool(["000001.SZ", "600001.SH"])
        prices = pd.DataFrame([_bar("000001.SZ"), _bar("600001.SH", "2026-09-18")])
        result = assess_price_coverage(prices, pool, "2026-09-21")
        self.assertEqual(result["covered_symbols"], 1)
        self.assertFalse(result["eligible"])

    def test_non_tradeable_pool_entries_are_excluded_and_pool_is_deduplicated(self):
        pool = _pool(["000001.SZ", "000001.SZ", "600001.SH", "600002.SH", "430001.BJ", "600003.SH"])
        pool.loc[2, "name"] = "*ST公司"
        pool.loc[3, "is_st"] = 1
        pool.loc[5, "name"] = "退市公司"
        result = assess_price_coverage(pd.DataFrame([_bar("000001.SZ")]), pool)
        self.assertEqual(result["expected_symbols"], 1)
        self.assertTrue(result["eligible"])

    def test_empty_or_malformed_inputs_fail_closed(self):
        pool = _pool(["000001.SZ"])
        for prices, expected in [
            (pd.DataFrame(), pool),
            (pd.DataFrame([_bar("000001.SZ")]), pd.DataFrame()),
            (pd.DataFrame([{"symbol": "000001.SZ", "date": "2026-09-21", "close": 10}]), pool),
            (pd.DataFrame([_bar("000001.SZ", "not-a-date")]), pool),
        ]:
            with self.subTest(prices=prices.to_dict("records")):
                result = assess_price_coverage(prices, expected)
                self.assertFalse(result["eligible"])
                json.dumps(result, allow_nan=False)

    def test_historical_coverage_uses_same_explicit_pool_for_each_date(self):
        pool = _pool(["000001.SZ", "600001.SH"])
        prices = pd.DataFrame([
            _bar("000001.SZ", "2026-09-18"), _bar("600001.SH", "2026-09-18"),
            _bar("000001.SZ"), _bar("000001.SZ"),
        ])
        original_prices, original_pool = prices.copy(deep=True), pool.copy(deep=True)
        result = historical_coverage_by_date(prices, pool)
        self.assertEqual(result.market_data_eligible.tolist(), [True, False])
        self.assertEqual(result.market_coverage_ratio.tolist(), [1.0, 0.5])
        self.assertEqual(result.market_expected_symbols.tolist(), [2, 2])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.date))
        pd.testing.assert_frame_equal(prices, original_prices)
        pd.testing.assert_frame_equal(pool, original_pool)

    def test_empty_history_has_mergeable_date_and_boolean_schema(self):
        result = historical_coverage_by_date(pd.DataFrame(), _pool(["000001.SZ"]))
        self.assertTrue(result.empty)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.date))
        self.assertTrue(pd.api.types.is_bool_dtype(result.market_data_eligible))

    def test_threshold_boundaries_and_validation(self):
        pool = _pool(["000001.SZ", "000002.SZ"])
        prices = pd.DataFrame([_bar("000001.SZ")])
        self.assertTrue(assess_price_coverage(prices, pool, min_total_coverage=0.5, min_exchange_coverage=0.5)["eligible"])
        for threshold in (-0.1, 1.1, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                assess_price_coverage(prices, pool, min_total_coverage=threshold)


if __name__ == "__main__":
    unittest.main()
