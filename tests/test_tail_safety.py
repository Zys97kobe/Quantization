from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from quant_limitup.intraday_features import (
    INTRADAY_FEATURE_COLUMNS,
    build_tail_features,
    merge_tail_features,
    normalize_minute_bars,
)
from quant_limitup.model import LogisticLimitUpModel, train_logistic


TAIL = "tail_ret_1430_1457"


def full_session(date: str = "2026-09-21", symbol: str = "000001.SZ") -> pd.DataFrame:
    times = list(pd.date_range(f"{date} 09:35", f"{date} 11:30", freq="5min"))
    times += list(pd.date_range(f"{date} 13:05", f"{date} 14:55", freq="5min"))
    return pd.DataFrame({
        "datetime": times,
        "symbol": symbol,
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "volume": 100.0,
        "amount": 1000.0,
    })


def daily(date: str = "2026-09-21") -> pd.DataFrame:
    return pd.DataFrame({"date": [date], "symbol": ["000001.SZ"], "limit_up_price": [11.0]})


class TailFeatureSafetyTests(unittest.TestCase):
    def test_complete_session_excludes_lunch_and_post_cutoff_data(self) -> None:
        bars = full_session()
        reference = build_tail_features(bars, daily()).iloc[0]
        extra = bars.iloc[[0, 0]].copy()
        extra["datetime"] = pd.to_datetime(["2026-09-21 12:00", "2026-09-21 15:00"])
        extra[["volume", "amount"]] = 1e12
        result = build_tail_features(pd.concat([bars, extra]), daily()).iloc[0]
        self.assertEqual(result["tail_available"], 1)
        self.assertEqual(result["tail_observed_bars"], 47)
        self.assertEqual(result["tail_expected_bars"], 47)
        self.assertEqual(result["tail_coverage_ratio"], 1)
        self.assertAlmostEqual(result["tail_volume_ratio"], 6 / 47)
        self.assertEqual(result["tail_vwap_deviation"], 0)
        np.testing.assert_allclose(
            result[INTRADAY_FEATURE_COLUMNS].to_numpy(dtype=float),
            reference[INTRADAY_FEATURE_COLUMNS].to_numpy(dtype=float),
        )

    def test_truncated_session_is_unavailable_even_when_tail_is_present(self) -> None:
        result = build_tail_features(full_session().iloc[15:], daily()).iloc[0]
        self.assertEqual(result["tail_available"], 0)
        self.assertEqual(result["tail_unavailable_reason"], "incomplete_session")
        self.assertAlmostEqual(result["tail_coverage_ratio"], 32 / 47)
        self.assertTrue(result[INTRADAY_FEATURE_COLUMNS].isna().all())

    def test_duplicate_bars_cannot_inflate_coverage_or_volume(self) -> None:
        bars = full_session()
        duplicated = pd.concat([bars, bars.iloc[[10]]], ignore_index=True)
        normalized = normalize_minute_bars(duplicated)
        result = build_tail_features(normalized, daily()).iloc[0]
        self.assertEqual(result["tail_available"], 0)
        self.assertEqual(result["tail_unavailable_reason"], "duplicate_bars")
        self.assertEqual(result["tail_observed_bars"], 47)
        self.assertEqual(result["tail_duplicate_bars"], 2)
        self.assertTrue(result[INTRADAY_FEATURE_COLUMNS].isna().all())

    def test_off_grid_and_invalid_prices_are_rejected(self) -> None:
        bars = full_session()
        off_grid = bars.iloc[[0]].copy()
        off_grid["datetime"] = pd.to_datetime(["2026-09-21 14:54"])
        result = build_tail_features(pd.concat([bars, off_grid]), daily()).iloc[0]
        self.assertEqual(result["tail_unavailable_reason"], "unexpected_timestamps")
        bars.loc[0, "high"] = 9.0
        result = build_tail_features(bars, daily()).iloc[0]
        self.assertEqual(result["tail_unavailable_reason"], "invalid_bars")
        self.assertEqual(result["tail_available"], 0)

    def test_serialized_false_duplicate_markers_do_not_reject_clean_sessions(self) -> None:
        bars = full_session()
        bars["minute_duplicate"] = "False"
        result = build_tail_features(bars, daily()).iloc[0]
        self.assertEqual(result["tail_available"], 1)
        bars.loc[10, "minute_duplicate"] = "True"
        result = build_tail_features(bars, daily()).iloc[0]
        self.assertEqual(result["tail_unavailable_reason"], "duplicate_bars")

    def test_missing_minutes_are_distinct_from_valid_zero_returns(self) -> None:
        missing = merge_tail_features(daily(), None).iloc[0]
        valid_zero = merge_tail_features(daily(), full_session()).iloc[0]
        self.assertEqual(missing["tail_available"], 0)
        self.assertTrue(pd.isna(missing[TAIL]))
        self.assertEqual(missing["tail_unavailable_reason"], "no_minute_data")
        self.assertEqual(valid_zero["tail_available"], 1)
        self.assertEqual(valid_zero[TAIL], 0)

    def test_historical_tail_volume_baseline_excludes_current_day(self) -> None:
        first = full_session("2026-09-21")
        second = full_session("2026-09-22")
        tail = second["datetime"].dt.strftime("%H:%M") >= "14:30"
        second.loc[tail, ["volume", "amount"]] *= 2
        inputs = pd.concat([daily("2026-09-21"), daily("2026-09-22")], ignore_index=True)
        result = build_tail_features(pd.concat([second, first], ignore_index=True), inputs).sort_values("date")
        self.assertEqual(result.iloc[0]["tail_volume_vs_5d"], 1)
        expected = (12 / 53) / (6 / 47)
        self.assertAlmostEqual(result.iloc[1]["tail_volume_vs_5d"], expected)


class TailModelSafetyTests(unittest.TestCase):
    @staticmethod
    def training_frame(rows: int = 1000, dates: int = 20) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.date_range("2026-08-01", periods=dates).take(np.arange(rows) % dates),
            "daily_signal": np.arange(rows) % 3,
            TAIL: np.where(np.arange(rows) % 2, 11.0, 9.0),
            "tail_available": 1,
            "target_limit_up_next": np.arange(rows) % 2,
        })

    def test_default_gate_requires_both_dates_and_rows(self) -> None:
        for frame in [self.training_frame(1000, 1), self.training_frame(999, 20)]:
            with self.subTest(rows=len(frame), dates=frame.date.nunique()):
                model, metrics = train_logistic(frame, feature_columns=["daily_signal", TAIL], epochs=2)
                self.assertEqual(model.feature_columns, ["daily_signal"])
                self.assertFalse(metrics["tail_features"]["enabled"])
                self.assertEqual(metrics["tail_features"]["reason"], "insufficient_complete_history")
        model, metrics = train_logistic(self.training_frame(), feature_columns=["daily_signal", TAIL], epochs=2)
        self.assertIn(TAIL, model.feature_columns)
        self.assertTrue(metrics["tail_features"]["enabled"])

    def test_missing_metadata_and_unlabeled_dates_do_not_pass_gate(self) -> None:
        legacy = self.training_frame().drop(columns="tail_available")
        model, metrics = train_logistic(legacy, feature_columns=["daily_signal", TAIL], epochs=2)
        self.assertNotIn(TAIL, model.feature_columns)
        self.assertEqual(metrics["tail_features"]["reason"], "missing_coverage_metadata")
        frame = self.training_frame()
        frame.loc[frame.date == frame.date.max(), "target_limit_up_next"] = np.nan
        _, metrics = train_logistic(frame, feature_columns=["daily_signal", TAIL], epochs=2)
        self.assertEqual(metrics["tail_features"]["available_dates"], 19)
        self.assertFalse(metrics["tail_features"]["enabled"])

    def test_minute_scaling_uses_complete_rows_and_missing_prediction_is_neutral(self) -> None:
        valid = self.training_frame(4, 2)
        missing = self.training_frame(100, 2)
        missing["tail_available"] = 0
        missing[TAIL] = 0.0
        frame = pd.concat([valid, missing], ignore_index=True)
        model, _ = train_logistic(
            frame, feature_columns=["daily_signal", TAIL], epochs=60, tail_min_dates=2, tail_min_rows=4,
        )
        tail_idx = model.feature_columns.index(TAIL)
        self.assertEqual(model.mean[tail_idx], 10)
        self.assertEqual(model.std[tail_idx], 1)
        query = pd.DataFrame({
            "daily_signal": [1.0] * 4,
            TAIL: [0.0, 1e9, np.nan, 10.0],
            "tail_available": [0, 0, 0, 1],
        })
        prediction = model.predict_proba(query)
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(prediction, np.repeat(prediction[-1], len(prediction)))
        np.testing.assert_allclose(model.predict_proba(query.drop(columns=TAIL)), prediction)

    def test_model_metadata_round_trip_and_legacy_json_loading(self) -> None:
        model, _ = train_logistic(
            self.training_frame(4, 2), feature_columns=["daily_signal", TAIL],
            epochs=4, tail_min_dates=2, tail_min_rows=4,
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = LogisticLimitUpModel.load(path)
            self.assertEqual(loaded.metadata, model.metadata)
            np.testing.assert_allclose(loaded.predict_proba(self.training_frame()), model.predict_proba(self.training_frame()))
            payload = json.loads(path.read_text())
            payload.pop("metadata")
            path.write_text(json.dumps(payload))
            legacy = LogisticLimitUpModel.load(path)
            query = pd.DataFrame({"daily_signal": [1, 1], TAIL: [0.0, 1e9], "tail_available": [0, 1]})
            np.testing.assert_allclose(legacy.predict_proba(query)[0], legacy.predict_proba(query)[1])


if __name__ == "__main__":
    unittest.main()
