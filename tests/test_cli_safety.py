import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from quant_limitup import cli


class DailyMinuteFallbackTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(cli, "ensure_dirs"))
        self.minute = self.root / "minute.csv"
        self.stack.enter_context(patch.object(cli, "MINUTE_FILE", self.minute))
        self.validated = self.stack.enter_context(patch.object(cli, "fetch_validated_buy_prices"))
        self.probe = self.stack.enter_context(patch.object(cli, "sina_minute_market_is_current", return_value=True))
        self.fetch = self.stack.enter_context(patch.object(cli, "run_with_retries"))
        self.current = self.stack.enter_context(patch.object(cli, "minute_data_is_current", return_value=True))
        self.paper = self.stack.enter_context(patch.object(cli, "run_paper_from_prices"))
        self.args = [
            "daily", "--provider", "sina", "--paper", "--use-minute", "--minute-mode", "all",
            "--prices", str(self.root / "daily.csv"), "--symbols", str(self.root / "pool.csv"),
        ]

    def test_buy_fetch_failure_continues_with_validated_daily_data(self):
        self.fetch.side_effect = OSError("fixture minute outage")
        cli.main([*self.args, "--phase", "buy"])
        self.validated.assert_called_once()
        self.probe.assert_not_called()
        self.current.assert_not_called()
        self.paper.assert_called_once()
        self.assertIsNone(self.paper.call_args.args[1])

    def test_stale_snapshot_falls_back_to_daily_for_buy_and_full(self):
        self.current.return_value = False
        for phase in ["buy", "full"]:
            with self.subTest(phase=phase):
                self.paper.reset_mock()
                cli.main([*self.args, "--phase", phase])
                self.paper.assert_called_once()
                self.assertIsNone(self.paper.call_args.args[1])
        self.probe.assert_not_called()

    def test_current_minute_data_reaches_buy_pipeline(self):
        cli.main([*self.args, "--phase", "buy"])
        self.assertEqual(self.paper.call_args.args[1], self.minute)

    def test_sell_keeps_minute_requirement_on_stale_data(self):
        self.current.return_value = False
        with patch.object(cli, "paper_account_has_positions", return_value=True), \
             patch.object(cli, "build_position_symbol_pool", return_value=self.root / "positions.csv"):
            cli.main([*self.args, "--phase", "sell-force"])
        self.probe.assert_called_once()
        self.validated.assert_not_called()
        self.paper.assert_not_called()

    def test_sell_fetch_failure_does_not_fall_back_to_buy_pipeline(self):
        self.fetch.side_effect = [None, OSError("fixture minute outage")]
        with patch.object(cli, "paper_account_has_positions", return_value=True), \
             patch.object(cli, "build_position_symbol_pool", return_value=self.root / "positions.csv"), \
             self.assertRaises(OSError):
            cli.main([*self.args, "--phase", "sell-force"])
        self.paper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
