import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from quant_limitup import cli
from quant_limitup.messaging import blocked_buy_text


def coverage_report():
    stale = {
        "provider": "sina", "date": "2026-09-22", "expected_date": "2026-09-23",
        "status": "blocked", "eligible": False, "reasons": ["stale_price_snapshot"],
        "coverage_ratio": 0.9978, "covered_symbols": 5009, "expected_symbols": 5020,
        "exchanges": {"SH": {"coverage_ratio": 0.9973}, "SZ": {"coverage_ratio": 0.9982}},
    }
    partial = {
        **stale, "provider": "tencent", "date": "2026-09-23", "coverage_ratio": 0.7096,
        "covered_symbols": 3562, "reasons": ["total_coverage_below_threshold", "SH_coverage_below_threshold"],
        "exchanges": {"SH": {"coverage_ratio": 0.3496}, "SZ": {"coverage_ratio": 0.9989}},
    }
    return {**stale, "attempts": [partial, stale]}


class BlockedBuyNotificationTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(cli, "REPORT_DIR", self.root))
        stack.enter_context(patch.object(cli, "ensure_dirs"))
        self.config = SimpleNamespace(
            sender="sender@example.test", recipient="recipient@example.test", smtp_host="smtp.example.test",
            smtp_port=587, username="fixture", password="fixture", use_tls=True,
        )
        self.email_config = stack.enter_context(patch.object(cli, "load_email_config", return_value=self.config))
        self.webhook = stack.enter_context(patch.object(cli, "load_feishu_webhook", return_value="fixture-webhook"))
        # Exercise the real formatters and dispatchers without external sends.
        self.smtp = stack.enter_context(patch("quant_limitup.emailing.smtplib.SMTP"))
        self.post = stack.enter_context(patch("quant_limitup.messaging._post_feishu"))
        for name in ("build_dataset", "train_model", "run_paper_day", "load_account"):
            stack.enter_context(patch.object(cli, name, side_effect=AssertionError(f"unexpected {name}")))

    def report(self, **kwargs):
        return cli.report_blocked_buy(coverage_report(), "buy", **kwargs)[1]

    def test_both_channels_send_clear_reasons_without_missing_account_or_rank_fields(self):
        result = self.report(send_email=True, send_feishu=True)
        self.post.assert_called_once()
        text = self.post.call_args.args[1]["content"]["text"]
        self.assertIn("Buy 暂停买入 - 2026-09-23", text)
        for expected in ("腾讯", "70.96%", "34.96%", "新浪", "2026-09-22", "99.78%", "行情日期过期", "新增买入 0 只"):
            self.assertIn(expected, text)
        self.assertNotIn("资金池余额", text)
        self.assertNotIn("优化阈值", text)
        message = self.smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
        self.assertEqual(message["Subject"], "Buy 暂停买入 - 2026-09-23")
        self.assertEqual(message.get_content().strip(), text)
        self.assertEqual(result["notification_status"], {"email": "sent", "feishu": "sent"})
        saved = json.loads((self.root / "paper_daily_summary.json").read_text())
        self.assertEqual(saved["notification_status"], result["notification_status"])

    def test_disabled_channels_do_not_load_configuration_or_send(self):
        result = self.report()
        self.email_config.assert_not_called()
        self.webhook.assert_not_called()
        self.smtp.assert_not_called()
        self.post.assert_not_called()
        self.assertEqual(result["skipped"], "buy_data_coverage")

    def test_missing_email_configuration_does_not_suppress_feishu(self):
        self.email_config.side_effect = RuntimeError("fixture missing configuration")
        result = self.report(send_email=True, send_feishu=True)
        self.post.assert_called_once()
        self.assertEqual(result["notification_status"], {"email": "failed", "feishu": "sent"})
        self.assertEqual(result["opened_positions"], 0)

    def test_delivery_failure_preserves_blocked_report_and_records_failure(self):
        def failed_post(*args):
            saved = json.loads((self.root / "paper_daily_summary.json").read_text())
            self.assertEqual(saved["skipped"], "buy_data_coverage")
            raise RuntimeError("fixture delivery failure")
        self.post.side_effect = failed_post
        result = self.report(send_feishu=True)
        self.assertEqual(result["notification_status"], {"feishu": "failed"})
        self.assertEqual(result["opened_positions"], 0)
        saved = json.loads((self.root / "paper_daily_summary.json").read_text())
        self.assertEqual(saved["notification_status"], {"feishu": "failed"})

    def test_daily_buy_and_full_forward_notification_switches_on_fetch_rejection(self):
        for phase in ("buy", "full"):
            with self.subTest(phase=phase), \
                 patch.object(cli, "fetch_validated_buy_prices", side_effect=cli.BuyCoverageError(coverage_report())), \
                 patch.object(cli, "settle_after_blocked_buy", return_value=None):
                self.post.reset_mock()
                cli.main([
                    "daily", "--provider", "sina", "--paper", "--phase", phase, "--send-feishu",
                    "--feishu-webhook-file", str(self.root / "webhook.txt"),
                ])
                self.post.assert_called_once()
                self.webhook.assert_called_with(self.root / "webhook.txt")
                self.assertIn(f"阶段: {phase}", self.post.call_args.args[1]["content"]["text"])

    def test_paper_daily_forwards_notification_switches_on_local_price_rejection(self):
        with patch.object(cli, "read_prices", return_value=pd.DataFrame()), \
             patch.object(cli, "assess_buy_prices", return_value=coverage_report()):
            cli.main([
                "paper-daily", "--phase", "buy", "--send-email", "--send-feishu",
                "--email-config", str(self.root / "email.json"),
            ])
        self.post.assert_called_once()
        self.email_config.assert_called_once_with(self.root / "email.json")
        self.smtp.return_value.__enter__.return_value.send_message.assert_called_once()

    def test_full_message_distinguishes_settlement_from_new_buys(self):
        base = {"date": "2026-09-23", "phase": "full", "data_quality": coverage_report()}
        text = blocked_buy_text({**base, "closed_trades": 2})
        self.assertIn("本次已结算持仓: 2 笔", text)
        text = blocked_buy_text({**base, "settlement_skipped": "missing_valid_current_prices"})
        self.assertIn("保留账户等待重试", text)


if __name__ == "__main__":
    unittest.main()
