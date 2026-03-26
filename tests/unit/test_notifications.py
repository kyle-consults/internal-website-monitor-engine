import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.notify import (
    NotificationError,
    build_email_subject,
    build_email_text,
    build_resend_payload,
    parse_recipients,
    send_notification,
    should_send_email,
)
from website_monitor.monitor import MonitorPaths


BASE_SUMMARY = {
    "homepage_url": "https://example.com/",
    "scanned_at": "2026-03-26T00:00:00+00:00",
    "baseline_created": False,
    "changes_detected": False,
    "counts": {
        "pages_scanned": 5,
        "added": 0,
        "removed": 0,
        "changed": 0,
    },
}


class NotificationCoreTests(unittest.TestCase):
    def test_should_send_email_defaults_to_changes_only(self) -> None:
        self.assertTrue(should_send_email({**BASE_SUMMARY, "baseline_created": True}, "changes_only"))
        self.assertTrue(should_send_email({**BASE_SUMMARY, "changes_detected": True}, "changes_only"))
        self.assertFalse(should_send_email(BASE_SUMMARY, "changes_only"))

    def test_should_send_email_supports_always_and_never_modes(self) -> None:
        self.assertTrue(should_send_email(BASE_SUMMARY, "always"))
        self.assertFalse(should_send_email({**BASE_SUMMARY, "baseline_created": True}, "never"))

    def test_parse_recipients_splits_and_trims_comma_separated_values(self) -> None:
        recipients = parse_recipients("one@example.com, two@example.com ,three@example.com")

        self.assertEqual(
            recipients,
            ["one@example.com", "two@example.com", "three@example.com"],
        )

    def test_build_email_subject_reflects_scan_state(self) -> None:
        baseline_subject = build_email_subject({**BASE_SUMMARY, "baseline_created": True})
        changed_subject = build_email_subject({**BASE_SUMMARY, "changes_detected": True})
        unchanged_subject = build_email_subject(BASE_SUMMARY)

        self.assertIn("Baseline established", baseline_subject)
        self.assertIn("Changes detected", changed_subject)
        self.assertIn("No changes", unchanged_subject)

    def test_build_email_text_includes_summary_and_report(self) -> None:
        text = build_email_text(
            {**BASE_SUMMARY, "changes_detected": True, "counts": {**BASE_SUMMARY["counts"], "changed": 2}},
            "# Website Change Report\n\n## Changed\n- https://example.com/\n",
        )

        self.assertIn("Homepage: https://example.com/", text)
        self.assertIn("Changed pages: 2", text)
        self.assertIn("# Website Change Report", text)

    def test_build_resend_payload_uses_sender_recipients_and_text_body(self) -> None:
        payload = build_resend_payload(
            {**BASE_SUMMARY, "changes_detected": True},
            report_text="# Website Change Report",
            sender="Website Monitor <alerts@example.com>",
            recipients=["one@example.com", "two@example.com"],
        )

        self.assertEqual(payload["from"], "Website Monitor <alerts@example.com>")
        self.assertEqual(payload["to"], ["one@example.com", "two@example.com"])
        self.assertIn("Changes detected", payload["subject"])
        self.assertIn("# Website Change Report", payload["text"])

    def test_send_notification_skips_when_configuration_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.make_paths(tmpdir)

            result = send_notification(
                paths=paths,
                env={"ALERT_EMAIL_TO": "one@example.com"},
                send_fn=lambda api_key, payload: self.fail("send_fn should not be called"),
            )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "email_not_configured")

    def test_send_notification_sends_for_baseline_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.make_paths(tmpdir, summary={**BASE_SUMMARY, "baseline_created": True})

            result = send_notification(
                paths=paths,
                env={
                    "ALERT_EMAIL_TO": "one@example.com",
                    "ALERT_EMAIL_FROM": "Website Monitor <alerts@example.com>",
                    "RESEND_API_KEY": "test-key",
                },
                send_fn=lambda api_key, payload: {"id": "email-1", "payload": payload},
            )

        self.assertEqual(result["status"], "sent")
        self.assertIn("Baseline established", result["payload"]["subject"])

    def test_send_notification_sends_for_changed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.make_paths(
                tmpdir,
                summary={
                    **BASE_SUMMARY,
                    "changes_detected": True,
                    "counts": {**BASE_SUMMARY["counts"], "changed": 1},
                },
            )

            result = send_notification(
                paths=paths,
                env={
                    "ALERT_EMAIL_TO": "one@example.com",
                    "ALERT_EMAIL_FROM": "Website Monitor <alerts@example.com>",
                    "RESEND_API_KEY": "test-key",
                },
                send_fn=lambda api_key, payload: {"id": "email-2", "payload": payload},
            )

        self.assertEqual(result["status"], "sent")
        self.assertIn("Changes detected", result["payload"]["subject"])

    def test_send_notification_skips_unchanged_runs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.make_paths(tmpdir)

            result = send_notification(
                paths=paths,
                env={
                    "ALERT_EMAIL_TO": "one@example.com",
                    "ALERT_EMAIL_FROM": "Website Monitor <alerts@example.com>",
                    "RESEND_API_KEY": "test-key",
                },
                send_fn=lambda api_key, payload: self.fail("send_fn should not be called"),
            )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "policy_no_send")

    def test_send_notification_wraps_provider_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.make_paths(tmpdir)

            with self.assertRaises(NotificationError):
                send_notification(
                    paths=paths,
                    env={
                        "ALERT_EMAIL_TO": "one@example.com",
                        "ALERT_EMAIL_FROM": "Website Monitor <alerts@example.com>",
                        "EMAIL_MODE": "always",
                        "RESEND_API_KEY": "test-key",
                    },
                    send_fn=lambda api_key, payload: (_ for _ in ()).throw(RuntimeError("boom")),
                )

    def make_paths(self, tmpdir: str, summary: dict | None = None) -> MonitorPaths:
        root = Path(tmpdir)
        (root / "config").mkdir()
        (root / "reports").mkdir()
        (root / "snapshots").mkdir()
        (root / "config" / "defaults.json").write_text("{}", encoding="utf-8")
        (root / "reports" / "latest-summary.json").write_text(
            json.dumps(summary or BASE_SUMMARY),
            encoding="utf-8",
        )
        (root / "reports" / "latest-report.md").write_text("# Website Change Report", encoding="utf-8")
        return MonitorPaths.for_root(root)


if __name__ == "__main__":
    unittest.main()
