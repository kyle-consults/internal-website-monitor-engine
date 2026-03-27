import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.monitor import MonitorPaths, run_monitor


def make_snapshot(homepage_url: str, scanned_at: str, pages: dict[str, dict]) -> dict:
    return {
        "homepage_url": homepage_url,
        "scanned_at": scanned_at,
        "pages": pages,
    }


class MonitorRunnerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "config").mkdir()
        (self.root / "reports").mkdir()
        (self.root / "snapshots").mkdir()
        (self.root / "config" / "defaults.json").write_text(
            json.dumps(
                {
                    "max_pages": 10,
                    "request_timeout_ms": 5000,
                    "archive_retention": 3,
                    "exclude_extensions": [".pdf"],
                    "exclude_url_contains": ["/login"],
                }
            ),
            encoding="utf-8",
        )
        self.paths = MonitorPaths.for_root(self.root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_first_run_creates_baseline_snapshot_and_report(self) -> None:
        current_snapshot = make_snapshot(
            homepage_url="https://example.com",
            scanned_at="2026-03-25T00:00:00+00:00",
            pages={
                "https://example.com/": {
                    "url": "https://example.com/",
                    "title": "Home",
                    "h1": "Welcome",
                    "text": "Hello world",
                    "hash": "home",
                    "status": 200,
                }
            },
        )

        result = run_monitor(
            paths=self.paths,
            env={"HOMEPAGE_URL": "https://example.com"},
            crawl_fn=lambda homepage_url, cfg: current_snapshot,
            archive_timestamp="2026-03-25T00-00-00Z",
        )

        self.assertTrue(result["baseline_created"])
        self.assertEqual(result["diff"]["added"], ["https://example.com/"])
        self.assertTrue(self.paths.latest_snapshot.exists())
        self.assertTrue(self.paths.latest_report.exists())
        self.assertTrue((self.paths.reports_dir / "latest-summary.json").exists())
        latest_summary = json.loads((self.paths.reports_dir / "latest-summary.json").read_text(encoding="utf-8"))
        self.assertTrue(latest_summary["baseline_created"])
        self.assertTrue(latest_summary["changes_detected"])
        self.assertEqual(
            latest_summary["counts"],
            {
                "pages_scanned": 1,
                "added": 1,
                "removed": 0,
                "changed": 0,
            },
        )
        self.assertIn("Initial baseline established.", self.paths.latest_report.read_text(encoding="utf-8"))

    def test_second_run_detects_added_removed_and_changed_pages(self) -> None:
        first_snapshot = make_snapshot(
            homepage_url="https://example.com",
            scanned_at="2026-03-25T00:00:00+00:00",
            pages={
                "https://example.com/": {
                    "url": "https://example.com/",
                    "title": "Home",
                    "h1": "Welcome",
                    "text": "Hello world",
                    "hash": "home-v1",
                    "status": 200,
                },
                "https://example.com/pricing": {
                    "url": "https://example.com/pricing",
                    "title": "Pricing",
                    "h1": "Pricing",
                    "text": "Old pricing",
                    "hash": "pricing-v1",
                    "status": 200,
                },
            },
        )
        second_snapshot = make_snapshot(
            homepage_url="https://example.com",
            scanned_at="2026-03-26T00:00:00+00:00",
            pages={
                "https://example.com/": {
                    "url": "https://example.com/",
                    "title": "Home Updated",
                    "h1": "Welcome Back",
                    "text": "Hello world. Pricing starts at $12. Chat with us today.",
                    "hash": "home-v2",
                    "status": 200,
                },
                "https://example.com/contact": {
                    "url": "https://example.com/contact",
                    "title": "Contact",
                    "h1": "Contact",
                    "text": "Talk to us",
                    "hash": "contact-v1",
                    "status": 200,
                },
            },
        )

        run_monitor(
            paths=self.paths,
            env={"HOMEPAGE_URL": "https://example.com"},
            crawl_fn=lambda homepage_url, cfg: first_snapshot,
            archive_timestamp="2026-03-25T00-00-00Z",
        )
        result = run_monitor(
            paths=self.paths,
            env={"HOMEPAGE_URL": "https://example.com"},
            crawl_fn=lambda homepage_url, cfg: second_snapshot,
            archive_timestamp="2026-03-26T00-00-00Z",
        )

        report_text = self.paths.latest_report.read_text(encoding="utf-8")
        latest_summary = json.loads((self.paths.reports_dir / "latest-summary.json").read_text(encoding="utf-8"))

        self.assertFalse(result["baseline_created"])
        self.assertEqual(result["diff"]["added"], ["https://example.com/contact"])
        self.assertEqual(result["diff"]["removed"], ["https://example.com/pricing"])
        self.assertEqual(result["diff"]["changed"], ["https://example.com/"])
        self.assertFalse(latest_summary["baseline_created"])
        self.assertTrue(latest_summary["changes_detected"])
        self.assertEqual(
            latest_summary["counts"],
            {
                "pages_scanned": 2,
                "added": 1,
                "removed": 1,
                "changed": 1,
            },
        )
        self.assertIn("## Added", report_text)
        self.assertIn("- https://example.com/contact | status: 200 | title: Contact", report_text)
        self.assertIn("## Removed", report_text)
        self.assertIn("- https://example.com/pricing | status: 200 | title: Pricing", report_text)
        self.assertIn("## Changed", report_text)
        self.assertNotIn("## All Pages Scraped", report_text)
        self.assertIn("### https://example.com/", report_text)
        self.assertIn('- Title changed: "Home" -> "Home Updated"', report_text)
        self.assertIn('- H1 changed: "Welcome" -> "Welcome Back"', report_text)
        self.assertIn('- Text modified: "Hello world" -> "Hello world."', report_text)
        self.assertIn("- Text added: Chat with us today.", report_text)

    def test_unchanged_second_run_does_not_persist_new_outputs(self) -> None:
        snapshot = make_snapshot(
            homepage_url="https://example.com",
            scanned_at="2026-03-25T00:00:00+00:00",
            pages={
                "https://example.com/": {
                    "url": "https://example.com/",
                    "title": "Home",
                    "h1": "Welcome",
                    "text": "Hello world",
                    "hash": "home-v1",
                    "status": 200,
                }
            },
        )

        first_result = run_monitor(
            paths=self.paths,
            env={"HOMEPAGE_URL": "https://example.com"},
            crawl_fn=lambda homepage_url, cfg: snapshot,
            archive_timestamp="2026-03-25T00-00-00Z",
        )
        original_report = self.paths.latest_report.read_text(encoding="utf-8")
        original_snapshot = self.paths.latest_snapshot.read_text(encoding="utf-8")
        original_summary = (self.paths.reports_dir / "latest-summary.json").read_text(encoding="utf-8")

        second_result = run_monitor(
            paths=self.paths,
            env={"HOMEPAGE_URL": "https://example.com"},
            crawl_fn=lambda homepage_url, cfg: {
                **snapshot,
                "scanned_at": "2026-03-26T00:00:00+00:00",
            },
            archive_timestamp="2026-03-26T00-00-00Z",
        )

        snapshot_archives = sorted(path.name for path in self.paths.snapshots_dir.glob("snapshot-*.json"))
        report_archives = sorted(path.name for path in self.paths.reports_dir.glob("report-*.md"))
        summary_archives = sorted(path.name for path in self.paths.reports_dir.glob("summary-*.json"))

        self.assertTrue(first_result["persisted"])
        self.assertFalse(second_result["persisted"])
        self.assertEqual(self.paths.latest_report.read_text(encoding="utf-8"), original_report)
        self.assertEqual(self.paths.latest_snapshot.read_text(encoding="utf-8"), original_snapshot)
        self.assertEqual((self.paths.reports_dir / "latest-summary.json").read_text(encoding="utf-8"), original_summary)
        self.assertEqual(snapshot_archives, ["snapshot-2026-03-25T00-00-00Z.json"])
        self.assertEqual(report_archives, ["report-2026-03-25T00-00-00Z.md"])
        self.assertEqual(summary_archives, ["summary-2026-03-25T00-00-00Z.json"])


if __name__ == "__main__":
    unittest.main()
