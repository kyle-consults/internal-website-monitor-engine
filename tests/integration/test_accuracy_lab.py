import json
import sys
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.monitor import MonitorPaths, run_monitor  # noqa: E402


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return


class AccuracyLabIntegrationTests(unittest.TestCase):
    """Run the real crawler against deterministic local pages.

    These fixtures simulate failure modes that are hard to reproduce safely on
    live customer sites: volatile widgets, query-scoped pages, and exact
    operational values such as appointment hours.
    """

    @classmethod
    def setUpClass(cls) -> None:
        fixture_dir = ROOT / "tests" / "fixtures" / "accuracy-lab"
        handler = partial(QuietHandler, directory=str(fixture_dir))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

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
                    "request_timeout_ms": 10000,
                    "archive_retention": 3,
                    "exclude_extensions": [".pdf", ".png", ".jpg"],
                    "exclude_url_contains": [],
                    "keep_url_query_params": ["zip"],
                    "content_include_selectors": [".monitor-content"],
                    "content_exclude_selectors": [".dynamic-widget", ".rotating-promo"],
                    "ignore_text_patterns": [
                        r"Visitor count: \d+",
                        r"Rendered at: \d{1,2}:\d{2}:\d{2}",
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.paths = MonitorPaths.for_root(self.root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_accuracy_lab_filters_dynamic_noise_and_keeps_operational_facts(self) -> None:
        result = run_monitor(
            paths=self.paths,
            env={
                "HOMEPAGE_URL": (
                    f"{self.base_url}/index.html?zip=95050&utm_source=ad"
                )
            },
            archive_timestamp="accuracy-lab",
        )

        pages = result["current"]["pages"]
        home_url = f"{self.base_url}/index.html?zip=95050"
        hours_url = f"{self.base_url}/hours.html"

        self.assertIn(home_url, pages)
        self.assertIn(hours_url, pages)
        self.assertEqual(result["current"]["homepage_url"], home_url)

        home_text = pages[home_url]["text"]
        self.assertIn("Clinic Alpha", home_text)
        self.assertIn("Phone: 555-0100", home_text)
        self.assertIn("Hours: 8:30 AM - 5:00 PM", home_text)
        self.assertNotIn("Visitor count", home_text)
        self.assertNotIn("Rendered at", home_text)
        self.assertNotIn("Flash sale", home_text)

        hours_text = pages[hours_url]["text"]
        self.assertIn("Weekday Hours: Monday-Friday 8:30 AM - 5:00 PM", hours_text)
        self.assertNotIn("Widget random", hours_text)

    def test_accuracy_lab_second_run_with_dynamic_noise_has_no_changes(self) -> None:
        env = {
            "HOMEPAGE_URL": f"{self.base_url}/index.html?zip=95050&utm_source=ad",
        }

        first = run_monitor(
            paths=self.paths,
            env=env,
            archive_timestamp="accuracy-lab-1",
        )
        second = run_monitor(
            paths=self.paths,
            env=env,
            archive_timestamp="accuracy-lab-2",
        )

        self.assertTrue(first["baseline_created"])
        self.assertFalse(second["baseline_created"])
        self.assertFalse(second["summary"]["changes_detected"])
        self.assertEqual(second["diff"]["added"], [])
        self.assertEqual(second["diff"]["removed"], [])
        self.assertEqual(second["diff"]["changed"], [])


if __name__ == "__main__":
    unittest.main()
