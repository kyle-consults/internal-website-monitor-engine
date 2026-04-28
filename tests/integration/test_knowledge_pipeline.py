import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.monitor import MonitorPaths, run_monitor  # noqa: E402

HOMEPAGE = "https://example.com"
SCAN_TIME_1 = "2026-04-01T00:00:00+00:00"
SCAN_TIME_2 = "2026-04-02T00:00:00+00:00"


def _make_snapshot(pages, scanned_at=SCAN_TIME_1):
    return {
        "homepage_url": HOMEPAGE,
        "scanned_at": scanned_at,
        "pages": pages,
    }


def _make_knowledge(pages, extracted_at=SCAN_TIME_1):
    return {
        "schema_version": 1,
        "homepage_url": HOMEPAGE,
        "extracted_at": extracted_at,
        "model": "gemini-2.0-flash-lite",
        "pages": pages,
    }


class TestKnowledgePipelineBaselineAndChangeDetection(unittest.TestCase):
    """Two runs: first creates baseline, second detects changes."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "config").mkdir()
        (self.root / "reports").mkdir()
        (self.root / "snapshots").mkdir()
        (self.root / "config" / "defaults.json").write_text(
            json.dumps({
                "max_pages": 10,
                "request_timeout_ms": 5000,
                "archive_retention": 3,
                "gemini_model": "gemini-2.0-flash-lite",
            }),
            encoding="utf-8",
        )
        self.paths = MonitorPaths.for_root(self.root)
        self.env = {"HOMEPAGE_URL": HOMEPAGE}

    def tearDown(self):
        self.tempdir.cleanup()

    def test_baseline_and_change_detection(self):
        page_v1 = {
            "url": f"{HOMEPAGE}/",
            "title": "Home",
            "h1": "Welcome",
            "text": "Our office hours are 9am to 5pm.",
            "hash": "hash_v1",
            "status": 200,
        }
        snapshot_v1 = _make_snapshot({f"{HOMEPAGE}/": page_v1})

        knowledge_v1 = _make_knowledge({
            f"{HOMEPAGE}/": {
                "url": f"{HOMEPAGE}/",
                "knowledge_units": [
                    {"label": "Office Hours", "value": "9am to 5pm", "category": "operations", "operational": True},
                ],
            }
        })

        mock_client = MagicMock()

        # --- Run 1: baseline ---
        with patch("website_monitor.monitor.extract_all_pages", return_value=knowledge_v1) as mock_extract, \
             patch("website_monitor.monitor.compare_knowledge", return_value={"added": [], "removed": [], "changed": [], "unchanged": []}):
            result1 = run_monitor(
                paths=self.paths,
                env=self.env,
                crawl_fn=lambda _url, _cfg: snapshot_v1,
                archive_timestamp="run1",
                gemini_client=mock_client,
            )

        self.assertTrue(result1["baseline_created"])
        self.assertTrue(result1["persisted"])
        self.assertTrue(self.paths.latest_knowledge.exists())
        mock_extract.assert_called_once()

        # --- Run 2: changed page ---
        page_v2 = dict(page_v1)
        page_v2["text"] = "Our office hours are 8am to 6pm."
        page_v2["hash"] = "hash_v2"
        snapshot_v2 = _make_snapshot({f"{HOMEPAGE}/": page_v2}, scanned_at=SCAN_TIME_2)

        knowledge_v2 = _make_knowledge({
            f"{HOMEPAGE}/": {
                "url": f"{HOMEPAGE}/",
                "knowledge_units": [
                    {"label": "Office Hours", "value": "8am to 6pm", "category": "operations", "operational": True},
                ],
            }
        }, extracted_at=SCAN_TIME_2)

        change_diff = {
            "added": [],
            "removed": [],
            "changed": [
                {"page": f"{HOMEPAGE}/", "label": "Office Hours", "old_value": "9am to 5pm", "new_value": "8am to 6pm"},
            ],
            "unchanged": [],
        }

        with patch("website_monitor.monitor.extract_all_pages", return_value=knowledge_v2), \
             patch("website_monitor.monitor.compare_knowledge", return_value=change_diff), \
             patch("website_monitor.monitor.quorum_verify_changes", side_effect=lambda diff, **kwargs: diff), \
             patch("website_monitor.monitor.verify_changes", side_effect=lambda diff, *a, **kw: diff):
            result2 = run_monitor(
                paths=self.paths,
                env=self.env,
                crawl_fn=lambda _url, _cfg: snapshot_v2,
                archive_timestamp="run2",
                gemini_client=mock_client,
            )

        self.assertFalse(result2["baseline_created"])
        self.assertTrue(result2["summary"].get("changes_detected"))
        self.assertTrue(result2["persisted"])

        # Report should mention the changed knowledge unit
        report_path = self.paths.reports_dir / "report-run2.md"
        self.assertTrue(report_path.exists())
        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("Office Hours", report_text)


class TestFallsBackToRawDiffWithoutGeminiKey(unittest.TestCase):
    """Without GEMINI_API_KEY or gemini_client, raw diff path is used."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "config").mkdir()
        (self.root / "reports").mkdir()
        (self.root / "snapshots").mkdir()
        (self.root / "config" / "defaults.json").write_text(
            json.dumps({
                "max_pages": 10,
                "request_timeout_ms": 5000,
                "archive_retention": 3,
            }),
            encoding="utf-8",
        )
        self.paths = MonitorPaths.for_root(self.root)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_no_knowledge_file_created(self):
        page_v1 = {
            "url": f"{HOMEPAGE}/",
            "title": "Home",
            "h1": "Welcome",
            "text": "Hello world",
            "hash": "hash_a",
            "status": 200,
        }
        snapshot_v1 = _make_snapshot({f"{HOMEPAGE}/": page_v1})

        # First run: baseline (no Gemini key in env)
        env = {"HOMEPAGE_URL": HOMEPAGE}
        result1 = run_monitor(
            paths=self.paths,
            env=env,
            crawl_fn=lambda _url, _cfg: snapshot_v1,
            archive_timestamp="run1",
        )

        self.assertTrue(result1["baseline_created"])
        self.assertTrue(result1["persisted"])
        # No knowledge file should exist
        self.assertFalse(self.paths.latest_knowledge.exists())
        # Raw diff report should be generated
        self.assertTrue(self.paths.latest_report.exists())
        report_text = self.paths.latest_report.read_text(encoding="utf-8")
        self.assertIn("Website Change Report", report_text)

        # Second run: changed page, detect via hash comparison
        page_v2 = dict(page_v1)
        page_v2["text"] = "Hello updated world with substantial new content that differs."
        page_v2["hash"] = "hash_b"
        snapshot_v2 = _make_snapshot({f"{HOMEPAGE}/": page_v2}, scanned_at=SCAN_TIME_2)

        result2 = run_monitor(
            paths=self.paths,
            env=env,
            crawl_fn=lambda _url, _cfg: snapshot_v2,
            verify_fn=lambda _urls, _cfg: {},
            archive_timestamp="run2",
        )

        self.assertFalse(result2["baseline_created"])
        self.assertTrue(result2["summary"]["changes_detected"])
        self.assertFalse(self.paths.latest_knowledge.exists())


class TestKnowledgeArchivesArePruned(unittest.TestCase):
    """Pre-seed 4 knowledge archive files, run with retention=3."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "config").mkdir()
        (self.root / "reports").mkdir()
        (self.root / "snapshots").mkdir()
        (self.root / "config" / "defaults.json").write_text(
            json.dumps({
                "max_pages": 10,
                "request_timeout_ms": 5000,
                "archive_retention": 3,
                "gemini_model": "gemini-2.0-flash-lite",
            }),
            encoding="utf-8",
        )
        self.paths = MonitorPaths.for_root(self.root)
        self.env = {"HOMEPAGE_URL": HOMEPAGE}

    def tearDown(self):
        self.tempdir.cleanup()

    def test_prunes_old_knowledge_archives(self):
        snapshots_dir = self.root / "snapshots"

        # Pre-seed 4 knowledge archive files
        for i in range(4):
            archive = snapshots_dir / f"knowledge-2026-04-0{i+1}T00-00-00Z.json"
            archive.write_text(json.dumps({"seeded": i}), encoding="utf-8")

        page = {
            "url": f"{HOMEPAGE}/",
            "title": "Home",
            "h1": "Welcome",
            "text": "Content here.",
            "hash": "hash_x",
            "status": 200,
        }
        snapshot = _make_snapshot({f"{HOMEPAGE}/": page})
        knowledge = _make_knowledge({
            f"{HOMEPAGE}/": {
                "url": f"{HOMEPAGE}/",
                "knowledge_units": [
                    {"label": "Info", "value": "content", "category": "general", "operational": False},
                ],
            }
        })

        mock_client = MagicMock()

        with patch("website_monitor.monitor.extract_all_pages", return_value=knowledge), \
             patch("website_monitor.monitor.compare_knowledge", return_value={"added": [], "removed": [], "changed": [], "unchanged": []}):
            run_monitor(
                paths=self.paths,
                env=self.env,
                crawl_fn=lambda _url, _cfg: snapshot,
                archive_timestamp="2026-04-05T00-00-00Z",
                gemini_client=mock_client,
            )

        # After the run: 4 pre-seeded + 1 new = 5, pruned to 3
        knowledge_archives = sorted(snapshots_dir.glob("knowledge-*.json"))
        self.assertLessEqual(len(knowledge_archives), 3)
        # The newest ones should survive
        archive_names = [a.name for a in knowledge_archives]
        self.assertIn("knowledge-2026-04-05T00-00-00Z.json", archive_names)


class TestTotalExtractionFailureDoesNotPersistKnowledge(unittest.TestCase):
    """When all Gemini extractions fail (empty knowledge_units), knowledge
    snapshot should NOT be persisted, but the raw snapshot should still be."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "config").mkdir()
        (self.root / "reports").mkdir()
        (self.root / "snapshots").mkdir()
        (self.root / "config" / "defaults.json").write_text(
            json.dumps({
                "max_pages": 10,
                "request_timeout_ms": 5000,
                "archive_retention": 3,
                "gemini_model": "gemini-2.0-flash-lite",
            }),
            encoding="utf-8",
        )
        self.paths = MonitorPaths.for_root(self.root)
        self.env = {"HOMEPAGE_URL": HOMEPAGE}

    def tearDown(self):
        self.tempdir.cleanup()

    def test_total_extraction_failure_does_not_persist_knowledge(self):
        page = {
            "url": f"{HOMEPAGE}/",
            "title": "Home",
            "h1": "Welcome",
            "text": "Our office hours are 9am to 5pm.",
            "hash": "hash_v1",
            "status": 200,
        }
        snapshot = _make_snapshot({f"{HOMEPAGE}/": page})

        # All extractions fail: knowledge_units are empty for every page
        empty_knowledge = _make_knowledge({
            f"{HOMEPAGE}/": {
                "url": f"{HOMEPAGE}/",
                "knowledge_units": [],
            }
        })

        mock_client = MagicMock()

        with patch("website_monitor.monitor.extract_all_pages", return_value=empty_knowledge), \
             patch("website_monitor.monitor.compare_knowledge", return_value={"added": [], "removed": [], "changed": [], "unchanged": []}):
            result = run_monitor(
                paths=self.paths,
                env=self.env,
                crawl_fn=lambda _url, _cfg: snapshot,
                archive_timestamp="run1",
                gemini_client=mock_client,
            )

        # Raw snapshot should still be persisted
        self.assertTrue(result["persisted"])
        self.assertTrue(self.paths.latest_snapshot.exists())

        # Knowledge snapshot should NOT be persisted
        self.assertFalse(self.paths.latest_knowledge.exists())

        # Report should mention extraction failure
        report_path = self.paths.reports_dir / "report-run1.md"
        self.assertTrue(report_path.exists())
        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("extraction", report_text.lower())


class TestFirstKnowledgeRunOnExistingSiteCreatesBaseline(unittest.TestCase):
    """When a site has a raw snapshot but no knowledge snapshot, the first
    knowledge run should create a baseline, NOT report false changes."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "config").mkdir()
        (self.root / "reports").mkdir()
        (self.root / "snapshots").mkdir()
        (self.root / "config" / "defaults.json").write_text(
            json.dumps({
                "max_pages": 10,
                "request_timeout_ms": 5000,
                "archive_retention": 3,
                "gemini_model": "gemini-2.0-flash-lite",
            }),
            encoding="utf-8",
        )
        self.paths = MonitorPaths.for_root(self.root)
        self.env = {"HOMEPAGE_URL": HOMEPAGE}

    def tearDown(self):
        self.tempdir.cleanup()

    def test_first_knowledge_run_on_existing_site_creates_baseline(self):
        page = {
            "url": f"{HOMEPAGE}/",
            "title": "Home",
            "h1": "Welcome",
            "text": "Our office hours are 9am to 5pm.",
            "hash": "hash_v1",
            "status": 200,
        }
        snapshot = _make_snapshot({f"{HOMEPAGE}/": page})

        # Pre-seed a raw snapshot (site already monitored without Gemini)
        raw_snapshot_path = self.paths.latest_snapshot
        raw_snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

        knowledge = _make_knowledge({
            f"{HOMEPAGE}/": {
                "url": f"{HOMEPAGE}/",
                "knowledge_units": [
                    {"label": "Office Hours", "value": "9am to 5pm", "category": "operations", "operational": True},
                ],
            }
        })

        mock_client = MagicMock()

        # compare_knowledge(None, knowledge) would normally report all units
        # as "added", but the fix should pass baseline_created=True so the
        # report says "Initial knowledge baseline established" instead.
        with patch("website_monitor.monitor.extract_all_pages", return_value=knowledge), \
             patch("website_monitor.monitor.compare_knowledge", return_value={"added": [], "removed": [], "changed": [], "unchanged": []}):
            result = run_monitor(
                paths=self.paths,
                env=self.env,
                crawl_fn=lambda _url, _cfg: snapshot,
                archive_timestamp="run1",
                gemini_client=mock_client,
            )

        # Should be treated as a baseline run for knowledge
        self.assertTrue(result["baseline_created"])
        self.assertFalse(result["summary"].get("changes_detected", False))

        # Knowledge should be persisted
        self.assertTrue(self.paths.latest_knowledge.exists())

        # Report should say baseline established
        report_path = self.paths.reports_dir / "report-run1.md"
        self.assertTrue(report_path.exists())
        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("baseline", report_text.lower())


class TestAfcStyleFalsePositiveFiltering(unittest.TestCase):
    """Regression coverage for unchanged pages where Gemini extraction drifts."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "config").mkdir()
        (self.root / "reports").mkdir()
        (self.root / "snapshots").mkdir()
        (self.root / "config" / "defaults.json").write_text(
            json.dumps({
                "max_pages": 10,
                "request_timeout_ms": 5000,
                "archive_retention": 3,
                "gemini_model": "gemini-2.0-flash-lite",
            }),
            encoding="utf-8",
        )
        self.paths = MonitorPaths.for_root(self.root)
        self.env = {"HOMEPAGE_URL": HOMEPAGE}

    def tearDown(self):
        self.tempdir.cleanup()

    def test_raw_text_evidence_filters_extraction_drift_before_llm_verify(self):
        urgent_url = f"{HOMEPAGE}/patient-services/urgent-care"
        program_url = f"{HOMEPAGE}/santa-clara/resources/scusd-student-program"
        insurance_policy = (
            "Patient insurance isn't necessary to be seen and treated. "
            "We accept all patients regardless of insurance type or status."
        )
        unchanged_snapshot = _make_snapshot({
            urgent_url: {
                "url": urgent_url,
                "title": "Urgent Care",
                "h1": "Urgent Care",
                "text": "You don't need an appointment or a primary care referral.",
                "hash": "urgent-same",
                "status": 200,
            },
            program_url: {
                "url": program_url,
                "title": "SCUSD Student Program",
                "h1": "SCUSD Student Program",
                "text": f"Program details. {insurance_policy}",
                "hash": "program-same",
                "status": 200,
            },
        })
        previous_knowledge = _make_knowledge({
            urgent_url: {
                "url": urgent_url,
                "knowledge_units": [
                    {
                        "label": "Appointment Requirement",
                        "value": "No",
                        "category": "policy",
                        "operational": True,
                    },
                ],
            },
            program_url: {
                "url": program_url,
                "knowledge_units": [],
            },
        })
        current_knowledge = _make_knowledge({
            urgent_url: {
                "url": urgent_url,
                "knowledge_units": [
                    {
                        "label": "Appointment Requirement",
                        "value": "do not require appointments or referrals",
                        "category": "policy",
                        "operational": True,
                    },
                ],
            },
            program_url: {
                "url": program_url,
                "knowledge_units": [
                    {
                        "label": "Insurance Policy",
                        "value": insurance_policy,
                        "category": "policy",
                        "operational": True,
                    },
                ],
            },
        })
        raw_diff = {
            "changed": [
                {
                    "page": urgent_url,
                    "category": "policy",
                    "label": "Appointment Requirement",
                    "old_value": "No",
                    "new_value": "do not require appointments or referrals",
                },
            ],
            "added": [
                {
                    "page": program_url,
                    "category": "policy",
                    "label": "Insurance Policy",
                    "value": insurance_policy,
                },
            ],
            "removed": [],
            "unchanged": [],
        }

        self.paths.latest_snapshot.write_text(json.dumps(unchanged_snapshot), encoding="utf-8")
        self.paths.latest_knowledge.write_text(json.dumps(previous_knowledge), encoding="utf-8")

        with patch("website_monitor.monitor.extract_all_pages", return_value=current_knowledge), \
             patch("website_monitor.monitor.compare_knowledge", return_value=raw_diff), \
             patch("website_monitor.monitor.quorum_verify_changes", side_effect=lambda diff, **kwargs: diff), \
             patch("website_monitor.monitor.verify_changes", side_effect=lambda diff, *a, **kw: diff) as mock_verify:
            result = run_monitor(
                paths=self.paths,
                env=self.env,
                crawl_fn=lambda _url, _cfg: unchanged_snapshot,
                archive_timestamp="afc-regression",
                gemini_client=MagicMock(),
            )

        mock_verify.assert_not_called()
        self.assertFalse(result["summary"]["changes_detected"])
        self.assertEqual(result["diff"]["changed"], [])
        self.assertEqual(result["diff"]["added"], [])
        self.assertEqual(len(result["diff"]["noise"]), 2)


if __name__ == "__main__":
    unittest.main()
