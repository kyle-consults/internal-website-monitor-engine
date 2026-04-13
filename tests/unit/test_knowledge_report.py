import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.knowledge_report import (  # noqa: E402
    build_knowledge_summary,
    render_knowledge_report,
)


def _snapshot(pages: dict, homepage_url: str = "https://example.com") -> dict:
    """Build a minimal knowledge snapshot for testing."""
    return {
        "schema_version": 1,
        "homepage_url": homepage_url,
        "extracted_at": "2026-04-11T12:00:00+00:00",
        "model": "gemini-2.0-flash-lite",
        "pages": {
            url: {"url": url, "knowledge_units": units}
            for url, units in pages.items()
        },
    }


def _diff(changed=None, added=None, removed=None, unchanged=None) -> dict:
    return {
        "changed": changed or [],
        "added": added or [],
        "removed": removed or [],
        "unchanged": unchanged or [],
    }


class TestRenderKnowledgeReport(unittest.TestCase):
    def test_renders_changed_added_removed(self):
        snapshot = _snapshot({
            "/hours": [
                {"label": "Weekday Hours", "value": "8am-9pm", "category": "hours", "operational": True},
            ],
            "/insurance": [
                {"label": "Provider", "value": "Aetna", "category": "insurance", "operational": True},
            ],
        })
        diff = _diff(
            changed=[
                {"page": "/hours", "category": "hours", "label": "Weekday Hours",
                 "old_value": "8am-8pm", "new_value": "8am-9pm"},
            ],
            added=[
                {"page": "/insurance", "category": "insurance", "label": "Provider",
                 "value": "Aetna"},
            ],
            removed=[
                {"page": "/hours", "category": "policy", "label": "Walk-ins",
                 "value": "Not accepted"},
            ],
        )
        report = render_knowledge_report(snapshot, diff, baseline_created=False)

        # All three sections should appear
        self.assertIn("## Changes Detected", report)
        self.assertIn("**Weekday Hours** changed", report)
        self.assertIn('"8am-8pm"', report)
        self.assertIn('"8am-9pm"', report)
        self.assertIn("**Provider** added", report)
        self.assertIn('"Aetna"', report)
        self.assertIn("**Walk-ins** removed", report)
        self.assertIn('"Not accepted"', report)
        # Category headers
        self.assertIn("### Hours", report)
        self.assertIn("### Insurance", report)
        self.assertIn("### Policy", report)

    def test_renders_no_changes(self):
        snapshot = _snapshot({"/about": []})
        diff = _diff()
        report = render_knowledge_report(snapshot, diff, baseline_created=False)
        self.assertIn("No knowledge changes detected", report)
        self.assertNotIn("## Changes Detected", report)

    def test_renders_baseline_established(self):
        snapshot = _snapshot({"/about": []})
        diff = _diff()
        report = render_knowledge_report(snapshot, diff, baseline_created=True)
        self.assertIn("Initial knowledge baseline established", report)

    def test_renders_raw_fallback_section(self):
        snapshot = _snapshot({"/contact": []})
        diff = _diff()
        fallback = [
            {"page": "/contact", "diff_summary": "Phone number changed from 555-0100 to 555-0199"},
        ]
        report = render_knowledge_report(snapshot, diff, baseline_created=False, raw_fallback_pages=fallback)
        self.assertIn("## Fallback: Raw Changes", report)
        self.assertIn("/contact", report)
        self.assertIn("555-0100", report)

    def test_renders_extraction_notes(self):
        snapshot = _snapshot({"/hours": []})
        diff = _diff()
        notes = ["Page /hours returned inconsistent extraction results across retries."]
        report = render_knowledge_report(snapshot, diff, baseline_created=False, extraction_notes=notes)
        self.assertIn("## Extraction Notes", report)
        self.assertIn("inconsistent extraction", report)


class TestBuildKnowledgeSummary(unittest.TestCase):
    def test_build_summary_with_change_counts(self):
        snapshot = _snapshot({
            "/hours": [{"label": "H", "value": "v", "category": "hours", "operational": True}],
            "/about": [{"label": "A", "value": "v", "category": "bg", "operational": False}],
        })
        diff = _diff(
            changed=[{"page": "/hours", "category": "hours", "label": "H", "old_value": "a", "new_value": "b"}],
            added=[{"page": "/ins", "category": "ins", "label": "P", "value": "Aetna"}],
            removed=[],
        )
        summary = build_knowledge_summary(snapshot, diff, baseline_created=False)

        self.assertEqual(summary["homepage_url"], "https://example.com")
        self.assertEqual(summary["scanned_at"], "2026-04-11T12:00:00+00:00")
        self.assertFalse(summary["baseline_created"])
        self.assertTrue(summary["changes_detected"])
        self.assertEqual(summary["counts"]["pages_scanned"], 2)
        self.assertEqual(summary["counts"]["changed"], 1)
        self.assertEqual(summary["counts"]["added"], 1)
        self.assertEqual(summary["counts"]["removed"], 0)

    def test_build_summary_baseline_created(self):
        snapshot = _snapshot({"/about": []})
        diff = _diff()
        summary = build_knowledge_summary(snapshot, diff, baseline_created=True)
        self.assertTrue(summary["baseline_created"])
        self.assertFalse(summary["changes_detected"])
        self.assertEqual(summary["counts"]["pages_scanned"], 1)


if __name__ == "__main__":
    unittest.main()
