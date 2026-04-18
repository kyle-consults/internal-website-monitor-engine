import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.knowledge_diff import (  # noqa: E402
    _fuzzy_reconcile,
    compare_knowledge,
    reconcile_knowledge_redirects,
)


def _snapshot(pages: dict) -> dict:
    """Build a minimal knowledge snapshot for testing."""
    return {
        "schema_version": 1,
        "homepage_url": "https://example.com",
        "extracted_at": "2026-04-11T00:00:00Z",
        "model": "gemini-2.0-flash-lite",
        "pages": {
            url: {
                "url": url,
                "knowledge_units": units,
            }
            for url, units in pages.items()
        },
    }


class TestCompareKnowledge(unittest.TestCase):
    """Tests for compare_knowledge."""

    def test_detects_added_changed_removed_unchanged(self):
        """All 4 change types in one diff."""
        previous = _snapshot({
            "https://example.com/hours": [
                {"category": "hours", "label": "Weekday Hours", "value": "Mon-Fri 8am-8pm", "operational": True},
                {"category": "hours", "label": "Weekend Hours", "value": "Sat-Sun 9am-5pm", "operational": True},
                {"category": "hours", "label": "Holiday Hours", "value": "Closed", "operational": True},
            ],
        })
        current = _snapshot({
            "https://example.com/hours": [
                {"category": "hours", "label": "Weekday Hours", "value": "Mon-Fri 9am-9pm", "operational": True},  # changed
                {"category": "hours", "label": "Weekend Hours", "value": "Sat-Sun 9am-5pm", "operational": True},  # unchanged
                # Holiday Hours removed
                {"category": "hours", "label": "Summer Hours", "value": "Mon-Fri 7am-10pm", "operational": True},  # added
            ],
        })
        result = compare_knowledge(previous, current)

        self.assertEqual(len(result["changed"]), 1)
        self.assertEqual(result["changed"][0]["label"], "Weekday Hours")
        self.assertEqual(result["changed"][0]["old_value"], "Mon-Fri 8am-8pm")
        self.assertEqual(result["changed"][0]["new_value"], "Mon-Fri 9am-9pm")

        self.assertEqual(len(result["unchanged"]), 1)
        self.assertEqual(result["unchanged"][0]["label"], "Weekend Hours")

        self.assertEqual(len(result["removed"]), 1)
        self.assertEqual(result["removed"][0]["label"], "Holiday Hours")

        self.assertEqual(len(result["added"]), 1)
        self.assertEqual(result["added"][0]["label"], "Summer Hours")

    def test_fuzzy_matches_similar_labels_as_change_not_add_remove(self):
        """'Weekday Hours' vs 'Monday-Friday Hours' with different values → changed."""
        previous = _snapshot({
            "https://example.com/hours": [
                {"category": "hours", "label": "Weekday Hours", "value": "8am-8pm", "operational": True},
            ],
        })
        current = _snapshot({
            "https://example.com/hours": [
                {"category": "hours", "label": "Monday-Friday Hours", "value": "9am-9pm", "operational": True},
            ],
        })
        result = compare_knowledge(previous, current, fuzzy_threshold=0.4)

        # Fuzzy match should treat this as a change, not add+remove
        self.assertEqual(len(result["changed"]), 1)
        self.assertEqual(result["changed"][0]["old_value"], "8am-8pm")
        self.assertEqual(result["changed"][0]["new_value"], "9am-9pm")
        self.assertEqual(len(result["added"]), 0)
        self.assertEqual(len(result["removed"]), 0)

    def test_baseline_treats_all_as_added(self):
        """previous=None → everything is added."""
        current = _snapshot({
            "https://example.com/hours": [
                {"category": "hours", "label": "Weekday Hours", "value": "Mon-Fri 8am-8pm", "operational": True},
                {"category": "hours", "label": "Weekend Hours", "value": "Sat-Sun 9am-5pm", "operational": True},
            ],
        })
        result = compare_knowledge(None, current)

        self.assertEqual(len(result["added"]), 2)
        self.assertEqual(len(result["removed"]), 0)
        self.assertEqual(len(result["changed"]), 0)
        self.assertEqual(len(result["unchanged"]), 0)

    def test_removed_page_reports_all_units_as_removed(self):
        """Page in previous but not current → all units removed."""
        previous = _snapshot({
            "https://example.com/hours": [
                {"category": "hours", "label": "Weekday Hours", "value": "Mon-Fri 8am-8pm", "operational": True},
                {"category": "hours", "label": "Weekend Hours", "value": "Sat-Sun 9am-5pm", "operational": True},
            ],
        })
        current = _snapshot({})
        result = compare_knowledge(previous, current)

        self.assertEqual(len(result["removed"]), 2)
        self.assertEqual(len(result["added"]), 0)

    def test_non_operational_units_are_skipped(self):
        """operational=False units ignored in comparison."""
        previous = _snapshot({
            "https://example.com/hours": [
                {"category": "hours", "label": "Weekday Hours", "value": "Mon-Fri 8am-8pm", "operational": True},
                {"category": "hours", "label": "Old Label", "value": "old", "operational": False},
            ],
        })
        current = _snapshot({
            "https://example.com/hours": [
                {"category": "hours", "label": "Weekday Hours", "value": "Mon-Fri 8am-8pm", "operational": True},
                {"category": "hours", "label": "New Label", "value": "new", "operational": False},
            ],
        })
        result = compare_knowledge(previous, current)

        # Non-operational units should not appear in any bucket
        self.assertEqual(len(result["unchanged"]), 1)
        self.assertEqual(len(result["added"]), 0)
        self.assertEqual(len(result["removed"]), 0)
        self.assertEqual(len(result["changed"]), 0)

    def test_cross_category_label_drift_reconciled_as_unchanged(self):
        """LLM relabels same fact under a different category → no add/remove pair.

        Regression for the AFC report: services listed under 'Urgent Care Clinic
        Services' in one run got re-extracted under 'service' in the next run
        with the same value. Should reconcile as unchanged, not as removed.
        """
        value = "Bleeding or cuts that need stitches"
        url = "https://example.com/clinic"
        previous = _snapshot({
            url: [
                {"category": "service", "label": "Urgent Care Clinic Services",
                 "value": value, "operational": True},
            ],
        })
        current = _snapshot({
            url: [
                {"category": "policy", "label": "Services Offered",
                 "value": value, "operational": True},
            ],
        })
        result = compare_knowledge(previous, current)

        self.assertEqual(len(result["removed"]), 0)
        self.assertEqual(len(result["added"]), 0)

    def test_empty_knowledge_units_list_no_crash(self):
        """Page with knowledge_units: [] works without error."""
        previous = _snapshot({"https://example.com/empty": []})
        current = _snapshot({"https://example.com/empty": []})
        result = compare_knowledge(previous, current)

        self.assertEqual(len(result["added"]), 0)
        self.assertEqual(len(result["removed"]), 0)
        self.assertEqual(len(result["changed"]), 0)
        self.assertEqual(len(result["unchanged"]), 0)


class TestFuzzyReconcile(unittest.TestCase):
    """Tests for _fuzzy_reconcile."""

    def test_fuzzy_reconcile_no_removed_keys_returns_unchanged(self):
        """Empty removed set returns immediately with no matches."""
        added = {
            ("https://example.com/hours", "hours", "Weekday Hours"): "8am-8pm",
        }
        removed = {}
        matched, remaining_added, remaining_removed = _fuzzy_reconcile(
            added, removed, threshold=0.75,
        )
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(remaining_added), 1)
        self.assertEqual(len(remaining_removed), 0)

    def test_fuzzy_reconcile_matches_across_categories_by_value(self):
        """Label+category drift: same value re-extracted under different category.

        Regression: AFC report showed "COVID-19 Vaccine Information" disappear and
        "COVID-19 Vaccine Contact" appear with the same value. Previously, grouping
        by (page, category) prevented cross-category reconciliation, so identical
        values slipped through as added+removed.
        """
        page = "https://example.com/covid"
        value = "Contact us about getting a COVID-19 vaccination."
        added = {
            (page, "service", "COVID-19 Vaccine Contact"): value,
        }
        removed = {
            (page, "policy", "COVID-19 Vaccine Information"): value,
        }
        matched, remaining_added, remaining_removed = _fuzzy_reconcile(
            added, removed, threshold=0.75,
        )

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["old_value"], value)
        self.assertEqual(matched[0]["new_value"], value)
        self.assertEqual(len(remaining_added), 0)
        self.assertEqual(len(remaining_removed), 0)

    def test_fuzzy_reconcile_picks_best_score_from_multiple_candidates(self):
        """Multiple similar labels — picks highest similarity."""
        added = {
            ("https://example.com/h", "hours", "Weekday Business Hours"): "9am-5pm",
        }
        removed = {
            ("https://example.com/h", "hours", "Weekday Hours"): "8am-8pm",
            ("https://example.com/h", "hours", "Weekday Hrs"): "7am-7pm",
        }
        matched, remaining_added, remaining_removed = _fuzzy_reconcile(
            added, removed, threshold=0.4,
        )

        self.assertEqual(len(matched), 1)
        # Should match "Weekday Hours" (higher similarity to "Weekday Business Hours")
        match = matched[0]
        self.assertEqual(match["old_label"], "Weekday Hours")
        self.assertEqual(match["new_label"], "Weekday Business Hours")
        self.assertEqual(len(remaining_added), 0)
        self.assertEqual(len(remaining_removed), 1)


class TestReconcileKnowledgeRedirects(unittest.TestCase):
    """Tests for reconcile_knowledge_redirects."""

    def test_redirect_reconciliation_identical_knowledge(self):
        """Same units on different URLs → redirect."""
        units = [
            {"category": "hours", "label": "Weekday Hours", "value": "Mon-Fri 8am-8pm", "operational": True},
        ]
        previous_pages = {
            "https://example.com/old-hours": {"url": "https://example.com/old-hours", "knowledge_units": units},
        }
        current_pages = {
            "https://example.com/new-hours": {"url": "https://example.com/new-hours", "knowledge_units": units},
        }
        redirected, remaining_added, remaining_removed = reconcile_knowledge_redirects(
            added_urls=["https://example.com/new-hours"],
            removed_urls=["https://example.com/old-hours"],
            previous_pages=previous_pages,
            current_pages=current_pages,
        )
        self.assertEqual(len(redirected), 1)
        self.assertEqual(redirected[0]["from_url"], "https://example.com/old-hours")
        self.assertEqual(redirected[0]["to_url"], "https://example.com/new-hours")
        self.assertEqual(len(remaining_added), 0)
        self.assertEqual(len(remaining_removed), 0)

    def test_redirect_reconciliation_different_knowledge_not_redirect(self):
        """Different units stay as add+remove."""
        previous_pages = {
            "https://example.com/old": {
                "url": "https://example.com/old",
                "knowledge_units": [
                    {"category": "hours", "label": "Hours", "value": "8am-8pm", "operational": True},
                ],
            },
        }
        current_pages = {
            "https://example.com/new": {
                "url": "https://example.com/new",
                "knowledge_units": [
                    {"category": "hours", "label": "Hours", "value": "9am-9pm", "operational": True},
                ],
            },
        }
        redirected, remaining_added, remaining_removed = reconcile_knowledge_redirects(
            added_urls=["https://example.com/new"],
            removed_urls=["https://example.com/old"],
            previous_pages=previous_pages,
            current_pages=current_pages,
        )
        self.assertEqual(len(redirected), 0)
        self.assertEqual(remaining_added, ["https://example.com/new"])
        self.assertEqual(remaining_removed, ["https://example.com/old"])

    def test_redirect_reconciliation_empty_fingerprint_not_redirect(self):
        """No operational units → no redirect match."""
        previous_pages = {
            "https://example.com/old": {
                "url": "https://example.com/old",
                "knowledge_units": [
                    {"category": "hours", "label": "Hours", "value": "8am-8pm", "operational": False},
                ],
            },
        }
        current_pages = {
            "https://example.com/new": {
                "url": "https://example.com/new",
                "knowledge_units": [
                    {"category": "hours", "label": "Hours", "value": "8am-8pm", "operational": False},
                ],
            },
        }
        redirected, remaining_added, remaining_removed = reconcile_knowledge_redirects(
            added_urls=["https://example.com/new"],
            removed_urls=["https://example.com/old"],
            previous_pages=previous_pages,
            current_pages=current_pages,
        )
        self.assertEqual(len(redirected), 0)
        self.assertEqual(remaining_added, ["https://example.com/new"])
        self.assertEqual(remaining_removed, ["https://example.com/old"])


if __name__ == "__main__":
    unittest.main()
