import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.knowledge import (  # noqa: E402
    build_gemini_client,
    extract_all_pages,
    extract_page_knowledge,
    quorum_verify_changes,
)


class TestExtractAllPages(unittest.TestCase):
    """Tests for extract_all_pages (parallel extraction with hash gating)."""

    def _make_extract_side_effect(
        self, results_by_text: dict[str, list[dict]],
    ):
        """Return a side_effect function for extract_page_knowledge mock."""
        def _side_effect(page_text, client, model):
            return results_by_text.get(page_text, [])
        return _side_effect

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_hash_gated_reuses_cache_for_unchanged(self, mock_extract: MagicMock) -> None:
        """Unchanged pages (same hash) reuse prior knowledge; changed pages call Gemini."""
        mock_extract.return_value = [{"label": "New fact", "value": "new value", "category": "info", "operational": True}]

        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/": {"text": "Home text", "hash": "aaa"},
                "https://example.com/about": {"text": "About changed", "hash": "bbb-new"},
            },
        }
        previous_snapshot = {
            "pages": {
                "https://example.com/": {"hash": "aaa"},
                "https://example.com/about": {"hash": "bbb-old"},
            },
        }
        previous_knowledge = {
            "pages": {
                "https://example.com/": {
                    "knowledge_units": [{"label": "Cached fact", "value": "cached value", "category": "hours", "operational": True}],
                },
                "https://example.com/about": {
                    "knowledge_units": [{"label": "Old about", "value": "old value", "category": "background", "operational": False}],
                },
            },
        }

        client = MagicMock()
        result = extract_all_pages(
            crawl_result, client, "gemini-2.0-flash-lite",
            previous_snapshot, previous_knowledge,
        )

        # Home page hash unchanged → cached
        self.assertEqual(
            result["pages"]["https://example.com/"]["knowledge_units"],
            [{"label": "Cached fact", "value": "cached value", "category": "hours", "operational": True}],
        )
        # About page hash changed → extracted
        self.assertEqual(
            result["pages"]["https://example.com/about"]["knowledge_units"],
            [{"label": "New fact", "value": "new value", "category": "info", "operational": True}],
        )
        # extract_page_knowledge called only for the changed page
        mock_extract.assert_called_once_with("About changed", client, "gemini-2.0-flash-lite")

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_first_run_extracts_all(self, mock_extract: MagicMock) -> None:
        """No previous snapshot → every page gets extracted."""
        mock_extract.return_value = [{"label": "A fact", "value": "a value", "category": "info", "operational": True}]

        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/": {"text": "Home", "hash": "aaa"},
                "https://example.com/contact": {"text": "Contact", "hash": "bbb"},
            },
        }
        client = MagicMock()
        result = extract_all_pages(crawl_result, client, "gemini-2.0-flash-lite", None, None)

        self.assertEqual(mock_extract.call_count, 2)
        self.assertIn("https://example.com/", result["pages"])
        self.assertIn("https://example.com/contact", result["pages"])
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["homepage_url"], "https://example.com")
        self.assertIn("extracted_at", result)
        self.assertEqual(result["model"], "gemini-2.0-flash-lite")

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_all_extractions_fail_returns_empty_units(self, mock_extract: MagicMock) -> None:
        """When every extraction fails, pages have empty units lists."""
        mock_extract.return_value = []

        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/": {"text": "Home", "hash": "aaa"},
            },
        }
        client = MagicMock()
        result = extract_all_pages(crawl_result, client, "gemini-2.0-flash-lite", None, None)

        self.assertEqual(result["pages"]["https://example.com/"]["knowledge_units"], [])

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_new_page_not_in_previous_triggers_extraction(self, mock_extract: MagicMock) -> None:
        """A page in the crawl but not in previous snapshot gets extracted."""
        mock_extract.return_value = [{"label": "Brand new", "value": "brand new value", "category": "info", "operational": True}]

        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/new-page": {"text": "New page text", "hash": "ccc"},
            },
        }
        previous_snapshot = {"pages": {}}
        previous_knowledge = {"pages": {}}

        client = MagicMock()
        result = extract_all_pages(
            crawl_result, client, "gemini-2.0-flash-lite",
            previous_snapshot, previous_knowledge,
        )

        mock_extract.assert_called_once_with("New page text", client, "gemini-2.0-flash-lite")
        self.assertEqual(
            result["pages"]["https://example.com/new-page"]["knowledge_units"],
            [{"label": "Brand new", "value": "brand new value", "category": "info", "operational": True}],
        )

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_removed_page_absent_from_result(self, mock_extract: MagicMock) -> None:
        """A page in previous but not in current crawl is absent from the result."""
        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/": {"text": "Home", "hash": "aaa"},
            },
        }
        previous_snapshot = {
            "pages": {
                "https://example.com/": {"hash": "aaa"},
                "https://example.com/old-page": {"hash": "zzz"},
            },
        }
        previous_knowledge = {
            "pages": {
                "https://example.com/": {
                    "knowledge_units": [{"label": "Cached", "value": "cached val", "category": "info", "operational": True}],
                },
                "https://example.com/old-page": {
                    "knowledge_units": [{"label": "Gone", "value": "gone val", "category": "info", "operational": True}],
                },
            },
        }

        client = MagicMock()
        result = extract_all_pages(
            crawl_result, client, "gemini-2.0-flash-lite",
            previous_snapshot, previous_knowledge,
        )

        self.assertIn("https://example.com/", result["pages"])
        self.assertNotIn("https://example.com/old-page", result["pages"])
        mock_extract.assert_not_called()


class TestBuildGeminiClient(unittest.TestCase):
    """Tests for build_gemini_client."""

    def test_no_key_returns_none(self) -> None:
        self.assertIsNone(build_gemini_client(None))

    def test_empty_key_returns_none(self) -> None:
        self.assertIsNone(build_gemini_client(""))

    @patch("website_monitor.knowledge.genai.Client")
    def test_valid_key_returns_client(self, mock_client_cls: MagicMock) -> None:
        sentinel = MagicMock()
        mock_client_cls.return_value = sentinel
        result = build_gemini_client("sk-real-key")
        mock_client_cls.assert_called_once_with(api_key="sk-real-key")
        self.assertIs(result, sentinel)


class TestExtractPageKnowledge(unittest.TestCase):
    """Tests for extract_page_knowledge."""

    def _make_client_with_response(self, parsed_data: object) -> MagicMock:
        """Build a mock genai Client whose generate_content returns parsed_data."""
        client = MagicMock()
        response = MagicMock()
        response.parsed = parsed_data
        client.models.generate_content.return_value = response
        return client

    def test_happy_path_extracts_units(self) -> None:
        units = [
            {
                "label": "Weekday Hours",
                "value": "Monday-Friday 9am-5pm",
                "category": "hours",
                "operational": True,
            },
            {
                "label": "Founded Year",
                "value": "2020",
                "category": "background",
                "operational": False,
            },
        ]
        parsed = MagicMock()
        parsed.knowledge_units = []
        for u in units:
            ku = MagicMock()
            ku.label = u["label"]
            ku.value = u["value"]
            ku.category = u["category"]
            ku.operational = u["operational"]
            parsed.knowledge_units.append(ku)

        client = self._make_client_with_response(parsed)
        result = extract_page_knowledge("Some page text", client, "gemini-2.0-flash-lite")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["label"], "Weekday Hours")
        self.assertEqual(result[0]["value"], "Monday-Friday 9am-5pm")
        self.assertTrue(result[0]["operational"])
        self.assertEqual(result[1]["label"], "Founded Year")
        self.assertEqual(result[1]["value"], "2020")
        self.assertFalse(result[1]["operational"])

    def test_empty_text_returns_empty(self) -> None:
        client = MagicMock()
        result = extract_page_knowledge("", client, "gemini-2.0-flash-lite")
        self.assertEqual(result, [])
        client.models.generate_content.assert_not_called()

    def test_whitespace_only_returns_empty(self) -> None:
        client = MagicMock()
        result = extract_page_knowledge("   \n\t  ", client, "gemini-2.0-flash-lite")
        self.assertEqual(result, [])
        client.models.generate_content.assert_not_called()

    def test_api_error_returns_empty(self) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = RuntimeError("API unavailable")
        result = extract_page_knowledge("Some text", client, "gemini-2.0-flash-lite")
        self.assertEqual(result, [])

    def test_malformed_response_returns_empty(self) -> None:
        """AttributeError when response.parsed has unexpected shape."""
        client = MagicMock()
        response = MagicMock()
        # parsed is a plain object with no knowledge_units attribute
        response.parsed = object()
        client.models.generate_content.return_value = response
        result = extract_page_knowledge("Some text", client, "gemini-2.0-flash-lite")
        self.assertEqual(result, [])

    def test_mixed_operational_flags_preserved(self) -> None:
        parsed = MagicMock()
        ku_op = MagicMock()
        ku_op.label = "Weekday Hours"
        ku_op.value = "Open Mon-Fri"
        ku_op.category = "hours"
        ku_op.operational = True

        ku_non = MagicMock()
        ku_non.label = "Company Motto"
        ku_non.value = "Be Bold"
        ku_non.category = "branding"
        ku_non.operational = False

        parsed.knowledge_units = [ku_op, ku_non]
        client = self._make_client_with_response(parsed)

        result = extract_page_knowledge("page content", client, "gemini-2.0-flash-lite")
        ops = [u for u in result if u["operational"]]
        non_ops = [u for u in result if not u["operational"]]
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["label"], "Weekday Hours")
        self.assertEqual(ops[0]["value"], "Open Mon-Fri")
        self.assertEqual(len(non_ops), 1)
        self.assertEqual(non_ops[0]["label"], "Company Motto")
        self.assertEqual(non_ops[0]["value"], "Be Bold")

    def test_prompt_contains_page_text_tags(self) -> None:
        """Verify the prompt wraps page text in <PAGE_TEXT> tags for injection safety."""
        parsed = MagicMock()
        parsed.knowledge_units = []
        client = self._make_client_with_response(parsed)

        extract_page_knowledge("Hello world", client, "gemini-2.0-flash-lite")

        call_args = client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents", call_args.args[0] if call_args.args else None)
        self.assertIn("<PAGE_TEXT>", contents)
        self.assertIn("</PAGE_TEXT>", contents)
        self.assertIn("Hello world", contents)


class TestQuorumVerifyChanges(unittest.TestCase):
    """Tests for quorum_verify_changes (multi-capture label-drift handling)."""

    def _knowledge(self, url: str, units: list[dict]) -> dict:
        return {"pages": {url: {"url": url, "knowledge_units": units}}}

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_removed_rejected_when_value_still_present_under_different_label(
        self, mock_extract: MagicMock,
    ) -> None:
        """Label drift: previous had 'Hours'='extended hours ...', current extracted
        same value under 'Weekend Hours'. compare_knowledge emits this as removed.
        Quorum recaptures keep producing the value under yet more labels. The
        'removed' candidate must fail quorum because the value is still present.
        """
        url = "https://example.com/hours"
        value = "extended hours on evenings and weekends"

        diff = {
            "changed": [],
            "added": [],
            "removed": [
                {"page": url, "category": "hours", "label": "Hours", "value": value},
            ],
            "unchanged": [],
        }
        current_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Weekend Hours", "value": value, "operational": True},
        ])
        previous_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Hours", "value": value, "operational": True},
        ])

        # Every recapture returns the same value under a drifting label
        mock_extract.return_value = [
            {"category": "hours", "label": "Evening Hours", "value": value, "operational": True},
        ]

        def fake_recrawl(urls, cfg):
            return {u: {"text": "nonempty"} for u in urls}

        result = quorum_verify_changes(
            diff=diff,
            current_knowledge=current_knowledge,
            previous_knowledge=previous_knowledge,
            recrawl_fn=fake_recrawl,
            client=MagicMock(),
            model="test-model",
            cfg={},
            captures=2,
            quorum=2,
        )

        self.assertEqual(len(result["removed"]), 0,
                         "Removed must be rejected when value still exists on page")

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_removed_confirmed_when_value_truly_gone(
        self, mock_extract: MagicMock,
    ) -> None:
        """Genuine removal: value is absent from every recapture → passes quorum."""
        url = "https://example.com/hours"
        value = "Open 24/7"

        diff = {
            "changed": [],
            "added": [],
            "removed": [
                {"page": url, "category": "hours", "label": "Hours", "value": value},
            ],
            "unchanged": [],
        }
        current_knowledge = self._knowledge(url, [])
        previous_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Hours", "value": value, "operational": True},
        ])

        mock_extract.return_value = [
            {"category": "hours", "label": "Weekday Hours", "value": "Mon-Fri 9-5", "operational": True},
        ]

        def fake_recrawl(urls, cfg):
            return {u: {"text": "nonempty"} for u in urls}

        result = quorum_verify_changes(
            diff=diff,
            current_knowledge=current_knowledge,
            previous_knowledge=previous_knowledge,
            recrawl_fn=fake_recrawl,
            client=MagicMock(),
            model="test-model",
            cfg={},
            captures=2,
            quorum=2,
        )

        self.assertEqual(len(result["removed"]), 1)
        self.assertEqual(result["removed"][0]["value"], value)


if __name__ == "__main__":
    unittest.main()
