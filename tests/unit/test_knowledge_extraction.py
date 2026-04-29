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
    filter_text_supported_noise,
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
                    "source_hash": "aaa",
                    "knowledge_units": [{"label": "Cached fact", "value": "cached value", "category": "hours", "operational": True}],
                },
                "https://example.com/about": {
                    "source_hash": "bbb-old",
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
    def test_source_hash_provenance_mismatch_triggers_extraction(self, mock_extract: MagicMock) -> None:
        """When the raw snapshot hash matches but knowledge was extracted from a
        different hash, the page must be re-extracted (not cached)."""
        mock_extract.return_value = [{"label": "Fresh", "value": "fresh val", "category": "info", "operational": True}]

        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/": {"text": "New content", "hash": "bbb"},
            },
        }
        previous_snapshot = {
            "pages": {
                "https://example.com/": {"hash": "bbb"},
            },
        }
        previous_knowledge = {
            "pages": {
                "https://example.com/": {
                    "source_hash": "aaa",
                    "knowledge_units": [{"label": "Stale", "value": "stale val", "category": "info", "operational": True}],
                },
            },
        }

        client = MagicMock()
        result = extract_all_pages(
            crawl_result, client, "gemini-2.0-flash-lite",
            previous_snapshot, previous_knowledge,
        )

        mock_extract.assert_called_once()
        self.assertEqual(
            result["pages"]["https://example.com/"]["knowledge_units"],
            [{"label": "Fresh", "value": "fresh val", "category": "info", "operational": True}],
        )

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_source_hash_match_reuses_cache(self, mock_extract: MagicMock) -> None:
        """When source_hash matches current hash, knowledge is cached (no extraction)."""
        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/": {"text": "Same content", "hash": "aaa"},
            },
        }
        previous_snapshot = {
            "pages": {
                "https://example.com/": {"hash": "aaa"},
            },
        }
        previous_knowledge = {
            "pages": {
                "https://example.com/": {
                    "source_hash": "aaa",
                    "knowledge_units": [{"label": "Cached", "value": "cached val", "category": "info", "operational": True}],
                },
            },
        }

        client = MagicMock()
        result = extract_all_pages(
            crawl_result, client, "gemini-2.0-flash-lite",
            previous_snapshot, previous_knowledge,
        )

        mock_extract.assert_not_called()
        self.assertEqual(
            result["pages"]["https://example.com/"]["knowledge_units"],
            [{"label": "Cached", "value": "cached val", "category": "info", "operational": True}],
        )

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_legacy_knowledge_without_source_hash_falls_back_to_raw_snapshot(self, mock_extract: MagicMock) -> None:
        """Legacy knowledge without source_hash falls back to raw snapshot hash.
        If the raw snapshot hash matches, knowledge is cached (not re-extracted)."""
        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/": {"text": "Content", "hash": "aaa"},
            },
        }
        previous_snapshot = {
            "pages": {
                "https://example.com/": {"hash": "aaa"},
            },
        }
        previous_knowledge = {
            "pages": {
                "https://example.com/": {
                    "knowledge_units": [{"label": "Legacy", "value": "old val", "category": "info", "operational": True}],
                },
            },
        }

        client = MagicMock()
        result = extract_all_pages(
            crawl_result, client, "gemini-2.0-flash-lite",
            previous_snapshot, previous_knowledge,
        )

        mock_extract.assert_not_called()
        self.assertEqual(
            result["pages"]["https://example.com/"]["knowledge_units"],
            [{"label": "Legacy", "value": "old val", "category": "info", "operational": True}],
        )

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_legacy_knowledge_without_source_hash_extracts_when_hash_differs(self, mock_extract: MagicMock) -> None:
        """Legacy knowledge without source_hash: if raw snapshot hash also differs
        from current, extraction is triggered."""
        mock_extract.return_value = [{"label": "Fresh", "value": "fresh val", "category": "info", "operational": True}]

        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/": {"text": "New content", "hash": "bbb"},
            },
        }
        previous_snapshot = {
            "pages": {
                "https://example.com/": {"hash": "aaa"},
            },
        }
        previous_knowledge = {
            "pages": {
                "https://example.com/": {
                    "knowledge_units": [{"label": "Legacy", "value": "old val", "category": "info", "operational": True}],
                },
            },
        }

        client = MagicMock()
        result = extract_all_pages(
            crawl_result, client, "gemini-2.0-flash-lite",
            previous_snapshot, previous_knowledge,
        )

        mock_extract.assert_called_once()
        self.assertEqual(
            result["pages"]["https://example.com/"]["knowledge_units"],
            [{"label": "Fresh", "value": "fresh val", "category": "info", "operational": True}],
        )

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_output_pages_include_source_hash(self, mock_extract: MagicMock) -> None:
        """Output knowledge pages must include source_hash from the crawl."""
        mock_extract.return_value = [{"label": "Fact", "value": "val", "category": "info", "operational": True}]

        crawl_result = {
            "homepage_url": "https://example.com",
            "pages": {
                "https://example.com/": {"text": "Home text", "hash": "aaa"},
                "https://example.com/about": {"text": "About text", "hash": "bbb"},
            },
        }
        previous_snapshot = {
            "pages": {
                "https://example.com/": {"hash": "aaa"},
            },
        }
        previous_knowledge = {
            "pages": {
                "https://example.com/": {
                    "source_hash": "aaa",
                    "knowledge_units": [{"label": "Cached", "value": "cached val", "category": "info", "operational": True}],
                },
            },
        }

        client = MagicMock()
        result = extract_all_pages(
            crawl_result, client, "gemini-2.0-flash-lite",
            previous_snapshot, previous_knowledge,
        )

        self.assertEqual(result["pages"]["https://example.com/"]["source_hash"], "aaa")
        self.assertEqual(result["pages"]["https://example.com/about"]["source_hash"], "bbb")

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
                    "source_hash": "aaa",
                    "knowledge_units": [{"label": "Cached", "value": "cached val", "category": "info", "operational": True}],
                },
                "https://example.com/old-page": {
                    "source_hash": "zzz",
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


class TestFilterTextSupportedNoise(unittest.TestCase):
    def test_added_value_already_in_previous_raw_text_is_noise(self) -> None:
        url = "https://example.com/student-program"
        value = (
            "Patient insurance isn't necessary to be seen and treated. "
            "We accept all patients regardless of insurance type or status."
        )
        diff = {
            "changed": [],
            "added": [
                {
                    "page": url,
                    "category": "policy",
                    "label": "Insurance Policy",
                    "value": value,
                },
            ],
            "removed": [],
            "unchanged": [],
        }
        previous_snapshot = {
            "pages": {
                url: {
                    "text": f"SCUSD student program. {value}",
                },
            },
        }
        current_snapshot = {
            "pages": {
                url: {
                    "text": f"SCUSD student program. {value}",
                },
            },
        }

        result = filter_text_supported_noise(diff, previous_snapshot, current_snapshot)

        self.assertEqual(result["added"], [])
        self.assertEqual(len(result["noise"]), 1)
        self.assertEqual(result["noise"][0]["_noise_reason"], "value_existed_in_previous_text")

    def test_removed_value_still_in_current_raw_text_is_noise(self) -> None:
        url = "https://example.com/urgent-care"
        value = "Walk-Ins Welcome"
        diff = {
            "changed": [],
            "added": [],
            "removed": [
                {
                    "page": url,
                    "category": "policy",
                    "label": "Walk-in Policy",
                    "value": value,
                },
            ],
            "unchanged": [],
        }
        previous_snapshot = {"pages": {url: {"text": value}}}
        current_snapshot = {"pages": {url: {"text": f"{value}. Open daily."}}}

        result = filter_text_supported_noise(diff, previous_snapshot, current_snapshot)

        self.assertEqual(result["removed"], [])
        self.assertEqual(len(result["noise"]), 1)
        self.assertEqual(result["noise"][0]["_noise_reason"], "value_still_in_current_text")

    def test_equivalent_no_appointment_requirement_change_is_noise(self) -> None:
        url = "https://example.com/patient-services/urgent-care"
        diff = {
            "changed": [
                {
                    "page": url,
                    "category": "policy",
                    "label": "Appointment Requirement",
                    "old_value": "No",
                    "new_value": "do not require appointments or referrals",
                },
            ],
            "added": [],
            "removed": [],
            "unchanged": [],
        }

        result = filter_text_supported_noise(diff, None, None)

        self.assertEqual(result["changed"], [])
        self.assertEqual(len(result["noise"]), 1)
        self.assertEqual(result["noise"][0]["_noise_reason"], "equivalent_appointment_requirement")

    def test_real_added_value_absent_from_previous_text_is_kept(self) -> None:
        url = "https://example.com/hours"
        diff = {
            "changed": [],
            "added": [
                {
                    "page": url,
                    "category": "hours",
                    "label": "Holiday Hours",
                    "value": "Closed December 25",
                },
            ],
            "removed": [],
            "unchanged": [],
        }
        previous_snapshot = {"pages": {url: {"text": "Open regular hours."}}}
        current_snapshot = {"pages": {url: {"text": "Closed December 25."}}}

        result = filter_text_supported_noise(diff, previous_snapshot, current_snapshot)

        self.assertEqual(len(result["added"]), 1)
        self.assertEqual(result.get("noise", []), [])

    def test_short_added_value_is_not_matched_inside_unrelated_words(self) -> None:
        url = "https://example.com/policy"
        diff = {
            "changed": [],
            "added": [
                {
                    "page": url,
                    "category": "policy",
                    "label": "Referral Required",
                    "value": "No",
                },
            ],
            "removed": [],
            "unchanged": [],
        }
        previous_snapshot = {"pages": {url: {"text": "Use the North entrance for check-in."}}}
        current_snapshot = {"pages": {url: {"text": "Use the North entrance. Referral required: No."}}}

        result = filter_text_supported_noise(diff, previous_snapshot, current_snapshot)

        self.assertEqual(len(result["added"]), 1)
        self.assertEqual(result.get("noise", []), [])

    def test_short_removed_value_is_not_matched_inside_unrelated_words(self) -> None:
        url = "https://example.com/policy"
        diff = {
            "changed": [],
            "added": [],
            "removed": [
                {
                    "page": url,
                    "category": "policy",
                    "label": "Walk-ins Accepted",
                    "value": "Yes",
                },
            ],
            "unchanged": [],
        }
        previous_snapshot = {"pages": {url: {"text": "Walk-ins accepted: Yes."}}}
        current_snapshot = {"pages": {url: {"text": "Yesterday's policy was updated."}}}

        result = filter_text_supported_noise(diff, previous_snapshot, current_snapshot)

        self.assertEqual(len(result["removed"]), 1)
        self.assertEqual(result.get("noise", []), [])

    def test_appointment_availability_change_is_not_requirement_equivalence(self) -> None:
        url = "https://example.com/appointments"
        diff = {
            "changed": [
                {
                    "page": url,
                    "category": "policy",
                    "label": "Appointment Availability",
                    "old_value": "Walk-ins welcome",
                    "new_value": "No appointments available",
                },
            ],
            "added": [],
            "removed": [],
            "unchanged": [],
        }

        result = filter_text_supported_noise(diff, None, None)

        self.assertEqual(len(result["changed"]), 1)
        self.assertEqual(result.get("noise", []), [])


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

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_added_confirmed_when_value_present_under_different_label(
        self, mock_extract: MagicMock,
    ) -> None:
        """Genuine add with label drift on recapture: previous had no such value,
        current extracted it under label X, recaptures extract the same value
        under label Y. The 'added' candidate must pass quorum because the value
        is genuinely new to the page, even if the label drifted.
        """
        url = "https://example.com/hours"
        value = "Now open until midnight on Fridays"

        diff = {
            "changed": [],
            "added": [
                {"page": url, "category": "hours", "label": "Friday Hours", "value": value},
            ],
            "removed": [],
            "unchanged": [],
        }
        current_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Friday Hours", "value": value, "operational": True},
        ])
        previous_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Weekday Hours", "value": "Mon-Thu 9-5", "operational": True},
        ])

        # Recaptures extract the same new value but under a drifted label
        mock_extract.return_value = [
            {"category": "hours", "label": "Extended Friday Hours", "value": value, "operational": True},
            {"category": "hours", "label": "Weekday Hours", "value": "Mon-Thu 9-5", "operational": True},
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

        self.assertEqual(len(result["added"]), 1,
                         "Added must pass quorum when value is present on recapture "
                         "even under a drifted label")

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_added_rejected_when_value_was_already_in_previous(
        self, mock_extract: MagicMock,
    ) -> None:
        """Phantom add: current has a fact with a new label, but the value was
        already in previous under a different label. Must fail quorum."""
        url = "https://example.com/hours"
        value = "Mon-Fri 9am-5pm"

        diff = {
            "changed": [],
            "added": [
                {"page": url, "category": "hours", "label": "Business Hours", "value": value},
            ],
            "removed": [],
            "unchanged": [],
        }
        current_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Business Hours", "value": value, "operational": True},
        ])
        previous_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Weekday Hours", "value": value, "operational": True},
        ])

        mock_extract.return_value = [
            {"category": "hours", "label": "Weekday Hours", "value": value, "operational": True},
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

        self.assertEqual(len(result["added"]), 0,
                         "Added must be rejected when value was already in previous")

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_changed_confirmed_when_new_value_present_under_different_label(
        self, mock_extract: MagicMock,
    ) -> None:
        """Genuine change with label drift on recapture: the new value is still
        present and the old value is absent, even though the model relabeled it.
        """
        url = "https://example.com/hours"
        old_value = "Mon-Fri 9am-5pm"
        new_value = "Mon-Fri 8am-6pm"

        diff = {
            "changed": [
                {
                    "page": url,
                    "category": "hours",
                    "label": "Weekday Hours",
                    "old_value": old_value,
                    "new_value": new_value,
                },
            ],
            "added": [],
            "removed": [],
            "unchanged": [],
        }
        current_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Weekday Hours", "value": new_value, "operational": True},
        ])
        previous_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Weekday Hours", "value": old_value, "operational": True},
        ])

        mock_extract.return_value = [
            {"category": "operations", "label": "Business Hours", "value": new_value, "operational": True},
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

        self.assertEqual(len(result["changed"]), 1)
        self.assertEqual(result["changed"][0]["new_value"], new_value)

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_changed_rejected_when_old_value_still_present_under_different_label(
        self, mock_extract: MagicMock,
    ) -> None:
        """Phantom change: recapture shows the old value still exists under a
        drifted label, so the changed candidate should fail quorum.
        """
        url = "https://example.com/hours"
        old_value = "Mon-Fri 9am-5pm"
        new_value = "Mon-Fri 8am-6pm"

        diff = {
            "changed": [
                {
                    "page": url,
                    "category": "hours",
                    "label": "Weekday Hours",
                    "old_value": old_value,
                    "new_value": new_value,
                },
            ],
            "added": [],
            "removed": [],
            "unchanged": [],
        }
        current_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Weekday Hours", "value": new_value, "operational": True},
        ])
        previous_knowledge = self._knowledge(url, [
            {"category": "hours", "label": "Weekday Hours", "value": old_value, "operational": True},
        ])

        mock_extract.return_value = [
            {"category": "operations", "label": "Business Hours", "value": old_value, "operational": True},
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

        self.assertEqual(len(result["changed"]), 0)

    @patch("website_monitor.knowledge.extract_page_knowledge")
    def test_changed_confirmed_when_long_new_value_is_similar_to_old_value(
        self, mock_extract: MagicMock,
    ) -> None:
        """A small real edit in a long relabeled value should still pass quorum.

        Regression: fuzzy-checking old_value against the new recaptured value
        made the verifier think the old fact was still present.
        """
        url = "https://example.com/contact"
        old_value = (
            "Call our clinic at 555-111-2222 for same-day urgent care appointments, "
            "walk-in availability, insurance questions, lab services, pediatric "
            "urgent care, and follow-up instructions after your visit."
        )
        new_value = (
            "Call our clinic at 555-111-3333 for same-day urgent care appointments, "
            "walk-in availability, insurance questions, lab services, pediatric "
            "urgent care, and follow-up instructions after your visit."
        )

        diff = {
            "changed": [
                {
                    "page": url,
                    "category": "contact",
                    "label": "Appointment Phone",
                    "old_value": old_value,
                    "new_value": new_value,
                },
            ],
            "added": [],
            "removed": [],
            "unchanged": [],
        }
        current_knowledge = self._knowledge(url, [
            {"category": "contact", "label": "Appointment Phone", "value": new_value, "operational": True},
        ])
        previous_knowledge = self._knowledge(url, [
            {"category": "contact", "label": "Appointment Phone", "value": old_value, "operational": True},
        ])

        mock_extract.return_value = [
            {"category": "policy", "label": "Contact Instructions", "value": new_value, "operational": True},
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

        self.assertEqual(len(result["changed"]), 1)
        self.assertEqual(result["changed"][0]["new_value"], new_value)


if __name__ == "__main__":
    unittest.main()
