import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.knowledge import (  # noqa: E402
    build_gemini_client,
    extract_page_knowledge,
)


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
                "fact": "Office hours are 9am-5pm",
                "category": "hours",
                "operational": True,
            },
            {
                "fact": "Founded in 2020",
                "category": "background",
                "operational": False,
            },
        ]
        parsed = MagicMock()
        parsed.knowledge_units = []
        for u in units:
            ku = MagicMock()
            ku.fact = u["fact"]
            ku.category = u["category"]
            ku.operational = u["operational"]
            parsed.knowledge_units.append(ku)

        client = self._make_client_with_response(parsed)
        result = extract_page_knowledge("Some page text", client, "gemini-2.0-flash-lite")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["fact"], "Office hours are 9am-5pm")
        self.assertTrue(result[0]["operational"])
        self.assertEqual(result[1]["fact"], "Founded in 2020")
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
        ku_op.fact = "Open Mon-Fri"
        ku_op.category = "hours"
        ku_op.operational = True

        ku_non = MagicMock()
        ku_non.fact = "Company motto: Be Bold"
        ku_non.category = "branding"
        ku_non.operational = False

        parsed.knowledge_units = [ku_op, ku_non]
        client = self._make_client_with_response(parsed)

        result = extract_page_knowledge("page content", client, "gemini-2.0-flash-lite")
        ops = [u for u in result if u["operational"]]
        non_ops = [u for u in result if not u["operational"]]
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["fact"], "Open Mon-Fri")
        self.assertEqual(len(non_ops), 1)
        self.assertEqual(non_ops[0]["fact"], "Company motto: Be Bold")

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


if __name__ == "__main__":
    unittest.main()
