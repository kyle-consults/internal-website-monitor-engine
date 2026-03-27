import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.monitor import (
    compare_snapshots,
    extract_page_data,
    normalize_url,
    prune_archives,
    render_report,
    resolve_runtime_root,
    summarize_text_changes,
    should_adopt_homepage_redirect_host,
    should_skip_url,
)


class MonitorCoreTests(unittest.TestCase):
    def test_normalize_url_removes_fragment_and_trailing_slash(self) -> None:
        normalized = normalize_url("HTTPS://Example.com/path/#section")

        self.assertEqual(normalized, "https://example.com/path")

    def test_should_skip_url_blocks_external_and_binary_targets(self) -> None:
        cfg = {
            "exclude_extensions": [".pdf", ".png"],
            "exclude_url_contains": ["/login"],
        }

        self.assertTrue(should_skip_url("https://cdn.example.net/file.pdf", cfg, "example.com"))
        self.assertTrue(should_skip_url("https://example.com/login", cfg, "example.com"))
        self.assertFalse(should_skip_url("https://example.com/about", cfg, "example.com"))

    def test_compare_snapshots_classifies_added_removed_and_changed_pages(self) -> None:
        previous = {
            "pages": {
                "https://example.com/": {"hash": "old-home"},
                "https://example.com/pricing": {"hash": "pricing"},
            }
        }
        current = {
            "pages": {
                "https://example.com/": {"hash": "new-home"},
                "https://example.com/contact": {"hash": "contact"},
            }
        }

        diff = compare_snapshots(previous, current)

        self.assertEqual(diff["added"], ["https://example.com/contact"])
        self.assertEqual(diff["removed"], ["https://example.com/pricing"])
        self.assertEqual(diff["changed"], ["https://example.com/"])
        self.assertEqual(diff["unchanged"], [])

    def test_extract_page_data_prefers_article_or_main_content_over_body_chrome(self) -> None:
        page = FakePage(
            title="Clinic Near Me",
            text_by_selector={
                "main article": "Core article content. Actual care information.",
                "main": "Core article content. Actual care information. Recent blogs.",
                "body": "Header links. Core article content. Actual care information. Footer links.",
            },
            headings=["Clinic Near Me"],
        )

        page_data = extract_page_data(page, "https://example.com/clinic-near-me")

        self.assertEqual(page_data["title"], "Clinic Near Me")
        self.assertEqual(page_data["h1"], "Clinic Near Me")
        self.assertEqual(page_data["text"], "Core article content. Actual care information.")

    def test_render_report_lists_all_pages_and_change_details(self) -> None:
        previous = {
            "homepage_url": "https://example.com",
            "scanned_at": "2026-03-25T00:00:00+00:00",
            "pages": {
                "https://example.com/": {
                    "url": "https://example.com/",
                    "title": "Home",
                    "h1": "Welcome",
                    "text": "Hello world. Pricing starts at $9.",
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
        }
        current = {
            "homepage_url": "https://example.com",
            "scanned_at": "2026-03-26T00:00:00+00:00",
            "pages": {
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
        }

        report = render_report(current, compare_snapshots(previous, current), baseline_created=False, previous=previous)

        self.assertIn("## All Pages Scraped", report)
        self.assertIn("- https://example.com/ | status: 200 | title: Home Updated", report)
        self.assertIn("- https://example.com/contact | status: 200 | title: Contact", report)
        self.assertIn("## Changed", report)
        self.assertIn("### https://example.com/", report)
        self.assertIn('- Title changed: "Home" -> "Home Updated"', report)
        self.assertIn('- H1 changed: "Welcome" -> "Welcome Back"', report)
        self.assertIn('- Text modified: "Pricing starts at $9." -> "Pricing starts at $12."', report)
        self.assertIn("- Text added: Chat with us today.", report)

    def test_summarize_text_changes_treats_sentence_splits_as_modifications(self) -> None:
        removed, modified, added = summarize_text_changes(
            "Pricing starts at $9 and includes support.",
            "Pricing starts at $12. It includes support.",
        )

        self.assertEqual(removed, [])
        self.assertEqual(
            modified,
            [("Pricing starts at $9 and includes support.", "Pricing starts at $12. It includes support.")],
        )
        self.assertEqual(added, [])

    def test_prune_archives_keeps_the_most_recent_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir)
            for index in range(5):
                path = archive_dir / f"snapshot-2026-03-2{index}T00-00-00Z.json"
                path.write_text(str(index), encoding="utf-8")

            removed = prune_archives(archive_dir, "snapshot-*.json", keep=2)

            remaining = sorted(path.name for path in archive_dir.glob("snapshot-*.json"))

        self.assertEqual(
            remaining,
            [
                "snapshot-2026-03-23T00-00-00Z.json",
                "snapshot-2026-03-24T00-00-00Z.json",
            ],
        )
        self.assertEqual(
            removed,
            [
                "snapshot-2026-03-20T00-00-00Z.json",
                "snapshot-2026-03-21T00-00-00Z.json",
                "snapshot-2026-03-22T00-00-00Z.json",
            ],
        )

    def test_resolve_runtime_root_prefers_working_directory_with_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd_root = Path(tmpdir)
            (cwd_root / "config").mkdir()
            (cwd_root / "config" / "defaults.json").write_text("{}", encoding="utf-8")
            installed_module = cwd_root / "venv" / "lib" / "python3.12" / "site-packages" / "website_monitor" / "monitor.py"
            installed_module.parent.mkdir(parents=True)
            installed_module.write_text("# module", encoding="utf-8")

            resolved = resolve_runtime_root(cwd=cwd_root, env={}, module_file=installed_module)

        self.assertEqual(resolved, cwd_root.resolve())

    def test_should_adopt_homepage_redirect_host_for_www_alias(self) -> None:
        self.assertTrue(
            should_adopt_homepage_redirect_host(
                current_allowed_host="google.com",
                final_host="www.google.com",
                pages_scanned=0,
            )
        )
        self.assertFalse(
            should_adopt_homepage_redirect_host(
                current_allowed_host="google.com",
                final_host="mail.google.com",
                pages_scanned=0,
            )
        )


if __name__ == "__main__":
    unittest.main()


class FakeLocator:
    def __init__(self, text: str | None = None, count: int = 0, headings: list[str] | None = None) -> None:
        self._text = text
        self._count = count
        self._headings = headings or []

    def count(self) -> int:
        return self._count

    @property
    def first(self) -> "FakeLocator":
        return self

    def inner_text(self, timeout: int = 5000) -> str:
        if self._text is None:
            raise RuntimeError("no text")
        return self._text

    def all_inner_texts(self) -> list[str]:
        return self._headings


class FakePage:
    def __init__(self, title: str, text_by_selector: dict[str, str], headings: list[str]) -> None:
        self._title = title
        self._text_by_selector = text_by_selector
        self._headings = headings

    def title(self) -> str:
        return self._title

    def locator(self, selector: str) -> FakeLocator:
        if selector == "h1":
            return FakeLocator(headings=self._headings)
        if selector in self._text_by_selector:
            return FakeLocator(text=self._text_by_selector[selector], count=1)
        return FakeLocator(text=None, count=0)
