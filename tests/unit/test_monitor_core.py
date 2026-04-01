import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.monitor import (
    clean_text,
    compare_snapshots,
    discover_links,
    extract_page_data,
    normalize_url,
    prune_archives,
    render_report,
    resolve_runtime_root,
    strip_boilerplate_js,
    summarize_text_changes,
    should_adopt_homepage_redirect_host,
    should_skip_url,
    wait_for_content_stable,
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

    def test_render_report_omits_all_pages_section_and_shows_changes(self) -> None:
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

        self.assertNotIn("## All Pages Scraped", report)
        self.assertIn("## Changed", report)
        self.assertIn("### https://example.com/", report)
        self.assertIn('- Title changed: "Home" -> "Home Updated"', report)
        self.assertIn('- H1 changed: "Welcome" -> "Welcome Back"', report)
        self.assertIn('- Text modified: "Pricing starts at $9." -> "Pricing starts at $12."', report)
        self.assertIn("- Text added: Chat with us today.", report)

    def test_render_report_shows_redirected_section(self) -> None:
        current = {
            "homepage_url": "https://example.com",
            "scanned_at": "2026-03-26T00:00:00+00:00",
            "pages": {
                "https://example.com/new-path": {
                    "url": "https://example.com/new-path",
                    "title": "Page",
                    "h1": "Page",
                    "text": "Content.",
                    "hash": "same-hash",
                    "status": 200,
                },
            },
        }
        diff = {
            "added": [],
            "removed": [],
            "changed": [],
            "unchanged": ["https://example.com/new-path"],
            "redirected": ["https://example.com/old-path -> https://example.com/new-path"],
        }

        report = render_report(current, diff, baseline_created=False)

        self.assertIn("## Redirected", report)
        self.assertIn("https://example.com/old-path -> https://example.com/new-path", report)

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


    def test_strip_boilerplate_js_removes_nav_header_footer_cookie_elements(self) -> None:
        js_snippet = strip_boilerplate_js()
        self.assertIn("nav", js_snippet)
        self.assertIn("header", js_snippet)
        self.assertIn("footer", js_snippet)
        self.assertIn("cookie", js_snippet.lower())

    def test_extract_page_data_strips_boilerplate_before_extracting_text(self) -> None:
        page = FakePageWithEvaluate(
            title="About Us",
            body_before_strip="Skip to content Nav links. Main content here. Footer copyright 2026.",
            body_after_strip="Main content here.",
            headings=["About Us"],
        )

        page_data = extract_page_data(page, "https://example.com/about")

        self.assertEqual(page_data["text"], "Main content here.")

    def test_clean_text_strips_skip_to_content(self) -> None:
        self.assertEqual(clean_text("Skip to content Main content here."), "Main content here.")
        self.assertEqual(clean_text("Skip to main Main content here."), "Main content here.")

    def test_clean_text_strips_copyright_notices(self) -> None:
        self.assertEqual(clean_text("Hello world. ©2026 Company. All Rights Reserved"), "Hello world.")
        self.assertEqual(clean_text("Hello world. © 2025 Company. All Rights Reserved"), "Hello world.")

    def test_clean_text_strips_manage_consent_text(self) -> None:
        self.assertEqual(clean_text("Hello world. Manage consent"), "Hello world.")

    def test_compare_snapshots_ignores_near_identical_pages(self) -> None:
        previous = {
            "pages": {
                "https://example.com/": {
                    "hash": "old-hash",
                    "text": "Welcome to our clinic. We provide urgent care services for the whole family.",
                },
            }
        }
        current = {
            "pages": {
                "https://example.com/": {
                    "hash": "new-hash",
                    "text": "Welcome to our clinic. We provide urgent care services for the whole family.",
                },
            }
        }

        diff = compare_snapshots(previous, current)

        self.assertEqual(diff["changed"], [])
        self.assertEqual(diff["unchanged"], ["https://example.com/"])

    def test_compare_snapshots_flags_pages_below_similarity_threshold(self) -> None:
        previous = {
            "pages": {
                "https://example.com/": {
                    "hash": "old-hash",
                    "text": "Old pricing page with completely different content.",
                },
            }
        }
        current = {
            "pages": {
                "https://example.com/": {
                    "hash": "new-hash",
                    "text": "New services page that has been totally rewritten.",
                },
            }
        }

        diff = compare_snapshots(previous, current)

        self.assertEqual(diff["changed"], ["https://example.com/"])

    def test_compare_snapshots_reconciles_redirect_pairs(self) -> None:
        previous = {
            "pages": {
                "https://example.com/patient-services": {
                    "hash": "services-hash",
                    "text": "Our urgent care services.",
                },
                "https://example.com/about": {
                    "hash": "about-hash",
                    "text": "About us.",
                },
            }
        }
        current = {
            "pages": {
                "https://example.com/santa-clara/patient-services": {
                    "hash": "services-hash",
                    "text": "Our urgent care services.",
                },
                "https://example.com/about": {
                    "hash": "about-hash",
                    "text": "About us.",
                },
            }
        }

        diff = compare_snapshots(previous, current)

        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], [])
        self.assertIn("https://example.com/patient-services -> https://example.com/santa-clara/patient-services", diff["redirected"])

    def test_compare_snapshots_keeps_genuine_adds_and_removes(self) -> None:
        previous = {
            "pages": {
                "https://example.com/old-page": {
                    "hash": "old-hash",
                    "text": "Old content that is completely unique.",
                },
            }
        }
        current = {
            "pages": {
                "https://example.com/new-page": {
                    "hash": "new-hash",
                    "text": "New content that is completely different.",
                },
            }
        }

        diff = compare_snapshots(previous, current)

        self.assertEqual(diff["added"], ["https://example.com/new-page"])
        self.assertEqual(diff["removed"], ["https://example.com/old-page"])
        self.assertEqual(diff["redirected"], [])

    def test_discover_links_filters_out_none_hrefs(self) -> None:
        page = FakePageWithLinks(hrefs=["/about", None, "None", "", "/contact"])

        links = discover_links(page, "https://example.com/")

        self.assertIn("https://example.com/about", links)
        self.assertIn("https://example.com/contact", links)
        for link in links:
            self.assertNotIn("None", link)

    def test_extract_page_data_does_not_break_link_discovery(self) -> None:
        page = FakePageWithEvaluate(
            title="Home",
            body_before_strip="Nav content. Main content.",
            body_after_strip="Main content.",
            headings=["Home"],
        )

        extract_page_data(page, "https://example.com/")

        self.assertTrue(page._stripped, "evaluate() should have been called")
        locator = page.locator("body")
        self.assertEqual(locator.count(), 1, "body locator should still work after stripping")

    def test_wait_for_content_stable_returns_when_text_matches(self) -> None:
        page = FakeStabilityPage(responses=["Hello world", "Hello world"])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 2)

    def test_wait_for_content_stable_waits_for_changing_content(self) -> None:
        page = FakeStabilityPage(responses=["Loading...", "Partial content", "Full content", "Full content"])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 4)

    def test_wait_for_content_stable_returns_after_timeout(self) -> None:
        call_counter = {"n": 0}

        class NeverStablePage:
            def evaluate(self, expression: str) -> str:
                call_counter["n"] += 1
                return f"text-{call_counter['n']}"

        page = NeverStablePage()

        import time as _time
        start = _time.monotonic()
        wait_for_content_stable(page, timeout_ms=500, interval_ms=100)
        elapsed = _time.monotonic() - start

        self.assertGreaterEqual(elapsed, 0.4)
        self.assertLess(elapsed, 2.0)

    def test_wait_for_content_stable_handles_empty_body(self) -> None:
        page = FakeStabilityPage(responses=["", ""])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 2)

    def test_wait_for_content_stable_handles_exception_on_first_call(self) -> None:
        page = FakeStabilityPage(responses=[RuntimeError("page closed")])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 1)

    def test_wait_for_content_stable_handles_mid_loop_exception(self) -> None:
        page = FakeStabilityPage(responses=["Hello world", RuntimeError("context destroyed")])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 2)


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


class FakePageWithEvaluate:
    """Simulates a Playwright page that supports evaluate() for boilerplate stripping."""

    def __init__(
        self,
        title: str,
        body_before_strip: str,
        body_after_strip: str,
        headings: list[str],
    ) -> None:
        self._title = title
        self._body_after_strip = body_after_strip
        self._headings = headings
        self._stripped = False

    def title(self) -> str:
        return self._title

    def evaluate(self, expression: str) -> None:
        self._stripped = True

    def locator(self, selector: str) -> FakeLocator:
        if selector == "h1":
            return FakeLocator(headings=self._headings)
        if selector == "body" and self._stripped:
            return FakeLocator(text=self._body_after_strip, count=1)
        return FakeLocator(text=None, count=0)


class FakeLinkLocator:
    def __init__(self, hrefs: list) -> None:
        self._hrefs = hrefs

    def evaluate_all(self, expression: str) -> list:
        return self._hrefs


class FakePageWithLinks:
    """Simulates a Playwright page for link discovery tests."""

    def __init__(self, hrefs: list) -> None:
        self._hrefs = hrefs

    def locator(self, selector: str) -> FakeLinkLocator:
        if selector == "a[href]":
            return FakeLinkLocator(self._hrefs)
        return FakeLinkLocator([])


class FakeStabilityPage:
    """Simulates a Playwright page for stability check tests."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = responses
        self.call_count = 0

    def evaluate(self, expression: str) -> str:
        if self.call_count >= len(self._responses):
            return self._responses[-1] if self._responses else ""
        result = self._responses[self.call_count]
        self.call_count += 1
        if isinstance(result, Exception):
            raise result
        return result
