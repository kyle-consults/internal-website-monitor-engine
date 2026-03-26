import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.monitor import (
    compare_snapshots,
    normalize_url,
    prune_archives,
    resolve_runtime_root,
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
