import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.monitor import ConfigurationError, MonitorPaths, run_monitor  # noqa: E402


class WorkflowContractTests(unittest.TestCase):
    def test_reusable_workflow_installs_engine_and_runs_monitor(self) -> None:
        workflow_text = (ROOT / ".github" / "workflows" / "reusable-monitor.yml").read_text(encoding="utf-8")

        self.assertIn("on:", workflow_text)
        self.assertIn("workflow_call:", workflow_text)
        self.assertIn("uses: actions/checkout@v6", workflow_text)
        self.assertIn("fetch-depth: 0", workflow_text)
        self.assertIn("ref: ${{ github.ref_name }}", workflow_text)
        self.assertIn("uses: actions/setup-python@v6", workflow_text)
        self.assertIn("repository: ${{ github.repository }}", workflow_text)
        self.assertIn("repository: kyle-consults/internal-website-monitor-engine", workflow_text)
        self.assertIn("path: _monitor_engine", workflow_text)
        self.assertIn("pip install -r _monitor_engine/requirements.txt", workflow_text)
        self.assertIn("pip install ./_monitor_engine", workflow_text)
        self.assertIn("WEBSITE_MONITOR_ROOT: ${{ github.workspace }}", workflow_text)
        self.assertIn("python -m website_monitor", workflow_text)

    def test_reusable_workflow_wires_optional_email_configuration(self) -> None:
        workflow_text = (ROOT / ".github" / "workflows" / "reusable-monitor.yml").read_text(encoding="utf-8")

        self.assertIn("RESEND_API_KEY: ${{ secrets.RESEND_API_KEY }}", workflow_text)
        self.assertIn("ALERT_EMAIL_TO: ${{ inputs.alert_email_to }}", workflow_text)
        self.assertIn("ALERT_EMAIL_FROM: ${{ inputs.alert_email_from }}", workflow_text)
        self.assertIn("EMAIL_MODE: ${{ inputs.email_mode }}", workflow_text)
        self.assertIn("python -m website_monitor.notify", workflow_text)

    def test_reusable_workflow_only_runs_notification_step_when_email_is_configured(self) -> None:
        workflow_text = (ROOT / ".github" / "workflows" / "reusable-monitor.yml").read_text(encoding="utf-8")

        self.assertIn(
            "if: ${{ inputs.alert_email_to != '' && inputs.alert_email_from != '' && env.RESEND_API_KEY != '' }}",
            workflow_text,
        )

    def test_reusable_workflow_persists_outputs_before_notification_step(self) -> None:
        workflow_text = (ROOT / ".github" / "workflows" / "reusable-monitor.yml").read_text(encoding="utf-8")

        self.assertLess(
            workflow_text.index("- name: Commit updated outputs"),
            workflow_text.index("- name: Send email notification"),
        )
        self.assertIn('git pull --rebase origin "${GITHUB_REF_NAME}"', workflow_text)
        self.assertIn('git push origin HEAD:"${GITHUB_REF_NAME}"', workflow_text)

    def test_reusable_workflow_wires_optional_gemini_api_key(self) -> None:
        workflow_text = (ROOT / ".github" / "workflows" / "reusable-monitor.yml").read_text(encoding="utf-8")

        self.assertIn("GEMINI_API_KEY:", workflow_text)
        self.assertIn("GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}", workflow_text)
        self.assertIn("required: false", workflow_text)

    def test_missing_homepage_url_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir()
            (root / "snapshots").mkdir()
            (root / "reports").mkdir()
            (root / "config" / "defaults.json").write_text(
                '{"max_pages": 5, "archive_retention": 2, "exclude_extensions": [], "exclude_url_contains": []}',
                encoding="utf-8",
            )

            with self.assertRaises(ConfigurationError):
                run_monitor(
                    paths=MonitorPaths.for_root(root),
                    env={},
                    crawl_fn=lambda homepage_url, cfg: {},
                    archive_timestamp="2026-03-25T00-00-00Z",
                )

    def test_fatal_failure_preserves_previous_latest_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir()
            (root / "snapshots").mkdir()
            (root / "reports").mkdir()
            (root / "config" / "defaults.json").write_text(
                '{"max_pages": 5, "archive_retention": 2, "exclude_extensions": [], "exclude_url_contains": []}',
                encoding="utf-8",
            )
            latest_snapshot = root / "snapshots" / "latest-snapshot.json"
            latest_report = root / "reports" / "latest-report.md"
            latest_snapshot.write_text('{"baseline": true}', encoding="utf-8")
            latest_report.write_text("baseline report", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                run_monitor(
                    paths=MonitorPaths.for_root(root),
                    env={"HOMEPAGE_URL": "https://example.com"},
                    crawl_fn=lambda homepage_url, cfg: (_ for _ in ()).throw(RuntimeError("boom")),
                    archive_timestamp="2026-03-25T00-00-00Z",
                )

            self.assertEqual(latest_snapshot.read_text(encoding="utf-8"), '{"baseline": true}')
            self.assertEqual(latest_report.read_text(encoding="utf-8"), "baseline report")


if __name__ == "__main__":
    unittest.main()
