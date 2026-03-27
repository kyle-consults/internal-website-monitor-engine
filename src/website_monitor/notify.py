from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from website_monitor.monitor import ConfigurationError, MonitorPaths, resolve_runtime_root


class NotificationError(RuntimeError):
    """Raised when notification delivery fails."""


@dataclass(frozen=True)
class NotificationSettings:
    recipients: list[str]
    sender: str
    api_key: str
    email_mode: str


def parse_recipients(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def should_send_email(summary: dict[str, object], email_mode: str) -> bool:
    normalized_mode = (email_mode or "changes_only").strip().lower()
    if normalized_mode == "always":
        return True
    if normalized_mode == "never":
        return False
    if normalized_mode != "changes_only":
        raise ConfigurationError(f"Unsupported EMAIL_MODE: {email_mode}")
    return bool(summary.get("baseline_created") or summary.get("changes_detected"))


def homepage_label(homepage_url: str) -> str:
    return urlparse(homepage_url).netloc or homepage_url


def build_email_subject(summary: dict[str, object]) -> str:
    label = homepage_label(str(summary["homepage_url"]))
    if summary.get("baseline_created"):
        prefix = "Baseline established"
    elif summary.get("changes_detected"):
        prefix = "Changes detected"
    else:
        prefix = "No changes"
    return f"[website-monitor] {prefix} for {label}"


def truncate_all_pages_section(report: str) -> str:
    marker_start = "## All Pages Scraped\n"
    start = report.find(marker_start)
    if start == -1:
        return report

    section_body_start = start + len(marker_start)
    next_heading = report.find("\n## ", section_body_start)
    if next_heading == -1:
        section_end = len(report)
    else:
        section_end = next_heading

    page_lines = [
        line for line in report[section_body_start:section_end].splitlines() if line.startswith("- ")
    ]
    total = len(page_lines)
    max_shown = 10
    if total <= max_shown:
        return report

    kept = "\n".join(page_lines[:max_shown])
    truncated_section = f"{kept}\n- ... and {total - max_shown} more pages\n"
    return report[:section_body_start] + truncated_section + report[section_end:]


def build_email_text(summary: dict[str, object], report_text: str) -> str:
    counts = summary["counts"]
    trimmed_report = truncate_all_pages_section(report_text.strip())
    lines = [
        "Website monitor result",
        "",
        f"Homepage: {summary['homepage_url']}",
        f"Scanned at: {summary['scanned_at']}",
        f"Pages scanned: {counts['pages_scanned']}",
        f"Added pages: {counts['added']}",
        f"Removed pages: {counts['removed']}",
        f"Changed pages: {counts['changed']}",
        "",
        "Report:",
        "",
        trimmed_report,
    ]
    return "\n".join(lines)


def build_resend_payload(
    summary: dict[str, object],
    report_text: str,
    sender: str,
    recipients: list[str],
) -> dict[str, object]:
    return {
        "from": sender,
        "to": recipients,
        "subject": build_email_subject(summary),
        "text": build_email_text(summary, report_text),
    }


def build_resend_request(api_key: str, payload: dict[str, object]) -> Request:
    return Request(
        url="https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "website-monitor-engine/0.1",
        },
        method="POST",
    )


def read_summary(paths: MonitorPaths) -> dict[str, object]:
    return json.loads(paths.latest_summary.read_text(encoding="utf-8"))


def read_report(paths: MonitorPaths) -> str:
    return paths.latest_report.read_text(encoding="utf-8")


def load_notification_settings(env: dict[str, str] | None = None) -> NotificationSettings | None:
    env = env or os.environ
    raw_to = env.get("ALERT_EMAIL_TO", "").strip()
    raw_from = env.get("ALERT_EMAIL_FROM", "").strip()
    api_key = env.get("RESEND_API_KEY", "").strip()
    email_mode = env.get("EMAIL_MODE", "changes_only").strip() or "changes_only"

    if not raw_to and not raw_from and not api_key:
        return None

    if not raw_to or not raw_from or not api_key:
        return None

    recipients = parse_recipients(raw_to)
    if not recipients:
        return None

    return NotificationSettings(
        recipients=recipients,
        sender=raw_from,
        api_key=api_key,
        email_mode=email_mode,
    )


def send_resend_email(api_key: str, payload: dict[str, object]) -> dict[str, object]:
    request = build_resend_request(api_key=api_key, payload=payload)

    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise NotificationError(f"Resend request failed with status {exc.code}: {details}") from exc
    except URLError as exc:
        raise NotificationError(f"Resend request failed: {exc.reason}") from exc


def send_notification(
    paths: MonitorPaths,
    env: dict[str, str] | None = None,
    send_fn: Callable[[str, dict[str, object]], dict[str, object]] | None = None,
) -> dict[str, object]:
    settings = load_notification_settings(env)
    if settings is None:
        return {"status": "skipped", "reason": "email_not_configured"}

    summary = read_summary(paths)
    if not should_send_email(summary, settings.email_mode):
        return {"status": "skipped", "reason": "policy_no_send"}

    payload = build_resend_payload(
        summary=summary,
        report_text=read_report(paths),
        sender=settings.sender,
        recipients=settings.recipients,
    )
    try:
        response = (send_fn or send_resend_email)(settings.api_key, payload)
    except NotificationError:
        raise
    except Exception as exc:
        raise NotificationError(f"Notification delivery failed: {exc}") from exc
    return {
        "status": "sent",
        "reason": "sent",
        "payload": payload,
        "response": response,
    }


def main() -> None:
    paths = MonitorPaths.for_root(resolve_runtime_root())
    result = send_notification(paths=paths)
    print(f"Notification {result['status']}: {result['reason']}")


if __name__ == "__main__":
    main()
