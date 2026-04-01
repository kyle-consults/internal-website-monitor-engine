from __future__ import annotations

import hashlib
import json
import os
import re
import time
from difflib import SequenceMatcher
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable
from urllib.parse import urldefrag, urljoin, urlparse


class ConfigurationError(RuntimeError):
    """Raised when required monitor configuration is missing or invalid."""


@dataclass(frozen=True)
class MonitorPaths:
    root: Path
    config_path: Path
    snapshots_dir: Path
    reports_dir: Path
    latest_snapshot: Path
    latest_report: Path
    latest_summary: Path

    @classmethod
    def for_root(cls, root: Path) -> "MonitorPaths":
        root = root.resolve()
        snapshots_dir = root / "snapshots"
        reports_dir = root / "reports"
        return cls(
            root=root,
            config_path=root / "config" / "defaults.json",
            snapshots_dir=snapshots_dir,
            reports_dir=reports_dir,
            latest_snapshot=snapshots_dir / "latest-snapshot.json",
            latest_report=reports_dir / "latest-report.md",
            latest_summary=reports_dir / "latest-summary.json",
        )


SnapshotDict = dict[str, object]
CrawlFunction = Callable[[str, dict[str, object]], SnapshotDict]


def load_config(paths: MonitorPaths) -> dict[str, object]:
    with paths.config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_runtime_root(
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    module_file: Path | None = None,
) -> Path:
    env = env or os.environ
    override = env.get("WEBSITE_MONITOR_ROOT", "").strip()
    if override:
        return Path(override).resolve()

    cwd = (cwd or Path.cwd()).resolve()
    if (cwd / "config" / "defaults.json").exists():
        return cwd

    module_path = (module_file or Path(__file__)).resolve()
    for candidate in module_path.parents:
        if (candidate / "config" / "defaults.json").exists():
            return candidate

    return cwd


def get_homepage_url(env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    url = env.get("HOMEPAGE_URL", "").strip()
    if not url:
        raise ConfigurationError("HOMEPAGE_URL is not set.")
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return normalize_url(url)


def normalize_url(url: str) -> str:
    stripped, _fragment = urldefrag(url)
    parsed = urlparse(stripped)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"{scheme}://{netloc}{path}"


def should_skip_url(url: str, cfg: dict[str, object], allowed_host: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return True
    if parsed.netloc.lower() != allowed_host.lower():
        return True

    path = parsed.path.lower()
    for ext in cfg.get("exclude_extensions", []):
        if path.endswith(str(ext).lower()):
            return True

    lower_url = url.lower()
    for token in cfg.get("exclude_url_contains", []):
        if str(token).lower() in lower_url:
            return True

    return False


def host_aliases(host: str) -> set[str]:
    host = host.lower().strip()
    aliases = {host}
    if host.startswith("www."):
        aliases.add(host[4:])
    else:
        aliases.add(f"www.{host}")
    return aliases


def should_adopt_homepage_redirect_host(current_allowed_host: str, final_host: str, pages_scanned: int) -> bool:
    if pages_scanned != 0:
        return False
    return final_host.lower() in host_aliases(current_allowed_host)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
    text = re.sub(r"\b(last updated|updated)\b[: ]+.*?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^skip to (?:content|main|navigation)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*©\s*\d{4}\b.*$", "", text)
    text = re.sub(r"\s*Manage consent\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_previous_snapshot(paths: MonitorPaths) -> dict[str, object] | None:
    if not paths.latest_snapshot.exists():
        return None
    with paths.latest_snapshot.open("r", encoding="utf-8") as handle:
        return json.load(handle)


SIMILARITY_THRESHOLD = 0.97


def reconcile_redirects(
    added: list[str],
    removed: list[str],
    previous_pages: dict,
    current_pages: dict,
) -> tuple[list[str], list[str], list[str]]:
    removed_by_hash: dict[str, str] = {}
    for url in removed:
        h = previous_pages[url].get("hash", "")
        if h:
            removed_by_hash[h] = url

    final_added: list[str] = []
    redirected: list[str] = []
    matched_removed: set[str] = set()

    for url in added:
        h = current_pages[url].get("hash", "")
        if h and h in removed_by_hash:
            old_url = removed_by_hash.pop(h)
            matched_removed.add(old_url)
            redirected.append(f"{old_url} -> {url}")
        else:
            final_added.append(url)

    final_removed = [url for url in removed if url not in matched_removed]
    return final_added, final_removed, redirected


def compare_snapshots(previous: dict[str, object] | None, current: dict[str, object]) -> dict[str, list[str]]:
    previous_pages = previous.get("pages", {}) if previous else {}
    current_pages = current.get("pages", {})

    prev_urls = set(previous_pages.keys())
    curr_urls = set(current_pages.keys())

    raw_added = sorted(curr_urls - prev_urls)
    raw_removed = sorted(prev_urls - curr_urls)
    changed: list[str] = []
    unchanged: list[str] = []

    for url in sorted(prev_urls & curr_urls):
        prev_hash = previous_pages[url].get("hash", "")
        curr_hash = current_pages[url].get("hash", "")
        if prev_hash != curr_hash:
            prev_text = str(previous_pages[url].get("text", ""))
            curr_text = str(current_pages[url].get("text", ""))
            if prev_text and curr_text and similarity_score(prev_text, curr_text) >= SIMILARITY_THRESHOLD:
                unchanged.append(url)
            else:
                changed.append(url)
        else:
            unchanged.append(url)

    if previous is None:
        return {
            "added": sorted(curr_urls),
            "removed": [],
            "changed": [],
            "unchanged": [],
            "redirected": [],
        }

    added, removed, redirected = reconcile_redirects(
        raw_added, raw_removed, previous_pages, current_pages,
    )

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
        "redirected": redirected,
    }


def split_text_units(text: str) -> list[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []

    parts = [item.strip() for item in re.split(r"(?<=[.!?])\s+", normalized) if item.strip()]
    return parts or [normalized]


def similarity_score(left: str, right: str) -> float:
    return SequenceMatcher(a=left, b=right).ratio()


def summarize_text_changes(previous_text: str, current_text: str) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    previous_units = split_text_units(previous_text)
    current_units = split_text_units(current_text)
    removed: list[str] = []
    modified: list[tuple[str, str]] = []
    added: list[str] = []

    matcher = SequenceMatcher(a=previous_units, b=current_units)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            removed.extend(previous_units[i1:i2])
            continue
        if tag == "insert":
            added.extend(current_units[j1:j2])
            continue

        previous_block = previous_units[i1:i2]
        current_block = current_units[j1:j2]
        previous_joined = " ".join(previous_block).strip()
        current_joined = " ".join(current_block).strip()
        if previous_joined and current_joined and similarity_score(previous_joined, current_joined) >= 0.67:
            modified.append((previous_joined, current_joined))
            continue

        pair_count = min(len(previous_block), len(current_block))
        modified.extend((previous_block[index], current_block[index]) for index in range(pair_count))
        removed.extend(previous_block[pair_count:])
        added.extend(current_block[pair_count:])

    return removed, modified, added


def render_page_listing(url: str, page: dict[str, object]) -> str:
    status = page.get("status", "unknown")
    title = page.get("title", "") or "(untitled)"
    line = f"- {url} | status: {status} | title: {title}"

    error = str(page.get("error", "")).strip()
    if error:
        line = f"{line} | error: {error}"

    return line


def describe_page_changes(previous_page: dict[str, object], current_page: dict[str, object]) -> list[str]:
    lines: list[str] = []

    if previous_page.get("title", "") != current_page.get("title", ""):
        lines.append(f'- Title changed: "{previous_page.get("title", "")}" -> "{current_page.get("title", "")}"')

    if previous_page.get("h1", "") != current_page.get("h1", ""):
        lines.append(f'- H1 changed: "{previous_page.get("h1", "")}" -> "{current_page.get("h1", "")}"')

    if previous_page.get("status") != current_page.get("status"):
        lines.append(f'- Status changed: "{previous_page.get("status")}" -> "{current_page.get("status")}"')

    previous_error = str(previous_page.get("error", "")).strip()
    current_error = str(current_page.get("error", "")).strip()
    if previous_error != current_error:
        if previous_error:
            lines.append(f"- Error removed: {previous_error}")
        if current_error:
            lines.append(f"- Error added: {current_error}")

    removed_text, modified_text, added_text = summarize_text_changes(
        str(previous_page.get("text", "")),
        str(current_page.get("text", "")),
    )
    lines.extend(f'- Text modified: "{before}" -> "{after}"' for before, after in modified_text)
    lines.extend(f"- Text removed: {item}" for item in removed_text)
    lines.extend(f"- Text added: {item}" for item in added_text)

    if not lines:
        lines.append("- Hash changed, but no field-level difference could be summarized.")

    return lines


_BOILERPLATE_SELECTORS = [
    "nav",
    "header",
    "footer",
    "[role='banner']",
    "[role='navigation']",
    "[role='contentinfo']",
    ".cookie-banner",
    "#cookie-consent",
    ".cookie-consent",
    "[class*='cookie']",
    "[id*='cookie']",
    "[class*='consent']",
]


def strip_boilerplate_js() -> str:
    selector_list = ", ".join(f"'{s}'" for s in _BOILERPLATE_SELECTORS)
    return (
        f"for (const sel of [{selector_list}]) {{"
        "  document.querySelectorAll(sel).forEach(el => el.remove());"
        "}"
    )


def extract_primary_text(page) -> str:
    try:
        page.evaluate(strip_boilerplate_js())
    except Exception:
        pass

    selectors = [
        "main article",
        "main",
        "[role='main']",
        "article",
        "body",
    ]

    for selector in selectors:
        try:
            locator = page.locator(selector)
            if locator.count() == 0:
                continue
            text = locator.first.inner_text(timeout=5000)
        except Exception:
            continue

        cleaned = clean_text(text)
        if cleaned:
            return cleaned

    return ""


def render_report(
    current: dict[str, object],
    diff: dict[str, list[str]],
    baseline_created: bool,
    previous: dict[str, object] | None = None,
) -> str:
    scanned_at = str(current["scanned_at"])
    homepage = str(current["homepage_url"])
    pages = current["pages"]

    lines = [
        "# Website Change Report",
        "",
        f"- Homepage: {homepage}",
        f"- Scanned at: {scanned_at}",
        f"- Pages scanned: {len(pages)}",
        f"- Added pages: {len(diff['added'])}",
        f"- Removed pages: {len(diff['removed'])}",
        f"- Changed pages: {len(diff['changed'])}",
        "",
    ]

    if baseline_created:
        lines.append("Initial baseline established.")
        lines.append("")

    if diff.get("redirected"):
        lines.append("## Redirected")
        for entry in diff["redirected"]:
            lines.append(f"- {entry}")
        lines.append("")

    if diff["added"]:
        lines.append("## Added")
        for url in diff["added"]:
            lines.append(render_page_listing(url, pages[url]))
        lines.append("")

    if diff["removed"] and previous is not None:
        lines.append("## Removed")
        previous_pages = previous.get("pages", {})
        for url in diff["removed"]:
            lines.append(render_page_listing(url, previous_pages[url]))
        lines.append("")

    if diff["changed"] and previous is not None:
        lines.append("## Changed")
        previous_pages = previous.get("pages", {})
        for url in diff["changed"]:
            lines.append(f"### {url}")
            lines.extend(describe_page_changes(previous_pages[url], pages[url]))
            lines.append("")
        lines.append("")

    if not diff["added"] and not diff["removed"] and not diff["changed"]:
        lines.append("No changes detected.")

    return "\n".join(lines).rstrip() + "\n"


def build_summary(current: dict[str, object], diff: dict[str, list[str]], baseline_created: bool) -> dict[str, object]:
    counts = {
        "pages_scanned": len(current["pages"]),
        "added": len(diff["added"]),
        "removed": len(diff["removed"]),
        "changed": len(diff["changed"]),
    }
    return {
        "homepage_url": current["homepage_url"],
        "scanned_at": current["scanned_at"],
        "baseline_created": baseline_created,
        "changes_detected": bool(counts["added"] or counts["removed"] or counts["changed"]),
        "counts": counts,
    }


def should_persist_run(diff: dict[str, list[str]], baseline_created: bool) -> bool:
    if baseline_created:
        return True
    return bool(diff["added"] or diff["removed"] or diff["changed"])


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def prune_archives(directory: Path, pattern: str, keep: int) -> list[str]:
    keep = max(0, keep)
    candidates = sorted(directory.glob(pattern))
    if len(candidates) <= keep:
        return []

    to_remove = candidates[: len(candidates) - keep]
    removed_names = [path.name for path in to_remove]
    for path in to_remove:
        path.unlink()
    return removed_names


def archive_timestamp_value(override: str | None = None) -> str:
    if override:
        return override
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def persist_outputs(
    paths: MonitorPaths,
    current: dict[str, object],
    report_text: str,
    summary: dict[str, object],
    archive_timestamp: str,
    keep_archives: int,
) -> None:
    paths.snapshots_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    snapshot_archive = paths.snapshots_dir / f"snapshot-{archive_timestamp}.json"
    report_archive = paths.reports_dir / f"report-{archive_timestamp}.md"
    summary_archive = paths.reports_dir / f"summary-{archive_timestamp}.json"

    # Write archives first so failures do not replace the previous latest outputs.
    write_json_atomic(snapshot_archive, current)
    write_text_atomic(report_archive, report_text)
    write_json_atomic(summary_archive, summary)
    write_json_atomic(paths.latest_snapshot, current)
    write_text_atomic(paths.latest_report, report_text)
    write_json_atomic(paths.latest_summary, summary)

    prune_archives(paths.snapshots_dir, "snapshot-*.json", keep_archives)
    prune_archives(paths.reports_dir, "report-*.md", keep_archives)
    prune_archives(paths.reports_dir, "summary-*.json", keep_archives)


def wait_for_content_stable(page, timeout_ms: int = 3000, interval_ms: int = 500) -> None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    interval_s = interval_ms / 1000.0
    try:
        previous_text = page.evaluate("() => document.body?.innerText || ''")
    except Exception:
        return
    while time.monotonic() < deadline:
        time.sleep(interval_s)
        try:
            current_text = page.evaluate("() => document.body?.innerText || ''")
        except Exception:
            return
        if current_text == previous_text:
            return
        previous_text = current_text


def extract_page_data(page, page_url: str) -> dict[str, object]:
    title = ""
    try:
        title = page.title() or ""
    except Exception:
        title = ""

    headings: list[str] = []
    try:
        headings = page.locator("h1").all_inner_texts()
    except Exception:
        headings = []

    cleaned_body = extract_primary_text(page)
    return {
        "url": page_url,
        "title": clean_text(title),
        "h1": " | ".join(clean_text(item) for item in headings if item.strip()),
        "text": cleaned_body,
        "hash": hash_text(cleaned_body),
    }


def discover_links(page, base_url: str) -> list[str]:
    try:
        hrefs = page.locator("a[href]").evaluate_all("(nodes) => nodes.map((node) => node.getAttribute('href'))")
    except Exception:
        return []

    links: list[str] = []
    for href in hrefs:
        if not href or href == "None":
            continue
        links.append(normalize_url(urljoin(base_url, href)))
    return links


def crawl(homepage_url: str, cfg: dict[str, object]) -> dict[str, object]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed. Install dependencies before running the scanner.") from exc

    allowed_host = urlparse(homepage_url).netloc.lower()
    homepage_seed = normalize_url(homepage_url)
    queue: deque[str] = deque([homepage_seed])
    seen: set[str] = set()
    pages: dict[str, dict[str, object]] = {}
    max_pages = int(cfg.get("max_pages", 100))
    timeout_ms = int(cfg.get("request_timeout_ms", 20000))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        try:
            while queue and len(pages) < max_pages:
                url = queue.popleft()
                if url in seen:
                    continue
                seen.add(url)

                if should_skip_url(url, cfg, allowed_host):
                    continue

                page = context.new_page()
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    final_url = normalize_url(page.url)
                    final_host = urlparse(final_url).netloc.lower()
                    if url == homepage_seed and should_adopt_homepage_redirect_host(allowed_host, final_host, len(pages)):
                        allowed_host = final_host
                    if should_skip_url(final_url, cfg, allowed_host):
                        continue

                    wait_for_content_stable(page)

                    discovered_links = discover_links(page, final_url)

                    page_data = extract_page_data(page, final_url)
                    page_data["status"] = response.status if response else None
                    pages[final_url] = page_data

                    for discovered in discovered_links:
                        if discovered in seen or discovered in queue:
                            continue
                        if should_skip_url(discovered, cfg, allowed_host):
                            continue
                        queue.append(discovered)
                except Exception as exc:
                    pages[url] = {
                        "url": url,
                        "title": "",
                        "h1": "",
                        "text": "",
                        "hash": "",
                        "status": None,
                        "error": str(exc),
                    }
                finally:
                    page.close()
        finally:
            browser.close()

    return {
        "homepage_url": homepage_url,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "pages": dict(sorted(pages.items())),
    }


def run_monitor(
    paths: MonitorPaths,
    env: dict[str, str] | None = None,
    crawl_fn: CrawlFunction | None = None,
    archive_timestamp: str | None = None,
) -> dict[str, object]:
    cfg = load_config(paths)
    homepage_url = get_homepage_url(env)
    previous = load_previous_snapshot(paths)
    current = (crawl_fn or crawl)(homepage_url, cfg)
    diff = compare_snapshots(previous, current)
    baseline_created = previous is None
    changes_detected = bool(diff["added"] or diff["removed"] or diff["changed"])
    if changes_detected and not baseline_created:
        changed_count = len(diff["added"]) + len(diff["removed"]) + len(diff["changed"])
        total_count = len(current.get("pages", {}))
        if changed_count > total_count * 0.5:
            print(
                f"Note: {changed_count}/{total_count} pages changed. "
                "If you recently updated the monitor engine, this is expected "
                "on the first scan and will resolve on the next run."
            )
    report_text = render_report(current, diff, baseline_created, previous=previous)
    summary = build_summary(current, diff, baseline_created)
    keep_archives = int(cfg.get("archive_retention", 12))
    persisted = should_persist_run(diff, baseline_created)
    if persisted:
        persist_outputs(
            paths=paths,
            current=current,
            report_text=report_text,
            summary=summary,
            archive_timestamp=archive_timestamp_value(archive_timestamp),
            keep_archives=keep_archives,
        )

    return {
        "homepage_url": homepage_url,
        "current": current,
        "diff": diff,
        "summary": summary,
        "baseline_created": baseline_created,
        "persisted": persisted,
    }


def main() -> None:
    paths = MonitorPaths.for_root(resolve_runtime_root())
    result = run_monitor(paths=paths)
    print(f"Scan complete for {result['homepage_url']}.")
