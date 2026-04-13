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

from website_monitor.knowledge import build_gemini_client, extract_all_pages, verify_changes
from website_monitor.knowledge_diff import compare_knowledge
from website_monitor.knowledge_report import render_knowledge_report, build_knowledge_summary
from website_monitor.webhook import send_webhook


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
    latest_knowledge: Path

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
            latest_knowledge=snapshots_dir / "latest-knowledge.json",
        )


SnapshotDict = dict[str, object]
CrawlFunction = Callable[[str, dict[str, object]], SnapshotDict]
VerifyFunction = Callable[[list[str], dict[str, object]], dict[str, dict[str, object]]]


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


def load_previous_knowledge(paths: MonitorPaths) -> dict[str, object] | None:
    if not paths.latest_knowledge.exists():
        return None
    with paths.latest_knowledge.open("r", encoding="utf-8") as handle:
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


def reconcile_verified_changes(
    diff: dict[str, list[str]],
    previous: dict[str, object],
    current: dict[str, object],
    verified: dict[str, dict[str, object]],
) -> dict[str, list[str]]:
    """Re-classify changed pages using a second capture.

    For each url in diff["changed"], compare the verified hash to the previous
    and current hashes:
      - matches previous hash -> flap, drop from changed and overwrite
        current["pages"][url] with the previous entry so the snapshot reflects
        the stable state.
      - matches current hash -> confirmed change, keep in changed.
      - matches neither -> page is unstable, keep in changed and record in
        diff["unstable"] so the reporter can flag it.
      - url not present in verified (re-fetch failed) -> leave unchanged.
    """
    previous_pages = previous.get("pages", {}) if previous else {}
    current_pages = current.get("pages", {})

    kept_changed: list[str] = []
    flapped: list[str] = []
    unstable: list[str] = []
    extraction_failed: list[str] = []

    # A page is treated as extraction_failed when the new capture text is
    # empty but the previous capture had substantial content. Empty extraction
    # is almost always a scraper failure (selector mismatch, timeout) rather
    # than a real "page deleted all its content" event. We restore the
    # previous snapshot so the next scan can try again on a clean baseline.
    extraction_min_previous_chars = 100

    for url in diff.get("changed", []):
        previous_entry = previous_pages.get(url, {}) if isinstance(previous_pages, dict) else {}
        current_entry = current_pages.get(url, {}) if isinstance(current_pages, dict) else {}
        previous_text = str(previous_entry.get("text", "")) if isinstance(previous_entry, dict) else ""
        current_text = str(current_entry.get("text", "")) if isinstance(current_entry, dict) else ""

        if not current_text and len(previous_text) >= extraction_min_previous_chars:
            extraction_failed.append(url)
            if isinstance(current_pages, dict) and isinstance(previous_entry, dict):
                current_pages[url] = dict(previous_entry)
            continue

        verified_entry = verified.get(url)
        if not verified_entry:
            kept_changed.append(url)
            continue

        verified_hash = verified_entry.get("hash", "")
        previous_hash = previous_entry.get("hash", "") if isinstance(previous_entry, dict) else ""
        current_hash = current_entry.get("hash", "") if isinstance(current_entry, dict) else ""

        if verified_hash and verified_hash == previous_hash:
            flapped.append(url)
            if isinstance(current_pages, dict) and isinstance(previous_entry, dict):
                current_pages[url] = dict(previous_entry)
            continue

        kept_changed.append(url)
        if verified_hash and verified_hash != current_hash:
            unstable.append(url)

    result = dict(diff)
    result["changed"] = kept_changed
    result["flapped"] = flapped
    result["unstable"] = unstable
    result["extraction_failed"] = extraction_failed
    return result


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
    # Semantic chrome
    "nav",
    "header",
    "footer",
    "aside",
    "[role='banner']",
    "[role='navigation']",
    "[role='contentinfo']",
    "[role='complementary']",
    # Cookie/consent banners
    ".cookie-banner",
    "#cookie-consent",
    ".cookie-consent",
    "[class*='cookie']",
    "[id*='cookie']",
    "[class*='consent']",
    # WordPress / Astra / Elementor sidebar and widget patterns. These often
    # live inside <main> on theme-heavy sites and are the primary source of
    # capture-to-capture flap noise (recent posts, related posts, sidebar
    # menus that lazy-load after first paint).
    "[class*='sidebar']",
    "[id*='sidebar']",
    "[class*='widget']",
    "[id*='widget']",
    "[class*='recent']",
    "[class*='related']",
    "[class*='menu']",
    "[id*='menu']",
    "[class*='subnav']",
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

    # Two-stage extraction:
    # 1. Walk the semantic main-content chain. The first selector with count>0
    #    wins. Fall through ONLY when a selector has zero matches (structural
    #    miss). If a selector matches but inner_text raises (race condition,
    #    timeout), return empty without falling through - the verifier will
    #    catch this case.
    # 2. If no semantic selector matched anywhere, fall back to body. This is
    #    safe for non-semantic pages (legal pages, WordPress page templates
    #    without <main>) because the absence of <main> is structural, not
    #    racy. The boilerplate strip removes nav/sidebar/widget patterns
    #    before we get here.
    semantic_selectors = [
        "main article",
        "main",
        "[role='main']",
        "article",
    ]

    for selector in semantic_selectors:
        try:
            locator = page.locator(selector)
            count = locator.count()
        except Exception:
            continue

        if count == 0:
            continue

        try:
            text = locator.first.inner_text(timeout=5000)
        except Exception:
            return ""

        return clean_text(text)

    try:
        body_locator = page.locator("body")
        if body_locator.count() == 0:
            return ""
        body_text = body_locator.first.inner_text(timeout=5000)
    except Exception:
        return ""

    return clean_text(body_text)


def diff_size_chars(previous_page: dict[str, object], current_page: dict[str, object]) -> int:
    """Sum of character lengths across all changed text content for a page.

    Used to flag pages with unusually large diffs as worth a manual look.
    """
    removed, modified, added = summarize_text_changes(
        str(previous_page.get("text", "")),
        str(current_page.get("text", "")),
    )
    total = sum(len(item) for item in removed)
    total += sum(len(before) + len(after) for before, after in modified)
    total += sum(len(item) for item in added)
    return total


def render_report(
    current: dict[str, object],
    diff: dict[str, list[str]],
    baseline_created: bool,
    previous: dict[str, object] | None = None,
    review_threshold_chars: int = 500,
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
        unstable_set = set(diff.get("unstable", []))
        for url in diff["changed"]:
            tags: list[str] = []
            if url in unstable_set:
                tags.append("unstable")
            if diff_size_chars(previous_pages[url], pages[url]) >= review_threshold_chars:
                tags.append("TO REVIEW")
            suffix = f" ({', '.join(tags)})" if tags else ""
            lines.append(f"### {url}{suffix}")
            lines.extend(describe_page_changes(previous_pages[url], pages[url]))
            lines.append("")
        lines.append("")

    if diff.get("flapped"):
        lines.append("## Flapped (auto-dismissed)")
        lines.append(
            "These pages reported a change in the first capture but matched the "
            "previous baseline on re-verification. Treated as noise and not persisted."
        )
        for url in diff["flapped"]:
            lines.append(f"- {url}")
        lines.append("")

    if diff.get("extraction_failed"):
        lines.append("## Extraction Failed (kept previous snapshot)")
        lines.append(
            "These pages had substantial content in the previous baseline but "
            "the new capture returned empty text. This is almost always a "
            "scraper failure (selector mismatch, timeout) rather than a real "
            "content removal. The previous snapshot has been kept and the next "
            "scan will retry."
        )
        for url in diff["extraction_failed"]:
            lines.append(f"- {url}")
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
    knowledge: dict[str, object] | None = None,
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

    if knowledge is not None:
        knowledge_archive = paths.snapshots_dir / f"knowledge-{archive_timestamp}.json"
        write_json_atomic(knowledge_archive, knowledge)
        write_json_atomic(paths.latest_knowledge, knowledge)
        prune_archives(paths.snapshots_dir, "knowledge-*.json", keep_archives)


def _default_stability_text_fn(page) -> str:
    return page.evaluate("() => document.body?.innerText || ''")


def wait_for_content_stable(
    page,
    timeout_ms: int = 15000,
    interval_ms: int = 750,
    required_matches: int = 3,
    text_fn: Callable[[object], str] | None = None,
) -> None:
    """Poll a text source until it returns the same value `required_matches`
    times in a row, or the deadline elapses.

    By default polls `document.body.innerText` so callers can stabilize a page
    without committing to an extraction strategy. The crawler injects
    `text_fn=extract_primary_text` so we stabilize on the *exact* text that
    will be hashed, eliminating the case where body settles before the locked
    main-content selector finishes rendering.

    `required_matches` defaults to 3 (not 2) because lazy-loading WordPress
    sidebars often arrive in waves with brief idle gaps; two consecutive
    matches can false-positive on a lull between waves.
    """
    if required_matches < 1:
        required_matches = 1
    fetch = text_fn if text_fn is not None else _default_stability_text_fn

    deadline = time.monotonic() + timeout_ms / 1000.0
    interval_s = interval_ms / 1000.0

    try:
        last_text = fetch(page)
    except Exception:
        return

    matches = 1
    if matches >= required_matches:
        return

    while time.monotonic() < deadline:
        time.sleep(interval_s)
        try:
            current_text = fetch(page)
        except Exception:
            return
        if current_text == last_text:
            matches += 1
            if matches >= required_matches:
                return
        else:
            matches = 1
            last_text = current_text


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
    timeout_ms = int(cfg.get("request_timeout_ms", 30000))

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
                    response = page.goto(url, wait_until="load", timeout=timeout_ms)
                    final_url = normalize_url(page.url)
                    final_host = urlparse(final_url).netloc.lower()
                    if url == homepage_seed and should_adopt_homepage_redirect_host(allowed_host, final_host, len(pages)):
                        allowed_host = final_host
                    if should_skip_url(final_url, cfg, allowed_host):
                        continue

                    wait_for_content_stable(page, text_fn=extract_primary_text)

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


def recrawl_urls(urls: list[str], cfg: dict[str, object]) -> dict[str, dict[str, object]]:
    """Re-fetch the given URLs once and return fresh page data keyed by URL.

    Used by run_monitor to double-capture pages that were flagged as "changed"
    in the primary scan. Pages that fail to re-fetch are omitted from the
    result, which the reconciler treats as "leave the original diff entry
    alone" (fail-safe: never drop a real change on infra error).
    """
    if not urls:
        return {}

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed. Install dependencies before running the scanner.") from exc

    timeout_ms = int(cfg.get("request_timeout_ms", 30000))
    verified: dict[str, dict[str, object]] = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        try:
            for url in urls:
                page = context.new_page()
                try:
                    response = page.goto(url, wait_until="load", timeout=timeout_ms)
                    wait_for_content_stable(page, text_fn=extract_primary_text)
                    final_url = normalize_url(page.url)
                    page_data = extract_page_data(page, final_url)
                    page_data["status"] = response.status if response else None
                    verified[url] = page_data
                except Exception:
                    # Fail-safe: omit. Reconciler will preserve the original diff entry.
                    continue
                finally:
                    page.close()
        finally:
            browser.close()

    return verified


def run_knowledge_pipeline(
    crawl_result: dict[str, object],
    cfg: dict[str, object],
    client: object,
    previous_snapshot: dict[str, object] | None,
    previous_knowledge: dict[str, object] | None,
    baseline_created: bool,
) -> tuple[dict[str, object], dict[str, list[dict]], str, dict[str, object]]:
    """Run knowledge extraction, comparison, and report.

    Returns (knowledge, diff, report_text, summary).
    """
    model = str(cfg.get("gemini_model", "gemini-2.5-flash-lite"))
    knowledge = extract_all_pages(crawl_result, client, model, previous_snapshot, previous_knowledge)
    knowledge_diff = compare_knowledge(previous_knowledge, knowledge)
    # LLM verification pass: filter noise from real changes
    if not baseline_created and (knowledge_diff.get("changed") or knowledge_diff.get("added") or knowledge_diff.get("removed")):
        knowledge_diff = verify_changes(knowledge_diff, client, model)
    report_text = render_knowledge_report(knowledge, knowledge_diff, baseline_created)
    summary = build_knowledge_summary(knowledge, knowledge_diff, baseline_created)
    return knowledge, knowledge_diff, report_text, summary


def run_monitor(
    paths: MonitorPaths,
    env: dict[str, str] | None = None,
    crawl_fn: CrawlFunction | None = None,
    verify_fn: VerifyFunction | None = None,
    archive_timestamp: str | None = None,
    gemini_client: object | None = None,
) -> dict[str, object]:
    cfg = load_config(paths)
    env = env or os.environ
    homepage_url = get_homepage_url(env)
    previous = load_previous_snapshot(paths)
    current = (crawl_fn or crawl)(homepage_url, cfg)
    baseline_created = previous is None

    # Determine if knowledge pipeline is available
    client = gemini_client
    if client is None:
        api_key = env.get("GEMINI_API_KEY", "").strip()
        if api_key:
            client = build_gemini_client(api_key)

    knowledge = None
    if client is not None:
        # Knowledge pipeline path
        previous_knowledge = load_previous_knowledge(paths)

        # Bug 2 fix: when adding Gemini to an already-monitored site,
        # previous_knowledge is None but previous (raw snapshot) exists.
        # Treat this as a knowledge baseline run, not a change detection run.
        knowledge_baseline = previous_knowledge is None and previous is not None
        if knowledge_baseline:
            baseline_created = True

        knowledge, diff, report_text, summary = run_knowledge_pipeline(
            current, cfg, client, previous, previous_knowledge, baseline_created,
        )

        # Bug 1 fix: if ALL pages have empty knowledge_units but the crawl
        # had pages with text content, the Gemini API likely failed entirely.
        # Do not persist the empty knowledge snapshot so the next run re-extracts.
        knowledge_pages = knowledge.get("pages", {}) if knowledge else {}
        has_any_units = any(
            p.get("knowledge_units") for p in knowledge_pages.values()
        )
        crawl_pages = current.get("pages", {})
        has_text_content = any(
            str(p.get("text", "")).strip() for p in crawl_pages.values()
        )
        if not has_any_units and has_text_content:
            # Re-render report with extraction failure note before nullifying
            failed_snapshot = dict(knowledge) if knowledge else dict(current)
            knowledge = None
            report_text = render_knowledge_report(
                failed_snapshot,
                diff,
                baseline_created,
                extraction_notes=["All knowledge extractions returned empty — possible Gemini API failure. "
                                  "Knowledge snapshot was not persisted; next run will retry."],
            )
            summary = build_knowledge_summary(
                failed_snapshot,
                diff,
                baseline_created,
            )
        # Optional webhook
        webhook_url = cfg.get("webhook_url")
        if webhook_url and (diff.get("changed") or diff.get("added") or diff.get("removed")):
            send_webhook(str(webhook_url), {
                "site": homepage_url,
                "scanned_at": current.get("scanned_at", ""),
                "changes": diff.get("changed", []) + diff.get("added", []) + diff.get("removed", []),
            })
    else:
        # Raw diff fallback path (existing behavior, unchanged)
        diff = compare_snapshots(previous, current)
        if not baseline_created and diff.get("changed"):
            verify_impl = verify_fn if verify_fn is not None else recrawl_urls
            try:
                verified = verify_impl(list(diff["changed"]), cfg)
            except Exception:
                verified = {}
            diff = reconcile_verified_changes(diff, previous or {}, current, verified)
        else:
            diff.setdefault("flapped", [])
            diff.setdefault("unstable", [])
            diff.setdefault("extraction_failed", [])
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
        review_threshold = int(cfg.get("review_threshold_chars", 500))
        report_text = render_report(
            current,
            diff,
            baseline_created,
            previous=previous,
            review_threshold_chars=review_threshold,
        )
        summary = build_summary(current, diff, baseline_created)

    changes_detected = summary.get("changes_detected", False)
    keep_archives = int(cfg.get("archive_retention", 12))
    persisted = baseline_created or changes_detected
    if persisted:
        persist_outputs(
            paths=paths,
            current=current,
            report_text=report_text,
            summary=summary,
            archive_timestamp=archive_timestamp_value(archive_timestamp),
            keep_archives=keep_archives,
            knowledge=knowledge,
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
