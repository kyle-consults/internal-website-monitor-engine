# Knowledge report: deterministic template rendering (no LLM).
#
#   knowledge_snapshot + diff ──> render_knowledge_report ──> markdown string
#   knowledge_snapshot + diff ──> build_knowledge_summary  ──> summary dict

from __future__ import annotations

from typing import Any, Callable


def _render_grouped_section(
    entries: list[dict],
    format_entry: Callable[[dict], str],
    suffix: str = "",
) -> str:
    """Group entries by category and render each with the format function.

    Returns a markdown string with ### Category headers and bulleted items.
    """
    by_category: dict[str, list[dict]] = {}
    for entry in entries:
        cat = entry.get("category", "other")
        by_category.setdefault(cat, []).append(entry)

    lines: list[str] = []
    for category in sorted(by_category):
        title = category.capitalize()
        if suffix:
            title = f"{title} ({suffix})"
        lines.append(f"### {title}")
        lines.append("")
        for entry in by_category[category]:
            lines.append(f"- {format_entry(entry)}")
        lines.append("")

    return "\n".join(lines)


def _fmt_changed(entry: dict) -> str:
    return (
        f'**{entry["label"]}** changed: '
        f'was "{entry["old_value"]}", now "{entry["new_value"]}" '
        f'(source: {entry["page"]})'
    )


def _fmt_added(entry: dict) -> str:
    return (
        f'**{entry["label"]}** added: '
        f'"{entry["value"]}" '
        f'(source: {entry["page"]})'
    )


def _fmt_removed(entry: dict) -> str:
    return (
        f'**{entry["label"]}** removed: '
        f'"{entry["value"]}" '
        f'(source: {entry["page"]})'
    )


def render_knowledge_report(
    knowledge_snapshot: dict[str, Any],
    diff: dict[str, list[dict]],
    baseline_created: bool,
    raw_fallback_pages: list[dict] | None = None,
    extraction_notes: list[str] | None = None,
) -> str:
    """Render a deterministic markdown report from a knowledge snapshot and diff.

    Returns a markdown string suitable for email or webhook payloads.
    """
    pages = knowledge_snapshot.get("pages", {})
    pages_scanned = len(pages)
    operational_count = sum(
        1 for p in pages.values()
        if any(u.get("operational", True) for u in p.get("knowledge_units", []))
    )

    changed = diff.get("changed", [])
    added = diff.get("added", [])
    removed = diff.get("removed", [])
    total_changes = len(changed) + len(added) + len(removed)

    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append(f"# Knowledge Report — {knowledge_snapshot.get('homepage_url', 'Unknown')}")
    lines.append("")
    lines.append(f"**Scanned at:** {knowledge_snapshot.get('extracted_at', 'N/A')}")
    lines.append(f"**Pages scanned:** {pages_scanned}")
    lines.append(f"**Operational pages:** {operational_count}")
    lines.append(f"**Knowledge changes:** {total_changes}")
    lines.append("")

    # ── Baseline notice ─────────────────────────────────────────────────────
    if baseline_created:
        lines.append("Initial knowledge baseline established.")
        lines.append("")

    # ── Changes ─────────────────────────────────────────────────────────────
    if total_changes > 0:
        lines.append("## Changes Detected")
        lines.append("")

        if changed:
            lines.append(_render_grouped_section(changed, _fmt_changed))
        if added:
            lines.append(_render_grouped_section(added, _fmt_added, suffix="New"))
        if removed:
            lines.append(_render_grouped_section(removed, _fmt_removed, suffix="Removed"))
    elif not baseline_created:
        lines.append("No knowledge changes detected.")
        lines.append("")

    # ── Raw fallback ────────────────────────────────────────────────────────
    if raw_fallback_pages:
        lines.append("## Fallback: Raw Changes")
        lines.append("")
        for fb in raw_fallback_pages:
            lines.append(f"**{fb.get('page', 'Unknown page')}**")
            lines.append("")
            lines.append(fb.get("diff_summary", ""))
            lines.append("")

    # ── Extraction notes ────────────────────────────────────────────────────
    if extraction_notes:
        lines.append("## Extraction Notes")
        lines.append("")
        for note in extraction_notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def build_knowledge_summary(
    knowledge_snapshot: dict[str, Any],
    diff: dict[str, list[dict]],
    baseline_created: bool,
) -> dict[str, Any]:
    """Build a summary dict from a knowledge snapshot and diff.

    Returns dict with: homepage_url, scanned_at, baseline_created,
    changes_detected, counts (pages_scanned, added, removed, changed).
    """
    pages = knowledge_snapshot.get("pages", {})
    changed = diff.get("changed", [])
    added = diff.get("added", [])
    removed = diff.get("removed", [])

    return {
        "homepage_url": knowledge_snapshot.get("homepage_url", ""),
        "scanned_at": knowledge_snapshot.get("extracted_at", ""),
        "baseline_created": baseline_created,
        "changes_detected": len(changed) + len(added) + len(removed) > 0,
        "counts": {
            "pages_scanned": len(pages),
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
        },
    }
