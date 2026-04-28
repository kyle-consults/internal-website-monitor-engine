# Knowledge report: deterministic template rendering (no LLM).
#
#   knowledge_snapshot + diff ──> render_knowledge_report ──> HTML string
#   knowledge_snapshot + diff ──> build_knowledge_summary  ──> summary dict

from __future__ import annotations

from html import escape
from typing import Any
from urllib.parse import urlparse


def _domain(url: str) -> str:
    return urlparse(url).netloc or url


def _path(url: str) -> str:
    return urlparse(url).path or url


def render_knowledge_report(
    knowledge_snapshot: dict[str, Any],
    diff: dict[str, list[dict]],
    baseline_created: bool,
    raw_fallback_pages: list[dict] | None = None,
    extraction_notes: list[str] | None = None,
) -> str:
    """Render a clean HTML report from a knowledge snapshot and diff."""
    domain = _domain(knowledge_snapshot.get("homepage_url", ""))
    extracted_at = escape(knowledge_snapshot.get("extracted_at", "N/A"))
    pages = knowledge_snapshot.get("pages", {})
    pages_scanned = len(pages)

    changed = diff.get("changed", [])
    added = diff.get("added", [])
    removed = diff.get("removed", [])
    total_changes = len(changed) + len(added) + len(removed)

    html = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 640px;
    margin: 0 auto;
    padding: 40px 20px;
    color: #1a1a1a;
    background: #ffffff;
    line-height: 1.6;
  }
  .header {
    border-bottom: 1px solid #e5e5e5;
    padding-bottom: 20px;
    margin-bottom: 32px;
  }
  .header h1 {
    font-size: 20px;
    font-weight: 600;
    margin: 0 0 4px 0;
  }
  .header .meta {
    font-size: 13px;
    color: #666;
  }
  .badge {
    display: inline-block;
    font-size: 12px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
  }
  .badge-changes {
    background: #fef3c7;
    color: #92400e;
  }
  .badge-clear {
    background: #d1fae5;
    color: #065f46;
  }
  .badge-baseline {
    background: #dbeafe;
    color: #1e40af;
  }
  .section {
    margin-bottom: 32px;
  }
  .section h2 {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #666;
    margin: 0 0 16px 0;
  }
  .change-item {
    padding: 16px 0;
    border-bottom: 1px solid #f0f0f0;
  }
  .change-item:last-child {
    border-bottom: none;
  }
  .change-label {
    font-weight: 600;
    font-size: 14px;
  }
  .change-category {
    font-size: 11px;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .change-detail {
    font-size: 14px;
    color: #444;
    margin-top: 4px;
  }
  .change-source {
    font-size: 12px;
    color: #999;
    margin-top: 8px;
  }
  .change-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    font-size: 14px;
    table-layout: fixed;
  }
  .change-table th {
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #666;
    padding: 6px 10px;
    width: 50%;
    border-bottom: 1px solid #e5e5e5;
  }
  .change-table td {
    padding: 8px 10px;
    vertical-align: top;
    word-wrap: break-word;
    border: 1px solid #e5e5e5;
    border-top: none;
  }
  .old-value {
    background: #fef2f2;
    color: #991b1b;
  }
  .new-value {
    background: #f0fdf4;
    color: #166534;
  }
  .added-tag {
    color: #16a34a;
    font-size: 12px;
    font-weight: 500;
  }
  .removed-tag {
    color: #dc2626;
    font-size: 12px;
    font-weight: 500;
  }
  .disclaimer {
    font-size: 12px;
    color: #999;
    border-left: 2px solid #e5e5e5;
    padding-left: 12px;
    margin-bottom: 24px;
  }
  .baseline-notice {
    font-size: 14px;
    color: #1e40af;
    background: #eff6ff;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 24px;
  }
  .no-changes {
    font-size: 14px;
    color: #666;
    padding: 24px 0;
  }
  .note {
    font-size: 13px;
    color: #999;
    padding: 8px 0;
  }
  .footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e5e5e5;
    font-size: 12px;
    color: #999;
  }
</style>
</head>
<body>
"""

    # ── Header
    badge = ""
    if baseline_created:
        badge = '<span class="badge badge-baseline">Baseline</span>'
    elif total_changes > 0:
        badge = f'<span class="badge badge-changes">{total_changes} change{"s" if total_changes != 1 else ""}</span>'
    else:
        badge = '<span class="badge badge-clear">No changes</span>'

    html += f"""\
<div class="header">
  <h1>{escape(domain)}{badge}</h1>
  <div class="meta">{extracted_at} &middot; {pages_scanned} pages scanned</div>
</div>
"""

    # ── Baseline
    if baseline_created:
        html += '<div class="baseline-notice">Initial knowledge baseline established. Future scans will report changes against this snapshot.</div>\n'

    # ── Disclaimer
    if total_changes > 0 and not baseline_created:
        html += '<div class="disclaimer">This report is generated by AI and may contain errors. Please verify reported changes against the actual website.</div>\n'

    # ── Changes
    if total_changes > 0:
        html += '<div class="section">\n<h2>Changes Detected</h2>\n'

        for entry in changed:
            label = escape(entry.get("label", ""))
            cat = escape(entry.get("category", "").capitalize())
            old_val = escape(entry.get("old_value", ""))
            new_val = escape(entry.get("new_value", ""))
            source = _path(entry.get("page", ""))
            html += f"""\
<div class="change-item">
  <div class="change-category">{cat}</div>
  <div class="change-label">{label}</div>
  <table class="change-table">
    <thead>
      <tr><th>Before</th><th>After</th></tr>
    </thead>
    <tbody>
      <tr>
        <td class="old-value">{old_val}</td>
        <td class="new-value">{new_val}</td>
      </tr>
    </tbody>
  </table>
  <div class="change-source">{escape(source)}</div>
</div>
"""

        for entry in added:
            label = escape(entry.get("label", ""))
            cat = escape(entry.get("category", "").capitalize())
            val = escape(entry.get("value", ""))
            source = _path(entry.get("page", ""))
            html += f"""\
<div class="change-item">
  <div class="change-category">{cat} <span class="added-tag">NEW</span></div>
  <div class="change-label">{label}</div>
  <div class="change-detail">{val}</div>
  <div class="change-source">{escape(source)}</div>
</div>
"""

        for entry in removed:
            label = escape(entry.get("label", ""))
            cat = escape(entry.get("category", "").capitalize())
            val = escape(entry.get("value", ""))
            source = _path(entry.get("page", ""))
            html += f"""\
<div class="change-item">
  <div class="change-category">{cat} <span class="removed-tag">REMOVED</span></div>
  <div class="change-label">{label}</div>
  <div class="change-detail" style="color: #991b1b;">{val}</div>
  <div class="change-source">{escape(source)}</div>
</div>
"""

        html += '</div>\n'

    elif not baseline_created:
        html += '<div class="no-changes">No knowledge changes detected.</div>\n'

    # ── Raw fallback
    if raw_fallback_pages:
        html += '<div class="section">\n<h2>Raw Changes (extraction failed)</h2>\n'
        for fb in raw_fallback_pages:
            page = escape(fb.get("page", "Unknown"))
            summary = escape(fb.get("diff_summary", ""))
            html += f'<div class="change-item"><div class="change-label">{page}</div><div class="change-detail">{summary}</div></div>\n'
        html += '</div>\n'

    # ── Extraction notes
    if extraction_notes:
        html += '<div class="section">\n<h2>Notes</h2>\n'
        for note in extraction_notes:
            html += f'<div class="note">{escape(note)}</div>\n'
        html += '</div>\n'

    # ── Footer
    html += '<div class="footer">Monitored by Website Knowledge Tracker</div>\n'
    html += '</body>\n</html>\n'

    return html


def build_knowledge_summary(
    knowledge_snapshot: dict[str, Any],
    diff: dict[str, list[dict]],
    baseline_created: bool,
) -> dict[str, Any]:
    """Build a summary dict from a knowledge snapshot and diff."""
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
