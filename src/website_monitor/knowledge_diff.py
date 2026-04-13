# Knowledge diff: compare two knowledge snapshots and detect changes.
#
#   previous_snapshot ──┐
#                       ├──> compare_knowledge ──> {added, removed, changed, unchanged}
#   current_snapshot  ──┘

from __future__ import annotations

import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def _operational_units(page: dict) -> list[dict]:
    """Return only operational knowledge units from a page."""
    return [u for u in page.get("knowledge_units", []) if u.get("operational", True)]


def _page_fingerprint(page: dict) -> frozenset[tuple]:
    """Build a fingerprint of operational units for redirect detection."""
    units = _operational_units(page)
    return frozenset(
        (u["category"], u["label"], u["value"]) for u in units
    )


def _fuzzy_reconcile(
    added: dict[tuple, str],
    removed: dict[tuple, str],
    threshold: float = 0.75,
) -> tuple[list[dict], dict[tuple, str], dict[tuple, str]]:
    """Reconcile add/remove pairs that are really label renames.

    Before declaring a unit as added and another as removed on the same
    page+category, check if labels are similar (SequenceMatcher ratio ≥
    threshold). If so, treat as a match and compare values.

    Args:
        added: {(page_url, category, label): value} for unmatched added units.
        removed: {(page_url, category, label): value} for unmatched removed units.
        threshold: minimum SequenceMatcher ratio to consider a fuzzy match.

    Returns:
        (matched, remaining_added, remaining_removed) where matched is a list
        of dicts with old_label, new_label, old_value, new_value, page, category.
    """
    if not removed:
        return [], dict(added), {}

    matched: list[dict] = []
    used_added: set[tuple] = set()
    used_removed: set[tuple] = set()

    # Group by (page, category) for efficient matching
    added_by_group: dict[tuple, list[tuple]] = {}
    for key in added:
        group = (key[0], key[1])  # (page_url, category)
        added_by_group.setdefault(group, []).append(key)

    removed_by_group: dict[tuple, list[tuple]] = {}
    for key in removed:
        group = (key[0], key[1])
        removed_by_group.setdefault(group, []).append(key)

    # For each group, find best fuzzy matches
    for group, added_keys in added_by_group.items():
        removed_keys = removed_by_group.get(group, [])
        if not removed_keys:
            continue

        # Build all candidate pairs with scores
        candidates: list[tuple[float, tuple, tuple]] = []
        for a_key in added_keys:
            for r_key in removed_keys:
                score = SequenceMatcher(None, r_key[2], a_key[2]).ratio()
                if score >= threshold:
                    candidates.append((score, a_key, r_key))

        # Greedily pick best matches
        candidates.sort(key=lambda x: x[0], reverse=True)
        for score, a_key, r_key in candidates:
            if a_key in used_added or r_key in used_removed:
                continue
            used_added.add(a_key)
            used_removed.add(r_key)
            matched.append({
                "page": a_key[0],
                "category": a_key[1],
                "old_label": r_key[2],
                "new_label": a_key[2],
                "old_value": removed[r_key],
                "new_value": added[a_key],
            })

    remaining_added = {k: v for k, v in added.items() if k not in used_added}
    remaining_removed = {k: v for k, v in removed.items() if k not in used_removed}

    return matched, remaining_added, remaining_removed


def compare_knowledge(
    previous: dict | None,
    current: dict,
    fuzzy_threshold: float = 0.75,
) -> dict[str, list[dict]]:
    """Compare two knowledge snapshots and return categorized changes.

    Args:
        previous: previous knowledge snapshot (None for baseline).
        current: current knowledge snapshot.
        fuzzy_threshold: SequenceMatcher ratio threshold for fuzzy label matching.

    Returns:
        dict with keys: added, removed, changed, unchanged. Each is a list of
        dicts describing the knowledge units in that category.
    """
    added: list[dict] = []
    removed: list[dict] = []
    changed: list[dict] = []
    unchanged: list[dict] = []

    current_pages = current.get("pages", {})
    previous_pages = previous.get("pages", {}) if previous else {}

    all_urls = set(list(current_pages.keys()) + list(previous_pages.keys()))

    for url in sorted(all_urls):
        prev_page = previous_pages.get(url, {})
        curr_page = current_pages.get(url, {})

        prev_units = _operational_units(prev_page) if prev_page else []
        curr_units = _operational_units(curr_page) if curr_page else []

        # Index by (category, label)
        prev_by_key: dict[tuple, str] = {}
        for u in prev_units:
            prev_by_key[(u["category"], u["label"])] = u["value"]

        curr_by_key: dict[tuple, str] = {}
        for u in curr_units:
            curr_by_key[(u["category"], u["label"])] = u["value"]

        prev_keys = set(prev_by_key.keys())
        curr_keys = set(curr_by_key.keys())

        # Exact matches
        for key in sorted(prev_keys & curr_keys):
            cat, label = key
            if prev_by_key[key] == curr_by_key[key]:
                unchanged.append({"page": url, "category": cat, "label": label, "value": curr_by_key[key]})
            else:
                changed.append({
                    "page": url, "category": cat, "label": label,
                    "old_value": prev_by_key[key], "new_value": curr_by_key[key],
                })

        # Preliminary added/removed (before fuzzy reconciliation)
        raw_added = {
            (url, k[0], k[1]): curr_by_key[k]
            for k in sorted(curr_keys - prev_keys)
        }
        raw_removed = {
            (url, k[0], k[1]): prev_by_key[k]
            for k in sorted(prev_keys - curr_keys)
        }

        # Fuzzy reconcile
        fuzzy_matched, remaining_added, remaining_removed = _fuzzy_reconcile(
            raw_added, raw_removed, threshold=fuzzy_threshold,
        )

        for m in fuzzy_matched:
            if m["old_value"] == m["new_value"]:
                unchanged.append({
                    "page": m["page"], "category": m["category"],
                    "label": m["new_label"], "value": m["new_value"],
                })
            else:
                changed.append({
                    "page": m["page"], "category": m["category"],
                    "label": m["new_label"],
                    "old_value": m["old_value"], "new_value": m["new_value"],
                })

        for key, value in remaining_added.items():
            added.append({"page": key[0], "category": key[1], "label": key[2], "value": value})

        for key, value in remaining_removed.items():
            removed.append({"page": key[0], "category": key[1], "label": key[2], "value": value})

    return {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged}


def reconcile_knowledge_redirects(
    added_urls: list[str],
    removed_urls: list[str],
    previous_pages: dict,
    current_pages: dict,
) -> tuple[list[dict], list[str], list[str]]:
    """Detect URL redirects where knowledge is identical but URL changed.

    Args:
        added_urls: page URLs present in current but not previous.
        removed_urls: page URLs present in previous but not current.
        previous_pages: pages dict from previous snapshot.
        current_pages: pages dict from current snapshot.

    Returns:
        (redirected, remaining_added, remaining_removed) where redirected is a
        list of {"from_url": ..., "to_url": ...} dicts.
    """
    # Fingerprint all pages
    removed_fps: dict[frozenset, str] = {}
    for url in removed_urls:
        page = previous_pages.get(url, {})
        fp = _page_fingerprint(page)
        if fp:  # skip empty fingerprints
            removed_fps[fp] = url

    redirected: list[dict] = []
    matched_added: set[str] = set()
    matched_removed: set[str] = set()

    for url in added_urls:
        page = current_pages.get(url, {})
        fp = _page_fingerprint(page)
        if not fp:
            continue
        if fp in removed_fps:
            old_url = removed_fps[fp]
            if old_url not in matched_removed:
                redirected.append({"from_url": old_url, "to_url": url})
                matched_added.add(url)
                matched_removed.add(old_url)

    remaining_added = [u for u in added_urls if u not in matched_added]
    remaining_removed = [u for u in removed_urls if u not in matched_removed]

    return redirected, remaining_added, remaining_removed
