# Knowledge extraction pipeline:
#
#   crawl_result ──> hash_gate ──> extract (parallel) ──> knowledge_snapshot
#                       │ (cache hit)
#                previous_knowledge

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-lite"

# ── Structured output schema for Gemini ──────────────────────────────────────

_KNOWLEDGE_UNIT_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "label": types.Schema(
            type="STRING",
            description="Concise normalized name for the knowledge unit, e.g. 'Weekday Hours'.",
        ),
        "value": types.Schema(
            type="STRING",
            description="The exact value as stated on the page, e.g. 'Monday-Friday 8:00 AM - 8:00 PM'.",
        ),
        "category": types.Schema(
            type="STRING",
            description=(
                "Category of the knowledge unit, e.g. hours, pricing, "
                "contact, location, policy, background, product, service."
            ),
        ),
        "operational": types.Schema(
            type="BOOLEAN",
            description=(
                "True if this fact is operational — something that could change "
                "and a customer/user would need to know about (hours, pricing, "
                "contact info, availability). False for static background info."
            ),
        ),
    },
    required=["label", "value", "category", "operational"],
)

_RESPONSE_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "knowledge_units": types.Schema(
            type="ARRAY",
            items=_KNOWLEDGE_UNIT_SCHEMA,
            description="List of knowledge units extracted from the page.",
        ),
    },
    required=["knowledge_units"],
)

_SYSTEM_PROMPT = """\
You are a knowledge extraction assistant. Given the text content of a web page, \
extract discrete, self-contained knowledge units. Each unit should capture a single \
piece of information as a label+value pair.

Rules:
- Use concise, stable labels. Prefer generic labels like 'Weekday Hours', 'Phone', \
'Address', 'Accepted Insurance' that will be the same every time you see this page.
- Preserve the EXACT values as stated on the page. Do not paraphrase, reformat, \
or change capitalization of numbers, times, addresses, or names.
- Mark operational=true for information that could change and a customer needs to \
know: hours, pricing, contact info, insurance accepted, services offered, policies, \
locations, availability.
- Mark operational=false for static background: company history, branding, taglines, \
general descriptions of what urgent care is.

Do NOT extract:
- Navigation menus, breadcrumbs, or page chrome
- Search widgets, location finders, or interactive UI elements
- Generic marketing copy or boilerplate disclaimers
- Lists of all locations/states (extract only the specific location's info)
- Call-to-action buttons or link text
- Cookie/consent banner text

Only extract information explicitly stated in the page text. Do not infer or fabricate."""


# ── Public API ───────────────────────────────────────────────────────────────


def build_gemini_client(api_key: str | None) -> genai.Client | None:
    """Return a genai Client for the given key, or None if the key is missing."""
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def extract_page_knowledge(
    page_text: str,
    client: genai.Client,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Extract knowledge units from *page_text* via Gemini structured output.

    Returns a list of dicts with keys ``label``, ``value``, ``category``, ``operational``.
    Returns ``[]`` for empty/whitespace text or on any API error.
    """
    if not page_text or not page_text.strip():
        return []

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"<PAGE_TEXT>\n{page_text}\n</PAGE_TEXT>"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
            ),
        )
        parsed = response.parsed
        # parsed can be a dict (Gemini 2.5+) or a Pydantic model (older SDKs)
        if isinstance(parsed, dict):
            raw_units = parsed.get("knowledge_units", [])
        else:
            raw_units = parsed.knowledge_units
        units: list[dict[str, Any]] = []
        for ku in raw_units:
            if isinstance(ku, dict):
                units.append({
                    "label": ku.get("label", ""),
                    "value": ku.get("value", ""),
                    "category": ku.get("category", ""),
                    "operational": ku.get("operational", True),
                })
            else:
                units.append({
                    "label": ku.label,
                    "value": ku.value,
                    "category": ku.category,
                    "operational": ku.operational,
                })
        return units
    except Exception:
        logger.exception("Gemini extraction failed")
        return []


def _operational_values_match(
    prev_units: list[dict[str, Any]],
    new_units: list[dict[str, Any]],
) -> bool:
    """Check if two unit lists have effectively identical operational content.

    Uses a two-pass approach:
    1. Exact match on (category, label, value) triples — if 100%, return True
    2. For remaining unmatched units, check if values are highly similar
       (>0.85 SequenceMatcher ratio). If all unmatched units have a similar
       counterpart, the page is treated as stable.

    This handles LLM nondeterminism where Gemini extracts slightly different
    wording for the same fact across runs (e.g., "extended hours" vs
    "extended hours on evenings and weekends").
    """
    def _op_list(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [u for u in units if u.get("operational", True)]

    prev_ops = _op_list(prev_units)
    new_ops = _op_list(new_units)

    if not prev_ops and not new_ops:
        return True

    # Pass 1: exact match
    prev_set = {
        (u.get("category", ""), u.get("label", ""), u.get("value", ""))
        for u in prev_ops
    }
    new_set = {
        (u.get("category", ""), u.get("label", ""), u.get("value", ""))
        for u in new_ops
    }
    if prev_set == new_set:
        return True

    # Pass 2: fuzzy value match for remaining units
    # Index by (category, label) for easy lookup
    prev_by_key: dict[tuple[str, str], str] = {}
    for u in prev_ops:
        prev_by_key[(u.get("category", ""), u.get("label", ""))] = u.get("value", "")

    new_by_key: dict[tuple[str, str], str] = {}
    for u in new_ops:
        new_by_key[(u.get("category", ""), u.get("label", ""))] = u.get("value", "")

    all_keys = set(prev_by_key.keys()) | set(new_by_key.keys())
    for key in all_keys:
        prev_val = prev_by_key.get(key)
        new_val = new_by_key.get(key)
        if prev_val is None or new_val is None:
            # Unit appeared or disappeared — check if there's a similar unit
            # in the other set with a different label but same category
            missing_val = prev_val if new_val is None else new_val
            other_vals = new_by_key if new_val is None else prev_by_key
            cat = key[0]
            best_sim = 0.0
            for other_key, other_val in other_vals.items():
                if other_key[0] == cat and other_key not in (set(prev_by_key) & set(new_by_key)):
                    sim = SequenceMatcher(a=missing_val, b=other_val).ratio()
                    best_sim = max(best_sim, sim)
            if best_sim < 0.7:
                return False
        elif prev_val != new_val:
            sim = SequenceMatcher(a=prev_val, b=new_val).ratio()
            if sim < 0.7:
                return False

    return True


def extract_all_pages(
    crawl_result: dict[str, Any],
    client: genai.Client,
    model: str = DEFAULT_MODEL,
    previous_snapshot: dict[str, Any] | None = None,
    previous_knowledge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract knowledge from all pages in a crawl result, with hash gating.

    If a page's hash matches the previous snapshot, its prior knowledge is
    reused instead of calling Gemini again.  Changed or new pages are
    extracted in parallel via a thread pool.

    Returns a knowledge snapshot dict::

        {
            "schema_version": 1,
            "homepage_url": ...,
            "extracted_at": ...,
            "model": ...,
            "pages": { url: {"url": ..., "knowledge_units": [...]} },
        }
    """
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone

    prev_pages = (previous_snapshot or {}).get("pages", {})
    prev_knowledge_pages = (previous_knowledge or {}).get("pages", {})
    current_pages = crawl_result.get("pages", {})

    # Partition into cached vs needs-extraction
    cached: dict[str, list[dict[str, Any]]] = {}
    to_extract: dict[str, str] = {}  # url → page_text

    for url, page_data in current_pages.items():
        current_hash = page_data.get("hash", "")
        previous_hash = prev_pages.get(url, {}).get("hash", "")

        if (
            current_hash
            and current_hash == previous_hash
            and url in prev_knowledge_pages
        ):
            cached[url] = prev_knowledge_pages[url].get("knowledge_units", [])
        else:
            to_extract[url] = str(page_data.get("text", ""))

    # Parallel extraction for changed / new pages
    extracted: dict[str, list[dict[str, Any]]] = {}
    if to_extract:
        def _do_extract(item: tuple[str, str]) -> tuple[str, list[dict[str, Any]]]:
            url, text = item
            return url, extract_page_knowledge(text, client, model)

        with ThreadPoolExecutor(max_workers=5) as pool:
            for url, units in pool.map(_do_extract, to_extract.items()):
                extracted[url] = units

    # Extraction stability check: if a page was re-extracted but all
    # operational values are identical to the previous extraction, keep the
    # previous extraction to avoid LLM nondeterminism noise.
    stabilized: list[str] = []
    for url, new_units in extracted.items():
        prev_units = prev_knowledge_pages.get(url, {}).get("knowledge_units", [])
        if prev_units and _operational_values_match(prev_units, new_units):
            cached[url] = prev_units
            stabilized.append(url)
    for url in stabilized:
        del extracted[url]

    # Assemble result
    pages_out: dict[str, dict[str, Any]] = {}
    for url in current_pages:
        if url in cached:
            pages_out[url] = {"url": url, "knowledge_units": cached[url]}
        else:
            pages_out[url] = {"url": url, "knowledge_units": extracted.get(url, [])}

    return {
        "schema_version": 1,
        "homepage_url": crawl_result.get("homepage_url", ""),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "pages": pages_out,
    }


# ── Change verification ─────────────────────────────────────────────────────

_VERIFY_PROMPT = """\
You are a strict change verification assistant for a website monitoring system. \
Many websites have dynamic pages (accordions, tabs, lazy-loaded content) that \
render differently on each visit. This causes our AI extractor to produce \
slightly different outputs each run, even when the website content has NOT \
actually changed. Your job is to aggressively filter this noise.

For each change, classify it as "real" or "noise".

A change is NOISE (most changes are noise) if:
- The old and new values convey the same information in different words
- The change is just capitalization, punctuation, whitespace, or formatting
- A list was extracted in a different order or with different item boundaries
- Content was extracted with a different label but the factual content is the same
- Generic service descriptions, marketing copy, or boilerplate changed wording
- A section appeared or disappeared that contains general information already \
covered elsewhere on the site (e.g., "services offered" list, FAQ answers)
- UI/navigation text like "Search by State", "Find a location", "About Us"

A change is REAL only if ALL of these are true:
- A specific, concrete fact changed (phone number, street address, hours of \
operation, insurance provider name, pricing, policy with specific terms)
- The change represents new information a customer would need to act on
- The old and new values are factually different, not just rephrased

Default to NOISE. Only classify as "real" when you are confident that a \
customer-facing fact genuinely changed. When in doubt, it is noise.

Respond with a JSON array of objects, one per change, each with:
- "index": the change number (starting from 0)
- "verdict": "real" or "noise"
- "reason": one sentence explaining why"""

_VERIFY_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "verdicts": types.Schema(
            type="ARRAY",
            items=types.Schema(
                type="OBJECT",
                properties={
                    "index": types.Schema(type="INTEGER"),
                    "verdict": types.Schema(type="STRING"),
                    "reason": types.Schema(type="STRING"),
                },
                required=["index", "verdict"],
            ),
        ),
    },
    required=["verdicts"],
)


def verify_changes(
    diff: dict[str, list[dict[str, Any]]],
    client: genai.Client,
    model: str = DEFAULT_MODEL,
) -> dict[str, list[dict[str, Any]]]:
    """Use an LLM to filter noise from real changes in a knowledge diff.

    Takes the raw diff from compare_knowledge and returns a filtered diff
    with only real changes. Noise is moved to a 'noise' key.
    """
    candidates = []
    for entry in diff.get("changed", []):
        candidates.append({
            "type": "changed",
            "label": entry.get("label", ""),
            "old_value": entry.get("old_value", ""),
            "new_value": entry.get("new_value", ""),
            "page": entry.get("page", ""),
        })
    for entry in diff.get("added", []):
        candidates.append({
            "type": "added",
            "label": entry.get("label", ""),
            "value": entry.get("value", ""),
            "page": entry.get("page", ""),
        })
    for entry in diff.get("removed", []):
        candidates.append({
            "type": "removed",
            "label": entry.get("label", ""),
            "value": entry.get("value", ""),
            "page": entry.get("page", ""),
        })

    if not candidates:
        return diff

    import json
    changes_text = json.dumps(candidates, indent=2)
    prompt = f"{_VERIFY_PROMPT}\n\nChanges to verify:\n{changes_text}"

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_VERIFY_SCHEMA,
            ),
        )
        parsed = response.parsed
        if isinstance(parsed, dict):
            verdicts = parsed.get("verdicts", [])
        else:
            verdicts = parsed.verdicts
    except Exception:
        logger.exception("Change verification failed, keeping all changes")
        return diff

    # Build a set of noise indices
    noise_indices: set[int] = set()
    for v in verdicts:
        if isinstance(v, dict):
            if v.get("verdict") == "noise":
                noise_indices.add(v.get("index", -1))
        else:
            if v.verdict == "noise":
                noise_indices.add(v.index)

    # Split diff into real and noise
    changed_count = len(diff.get("changed", []))
    added_count = len(diff.get("added", []))

    real_changed = []
    real_added = []
    real_removed = []
    noise = []

    for i, entry in enumerate(diff.get("changed", [])):
        if i in noise_indices:
            noise.append({**entry, "_noise_type": "changed"})
        else:
            real_changed.append(entry)

    for i, entry in enumerate(diff.get("added", [])):
        idx = changed_count + i
        if idx in noise_indices:
            noise.append({**entry, "_noise_type": "added"})
        else:
            real_added.append(entry)

    for i, entry in enumerate(diff.get("removed", [])):
        idx = changed_count + added_count + i
        if idx in noise_indices:
            noise.append({**entry, "_noise_type": "removed"})
        else:
            real_removed.append(entry)

    return {
        "changed": real_changed,
        "added": real_added,
        "removed": real_removed,
        "unchanged": diff.get("unchanged", []),
        "noise": noise,
    }


# ── Multi-capture quorum ────────────────────────────────────────────────────


def _fact_key(entry: dict[str, Any], change_type: str) -> tuple[str, str, str, str]:
    """Unique key identifying a candidate change across captures."""
    if change_type == "changed":
        return (change_type, entry.get("page", ""), entry.get("category", ""),
                entry.get("label", ""))
    return (change_type, entry.get("page", ""), entry.get("category", ""),
            entry.get("label", ""))


def quorum_verify_changes(
    diff: dict[str, list[dict[str, Any]]],
    current_knowledge: dict[str, Any],
    previous_knowledge: dict[str, Any] | None,
    recrawl_fn,
    client: genai.Client,
    model: str,
    cfg: dict[str, Any],
    captures: int = 2,
    quorum: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Re-crawl pages with candidate changes and only keep changes that
    appear in at least `quorum` out of (captures + 1) total captures.

    The first capture is what's already in current_knowledge. We do
    `captures` additional re-crawls (default 2 for 3 total), re-extract,
    and check whether each candidate change survives.

    Pages with inconsistent extraction across captures are marked unstable
    and all their changes are dropped.
    """
    all_changes = (
        [(e, "changed") for e in diff.get("changed", [])]
        + [(e, "added") for e in diff.get("added", [])]
        + [(e, "removed") for e in diff.get("removed", [])]
    )
    if not all_changes:
        return diff

    # Pages that have at least one candidate change
    affected_urls = sorted({e.get("page", "") for e, _ in all_changes if e.get("page")})
    if not affected_urls:
        return diff

    prev_pages = (previous_knowledge or {}).get("pages", {})

    # Build per-page vote counts: {url: {fact_key: vote_count}}
    # Initialize with votes from the primary extraction (current_knowledge)
    page_change_votes: dict[str, dict[tuple, int]] = {url: {} for url in affected_urls}
    page_change_entries: dict[tuple, dict] = {}
    page_change_types: dict[tuple, str] = {}

    for entry, change_type in all_changes:
        url = entry.get("page", "")
        key = _fact_key(entry, change_type)
        page_change_votes[url][key] = 1  # primary extraction = 1 vote
        page_change_entries[key] = entry
        page_change_types[key] = change_type

    # Do N additional captures
    for _capture_idx in range(captures):
        try:
            recaptured = recrawl_fn(affected_urls, cfg)
        except Exception:
            logger.exception("Recrawl failed during quorum verification")
            continue

        for url in affected_urls:
            page_data = recaptured.get(url)
            if not page_data:
                # Couldn't re-crawl this URL; don't count a vote
                continue
            page_text = str(page_data.get("text", ""))
            new_units = extract_page_knowledge(page_text, client, model)
            if not new_units:
                # Extraction failed on this capture; don't use it
                continue

            # Build lookup of this capture's operational units
            new_by_key: dict[tuple[str, str], str] = {}
            new_values: list[str] = []
            for u in new_units:
                if u.get("operational", True):
                    new_by_key[(u.get("category", ""), u.get("label", ""))] = u.get("value", "")
                    new_values.append(u.get("value", ""))

            # For each candidate change on this page, check if the new capture
            # also exhibits it
            prev_units = prev_pages.get(url, {}).get("knowledge_units", [])
            prev_by_key: dict[tuple[str, str], str] = {}
            for u in prev_units:
                if u.get("operational", True):
                    prev_by_key[(u.get("category", ""), u.get("label", ""))] = u.get("value", "")

            for key in list(page_change_votes[url].keys()):
                change_type = page_change_types[key]
                entry = page_change_entries[key]
                cat = entry.get("category", "")
                label = entry.get("label", "")
                cap_key = (cat, label)

                if change_type == "changed":
                    new_val = entry.get("new_value", "")
                    if new_by_key.get(cap_key) == new_val:
                        page_change_votes[url][key] += 1
                elif change_type == "added":
                    val = entry.get("value", "")
                    # Added if it's in this capture AND not in previous
                    if new_by_key.get(cap_key) == val and cap_key not in prev_by_key:
                        page_change_votes[url][key] += 1
                elif change_type == "removed":
                    old_val = entry.get("value", "")
                    # A removal is only genuine if the value is absent from this
                    # capture under ANY label. Label/category drift (same fact,
                    # relabeled) would otherwise falsely confirm the removal
                    # because the old (cat, label) slot is empty.
                    if cap_key in new_by_key:
                        continue
                    value_still_present = any(
                        v == old_val
                        or SequenceMatcher(None, str(v), str(old_val)).ratio() >= 0.85
                        for v in new_values
                    )
                    if not value_still_present:
                        page_change_votes[url][key] += 1

    # A page is unstable if its captures produced wildly different extractions.
    # Simple heuristic: if ANY candidate change on the page only has 1 vote
    # (only showed up in the primary), AND we got at least one successful
    # recapture, flag the page as unstable.
    #
    # Actually, simpler: just apply quorum threshold per-change. The page
    # itself isn't unstable — the individual change is.

    real_changed: list[dict] = []
    real_added: list[dict] = []
    real_removed: list[dict] = []
    noise: list[dict] = diff.get("noise", []) or []

    for url, votes in page_change_votes.items():
        for key, vote_count in votes.items():
            entry = page_change_entries[key]
            change_type = page_change_types[key]
            if vote_count >= quorum:
                if change_type == "changed":
                    real_changed.append(entry)
                elif change_type == "added":
                    real_added.append(entry)
                elif change_type == "removed":
                    real_removed.append(entry)
            else:
                # Failed quorum — treat as noise
                noise.append({**entry, "_noise_type": change_type, "_noise_reason": "quorum_failed"})

    return {
        "changed": real_changed,
        "added": real_added,
        "removed": real_removed,
        "unchanged": diff.get("unchanged", []),
        "noise": noise,
    }
