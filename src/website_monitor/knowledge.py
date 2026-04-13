# Knowledge extraction pipeline:
#
#   crawl_result ──> hash_gate ──> extract (parallel) ──> knowledge_snapshot
#                       │ (cache hit)
#                previous_knowledge

from __future__ import annotations

import logging
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
    """Check if two unit lists have identical operational values.

    Returns True only if every operational unit's (category, label, value)
    triple is the same in both lists. Any difference means the page genuinely
    changed and the new extraction should be kept.
    """
    def _op_set(units: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
        return {
            (u.get("category", ""), u.get("label", ""), u.get("value", ""))
            for u in units
            if u.get("operational", True)
        }

    return _op_set(prev_units) == _op_set(new_units)


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
You are a change verification assistant. You are given a list of detected \
knowledge changes from a website monitoring system. Some of these changes \
are REAL (the website actually updated information) and some are NOISE \
(the extraction interpreted the same content differently between runs).

For each change, classify it as "real" or "noise".

A change is NOISE if:
- The old and new values say the same thing in different words or formatting
- The change is just capitalization, punctuation, or whitespace differences
- A list was reordered but contains the same items
- Content moved from one label to another but the information is the same
- Boilerplate, navigation, or UI text was extracted inconsistently

A change is REAL if:
- Actual factual information changed (hours, phone numbers, addresses, prices)
- New services, policies, or providers were genuinely added or removed
- Contact information was updated

Be conservative: when in doubt, classify as noise. Only flag changes where \
the actual meaning or facts changed for a real customer.

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
