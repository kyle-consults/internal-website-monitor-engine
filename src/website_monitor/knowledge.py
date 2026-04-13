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

DEFAULT_MODEL = "gemini-2.0-flash-lite"

# ── Structured output schema for Gemini ──────────────────────────────────────

_KNOWLEDGE_UNIT_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "fact": types.Schema(
            type="STRING",
            description="A single, self-contained piece of knowledge from the page.",
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
    required=["fact", "category", "operational"],
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
fact, detail, or piece of information.

Classify each unit with a category and mark whether it is *operational* \
(something that changes and matters to users, like hours, pricing, contact info) \
or non-operational (static background, company history, branding).

Only extract facts that are explicitly stated in the page text. Do not infer \
or fabricate information."""


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

    Returns a list of dicts with keys ``fact``, ``category``, ``operational``.
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
        units: list[dict[str, Any]] = []
        for ku in parsed.knowledge_units:
            units.append(
                {
                    "fact": ku.fact,
                    "category": ku.category,
                    "operational": ku.operational,
                }
            )
        return units
    except Exception:
        logger.exception("Gemini extraction failed")
        return []
