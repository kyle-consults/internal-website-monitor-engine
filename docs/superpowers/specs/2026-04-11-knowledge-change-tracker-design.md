# Knowledge Change Tracker - Design Spec

**Date:** 2026-04-11
**Status:** Approved

## Problem

The current website monitor engine detects that pages changed via text hashing and diffs, but it doesn't understand *what* changed. A client (e.g., an urgent care site) receives reports saying "page /hours changed" with raw text diffs. What they actually need is: "Your weekday hours changed from 8am-8pm to 8am-9pm."

## Solution

Add an LLM-powered knowledge extraction layer that transforms raw page text into structured knowledge units, compares knowledge across runs, and produces plain-English change summaries.

## Consumer

Human readers (client staff) receive daily email reports describing knowledge changes in plain language. Structured knowledge snapshots are persisted in Git for potential downstream system consumption later.

## Pipeline

```
Crawl (existing, Playwright)
  -> Knowledge Extraction (NEW, Gemini Flash Lite)
  -> Knowledge Comparison (NEW, structural diff)
  -> Change Summarization (NEW, Gemini Flash Lite)
  -> Report Generation (modified)
  -> Persistence (modified, knowledge snapshots added)
  -> Notification (existing, Resend)
```

### What stays the same

- Playwright crawl with content stability, boilerplate removal, BFS link following
- Git-backed storage in client repos
- Reusable GitHub Actions workflow pattern
- Email notification via Resend
- Archive retention and atomic writes

### What changes

- After crawling, each page's text goes to Gemini for knowledge extraction
- Snapshots store extracted knowledge units alongside raw page text
- Comparison operates on knowledge units, not text hashes
- Report is rendered from a structured change summary, not raw text diffs
- New dependency: `google-genai`

### What may be removed

- The 97% similarity threshold becomes less important (knowledge comparison is semantic)
- The re-verification/flap detection phase may become unnecessary since LLM extraction naturally absorbs minor text variations. Evaluate during implementation.

## Knowledge Extraction

Each crawled page's text is sent to Gemini Flash Lite to classify the page and extract knowledge.

### Page Classification

- **operational** - hours, policies, insurance, locations, contact info, services, pricing, FAQs about operations
- **non-operational** - blog posts, news, staff bios, testimonials, marketing copy

Non-operational pages are stored with their classification but no extracted knowledge. They are skipped in comparison.

### Knowledge Unit Structure

```json
{
  "page_url": "/hours",
  "classification": "operational",
  "knowledge_units": [
    {
      "category": "hours",
      "label": "Weekday Hours",
      "value": "Monday-Friday 8:00 AM - 8:00 PM"
    },
    {
      "category": "hours",
      "label": "Weekend Hours",
      "value": "Saturday-Sunday 9:00 AM - 5:00 PM"
    },
    {
      "category": "policy",
      "label": "Holiday Hours - Christmas",
      "value": "Closed December 25"
    }
  ]
}
```

### Extraction Rules

- Categories are not predefined - the LLM assigns them naturally (hours, policy, insurance, contact, services, pricing, etc.)
- Labels should be concise and normalized for consistency across runs
- Values must preserve exact wording from the page (no paraphrasing numbers, times, names)
- Non-operational pages return empty knowledge_units

### Model

Gemini 2.0 Flash Lite. Configurable per client via `gemini_model` in `defaults.json`. Can bump to Flash or Pro if extraction quality is insufficient.

## Knowledge Comparison

Compares current knowledge snapshot against the previous one using a structural diff on knowledge units.

### Matching

Knowledge units are matched by composite key: `(page_url, category, label)`.

### Change Types

- **added** - unit exists in current but not previous (new page or new info)
- **removed** - unit exists in previous but not current (page removed or info deleted)
- **changed** - same key, different value (the core case)
- **unchanged** - same key, same value

### Handling LLM Inconsistency

The main risk is Gemini labeling the same knowledge differently across runs (e.g., "Weekday Hours" vs "Monday-Friday Hours"), causing false adds/removes.

Mitigations:

1. **Fuzzy label matching** - before declaring an add+remove pair, check if any removed unit on the same page has a similar label (string similarity) and same category. If so, treat as a match and compare values.
2. **Prompt engineering** - the extraction prompt emphasizes consistent, normalized labeling with examples.
3. **Page-level fallback** - if a page has many apparent adds/removes but values are mostly the same, flag as "extraction inconsistency" rather than reporting false changes.

### Page-Level Events

- Page classification changing (operational <-> non-operational) is reported as a notable event
- New operational page: all units reported as "added"
- Removed page: all units reported as "removed"

### No Hash Gating

Every operational page is extracted every run. Daily frequency + Flash Lite cost makes this viable and avoids complexity of conditional extraction.

## Change Summarization & Report

When changes are detected, the structured diff is sent to Gemini Flash Lite to produce a human-readable change summary.

### Example

Input diff:
```json
{
  "changed": [
    {"page": "/hours", "category": "hours", "label": "Weekday Hours",
     "old_value": "Mon-Fri 8:00 AM - 8:00 PM",
     "new_value": "Mon-Fri 8:00 AM - 9:00 PM"}
  ],
  "added": [
    {"page": "/insurance", "category": "insurance", "label": "Accepted Provider",
     "value": "Blue Cross Blue Shield"}
  ]
}
```

Output report:
```markdown
## Changes Detected - April 11, 2026

### Hours Updated
- **Weekday Hours** changed: was "Mon-Fri 8:00 AM - 8:00 PM",
  now "Mon-Fri 8:00 AM - 9:00 PM" (source: /hours)

### Insurance Updated
- **New accepted provider added:** Blue Cross Blue Shield (source: /insurance)
```

### Report Structure

1. Header - site, scan time, change count
2. Change summary - LLM-generated plain-English section
3. Extraction notes - pages flagged as inconsistent, newly classified, or failed to extract
4. Pages scanned - count of operational vs non-operational pages

### Email Behavior

- Only send if changes detected (default mode), configurable to "always" or "never"
- Subject: "MyUrgentCare.com - 3 knowledge changes detected"
- Body: rendered report
- No-change runs: no email sent, knowledge snapshot still persisted

## Storage & Persistence

### Knowledge Snapshot Format

```json
{
  "homepage_url": "https://myurgentcare.com",
  "extracted_at": "2026-04-11T08:00:00Z",
  "model": "gemini-2.0-flash-lite",
  "pages": {
    "/hours": {
      "url": "https://myurgentcare.com/hours",
      "classification": "operational",
      "knowledge_units": [
        {"category": "hours", "label": "Weekday Hours", "value": "Mon-Fri 8:00 AM - 9:00 PM"},
        {"category": "hours", "label": "Weekend Hours", "value": "Sat-Sun 9:00 AM - 5:00 PM"}
      ]
    },
    "/blog/spring-tips": {
      "url": "https://myurgentcare.com/blog/spring-tips",
      "classification": "non-operational",
      "knowledge_units": []
    }
  }
}
```

### File Structure

```
{client_repo}/
├── config/
│   └── defaults.json
├── snapshots/
│   ├── latest-snapshot.json          (raw crawl, kept for debugging)
│   ├── latest-knowledge.json         (extracted knowledge)
│   ├── knowledge-{timestamp}.json    (archived)
│   └── snapshot-{timestamp}.json     (raw crawl archive)
├── reports/
│   ├── latest-report.md
│   ├── latest-summary.json
│   ├── report-{timestamp}.md
│   └── summary-{timestamp}.json
```

- Raw snapshots kept alongside knowledge snapshots for debugging extraction quality
- Same archive retention policy (default 12) applies to all file types
- Same atomic write pattern via temp files
- Summary JSON extended with knowledge-specific counts (operational pages, knowledge units tracked, changes by category)

### Webhook (Optional)

If `webhook_url` is configured in `defaults.json`, POST a JSON payload when changes are detected:

```json
{
  "site": "https://myurgentcare.com",
  "scanned_at": "2026-04-11T08:00:00Z",
  "changes": [
    {"type": "changed", "page": "/hours", "category": "hours",
     "label": "Weekday Hours", "old_value": "...", "new_value": "..."}
  ]
}
```

No webhook configured = no POST.

## Configuration

### New Config Fields

```json
{
  "gemini_model": "gemini-2.0-flash-lite",
  "webhook_url": null
}
```

Added to `defaults.json`. Only two new fields. `gemini_model` allows per-client override. `webhook_url` is null by default (disabled).

### New Secrets

- `GEMINI_API_KEY` (required) - Google AI API key, passed as a workflow secret

### New Dependency

- `google-genai` - Google's Python SDK for Gemini

### GitHub Actions Usage

```yaml
jobs:
  scan:
    uses: kyle-consults/internal-website-monitor-engine/.github/workflows/reusable-monitor.yml@main
    with:
      homepage_url: https://myurgentcare.com
      alert_email_to: alerts@client.com
      alert_email_from: Monitor <noreply@client.com>
    secrets:
      RESEND_API_KEY: ${{ secrets.RESEND_API_KEY }}
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

Same pattern as today, just with the additional secret.

## Error Handling & Graceful Degradation

- If Gemini fails for a single page: that page falls back to raw text hash comparison and text diff in the report, flagged as "extraction failed - showing raw diff"
- If Gemini is completely unreachable (bad key, quota exhausted): the entire run falls back to the current raw-diff pipeline. Report header notes "Knowledge extraction unavailable - raw diff mode"
- The system never produces no report due to LLM failure

## Testing Strategy

### Unit Tests

- **Knowledge extraction** - mock Gemini responses, verify parsing of knowledge units, page classification, handling of malformed LLM responses
- **Knowledge comparison** - all four change types, fuzzy label matching, page classification changes, page-level add/remove
- **Change summarization** - mock Gemini responses, verify report rendering from structured diffs
- **Graceful degradation** - verify fallback to raw diff when extraction fails per page and when API is unreachable

### Integration Tests

- Full pipeline: crawl (mocked) -> extract (mocked Gemini) -> compare -> summarize (mocked Gemini) -> report -> persist
- Baseline establishment (first run, no previous knowledge snapshot)
- Change detection across runs
- Webhook delivery (mocked HTTP endpoint)
- Archive retention for knowledge files

### LLM Quality Evaluation (Manual)

- Run extraction against real pages and review knowledge unit quality
- Verify label consistency across multiple runs of the same page
- Not CI tests - evaluation runs during development and periodically

### Existing Tests

- Crawl, notification, and workflow contract tests remain as-is
- Some comparison tests updated since comparison now operates on knowledge units
