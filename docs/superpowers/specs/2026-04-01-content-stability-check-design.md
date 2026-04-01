# Content Stability Check for Page Fetching

## Problem

The website monitor uses Playwright to crawl pages and extract text for change detection. The current `wait_until="networkidle"` strategy races against client-side JavaScript rendering on SPA sites, and fails entirely on sites with persistent connections (WebSockets, analytics beacons). This causes flapping diffs where the same content appears as "removed" then "added" on consecutive scans.

## Solution

Replace `networkidle` with `domcontentloaded` + a text-level stability polling loop. After the DOM is parsed, poll the page's body text until two consecutive readings match, then proceed with extraction.

## Design

### New function: `wait_for_content_stable(page, timeout_ms=3000, interval_ms=500)`

Located in `monitor.py`, above `extract_page_data()`.

**Algorithm:**

1. Read `document.body?.innerText || ""` via `page.evaluate()`
2. Sleep `interval_ms`
3. Read again
4. If the two readings match, return (content is stable)
5. If not, set the new reading as the baseline and go to step 2
6. If cumulative wait exceeds `timeout_ms`, return (use whatever we have)

**JS evaluation:** `page.evaluate("() => document.body?.innerText || ''")` . Lightweight, no DOM mutation, returns raw text including boilerplate. We only need string equality for stability, not clean text.

**Parameters:** `timeout_ms=3000`, `interval_ms=500`. Not exposed as user config. These are internal implementation details. Max 6 polls per page, worst case 3s added per page.

### Integration

In `crawl()` at line 645:

```python
# Before (current):
response = page.goto(url, wait_until="networkidle", timeout=timeout_ms)

# After:
response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
```

In `extract_page_data()` at line 579, before `extract_primary_text()`:

```python
def extract_page_data(page, page_url: str) -> dict[str, object]:
    wait_for_content_stable(page)  # NEW: wait for JS rendering to settle
    title = ""
    ...
```

### Timeout behavior

If content never stabilizes (live ticker, animation text), we proceed after 3s with whatever text is present. The existing 97% similarity threshold in `compare_snapshots` provides a secondary safety net for minor residual jitter between scans.

### Performance impact

- Server-rendered pages: one 500ms wait, then stable. Adds ~500ms per page.
- SPA pages: typically 2-3 cycles (1-1.5s).
- Worst case (100 pages, all timing out): ~5 min added to scan. Realistic case is much lower since most pages stabilize quickly.

## Files changed

- `src/website_monitor/monitor.py`: Add `wait_for_content_stable()`, revert `crawl()` to `domcontentloaded`, call stability check in `extract_page_data()`
- `tests/unit/test_monitor_core.py`: Add tests for the new function

## Test plan

1. **Immediate stability:** Mock page returns same text twice. Function exits after one interval.
2. **Delayed stability:** Mock page returns different text for 2 cycles, then stabilizes. Function waits and returns.
3. **Timeout:** Mock page never returns the same text twice. Function returns after timeout_ms.
4. **Empty body:** Mock page returns empty string. Function handles gracefully.
5. **Exception handling:** Mock page.evaluate raises. Function proceeds without crashing.
6. **Integration in extract_page_data:** Verify `wait_for_content_stable` is called before text extraction.
