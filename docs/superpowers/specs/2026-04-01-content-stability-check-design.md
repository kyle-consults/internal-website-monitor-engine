# Content Stability Check for Page Fetching

## Problem

The website monitor uses Playwright to crawl pages and extract text for change detection. The current `wait_until="networkidle"` strategy races against client-side JavaScript rendering on SPA sites, and fails entirely on sites with persistent connections (WebSockets, analytics beacons). This causes flapping diffs where the same content appears as "removed" then "added" on consecutive scans.

## Solution

Replace `networkidle` with `domcontentloaded` + a text-level stability polling loop. After the DOM is parsed, poll the page's body text until two consecutive readings match, then proceed with extraction.

## Design

### New function: `wait_for_content_stable(page, timeout_ms=3000, interval_ms=500)`

Located in `monitor.py`, above `extract_page_data()`.

**Algorithm:**

1. Record wall-clock start time
2. Read `document.body?.innerText || ""` via `page.evaluate()`
3. Sleep `interval_ms`
4. Read again
5. If the two readings match, return (content is stable)
6. If not, set the new reading as the baseline and go to step 3
7. If wall-clock elapsed time from step 1 exceeds `timeout_ms`, return (use whatever we have)

**JS evaluation:** `page.evaluate("() => document.body?.innerText || ''")`. Lightweight, no DOM mutation, returns raw text including boilerplate. We only need string equality for stability, not clean text.

**Parameters:** `timeout_ms=3000`, `interval_ms=500`. Not exposed as user config. These are internal implementation details. Max 6 polls per page, worst case 3s added per page.

**Timeout measurement:** Wall clock from function entry, including `page.evaluate()` overhead. This means on slow CI runners where `page.evaluate()` takes 100-200ms, the effective number of polls may be fewer than 6, but the total wall time stays bounded at 3s.

**Minimum wait:** Every page incurs at least one 500ms sleep cycle, even if already stable. This is a deliberate choice: the simplicity of a uniform path outweighs the 50s overhead on a 100-page fully-static scan. No early-exit heuristic.

### Integration

In `crawl()`, the stability wait is called **after** the host-redirect and skip checks, but **before** both `discover_links()` and `extract_page_data()`. This ensures:
- We don't waste time stabilizing pages that will be skipped (redirects, off-host)
- JS-rendered navigation links are captured by `discover_links()`
- Content is fully rendered before text extraction

```python
# In crawl(), after skip checks:
response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
final_url = normalize_url(page.url)
# ... host redirect checks ...
if should_skip_url(final_url, cfg, allowed_host):
    continue

wait_for_content_stable(page)  # NEW: after skip checks, before extraction
discovered_links = discover_links(page, final_url)
page_data = extract_page_data(page, final_url)
```

### Timeout behavior

If content never stabilizes (live ticker, animation text, rotating carousel), we proceed after 3s with whatever text is present. The existing 97% similarity threshold in `compare_snapshots` provides a secondary safety net for minor residual jitter between scans.

### Performance impact

- Server-rendered pages: one 500ms sleep + evaluate overhead (~50-200ms on CI). Adds ~500-700ms per page.
- SPA pages: typically 2-3 cycles (1-1.5s).
- Worst case (100 pages, all timing out): ~5 min added to scan, plus evaluate() overhead. Realistic case is much lower since most pages stabilize quickly and skipped pages incur no wait.

## Files changed

- `src/website_monitor/monitor.py`: Add `wait_for_content_stable()`, revert `crawl()` to `domcontentloaded`, call stability check in `crawl()` after skip checks
- `tests/unit/test_monitor_core.py`: Add tests for the new function

## Test plan

1. **Immediate stability:** Mock page returns same text twice. Function exits after one interval.
2. **Delayed stability:** Mock page returns different text for 2 cycles, then stabilizes. Function waits and returns.
3. **Timeout:** Mock page never returns the same text twice. Function returns after timeout_ms.
4. **Empty body:** Mock page returns empty string. Function handles gracefully.
5. **Exception on first call:** Mock page.evaluate raises immediately. Function proceeds without crashing.
6. **Exception mid-loop:** Mock page.evaluate succeeds on first call, raises on second. Function returns gracefully.
7. **Integration in crawl():** Verify `wait_for_content_stable` is called after skip checks and before `discover_links()`.
