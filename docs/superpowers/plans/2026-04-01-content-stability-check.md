# Content Stability Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `networkidle` with `domcontentloaded` + text-level stability polling so the crawler waits for JS-rendered content to stabilize before extracting text, eliminating flapping diffs on SPA sites.

**Architecture:** One new function `wait_for_content_stable()` polls `document.body.innerText` every 500ms until two consecutive readings match or 3s elapses. Called in `crawl()` after skip checks, before link discovery and text extraction.

**Tech Stack:** Python 3.13, Playwright (existing), `time` stdlib module (new import)

---

### Task 1: Add `wait_for_content_stable()` with TDD

**Files:**
- Create tests in: `tests/unit/test_monitor_core.py` (append to existing)
- Create function in: `src/website_monitor/monitor.py:577` (above `extract_page_data`)

- [ ] **Step 1: Add the `time` import to monitor.py**

In `src/website_monitor/monitor.py`, add `import time` after the existing `import re` on line 6:

```python
import re
import time
```

- [ ] **Step 2: Write failing test for immediate stability**

Append to `tests/unit/test_monitor_core.py`, inside the `MonitorCoreTests` class (before the `if __name__` block at line 483). Also add `wait_for_content_stable` to the import list at the top of the file.

Add to the import block at line 10:

```python
from website_monitor.monitor import (
    clean_text,
    compare_snapshots,
    discover_links,
    extract_page_data,
    normalize_for_hash,
    normalize_url,
    prune_archives,
    render_report,
    resolve_runtime_root,
    strip_boilerplate_js,
    summarize_text_changes,
    should_adopt_homepage_redirect_host,
    should_skip_url,
    wait_for_content_stable,
)
```

Add the test before the `if __name__` block:

```python
    def test_wait_for_content_stable_returns_when_text_matches(self) -> None:
        page = FakeStabilityPage(responses=["Hello world", "Hello world"])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 2)
```

And add this helper class at the bottom of the file, after the `FakePageWithLinks` class:

```python
class FakeStabilityPage:
    """Simulates a Playwright page for stability check tests."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = responses
        self.call_count = 0

    def evaluate(self, expression: str) -> str:
        if self.call_count >= len(self._responses):
            return self._responses[-1] if self._responses else ""
        result = self._responses[self.call_count]
        self.call_count += 1
        if isinstance(result, Exception):
            raise result
        return result
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/unit/test_monitor_core.py::MonitorCoreTests::test_wait_for_content_stable_returns_when_text_matches -v`

Expected: FAIL with `ImportError: cannot import name 'wait_for_content_stable'`

- [ ] **Step 4: Write minimal implementation**

Add this function in `src/website_monitor/monitor.py` at line 578, right above `extract_page_data`:

```python
def wait_for_content_stable(page, timeout_ms: int = 3000, interval_ms: int = 500) -> None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    interval_s = interval_ms / 1000.0
    try:
        previous_text = page.evaluate("() => document.body?.innerText || ''")
    except Exception:
        return
    while time.monotonic() < deadline:
        time.sleep(interval_s)
        try:
            current_text = page.evaluate("() => document.body?.innerText || ''")
        except Exception:
            return
        if current_text == previous_text:
            return
        previous_text = current_text
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/unit/test_monitor_core.py::MonitorCoreTests::test_wait_for_content_stable_returns_when_text_matches -v`

Expected: PASS

- [ ] **Step 6: Write failing test for delayed stability**

Add to `MonitorCoreTests`:

```python
    def test_wait_for_content_stable_waits_for_changing_content(self) -> None:
        page = FakeStabilityPage(responses=["Loading...", "Partial content", "Full content", "Full content"])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 4)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/unit/test_monitor_core.py::MonitorCoreTests::test_wait_for_content_stable_waits_for_changing_content -v`

Expected: PASS (implementation already handles this case)

- [ ] **Step 8: Write failing test for timeout**

Add to `MonitorCoreTests`:

```python
    def test_wait_for_content_stable_returns_after_timeout(self) -> None:
        call_counter = {"n": 0}
        class NeverStablePage:
            def evaluate(self, expression: str) -> str:
                call_counter["n"] += 1
                return f"text-{call_counter['n']}"

        page = NeverStablePage()

        import time as _time
        start = _time.monotonic()
        wait_for_content_stable(page, timeout_ms=500, interval_ms=100)
        elapsed = _time.monotonic() - start

        self.assertGreaterEqual(elapsed, 0.4)
        self.assertLess(elapsed, 2.0)
```

- [ ] **Step 9: Run test to verify it passes**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/unit/test_monitor_core.py::MonitorCoreTests::test_wait_for_content_stable_returns_after_timeout -v`

Expected: PASS

- [ ] **Step 10: Write failing test for empty body**

Add to `MonitorCoreTests`:

```python
    def test_wait_for_content_stable_handles_empty_body(self) -> None:
        page = FakeStabilityPage(responses=["", ""])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 2)
```

- [ ] **Step 11: Run test to verify it passes**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/unit/test_monitor_core.py::MonitorCoreTests::test_wait_for_content_stable_handles_empty_body -v`

Expected: PASS

- [ ] **Step 12: Write failing test for exception on first call**

Add to `MonitorCoreTests`:

```python
    def test_wait_for_content_stable_handles_exception_on_first_call(self) -> None:
        page = FakeStabilityPage(responses=[RuntimeError("page closed")])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 1)
```

- [ ] **Step 13: Run test to verify it passes**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/unit/test_monitor_core.py::MonitorCoreTests::test_wait_for_content_stable_handles_exception_on_first_call -v`

Expected: PASS

- [ ] **Step 14: Write failing test for mid-loop exception**

Add to `MonitorCoreTests`:

```python
    def test_wait_for_content_stable_handles_mid_loop_exception(self) -> None:
        page = FakeStabilityPage(responses=["Hello world", RuntimeError("context destroyed")])

        wait_for_content_stable(page, timeout_ms=3000, interval_ms=50)

        self.assertEqual(page.call_count, 2)
```

- [ ] **Step 15: Run test to verify it passes**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/unit/test_monitor_core.py::MonitorCoreTests::test_wait_for_content_stable_handles_mid_loop_exception -v`

Expected: PASS

- [ ] **Step 16: Run full test suite**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/ -v`

Expected: All tests pass (58 existing + 6 new = 64 total)

- [ ] **Step 17: Commit**

```bash
cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine
git add src/website_monitor/monitor.py tests/unit/test_monitor_core.py
git commit -m "feat: add wait_for_content_stable() with full test coverage

Polls document.body.innerText every 500ms until two consecutive
readings match or 3s timeout. Handles exceptions, empty bodies,
and mid-loop failures gracefully."
```

---

### Task 2: Integrate stability check into `crawl()` and revert to `domcontentloaded`

**Files:**
- Modify: `src/website_monitor/monitor.py:645` (crawl function)

- [ ] **Step 1: Revert `wait_until` from `networkidle` to `domcontentloaded`**

In `src/website_monitor/monitor.py`, in the `crawl()` function, change line 645:

```python
# Before:
                    response = page.goto(url, wait_until="networkidle", timeout=timeout_ms)

# After:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
```

- [ ] **Step 2: Add `wait_for_content_stable()` call after skip checks**

In `src/website_monitor/monitor.py`, in the `crawl()` function, add the stability check call after the `should_skip_url(final_url)` check (after line 651) and before `discover_links` (line 653):

```python
                    if should_skip_url(final_url, cfg, allowed_host):
                        continue

                    wait_for_content_stable(page)

                    discovered_links = discover_links(page, final_url)
```

- [ ] **Step 3: Run the full test suite**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/ -v`

Expected: All 64 tests pass

- [ ] **Step 4: Commit**

```bash
cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine
git add src/website_monitor/monitor.py
git commit -m "feat: integrate content stability check into crawl()

Revert wait_until from networkidle to domcontentloaded and add
wait_for_content_stable() call after skip checks, before link
discovery and text extraction. Fixes flapping diffs on JS-rendered
sites like AFC urgent care."
```

---

### Task 3: Add baseline reset warning to `run_monitor()`

**Files:**
- Modify: `src/website_monitor/monitor.py` (run_monitor function, ~line 697)

- [ ] **Step 1: Add a print warning when changes are detected on a non-baseline run**

In `src/website_monitor/monitor.py`, in the `run_monitor()` function, after `diff = compare_snapshots(previous, current)` (line 697) and `baseline_created = previous is None` (line 698), add:

```python
    diff = compare_snapshots(previous, current)
    baseline_created = previous is None
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
```

This only warns when more than 50% of pages changed, which is the signature of a baseline reset (not normal editorial changes).

- [ ] **Step 2: Run the full test suite**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/ -v`

Expected: All 64 tests pass

- [ ] **Step 3: Commit**

```bash
cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine
git add src/website_monitor/monitor.py
git commit -m "feat: warn when mass changes suggest baseline reset

Prints a note when >50% of pages show changes, which is the
typical signature of a first scan after engine updates. Helps
users distinguish expected baseline resets from real site changes."
```

---

### Task 4: Final verification

- [ ] **Step 1: Run the full test suite one final time**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && uv run pytest tests/ -v`

Expected: All 64 tests pass, no warnings

- [ ] **Step 2: Verify git log looks clean**

Run: `cd /Users/kylezhang/Developer/kyle-consults/internal-website-monitor-engine && git log --oneline -5`

Expected: 3 new commits on top of the spec commits
