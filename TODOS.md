# TODOS

## Knowledge extraction quality eval
**What:** Build a simple eval script that runs extraction against a known test page and checks output consistency across runs.
**Why:** Google may update Flash Lite at any time. If extraction quality degrades (label drift, missing units, classification errors), the only way to catch it is a repeatable eval against a known baseline.
**Pros:** Catches model regressions early, before clients see false positives in their reports. Provides confidence when bumping model versions.
**Cons:** Requires maintaining a test page and expected-output fixture. Small ongoing maintenance cost.
**Context:** The knowledge extraction pipeline uses Gemini Flash Lite with structured output to extract operational knowledge units from web pages. The extraction is autonomous (no per-client config), so the model's consistency IS the system's reliability. A simple eval that extracts from a fixed HTML fixture and asserts the output matches expected units (category, label, value) would catch regressions. Run manually during development and optionally as a CI check.
**Depends on:** Knowledge extraction pipeline (v1) must be implemented first.
