# Accuracy Lab Fixtures

Synthetic pages for evaluating monitor accuracy without changing live sites.

Run the automated checks:

```bash
uv run python -m pytest tests/integration/test_accuracy_lab.py -q
```

What these pages simulate:

- `index.html`: query-scoped homepage, exact operational hours, phone number,
  volatile visitor count, volatile rendered time, rotating promo, and dynamic
  widget text.
- `hours.html`: exact hours with `HH:MM` values plus a constantly changing
  widget inside the monitored content area.
- `billing.html`: extra linked page used to verify normal crawl discovery still
  works while the monitor strips dynamic content for snapshots.

Expected behavior:

- `?zip=95050` is preserved from the initial `HOMEPAGE_URL`.
- `utm_source` is dropped.
- dynamic widgets, visitor counts, rendered timestamps, and rotating promos do
  not appear in snapshots.
- exact facts like `8:30 AM - 5:00 PM` remain in snapshots.
- two back-to-back runs produce no changes when only dynamic noise changes.
