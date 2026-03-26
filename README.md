# Website Monitor Engine

Shared website monitor engine and reusable GitHub Actions workflow for `kyle-consults` automation repositories.

## What Lives Here

- reusable workflow: `.github/workflows/reusable-monitor.yml`
- monitor package: `src/website_monitor/`
- automated tests for scan, notification, and workflow behavior

## Consumption Model

Client repositories and thin templates call the reusable workflow from this repository:

```yaml
jobs:
  scan:
    uses: kyle-consults/internal-website-monitor-engine/.github/workflows/reusable-monitor.yml@main
    with:
      homepage_url: ${{ vars.HOMEPAGE_URL }}
      alert_email_to: ${{ vars.ALERT_EMAIL_TO }}
      alert_email_from: ${{ vars.ALERT_EMAIL_FROM }}
      email_mode: ${{ vars.EMAIL_MODE }}
    secrets: inherit
```

The reusable workflow checks out the caller repository, runs the scan against the caller repo's config and output directories, commits updated outputs back to the caller repo, and optionally sends email notifications.

## Local Validation

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
