# Webhook notification: POST JSON payloads to an external URL.
#
# Uses stdlib only (urllib.request). Never raises — always returns a status dict.

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def send_webhook(
    url: str | None,
    payload: dict[str, Any],
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """POST a JSON payload to *url*. Never raises.

    Returns a status dict:
        - ``{"sent": True, "status": 200}`` on success.
        - ``{"sent": False, "reason": "no_webhook_url"}`` if *url* is None or empty.
        - ``{"sent": False, "reason": "http_error_NNN"}`` on HTTP errors.
        - ``{"sent": False, "reason": "connection_error"}`` on network errors.
    """
    if not url:
        return {"sent": False, "reason": "no_webhook_url"}

    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return {"sent": True, "status": response.status}
    except HTTPError as exc:
        logger.warning("Webhook HTTP error %d for %s", exc.code, url)
        return {"sent": False, "reason": f"http_error_{exc.code}"}
    except (URLError, OSError) as exc:
        logger.warning("Webhook connection error for %s: %s", url, exc)
        return {"sent": False, "reason": "connection_error"}
    except Exception:
        logger.exception("Unexpected webhook error for %s", url)
        return {"sent": False, "reason": "unexpected_error"}
