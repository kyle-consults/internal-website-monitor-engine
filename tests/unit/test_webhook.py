import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from website_monitor.webhook import send_webhook  # noqa: E402


class TestSendWebhook(unittest.TestCase):
    @patch("website_monitor.webhook.urlopen")
    def test_posts_payload_to_webhook_url(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = {"event": "scan_complete", "changes": 3}
        result = send_webhook("https://hooks.example.com/notify", payload)

        self.assertTrue(result["sent"])
        self.assertEqual(result["status"], 200)

        # Verify the request was a POST with JSON
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(request.data), payload)

    def test_skips_when_url_is_none(self):
        result = send_webhook(None, {"event": "test"})
        self.assertFalse(result["sent"])
        self.assertEqual(result["reason"], "no_webhook_url")

    def test_skips_when_url_is_empty(self):
        result = send_webhook("", {"event": "test"})
        self.assertFalse(result["sent"])
        self.assertEqual(result["reason"], "no_webhook_url")

    @patch("website_monitor.webhook.urlopen")
    def test_returns_failure_on_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            url="https://hooks.example.com/notify",
            code=502,
            msg="Bad Gateway",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

        result = send_webhook("https://hooks.example.com/notify", {"event": "test"})
        self.assertFalse(result["sent"])
        self.assertEqual(result["reason"], "http_error_502")


if __name__ == "__main__":
    unittest.main()
