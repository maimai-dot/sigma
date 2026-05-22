"""
HTTP API Tool — make HTTP requests with JSON support.

Allows agents to call external APIs. Validates URLs, enforces timeouts,
and returns structured results.
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from sigma.agent import BaseTool

ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_TIMEOUT = 30
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB


def _validate_url(url: str) -> str | None:
    """Return error message if URL is invalid, None otherwise."""
    if not url or not isinstance(url, str):
        return "URL is required and must be a string"
    if not re.match(r"^https?://", url):
        return f"URL must start with http:// or https://, got: {url[:80]}"
    return None


def _build_request(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: str | None = None,
) -> urllib.request.Request:
    """Build a urllib Request with JSON content negotiation."""
    req_headers = {
        "Accept": "application/json",
        "User-Agent": "Sigma-HttpApiTool/1.0",
    }
    if headers:
        req_headers.update(headers)

    if body is not None:
        req_headers.setdefault("Content-Type", "application/json")

    data = body.encode("utf-8") if body is not None else None
    return urllib.request.Request(url, data=data, headers=req_headers, method=method.upper())


@dataclass
class HttpApiTool(BaseTool):
    """Make HTTP requests to external APIs.

    Supports GET, POST, PUT, DELETE with JSON request/response handling.
    """

    name: str = "http_api"
    description: str = (
        "Make HTTP requests to external APIs. Supports GET, POST, PUT, DELETE. "
        "Request body can be JSON string. Returns status_code, headers, and parsed "
        "JSON body (or raw text if not JSON)."
    )
    timeout: int = DEFAULT_TIMEOUT

    def _run(
        self,
        url: str = "",
        method: str = "GET",
        headers: dict | None = None,
        body: str | None = None,
        **kwargs,
    ) -> dict:
        """Execute an HTTP request.

        Args:
            url: Target URL (must start with http:// or https://).
            method: HTTP method (GET, POST, PUT, DELETE).
            headers: Optional dict of request headers.
            body: Optional JSON string for the request body.

        Returns:
            dict with: success, status_code, headers, body (parsed JSON or raw text),
            error (on failure).
        """
        err = _validate_url(url)
        if err:
            return {"success": False, "error": err, "status_code": None}

        method = method.upper()
        if method not in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"}:
            return {"success": False, "error": f"Unsupported method: {method}", "status_code": None}

        try:
            req = _build_request(url, method, headers, body)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read(MAX_RESPONSE_BYTES)
                text = raw.decode("utf-8", errors="replace")

                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    parsed = text

                return {
                    "success": True,
                    "status_code": resp.status,
                    "headers": dict(resp.headers),
                    "body": parsed,
                }
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
            except Exception:
                pass
            return {
                "success": False,
                "status_code": e.code,
                "error": str(e),
                "body": error_body[:4096] if error_body else None,
            }
        except urllib.error.URLError as e:
            return {"success": False, "error": f"Connection failed: {e.reason}", "status_code": None}
        except Exception as e:
            return {"success": False, "error": str(e), "status_code": None}
