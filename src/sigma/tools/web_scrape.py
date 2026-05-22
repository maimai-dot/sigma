"""
Web Scrape Tool — fetch and parse web pages.

Downloads a URL, extracts plain text from HTML (no external parser needed).
Also supports fetching raw JSON from APIs.
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from sigma.agent import BaseTool

DEFAULT_TIMEOUT = 30
MAX_RESPONSE_BYTES = 4 * 1024 * 1024  # 4 MB

# Minimal HTML-to-text conversion patterns
SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|iframe|svg|canvas|video|audio|nav|footer|header|aside)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s{3,}")
ENTITY_RE = re.compile(r"&[a-z]+;")


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return readable text."""
    text = SCRIPT_STYLE_RE.sub(" ", html)
    text = TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&apos;", "'")
    text = ENTITY_RE.sub(" ", text)
    text = WHITESPACE_RE.sub("\n", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()[:50000]  # max 50KB of text


@dataclass
class WebScrapeTool(BaseTool):
    """Fetch and parse web pages.

    Fetches a URL and extracts plain text from HTML. Also supports
    fetching raw JSON from API endpoints.
    """

    name: str = "web_scrape"
    description: str = (
        "Fetch and extract text from a web page. "
        "Operations: 'fetch' (HTML → plain text), 'json' (parse JSON API response). "
        "Returns extracted text or parsed JSON data."
    )
    timeout: int = DEFAULT_TIMEOUT

    def _run(
        self, url: str = "", operation: str = "fetch", **kwargs
    ) -> dict:
        """Fetch and parse a web page.

        Args:
            url: Target URL to fetch.
            operation: 'fetch' (HTML to text) or 'json' (parse as JSON API).

        Returns:
            dict with: success, url, status_code, content (text or parsed JSON).
        """
        if not url:
            return {"success": False, "error": "url is required"}

        if not re.match(r"^https?://", url):
            return {"success": False, "error": f"URL must start with http:// or https://"}

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Sigma-WebScrapeTool/1.0", "Accept": "text/html,application/json,*/*"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read(MAX_RESPONSE_BYTES)
                text = raw.decode("utf-8", errors="replace")
                content_type = resp.headers.get("Content-Type", "")

                if operation == "json" or "application/json" in content_type:
                    try:
                        data = json.loads(text)
                        return {"success": True, "url": url, "status_code": resp.status,
                                "data": data, "format": "json"}
                    except json.JSONDecodeError:
                        return {"success": False, "error": "Response is not valid JSON",
                                "status_code": resp.status}

                plain_text = _html_to_text(text)
                return {
                    "success": True,
                    "url": url,
                    "status_code": resp.status,
                    "content": plain_text,
                    "char_count": len(plain_text),
                    "format": "text",
                }

        except urllib.error.HTTPError as e:
            return {"success": False, "error": f"HTTP {e.code}: {e.reason}", "status_code": e.code, "url": url}
        except urllib.error.URLError as e:
            return {"success": False, "error": f"Connection failed: {e.reason}", "url": url}
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}


@dataclass
class DirectorySearchTool(BaseTool):
    """Recursively search for files by name pattern.

    Finds files matching a glob pattern within a directory tree.
    """

    name: str = "directory_search"
    description: str = (
        "Search for files by name pattern. Supports * and ** globs. "
        "Returns matching file paths, sizes, and modification times."
    )
    max_results: int = 200

    def _run(self, directory: str = "", pattern: str = "*", **kwargs) -> dict:
        """Search for files matching a glob pattern.

        Args:
            directory: Root directory to search from.
            pattern: File name pattern with wildcards (e.g., '*.py', 'test_*.ts').

        Returns:
            dict with: success, directory, pattern, match_count, files (list of {path, size, mtime}).
        """
        import datetime
        from pathlib import Path

        if not directory:
            directory = "."

        root = Path(directory)
        if not root.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}
        if not root.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}

        matches = []
        for filepath in root.rglob(pattern):
            if filepath.is_file():
                stat = filepath.stat()
                matches.append({
                    "path": str(filepath),
                    "size": stat.st_size,
                    "mtime": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            if len(matches) >= self.max_results:
                break

        return {
            "success": True,
            "directory": str(root),
            "pattern": pattern,
            "match_count": len(matches),
            "files": sorted(matches, key=lambda x: x["path"]),
        }
