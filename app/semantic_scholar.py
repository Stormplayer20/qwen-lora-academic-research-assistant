"""
Client for the Semantic Scholar Graph API.

Supports:
- Optional API key from .env
- Automatic retries for rate limits
- Clear error messages instead of generic server crashes
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

BASE_URL = "https://api.semanticscholar.org/graph/v1"

DEFAULT_FIELDS = [
    "title",
    "abstract",
    "authors",
    "year",
    "citationCount",
    "url",
    "venue",
    "externalIds",
]


class SemanticScholarError(RuntimeError):
    """General Semantic Scholar API failure."""


class SemanticScholarRateLimitError(SemanticScholarError):
    """Semantic Scholar rejected the request because of rate limiting."""


class SemanticScholarClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 20.0,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries

        self.headers = {
            "User-Agent": "Academic-Research-Assistant/1.0"
        }

        key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if key:
            self.headers["x-api-key"] = key

    def search_papers(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search Semantic Scholar and return raw paper records."""

        cleaned_query = query.strip().strip('"').strip("'")

        if not cleaned_query:
            return []

        safe_limit = max(1, min(limit, 100))

        params = {
            "query": cleaned_query,
            "limit": safe_limit,
            "fields": ",".join(fields or DEFAULT_FIELDS),
        }

        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers=self.headers,
        ) as client:

            for attempt in range(self.max_retries + 1):
                try:
                    response = client.get(
                        f"{BASE_URL}/paper/search",
                        params=params,
                    )
                except httpx.RequestError as exc:
                    if attempt >= self.max_retries:
                        raise SemanticScholarError(
                            f"Could not connect to Semantic Scholar: {exc}"
                        ) from exc

                    time.sleep(2**attempt)
                    continue

                if response.status_code == 429:
                    if attempt >= self.max_retries:
                        raise SemanticScholarRateLimitError(
                            "Semantic Scholar is temporarily rate-limiting "
                            "requests. Wait briefly or configure an API key."
                        )

                    retry_after = response.headers.get("retry-after")

                    try:
                        wait_seconds = (
                            float(retry_after)
                            if retry_after
                            else float(2**attempt)
                        )
                    except ValueError:
                        wait_seconds = float(2**attempt)

                    wait_seconds = min(max(wait_seconds, 1.0), 30.0)
                    time.sleep(wait_seconds)
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise SemanticScholarError(
                        "Semantic Scholar returned "
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    ) from exc

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise SemanticScholarError(
                        "Semantic Scholar returned invalid JSON."
                    ) from exc

                return payload.get("data", [])

        return []


if __name__ == "__main__":
    client = SemanticScholarClient()

    try:
        results = client.search_papers(
            "algorithmic bias in hiring",
            limit=3,
        )

        for paper in results:
            print(
                paper.get("title"),
                "-",
                paper.get("year"),
                "-",
                paper.get("citationCount"),
                "citations",
            )

    except SemanticScholarError as exc:
        print(f"Semantic Scholar error: {exc}")