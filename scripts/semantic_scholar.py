#!/usr/bin/env python3
# pylint: disable=broad-exception-caught,too-many-branches,too-many-statements,too-many-locals
"""
Semantic Scholar discovery helper.

Two modes:

- search:    keyword search across Semantic Scholar's corpus.
- recommend: given a seed paper (DOI or Semantic Scholar id), return recommended related papers.

Reads an optional SEMANTIC_SCHOLAR_API_KEY from the project .env (raises the rate limit; the API
also works without a key, but more slowly). Results print title, year, venue, and DOI so they can be
fed back into references.bib via the normal clean/rename/digest pipeline.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
REC_BASE = "https://api.semanticscholar.org/recommendations/v1"
FIELDS = "title,year,venue,authors,externalIds,openAccessPdf"


def load_env_file():
    """Load env vars from the project .env file if present."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def request_json(url, max_retries=5):
    """GET a URL and return parsed JSON, attaching the API key header if available.

    Semantic Scholar allows roughly one request per second across all endpoints, so this retries
    on HTTP 429 with exponential backoff to stay within the limit when called in a loop.
    """
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    backoff = 1.0
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if api_key:
            req.add_header("x-api-key", api_key)
        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                print(f"Rate limited (429); waiting {backoff:.0f}s and retrying...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            if e.code == 429:
                print(
                    "Still rate limited after retries. Wait a moment and try again, or add an API key."
                )
            else:
                print(f"HTTP error {e.code}: {e.reason}")
            return None
        except Exception as e:
            print(f"Error querying Semantic Scholar: {e}")
            return None
    return None


def print_papers(papers):
    """Pretty-print a list of paper records."""
    if not papers:
        print("No results.")
        return
    for idx, paper in enumerate(papers):
        if paper is None:
            continue
        title = paper.get("title", "(no title)")
        year = paper.get("year", "?")
        venue = paper.get("venue", "")
        ext = paper.get("externalIds", {}) or {}
        doi = ext.get("DOI", "")
        oa = paper.get("openAccessPdf") or {}
        oa_url = oa.get("url", "")
        print(f"{idx + 1}. {title} ({year})")
        if venue:
            print(f"   Venue: {venue}")
        if doi:
            print(f"   DOI: {doi}")
        if oa_url:
            print(f"   Open-access PDF: {oa_url}")
        print()


def do_search(query, limit):
    """Keyword search."""
    params = {"query": query, "fields": FIELDS, "limit": limit}
    url = f"{GRAPH_BASE}/paper/search?" + urllib.parse.urlencode(params)
    print(f"Searching Semantic Scholar for: {query}\n")
    data = request_json(url)
    if data:
        print_papers(data.get("data", []))


def do_recommend(seed, limit):
    """Recommendations for a seed paper (DOI or Semantic Scholar id)."""
    paper_id = seed
    if seed.lower().startswith("10."):
        paper_id = f"DOI:{seed}"
    params = {"fields": FIELDS, "limit": limit}
    url = (
        f"{REC_BASE}/papers/forpaper/{urllib.parse.quote(paper_id, safe=':')}?"
        + urllib.parse.urlencode(params)
    )
    print(f"Recommendations related to: {seed}\n")
    data = request_json(url)
    if data:
        print_papers(data.get("recommendedPapers", []))


def main():
    """CLI entry point."""
    load_env_file()
    if len(sys.argv) < 3:
        print('Usage: python3 semantic_scholar.py search "<query>" [limit]')
        print('       python3 semantic_scholar.py recommend "<DOI or S2 id>" [limit]')
        sys.exit(1)

    mode = sys.argv[1]
    arg = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    if mode == "search":
        do_search(arg, limit)
    elif mode == "recommend":
        do_recommend(arg, limit)
    else:
        print(f"Unknown mode '{mode}'. Use 'search' or 'recommend'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
