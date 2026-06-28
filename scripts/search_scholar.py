# pylint: disable=broad-exception-caught
import json
import os
import sys
import urllib.parse
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def search_scholar(query, year_start=None, num_results=10):
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        print("Error: SERPAPI_API_KEY not found. Add it to the project .env file.")
        return None
    params = {"engine": "google_scholar", "q": query, "api_key": api_key, "num": num_results}
    if year_start:
        params["as_ylo"] = year_start

    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(params)
    print(f"Querying SerpAPI with: {query} (year_start: {year_start})")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        print(f"Error querying SerpAPI: {e}")
        return None


def main():
    load_env_file()
    if len(sys.argv) < 2:
        print("Usage: python3 search_scholar.py <query> [year_start] [num_results]")
        sys.exit(1)

    query = sys.argv[1]
    year_start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    num_results = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    results = search_scholar(query, year_start, num_results)
    if not results:
        print("No results returned.")
        return

    organic_results = results.get("organic_results", [])
    print(f"\nFound {len(organic_results)} results:\n")

    for idx, paper in enumerate(organic_results):
        title = paper.get("title")
        link = paper.get("link")
        snippet = paper.get("snippet")
        publication_info = paper.get("publication_info", {})
        summary = publication_info.get("summary", "")

        print(f"{idx + 1}. {title}")
        if link:
            print(f"   Link: {link}")
        if summary:
            print(f"   Summary: {summary}")
        if snippet:
            print(f"   Snippet: {snippet}")
        print()


if __name__ == "__main__":
    main()
