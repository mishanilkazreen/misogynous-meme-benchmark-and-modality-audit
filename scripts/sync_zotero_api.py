#!/usr/bin/env python3
# pylint: disable=broad-exception-caught,too-many-statements
"""
Sync references directly from Zotero API and merge into references.bib.
"""

import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRANT = os.path.join(PROJECT_ROOT, "application", "horizon_drs_2026")
grant_dir = os.environ.get("GRANT_DIR", DEFAULT_GRANT)
bib_path = os.environ.get("BIB_PATH", os.path.join(grant_dir, "references.bib"))


def load_env_file():
    """Load env vars from .env file if it exists."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()


def main():
    """Fetch all Zotero references and append to references.bib."""
    load_env_file()
    user_id = os.environ.get("ZOTERO_USER_ID")
    api_key = os.environ.get("ZOTERO_API_KEY")

    if not user_id or not api_key:
        print("Error: ZOTERO_USER_ID and ZOTERO_API_KEY environment variables not found.")
        print("\nPlease create a '.env' file in the project root containing:")
        print("ZOTERO_USER_ID=your_numerical_userid")
        print("ZOTERO_API_KEY=your_private_api_key")
        print("\nTo generate these keys:")
        print("1. Log in to your Zotero account at https://www.zotero.org")
        print("2. Go to Settings -> Feeds/API (https://www.zotero.org/settings/keys)")
        print("3. Note your User ID and click 'Create new private key'")
        print("4. Check 'Allow write access' if you want, save, and copy the key.")
        sys.exit(1)

    print(f"Connecting to Zotero API for User ID: {user_id}...")

    all_bibtex = []
    start = 0
    limit = 100

    while True:
        url = f"https://api.zotero.org/users/{user_id}/items?format=bibtex&limit={limit}&start={start}"
        req = urllib.request.Request(url)
        req.add_header("Zotero-API-Key", api_key)
        req.add_header("User-Agent", "Mozilla/5.0")

        try:
            with urllib.request.urlopen(req) as response:
                content = response.read().decode("utf-8").strip()
                if not content:
                    break
                all_bibtex.append(content)
                # Check if there's more data
                link_header = response.headers.get("Link", "")
                if 'rel="next"' not in link_header:
                    break
                start += limit
        except urllib.error.HTTPError as e:
            print(f"HTTP Error querying Zotero API: {e.code} {e.reason}")
            if e.code == 403:
                print("Please check if your API Key is correct and has access permission.")
            sys.exit(1)
        except Exception as e:
            print(f"Error querying Zotero API: {e}")
            sys.exit(1)

    if not all_bibtex:
        print("No items found in your Zotero library.")
        sys.exit(0)

    combined_bibtex = "\n\n".join(all_bibtex)

    # Let's append to references.bib
    print(f"Fetched Zotero references. Merging into {bib_path}...")
    with open(bib_path, "a", encoding="utf-8") as f:
        f.write("\n\n" + combined_bibtex + "\n")

    # Run clean references
    print("Running clean_references.py to deduplicate and format the bibliography...")
    result = subprocess.run(
        ["python3", os.path.join(PROJECT_ROOT, "scripts", "clean_references.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout)
    print("Sync complete!")


if __name__ == "__main__":
    main()
