"""Download the Cricsheet IPL match files (JSON) into cricket_data/ipl_json.

Makes one request, for https://cricsheet.org/downloads/ipl_json.zip, and only if the
files are not already on disk. Cricsheet's robots.txt allows /downloads/.

The data is licensed under the Open Data Commons Attribution License 1.0
(https://opendatacommons.org/licenses/by/1-0/); see the README for attribution.

The zip on Cricsheet grows as new seasons finish. The analysis uses the 2008 to 2025
seasons (1,169 matches), which `python -m cricket_ba.parse` selects by default, so a
newer download gives the same tables.
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://cricsheet.org/downloads/ipl_json.zip"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dest", default="cricket_data/ipl_json")
    p.add_argument("--force", action="store_true", help="download even if files exist")
    a = p.parse_args()

    dest = Path(a.dest)
    if dest.exists() and any(dest.glob("*.json")) and not a.force:
        print(f"{dest} already has match files; use --force to download again")
        return

    req = urllib.request.Request(URL, headers={"User-Agent": "cricket_ba data script"})
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = r.read()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        z.extractall(dest)
    print(f"extracted {len(list(dest.glob('*.json')))} match files to {dest}")


if __name__ == "__main__":
    main()
