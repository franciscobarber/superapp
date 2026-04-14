# Databricks Notebook — Step 1: Download SEPA Precios data
# Run this as a Job step BEFORE the pipeline.
# Scrapes datos.produccion.gob.ar, downloads Thursday ZIPs, writes CSVs to a Volume.

import requests
import zipfile
import io
import os
import re
from datetime import datetime
from html.parser import HTMLParser

# ── Config ──────────────────────────────────────────────────────────────────

CATALOG     = "workspace"
SCHEMA      = "superapp"
SOURCE_URL  = "https://datos.produccion.gob.ar/dataset/sepa-precios"
TARGET_CSVS = {"comercio.csv", "sucursales.csv", "productos.csv"}

# Auto-detect environment: Databricks Volume path vs local fallback.
# Change LOCAL_DATA_DIR to any local folder you want when running on your machine.
LOCAL_DATA_DIR = r"C:\Users\2371180\OneDrive - Cognizant\Documents\sepa_raw"

_in_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
VOLUME_BASE    = f"/Volumes/{CATALOG}/{SCHEMA}/sepa_raw" if _in_databricks else LOCAL_DATA_DIR
print(f"{'Databricks' if _in_databricks else 'Local'} mode. Output: {VOLUME_BASE}")

# ── HTML parser ─────────────────────────────────────────────────────────────
# Replicates the Go logic: find div.pkg-container whose h3 == "Jueves",
# then grab the first href containing "/download/".

class SepaPageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls: list[str] = []
        self._in_pkg_container = False
        self._depth = 0          # tracks nesting depth inside a pkg-container
        self._is_thursday = False
        self._candidate_url: str | None = None

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)

        if tag == "div" and attr_dict.get("class") == "pkg-container":
            self._in_pkg_container = True
            self._depth = 0
            self._is_thursday = False
            self._candidate_url = None
            return

        if self._in_pkg_container:
            self._depth += 1
            if tag == "a":
                href = attr_dict.get("href", "")
                if "/download/" in href and self._candidate_url is None:
                    self._candidate_url = href

    def handle_endtag(self, tag):
        if self._in_pkg_container:
            if self._depth == 0:
                # Closing the pkg-container div itself
                if self._is_thursday and self._candidate_url:
                    self.urls.append(self._candidate_url)
                self._in_pkg_container = False
            else:
                self._depth -= 1

    def handle_data(self, data):
        if self._in_pkg_container and data.strip() == "Jueves":
            self._is_thursday = True


def get_thursday_urls() -> list[str]:
    """Return all Thursday download URLs from the SEPA page."""
    resp = requests.get(SOURCE_URL, timeout=30)
    resp.raise_for_status()
    parser = SepaPageParser()
    parser.feed(resp.text)
    if not parser.urls:
        raise ValueError("No Thursday ZIP files found on the SEPA page")
    return parser.urls


# ── Download with progress ───────────────────────────────────────────────────

def download_to_memory(url: str) -> bytes:
    """Download a URL into memory, printing progress like the Go version."""
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunks = []

    for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            print(f"\rDownloading... {downloaded * 100 // total}%", end="", flush=True)
        else:
            print(f"\rDownloading... {downloaded / 1024 / 1024:.1f} MB", end="", flush=True)

    print()  # newline after progress
    return b"".join(chunks)


# ── ZIP extraction ───────────────────────────────────────────────────────────

def extract_date_from_name(name: str) -> str:
    """Pull YYYY-MM-DD from a filename like '2024-12-05.zip' or 'sepa_2024-12-05_jueves.zip'."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return match.group(1) if match else datetime.now().strftime("%Y-%m-%d")


def strip_bom(data: bytes) -> bytes:
    """Remove UTF-8 BOM if present — mirrors the Go unicode.IsGraphic trimming."""
    return data.lstrip(b"\xef\xbb\xbf")


def process_top_level_zip(url: str) -> list[str]:
    """
    Download the top-level ZIP, iterate nested ZIPs inside it, and for each
    nested ZIP extract the three required CSVs and APPEND them together.
    Each nested ZIP = one comercio, so we concatenate all into single files.
    Returns list of date folders written.
    """
    raw = download_to_memory(url)
    written_dates = []

    with zipfile.ZipFile(io.BytesIO(raw)) as top_zip:
        nested_zips = [n for n in top_zip.namelist() if n.lower().endswith(".zip")]

        if not nested_zips:
            print(f"  Warning: no nested ZIPs found in {url}")
            return []

        # Group by date (in case multiple dates in one top-level ZIP)
        date_to_zips = {}
        for nested_name in nested_zips:
            date_str = extract_date_from_name(nested_name)
            date_to_zips.setdefault(date_str, []).append(nested_name)

        for date_str, zips_for_date in date_to_zips.items():
            dest_dir = f"{VOLUME_BASE}/{date_str}"
            os.makedirs(dest_dir, exist_ok=True)

            print(f"\n  Processing {len(zips_for_date)} nested ZIPs for date: {date_str}")

            # Accumulate CSV content for this date (one per target CSV)
            csv_data = {target: [] for target in TARGET_CSVS}
            csv_headers = {target: None for target in TARGET_CSVS}

            for nested_name in zips_for_date:
                print(f"    Reading: {nested_name}")
                nested_bytes = io.BytesIO(top_zip.read(nested_name))

                with zipfile.ZipFile(nested_bytes) as nested_zip:
                    members = nested_zip.namelist()

                    for target in TARGET_CSVS:
                        matches = [m for m in members if m.endswith(target)]
                        if not matches:
                            print(f"      Warning: {target} not found in {nested_name}")
                            continue

                        with nested_zip.open(matches[0]) as src:
                            content = strip_bom(src.read()).decode("utf-8")

                        lines = content.strip().split("\n")
                        if not lines:
                            continue

                        # First file: save header
                        if csv_headers[target] is None:
                            csv_headers[target] = lines[0]
                            csv_data[target].extend(lines[1:])  # skip header
                        else:
                            # Subsequent files: skip header, append data
                            csv_data[target].extend(lines[1:])

            # Write concatenated CSVs to Volume
            for target in TARGET_CSVS:
                if csv_headers[target] is None:
                    print(f"    Warning: No data found for {target}")
                    continue

                dest_path = f"{dest_dir}/{target}"
                full_content = "\n".join([csv_headers[target]] + csv_data[target]) + "\n"

                with open(dest_path, "w", encoding="utf-8") as dst:
                    dst.write(full_content)

                size_kb = len(full_content.encode("utf-8")) / 1024
                row_count = len(csv_data[target])
                print(f"    Written: {dest_path}  ({size_kb:,.0f} KB, {row_count:,} rows)")

            written_dates.append(date_str)

    return written_dates


# ── Main ─────────────────────────────────────────────────────────────────────

os.makedirs(VOLUME_BASE, exist_ok=True)

print(f"Scraping {SOURCE_URL} ...")
urls = get_thursday_urls()
print(f"Found {len(urls)} Thursday URL(s):\n" + "\n".join(f"  {u}" for u in urls))

all_dates = []
for url in urls:
    print(f"\nProcessing: {url}")
    try:
        dates = process_top_level_zip(url)
        all_dates.extend(dates)
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nDone. Dates written to Volume: {all_dates}")