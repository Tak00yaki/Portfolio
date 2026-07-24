"""
Fetches a real photo gallery for each castle and saves it into
assets/castles/<slug>_N.jpg, crediting source and license.

Strategy, per castle: query the English Wikipedia article for the castle
(by its known wiki_title) and list the images actually used on that page
via generator=images. This is far more reliable than geosearch here,
since we know the exact subject - editors already curated these images
for this specific castle, rather than picking up whatever happens to be
near a coordinate. Results are filtered (skip logos/maps/icons/svgs,
tiny thumbnails, extreme aspect ratios) and the largest PHOTOS_PER_CASTLE
remaining images are downloaded.

Run with:  .venv/bin/python scripts/fetch_castle_photos.py
"""

import html
import io
import json
import pathlib
import re
import time

import requests
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "castles.json"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "CastleMapApp/1.0 (educational hobby project; no contact endpoint)"}

PHOTOS_PER_CASTLE = 3
MIN_BYTES = 60_000  # skip tiny thumbnails/icons
BAD_HINTS = (
    "logo", "icon", "flag", "map", "symbol", "coat of arms", "seal of",
    "locator", "plan", "diagram", "crest", "arms of", "escutcheon",
    ".svg", ".gif", "wiki", "commons-logo", "question_book",
)

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def request_with_backoff(method: str, url: str, **kwargs) -> requests.Response:
    max_attempts = 3
    delay = 5.0
    for attempt in range(1, max_attempts + 1):
        resp = SESSION.request(method, url, timeout=20, **kwargs)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = min(float(resp.headers.get("Retry-After", delay)), 20.0)
        print(f"    (429 rate-limited, waiting {wait:.0f}s, attempt {attempt}/{max_attempts})")
        time.sleep(wait)
        delay *= 1.7
    resp.raise_for_status()
    return resp


def article_images(title: str) -> list[dict]:
    """Images used on the Wikipedia article, with url/size/mime/credit info."""
    resp = request_with_backoff(
        "GET",
        WIKIPEDIA_API,
        params={
            "action": "query",
            "titles": title,
            "generator": "images",
            "gimlimit": 40,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1200,
            "format": "json",
        },
    )
    pages = resp.json().get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        infos = page.get("imageinfo")
        if not infos:
            continue
        info = infos[0]
        out.append({"title": page.get("title", ""), **info})
    return out


def pick_best(candidates: list[dict]) -> list[dict]:
    picked = []
    for cand in candidates:
        if cand.get("mime") not in ("image/jpeg", "image/png"):
            continue
        size = cand.get("size", 0)
        if size < MIN_BYTES:
            continue
        lower = cand["title"].lower()
        if any(hint in lower for hint in BAD_HINTS):
            continue
        w, h = cand.get("width", 0), cand.get("height", 0)
        if w and h and (w / h > 2.2 or h / w > 2.2):
            continue
        picked.append(cand)
    picked.sort(key=lambda c: c.get("size", 0), reverse=True)
    return picked[:PHOTOS_PER_CASTLE]


def credit_from_extmetadata(info: dict) -> dict:
    meta = info.get("extmetadata", {})

    def clean(key):
        val = meta.get(key, {}).get("value", "")
        if not val:
            return ""
        val = re.sub(r"<[^>]+>", "", val)  # strip HTML tags, e.g. <a><bdi>Name</bdi></a>
        val = html.unescape(val.replace("&nbsp;", " "))
        return " ".join(val.split())[:200]

    return {
        "author": clean("Artist") or "Wikimedia Commons contributor",
        "license": clean("LicenseShortName") or "See file page",
        "source_url": info.get("descriptionurl", ""),
    }


def download(url: str, dest: pathlib.Path) -> None:
    resp = request_with_backoff("GET", url)
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img.thumbnail((1200, 1200))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=85)


def main() -> None:
    data = json.loads(DATA_FILE.read_text())
    castles = data["castles"]

    total_hits, total_misses = 0, 0

    for castle in castles:
        if all(p.get("source") == "wikipedia" for p in castle.get("photos", [])):
            print(f"\n{castle['name']}: already has real photos, skipping")
            continue
        print(f"\n{castle['name']}")
        try:
            candidates = article_images(castle["wiki_title"])
        except requests.RequestException as e:
            print(f"  ! wikipedia query error: {e}")
            candidates = []
        best = pick_best(candidates)

        photos = []
        for i in range(1, PHOTOS_PER_CASTLE + 1):
            rel_path = f"assets/castles/{castle['slug']}_{i}.jpg"
            dest = ROOT / rel_path
            if i <= len(best):
                info = best[i - 1]
                try:
                    time.sleep(1.0)
                    download(info.get("thumburl") or info["url"], dest)
                    photos.append({
                        "path": rel_path,
                        "source": "wikipedia",
                        "credit": credit_from_extmetadata(info),
                    })
                    print(f"  HIT  {i}: {info['title']}")
                    total_hits += 1
                    continue
                except Exception as e:
                    print(f"  MISS {i}: download failed ({e})")
            else:
                print(f"  MISS {i}: no more usable candidates, kept placeholder")
            photos.append({"path": rel_path, "source": "placeholder"})
            total_misses += 1

        castle["photos"] = photos
        DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        time.sleep(0.5)

    print(f"\n{total_hits} real photos fetched, {total_misses} placeholders kept "
          f"out of {total_hits + total_misses} total.")


if __name__ == "__main__":
    main()
