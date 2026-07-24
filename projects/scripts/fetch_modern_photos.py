"""
Fetches a real present-day photo for each filming location and saves it into
assets/modern/<slug>.jpg.

Strategy, per location:
1. Geosearch Wikipedia articles near the coordinate and take the curated
   infobox/lead image of the best-matching article (by name overlap with the
   location, falling back to nearest). This is far more reliable than raw
   Commons geotags, which pick up *any* photo taken near the coordinate
   (wildlife, cars, unrelated buildings) regardless of relevance.
2. If no relevant Wikipedia article/image is found, fall back to Wikimedia
   Commons geosearch, but only accept a result whose file title shares a
   meaningful word with the location name (or the movie title) - random
   nearby photos are rejected rather than used.
3. If nothing passes, keep the placeholder image and report it as a miss.

Run with:  .venv/bin/python scripts/fetch_modern_photos.py
"""

import io
import json
import pathlib
import re
import time

import requests
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "movies.json"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "FilmLocationMapApp/1.0 (educational hobby project; no contact endpoint)"}

RADII = [1000, 5000, 20000]
BAD_TITLE_HINTS = ("logo", "icon", "flag", "map", "symbol", "coat of arms", "seal of")
MIN_BYTES = 40_000  # skip tiny thumbnails/icons
STOPWORDS = {
    "the", "and", "of", "a", "an", "at", "in", "on", "near", "former", "site",
    "set", "movie", "studio", "studios", "stage", "backlot", "used", "to",
    "recreate", "scenes", "block", "st", "street", "avenue", "blvd", "road",
    "east", "west", "north", "south", "upper", "lower", "old", "new",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def request_with_backoff(method: str, url: str, **kwargs) -> requests.Response:
    max_attempts = 6
    delay = 3.0
    for attempt in range(1, max_attempts + 1):
        resp = SESSION.request(method, url, timeout=20, **kwargs)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = min(float(resp.headers.get("Retry-After", delay)), 90.0)
        print(f"    (429 rate-limited, waiting {wait:.0f}s, attempt {attempt}/{max_attempts})")
        time.sleep(wait)
        delay *= 1.7
    resp.raise_for_status()
    return resp


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def landmark_name(loc_name: str) -> str:
    """The specific proper-noun part of a location string, e.g. 'Fort Ricasoli' out
    of 'Fort Ricasoli, Kalkara, Malta' - excludes the trailing city/region/country,
    which would otherwise trivially "match" anything else in the same city."""
    return re.split(r"[,(]", loc_name, maxsplit=1)[0]


def shares_keyword(candidate_title: str, loc_name: str, movie_title: str) -> bool:
    cand_tokens = tokens(candidate_title)
    if cand_tokens & tokens(landmark_name(loc_name)):
        return True
    if cand_tokens & tokens(movie_title):
        return True
    return False


# ---------- Wikipedia (curated article images) ----------

def wikipedia_geosearch(lat: float, lon: float, radius: int) -> list[dict]:
    resp = request_with_backoff(
        "GET",
        WIKIPEDIA_API,
        params={
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": radius,
            "gslimit": 10,
            "gsnamespace": 0,
            "format": "json",
        },
    )
    return resp.json().get("query", {}).get("geosearch", [])


def wikipedia_page_image(pageid: int) -> str | None:
    resp = request_with_backoff(
        "GET",
        WIKIPEDIA_API,
        params={
            "action": "query",
            "pageids": pageid,
            "prop": "pageimages",
            "piprop": "original",
            "format": "json",
        },
    )
    pages = resp.json().get("query", {}).get("pages", {})
    page = pages.get(str(pageid), {})
    original = page.get("original")
    return original["source"] if original else None


def try_wikipedia(loc_name: str, movie_title: str, lat: float, lon: float):
    for radius in RADII:
        time.sleep(0.4)
        try:
            candidates = wikipedia_geosearch(lat, lon, radius)
        except requests.RequestException as e:
            print(f"    ! wikipedia geosearch error: {e}")
            continue
        if not candidates:
            continue

        # Trust a candidate only if it's genuinely close to the coordinate, or
        # close-ish AND either its article title or its actual image filename
        # shares a specific (non-generic) word with the location/movie name -
        # a shared city/country name alone isn't enough, since that just means
        # two unrelated things happen to be in the same city.
        ranked = sorted(candidates, key=lambda c: c["dist"])
        for cand in ranked[:6]:
            if cand["dist"] > 5000:
                continue
            time.sleep(0.3)
            try:
                url = wikipedia_page_image(cand["pageid"])
            except requests.RequestException:
                continue
            if not (url and url.lower().endswith((".jpg", ".jpeg", ".png"))):
                continue
            filename = url.rsplit("/", 1)[-1]
            matched = shares_keyword(cand["title"], loc_name, movie_title) or shares_keyword(
                filename, loc_name, movie_title
            )
            if cand["dist"] <= 80 or (matched and cand["dist"] <= 5000):
                return url, f"wikipedia:{cand['title']} ({cand['dist']:.0f}m)"
    return None, None


# ---------- Wikimedia Commons (fallback, geotagged files) ----------

def commons_geosearch(lat: float, lon: float, radius: int) -> list[dict]:
    resp = request_with_backoff(
        "GET",
        COMMONS_API,
        params={
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": radius,
            "gslimit": 10,
            "gsnamespace": 6,
            "format": "json",
        },
    )
    return resp.json().get("query", {}).get("geosearch", [])


def commons_image_info(titles: list[str]) -> dict:
    if not titles:
        return {}
    resp = request_with_backoff(
        "GET",
        COMMONS_API,
        params={
            "action": "query",
            "titles": "|".join(titles),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1200,
            "format": "json",
        },
    )
    pages = resp.json().get("query", {}).get("pages", {})
    out = {}
    for page in pages.values():
        infos = page.get("imageinfo")
        if infos:
            out[page["title"]] = infos[0]
    return out


def pick_best_commons(candidates: list[dict], infos: dict, loc_name: str, movie_title: str):
    for cand in candidates:
        title = cand["title"]
        info = infos.get(title)
        if not info:
            continue
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        if info.get("size", 0) < MIN_BYTES:
            continue
        lower = title.lower()
        if any(hint in lower for hint in BAD_TITLE_HINTS):
            continue
        w, h = info.get("width", 0), info.get("height", 0)
        if w and h and (w / h > 2.5 or h / w > 2.5):
            continue
        if not shares_keyword(title, loc_name, movie_title):
            continue  # random nearby photo (plant, car, passerby) - not trustworthy by proximity alone
        return title, info
    return None, None


def try_commons(loc_name: str, movie_title: str, lat: float, lon: float):
    for radius in RADII:
        time.sleep(0.4)
        try:
            candidates = commons_geosearch(lat, lon, radius)
        except requests.RequestException as e:
            print(f"    ! commons geosearch error: {e}")
            continue
        if not candidates:
            continue
        time.sleep(0.4)
        infos = commons_image_info([c["title"] for c in candidates])
        found_title, found_info = pick_best_commons(candidates, infos, loc_name, movie_title)
        if found_title:
            return found_info["url"], found_title, found_info
    return None, None, None


def credit_from_extmetadata(info: dict) -> dict:
    meta = info.get("extmetadata", {})

    def clean(key):
        val = meta.get(key, {}).get("value", "")
        return " ".join(val.replace("&nbsp;", " ").split())[:200] if val else ""

    return {
        "author": clean("Artist") or "Wikimedia Commons contributor",
        "license": clean("LicenseShortName") or "See file page",
        "source_url": info.get("descriptionurl", ""),
    }


def download(url: str, dest: pathlib.Path) -> None:
    resp = request_with_backoff("GET", url)
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img.thumbnail((1000, 1000))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=85)


def main() -> None:
    data = json.loads(DATA_FILE.read_text())
    movies = data["movies"]

    hits, misses = [], []

    for movie in movies:
        for loc in movie["locations"]:
            label = f"{movie['title']} — {loc['name']}"
            dest = ROOT / loc["modern"]

            # 1. Wikipedia curated image
            wiki_url, wiki_source = try_wikipedia(loc["name"], movie["title"], loc["lat"], loc["lon"])
            if wiki_url:
                try:
                    download(wiki_url, dest)
                except Exception as e:
                    print(f"MISS  {label} (wikipedia download failed: {e})")
                    misses.append(label)
                    loc["modern_source"] = "placeholder"
                    continue
                loc["modern_source"] = "wikipedia"
                loc["modern_credit"] = {
                    "author": "Wikipedia contributors",
                    "license": "See Wikipedia article",
                    "source_url": wiki_url,
                }
                print(f"HIT   {label}  <-  {wiki_source}")
                hits.append(label)
                time.sleep(0.5)
                continue

            # 2. Commons fallback, relevance-filtered
            commons_url, commons_title, commons_info = try_commons(
                loc["name"], movie["title"], loc["lat"], loc["lon"]
            )
            if commons_url:
                try:
                    download(commons_url, dest)
                except Exception as e:
                    print(f"MISS  {label} (commons download failed: {e})")
                    misses.append(label)
                    loc["modern_source"] = "placeholder"
                    continue
                loc["modern_source"] = "wikimedia_commons"
                loc["modern_credit"] = credit_from_extmetadata(commons_info)
                print(f"HIT   {label}  <-  {commons_title} (commons fallback)")
                hits.append(label)
                time.sleep(0.5)
                continue

            print(f"MISS  {label}")
            misses.append(label)
            loc["modern_source"] = "placeholder"

    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{len(hits)} hits, {len(misses)} misses out of {len(hits) + len(misses)}")
    if misses:
        print("\nKept placeholder for:")
        for m in misses:
            print(" -", m)


if __name__ == "__main__":
    main()
