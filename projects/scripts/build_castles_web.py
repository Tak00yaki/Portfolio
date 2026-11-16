"""
Builds the static Castle Map at khushikhan.com/castles/ from the same data
the Streamlit app uses.

Reads  projects/data/castles.json and projects/assets/castles/*.jpg
Writes public/castles/castles.json and public/castles/photos/
       (<slug>_N.jpg for the gallery, <slug>_N_t.jpg thumbnails for pins)

Placeholder photos are skipped, so castles without real photos show "no photo".
Re-run after changing the data or fetching new photos:

    .venv/bin/python scripts/build_castles_web.py      (from projects/)
"""

import json
import pathlib
import shutil

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent          # projects/
SITE = ROOT.parent / "public" / "castles"
DATA_FILE = ROOT / "data" / "castles.json"
OUT_PHOTOS = SITE / "photos"

FULL_MAX = 800     # longest side of gallery photos, px
THUMB_MAX = 360    # longest side of map-popup thumbnails, px


def save_resized(src: pathlib.Path, dest: pathlib.Path, max_side: int, quality: int) -> None:
    img = Image.open(src).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(dest, "JPEG", quality=quality, optimize=True, progressive=True)


def main() -> None:
    castles = json.loads(DATA_FILE.read_text())["castles"]

    if OUT_PHOTOS.exists():
        shutil.rmtree(OUT_PHOTOS)
    OUT_PHOTOS.mkdir(parents=True)

    out = []
    for castle in castles:
        photos = []
        for photo in castle.get("photos", []):
            src = ROOT / photo["path"]
            if photo.get("source") == "placeholder" or not src.exists():
                continue
            stem = src.stem
            save_resized(src, OUT_PHOTOS / f"{stem}.jpg", FULL_MAX, 72)
            save_resized(src, OUT_PHOTOS / f"{stem}_t.jpg", THUMB_MAX, 68)
            credit = photo.get("credit") or {}
            photos.append({
                "src": f"photos/{stem}.jpg",
                "thumb": f"photos/{stem}_t.jpg",
                "author": credit.get("author"),
                "license": credit.get("license"),
                "url": credit.get("source_url"),
            })

        entry = {k: v for k, v in castle.items() if k not in ("photos", "wiki_title", "century_built")}
        entry["photos"] = photos
        out.append(entry)

    (SITE / "castles.json").write_text(json.dumps({"castles": out}, ensure_ascii=False, separators=(",", ":")))
    n_photos = sum(len(c["photos"]) for c in out)
    print(f"Wrote {len(out)} castles and {n_photos} photos to {SITE}")


if __name__ == "__main__":
    main()
