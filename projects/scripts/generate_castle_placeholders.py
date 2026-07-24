"""
One-off helper: generates placeholder photos for every castle in
data/castles.json, so the app has a gallery to show before real images are
fetched. Each castle gets PHOTOS_PER_CASTLE placeholder images under
assets/castles/<slug>_N.jpg.

Run once with:  .venv/bin/python scripts/generate_castle_placeholders.py
Then run scripts/fetch_castle_photos.py to replace them with real,
freely-licensed photos from Wikipedia/Wikimedia Commons.
"""

import json
import pathlib
import random

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "castles.json"
PHOTOS_PER_CASTLE = 3

BG_COLORS = ["#2b2118", "#1c2b33", "#241c33", "#332318", "#182b22", "#33291c"]
FG = "#f2e9d8"


def draw_placeholder(path: pathlib.Path, title: str, subtitle: str, bg: str) -> None:
    img = Image.new("RGB", (900, 600), bg)
    draw = ImageDraw.Draw(img)
    font_big = ImageFont.load_default(size=32)
    font_small = ImageFont.load_default(size=18)

    draw.rectangle([0, 0, 899, 599], outline=FG, width=3)
    draw.text((40, 260), title, fill=FG, font=font_big)
    draw.text((40, 305), subtitle, fill=FG, font=font_small)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=85)


def main() -> None:
    data = json.loads(DATA_FILE.read_text())
    castles = data["castles"]

    for castle in castles:
        photos = []
        for i in range(1, PHOTOS_PER_CASTLE + 1):
            rel_path = f"assets/castles/{castle['slug']}_{i}.jpg"
            draw_placeholder(
                ROOT / rel_path,
                castle["name"],
                f"PLACEHOLDER PHOTO {i} of {PHOTOS_PER_CASTLE}",
                bg=random.choice(BG_COLORS),
            )
            photos.append({"path": rel_path, "source": "placeholder"})
        castle["photos"] = photos

    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Generated {PHOTOS_PER_CASTLE} placeholder photos each for {len(castles)} castles.")


if __name__ == "__main__":
    main()
