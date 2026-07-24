"""
One-off helper: generates placeholder split-screen images (movie still + modern
photo) for every location in data/movies.json, so the app has something to show
before real images are dropped in.

Run once with:  .venv/bin/python scripts/generate_placeholders.py
Then replace files under assets/stills and assets/modern with real images
(same filenames) whenever you have them - real movie stills are copyrighted,
so ship your own or license-cleared photos for production use.
"""

import json
import pathlib
import random

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "movies.json"

STILL_COLORS = ["#2b2118", "#1c2b33", "#241c33", "#332318", "#182b22"]
MODERN_COLORS = ["#dfe7ec", "#e8e3d8", "#dce8dc", "#e8dcdc", "#d8e0e8"]


def draw_placeholder(path: pathlib.Path, label: str, subtitle: str, bg: str, fg: str) -> None:
    img = Image.new("RGB", (800, 450), bg)
    draw = ImageDraw.Draw(img)
    font_big = ImageFont.load_default(size=28)
    font_small = ImageFont.load_default(size=16)

    draw.rectangle([0, 0, 799, 449], outline=fg, width=3)
    draw.text((40, 190), label, fill=fg, font=font_big)
    draw.text((40, 230), subtitle, fill=fg, font=font_small)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=85)


def main() -> None:
    movies = json.loads(DATA_FILE.read_text())["movies"]
    for movie in movies:
        for loc in movie["locations"]:
            still_bg = random.choice(STILL_COLORS)
            modern_bg = random.choice(MODERN_COLORS)

            draw_placeholder(
                ROOT / loc["still"],
                f"{movie['title']} ({movie['year']})",
                f"PLACEHOLDER STILL - {loc['name']}",
                bg=still_bg,
                fg="#f2e9d8",
            )
            draw_placeholder(
                ROOT / loc["modern"],
                "Present day",
                f"PLACEHOLDER PHOTO - {loc['name']}",
                bg=modern_bg,
                fg="#2b2b2b",
            )

    print("Placeholder images generated under assets/stills and assets/modern.")


if __name__ == "__main__":
    main()
