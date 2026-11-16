"""
Builds public/sky/sky.json, the real star map behind khushikhan.com.

Source: d3-celestial data by Olaf Frohn (BSD-3-Clause), from the Hipparcos /
Yale Bright Star catalogues and IAU constellation figures. Its licence is copied
to public/sky/LICENSE-d3-celestial.txt alongside the data.

Output (angles in degrees, RA in -180..180 as in the source):
  stars: [[ra, dec, mag, bv], ...]              all stars to magnitude 6
  names: [[ra, dec, mag, "en", "ja"], ...]      proper names of the brightest stars
  lines: [[[[ra, dec], ...], ...], ...]         constellation stick figures
  cons:  [[ra, dec, rank, "en", "ja"], ...]     constellation label positions

Run from projects/:   python3 scripts/build_sky_data.py
"""

import json
import pathlib
import urllib.request

VERSION = "0.7.35"
BASE = f"https://cdn.jsdelivr.net/npm/d3-celestial@{VERSION}"
OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "public" / "sky"
NAME_MAG_LIMIT = 1.6   # label stars at least this bright


def get(path: str):
    with urllib.request.urlopen(f"{BASE}/{path}", timeout=30) as r:
        return r.read()


def main() -> None:
    stars = json.loads(get("data/stars.6.json"))["features"]
    starnames = json.loads(get("data/starnames.json"))
    lines = json.loads(get("data/constellations.lines.json"))["features"]
    cons = json.loads(get("data/constellations.json"))["features"]
    licence = get("LICENSE").decode()

    r2 = lambda v: round(v, 2)

    out_stars, out_names = [], []
    for f in stars:
        ra, dec = f["geometry"]["coordinates"]
        mag = f["properties"]["mag"]
        try:
            bv = round(float(f["properties"].get("bv") or 0.6), 2)
        except ValueError:
            bv = 0.6
        out_stars.append([r2(ra), r2(dec), round(mag, 2), bv])
        name = starnames.get(str(f["id"]), {})
        if mag <= NAME_MAG_LIMIT and name.get("name"):
            out_names.append([r2(ra), r2(dec), round(mag, 2), name["name"], name.get("ja") or name["name"]])
    out_stars.sort(key=lambda s: s[2])   # brightest first

    out_lines = [[[[r2(p[0]), r2(p[1])] for p in seg] for seg in f["geometry"]["coordinates"]] for f in lines]

    out_cons = []
    for f in cons:
        p = f["properties"]
        ra, dec = f["geometry"]["coordinates"]
        out_cons.append([r2(ra), r2(dec), int(p.get("rank", 3)), p["name"], p.get("ja") or p["name"]])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "source": f"d3-celestial {VERSION} by Olaf Frohn, BSD-3-Clause (see LICENSE-d3-celestial.txt)",
        "stars": out_stars, "names": out_names, "lines": out_lines, "cons": out_cons,
    }
    (OUT_DIR / "sky.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT_DIR / "LICENSE-d3-celestial.txt").write_text(licence)
    print(f"{len(out_stars)} stars, {len(out_names)} named, {len(out_lines)} figures, {len(out_cons)} constellations"
          f" -> {OUT_DIR / 'sky.json'}")


if __name__ == "__main__":
    main()
