"""
Fetch Award Winners
--------------------
One-off data prep script: resolves a hardcoded list of Academy Award Best
Picture winners to their TMDB ids and writes data/award_winners.json for
movies_rec.py to read at runtime - same "curate once, app reads a static
file" pattern used by scripts/fetch_modern_photos.py for Film_location_map.py.

Requires TMDB_API_KEY in .streamlit/secrets.toml (or the TMDB_API_KEY env var).

Run with:  .venv/bin/python scripts/fetch_award_winners.py

Note: the title/year list below was compiled from memory, not copy-pasted
from a source - worth a quick sanity check against Wikipedia's "Academy
Award for Best Picture" list before you rely on it for anything important.
Extending it with other categories (Best Director, BAFTA, etc.) just means
adding more (title, year, category) tuples below.
"""

import json
import os
import pathlib
import re
import time
import tomllib

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "award_winners.json"
CATEGORY = "Academy Award for Best Picture"

# (title, release year)
BEST_PICTURE_WINNERS = [
    ("Wings", 1927),
    ("The Broadway Melody", 1928),
    ("All Quiet on the Western Front", 1929),
    ("Cimarron", 1930),
    ("Grand Hotel", 1931),
    ("Cavalcade", 1932),
    ("It Happened One Night", 1934),
    ("Mutiny on the Bounty", 1935),
    ("The Great Ziegfeld", 1936),
    ("The Life of Emile Zola", 1937),
    ("You Can't Take It with You", 1938),
    ("Gone with the Wind", 1939),
    ("Rebecca", 1940),
    ("How Green Was My Valley", 1941),
    ("Mrs. Miniver", 1942),
    ("Casablanca", 1943),
    ("Going My Way", 1944),
    ("The Lost Weekend", 1945),
    ("The Best Years of Our Lives", 1946),
    ("Gentleman's Agreement", 1947),
    ("Hamlet", 1948),
    ("All the King's Men", 1949),
    ("All About Eve", 1950),
    ("An American in Paris", 1951),
    ("The Greatest Show on Earth", 1952),
    ("From Here to Eternity", 1953),
    ("On the Waterfront", 1954),
    ("Marty", 1955),
    ("Around the World in 80 Days", 1956),
    ("The Bridge on the River Kwai", 1957),
    ("Gigi", 1958),
    ("Ben-Hur", 1959),
    ("The Apartment", 1960),
    ("West Side Story", 1961),
    ("Lawrence of Arabia", 1962),
    ("Tom Jones", 1963),
    ("My Fair Lady", 1964),
    ("The Sound of Music", 1965),
    ("A Man for All Seasons", 1966),
    ("In the Heat of the Night", 1967),
    ("Oliver!", 1968),
    ("Midnight Cowboy", 1969),
    ("Patton", 1970),
    ("The French Connection", 1971),
    ("The Godfather", 1972),
    ("The Sting", 1973),
    ("The Godfather Part II", 1974),
    ("One Flew Over the Cuckoo's Nest", 1975),
    ("Rocky", 1976),
    ("Annie Hall", 1977),
    ("The Deer Hunter", 1978),
    ("Kramer vs. Kramer", 1979),
    ("Ordinary People", 1980),
    ("Chariots of Fire", 1981),
    ("Gandhi", 1982),
    ("Terms of Endearment", 1983),
    ("Amadeus", 1984),
    ("Out of Africa", 1985),
    ("Platoon", 1986),
    ("The Last Emperor", 1987),
    ("Rain Man", 1988),
    ("Driving Miss Daisy", 1989),
    ("Dances with Wolves", 1990),
    ("The Silence of the Lambs", 1991),
    ("Unforgiven", 1992),
    ("Schindler's List", 1993),
    ("Forrest Gump", 1994),
    ("Braveheart", 1995),
    ("The English Patient", 1996),
    ("Titanic", 1997),
    ("Shakespeare in Love", 1998),
    ("American Beauty", 1999),
    ("Gladiator", 2000),
    ("A Beautiful Mind", 2001),
    ("Chicago", 2002),
    ("The Lord of the Rings: The Return of the King", 2003),
    ("Million Dollar Baby", 2004),
    ("Crash", 2005),
    ("The Departed", 2006),
    ("No Country for Old Men", 2007),
    ("Slumdog Millionaire", 2008),
    ("The Hurt Locker", 2009),
    ("The King's Speech", 2010),
    ("The Artist", 2011),
    ("Argo", 2012),
    ("12 Years a Slave", 2013),
    ("Birdman", 2014),
    ("Spotlight", 2015),
    ("Moonlight", 2016),
    ("The Shape of Water", 2017),
    ("Green Book", 2018),
    ("Parasite", 2019),
    ("Nomadland", 2020),
    ("CODA", 2021),
    ("Everything Everywhere All at Once", 2022),
    ("Oppenheimer", 2023),
    ("Anora", 2024),
]


def get_api_key() -> str:
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as f:
            data = tomllib.load(f)
        if data.get("TMDB_API_KEY"):
            return data["TMDB_API_KEY"]

    key = os.environ.get("TMDB_API_KEY")
    if not key:
        raise SystemExit(
            "TMDB_API_KEY not found. Add it to .streamlit/secrets.toml "
            "or export it as an environment variable before running this script."
        )
    return key


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def search(api_key: str, title: str, year: int | None) -> list[dict]:
    params = {"api_key": api_key, "query": title}
    if year:
        params["year"] = year
    resp = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=10)
    resp.raise_for_status()
    time.sleep(0.05)
    return resp.json().get("results", [])


def resolve_movie(api_key: str, title: str, year: int) -> dict | None:
    """The most-voted film whose title starts with `title`, released that year or the next
    (early winners often went on general release a year later), else the first result.
    Most-voted, because obscure films share titles ("Birdman", "Grand Hotel")."""
    wanted = _norm(title)
    candidates = search(api_key, title, year) + search(api_key, title, year + 1)
    matching = [r for r in candidates if _norm(r.get("title", "")).startswith(wanted)]
    if matching:
        return max(matching, key=lambda r: r.get("vote_count", 0))
    if candidates:
        return candidates[0]
    results = search(api_key, title, None)
    return results[0] if results else None


def main() -> None:
    api_key = get_api_key()
    winners = []
    for title, year in BEST_PICTURE_WINNERS:
        movie = resolve_movie(api_key, title, year)
        if movie is None:
            print(f"  ! no TMDB match for {title} ({year}) - skipping")
            continue
        winners.append({
            "title": title,
            "year": year,
            "tmdb_id": movie["id"],
            "poster_path": movie.get("poster_path"),
            "release_date": movie.get("release_date") or "",
            "category": CATEGORY,
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"winners": winners}, indent=2))
    print(f"Wrote {len(winners)}/{len(BEST_PICTURE_WINNERS)} winners to {OUT_FILE}")


if __name__ == "__main__":
    main()
