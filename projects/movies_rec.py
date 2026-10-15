"""
Movie & Music Recommender
--------------------------
A Pinterest-style feed of movie posters. It starts random (popular, top rated,
Best Picture winners); once you open a film or save it to Favourites / Watchlist,
the feed mixes in more from the same director, the top-billed cast, the author
of the book it is based on, and TMDB's own recommendations.

Each film's page shows where to watch it (pick a country; Japan by default),
the trailer, the director and cast, Wikipedia context for films about real
events, and the official soundtrack on Spotify.

Owner mode (passcode = OWNER_PASSCODE in secrets.toml) unlocks the lists and
"Open in Spotify". The lists are kept in data/my_lists.json on this computer
(gitignored), so they only exist where you run the app yourself. Visitors get
the feed and the Spotify embed player (30-second previews unless they are
logged in to Spotify themselves).

Data sources (all live, not pre-curated - see data/award_winners.json for the
one exception):
  - TMDB API        search, discover, credits, recommendations, trailers,
                    where to watch (provider data by JustWatch)
  - Wikipedia API   historical context for period pieces, no key needed
  - Spotify API     soundtrack album / playlist search, Client Credentials flow

Requires TMDB_API_KEY, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET (and optionally
OWNER_PASSCODE) in .streamlit/secrets.toml - see .streamlit/secrets.toml.example.

Run with:  .venv/bin/streamlit run movies_rec.py
"""

import hmac
import json
import pathlib
import random
import re
import time
from urllib.parse import quote

import requests
import streamlit as st
import streamlit.components.v1 as components

ROOT = pathlib.Path(__file__).resolve().parent
AWARD_WINNERS_FILE = ROOT / "data" / "award_winners.json"
MY_LISTS_FILE = ROOT / "data" / "my_lists.json"

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p"
WIKI_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
# Wikipedia refuses requests without a descriptive User-Agent
WIKI_HEADERS = {"User-Agent": "KKMovieRecommender/1.0 (https://khushikhan.com)"}
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API = "https://api.spotify.com/v1"

DEFAULT_REGION = "JP"
GRID_COLS = 5
PAGE_SIZE = 20                 # cards per "Load more"
SEEDS_USED = 6                 # most recent saved / opened films that steer the feed
PERSONAL_SHARE = 0.7           # share of the feed that is personal once there are seeds

LISTS = {"favourites": "♥ Favourites", "watchlist": "＋ Watchlist"}

HISTORY_GENRE_IDS = {36, 10752}  # History, War
HISTORY_KEYWORDS = {"based on a true story", "true story", "biography", "biopic"}
# TMDB crew jobs that credit the writer of the source book
AUTHOR_JOBS = {"Novel", "Author", "Book", "Short Story", "Comic Book", "Graphic Novel"}
# role -> (cast credits?, crew jobs, heading) for a person's page
ROLES = {
    "director": (False, {"Director"}, "Directed by"),
    "author": (False, AUTHOR_JOBS, "Based on the writing of"),
    "actor": (True, None, "Starring"),
}

# Poster cards: an invisible button stretched over the poster makes it clickable,
# and the list buttons are small circles.
CSS = """
<style>
div[class*="st-key-pcard-"] { position: relative; }
div[class*="st-key-pcard-"]:hover img { filter: brightness(0.85); }
div[class*="st-key-pclick-"] { position: absolute; inset: 0; z-index: 2; }
div[class*="st-key-pclick-"] div, div[class*="st-key-pclick-"] button { width: 100%; height: 100%; }
div[class*="st-key-pclick-"] button { opacity: 0; cursor: pointer; }
div[class*="st-key-listbtn-"] button {
    width: 2.25rem; height: 2.25rem; min-height: 0; padding: 0; border-radius: 50%;
}
div[class*="st-key-person-"] button p { text-decoration: underline dotted; }
</style>
"""


class MissingAPIKey(Exception):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


# --------------------------------------------------------------------------- #
# Secrets / low-level API wrappers
# --------------------------------------------------------------------------- #

def get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or None


def require_secret(name: str) -> str:
    value = get_secret(name)
    if not value:
        raise MissingAPIKey(name)
    return value


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def tmdb_get(path: str, **params) -> dict:
    params["api_key"] = require_secret("TMDB_API_KEY")
    resp = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=86400, show_spinner=False)
def wikipedia_search(query: str, limit: int = 3) -> list[dict]:
    resp = requests.get(WIKI_SEARCH_URL, params={"q": query, "limit": limit}, headers=WIKI_HEADERS, timeout=10)
    resp.raise_for_status()
    pages = resp.json().get("pages", [])
    return [
        {
            "title": p["title"],
            "excerpt": re.sub("<[^>]+>", "", p.get("excerpt", "")),
            "url": f"https://en.wikipedia.org/wiki/{quote(p['key'])}",
        }
        for p in pages
    ]


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def wikipedia_url(wikidata_id: str | None, title: str, year: str) -> str | None:
    """The film's English Wikipedia page, via its Wikidata item, else the top search hit."""
    if wikidata_id:
        resp = requests.get(
            WIKIDATA_API,
            params={"action": "wbgetentities", "ids": wikidata_id, "props": "sitelinks/urls",
                    "sitefilter": "enwiki", "format": "json"},
            headers=WIKI_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        link = resp.json().get("entities", {}).get(wikidata_id, {}).get("sitelinks", {}).get("enwiki")
        if link:
            return link["url"]
    hits = wikipedia_search(f"{title} {year} film".strip(), limit=1)
    return hits[0]["url"] if hits else None


@st.cache_data(ttl=3000, show_spinner=False)   # Spotify tokens last an hour
def spotify_token() -> str:
    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(require_secret("SPOTIFY_CLIENT_ID"), require_secret("SPOTIFY_CLIENT_SECRET")),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def spotify_search(query: str, kind: str, limit: int = 10) -> list[dict]:
    resp = requests.get(
        f"{SPOTIFY_API}/search",
        params={"q": query, "type": kind, "limit": limit},
        headers={"Authorization": f"Bearer {spotify_token()}"},
        timeout=10,
    )
    resp.raise_for_status()
    return [item for item in resp.json().get(f"{kind}s", {}).get("items", []) if item]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@st.cache_data(ttl=86400, show_spinner=False)
def spotify_soundtrack(title: str, year: str) -> dict | None:
    """The official soundtrack album if Spotify has one, else the best playlist."""
    wanted = _norm(title)
    best, best_score = None, 0
    for album in spotify_search(f"{title} original motion picture soundtrack", "album"):
        name = _norm(album["name"])
        if f" {wanted} " not in f" {name} ":
            continue
        score = 1
        if any(w in name for w in ("soundtrack", "motion picture", "original score", "music from")):
            score += 2
        album_year = (album.get("release_date") or "")[:4]
        if year and album_year and abs(int(album_year) - int(year)) <= 1:
            score += 2
        if score > best_score:
            best, best_score = album, score
    if best and best_score >= 3:
        return {
            "kind": "album",
            "id": best["id"],
            "name": best["name"],
            "by": ", ".join(a["name"] for a in best.get("artists", [])),
            "url": best["external_urls"]["spotify"],
        }

    playlists = spotify_search(f"{title} soundtrack", "playlist", limit=5)
    if not playlists:
        return None
    p = playlists[0]
    return {
        "kind": "playlist",
        "id": p["id"],
        "name": p["name"],
        "by": p["owner"]["display_name"],
        "url": p["external_urls"]["spotify"],
    }


# --------------------------------------------------------------------------- #
# TMDB queries
# --------------------------------------------------------------------------- #

def tmdb_image(path: str | None, size: str = "w342") -> str | None:
    return f"{TMDB_IMG_BASE}/{size}{path}" if path else None


def search_movies(query: str) -> list[dict]:
    return tmdb_get("/search/movie", query=query, include_adult=False).get("results", [])


def get_movie_details(movie_id: int) -> dict:
    return tmdb_get(
        f"/movie/{movie_id}",
        append_to_response="credits,keywords,videos,recommendations,watch/providers,external_ids",
    )


def get_person_bio(person_id: int) -> dict:
    return tmdb_get(f"/person/{person_id}")


def person_films(person_id: int, as_cast: bool, jobs: set[str] | None = None) -> list[dict]:
    credits = tmdb_get(f"/person/{person_id}/movie_credits")
    if as_cast:
        films = credits.get("cast", [])
    else:
        films = [c for c in credits.get("crew", []) if c.get("job") in jobs]
    films = [f for f in films if f.get("poster_path") and not f.get("adult")]
    films = list({f["id"]: f for f in films}.values())   # one card per film, even with several roles
    return sorted(films, key=lambda f: f.get("popularity", 0), reverse=True)


@st.cache_data(ttl=86400, show_spinner=False)
def get_regions() -> dict[str, str]:
    data = tmdb_get("/watch/providers/regions")
    return {r["iso_3166_1"]: r["english_name"] for r in data.get("results", [])}


def load_award_winners() -> list[dict]:
    # Not cached: the file is small, and a cache would hide a freshly fetched file until restart
    if not AWARD_WINNERS_FILE.exists():
        return []
    return json.loads(AWARD_WINNERS_FILE.read_text())["winners"]


def winner_card(w: dict, reason: str) -> dict:
    """Card straight from award_winners.json; older files without posters fall back to TMDB."""
    if not w.get("poster_path"):
        return card(get_movie_details(w["tmdb_id"]), reason)
    return {
        "id": w["tmdb_id"],
        "title": w["title"],
        "release_date": w.get("release_date") or str(w.get("year", "")),
        "poster_path": w["poster_path"],
        "reason": reason,
    }


def is_period_piece(details: dict) -> bool:
    genre_ids = {g["id"] for g in details.get("genres", [])}
    if genre_ids & HISTORY_GENRE_IDS:
        return True
    keywords = {k["name"].lower() for k in details.get("keywords", {}).get("keywords", [])}
    return any(hk in kw for kw in keywords for hk in HISTORY_KEYWORDS)


def authors_of(details: dict) -> list[dict]:
    crew = details.get("credits", {}).get("crew", [])
    seen, authors = set(), []
    for c in crew:
        if c.get("job") in AUTHOR_JOBS and c["id"] not in seen:
            seen.add(c["id"])
            authors.append(c)
    return authors


def card(movie: dict, reason: str) -> dict:
    """The few fields a poster card needs (also what gets saved to the lists)."""
    return {
        "id": movie["id"],
        "title": movie.get("title") or movie.get("name", ""),
        "release_date": movie.get("release_date") or "",
        "poster_path": movie.get("poster_path"),
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Feed
# --------------------------------------------------------------------------- #

def personal_sources(seed_id: int) -> list[list[dict]]:
    """Lists of cards that follow from one seed film, one list per reason."""
    details = get_movie_details(seed_id)
    title = details["title"]
    credits = details.get("credits", {})
    sources = []

    for d in [c for c in credits.get("crew", []) if c.get("job") == "Director"][:2]:
        films = person_films(d["id"], as_cast=False, jobs={"Director"})
        sources.append([card(f, f"Directed by {d['name']} · because of {title}") for f in films])

    for a in authors_of(details)[:2]:
        films = person_films(a["id"], as_cast=False, jobs=AUTHOR_JOBS)
        sources.append([card(f, f"Based on {a['name']}'s writing · because of {title}") for f in films])

    for actor in credits.get("cast", [])[:3]:
        films = person_films(actor["id"], as_cast=True)
        sources.append([card(f, f"With {actor['name']} · because of {title}") for f in films])

    recs = [m for m in details.get("recommendations", {}).get("results", []) if m.get("poster_path")]
    sources.append([card(m, f"If you liked {title}") for m in recs])
    return sources


def random_pages(rng: random.Random):
    """Endless (endpoint, params, reason) pages of discover results, in a random order."""
    while True:
        roll = rng.random()
        if roll < 0.5:
            yield "/discover/movie", {"sort_by": "popularity.desc", "page": rng.randint(1, 25)}, "Popular"
        else:
            yield (
                "/discover/movie",
                {"sort_by": "vote_average.desc", "vote_count.gte": 1000, "page": rng.randint(1, 25)},
                "Top rated",
            )


def build_feed(seeds: tuple[int, ...], rng_seed: int, count: int) -> list[dict]:
    rng = random.Random(rng_seed)
    exclude = set(seeds)
    feed, seen = [], set()

    def take(c: dict) -> bool:
        if c["id"] in seen or c["id"] in exclude or not c.get("poster_path"):
            return False
        seen.add(c["id"])
        feed.append(c)
        return True

    # Personal cards: round-robin across every reason so no one actor floods the feed
    queues = []
    for seed_id in seeds[-SEEDS_USED:][::-1]:
        try:
            queues.extend(iter(s) for s in personal_sources(seed_id))
        except requests.RequestException:
            continue

    def next_personal() -> bool:
        while queues:
            q = queues.pop(0)
            for c in q:
                if take(c):
                    queues.append(q)
                    return True
        return False

    # Random cards: Best Picture winners sprinkled into discover pages
    winners = load_award_winners()[:]
    rng.shuffle(winners)
    pages = random_pages(rng)
    random_buffer: list[dict] = []

    def next_random() -> bool:
        for _ in range(10):
            if not random_buffer:
                if winners and rng.random() < 0.15:
                    w = winners.pop()
                    try:
                        random_buffer.append(winner_card(w, "Best Picture winner"))
                    except requests.RequestException:
                        pass
                    continue
                path, params, reason = next(pages)
                results = tmdb_get(path, include_adult=False, **params).get("results", [])
                rng.shuffle(results)
                random_buffer.extend(card(m, reason) for m in results)
            while random_buffer:
                if take(random_buffer.pop()):
                    return True
        return False

    while len(feed) < count:
        want_personal = queues and rng.random() < PERSONAL_SHARE
        if not ((want_personal and next_personal()) or next_random() or next_personal()):
            break
    return feed


# --------------------------------------------------------------------------- #
# Owner mode and lists (local file, gitignored)
# --------------------------------------------------------------------------- #

def is_owner() -> bool:
    return st.session_state.get("owner", False)


def load_lists() -> dict:
    if "lists" not in st.session_state:
        data = {"favourites": {}, "watchlist": {}, "region": DEFAULT_REGION}
        if MY_LISTS_FILE.exists():
            try:
                data.update(json.loads(MY_LISTS_FILE.read_text()))
            except json.JSONDecodeError:
                st.warning(f"Couldn't read {MY_LISTS_FILE.name}; starting with empty lists.")
        st.session_state.lists = data
    return st.session_state.lists


def save_lists() -> None:
    tmp = MY_LISTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st.session_state.lists, indent=2, ensure_ascii=False))
    tmp.replace(MY_LISTS_FILE)


def toggle_in_list(name: str, c: dict) -> None:
    items = load_lists()[name]
    key = str(c["id"])
    if key in items:
        del items[key]
    else:
        items[key] = {**c, "added": time.time()}
    save_lists()


def seeds() -> tuple[int, ...]:
    """Films that steer the feed, oldest first: saved films (owner) and films opened this visit."""
    ids: list[int] = []
    if is_owner():
        saved = [c for name in LISTS for c in load_lists()[name].values()]
        ids += [c["id"] for c in sorted(saved, key=lambda c: c.get("added", 0))]
    ids += st.session_state.get("opened", [])
    out: list[int] = []
    for i in ids:
        if i in out:
            out.remove(i)
        out.append(i)
    return tuple(out)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def nav() -> list[dict]:
    """Pages opened on top of the feed, newest last: {"movie": id} or {"person": id, "role": role}."""
    return st.session_state.setdefault("nav", [])


def open_movie(movie_id: int) -> None:
    nav().append({"movie": movie_id})
    opened = st.session_state.setdefault("opened", [])
    if movie_id in opened:
        opened.remove(movie_id)
    opened.append(movie_id)


def open_person(person_id: int, role: str) -> None:
    nav().append({"person": person_id, "role": role})


def back_buttons() -> None:
    with st.container(horizontal=True, gap="small"):
        if st.button("← Back"):
            nav().pop()
            st.rerun()
        if len(nav()) > 1 and st.button("Feed"):
            nav().clear()
            st.rerun()


def person_button(person: dict, role: str, label: str, key: str) -> None:
    """A person's name as a link-style button that opens the films they made in that role."""
    if st.button(label, key=f"person-{key}-{role}-{person['id']}", type="tertiary",
                 help=f"More films with {person['name']}"):
        open_person(person["id"], role)
        st.rerun()


def list_buttons(c: dict, key: str) -> None:
    lists = load_lists()
    with st.container(horizontal=True, gap="small"):
        for name in LISTS:
            on = str(c["id"]) in lists[name]
            if st.button(
                LISTS[name].split()[0],
                key=f"listbtn-{key}-{name}",
                type="primary" if on else "secondary",
                help=("Remove from " if on else "Add to ") + LISTS[name].split(maxsplit=1)[1],
            ):
                toggle_in_list(name, c)
                st.rerun()


def render_grid(cards: list[dict], key: str) -> None:
    for start in range(0, len(cards), GRID_COLS):
        for col, c in zip(st.columns(GRID_COLS), cards[start:start + GRID_COLS]):
            with col, st.container(border=True):
                with st.container(key=f"pcard-{key}-{c['id']}"):
                    st.image(tmdb_image(c["poster_path"]), width="stretch")
                    if st.button(f"Open {c['title']}", key=f"pclick-{key}-{c['id']}"):
                        open_movie(c["id"])
                        st.rerun()
                year = c["release_date"][:4] or "TBA"
                st.markdown(f"**{c['title']}** ({year})")
                if c.get("reason"):
                    st.caption(c["reason"])
                if is_owner():
                    list_buttons(c, f"{key}-{c['id']}")


def render_person(person_id: int, role: str, link: bool = True) -> None:
    person = get_person_bio(person_id)
    cols = st.columns([1, 4])
    with cols[0]:
        photo = tmdb_image(person.get("profile_path"), "w185")
        if photo:
            st.image(photo, width="stretch")
    with cols[1]:
        label = f"**{person['name']}** — {role.capitalize()}"
        if link:
            person_button(person, role, label, "bio")
        else:
            st.markdown(label)
        bio = person.get("biography") or "No biography available."
        st.caption(bio[:400] + ("…" if len(bio) > 400 else ""))


def render_where_to_watch(details: dict, region: str, region_name: str) -> None:
    st.subheader(f"Where to watch · {region_name}")
    offers = details.get("watch/providers", {}).get("results", {}).get(region)
    if not offers:
        st.caption(f"No streaming, rental or purchase options listed for {region_name}.")
        return
    for kind, label in (("flatrate", "Stream"), ("rent", "Rent"), ("buy", "Buy")):
        providers = offers.get(kind, [])
        if providers:
            st.markdown(f"**{label}:** " + " · ".join(p["provider_name"] for p in providers))
    st.link_button("See where to watch", offers["link"])
    st.caption("Where-to-watch data by JustWatch.")


def render_trailer(details: dict) -> None:
    videos = details.get("videos", {}).get("results", [])
    trailers = [v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer"]
    trailers.sort(key=lambda v: not v.get("official"))
    if trailers:
        st.subheader("Trailer")
        st.video(f"https://www.youtube.com/watch?v={trailers[0]['key']}")


def render_soundtrack(details: dict, year: str) -> None:
    st.subheader("Soundtrack")
    try:
        soundtrack = spotify_soundtrack(details["title"], year)
    except MissingAPIKey:
        st.caption("Add Spotify API keys to see the soundtrack (see .streamlit/secrets.toml.example).")
        return
    except requests.RequestException as exc:
        st.caption(f"Couldn't reach Spotify: {exc}")
        return
    if not soundtrack:
        st.caption("No soundtrack found on Spotify.")
        return
    kind = "Official soundtrack" if soundtrack["kind"] == "album" else "Soundtrack playlist"
    st.caption(f"{kind}: {soundtrack['name']} · {soundtrack['by']}")
    if is_owner():
        st.link_button("Open in Spotify", soundtrack["url"], type="primary")
    components.iframe(
        f"https://open.spotify.com/embed/{soundtrack['kind']}/{soundtrack['id']}", height=352
    )


def render_detail(movie_id: int, region: str, region_name: str) -> None:
    back_buttons()
    details = get_movie_details(movie_id)
    year = (details.get("release_date") or "")[:4]

    poster_col, info_col = st.columns([1, 2])
    with poster_col:
        poster = tmdb_image(details.get("poster_path"), "w500")
        if poster:
            st.image(poster, width="stretch")
    with info_col:
        st.header(f"{details['title']} ({year or 'TBA'})")
        facts = [g["name"] for g in details.get("genres", [])]
        if details.get("runtime"):
            facts.append(f"{details['runtime']} min")
        st.caption(" · ".join(facts))
        authors = authors_of(details)
        if authors:
            with st.container(horizontal=True, gap="small", vertical_alignment="center"):
                st.caption("Based on the work of", width="content")
                for a in authors:
                    person_button(a, "author", a["name"], "detail")
        st.write(details.get("overview", ""))
        rating = details.get("vote_average")
        if rating:
            st.caption(f"⭐ {rating:.1f}/10 · {details.get('vote_count', 0)} votes")
        try:
            wiki = wikipedia_url(details.get("external_ids", {}).get("wikidata_id"), details["title"], year)
        except requests.RequestException:
            wiki = None
        if wiki:
            st.link_button("Wikipedia", wiki)
        if is_owner():
            list_buttons(card(details, ""), f"detail_{movie_id}")
        render_where_to_watch(details, region, region_name)

    render_trailer(details)
    render_soundtrack(details, year)

    if is_period_piece(details):
        st.subheader("Historical Context")
        try:
            hits = wikipedia_search(f"{details['title']} {year}".strip())
        except requests.RequestException:
            hits = []
        if hits:
            for hit in hits:
                st.markdown(f"**[{hit['title']}]({hit['url']})**")
                st.caption(hit["excerpt"])
        else:
            st.caption("No related Wikipedia articles found.")

    st.subheader("Director & Cast")
    credits = details.get("credits", {})
    for director in [c for c in credits.get("crew", []) if c.get("job") == "Director"]:
        render_person(director["id"], role="director")

    cast = credits.get("cast", [])[:6]
    if cast:
        st.caption("Top cast")
        for col, actor in zip(st.columns(len(cast)), cast):
            with col:
                photo = tmdb_image(actor.get("profile_path"), "w185")
                if photo:
                    st.image(photo, width="stretch")
                person_button(actor, "actor", actor["name"], "cast")
                if actor.get("character"):
                    st.caption(f"as {actor['character']}")


def render_person_page(person_id: int, role: str) -> None:
    back_buttons()
    render_person(person_id, role, link=False)
    as_cast, jobs, heading = ROLES[role]
    films = person_films(person_id, as_cast=as_cast, jobs=jobs)
    st.subheader(f"{heading} {get_person_bio(person_id)['name']}")
    if not films:
        st.info("No other films found.")
        return
    count_key = f"person_count_{person_id}_{role}"
    count = st.session_state.setdefault(count_key, PAGE_SIZE)
    render_grid([card(f, "") for f in films[:count]], f"person{person_id}")
    if count < len(films) and st.button("Load more films"):
        st.session_state[count_key] = count + PAGE_SIZE
        st.rerun()


def render_feed() -> None:
    query = st.text_input("Search", placeholder="Search movies by title", label_visibility="collapsed")
    if query:
        results = [m for m in search_movies(query) if m.get("poster_path")]
        if not results:
            st.info("No movies found.")
        render_grid([card(m, "") for m in results], "search")
        return

    count = st.session_state.setdefault("feed_count", PAGE_SIZE)
    rng_seed = st.session_state.setdefault("feed_seed", random.randrange(1 << 30))
    current = seeds()
    if current:
        st.caption(f"Tuned to {len(current[-SEEDS_USED:])} film(s) you saved or opened.")
    with st.spinner("Finding films…"):
        feed = build_feed(current, rng_seed, count)
    render_grid(feed, "feed")
    left, right = st.columns(2)
    if left.button("Load more", width="stretch"):
        st.session_state.feed_count = count + PAGE_SIZE
        st.rerun()
    if right.button("Shuffle", width="stretch"):
        st.session_state.feed_seed = random.randrange(1 << 30)
        st.session_state.feed_count = PAGE_SIZE
        st.rerun()


def render_list(name: str) -> None:
    items = sorted(load_lists()[name].values(), key=lambda c: c.get("added", 0), reverse=True)
    if not items:
        st.info(f"Nothing in your {LISTS[name].split(maxsplit=1)[1]} yet.")
        return
    render_grid([{**c, "reason": ""} for c in items], name)


def render_awards() -> None:
    winners = load_award_winners()
    if not winners:
        st.info(
            "No award winners loaded yet — run "
            "`.venv/bin/python scripts/fetch_award_winners.py` once your TMDB key is set up."
        )
        return
    count = st.session_state.setdefault("awards_count", PAGE_SIZE)
    cards = [winner_card(w, w.get("category", "")) for w in winners[:count]]
    render_grid(cards, "awards")
    if count < len(winners) and st.button("Load more winners"):
        st.session_state.awards_count = count + PAGE_SIZE
        st.rerun()


def render_sidebar() -> tuple[str, str]:
    regions = get_regions()
    lists = load_lists()
    codes = sorted(regions, key=regions.get)
    current = lists.get("region", DEFAULT_REGION)
    region = st.sidebar.selectbox(
        "Where to watch: country",
        codes,
        index=codes.index(current) if current in codes else 0,
        format_func=regions.get,
    )
    if region != current:
        lists["region"] = region
        if is_owner():
            save_lists()

    passcode = get_secret("OWNER_PASSCODE")
    if passcode:
        if is_owner():
            if st.sidebar.button("Lock owner mode"):
                st.session_state.owner = False
                st.rerun()
        else:
            typed = st.sidebar.text_input("Owner passcode", type="password")
            if typed:
                if hmac.compare_digest(typed, passcode):
                    st.session_state.owner = True
                    st.rerun()
                st.sidebar.error("Wrong passcode.")

    st.sidebar.divider()
    st.sidebar.caption(
        "This product uses the TMDB API but is not endorsed or certified by TMDB. "
        "[themoviedb.org](https://www.themoviedb.org)"
    )
    return region, regions.get(region, region)


def main() -> None:
    st.set_page_config(page_title="Movie & Music Recommender", page_icon="🎬", layout="wide")
    st.html(CSS)
    st.title("🎬 Movie & Music Recommender")

    try:
        region, region_name = render_sidebar()
    except MissingAPIKey:
        st.error(
            "Missing TMDB_API_KEY. Copy `.streamlit/secrets.toml.example` to "
            "`.streamlit/secrets.toml`, fill in your key, and restart the app."
        )
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Couldn't reach TMDB: {exc}")
        st.stop()

    if nav():
        page = nav()[-1]
        if "movie" in page:
            render_detail(page["movie"], region, region_name)
        else:
            render_person_page(page["person"], page["role"])
        return

    names = list(LISTS) if is_owner() else []
    tabs = st.tabs(["For you", "Award winners", *(LISTS[n] for n in names)])
    with tabs[0]:
        render_feed()
    with tabs[1]:
        render_awards()
    for tab, name in zip(tabs[2:], names):
        with tab:
            render_list(name)


if __name__ == "__main__":
    main()
