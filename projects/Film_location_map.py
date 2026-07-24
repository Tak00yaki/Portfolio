"""
Film Location Map
------------------
Search for a movie, see pins dropped at its real-world filming locations, and
click a pin to compare the movie still against a present-day photo of the
same spot.

Run with:  .venv/bin/streamlit run Film_location_map.py

Data lives in data/movies.json. Images live in assets/stills (movie stills)
and assets/modern (present-day photos) - the ones shipped here are generated
placeholders (see scripts/generate_placeholders.py). Swap in real images
under the same filenames to replace them; movie stills are copyrighted, so
use your own licensed/cleared photos for anything beyond local experimenting.
"""

import json
import pathlib

import folium
import streamlit as st
from streamlit_folium import st_folium

ROOT = pathlib.Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "movies.json"

MARKER_COLORS = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]


@st.cache_data
def load_movies() -> list[dict]:
    return json.loads(DATA_FILE.read_text())["movies"]


def build_map(movies: list[dict]) -> folium.Map:
    all_locs = [loc for movie in movies for loc in movie["locations"]]
    if not all_locs:
        center, zoom = (20, 0), 2
    elif len(all_locs) == 1:
        center, zoom = (all_locs[0]["lat"], all_locs[0]["lon"]), 6
    else:
        lats = [loc["lat"] for loc in all_locs]
        lons = [loc["lon"] for loc in all_locs]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))
        zoom = 2

    fmap = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    for i, movie in enumerate(movies):
        color = MARKER_COLORS[i % len(MARKER_COLORS)]
        for loc in movie["locations"]:
            tooltip_html = f"""
                <div style="max-width:220px; white-space:normal; line-height:1.3;">
                    <div style="font-weight:700; color:#c0392b;">{movie['title']} ({movie['year']})</div>
                    <div style="color:#2a6f97; font-style:italic; font-size:12px;">{loc['name']}</div>
                </div>
            """
            popup_html = f"""
                <div style="max-width:230px; white-space:normal; line-height:1.35;">
                    <div style="font-weight:700; color:#c0392b; margin-bottom:2px;">{movie['title']} ({movie['year']})</div>
                    <div style="color:#2a6f97; font-style:italic; font-size:12px; margin-bottom:6px;">{loc['name']}</div>
                    <div style="color:#333; font-size:12px;">{loc['scene']}</div>
                </div>
            """
            folium.Marker(
                location=(loc["lat"], loc["lon"]),
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=260),
                icon=folium.Icon(color=color, icon="film", prefix="fa"),
            ).add_to(fmap)

    if len(all_locs) > 1:
        fmap.fit_bounds([(loc["lat"], loc["lon"]) for loc in all_locs], padding=(30, 30))

    return fmap


def find_clicked_location(movies: list[dict], lat: float, lon: float, tolerance: float = 1e-4):
    for movie in movies:
        for loc in movie["locations"]:
            if abs(loc["lat"] - lat) < tolerance and abs(loc["lon"] - lon) < tolerance:
                return movie, loc
    return None, None


def main() -> None:
    st.set_page_config(page_title="Film Location Map", page_icon="🎬", layout="wide")
    st.title("🎬 Film Location Map")
    st.caption("Search a movie, click a pin, compare the scene to the spot today.")

    movies = load_movies()
    titles = [movie["title"] for movie in movies]

    selected_titles = st.sidebar.multiselect(
        "Search movies", options=titles, default=titles,
        help="Pick one or more movies to plot their filming locations.",
    )
    st.sidebar.caption(f"{len(selected_titles)} of {len(titles)} movies shown")

    selected_movies = [m for m in movies if m["title"] in selected_titles]

    if "selected_location" not in st.session_state:
        st.session_state.selected_location = None

    map_col, detail_col = st.columns([3, 2])

    with map_col:
        if not selected_movies:
            st.info("Select at least one movie in the sidebar to see its pins.")
        else:
            fmap = build_map(selected_movies)
            map_state = st_folium(fmap, height=520, use_container_width=True, key="film_map")

            clicked = map_state.get("last_object_clicked") if map_state else None
            if clicked:
                movie, loc = find_clicked_location(selected_movies, clicked["lat"], clicked["lng"])
                if loc:
                    st.session_state.selected_location = (movie["title"], movie["year"], loc)

    with detail_col:
        st.subheader("Then vs. Now")
        selection = st.session_state.selected_location
        if not selection:
            st.write("Click a pin on the map to reveal the scene comparison.")
        else:
            title, year, loc = selection
            st.markdown(f"**{title} ({year})** — {loc['name']}")
            st.caption(loc["scene"])
            st.image(str(ROOT / loc["still"]), caption="On screen", use_container_width=True)
            st.image(str(ROOT / loc["modern"]), caption="Present day", use_container_width=True)

            credit = loc.get("modern_credit")
            if loc.get("modern_source") in ("wikipedia", "wikimedia_commons") and credit:
                st.caption(
                    f"Present-day photo: {credit['author']} · {credit['license']} · "
                    f"[source]({credit['source_url']})"
                )
            elif loc.get("modern_source") == "placeholder":
                st.caption("Present-day photo: placeholder — no verified real photo found yet for this spot.")


if __name__ == "__main__":
    main()
