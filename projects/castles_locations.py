"""
Castle Map
----------
Browse real castles around the world on a map. Click a pin to see when it
was built, its architecture style, who built it, why it was built, and a
preview photo, all inside the map's info window. Filter/sort from the
dropdown by architecture style, era, the dynasty/empire each castle
belonged to, or who built it.

Run with:  .venv/bin/streamlit run castles_locations.py

Data lives in data/castles.json. Photos live in assets/castles/ - the ones
shipped here are generated placeholders (see scripts/generate_castle_placeholders.py).
Run scripts/fetch_castle_photos.py to replace them with real, freely-licensed
photos pulled from Wikipedia/Wikimedia Commons.
"""

import base64
import io
import json
import pathlib

import folium
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

ROOT = pathlib.Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "castles.json"

ERA_ORDER = [
    "5th–8th century",
    "9th–10th century",
    "11th–13th century",
    "14th–15th century",
    "16th–17th century",
    "18th–19th century",
    "20th century",
]

MARKER_COLORS = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]
POPUP_THUMB_SIZE = (240, 240)


@st.cache_data
def load_castles() -> list[dict]:
    return json.loads(DATA_FILE.read_text())["castles"]


@st.cache_data
def popup_thumbnail_b64(path_str: str, mtime: float) -> str | None:
    path = pathlib.Path(path_str)
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    img.thumbnail(POPUP_THUMB_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sorted_unique(castles: list[dict], key: str, order: list[str] | None = None) -> list[str]:
    values = {c[key] for c in castles}
    if order:
        return [v for v in order if v in values]
    return sorted(values)


def checkbox_filter(label: str, options: list[str], key_prefix: str) -> list[str]:
    """A dropdown button that opens a searchable checkbox list, all checked by default."""
    for option in options:
        st.session_state.setdefault(f"{key_prefix}_cb_{option}", True)

    selected_count = sum(1 for o in options if st.session_state.get(f"{key_prefix}_cb_{o}", True))

    with st.popover(f"{label} ({selected_count}/{len(options)})", use_container_width=True):
        search = st.text_input(
            "Search", key=f"{key_prefix}_search", placeholder=f"Search {label.lower()}…",
            label_visibility="collapsed",
        )
        col_a, col_b = st.columns(2)
        if col_a.button("Select all", key=f"{key_prefix}_selall", use_container_width=True):
            for option in options:
                st.session_state[f"{key_prefix}_cb_{option}"] = True
        if col_b.button("Clear", key=f"{key_prefix}_clearall", use_container_width=True):
            for option in options:
                st.session_state[f"{key_prefix}_cb_{option}"] = False

        st.divider()
        visible = [o for o in options if search.lower() in o.lower()] if search else options
        if not visible:
            st.caption("No matches.")
        for option in visible:
            st.checkbox(option, key=f"{key_prefix}_cb_{option}")

    return [o for o in options if st.session_state.get(f"{key_prefix}_cb_{o}", True)]


def year_label(castle: dict) -> str:
    start, end = castle["year_built_start"], castle["year_built_end"]
    return str(start) if start == end else f"{start}–{end}"


def build_popup_html(castle: dict) -> str:
    photos = castle.get("photos", [])[:1]
    img_html = ""
    for photo in photos:
        img_path = ROOT / photo["path"]
        if img_path.exists():
            b64 = popup_thumbnail_b64(str(img_path), img_path.stat().st_mtime)
            if b64:
                img_html += (
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    f'style="width:100%;height:140px;object-fit:cover;border-radius:6px;margin-bottom:8px;" />'
                )

    return f"""
        <div style="width:250px; font-family:-apple-system,Helvetica,Arial,sans-serif;">
            {img_html}
            <div style="font-weight:700; font-size:15px; color:#5b3a29; margin-bottom:1px;">{castle['name']}</div>
            <div style="color:#6b6b6b; font-style:italic; font-size:12px; margin-bottom:6px;">{castle['country']}</div>
            <div style="font-size:12px; line-height:1.6; color:#333;">
                <b>Built:</b> {year_label(castle)} ({castle['era']})<br>
                <b>Style:</b> {castle['architecture_style']}<br>
                <b>Built by:</b> {castle['built_by']}<br>
                <b>Dynasty / empire:</b> {castle['dynasty_empire']}<br>
                <b>Why:</b> {castle['purpose']}
            </div>
        </div>
    """


def build_map(castles: list[dict], color_by: str) -> folium.Map:
    if not castles:
        center, zoom = (30, 10), 2
    elif len(castles) == 1:
        center, zoom = (castles[0]["lat"], castles[0]["lon"]), 6
    else:
        lats = [c["lat"] for c in castles]
        lons = [c["lon"] for c in castles]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))
        zoom = 3

    fmap = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    color_keys = sorted({c[color_by] for c in castles})
    color_map = {key: MARKER_COLORS[i % len(MARKER_COLORS)] for i, key in enumerate(color_keys)}

    for castle in castles:
        tooltip_html = f"""
            <div style="max-width:220px; white-space:normal; line-height:1.3;">
                <div style="font-weight:700; color:#5b3a29;">{castle['name']}</div>
                <div style="color:#6b6b6b; font-style:italic; font-size:12px;">{castle['country']}</div>
            </div>
        """
        folium.Marker(
            location=(castle["lat"], castle["lon"]),
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
            popup=folium.Popup(build_popup_html(castle), max_width=280),
            icon=folium.Icon(color=color_map[castle[color_by]], icon="chess-rook", prefix="fa"),
        ).add_to(fmap)

    if len(castles) > 1:
        fmap.fit_bounds([(c["lat"], c["lon"]) for c in castles], padding=(30, 30))

    return fmap


def find_clicked_castle(castles: list[dict], lat: float, lon: float, tolerance: float = 1e-4):
    for castle in castles:
        if abs(castle["lat"] - lat) < tolerance and abs(castle["lon"] - lon) < tolerance:
            return castle
    return None


def render_gallery(castle: dict) -> None:
    photos = castle.get("photos", [])
    if not photos:
        return
    cols = st.columns(len(photos))
    for col, photo in zip(cols, photos):
        img_path = ROOT / photo["path"]
        if img_path.exists():
            col.image(str(img_path), use_container_width=True)
        credit = photo.get("credit")
        if photo.get("source") == "wikipedia" and credit:
            col.caption(f"{credit['author']} · {credit['license']}")
        elif photo.get("source") == "placeholder":
            col.caption("Placeholder — run scripts/fetch_castle_photos.py")


def main() -> None:
    st.set_page_config(page_title="Castle Map", page_icon="🏰", layout="wide")
    st.title("🏰 Castle Map")
    st.caption("Explore real castles around the world. Click a pin for its history, architecture, and a preview photo.")

    castles = load_castles()

    styles = sorted_unique(castles, "architecture_style")
    eras = sorted_unique(castles, "era", order=ERA_ORDER)
    dynasties = sorted_unique(castles, "dynasty_empire")

    style_col, era_col, dynasty_col, color_col, sort_col = st.columns(5)
    with style_col:
        selected_styles = checkbox_filter("Architecture style", styles, "style")
    with era_col:
        selected_eras = checkbox_filter("Era of build", eras, "era")
    with dynasty_col:
        selected_dynasties = checkbox_filter("Dynasty / empire", dynasties, "dynasty")
    with color_col:
        color_by_label = st.selectbox(
            "Color pins by",
            options=["Architecture style", "Era of build", "Dynasty / empire"],
        )
    with sort_col:
        sort_option = st.selectbox(
            "Sort castle list by",
            options=["Name", "Year built", "Architecture style", "Dynasty / empire", "Built by"],
        )

    color_by_key = {
        "Architecture style": "architecture_style",
        "Era of build": "era",
        "Dynasty / empire": "dynasty_empire",
    }[color_by_label]

    sort_key = {
        "Name": lambda c: c["name"],
        "Year built": lambda c: c["year_built_start"],
        "Architecture style": lambda c: c["architecture_style"],
        "Dynasty / empire": lambda c: c["dynasty_empire"],
        "Built by": lambda c: c["built_by"],
    }[sort_option]

    filtered = [
        c for c in castles
        if c["architecture_style"] in selected_styles
        and c["era"] in selected_eras
        and c["dynasty_empire"] in selected_dynasties
    ]
    filtered_sorted = sorted(filtered, key=sort_key)

    st.caption(f"{len(filtered)} of {len(castles)} castles shown · sorted by {sort_option.lower()}")

    if "selected_castle_slug" not in st.session_state:
        st.session_state.selected_castle_slug = None

    if not filtered:
        st.info("No castles match the current filters. Widen your selection in Filters & sort.")
    else:
        fmap = build_map(filtered, color_by_key)
        map_state = st_folium(fmap, height=720, use_container_width=True, key="castle_map")

        clicked = map_state.get("last_object_clicked") if map_state else None
        if clicked:
            castle = find_clicked_castle(filtered, clicked["lat"], clicked["lng"])
            if castle:
                st.session_state.selected_castle_slug = castle["slug"]

    st.divider()

    pick_col, gallery_header_col = st.columns([2, 3])
    with pick_col:
        names = [c["name"] for c in filtered_sorted]
        current = next((c["name"] for c in filtered_sorted if c["slug"] == st.session_state.selected_castle_slug), None)
        picked_name = st.selectbox(
            "Or pick a castle to view its photo gallery",
            options=names,
            index=names.index(current) if current in names else None,
            placeholder="Choose a castle…",
        )
        if picked_name:
            picked = next(c for c in filtered_sorted if c["name"] == picked_name)
            st.session_state.selected_castle_slug = picked["slug"]

    slug = st.session_state.selected_castle_slug
    selected = next((c for c in castles if c["slug"] == slug), None)

    if selected:
        st.subheader(f"📸 {selected['name']} — photo gallery")
        st.write(selected["description"])
        render_gallery(selected)


if __name__ == "__main__":
    main()
