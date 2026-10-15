"""Week 2: fit the Week 1 race model to real data (2024 Bahrain GP) with FastF1.

1. Download the race and keep clean green-flag laps.
2. Fit lap time = driver offset + fuel/track term * lap number + k0[compound]
   + k1[compound] * tyre age (k0 of soft fixed at 0, as in Paper 1).
3. Estimate pit loss and start-tyre ages from the same race.
4. Run the brute-force optimiser with the fitted numbers and compare its best
   strategy with what the teams actually did.

Figures are written to figures/.
"""

import json
import os
import time
import urllib.parse
import urllib.request
from collections import Counter
from itertools import combinations, product

import fastf1
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "fastf1_cache")
FIG_DIR = os.path.join(HERE, "figures")

YEAR, EVENT = 2024, "Bahrain"
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MIN_STOPS, MAX_STOPS = 1, 3
OUTLIER_PCT = 1.03       # drop laps slower than 103% of the driver's median clean lap

# Compound colours: categorical slots 1-3 of the validated palette (always labelled too)
COLOURS = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


# --- 1. Data -----------------------------------------------------------------
# Both loaders return the same two tables:
#   laps:    Driver, LapNumber, LapTimeS, Compound, Stint, Age, InLap, OutLap, Green
#   results: Driver, Position (finishing order)
# Tyre age is counted from 0 on a tyre's first lap (Week 1 convention), and
# includes laps the tyre already did in qualifying.

def load_fastf1():
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.set_log_level("ERROR")
    session = fastf1.get_session(YEAR, EVENT, "R")
    session.load(telemetry=False, weather=False, messages=False)
    raw = session.laps
    laps = pd.DataFrame({
        "Driver": raw["Driver"],
        "LapNumber": raw["LapNumber"].astype(int),
        "LapTimeS": raw["LapTime"].dt.total_seconds(),
        "Compound": raw["Compound"],
        "Stint": raw["Stint"],
        "Age": raw["TyreLife"] - 1,          # FastF1's TyreLife is 1 on a new tyre's first lap
        "InLap": raw["PitInTime"].notna(),
        "OutLap": raw["PitOutTime"].notna(),
        "Green": raw["TrackStatus"] == "1",
    })
    results = session.results.rename(columns={"Abbreviation": "Driver"})[["Driver", "Position"]]
    return laps, results


def openf1_get(endpoint, session_key=None, **params):
    """GET an OpenF1 endpoint, cached as JSON next to the FastF1 cache."""
    params = {"session_key": session_key, **params} if session_key else params
    key = "_".join(f"{k}-{v}" for k, v in params.items())
    path = os.path.join(CACHE_DIR, "openf1", f"{endpoint}_{key}.json")
    if not os.path.exists(path):
        url = f"https://api.openf1.org/v1/{endpoint}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        time.sleep(0.5)                       # stay well under OpenF1's rate limit
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def green_laps(race_control, race_laps):
    """Laps run under green: not under SC / VSC / red flag (from race control)."""
    green, state = {}, True
    events = {n: m for n, m in race_control.groupby("lap_number")["message"]}
    for n in range(1, race_laps + 1):
        lap_green = state
        for msg in events.get(n, []):
            msg = (msg or "").upper()
            if "SAFETY CAR DEPLOYED" in msg or "RED FLAG" in msg:
                state, lap_green = False, False
            elif "ENDING" in msg or "IN THIS LAP" in msg or "TRACK CLEAR" in msg:
                state = True                  # racing resumes from the next lap
        green[n] = lap_green
    return green


def load_openf1():
    session = openf1_get("sessions", year=YEAR, country_name=EVENT, session_name="Race")
    key = int(session["session_key"].iloc[0])
    drivers = openf1_get("drivers", key).set_index("driver_number")["name_acronym"]
    raw = openf1_get("laps", key)
    stints = openf1_get("stints", key)
    result = openf1_get("session_result", key)
    race_laps = int(raw["lap_number"].max())
    green = green_laps(openf1_get("race_control", key), race_laps)

    rows = []
    for _, st in stints.iterrows():
        last_stint = st["stint_number"] == stints.loc[
            stints["driver_number"] == st["driver_number"], "stint_number"].max()
        for n in range(int(st["lap_start"]), int(st["lap_end"]) + 1):
            rows.append({
                "driver_number": st["driver_number"], "LapNumber": n,
                "Compound": st["compound"], "Stint": st["stint_number"],
                "Age": st["tyre_age_at_start"] + n - st["lap_start"],
                "InLap": n == st["lap_end"] and not last_stint,
                "OutLap": n == st["lap_start"] and st["stint_number"] > 1,
            })
    laps = pd.DataFrame(rows).merge(
        raw[["driver_number", "lap_number", "lap_duration"]].rename(
            columns={"lap_number": "LapNumber", "lap_duration": "LapTimeS"}),
        on=["driver_number", "LapNumber"], how="inner")
    laps["Driver"] = laps["driver_number"].map(drivers)
    laps["Green"] = laps["LapNumber"].map(green)
    results = pd.DataFrame({"Driver": result["driver_number"].map(drivers),
                            "Position": result["position"]})
    return laps.drop(columns="driver_number"), results


def load_race():
    """FastF1 first; fall back to OpenF1 if the F1 live-timing server refuses us."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        laps, results = load_fastf1()
        source = "FastF1"
    except Exception as e:                    # FastF1 raises DataNotLoadedError on a blocked download
        print(f"FastF1 download failed ({type(e).__name__}); using OpenF1 instead")
        laps, results = load_openf1()
        source = "OpenF1"
    laps["Position"] = laps["Driver"].map(results.set_index("Driver")["Position"])
    return laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True), results, source


def clean_laps(laps):
    """Green-flag racing laps only: no lap 1, no in/out laps, no SC/VSC/red flag."""
    ok = (
        laps["LapTimeS"].notna()
        & laps["Compound"].isin(DRY_COMPOUNDS)
        & (laps["LapNumber"] > 1)
        & ~laps["InLap"]
        & ~laps["OutLap"]
        & laps["Green"].astype(bool)
        & laps["Age"].notna()
    )
    clean = laps[ok].copy()
    median = clean.groupby("Driver")["LapTimeS"].transform("median")
    return clean[clean["LapTimeS"] < OUTLIER_PCT * median]


# --- 2. Fit k0 / k1 ----------------------------------------------------------

def fit_tyre_model(clean):
    """Least squares on all clean laps at once.

    Columns: one offset per driver, lap number (fuel burn + track evolution),
    k0 for medium and hard (soft is the reference, k0 = 0), k1 per compound.
    """
    drivers = sorted(clean["Driver"].unique())
    compounds = [c for c in DRY_COMPOUNDS if c in set(clean["Compound"])]
    cols = [f"drv_{d}" for d in drivers] + ["lap"] \
        + [f"k0_{c}" for c in compounds[1:]] + [f"k1_{c}" for c in compounds]

    X = np.zeros((len(clean), len(cols)))
    idx = {c: i for i, c in enumerate(cols)}
    for row, (_, lap) in enumerate(clean.iterrows()):
        X[row, idx[f"drv_{lap['Driver']}"]] = 1
        X[row, idx["lap"]] = lap["LapNumber"]
        if lap["Compound"] != compounds[0]:
            X[row, idx[f"k0_{lap['Compound']}"]] = 1
        X[row, idx[f"k1_{lap['Compound']}"]] = lap["Age"]
    y = clean["LapTimeS"].to_numpy()

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef = dict(zip(cols, beta))
    residuals = y - X @ beta
    # Standard errors: sigma^2 (X'X)^-1, to see which parameters the data pins down
    sigma2 = residuals @ residuals / (len(y) - len(cols))
    se = dict(zip(cols, np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))))

    params = {c: (coef.get(f"k0_{c}", 0.0), coef[f"k1_{c}"]) for c in compounds}
    driver_offsets = {d: coef[f"drv_{d}"] for d in drivers}
    param_se = {c: (se.get(f"k0_{c}", 0.0), se[f"k1_{c}"]) for c in compounds}
    return params, param_se, coef["lap"], se["lap"], driver_offsets, residuals


# --- 3. Pit loss and start ages ---------------------------------------------

def estimate_pit_loss(laps, clean):
    """(in-lap + out-lap) - 2 x driver's median clean lap, green-flag stops only."""
    median = clean.groupby("Driver")["LapTimeS"].median()
    losses = []
    for drv, d in laps.groupby("Driver"):
        d = d.set_index("LapNumber")
        for n in d.index[d["InLap"]]:
            if n + 1 not in d.index or drv not in median:
                continue
            in_lap, out_lap = d.loc[n], d.loc[n + 1]
            if not (in_lap["Green"] and out_lap["Green"]):
                continue
            total = in_lap["LapTimeS"] + out_lap["LapTimeS"]
            if np.isfinite(total):
                losses.append(total - 2 * median[drv])
    return float(np.median(losses)), losses


def start_ages(laps, drivers):
    """Median tyre age at the start among `drivers`, per compound (0 if nobody started on it).

    Cars that reached Q3 start on used qualifying tyres, the rest often on new ones,
    so this is taken over the drivers whose strategies we compare against.
    """
    first = laps[(laps["LapNumber"] == 1) & laps["Driver"].isin(drivers)]
    ages = first.groupby("Compound")["Age"].median()
    return {c: int(ages.get(c, 0)) for c in DRY_COMPOUNDS}


# --- 4. Optimiser (Week 1, generalised) --------------------------------------

class RaceModel:
    def __init__(self, params, base_lap, pit_loss, race_laps, start_age):
        self.params, self.base_lap, self.pit_loss = params, base_lap, pit_loss
        self.race_laps, self.start_age = race_laps, start_age

    def stint_time(self, compound, laps, start_age=0):
        k0, k1 = self.params[compound]
        return laps * (self.base_lap + k0) + k1 * (laps * start_age + laps * (laps - 1) / 2)

    def race_time(self, compounds, pit_laps):
        bounds = (0, *pit_laps, self.race_laps)
        total = self.pit_loss * len(pit_laps)
        for i, c in enumerate(compounds):
            age = self.start_age[c] if i == 0 else 0
            total += self.stint_time(c, bounds[i + 1] - bounds[i], age)
        return total

    def best_strategy(self, n_stops, start_compound=None):
        """Every compound sequence (>= 2 compounds) x every pit-lap set.

        `start_compound` fixes the first tyre; None lets the optimiser choose it.
        """
        best = None
        for compounds in product(self.params, repeat=n_stops + 1):
            if len(set(compounds)) < 2 or (start_compound and compounds[0] != start_compound):
                continue
            for pit_laps in combinations(range(1, self.race_laps), n_stops):
                t = self.race_time(compounds, pit_laps)
                if best is None or t < best[0]:
                    best = (t, compounds, pit_laps)
        return best


def describe(compounds, pit_laps, race_laps):
    bounds = (0, *pit_laps, race_laps)
    return " -> ".join(f"{c[0]}{bounds[i + 1] - bounds[i]}" for i, c in enumerate(compounds))


def actual_strategies(results, laps, race_laps):
    """Each finisher's stints as (compounds, pit laps), in finishing order."""
    out = []
    for _, r in results.dropna(subset=["Position"]).sort_values("Position").iterrows():
        d = laps[laps["Driver"] == r["Driver"]].sort_values("LapNumber")
        if d["LapNumber"].max() != race_laps:
            continue  # lapped or retired: not comparable over the full distance
        stints = d.groupby("Stint").agg(compound=("Compound", "first"),
                                        last=("LapNumber", "max"))
        out.append((int(r["Position"]), r["Driver"],
                    tuple(stints["compound"]), tuple(stints["last"].iloc[:-1].astype(int))))
    return out


# --- Plots -------------------------------------------------------------------

def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_driver_stints(laps, race_laps, path):
    """Each driver's raw lap times across the race, coloured by compound."""
    drivers = laps.groupby("Driver")["Position"].first().sort_values().index
    fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharex=True, sharey=True)
    for ax, drv in zip(axes.flat, drivers):
        d = laps[laps["Driver"] == drv]
        for _, stint in d.groupby("Stint"):
            c = stint["Compound"].iloc[0]
            ax.plot(stint["LapNumber"], stint["LapTimeS"], "-o", color=COLOURS.get(c, MUTED),
                    linewidth=2, markersize=3)
        ax.set_title(drv, fontsize=10, color=INK, loc="left")
        style(ax)
    lo = laps["LapTimeS"].quantile(0.01)
    axes.flat[0].set_ylim(lo - 0.5, lo + 7)
    axes.flat[0].set_xlim(0, race_laps + 1)
    handles = [plt.Line2D([], [], color=COLOURS[c], linewidth=2, label=c.title())
               for c in DRY_COMPOUNDS if c in set(laps["Compound"])]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3)
    fig.suptitle(f"{YEAR} {EVENT} GP: lap time by stint (s)", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.supxlabel("Lap", color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tyre_fit(clean, params, fuel, offsets, path):
    """Lap time minus driver offset and fuel term, against tyre age, per compound."""
    compounds = list(params)
    fig, axes = plt.subplots(1, len(compounds), figsize=(5 * len(compounds), 4.5), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), compounds):
        d = clean[clean["Compound"] == c]
        loss = d["LapTimeS"] - d["Driver"].map(offsets) - fuel * d["LapNumber"]
        ax.scatter(d["Age"], loss, s=10, color=COLOURS[c], alpha=0.35, linewidths=0)
        k0, k1 = params[c]
        age = np.linspace(0, d["Age"].max(), 50)
        ax.plot(age, k0 + k1 * age, color=COLOURS[c], linewidth=2)
        ax.set_title(f"{c.title()}   k0 = {k0:.2f} s,  k1 = {k1:.3f} s/lap",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Tyre age (laps)", color=MUTED)
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Tyre loss (s, relative to new soft)", color=MUTED)
    fig.suptitle("Fitted tyre model: points are fuel- and driver-corrected laps",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_strategies(rows, race_laps, path):
    """Gantt of stints: model optima on top, then real finishers."""
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(rows) + 1.5))
    for y, (label, compounds, pit_laps, gap) in enumerate(rows):
        bounds = (0, *pit_laps, race_laps)
        for i, c in enumerate(compounds):
            ax.barh(y, bounds[i + 1] - bounds[i], left=bounds[i], height=0.7,
                    color=COLOURS[c], edgecolor="white", linewidth=2)
            ax.text((bounds[i] + bounds[i + 1]) / 2, y, c[0], ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        ax.text(race_laps + 0.8, y, gap, va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, race_laps + 9)
    ax.set_xlabel("Lap", color=MUTED)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title("Model-optimal vs actual strategies (right: model time lost vs best plan)",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    laps, results, source = load_race()
    race_laps = int(laps["LapNumber"].max())
    clean = clean_laps(laps)
    print(f"{YEAR} {EVENT} GP ({source}): {race_laps} laps, "
          f"{len(clean)} clean laps of {len(laps)}")

    params, param_se, fuel, fuel_se, offsets, residuals = fit_tyre_model(clean)
    print(f"\nFitted model, +/- 1 standard error (residual std {residuals.std():.3f} s)")
    print(f"  fuel + track evolution: {fuel:+.4f} +/- {fuel_se:.4f} s per lap")
    for c, (k0, k1) in params.items():
        n = (clean["Compound"] == c).sum()
        k0_se, k1_se = param_se[c]
        k0_txt = f"{k0_se:.3f}" if k0_se else "fixed"
        print(f"  {c:<6}  k0 = {k0:6.3f} +/- {k0_txt:<5} s   "
              f"k1 = {k1:6.4f} +/- {k1_se:.4f} s/lap   ({n} laps)")
    missing = [c for c in DRY_COMPOUNDS if c not in params]
    if missing:
        print(f"  not used in this race, so not fitted: {', '.join(missing)}")

    actual = actual_strategies(results, laps, race_laps)
    pit_loss, losses = estimate_pit_loss(laps, clean)
    ages = start_ages(laps, [drv for _, drv, _, _ in actual])
    print(f"\nPit loss: {pit_loss:.2f} s (median of {len(losses)} green-flag stops)")
    print(f"Start tyre age (median): {ages}")

    # Effective base: average driver, fuel term averaged over the race (it is the
    # same for every strategy, so it only shifts the totals).
    base_lap = np.mean(list(offsets.values())) + fuel * (race_laps + 1) / 2
    model = RaceModel(params, base_lap, pit_loss, race_laps, ages)

    print("\nModel optimum per number of stops:")
    best = {}
    for n in range(MIN_STOPS, MAX_STOPS + 1):
        best[n] = model.best_strategy(n)
        t, comps, pits = best[n]
        print(f"  {n}-stop: {describe(comps, pits, race_laps):<22} {t:9.2f} s")
    overall = min(best.values())

    # Every car started on softs, so also find the best plan under that constraint.
    start = Counter(c[0] for _, _, c, _ in actual).most_common(1)[0][0]
    best_fixed = min(model.best_strategy(n, start) for n in range(MIN_STOPS, MAX_STOPS + 1))
    t, comps, pits = best_fixed
    print(f"  best starting on {start.lower()} (as the teams did): "
          f"{describe(comps, pits, race_laps)}  {t:9.2f} s  (+{t - overall[0]:.1f} s)")

    print("\nWhat the teams did (model time vs model optimum):")
    rows = [(f"Model {n}-stop", b[1], b[2], f"+{b[0] - overall[0]:.1f} s")
            for n, b in best.items()]
    rows.append((f"Model, {start.lower()} start", comps, pits, f"+{t - overall[0]:.1f} s"))
    for pos, drv, comps, pits in actual:
        if not set(comps) <= set(params):
            continue
        gap = model.race_time(comps, pits) - overall[0]
        print(f"  P{pos:<2} {drv}  {describe(comps, pits, race_laps):<22} +{gap:5.1f} s")
        rows.append((f"P{pos} {drv}", comps, pits, f"+{gap:.1f} s"))

    shapes = Counter(len(c) - 1 for _, _, c, _ in actual)
    print(f"\nStops used by full-distance finishers: {dict(sorted(shapes.items()))}")

    plot_driver_stints(laps, race_laps, os.path.join(FIG_DIR, "week2_driver_stints.png"))
    plot_tyre_fit(clean, params, fuel, offsets, os.path.join(FIG_DIR, "week2_tyre_fit.png"))
    plot_strategies(rows, race_laps, os.path.join(FIG_DIR, "week2_strategies.png"))
    print(f"\nFigures saved to {FIG_DIR}")
