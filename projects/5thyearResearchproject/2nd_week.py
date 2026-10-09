"""Week 2: fit the Week 1 race model to real data (2024 Bahrain GP) with FastF1.

1. Download the race and keep clean green-flag laps.
2. Fit lap time = driver offset + fuel/track term * lap number + k0[compound]
   + k1[compound] * tyre age (k0 of soft fixed at 0, as in Paper 1).
3. Estimate pit loss and start-tyre ages from the same race.
4. Run the brute-force optimiser with the fitted numbers and compare its best
   strategy with what the teams actually did.
5. Robustness: refit without early-race laps and on free-air laps only (is the
   hard-vs-soft pace result caused by traffic?), test the model on drivers it
   was not fitted on, and fit on the 2023 race to predict 2024.

Main fit: free-air laps only (gap to the car ahead >= FREE_AIR_GAP), with the
all-laps fit reported as a sensitivity check. Free-air laps are not a random
sample: many come from the leaders or just after a stop, on fresh tyres.

Not modelled yet: the limited number of tyre sets per compound (the optimiser
may use as many new sets as it likes). Needed from Week 3.

Figures are written to figures/.
"""

import json
import os
import time
import urllib.parse
import urllib.request
from collections import Counter
from functools import lru_cache
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
COMPARE_YEAR = 2023      # cross-year test: fit on this race, predict YEAR
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MIN_STOPS, MAX_STOPS = 1, 3
OUTLIER_PCT = 1.03       # drop laps slower than 103% of the driver's median clean lap
FREE_AIR_GAP = 2.0       # s to the car ahead for a lap to count as "free air"
MIN_COMPOUND_LAPS = 30   # compounds with fewer free-air laps are not fitted
SKIP_EARLY_LAPS = (3, 5) # refit without the first N race laps (traffic test)
N_SPLITS = 50            # random half/half driver splits for the out-of-sample test
N_BOOT = 200             # bootstrap resamples (drivers, with replacement) for the regret interval

# Compound colours: categorical slots 1-3 of the validated palette (always labelled too)
COLOURS = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


# --- 1. Data -----------------------------------------------------------------
# Both loaders return the same two tables:
#   laps:    Driver, LapNumber, LapTimeS, Compound, Stint, Age, InLap, OutLap, Green
#   results: Driver, Position (finishing order)
# Tyre age is counted from 0 on a tyre's first lap (Week 1 convention), and
# includes laps the tyre already did in qualifying.

def load_fastf1(year):
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.set_log_level("ERROR")
    session = fastf1.get_session(year, EVENT, "R")
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


def openf1_session_key(year):
    session = openf1_get("sessions", year=year, country_name=EVENT, session_name="Race")
    return int(session["session_key"].iloc[0])


def load_openf1(year):
    key = openf1_session_key(year)
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


def load_race(year):
    """FastF1 first; fall back to OpenF1 if the F1 live-timing server refuses us."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        laps, results = load_fastf1(year)
        source = "FastF1"
    except Exception as e:                    # FastF1 raises DataNotLoadedError on a blocked download
        print(f"FastF1 download failed ({type(e).__name__}); using OpenF1 instead")
        laps, results = load_openf1(year)
        source = "OpenF1"
    laps["Position"] = laps["Driver"].map(results.set_index("Driver")["Position"])
    return laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True), results, source


def add_free_air(laps, year):
    """Smallest gap to the car ahead during each lap, from OpenF1's intervals.

    Lap start times also come from OpenF1 (FastF1 only has them with telemetry
    loaded), so this works whichever source the laps came from. The race leader
    has no car ahead, so its gap is infinite. Laps with no interval data get NaN
    and are never counted as free air.
    """
    key = openf1_session_key(year)
    iv = openf1_get("intervals", key)
    numbers = openf1_get("drivers", key).set_index("name_acronym")["driver_number"]
    iv["date"] = pd.to_datetime(iv["date"], utc=True, format="ISO8601")
    iv["gap"] = pd.to_numeric(iv["interval"], errors="coerce")
    iv.loc[iv["gap_to_leader"] == 0, "gap"] = np.inf

    starts = openf1_get("laps", key).dropna(subset=["date_start"]).rename(
        columns={"lap_number": "LapNumber", "date_start": "LapStart"})
    starts["LapStart"] = pd.to_datetime(starts["LapStart"], utc=True, format="ISO8601")
    starts = starts[["driver_number", "LapNumber", "LapStart"]]
    # Assign each interval sample to the lap that was running at that moment
    tagged = pd.merge_asof(iv.sort_values("date"), starts.sort_values("LapStart"),
                           left_on="date", right_on="LapStart", by="driver_number")
    gap = tagged.groupby(["driver_number", "LapNumber"])["gap"].min()
    key = list(zip(laps["Driver"].map(numbers), laps["LapNumber"]))
    laps["GapAhead"] = [gap.get(k, np.nan) for k in key]
    return laps


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

def design_matrix(clean, tyres=True):
    """Columns: one offset per driver, lap number (fuel burn + track evolution),
    and if `tyres`: k0 for medium and hard (soft is the reference, k0 = 0), k1 per compound.
    """
    drivers = sorted(clean["Driver"].unique())
    compounds = [c for c in DRY_COMPOUNDS if c in set(clean["Compound"])]
    cols = [f"drv_{d}" for d in drivers] + ["lap"]
    if tyres:
        cols += [f"k0_{c}" for c in compounds[1:]] + [f"k1_{c}" for c in compounds]

    X = np.zeros((len(clean), len(cols)))
    idx = {c: i for i, c in enumerate(cols)}
    for row, (_, lap) in enumerate(clean.iterrows()):
        X[row, idx[f"drv_{lap['Driver']}"]] = 1
        X[row, idx["lap"]] = lap["LapNumber"]
        if tyres:
            if lap["Compound"] != compounds[0]:
                X[row, idx[f"k0_{lap['Compound']}"]] = 1
            X[row, idx[f"k1_{lap['Compound']}"]] = lap["Age"]
    return X, cols, drivers, compounds


def fit_tyre_model(clean):
    """Least squares on all clean laps at once (columns: see design_matrix)."""
    X, cols, drivers, compounds = design_matrix(clean)
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

    def race_time(self, compounds, pit_laps, ages=None):
        """`ages`: tyre age at the start of each stint. Default: the start age
        for the first stint, new tyres at every stop (what the optimiser assumes)."""
        bounds = (0, *pit_laps, self.race_laps)
        if ages is None:
            ages = [self.start_age[compounds[0]]] + [0] * len(pit_laps)
        total = self.pit_loss * len(pit_laps)
        for i, c in enumerate(compounds):
            total += self.stint_time(c, bounds[i + 1] - bounds[i], ages[i])
        return total

    def best_strategy(self, n_stops, start_compound=None):
        """Every compound sequence (>= 2 compounds) x every pit-lap set.

        `start_compound` fixes the first tyre; None lets the optimiser choose it.
        All pit-lap sets for one compound sequence are scored at once with numpy
        (stint_time works on arrays of stint lengths).
        """
        pit_sets = pit_lap_sets(self.race_laps, n_stops)
        bounds = np.column_stack([np.zeros(len(pit_sets), int), pit_sets,
                                  np.full(len(pit_sets), self.race_laps)])
        lengths = np.diff(bounds, axis=1)
        best = None
        for compounds in product(self.params, repeat=n_stops + 1):
            if len(set(compounds)) < 2 or (start_compound and compounds[0] != start_compound):
                continue
            total = self.pit_loss * n_stops + sum(
                self.stint_time(c, lengths[:, i], self.start_age[c] if i == 0 else 0)
                for i, c in enumerate(compounds))
            i = int(np.argmin(total))
            if best is None or total[i] < best[0]:
                best = (float(total[i]), compounds, tuple(int(x) for x in pit_sets[i]))
        return best

    def best_plan(self, start_compound=None):
        """Fastest plan over MIN_STOPS..MAX_STOPS stops."""
        return min(self.best_strategy(n, start_compound) for n in range(MIN_STOPS, MAX_STOPS + 1))

    def with_(self, **changes):
        """Copy of this model with some attributes replaced."""
        kw = dict(params=self.params, base_lap=self.base_lap, pit_loss=self.pit_loss,
                  race_laps=self.race_laps, start_age=self.start_age)
        return RaceModel(**{**kw, **changes})


@lru_cache(maxsize=None)
def pit_lap_sets(race_laps, n_stops):
    """All pit-lap combinations as an (n_sets, n_stops) array, in itertools order."""
    return np.array(list(combinations(range(1, race_laps), n_stops)), dtype=int)


def describe(compounds, pit_laps, race_laps):
    bounds = (0, *pit_laps, race_laps)
    return " -> ".join(f"{c[0]}{bounds[i + 1] - bounds[i]}" for i, c in enumerate(compounds))


def actual_strategies(results, laps, race_laps):
    """Each finisher's stints as (compounds, pit laps, tyre age at each stint start),
    in finishing order. Some sets fitted at a stop were already used in practice."""
    out = []
    for _, r in results.dropna(subset=["Position"]).sort_values("Position").iterrows():
        d = laps[laps["Driver"] == r["Driver"]].sort_values("LapNumber")
        if d["LapNumber"].max() != race_laps:
            continue  # lapped or retired: not comparable over the full distance
        stints = d.groupby("Stint").agg(compound=("Compound", "first"),
                                        last=("LapNumber", "max"), age=("Age", "first"))
        out.append((int(r["Position"]), r["Driver"],
                    tuple(stints["compound"]), tuple(stints["last"].iloc[:-1].astype(int)),
                    tuple(stints["age"].astype(int))))
    return out


# --- 5. Robustness checks ---------------------------------------------------

def traffic_tests(clean):
    """Refit on subsets that remove traffic, to see whether hard k0 moves."""
    subsets = {"all clean laps": clean}
    for n in SKIP_EARLY_LAPS:
        subsets[f"drop first {n} laps"] = clean[clean["LapNumber"] > n]
    free = clean["GapAhead"] >= FREE_AIR_GAP
    subsets[f"free air (gap >= {FREE_AIR_GAP:g} s)"] = clean[free]
    subsets[f"free air + drop first {SKIP_EARLY_LAPS[-1]}"] = \
        clean[free & (clean["LapNumber"] > SKIP_EARLY_LAPS[-1])]

    rows = []
    for name, d in subsets.items():
        params, se, fuel, _, _, res = fit_tyre_model(d)
        row = {"subset": name, "laps": len(d)}
        for c in params:
            row[f"n {c.lower()}"] = int((d["Compound"] == c).sum())
        row.update({"fuel+track": fuel, "resid std": res.std()})
        for c in params:
            if c != DRY_COMPOUNDS[0]:
                row[f"k0 {c.lower()}"] = f"{params[c][0]:+.3f} ± {se[c][0]:.3f}"
            row[f"k1 {c.lower()}"] = f"{params[c][1]:.4f} ± {se[c][1]:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("subset")


def out_of_sample(clean, seed=0):
    """Fit on a random half of the drivers, predict the other half's lap times.

    A held-out driver's own pace is unknown to the model, so each gets one
    offset (the mean of its residuals); everything else comes from the training
    half. The baseline model has driver offset + lap trend but no tyre terms.
    """
    rng = np.random.default_rng(seed)
    drivers = np.array(sorted(clean["Driver"].unique()))
    out = {"tyre model": [], "no tyre terms": [], "tyre model, in-sample": []}
    for _ in range(N_SPLITS):
        test = set(rng.choice(drivers, len(drivers) // 2, replace=False))
        tr = clean[~clean["Driver"].isin(test)]
        te = clean[clean["Driver"].isin(test)]
        for name, tyres in (("tyre model", True), ("no tyre terms", False)):
            X, cols, _, _ = design_matrix(tr, tyres)
            beta, *_ = np.linalg.lstsq(X, tr["LapTimeS"].to_numpy(), rcond=None)
            coef = dict(zip(cols, beta))
            if tyres:
                out["tyre model, in-sample"].append(
                    np.sqrt(np.mean((tr["LapTimeS"].to_numpy() - X @ beta) ** 2)))
            shape = coef["lap"] * te["LapNumber"]
            if tyres:
                shape = shape + te["Compound"].map(lambda c: coef.get(f"k0_{c}", 0.0)) \
                    + te["Compound"].map(lambda c: coef.get(f"k1_{c}", np.nan)) * te["Age"]
            resid = te["LapTimeS"] - shape
            resid = resid - resid.groupby(te["Driver"]).transform("mean")
            out[name].append(np.sqrt(np.nanmean(resid ** 2)))
    return {k: (np.mean(v), np.std(v)) for k, v in out.items()}


def cross_year_test(old_fit, new_fit, old_model, new_model, start):
    """Fit on one year's race, use it on another: a small version of the research question.

    Lap times: the old model's tyre and lap-trend terms predict the new race's
    free-air laps; each driver again gets one offset for their own pace.
    Strategy: the old model's best plan is scored in the new model (taken as the
    truth), and the time lost vs the new model's own best plan is the "regret".
    Only compounds fitted in both years are used.
    """
    common = [c for c in old_model.params if c in new_model.params]
    old_p, _, old_fuel, _, _, _ = fit_tyre_model(old_fit)
    new_p, _, new_fuel, _, _, new_res = fit_tyre_model(new_fit)
    te = new_fit[new_fit["Compound"].isin(common)]

    def rmse(params, fuel, tyres=True):
        shape = fuel * te["LapNumber"]
        if tyres:
            shape = shape + te["Compound"].map(lambda c: params[c][0]) \
                + te["Compound"].map(lambda c: params[c][1]) * te["Age"]
        resid = te["LapTimeS"] - shape
        resid = resid - resid.groupby(te["Driver"]).transform("mean")
        return float(np.sqrt(np.mean(resid ** 2)))

    X, cols, _, _ = design_matrix(te, tyres=False)
    beta, *_ = np.linalg.lstsq(X, te["LapTimeS"].to_numpy(), rcond=None)
    errors = {
        f"{COMPARE_YEAR} model on {YEAR} laps": rmse(old_p, old_fuel),
        f"{YEAR} model on {YEAR} laps (in-sample)": rmse(new_p, new_fuel),
        f"no tyre terms (fitted on {YEAR})": rmse(None, dict(zip(cols, beta))["lap"], tyres=False),
    }

    return errors, regret_parts(old_model, new_model, start), len(te), len(new_fit) - len(te)


def regret_parts(old_model, new_model, start):
    """Time lost in the new model by plans chosen with (parts of) the old model.

    "both": the old model's own best plan. "tyres only": best plan of the new
    model with the old tyre parameters swapped in. "pit loss only": best plan of
    the new model with the old pit loss swapped in. Only compounds fitted in
    both years are used. Returns {constraint: {cause: (plan, regret)}} plus the
    new model's own best plan per constraint.
    """
    common = [c for c in old_model.params if c in new_model.params]
    truth = new_model.with_(params={c: new_model.params[c] for c in common})
    old_tyres = {c: old_model.params[c] for c in common}
    planners = {
        "both": old_model.with_(params=old_tyres),
        "tyres only": truth.with_(params=old_tyres),
        "pit loss only": truth.with_(pit_loss=old_model.pit_loss),
    }
    out, best_new = {}, {}
    for label, sc in (("free start compound", None), (f"{start.lower()} start", start)):
        new_t, new_c, new_pits = truth.best_plan(sc)
        best_new[label] = describe(new_c, new_pits, truth.race_laps)
        out[label] = {}
        for cause, planner in planners.items():
            _, c, pits = planner.best_plan(sc)
            # Can't be below 0 (new_t is the optimum); clip floating-point noise
            out[label][cause] = (describe(c, pits, truth.race_laps),
                                 max(0.0, truth.race_time(c, pits) - new_t))
    return out, best_new


def resample_drivers(rng, *frames):
    """Bootstrap: draw drivers with replacement and return each frame rebuilt from
    those drivers. A driver drawn twice appears twice under distinct names
    (VER#0, VER#1), so it gets its own offset and its stops count twice."""
    drivers = sorted(frames[0]["Driver"].unique())
    picks = rng.choice(drivers, len(drivers), replace=True)
    out = []
    for f in frames:
        groups = {d: g for d, g in f.groupby("Driver")}
        out.append(pd.concat([groups[d].assign(Driver=f"{d}#{i}")
                              for i, d in enumerate(picks) if d in groups], ignore_index=True))
    return out


def bootstrap_regret(old, new, old_model, new_model, start, seed=0):
    """Resample drivers in both years, refit tyres and pit loss, recompute the regret.

    `old`/`new` are (laps, clean, fit_laps). Start ages and race length stay fixed.
    Returns {constraint: {cause: array of N_BOOT regrets}}.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for _ in range(N_BOOT):
        models = []
        for (laps, clean, fit_laps), base in ((old, old_model), (new, new_model)):
            r_laps, r_clean, r_fit = resample_drivers(rng, laps, clean, fit_laps)
            params, *_ = fit_tyre_model(r_fit)
            pit_loss, _ = estimate_pit_loss(r_laps, r_clean)
            models.append(base.with_(params=params, pit_loss=pit_loss))
        parts, _ = regret_parts(*models, start)
        for label, causes in parts.items():
            for cause, (_, lost) in causes.items():
                samples.setdefault(label, {}).setdefault(cause, []).append(lost)
    return {lab: {c: np.array(v) for c, v in causes.items()} for lab, causes in samples.items()}


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
    fig.suptitle("Fitted tyre model (free-air laps): points are fuel- and driver-corrected",
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

def prepare(year):
    """Load a race and return its laps, clean laps and free-air laps (the main fit sample)."""
    laps, results, source = load_race(year)
    race_laps = int(laps["LapNumber"].max())
    laps = add_free_air(laps, year)
    clean = clean_laps(laps)
    fit_laps = clean[clean["GapAhead"] >= FREE_AIR_GAP]
    counts = fit_laps["Compound"].value_counts()
    too_few = counts[counts < MIN_COMPOUND_LAPS]
    fit_laps = fit_laps[~fit_laps["Compound"].isin(too_few.index)]
    print(f"{year} {EVENT} GP ({source}): {race_laps} laps, {len(clean)} clean laps of "
          f"{len(laps)}, {len(fit_laps)} in free air used for the fit")
    for c, n in too_few.items():
        print(f"  {c.lower()} left out: only {n} free-air laps (< {MIN_COMPOUND_LAPS})")
    return laps, results, race_laps, clean, fit_laps


def build_model(laps, results, race_laps, clean, fit_laps):
    """Fitted race model plus the pieces needed to report on it."""
    params, param_se, fuel, fuel_se, offsets, residuals = fit_tyre_model(fit_laps)
    actual = actual_strategies(results, laps, race_laps)
    pit_loss, losses = estimate_pit_loss(laps, clean)
    ages = start_ages(laps, [a[1] for a in actual])
    # Effective base: average driver, fuel term averaged over the race (it is the
    # same for every strategy, so it only shifts the totals).
    base_lap = np.mean(list(offsets.values())) + fuel * (race_laps + 1) / 2
    model = RaceModel(params, base_lap, pit_loss, race_laps, ages)
    return model, dict(param_se=param_se, fuel=fuel, fuel_se=fuel_se, offsets=offsets,
                       residuals=residuals, actual=actual, losses=losses)


def print_fit(name, fit_laps, params, param_se, fuel, fuel_se, residuals):
    print(f"\n{name}, +/- 1 standard error (residual std {residuals.std():.3f} s)")
    print(f"  fuel + track evolution: {fuel:+.4f} +/- {fuel_se:.4f} s per lap")
    for c, (k0, k1) in params.items():
        n = (fit_laps["Compound"] == c).sum()
        k0_se, k1_se = param_se[c]
        k0_txt = f"{k0_se:.3f}" if k0_se else "fixed"
        print(f"  {c:<6}  k0 = {k0:6.3f} +/- {k0_txt:<5} s   "
              f"k1 = {k1:6.4f} +/- {k1_se:.4f} s/lap   ({n} laps)")
    missing = [c for c in DRY_COMPOUNDS if c not in params]
    if missing:
        print(f"  not in this sample, so not fitted: {', '.join(missing)}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    laps, results, race_laps, clean, fit_laps = prepare(YEAR)
    model, info = build_model(laps, results, race_laps, clean, fit_laps)
    params, offsets, actual = model.params, info["offsets"], info["actual"]
    fuel = info["fuel"]

    print_fit(f"Main fit: free-air laps (gap ahead >= {FREE_AIR_GAP:g} s)", fit_laps, params,
              info["param_se"], fuel, info["fuel_se"], info["residuals"])
    p_all, se_all, f_all, fse_all, _, r_all = fit_tyre_model(clean)
    print_fit("Sensitivity check: all clean laps", clean, p_all, se_all, f_all, fse_all, r_all)

    print(f"\nPit loss: {model.pit_loss:.2f} s (median of {len(info['losses'])} green-flag stops)")
    print(f"Start tyre age (median): {model.start_age}")

    print("\nModel optimum per number of stops:")
    best = {}
    for n in range(MIN_STOPS, MAX_STOPS + 1):
        best[n] = model.best_strategy(n)
        t, comps, pits = best[n]
        print(f"  {n}-stop: {describe(comps, pits, race_laps):<22} {t:9.2f} s")
    overall = min(best.values())

    # Every car started on softs, so also find the best plan under that constraint.
    start = Counter(a[2][0] for a in actual).most_common(1)[0][0]
    best_fixed = min(model.best_strategy(n, start) for n in range(MIN_STOPS, MAX_STOPS + 1))
    t, comps, pits = best_fixed
    print(f"  best starting on {start.lower()} (as the teams did): "
          f"{describe(comps, pits, race_laps)}  {t:9.2f} s  (+{t - overall[0]:.1f} s)")

    print("\nWhat the teams did (model time vs model optimum; tyre age at each stint start "
          "from the data, so sets used in practice count as used):")
    rows = [(f"Model {n}-stop", b[1], b[2], f"+{b[0] - overall[0]:.1f} s")
            for n, b in best.items()]
    rows.append((f"Model, {start.lower()} start", comps, pits, f"+{t - overall[0]:.1f} s"))
    for pos, drv, comps, pits, stint_ages in actual:
        if not set(comps) <= set(params):
            continue
        gap = model.race_time(comps, pits, stint_ages) - overall[0]
        print(f"  P{pos:<2} {drv}  {describe(comps, pits, race_laps):<22} +{gap:5.1f} s   "
              f"ages {'/'.join(map(str, stint_ages))}")
        rows.append((f"P{pos} {drv}", comps, pits, f"+{gap:.1f} s"))

    shapes = Counter(len(a[2]) - 1 for a in actual)
    print(f"\nStops used by full-distance finishers: {dict(sorted(shapes.items()))}")

    print("\nTraffic test: does hard k0 move when traffic is removed?")
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(traffic_tests(clean))

    print(f"\nOut-of-sample test: fit on half the drivers, predict the other half "
          f"({N_SPLITS} random splits, free-air laps, lap-time RMSE, mean ± std)")
    for name, (m, sd) in out_of_sample(fit_laps).items():
        print(f"  {name:<22} {m:.3f} ± {sd:.3f} s")

    print(f"\nCross-year test: fit on {COMPARE_YEAR}, use on {YEAR}")
    old = prepare(COMPARE_YEAR)
    old_model, old_info = build_model(*old)
    print_fit(f"{COMPARE_YEAR} free-air fit", old[4], old_model.params, old_info["param_se"],
              old_info["fuel"], old_info["fuel_se"], old_info["residuals"])
    print(f"  pit loss {old_model.pit_loss:.2f} s, start tyre age {old_model.start_age}")
    errors, regret, n_used, n_dropped = cross_year_test(old[4], fit_laps, old_model, model, start)
    print(f"\n  Lap-time RMSE on {YEAR} free-air laps ({n_used} laps"
          + (f"; {n_dropped} on compounds not fitted in both years left out" if n_dropped else "")
          + ")")
    for name, e in errors.items():
        print(f"    {name:<36} {e:.3f} s")
    parts, best_new = regret
    print(f"\n  Running {N_BOOT} bootstrap resamples (drivers, with replacement)...")
    boot = bootstrap_regret((old[0], old[3], old[4]), (laps, clean, fit_laps),
                            old_model, model, start)
    print(f"\n  Strategy regret: plan chosen with {COMPARE_YEAR} numbers, scored in the {YEAR} "
          f"model (90% bootstrap interval; share of resamples where nothing is lost)")
    for label, causes in parts.items():
        print(f"    {label} ({YEAR} best: {best_new[label]})")
        for cause, (plan, lost) in causes.items():
            b = boot[label][cause]
            lo, hi = np.percentile(b, [5, 95])
            print(f"      {cause:<14} plan {plan:<18} lost {lost:4.1f} s  ({lo:.1f}-{hi:.1f}; "
                  f"zero in {np.mean(b < 0.05):.0%})")

    plot_driver_stints(laps, race_laps, os.path.join(FIG_DIR, "week2_driver_stints.png"))
    plot_tyre_fit(fit_laps, params, fuel, offsets, os.path.join(FIG_DIR, "week2_tyre_fit.png"))
    plot_strategies(rows, race_laps, os.path.join(FIG_DIR, "week2_strategies.png"))
    print(f"\nFigures saved to {FIG_DIR}")
