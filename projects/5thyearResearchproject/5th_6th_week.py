"""Weeks 5-6: the ETH-style race simulator with old-era (2022-25) energy rules.

RaceSim (racesim.py) steps one lap at a time. State: battery, fuel, car mass, compound,
tyre wear, race time and the "next lap is an out-lap" flag. Inputs each lap: fuel use,
battery use and the pit choice 0-3 (no stop / soft / medium / hard). Lap time =
base + mass effect + energy terms (diminishing returns) + fitted tyre loss + in-lap /
out-lap loss + standing start on lap 1.

Each step is checked and plotted (figures/week5_*.png, week6_*.png):

1. FIA energy limits (2024 Technical Regulations, cited in the settings file). Check:
   the simulator's per-lap bounds are the regulation caps, and the caps that are not
   modelled directly (MGU-K power, fuel flow) never bind.
2. Energy terms. Check: using more fuel or battery always saves time, with diminishing
   returns, and both terms are zero at nominal use.
3. In-lap, out-lap and start losses measured from FastF1 (2024 Bahrain). Check: the
   settings file matches, the step-by-step simulator charges them on the right laps,
   and a stop costs about 25-30 s (ETH: 26.6 s). Bahrain comes out just below that, so
   the Week 2 method is rerun on other Bahrain years and on other tracks to show
   the low value comes from the track, not the method.
4. Replay every lead-lap finisher's real strategy and compare the simulated race time
   with the real one.

All era numbers come from eras/legacy_2022_25.toml; assumptions are in ASSUMPTIONS.md.
"""

import importlib.util
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import racesim  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
SETTINGS = os.path.join(HERE, "eras", "legacy_2022_25.toml")
YEAR = 2024
NEIGHBOUR_LAPS = 3          # clean laps either side of a stop used as its reference
MIN_NEIGHBOURS = 2          # stops with fewer clean neighbours on a side are skipped
ETH_STOP_S = 26.6           # ETH's in-lap + out-lap loss per stop
STOP_RANGE = (25.0, 30.0)   # reviewer's expected range per stop
# Races for the pit-loss cross-check: Bahrain in other years, plus Singapore (a long
# pit lane) and Zandvoort (a short one). FastF1 first, OpenF1 if it is blocked.
OTHER_RACES = ((2022, "Bahrain"), (2023, "Bahrain"), (2024, "Singapore"), (2024, "Netherlands"))
LONG_PIT_LANE = (2024, "Singapore")
SAME_TRACK_TOL = 1.5        # s: Bahrain years agree with 2024 to within this
REPLAY_TOL = 0.001          # replayed race time within 0.1% of the real one
TOL = 1e-6

# Categorical slots 1-4 of the validated palette, always with direct labels
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def load_week2():
    spec = importlib.util.spec_from_file_location("week2", os.path.join(HERE, "2nd_week.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def style(ax, title=None):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def describe(compounds, pits, laps):
    bounds = (0, *pits, laps)
    return " -> ".join(f"{c[0]}{bounds[i + 1] - bounds[i]}" for i, c in enumerate(compounds))


def check(cond, msg):
    assert cond, msg
    print(f"  [ok] {msg}")


# --- Step 1: FIA energy limits ----------------------------------------------------

def step1_rules(settings, sim):
    print("\nStep 1: FIA energy limits (2022-25 power unit)")
    r = settings["rules"]
    print(f"  MGU-K {r['mguk_max_kw']:g} kW; ES -> MGU-K <= {r['es_to_mguk_max_mj_per_lap']:g} MJ/lap "
          f"(deploy); MGU-K -> ES <= {r['mguk_to_es_max_mj_per_lap']:g} MJ/lap (harvest); "
          f"ES window {r['es_soc_window_mj']:g} MJ")
    print(f"  fuel flow <= {r['fuel_flow_max_kg_per_h']:g} kg/h; race fuel <= {r['race_fuel_max_kg']:g} kg")
    check(sim.b_max == r["es_to_mguk_max_mj_per_lap"] and sim.b_min == -r["mguk_to_es_max_mj_per_lap"]
          and sim.b_cap == r["es_soc_window_mj"],
          "simulator battery bounds are the regulation caps (deploy, harvest, window)")
    full_k = sim.b_max / (r["mguk_max_kw"] / 1000)
    check(full_k < sim.base, f"deploying {sim.b_max:g} MJ needs {full_k:.1f} s at full MGU-K power, "
                             f"less than a {sim.base:.1f} s lap, so the per-lap cap binds, not the power")
    kg = sim.lhv
    f_band = settings["energy"]["fuel"]["alloc_max_frac"] * sim.fuel_nominal
    print(f"  fuel per lap: nominal {sim.fuel_nominal / kg:.2f} kg, max allocation {f_band / kg:.2f} kg, "
          f"flow limit held all lap {sim.fuel_flow_cap / kg:.2f} kg")
    check(f_band < sim.fuel_flow_cap and sim.f_max == f_band,
          "the fuel-flow limit does not bind (allocation band is tighter)")
    check(sim.fuel0 / kg <= r["race_fuel_max_kg"] and sim.m_empty >= r["min_mass_kg"],
          "start fuel within the race maximum, empty car at the minimum mass")

    # Every lap obeys the limits however hard the controls are pushed
    state, worst = sim.initial_state(), []
    rng = np.random.default_rng(0)
    for _ in range(sim.laps):
        u_f, u_b = rng.uniform(0, 3 * sim.fuel_nominal), rng.uniform(-10, 10)
        new, _, parts = sim.step(state, u_f, u_b)
        worst.append((parts["u_b"], parts["u_f"], new.batt_mj, new.fuel_mj))
        state = new
    u_b, u_f, soc, fuel = map(np.array, zip(*worst))
    check(u_b.max() <= sim.b_max + TOL and u_b.min() >= sim.b_min - TOL
          and soc.min() >= -TOL and soc.max() <= sim.b_cap + TOL and fuel.min() >= -TOL
          and u_f.max() <= sim.f_max + TOL,
          "random extreme inputs are clipped: per-lap caps, battery window and fuel all hold")


# --- Step 2: energy terms ------------------------------------------------------------

def step2_energy_terms(sim):
    print("\nStep 2: energy terms with diminishing returns")
    check(abs(sim.fuel_term(sim.fuel_nominal)) < TOL and abs(sim.batt_term(0.0)) < TOL,
          "both terms are zero at nominal use, so nominal energy reproduces Weeks 3-4")
    u_f = np.linspace(sim.f_min, sim.f_max, 201)
    u_b = np.linspace(sim.b_min, sim.b_max, 201)
    for name, t in (("fuel", sim.fuel_term(u_f)), ("battery", sim.batt_term(u_b))):
        d1, d2 = np.diff(t), np.diff(t, 2)
        check(np.all(d1 < 0) and np.all(d2 > 0),
              f"{name}: more energy always saves time, and each extra unit saves less")
    print(f"  fuel: -10% costs {sim.fuel_term(sim.f_min):+.2f} s, +10% gains {sim.fuel_term(sim.f_max):+.2f} s")
    print(f"  battery: full harvest {sim.b_min:g} MJ costs {sim.batt_term(sim.b_min):+.2f} s, "
          f"full deploy {sim.b_max:g} MJ gains {sim.batt_term(sim.b_max):+.2f} s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, x, y, color, title, xlabel in (
            (axes[0], u_f / sim.fuel_nominal, sim.fuel_term(u_f), C1,
             "Fuel term (s) vs fuel used, x nominal", "Fuel used this lap (x nominal)"),
            (axes[1], u_b, sim.batt_term(u_b), C2,
             "Battery term (s) vs net battery energy", "Net battery energy this lap (MJ, + = deploy)")):
        ax.plot(x, y, color=color, linewidth=2)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
        for end, side in ((x[0], "left"), (x[-1], "right")):
            ax.axvline(end, color=MUTED, linewidth=0.8, linestyle=":")
        style(ax, title)
        ax.set_xlabel(xlabel, color=MUTED)
    axes[1].annotate("FIA harvest cap\n2 MJ/lap", (u_b[0], 0.45), xytext=(4, 0),
                     textcoords="offset points", fontsize=8, color=INK)
    axes[1].annotate("FIA deploy cap\n4 MJ/lap", (u_b[-1], 0.45), xytext=(-4, 0),
                     textcoords="offset points", fontsize=8, color=INK, ha="right")
    axes[0].annotate("ETH band\n0.9-1.1", (u_f[0] / sim.fuel_nominal, 0.35), xytext=(4, 0),
                     textcoords="offset points", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week5_energy_terms.png"), dpi=150)
    plt.close(fig)


# --- Step 3: in-lap, out-lap and start losses from FastF1 ------------------------------

def driver_plan(d):
    """A driver's stints as (compounds, pit laps, tyre age at each stint start)."""
    stints = d.groupby("Stint").agg(compound=("Compound", "first"), last=("LapNumber", "max"),
                                    age=("Age", "first"))
    return (tuple(stints["compound"]), tuple(stints["last"].iloc[:-1].astype(int)),
            tuple(stints["age"].astype(float)))


def residuals(sim, laps, clean):
    """Every lap's time minus the simulator's flying-lap time for that driver's actual
    tyres and lap (no start, in-lap or out-lap term). Mean-driver base, nominal fuel."""
    laps = laps.copy()
    laps["Clean"] = laps.index.isin(clean.index)
    out = []
    for drv, d in laps.groupby("Driver"):
        d = d.sort_values("LapNumber")
        _, rows = sim.simulate(*driver_plan(d)[:2], ages=driver_plan(d)[2])
        flying = {r["lap"]: r["time"] - r["start"] - r["in-lap"] - r["out-lap"] for r in rows}
        d = d.assign(Flying=d["LapNumber"].map(flying))
        out.append(d.assign(Resid=d["LapTimeS"] - d["Flying"]))
    return pd.concat(out)


def measure_losses(res):
    """In-lap and out-lap loss per green-flag stop, and the lap-1 loss per driver:
    the lap's residual minus the median residual of nearby clean laps (before the stop
    for the in-lap, after it for the out-lap, laps 2-4 for the start). Using nearby
    laps keeps each driver's pace and traffic out of the loss."""
    stops, starts = [], []
    for drv, d in res.groupby("Driver"):
        d = d.set_index("LapNumber")
        clean = d[d["Clean"]]["Resid"]

        def ref(lo, hi):
            r = clean[(clean.index >= lo) & (clean.index <= hi)]
            return r.median() if len(r) >= MIN_NEIGHBOURS else np.nan

        for n in d.index[d["InLap"]]:
            if n + 1 not in d.index or not (d.loc[n, "Green"] and d.loc[n + 1, "Green"]):
                continue
            before, after = ref(n - NEIGHBOUR_LAPS, n - 1), ref(n + 2, n + 1 + NEIGHBOUR_LAPS)
            times = d.loc[[n, n + 1], "LapTimeS"]
            if np.isfinite(before) and np.isfinite(after) and np.isfinite(times).all():
                stops.append({"Driver": drv, "lap": n, "inlap": d.loc[n, "Resid"] - before,
                              "outlap": d.loc[n + 1, "Resid"] - after})
        if 1 in d.index and np.isfinite(d.loc[1, "LapTimeS"]):
            r0 = ref(2, 1 + NEIGHBOUR_LAPS)
            if np.isfinite(r0):
                starts.append(d.loc[1, "Resid"] - r0)
    stops = pd.DataFrame(stops)
    stops["total"] = stops["inlap"] + stops["outlap"]
    return stops, np.array(starts)


def step3_pit_loss(settings, sim, res, w2_losses):
    print("\nStep 3: in-lap, out-lap and start losses (FastF1)")
    stops, starts = measure_losses(res)
    inlap, outlap, start = stops["inlap"].median(), stops["outlap"].median(), float(np.median(starts))
    print(f"  {len(stops)} green-flag stops: in-lap {inlap:.2f} s, out-lap {outlap:.2f} s "
          f"(medians), in + out per stop {stops['total'].median():.2f} s "
          f"(IQR {stops['total'].quantile(0.25):.1f}-{stops['total'].quantile(0.75):.1f})")
    print(f"  Week 2 method (in + out - 2 x driver median lap): {np.median(w2_losses):.2f} s")
    print(f"  lap-1 standing start: {start:+.2f} s vs a flying lap ({len(starts)} drivers, median)")
    p = settings["pit"]
    check(abs(p["inlap_loss_s"] - inlap) < 5e-3 and abs(p["outlap_loss_s"] - outlap) < 5e-3
          and abs(settings["race"]["start_loss_s"] - start) < 5e-3,
          "settings file matches these measurements (in-lap, out-lap, start)")
    check(inlap > 0 and outlap > 0, "both the in-lap and the out-lap lose time")

    # The step-by-step simulator charges them on the right laps
    plan = (("SOFT", "HARD", "HARD"), (15, 36))
    _, rows = sim.simulate(*plan)
    on = {r["lap"]: (r["in-lap"], r["out-lap"], r["start"]) for r in rows}
    check(all(on[n][0] == sim.inlap_loss and on[n + 1][1] == sim.outlap_loss for n in plan[1])
          and sum(v[0] + v[1] for v in on.values()) == sim.pit_loss * len(plan[1])
          and on[1][2] == sim.start_loss and sum(v[2] for v in on.values()) == sim.start_loss,
          "in-lap on the stop lap, out-lap on the next lap, start loss on lap 1 only")
    fuel, batt = sim.nominal_alloc()
    check(abs(sim.simulate(*plan)[0] - sim.race_time(*plan, fuel, batt)) < TOL,
          "lap-by-lap simulation still equals the vectorised race time")
    # Pit choice as a number 0-3 gives the same race as the compound name
    st_n, st_c = sim.initial_state(), sim.initial_state()
    for k in range(1, sim.laps + 1):
        choice = 3 if k in plan[1] else 0
        st_n, *_ = sim.step(st_n, sim.fuel_nominal, 0.0, choice)
        st_c, *_ = sim.step(st_c, sim.fuel_nominal, 0.0, racesim.PIT_CHOICES[choice])
    check(st_n == st_c, "pit choice 0-3 gives the same race as naming the compound")
    no_stop = racesim.RaceSim(racesim.override(settings, **{"pit.inlap_loss_s": 0.0,
                                                            "pit.outlap_loss_s": 0.0}))
    per_stop = (sim.simulate(*plan)[0] - no_stop.simulate(*plan)[0]) / len(plan[1])
    lo, hi = STOP_RANGE
    print(f"  simulated cost per stop: {per_stop:.2f} s (ETH {ETH_STOP_S} s, expected {lo:g}-{hi:g} s)")
    if lo <= per_stop <= hi:
        print(f"  [ok] in-lap + out-lap loss is within {lo:g}-{hi:g} s per stop")
    else:
        print(f"  [!!] in-lap + out-lap loss is outside {lo:g}-{hi:g} s per stop "
              f"(see step 3b and ASSUMPTIONS A28)")
    check(abs(per_stop - sim.pit_loss) < TOL, "simulated cost per stop = in-lap + out-lap loss")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, (key, color, name) in enumerate((("inlap", C1, "In-lap"), ("outlap", C2, "Out-lap"),
                                            ("total", C3, "In + out"))):
        y = stops[key].to_numpy()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y, s=14, color=color, alpha=0.7)
        ax.plot([i - 0.25, i + 0.25], [np.median(y)] * 2, color=INK, linewidth=2)
        ax.annotate(f"{name}\nmedian {np.median(y):.1f} s", (i + 0.28, np.median(y)),
                    fontsize=8, color=INK, va="center")
    ax.axhspan(lo, hi, color=GRID, alpha=0.6, zorder=0)
    ax.axhline(ETH_STOP_S, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"ETH {ETH_STOP_S} s", (-0.45, ETH_STOP_S), xytext=(0, 3), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.set_xticks([])
    ax.set_xlim(-0.5, 3.1)
    style(ax, f"Time lost per green-flag stop, {YEAR} Bahrain ({len(stops)} stops, s); "
              f"band = {lo:g}-{hi:g} s")
    ax = axes[1]
    ax.hist(starts, bins=12, color=C4, edgecolor="white")
    ax.axvline(start, color=INK, linewidth=2)
    ax.annotate(f"median {start:+.1f} s", (start, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK)
    style(ax, "Lap 1 vs a flying lap (s), per driver")
    ax.set_xlabel("Seconds lost on lap 1", color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week5_pit_loss.png"), dpi=150)
    plt.close(fig)
    return stops


def step3b_other_races(w2, w2_losses):
    """Week 2 pit-loss method (in + out - 2 x driver median lap) on other races. It needs
    no tyre fit, so it works on any track; on 2024 Bahrain it is within 0.7 s of step 3."""
    print("\nStep 3b: is the low Bahrain pit loss the track or the method? (Week 2 method)")
    races = {(YEAR, w2.EVENT): np.asarray(w2_losses)}
    for year, event in OTHER_RACES:
        laps, _, source = w2.load_race(year, event)
        races[(year, event)] = np.asarray(w2.estimate_pit_loss(laps, w2.clean_laps(laps))[1])
    races = dict(sorted(races.items(), key=lambda kv: np.median(kv[1])))
    for (year, event), x in races.items():
        q1, q3 = np.percentile(x, [25, 75])
        print(f"  {year} {event:<12} {len(x):3d} stops: median {np.median(x):5.2f} s (IQR {q1:.1f}-{q3:.1f})")
    ref = np.median(races[(YEAR, w2.EVENT)])
    bahrain = [np.median(x) for (y, e), x in races.items() if e == w2.EVENT]
    long_lane = np.median(races[LONG_PIT_LANE])
    check(max(abs(b - ref) for b in bahrain) < SAME_TRACK_TOL,
          f"every Bahrain year is within {SAME_TRACK_TOL:g} s of {YEAR}: the value is stable at this track")
    check(long_lane > ETH_STOP_S and all(b < long_lane for b in bahrain),
          f"the same method gives {long_lane:.1f} s at {LONG_PIT_LANE[1]} (> ETH {ETH_STOP_S} s), "
          f"so it can measure long pit lanes; Bahrain's is shorter")

    lo, hi = STOP_RANGE
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(races))
    ax.axvspan(lo, hi, color=GRID, alpha=0.6, zorder=0)
    ax.axvline(ETH_STOP_S, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"ETH {ETH_STOP_S} s", (ETH_STOP_S, y[-1] + 0.4), xytext=(3, 0),
                textcoords="offset points", fontsize=8, color=MUTED)
    for i, ((year, event), x) in enumerate(races.items()):
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        color = C1 if event == w2.EVENT else C2
        ax.plot([q1, q3], [i, i], color=color, linewidth=3, alpha=0.5)
        ax.plot(med, i, "o", color=color, markersize=7)
        ax.annotate(f"{med:.1f} s", (q3, i), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=INK, va="center")
    ax.set_yticks(y, [f"{year} {event}" for year, event in races])
    ax.set_ylim(-0.6, len(races) - 0.2)
    style(ax, f"Pit loss per green-flag stop, Week 2 method (median, IQR, s); "
              f"band = {lo:g}-{hi:g} s")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week5_pit_loss_tracks.png"), dpi=150)
    plt.close(fig)


# --- Step 4: replay real strategies ------------------------------------------------------

def step4_replay(settings, sim, res, results):
    print(f"\nStep 4: replay real strategies, {YEAR} Bahrain (nominal energy)")
    rows = []
    for _, r in results.dropna(subset=["Position"]).sort_values("Position").iterrows():
        d = res[res["Driver"] == r["Driver"]].sort_values("LapNumber")
        if d["LapNumber"].max() != sim.laps or d["LapTimeS"].isna().any():
            continue  # lapped: not comparable over the full distance
        compounds, pits, ages = driver_plan(d)
        pace = d.loc[d["Clean"], "Resid"].median()  # driver's usual lap vs the mean driver
        own = racesim.RaceSim(racesim.override(settings, **{"track.base_lap_s": sim.base + pace}))
        actual = d["LapTimeS"].sum()
        t_mean, _ = sim.simulate(compounds, pits, ages=ages)
        t_own, laps_own = own.simulate(compounds, pits, ages=ages)
        rows.append({"pos": int(r["Position"]), "driver": r["Driver"],
                     "plan": describe(compounds, pits, sim.laps), "actual": actual,
                     "mean": t_mean - actual, "own": t_own - actual, "pace": pace,
                     "lap1": laps_own[0]["time"] - d["LapTimeS"].iloc[0],
                     "laps": np.array([x["time"] for x in laps_own]),
                     "real": d["LapTimeS"].to_numpy()})
    print(f"  {'':3} {'drv':4} {'strategy':<22} {'real (s)':>9} {'pace':>6} "
          f"{'sim-real, mean driver':>22} {'own pace':>9} {'of which lap 1':>15}")
    for x in rows:
        print(f"  P{x['pos']:<2} {x['driver']:<4} {x['plan']:<22} {x['actual']:9.1f} {x['pace']:+6.2f} "
              f"{x['mean']:+18.1f} s {x['own']:+8.1f} s {x['lap1']:+13.1f} s")
    own_err = np.array([x["own"] for x in rows])
    real = np.array([x["actual"] for x in rows])
    win = rows[0]
    print(f"  winner {win['driver']}: simulated {win['actual'] + win['own']:.1f} s vs real "
          f"{win['actual']:.1f} s ({win['own']:+.1f} s, {win['own'] / win['actual']:+.3%})")
    print(f"  all {len(rows)} lead-lap finishers, own pace: median |error| "
          f"{np.median(np.abs(own_err)):.1f} s, worst {own_err[np.argmax(np.abs(own_err))]:+.1f} s")
    check(abs(win["own"]) < REPLAY_TOL * win["actual"],
          f"winner's replayed race time within {REPLAY_TOL:.1%} of the real one")
    check(np.median(np.abs(own_err) / real) < REPLAY_TOL,
          f"median replay error within {REPLAY_TOL:.1%} across lead-lap finishers")
    # Lap 1 gets one average start loss (A29), but front-runners lose ~1 s and cars in the
    # pack much more, so take it out to test tyres, fuel and stops on their own
    rest = own_err - np.array([x["lap1"] for x in rows])
    worst = rows[np.argmax(np.abs(rest))]
    w = np.argmax(np.abs(rest))
    print(f"  without lap 1: median |error| {np.median(np.abs(rest)):.1f} s, worst "
          f"{worst['driver']} {rest[w]:+.1f} s ({rest[w] / real[w]:+.3%}); see ASSUMPTIONS A29-A30")
    check(np.abs(rest).max() < np.abs(own_err).max(),
          f"the worst replay error ({rows[np.argmax(np.abs(own_err))]['driver']}) is mostly its lap 1: "
          f"{np.abs(own_err).max():.1f} s -> {np.abs(rest).max():.1f} s without it")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.18, [x["mean"] for x in rows], height=0.36, color=C1)
    ax.barh(y - 0.18, [x["own"] for x in rows], height=0.36, color=C2)
    ax.set_yticks(y, [f"P{x['pos']} {x['driver']}" for x in rows])
    ax.axvline(0, color=INK, linewidth=1)
    style(ax, "Simulated minus real race time (s)")
    ax.annotate("mean-driver pace", (rows[0]["mean"], y[0] + 0.18), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=C1, va="center")
    ax.annotate("driver's own pace", (rows[0]["own"], y[0] - 0.18), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=C2, va="center")
    ax = axes[1]
    lap = np.arange(1, sim.laps + 1)
    gap = np.cumsum(win["laps"]) - np.cumsum(win["real"])
    ax.plot(lap, gap, color=C2, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    compounds, pits, _ = driver_plan(res[res["Driver"] == win["driver"]])
    for p in pits:
        ax.axvline(p, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate("stops", (pits[0], gap.max()), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=MUTED, va="top")
    ax.annotate(f"{gap[-1]:+.1f} s at the flag", (lap[-1], gap[-1]), xytext=(-4, 6),
                textcoords="offset points", fontsize=8, color=INK, ha="right")
    style(ax, f"{win['driver']} ({win['plan']}): simulated minus real, cumulative (s)")
    ax.set_xlabel("Lap", color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week6_replay.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    settings = racesim.load_settings(SETTINGS)
    print(f"Era settings: {settings['name']} ({settings['race']['calibration_race']})")
    sim = racesim.RaceSim(settings)
    step1_rules(settings, sim)
    step2_energy_terms(sim)
    w2 = load_week2()
    laps, results, race_laps, clean, _ = w2.prepare(YEAR)
    assert race_laps == sim.laps
    res = residuals(sim, laps, clean)
    w2_losses = w2.estimate_pit_loss(laps, clean)[1]
    step3_pit_loss(settings, sim, res, w2_losses)
    step3b_other_races(w2, w2_losses)
    step4_replay(settings, sim, res, results)
    print(f"\nAll checks passed. Figures in {FIG_DIR}")
