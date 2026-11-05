"""Week 11: the 2026 simulator and benchmark, and a transfer test with no RL.

The plan was 2026 Bahrain. That race was cancelled and the "Bahrain GP" of October ran in
Kuala Lumpur, so both eras are calibrated on a track raced under both sets of rules:
Barcelona (2024 Spanish GP, 2026 Barcelona GP). Each year goes through the same pipeline:

1. Which race? Check: Barcelona ran in both eras with usable telemetry.
2. Tyres, base lap, pit and start losses from the race's laps (Weeks 2, 3-4, 5-6).
3. Car, mini-sim curves, free harvest, lifting (Weeks 9-10), with each era's MGU-K rules.
4. Closed forms. Check: settings files hold what was measured.
5. Exact optimiser, every 1-2 stop plan and every 3-stop plan that could win, both eras;
   convexity with random starts.
6. Transfer test: the 2024 optimum run in the 2026 simulator, split into seconds lost
   to the pit plan and to the energy pattern. If that is near zero, RL has little to show.
7. Sensitivity of the test to what is not published: harvest cap, fuel load, rear share.

Numbers come from eras/legacy_2022_25_barcelona.toml and eras/2026.toml; ASSUMPTIONS.md.
With CALIBRATE=1 the settings checks only print, for pasting in measured values.
"""

import importlib.util
import json
import os
import sys
import time
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pointmass as pm  # noqa: E402
import racesim  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RESULTS_DIR = os.path.join(HERE, "results")
EVENT = "Spain"                     # OpenF1 country; Barcelona is Spain's first race each year
CIRCUIT = "Catalunya"
ERAS = {2024: "legacy_2022_25_barcelona.toml", 2026: "2026.toml"}
CURVE_EVERY = 3
MAX_STOPS = 2
N_PLANS_CONVEX, N_STARTS = 5, 10
AGE_BIN, AGE_MIN_LAPS = 4, 10        # A48: a compound is used up to the last 4-lap age band with 10+ clean laps
SUCCESS = "Solve_Succeeded"
TOL = 1e-6
CALIBRATE = bool(os.environ.get("CALIBRATE"))

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w2, w5, w9 = load("week2", "2nd_week.py"), load("week5", "5th_6th_week.py"), load("week9", "9th_10th_week.py")
check, report, describe, style = w9.check, w9.report, w9.describe, w9.style


def check_file(found, section, name, rel=0.02):
    bad = {k: (section.get(k), round(float(v), 5)) for k, v in found.items()
           if k not in section or abs(section[k] - v) > rel * max(abs(v), 1e-3)}
    if bad and CALIBRATE:
        print(f"  [CALIBRATE] {name} file, measured: {bad}")
        return
    check(not bad, f"{name}: settings file matches {', '.join(f'{k} = {v:.4g}' for k, v in found.items())}"
          + (f"; MISMATCH (file, measured) {bad}" if bad else ""))


# --- Step 1: which race ---------------------------------------------------------------

def lap_start_spread(year, country, circuit):
    """Spread (10-90%, km/h) of the speed at the timing line over a driver's green laps.
    Telemetry that is on the same clock as the lap times gives a tight band."""
    s = w2.openf1_get("sessions", year=year, country_name=country, session_name="Race")
    r = s[s["circuit_short_name"] == circuit].iloc[0]
    k = int(r["session_key"])
    T = w2.openf1_get("laps", k).dropna(subset=["date_start", "lap_duration"])
    T["t0"] = pd.to_datetime(T["date_start"], utc=True, format="ISO8601")
    res = w2.openf1_get("session_result", k).sort_values("position")
    spreads = []
    for n in res["driver_number"].head(4):
        car = w2.openf1_get("car_data", k, driver_number=int(n))
        car["t"] = pd.to_datetime(car["date"], utc=True, format="ISO8601")
        car = car.sort_values("t")
        t = T[(T["driver_number"] == n) & (T["lap_number"] > 1)].sort_values("lap_number")
        v = car["speed"].to_numpy()[np.clip(car["t"].searchsorted(t["t0"]), 0, len(car) - 1)]
        v = v[v > 150]                      # in-laps cross the line in the pit lane
        spreads.append(np.quantile(v, 0.9) - np.quantile(v, 0.1))
    return float(np.median(spreads)), k


def step1_race():
    print("\nStep 1: which race? 2026 Bahrain did not run at Sakhir")
    s = w2.openf1_get("sessions", year=2026, session_name="Race")
    sak = s[s["circuit_short_name"] == "Sakhir"]
    check(bool(sak["is_cancelled"].iloc[0]), "2026 Sakhir race cancelled (OpenF1); October's 'Bahrain GP' ran at "
                                             "Kuala Lumpur, which was not raced in 2024")
    bcn24, k24 = lap_start_spread(2024, EVENT, CIRCUIT)
    bcn26, k26 = lap_start_spread(2026, EVENT, CIRCUIT)
    hun26, _ = lap_start_spread(2026, "Hungary", "Hungaroring")
    print(f"  speed at the line, 10-90% spread: Barcelona 2024 {bcn24:.0f}, Barcelona 2026 {bcn26:.0f}, "
          f"Hungary 2026 {hun26:.0f} km/h")
    check(max(bcn24, bcn26) < 25, "Barcelona telemetry is on the lap-time clock in both years")
    check(hun26 > 50, "Hungary 2026 telemetry is not (the other race with no safety car in either year)")
    return k24, k26


# --- Step 2: tyres, base lap, pit and start losses ---------------------------------------

def load_year(year):
    laps, results = w2.load_openf1(year, EVENT)
    laps["Position"] = laps["Driver"].map(results.set_index("Driver")["Position"])
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    race_laps = int(laps["LapNumber"].max())
    laps = w2.add_free_air(laps, year, EVENT)
    clean = w2.clean_laps(laps)
    fit_laps = clean[clean["GapAhead"] >= w2.FREE_AIR_GAP]
    return laps, results, race_laps, clean, fit_laps


def step2_race(year, settings, data):
    laps, results, race_laps, clean, fit_laps = data
    car = settings["car"]
    print(f"\nStep 2, {year}: {race_laps} laps, {len(clean)} clean, {len(fit_laps)} in free air "
          f"{dict(fit_laps['Compound'].value_counts())}")
    burn = car["fuel_start_kg"] / race_laps
    max_age = {}
    for c, d in clean.groupby("Compound"):
        n = d.groupby(d["Age"] // AGE_BIN).size()
        dense = n.index[n >= AGE_MIN_LAPS]
        max_age[c] = int((dense.max() + 1) * AGE_BIN - 1)
    print(f"  oldest age with data (A48): {max_age}; clean laps beyond: "
          f"{ {c: int((clean[clean['Compound'] == c]['Age'] > a).sum()) for c, a in max_age.items()} }")
    clean = clean[clean["Age"] <= clean["Compound"].map(max_age)]
    corrected = clean.copy()            # A43: 2026 has too few free-air laps on softs; same method both years
    corrected["LapTimeS"] -= car["mass_time_s_per_kg"] * (car["fuel_start_kg"] - burn * (corrected["LapNumber"] - 0.5))
    params, se, track, track_se, offsets, _ = w2.fit_tyre_model(corrected)
    base = float(np.mean(list(offsets.values())))
    print(f"  base lap {base:.3f} s, track evolution {track:+.5f} +/- {track_se:.5f} s/lap")
    for c, (k0, k1) in params.items():
        print(f"  {c.lower():<7} k0 {k0:+.3f}, k1 {k1:.4f} s/lap (+/- {se[c][1] if c in se else float('nan'):.4f})")
    top = [a[1] for a in w2.actual_strategies(results, laps, race_laps)][:10]
    first = laps[(laps["LapNumber"] == 1) & laps["Driver"].isin(top)]
    start_c = first["Compound"].mode().iloc[0]
    age = float(first[first["Compound"] == start_c]["Age"].median())
    print(f"  top 10 started on: {dict(first['Compound'].value_counts())}, {start_c.lower()} age {age:.0f}")
    found = {"base_lap_s": base, "evolution_s_per_lap": track}
    check_file(found, settings["track"], f"{year} track", rel=0.05)
    r = settings["race"]
    if not CALIBRATE:
        check(r["laps"] == race_laps and r["start_compound"] == start_c and r["start_tyre_age"] == age,
              f"{year}: settings race matches ({race_laps} laps, starts {start_c.lower()} age {age:.0f})")
    for c, (k0, k1) in params.items():
        check_file({"k0": k0, "k1": k1, "max_age": max_age[c]}, settings["tyres"].get(c, {}), f"{year} {c.lower()}", rel=0.05)

    if CALIBRATE and "energy" not in settings:
        return params
    sim = racesim.RaceSim(settings)
    res = w5.residuals(sim, laps[laps["Compound"].isin(sim.tyre)], clean)
    stops, starts = w5.measure_losses(res)
    inlap, outlap, start = stops["inlap"].median(), stops["outlap"].median(), float(np.median(starts))
    print(f"  {len(stops)} green stops: in-lap {inlap:.2f} s, out-lap {outlap:.2f} s; lap 1 {start:+.2f} s")
    check_file({"inlap_loss_s": inlap, "outlap_loss_s": outlap}, settings["pit"], f"{year} pit", rel=0.05)
    check_file({"start_loss_s": start}, settings["race"], f"{year} start", rel=0.05)
    return params


# --- Step 3: car, curves, harvest ---------------------------------------------------------

def era_rates(settings):
    r = settings["rules"]
    flow = r.get("fuel_energy_flow_max_mj_per_h")
    flow = r["fuel_flow_max_kg_per_h"] * settings["car"]["fuel_lhv_mj_per_kg"] if flow is None else flow
    return flow / 3600                                                  # MJ/s at full flow


def deploy_limit(settings):
    a, b = (settings["rules"].get(k) for k in ("mguk_taper_a_kw", "mguk_taper_b_kw_per_kmh"))
    if a is None:
        return None
    return lambda v: max(a - b * v * 3.6, 0.0) * 1e3


def fit_car(settings, traces, year):
    ps = settings["pointmass"]
    crr, rho = ps["crr"], ps["rho_kg_m3"]
    clean = [t for t in traces if t["kind"] == "clean"]
    V, A, M, per = [], [], [], {}
    for drv in sorted({t["driver"] for t in clean}):
        own = [t for t in clean if t["driver"] == drv]
        pairs = [w9.full_throttle_samples(t) for t in own]
        v, a = (np.concatenate(x) for x in zip(*pairs))
        m = np.concatenate([np.full(len(p[0]), t["mass"]) for p, t in zip(pairs, own)])
        per[drv] = pm.fit_power_drag(v, a, m, crr, rho)[0]
        V.append(v), A.append(a), M.append(m)
    V, A, M = map(np.concatenate, (V, A, M))
    power, cda, sd = pm.fit_power_drag(V, A, M, crr, rho)
    vb, ab = map(np.concatenate, zip(*(w9.braking_samples(t) for t in clean)))
    ba, bb = pm.fit_braking(vb, ab)
    print(f"  car: {power / 1e3:.0f} kW, CdA {cda:.3f} m^2, {len(V)} samples, drivers "
          f"{ {d: round(p / 1e3) for d, p in per.items()} }; braking {ba:.2f} + {bb:.5f} v^2")
    bins = np.arange(160, 340, 20)
    print("  mean full-throttle acceleration by speed: " + ", ".join(
        f"{lo}: {A[(V * 3.6 >= lo) & (V * 3.6 < lo + 20)].mean():.1f}" for lo in bins
        if ((V * 3.6 >= lo) & (V * 3.6 < lo + 20)).sum() > 20))
    return {"power": power, "cda": cda, "crr": crr, "rho": rho, "brake_a": ba, "brake_b": bb}


def lap_curves(fit, tr, nominal_deploy, kw, rate_mj, limit, grids, ice_w=None):
    """As Weeks 9-10 (A35). With `ice_w` (2026, A45) the no-deploy car has that power, and
    the deploy of the real lap is what the fitted power has above it."""
    car, sts = w9.lap_straights(fit, tr)
    on = sum(pm.run(car, st)[4].sum() for st in sts)
    power = fit["power"] - nominal_deploy * 1e6 / on if ice_w is None else ice_w
    base = w9.lap_car(fit, tr["mass"], power)
    out = {"on_throttle": on, "base_power": base.power, "real_deploy": (fit["power"] - power) * on / 1e6}
    for name, power, epj, lim in (("deploy", kw * 1e3, 1.0, limit), ("clip", -kw * 1e3, 1.0, None),
                                  ("coast", -base.power, rate_mj * 1e6 / base.power, None)):
        E, T = pm.energy_curve(base, sts, power, epj, deploy_limit=lim)
        E = np.maximum.accumulate(E)
        out[name] = np.interp(grids[name], E, T, right=np.nan)
        out[name + "_max"] = E[-1]
    return out


def step3_energy(year, settings, traces, fit):
    r, ps = settings["rules"], settings["pointmass"]
    kw, cap = r["mguk_max_kw"], r["mguk_to_es_max_mj_per_lap"]
    rate = era_rates(settings)
    clean = [t for t in traces if t["kind"] == "clean"]
    share = ps["rear_brake_share"]
    df = pd.DataFrame([{"driver": t["driver"], "kind": t["kind"], "free": w9.free_harvest(t, fit, kw, share),
                        "free_half": w9.free_harvest(t, fit, kw, 0.5), "lift": w9.lift_time(t),
                        "thr": float(np.sum(np.clip(t["throttle"], 0, 100) / 100 * np.gradient(t["t"])))}
                       for t in traces])
    c = df[df["kind"] == "clean"]
    own = c.groupby("driver")["free"].median()
    bonus = {k: float((df[df["kind"] == k]["free"] - df[df["kind"] == k]["driver"].map(own)).median())
             for k in ("in", "out")}
    h_free, lift = float(c["free"].median()), float(c["lift"].mean())
    print(f"  free harvest {h_free:.2f} MJ/lap (10-90% {c['free'].quantile(.1):.2f}-{c['free'].quantile(.9):.2f}, rear "
          f"share 0.5: {c['free_half'].median():.2f}) vs cap {cap:g}: reaches it on {(c['free'] >= cap).mean():.0%} of laps; "
          f"in {bonus['in']:+.2f}, out {bonus['out']:+.2f}; lift {lift:.2f} s/lap; throttle time {c['thr'].mean():.1f} s/lap")
    nominal = min(h_free, cap)
    eff = ps.get("ice_fuel_to_wheel")
    ice_w = None if eff is None else eff * rate * 1e6
    grids = {"deploy": np.linspace(0, ps["deploy_grid_mj"], 41), "clip": np.linspace(0, cap, 21),
             "coast": np.linspace(0, 0.14 * settings["car"]["fuel_start_kg"] * settings["car"]["fuel_lhv_mj_per_kg"]
                                  / settings["race"]["laps"], 41)}
    sample = [t for drv in sorted({t["driver"] for t in clean}) for t in [x for x in clean if x["driver"] == drv][::CURVE_EVERY]]
    t0 = time.time()
    res = [lap_curves(fit, t, nominal, kw, rate, deploy_limit(settings), grids, ice_w) for t in sample]
    curves = {n: np.nanmean([x[n] for x in res], axis=0) for n in ("deploy", "clip", "coast")}
    room = float(np.quantile([x["deploy_max"] for x in res], 0.1))
    base_w = float(np.mean([x["base_power"] for x in res]))
    real_d = float(np.mean([x["real_deploy"] for x in res]))
    print(f"  curves on {len(sample)} laps, {time.time() - t0:.0f} s; no-deploy car {base_w / 1e3:.0f} kW "
          f"(fuel to wheel {base_w / (rate * 1e6):.3f}); real deploy on the straights {real_d:.2f} MJ/lap; on the throttle "
          f"{np.mean([x['on_throttle'] for x in res]):.1f} s; room to deploy {room:.1f} MJ (10th percentile of laps)")
    for n in ("deploy", "clip", "coast"):
        print(f"    {n}: " + ", ".join(f"{e:.1f} MJ {t:+.3f} s" for e, t in list(zip(grids[n], curves[n]))[10::10]))
    return {"h_free": h_free, "bonus": bonus, "lift": lift, "room": room, "thr": float(c["thr"].mean()),
            "curves": curves, "grids": grids, "rate": rate, "nominal": nominal, "base_w": base_w, "real_deploy": real_d}


def step4_forms(year, settings, m):
    D, X, F = m["grids"]["deploy"], m["grids"]["clip"], m["grids"]["coast"]
    d, x, f = m["curves"]["deploy"], m["curves"]["clip"], m["curves"]["coast"]
    keep = np.isfinite(d) & (D <= m["room"])
    (a, b), _ = curve_fit(lambda e, a, b: -a * (1 - np.exp(-e / b)), D[keep], d[keep], p0=(5, 5))
    use = X <= m["room"]                    # clipping more than can be deployed is pointless
    X, x = X[use], x[use]
    (c1, c2), _ = curve_fit(lambda e, c1, c2: c1 * e + c2 * e ** 2, X, x, p0=(0.01, 0.05), bounds=([0, 0], [np.inf, np.inf]))
    lift0 = m["lift"] * m["rate"]
    ok = np.isfinite(f)
    (k, p), _ = curve_fit(lambda e, k, p: k * ((lift0 + e) ** p - lift0 ** p), F[ok], f[ok], p0=(0.001, 2.5))
    err = max(np.max(np.abs(-a * (1 - np.exp(-D[keep] / b)) - d[keep])), np.max(np.abs(c1 * X + c2 * X ** 2 - x)),
              np.max(np.abs(k * ((lift0 + F[ok]) ** p - lift0 ** p) - f[ok])))
    print(f"  closed forms: deploy {a:.3f}(1-exp(-D/{b:.3f})), clip {c1:.4f}X+{c2:.4f}X^2, coast {k:.6f}F^{p:.3f}; "
          f"largest error {err:.3f} s")
    if not CALIBRATE:
        check(err < 0.05, f"{year}: closed forms within 0.05 s of the curves")
    e = settings["energy"]
    check_file({"deploy_a_s": a, "deploy_b_mj": b, "clip_c1_s_per_mj": c1, "clip_c2_s_per_mj2": c2,
                "free_harvest_mj": m["h_free"], "inlap_bonus_mj": m["bonus"]["in"], "outlap_bonus_mj": m["bonus"]["out"]},
               e["battery"], f"{year} battery", rel=0.05)
    check_file({"coast_k_s": k, "coast_p": p, "lift_nominal_mj": lift0}, e["fuel"], f"{year} fuel", rel=0.05)
    if "deploy_max_mj" in e["battery"]:
        check_file({"deploy_max_mj": m["room"]}, e["battery"], f"{year} deploy room", rel=0.05)


# --- Step 5: exact optimiser ---------------------------------------------------------------

def plan_table(sim, year):
    t0 = time.time()
    best, table = sim.exact(MAX_STOPS)
    df = pd.DataFrame(table, columns=["compounds", "pits", "time", "status"]).sort_values("time").reset_index(drop=True)
    print(f"\nStep 5, {year}: {len(df)} plans in {time.time() - t0:.0f} s")
    statuses = Counter(df["status"])
    bad = len(df) - statuses[SUCCESS]
    print(f"  solver status {dict(statuses)}")
    check(bad == 0 or df["status"].iloc[0] == SUCCESS and bad / len(df) < 0.01,
          f"{year}: solves succeeded on {statuses[SUCCESS]} of {len(df)} plans, including the fastest")
    # 3 stops (A32 re-checked under the age limits, A48): exact energy can save at most
    # `gain` on any plan, so only 3-stop plans within twice that at nominal energy can win
    nominal = np.array([sim.race_time(c, p, *sim.nominal_alloc()) for c, p in zip(df["compounds"], df["pits"])])
    gain = float(np.max(nominal - df["time"].to_numpy()))
    three = racesim.RaceSim(racesim.override(sim.settings, **{"race.min_stops": 3}))
    nom3 = [(sim.race_time(c, p, *sim.nominal_alloc()), c, p) for c, p in three.plans(3)]
    near = [(c, p) for t, c, p in nom3 if t - best[0] < 2 * gain]
    rows3 = [(c, p, *sim.optimal_energy(c, p)[::3]) for c, p in near]
    print(f"  3 stops: {len(nom3)} legal plans, best at nominal energy {min(t for t, *_ in nom3) - best[0]:+.2f} s; "
          f"{len(near)} within 2 x the largest energy gain ({gain:.2f} s) solved exactly")
    if rows3:
        df = pd.concat([df, pd.DataFrame(rows3, columns=["compounds", "pits", "time", "status"])])
        df = df.sort_values("time").reset_index(drop=True)
        if df["time"].iloc[0] < best[0]:
            c, p = df.loc[0, "compounds"], df.loc[0, "pits"]
            t, fuel, batt, _ = sim.optimal_energy(c, p)
            best = (t, c, p, fuel, batt)
    check((df["status"] == SUCCESS).all(), f"{year}: every 3-stop solve succeeded")
    df["gap"] = df["time"] - best[0]
    df["plan"] = [describe(c, p, sim.laps) for c, p in zip(df["compounds"], df["pits"])]
    print(f"  OPTIMUM {year} (1-3 stops): {describe(best[1], best[2], sim.laps)}, {best[0]:.3f} s")
    for _, r in df.head(3).iterrows():
        print(f"    {r['plan']:<20} {r['time']:10.3f} s {r['gap']:+7.3f} s")
    return best, df


def energy_rows(sim, best):
    t, c, p, fuel, batt = best
    _, rows = sim.simulate(c, p, fuel, batt)
    r = pd.DataFrame(rows)
    d = r["deploy"].median()
    print(f"  energy: deploy {r['deploy'].min():.2f}-{r['deploy'].max():.2f} MJ (median {d:.2f}), free "
          f"{r['harvest_free'].median():.2f}, clip {r['harvest_extra'].min():.2f}-{r['harvest_extra'].max():.2f}; harvest cap "
          f"reached on {(r['harvest_free'] + r['harvest_extra'] >= sim.h_cap - 1e-6).mean():.0%} of laps; fuel "
          f"{100 * fuel.min() / sim.fuel_nominal:.1f}-{100 * fuel.max() / sim.fuel_nominal:.1f}% of nominal")
    for q in p:
        x = r[r["lap"].isin([q - 1, q, q + 1, q + 2])]
        print("    laps " + "; ".join(f"{int(a.lap)}: D {a.deploy:.2f} h {a.harvest_free:.2f} X {a.harvest_extra:.2f} net {a.u_b:+.2f}"
                                      for a in x.itertuples()))
    return r


def convexity(sim, best, df, year):
    rng = np.random.default_rng(1)
    plans = [(best[1], best[2])] + [tuple(df.loc[i, ["compounds", "pits"]]) for i in
                                    rng.choice(len(df), N_PLANS_CONVEX - 1, replace=False)]
    spread = 0.0
    for c, p in plans:
        ts = [sim.optimal_energy(c, p, w9.random_start(sim, c, p, rng))[0] for _ in range(N_STARTS)]
        spread = max(spread, np.ptp(ts))
    check(spread < 1e-4, f"{year}: {N_STARTS} random starts on {N_PLANS_CONVEX} plans reach the same time "
                         f"(spread {spread:.1e} s), consistent with convex (A31)")


# --- Step 6: transfer test --------------------------------------------------------------

def literal_time(new, old, plan, fuel, r_old):
    """2026 lap times if the car also keeps the old lap's deploy and harvest amounts,
    rather than letting the simulator re-split the net number."""
    T = new.energy_problem()[2]
    c, p = plan
    free = new.lap_free(p)
    harvest = (r_old["harvest_free"] + r_old["harvest_extra"]).to_numpy()
    h = np.minimum(harvest, free)
    x = np.concatenate([fuel, r_old["deploy"].to_numpy(), h, harvest - h])
    return float(T(x, new.plan_params(c, p))) + new.pit_loss * len(p)


def transfer(new, old, b_new, b_old, r_old):
    plan_old, plan_new = (b_old[1], b_old[2]), (b_new[1], b_new[2])
    fuel = np.clip(b_old[3] / old.fuel_nominal * new.fuel_nominal, new.f_min, new.f_max)
    fuel *= min(1.0, new.fuel0 / fuel.sum())
    batt = np.clip(b_old[4], new.b_min, new.b_max)
    opt_old_plan = new.optimal_energy(*plan_old)[0]
    out = {"optimum": b_new[0],
           "plan": opt_old_plan - b_new[0],
           "energy_net": new.race_time(*plan_new, fuel, batt) - b_new[0],
           "both_net": new.race_time(*plan_old, fuel, batt) - b_new[0],
           "both_literal": literal_time(new, old, plan_old, fuel, r_old) - b_new[0]}
    out["energy_literal"] = out["both_literal"] - out["plan"]
    return out


def real_strategies(sim, data, t_best):
    laps, results, race_laps, _, _ = data
    rows = []
    for pos, drv, c, p, _ in w2.actual_strategies(results, laps, race_laps):
        rows.append((pos, drv, describe(c, p, sim.laps), sim.optimal_energy(c, p)[0] - t_best))
    for pos, drv, plan, lost in rows:
        print(f"    P{pos:<2} {drv:<4} {plan:<28} {lost:+6.2f} s")
    return rows


def step6b_tyres(sim, old, b_new, b_old, r_old, df, data):
    """The linear soft-tyre fit (A43) makes every real strategy lose 15+ s, so the test is
    repeated with soft stints after the start ruled out, as the real field raced."""
    print("\nStep 6b: is the plan loss real? Real 2026 strategies against the benchmark:")
    real = real_strategies(sim, data, b_new[0])
    med = np.median([x[3] for x in real])
    print(f"  median {med:.1f} s lost (Bahrain 2024: 1.6 s)" + (": too much to be strategy" if med > 10 else ""))
    no_soft = df[[all(x != "SOFT" for x in c) for c in df["compounds"]]].iloc[0]       # A43 check
    t, f, b, _ = sim.optimal_energy(no_soft["compounds"], no_soft["pits"])
    alt = (t, no_soft["compounds"], no_soft["pits"], f, b)
    print(f"  fastest plan without softs: {no_soft['plan']}, {t:.3f} s (+{t - b_new[0]:.2f} s); real strategies against it:")
    real_alt = real_strategies(sim, data, t)
    r_alt = transfer(sim, old, alt, b_old, r_old)
    same_laps = [(c, p) for c, p in zip(df["compounds"], df["pits"]) if p == b_old[2]]
    timing_all = min(sim.optimal_energy(c, p)[0] for c, p in same_laps) - b_new[0]
    no_soft_laps = [(c, p) for c, p in same_laps if "SOFT" not in c]
    timing_alt = min(sim.optimal_energy(c, p)[0] for c, p in no_soft_laps) - t
    print(f"  old stop laps {b_old[2]} with the best 2026 compounds: {timing_all:+.2f} s; without softs {timing_alt:+.2f} s")
    print(f"  against the no-soft benchmark: old plan {r_alt['plan']:+.2f} s, energy net {r_alt['energy_net']:+.2f} s, "
          f"old amounts {r_alt['energy_literal']:+.2f} s, all {r_alt['both_literal']:+.2f} s")
    return {"no_soft_plan": no_soft["plan"], "no_soft": r_alt, "timing_all_tyres": timing_all,
            "timing_no_soft": timing_alt, "real_lost": [x[3] for x in real], "real_lost_no_soft": [x[3] for x in real_alt]}


def energy_bounds(sim, old_rows, real_deploy, kept):
    """The 'old amounts' loss assumes the 2026 optimum deploys what the model says, which
    is about twice what the real cars show. Lower bound: gain up to the real deploy only."""
    d_old = old_rows["deploy"].median()
    gain = float(sim._deploy_gain(real_deploy) - sim._deploy_gain(d_old))
    low = gain * sim.laps
    print(f"\nEnergy with old amounts kept: {kept:+.1f} s if the 2026 model optimum is right; "
          f"{low:+.1f} s if real cars' {real_deploy:.2f} MJ is the right target ({d_old:.2f} MJ kept)")
    return low


def step7_sensitivity(s26, old, b_old, r_old, df, b_ref, kept):
    print("\nStep 7: sensitivity of the transfer test to what is not published")
    top = [tuple(x) for x in df[["compounds", "pits"]].head(150).to_numpy()]
    cases = {"harvest cap 7 MJ (C5.2.10i)": {"rules.mguk_to_es_max_mj_per_lap": 7.0},
             "fuel load 65 kg": {"car.fuel_start_kg": 65.0}, "fuel load 85 kg": {"car.fuel_start_kg": 85.0},
             "free harvest +12% (rear share 0.5)": {"energy.battery.free_harvest_mj": s26["energy"]["battery"]["free_harvest_mj"] * 1.12},
             "deploy gain -10% (A35)": {"energy.battery.deploy_a_s": s26["energy"]["battery"]["deploy_a_s"] * 0.9},
             "tyre wear +20%": {f"tyres.{c}.k1": s26["tyres"][c]["k1"] * 1.2 for c in racesim.COMPOUNDS}}
    rows = {}
    for name, ch in cases.items():
        sim = racesim.RaceSim(racesim.override(s26, **ch))
        sim.b_ref = b_ref                   # A40
        sols = [(sim.optimal_energy(c, p), c, p) for c, p in top]
        (t, f, b, _), c, p = min(sols, key=lambda z: z[0][0])
        r = transfer(sim, old, (t, c, p, f, b), b_old, r_old)
        rows[name] = r
        print(f"  {name:<36} optimum {describe(c, p, sim.laps)}: plan {r['plan']:+.2f}, energy net {r['energy_net']:+.2f}, "
              f"both {r['both_net']:+.2f}, literal {r['both_literal']:+.2f} s")
    return rows


def save(s, sim, best, df, name):
    t, c, p, fuel, batt = best
    d, h, x = sim.split(batt, sim.lap_free(p))
    with open(os.path.join(RESULTS_DIR, f"optimum_{name}.json"), "w") as f:
        json.dump({"era": s["name"], "race": s["race"]["calibration_race"], "energy_model": sim.model,
                   "race_time_s": round(t, 6), "plan": describe(c, p, sim.laps), "compounds": list(c), "pit_laps": list(p),
                   "fuel_mj_per_lap": np.round(fuel, 6).tolist(), "battery_mj_per_lap": np.round(batt, 6).tolist(),
                   "deploy_mj_per_lap": np.round(d, 6).tolist(), "harvest_free_mj_per_lap": np.round(h, 6).tolist(),
                   "harvest_extra_mj_per_lap": np.round(x, 6).tolist()}, f, indent=1)
    df.assign(pits=df["pits"].map(lambda q: " ".join(map(str, q))))[["plan", "pits", "time", "gap"]].round(6).to_csv(
        os.path.join(RESULTS_DIR, f"all_plans_{name}.csv"), index=False)


def plot_battery(rows, sims):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, year in zip(axes, rows):
        r, sim = rows[year], sims[year]
        ax.plot(r["lap"], r["deploy"], color=C1, linewidth=2, label="deploy")
        ax.plot(r["lap"], r["harvest_free"] + r["harvest_extra"], color=C2, linewidth=2, label="harvest (free + clip)")
        ax.plot(r["lap"], r["harvest_free"], color=C2, linewidth=1, linestyle="--", label="of which free (braking)")
        ax.plot(r["lap"], r["batt_mj"], color=C4, linewidth=1.5, label="battery after the lap")
        ax.axhline(sim.h_cap, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xlabel("Lap", color=MUTED)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", ncol=2)
        style(ax, f"{year} Barcelona, optimum (MJ per lap; dotted = harvest cap)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week11_energy.png"), dpi=150)
    plt.close(fig)


def plot_transfer(r, t, low):
    items = [("Pit plan, linear tyre fit", r["plan"]), ("Pit plan, no soft stints", t["no_soft"]["plan"]),
             ("Energy, net number only (2026 split)", r["energy_net"]),
             ("Energy, old amounts, real-deploy target", low), ("Energy, old amounts, model optimum", r["energy_literal"])]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    y = np.arange(len(items))[::-1]
    ax.barh(y, [v for _, v in items], color=C1, height=0.55)
    ax.set_yticks(y, [k for k, _ in items])
    for yi, (_, v) in zip(y, items):
        ax.annotate(f"{v:.1f} s", (v, yi), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    style(ax, "2024 optimum in the 2026 simulator: seconds lost per race")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week11_transfer.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    step1_race()
    settings = {y: racesim.load_settings(os.path.join(HERE, "eras", f)) for y, f in ERAS.items()}
    measured = {}
    for year in ERAS:
        s = settings[year]
        print(f"\n===== {year}: {s['name']} =====")
        data = load_year(year)
        step2_race(year, s, data)
        laps, results, race_laps, clean, _ = data
        top = results.sort_values("Position")["Driver"].head(4).tolist()
        traces = w9.load_traces(w2, laps, clean, racesim.RaceSim(s), year, EVENT, top)
        print(f"Step 3, {year}: telemetry {Counter(t['kind'] for t in traces)} for {top}")
        fit = fit_car(s, traces, year)
        check_file({"power_kw": fit["power"] / 1e3, "cda_m2": fit["cda"], "brake_a_m_s2": fit["brake_a"],
                    "brake_b_per_m": fit["brake_b"]}, s["pointmass"], f"{year} car", rel=0.03)
        measured[year] = step3_energy(year, s, traces, fit)
        step4_forms(year, s, measured[year])
    s24, s26 = settings[2024], settings[2026]
    m24, m26 = measured[2024], measured[2026]
    per_thr = s24["car"]["fuel_start_kg"] * s24["car"]["fuel_lhv_mj_per_kg"] / s24["race"]["laps"] / (m24["thr"] * era_rates(s24))
    kg_26 = m26["thr"] * era_rates(s26) * per_thr * s26["race"]["laps"] / s26["car"]["fuel_lhv_mj_per_kg"]
    eff = m24["base_w"] / (era_rates(s24) * 1e6)
    print(f"\nDerived for 2026: fuel load {kg_26:.1f} kg (A44); fuel-to-wheel efficiency 2024 {eff:.3f} -> "
          f"engine {eff * era_rates(s26) * 1e3:.0f} kW (A45)")
    check_file({"fuel_start_kg": kg_26}, s26["car"], "2026 fuel load", rel=0.03)
    check_file({"ice_fuel_to_wheel": eff}, s26["pointmass"], "2026 engine", rel=0.03)
    if CALIBRATE:
        sys.exit()
    print(f"Real 2026 deploy on the straights {m26['real_deploy']:.2f} MJ/lap against {m26['h_free']:.2f} MJ of free harvest "
          f"(part-throttle deploy is not seen by this measure)")
    sims = {y: racesim.RaceSim(settings[y]) for y in ERAS}
    bests, tables, rows = {}, {}, {}
    for y, sim in sims.items():
        bests[y], tables[y] = plan_table(sim, y)
        rows[y] = energy_rows(sim, bests[y])
        convexity(sim, bests[y], tables[y], y)
        save(settings[y], sim, bests[y], tables[y], settings[y]["name"])

    print("\nStep 6: transfer test, 2024 optimum in the 2026 simulator (no RL)")
    r = transfer(sims[2026], sims[2024], bests[2026], bests[2024], rows[2024])
    print(f"  2026 optimum {describe(bests[2026][1], bests[2026][2], 66)}, {bests[2026][0]:.3f} s; "
          f"2024 optimum {describe(bests[2024][1], bests[2024][2], 66)}")
    print(f"  pit plan only (old plan, best 2026 energy):           {r['plan']:+.2f} s")
    print(f"  energy only, net pattern, simulator re-splits:        {r['energy_net']:+.2f} s")
    print(f"  energy only, old deploy and harvest amounts kept:     {r['energy_literal']:+.2f} s")
    print(f"  both, net pattern (what the current env can show):    {r['both_net']:+.2f} s")
    print(f"  both, old amounts kept:                               {r['both_literal']:+.2f} s")
    check(min(r["plan"], r["energy_net"], r["both_net"], r["both_literal"]) >= -TOL, "nothing beats the 2026 optimum")
    tyres = step6b_tyres(sims[2026], sims[2024], bests[2026], bests[2024], rows[2024], tables[2026], data)
    low = energy_bounds(sims[2026], rows[2024], m26["real_deploy"], r["energy_literal"])
    sens = step7_sensitivity(s26, sims[2024], bests[2024], rows[2024], tables[2026], sims[2026].b_ref, None)
    with open(os.path.join(RESULTS_DIR, "transfer_barcelona_2024_to_2026.json"), "w") as f:
        json.dump({"main": r, "tyres": tyres, "energy_old_amounts_low": low, "sensitivity": sens}, f, indent=1, default=float)
    plot_battery(rows, sims)
    plot_transfer(r, tyres, low)
    print(f"\nAll checks passed. Figures in {FIG_DIR}")
