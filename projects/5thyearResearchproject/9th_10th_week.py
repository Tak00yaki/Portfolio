"""Weeks 9-10: a real energy trade-off, from physics instead of guesses.

Weeks 7-8 found the battery deploys the same 0.07 MJ every lap: with the placeholder
terms (A16, A19) there is no energy decision to get right, so an RL agent could not fail
at it and the hypothesis could not be tested. This replaces both terms with numbers from
a point-mass straight-line simulation (pointmass.py) driven along the 2024 Bahrain
straights from OpenF1 speed traces, and adds the parts of the battery problem that make
it worth planning:

    deploy       battery energy saves lap time, with diminishing returns
    free harvest braking puts energy back at no cost, up to a track-specific amount
    extra        harvesting more at full throttle ("clipping") costs lap time
    pit laps     in- and out-laps brake more, so harvest more for free
    carry-over   the 4 MJ battery window lets energy saved on one lap be spent later
    fuel         using less fuel means lifting and coasting, which costs lap time

Each step is checked and plotted (figures/week9_*.png, week10_*.png):

1. Fit the car (wheel power, drag, braking) to full-throttle and braking samples.
   Check: the same car for every driver; plausible braking.
2. Mini-sim on every sampled lap: deploy, clip and coast curves. Check: the unchanged
   car reproduces the measured straight times; diminishing returns; deploy and clip
   on the same lap add up.
3. Free harvest from braking, the in-/out-lap bonus and the nominal lift. Check:
   settings file holds these numbers.
4. Closed forms for the simulator. Check: they fit the curves; settings file matches.
5. Rerun the exact optimiser (every 1-2 stop plan). Check: model equality, limits;
   what the battery does around the stops; real strategies against the new benchmark.
6. Sensitivity: harvest cap raised so it no longer binds. Check: the in-lap charges.
7. Convexity (A31) with random starts on several plans, both cases.

All era numbers come from eras/legacy_2022_25.toml; assumptions are in ASSUMPTIONS.md.
"""

import importlib.util
import json
import os
import time
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pointmass as pm  # noqa: E402
import racesim  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RESULTS_DIR = os.path.join(HERE, "results")
SETTINGS = os.path.join(HERE, "eras", "legacy_2022_25.toml")
YEAR = 2024
DRIVERS = ("VER", "PER", "SAI", "LEC")    # the top four; ~5.7 MB of telemetry each
CURVE_EVERY = 3                         # mini-sim curves on every 3rd clean lap per driver
FULL_THROTTLE_MIN_KMH = 160             # below this the car can be traction-limited
BRAKE_MIN_KMH = 150
MAX_STOPS = 2
N_PLANS_CONVEX = 5                      # plans in the convexity check (optimum + random)
N_STARTS = 10                           # random starting points per plan
SENS_HARVEST_CAP = 4.0                  # step 6: harvest cap (MJ/lap) that no longer binds
KW_2026 = 350.0                         # MGU-K power in 2026, for the preview curve only
SUCCESS = "Solve_Succeeded"
TOL = 1e-6

# Categorical slots 1-4 of the validated palette, always with direct labels
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

D_GRID = np.linspace(0, 4, 41)          # deploy, MJ (FIA cap 4 MJ/lap)
X_GRID = np.linspace(0, 2, 21)          # extra harvest, MJ (FIA cap 2 MJ/lap)
F_GRID = np.linspace(0, 10, 51)         # fuel saved by coasting, MJ (10% of nominal ~ 8.3 MJ)


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


def report(cond, msg):
    """An expectation being tested, not a correctness check: print the answer either way."""
    print(f"  [{'yes' if cond else 'NO '}] {msg}")
    return cond


def check_settings(found, section, keys, rel=0.01):
    """The settings file holds what this script measured (to rounding)."""
    bad = {k: (section[k], round(found[k], 4)) for k in keys
           if abs(section[k] - found[k]) > rel * max(abs(found[k]), 1e-3)}
    if bad and os.environ.get("CALIBRATE"):     # first run only: print the numbers to paste in
        print(f"  [CALIBRATE] settings file, measured: {bad}")
        return
    check(not bad, f"settings file matches ({', '.join(f'{k} = {found[k]:.4g}' for k in keys)})"
          + (f"; MISMATCH (file, measured): {bad}" if bad else ""))


# --- Telemetry ---------------------------------------------------------------------

def load_traces(w2, laps, clean, sim):
    """Every green-flag lap of DRIVERS as a speed trace: time, distance, smoothed speed,
    acceleration, throttle, brake, DRS, car mass. Kind: clean (Week 2 definition, A5),
    in- or out-lap."""
    key = w2.openf1_session_key(YEAR)
    numbers = w2.openf1_get("drivers", key).set_index("name_acronym")["driver_number"]
    timing = w2.openf1_get("laps", key).dropna(subset=["date_start", "lap_duration"])
    timing["t0"] = pd.to_datetime(timing["date_start"], utc=True, format="ISO8601")
    clean_set = set(zip(clean["Driver"], clean["LapNumber"]))
    green = laps[laps["Green"].astype(bool)].set_index(["Driver", "LapNumber"])
    fuel_kg = sim.fuel0 / sim.lhv
    traces = []
    for drv in DRIVERS:
        n = int(numbers[drv])
        car = w2.openf1_get("car_data", key, driver_number=n)
        car["t"] = pd.to_datetime(car["date"], utc=True, format="ISO8601")
        car = car.sort_values("t")
        for _, r in timing[timing["driver_number"] == n].iterrows():
            lap = int(r["lap_number"])
            if (drv, lap) in clean_set:
                kind = "clean"
            elif (drv, lap) in green.index and green.loc[(drv, lap), "InLap"]:
                kind = "in"
            elif (drv, lap) in green.index and green.loc[(drv, lap), "OutLap"]:
                kind = "out"
            else:
                continue
            c = car[(car["t"] >= r["t0"]) & (car["t"] < r["t0"] + pd.Timedelta(seconds=float(r["lap_duration"])))]
            if len(c) < 300:
                continue
            t = (c["t"] - r["t0"]).dt.total_seconds().to_numpy()
            v = savgol_filter(c["speed"].to_numpy() / 3.6, 7, 2)
            traces.append({
                "driver": drv, "lap": lap, "kind": kind, "lap_time": float(r["lap_duration"]),
                "t": t, "v": v, "a": np.gradient(v, t),
                "s": np.r_[0, np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))],
                "throttle": c["throttle"].to_numpy(), "brake": c["brake"].to_numpy(),
                "drs": c["drs"].to_numpy(),
                "mass": sim.m_empty + fuel_kg * (1 - (lap - 0.5) / sim.laps),   # A11: constant burn
            })
    return traces


# --- Step 1: the car -----------------------------------------------------------------

def full_throttle_samples(tr):
    thr = tr["throttle"]
    ok = ((thr >= 99) & (np.r_[thr[1:], 0] >= 99) & (np.r_[0, thr[:-1]] >= 99)
          & (tr["drs"] < 10) & (tr["v"] > FULL_THROTTLE_MIN_KMH / 3.6))
    return tr["v"][ok], tr["a"][ok]


def braking_samples(tr):
    """2nd-4th sample of each braking zone: straight-line braking, before turn-in."""
    b = tr["brake"] > 0
    starts = np.flatnonzero(np.diff(np.r_[0, b.astype(int)]) == 1)
    idx = [i + k for i in starts for k in (1, 2, 3)
           if i + k < len(b) and b[i + k] and tr["v"][i + k] > BRAKE_MIN_KMH / 3.6]
    return tr["v"][idx], tr["a"][idx]


def step1_car(settings, traces):
    print("\nStep 1: fit the car (wheel power, drag, braking) on full-throttle and braking samples")
    pmset = settings["pointmass"]
    crr, rho = pmset["crr"], pmset["rho_kg_m3"]
    clean = [tr for tr in traces if tr["kind"] == "clean"]
    per_driver = {}
    V, A, M = [], [], []
    for drv in DRIVERS:
        own = [tr for tr in clean if tr["driver"] == drv]
        v, a = map(np.concatenate, zip(*(full_throttle_samples(tr) for tr in own)))
        m = np.concatenate([np.full(len(full_throttle_samples(tr)[0]), tr["mass"]) for tr in own])
        per_driver[drv] = pm.fit_power_drag(v, a, m, crr, rho)
        V.append(v), A.append(a), M.append(m)
    V, A, M = map(np.concatenate, (V, A, M))
    power, cda, sd = pm.fit_power_drag(V, A, M, crr, rho)
    print(f"  {len(V)} full-throttle samples (DRS closed, > {FULL_THROTTLE_MIN_KMH} km/h) from "
          f"{len(clean)} clean laps: wheel power {power / 1e3:.0f} kW, CdA {cda:.3f} m^2 "
          f"(Crr fixed at {crr}), residual sd {sd:.0f} N")
    for drv, (p, c, _) in per_driver.items():
        print(f"    {drv}: {p / 1e3:.0f} kW, CdA {c:.3f}")
    powers = np.array([p for p, _, _ in per_driver.values()])
    check(np.ptp(powers) / power < 0.05, f"one car fits every driver: wheel power within "
                                         f"{np.ptp(powers) / power:.1%} across the four")
    vb, ab = map(np.concatenate, zip(*(braking_samples(tr) for tr in clean)))
    ba, bb = pm.fit_braking(vb, ab)
    g300 = (ba + bb * (300 / 3.6) ** 2) / pm.G
    print(f"  braking ({len(vb)} samples): deceleration {ba:.2f} + {bb:.5f} v^2 m/s^2 "
          f"({ba / pm.G:.1f} g at low speed, {g300:.1f} g at 300 km/h)")
    check(ba > 0 and bb >= 0 and 2 < g300 < 6, "braking grows with speed (downforce) and is 2-6 g at 300 km/h")
    found = {"power_kw": power / 1e3, "cda_m2": cda, "brake_a_m_s2": ba, "brake_b_per_m": bb}
    check_settings(found, pmset, list(found), rel=0.02)
    return {"power": power, "cda": cda, "crr": crr, "rho": rho, "brake_a": ba, "brake_b": bb,
            "samples": (V, A, M, vb, ab)}


# --- Step 2: the mini-sim -----------------------------------------------------------------

def lap_car(fit, mass, power=None):
    return pm.Car(mass, fit["power"] if power is None else power, fit["cda"], fit["crr"],
                  fit["rho"], fit["brake_a"], fit["brake_b"])


def lap_straights(fit, tr, match_time=False):
    """The lap's straights, each calibrated to its measured braking-point speed (A34);
    with `match_time` also to its measured time on the throttle (a per-straight power
    change, used only as a robustness check: it is badly conditioned)."""
    car = lap_car(fit, tr["mass"])
    raw = pm.straights_from_trace(tr["s"], tr["v"], tr["throttle"], tr["brake"], t=tr["t"])
    return car, [pm.calibrate(car, st, measured_time(tr, st, True) if match_time else None) for st in raw]


def measured_time(tr, st, on_throttle=False):
    end = st.brake_s[0] if on_throttle and st.brakes else st.s1
    return float(np.interp(end, tr["s"], tr["t"]) - np.interp(st.s0, tr["s"], tr["t"]))


def mid_speed_error(tr, car, st):
    """Held-out check: simulated vs measured speed halfway along the time on the throttle."""
    end = st.brake_s[0] if st.brakes else st.s1
    mid = 0.5 * (st.s0 + end)
    _, _, s, v, _ = pm.run(car, st)
    return float(np.interp(mid, s, v) / np.interp(mid, tr["s"], tr["v"]) - 1)


def curves_for_lap(sim, fit, tr, deploy_mj, k_kw, match_time=False):
    """Deploy, clip and coast curves for one lap, on the common grids. The curves start
    from the car with no MGU-K deploy: the fitted (real-lap) power minus the model's
    nominal deploy spread over the time on the throttle (A35)."""
    car, sts = lap_straights(fit, tr, match_time)
    on_throttle = sum(pm.run(car, st)[4].sum() for st in sts)
    base = lap_car(fit, tr["mass"], fit["power"] - deploy_mj * 1e6 / on_throttle)
    k = k_kw * 1e3
    fuel_rate = sim.settings["rules"]["fuel_flow_max_kg_per_h"] / 3600 * sim.lhv * 1e6     # J/s
    out = {"on_throttle": on_throttle, "straights": sts, "car": base}
    for name, power, epj, grid in (("deploy", k, 1.0, D_GRID), ("clip", -k, 1.0, X_GRID),
                                   ("coast", -base.power, fuel_rate / base.power, F_GRID)):
        E, T, order = pm.energy_curve(base, sts, power, epj, with_order=True)
        E = np.maximum.accumulate(E)
        out[name] = np.interp(grid, E, T, right=np.nan)
        out[name + "_max"] = E[-1]
        out[name + "_order"] = (E, order)
        out[name + "_raw"] = (E, T)
    return out


def joint_error(sim, res, d_mj, x_mj, k_kw):
    """Deploy d_mj on the best stretches and clip x_mj on the cheapest of the others,
    on the same lap: lap-time change vs the sum of the two separate curves."""
    car, sts = res["car"], res["straights"]
    k = k_kw * 1e3
    stretches = [dict() for _ in sts]
    E, order = res["deploy_order"]
    for (kk, b), e in zip(order, E[1:]):
        if e > d_mj:
            break
        stretches[kk][b] = k
    E, order = res["clip_order"]
    for (kk, b), e in zip(order, E[1:]):
        if e > x_mj:
            break
        stretches[kk].setdefault(b, -k)
    joint = sum(pm.run_mixed(car, st, s)[0] - pm.run(car, st)[0] for st, s in zip(sts, stretches))
    separate = np.interp(d_mj, D_GRID, res["deploy"]) + np.interp(x_mj, X_GRID, res["clip"])
    return joint, separate


def step2_minisim(sim, fit, traces):
    print("\nStep 2: point-mass mini-sim along the full-throttle straights of every lap")
    clean = [tr for tr in traces if tr["kind"] == "clean"]
    sample = [tr for drv in DRIVERS for tr in [t for t in clean if t["driver"] == drv][::CURVE_EVERY]]
    t_err, v_err, n_st = [], [], []
    for tr in sample:
        car, sts = lap_straights(fit, tr)
        sim_t = sum(pm.run(car, st)[0] for st in sts)
        t_err.append(sim_t / sum(measured_time(tr, st) for st in sts) - 1)
        v_err += [mid_speed_error(tr, car, st) for st in sts]
        n_st.append(len(sts))
    t_err, v_err = np.array(t_err), np.array(v_err)
    print(f"  {len(sample)} laps (every {CURVE_EVERY}rd clean lap per driver), {min(n_st)}-{max(n_st)} straights each")
    print(f"  simulated vs measured time over the straights (incl. braking): median {np.median(t_err):+.2%}, "
          f"90% within {np.quantile(np.abs(t_err), 0.9):.2%}; speed halfway along: median {np.median(v_err):+.2%}")
    print("  the point mass is slightly slow mid-straight: the real car accelerates harder early and de-rates")
    print("  at the end (its own deploy and clipping), which one constant power cannot show")

    nominal_deploy = min(sim.h_free, sim.h_cap)
    kw = sim.settings["rules"]["mguk_max_kw"]
    t0 = time.time()
    res = [curves_for_lap(sim, fit, tr, nominal_deploy, kw) for tr in sample]
    res26 = [curves_for_lap(sim, fit, tr, nominal_deploy, KW_2026) for tr in sample[::4]]
    print(f"  curves on {len(sample)} laps, {time.time() - t0:.0f} s")
    curves = {name: np.nanmean([r[name] for r in res], axis=0) for name in ("deploy", "clip", "coast")}
    curves.update({name + "_band": np.nanquantile([r[name] for r in res], [0.1, 0.9], axis=0)
                   for name in ("deploy", "clip", "coast")})
    grid26 = np.linspace(0, 9, 46)
    curves["grid_2026"] = grid26
    curves["deploy_2026"] = np.nanmean([np.interp(grid26, *r["deploy_raw"], right=np.nan) for r in res26], axis=0)

    d, x, f = curves["deploy"], curves["clip"], curves["coast"]
    print(f"  deploy: 1 MJ {d[10]:+.2f} s, 2 MJ {d[20]:+.2f} s, 4 MJ {d[40]:+.2f} s "
          f"(placeholder A19: 4 MJ {sim_placeholder(sim).batt_term(4.0):+.2f} s); "
          f"time on the throttle {np.mean([r['on_throttle'] for r in res]):.1f} s/lap")
    print(f"  clip:   0.5 MJ {x[5]:+.3f} s, 1 MJ {x[10]:+.3f} s, 2 MJ {x[20]:+.3f} s")
    print(f"  coast:  2 MJ fuel {f[10]:+.3f} s, 4 MJ {f[20]:+.3f} s, 8 MJ {f[40]:+.3f} s "
          f"(placeholder A16: 10% = {0.1 * sim.fuel_nominal:.1f} MJ costs "
          f"{sim_placeholder(sim).fuel_term(0.9 * sim.fuel_nominal):+.2f} s)")
    check(min(r["deploy_max"] for r in res) > sim.d_cap,
          f"there is room to deploy the FIA's {sim.d_cap:g} MJ on every lap "
          f"(least {min(r['deploy_max'] for r in res):.1f} MJ at {kw:g} kW): the cap binds, not the straights")
    check(np.all(np.diff(d, 2) > -1e-3) and np.all(np.diff(d) < 0),
          "deploy: every MJ saves time, each extra MJ saves less (diminishing returns)")
    check(np.all(np.diff(x, 2) > -1e-3) and np.all(np.diff(f, 2) > -1e-3),
          "clip and coast: each extra MJ costs more than the last (convex)")
    print(f"  first MJ deployed saves {-d[10]:.2f} s; first MJ clipped costs {x[10]:.3f} s: clipping to "
          f"deploy more pays until the harvest cap")

    mass_effect = []
    for tr in sample[:: max(len(sample) // 6, 1)]:
        g = []
        for m in (sim.m_empty, sim.m_empty + sim.fuel0 / sim.lhv):
            r = curves_for_lap(sim, fit, {**tr, "mass": m}, nominal_deploy, kw)
            g.append(r["deploy"][-1])
        mass_effect.append(g[1] / g[0] - 1)
    print(f"  full tank vs empty: 4 MJ deploy gain changes by {np.mean(mass_effect):+.1%} (A37: ignored)")
    check(abs(np.mean(mass_effect)) < 0.1, "deploy gain depends little on car mass (< 10% from full to empty)")

    # robustness: straights made to take exactly the measured time (per-straight power change)
    timed = [curves_for_lap(sim, fit, tr, nominal_deploy, kw, match_time=True) for tr in sample[::8]]
    base = [r for r in res[::8]]
    rel_d = np.array([[a["deploy"][i] / b["deploy"][i] - 1 for i in (20, 40)] for a, b in zip(timed, base)])
    curves["deploy_uncertainty"] = float(rel_d.mean())
    print(f"  robustness: if every straight is made to take exactly the measured time (a per-straight power "
          f"change), the deploy gain at 2-4 MJ is {rel_d.mean():+.0%} on average ({rel_d.min():+.0%} at most, "
          f"{len(timed)} laps); A35 carries this as its uncertainty")
    i_nom = int(round(nominal_deploy * 10))

    def deploy_beats_clip(r):     # next MJ deployed at the nominal vs the last MJ clipped at the cap
        return -(r["deploy"][i_nom + 1] - r["deploy"][i_nom]) > r["clip"][-1] - r["clip"][-2]
    check(all(deploy_beats_clip(r) for r in timed + base),
          f"either way, on every lap a MJ deployed above the nominal {nominal_deploy:g} MJ saves more than the "
          f"last MJ clipped below the {sim.h_cap:g} MJ cap costs, so the optimum always fills the harvest cap")

    joint = [joint_error(sim, r, nominal_deploy, 1.0, kw) for r in res[:10]]
    rel = max(abs(j - s) / abs(s) for j, s in joint)
    check(rel < 0.05, f"deploying {nominal_deploy:g} MJ and clipping 1 MJ on the same lap add up "
                      f"to within {rel:.1%} of the two curves (A35: terms are separate)")
    print(f"  2026 preview ({KW_2026:g} kW, same car, other 2026 rules not applied): "
          f"4 MJ {np.interp(4, grid26, curves['deploy_2026']):+.2f} s, "
          f"8 MJ {np.interp(8, grid26, curves['deploy_2026']):+.2f} s")
    return curves, res, sample


def sim_placeholder(sim):
    return racesim.RaceSim(racesim.override(sim.settings, **{"energy.model": "placeholder"}))


# --- Step 3: harvest from braking, lifting -------------------------------------------------

def free_harvest(tr, fit, k_kw, rear_share):
    """Energy the MGU-K could recover under braking: min(MGU-K power, rear axle's share of
    the braking power) on every braking sample. Braking power excludes drag and rolling."""
    v, a, m = tr["v"], tr["a"], tr["mass"]
    brake_power = np.clip(-m * a - 0.5 * fit["rho"] * fit["cda"] * v ** 2 - fit["crr"] * m * pm.G, 0, None) * v
    dt = np.gradient(tr["t"])
    return float(np.sum(np.minimum(k_kw * 1e3, rear_share * brake_power) * dt * (tr["brake"] > 0)) / 1e6)


def lift_time(tr):
    """Time off the throttle and the brake above 200 km/h just before a braking zone."""
    b = tr["brake"] > 0
    soon = np.array([b[i:i + 5].any() for i in range(len(b))])
    lift = (tr["throttle"] < 1) & ~b & (tr["v"] > 200 / 3.6) & soon
    return float(np.sum(np.gradient(tr["t"]) * lift))


def step3_harvest(settings, sim, fit, traces):
    print("\nStep 3: free harvest from braking, the pit-lap bonus, and how much the nominal lap lifts")
    rules, share = settings["rules"], settings["pointmass"]["rear_brake_share"]
    kw = rules["mguk_max_kw"]
    df = pd.DataFrame([{"driver": tr["driver"], "lap": tr["lap"], "kind": tr["kind"],
                        "free": free_harvest(tr, fit, kw, share),
                        "free_all": free_harvest(tr, fit, kw, 1.0),
                        "free_half": free_harvest(tr, fit, kw, 0.5),
                        "free_2026": free_harvest(tr, fit, KW_2026, share),
                        "lift": lift_time(tr)} for tr in traces])
    clean = df[df["kind"] == "clean"]
    own = clean.groupby("driver")["free"].median()
    bonus = {k: (df[df["kind"] == k]["free"] - df[df["kind"] == k]["driver"].map(own)).median()
             for k in ("in", "out")}
    h_free = clean["free"].median()
    cap = rules["mguk_to_es_max_mj_per_lap"]
    print(f"  normal lap ({len(clean)} clean laps, rear share {share}): median {h_free:.2f} MJ "
          f"(10-90% {clean['free'].quantile(0.1):.2f}-{clean['free'].quantile(0.9):.2f}); "
          f"rear share 0.5: {clean['free_half'].median():.2f}, all braking power: {clean['free_all'].median():.2f}")
    print(f"  in-laps ({(df['kind'] == 'in').sum()}): {bonus['in']:+.2f} MJ, out-laps "
          f"({(df['kind'] == 'out').sum()}): {bonus['out']:+.2f} MJ against the driver's normal lap")
    at_cap = (clean["free"] >= cap).mean()
    print(f"  FIA harvest cap {cap:g} MJ/lap (MGU-K -> ES): braking alone reaches it on {at_cap:.0%} of laps; "
          f"with 5% electrical loss the median would be {0.95 * h_free:.2f} MJ")
    fuel_rate = rules["fuel_flow_max_kg_per_h"] / 3600 * sim.lhv          # MJ/s at full flow
    lift = clean["lift"].mean()
    print(f"  lift before braking: mean {lift:.2f} s/lap (median {clean['lift'].median():.2f}): the nominal "
          f"lap saves {lift * fuel_rate:.2f} MJ of fuel ({lift * fuel_rate / sim.fuel_nominal:.1%} of nominal), "
          f"so using more than nominal is barely possible")
    print(f"  2026 preview ({KW_2026:g} kW): braking could recover {clean['free_2026'].median():.1f} MJ/lap")
    found = {"free_harvest_mj": h_free, "inlap_bonus_mj": bonus["in"], "outlap_bonus_mj": bonus["out"]}
    check_settings(found, settings["energy"]["battery"], list(found), rel=0.03)
    check_settings({"lift_nominal_mj": lift * fuel_rate}, settings["energy"]["fuel"], ["lift_nominal_mj"], rel=0.03)
    return df


# --- Step 4: closed forms ------------------------------------------------------------------

def step4_forms(settings, sim, curves):
    print("\nStep 4: closed forms for the simulator")
    d, x, f = curves["deploy"], curves["clip"], curves["coast"]
    (a, b), _ = curve_fit(lambda D, a, b: -a * (1 - np.exp(-D / b)), D_GRID, d, p0=(3, 3))
    (c1, c2), _ = curve_fit(lambda X, c1, c2: c1 * X + c2 * X ** 2, X_GRID, x, p0=(0.01, 0.1),
                            bounds=([0, 0], [np.inf, np.inf]))
    lift0 = settings["energy"]["fuel"]["lift_nominal_mj"]
    ok = np.isfinite(f)
    (k, p), _ = curve_fit(lambda F, k, p: k * ((lift0 + F) ** p - lift0 ** p), F_GRID[ok], f[ok], p0=(0.005, 2))
    err = {"deploy": np.max(np.abs(-a * (1 - np.exp(-D_GRID / b)) - d)),
           "clip": np.max(np.abs(c1 * X_GRID + c2 * X_GRID ** 2 - x)),
           "coast": np.max(np.abs(k * ((lift0 + F_GRID[ok]) ** p - lift0 ** p) - f[ok]))}
    print(f"  deploy gain  {a:.3f} * (1 - exp(-D / {b:.3f}))   max error {err['deploy']:.3f} s")
    print(f"  clip cost    {c1:.4f} X + {c2:.4f} X^2            max error {err['clip']:.3f} s")
    print(f"  coast cost   {k:.5f} * F^{p:.3f} (F from flat out)  max error {err['coast']:.3f} s")
    check(max(err.values()) < 0.03, "every closed form is within 0.03 s of its curve")
    check(p > 2, "coast exponent > 2, so the cost is smooth enough for IPOPT at the 'no lifting' bound")
    check_settings({"deploy_a_s": a, "deploy_b_mj": b, "clip_c1_s_per_mj": c1, "clip_c2_s_per_mj2": c2},
                   settings["energy"]["battery"],
                   ["deploy_a_s", "deploy_b_mj", "clip_c1_s_per_mj", "clip_c2_s_per_mj2"], rel=0.02)
    check_settings({"coast_k_s": k, "coast_p": p}, settings["energy"]["fuel"], ["coast_k_s", "coast_p"], rel=0.02)
    m = sim.fuel_nominal
    print(f"  in the simulator: marginal value of battery energy at the nominal {sim.split(0, sim.h_free)[0]:.2f} MJ "
          f"deploy: {a / b * np.exp(-sim.split(0, sim.h_free)[0] / b):.2f} s/MJ (placeholder 0.25); fuel band now "
          f"{sim.f_min / m:.0%}-{sim.f_max / m:.1%} of nominal (was 90-110%)")


# --- Step 5: the exact optimiser with the physics terms --------------------------------------

def energy_table(sim, compounds, pits, fuel, batt):
    _, rows = sim.simulate(compounds, pits, fuel, batt)
    t = pd.DataFrame(rows)
    t["soc"] = t["batt_mj"]
    t["kind"] = ["in" if l in pits else "out" if l - 1 in pits else "" for l in t["lap"]]
    return t


def print_stops(t, pits):
    print("    lap  kind  deploy  free  extra   net   battery after")
    for p in pits:
        for _, r in t[(t["lap"] >= p - 1) & (t["lap"] <= p + 2)].iterrows():
            print(f"    {r['lap']:>3}  {r['kind']:<4}  {r['deploy']:6.2f} {r['harvest_free']:5.2f} "
                  f"{r['harvest_extra']:6.2f} {r['u_b']:+6.2f}  {r['soc']:6.2f}")
    normal = t[t["kind"] == ""]
    print(f"    median normal lap: deploy {normal['deploy'].median():.2f}, net {normal['u_b'].median():+.2f}")


def step5_optimum(sim, w2_data, w2):
    print(f"\nStep 5: exact optimiser, every legal 1-{MAX_STOPS} stop plan, physics energy terms")
    plans = list(sim.plans(MAX_STOPS))
    rng = np.random.default_rng(0)
    sample = [plans[i] for i in rng.choice(len(plans), 10, replace=False)]
    T = sim.energy_problem()[2]
    gap = 0.0
    for c, p in sample:
        t_nom = float(T(sim.energy_start(c, p), sim.plan_params(c, p))) + sim.pit_loss * len(p)
        t, fuel, batt, _ = sim.optimal_energy(c, p)
        gap = max(gap, abs(t_nom - sim.race_time(c, p, *sim.nominal_alloc())),
                  abs(t - sim.race_time(c, p, fuel, batt)), abs(t - sim.simulate(c, p, fuel, batt)[0]))
    check(gap < TOL, f"solver objective = simulator on 10 random plans, nominal and optimal energy "
                     f"(the simulator's own deploy/harvest split matches the solver's; max diff {gap:.1e} s)")
    worse = [sim.optimal_energy(c, p)[0] - sim.best_energy(c, p)[0] for c, p in sample[:3]]
    check(max(worse) < TOL, f"IPOPT never worse than SciPy on the net-battery form ({max(worse):+.1e} s)")

    t0 = time.time()
    best, table = sim.exact(MAX_STOPS)
    print(f"  {len(table)} plans in {time.time() - t0:.0f} s")
    statuses = Counter(s for *_, s in table)
    check(statuses[SUCCESS] == len(table), f"every solve succeeded ({dict(statuses)})")
    df = pd.DataFrame(table, columns=["compounds", "pits", "time", "status"]).sort_values("time").reset_index(drop=True)
    t_best, compounds, pits, fuel, batt = best
    df["gap"] = df["time"] - t_best
    df["plan"] = [describe(c, p, sim.laps) for c, p in zip(df["compounds"], df["pits"])]
    old = json.load(open(os.path.join(RESULTS_DIR, f"optimum_{sim.settings['name']}.json")))
    print(f"  OPTIMUM: {describe(compounds, pits, sim.laps)}, {t_best:.3f} s "
          f"(placeholder energy, Weeks 7-8: {old['plan']}, {old['race_time_s']:.3f} s)")
    for _, r in df.head(5).iterrows():
        print(f"    {r['plan']:<20} {r['time']:10.3f} s  {r['gap']:+7.3f} s")
    t_nom = sim.race_time(compounds, pits, *sim.nominal_alloc())
    print(f"  optimising energy saves {t_nom - t_best:.2f} s on it (Weeks 7-8: 3.49 s)")
    soc = sim.b_cap - np.cumsum(batt)
    check(np.all(fuel >= sim.f_min - TOL) and np.all(fuel <= sim.f_max + TOL) and fuel.sum() <= sim.fuel0 + TOL
          and np.all(soc >= -TOL) and np.all(soc <= sim.b_cap + TOL), "solution obeys every limit")

    t = energy_table(sim, compounds, pits, fuel, batt)
    print("  battery around the stops (MJ):")
    print_stops(t, pits)
    normal = t[t["kind"] == ""]
    inl, outl = t[t["kind"] == "in"], t[t["kind"] == "out"]
    print("  reviewer's expectation, tested:")
    report((inl["u_b"] < normal["u_b"].median() - 0.05).all(), "the in-laps charge more than a normal lap")
    report((outl["deploy"] > normal["deploy"].median() + 0.05).all(), "the out-laps deploy more than a normal lap")
    check((t["harvest_free"] + t["harvest_extra"] >= sim.h_cap - 1e-6).all(),
          f"why not: every lap already harvests the FIA's {sim.h_cap:g} MJ (braking alone gives "
          f"{sim.h_free:.2f}), so the pit laps' extra braking energy has nowhere to go")
    print(f"  deploy: {normal['deploy'].min():.3f}-{normal['deploy'].max():.3f} MJ on every normal lap; "
          f"fuel {100 * fuel.min() / sim.fuel_nominal:.1f}-{100 * fuel.max() / sim.fuel_nominal:.1f}% of nominal")

    laps, results, race_laps = w2_data
    rows = []
    for pos, drv, c, p, _ in w2.actual_strategies(results, laps, race_laps):
        legal = c[0] == sim.start_compound and sim.legal(c, p)
        rows.append({"pos": pos, "driver": drv, "plan": describe(c, p, sim.laps),
                     "lost": sim.optimal_energy(c, p)[0] - t_best if legal else np.nan})
    real = pd.DataFrame(rows).dropna()
    print(f"  real {YEAR} strategies vs the new benchmark: median {real['lost'].median():.2f} s lost "
          f"(Weeks 7-8: 1.6 s), {real['lost'].min():.2f}-{real['lost'].max():.2f} s")
    check((real["lost"] >= -TOL).all(), "no real strategy beats the benchmark")
    return best, df, t


# --- Step 6: sensitivity, a harvest cap that doesn't bind ------------------------------------

def step6_sensitivity(settings, sim, best):
    print(f"\nStep 6: sensitivity, harvest cap raised from 2 to {SENS_HARVEST_CAP:g} MJ/lap so it no longer binds")
    sens = racesim.RaceSim(racesim.override(settings, **{"rules.mguk_to_es_max_mj_per_lap": SENS_HARVEST_CAP}))
    sens.b_ref = sim.b_ref          # the base lap was measured under the real cap: keep its zero point
    _, compounds, pits, *_ = best
    t_s, fuel, batt, status = sens.optimal_energy(compounds, pits)
    check(status == SUCCESS and t_s < best[0], f"benchmark plan solved: {t_s:.3f} s, {best[0] - t_s:.1f} s faster "
                                               f"than under the real cap (clipping the extra pays)")
    t = energy_table(sens, compounds, pits, fuel, batt)
    print_stops(t, pits)
    normal = t[t["kind"] == ""]
    inl, outl = t[t["kind"] == "in"], t[t["kind"] == "out"]
    check(((normal["harvest_free"] + normal["harvest_extra"]) < sens.h_cap - 0.1).all(),
          "the cap no longer binds: every normal lap stops harvesting below it")
    check((inl["u_b"] < normal["u_b"].median() - 0.5 * sens.h_in).all(),
          f"the in-laps charge more: the {sens.h_in:.2f} MJ of extra free braking energy is kept")
    report((outl["deploy"] > normal["deploy"].median() + 0.05).all(), "the out-laps deploy more than a normal lap")
    print(f"  deploy {normal['deploy'].min():.2f}-{normal['deploy'].max():.2f} MJ on every normal lap: the "
          f"energy kept on the in-lap is spread over all later laps, because a MJ is worth the same on any lap")
    return sens, t


# --- Step 7: convexity ------------------------------------------------------------------------

def random_start(sim, compounds, pits, rng):
    n = sim.laps
    free = np.minimum(sim.lap_free(pits), sim.h_cap)
    h = rng.uniform(0, free)
    return np.concatenate([rng.uniform(sim.f_min, sim.f_max, n), rng.uniform(0, sim.d_cap, n), h,
                           rng.uniform(0, sim.h_cap - h)])


def step7_convexity(sims, best, df):
    print(f"\nStep 7: convexity (A31): {N_STARTS} random starting points on {N_PLANS_CONVEX} plans per case")
    rng = np.random.default_rng(1)
    plans = [(best[1], best[2])] + [tuple(df.loc[i, ["compounds", "pits"]]) for i in
                                    rng.choice(len(df), N_PLANS_CONVEX - 1, replace=False)]
    for name, sim in sims:
        spread = 0.0
        for c, p in plans:
            ts = [sim.optimal_energy(c, p, random_start(sim, c, p, rng)) for _ in range(N_STARTS)]
            assert all(s == SUCCESS for *_, s in ts)
            spread = max(spread, np.ptp([x[0] for x in ts]))
        check(spread < TOL, f"{name}: every start reaches the same race time (max spread {spread:.1e} s)")
    print("  as expected: deploy gain is concave, clip and coast costs convex, every limit linear,")
    print("  so for a fixed plan the problem is still convex and the exact optimum is still exact")


# --- Output ---------------------------------------------------------------------------------

def save(settings, sim, best, df, curves):
    t_best, compounds, pits, fuel, batt = best
    d, h, x = sim.split(batt, sim.lap_free(pits))
    out = {
        "era": settings["name"], "race": settings["race"]["calibration_race"],
        "energy_model": sim.model,
        "search": f"every legal plan with 1-{MAX_STOPS} stops, exact energy per plan (9th_10th_week.py)",
        "race_time_s": round(t_best, 6), "plan": describe(compounds, pits, sim.laps),
        "compounds": list(compounds), "pit_laps": list(pits),
        "fuel_mj_per_lap": np.round(fuel, 6).tolist(), "battery_mj_per_lap": np.round(batt, 6).tolist(),
        "deploy_mj_per_lap": np.round(d, 6).tolist(), "harvest_free_mj_per_lap": np.round(h, 6).tolist(),
        "harvest_extra_mj_per_lap": np.round(x, 6).tolist(),
    }
    name = f"{settings['name']}_physics"
    with open(os.path.join(RESULTS_DIR, f"optimum_{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    table = df[["plan", "time", "gap"]].assign(pits=df["pits"].map(lambda p: " ".join(map(str, p))))
    table.round(6).to_csv(os.path.join(RESULTS_DIR, f"all_plans_{name}.csv"), index=False)
    pd.DataFrame({"energy_mj": D_GRID, "deploy_s": curves["deploy"]}).merge(
        pd.DataFrame({"energy_mj": X_GRID, "clip_s": curves["clip"]}), how="outer").merge(
        pd.DataFrame({"energy_mj": F_GRID, "coast_s": curves["coast"]}), how="outer").round(5).to_csv(
        os.path.join(RESULTS_DIR, f"energy_curves_{settings['name']}.csv"), index=False)
    print(f"\nBenchmark saved: results/optimum_{name}.json, all_plans_{name}.csv, "
          f"energy_curves_{settings['name']}.csv")


def plot_car(fit, traces):
    V, A, M, vb, ab = fit["samples"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    bins = np.arange(FULL_THROTTLE_MIN_KMH, 330, 10)
    kmh = V * 3.6
    mid = 0.5 * (bins[1:] + bins[:-1])
    mean = [A[(kmh >= lo) & (kmh < hi)].mean() if ((kmh >= lo) & (kmh < hi)).sum() > 20 else np.nan
            for lo, hi in zip(bins[:-1], bins[1:])]
    ax.plot(mid, mean, "o", color=C1, markersize=5, label="measured (mean per 10 km/h)")
    v = np.linspace(FULL_THROTTLE_MIN_KMH, 325, 100) / 3.6
    m = M.mean()
    model = (fit["power"] / v - 0.5 * fit["rho"] * fit["cda"] * v ** 2 - fit["crr"] * m * pm.G) / m
    ax.plot(v * 3.6, model, color=INK, linewidth=2,
            label=f"fit: {fit['power'] / 1e3:.0f} kW, CdA {fit['cda']:.2f} m²")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_xlabel("Speed (km/h)", color=MUTED)
    style(ax, "Full-throttle acceleration (m/s²)")
    ax = axes[1]
    ax.plot(vb * 3.6, -ab, ".", color=C2, markersize=2, alpha=0.4, label="measured")
    v = np.linspace(BRAKE_MIN_KMH, 330, 100) / 3.6
    ax.plot(v * 3.6, fit["brake_a"] + fit["brake_b"] * v ** 2, color=INK, linewidth=2, label="fit")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_xlabel("Speed (km/h)", color=MUTED)
    style(ax, "Straight-line braking deceleration (m/s²)")
    ax = axes[2]
    tr = next(t for t in traces if t["kind"] == "clean" and t["driver"] == DRIVERS[0])
    car, sts = lap_straights(fit, tr)
    ax.plot(tr["s"], tr["v"] * 3.6, color=MUTED, linewidth=1.2, label="measured")
    for i, st in enumerate(sts):
        _, _, s, vv, _ = pm.run(car, st)
        ax.plot(s, vv * 3.6, color=C1, linewidth=2, label="mini-sim" if i == 0 else None)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_xlabel("Distance (m)", color=MUTED)
    style(ax, f"Speed (km/h), {tr['driver']} lap {tr['lap']}: straights simulated")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week9_car_fit.png"), dpi=150)
    plt.close(fig)


def plot_curves(sim, curves):
    s = sim.settings["energy"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ph = sim_placeholder(sim)
    for ax, grid, key, color, title, fit, old in (
            (axes[0], D_GRID, "deploy", C1, "Deploy: lap time vs battery energy used (s)",
             -s["battery"]["deploy_a_s"] * (1 - np.exp(-D_GRID / s["battery"]["deploy_b_mj"])),
             ph.batt_term(D_GRID)),
            (axes[1], X_GRID, "clip", C2, "Clip: lap time vs extra energy harvested (s)",
             s["battery"]["clip_c1_s_per_mj"] * X_GRID + s["battery"]["clip_c2_s_per_mj2"] * X_GRID ** 2, None),
            (axes[2], F_GRID, "coast", C3, "Coast: lap time vs fuel saved (s)",
             s["fuel"]["coast_k_s"] * ((s["fuel"]["lift_nominal_mj"] + F_GRID) ** s["fuel"]["coast_p"]
                                       - s["fuel"]["lift_nominal_mj"] ** s["fuel"]["coast_p"]),
             ph.fuel_term(ph.fuel_nominal - F_GRID))):
        lo, hi = curves[key + "_band"]
        ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, curves[key], color=color, linewidth=2, label="mini-sim (mean, 10-90% of laps)")
        ax.plot(grid, fit, color=INK, linewidth=1.2, linestyle="--", label="closed form in the simulator")
        if old is not None:
            ax.plot(grid, old, color=MUTED, linewidth=1.2, linestyle=":", label="Weeks 3-8 placeholder")
        ax.set_xlabel("MJ per lap", color=MUTED)
        style(ax, title)
    g26 = curves["grid_2026"]
    axes[0].plot(g26, curves["deploy_2026"], color=C4, linewidth=2, label=f"{KW_2026:g} kW preview (2026)")
    axes[0].axvline(4, color=MUTED, linewidth=0.8)
    axes[0].annotate("FIA cap 4 MJ", (4, 0), xytext=(4, -10), textcoords="offset points", fontsize=8, color=MUTED)
    axes[1].axvline(2, color=MUTED, linewidth=0.8)
    for ax in axes:
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left" if ax is axes[0] else "upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week9_energy_curves.png"), dpi=150)
    plt.close(fig)


def plot_battery(cases, pits):
    fig, axes = plt.subplots(1, len(cases), figsize=(7 * len(cases), 4), sharey=True)
    for ax, (title, t) in zip(axes, cases):
        lap = t["lap"]
        ax.plot(lap, t["deploy"], color=C1, linewidth=2, label="deploy")
        ax.plot(lap, t["harvest_free"] + t["harvest_extra"], color=C2, linewidth=2, label="harvest (free + clip)")
        ax.plot(lap, t["harvest_free"], color=C2, linewidth=1, linestyle="--", label="of which free (braking)")
        ax.plot(lap, t["soc"], color=C4, linewidth=1.5, label="battery after the lap")
        for p in pits:
            ax.axvline(p, color=MUTED, linewidth=0.8, linestyle=":")
        ax.annotate("in-laps", (pits[0], 0.1), xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_xlabel("Lap", color=MUTED)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", ncol=2)
        ax.set_ylim(0, 5.2)
        style(ax, title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week10_battery.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    settings = racesim.load_settings(SETTINGS)
    print(f"Era settings: {settings['name']} ({settings['race']['calibration_race']}), "
          f"energy model {settings['energy']['model']}")
    sim = racesim.RaceSim(settings)
    assert sim.model == "physics"
    w2 = load_week2()
    laps, results, race_laps, clean, _ = w2.prepare(YEAR)
    traces = load_traces(w2, laps, clean, sim)
    print(f"Telemetry: {len(traces)} green-flag laps of {', '.join(DRIVERS)} from OpenF1 car_data "
          f"({Counter(tr['kind'] for tr in traces)})")
    fit = step1_car(settings, traces)
    curves, res, sample = step2_minisim(sim, fit, traces)
    step3_harvest(settings, sim, fit, traces)
    step4_forms(settings, sim, curves)
    best, df, t_main = step5_optimum(sim, (laps, results, race_laps), w2)
    sens, t_sens = step6_sensitivity(settings, sim, best)
    step7_convexity([("FIA caps", sim), (f"harvest cap {SENS_HARVEST_CAP:g} MJ", sens)], best, df)
    save(settings, sim, best, df, curves)
    plot_car(fit, traces)
    plot_curves(sim, curves)
    plot_battery([("FIA caps: the 2 MJ harvest cap binds every lap", t_main),
                  (f"Sensitivity: harvest cap {SENS_HARVEST_CAP:g} MJ/lap", t_sens)], best[2])
    print(f"\nAll checks passed. Figures in {FIG_DIR}")
