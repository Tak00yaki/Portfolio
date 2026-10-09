"""Weeks 3-4: grow the Week 2 model into the ETH model, one feature at a time.

Each step is checked against the step before it and plotted (figures/week3_*.png):

1. Fuel: the car gets lighter each lap. Refit Week 2's lap trend with the fuel
   effect fixed, so the leftover is track evolution. Check: pit-plan rankings are
   unchanged from Week 2 (a fuel effect linear in lap number can't change them).
2. Tyre wear depends on car mass (ETH Eq. 26). Check: beta = 0 reproduces step 1.
3. Fuel and battery energy per lap become decisions with their limits. Check: the
   nominal allocation reproduces step 2, and the optimised one is never slower.
4. Gymnasium environment. Check: gymnasium's env checker passes, and replaying the
   reference strategy through the environment gives the simulator's race time.

All era numbers come from eras/legacy_2022_25.toml; assumptions are in ASSUMPTIONS.md.
"""

import importlib.util
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402

import racesim  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
SETTINGS = os.path.join(HERE, "eras", "legacy_2022_25.toml")
N_RANDOM = 300           # random-policy races for the environment baseline
RANDOM_STOPS = 2         # random policy pits with probability RANDOM_STOPS / laps each lap
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


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK)
    ax.plot(x[-1], y[-1], "o", color=color, markersize=4)


def describe(compounds, pits, laps):
    bounds = (0, *pits, laps)
    return " -> ".join(f"{c[0]}{bounds[i + 1] - bounds[i]}" for i, c in enumerate(compounds))


def check(cond, msg):
    assert cond, msg
    print(f"  [ok] {msg}")


# --- Step 1: fuel ------------------------------------------------------------

def step1_fuel(settings, w2):
    print("\nStep 1: fuel mass")
    car, laps = settings["car"], settings["race"]["laps"]
    _, _, race_laps, _, fit_laps = w2.prepare(2024)
    burn = car["fuel_start_kg"] / race_laps
    corrected = fit_laps.copy()
    # Fuel on board mid-lap, assuming a constant burn rate (ASSUMPTIONS A11)
    corrected["LapTimeS"] -= car["mass_time_s_per_kg"] * (
        car["fuel_start_kg"] - burn * (corrected["LapNumber"] - 0.5))
    params, se, track, track_se, offsets, _ = w2.fit_tyre_model(corrected)
    base = float(np.mean(list(offsets.values())))
    print(f"  fuel effect fixed at {car['mass_time_s_per_kg']} s/kg x {burn:.2f} kg/lap "
          f"= {car['mass_time_s_per_kg'] * burn:.4f} s/lap")
    print(f"  leftover track evolution: {track:+.5f} +/- {track_se:.5f} s/lap "
          f"(negative = track getting faster)")
    print(f"  base lap (empty tank, lap 0): {base:.3f} s")

    t = settings["track"]
    check(abs(t["base_lap_s"] - base) < 1e-3 and abs(t["evolution_s_per_lap"] - track) < 1e-5,
          "settings file matches this refit (base lap, track evolution)")
    for c, (k0, k1) in params.items():
        s = settings["tyres"][c]
        check(abs(s["k0"] - k0) < 1e-3 and abs(s["k1"] - k1) < 1e-3,
              f"settings file matches the refit for {c.lower()} (k0, k1)")
    check(track < 0 and abs(track) < 0.02,
          "leftover track term is small and the track gets faster")

    # Pit-plan rankings must match Week 2's model with the same tyre numbers
    sim = racesim.RaceSim(racesim.override(settings, **{"tyres.wear_mass_exponent": 0.0}))
    week2 = w2.RaceModel(sim.tyre, 0.0, sim.pit_loss, laps,
                         {c: sim.start_age if c == sim.start_compound else 0 for c in sim.tyre})
    rng = np.random.default_rng(0)
    ref = (("SOFT", "HARD", "HARD"), (15, 36))
    diffs = []
    for _ in range(200):
        n = int(rng.integers(1, 4))
        pits = tuple(sorted(rng.choice(np.arange(1, laps), n, replace=False).tolist()))
        comps = ("SOFT", *rng.choice(list(sim.tyre), n).tolist())
        if len(set(comps)) < 2:
            continue
        d_sim = sim.simulate(comps, pits)[0] - sim.simulate(*ref)[0]
        d_w2 = week2.race_time(comps, pits) - week2.race_time(*ref)
        diffs.append(abs(d_sim - d_w2))
    check(max(diffs) < TOL, f"time differences between {len(diffs)} random plans equal Week 2's")
    t1, comps, pits = sim.best_plan()
    print(f"  best plan: {describe(comps, pits, laps)}  {t1:.2f} s")
    w2_best = week2.best_plan(sim.start_compound)
    check(abs((t1 - sim.simulate(*ref)[0]) - (w2_best[0] - week2.race_time(*ref))) < TOL,
          f"best plan is as fast as Week 2's optimiser's ({describe(*w2_best[1:], laps)})")

    _, rows = sim.simulate(comps, pits)
    lap = np.array([r["lap"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for key, color, name in (("fuel mass", C1, "Fuel mass"), ("tyres", C2, "Tyre loss"),
                             ("track", C3, "Track evolution")):
        y = np.array([r[key] for r in rows])
        ax.plot(lap, y, color=color, linewidth=2)
        label_end(ax, lap, y, name, color)
    style(ax, f"Step 1: lap-time components, best plan {describe(comps, pits, laps)} (s)")
    ax.set_xlabel("Lap", color=MUTED)
    ax.set_xlim(0, laps + 9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week3_step1_fuel.png"), dpi=150)
    plt.close(fig)
    return sim, t1, (comps, pits)


# --- Step 2: mass-dependent tyre wear --------------------------------------------

def step2_wear(settings, step1_sim, step1_plan):
    print("\nStep 2: tyre wear depends on car mass")
    sim = racesim.RaceSim(settings)
    laps = sim.laps
    check(abs(step1_sim.simulate(*step1_plan)[0]
              - racesim.RaceSim(racesim.override(settings, **{"tyres.wear_mass_exponent": 0.0}))
              .simulate(*step1_plan)[0]) < TOL, "beta = 0 reproduces step 1")
    for plan in (step1_plan, (("SOFT", "MEDIUM", "HARD"), (12, 30))):
        fuel, batt = sim.nominal_alloc()
        check(abs(sim.simulate(*plan)[0] - sim.race_time(*plan, fuel, batt)) < TOL,
              f"vectorised race time equals lap-by-lap simulation ({describe(*plan, laps)})")
    t2, comps, pits = sim.best_plan()
    t_old = sim.simulate(*step1_plan)[0]
    print(f"  beta = {sim.beta}: best plan {describe(comps, pits, laps)}  {t2:.2f} s "
          f"(step-1 plan now {t_old:.2f} s, {t_old - t2:+.2f} s)")
    check(t2 <= t_old + TOL, "best plan is no slower than the step-1 plan in the new model")

    _, rows = sim.simulate(comps, pits)
    _, rows0 = step1_sim.simulate(comps, pits)
    lap = np.arange(1, laps + 1)
    mass = np.array([r["mass"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    rate = 1 + sim.beta * (mass / sim.m_ref - 1)
    axes[0].plot(lap, rate, color=C1, linewidth=2)
    axes[0].axhline(1, color=MUTED, linewidth=1, linestyle="--")
    label_end(axes[0], lap, rate, f"beta = {sim.beta:g}", C1)
    style(axes[0], "Wear added per lap (laps at reference mass)")
    for r, color, name in ((rows0, C2, "beta = 0 (step 1)"), (rows, C1, f"beta = {sim.beta:g}")):
        y = np.array([x["tyres"] for x in r])
        axes[1].plot(lap, y, color=color, linewidth=2)
        label_end(axes[1], lap, y, name, color)
    style(axes[1], f"Tyre loss per lap, plan {describe(comps, pits, laps)} (s)")
    for ax in axes:
        ax.set_xlabel("Lap", color=MUTED)
        ax.set_xlim(0, laps + 10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week3_step2_wear.png"), dpi=150)
    plt.close(fig)
    return sim, t2, (comps, pits)


# --- Step 3: energy as decisions ---------------------------------------------------

def step3_energy(sim, t2, plan):
    print("\nStep 3: fuel and battery energy per lap as decisions")
    fuel, batt = sim.nominal_alloc()
    check(abs(sim.race_time(*plan, fuel, batt) - t2) < TOL,
          "nominal allocation (constant fuel, battery net zero) reproduces step 2")
    t_e, f_opt, b_opt = sim.best_energy(*plan)
    print(f"  best energy for plan {describe(*plan, sim.laps)}: {t_e:.2f} s ({t_e - t2:+.2f} s)")
    check(t_e <= t2 + TOL, "optimised allocation is never slower than nominal")
    soc = sim.b_cap - np.cumsum(b_opt)
    check(f_opt.sum() <= sim.fuel0 + 1e-6 and soc.min() >= -1e-6 and soc.max() <= sim.b_cap + 1e-6,
          "fuel total and battery limits hold")
    t_sim = sim.simulate(*plan, f_opt, b_opt)[0]
    check(abs(t_sim - t_e) < 1e-5, "lap-by-lap simulation agrees (no clipping needed)")

    t_ref, ref_plan, ref_f, ref_b = sim.reference()
    print(f"  reference (plan and energy together): {describe(*ref_plan, sim.laps)}  "
          f"{t_ref:.2f} s ({t_ref - t2:+.2f} s vs step 2)")

    lap = np.arange(1, sim.laps + 1)
    _, rows = sim.simulate(*ref_plan, ref_f, ref_b)
    _, rows_nom = sim.simulate(*ref_plan)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    ax = axes[0, 0]
    ax.plot(lap, ref_f / sim.fuel_nominal, color=C1, linewidth=2)
    ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
    ax.axhline(sim.f_min / sim.fuel_nominal, color=MUTED, linewidth=0.8, linestyle=":")
    ax.axhline(sim.f_max / sim.fuel_nominal, color=MUTED, linewidth=0.8, linestyle=":")
    style(ax, "Fuel used per lap (x nominal; dotted = limits)")
    ax = axes[0, 1]
    ax.plot(lap, ref_b, color=C2, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    style(ax, "Net battery energy per lap (MJ, + = discharge)")
    ax = axes[1, 0]
    ax.plot(lap, [r["batt_mj"] for r in rows], color=C2, linewidth=2)
    ax.set_ylim(-0.2, sim.b_cap + 0.2)
    style(ax, f"Battery energy left (MJ, capacity {sim.b_cap:g})")
    ax = axes[1, 1]
    gain = np.array([r["time"] for r in rows]) - np.array([r["time"] for r in rows_nom])
    ax.plot(lap, gain, color=C3, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    style(ax, f"Lap time vs nominal energy (s; total {gain.sum():+.2f} s)")
    for ax in axes[1]:
        ax.set_xlabel("Lap", color=MUTED)
    fig.suptitle(f"Step 3: reference energy use, plan {describe(*ref_plan, sim.laps)}",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week3_step3_energy.png"), dpi=150)
    plt.close(fig)
    return t_ref, ref_plan, ref_f, ref_b


# --- Step 4: Gymnasium environment ---------------------------------------------------

def step4_env(settings, t_ref, ref_plan, ref_f, ref_b, t2, plan2):
    print("\nStep 4: Gymnasium environment")
    env = racesim.F1StrategyEnv(settings)
    check_env(env, skip_render_check=True)
    print("  [ok] gymnasium env checker passes")

    t_replay, infos = racesim.run_episode(env, racesim.plan_policy(*ref_plan, ref_f, ref_b))
    check(abs(t_replay - t_ref) < 1e-3, f"replaying the reference gives its race time "
          f"({t_replay:.3f} vs {t_ref:.3f} s)")
    check(not any(i["forced_change"] for i in infos), "reference needs no forced compound change")
    t_nom, _ = racesim.run_episode(env, racesim.plan_policy(*plan2))
    check(abs(t_nom - t2) < 1e-3, "replaying the step-2 plan with nominal energy gives step 2's time")

    # "Never pit unless forced": also checks the forced compound change
    t_lazy, infos = racesim.run_episode(env, lambda obs, e: e.encode(e.sim.fuel_nominal, 0.0, None))
    forced_laps = [i + 1 for i, x in enumerate(infos) if x["forced_change"]]
    check(forced_laps == [env.sim.laps - env.sim.force_laps_left],
          f"a car that never pits is forced to change compound on lap {forced_laps}")
    print(f"  no-stop-unless-forced heuristic: {t_lazy - t_ref:+.1f} s vs reference")

    # Random policy: random energy every lap, a stop to a random compound now and then
    rng = np.random.default_rng(0)
    p_stop = RANDOM_STOPS / env.sim.laps

    def random_policy(obs, e):
        F, B = rng.uniform(0, 1), rng.uniform(-1, 1)
        P = rng.uniform(0.25, 1) if rng.random() < p_stop else 0.0
        return np.array([F, B, P], np.float32)

    random_times, forced = [], 0
    for i in range(N_RANDOM):
        t, infos = racesim.run_episode(env, random_policy, seed=i)
        random_times.append(t)
        forced += any(x["forced_change"] for x in infos)
    random_times = np.array(random_times)
    print(f"  random policy over {N_RANDOM} races: {np.median(random_times) - t_ref:+.1f} s median "
          f"vs reference (best {random_times.min() - t_ref:+.1f} s); "
          f"{forced / N_RANDOM:.0%} needed a forced compound change")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    gaps = random_times - t_ref
    ax.hist(gaps, bins=40, color=C1, edgecolor="white", linewidth=1)
    top = ax.get_ylim()[1]
    for j, (x, color, name) in enumerate((
            (0, INK, "reference"), (t_nom - t_ref, C3, "step-2 plan, nominal energy"),
            (t_lazy - t_ref, C2, "no stop unless forced"))):
        ax.axvline(x, color=color, linewidth=2)
        ax.annotate(f"{name}: {x:+.1f} s", (x, top * (0.95 - 0.1 * j)), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK, va="top")
    style(ax, f"Step 4: race time vs reference, {N_RANDOM} random-policy races "
              f"(random energy, ~{RANDOM_STOPS} random stops)")
    ax.set_xlabel("Race time minus reference (s)", color=MUTED)
    ax.set_ylabel("Races", color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week4_env.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    settings = racesim.load_settings(SETTINGS)
    print(f"Era settings: {settings['name']} ({settings['race']['calibration_race']})")
    w2 = load_week2()
    sim1, t1, plan1 = step1_fuel(settings, w2)
    sim2, t2, plan2 = step2_wear(settings, sim1, plan1)
    t_ref, ref_plan, ref_f, ref_b = step3_energy(sim2, t2, plan2)
    step4_env(settings, t_ref, ref_plan, ref_f, ref_b, t2, plan2)
    print(f"\nAll checks passed. Figures in {FIG_DIR}")
