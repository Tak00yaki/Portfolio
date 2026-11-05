"""Weeks 13-14: the main experiments (Step 4), with the Week 12 agents.

1. Transfer: the 2024 agents in 2026 with no retraining, total loss split into pit/tyre
   and energy (as Week 12), and a second reading of the same agents with the battery
   action decoded in absolute MJ instead of fractions of each era's limits (A54).
2. Adaptation: the 2024 agents trained on 2026 for FINETUNE_STEPS (as long as the
   from-scratch agents), scored every 10,000 steps. Table at 0, 10k, 50k, 100k and 200k
   steps against the agents trained on 2026 from scratch with the same budget.
   Check: the first 100,000 steps reproduce Week 12's 100,000-step runs (same seeds).
3. Context: the size of all this against a normal season's change, the ~2 s lost by
   choosing the 2024 Bahrain plan with 2023 numbers (Weeks 1-2), and the no-RL transfer
   test (Week 11).
4. Sources for two placeholders, and what they change:
   A8 (lap time per kg): the published TUM race-simulation values, and our own estimate
   from the point-mass straights (Weeks 9-11), which see only part of the lap.
   A12 (fuel lower heating value): textbook petrol and ethanol values, blended as the
   E10 fuel used since 2022. Both: the Barcelona benchmarks re-solved with the range.

Usage: python 13th_14th_week.py            (trains what is missing, then the analysis)
       python 13th_14th_week.py analyse    (only the tables and figure, from results/rl/)
       python 13th_14th_week.py train      (only the training)
"""

import importlib.util
import json
import os
import sys
from multiprocessing import Pool

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w12 = load("week12", "12th_week.py")
check, report = w12.check, w12.report
RL_DIR, MODEL_DIR, FIG_DIR = w12.RL_DIR, w12.MODEL_DIR, w12.FIG_DIR
N_SEEDS = w12.N_SEEDS
FINETUNE_STEPS = w12.STEPS          # same budget as from scratch
BUDGETS = (0, 10_000, 50_000, 100_000, 200_000)
YEAR_TO_YEAR_S = 2.0                # Weeks 1-2: 2024 Bahrain plan chosen with 2023 numbers (90% 0-5 s)
LONG = "sac_2024to2026_long"
# A8: lap time mass sensitivity t_lap_sens_mass (s/kg) in the TUM race simulation's 2019 parameter
# files (github.com/TUMFTM/race-simulation, racesim/input/parameters; Heilmeier et al. 2018, 2020),
# from their quasi-steady-state lap time simulation (Heilmeier, Geisslinger and Betz 2019).
TUM_MASS = {"Catalunya 2019": 0.033, "Sakhir 2019": 0.023, "Budapest 2019": 0.028,
            "Monza 2019": 0.027, "Singapore 2019": 0.030}
# A12: lower heating values, MJ/kg (Heywood, Internal Combustion Engine Fundamentals, App. D:
# gasoline 44.0, ethanol 26.9; ORNL Bioenergy conversion factors: 43.5 and 26.7), and the
# E10 share by mass (2022 Technical Regulations, at least 10% m/m advanced sustainable ethanol).
LHV = {"Heywood": (44.0, 26.9), "ORNL": (43.5, 26.7)}
ETHANOL_MASS_SHARE = 0.10
N_TOP = 60                          # best plans re-solved in the sensitivity
SENS_ERAS = {"2024 Bahrain": ("legacy_2022_25", "legacy_2022_25_physics"),     # (settings, saved plans)
             "2024 Barcelona": ("legacy_2022_25_barcelona", "legacy_2022_25_barcelona"),
             "2026 Barcelona": ("2026", "2026")}
C1, C2, C3, C4, INK, MUTED, GRID = w12.C1, w12.C2, w12.C3, w12.C4, w12.INK, w12.MUTED, w12.GRID


# --- 1. Transfer with the battery in absolute MJ ---------------------------------------------

def zero_shot_absolute():
    """The 2024 agents in 2026, deploy and extra harvest read as 2024 MJ (fraction x 2024
    cap), then given to the 2026 environment as that many MJ (clipped to the 2026 caps).
    Fuel stays a fraction of each era's range: 2024 and 2026 fuel MJ are not comparable."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    s24, s26 = w12.make_env("2024").sim, w12.make_env("2026").sim
    scale = np.array([1.0, s24.d_cap / s26.d_cap, s24.h_cap / s26.h_cap, 1.0], np.float32)
    out = []
    for s in range(N_SEEDS):
        path = os.path.join(MODEL_DIR, f"sac_2024_s{s}")
        vec = VecNormalize.load(os.path.join(path, "vecnormalize.pkl"), DummyVecEnv([lambda: w12.make_env("2026")]))
        vec.training = False
        model = SAC.load(os.path.join(path, "model"))
        env = w12.make_env("2026")

        def policy(obs):
            a = model.predict(vec.normalize_obs(obs[None, :]), deterministic=True)[0][0]
            return np.clip(a * scale, 0.0, 1.0)
        t, c, p, infos = w12.play(env, policy)
        r = w12.split_loss(env.sim, w12.benchmark("2026"), t, c, p)
        r["deploy_mj"] = float(np.median([i["deploy"] for i in infos]))
        r["deploy_total_mj"] = float(sum(i["deploy"] for i in infos))
        out.append(r)
    return out, scale


# --- 2. Adaptation ---------------------------------------------------------------------------

def train(job):
    """Week 12's training run, from this module so the worker processes can find it."""
    return w12.train(job)


def run_long():
    jobs = [(f"{LONG}_s{s}", "2026", s, FINETUNE_STEPS, f"sac_2024_s{s}") for s in range(N_SEEDS)]
    todo = [j for j in jobs if not os.path.exists(os.path.join(RL_DIR, f"{j[0]}.json"))]
    if todo:
        print(f"Training {len(todo)} runs: 2024 agents on 2026 for {FINETUNE_STEPS:,} steps ({w12.WORKERS} at a time)")
        with Pool(w12.WORKERS) as pool:
            pool.map(train, todo, chunksize=1)


def at(runs, steps):
    rows = [next(c for c in r["curve"] if c["steps"] == steps) for r in runs]
    return pd.DataFrame(rows)


def ms(x):
    return f"{x.mean():+6.2f} ± {x.std(ddof=1):5.2f} (median {x.median():+6.2f})"


def analyse():
    r24, r26 = w12.load_runs("sac_2024"), w12.load_runs("sac_2026")
    rt, long = w12.load_runs("sac_2024to2026"), w12.load_runs(LONG)
    check(len(r24) == len(r26) == len(long) == N_SEEDS, f"all {N_SEEDS} seeds of every run are there")
    with open(os.path.join(RL_DIR, "baselines.json")) as f:
        base = json.load(f)

    print("\nStep 1: transfer, the 2024 agents in 2026 with no retraining (seconds behind the 2026 benchmark)")
    z = pd.DataFrame([r["zero_shot_2026"] for r in r24])
    za, scale = zero_shot_absolute()
    za = pd.DataFrame(za)
    for label, d in (("battery as fractions of each era's caps (A49)", z), ("battery in absolute MJ (A54)", za)):
        print(f"  {label}")
        for k in ("lost", "pit", "energy"):
            print(f"    {k:<7} {ms(d[k])}")
    print(f"  absolute MJ: deploy fraction x {scale[1]:.3f}, harvest x {scale[2]:.3f}; median lap deploys "
          + ", ".join(f"{x:.2f}" for x in za["deploy_mj"]) + f" MJ (2026 optimum ~5.27)")
    print("  plans (fractions): " + ", ".join(z["plan"]))
    print("  plans (absolute):  " + ", ".join(za["plan"]))
    report(za["energy"].mean() > z["energy"].mean(),
           "energy loss is larger when the old agent keeps its 2024 MJ than when it scales with the caps")

    print(f"\nStep 2: adaptation, 2024 agents trained on 2026 against agents trained on 2026 from scratch")
    diff = max(abs(a["lost"] - b["lost"]) for r1, r2 in zip(rt, long) for a, b in zip(r1["curve"], r2["curve"]))
    report(diff < 1e-6, f"the first 100,000 steps reproduce Week 12's runs (largest difference {diff:.2g} s)")
    rows = []
    for n in BUDGETS:
        a = at(long, n)
        b = at(r26, n) if n else None
        rows.append({"steps": n, **{f"retrained_{k}": a[k].mean() for k in ("lost", "pit", "energy")},
                     "retrained_lost_sd": a["lost"].std(ddof=1), "retrained_lost_median": a["lost"].median(),
                     **({f"scratch_{k}": b[k].mean() for k in ("lost", "pit", "energy")} if n else {}),
                     **({"scratch_lost_sd": b["lost"].std(ddof=1), "scratch_lost_median": b["lost"].median()} if n else {})})
        print(f"  {n:>7,} steps  retrained {ms(a['lost'])}  [pit {a['pit'].mean():+6.2f}, energy {a['energy'].mean():+6.2f}]")
        if n:
            print(f"  {'':>13}  scratch   {ms(b['lost'])}  [pit {b['pit'].mean():+6.2f}, energy {b['energy'].mean():+6.2f}]")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RL_DIR, "adaptation.csv"), index=False)
    print("  each retrained run: lost at " + " / ".join(f"{n // 1000}k" for n in BUDGETS) + ", final plan")
    for r in long:
        print(f"    {r['name']:<26} " + " / ".join(f"{at([r], n)['lost'][0]:+6.2f}" for n in BUDGETS)
              + f"   {r['curve'][-1]['plan']}")
    bar = min(base["2026"]["never_stop"]["lost"], base["2026"]["random_median"])
    bad = [r["name"] for r in long if r["curve"][-1]["lost"] >= bar]
    report(not bad, "every retrained seed beats never-stop and random in 2026" + (f"; not: {', '.join(bad)}" if bad else ""))

    print(f"\nStep 3: context (a normal season's change: ~{YEAR_TO_YEAR_S:.0f} s, 2023 -> 2024 Bahrain plan, Weeks 1-2)")
    with open(os.path.join(HERE, "results", "transfer_barcelona_2024_to_2026.json")) as f:
        tr = json.load(f)
    own = np.median([r["curve"][-1]["lost"] for r in r24])
    floor = np.median([r["curve"][-1]["lost"] for r in r26])
    print(f"  RL agent in its own era: 2024 {own:+.1f} s, 2026 from scratch {floor:+.1f} s (medians, 200k steps)")
    print(f"  no RL, 2024 decisions in 2026 (Week 11): stop laps {tr['tyres']['timing_all_tyres']:+.1f} s, "
          f"old plan {tr['main']['plan']:+.1f} s, old MJ amounts {tr['energy_old_amounts_low']:+.0f} to "
          f"{tr['main']['energy_literal']:+.0f} s")
    print(f"  RL zero-shot: {z['lost'].median():+.1f} s (fractions), {za['lost'].median():+.1f} s (absolute MJ), "
          f"= {z['lost'].median() / YEAR_TO_YEAR_S:.0f}x and {za['lost'].median() / YEAR_TO_YEAR_S:.0f}x a normal season")
    med = table.set_index("steps")["retrained_lost_median"]
    gap = {n: med[n] - floor for n in BUDGETS}
    print("  transfer cost (retrained median - from-scratch 200k median): "
          + ", ".join(f"{n // 1000}k {g:+.1f} s" for n, g in gap.items()))
    steps = [c["steps"] for c in long[0]["curve"]]
    curve = np.median([[c["lost"] for c in r["curve"]] for r in long], axis=0)
    within = [s for s, v in zip(steps, curve) if v - floor <= YEAR_TO_YEAR_S]
    first = within[0] if within else None
    report(first is not None, "the retrained agents get within a normal season's change of the from-scratch "
           "level" + (f": first at {first:,} steps" if first is not None else " within 200,000 steps"))
    scratch_curve = np.median([[c["lost"] for c in r["curve"]] for r in r26], axis=0)
    ahead = [s for s, a, b in zip(steps[1:], curve[1:], scratch_curve) if a < b]
    print("  steps where the retrained median is ahead of from scratch: "
          + (", ".join(f"{s // 1000}k" for s in ahead) if ahead else "none"))
    with open(os.path.join(RL_DIR, "step4_summary.json"), "w") as f:
        json.dump({"zero_shot_fraction": z[["lost", "pit", "energy"]].describe().to_dict(),
                   "zero_shot_absolute": za[["lost", "pit", "energy"]].describe().to_dict(),
                   "zero_shot_absolute_plans": list(za["plan"]),
                   "adaptation": rows, "year_to_year_s": YEAR_TO_YEAR_S, "own_era_2024": own,
                   "scratch_2026": floor, "within_year_to_year_at": first, "retrained_ahead_at": ahead,
                   "reproduces_week12": diff}, f, indent=1)
    plot(long, r26, floor, z, za)


def plot(long, r26, floor, z, za):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ("lost", "pit", "energy"),
                              ("Slower than the 2026 benchmark (s)", "Pit/tyre loss (s)", "Energy loss (s)")):
        for runs, color, label in ((r26, C1, "trained on 2026 from scratch"), (long, C2, "2024 agent, then 2026 training")):
            steps = np.array([c["steps"] for c in runs[0]["curve"]])
            y = np.array([[c[key] for c in r["curve"]] for r in runs])
            ax.plot(steps / 1e3, np.median(y, axis=0), color=color, linewidth=2, label=label)
            ax.fill_between(steps / 1e3, y.min(axis=0), y.max(axis=0), color=color, alpha=0.15, linewidth=0)
            if runs is long:
                marks = [i for i, s in enumerate(steps) if s in BUDGETS]
                ax.plot(steps[marks] / 1e3, np.median(y, axis=0)[marks], "o", color=color, markersize=5)
        if key == "lost":
            ax.axhspan(floor, floor + YEAR_TO_YEAR_S, color=C3, alpha=0.2, linewidth=0,
                       label=f"within {YEAR_TO_YEAR_S:.0f} s (a normal season) of from scratch")
            ax.plot([0], [za["lost"].median()], "D", color=C4, markersize=6, label="zero-shot, battery in absolute MJ")
        if key == "energy":
            ax.plot([0], [za["energy"].median()], "D", color=C4, markersize=6)
        ax.set_xlabel("Training steps on 2026 (thousands)", color=MUTED)
        ax.set_yscale("symlog", linthresh=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(title + ", median and range over seeds", fontsize=10, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week13_adaptation.png"), dpi=150)
    plt.close(fig)


# --- 4. Sources for A8 and A12 -----------------------------------------------------------------

def mass_on_straights(settings):
    """Our own A8 estimate: 2024 Barcelona clean laps of the top 4 (Week 11 telemetry), each
    straight re-run 10 kg heavier with everything else as calibrated. Corners and the
    traction phase out of them are not modelled, so this is a lower bound."""
    import pointmass as pm
    import racesim
    from dataclasses import replace
    w11 = load("week11", "11th_week.py")
    laps, results, race_laps, clean, _ = w11.load_year(2024)
    top = results.sort_values("Position")["Driver"].head(4).tolist()
    traces = w11.w9.load_traces(w11.w2, laps, clean, racesim.RaceSim(settings), 2024, w11.EVENT, top)
    ps = settings["pointmass"]
    fit = {"power": ps["power_kw"] * 1e3, "cda": ps["cda_m2"], "crr": ps["crr"], "rho": ps["rho_kg_m3"],
           "brake_a": ps["brake_a_m_s2"], "brake_b": ps["brake_b_per_m"]}
    rows = []
    for tr in (t for t in traces if t["kind"] == "clean"):
        car, sts = w11.w9.lap_straights(fit, tr)
        heavy = replace(car, mass=car.mass + 10.0)
        t0 = sum(pm.run(car, st)[0] for st in sts)
        rows.append(((sum(pm.run(heavy, st)[0] for st in sts) - t0) / 10.0, t0 / tr["lap_time"]))
    return np.array(rows)


def plans_from_csv(name, n):
    full = {"S": "SOFT", "M": "MEDIUM", "H": "HARD"}
    df = pd.read_csv(os.path.join(HERE, "results", f"all_plans_{name}.csv")).head(n)
    return [(tuple(full[x.strip()[0]] for x in plan.split("->")), tuple(int(p) for p in str(pits).split()))
            for plan, pits in zip(df["plan"], df["pits"])]


def resolve(settings, plans):
    import racesim
    sim = racesim.RaceSim(settings)
    t, c, p = min((sim.optimal_energy(c, p)[0], c, p) for c, p in plans)
    return t, w12.split_loss(sim, t, t, c, p)["plan"]


def step4_sources():
    import racesim
    print("\nStep 4: sources for A8 (lap time per kg) and A12 (fuel heating value)")
    print("  A8, TUM race simulation (2019 cars): " + ", ".join(f"{k} {v:.3f}" for k, v in TUM_MASS.items()) + " s/kg")
    s24 = racesim.load_settings(os.path.join(HERE, "eras", f"{w12.ERAS['2024']}.toml"))
    est = mass_on_straights(s24)
    print(f"  A8, our point mass on the straights ({len(est)} laps): {np.median(est[:, 0]):.4f} s/kg "
          f"(IQR {np.quantile(est[:, 0], 0.25):.4f}-{np.quantile(est[:, 0], 0.75):.4f}), straights "
          f"{np.median(est[:, 1]):.0%} of the lap: drag-limited, so most of the mass cost is in the corners")
    report(np.median(est[:, 0]) < min(TUM_MASS.values()),
           "the straights alone give less than every published value (a lower bound, as expected)")
    blends = {k: (1 - ETHANOL_MASS_SHARE) * g + ETHANOL_MASS_SHARE * e for k, (g, e) in LHV.items()}
    print("  A12, E10 blend: " + ", ".join(f"{k} {v:.1f} MJ/kg" for k, v in blends.items())
          + f"; petrol alone {min(g for g, _ in LHV.values()):.1f}-{max(g for g, _ in LHV.values()):.1f}")
    out = {"tum_mass": TUM_MASS, "straights_s_per_kg": float(np.median(est[:, 0])), "lhv_e10": blends, "cases": {}}
    for era, (toml, saved) in SENS_ERAS.items():
        s = racesim.load_settings(os.path.join(HERE, "eras", f"{toml}.toml"))
        plans = plans_from_csv(saved, N_TOP)
        t0, plan0 = resolve(s, plans)
        with open(os.path.join(HERE, "results", f"optimum_{saved}.json")) as f:
            bench = json.load(f)["race_time_s"]
        check(abs(t0 - bench) < 1e-3, f"{era}: the top {N_TOP} plans re-solve to the benchmark ({plan0})")
        burn = s["car"]["fuel_start_kg"] / s["race"]["laps"]
        m0, ev0, b0, f0 = (s["car"]["mass_time_s_per_kg"], s["track"]["evolution_s_per_lap"],
                           s["track"]["base_lap_s"], s["car"]["fuel_start_kg"])
        # as a refit would: base and evolution move so a lap at nominal fuel keeps its measured time
        cases = {f"A8 {m:.3f} s/kg": {"car.mass_time_s_per_kg": m, "track.evolution_s_per_lap": ev0 + (m - m0) * burn,
                                      "track.base_lap_s": b0 - (m - m0) * (f0 + 0.5 * burn)}
                 for m in (min(TUM_MASS.values()), max(TUM_MASS.values()))}
        if era.startswith("2024"):      # 2026 fuel is set by its own rules (A44)
            cases.update({f"A12 {v:.1f} MJ/kg ({k} E10)": {"car.fuel_lhv_mj_per_kg": v} for k, v in blends.items()})
            cases["A12 44.0 MJ/kg (petrol)"] = {"car.fuel_lhv_mj_per_kg": 44.0}
        out["cases"][era] = {}
        for name, ch in cases.items():
            t, plan = resolve(racesim.override(s, **ch), plans)
            out["cases"][era][name] = {"time_change_s": t - t0, "plan": plan, "same_plan": plan == plan0}
            print(f"    {era:<14} {name:<28} race {t - t0:+6.2f} s, optimum {plan}{'' if plan == plan0 else '  (changes)'}")
        same = all(c["same_plan"] for c in out["cases"][era].values())
        report(same, f"{era}: the benchmark plan is the same across the A8 and A12 ranges")
    with open(os.path.join(HERE, "results", "a8_a12_sources.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1:] != ["analyse"]:
        run_long()
    if sys.argv[1:] != ["train"]:
        analyse()
        step4_sources()
    print(f"\nAll checks passed. Results in {RL_DIR}")
