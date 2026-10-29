"""Weeks 7-8: the exact optimiser, the benchmark ("ruler") for everything else.

Instead of ETH's mixed-integer solver (BONMIN), split the problem in two:

1. List every sensible pit plan: 1 or 2 stops, every compound order, every stop lap
   (starting on softs, legal under the compound-change rule).
2. For each plan, find the best fuel and battery use per lap with CasADi + IPOPT.
3. The best of all plans is the optimum.

This is exact because step 2 is convex (ASSUMPTIONS A31): IPOPT's optimum for a plan is
that plan's global optimum, and every plan is tried.

Each step is checked and plotted (figures/week7_*.png, week8_*.png):

1. Energy solver. Check: it is the simulator's model (same race time), never worse than
   the Weeks 3-4 SciPy solver, and lands on the same optimum from any starting point.
2. Solve every plan. Check: every solve succeeds; report the optimum and the runners-up.
3. Is it really the best? Check: it is at least as fast as the Weeks 3-4 alternating
   heuristic and the best plan with nominal energy, and no 3-stop plan beats it.
4. Time lost per race by the real 2024 Bahrain strategies against the benchmark.

The benchmark is written to results/ for later weeks. All era numbers come from
eras/legacy_2022_25.toml; assumptions are in ASSUMPTIONS.md.
"""

import importlib.util
import json
import os
import time
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import racesim  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RESULTS_DIR = os.path.join(HERE, "results")
SETTINGS = os.path.join(HERE, "eras", "legacy_2022_25.toml")
YEAR = 2024
MAX_STOPS = 2           # the search space from the brief: 1 or 2 stops
N_SAMPLE = 20           # random plans for the model-equality checks
N_STARTS = 10           # random starting points per plan for the convexity check
N_THREE_STOP = 50       # best 3-stop plans (by nominal energy) re-solved with optimal energy
SUCCESS = "Solve_Succeeded"
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


def order(compounds):
    return "-".join(c[0] for c in compounds)


def check(cond, msg):
    assert cond, msg
    print(f"  [ok] {msg}")


# --- Step 1: the energy solver -------------------------------------------------------

def step1_solver(sim, plans):
    print("\nStep 1: energy solver (CasADi + IPOPT, one problem reused for every plan)")
    T = sim.energy_problem()[2]
    rng = np.random.default_rng(0)
    sample = [plans[i] for i in rng.choice(len(plans), N_SAMPLE, replace=False)]
    x_nom = np.concatenate(sim.nominal_alloc())
    gap_nom, gap_opt = 0.0, 0.0
    for compounds, pits in sample:
        t_nom = float(T(x_nom, sim.plan_params(compounds, pits))) + sim.pit_loss * len(pits)
        gap_nom = max(gap_nom, abs(t_nom - sim.race_time(compounds, pits, *sim.nominal_alloc())))
        t, fuel, batt, _ = sim.optimal_energy(compounds, pits)
        gap_opt = max(gap_opt, abs(t - sim.race_time(compounds, pits, fuel, batt)))
    check(gap_nom < TOL and gap_opt < TOL,
          f"CasADi objective = simulator race time on {N_SAMPLE} random plans, at nominal and "
          f"optimal energy (max diff {max(gap_nom, gap_opt):.1e} s)")

    t, fuel, batt, _ = sim.optimal_energy(*sample[0])
    soc = sim.b_cap - np.cumsum(batt)
    check(np.all(fuel >= sim.f_min - TOL) and np.all(fuel <= sim.f_max + TOL)
          and fuel.sum() <= sim.fuel0 + TOL and np.all(batt >= sim.b_min - TOL)
          and np.all(batt <= sim.b_max + TOL) and np.all(soc >= -TOL) and np.all(soc <= sim.b_cap + TOL),
          "solution obeys every limit (fuel band and race total, battery deploy/harvest and window)")
    _, rows = sim.simulate(*sample[0], fuel, batt)
    check(all(abs(r["u_f"] - f) < 1e-6 and abs(r["u_b"] - b) < 1e-6 for r, f, b in zip(rows, fuel, batt)),
          "lap-by-lap simulator runs the solution without clipping anything")

    worse = []
    for compounds, pits in sample[:5]:
        t_ipopt = sim.optimal_energy(compounds, pits)[0]
        t_slsqp = sim.best_energy(compounds, pits)[0]
        worse.append(t_ipopt - t_slsqp)
    check(max(worse) < TOL, f"IPOPT never worse than the Weeks 3-4 SciPy solver on 5 plans "
                            f"(IPOPT - SciPy: {min(worse):+.1e} to {max(worse):+.1e} s)")

    spread = 0.0
    for compounds, pits in sample[:3]:
        ts = []
        for _ in range(N_STARTS):
            x0 = np.concatenate([rng.uniform(sim.f_min, sim.fuel_nominal, sim.laps),
                                 rng.uniform(sim.b_min, sim.b_max, sim.laps)])
            ts.append(sim.optimal_energy(compounds, pits, x0)[0])
        spread = max(spread, np.ptp(ts))
    check(spread < TOL, f"{N_STARTS} random starting points per plan reach the same optimum "
                        f"(max spread {spread:.1e} s): consistent with a convex problem (A31)")


# --- Step 2: every plan ----------------------------------------------------------------

def step2_search(sim):
    print(f"\nStep 2: every legal plan with 1-{MAX_STOPS} stops, exact energy for each")
    t0 = time.time()
    best, table = sim.exact(MAX_STOPS)
    elapsed = time.time() - t0
    df = pd.DataFrame(table, columns=["compounds", "pits", "time", "status"])
    df["stops"] = df["pits"].map(len)
    df["order"] = df["compounds"].map(order)
    df["plan"] = [describe(c, p, sim.laps) for c, p in zip(df["compounds"], df["pits"])]
    counts = df.groupby("stops").size()
    print(f"  {len(df)} plans ({', '.join(f'{n} with {k} stop' + 's' * (k > 1) for k, n in counts.items())}), "
          f"solved in {elapsed:.0f} s ({elapsed / len(df) * 1000:.1f} ms each)")
    statuses = Counter(df["status"])
    check(statuses[SUCCESS] == len(df), f"every solve succeeded ({dict(statuses)})")
    t_best, compounds, pits, *_ = best
    check(abs(df["time"].min() - t_best) < TOL, "optimum = fastest plan in the table")
    df = df.sort_values("time").reset_index(drop=True)
    df["gap"] = df["time"] - t_best

    print(f"  OPTIMUM: {describe(compounds, pits, sim.laps)} (stops after laps "
          f"{', '.join(map(str, pits))}), race time {t_best:.3f} s")
    print("  ten fastest plans:")
    for _, r in df.head(10).iterrows():
        print(f"    {r['plan']:<20} {r['time']:10.3f} s  {r['gap']:+7.3f} s")
    print("  fastest plan per compound order:")
    per_order = df.groupby("order", sort=False).head(1)
    for _, r in per_order.iterrows():
        print(f"    {r['order']:<6} {r['plan']:<20} {r['gap']:+7.2f} s")
    print(f"  fastest 1-stop: {df[df['stops'] == 1].iloc[0]['plan']}, "
          f"{df[df['stops'] == 1].iloc[0]['gap']:+.2f} s")
    return best, df


# --- Step 3: is it really the best? ------------------------------------------------------

def step3_compare(settings, sim, best, df):
    print("\nStep 3: compare with the earlier optimisers")
    t_best, compounds, pits, *_ = best

    t_ref, ref_plan, _, _ = sim.reference()
    print(f"  Weeks 3-4 alternating heuristic (A20): {describe(*ref_plan, sim.laps)}, {t_ref:.3f} s "
          f"({t_ref - t_best:+.1e} s vs exact)")
    check(t_best <= t_ref + TOL, "exact optimum is at least as fast as the Weeks 3-4 heuristic")
    same = ref_plan == (compounds, pits)
    print(f"  the heuristic {'found the same plan' if same else 'found a DIFFERENT plan'}: "
          f"A20's 'not guaranteed' now {'confirmed harmless for this race' if same else 'matters'}")

    nominal = np.array([sim.race_time(c, p, *sim.nominal_alloc()) for c, p in zip(df["compounds"], df["pits"])])
    i_nom = int(np.argmin(nominal))
    gain = nominal - df["time"].to_numpy()
    print(f"  best plan with nominal energy: {df.loc[i_nom, 'plan']}, {nominal[i_nom]:.3f} s; optimising "
          f"energy saves {nominal[i_nom] - t_best:.2f} s on the optimum "
          f"(between {gain.min():.2f} and {gain.max():.2f} s across all plans)")
    check(t_best <= nominal.min() + TOL, "exact optimum beats the best plan with nominal energy")

    # 3 stops: not in the search, so check the best 3-stop plans with exact energy
    three = racesim.RaceSim(racesim.override(settings, **{"race.min_stops": 3}))
    nom3 = sorted((sim.race_time(c, p, *sim.nominal_alloc()), c, p) for c, p in three.plans(3))
    solved3 = [(sim.optimal_energy(c, p)[0], t_nom, c, p) for t_nom, c, p in nom3[:N_THREE_STOP]]
    t3 = min(solved3)
    gain3 = max(t_nom - t for t, t_nom, _, _ in solved3)
    print(f"  {len(nom3)} 3-stop plans; best of the {N_THREE_STOP} fastest at nominal energy, with exact "
          f"energy: {describe(t3[2], t3[3], sim.laps)}, {t3[0]:.3f} s ({t3[0] - t_best:+.2f} s); "
          f"energy saved on them at most {gain3:.2f} s")
    check(t3[0] > t_best, f"none of the {N_THREE_STOP} best 3-stop plans beats it with exact energy")
    margin = nom3[0][0] - t_best
    check(margin > 2 * max(gain.max(), gain3),
          f"every 3-stop plan with nominal energy is >= {margin:.1f} s slower than the optimum, over "
          f"twice the most energy has saved on any plan ({max(gain.max(), gain3):.2f} s)")
    return nominal


# --- Step 4: time lost by the real strategies --------------------------------------------

def step4_real(sim, best, w2):
    print(f"\nStep 4: time lost per race by the real {YEAR} Bahrain strategies (mean driver, new sets)")
    laps, results, race_laps, _, _ = w2.prepare(YEAR)
    assert race_laps == sim.laps
    t_best = best[0]
    rows = []
    for pos, drv, compounds, pits, _ in w2.actual_strategies(results, laps, race_laps):
        legal = compounds[0] == sim.start_compound and sim.legal(compounds, pits)
        t = sim.optimal_energy(compounds, pits)[0] if legal else np.nan
        rows.append({"pos": pos, "driver": drv, "plan": describe(compounds, pits, sim.laps),
                     "stops": len(pits), "lost": t - t_best})
    real = pd.DataFrame(rows)
    for _, r in real.iterrows():
        lost = "not comparable" if np.isnan(r["lost"]) else f"{r['lost']:+6.2f} s"
        print(f"    P{r['pos']:<2} {r['driver']:<4} {r['plan']:<22} {lost}")
    ok = real.dropna()
    print(f"  {len(ok)} lead-lap finishers: median {ok['lost'].median():.2f} s lost, "
          f"best {ok['lost'].min():.2f} s, worst {ok['lost'].max():.2f} s")
    check((ok["lost"] >= -TOL).all(), "no real strategy beats the benchmark")
    return real


# --- Output --------------------------------------------------------------------------

def save(settings, sim, best, df):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_best, compounds, pits, fuel, batt = best
    out = {
        "era": settings["name"], "race": settings["race"]["calibration_race"],
        "search": f"every legal plan with 1-{MAX_STOPS} stops, exact energy per plan (7th_8th_week.py)",
        "race_time_s": round(t_best, 6), "plan": describe(compounds, pits, sim.laps),
        "compounds": list(compounds), "pit_laps": list(pits),
        "fuel_mj_per_lap": np.round(fuel, 6).tolist(), "battery_mj_per_lap": np.round(batt, 6).tolist(),
    }
    path = os.path.join(RESULTS_DIR, f"optimum_{settings['name']}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    table = df[["plan", "order", "stops", "time", "gap"]].assign(
        pits=df["pits"].map(lambda p: " ".join(map(str, p))))
    csv = os.path.join(RESULTS_DIR, f"all_plans_{settings['name']}.csv")
    table.round(6).to_csv(csv, index=False)
    print(f"\nBenchmark saved: {os.path.relpath(path, HERE)}, every plan's time in {os.path.relpath(csv, HERE)}")


def plot_landscape(sim, best, df):
    _, compounds, pits, *_ = best
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    first = df.assign(first=df["pits"].map(lambda p: p[0]))
    best_order = first.groupby(["stops", "order"])["gap"].min().sort_values()
    top = [*best_order[2].index[:3], best_order[1].index[0]]     # 3 best 2-stop orders, best 1-stop
    for o, color in zip(top, (C1, C2, C3, C4)):
        d = first[first["order"] == o].groupby("first")["gap"].min()
        ax.plot(d.index, d.values, color=color, linewidth=2,
                label=f"{o}, {'1' if o.count('-') == 1 else '2'}-stop (best {d.min():+.1f} s)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", labelcolor=INK)
    ax.plot(pits[0], 0, "o", color=INK, markersize=6)
    ax.annotate(f"optimum {describe(compounds, pits, sim.laps)}", (pits[0], 0), xytext=(0, -14),
                textcoords="offset points", fontsize=8, color=INK, ha="center")
    ax.set_ylim(-2.5, 20)
    ax.set_xlabel("First stop after lap", color=MUTED)
    style(ax, "Slower than the optimum (s), best plan for each first-stop lap")

    ax = axes[1]
    o = order(compounds)
    d = df[(df["order"] == o) & (df["stops"] == 2)]
    grid = np.full((sim.laps, sim.laps), np.nan)
    for p, g in zip(d["pits"], d["gap"]):
        grid[p[1] - 1, p[0] - 1] = g
    im = ax.imshow(np.minimum(grid, 10), origin="lower", cmap="Blues_r", aspect="auto",
                   extent=(0.5, sim.laps + 0.5, 0.5, sim.laps + 0.5))
    ax.plot(pits[0], pits[1], "o", color=C2, markersize=6)
    ax.annotate("optimum", (pits[0], pits[1]), xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=INK)
    ax.set_xlabel("First stop after lap", color=MUTED)
    ax.set_ylabel("Second stop after lap", color=MUTED)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("slower than the optimum (s, capped at 10)", color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    style(ax, f"{o}: every pair of stop laps")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week7_landscape.png"), dpi=150)
    plt.close(fig)


def plot_optimum(sim, best):
    t_best, compounds, pits, fuel, batt = best
    _, rows = sim.simulate(compounds, pits, fuel, batt)
    lap = np.arange(1, sim.laps + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    series = ((100 * fuel / sim.fuel_nominal, C1, "Fuel per lap (% of nominal)"),
              (batt, C2, "Net battery use per lap (MJ, + = deploy)"),
              ([r["time"] for r in rows], C3, "Lap time (s)"))
    for ax, (y, color, title) in zip(axes, series):
        ax.plot(lap, y, color=color, linewidth=2)
        for p in pits:
            ax.axvline(p + 0.5, color=MUTED, linewidth=0.8, linestyle=":")
        style(ax, title)
        ax.set_xlabel("Lap", color=MUTED)
    axes[1].plot(lap, sim.b_cap - np.cumsum(batt), color=C4, linewidth=1.5)
    axes[1].annotate("battery left (MJ)", (lap[0], sim.b_cap - batt[0]), xytext=(6, 2),
                     textcoords="offset points", fontsize=8, color=C4)
    axes[1].annotate(f"used per lap: {batt.mean():.2f} MJ", (lap[-1], batt[-1]), xytext=(-4, 6),
                     textcoords="offset points", fontsize=8, color=C2, ha="right")
    axes[2].set_ylim(min(r["time"] for r in rows) - 1, 101)
    axes[2].annotate("stops", (pits[0] + 0.5, 100.5), xytext=(4, 0), textcoords="offset points",
                     fontsize=8, color=MUTED, va="top")
    fig.suptitle(f"Optimum {describe(compounds, pits, sim.laps)}: {t_best:.2f} s", x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week8_optimum.png"), dpi=150)
    plt.close(fig)


def plot_real(real):
    ok = real.dropna().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(ok))
    ax.barh(y, ok["lost"], color=C1, height=0.6)
    ax.set_yticks(y, [f"P{p} {d}  {s}" for p, d, s in zip(ok["pos"], ok["driver"], ok["plan"])])
    for yi, v in zip(y, ok["lost"]):
        ax.annotate(f"{v:.1f} s", (v, yi), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK, va="center")
    style(ax, f"Time lost vs the benchmark, real {YEAR} Bahrain plans (s, model's mean driver)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week8_real_vs_optimum.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    # Weeks 3-8 used the placeholder energy terms (A16, A19); pinned so they stay reproducible
    settings = racesim.override(racesim.load_settings(SETTINGS), **{"energy.model": "placeholder"})
    print(f"Era settings: {settings['name']} ({settings['race']['calibration_race']})")
    sim = racesim.RaceSim(settings)
    plans = list(sim.plans(MAX_STOPS))
    step1_solver(sim, plans)
    best, df = step2_search(sim)
    step3_compare(settings, sim, best, df)
    real = step4_real(sim, best, load_week2())
    save(settings, sim, best, df)
    plot_landscape(sim, best, df)
    plot_optimum(sim, best)
    plot_real(real)
    print(f"\nAll checks passed. Figures in {FIG_DIR}")
