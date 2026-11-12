"""Week 15: how much of the transfer loss is the tyres, and how much the power unit?

The 2024 agents lose +18 s on pit/tyre decisions in 2026 (Weeks 12-14), mostly because
2024 stint lengths outlast 2026 tyres (A48). That is Pirelli's change, not the 2026
power-unit rules the hypothesis is about. Two mixed simulators split them (A57):

    "2026 PU, 2024 tyres"   2026 settings with the 2024 [tyres] and start tyre
    "2024 PU, 2026 tyres"   2024 settings with the 2026 [tyres] and start tyre

Everything else (rules, car, fuel, battery, track, pit and start losses) stays with the
host era. Steps:
1. Build both and check that only the tyre settings moved.
2. The exact benchmark of each (Week 11's search: 1-2 stops exactly, 3-stop screen).
3. The 2024 agents dropped into each with no retraining, battery read both ways
   (fractions of the caps, A49; absolute 2024 MJ, A54), loss split against that
   world's own benchmark.
4. Decomposition: 2024 -> tyres only / power unit only -> both (2026), for the total,
   pit/tyre and energy loss, mean over seeds, with what is left as interaction.
5. Recovery from the rules change alone: in "2026 PU, 2024 tyres", the 2024 agents
   trained 200,000 steps and agents trained from scratch for the same, as Weeks 13-14
   (A55), next to the same curves in 2026 (both changes).

Usage: python 15th_week.py            (trains what is missing, then everything else)
       python 15th_week.py analyse    (no training; step 5 from the saved runs)
       python 15th_week.py train      (only the training)
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
w11 = load("week11", "11th_week.py")
import racesim  # noqa: E402

check, report = w12.check, w12.report
N_SEEDS = w12.N_SEEDS
TYRE_KEYS = ("race.start_compound", "race.start_tyre_age")      # move with the tyres, plus [tyres]
WORLDS = {"2026 PU, 2024 tyres": ("2026", "2024", "hybrid_2026pu_2024tyres"),       # (host, tyre donor, name)
          "2024 PU, 2026 tyres": ("2024", "2026", "hybrid_2024pu_2026tyres")}


def get(s, key):
    for k in key.split("."):
        s = s[k]
    return s


def hybrid(host, donor, name):
    s, d = w12.settings(host), w12.settings(donor)
    out = racesim.override(s, tyres=d["tyres"], name=name, **{k: get(d, k) for k in TYRE_KEYS})
    return out


PU_ONLY = WORLDS["2026 PU, 2024 tyres"][2]
w12.WORLDS[PU_ONLY] = hybrid(*WORLDS["2026 PU, 2024 tyres"])     # module level: worker processes need it too
BUDGETS = (0, 10_000, 50_000, 100_000, 200_000)


def flat(s, prefix=""):
    out = {}
    for k, v in s.items():
        out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


# --- 1-2. Build and benchmark -----------------------------------------------------------------

def build():
    worlds = {}
    for label, (host, donor, name) in WORLDS.items():
        s = hybrid(host, donor, name)
        fs, fh, fd = flat(s), flat(w12.settings(host)), flat(w12.settings(donor))
        moved = {k for k in fs if fs[k] != fh.get(k)} - {"name"}
        tyre_keys = {k for k in fs if k.startswith("tyres.")} | set(TYRE_KEYS)
        check(moved <= tyre_keys and all(fs[k] == fd[k] for k in tyre_keys),
              f"{label}: only the tyre settings differ from {host} ({len(moved)} changed), all equal to {donor}'s")
        worlds[label] = s
    return worlds


def benchmark(label, s):
    sim = racesim.RaceSim(s)
    best, df = w11.plan_table(sim, label)
    w11.save(s, sim, best, df, s["name"])
    return best[0], w12.split_loss(sim, best[0], best[0], best[1], best[2])["plan"]


# --- 3. Zero-shot -----------------------------------------------------------------------------

def zero_shot(s, t_bench, prefix="sac_2024"):
    """The 2024 agents in world `s`, battery as fractions (A49) and as absolute 2024 MJ (A54)."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def env():
        return racesim.F1StrategyEnv(s, split_battery=True, no_stop_share=w12.NO_STOP_SHARE)
    s24, host = w12.make_env("2024").sim, env().sim
    readings = {"fractions": np.ones(4, np.float32),
                "absolute": np.array([1.0, s24.d_cap / host.d_cap, s24.h_cap / host.h_cap, 1.0], np.float32)}
    out = {k: [] for k in readings}
    for seed in range(N_SEEDS):
        path = os.path.join(w12.MODEL_DIR, f"{prefix}_s{seed}")
        vec = VecNormalize.load(os.path.join(path, "vecnormalize.pkl"), DummyVecEnv([env]))
        vec.training = False
        model = SAC.load(os.path.join(path, "model"))
        for k, scale in readings.items():
            def policy(obs):
                a = model.predict(vec.normalize_obs(obs[None, :]), deterministic=True)[0][0]
                return np.clip(a * scale, 0.0, 1.0)
            e = env()
            t, c, p, infos = w12.play(e, policy)
            r = w12.split_loss(e.sim, t_bench, t, c, p)
            r["forced"] = int(sum(i["forced_change"] for i in infos))
            out[k].append(r)
    return out, readings["absolute"]


# --- 4. Decomposition -------------------------------------------------------------------------

def known():
    """The corners already measured: 2024 agents in 2024 and in 2026 (Weeks 12-14)."""
    with open(os.path.join(w12.RL_DIR, "step4_summary.json")) as f:
        step4 = json.load(f)
    own = [r["curve"][-1] for r in w12.load_runs("sac_2024")]
    full_frac = [r["zero_shot_2026"] for r in w12.load_runs("sac_2024")]
    mean = {k: v["mean"] for k, v in step4["zero_shot_absolute"].items()}
    return own, full_frac, mean


def means(rows):
    d = pd.DataFrame(rows)
    return {k: float(d[k].mean()) for k in ("lost", "pit", "energy")}, {k: float(d[k].std(ddof=1)) for k in ("lost", "pit", "energy")}


def main():
    print("Step 1: mixed simulators (A57)")
    worlds = build()
    bench, shots = {}, {}
    for label, s in worlds.items():
        print(f"\n===== {label} =====")
        bench[label] = benchmark(label, s)
        print(f"\nStep 3, {label}: the 2024 agents with no retraining")
        shots[label], scale = zero_shot(s, bench[label][0])
        for k, rows in shots[label].items():
            m, sd = means(rows)
            print(f"  battery as {k:<9} lost {m['lost']:+6.2f} ± {sd['lost']:5.2f}   pit {m['pit']:+6.2f} ± {sd['pit']:5.2f}"
                  f"   energy {m['energy']:+6.2f} ± {sd['energy']:5.2f}   forced stops "
                  f"{sum(r['forced'] for r in rows)} in {len(rows)} races")
            print("    plans: " + ", ".join(r["plan"] for r in rows))
        if label.startswith("2024 PU"):
            check(np.allclose(scale, 1.0), f"{label}: both readings are the same (2024 power unit, scale 1)")
    with open(os.path.join(w12.RL_DIR, "baselines.json")) as f:
        b24 = json.load(f)["2024"]
    print("\nBenchmarks: 2024 S20 -> M23 -> M23, 2026 M14 -> H17 -> H17 -> H18; "
          + "; ".join(f"{k} {p} ({t:.2f} s)" for k, (t, p) in bench.items()))

    print("\nStep 4: decomposition of the 2024 agents' loss, mean over seeds (s behind each world's benchmark)")
    own, full_frac, full_abs = known()
    rows = {}
    for k in ("fractions", "absolute"):
        o = means(own)[0]
        t = means(shots["2024 PU, 2026 tyres"]["fractions"])[0]          # 2024 PU: readings agree
        p = means(shots["2026 PU, 2024 tyres"][k])[0]
        f = means(full_frac)[0] if k == "fractions" else full_abs
        rows[k] = {"own era": o, "tyres only": t, "power unit only": p, "both (2026)": f,
                   "interaction": {x: f[x] - t[x] - p[x] + o[x] for x in o}}
        print(f"  battery as {k}")
        print(f"    {'':<18} {'lost':>8} {'pit/tyre':>9} {'energy':>8}")
        for name, v in rows[k].items():
            print(f"    {name:<18} {v['lost']:+8.2f} {v['pit']:+9.2f} {v['energy']:+8.2f}")
        tyre_part = {x: t[x] - o[x] for x in o}
        pu_part = {x: p[x] - o[x] for x in o}
        print(f"    added by the tyres:      lost {tyre_part['lost']:+6.2f}, pit {tyre_part['pit']:+6.2f}, energy {tyre_part['energy']:+6.2f}")
        print(f"    added by the power unit: lost {pu_part['lost']:+6.2f}, pit {pu_part['pit']:+6.2f}, energy {pu_part['energy']:+6.2f}")
        rows[k]["added by tyres"], rows[k]["added by power unit"] = tyre_part, pu_part
    pu = rows["fractions"]["added by power unit"]
    tyre = rows["fractions"]["added by tyres"]
    report(abs(pu["pit"]) < abs(tyre["pit"]), "the pit/tyre loss comes more from the tyres than from the power unit")
    report(abs(pu["energy"]) > abs(tyre["energy"]), "the energy loss comes more from the power unit than from the tyres")
    print(f"  (never stop in 2024: {b24['never_stop']['lost']:+.1f} s, for scale)")
    with open(os.path.join(w12.RL_DIR, "week15_decomposition.json"), "w") as f:
        json.dump({"benchmarks": bench, "zero_shot": shots, "decomposition": rows}, f, indent=1)


# --- 5. Recovery from the power unit alone --------------------------------------------------

def train(job):
    """Week 12's training run, from this module so the worker processes can find it."""
    return w12.train(job)


def run_training():
    jobs = ([(f"sac_pu_s{s}", PU_ONLY, s, w12.STEPS, None) for s in range(N_SEEDS)]
            + [(f"sac_2024topu_s{s}", PU_ONLY, s, w12.STEPS, f"sac_2024_s{s}") for s in range(N_SEEDS)])
    todo = [j for j in jobs if not os.path.exists(os.path.join(w12.RL_DIR, f"{j[0]}.json"))]
    if todo:
        print(f"Training {len(todo)} runs in '2026 PU, 2024 tyres', {w12.STEPS:,} steps ({w12.WORKERS} at a time)")
        with Pool(w12.WORKERS) as pool:
            pool.map(train, todo, chunksize=1)


def at(runs, n):
    return pd.DataFrame([next(c for c in r["curve"] if c["steps"] == n) for r in runs])


def recovery():
    print("\nStep 5: recovery from the power unit alone ('2026 PU, 2024 tyres'), mean ± sd (median) over seeds")
    pu, pu_t = w12.load_runs("sac_pu"), w12.load_runs("sac_2024topu")
    full, full_t = w12.load_runs("sac_2026"), w12.load_runs("sac_2024to2026_long")
    check(len(pu) == len(pu_t) == N_SEEDS, f"all {N_SEEDS} seeds of both runs are there")
    with open(os.path.join(HERE, "results", f"optimum_{PU_ONLY}.json")) as f:
        t_b = json.load(f)["race_time_s"]
    check(abs(t_b - w12.benchmark(PU_ONLY)) < 1e-9, "runs are scored against this world's benchmark")
    rows = []
    for n in BUDGETS:
        line = {"steps": n}
        for label, runs in (("pu_retrained", pu_t), ("pu_scratch", pu), ("full_retrained", full_t), ("full_scratch", full)):
            if n == 0 and "scratch" in label:
                continue
            d = at(runs, n)
            line.update({f"{label}_{k}": float(d[k].mean()) for k in ("lost", "pit", "energy")})
            line[f"{label}_lost_median"] = float(d["lost"].median())
        rows.append(line)
        d, txt = at(pu_t, n), f"  {n:>7,} steps  PU only: retrained {at(pu_t, n)['lost'].mean():+6.2f} ± {at(pu_t, n)['lost'].std(ddof=1):5.2f} "
        txt += f"({d['lost'].median():+6.2f}) [pit {d['pit'].mean():+5.2f}, energy {d['energy'].mean():+5.2f}]"
        if n:
            e = at(pu, n)
            txt += f"   scratch {e['lost'].mean():+6.2f} ({e['lost'].median():+6.2f}) [pit {e['pit'].mean():+5.2f}, energy {e['energy'].mean():+5.2f}]"
        print(txt)
        f2 = at(full_t, n)
        print(f"  {'':>13}  2026 (both): retrained {f2['lost'].mean():+6.2f} ({f2['lost'].median():+6.2f})"
              + (f"   scratch {at(full, n)['lost'].mean():+6.2f} ({at(full, n)['lost'].median():+6.2f})" if n else ""))
    pd.DataFrame(rows).to_csv(os.path.join(w12.RL_DIR, "week15_recovery.csv"), index=False)
    print("  each run: lost at " + " / ".join(f"{n // 1000}k" for n in BUDGETS[1:]) + " (retrained from 0k), final plan")
    for r in pu_t + pu:
        print(f"    {r['name']:<18} " + " / ".join(f"{at([r], n)['lost'][0]:+6.2f}" for n in BUDGETS if n or r["init"])
              + f"   {r['curve'][-1]['plan']}")
    steps = [c["steps"] for c in pu[0]["curve"]]
    med_s = np.median([[c["lost"] for c in r["curve"]] for r in pu], axis=0)
    med_t = np.median([[c["lost"] for c in r["curve"][1:]] for r in pu_t], axis=0)
    floor = float(np.median([r["curve"][-1]["lost"] for r in pu]))
    ahead = [s for s, a, b in zip(steps, med_t, med_s) if a < b]
    within = [s for s, v in zip(steps, med_t) if v - floor <= YEAR_TO_YEAR_S]
    print(f"  retrained median ahead of from scratch at: {', '.join(f'{s // 1000}k' for s in ahead) or 'never'}")
    report(bool(within), "retrained agents get within a normal season (2 s) of the from-scratch level"
           + (f": first at {within[0]:,} steps" if within else " within 200,000 steps"))
    report(ahead and len(ahead) >= len(steps) // 2, "the 2024 start helps for at least half of training (pit timing transfers)")
    plot_recovery(pu, pu_t, full, full_t)
    return rows


YEAR_TO_YEAR_S = 2.0                # A56


def plot_recovery(pu, pu_t, full, full_t):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ("lost", "pit", "energy"),
                              ("Slower than the world's benchmark (s)", "Pit/tyre loss (s)", "Energy loss (s)")):
        for runs, color, ls, label in ((pu, w12.C1, "-", "PU only: from scratch"), (pu_t, w12.C2, "-", "PU only: 2024 agent retrained"),
                                       (full, w12.C1, ":", "2026 (both): from scratch"),
                                       (full_t, w12.C2, ":", "2026 (both): 2024 agent retrained")):
            steps = np.array([c["steps"] for c in runs[0]["curve"]])
            y = np.array([[c[key] for c in r["curve"]] for r in runs])
            ax.plot(steps / 1e3, np.median(y, axis=0), color=color, linestyle=ls, linewidth=2 if ls == "-" else 1.5, label=label)
            if ls == "-":
                ax.fill_between(steps / 1e3, y.min(axis=0), y.max(axis=0), color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel("Training steps (thousands)", color=w12.MUTED)
        ax.set_yscale("symlog", linthresh=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=w12.GRID, linewidth=0.8)
        ax.tick_params(colors=w12.MUTED, labelsize=9)
        ax.set_title(title + ", median over seeds", fontsize=10, color=w12.INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=w12.INK)
    fig.tight_layout()
    fig.savefig(os.path.join(w12.FIG_DIR, "week15_recovery.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args != ["analyse"]:
        run_training()
    if args != ["train"]:
        main()
        recovery()
        print(f"\nAll checks passed. Results in {w12.RL_DIR}")
