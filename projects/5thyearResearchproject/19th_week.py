"""Week 19: the main experiments again at learning rate 1e-4.

Week 18 found constant lr 3e-4 (used in Weeks 12-17) leaves the policy moving late in
training: single flipped races, one seed ending +36.6 s. Constant 1e-4 gave +3.25 ± 0.18 s
from scratch in 2026 against +11.82 ± 13.93 s (A61). Weeks 12-17 compare groups all
trained at 3e-4, so their comparisons are fair, but the numbers are noisy. This reruns
the runs behind the main results at 1e-4 (same seeds, steps and settings; A62):

    stage 1   sac_2024_lr1e4       2024 from scratch (the agents that get transferred)
              sac_pu_lr1e4         '2026 PU, 2024 tyres' from scratch (Week 15)
    stage 2   sac_2024to2026_long_lr1e4   2024 agents retrained on 2026 (Weeks 13-14)
              sac_2024topu_lr1e4          ... on '2026 PU, 2024 tyres' (Week 15)
              sac_2024to2026_pit_lr1e4    ... with a fresh pit output (Week 16)

2026 from scratch at 1e-4 is Week 18's sac_2026_lr1e4. Not rerun: Week 12's 100k
retraining (the 200k runs cover it) and Week 16's fresh pit + critic (the weakest variant).
Stage 2 starts from stage 1's agents, so the stages run in order.

Usage: python 19th_week.py            (trains what is missing, then the analysis)
       python 19th_week.py train      (only training)
       python 19th_week.py analyse    (only the analysis)
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
w15 = load("week15", "15th_week.py")
w16 = load("week16", "16th_week.py")
check, report = w12.check, w12.report
N_SEEDS = w12.N_SEEDS
LR = "lr1e4"
PU_ONLY = w15.PU_ONLY
# module level: the worker processes need these in this module's copy of Week 12
w12.LR_VARIANTS[LR] = {"lr": 1e-4}
w12.WORLDS[PU_ONLY] = w15.hybrid(*w15.WORLDS["2026 PU, 2024 tyres"])
w12.RESETS["pit"] = w16.reset_pit


def train(job):
    """Week 12's training run, from this module so the worker processes can find it."""
    return w12.train(job)


def stages():
    s1 = ([(f"sac_2024_{LR}_s{s}", "2024", s, w12.STEPS, None, None, LR) for s in range(N_SEEDS)]
          + [(f"sac_pu_{LR}_s{s}", PU_ONLY, s, w12.STEPS, None, None, LR) for s in range(N_SEEDS)])
    s2 = [(f"{name}_{LR}_s{s}", era, s, w12.STEPS, f"sac_2024_{LR}_s{s}", reset, LR)
          for name, era, reset in (("sac_2024to2026_long", "2026", None), ("sac_2024topu", PU_ONLY, None),
                                   ("sac_2024to2026_pit", "2026", "pit"))
          for s in range(N_SEEDS)]
    return s1, s2


def run_training():
    for i, jobs in enumerate(stages(), 1):
        todo = [j for j in jobs if not os.path.exists(os.path.join(w12.RL_DIR, f"{j[0]}.json"))]
        if todo:
            print(f"Stage {i}: training {len(todo)} runs at lr 1e-4, {w12.STEPS:,} steps ({w12.WORKERS} at a time)",
                  flush=True)
            with Pool(w12.WORKERS) as pool:
                pool.map(train, todo, chunksize=1)


# --- Analysis: every headline number at both learning rates ----------------------------------

RATES = {"3e-4": "", "1e-4": f"_{LR}"}          # suffix on each group's run names
GROUPS = {"2024 from scratch": "sac_2024", "2026 from scratch": "sac_2026",
          "2026 retrained": "sac_2024to2026_long", "2026 fresh pit output": "sac_2024to2026_pit",
          "PU only from scratch": "sac_pu", "PU only retrained": "sac_2024topu"}
BUDGETS = (0, 10_000, 50_000, 100_000, 200_000)
YEAR_TO_YEAR_S = 2.0                # A56
KEYS = ("lost", "pit", "energy")


def runs(group, rate):
    return w12.load_runs(GROUPS[group] + RATES[rate])


def mean(rows):
    d = pd.DataFrame(rows)
    return {k: float(d[k].mean()) for k in KEYS}


def finals():
    print("Step 1: final scores at 200,000 steps, s behind each world's benchmark, mean ± sd (median) over seeds")
    out = {}
    for g in GROUPS:
        out[g] = {}
        for rate in RATES:
            rs = runs(g, rate)
            check(len(rs) == N_SEEDS, f"{g}, lr {rate}: all {N_SEEDS} seeds are there")
            if rate == "1e-4":
                check(all(r["lr"] == LR for r in rs), f"{g}, lr {rate}: trained at that rate")
            d = pd.DataFrame([r["curve"][-1] for r in rs])
            out[g][rate] = {"mean": float(d["lost"].mean()), "sd": float(d["lost"].std(ddof=1)),
                            "median": float(d["lost"].median()), "pit": float(d["pit"].mean()),
                            "energy": float(d["energy"].mean()), "seeds": [float(x) for x in d["lost"]]}
        o, n = out[g]["3e-4"], out[g]["1e-4"]
        print(f"  {g:<22} 3e-4 {o['mean']:+6.2f} ± {o['sd']:5.2f} ({o['median']:+6.2f})   "
              f"1e-4 {n['mean']:+6.2f} ± {n['sd']:5.2f} ({n['median']:+6.2f})   [pit {n['pit']:+5.2f}, energy {n['energy']:+5.2f}]")
    narrower = sum(out[g]["1e-4"]["sd"] < out[g]["3e-4"]["sd"] for g in GROUPS)
    report(narrower >= len(GROUPS) - 1, f"seeds agree more at 1e-4 ({narrower} of {len(GROUPS)} groups have a smaller sd)")
    better = sum(out[g]["1e-4"]["median"] <= out[g]["3e-4"]["median"] + 1.0 for g in GROUPS)
    report(better == len(GROUPS), f"no group's median is more than 1 s worse at 1e-4 ({better} of {len(GROUPS)})")
    return out


def transfer():
    """Zero-shot transfer and the Week 15 decomposition, for the 2024 agents of each rate."""
    print("\nStep 2: the 2024 agents with no retraining, and what the tyres and power unit each add (means over seeds)")
    worlds = {"2026": (w12.settings("2026"), w12.benchmark("2026")),
              **{label: (w15.hybrid(*w15.WORLDS[label]), w12.benchmark(w15.WORLDS[label][2])) for label in w15.WORLDS}}
    out = {}
    for rate, suffix in RATES.items():
        prefix = f"sac_2024{suffix}"
        r24 = runs("2024 from scratch", rate)
        shots = {w: w15.zero_shot(s, t, prefix)[0] for w, (s, t) in worlds.items()}
        frac = [r["zero_shot_2026"] for r in r24]
        check(max(abs(a["lost"] - b["lost"]) for a, b in zip(shots["2026"]["fractions"], frac)) < 1e-6,
              f"lr {rate}: zero-shot in 2026 matches the score saved at the end of training")
        o = mean([r["curve"][-1] for r in r24])
        t = mean(shots["2024 PU, 2026 tyres"]["fractions"])          # 2024 power unit: readings agree
        res = {"own era": o, "tyre loss part": {}, "power unit part": {}}
        for k in ("fractions", "absolute"):
            p, f = mean(shots["2026 PU, 2024 tyres"][k]), mean(shots["2026"][k])
            res[f"2026, battery as {k}"] = f
            res["tyre loss part"] = {x: t[x] - o[x] for x in KEYS}
            res["power unit part"][k] = {x: p[x] - o[x] for x in KEYS}
        out[rate] = res
    with open(os.path.join(w12.RL_DIR, "week15_decomposition.json")) as f:
        saved = json.load(f)["decomposition"]
    check(all(abs(out["3e-4"]["power unit part"][k][x] - saved[k]["added by power unit"][x]) < 1e-6
              and abs(out["3e-4"]["tyre loss part"][x] - saved[k]["added by tyres"][x]) < 1e-6 for k in saved for x in KEYS),
          "lr 3e-4: the decomposition reproduces Week 15's saved numbers")
    print(f"  {'':<40} {'3e-4: lost / pit / energy':>27}   {'1e-4: lost / pit / energy':>27}")
    lines = [("2024 agent in 2024", lambda r: r["own era"]),
             ("2026, battery as fractions (A49)", lambda r: r["2026, battery as fractions"]),
             ("2026, battery in absolute MJ (A54)", lambda r: r["2026, battery as absolute"]),
             ("added by the 2026 tyres", lambda r: r["tyre loss part"]),
             ("added by the 2026 PU (fractions)", lambda r: r["power unit part"]["fractions"]),
             ("added by the 2026 PU (absolute MJ)", lambda r: r["power unit part"]["absolute"])]
    for name, get in lines:
        print(f"  {name:<40} " + "   ".join(" / ".join(f"{get(out[r])[x]:+7.2f}" for x in KEYS) for r in RATES))
    n = out["1e-4"]
    report(n["2026, battery as fractions"]["pit"] > n["2026, battery as fractions"]["energy"],
           "lr 1e-4, battery as fractions: more of the zero-shot loss is pit/tyre than energy (as at 3e-4)")
    report(n["2026, battery as absolute"]["energy"] > n["2026, battery as absolute"]["pit"],
           "lr 1e-4, absolute MJ: more of the zero-shot loss is energy than pit/tyre (as at 3e-4)")
    report(abs(n["tyre loss part"]["pit"]) > abs(n["power unit part"]["fractions"]["pit"]),
           "lr 1e-4: the pit/tyre loss comes more from the tyres than from the power unit")
    report(abs(n["power unit part"]["fractions"]["energy"]) > abs(n["tyre loss part"]["energy"]),
           "lr 1e-4: the energy loss comes more from the power unit than from the tyres")
    return out


def median_curve(rs, key="lost", skip=0):
    return np.median([[c[key] for c in r["curve"][skip:]] for r in rs], axis=0)


def adaptation():
    print(f"\nStep 3: retrained against from scratch, mean (median) over seeds, s behind each world's benchmark")
    out, rows = {}, []
    for world, (rt_g, sc_g) in {"2026": ("2026 retrained", "2026 from scratch"),
                                "PU only": ("PU only retrained", "PU only from scratch")}.items():
        print(f"  {world}")
        out[world] = {}
        for rate in RATES:
            rt, sc = runs(rt_g, rate), runs(sc_g, rate)
            line = []
            for n in BUDGETS:
                a = w15.at(rt, n)
                row = {"world": world, "lr": rate, "steps": n, "retrained": float(a["lost"].mean()),
                       "retrained_median": float(a["lost"].median()),
                       "retrained_pit": float(a["pit"].mean()), "retrained_energy": float(a["energy"].mean())}
                txt = f"{n // 1000:>3}k {a['lost'].mean():+6.2f} ({a['lost'].median():+6.2f})"
                if n:
                    b = w15.at(sc, n)
                    row.update({"scratch": float(b["lost"].mean()), "scratch_median": float(b["lost"].median())})
                    txt += f" vs {b['lost'].mean():+6.2f} ({b['lost'].median():+6.2f})"
                rows.append(row)
                line.append(txt)
            steps = np.array([c["steps"] for c in sc[0]["curve"]])
            m_rt, m_sc = median_curve(rt, skip=1), median_curve(sc)
            floor = float(np.median([r["curve"][-1]["lost"] for r in sc]))
            within = [int(s) for s, v in zip(steps, m_rt) if v - floor <= YEAR_TO_YEAR_S]
            ahead = [int(s) for s, a, b in zip(steps, m_rt, m_sc) if a < b]
            out[world][rate] = {"first_within_2s": within[0] if within else None, "ahead_at": ahead}
            print(f"    lr {rate}: retrained vs scratch  " + ";  ".join(line))
            print(f"    {'':>9} retrained median within {YEAR_TO_YEAR_S:.0f} s of from scratch's final median first at "
                  f"{within[0] // 1000 if within else '-'}k; ahead of from scratch at "
                  f"{', '.join(f'{s // 1000}k' for s in ahead) or 'no checkpoint'}")
    for world in out:
        n = out[world]["1e-4"]
        report(n["first_within_2s"] is not None, f"lr 1e-4, {world}: retrained agents get within a normal season "
               f"(2 s) of from scratch" + (f", first at {n['first_within_2s']:,} steps" if n["first_within_2s"] else ""))
        report(10_000 in n["ahead_at"], f"lr 1e-4, {world}: the 2024 start is ahead after 10,000 steps")
    return out, pd.DataFrame(rows)


def fresh_pit():
    print("\nStep 4: fresh pit output (Week 16) against full retraining, 2026")
    out = {}
    for rate in RATES:
        fp, full, sc = runs("2026 fresh pit output", rate), runs("2026 retrained", rate), runs("2026 from scratch", rate)
        below = float(np.mean(median_curve(fp, "pit", 1) < median_curve(full, "pit", 1)))
        ahead = float(np.mean(median_curve(fp, skip=1) < median_curve(sc)))
        last = {k: float(np.mean([r["curve"][-1]["pit"] for r in rs])) for k, rs in (("fresh pit", fp), ("full", full))}
        out[rate] = {"pit_below_full_share": below, "ahead_of_scratch_share": ahead, "final_pit": last}
        print(f"  lr {rate}: pit/tyre median below full retraining at {below:.0%} of checkpoints; lost median ahead of "
              f"from scratch at {ahead:.0%}; final pit/tyre {last['fresh pit']:+.2f} vs {last['full']:+.2f} s")
    report(out["1e-4"]["final_pit"]["fresh pit"] < out["1e-4"]["final_pit"]["full"],
           "lr 1e-4: a fresh pit decision ends with a lower pit/tyre loss than full retraining (as at 3e-4)")
    return out


def stops():
    """Week 17's stop test for the 2024 agents of both rates: is a first stop chosen, or forced by the age limit (A48, A60)?"""
    w17 = load("week17", "17th_week.py")
    print("\nStep 5: the 2024 agents' stops, chosen (C) or forced (F), no retraining")
    out = {}
    for rate, suffix in RATES.items():
        for era in ("2024", "2026"):
            races = [w17.race(f"sac_2024{suffix}_s{s}", era)[1] for s in range(N_SEEDS)]
            first_forced = sum(bool(st) and not st[0]["chosen"] for st in races)
            first_age = [float(st[0]["age"]) for st in races if st]
            out[f"{rate} {era}"] = {"first_forced": first_forced, "first_stint_laps": first_age,
                                    "stops": [" ".join(f"{x['compound'][0]}{x['age']:.0f}{'C' if x['chosen'] else 'F'}"
                                                       for x in st) for st in races]}
            print(f"  lr {rate}, {era}: first stop forced in {first_forced} of {N_SEEDS}; first stint "
                  f"{', '.join(f'{a:.0f}' for a in first_age)} laps; " + " ; ".join(out[f"{rate} {era}"]["stops"]))
    b26 = w12.settings("2026")
    print(f"  (2026 benchmark first stint: 14 laps; 2026 medium age limit {b26['tyres']['MEDIUM']['max_age']} laps)")
    report(out["1e-4 2024"]["first_forced"] < out["3e-4 2024"]["first_forced"],
           "at 1e-4 the 2024 agents choose more of their own first stops in 2024 than at 3e-4")
    report(np.median(out["1e-4 2026"]["first_stint_laps"]) > 16,
           "at 1e-4 they still run 2024-length first stints in 2026 (the stint habit carries over)")
    return out


def plot(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, world, (rt_g, sc_g) in zip(axes, ("2026", "PU only"), (("2026 retrained", "2026 from scratch"),
                                                                   ("PU only retrained", "PU only from scratch"))):
        for g, color, label in ((sc_g, w12.C1, "from scratch"), (rt_g, w12.C2, "2024 agent retrained")):
            for rate, ls, lw, alpha in (("3e-4", ":", 1.3, 0.7), ("1e-4", "-", 2.0, 1.0)):
                rs = runs(g, rate)
                skip = 1 if rs[0]["init"] else 0              # the 0-step score is the zero-shot, not training
                steps = np.array([c["steps"] for c in rs[0]["curve"][skip:]])
                ax.plot(steps / 1e3, median_curve(rs, skip=skip), color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                        label=f"{label}, lr {rate}")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Training steps (thousands)", color=w12.MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=w12.GRID, linewidth=0.8)
        ax.tick_params(colors=w12.MUTED, labelsize=9)
        ax.set_title(f"{'2026' if world == '2026' else '2026 PU, 2024 tyres'}: median over seeds", fontsize=10,
                     color=w12.INK, loc="left")
    axes[0].set_ylabel("Slower than the world's benchmark (s)", color=w12.MUTED)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=w12.INK)
    fig.tight_layout()
    fig.savefig(os.path.join(w12.FIG_DIR, "week19_rerun.png"), dpi=150)
    plt.close(fig)


def analyse():
    f = finals()
    t = transfer()
    a, table = adaptation()
    p = fresh_pit()
    st = stops()
    table.to_csv(os.path.join(w12.RL_DIR, "week19_adaptation.csv"), index=False)
    with open(os.path.join(w12.RL_DIR, "week19_rerun.json"), "w") as fh:
        json.dump({"finals": f, "transfer": t, "adaptation": a, "fresh_pit": p, "stops": st}, fh, indent=1)
    plot(table)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args != ["analyse"]:
        run_training()
    if args != ["train"]:
        analyse()
        print(f"\nAll checks passed. Results in {w12.RL_DIR}")
