"""Week 18: more stable training.

Three runs ended far above their best (2026 from scratch seed 3, +36.6 s; retrained seed 3,
+14.9 s; fresh pit + critic seed 3, +17.5 s). Each is one bad deterministic race in which a
pit decision or the energy plan flips, with good scores 10-20k steps either side; smaller
swings appear in other seeds too. With a constant learning rate (3e-4) the policy keeps
moving to the end. Two standard fixes, tested on 2026 from scratch with the same 5 seeds
(A61):

    "decaying lr"   3e-4 at the start, linearly down to 3e-5 at 200,000 steps
    "lr 1e-4"       a constant, lower learning rate

against the existing runs (constant 3e-4). Everything else as Week 12.

Measures, per run, over the last 100,000 steps (10 scores):
    late drops   scores more than 5 s worse than the run's best so far
    swing        median change between consecutive scores
and the final score, the best, and the mean of the last 5.

Usage: python 18th_week.py            (trains what is missing, then the analysis)
       python 18th_week.py analyse    (only the analysis)
"""

import importlib.util
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
N_SEEDS = w12.N_SEEDS
LR0 = w12.SAC_KW["learning_rate"]
VARIANTS = {"lrdecay": {"lr": LR0, "lr_end": LR0 / 10}, "lr1e4": {"lr": 1e-4}}
w12.LR_VARIANTS.update(VARIANTS)          # module level: worker processes need it too
LABELS = {"sac_2026": "constant 3e-4 (Week 12)", "sac_2026_lrdecay": "decaying 3e-4 -> 3e-5",
          "sac_2026_lr1e4": "constant 1e-4"}
LATE = 100_000
DROP_S = 5.0


def train(job):
    """Week 12's training run, from this module so the worker processes can find it."""
    return w12.train(job)


def jobs():
    return [(f"sac_2026_{v}_s{s}", "2026", s, w12.STEPS, None, None, v) for v in VARIANTS for s in range(N_SEEDS)]


def run_stats(r):
    steps = np.array([c["steps"] for c in r["curve"]])
    lost = np.array([c["lost"] for c in r["curve"]])
    best_so_far = np.minimum.accumulate(lost)
    late = steps > w12.STEPS - LATE
    return {"final": lost[-1], "best": lost.min(), "last5": lost[-5:].mean(),
            "late_drops": int(np.sum(late & (lost > best_so_far + DROP_S))),
            "swing": float(np.median(np.abs(np.diff(lost[late])))),
            "ends_bad": bool(lost[-1] > lost.min() + DROP_S)}


def analyse():
    groups = {k: w12.load_runs(k) for k in LABELS}
    check(all(len(r) == N_SEEDS for r in groups.values()), f"all {N_SEEDS} seeds of every group are there")
    for k in ("sac_2026_lrdecay", "sac_2026_lr1e4"):
        check(all(r["lr"] == k.split("_")[-1] for r in groups[k]), f"{LABELS[k]}: runs were trained with that schedule")
    print(f"\nPer run (2026 from scratch): final / best / mean of last 5 (s), late drops > {DROP_S:.0f} s "
          f"and median swing over the last {LATE:,} steps")
    rows = []
    for k, runs in groups.items():
        for r in runs:
            st = run_stats(r)
            rows.append({"group": k, "seed": r["seed"], **st})
            print(f"  {LABELS[k]:<24} s{r['seed']}  {st['final']:+7.2f} / {st['best']:+6.2f} / {st['last5']:+6.2f}   "
                  f"drops {st['late_drops']:2d}   swing {st['swing']:5.2f}" + ("   ends > best + 5 s" if st["ends_bad"] else ""))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(w12.RL_DIR, "week18_stability.csv"), index=False)
    print("\nPer group, mean ± sd over seeds (median)")
    summ = {}
    for k in LABELS:
        g = d[d["group"] == k]
        summ[k] = {c: float(g[c].mean()) for c in ("final", "best", "last5", "late_drops", "swing")}
        summ[k]["ends_bad"] = int(g["ends_bad"].sum())
        print(f"  {LABELS[k]:<24} final {g['final'].mean():+6.2f} ± {g['final'].std(ddof=1):5.2f} ({g['final'].median():+6.2f})"
              f"   last 5 {g['last5'].mean():+6.2f}   late drops {g['late_drops'].sum():2d}   swing {g['swing'].median():4.2f}"
              f"   runs ending > best + 5 s: {summ[k]['ends_bad']}")
    base = summ["sac_2026"]
    for k in ("sac_2026_lrdecay", "sac_2026_lr1e4"):
        s = summ[k]
        report(d[d["group"] == k]["late_drops"].sum() < d[d["group"] == "sac_2026"]["late_drops"].sum()
               and s["swing"] < base["swing"], f"{LABELS[k]}: fewer late drops and smaller swings than constant 3e-4")
        report(np.median(d[d["group"] == k]["final"]) <= np.median(d[d["group"] == "sac_2026"]["final"]) + 1.0,
               f"{LABELS[k]}: final score no worse (median within 1 s)")
    plot(groups)


def plot(groups):
    colors = {"sac_2026": w12.C1, "sac_2026_lrdecay": w12.C3, "sac_2026_lr1e4": w12.C4}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (k, runs) in zip(axes, groups.items()):
        for r in runs:
            st = [c["steps"] / 1e3 for c in r["curve"]]
            ax.plot(st, [c["lost"] for c in r["curve"]], color=colors[k], linewidth=1.2, alpha=0.85)
        ax.axvspan((w12.STEPS - LATE) / 1e3, w12.STEPS / 1e3, color=w12.GRID, alpha=0.5, linewidth=0)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Training steps (thousands)", color=w12.MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=w12.GRID, linewidth=0.8)
        ax.tick_params(colors=w12.MUTED, labelsize=9)
        ax.set_title(f"{LABELS[k]}: each seed", fontsize=10, color=w12.INK, loc="left")
    axes[0].set_ylabel("Slower than the 2026 benchmark (s)", color=w12.MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(w12.FIG_DIR, "week18_stability.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args != ["analyse"]:
        todo = [j for j in jobs() if not os.path.exists(os.path.join(w12.RL_DIR, f"{j[0]}.json"))]
        if todo:
            print(f"Training {len(todo)} runs on 2026 from scratch, {w12.STEPS:,} steps ({w12.WORKERS} at a time)")
            with Pool(w12.WORKERS) as pool:
                pool.map(train, todo, chunksize=1)
    if args != ["train"]:
        analyse()
        print(f"\nAll checks passed. Results in {w12.RL_DIR}")
