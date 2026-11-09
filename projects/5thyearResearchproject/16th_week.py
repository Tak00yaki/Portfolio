"""Week 16: retraining in 2026 with a fresh pit decision.

In real 2026 the 2024 agents' energy comes back within ~50,000 steps but their pit/tyre
loss stays at +5 s after 200,000 (Weeks 13-14), because 2024 stint lengths outlast 2026
tyres; with 2024 tyres kept, retraining recovers in 10,000 steps (Week 15). So the slow
part is unlearning the stint habit. Two ways to drop it before retraining (A59):

    "fresh pit output"          the actor's pit output (its row of the mean and log-std
                                layers) re-initialised; shared layers, energy outputs and
                                critic kept
    "fresh pit output + critic" the same, and the critic (Q-networks and target) re-
                                initialised too, since it holds the values of 2024 stints

Both: the actor's optimiser state is reset, then trained 200,000 steps on 2026 exactly
as the full retraining (A55), 5 seeds, compared with that and with from scratch.

1. Check the resets: only the intended parameters change, the energy outputs are the
   same as the 2024 agent's on 2026 states.
2. Train (10 runs).
3. Table at 0, 10k, 50k, 100k, 200k steps: lost, pit/tyre and energy, against full
   retraining and from scratch; plot.

Usage: python 16th_week.py            (trains what is missing, then the analysis)
       python 16th_week.py analyse    (only the analysis, from results/rl/)
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
N_SEEDS = w12.N_SEEDS
PIT = 3                             # index of the pit output in [F, D, X, P]
BUDGETS = (0, 10_000, 50_000, 100_000, 200_000)
VARIANTS = {"pit": "fresh pit output", "pit_critic": "fresh pit output + critic"}
COMPARE = {"sac_2024to2026_long": "full retraining (Weeks 13-14)", "sac_2026": "from scratch"}


# --- Resets (registered at module level so the worker processes have them) -------------------

def _fresh_rows(layer, rows):
    import torch
    new = torch.nn.Linear(layer.in_features, layer.out_features)     # PyTorch's default initialisation
    with torch.no_grad():
        layer.weight[rows] = new.weight[rows]
        layer.bias[rows] = new.bias[rows]


def _fresh_actor_optimiser(model):
    model.actor.optimizer = model.actor.optimizer.__class__(model.actor.parameters(),
                                                            **model.actor.optimizer.defaults)


def reset_pit(model):
    _fresh_rows(model.actor.mu, [PIT])
    _fresh_rows(model.actor.log_std, [PIT])
    _fresh_actor_optimiser(model)


def reset_pit_critic(model):
    reset_pit(model)
    for q in model.critic.q_networks:
        for layer in q:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    model.critic_target.load_state_dict(model.critic.state_dict())
    model.critic.optimizer = model.critic.optimizer.__class__(model.critic.parameters(),
                                                              **model.critic.optimizer.defaults)


w12.RESETS.update({"pit": reset_pit, "pit_critic": reset_pit_critic})


def train(job):
    """Week 12's training run, from this module so the worker processes can find it."""
    return w12.train(job)


def jobs():
    return [(f"sac_2024to2026_{v}_s{s}", "2026", s, w12.STEPS, f"sac_2024_s{s}", v)
            for v in VARIANTS for s in range(N_SEEDS)]


# --- 1. Check the resets ------------------------------------------------------------------------

def check_resets():
    import torch
    from stable_baselines3 import SAC
    print("Step 1: what each reset changes")
    path = os.path.join(w12.MODEL_DIR, "sac_2024_s0", "model")
    env = w12.make_env("2026")
    states = []
    obs, _ = env.reset()
    for _ in range(30):                 # some 2026 states from a nominal race
        states.append(obs)
        obs, *_ = env.step(env.encode(env.sim.fuel_nominal, env.sim.split(0.0, env.sim.h_free)[::2], None))
    x = torch.as_tensor(np.array(states))
    for v, f in w12.RESETS.items():
        if v not in VARIANTS:
            continue
        base, new = SAC.load(path), SAC.load(path)
        torch.manual_seed(0)
        f(new)
        changed = {}
        for (name, a), (_, b) in zip(base.policy.named_parameters(), new.policy.named_parameters()):
            if not torch.equal(a, b):
                changed[name] = (a != b).any(dim=-1).nonzero().ravel().tolist() if a.dim() > 1 else "all"
        actor = {k: r for k, r in changed.items() if k.startswith("actor.")}
        critic = {k for k in changed if k.startswith("critic")}
        check(set(actor) == {"actor.mu.weight", "actor.mu.bias", "actor.log_std.weight", "actor.log_std.bias"}
              and actor["actor.mu.weight"] == [PIT] and actor["actor.log_std.weight"] == [PIT],
              f"{VARIANTS[v]}: in the actor only the pit row of the mean and log-std layers changes")
        check(bool(critic) == (v == "pit_critic"),
              f"{VARIANTS[v]}: critic {'re-initialised' if critic else 'kept'} ({len(critic)} tensors changed)")
        with torch.no_grad():
            a0 = base.actor(x, deterministic=True).numpy()
            a1 = new.actor(x, deterministic=True).numpy()
        same = np.abs(a0[:, :PIT] - a1[:, :PIT]).max()
        check(same < 1e-6, f"{VARIANTS[v]}: fuel, deploy and harvest outputs unchanged on 2026 states "
              f"(largest difference {same:.1g})")
        print(f"    pit output before {a0[:, PIT].mean():+.2f}, after {a1[:, PIT].mean():+.2f} (mean over states, -1..1)")


# --- 3. Analysis --------------------------------------------------------------------------------

def at(runs, n):
    return pd.DataFrame([next(c for c in r["curve"] if c["steps"] == n) for r in runs])


def analyse():
    runs = {v: w12.load_runs(f"sac_2024to2026_{v}") for v in VARIANTS}
    runs.update({k: w12.load_runs(k) for k in COMPARE})
    labels = {**VARIANTS, **COMPARE}
    check(all(len(r) == N_SEEDS for r in runs.values()), f"all {N_SEEDS} seeds of every run are there")
    print("\nStep 3: 2026, seconds behind the benchmark, mean ± sd (median) over seeds [pit/tyre, energy]")
    rows = []
    for n in BUDGETS:
        print(f"  {n:>7,} steps")
        for k, rs in runs.items():
            if n == 0 and k == "sac_2026":
                continue
            d = at(rs, n)
            rows.append({"steps": n, "run": k, **{x: float(d[x].mean()) for x in ("lost", "pit", "energy")},
                         "lost_sd": float(d["lost"].std(ddof=1)), "lost_median": float(d["lost"].median())})
            print(f"    {labels[k]:<30} {d['lost'].mean():+6.2f} ± {d['lost'].std(ddof=1):5.2f} ({d['lost'].median():+6.2f})"
                  f"  [{d['pit'].mean():+6.2f}, {d['energy'].mean():+6.2f}]"
                  + ("   (untrained pit output: stops at random, the cost of the reset)" if n == 0 and k in VARIANTS else ""))
    pd.DataFrame(rows).to_csv(os.path.join(w12.RL_DIR, "week16_fresh_pit.csv"), index=False)
    print("  each run: lost at " + " / ".join(f"{n // 1000}k" for n in BUDGETS) + ", final plan")
    for v in VARIANTS:
        for r in runs[v]:
            print(f"    {r['name']:<30} " + " / ".join(f"{at([r], n)['lost'][0]:+6.2f}" for n in BUDGETS)
                  + f"   {r['curve'][-1]['plan']}")
    bar = json.load(open(os.path.join(w12.RL_DIR, "baselines.json")))["2026"]
    bar = min(bar["never_stop"]["lost"], bar["random_median"])
    for v in VARIANTS:
        bad = [r["name"] for r in runs[v] if r["curve"][-1]["lost"] >= bar]
        report(not bad, f"{VARIANTS[v]}: every seed beats never-stop and random" + (f"; not: {', '.join(bad)}" if bad else ""))

    def med(k, key="lost", skip=0):
        """Median curve over seeds; skip=1 drops the 0-step score so curves line up with from scratch."""
        return np.median([[c[key] for c in r["curve"][skip:]] for r in runs[k]], axis=0)
    steps = np.array([c["steps"] for c in runs["sac_2026"][0]["curve"]])
    full, fp, scratch = med("sac_2024to2026_long", skip=1), med("sac_2024to2026_long", "pit", 1), med("sac_2026")
    for v in VARIANTS:
        m, mp = med(v, skip=1), med(v, "pit", 1)
        print(f"  {VARIANTS[v]}: median ahead of full retraining at {np.mean(m < full):.0%} of checkpoints, of from scratch "
              f"at {np.mean(m < scratch):.0%}; pit/tyre median below full retraining at {np.mean(mp < fp):.0%}")
    last = {k: float(np.mean([r["curve"][-1]["pit"] for r in rs])) for k, rs in runs.items()}
    report(min(last[v] for v in VARIANTS) < last["sac_2024to2026_long"],
           "a fresh pit decision ends with a lower pit/tyre loss than full retraining")
    plot(runs, labels, steps)


def plot(runs, labels, steps):
    colors = {"pit": w12.C3, "pit_critic": w12.C4, "sac_2024to2026_long": w12.C2, "sac_2026": w12.C1}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ("lost", "pit", "energy"),
                              ("Slower than the 2026 benchmark (s)", "Pit/tyre loss (s)", "Energy loss (s)")):
        for k, rs in runs.items():
            skip = 1 if k in VARIANTS else 0     # 0 steps with a fresh pit output: random stops, not a transfer score
            st = np.array([c["steps"] for c in rs[0]["curve"][skip:]])
            y = np.array([[c[key] for c in r["curve"][skip:]] for r in rs])
            ax.plot(st / 1e3, np.median(y, axis=0), color=colors[k], linewidth=2, label=labels[k])
            if k in VARIANTS:
                ax.fill_between(st / 1e3, y.min(axis=0), y.max(axis=0), color=colors[k], alpha=0.12, linewidth=0)
        ax.set_xlabel("Training steps on 2026 (thousands)", color=w12.MUTED)
        ax.set_yscale("symlog", linthresh=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=w12.GRID, linewidth=0.8)
        ax.tick_params(colors=w12.MUTED, labelsize=9)
        ax.set_title(title + ", median over seeds", fontsize=10, color=w12.INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=w12.INK)
    fig.tight_layout()
    fig.savefig(os.path.join(w12.FIG_DIR, "week16_fresh_pit.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    check_resets()
    if args != ["analyse"]:
        todo = [j for j in jobs() if not os.path.exists(os.path.join(w12.RL_DIR, f"{j[0]}.json"))]
        if todo:
            print(f"\nStep 2: training {len(todo)} runs on 2026, {w12.STEPS:,} steps ({w12.WORKERS} at a time)")
            with Pool(w12.WORKERS) as pool:
                pool.map(train, todo, chunksize=1)
    if args != ["train"]:
        analyse()
        print(f"\nAll checks passed. Results in {w12.RL_DIR}")
