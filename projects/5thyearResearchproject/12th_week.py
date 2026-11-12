"""Week 12: the RL agent, and the hypothesis test.

Stable-Baselines3 SAC on F1StrategyEnv with the split battery action (A49): fuel, deploy,
extra harvest and pit, all continuous, the pit output binned into no stop / soft / medium
/ hard. Observations normalised with VecNormalize; reward = base lap - lap time, in
seconds (about -1 to -25 per lap rather than -80).

1. Baselines per era: "never stop unless forced", random policy, the exact benchmark.
2. Train on 2024 Barcelona, N_SEEDS seeds. Check: every seed beats the baselines.
3. Train on 2026 Barcelona from scratch, same seeds and budget: what an agent can reach.
4. Drop the 2024 agents into 2026 unchanged (zero-shot), then keep training them on
   2026, scoring every EVAL_EVERY steps: the time lost and how fast it comes back.
5. Split every agent's loss against the benchmark (the hypothesis test):
       pit/tyre loss = best time for the agent's own plan - benchmark
       energy loss   = agent's time - best time for its plan
   "pit transfers, energy breaks" means the zero-shot energy loss is large and the pit
   loss small. Mean and spread over seeds, never one run.

Usage: python 12th_week.py            (all of it, in parallel; models in models/, git-ignored)
       python 12th_week.py analyse    (only the tables and figures, from results/rl/)
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import racesim  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RL_DIR = os.path.join(HERE, "results", "rl")
MODEL_DIR = os.path.join(HERE, "models")
ERAS = {"2024": "legacy_2022_25_barcelona", "2026": "2026"}
N_SEEDS = 5
STEPS = 200_000             # per training run (~3,000 races)
FINETUNE_STEPS = 100_000    # 2024 agents kept training on 2026
EVAL_EVERY = 10_000
N_RANDOM = 200              # random-policy races per era
WORKERS = 4
NO_STOP_SHARE = 0.5         # A52: "no stop" gets half the pit output (v1, equal bins: results/rl_v1_equal_pit_bins)
SAC_KW = dict(learning_rate=3e-4, buffer_size=300_000, batch_size=256, gamma=0.995,
              learning_starts=5_000, train_freq=1, policy_kwargs=dict(net_arch=[256, 256]))

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


WORLDS = {}                 # other settings by name, registered by later weeks (Week 15's mixed simulators)
RESETS = {}                 # name -> f(model): changes to a loaded agent before retraining (Week 16)
LR_VARIANTS = {}            # name -> {"lr": start, "lr_end": end or None}: learning-rate schedules (Week 18)


def settings(era):
    if era in WORLDS:
        return WORLDS[era]
    return racesim.load_settings(os.path.join(HERE, "eras", f"{ERAS[era]}.toml"))


def make_env(era, start=None):
    """`start`: override the start compound (zero-shot check with the training era's start)."""
    s = settings(era)
    if start is not None:
        s = racesim.override(s, **{"race.start_compound": start})
    return racesim.F1StrategyEnv(s, split_battery=True, no_stop_share=NO_STOP_SHARE)


def benchmark(era):
    with open(os.path.join(HERE, "results", f"optimum_{ERAS.get(era, era)}.json")) as f:
        return json.load(f)["race_time_s"]


def check(cond, msg):
    assert cond, msg
    print(f"  [ok] {msg}")


def report(cond, msg):
    """An expectation about learning, not a correctness check: print the answer either way."""
    print(f"  [{'yes' if cond else 'NO '}] {msg}")
    return cond


# --- Running a policy and splitting its loss ------------------------------------------------

def play(env, policy):
    """One race. Returns (race time, compounds, pit laps, per-lap info)."""
    obs, _ = env.reset()
    compounds, pits, infos, done = [env.state.compound], [], [], False
    while not done:
        lap_before = env.state.lap
        compound_before, age_before = env.state.compound, env.state.age
        obs, _, done, _, info = env.step(policy(obs))
        infos.append(info)
        if info["in-lap"] > 0:
            pits.append(lap_before + 1)
            compounds.append(env.state.compound)
    return env.state.race_time, tuple(compounds), tuple(pits), infos


def split_loss(sim, t_bench, race_time, compounds, pits):
    """Pit/tyre loss = best time for this plan - benchmark; energy loss = the rest."""
    plan_best = sim.optimal_energy(compounds, pits)[0]
    return {"time": race_time, "lost": race_time - t_bench, "pit": plan_best - t_bench,
            "energy": race_time - plan_best, "stops": len(pits),
            "plan": " -> ".join(f"{c[0]}{b - a}" for c, a, b in
                                zip(compounds, (0, *pits), (*pits, sim.laps)))}


def baselines(era):
    env = make_env(era)
    sim, t_b = env.sim, benchmark(era)
    nominal = env.encode(sim.fuel_nominal, sim.split(0.0, sim.h_free)[::2], None)
    t, c, p, _ = play(env, lambda o: nominal)
    never = split_loss(sim, t_b, t, c, p)
    rng = np.random.default_rng(0)
    p_stop = 2 / sim.laps               # as Weeks 3-4: random energy, a stop now and then

    def random_policy(obs):
        a = rng.uniform(0, 1, 4)
        a[3] = rng.uniform(0.25, 1) if rng.random() < p_stop else 0.0
        return a.astype(np.float32)
    rand = [split_loss(sim, t_b, *play(env, random_policy)[:3]) for _ in range(N_RANDOM)]
    return {"never_stop": never, "random_median": float(np.median([r["lost"] for r in rand])),
            "random_iqr": [float(np.quantile([r["lost"] for r in rand], q)) for q in (0.25, 0.75)]}


# --- Training ------------------------------------------------------------------------------

def evaluate(model, vec, era, start=None):
    """Deterministic race with the model and the normalisation it was trained with (frozen)."""
    env = make_env(era, start)

    def policy(obs):
        o = vec.normalize_obs(obs[None, :])
        return model.predict(o, deterministic=True)[0][0]
    t, c, p, infos = play(env, policy)
    out = split_loss(env.sim, benchmark(era), t, c, p)
    out["deploy_mj"] = float(np.median([i["deploy"] for i in infos]))
    out["harvest_mj"] = float(np.median([i["harvest_free"] + i["harvest_extra"] for i in infos]))
    return out


def train(job):
    """One run: name, era to train on, seed, steps, and optionally the run to start from."""
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.set_num_threads(1)
    name, era, seed, steps, init, *opts = job
    reset = opts[0] if opts else None
    lr_name = opts[1] if len(opts) > 1 else None
    lr0 = LR_VARIANTS[lr_name]["lr"] if lr_name else None
    lr1 = LR_VARIANTS[lr_name].get("lr_end") if lr_name else None
    path = os.path.join(MODEL_DIR, name)
    log = os.path.join(RL_DIR, f"{name}.json")
    if os.path.exists(log):
        return log
    venv = DummyVecEnv([lambda: make_env(era)])
    if init is None:
        vec = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        model = SAC("MlpPolicy", vec, seed=seed, verbose=0, **{**SAC_KW, **({"learning_rate": lr0} if lr0 else {})})
    else:
        vec = VecNormalize.load(os.path.join(MODEL_DIR, init, "vecnormalize.pkl"), venv)
        model = SAC.load(os.path.join(MODEL_DIR, init, "model"), env=vec, seed=seed)
        model.learning_starts = 2_000           # fresh buffer: learn from 2026 races, not stale 2024 ones
        if reset is not None:
            RESETS[reset](model)
    curve = []
    if init is not None:            # zero-shot score before any 2026 training
        vec.training = False
        curve.append({"steps": 0, **evaluate(model, vec, era)})
        vec.training = True
    t0 = time.time()
    for done in range(EVAL_EVERY, steps + 1, EVAL_EVERY):
        if lr0 is not None:             # set per chunk: SB3's own progress resets with each learn() call
            from stable_baselines3.common.utils import ConstantSchedule
            frac = (done - EVAL_EVERY) / steps
            model.lr_schedule = ConstantSchedule(lr0 if lr1 is None else lr0 + (lr1 - lr0) * frac)
        model.learn(EVAL_EVERY, reset_num_timesteps=False)
        vec.training = False
        curve.append({"steps": done, **evaluate(model, vec, era)})
        vec.training = True
    os.makedirs(path, exist_ok=True)
    model.save(os.path.join(path, "model"))
    vec.save(os.path.join(path, "vecnormalize.pkl"))
    out = {"name": name, "era": era, "seed": seed, "init": init, "reset": reset, "lr": lr_name, "no_stop_share": NO_STOP_SHARE,
           "minutes": (time.time() - t0) / 60, "curve": curve}
    if init is None and era == "2024":      # the 2024 agent dropped into 2026 as it is
        vec.training = False
        out["zero_shot_2026"] = evaluate(model, vec, "2026")
    with open(log, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  {name}: {out['minutes']:.0f} min, final {curve[-1]['lost']:+.2f} s "
          f"(pit {curve[-1]['pit']:+.2f}, energy {curve[-1]['energy']:+.2f})", flush=True)
    return log


def run_all():
    os.makedirs(RL_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Step 1: baselines")
    base = {}
    for era in ERAS:
        base[era] = baselines(era)
        b = base[era]
        print(f"  {era}: never stop {b['never_stop']['lost']:+.1f} s ({b['never_stop']['plan']}), random median "
              f"{b['random_median']:+.1f} s (IQR {b['random_iqr'][0]:+.1f} to {b['random_iqr'][1]:+.1f})")
    with open(os.path.join(RL_DIR, "baselines.json"), "w") as f:
        json.dump(base, f, indent=1)
    print(f"\nSteps 2-3: SAC, {N_SEEDS} seeds x {STEPS:,} steps on each era ({WORKERS} at a time)")
    first = [(f"sac_{era}_s{s}", era, s, STEPS, None) for era in ERAS for s in range(N_SEEDS)]
    with Pool(WORKERS) as pool:
        pool.map(train, first, chunksize=1)
    print(f"\nStep 4: 2024 agents kept training on 2026, {FINETUNE_STEPS:,} steps")
    tune = [(f"sac_2024to2026_s{s}", "2026", s, FINETUNE_STEPS, f"sac_2024_s{s}") for s in range(N_SEEDS)]
    with Pool(WORKERS) as pool:
        pool.map(train, tune, chunksize=1)


# --- Analysis ------------------------------------------------------------------------------

def zero_shot_same_start():
    """2024 agents in 2026, but starting on softs as every 2024 car did (A51). Scored
    against the 2026 benchmark, which starts on mediums, so the pit loss includes what
    the soft start itself costs: that is printed too."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    sim = make_env("2026", "SOFT").sim
    plans = pd.read_csv(os.path.join(HERE, "results", "all_plans_2026.csv"))
    out = []
    for s in range(N_SEEDS):
        path = os.path.join(MODEL_DIR, f"sac_2024_s{s}")
        if not os.path.exists(path):
            continue
        vec = VecNormalize.load(os.path.join(path, "vecnormalize.pkl"), DummyVecEnv([lambda: make_env("2026", "SOFT")]))
        vec.training = False
        out.append(evaluate(SAC.load(os.path.join(path, "model")), vec, "2026", "SOFT"))
    best_soft = min(sim.optimal_energy(c, p)[0] for c, p in sim.plans(2))
    return out, best_soft - benchmark("2026")


def load_runs(prefix):
    runs = []
    for s in range(N_SEEDS):
        p = os.path.join(RL_DIR, f"{prefix}_s{s}.json")
        if os.path.exists(p):
            with open(p) as f:
                runs.append(json.load(f))
    return runs


def summarise(rows, label):
    d = pd.DataFrame(rows)
    out = {k: (float(d[k].mean()), float(d[k].std(ddof=1)) if len(d) > 1 else 0.0) for k in ("lost", "pit", "energy")}
    print(f"  {label:<38} lost {out['lost'][0]:+6.2f} ± {out['lost'][1]:.2f} s   pit {out['pit'][0]:+6.2f} ± "
          f"{out['pit'][1]:.2f}   energy {out['energy'][0]:+6.2f} ± {out['energy'][1]:.2f}   (n = {len(d)})")
    return out


def analyse():
    with open(os.path.join(RL_DIR, "baselines.json")) as f:
        base = json.load(f)
    r24, r26, rt = load_runs("sac_2024"), load_runs("sac_2026"), load_runs("sac_2024to2026")
    print("\nStep 5: losses against each era's benchmark, mean ± sd over seeds")
    table = {}
    table["2024 agent in 2024"] = summarise([r["curve"][-1] for r in r24], "2024 agent in 2024")
    table["2026 agent in 2026"] = summarise([r["curve"][-1] for r in r26], "2026 agent in 2026 (from scratch)")
    table["2024 agent in 2026, zero-shot"] = summarise([r["zero_shot_2026"] for r in r24], "2024 agent in 2026, zero-shot")
    same, soft_cost = zero_shot_same_start()
    table["2024 agent in 2026, zero-shot, soft start"] = summarise(same, "  same, starting on softs as in 2024")
    print(f"    (the best soft-start plan itself is {soft_cost:+.2f} s off the 2026 benchmark)")
    table["2024 agent in 2026, retrained"] = summarise([r["curve"][-1] for r in rt], "2024 agent after 2026 training")
    for era in ERAS:
        b = base[era]
        print(f"  baselines {era}: never stop {b['never_stop']['lost']:+.1f} s, random median {b['random_median']:+.1f} s")
    print("  each run: final / best / median of the last 5 scores (s); a run can learn and then collapse (A53)")
    for runs in (r24, r26, rt):
        for r in runs:
            c = [x["lost"] for x in r["curve"]]
            print(f"    {r['name']:<20} {c[-1]:+7.2f} / {min(c):+6.2f} / {np.median(c[-5:]):+7.2f}")
    failed = {}
    for name, runs in (("2024", r24), ("2026", r26)):
        bar = min(base[name]["never_stop"]["lost"], base[name]["random_median"])
        failed[name] = [r["name"] for r in runs if r["curve"][-1]["lost"] >= bar]
        report(not failed[name], f"every {name} seed beats never-stop and random in its own era"
               + (f"; not: {', '.join(failed[name])}" if failed[name] else ""))
    print("  plans: 2024 " + ", ".join(r["curve"][-1]["plan"] for r in r24))
    print("         zero-shot in 2026 " + ", ".join(r["zero_shot_2026"]["plan"] for r in r24))
    print("         zero-shot, soft start " + ", ".join(r["plan"] for r in same))
    lap1 = sum(r["curve"][-1]["plan"].split(" -> ")[0][1:] == "1" for r in r24 + r26)
    print(f"  {lap1} of {len(r24) + len(r26)} agents stop after lap 1 in their own era (a learned habit, A52)")
    print("         2026 " + ", ".join(r["curve"][-1]["plan"] for r in r26))
    print("  deploy MJ/lap (median lap): 2024 agents " + ", ".join(f"{r['curve'][-1]['deploy_mj']:.2f}" for r in r24)
          + "; zero-shot in 2026 " + ", ".join(f"{r['zero_shot_2026']['deploy_mj']:.2f}" for r in r24)
          + "; 2026 agents " + ", ".join(f"{r['curve'][-1]['deploy_mj']:.2f}" for r in r26))
    z = table["2024 agent in 2026, zero-shot"]
    print("\n  Hypothesis 'pit transfers, energy breaks', zero-shot, mean over seeds:")
    print(f"    pit/tyre loss {z['pit'][0]:+.2f} s, energy loss {z['energy'][0]:+.2f} s")
    with open(os.path.join(RL_DIR, "summary.json"), "w") as f:
        json.dump({"table": table, "baselines": base, "below_baselines": failed}, f, indent=1)
    plot_curves(r26, rt, base)


def plot_curves(r26, rt, base):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, key, title in zip(axes, ("lost", "pit", "energy"),
                              ("Slower than the 2026 benchmark (s)", "Pit/tyre loss (s)", "Energy loss (s)")):
        for runs, color, label in ((r26, C1, "trained on 2026 from scratch"), (rt, C2, "2024 agent, then 2026 training")):
            steps = np.array([c["steps"] for c in runs[0]["curve"]])
            y = np.array([[c[key] for c in r["curve"]] for r in runs])
            ax.plot(steps / 1e3, np.median(y, axis=0), color=color, linewidth=2, label=label)
            ax.fill_between(steps / 1e3, y.min(axis=0), y.max(axis=0), color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel("Training steps on 2026 (thousands)", color=MUTED)
        ax.set_yscale("symlog", linthresh=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(title + ", median and range over seeds", fontsize=10, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "week12_recovery.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    if sys.argv[1:] != ["analyse"]:
        run_all()
    analyse()
    print(f"\nAll checks passed. Results in {RL_DIR}")
