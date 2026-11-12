"""Week 17: what makes the 2024 agents keep their stint lengths in 2026?

The 2024 agents run ~20-lap medium stints in 2026 although the 2026 medium only lasts 19
laps and the agents see tyre age as a fraction of each compound's limit (A48). Three
tests, with the 5 trained 2024 agents (deterministic, battery as fractions, A49):

1. Every stop in 2024 and in 2026 (no retraining): chosen by the agent (pit output in a
   stop bin) or forced by the environment (age limit, A48; or no compound change 20 laps
   from the end, A15), and what the agent saw then (age fraction, wear, laps left).
2. One input at a time in the first stint: each input set to every value it takes in the
   race (11 points) with the others as they were; the share of first-stint laps it can
   turn into a chosen stop. A single cue would turn many.
3. Forced stops after retraining: the share of stops forced in the final race of every
   2026 run (from scratch, full retraining, fresh pit output).

Usage: python 17th_week.py            (a few seconds, no training)
"""

import importlib.util
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w12 = load("week12", "12th_week.py")
check, report = w12.check, w12.report
N_SEEDS = w12.N_SEEDS
OBS = ["battery", "fuel", "fuel mass", "race time", "compound changed", "compound", "wear", "out-lap",
       "last lap", "laps left", "age fraction"]
CUES = {"age fraction": 10, "wear": 6, "laps left": 9}
SWEEP_POINTS = 11
GROUPS = {"from scratch": "sac_2026", "full retraining": "sac_2024to2026_long", "fresh pit output": "sac_2024to2026_pit"}


def agent(name, era):
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    path = os.path.join(w12.MODEL_DIR, name)
    vec = VecNormalize.load(os.path.join(path, "vecnormalize.pkl"), DummyVecEnv([lambda: w12.make_env(era)]))
    vec.training = False
    model = SAC.load(os.path.join(path, "model"))

    def pit_output(obs):
        return model.predict(vec.normalize_obs(np.atleast_2d(obs)), deterministic=True)[0][:, 3]

    def policy(obs):
        return model.predict(vec.normalize_obs(obs[None, :]), deterministic=True)[0][0]
    return policy, pit_output


def race(name, era):
    """One deterministic race: every lap's observation and stint, and every stop."""
    policy, pit_output = agent(name, era)
    env = w12.make_env(era)
    share = env.pit_edges[1]                    # pit output below this: no stop chosen
    obs, _ = env.reset()
    laps, stops, done = [], [], False
    while not done:
        a = policy(obs)
        lap, compound, age = env.state.lap, env.state.compound, env.state.age
        laps.append({"obs": obs.copy(), "first_stint": not stops})
        before = obs.copy()
        obs, _, done, _, info = env.step(a)
        if info["in-lap"] > 0:
            stops.append({"lap": lap + 1, "compound": compound, "age": age + 1, "chosen": bool(a[3] >= share),
                          "pit_output": float(a[3]), **{k: float(before[i]) for k, i in CUES.items()}})
    return laps, stops, pit_output, share


# --- 1. Every stop ------------------------------------------------------------------------------

def step1():
    print("Step 1: the 2024 agents' stops, chosen (C) or forced (F) by the age limit (A48) or the compound rule (A15)")
    rows = []
    for era in ("2024", "2026"):
        for s in range(N_SEEDS):
            _, stops, _, _ = race(f"sac_2024_s{s}", era)
            print(f"  {era} seed {s}: " + ", ".join(
                f"lap {x['lap']} {x['compound'][0]}{x['age']:.0f} {'C' if x['chosen'] else 'F'}" for x in stops))
            rows += [{"era": era, "seed": s, "n": i, **x} for i, x in enumerate(stops)]
    d = pd.DataFrame(rows)
    first = d[d["n"] == 0]
    out = {}
    for era in ("2024", "2026"):
        f, c = first[first["era"] == era], d[(d["era"] == era) & d["chosen"]]
        out[era] = {"first_forced": int((~f["chosen"]).sum()), "chosen": len(c), "stops": int((d["era"] == era).sum()),
                    **{f"{k} at chosen stops": [float(c[k].min()), float(c[k].median()), float(c[k].max())] for k in CUES}}
        print(f"  {era}: first stop forced in {out[era]['first_forced']} of {len(f)} races; {len(c)} of "
              f"{out[era]['stops']} stops chosen; at chosen stops (min / median / max) "
              + "; ".join(f"{k} {lo:.2f} / {m:.2f} / {hi:.2f}" for k in CUES
                          for lo, m, hi in [out[era][f'{k} at chosen stops']]))
    report(out["2024"]["first_forced"] >= N_SEEDS // 2 + 1,
           "in 2024 most agents never choose their first stop: the soft's age limit stops them")
    report(out["2026"]["first_forced"] >= N_SEEDS // 2 + 1, "in 2026 they do the same with the medium")
    a24, a26 = (np.median(d[(d["era"] == e) & d["chosen"]]["age fraction"]) for e in ("2024", "2026"))
    report(abs(a24 - a26) < 0.1, f"chosen stops come at the same age fraction in both eras ({a24:.2f}, {a26:.2f})")
    return d, out


# --- 2. One input at a time ---------------------------------------------------------------------

def step2():
    print(f"\nStep 2: share of first-stint laps one input can turn into a chosen stop ({SWEEP_POINTS} values each)")
    table = {}
    for era in ("2024", "2026"):
        for s in range(N_SEEDS):
            laps, _, pit_output, share = race(f"sac_2024_s{s}", era)
            X = np.array([l["obs"] for l in laps])
            first = np.array([l["first_stint"] for l in laps])
            base = float((pit_output(X[first]) >= share).mean())
            row = {"as raced": base}
            for j, name in enumerate(OBS):
                lo, hi = X[:, j].min(), X[:, j].max()
                if hi - lo < 1e-6:
                    lo, hi = 0.0, 1.0           # constant in this race: its full range
                best = 0.0
                for v in np.linspace(lo, hi, SWEEP_POINTS):
                    Z = X[first].copy()
                    Z[:, j] = v
                    best = max(best, float((pit_output(Z) >= share).mean()))
                row[name] = best
            table[(era, s)] = row
    df = pd.DataFrame(table).T
    with pd.option_context("display.width", 250, "display.float_format", "{:.2f}".format):
        print(df.to_string())
    top = df.drop(columns="as raced").max(axis=1)
    print("  most any one input achieves, per race: " + ", ".join(f"{e} s{s} {v:.2f}" for (e, s), v in top.items()))
    report(float(top.median()) < 0.3, "no single input makes the agents choose a first-stint stop on most laps "
           f"(median best {top.median():.2f}): no one cue, the habit is to wait for the limit")
    return df


# --- 3. Forced stops after retraining -----------------------------------------------------------

def step3():
    print("\nStep 3: forced stops in each run's final 2026 race")
    out = {}
    for label, prefix in {"2024 agents, no retraining": "sac_2024", **GROUPS}.items():
        forced = total = first_forced = 0
        for s in range(N_SEEDS):
            _, stops, _, _ = race(f"{prefix}_s{s}", "2026")
            forced += sum(not x["chosen"] for x in stops)
            total += len(stops)
            first_forced += bool(stops) and not stops[0]["chosen"]
        out[label] = {"forced": forced, "stops": total, "first_forced": first_forced}
        print(f"  {label:<28} {forced:2d} of {total:2d} stops forced; first stop forced in {first_forced} of {N_SEEDS}")
    report(out["full retraining"]["first_forced"] < out["2024 agents, no retraining"]["first_forced"],
           "retraining teaches the agents to choose their first stop before the limit")
    return out


def main():
    d, s1 = step1()
    s2 = step2()
    s3 = step3()
    with open(os.path.join(w12.RL_DIR, "week17_stint_cue.json"), "w") as f:
        json.dump({"stops": d.to_dict("records"), "summary": s1,
                   "one_input_sweep": {f"{e} s{s}": r for (e, s), r in s2.to_dict("index").items()},
                   "forced_after_retraining": s3}, f, indent=1)


if __name__ == "__main__":
    main()
    print(f"\nAll checks passed. Results in {w12.RL_DIR}")
