"""Results pack: every headline number of the project, where it comes from, and what it rests on.

Writes RESULTS.md from the saved results only (no training, no simulation), so each number
traces to a file. The 1e-4 column (Week 19's rerun, A61-A62) fills in once
results/rl/week19_rerun.json exists; until then it reads "pending".

It also checks the numbers:
  - every file and figure it cites exists;
  - every assumption it cites is in ASSUMPTIONS.md;
  - the 3e-4 headline numbers appear in PROGRESS.md as written there;
  - once Week 19 has run, its recomputed 3e-4 numbers match the older weeks' saved files.
Failures are listed in RESULTS.md (section 7), not hidden.

Usage: python results_pack.py
"""

import datetime
import glob
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(HERE, "results", "rl")
N_SEEDS = 5
BUDGETS = (10_000, 50_000, 100_000, 200_000)
YEAR_TO_YEAR_S = 2.0                # A56
C1, C2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e6e5e0"     # as 12th_week.py
FIG_DECOMP = "figures/results_decomposition.png"
CHECKS = []                         # (ok, message)
CITED = set()                       # files and figures cited


def path(*p):
    return os.path.join(HERE, *p)


def load(rel):
    CITED.add(rel)
    with open(path(rel)) as f:
        return json.load(f)


def csv(rel, **kw):
    CITED.add(rel)
    return pd.read_csv(path(rel), **kw)


def runs(prefix):
    out = []
    for s in range(N_SEEDS):
        p = os.path.join(RL, f"{prefix}_s{s}.json")
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def fmt(x, nd=1, sign=True):
    if x is None:
        return "pending"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ks(n):
    return "never" if n is None else f"{n // 1000}k"


def check(ok, msg):
    CHECKS.append((bool(ok), msg))


# --- Numbers at 3e-4, from the files Weeks 11-18 saved ------------------------------------------

def curve_stats(rt, sc):
    """First step where the retrained median is within 2 s of from scratch's final median, and
    the steps where it is ahead of the from-scratch median (as Weeks 13-15 compute them)."""
    steps = [c["steps"] for c in sc[0]["curve"]]
    m_rt = np.median([[c["lost"] for c in r["curve"][1:]] for r in rt], axis=0)
    m_sc = np.median([[c["lost"] for c in r["curve"]] for r in sc], axis=0)
    floor = float(np.median([r["curve"][-1]["lost"] for r in sc]))
    within = [s for s, v in zip(steps, m_rt) if v - floor <= YEAR_TO_YEAR_S]
    return {"first_within_2s": within[0] if within else None,
            "ahead_at": [s for s, a, b in zip(steps, m_rt, m_sc) if a < b]}


def old_numbers():
    dec = load("results/rl/week15_decomposition.json")["decomposition"]
    r24 = runs("sac_2024")
    own = {k: float(np.mean([r["curve"][-1][k] for r in r24])) for k in ("lost", "pit", "energy")}
    transfer = {"own era": own,
                "2026, battery as fractions": dec["fractions"]["both (2026)"],
                "2026, battery as absolute": dec["absolute"]["both (2026)"],
                "tyre loss part": dec["fractions"]["added by tyres"],
                "power unit part": {k: dec[k]["added by power unit"] for k in ("fractions", "absolute")}}
    ad, rec = csv("results/rl/adaptation.csv").set_index("steps"), csv("results/rl/week15_recovery.csv").set_index("steps")
    medians = {"2026": {n: (ad.loc[n, "retrained_lost_median"], ad.loc[n, "scratch_lost_median"]) for n in BUDGETS},
               "PU only": {n: (rec.loc[n, "pu_retrained_lost_median"], rec.loc[n, "pu_scratch_lost_median"]) for n in BUDGETS}}
    zero = {"2026": ad.loc[0, "retrained_lost_median"], "PU only": rec.loc[0, "pu_retrained_lost_median"]}
    adaptation = {"2026": curve_stats(runs("sac_2024to2026_long"), runs("sac_2026")),
                  "PU only": curve_stats(runs("sac_2024topu"), runs("sac_pu"))}
    fp, full, sc = runs("sac_2024to2026_pit"), runs("sac_2024to2026_long"), runs("sac_2026")
    med = lambda rs, k, skip: np.median([[c[k] for c in r["curve"][skip:]] for r in rs], axis=0)  # noqa: E731
    fresh = {"pit_below_full_share": float(np.mean(med(fp, "pit", 1) < med(full, "pit", 1))),
             "ahead_of_scratch_share": float(np.mean(med(fp, "lost", 1) < med(sc, "lost", 0))),
             "final_pit": {"fresh pit": float(np.mean([r["curve"][-1]["pit"] for r in fp])),
                           "full": float(np.mean([r["curve"][-1]["pit"] for r in full]))}}
    groups = {"2024 from scratch": "sac_2024", "2026 from scratch": "sac_2026", "2026 retrained": "sac_2024to2026_long",
              "2026 fresh pit output": "sac_2024to2026_pit", "PU only from scratch": "sac_pu",
              "PU only retrained": "sac_2024topu"}
    finals = {}
    for g, p in groups.items():
        x = pd.Series([r["curve"][-1]["lost"] for r in runs(p)])
        finals[g] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "median": float(x.median())}
    return {"transfer": transfer, "medians": medians, "zero": zero, "adaptation": adaptation, "fresh_pit": fresh,
            "finals": finals}


def new_numbers(old):
    """Week 19's numbers at 1e-4, or None while the rerun is not finished."""
    p = os.path.join(RL, "week19_rerun.json")
    if not os.path.exists(p):
        return None
    w = load("results/rl/week19_rerun.json")
    t = csv("results/rl/week19_adaptation.csv", dtype={"world": str, "lr": str})
    new = {"transfer": w["transfer"]["1e-4"], "adaptation": {k: v["1e-4"] for k, v in w["adaptation"].items()},
           "fresh_pit": w["fresh_pit"]["1e-4"], "finals": {g: v["1e-4"] for g, v in w["finals"].items()}}
    q = t[t["lr"] == "1e-4"].set_index(["world", "steps"])
    new["medians"] = {wd: {n: (q.loc[(wd, n), "retrained_median"], q.loc[(wd, n), "scratch_median"]) for n in BUDGETS}
                      for wd in ("2026", "PU only")}
    new["zero"] = {wd: q.loc[(wd, 0), "retrained_median"] for wd in ("2026", "PU only")}
    # Week 19 recomputes 3e-4 from the runs: it must agree with what the older weeks saved
    w_old = w["transfer"]["3e-4"]
    for key in ("2026, battery as fractions", "2026, battery as absolute", "tyre loss part"):
        check(all(abs(w_old[key][k] - old["transfer"][key][k]) < 1e-6 for k in ("lost", "pit", "energy")),
              f"Week 19's 3e-4 '{key}' equals Week 15's saved numbers")
    for g in old["finals"]:
        check(abs(w["finals"][g]["3e-4"]["mean"] - old["finals"][g]["mean"]) < 1e-9,
              f"Week 19's 3e-4 final for '{g}' equals the saved runs")
    return new


def worse_at_1e4():
    """Seeds whose 1e-4 run ends more than 5 s worse than the same seed at 3e-4, with its final plan."""
    out = []
    for g, prefix in (("2024 from scratch", "sac_2024"), ("2026 from scratch", "sac_2026"),
                      ("2026 retrained", "sac_2024to2026_long"), ("2026 fresh pit output", "sac_2024to2026_pit"),
                      ("PU only from scratch", "sac_pu"), ("PU only retrained", "sac_2024topu")):
        old = {r["seed"]: r["curve"][-1]["lost"] for r in runs(prefix)}
        for r in runs(prefix + "_lr1e4"):
            c = r["curve"][-1]
            if c["lost"] > old[r["seed"]] + 5:
                out.append(f"- {g}, seed {r['seed']}: {fmt(c['lost'])} s at 1e-4 against {fmt(old[r['seed']])} s at "
                           f"3e-4, final plan {c['plan']} (`results/rl/{prefix}_lr1e4_s{r['seed']}.json`)")
    return "\n".join(out) or "- none"


def plot_decomposition(T, Tn):
    """C4 as a figure: what the 2026 tyres and the 2026 power unit each add to the 2024
    agents' loss, split into pit/tyre and energy. One panel per learning rate available."""
    rows = [("2026 tyres only", lambda t: t["tyre loss part"]),
            ("2026 power unit only,\nbattery as fractions", lambda t: t["power unit part"]["fractions"]),
            ("2026 power unit only,\nbattery in absolute MJ", lambda t: t["power unit part"]["absolute"])]
    panels = [("lr 3e-4 (Weeks 12–17)", T)] + ([("lr 1e-4 (Week 19)", Tn)] if Tn else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 3.6), sharey=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    h = 0.36
    hi = max(max(abs(get(t)[k]) for _, get in rows for k in ("pit", "energy")) for _, t in panels)
    for ax, (title, t) in zip(axes[0], panels):
        for off, key, color, label in ((h / 2, "pit", C1, "pit/tyre"), (-h / 2, "energy", C2, "energy")):
            vals = [get(t)[key] for _, get in rows]
            ax.barh(y + off, vals, height=h, color=color, edgecolor="white", linewidth=2, label=label)
            for yy, v in zip(y + off, vals):
                ax.text(v + (0.012 * hi if v >= 0 else -0.012 * hi), yy, f"{v:+.1f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=9, color=INK)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlim(min(0, -0.05 * hi), hi * 1.18)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK)
        ax.set_xlabel("Seconds added to the 2024 agents' loss (mean over 5 seeds)", color=MUTED, fontsize=9)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9, left=False)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK, ncol=2, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path(FIG_DECOMP), dpi=150)
    plt.close(fig)


def stops_table(new):
    """Week 19's stop test (chosen or forced first stops) at both rates, once it has run."""
    if not new:
        return "- At 1e-4: pending (Week 19 step 5)."
    st = load("results/rl/week19_rerun.json")["stops"]
    rows = ["- **Chosen or forced, at both rates** (Week 19 step 5; 2026 benchmark first stint 14 laps, 2026 medium limit 19):",
            "", "| 2024 agents, no retraining | first stop forced | first stint (laps) |", "|---|---|---|"]
    for k, v in st.items():
        rate, era = k.split()
        rows.append(f"| lr {rate}, in {era} | {v['first_forced']} of {N_SEEDS} | "
                    f"{', '.join(f'{a:.0f}' for a in v['first_stint_laps'])} |")
    rows += ["", "  At 1e-4 the agents choose every first stop in 2024 but still run 2024-length first stints in 2026: "
             "**the waiting-for-the-limit habit was partly a 3e-4 effect; the stint length habit is not.**", ""]
    return "\n".join(rows)


def tyre_vs_pu(ten):
    """Whether the tyres' pit/tyre part is clearly larger than the power unit's, for each reading."""
    if not ten:
        return ""
    t = ten["parts"]["tyres: pit"]
    out = []
    for k, name in (("absolute", "absolute-MJ"), ("fractions", "fraction")):
        p = ten["parts"][f"power unit, {k}: pit"]
        if t["lo"] > p["hi"]:
            out.append(f"with the {name} reading the tyres' part is clearly larger (intervals apart)")
        elif t["mean"] > p["mean"]:
            out.append(f"with the {name} reading it is larger but the intervals overlap")
        else:
            out.append(f"with the {name} reading it is not larger")
    return ". " + "; ".join(out)[0].upper() + "; ".join(out)[1:]


def split_sentence(new, T):
    """Claim 3's second half: which way the fraction-reading split goes, from the 10-seed interval when it exists."""
    if not new:
        return ""
    old = (f"at 3e-4 (5 seeds) it was pit/tyre {fmt(T['2026, battery as fractions']['pit'])} s against energy "
           f"{fmt(T['2026, battery as fractions']['energy'])} s, because those agents left their first stop to the age limit (C6)")
    ten = load("results/rl/week19_rerun.json").get("ten_seeds")
    if not ten:
        return ("\n   At 1e-4 (5 seeds) the fraction-reading loss is "
                f"{fmt(new['transfer']['2026, battery as fractions']['energy'])} s energy against "
                f"{fmt(new['transfer']['2026, battery as fractions']['pit'])} s pit/tyre; " + old + ".")
    sp = ten["split"]["fractions"]
    d = sp["energy_minus_pit"]
    if d["lo"] > 0:
        verdict = "more energy than pit/tyre"
    elif d["hi"] < 0:
        verdict = "more pit/tyre than energy"
    else:
        verdict = "about equal between energy and pit/tyre"
    return (f"\n   With fractions, at 1e-4 over {d['n']} seeds the loss is **{verdict}**: energy {fmt(sp['energy']['mean'])} s, "
            f"pit/tyre {fmt(sp['pit']['mean'])} s, difference {fmt(d['mean'])} s (95% CI {fmt(d['lo'])} to {fmt(d['hi'])}, "
            f"{d['positive']} of {d['n']} seeds more energy); " + old + ".")


def ten_seed_table(new):
    """C3/C4 at 1e-4 with 10 seeds: means and 95% intervals of the paired differences (Week 19 step 6, A63)."""
    if not new:
        return ""
    ten = load("results/rl/week19_rerun.json").get("ten_seeds")
    if not ten:
        return "10 seeds at 1e-4 (Week 19 step 6): pending.\n\n"
    n = ten["split"]["fractions"]["pit"]["n"]
    row = lambda name, c: (f"| {name} | {fmt(c['mean'])} | {fmt(c['lo'])} to {fmt(c['hi'])} | "  # noqa: E731
                           f"{c['positive']} of {c['n']} |")
    lines = [f"**1e-4 with {n} seeds** (Week 19 step 6, A63). Differences are paired within a seed: energy minus "
             "pit/tyre in the same race, or a world minus the same agent's own-era score.", "",
             "| Quantity | Mean (s) | 95% CI (s) | Seeds above 0 |", "|---|---|---|---|"]
    for k in ("fractions", "absolute"):
        sp = ten["split"][k]
        lines += [row(f"2026, battery as {k}: pit/tyre", sp["pit"]), row(f"2026, battery as {k}: energy", sp["energy"]),
                  row(f"2026, battery as {k}: energy minus pit/tyre", sp["energy_minus_pit"])]
    lines += [row(f"added by the {name}", c) for name, c in ten["parts"].items()]
    return "\n".join(lines) + "\n\n"


def progress_has(text, label):
    with open(path("PROGRESS.md")) as f:
        body = f.read().replace("−", "-")
    check(text.replace("−", "-") in body, f"PROGRESS.md states {label} as {text}")


# --- Assumptions ---------------------------------------------------------------------------

def assumptions():
    rows = {}
    with open(path("ASSUMPTIONS.md")) as f:
        for line in f:
            m = re.match(r"\| (A\d+[a-z]?) \| (.*) \| ([^|]*) \| ([^|]*) \|\s*$", line)
            if m:
                rows[m.group(1)] = {"text": m.group(2), "status": m.group(3).strip()}
    return rows


def short(text, n=170):
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + " …"


# --- The document --------------------------------------------------------------------------

def both(old, new, get, nd=1):
    """'old | new' cells for one number."""
    return f"{fmt(get(old), nd)} | {fmt(get(new), nd) if new else 'pending'}"


def build():
    old = old_numbers()
    new = new_numbers(old)
    A = assumptions()
    tr = load("results/transfer_barcelona_2024_to_2026.json")
    base = load("results/rl/baselines.json")
    cue = load("results/rl/week17_stint_cue.json")
    stab = csv("results/rl/week18_stability.csv")
    o24, o26 = load("results/optimum_legacy_2022_25_barcelona.json"), load("results/optimum_2026.json")
    opu = load("results/optimum_hybrid_2026pu_2024tyres.json")
    oty = load("results/optimum_hybrid_2024pu_2026tyres.json")
    n_new = len([p for p in glob.glob(os.path.join(RL, "*_lr1e4_s*.json"))
                 if not os.path.basename(p).startswith("sac_2026_lr1e4")])
    T, Tn = old["transfer"], (new or {}).get("transfer")
    P, PA, prate = (Tn, new["adaptation"], "1e-4") if new else (T, old["adaptation"], "3e-4")
    TEN = load("results/rl/week19_rerun.json").get("ten_seeds") if new else None

    def q(part, key, five):
        """Short-answer number: the 10-seed mean and 95% CI when Step 6 has run, else the 5-seed mean."""
        if not TEN:
            return f"{fmt(five)} s"
        c = TEN["parts"][part] if key is None else TEN["split"][part][key]
        return f"{fmt(c['mean'])} s (95% CI {fmt(c['lo'])} to {fmt(c['hi'])})"
    status = ("complete: the 1e-4 column is final" if new else
              f"running: {n_new} of 25 Week 19 runs saved, the 1e-4 column reads \"pending\" until "
              "`19th_week.py analyse` has run")

    # numbers the progress log states, checked against the files
    progress_has(f"+{T['2026, battery as fractions']['lost']:.1f} ± ", "zero-shot loss (fractions)")
    progress_has(f"+{T['2026, battery as absolute']['lost']:.1f} ± ", "zero-shot loss (absolute MJ)")
    progress_has(f"{old['adaptation']['2026']['first_within_2s']:,} steps", "first step within 2 s in 2026")
    progress_has(f"{old['fresh_pit']['pit_below_full_share']:.0%} of checkpoints", "fresh pit below full retraining")
    progress_has(f"stop laps +{tr['tyres']['timing_all_tyres']:.2f} s", "no-RL stop-lap loss")
    real = sorted(tr["tyres"]["real_lost"])
    progress_has(f"lose {real[0]:.1f}–{sorted(real)[-2]:.1f} s", "real 2026 strategies against the benchmark")
    check(cue["summary"]["2026"]["first_forced"] == 4, "Week 17: first stop forced in 4 of 5 races in 2026")

    def tline(name, get):
        a = get(T)
        b = get(Tn) if Tn else None
        return (f"| {name} | " + " / ".join(fmt(a[k]) for k in ("lost", "pit", "energy")) + " | "
                + (" / ".join(fmt(b[k]) for k in ("lost", "pit", "energy")) if b else "pending") + " |")

    def recovery_rows(world):
        lines = [f"| 0 (no retraining) | {fmt(old['zero'][world])} | — | "
                 f"{fmt(new['zero'][world]) if new else 'pending'} | — |"]
        for n in BUDGETS:
            a, b = old["medians"][world][n]
            c, d = new["medians"][world][n] if new else (None, None)
            lines.append(f"| {n:,} | {fmt(a)} | {fmt(b)} | {fmt(c)} | {fmt(d)} |")
        return "\n".join(lines)

    def ad(world, rate):
        src = old if rate == "old" else new
        if not src:
            return "pending", "pending"
        x = src["adaptation"][world]
        return ks(x["first_within_2s"]), (", ".join(ks(s) for s in x["ahead_at"]) or "never") \
            if len(x["ahead_at"]) < 20 else "every checkpoint"

    F = old["fresh_pit"]
    Fn = new["fresh_pit"] if new else None
    s18 = stab.groupby("group").agg(final_mean=("final", "mean"), final_sd=("final", "std"),
                                    final_median=("final", "median"), drops=("late_drops", "sum"))
    for f in ("figures/week11_transfer.png", "figures/week13_adaptation.png", "figures/week15_recovery.png",
              "figures/week16_fresh_pit.png", "figures/week18_stability.png", "figures/week8_real_vs_optimum.png",
              "figures/week6_replay.png", "figures/week2_tyre_fit.png", "figures/week10_battery.png",
              "figures/week11_energy.png"):
        CITED.add(f)
    fig19 = "figures/week19_rerun.png"
    plot_decomposition(T, Tn)
    CITED.add(FIG_DECOMP)
    if new:
        CITED.add(fig19)

    used = ["A1", "A4", "A8", "A12", "A15", "A30", "A31", "A32", "A39", "A43", "A45", "A48", "A49", "A50", "A51",
            "A52", "A53", "A54", "A55", "A56", "A57", "A58", "A59", "A60", "A61", "A62", "A63"]
    missing = [a for a in used if a not in A]
    check(not missing, "every assumption cited is in ASSUMPTIONS.md" + (f" (missing: {', '.join(missing)})" if missing else ""))
    for c in sorted(CITED):
        check(os.path.exists(path(c)), f"cited file exists: {c}")
    with open(path("PROGRESS.md")) as f:
        stale_real = "3.6–5.2 s" in f.read()

    md = f"""# Results pack

Every headline number of the project, the file it comes from, and the assumptions it rests on.
Written by `results_pack.py` from the saved results on {datetime.date.today():%d %b %Y}; rerun it after any
result changes, and do not edit this file by hand.

**Status of the learning-rate rerun (Week 19, A62):** {status}.

**How to read the numbers.** Seconds slower over the race than the exact benchmark of the same world (Weeks 7–8
optimiser), mean over 5 seeds unless marked *median*. Two columns where training is involved:
**3e-4** is the runs of Weeks 12–17; **1e-4** is the same experiments rerun at the more stable learning rate
(Week 18 found 3e-4 makes late single-race collapses, A61). Write from the 1e-4 column once it is complete; the 3e-4
column then shows whether a conclusion depends on the training setting.

---

## 1. The question and the short answer

**Question:** an RL strategist trained under the 2022–25 rules is dropped into a 2026-rules simulator of the same race
(Barcelona, 66 laps). How much time does it lose, which decisions break, and how fast does retraining recover it?

**Hypothesis:** pit timing transfers, energy management breaks.

**What the evidence supports** (lr {prate} numbers{", 10 seeds for claims 1-3 (A63); the 3e-4 runs agree unless the claim says so" if TEN else ("; the 3e-4 runs agree unless the claim says so" if new else "; check them against 1e-4 when it is in")}):
1. **For the power-unit rules alone, the hypothesis holds in proportion: energy breaks far more than pit timing.** With
   the 2024 tyres kept and the battery read in absolute MJ, the old agents lose {q('power unit, absolute: pit', None, P['power unit part']['absolute']['pit'])} on pit/tyre and
   {q('power unit, absolute: energy', None, P['power unit part']['absolute']['energy'])} on energy (C4), and recover within
   {ks(PA['PU only']['first_within_2s'])} steps (C5). Pit timing does not transfer perfectly: its loss is small but above zero{' and above a normal season (2 s)' if TEN and TEN['parts']['power unit, absolute: pit']['mean'] > YEAR_TO_YEAR_S else ''}.
2. **In the real 2026 change, pit timing breaks too, mostly because of the tyres.** The 2026 tyres add
   {q('tyres: pit', None, P['tyre loss part']['pit'])} of pit/tyre loss; the power unit adds
   {q('power unit, absolute: pit', None, P['power unit part']['absolute']['pit'])} (absolute MJ) or {q('power unit, fractions: pit', None, P['power unit part']['fractions']['pit'])} (fractions) (C4){tyre_vs_pu(TEN)}.
3. **How badly energy breaks depends on how the old agent reads the battery:** {q('fractions', 'energy', P['2026, battery as fractions']['energy'])}
   if its actions scale with the new limits, {q('absolute', 'energy', P['2026, battery as absolute']['energy'])} if it keeps its 2024 MJ (C3).
   The two readings bracket the answer (A49, A54).{split_sentence(new, T)}
4. **Energy is relearned quickly; the stint habit is what holds retraining back** (C5, C6): in 2026 the retrained
   agents first come within 2 s of from scratch at {ks(PA['2026']['first_within_2s'])} steps.

---

## 2. Claims and evidence

### C1. The simulator and its benchmark are credible (Weeks 1–11)

| Check | Result | Source |
|---|---|---|
| Paper 1's published optimum | reproduced exactly for 1, 2 and 3 stops | `1st_week.py`, PROGRESS Week 1 (script output) |
| Tyre model, out of sample (2024 Bahrain) | 0.395 s/lap error, against 0.671 s with no tyre terms | `2nd_week.py`, PROGRESS Week 2 |
| A normal season's change (2023 numbers used for 2024) | ~2 s lost (90% interval 0–5 s) | `2nd_week.py`, A56 |
| Replay of real 2024 races | Verstappen −0.5 s over the race; median error 2.9 s over 10 drivers | `5th_6th_week.py`, `figures/week6_replay.png` |
| Exact optimiser | every plan solved; convexity spread ≤ 1e-10 s (A31) | `7th_8th_week.py`, `9th_10th_week.py` |
| Real 2024 strategies against the benchmark | median 1.6 s lost | `figures/week8_real_vs_optimum.png` |
| Real 2026 strategies against the 2026 benchmark | {real[0]:.1f}–{real[-2]:.1f} s lost; one at {real[-1]:.1f} s (Verstappen) | `results/transfer_barcelona_2024_to_2026.json` |

Benchmarks (Barcelona): 2024 **{o24['plan']}** ({o24['race_time_s']:.2f} s); 2026 **{o26['plan']}** ({o26['race_time_s']:.2f} s).

Rests on: A1, A4, A31, A43, A48. Caveats: one track; Alonso's −6 s replay error and the lap-1 grid effect are
unexplained (A30); the model's 2026 deploy (5.3 MJ/lap) is about twice the real cars' 2.8 MJ (A45).

### C2. Without RL, the 2026 change is in energy, not stop timing (Week 11)

The 2024 benchmark plan replayed in the 2026 simulator (no learning involved):

| What is carried over from 2024 | Seconds lost in 2026 |
|---|---|
| The stop laps only, best 2026 compounds | {fmt(tr['tyres']['timing_all_tyres'], 2)} |
| The whole pit plan (laps and compounds) | {fmt(tr['main']['plan'])} |
| Energy as one net number per lap, simulator re-splits | {fmt(tr['main']['energy_net'], 2)} |
| Energy with the 2024 deploy and harvest amounts kept | {fmt(tr['energy_old_amounts_low'])} to {fmt(tr['main']['energy_literal'])} |

Source: `results/transfer_barcelona_2024_to_2026.json`, `figures/week11_transfer.png`. Rests on: A43, A45, A48.
Caveat: the upper energy figure assumes the model's 2026 optimum deploy is right (A45); the lower uses the real cars'.
This test also showed that a single net battery action would hide the energy break, which is why the agent chooses
deploy and harvest separately (A49).

### C3. The 2024 agents in 2026 with no retraining (Weeks 12–14)

Mean over 5 seeds, lost / pit-tyre / energy (s).

| Case | 3e-4 | 1e-4 |
|---|---|---|
{tline("2024 agent in its own era (2024)", lambda t: t["own era"])}
{tline("2026, battery as fractions of each era's limits (A49)", lambda t: t["2026, battery as fractions"])}
{tline("2026, battery in absolute 2024 MJ (A54)", lambda t: t["2026, battery as absolute"])}

{ten_seed_table(new)}For scale (2026): never stop {fmt(base['2026']['never_stop']['lost'])} s, random policy (median)
{fmt(base['2026']['random_median'])} s. A normal season's change: ~{YEAR_TO_YEAR_S:.0f} s (A56).

Source: `results/rl/step4_summary.json`, `results/rl/week15_decomposition.json`, `results/rl/baselines.json`
(1e-4: `results/rl/week19_rerun.json`). Rests on: A48–A51, A54. Caveat: seed spread at 3e-4 is large (sd 7.9 s
fractions, 5.9 s absolute).

### C4. Tyres against power unit (Week 15)

What each part of the 2026 change adds to the 2024 agents' loss, against their own-era score. Mean over seeds,
lost / pit-tyre / energy (s).

| Change | 3e-4 | 1e-4 |
|---|---|---|
{tline("2026 tyres only", lambda t: t["tyre loss part"])}
{tline("2026 power unit only, battery as fractions", lambda t: t["power unit part"]["fractions"])}
{tline("2026 power unit only, battery in absolute MJ", lambda t: t["power unit part"]["absolute"])}

Benchmarks of the mixed worlds: power unit only **{opu['plan']}** (the 2024 plan); tyres only **{oty['plan']}**
(close to the 2026 plan). **The best pit plan follows the tyres.**

Source: `results/rl/week15_decomposition.json`, `results/optimum_hybrid_*.json`; figure `{FIG_DECOMP}`. Rests on: A57. Caveats: the mixed
worlds are not real seasons; the parts do not add up (interaction −12 s fractions, −6 s absolute), because both
changes push the agents into extra stops.

### C5. Recovery: retrained 2024 agents against agents trained from scratch (Weeks 13–15)

*Median* over seeds (s behind the world's benchmark).

**2026 (tyres and power unit):**

| Steps on the new world | Retrained, 3e-4 | From scratch, 3e-4 | Retrained, 1e-4 | From scratch, 1e-4 |
|---|---|---|---|---|
{recovery_rows("2026")}

**2026 power unit, 2024 tyres:**

| Steps on the new world | Retrained, 3e-4 | From scratch, 3e-4 | Retrained, 1e-4 | From scratch, 1e-4 |
|---|---|---|---|---|
{recovery_rows("PU only")}

| | 3e-4 | 1e-4 |
|---|---|---|
| 2026: retrained first within 2 s of from scratch's final level | {ad("2026", "old")[0]} | {ad("2026", "new")[0]} |
| 2026: retrained ahead of from scratch at | {ad("2026", "old")[1]} | {ad("2026", "new")[1]} |
| PU only: retrained first within 2 s | {ad("PU only", "old")[0]} | {ad("PU only", "new")[0]} |
| PU only: retrained ahead of from scratch at | {ad("PU only", "old")[1]} | {ad("PU only", "new")[1]} |

Source: `results/rl/adaptation.csv`, `results/rl/week15_recovery.csv`, `figures/week13_adaptation.png`,
`figures/week15_recovery.png` (1e-4: `results/rl/week19_adaptation.csv`{", `" + fig19 + "`" if new else ""}).
Rests on: A55, A56, A58. Caveat: "faster" is in steps on the new world only; the retrained agents also had 200k
steps of 2024 training (fair-budget check still to do, A58).

### C6. Why pit timing fails in 2026: the stint habit (Weeks 16–17, 19)

Week 17, lr 3e-4 agents:
- The 2024 agents choose their first stop in only {N_SEEDS - cue['summary']['2024']['first_forced']} of 5 races in 2024 and {N_SEEDS - cue['summary']['2026']['first_forced']} of 5 in 2026: otherwise the tyre age limit (A48)
  stops them (first stop forced in {cue['summary']['2024']['first_forced']} of 5 races in 2024,
  {cue['summary']['2026']['first_forced']} of 5 in 2026). In 2024 that limit falls where the benchmark stops anyway.
- No single input of the agent can turn most first-stint laps into a chosen stop: the habit is to wait for the limit.
- Stops the agents do choose come at the same tyre-age fraction in both eras (median
  {cue['summary']['2024']['age fraction at chosen stops'][1]:.2f} in 2024, {cue['summary']['2026']['age fraction at chosen stops'][1]:.2f} in 2026).
- After 2026 training the first stop is chosen in every from-scratch and fully retrained run.

{stops_table(new)}
- Resetting only the pit output before retraining (fresh pit output, A59):

| | 3e-4 | 1e-4 |
|---|---|---|
| Pit/tyre median below full retraining, share of checkpoints | {F['pit_below_full_share']:.0%} | {f"{Fn['pit_below_full_share']:.0%}" if Fn else "pending"} |
| Lost median ahead of from scratch, share of checkpoints | {F['ahead_of_scratch_share']:.0%} | {f"{Fn['ahead_of_scratch_share']:.0%}" if Fn else "pending"} |
| Final pit/tyre: fresh pit output / full retraining (s) | {fmt(F['final_pit']['fresh pit'])} / {fmt(F['final_pit']['full'])} | {(fmt(Fn['final_pit']['fresh pit']) + " / " + fmt(Fn['final_pit']['full'])) if Fn else "pending"} |

Source: `results/rl/week17_stint_cue.json`, `results/rl/week19_rerun.json` (stops), `results/rl/week16_fresh_pit.csv`, `figures/week16_fresh_pit.png`.
Rests on: A48, A59, A60. **Caveat for the write-up:** part of the old agents' "pit/tyre" skill is the environment's
age limit doing their stopping (A60); without forced stops they would have had to learn the first stop.

### C7. Training stability (Weeks 18–19)

2026 from scratch, final score at 200k steps:

| Learning rate | Mean ± sd (median) | Late drops > 5 s |
|---|---|---|
""" + "\n".join(f"| {lab} | {fmt(s18.loc[g, 'final_mean'], 2)} ± {s18.loc[g, 'final_sd']:.2f} ({fmt(s18.loc[g, 'final_median'], 2)}) | {int(s18.loc[g, 'drops'])} |"
                for g, lab in (("sac_2026", "constant 3e-4 (Weeks 12–17)"), ("sac_2026_lrdecay", "decaying 3e-4 → 3e-5"),
                               ("sac_2026_lr1e4", "constant 1e-4"))) + """

Every group's final score at both rates (mean ± sd (median)):

| Group | 3e-4 | 1e-4 |
|---|---|---|
""" + "\n".join(f"| {g} | {fmt(v['mean'])} ± {v['sd']:.1f} ({fmt(v['median'])}) | "
                + (f"{fmt(new['finals'][g]['mean'])} ± {new['finals'][g]['sd']:.1f} ({fmt(new['finals'][g]['median'])})" if new else "pending") + " |"
                for g, v in old["finals"].items()) + f"""

Source: `results/rl/week18_stability.csv`, `figures/week18_stability.png` (1e-4: `results/rl/week19_rerun.json`).
Rests on: A50, A53, A61, A62. Caveat: a lower learning rate removes late collapses but can lock a seed into a poor
plan early. Seeds ending more than 5 s worse at 1e-4 than at 3e-4 (from the saved runs so far):
{worse_at_1e4()}

---

## 3. Figures

| Figure | Shows | Claim | Status |
|---|---|---|---|
| `figures/week2_tyre_fit.png` | tyre degradation fit | C1 | final |
| `figures/week6_replay.png` | replay of real 2024 races | C1 | final |
| `figures/week8_real_vs_optimum.png` | real strategies against the benchmark | C1 | final (Bahrain) |
| `figures/week10_battery.png`, `figures/week11_energy.png` | energy trade-off, 2024 against 2026 | C2 | final |
| `figures/week11_transfer.png` | no-RL transfer test | C2 | final |
| `figures/week13_adaptation.png` | recovery in 2026 | C5 | 3e-4; replace with `{fig19}` |
| `figures/week15_recovery.png` | recovery, power unit only | C5 | 3e-4; replace with `{fig19}` |
| `{FIG_DECOMP}` | what tyres and power unit each add (C4), drawn by this script | C4 | {"both rates" if new else "3e-4; 1e-4 panel added when Week 19 is in"} |
| `figures/week16_fresh_pit.png` | fresh pit output | C6 | 3e-4 |
| `figures/week18_stability.png` | learning-rate test | C7 | final |
| `{fig19}` | recovery at both rates, both worlds | C5, C7 | {"final" if new else "pending"} |

`figures/week12_recovery_v1.png` is superseded (equal pit bins, A52): do not use.

---

## 4. Assumptions the conclusions rest on

| ID | Assumption (start of the entry) | Status |
|---|---|---|
""" + "\n".join(f"| {a} | {short(A[a]['text'])} | {A[a]['status']} |" for a in used if a in A) + """

Full text: `ASSUMPTIONS.md`.

---

## 5. Limitations (for the threats-to-validity section)

- **One race, one track** (Barcelona 2024 and 2026, dry, no safety car). Nothing here shows the result holds elsewhere.
- **Simulator, not cars:** lap times and energy come from a calibrated model; 2026 deploy is about twice the real
  cars' (A45), so absolute-MJ energy losses are upper bounds.
- **Two battery readings bracket the energy result** (A49, A54); which a real old strategist resembles is a judgement.
- **Forced stops:** the age limit stops old agents for them (A48, A60), which flatters their pit/tyre skill.
- **Training:** SAC only, 5 seeds, 200k steps (A50); the 3e-4 runs carry late collapses (A53, A61).
- **Budget:** retrained agents had 2024 training on top of their 2026 steps (A58).
- **Mixed worlds** are counterfactual constructions (A57).
- **Unexplained:** Alonso's −6 s replay error and the lap-1 grid effect (A30); A12's original sources not yet checked.

---

## 6. Open items before writing

- Week 19 rerun: finish, run `19th_week.py analyse`, rerun this script.
- Fair-budget check (A58) and the age limit without a free stop (A60): PROGRESS.md next steps 3–4.
- Deploy gap (A45), Barcelona race recharge notice (A42), A12 originals, A8 for heavier 2026 cars.
""" + ("- **Fix:** A48 and PROGRESS.md Week 12 say real 2026 strategies lose \"3.6–5.2 s (VER 17.9 s)\"; the current "
       f"saved result is {real[0]:.1f}–{real[-2]:.1f} s (VER {real[-1]:.1f} s).\n" if stale_real else "") + """
---

## 7. Number checks

""" + "\n".join(f"- {'ok' if ok else '**FAILED**'}: {msg}" for ok, msg in CHECKS) + "\n"
    with open(path("RESULTS.md"), "w") as f:
        f.write(md)
    bad = [m for ok, m in CHECKS if not ok]
    print(f"RESULTS.md written: {len(CHECKS)} number checks, {len(bad)} failed; 1e-4 column "
          + ("complete" if new else f"pending ({n_new} of 25 runs)"))
    for m in bad:
        print(f"  FAILED: {m}")


if __name__ == "__main__":
    build()
