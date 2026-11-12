# Progress log

One entry per week block: what was built, the results, what went wrong, and what is still open.
Assumptions are numbered in [ASSUMPTIONS.md](ASSUMPTIONS.md) (A1–A63); every number below can be
reproduced by running that week's script with the project's `.venv`.

**Project:** train an RL strategist on a 2022–25-rules race simulator, drop it into a 2026-rules
simulator, and measure the time lost per race and how fast retraining recovers it.
Hypothesis: pit timing transfers, energy management breaks.

**Calibration race:** 2024 Bahrain GP, 57 laps (Weeks 1–10); 2024 and 2026 Spanish GP at Barcelona, 66 laps (Week 11 on).

---

## Week 1: simplest race model and brute-force optimiser (`1st_week.py`)

**Built**
- Paper 1's race model: lap time = base + `k0 + k1·tyre age`.
- A brute-force search over every 1–3 stop plan.

**Results**
- Reproduces Paper 1's published optimum exactly for every stop count:

  | Stops | Plan | Race time |
  |---|---|---|
  | 1 | S22 → M33 | 5237.65 s |
  | 2 | S16 → M21 → S18 | 5240.95 s |
  | 3 | S12 → S14 → S14 → M15 | 5252.23 s |

**Problems**
- Paper 1 writes the tyre formula with age counted from 1, but only counting from 0 reproduces its
  numbers (A1).
- The base lap and pit loss are not published, so they were inferred from the paper's race totals (A3).

---

## Week 2: fit the model to real data (`2nd_week.py`, commit `fc25019`)

**Built**
- 2024 Bahrain from FastF1, with an OpenF1 fallback when live timing is blocked.
- Tyre fit on free-air laps (≥ 2 s to the car ahead), with an all-laps fit as a sensitivity check.
- Pit loss and start-tyre ages from the same race.
- Optimiser run with the fitted numbers and compared with the real strategies.
- Robustness tests: traffic, out-of-sample drivers, and fitting on 2023 to predict 2024.

**Results**
- Tyre fit (584 free-air laps):

  | Compound | k1 (degradation per lap) | k0 (pace offset vs soft) |
  |---|---|---|
  | Soft | 0.1318 ± 0.0079 s/lap | 0, fixed |
  | Hard | 0.1033 ± 0.0035 s/lap | +0.159 ± 0.094 s |

- Medium was not run, so it was not fitted (A14).
- Pit loss 24.88 s (median of 37 stops). Start tyres: 3-lap-old softs.
- Optimum if any start tyre is allowed: H20 → S17 → H20 (5563.90 s).
- Optimum starting on softs, as every team did: S15 → H21 → H21 (+6.3 s).
- Out-of-sample error 0.395 s per lap, against 0.671 s with no tyre terms.
- 2023 → 2024: the 2023 model predicts 2024 laps almost as well as the 2024 model itself
  (0.389 s vs 0.381 s).
- Choosing the plan with 2023 numbers loses about 2 s in 2024 (90% interval 0–5 s). The loss
  comes entirely from the tyres; the pit-loss change costs nothing.

**Problems**
- Hard-tyre k0 moves with traffic (−0.01 s on all laps, +0.16 s in free air). This is why the main
  fit uses free-air laps only (A4).
- FastF1 live timing is blocked on some networks, so the OpenF1 fallback was added (2023+ only).

---

## Weeks 3–4: ETH-style simulator (`racesim.py`, `3rd_4th_week.py`, commit `f4a632e`)

**Built** (each feature added and checked one at a time)
- Fuel mass.
- Tyre wear that depends on car mass.
- Fuel and battery use per lap as decisions, with their limits.
- A Gymnasium environment.
- All era numbers moved into `eras/legacy_2022_25.toml`; the assumptions log started (A1–A25).

**Results**
- Fuel fixed at 0.03 s/kg leaves a track-evolution term of −0.0069 s/lap; base lap 94.176 s.
- Adding fuel and mass-dependent wear leaves the best plan unchanged: S15 → H21 → H21.
- Optimising energy for that plan saves 3.49 s, giving 5571.28 s.
- The environment passes Gymnasium's checker and replays the reference exactly.
- Baselines: "never stop unless forced" is +36.8 s; a random policy is +39.3 s median.

**Problems / limits**
- The reference optimiser alternates between plan and energy, so it was not guaranteed optimal
  (A20). Weeks 7–8 showed it was.
- Several values are guesses that need sources: the 0.03 s/kg fuel effect (A8), the fuel energy
  content (A12) and the energy lap-time effects (A16, A19).

---

## Weeks 5–6: old-era energy rules, in/out laps, replay check (`5th_6th_week.py`)

Commits `1b9b873` and `6912532`, 15 Oct.

**Built**
- FIA 2022–25 limits with article citations (2024 Technical Regulations Issue 4):

  | Limit | Value | Source |
  |---|---|---|
  | MGU-K power | 120 kW | Art. 5.3.3 |
  | Battery deploy | 4 MJ/lap | Art. 5.3.2 |
  | Battery harvest | 2 MJ/lap | Art. 5.3.2 |
  | Battery window | 4 MJ | Art. 5.3.2 |
  | Fuel flow | 100 kg/h | Art. 5.2.3 |
  | Race fuel | 110 kg | 2021 Sporting Regs Art. 30.5, A26 |

- Energy terms with diminishing returns.
- Separate in-lap and out-lap losses, measured from FastF1.
- A replay of every lead-lap finisher's real strategy.
- **Step 3b** (added 15 Oct): the same pit-loss method on other races.

**Results**
- The FIA caps that aren't modelled directly never bind: full deploy needs 33 s at 120 kW, and the
  fuel band is tighter than the flow limit.
- Pit loss: in-lap 2.47 s + out-lap 21.83 s = **24.3 s per stop**, just under the 25–30 s
  expected (ETH: 26.6 s).
- Step 3b shows this is the track, not the method:

  | Race | Pit loss per stop |
  |---|---|
  | 2024 Zandvoort | 24.0 s |
  | 2022 Bahrain | 24.2 s |
  | 2024 Bahrain | 24.9 s |
  | 2023 Bahrain | 26.2 s |
  | 2024 Singapore | 30.3 s |

- Replay: Verstappen is −0.5 s over his whole race (0.009%); median error 2.9 s across 10 drivers.
- Stroll's −10.4 s error is mostly lap 1. He lost 12.1 s there, against the 5.4 s the simulator
  charges everyone; Hülkenberg lost 16.4 s.

**Problems**
- **Commit `1b9b873` accidentally replaced `2nd_week.py` with an older version.** That removed
  `prepare()` and broke Weeks 3–4 and 5–6. Found and restored in `6912532`.
- FastF1 downloads were blocked; other races came from OpenF1.
- One call was missed when the Week 2 loaders were changed to take a race name; fixed.
- Stroll's error was first blamed on his 30-lap hard stint. That was wrong; it was lap 1.
- The check "every driver within 0.1% without lap 1" failed: Alonso is −6.0 s off (0.107%), and
  no cause was found. The check was replaced with "lap 1 explains the worst case", not loosened,
  and Alonso is logged as unexplained (A30).

---

## Weeks 7–8: the exact optimiser, the benchmark (`7th_8th_week.py`, commit `ac7a74d`, 16 Oct)

**Built**
- `RaceSim.exact` lists every legal 1–2 stop plan: 9,620 in all (74 with 1 stop, 9,546 with 2).
- For each plan, CasADi + IPOPT finds the best fuel and battery use per lap.
- The energy problem is convex for a fixed plan (A31), so the best over all plans is the exact
  optimum.
- The benchmark is saved in `results/`.

**Results**
- **Benchmark: S15 → H21 → H21, 5571.280 s.** All 9,620 solves succeed, in 56 s total.
- Checks:
  - race times match the simulator to within 3e-12 s;
  - IPOPT is never worse than the old SciPy solver;
  - random starting points all reach the same optimum;
  - all limits hold exactly.
- The Weeks 3–4 heuristic found the same plan.
- The 10 fastest plans are within 0.26 s of each other.
- Other plan types are clearly worse: medium instead of hard +1.2 s, best 1-stop +12.0 s, every
  3-stop plan ≥ 10.7 s slower before energy is optimised (A32).
- Optimising energy saves 3.5–3.9 s on any plan.
- Real 2024 strategies (A33): median 1.6 s lost; Sainz 0.01 s, Verstappen 2.6 s, Stroll 6.3 s.
- **What the optimum does with energy:**
  - fuel runs at 110% until lap 14 (shedding weight early), then tapers to 90%;
  - the battery deploys the same 0.07 MJ every lap.

  Both are set by the model's simple energy terms (A16, A19), not by anything learned from data.

**Problems**
- IPOPT relaxes limits slightly by default, so it used about 1 mg of fuel too much. It now
  respects the limits exactly (`bound_relax_factor = 0`).
- The first 3-stop check compared the wrong numbers and failed; it now compares the best 3-stop
  plan with the optimum.
- A clean-up edit matched too much and broke two functions. Caught and fixed before committing.
- Figure labels overlapped; fixed.

---

## Weeks 9–10: a real energy trade-off, from physics (`9th_10th_week.py`, `pointmass.py`, 29 Oct)

**Why:** in Weeks 7–8 the battery deployed the same 0.07 MJ every lap. With the guessed energy terms
(A16, A19) there was no energy decision to get right, so energy could not be what "breaks" in 2026.

**Built**
- `pointmass.py`: a point-mass car (power, drag, mass) driven along each lap's full-throttle straights.
  Each straight starts at the measured full-throttle speed and ends with the lap's measured braking
  profile, so more speed means braking a little earlier.
- Speed traces from OpenF1 `car_data` (FastF1 telemetry is still blocked): VER, PER, SAI and LEC,
  203 clean laps, 8 in-laps and 8 out-laps.
- From the mini-sim, three curves of lap time against energy:
  - **deploy** (+120 kW where it saves most);
  - **clip** (harvesting at full throttle where it costs least);
  - **coast** (lift and coast to save fuel).
- From the braking samples:
  - **free harvest**: what the MGU-K can recover under braking;
  - the in-lap and out-lap bonus;
  - how much the nominal lap already lifts.
- A new `"physics"` energy model in `racesim.py`, selected by `energy.model` in the settings file. The
  agent still chooses one net battery number per lap; within the lap the simulator splits it into
  deploy, free harvest and extra harvest (A39). The exact optimiser has all three as variables.
- The old model is kept as `"placeholder"`, and the Weeks 3–8 scripts pin it. Rerunning them gives the
  same numbers as before (benchmark 5571.280 s).

**Results**
- Car: 627 kW at the wheels, CdA 1.75 m² (the four drivers within 2.8%); braking 2.3–2.8 g.
- Energy terms, physics against the old guesses:

  | | Physics | Weeks 3–8 guess |
  |---|---|---|
  | Deploy 4 MJ | −2.20 s | −0.83 s |
  | Value of the next MJ at the nominal deploy | 0.53 s/MJ | 0.25 s/MJ |
  | Clip 1 MJ / 2 MJ | +0.11 s / +0.39 s | (no such term) |
  | Save 10% fuel by coasting | +0.2 s | +0.47 s |
  | Most fuel per lap | 100.6% of nominal | 110% |

- **Free harvest from braking: 2.04 MJ/lap (median), above the FIA's 2 MJ cap on 65% of laps.**
  In-laps +0.29 MJ, out-laps +0.22 MJ.
- Lift before braking: 0.43 s/lap (median 0), so the cars ran flat out and fuel can barely go above nominal.
- **New benchmark: S15 → H21 → H21 again, 5572.285 s** (`results/optimum_legacy_2022_25_physics.json`).
  The time isn't comparable with 5571.280 s, because the energy terms have a different zero point. Energy
  optimisation now saves 2.49 s (was 3.49 s), almost all of it from spending the starting 4 MJ.
- Real 2024 strategies lose a median of 1.59 s against it (was 1.6 s), between 0.01 and 6.39 s.
- **The reviewer's expected pattern does not appear under the 2022–25 rules:**
  - every lap deploys 2.07 MJ and harvests exactly the 2 MJ cap, pit laps included;
  - the in-laps don't charge more, and the out-laps don't deploy more.

  The reason is physical, not a modelling gap. Braking alone nearly fills the 2 MJ cap, and clipping the
  rest costs less (at most 0.39 s/MJ) than deploying it gains (0.53 s/MJ). So the optimum always fills
  the cap, and the extra braking energy on the pit laps has nowhere to go. This holds on every lap under
  both mini-sim calibrations.
- **Sensitivity, harvest cap raised to 4 MJ/lap:** the cap stops binding and the trade-off appears.
  - Every lap clips 1.7 MJ and deploys 3.84 MJ, 25.3 s faster over the race.
  - The in-laps now charge (net −0.16 MJ against +0.09 on a normal lap), keeping the 0.29 MJ bonus.
  - The out-laps still don't deploy more: a MJ is worth the same on every lap, so the bonus is spread
    over all later laps.
- **Convexity (A31) still holds:** 10 random starts on 5 plans each, in both cases, reach the same time
  (spread ≤ 1e-10 s). Deploy gain is concave, clip and coast costs are convex, and every limit is linear.
- 2026 preview (same car, 350 kW, nothing else changed): braking could recover 4.1 MJ/lap, and deploying
  8 MJ saves 4.6 s.

**What this means for the hypothesis**
- Under the 2022–25 rules the energy decision is trivial: harvest 2 MJ, deploy evenly, spend the
  starting charge.
- An old-era agent can therefore learn only "deploy flat". If the 2026 harvest cap doesn't bind the way
  the old one does, that habit is exactly what should break.
- So the test now depends on the 2026 settings, not on making the old era harder.

**Problems**
- First mini-sim (straight-line braking model) was 6% fast. Straight-line braking can't model
  trail-braking into a corner, so braking now follows the measured profile.
- **The validation check "90% of laps within 2% of the measured straight times" failed (2.9%).** The point
  mass is slow mid-straight: the real car accelerates harder early and de-rates late.
  - Matching each straight's time exactly needs a per-straight power change. That fit is badly
    conditioned: +190 kW to remove 0.09 s.
  - The check was replaced, not loosened. The new check asks whether that alternative calibration changes
    the conclusion; it doesn't.
  - It does lower the deploy gain by 10% on average (17% at most), now logged as A35's uncertainty.
- The first sensitivity run came out 0.45 s *slower* with a looser cap, which can't be right. The battery
  term's zero point had moved with the cap; it is now held at the real-rules value (A40).
- The rear-axle share of braking (0.4) is a guess. At 0.5, or with a 5% electrical loss, the free harvest
  is 2.12 or 1.94 MJ. Either way the cap still binds, because clipping the gap is cheap.

---

## Week 11: 2026 settings, benchmark and a transfer test with no RL (`11th_week.py`, 2 Nov)

**Built**
- `eras/2026.toml`, with limits from the FIA 2026 Technical Regulations Issue 18 and article citations.
- `eras/legacy_2022_25_barcelona.toml`: 2024 recalibrated on the same track. The Bahrain file is unchanged.
- One pipeline for both years: tyres, base lap, pit and start losses, car, mini-sim curves, free harvest.
- The mini-sim now applies the 2026 deploy taper with speed (Art. C5.2.8).

**Which race:** the 2026 Sakhir race was cancelled, and October's "Bahrain GP" ran in Kuala Lumpur. Barcelona
replaces it: raced in both eras, dry, no safety car in either year (two VSCs in 2026), same 66 laps. Hungary was the other
candidate, but its 2026 telemetry drifts off the lap-time clock from about lap 6.

**Results**

| | 2024 Barcelona | 2026 Barcelona |
|---|---|---|
| Engine, no deploy | 570 kW (measured) | 398 kW (from fuel flow) |
| Drag area | 1.61 m² | 0.92 m² |
| Free harvest per lap | 1.45 MJ | 3.08 MJ |
| Harvest cap per lap | 2 MJ, reached every lap | 8.5 MJ, never reached |
| Optimum deploy per lap | 2.06 MJ | 5.26 MJ (limited by the straights) |
| Clipping per lap | 0.18–0.55 MJ | 2.10 MJ |
| In-lap net battery | +0.06 MJ, same as any lap | **−0.77 MJ, charges** |
| Benchmark | S20 → M23 → M23, 5366.24 s | M15 → S25 → S26, 5588.03 s |

- All 13,616 plans solved in both eras. Convexity holds in both (spread 3.6e-12 s).
- **The in-lap charging the reviewer expected appears under the real 2026 rules.** The harvest cap no longer
  binds, as Weeks 9–10 predicted. The out-lap still doesn't deploy more.
- Transfer test, 2024 optimum in the 2026 simulator:

  | Seconds lost per race | Linear tyre fit | No soft stints |
  |---|---|---|
  | Stop laps only, best 2026 compounds | 2.3 | 0.4 |
  | Whole old pit plan | 16.9 | 5.6 |
  | Energy, net number, simulator re-splits | 0.25 | 0.25 |
  | Energy, old deploy and harvest amounts kept | 22–58 | 22–58 |

- **2026 is a big change, and only in energy. Stop timing transfers (0.4–2.3 s); the deploy and harvest amounts don't.**
- **The current environment can't show this.** Its one net battery number lets the simulator re-split each lap perfectly,
  so an old agent would lose only 0.25 s on energy. RL on the current action space would wrongly reject the hypothesis.
- None of the unpublished inputs changes the picture: 7 MJ cap, fuel 65–85 kg, rear share and deploy gain −10% leave
  net energy at 0.2–0.3 s and the plan loss at about 17 s; tyre wear +20% raises the plan loss to 22.7 s.

**Problems**
- First 2026 fit gave a no-deploy car of 163 kW, because I assumed real deploy of 8.5 MJ. I called CdA 0.93 unphysical too;
  that was wrong (a hand check supports it). Engine power now comes from fuel flow, and real deploy is measured.
- An upper-envelope car fit gave an impossible 174–246 kW engine and was dropped.
- The clip fit missed by 0.055 s at amounts nobody would use. It is now fitted only up to what can be deployed.
- The 2026 benchmark passes every check but isn't believable: all real strategies lose 15–29 s. The linear soft fit is the
  likely cause, so results are given both ways (A43). This needs a real fix before RL scores against it.
- The 58 s energy figure rests on the model deploying nearly twice the real cars' 2.77 MJ, so it is an upper bound (A45).
- I first read Section C Issue 17, then Issue 19, but the Barcelona race (14 June) ran under Issue 18 (7 May). The
  articles we cite were compared across 18 and 19: the same apart from a per-cylinder flow limit and a Low Grip sentence.
- Hungary was the first choice (no safety car in either year), but its 2026 telemetry drifts off the lap-time clock
  from about lap 6: cars appeared to cross the line at 83–311 km/h. Barcelona passes the same test (8 km/h spread).

---

## Week 12: stint limits, split battery action, the RL agent (`12th_week.py`, 4–5 Nov)

**Left from Week 11, done first**
- **Battery action split:** the agent now chooses deploy and extra harvest itself (A49). The environment replays each
  era's exact optimum to 1e-4 s.
- **2026 soft tyre fixed:** each compound is only used up to the age the data covers (A48); longer stints are illegal for
  the optimiser and forced stops in the environment. 2026 soft wear went from 0.089 to 0.165 s/lap.
- **3 stops rechecked (A32b):** with the age limits a 3-stop plan wins in 2026. New benchmarks:
  2024 S20 → M23 → M23, 5366.38 s; 2026 M14 → H17 → H17 → H18, 5599.34 s. Real 2026 strategies now lose 4.3–5.9 s
  (Verstappen 18.6 s), not 15–29 s. (Corrected in Week 19: this entry first said 3.6–5.2 s and 17.9 s, which the
  saved results don't support.)
- No-RL transfer test, final: stop laps +1.15 s, whole old plan +13.3 s, energy with the net number +0.28 s, energy
  with old amounts kept +22 to +58 s.

**Built:** SAC (Stable-Baselines3 2.7.0) on the split-action environment, VecNormalize, reward in seconds per lap,
5 seeds per case, 200,000 steps per run (A50). Cases: train 2024; train 2026 from scratch; 2024 agents dropped into
2026 unchanged; those agents then trained 100,000 more steps on 2026. Losses split into pit/tyre and energy.

**Results (v2, mean ± sd over 5 seeds, seconds behind each era's benchmark)**

| Case | Lost | Pit/tyre | Energy |
|---|---|---|---|
| 2024 agent in 2024 | +2.4 ± 0.3 | +1.0 ± 0.5 | +1.4 ± 0.4 |
| 2026 agent in 2026, from scratch | +11.8 ± 13.9 | +8.6 ± 13.3 | +3.3 ± 1.2 |
| 2024 agent in 2026, zero-shot | +26.8 ± 7.9 | +18.7 ± 9.1 | +8.1 ± 2.5 |
| …same, starting on softs (A51) | +12.9 ± 7.5 | +4.5 ± 8.2 | +8.5 ± 5.3 |
| 2024 agent after 100k steps on 2026 | +12.0 ± 8.5 | +9.8 ± 9.2 | +2.2 ± 0.9 |

- Baselines: never stop (unless forced) 2024 +27.7 s, 2026 +17.8 s; random 2024 +46.8 s, 2026 +80.2 s.
  Every 2024 seed beats both; 2026 seed 3 does not (it learned, then collapsed, A53). Without it, 2026 from scratch is
  +5.6 ± 2.0 s.
- **Energy breaks on arrival:** zero-shot energy loss +8 s, against +1.4 s (2024) and +3.3 s (2026) for agents in their
  own era. The typical lap is close to the 2026 optimum (deploy 4.3–5.2 MJ against 5.27) because actions are fractions of
  each era's limits (A49). The loss comes from uneven laps and battery left unused at the flag (up to 4 MJ). So the
  scaled action hides most of the 22–58 s the no-RL test found with the old MJ amounts kept.
- **Pit/tyre also breaks, for a tyre reason:** the 2024 agents keep 2024 stint lengths (about 21 laps on softs), which
  2026's faster-wearing tyres don't allow, so the environment forces stops (S16 → S16 → S14 → H20 on 4 of 5 soft-start
  runs). That is a tyre change between seasons, not the power-unit rules.
- **Recovery:** the 2024 agents start ahead (+13.7 s against +24.0 s after 10,000 steps) but end behind (+7.6 s against
  +4.8 s at 100,000 steps, medians). Their energy loss recovers as fast as from scratch (about 40,000 steps); the
  pit/tyre loss is what stays high.

**Problems**
- **v1 pit output (equal quarters):** 4 of 5 2024 agents learned to stop after lap 1, about 17 s each. Half the range now
  means "no stop" (A52); v1 is kept in `results/rl_v1_equal_pit_bins` and `models/v1_equal_pit_bins`.
- **v1 zero-shot:** some agents stopped on almost every lap, because the 2026 start compound (medium) was never seen at lap 0
  in 2024. The soft-start check (A51) separates that out.
- The random baseline first stopped on most laps (+1042 s). It now matches Weeks 3–4 (a stop with probability 2/laps).
- My first pit-bin code put "no stop" on half the range even by default; caught by checking the old mapping.
- The check "every 2026 seed beats the baselines" failed (seed 3). It is reported, not loosened or hidden, with every
  run's final, best and last-5 scores.
- The session restarted mid-run; the training job survived and finished.

---

## Weeks 13–14: the main experiments, and small fixes (`13th_14th_week.py`, `run_all.py`, 5 Nov)

**Built**
- **Transfer, a second reading (A54):** the same 2024 agents in 2026 with the battery action kept in 2024 MJ rather
  than scaled to the 2026 limits.
- **Adaptation (A55):** the 2024 agents trained on 2026 for 200,000 steps, the same budget as from scratch. Scored every
  10,000 steps. The first 100,000 steps reproduce Week 12 exactly.
- **Context (A56):** everything compared with a normal season's change, the ~2 s lost by planning 2024 Bahrain with
  2023 numbers (Weeks 1–2).
- **A8 and A12 sourced and checked** (step 4), and `run_all.py`, which runs every week in order, stops at the first
  failure, and lists every saved result or figure that changed.

**Results** (seconds behind the 2026 benchmark, mean ± sd over 5 seeds)

| 2024 agent in 2026 | Lost | Pit/tyre | Energy |
|---|---|---|---|
| zero-shot, battery as fractions of each era's limits (A49) | +26.8 ± 7.9 | +18.7 ± 9.1 | +8.1 ± 2.5 |
| zero-shot, battery in absolute 2024 MJ (A54) | +49.3 ± 5.9 | +18.0 ± 8.0 | +31.3 ± 5.7 |

| Steps on 2026 | Retrained 2024 agent | From scratch |
|---|---|---|
| 0 | +26.8 ± 7.9 (median +21.9) | — |
| 10,000 | +18.9 ± 9.9 (+13.7) | +28.2 ± 9.9 (+24.0) |
| 50,000 | +14.6 ± 7.9 (+13.7) | +7.6 ± 2.8 (+7.0) |
| 100,000 | +12.0 ± 8.5 (+7.6) | +5.4 ± 1.3 (+4.8) |
| 200,000 | +9.1 ± 4.2 (+8.4) | +11.8 ± 13.9 (+5.0) |

- **Energy breaks, and how much depends on the reading:** if the old strategist keeps its 2024 MJ, the energy loss is
  +31 s (the no-RL test gave +22 to +58 s); if its actions scale with the new limits, +8 s. Pit/tyre loss is the same,
  +18 s, either way.
- **Pit/tyre does not transfer either:** the 2024 stint lengths exceed 2026 tyre life (A48), so stops are forced.
- **Recovery is fast for energy, slow for pit stops:** the energy loss falls to the from-scratch level within
  10,000–50,000 steps. The pit/tyre loss is still +5.4 s at 200,000 steps (from scratch: +2.6 s at 100,000). The 2024
  start helps only for the first 20,000 steps; after that the agents trained from scratch are ahead. The retrained median
  first comes within 2 s of the from-scratch level at 140,000 steps.
- **Context:** zero-shot loses 11× (fractions) to 24× (absolute MJ) a normal season's change. Even after 200,000 steps the
  transfer still costs +3.3 s (median), more than a normal season.
- Every retrained seed beats never-stop and random. Seed 3 reached +4.8 s and ended at +14.9 s (energy +12 s): the same
  late collapse as Week 12 (A53).
- **A8 (0.03 s/kg):** the TUM race simulation's published values are 0.023 (Sakhir) to 0.033 s/kg (Catalunya), for 2019
  cars. Our data can't separate mass from track evolution; the point-mass straights give only 0.003 s/kg, because mass
  costs time mainly in corners. Over 0.023–0.033 every benchmark keeps its plan and moves ≤ 0.2 s. Kept.
- **A12 (43 MJ/kg):** petrol 43.5–44.0 MJ/kg (Heywood; ORNL), E10 fuel 41.8–42.3. Over that range every 2024 benchmark
  keeps its plan and moves ≤ 0.03 s. Kept.

**Problems**
- The first training launch failed: the worker processes could not import the Week 12 module loaded by file path. A
  wrapper in the Week 13–14 file fixed it.
- The first A8 check moved race times by −51 to +22 s, because only the lap trend was refitted and not the base lap.
  With both moved as a refit would, the change is ≤ 0.2 s.
- I could not open Heywood's table or the 2022 fuel article; both are marked to check against the originals (A12).
- Alonso's −6 s replay error and the lap-1 grid effect are left as logged limitations, as the reviewer suggested.

---

## Week 15: tyres against power unit (`15th_week.py`, 6 Nov)

**Built**
- Two mixed simulators (A57): 2026 rules with 2024 tyres ("power unit only"), and 2024 rules with 2026 tyres ("tyres
  only"). Each has its own exact benchmark. The 2024 agents are dropped into each with no retraining, and the battery is
  read both ways (fractions, A49; absolute 2024 MJ, A54).
- Checks: only the tyre settings differ from the host era, and both readings agree in the 2024-PU world (scale 1).

**Results**
- **Benchmarks:** "power unit only" S20 → M23 → M23, the 2024 plan (2026 is M14 → H17 → H17 → H18); "tyres only"
  M13 → H17 → H18 → H18, close to the 2026 plan. **The best pit plan is set by the tyres, not the power-unit rules.**
- 2024 agents, no retraining, mean over 5 seeds (s behind each world's own benchmark):

  | World | Reading | Lost | Pit/tyre | Energy |
  |---|---|---|---|---|
  | own era (2024) | — | +2.4 | +1.0 | +1.4 |
  | tyres only | either | +22.3 ± 4.9 | +19.1 ± 5.0 | +3.2 ± 0.3 |
  | power unit only | fractions | +18.6 ± 10.3 | +9.1 ± 9.9 | +9.5 ± 4.5 |
  | power unit only | absolute MJ | +35.1 ± 4.7 | +3.0 ± 4.6 | +32.1 ± 4.3 |
  | both (2026) | fractions | +26.8 ± 7.9 | +18.7 ± 9.1 | +8.1 ± 2.5 |
  | both (2026) | absolute MJ | +49.3 ± 5.9 | +18.0 ± 8.0 | +31.3 ± 5.7 |

- **The hypothesis holds for the power-unit rules:** with the old tyres kept and the battery in absolute MJ, pit timing
  loses +3.0 s and energy +32.1 s. The +18 s pit/tyre loss in 2026 comes from the tyres (+18 s added by tyres, +2 s by
  the power unit).
- With the fraction reading, the power-unit-only pit loss is larger (+9.1 s, two seeds stop 3 times). The energy actions
  change the laps, which pushes some agents into extra stops.
- The effects don't add (interaction −12 s fractions, −6 s absolute): both changes push the agents into extra stops, and
  those overlap.
- **Recovery from the power unit alone (A58):** the 2024 agents retrained in "2026 PU, 2024 tyres" against agents trained
  there from scratch (mean, median in brackets; s behind that world's benchmark):

  | Steps | Retrained, PU only | Scratch, PU only | Retrained, 2026 (both) | Scratch, 2026 (both) |
  |---|---|---|---|---|
  | 0 | +18.6 (+21.3) | — | +26.8 (+21.9) | — |
  | 10,000 | +7.2 (+3.0) | +34.6 (+33.6) | +18.9 (+13.7) | +28.2 (+24.0) |
  | 50,000 | +3.0 (+3.0) | +6.8 (+6.2) | +14.6 (+13.7) | +7.6 (+7.0) |
  | 100,000 | +2.8 (+3.0) | +6.3 (+5.8) | +12.0 (+7.6) | +5.4 (+4.8) |
  | 200,000 | +2.4 (+2.9) | +4.3 (+4.5) | +9.1 (+8.4) | +11.8 (+5.0) |

  Without the tyre change, the old agents recover within 10,000 steps (median +3.0 s, inside a normal season's 2 s of
  the from-scratch level) and stay ahead of agents trained from scratch for all 200,000 steps. Energy recovers in about
  10,000 steps and pit/tyre stays near +1 s: the old pit timing is kept and only the energy has to be relearned. In
  real 2026 the 2024 start helps only for 20,000 steps, because the stint habits have to be unlearned. No run collapsed.
  Figure: `figures/week15_recovery.png`.

**Problems**
- None new. The tyres-only world starts on new mediums, which the agents never saw (A51); that start is counted as tyre
  loss.

---

## Week 16: retraining in 2026 with a fresh pit decision (`16th_week.py`, 9 Nov)

**Built**
- Two ways to drop the 2024 stint habit before retraining in 2026 (A59): re-initialise only the actor's pit output
  ("fresh pit output"), or that and the critic ("+ critic"). 5 seeds each, 200,000 steps, as the full retraining
  (Weeks 13–14).
- Checks: only the intended parameters change, and the energy outputs are unchanged on 2026 states.
- `12th_week.RESETS` lets the Week 12 trainer apply a change to a loaded agent before retraining.

**Results** (s behind the 2026 benchmark, mean (median) over 5 seeds)

| Steps | Fresh pit output | + critic | Full retraining | From scratch |
|---|---|---|---|---|
| 10,000 | +18.9 (+22.6) | +39.7 (+36.0) | +18.9 (+13.7) | +28.2 (+24.0) |
| 50,000 | **+5.7 (+6.3)** | +18.2 (+12.3) | +14.6 (+13.7) | +7.6 (+7.0) |
| 100,000 | +5.5 (+4.8) | +16.4 (+9.8) | +12.0 (+7.6) | +5.4 (+4.8) |
| 200,000 | +6.5 (+7.2) | +8.7 (+6.6) | +9.1 (+8.4) | +11.8 (+5.0) |

- **The stint habit is in the pit output:** resetting only that row removes most of the stuck pit/tyre loss. The
  pit/tyre median is below full retraining at 95% of checkpoints, and the agents reach +5.7 s at 50,000 steps, where
  full retraining is at +14.6 s.
- **But it is no better than starting from scratch:** ahead of from scratch at only half the checkpoints, and level at
  100,000 steps. In real 2026 the kept energy layers save little, because energy is relearned within ~50,000 steps
  anyway (Weeks 13–14). In the power-unit-only world (Week 15) the old agent was worth much more, because its pit
  timing still fitted.
- **Keep the critic:** resetting it as well is the slowest option of all until ~200,000 steps (energy +13.8 s at 10,000
  steps). The actor follows an untrained critic and loses its energy skill for a while. Seed 3 ends at +17.5 s.
- The pit/tyre loss of the fresh-pit agents drifts up slightly after 100,000 steps (+3.3 → +4.4 s, mean).

**Problems**
- The 0-step score with a fresh pit output is +1,156 s (random stops on most laps). It is labelled as the cost of the
  reset and left out of the plot (A59), not reported as a transfer result.

---

## Week 17: what cues the stint habit (`17th_week.py`, 10 Nov)

**Built**
- Every stop of the 2024 agents in 2024 and in 2026 (no retraining), chosen or forced (A60), with what the agent saw
  then. A one-input test of the first stint, and the forced stops of every 2026 run after training. No training; a few
  seconds.

**Results**
- **The 2024 agents never choose their first stop:** in 4 of 5 races in 2024 the soft's age limit (A48) stops them on
  lap 21, which is where the 2024 benchmark stops anyway (S20), so relying on it cost nothing. In 2026 they do the same
  on mediums in 4 of 5 races: the medium limit stops them on lap 20 (benchmark: lap 14), and the forced stop puts on
  another medium.
- **No single input triggers it:** setting any one input to any value it takes in the race turns at most 10–30% of
  first-stint laps into a chosen stop (median 26%, one race 56%). The habit is to wait for the limit, not a cue
  that changes in 2026.
- **The stops they do choose transfer:** chosen stops come at the same age fraction in both eras (median 0.87 in 2024,
  0.88 in 2026), so that part of the pit decision is era-independent.
- **Retraining fixes the first stop:** after 2026 training the first stop is chosen in every run, from scratch (0 of 5
  forced) and fully retrained (0 of 5), and in 4 of 5 fresh-pit runs.
- So the "pit/tyre" loss of the 2024 agents is partly the environment's age limit doing their stopping (A60). This
  should go in the write-up as a limitation of A48: with a penalty or no forced stop, the agents would have had to
  learn the first stop themselves.

---

## Week 18: more stable training (`18th_week.py`, 10 Nov)

**What the "collapses" were:** each is one bad deterministic race in which a pit decision flips (a lap-1 stop at
170,000 steps in 2026 seed 3) or the energy plan jumps, with good scores 10–20k steps either side. Smaller swings
appear in other seeds too. The two retrained seed-3 runs start from the same 2024 agent with the same seed, so the
three are not independent failures.

**Built**
- Two learning-rate settings on 2026 from scratch, the same 5 seeds (A61): decaying 3e-4 → 3e-5, and constant 1e-4.
  `12th_week.LR_VARIANTS` lets the Week 12 trainer use them; earlier runs are unchanged. Checked that the rate reaches
  both optimisers.

**Results** (2026 from scratch, 200,000 steps; s behind the benchmark)

| Learning rate | Final, mean ± sd (median) | Mean of last 5 | Late drops > 5 s | Median swing | Runs ending > best + 5 s |
|---|---|---|---|---|---|
| constant 3e-4 (Weeks 12–17) | +11.8 ± 13.9 (+5.0) | +30.0 | 6 | 0.43 s | 1 |
| decaying 3e-4 → 3e-5 | +4.7 ± 0.4 (+4.8) | +4.9 | 0 | 0.33 s | 0 |
| **constant 1e-4** | **+3.3 ± 0.2 (+3.3)** | +4.1 | 1 | 0.20 s | 0 |

- **Both fix the instability; 1e-4 also learns better:** every 1e-4 seed ends between +3.0 and +3.4 s, 1.7 s better than
  the old median, with the spread over seeds cut from 13.9 s to 0.2 s. One 1e-4 seed had a single bad race at
  180,000 steps (+21 s) and recovered.
- **What this means for Weeks 12–17:** those agents were trained at 3e-4, so their scores carry extra noise, and the
  from-scratch 2026 level was about 1.7 s too pessimistic. The conclusions compare groups trained the same way, but the
  numbers in the write-up should come from a rerun at 1e-4 (next steps).

---

## Week 19: the main experiments at learning rate 1e-4, and the results pack (`19th_week.py`, `results_pack.py`, 11–12 Nov)

**Built**
- `19th_week.py` reruns the runs behind the main results at constant lr 1e-4 (A62), under `_lr1e4` names so the 3e-4
  runs stay as they were: 2024 and "2026 PU, 2024 tyres" from scratch, then from the new 2024 agents full retraining
  on 2026, retraining on "2026 PU, 2024 tyres", and the fresh pit output. 25 runs; 2026 from scratch is Week 18's.
  Its analysis puts every headline number at both rates side by side and reports whether each conclusion still holds.
  Tested on the old runs first: it reproduces every saved 3e-4 number, including Week 15's decomposition exactly.
- `15th_week.zero_shot` takes which agents to load (default the original ones; Week 15's results unchanged).
- `results_pack.py` writes `RESULTS.md`: claims C1–C7 with their numbers at both rates, source files, figures,
  assumptions and limitations, from the saved results only, plus 30 number checks (files and assumptions cited exist,
  numbers match PROGRESS.md, Week 19's 3e-4 numbers match the older weeks). It draws
  `figures/results_decomposition.png` (C4). Last script in `run_all.py`.
- **A8 for the heavier 2026 car:** the 2026 check now also runs 0.040 s/kg (+20% on the highest published value; 2026
  cars are 768 kg against 2019's 743 kg). The 2026 benchmark keeps its plan and moves 0.11 s (0.045 s/kg: 0.16 s). Closed.

**Results** (s behind each world's benchmark; mean (median) over 5 seeds; 3e-4 → 1e-4)

| | 3e-4 | 1e-4 |
|---|---|---|
| 2024 agent in 2024 | +2.4 | +1.5 |
| Zero-shot 2026, fractions: lost / pit / energy | +26.8 / +18.7 / +8.1 | +18.4 / +8.0 / +10.5 |
| Zero-shot 2026, absolute MJ: lost / pit / energy | +49.3 / +18.0 / +31.3 | +42.7 / +8.2 / +34.5 |
| Added by the 2026 tyres: pit / energy | +18.2 / +1.8 | +12.5 / +4.7 |
| Added by the 2026 power unit, absolute MJ: pit / energy | +2.0 / +30.7 | +2.0 / +31.3 |
| 2026: retrained vs from scratch at 200k | +9.1 (+8.4) vs +11.8 (+5.0) | +5.5 (+4.2) vs +3.2 (+3.3) |
| 2026: retrained first within 2 s of from scratch | 140k | 160k |
| PU only: retrained vs from scratch at 200k | +2.4 (+2.9) vs +4.3 (+4.5) | +1.4 (+0.9) vs +6.8 (+3.4) |
| PU only: retrained within 2 s / ahead | 10k / every checkpoint | 10k / every checkpoint |
| Fresh pit output: pit/tyre below full retraining | 95% of checkpoints | 95% |
| 2024 agents' first stop forced (in 2024 / in 2026) | 4 / 4 of 5 | 0 / 2 of 5 |

- **Every conclusion holds except one:** with the battery as fractions, the zero-shot loss is no longer mostly
  pit/tyre (5 seeds: energy +10.5, pit/tyre +8.0 s; 10 seeds: about equal, see below). The 1e-4 2024 agents choose their own first stops (none forced in 2024), so in 2026 fewer stops land where
  the age limit puts them, and the pit/tyre loss halves (+18.7 → +8.0 s). The hypothesis's core is unchanged and
  clearer: the power unit alone costs +2.0 s pit/tyre and +31.3 s energy (absolute MJ), and the tyres drive the
  pit/tyre loss.
- **Week 17 revised:** "the agents wait for the age limit" was partly a 3e-4 effect. The stint-length habit is not:
  at 1e-4 the first 2026 stint is still 18–20 laps (benchmark 14, medium limit 19).
- **Retraining in 2026:** the 2024 start still helps only for the first 20k steps, and comes within 2 s of from
  scratch at 160k. Part of the early lead is from scratch learning more slowly at 1e-4 (+57 s median at 10k against
  +24 s at 3e-4). In the power-unit-only world the old agents are ahead at every checkpoint and end at +0.9 s (median).
- **Spread:** smaller in 3 of 6 groups. Two single seeds widen the others: the stuck "PU only" seed 2 (+21.4 s) and
  fresh pit output seed 3 (+11.8 s). Medians are no worse at 1e-4 in all six groups.
- **Ten seeds for the 2024 agents (A63, step 6):** with 5 seeds the fraction-reading split was +2.5 ± 9.0 s (energy minus
  pit/tyre), too noisy to call. 5 more 2024 agents at 1e-4 (seeds 5–9, all at once, 23 min; own-era +1.0 to +2.0 s).
  Paired within a seed, mean [95% CI] over 10 seeds:

  | | Mean (s) | 95% CI (s) |
  |---|---|---|
  | 2026, fractions: energy minus pit/tyre | +0.8 | −5.7 to +7.4 (6 of 10 seeds above 0) |
  | 2026, absolute MJ: energy minus pit/tyre | +24.1 | +15.1 to +33.0 (9 of 10) |
  | Added by the tyres: pit/tyre | +17.0 | +9.4 to +24.6 |
  | Added by the power unit (absolute MJ): pit/tyre / energy | +4.5 / +33.4 | +1.0 to +8.0 / +29.1 to +37.7 |
  | Added by the power unit (fractions): pit/tyre / energy | +6.5 / +9.9 | +2.1 to +10.9 / +4.8 to +15.0 |

  - **With fractions, energy and pit/tyre losses are about equal** at 1e-4; neither dominates. The pit/tyre losses are
    bimodal (about 5 s or about 23 s per seed): a seed either lands on a workable plan or not.
  - **The power unit alone costs some pit/tyre time too** (+4.5 s, above zero and above a normal season's 2 s), but far
    less than energy (+33.4 s). The tyres' pit/tyre part is clearly larger than the power unit's with absolute MJ;
    with fractions the intervals overlap. The write-up wording is softened to match (`RESULTS.md` section 1).
- `RESULTS.md` now uses the 1e-4 numbers (10 seeds for the transfer split); figure `figures/week19_rerun.png`.

**Problems**
- **The Mac slept on battery during training:** runs that take ~22 min took 147 min to 19 h. Nothing was lost (finished
  runs are saved, a restart skips them). Long runs now go with `caffeinate -i -w <pid>`.
- **A lower learning rate can lock a seed in early:** "2026 PU, 2024 tyres" from scratch, seed 2, settled smoothly on a
  lap-1 stop (S1 → M24 → S18 → M23, +21.4 s) and stayed there; at 3e-4 the same seed ended at +4.1 s. So 1e-4 trades
  late collapses for early lock-in, which belongs in the write-up.
- **A stale number found by the pack's checks:** A48 and the Week 12 entry said real 2026 strategies lose 3.6–5.2 s
  (Verstappen 17.9 s); the saved results and the current code give 4.3–5.9 s (18.6 s). Both corrected.

---

## Reference: IAmTomShaw/f1-race-replay

https://github.com/IAmTomShaw/f1-race-replay, checked 16 Oct.

`resources/` holds only screenshots and a GIF, so there's nothing there to use in the research.
The useful parts are in `src/`:

| What | Where | Possible use here |
|---|---|---|
| Tyre warm-up penalty (slicks lose about 0.3 s on the first laps, fading out) | `src/bayesian_tyre_model.py` | Our model puts all cold-tyre loss in the out-lap (A28). Measure the loss on the second lap of new tyres from our own data. |
| Robust per-stint degradation slope (Theil–Sen) | same file | A cross-check of our fitted wear rates that's less sensitive to outlier laps. |
| Hidden-pace filter (tracks each driver's true pace through the race) | same file | Could explain driver-level errors such as Alonso's, or estimate tyre condition for the agent's observations. |
| Track abrasion factor | same file | Scales wear rates between tracks, if more tracks are added. |
| Fuel effect 0.032 s/kg | `StateSpaceConfig` | Close to A8's 0.03, but not a citable source either. |
| Telemetry stream + PitWallWindow | `telemetry.md`, `docs/PitWallWindow.md` | A demo window showing the benchmark or agent's strategy next to a real race replay. |
| Throttle, speed and position telemetry | stream format | Could help replace the placeholder battery model (A19). FastF1 provides the same data directly. |

**Caveats**
- The README says MIT, but the repo has no LICENSE file. Credit it, and check before copying code.
- Its tyre parameters are hand-set starting values, not fitted to data.
- It depends on heavy GUI libraries (arcade, PySide6); keep them out of `.venv`.

---

## Next steps

1. **Write-up (you write; `RESULTS.md` has the numbers):** the hypothesis as two numbers per reading (fractions, absolute MJ), split into tyres and power unit
   (Week 15), the recovery curves (Weeks 13–16), and the 2 s context.
2. **Write-up check:** read `RESULTS.md` section 1 against the claims you want to make; the fraction-reading result
   changed at 1e-4, and with 10 seeds energy and pit/tyre are about equal there (Week 19).
3. **Fair-budget check:** compare a retrained agent with a from-scratch agent at equal total steps (A58).
4. **Age limit without a free stop (A60):** retrain the 2024 agents in an environment where reaching the age limit
   costs time instead of forcing a stop, to see whether they learn to choose the first stop and whether the pit
   timing then transfers.
5. **Still open:** deploy gap (model 5.3 MJ, real ~2.8 MJ, A45); Barcelona's race recharge notice (A42); Nominal Tyre
   Mass (A46); A36; check Heywood's table and the 2022 E10 article (A12); Alonso's replay
   error and lap-1 grid effect (logged limitations); `projects/shot.py`.

## Lessons

- Run every week's script before each commit, not just the current one. The broken `2nd_week.py`
  would have been caught immediately.
- Read `git diff --stat` before committing; a −400 line change to an older week's file is a
  warning sign.
- Check the data before explaining an outlier, and don't loosen a failing check to make it pass.
- Make targeted edits rather than broad find-and-replace.
