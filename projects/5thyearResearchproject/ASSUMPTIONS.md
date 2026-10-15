# Assumptions log

Every modelling assumption, numbered so code and settings files can point to it (`[A8]`).
Status: **data** = fitted from race data · **paper** = taken from a cited paper ·
**rules** = from the FIA regulations (verify against the exact edition) ·
**heuristic** = a reasoned choice · **placeholder** = a guess that must be replaced or
sourced before results depend on it.

## Weeks 1–2: tyre model and data

| # | Assumption | Status | Where |
|---|---|---|---|
| A1 | Tyre age counts from 0 on a tyre's first lap. Paper 1 writes the formula from 1, but only the 0 version reproduces its Table 8 / Section 4.1 numbers. | paper | `1st_week.py`, all later code |
| A2 | Tyre loss is linear in age, `k0 + k1·age`, with no "cliff" at high age. | paper (Paper 1) | everywhere |
| A3 | Week 1 base lap time (93.59 s) and pit loss (22.5 s) are inferred from Paper 1's published race totals; the base is an "effective" value absorbing shared terms such as fuel. | paper (derived) | `1st_week.py` |
| A4 | Tyre parameters are fitted on free-air laps only (≥ 2 s to the car ahead, from OpenF1 intervals). Free-air laps are not a random sample: many come from the leaders or just after a stop, on fresh tyres. All-laps fit kept as a sensitivity check. | data | `2nd_week.py` |
| A5 | Clean laps: no lap 1, no in/out laps, green flag only, slower than 103% of the driver's median dropped. | heuristic | `2nd_week.py` |
| A6 | A compound with fewer than 30 free-air laps is not fitted. | heuristic | `2nd_week.py` |
| A7 | Pit loss = in-lap + out-lap − 2 × the driver's median clean lap, median over green-flag stops; one constant per race. | data | `2nd_week.py` |

## Weeks 3–4: ETH-style simulator (`racesim.py`, `eras/*.toml`)

| # | Assumption | Status | Where |
|---|---|---|---|
| A8 | Each kg of car mass costs 0.03 s per lap. Rule of thumb; **find a citable source** before relying on it. All of the fitted lap trend except A9 is attributed to it. | placeholder | `[car] mass_time_s_per_kg` |
| A9 | Track evolution is linear in lap number: what is left of the lap trend after A8 (−0.0069 ± 0.0014 s/lap in 2024 Bahrain). | data | `[track] evolution_s_per_lap` |
| A10 | Empty car mass = FIA minimum mass including driver (798 kg, 2024 Technical Regulations Art. 4.1); cars assumed to run at the minimum. | rules | `[car] mass_empty_kg` |
| A11 | Cars start with the 110 kg maximum and burn it at a constant rate in the step-1 refit. Real start loads are lower and not public. | rules / heuristic | `[car] fuel_start_kg`, `3rd_4th_week.py` step 1 |
| A12 | Fuel lower heating value 43 MJ/kg. Approximate; **find a source**. Only converts between fuel energy and fuel mass. | placeholder | `[car] fuel_lhv_mj_per_kg` |
| A13 | Wear per lap = 1 + β(m/m_ref − 1) with β = 1 (wear proportional to load) and m_ref = mid-race mass, so the race-average wear rate equals the fitted one. ETH Eq. 26 with a = 1; ETH also chose its mass coefficient heuristically. | heuristic (paper form) | `[tyres] wear_mass_exponent` |
| A14 | Medium tyre (not run in 2024 Bahrain) = midpoint of soft and hard k0 and k1. ETH also derives the medium heuristically from soft and hard. | heuristic | `[tyres.MEDIUM]` |
| A15 | If no compound change has been made with 20 laps left, one is forced (ETH Sec. 4.7). The reference optimiser obeys the same rule. | paper | `[race] force_change_laps_left` |
| A16 | Fuel allocation per lap within 0.9–1.1 × nominal (ETH Eq. 2). Lap-time effect has diminishing returns, −2·g·s·(√(1 + x/s) − 1) for x = fraction above nominal, g = 4 s, s = 0.2 (saving 10% costs 0.47 s, using 10% more gains 0.36 s; Weeks 3–4 used the quadratic −4x + 10x² with about the same values). **Values are guesses.** | paper (bounds) / placeholder (effect) | `[energy.fuel]` |
| A17 | Usable battery window 4 MJ (Art. 5.3.2 diagram: max − min state of charge ≤ 4 MJ); battery starts full (ETH Eq. 8) and may finish empty. | rules / paper | `[energy.battery] capacity_mj` |
| A18 | One "net" battery decision per lap within −2 to +4 MJ: the FIA deploy cap (ES → MGU-K ≤ 4 MJ/lap) and harvest cap (MGU-K → ES ≤ 2 MJ/lap), Art. 5.3.2 diagram. Each is also limited by 120 kW (Art. 5.3.3) held for a whole lap, which never binds (4 MJ needs 33 s at full power). Weeks 3–4 used ±2 MJ. | rules / heuristic | `[energy.battery] net_*_mj` |
| A19 | Battery lap-time effect has diminishing returns, −2·g·s·(√(1 + u/s) − 1), g = 0.25 s/MJ, s = 4 MJ (u in MJ; full 4 MJ deploy −0.83 s, full 2 MJ harvest +0.59 s). Weeks 3–4 used −0.25u + 0.02u². **Values are guesses.** Because it doesn't depend on the lap or the charge level, the optimal use is to spread the starting charge evenly (flat line in `week3_step3_energy.png`), so the battery decision is trivial in this model. **Must be enriched before the 2026 comparison**, since the hypothesis is that energy management is what changes. | placeholder | `[energy.battery]` |
| A20 | The "reference" (near-perfect) strategy alternates between the exact best pit plan for a fixed energy allocation and the best energy allocation (SLSQP) for a fixed plan, until the plan stops changing. Not guaranteed to be the global optimum (ETH solve the joint MINLP instead). | heuristic | `RaceSim.reference` |
| A21 | One average car: mean driver offset, no traffic, overtaking, safety cars or randomness. | heuristic | `racesim.py` |
| A22 | Every set fitted at a stop is new, and the number of sets per compound is unlimited. | heuristic | `racesim.py` |
| A23 | ~~The whole pit loss is charged to the in-lap.~~ Replaced in Weeks 5–6 by separate in-lap and out-lap losses (A28). | superseded | |
| A24 | Every car starts on 3-lap-old softs (2024 top-10 median). | data | `[race] start_compound`, `start_tyre_age` |
| A25 | Environment action: a 3-number Box; the pit decision is the third number binned into 4 options (ETH uses a separate discrete head). Infeasible energy is clipped to the feasible range (ETH Sec. 4.5). Reward = base lap − lap time (ETH Eq. 67). | heuristic (paper form) | `F1StrategyEnv` |

## Weeks 5–6: old-era energy rules, in/out laps, replay check (`5th_6th_week.py`)

FIA source: 2024 Formula 1 Technical Regulations, Issue 4, 13 Feb 2024 (in force at the 2024 Bahrain GP). All numbers are in the `[rules]` section of the settings file.

| # | Assumption | Status | Where |
|---|---|---|---|
| A26 | Race fuel ≤ 110 kg. This is 2021 Sporting Regulations Art. 30.5; the limit does **not** appear in the text of the 2024 Technical or Sporting Regulations (Issue 4), so for 2022–25 it is kept as the practical maximum load. Fuel flow ≤ 100 kg/h (Art. 5.2.3) is checked and never binds (2.12 kg/lap max allocation vs 2.62 kg/lap at full flow all lap). | rules (older edition) | `[rules] race_fuel_max_kg` |
| A27 | MGU-H recharging the battery (unlimited in the rules) is ignored: all battery charging goes through the MGU-K's 2 MJ/lap cap. MGU-H energy is treated as part of the base lap. | heuristic | `[rules] mguk_to_es_max_mj_per_lap` |
| A28 | Pit loss is split into an in-lap loss (2.47 s) and an out-lap loss (21.83 s): each lap's time minus the simulator's flying-lap time for that driver's tyres, compared with the median of the same quantity over up to 3 clean laps before (in-lap) or after (out-lap) the stop. Medians over 37 green-flag 2024 Bahrain stops. **24.3 s per stop, just below the 25–30 s expected range** (ETH: 26.6 s). Bahrain's pit lane is short; Week 2's cruder method gave 24.9 s. Kept as measured, not adjusted. | data | `[pit]` |
| A29 | Lap 1 is 5.37 s slower than a flying lap (standing start, measured as A28 against laps 2–4, median over 20 drivers). The same for every strategy, so it changes totals, not rankings. | data | `[race] start_loss_s` |
| A30 | Replay check: each driver's pace = median of (clean lap − simulated flying lap) over their own clean laps, applied as a base-lap shift; nominal energy every lap; used sets fitted at stops keep their real age. The pace is measured on the same race, so the check tests whether tyres, fuel, stops and the start add up correctly over a race, not whether pace is predicted. | heuristic | `5th_6th_week.py` step 4 |
