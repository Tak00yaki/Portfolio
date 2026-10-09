"""Week 1: simplest race model (Paper 1) and brute-force strategy optimiser.

Lap time = base time + tyre loss, with tyre loss = k0 + k1 * tyre_age.
Expected results (Paper 1, Section 4.1):
    1 stop: Soft 22 -> Medium 33                     5237.65 s
    2 stop: Soft 16 -> Soft 18 -> Medium 21          5240.95 s
    3 stop: Soft 12 -> Soft 14 -> Soft 14 -> Medium 15  5252.23 s
"""

from itertools import combinations, product

# --- Paper 1, Table 8 -------------------------------------------------------
RACE_LAPS = 55
MIN_STOPS, MAX_STOPS = 1, 3
COMPOUNDS = {          # compound: (k0 [s], k1 [s/lap])
    "soft":   (0.0, 0.09),
    "medium": (0.5, 0.05),
    "hard":   (1.2, 0.016),
}
START_COMPOUND = "soft"   # Table 8: start compound
START_AGE = 2             # Table 8: tyre age at race start (used qualifying tyres)

# --- Not in Table 8: derived from the paper's published race totals ---------
# The 2-stop (5240.95 s) and 3-stop (5252.23 s) totals both imply a 22.5 s pit
# loss. BASE_LAP is an "effective" base that also absorbs strategy-independent
# lap time (e.g. fuel), fixed by 55 * base + one pit loss = 5170 s.
PIT_LOSS = 22.5
BASE_LAP = (5170.0 - PIT_LOSS) / RACE_LAPS   # ~93.59 s

# Tyre age is counted from 0 on a tyre's first lap. The paper writes the formula
# as if ages start at 1, but age 0 is the convention that reproduces Table 8
# and Section 4.1.


def stint_time(compound, laps, start_age=0):
    """Total time for one stint: base + linear tyre loss on every lap.

    Closed form of sum(BASE_LAP + k0 + k1 * (start_age + i) for i in range(laps)).
    """
    k0, k1 = COMPOUNDS[compound]
    return laps * (BASE_LAP + k0) + k1 * (laps * start_age + laps * (laps - 1) / 2)


def race_time(compounds, pit_laps):
    """Race time for a compound sequence, pitting at the end of each lap in `pit_laps`."""
    bounds = (0, *pit_laps, RACE_LAPS)
    total = PIT_LOSS * len(pit_laps)
    for i, compound in enumerate(compounds):
        start_age = START_AGE if i == 0 else 0
        total += stint_time(compound, bounds[i + 1] - bounds[i], start_age)
    return total


def best_strategy(n_stops):
    """Try every compound sequence and every set of pit laps for `n_stops` stops.

    Sequences must start on START_COMPOUND and use at least two different compounds.
    """
    best = None
    for rest in product(COMPOUNDS, repeat=n_stops):
        compounds = (START_COMPOUND, *rest)
        if len(set(compounds)) < 2:
            continue
        for pit_laps in combinations(range(1, RACE_LAPS), n_stops):
            t = race_time(compounds, pit_laps)
            if best is None or t < best[0]:
                best = (t, compounds, pit_laps)
    return best


def describe(compounds, pit_laps):
    bounds = (0, *pit_laps, RACE_LAPS)
    return " -> ".join(f"{c.capitalize()} {bounds[i + 1] - bounds[i]}"
                       for i, c in enumerate(compounds))


if __name__ == "__main__":
    # Paper 1 plans as (compounds, pit laps). Fresh-tyre stints can be swapped
    # without changing the total, so the search may return an equally fast
    # reordering; the check is that the paper's plan is optimal and its total matches.
    S, M = "soft", "medium"
    expected = {
        1: ((S, M), (22,), 5237.65),
        2: ((S, S, M), (16, 34), 5240.95),
        3: ((S, S, S, M), (12, 26, 40), 5252.23),
    }
    for n_stops in range(MIN_STOPS, MAX_STOPS + 1):
        total, compounds, pit_laps = best_strategy(n_stops)
        print(f"Best {n_stops}-stop: {describe(compounds, pit_laps)}  ->  {total:.2f} s")

        exp_compounds, exp_pits, exp_total = expected[n_stops]
        paper_total = race_time(exp_compounds, exp_pits)
        assert abs(paper_total - total) < 1e-6, \
            f"paper plan {describe(exp_compounds, exp_pits)} is not optimal ({paper_total:.2f} s)"
        assert abs(total - exp_total) < 1e-6, f"expected {exp_total}, got {total:.2f}"
    print("All checks passed (1, 2 and 3 stops match Paper 1)")
