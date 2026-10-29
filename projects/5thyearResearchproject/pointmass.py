"""Point-mass straight-line simulator: what the energy decisions are worth in lap time.

A car of mass m with power P at the wheels, aerodynamic drag and rolling resistance,
driven along a lap's full-throttle straights (from FastF1/OpenF1 speed traces):

    m dv/dt = P / v - 0.5 rho CdA v^2 - Crr m g                    accelerating
    dv/dt   = -(brake_a + brake_b v^2)                             braking (downforce grows with v^2)

Each straight starts where the driver reaches full throttle (measured speed) and ends at
the slowest point of the next corner. Braking follows the lap's measured braking profile
(so trail-braking into the corner is as measured), extended above the measured braking
speed with the straight-line deceleration fitted to the data: a faster car brakes a little
earlier. Straights that end in a lift instead of braking end at the measured lift point.
A power change and a constant force per straight (`power_adj`, `offset`, fitted by
`calibrate`) absorb what a flat point mass leaves out: the real car's own deploy pattern
(more power early in a straight, less at the end), gradient and wind. With them the
unchanged car reproduces the measured time on the throttle and the braking-point speed.
Extra power on part of a straight then changes the time over it:

    deploy   +MGU-K power on a stretch     -> time saved per MJ from the battery
    clip     -MGU-K power on a stretch     -> time lost per MJ harvested at full throttle
    coast    no power on a stretch         -> time lost per MJ of fuel saved (lift and coast)

Each is applied where it is worth most first (deploy) or costs least first (clip, coast),
which gives a curve of time against energy with diminishing returns that comes from the
physics, not a guess. The MGU-K power is a parameter, so the same code gives the 2026 curve.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

G = 9.81
DS = 2.0            # integration step along the straight (m)
BIN_M = 25.0        # length of the stretches energy is assigned to (m)


@dataclass(frozen=True)
class Car:
    mass: float     # kg
    power: float    # W at the wheels, full throttle
    cda: float      # m^2
    crr: float
    rho: float      # kg/m^3
    brake_a: float  # m/s^2, braking deceleration = brake_a + brake_b * v^2
    brake_b: float  # 1/m


@dataclass(frozen=True)
class Straight:
    s0: float               # m from the lap start, where full throttle starts
    v0: float               # m/s there
    s1: float               # m, slowest point of the next corner (or the lift point)
    v1: float               # m/s there
    brake_s: tuple = ()     # measured braking profile, braking point to s1 (empty: ends in a lift)
    brake_v: tuple = ()
    offset: float = 0.0     # N, extra resistance on this straight (see calibrate)
    power_adj: float = 0.0  # W, power change on this straight (see calibrate)

    @property
    def brakes(self):
        return bool(self.brake_s)


def fit_power_drag(v, a, mass, crr, rho):
    """Least-squares wheel power and CdA from full-throttle samples:
    m a + Crr m g = P / v - 0.5 rho v^2 CdA. Returns (power W, CdA, residual sd in N)."""
    A = np.column_stack([1 / v, -0.5 * rho * v ** 2])
    y = mass * a + crr * mass * G
    (power, cda), *_ = np.linalg.lstsq(A, y, rcond=None)
    return power, cda, float(np.std(y - A @ np.array([power, cda])))


def fit_braking(v, a):
    """Least-squares braking deceleration -a = brake_a + brake_b v^2."""
    A = np.column_stack([np.ones_like(v), v ** 2])
    (ba, bb), *_ = np.linalg.lstsq(A, -a, rcond=None)
    return ba, bb


def straights_from_trace(s, v, throttle, brake, min_len_s=2.0, t=None):
    """Full-throttle straights in one lap's trace (distance s, speed v in m/s).

    A straight is a run of throttle >= 99% lasting min_len_s or longer. If braking
    follows within 1 s it ends at the slowest point before the throttle is reapplied;
    otherwise it ends where the throttle came off."""
    full = throttle >= 99
    edges = np.flatnonzero(np.diff(np.r_[0, full.astype(int), 0]))
    out = []
    for i0, i1 in zip(edges[::2], edges[1::2]):            # samples i0 .. i1-1 at full throttle
        if t is not None and t[i1 - 1] - t[i0] < min_len_s:
            continue
        after = np.arange(i1, len(s))
        if not len(after):
            out.append(Straight(s[i0], v[i0], s[-1], v[-1]))
            continue
        brakes = after[:5][brake[after[:5]] > 0]             # braking within ~1 s (0.24 s samples)
        if len(brakes):
            back_on = after[(throttle[after] > 20) & (after > brakes[0])]
            j_end = back_on[0] if len(back_on) else len(s) - 1
            seg = np.arange(brakes[0], j_end + 1)
            j = seg[np.argmin(v[seg])]
            prof = np.arange(i1, j + 1)                  # throttle off (lift) to the corner
            out.append(Straight(s[i0], v[i0], s[j], v[j], tuple(s[prof]), tuple(v[prof])))
        else:
            out.append(Straight(s[i0], v[i0], s[i1], v[i1]))
    return out


def _braking_limit(car, st, s):
    """Fastest speed at each point from which the car can still make the corner."""
    vb = np.full(len(s), np.inf)
    if not st.brakes:
        return vb
    bs, bv = np.asarray(st.brake_s), np.asarray(st.brake_v)
    on = s >= bs[0]
    vb[on] = np.interp(s[on], bs, bv)
    before = np.flatnonzero(~on)[::-1]                   # back from the measured braking point
    v_next, s_next = bv[0], bs[0]
    for i in before:
        ds = s_next - s[i]
        v_next = np.sqrt(v_next ** 2 + 2 * (car.brake_a + car.brake_b * v_next ** 2) * ds)
        vb[i], s_next = v_next, s[i]
    return vb


def run(car, st, extra=None, end=None):
    """One straight. `extra(s_grid) -> W` is added to the car's power while it is on the
    throttle (None = none); `end` stops early, with no braking (used by calibrate).
    Returns (time s, extra energy J, grid, speed, time on the throttle per step)."""
    s1 = st.s1 if end is None else end
    n = max(int(np.ceil((s1 - st.s0) / DS)), 2)
    s = np.linspace(st.s0, s1, n + 1)
    ds = s[1] - s[0]
    dP = np.zeros(n) if extra is None else extra(s[:-1])
    vb = _braking_limit(car, st, s) if end is None else np.full(n + 1, np.inf)
    v = np.empty(n + 1)
    v[0] = min(st.v0, vb[0])
    acc = np.zeros(n, bool)
    resist = car.crr * car.mass * G + st.offset
    for i in range(n):
        p = max(car.power + st.power_adj + dP[i], 0.0)
        a = (p / v[i] - 0.5 * car.rho * car.cda * v[i] ** 2 - resist) / car.mass
        va = np.sqrt(max(v[i] ** 2 + 2 * a * ds, 1.0))
        acc[i] = va <= vb[i + 1]
        v[i + 1] = min(va, vb[i + 1])
    dt = 2 * ds / (v[:-1] + v[1:])
    return float(dt.sum()), float(np.sum(dP * dt * acc)), s, v, dt * acc


def calibrate(car, st, t_on=None):
    """Copy of the straight with the constant force that makes the car reach the measured
    speed where the driver lifted for braking (or the end of the throttle). With `t_on`
    (measured time from full throttle to that point), also the power change that makes
    it take that long."""
    s_end, v_end = (st.brake_s[0], st.brake_v[0]) if st.brakes else (st.s1, st.v1)

    def with_offset(adj):
        def miss(f):
            return run(car, replace(st, offset=f, power_adj=adj), end=s_end)[3][-1] - v_end
        lo, hi = -40e3, 40e3
        if miss(lo) * miss(hi) > 0:                 # cannot reach it: closest end
            return replace(st, offset=lo if abs(miss(lo)) < abs(miss(hi)) else hi, power_adj=adj)
        return replace(st, offset=brentq(miss, lo, hi, xtol=0.5), power_adj=adj)

    if t_on is None:
        return with_offset(0.0)

    def slow(adj):
        return run(car, with_offset(adj), end=s_end)[0] - t_on
    lo, hi = -0.6 * car.power, 0.6 * car.power
    if slow(lo) * slow(hi) > 0:
        return with_offset(lo if abs(slow(lo)) < abs(slow(hi)) else hi)
    return with_offset(brentq(slow, lo, hi, xtol=100.0))


def _bins(st):
    edges = np.arange(st.s0, st.s1, BIN_M)
    return [(lo, min(lo + BIN_M, st.s1)) for lo in edges]


def _power_on(bins, power):
    def extra(s):
        out = np.zeros_like(s)
        for lo, hi in bins:
            out[(s >= lo) & (s < hi)] = power
        return out
    return extra


def energy_curve(car, straights, power, energy_per_j=1.0, with_order=False):
    """Lap-time change against energy for one kind of decision, applied 25 m at a time to
    the stretches worth most (deploy) or costing least (clip, coast) first.

    power > 0: deploy; power < 0: clip; power = -car.power: coast. Energy = |power change|
    x time on power, times energy_per_j (1 for the battery; fuel J per wheel J not produced
    for coasting). Returns (energy MJ, lap-time change s), starting at (0, 0), and with
    `with_order` the stretches in the order they were added, as (straight, stretch)."""
    base = [run(car, st)[0] for st in straights]
    cand = []                                       # (time change per J, straight, stretch)
    for k, st in enumerate(straights):
        for b in _bins(st):
            t, e, *_ = run(car, st, _power_on([b], power))
            if abs(e) > 1.0:
                cand.append(((t - base[k]) / abs(e), k, b))
    cand.sort(key=lambda c: c[0])                   # most time saved, or least lost, per J first
    chosen = [[] for _ in straights]
    times, energies = list(base), [0.0] * len(straights)
    E, T = [0.0], [0.0]
    for _, k, b in cand:
        chosen[k].append(b)
        times[k], e, *_ = run(car, straights[k], _power_on(chosen[k], power))
        energies[k] = abs(e)
        E.append(sum(energies) * energy_per_j / 1e6)
        T.append(sum(times) - sum(base))
    if with_order:
        return np.array(E), np.array(T), [(k, b) for _, k, b in cand]
    return np.array(E), np.array(T)


def run_mixed(car, st, stretches):
    """One straight with a different power change on each stretch: {(lo, hi): W}."""
    def extra(s):
        out = np.zeros_like(s)
        for (lo, hi), power in stretches.items():
            out[(s >= lo) & (s < hi)] = power
        return out
    return run(car, st, extra)
