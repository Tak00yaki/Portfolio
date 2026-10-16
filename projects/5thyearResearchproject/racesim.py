"""Lap-by-lap race simulator in the style of the ETH model, plus its Gymnasium environment.

Model (Fieni & Onder, "Towards Learning-Based Formula 1 Race Strategies",
arXiv 2512.21570), simplified to one car and calibrated on real lap data:

    lap time = base + track evolution * lap
             + mass_time * (car mass mid-lap - empty mass)     fuel: lighter car, faster lap
             + k0[compound] + k1[compound] * wear               tyre loss (Weeks 1-2)
             + fuel term + battery term                         energy decisions, diminishing
                                                                returns (square-root shaped)
             + in-lap loss (lap of a stop) + out-lap loss (lap after) + standing start (lap 1)

    car mass  -= fuel energy used / LHV                          (ETH Eq. 6)
    wear      += 1 + beta * (mass mid-lap / reference mass - 1)  (ETH Eq. 26, a = 1)
    battery   -= net battery energy used, 0 <= battery <= capacity (ETH Eqs. 4, 9)

Energy limits per lap (battery deploy and harvest, MGU-K power, fuel flow, race fuel)
come from the era's FIA regulations, cited in the settings file's [rules] section.

Every era-specific number comes from a settings file in eras/; nothing is hard-coded
here, so the 2026 simulator is this code with a different file.
"""

import copy
import tomllib
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations, product

import casadi as ca
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.optimize import minimize

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
PIT_CHOICES = (None, *COMPOUNDS)    # pit decision 0-3: no stop / soft / medium / hard


def load_settings(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def diminishing(u, slope, scale):
    """Lap-time change from using energy u above nominal: -slope * u for small u, with
    diminishing returns (square-root shaped). Using more always saves time, but each
    extra unit saves less; saving energy costs more than the same amount used gains.
    Defined for u > -scale."""
    return -2 * slope * scale * (np.sqrt(1 + np.asarray(u) / scale) - 1)


def override(settings, **changes):
    """Copy of `settings` with dotted keys replaced, e.g. override(s, **{"tyres.wear_mass_exponent": 0})."""
    s = copy.deepcopy(settings)
    for key, value in changes.items():
        *path, last = key.split(".")
        d = s
        for p in path:
            d = d[p]
        d[last] = value
    return s


@dataclass(frozen=True)
class RaceState:
    lap: int            # laps completed
    fuel_mj: float      # fuel energy left
    batt_mj: float      # battery energy left
    mass_kg: float      # car mass (empty car + fuel left)
    compound: str
    wear: float         # tyre wear, in laps at reference mass
    changes: int        # pit stops onto a different compound (ETH b_comp)
    race_time: float
    outlap: bool        # previous lap was an in-lap, so the next lap is an out-lap


class RaceSim:
    def __init__(self, settings):
        self.settings = settings
        r, c, t, e = settings["race"], settings["car"], settings["tyres"], settings["energy"]
        rules, pit = settings["rules"], settings["pit"]
        self.laps = r["laps"]
        self.base, self.evolution = settings["track"]["base_lap_s"], settings["track"]["evolution_s_per_lap"]
        self.start_loss = r["start_loss_s"]
        self.m_empty, self.lhv, self.mass_time = c["mass_empty_kg"], c["fuel_lhv_mj_per_kg"], c["mass_time_s_per_kg"]
        if c["fuel_start_kg"] > rules["race_fuel_max_kg"] or self.m_empty < rules["min_mass_kg"]:
            raise ValueError("start fuel above the race maximum, or car below the minimum mass")
        self.fuel0 = c["fuel_start_kg"] * self.lhv
        self.fuel_nominal = self.fuel0 / self.laps
        self.inlap_loss, self.outlap_loss = pit["inlap_loss_s"], pit["outlap_loss_s"]
        self.pit_loss = self.inlap_loss + self.outlap_loss     # per stop
        self.tyre = {k: (t[k]["k0"], t[k]["k1"]) for k in COMPOUNDS if k in t}
        self.beta = t["wear_mass_exponent"]
        self.m_ref = self.m_empty + c["fuel_start_kg"] / 2      # "mid_race"
        f, b = e["fuel"], e["battery"]
        # Fuel per lap: ETH's allocation band, and never above the fuel-flow limit held
        # for a whole lap (Art. 5.2.3); the band binds first at every track we model.
        self.fuel_flow_cap = rules["fuel_flow_max_kg_per_h"] * self.base / 3600 * self.lhv
        self.f_min = f["alloc_min_frac"] * self.fuel_nominal
        self.f_max = min(f["alloc_max_frac"] * self.fuel_nominal, self.fuel_flow_cap)
        self.f_gain, self.f_scale = f["gain_s"], f["scale_frac"]
        # Battery per lap (net, + = discharge): deploy and harvest caps, each also
        # limited by the MGU-K power held for a whole lap (Art. 5.3.2 diagram)
        k_mj = rules["mguk_max_kw"] / 1000 * self.base
        self.b_cap = rules["es_soc_window_mj"]
        self.b_max = min(rules["es_to_mguk_max_mj_per_lap"], k_mj)
        self.b_min = -min(rules["mguk_to_es_max_mj_per_lap"], k_mj)
        self.b_gain, self.b_scale = b["gain_s_per_mj"], b["scale_mj"]
        if self.b_scale <= -self.b_min or self.f_scale <= 1 - f["alloc_min_frac"]:
            raise ValueError("energy-term scale must exceed the largest saving allowed")
        self.start_compound, self.start_age = r["start_compound"], r["start_tyre_age"]
        self.force_laps_left = r["force_change_laps_left"]
        self.min_stops, self.max_stops = r["min_stops"], r["max_stops"]

    # --- one lap ---------------------------------------------------------------

    def initial_state(self, compound=None, age=None):
        compound = compound or self.start_compound
        age = self.start_age if age is None else age
        return RaceState(0, self.fuel0, self.b_cap, self.mass(self.fuel0), compound,
                         float(age), 0, 0.0, False)

    def mass(self, fuel_mj):
        return self.m_empty + fuel_mj / self.lhv

    def fuel_term(self, u_f):
        """x = fraction of fuel used above nominal this lap."""
        return diminishing((u_f - self.fuel_nominal) / self.fuel_nominal, self.f_gain, self.f_scale)

    def batt_term(self, u_b):
        return diminishing(u_b, self.b_gain, self.b_scale)

    def feasible(self, state, u_f, u_b):
        """Clip energy use to what keeps the rest of the race feasible (ETH Sec. 4.5):
        enough fuel left for the minimum allocation on every remaining lap, and the
        battery inside [0, capacity]."""
        left_after = self.laps - state.lap - 1
        f_hi = min(self.f_max, state.fuel_mj - self.f_min * left_after)
        f_lo = min(self.f_min, f_hi)
        b_lo = max(self.b_min, -(self.b_cap - state.batt_mj))
        b_hi = min(self.b_max, state.batt_mj)
        return float(np.clip(u_f, f_lo, f_hi)), float(np.clip(u_b, b_lo, b_hi))

    def step(self, state, u_f, u_b, pit_to=None, fit_age=0.0):
        """Run one lap. Inputs: fuel energy u_f and net battery energy u_b (MJ) this lap,
        and the pit decision: 0-3 (PIT_CHOICES) or the compound fitted at the end of the
        lap (None = no stop). `fit_age`: laps already on the set fitted (0 = new).
        Returns (new state, lap time, components)."""
        if isinstance(pit_to, (int, np.integer)):
            pit_to = PIT_CHOICES[pit_to]
        u_f, u_b = self.feasible(state, u_f, u_b)
        k = state.lap + 1
        if k == self.laps:
            pit_to = None                       # a stop after the flag does nothing
        m_mid = self.mass(state.fuel_mj - u_f / 2)
        k0, k1 = self.tyre[state.compound]
        parts = {
            "base": self.base,
            "track": self.evolution * k,
            "fuel mass": self.mass_time * (m_mid - self.m_empty),
            "tyres": k0 + k1 * state.wear,
            "fuel use": self.fuel_term(u_f),
            "battery": self.batt_term(u_b),
            "start": self.start_loss if k == 1 else 0.0,
            "in-lap": self.inlap_loss if pit_to else 0.0,
            "out-lap": self.outlap_loss if state.outlap else 0.0,
        }
        lap_time = sum(parts.values())
        wear = state.wear + 1 + self.beta * (m_mid / self.m_ref - 1)
        new = replace(
            state, lap=k, fuel_mj=state.fuel_mj - u_f, batt_mj=state.batt_mj - u_b,
            mass_kg=self.mass(state.fuel_mj - u_f),
            race_time=state.race_time + lap_time, outlap=bool(pit_to),
            compound=pit_to or state.compound, wear=float(fit_age) if pit_to else wear,
            changes=state.changes + int(bool(pit_to) and pit_to != state.compound))
        return new, lap_time, {**parts, "u_f": u_f, "u_b": u_b, "mass": m_mid}

    # --- whole race --------------------------------------------------------------

    def nominal_alloc(self):
        return np.full(self.laps, self.fuel_nominal), np.zeros(self.laps)

    def simulate(self, compounds, pit_laps, fuel=None, batt=None, ages=None):
        """Lap-by-lap run of a plan (Week 2 format). `ages`: tyre age at the start of each
        stint, for replaying a real race with used sets (default: settings start age, new
        sets at stops). Returns total time and per-lap records."""
        fuel, batt = (fuel, batt) if fuel is not None else self.nominal_alloc()
        stops = dict(zip(pit_laps, compounds[1:]))
        fit_ages = dict(zip(pit_laps, ages[1:])) if ages is not None else {}
        if ages is None:
            state = self.initial_state()
            assert compounds[0] == state.compound, "plan must start on the start compound"
        else:
            state = self.initial_state(compounds[0], ages[0])
        rows = []
        for i in range(self.laps):
            wear, compound = state.wear, state.compound
            state, t, parts = self.step(state, fuel[i], batt[i], stops.get(i + 1),
                                        fit_ages.get(i + 1, 0.0))
            rows.append({"lap": i + 1, "time": t, "compound": compound, "wear": wear,
                         "fuel_mj": state.fuel_mj, "batt_mj": state.batt_mj, **parts})
        return state.race_time, rows

    def race_time(self, compounds, pit_laps, fuel, batt):
        """Vectorised total race time (same model as simulate, no clipping) for the optimiser."""
        k = np.arange(1, self.laps + 1)
        fuel_before = self.fuel0 - np.concatenate([[0.0], np.cumsum(fuel)[:-1]])
        m_mid = self.mass(fuel_before - fuel / 2)
        w = 1 + self.beta * (m_mid / self.m_ref - 1)
        total = np.sum(self.base + self.evolution * k + self.mass_time * (m_mid - self.m_empty)
                       + self.fuel_term(fuel) + self.batt_term(batt))
        total += self.start_loss + self.pit_loss * len(pit_laps)   # every stop has its out-lap
        bounds = (0, *pit_laps, self.laps)
        for i, c in enumerate(compounds):
            k0, k1 = self.tyre[c]
            ws = w[bounds[i]:bounds[i + 1]]
            age0 = self.start_age if i == 0 else 0.0
            wear = age0 + np.concatenate([[0.0], np.cumsum(ws)[:-1]])
            total += np.sum(k0 + k1 * wear)
        return float(total)

    # --- reference optimiser ("mathematically perfect" target for the agent) ----

    def legal(self, compounds, pit_laps):
        """>= 2 compounds, and the first change no later than the forced-change lap."""
        changes = [p for p, (a, b) in zip(pit_laps, zip(compounds, compounds[1:])) if a != b]
        return bool(changes) and changes[0] <= self.laps - self.force_laps_left

    def best_plan(self, fuel=None, batt=None, start_compound=None):
        """Exact best pit plan for a fixed energy allocation.

        With the allocation fixed, car mass and hence the wear added per lap are known
        in advance, so each stint's tyre loss follows from prefix sums and every plan
        can be scored at once. Returns (time, compounds, pit laps).
        """
        fuel = self.nominal_alloc()[0] if fuel is None else fuel
        batt = np.zeros(self.laps) if batt is None else batt
        start = start_compound or self.start_compound
        fuel_before = self.fuel0 - np.concatenate([[0.0], np.cumsum(fuel)[:-1]])
        w = 1 + self.beta * (self.mass(fuel_before - fuel / 2) / self.m_ref - 1)
        C = np.concatenate([[0.0], np.cumsum(w)])       # C[k] = wear added on laps 1..k
        P = np.concatenate([[0.0], np.cumsum(C)])       # P[i] = C[0] + ... + C[i-1]
        best = None
        for n in range(self.min_stops, self.max_stops + 1):
            sets = _pit_sets(self.laps, n)
            a = np.column_stack([np.ones(len(sets), int), sets + 1])          # stint first laps
            b = np.column_stack([sets, np.full(len(sets), self.laps)])        # stint last laps
            L = b - a + 1
            for compounds in product(self.tyre, repeat=n + 1):
                if compounds[0] != start or len(set(compounds)) < 2:
                    continue
                age0 = np.zeros(n + 1)
                age0[0] = self.start_age
                k0 = np.array([self.tyre[c][0] for c in compounds])
                k1 = np.array([self.tyre[c][1] for c in compounds])
                # sum of wear over a stint = age0*L + sum_{k=a..b} (C[k-1] - C[a-1])
                wear_sum = age0 * L + (P[b] - P[a - 1]) - L * C[a - 1]
                total = (k0 * L + k1 * wear_sum).sum(axis=1) + self.pit_loss * n
                first = _first_change(compounds, sets)
                total[first > self.laps - self.force_laps_left] = np.inf
                i = int(np.argmin(total))
                if np.isfinite(total[i]) and (best is None or total[i] < best[0]):
                    best = (total[i], compounds, tuple(int(x) for x in sets[i]))
        _, compounds, pits = best
        return self.race_time(compounds, pits, fuel, batt), compounds, pits

    def best_energy(self, compounds, pit_laps):
        """Best fuel and battery allocation per lap for a fixed pit plan (SLSQP).

        Constraints: per-lap bounds, total fuel <= start fuel, battery in [0, capacity]
        after every lap. Returns (time, fuel per lap, battery per lap).
        """
        n = self.laps
        x0 = np.concatenate(self.nominal_alloc())
        lower_tri = np.tril(np.ones((n, n)))
        cons = [
            {"type": "ineq", "fun": lambda x: self.fuel0 - x[:n].sum()},
            {"type": "ineq", "fun": lambda x: self.b_cap - lower_tri @ x[n:]},           # soc >= 0
            {"type": "ineq", "fun": lambda x: lower_tri @ x[n:]},                         # soc <= cap
        ]
        bounds = [(self.f_min, self.f_max)] * n + [(self.b_min, self.b_max)] * n
        res = minimize(lambda x: self.race_time(compounds, pit_laps, x[:n], x[n:]), x0,
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if not res.success:
            raise RuntimeError(f"energy optimisation failed: {res.message}")
        return res.fun, res.x[:n], res.x[n:]

    def reference(self, max_rounds=10):
        """Near-optimal strategy by alternating: best plan for the energy allocation,
        then best energy allocation for that plan, until the plan stops changing.
        Not guaranteed globally optimal (ASSUMPTIONS A20)."""
        fuel, batt = self.nominal_alloc()
        plan = None
        for _ in range(max_rounds):
            _, compounds, pits = self.best_plan(fuel, batt)
            if plan == (compounds, pits):
                break
            plan = (compounds, pits)
            t, fuel, batt = self.best_energy(*plan)
        return self.race_time(*plan, fuel, batt), plan, fuel, batt


    # --- exact optimiser (Weeks 7-8): every pit plan, exact energy for each --------

    def plans(self, max_stops=2):
        """Every legal pit plan with min_stops..max_stops stops: start on the start
        compound, every compound order, every stop lap."""
        for n in range(self.min_stops, max_stops + 1):
            for rest in product(self.tyre, repeat=n):
                compounds = (self.start_compound, *rest)
                for pits in combinations(range(1, self.laps), n):
                    if self.legal(compounds, pits):
                        yield compounds, pits

    def plan_params(self, compounds, pit_laps):
        """A plan as the energy problem's parameters, per lap: new set fitted before this
        lap (0/1), and the tyre's k0 and k1."""
        reset, k0, k1 = np.zeros(self.laps), np.zeros(self.laps), np.zeros(self.laps)
        reset[list(pit_laps)] = 1
        bounds = (0, *pit_laps, self.laps)
        for i, c in enumerate(compounds):
            k0[bounds[i]:bounds[i + 1]], k1[bounds[i]:bounds[i + 1]] = self.tyre[c]
        return np.concatenate([reset, k0, k1])

    def energy_problem(self):
        """CasADi + IPOPT: best fuel and battery use per lap for a pit plan passed as
        parameters (plan_params). Same objective and limits as race_time / best_energy,
        built once and reused for every plan. Returns (solver, bounds, objective function).

        The objective is linear in the energy except for the fuel and battery terms,
        which are convex, and every limit is linear, so the problem is convex and IPOPT's
        optimum is the global one (ASSUMPTIONS A31)."""
        if getattr(self, "_energy", None) is None:
            n = self.laps
            uf, ub = ca.SX.sym("fuel", n), ca.SX.sym("batt", n)
            p = ca.SX.sym("plan", 3 * n)
            reset, k0, k1 = p[:n], p[n:2 * n], p[2 * n:]

            def dim(u, slope, scale):       # diminishing() for CasADi symbols
                return -2 * slope * scale * (ca.sqrt(1 + u / scale) - 1)

            f, fuel, wear = self.start_loss, self.fuel0, self.start_age
            for i in range(n):
                m_mid = self.mass(fuel - uf[i] / 2)
                wear = (1 - reset[i]) * wear
                f += (self.base + self.evolution * (i + 1) + self.mass_time * (m_mid - self.m_empty)
                      + k0[i] + k1[i] * wear
                      + dim((uf[i] - self.fuel_nominal) / self.fuel_nominal, self.f_gain, self.f_scale)
                      + dim(ub[i], self.b_gain, self.b_scale))
                wear = wear + 1 + self.beta * (m_mid / self.m_ref - 1)
                fuel = fuel - uf[i]
            x = ca.vertcat(uf, ub)
            g = ca.vertcat(ca.sum1(uf), ca.cumsum(ub))      # fuel used; battery used so far
            solver = ca.nlpsol("energy", "ipopt", {"x": x, "p": p, "f": f, "g": g},
                               {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
                                "ipopt.tol": 1e-10, "ipopt.bound_relax_factor": 0.0})  # limits exactly
            bounds = {"lbx": [self.f_min] * n + [self.b_min] * n,
                      "ubx": [self.f_max] * n + [self.b_max] * n,
                      "lbg": [-np.inf] + [0.0] * n,                 # battery never above full
                      "ubg": [self.fuel0] + [self.b_cap] * n}       # nor below empty
            self._energy = solver, bounds, ca.Function("T", [x, p], [f])
        return self._energy

    def optimal_energy(self, compounds, pit_laps, x0=None):
        """Exact best energy use for one plan. Returns (race time, fuel, battery, status)."""
        solver, bounds, _ = self.energy_problem()
        x0 = np.concatenate(self.nominal_alloc()) if x0 is None else x0
        sol = solver(x0=x0, p=self.plan_params(compounds, pit_laps), **bounds)
        x = np.asarray(sol["x"]).ravel()
        status = solver.stats()["return_status"]
        return float(sol["f"]) + self.pit_loss * len(pit_laps), x[:self.laps], x[self.laps:], status

    def exact(self, max_stops=2):
        """The exact optimum: optimal_energy for every plan, best of all. Returns the
        optimum (time, compounds, pit laps, fuel, battery) and every plan's
        (compounds, pit laps, time, status)."""
        table, best = [], None
        for compounds, pits in self.plans(max_stops):
            t, fuel, batt, status = self.optimal_energy(compounds, pits)
            table.append((compounds, pits, t, status))
            if status == "Solve_Succeeded" and (best is None or t < best[0]):
                best = (t, compounds, pits, fuel, batt)
        return best, table

@lru_cache(maxsize=None)
def _pit_sets(laps, n):
    return np.array(list(combinations(range(1, laps), n)), dtype=int)


def _first_change(compounds, sets):
    """Pit lap of the first compound change for every pit-lap set (inf if none)."""
    for j in range(1, len(compounds)):
        if compounds[j] != compounds[j - 1]:
            return sets[:, j - 1]
    return np.full(len(sets), np.inf)


# --- Gymnasium environment ---------------------------------------------------------

class F1StrategyEnv(gym.Env):
    """One step = one lap (ETH Sec. 4).

    Action (Box, 3): [F in 0..1, B in -1..1, P in 0..1]
        F: fuel energy this lap, linear between the min and max allocation
        B: net battery energy this lap, -1 = most charging, +1 = most discharge
        P: pit decision, binned into 4: no stop / soft / medium / hard
    Infeasible energy is clipped to the feasible range, and a compound change is
    forced when none has been made `force_change_laps_left` laps from the end.

    Observation (10, roughly 0..1): battery, fuel, fuel mass, race time, compound
    changed, compound, wear, out-lap flag, last lap time vs base, laps left.

    Reward: base lap time - lap time (ETH Eq. 67), so the return is
    -(race time - laps * base) and maximising it minimises race time.
    """

    metadata = {"render_modes": []}

    def __init__(self, settings):
        self.sim = RaceSim(settings)
        self.pit_options = tuple(c for c in PIT_CHOICES if c is None or c in self.sim.tyre)
        self.action_space = spaces.Box(np.array([0, -1, 0], np.float32),
                                       np.array([1, 1, 1], np.float32))
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(10,), dtype=np.float32)
        self.state, self.last_lap = None, 0.0

    def decode(self, action):
        F, B, P = np.clip(action, self.action_space.low, self.action_space.high)
        s = self.sim
        u_f = s.f_min + F * (s.f_max - s.f_min)
        u_b = s.b_min + (B + 1) / 2 * (s.b_max - s.b_min)
        pit = self.pit_options[min(int(P * len(self.pit_options)), len(self.pit_options) - 1)]
        return u_f, u_b, pit

    def encode(self, u_f, u_b, pit):
        """Inverse of decode: the action that produces (u_f, u_b, pit)."""
        s = self.sim
        F = (u_f - s.f_min) / (s.f_max - s.f_min)
        B = 2 * (u_b - s.b_min) / (s.b_max - s.b_min) - 1
        P = (self.pit_options.index(pit) + 0.5) / len(self.pit_options)
        return np.array([F, B, P], np.float32)

    def _obs(self):
        s, st = self.sim, self.state
        return np.array([
            st.batt_mj / s.b_cap,
            st.fuel_mj / s.fuel0,
            (s.mass(st.fuel_mj) - s.m_empty) / (s.fuel0 / s.lhv),
            st.race_time / (s.laps * s.base),
            float(st.changes > 0),
            COMPOUNDS.index(st.compound) / 2,
            st.wear / s.laps,
            float(st.outlap),
            (self.last_lap - s.base) / s.pit_loss,
            (s.laps - st.lap) / s.laps,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state, self.last_lap = self.sim.initial_state(), self.sim.base
        return self._obs(), {}

    def step(self, action):
        u_f, u_b, pit = self.decode(action)
        s, st = self.sim, self.state
        forced = False
        left_after = s.laps - st.lap - 1
        if st.changes == 0 and left_after <= s.force_laps_left and (pit is None or pit == st.compound):
            pit = "HARD" if st.compound != "HARD" else "MEDIUM"
            forced = True
        self.state, lap_time, parts = s.step(st, u_f, u_b, pit)
        self.last_lap = lap_time
        terminated = self.state.lap == s.laps
        info = {"lap_time": lap_time, "forced_change": forced, **parts}
        return self._obs(), s.base - lap_time, terminated, False, info


def run_episode(env, policy, seed=None):
    """Play one race with `policy(obs, env) -> action`. Returns race time and per-lap info."""
    obs, _ = env.reset(seed=seed)
    infos, done = [], False
    while not done:
        obs, _, done, _, info = env.step(policy(obs, env))
        infos.append(info)
    return env.state.race_time, infos


def plan_policy(compounds, pit_laps, fuel=None, batt=None):
    """Policy that replays a fixed plan and energy allocation through the environment."""
    stops = dict(zip(pit_laps, compounds[1:]))

    def policy(obs, env):
        lap = env.state.lap
        f = env.sim.fuel_nominal if fuel is None else fuel[lap]
        b = 0.0 if batt is None else batt[lap]
        return env.encode(f, b, stops.get(lap + 1))
    return policy
