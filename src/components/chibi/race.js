// Race logic for the chibi grid, kept free of React and the DOM so it can be tested.
//
// Racing rules: every car keeps a safe gap to the car ahead on its line. A faster
// car pulls out to the passing line when it is clear, drives past, and cuts back
// in once there is room; if the passing line is blocked it follows instead.

export const CAR_W = 46;
export const CAR_H = (CAR_W * 130) / 80;
export const SAFE = CAR_H + 14;     // closest nose-to-tail spacing on the same line (px)
export const LINE = CAR_W + 4;      // distance between the racing line and the passing line
const LOOK = SAFE + 60;             // start thinking about a pass when this close
const LANE_RATE = 2.2;              // lane changes per second
export const BASE_SPEED = 60;       // px per second

// Cars clash if they are on (or moving between) overlapping lines
export const sideBySide = (a, b) => Math.abs(a.lat - b.lat) < (CAR_W + 2) / LINE;

// start "spread": evenly round the lap. start "grid": a two-wide grid queued just
// behind position 0 (the top of the right-hand lane), in random order.
export function createRace(n, lap, random = Math.random, start = "spread") {
  const race = { lap };
  race.state = Array.from({ length: n }, (_, i) => ({
    i,
    s: (i * lap) / n,
    v: BASE_SPEED,
    base: BASE_SPEED * (0.85 + random() * 0.3),
    phase: random() * Math.PI * 2,
    lat: 0,
    target: 0,
  }));
  const state = race.state;
  if (start === "grid") {
    const order = state.map((c) => c.i).sort(() => random() - 0.5);
    order.forEach((idx, slot) => {
      const c = state[idx];
      c.s = lap - (Math.floor(slot / 2) + 1) * SAFE - (slot % 2) * (SAFE / 2);   // staggered rows
      c.lat = c.target = slot % 2;
    });
  }
  const ahead = (a, b) => (((b.s - a.s) % race.lap) + race.lap) % race.lap;   // a forward to b
  race.ahead = ahead;
  const lineClear = (c, line, margin) =>
    state.every((o) => o === c || Math.abs(o.lat - line) >= (CAR_W + 2) / LINE
      || (ahead(c, o) > margin && ahead(o, c) > margin));

  race.resize = (newLap) => {
    state.forEach((c) => (c.s = (c.s / race.lap) * newLap));
    race.lap = newLap;
  };

  // Advance dt seconds. `paused` is the index of a car held in place (hovered), or null.
  race.step = (now, dt, paused = null) => {
    // Work from the front of the queue backwards, starting behind the biggest gap
    const order = [...state].sort((a, b) => b.s - a.s);
    let lead = 0, gap = -1;
    order.forEach((c, k) => {
      const d = ahead(c, order[(k - 1 + n) % n]) || race.lap;
      if (d > gap) { gap = d; lead = k; }
    });
    for (let k = 0; k < n; k++) {
      const c = order[(lead + k) % n];
      if (paused === c.i) { c.v = 0; continue; }
      const want = c.base * (1 + 0.2 * Math.sin(now / 2600 + c.phase));
      let front = null, dFront = Infinity;
      state.forEach((o) => {
        if (o === c || !sideBySide(c, o)) return;
        const d = ahead(c, o);
        if (d < dFront) { dFront = d; front = o; }
      });
      const settled = Math.abs(c.lat - c.target) < 0.01;
      if (settled && front && dFront < LOOK && want > front.v + 2 && lineClear(c, 1 - c.target, SAFE + 6)) {
        c.target = 1 - c.target;                          // pull out and go for the pass
      } else if (settled && c.target === 1 && lineClear(c, 0, SAFE + 20)
        && state.every((o) => o === c || Math.abs(o.lat) > 0.5 || ahead(c, o) > LOOK || o.v >= want - 2)) {
        c.target = 0;                                     // pass done: back to the racing line
      }
      c.v = want;
      if (front && dFront < LOOK) c.v = Math.max(0, Math.min(want, front.v + (dFront - SAFE) * 1.5));
      c.s = (c.s + c.v * dt) % race.lap;
      if (front && ahead(c, front) < SAFE) c.s = (front.s - SAFE + race.lap) % race.lap;   // never touch
      const dl = LANE_RATE * dt;
      c.lat += Math.max(-dl, Math.min(dl, c.target - c.lat));
    }
  };
  return race;
}
