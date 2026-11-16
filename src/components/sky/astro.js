// Low-precision astronomy for the background sky (good to a fraction of a
// degree, plenty for a picture). Angles in degrees unless noted.

const RAD = Math.PI / 180;
const OBLIQUITY = 23.4393 * RAD;

const norm360 = (d) => ((d % 360) + 360) % 360;

// Days since J2000.0 (2000-01-01 12:00 TT ~ UTC).
export const daysSinceJ2000 = (date) => date.getTime() / 86400000 - 10957.5;

// Local sidereal time: the right ascension currently on the meridian.
export function localSiderealTime(date, longitude) {
  return norm360(280.46061837 + 360.98564736629 * daysSinceJ2000(date) + longitude);
}

// Ecliptic (lon, lat) -> equatorial (ra, dec).
function eclipticToEquatorial(lon, lat) {
  const l = lon * RAD, b = lat * RAD;
  const x = Math.cos(b) * Math.cos(l);
  const y = Math.cos(b) * Math.sin(l) * Math.cos(OBLIQUITY) - Math.sin(b) * Math.sin(OBLIQUITY);
  const z = Math.cos(b) * Math.sin(l) * Math.sin(OBLIQUITY) + Math.sin(b) * Math.cos(OBLIQUITY);
  return { ra: norm360(Math.atan2(y, x) / RAD), dec: Math.asin(z) / RAD };
}

export function sunLongitude(date) {
  const d = daysSinceJ2000(date);
  const L = 280.46 + 0.9856474 * d;
  const g = (357.528 + 0.9856003 * d) * RAD;
  return norm360(L + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g));
}

// Moon position plus phase (illuminated fraction 0..1, waxing or waning).
export function moon(date) {
  const T = daysSinceJ2000(date) / 36525;
  const s = (deg) => Math.sin(deg * RAD);
  const lon = norm360(
    218.32 + 481267.881 * T +
    6.29 * s(135.0 + 477198.87 * T) - 1.27 * s(259.3 - 413335.36 * T) +
    0.66 * s(235.7 + 890534.22 * T) + 0.21 * s(269.9 + 954397.74 * T) -
    0.19 * s(357.5 + 35999.05 * T) - 0.11 * s(186.5 + 966404.03 * T)
  );
  const lat =
    5.13 * s(93.3 + 483202.02 * T) + 0.28 * s(228.2 + 960400.89 * T) -
    0.28 * s(318.3 + 6003.15 * T) - 0.17 * s(217.6 - 407332.21 * T);
  const elong = norm360(lon - sunLongitude(date));
  return {
    ...eclipticToEquatorial(lon, lat),
    illumination: (1 - Math.cos(elong * RAD)) / 2,
    waxing: elong < 180,
  };
}

// Keplerian elements and rates per century (JPL "Approximate Positions of the
// Planets", 1800-2050): a, e, I, L, longitude of perihelion, longitude of node.
const ELEMENTS = {
  mercury: [[0.38709927, 0.20563593, 7.00497902, 252.2503235, 77.45779628, 48.33076593],
            [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081]],
  venus:   [[0.72333566, 0.00677672, 3.39467605, 181.9790995, 131.60246718, 76.67984255],
            [0.0000039, -0.00004107, -0.0007889, 58517.81538729, 0.00268329, -0.27769418]],
  earth:   [[1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0],
            [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0]],
  mars:    [[1.52371034, 0.0933941, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
            [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343]],
  jupiter: [[5.202887, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
            [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106]],
  saturn:  [[9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
            [-0.0012506, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794]],
};

function heliocentric(name, T) {
  const [base, rate] = ELEMENTS[name];
  const [a, e, I, L, peri, node] = base.map((v, i) => v + rate[i] * T);
  const M = norm360(L - peri) * RAD;
  const w = (peri - node) * RAD, O = node * RAD, inc = I * RAD;
  let E = M + e * Math.sin(M);
  for (let i = 0; i < 6; i++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  const xp = a * (Math.cos(E) - e), yp = a * Math.sqrt(1 - e * e) * Math.sin(E);
  const cw = Math.cos(w), sw = Math.sin(w), cO = Math.cos(O), sO = Math.sin(O), cI = Math.cos(inc), sI = Math.sin(inc);
  return [
    (cw * cO - sw * sO * cI) * xp + (-sw * cO - cw * sO * cI) * yp,
    (cw * sO + sw * cO * cI) * xp + (-sw * sO + cw * cO * cI) * yp,
    sw * sI * xp + cw * sI * yp,
  ];
}

// Geocentric RA/Dec of the naked-eye planets.
export function planets(date) {
  const T = daysSinceJ2000(date) / 36525;
  const earth = heliocentric("earth", T);
  return ["mercury", "venus", "mars", "jupiter", "saturn"].map((name) => {
    const p = heliocentric(name, T);
    const x = p[0] - earth[0], y = p[1] - earth[1], z = p[2] - earth[2];
    const ye = y * Math.cos(OBLIQUITY) - z * Math.sin(OBLIQUITY);
    const ze = y * Math.sin(OBLIQUITY) + z * Math.cos(OBLIQUITY);
    return { name, ra: norm360(Math.atan2(ye, x) / RAD), dec: Math.atan2(ze, Math.hypot(x, ye)) / RAD };
  });
}
