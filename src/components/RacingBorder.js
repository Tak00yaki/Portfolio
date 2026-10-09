import { useEffect, useMemo, useRef, useState } from "react";
import { DRIVERS, TEAMS } from "./chibi/drivers";
import { carSVG, stickerSVG } from "./chibi/art";
import { CAR_H, CAR_W, LINE, createRace } from "./chibi/race";

// The 2026 grid as chibi cars racing down the right edge, round an off-screen
// hairpin, and back up the left edge, overtaking each other (rules in chibi/race.js).
// Shown at chaos level 1; the cars roll in from a grid just above the screen.
// Hover a car to pause it and meet the driver.

const TURN = 1100;           // off-screen hairpin at the top and bottom (px); fits the starting grid
const MIN_WIDTH = 1100;      // below this the side margins are too narrow

function lapGeometry() {
  const W = window.innerWidth, H = window.innerHeight;
  const straight = H + CAR_H;
  return { W, H, straight, lap: 2 * straight + 2 * TURN, gutter: W * 0.06 };
}

// Position on the loop -> screen position and direction (null while off-screen).
// lat 0 = racing line, hugging the inside edge of the lane; 1 = passing line, outside.
function place(s, lat, g) {
  const { W, H, straight, gutter } = g;
  const inner = CAR_W / 2 + 3;
  if (s < straight) return { x: W - gutter + inner + lat * LINE, y: -CAR_H + s, dir: "down" };
  if (s < straight + TURN) return null;
  if (s < 2 * straight + TURN) return { x: gutter - inner - lat * LINE, y: H - (s - straight - TURN), dir: "up" };
  return null;
}

export default function RacingBorder() {
  const cars = useRef([]);
  const hovered = useRef(null);
  const [card, setCard] = useState(null);

  const art = useMemo(
    () => DRIVERS.map((d) => ({ down: carSVG(d, "down"), up: carSVG(d, "up") })),
    []
  );

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let g = lapGeometry();
    const n = DRIVERS.length;
    const race = createRace(n, g.lap, Math.random, reduce ? "spread" : "grid");
    const state = race.state;
    if (reduce) {
      // Parked: half the grid on each side, evenly spaced, no movement
      state.forEach((c, i) => {
        const side = i % 2, slot = Math.floor(i / 2);
        const spacing = g.H / Math.ceil(n / 2);
        c.s = side === 0 ? CAR_H + slot * spacing : g.straight + TURN + CAR_H + slot * spacing;
      });
    }

    const draw = () => {
      state.forEach((c, i) => {
        const el = cars.current[i];
        if (!el) return;
        const p = place(c.s, c.lat, g);
        if (!p) {
          el.style.visibility = "hidden";
          return;
        }
        el.style.visibility = "visible";
        el.dataset.dir = p.dir;
        el.style.transform = `translate(${p.x - CAR_W / 2}px, ${p.y}px)`;
      });
    };

    let raf, last = performance.now();
    const tick = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (window.innerWidth >= MIN_WIDTH) {
        race.step(now, dt, hovered.current);
        draw();
      }
      raf = requestAnimationFrame(tick);
    };

    const onResize = () => {
      g = lapGeometry();
      race.resize(g.lap);
      draw();
    };
    window.addEventListener("resize", onResize);
    if (reduce) draw();
    else raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  const enter = (i) => (e) => {
    hovered.current = i;
    const r = e.currentTarget.getBoundingClientRect();
    const onRight = r.left > window.innerWidth / 2;
    setCard({
      i,
      top: Math.min(Math.max(12, r.top - 40), window.innerHeight - 330),
      [onRight ? "right" : "left"]: window.innerWidth * 0.06 + 8,
    });
  };
  const leave = () => {
    hovered.current = null;
    setCard(null);
  };

  const d = card && DRIVERS[card.i];
  return (
    <div className="racing" aria-hidden="true">
      <div className="racing__lane racing__lane--left" />
      <div className="racing__lane racing__lane--right" />
      {DRIVERS.map((drv, i) => (
        <div
          key={drv.code}
          className="racing__car"
          ref={(el) => (cars.current[i] = el)}
          onMouseEnter={enter(i)}
          onMouseLeave={leave}
        >
          <span className="racing__down" dangerouslySetInnerHTML={{ __html: art[i].down }} />
          <span className="racing__up" dangerouslySetInnerHTML={{ __html: art[i].up }} />
        </div>
      ))}
      {d && (
        <div className="racing__card" style={{ top: card.top, left: card.left, right: card.right }}>
          <div className="racing__sticker" dangerouslySetInnerHTML={{ __html: stickerSVG(d, "-card") }} />
          <p className="racing__name">
            <span>{d.num}</span> {d.name}
          </p>
          <p className="racing__team">{TEAMS[d.team].name}</p>
          <p className="racing__note">{d.note}</p>
        </div>
      )}
    </div>
  );
}
