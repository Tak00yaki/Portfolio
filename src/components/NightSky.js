import { useEffect, useRef } from "react";
import { useLanguage } from "../i18n/LanguageContext";
import { localSiderealTime, moon as moonAt, planets as planetsAt } from "./sky/astro";

// Fixed night-sky backdrop behind the whole site: the real sky (about 5,000
// stars, the 88 constellations, the Moon in its current phase and the bright
// planets), centred on what is overhead for the visitor right now. Scrolling
// the page pans the view from the northern sky down to the southern sky.
// Static (no twinkle, drift or shooting stars) when reduced motion is on.

const RAD = Math.PI / 180;
const DATA_URL = `${process.env.PUBLIC_URL}/sky/sky.json`;
const DEC_TOP = 50;     // view centre declination at the top of the page
const DEC_BOTTOM = -45; // ...and at the bottom
const DRIFT = 0.12;     // extra westward drift, degrees per second
const FPS = 40;

const PLANET_STYLE = {
  mercury: { r: 1.6, c: "220,210,200", en: "Mercury", ja: "水星" },
  venus:   { r: 2.8, c: "255,248,225", en: "Venus",   ja: "金星" },
  mars:    { r: 2.1, c: "255,150,110", en: "Mars",    ja: "火星" },
  jupiter: { r: 2.6, c: "255,232,200", en: "Jupiter", ja: "木星" },
  saturn:  { r: 2.2, c: "240,220,170", en: "Saturn",  ja: "土星" },
};

// B-V colour index -> star tint.
function starColor(bv) {
  if (bv < 0) return "175,200,255";
  if (bv < 0.3) return "210,222,255";
  if (bv < 0.6) return "250,248,255";
  if (bv < 1.0) return "255,238,215";
  if (bv < 1.4) return "255,215,170";
  return "255,192,145";
}

// Unit-vector form of (ra, dec) so per-frame projection is cheap.
const point = (ra, dec) => ({ ra: ra * RAD, sd: Math.sin(dec * RAD), cd: Math.cos(dec * RAD) });

function prepare(data) {
  return {
    stars: data.stars.map(([ra, dec, mag, bv]) => ({
      ...point(ra, dec),
      r: Math.min(3, Math.max(0.4, 0.4 + (6 - mag) * 0.38)),
      a: Math.min(1, Math.max(0.28, 0.3 + (6 - mag) * 0.14)),
      c: starColor(bv),
      glow: mag < 1.6,
      tw: 0.6 + Math.random() * 1.8,
      ph: Math.random() * Math.PI * 2,
    })),
    names: data.names.map(([ra, dec, , en, ja]) => ({ ...point(ra, dec), en, ja })),
    lines: data.lines.map((fig) => fig.map((seg) => seg.map(([ra, dec]) => point(ra, dec)))),
    cons: data.cons.map(([ra, dec, rank, en, ja]) => ({ ...point(ra, dec), rank, en, ja })),
  };
}

function drawMoon(ctx, x, y, r, illum, waxing) {
  const glow = ctx.createRadialGradient(x, y, r * 0.8, x, y, r * 4);
  glow.addColorStop(0, `rgba(244,230,200,${0.08 + 0.18 * illum})`);
  glow.addColorStop(1, "rgba(244,230,200,0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, r * 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.save();
  ctx.translate(x, y);
  if (!waxing) ctx.scale(-1, 1); // waning: lit side on the left
  ctx.fillStyle = "rgba(40,44,72,0.95)";
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fill();
  // Lit right half, then the terminator as half an ellipse.
  const rx = r * Math.abs(1 - 2 * illum);
  ctx.fillStyle = "rgb(246,236,212)";
  ctx.beginPath();
  ctx.arc(0, 0, r, -Math.PI / 2, Math.PI / 2, false);
  ctx.ellipse(0, 0, rx, r, 0, Math.PI / 2, -Math.PI / 2, illum < 0.5);
  ctx.fill();
  ctx.restore();
}

export default function NightSky() {
  const canvasRef = useRef(null);
  const { lang } = useLanguage();
  const langRef = useRef(lang);
  const redrawRef = useRef(() => {});

  useEffect(() => {
    langRef.current = lang;
    redrawRef.current();
  }, [lang]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
    let w = 0, h = 0, scale = 1, sky = null, raf = 0, lastFrame = 0;
    let shooting = null, nextShot = 0, cancelled = false;

    const now = new Date();
    // Approximate longitude from the visitor's time zone, so the sky roughly
    // matches what is above them tonight.
    const longitude = (-now.getTimezoneOffset() / 60) * 15;
    const lstStart = localSiderealTime(now, longitude);
    const startTime = performance.now();
    const moon = moonAt(now);
    const moonPt = point(moon.ra, moon.dec);
    const planets = planetsAt(now).map((p) => ({ ...p, ...point(p.ra, p.dec), ...PLANET_STYLE[p.name] }));

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // About 120 degrees across the longer side of the screen.
      scale = Math.max(w, h * 1.1) / (2 * 2 * Math.tan(30 * RAD));
      draw(performance.now());
    }

    function view(t) {
      const doc = document.documentElement;
      const max = Math.max(1, doc.scrollHeight - window.innerHeight);
      const frac = Math.min(1, Math.max(0, (window.scrollY || 0) / max));
      const drift = reduce.matches ? 0 : ((t - startTime) / 1000) * DRIFT;
      const dec0 = (DEC_TOP + (DEC_BOTTOM - DEC_TOP) * frac) * RAD;
      return { ra0: (lstStart + drift) * RAD, sd0: Math.sin(dec0), cd0: Math.cos(dec0) };
    }

    // Stereographic projection; returns null for points far behind the viewer.
    function project(p, v, out) {
      const dra = p.ra - v.ra0;
      const cra = Math.cos(dra);
      const cosc = v.sd0 * p.sd + v.cd0 * p.cd * cra;
      if (cosc < -0.3) return null;
      const k = (2 / (1 + cosc)) * scale;
      out.x = w / 2 - k * p.cd * Math.sin(dra); // east is to the left, as when facing south
      out.y = h / 2 - k * (v.cd0 * p.sd - v.sd0 * p.cd * cra);
      return out;
    }

    const A = { x: 0, y: 0 }, B = { x: 0, y: 0 };
    const onScreen = (q, m = 40) => q.x > -m && q.x < w + m && q.y > -m && q.y < h + m;

    function draw(t) {
      ctx.clearRect(0, 0, w, h);
      if (!sky) return;
      const v = view(t);
      const ja = langRef.current === "ja";
      const time = t / 1000;
      const small = w < 640;

      // Constellation figures.
      ctx.strokeStyle = "rgba(244,230,200,0.13)";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      for (const fig of sky.lines) {
        for (const seg of fig) {
          let prev = null;
          for (const p of seg) {
            const q = project(p, v, prev === A ? B : A);
            if (q && prev && Math.hypot(q.x - prev.x, q.y - prev.y) < w) {
              ctx.moveTo(prev.x, prev.y);
              ctx.lineTo(q.x, q.y);
            }
            prev = q;
          }
        }
      }
      ctx.stroke();

      // Stars, brightest last so they sit on top.
      for (let i = sky.stars.length - 1; i >= 0; i--) {
        const s = sky.stars[i];
        const q = project(s, v, A);
        if (!q || !onScreen(q, 10)) continue;
        const alpha = s.a * (reduce.matches ? 1 : 0.72 + 0.28 * Math.sin(time * s.tw + s.ph));
        if (s.glow) {
          const g = ctx.createRadialGradient(q.x, q.y, 0, q.x, q.y, s.r * 4.5);
          g.addColorStop(0, `rgba(${s.c},${alpha * 0.55})`);
          g.addColorStop(1, `rgba(${s.c},0)`);
          ctx.fillStyle = g;
          ctx.fillRect(q.x - s.r * 4.5, q.y - s.r * 4.5, s.r * 9, s.r * 9);
        }
        ctx.fillStyle = `rgba(${s.c},${alpha})`;
        if (s.r < 0.9) {
          ctx.fillRect(q.x - s.r, q.y - s.r, s.r * 2, s.r * 2);
        } else {
          ctx.beginPath();
          ctx.arc(q.x, q.y, s.r, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Constellation names (smaller ones hidden on phones to avoid clutter).
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = `${small ? 9 : 10}px "Space Mono", monospace`;
      if ("letterSpacing" in ctx) ctx.letterSpacing = ja ? "1px" : "2px";
      for (const c of sky.cons) {
        if (c.rank > (small ? 1 : 2)) continue;
        const q = project(c, v, A);
        if (!q || !onScreen(q)) continue;
        ctx.fillStyle = `rgba(205,187,150,${c.rank === 1 ? 0.4 : 0.28})`;
        ctx.fillText(ja ? c.ja : c.en.toUpperCase(), q.x, q.y);
      }
      if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";

      // Bright star names.
      ctx.textAlign = "left";
      ctx.font = `${small ? 9 : 10}px "Space Mono", monospace`;
      ctx.fillStyle = "rgba(255,236,210,0.5)";
      for (const n of sky.names) {
        const q = project(n, v, A);
        if (q && onScreen(q)) ctx.fillText(ja ? n.ja : n.en, q.x + 7, q.y - 7);
      }

      // Planets. Labels step down when planets sit close together (conjunctions).
      const labels = [];
      const placeLabel = (x, y) => {
        for (const l of labels) {
          if (Math.abs(l.x - x) < 60 && Math.abs(l.y - y) < 11) y = l.y + 12;
        }
        labels.push({ x, y });
        return y;
      };
      for (const p of planets) {
        const q = project(p, v, A);
        if (!q || !onScreen(q)) continue;
        const g = ctx.createRadialGradient(q.x, q.y, 0, q.x, q.y, p.r * 5);
        g.addColorStop(0, `rgba(${p.c},0.45)`);
        g.addColorStop(1, `rgba(${p.c},0)`);
        ctx.fillStyle = g;
        ctx.fillRect(q.x - p.r * 5, q.y - p.r * 5, p.r * 10, p.r * 10);
        ctx.fillStyle = `rgb(${p.c})`;
        ctx.beginPath();
        ctx.arc(q.x, q.y, p.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = `rgba(${p.c},0.7)`;
        ctx.fillText(ja ? p.ja : p.en, q.x + 8, placeLabel(q.x + 8, q.y + 9));
      }

      // Moon, drawn larger than life so its phase reads.
      const mq = project(moonPt, v, A);
      if (mq && onScreen(mq, 80)) {
        const r = small ? 11 : 15;
        drawMoon(ctx, mq.x, mq.y, r, moon.illumination, moon.waxing);
        ctx.textAlign = "center";
        ctx.fillStyle = "rgba(246,236,212,0.6)";
        ctx.fillText(ja ? "月" : "Moon", mq.x, mq.y + r + 12);
      }

      if (!reduce.matches) drawShootingStar(t);
    }

    function drawShootingStar(t) {
      if (!shooting && t > nextShot) {
        const angle = (20 + Math.random() * 20) * RAD;
        const dir = Math.random() < 0.5 ? 1 : -1;
        const speed = 700 + Math.random() * 400;
        shooting = {
          x: dir > 0 ? Math.random() * w * 0.6 : w * 0.4 + Math.random() * w * 0.6,
          y: Math.random() * h * 0.45,
          vx: Math.cos(angle) * dir * speed,
          vy: Math.sin(angle) * speed,
          start: t,
          life: 0.9 + Math.random() * 0.5,
        };
      }
      if (!shooting) return;
      const age = (t - shooting.start) / 1000;
      if (age > shooting.life) {
        shooting = null;
        nextShot = t + 6000 + Math.random() * 8000;
        return;
      }
      const p = age / shooting.life;
      const fade = p < 0.15 ? p / 0.15 : 1 - (p - 0.15) / 0.85;
      const hx = shooting.x + shooting.vx * age, hy = shooting.y + shooting.vy * age;
      const tx = hx - shooting.vx * 0.12, ty = hy - shooting.vy * 0.12;
      const g = ctx.createLinearGradient(hx, hy, tx, ty);
      g.addColorStop(0, `rgba(255,240,220,${0.9 * fade})`);
      g.addColorStop(1, "rgba(255,180,120,0)");
      ctx.strokeStyle = g;
      ctx.lineWidth = 1.6;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(tx, ty);
      ctx.stroke();
    }

    function loop(t) {
      if (t - lastFrame >= 1000 / FPS - 2) {
        lastFrame = t;
        draw(t);
      }
      raf = requestAnimationFrame(loop);
    }

    function start() {
      cancelAnimationFrame(raf);
      if (reduce.matches) {
        draw(performance.now());
      } else {
        nextShot = performance.now() + 3000;
        raf = requestAnimationFrame(loop);
      }
    }

    const onScroll = () => { if (reduce.matches) draw(performance.now()); };
    redrawRef.current = () => draw(performance.now());

    resize();
    start();
    window.addEventListener("resize", resize);
    window.addEventListener("scroll", onScroll, { passive: true });
    reduce.addEventListener?.("change", start);

    fetch(DATA_URL)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((data) => {
        if (cancelled) return;
        sky = prepare(data);
        draw(performance.now());
      })
      .catch(() => { /* sky stays as the plain gradient */ });

    // Redraw once the label font has loaded.
    document.fonts?.ready.then(() => !cancelled && draw(performance.now()));

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      redrawRef.current = () => {};
      window.removeEventListener("resize", resize);
      window.removeEventListener("scroll", onScroll);
      reduce.removeEventListener?.("change", start);
    };
  }, []);

  return (
    <div className="night-sky" aria-hidden="true">
      <div className="night-sky__milkyway" />
      <canvas ref={canvasRef} className="night-sky__stars" />
      <div className="night-sky__horizon" />
    </div>
  );
}
