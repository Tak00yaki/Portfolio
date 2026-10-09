import { useEffect, useRef, useState } from "react";
import { LEVEL_COUNT, UNLOCKED, useChaos } from "../chaos/ChaosContext";
import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

// Navbar "CHAOS" control, styled like a rotary switch on an F1 steering wheel.
// Click to open: pick a level, or SPIN and let the dial decide.

const SWEEP = 270;
const SPIN_MS = 1800;
const angleFor = (level) => -SWEEP / 2 + (level * SWEEP) / (LEVEL_COUNT - 1);
const shortest = (d) => ((((d + 180) % 360) + 360) % 360) - 180;

function Knob({ angle, spinning, big }) {
  const ticks = Array.from({ length: LEVEL_COUNT }, (_, i) => {
    const a = ((angleFor(i) - 90) * Math.PI) / 180;
    const locked = !UNLOCKED.includes(i);
    return (
      <g key={i} className={locked ? "is-locked" : ""}>
        <line x1={Math.cos(a) * 38} y1={Math.sin(a) * 38} x2={Math.cos(a) * 45} y2={Math.sin(a) * 45} />
        {big && (
          <text x={Math.cos(a) * 31} y={Math.sin(a) * 31 + 3}>
            {i}
          </text>
        )}
      </g>
    );
  });
  return (
    <svg className="chaos-knob" viewBox="-50 -50 100 100" aria-hidden="true">
      <circle className="chaos-knob__ring" r="47" />
      {ticks}
      <g
        className="chaos-knob__cap"
        style={{
          transform: `rotate(${angle}deg)`,
          transition: `transform ${spinning ? SPIN_MS : 350}ms cubic-bezier(.15,.7,.2,1.08)`,
        }}
      >
        <circle r={big ? 22 : 26} />
        {[0, 60, 120, 180, 240, 300].map((d) => (
          <rect key={d} x="-2" y={big ? -22 : -26} width="4" height="5" transform={`rotate(${d})`} />
        ))}
        <rect className="chaos-knob__pointer" x="-2.5" y={big ? -20 : -24} width="5" height={big ? 13 : 15} rx="2" />
      </g>
    </svg>
  );
}

export default function ChaosDial() {
  const { level, setLevel } = useChaos();
  const { lang } = useLanguage();
  const t = translations[lang].chaos;
  const [open, setOpen] = useState(false);
  const [angle, setAngle] = useState(() => angleFor(level));
  const [spinning, setSpinning] = useState(false);
  const root = useRef(null);

  // Turn the knob the short way to the current level (unless a spin is running)
  useEffect(() => {
    if (!spinning) setAngle((a) => a + shortest(angleFor(level) - a));
  }, [level, spinning]);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => root.current && !root.current.contains(e.target) && setOpen(false);
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const spin = () => {
    if (spinning) return;
    const target = UNLOCKED[Math.floor(Math.random() * UNLOCKED.length)];
    setSpinning(true);
    setAngle((a) => a + 3 * 360 + shortest(angleFor(target) - a));
    setTimeout(() => {
      setSpinning(false);
      setLevel(target);
    }, SPIN_MS);
  };

  return (
    <div className="chaos" ref={root}>
      <button
        className={`chaos__button ${open ? "is-open" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label={`${t.label} ${level}: ${t.levels[level]}`}
      >
        <Knob angle={angle} spinning={spinning} />
        <span>
          {t.label} <b>{level}</b>
        </span>
      </button>

      {open && (
        <div className="chaos__panel" role="dialog" aria-label={t.label}>
          <p className="chaos__hint">{t.hint}</p>
          <div className="chaos__big">
            <Knob angle={angle} spinning={spinning} big />
          </div>
          <ul className="chaos__levels">
            {Array.from({ length: LEVEL_COUNT }, (_, i) => {
              const locked = !UNLOCKED.includes(i);
              return (
                <li key={i}>
                  <button
                    className={i === level ? "is-active" : ""}
                    disabled={locked || spinning}
                    onClick={() => setLevel(i)}
                  >
                    <b>{i}</b> {locked ? t.locked : t.levels[i]}
                  </button>
                </li>
              );
            })}
          </ul>
          <button className="chaos__spin" onClick={spin} disabled={spinning}>
            {spinning ? t.spinning : t.spin}
          </button>
        </div>
      )}
    </div>
  );
}
