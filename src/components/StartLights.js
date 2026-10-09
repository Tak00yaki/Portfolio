import { useEffect, useState } from "react";
import { useChaos } from "../chaos/ChaosContext";
import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

// F1 start sequence: five red lights come on one per second, hold for a random
// moment, then go out and the race starts.

const STEP_MS = 900;
const OUT_MESSAGE_MS = 1800;

export default function StartLights() {
  const { lights, setLights } = useChaos();
  const { lang } = useLanguage();
  const t = translations[lang].chaos;
  const [lit, setLit] = useState(0);

  useEffect(() => {
    if (lights !== "on") return undefined;
    setLit(0);
    const timers = [1, 2, 3, 4, 5].map((n) => setTimeout(() => setLit(n), n * STEP_MS));
    const hold = 5 * STEP_MS + 400 + Math.random() * 1200;
    timers.push(setTimeout(() => setLights("out"), hold));
    return () => timers.forEach(clearTimeout);
  }, [lights, setLights]);

  useEffect(() => {
    if (lights !== "out") return undefined;
    const t = setTimeout(() => setLights(null), OUT_MESSAGE_MS);
    return () => clearTimeout(t);
  }, [lights, setLights]);

  if (!lights) return null;
  return (
    <div className={`start-lights ${lights === "out" ? "is-out" : ""}`} role="status" aria-live="polite">
      <div className="start-lights__gantry" aria-hidden="true">
        {[0, 1, 2, 3, 4].map((i) => (
          <div className="start-lights__unit" key={i}>
            <span />
            <span className={lights === "on" && i < lit ? "is-lit" : ""} />
          </div>
        ))}
      </div>
      <p className="start-lights__text">{lights === "out" ? t.lightsOut : t.lightsOn}</p>
    </div>
  );
}
