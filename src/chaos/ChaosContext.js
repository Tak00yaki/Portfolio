import { createContext, useCallback, useContext, useState } from "react";

// Site-wide "chaos level", picked with the dial in the navbar.
// 0 = calm (default), 1 = the chibi F1 grid races round the edges. 2-5 are locked for now.
// The choice is remembered per visitor in localStorage.

export const LEVEL_COUNT = 6;
export const UNLOCKED = [0, 1];
const KEY = "kk-chaos-level";

function readLevel() {
  try {
    const v = Number(window.localStorage.getItem(KEY));
    return UNLOCKED.includes(v) ? v : 0;
  } catch {
    return 0;
  }
}

const ChaosContext = createContext(null);

export function ChaosProvider({ children }) {
  const [level, setLevelState] = useState(readLevel);
  // Start-lights sequence before the race: null | "on" (lights coming on) | "out" (race started)
  const [lights, setLights] = useState(null);

  const setLevel = useCallback(
    (next) => {
      if (next === 1 && level !== 1) setLights("on");
      if (next !== 1) setLights(null);
      setLevelState(next);
      try {
        window.localStorage.setItem(KEY, String(next));
      } catch {
        // storage blocked: the level just won't be remembered
      }
    },
    [level]
  );

  return (
    <ChaosContext.Provider value={{ level, setLevel, lights, setLights }}>
      {children}
    </ChaosContext.Provider>
  );
}

export function useChaos() {
  return useContext(ChaosContext);
}
