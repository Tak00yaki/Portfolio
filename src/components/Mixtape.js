import { useState } from "react";
import "./Mixtape.css";

// Interests as a mixtape J-card: a cassette on the left whose reels spin while
// a track is hovered, and the interests listed as Side A / Side B tracks.
export default function Mixtape({ items, onSelect }) {
  const [playing, setPlaying] = useState(null);
  const half = Math.ceil(items.length / 2);
  const sides = [["A", items.slice(0, half)], ["B", items.slice(half)]];

  return (
    <div className={`mixtape${playing !== null ? " is-playing" : ""}`}>
      <div className="mixtape__tape" aria-hidden="true">
        <div className="mixtape__label">
          <span>KK · MIX VOL.1</span>
          <span>{playing !== null ? `▶ ${items[playing].title}` : "INTERESTS"}</span>
        </div>
        <div className="mixtape__window">
          <span className="mixtape__reel" />
          <span className="mixtape__ribbon" />
          <span className="mixtape__reel" />
        </div>
        <div className="mixtape__foot" />
      </div>

      <div className="mixtape__sides">
        {sides.map(([side, list], s) => (
          <div className="mixtape__side" key={side}>
            <h3>SIDE {side}</h3>
            <ol>
              {list.map((item, j) => {
                const idx = s * half + j;
                return (
                  <li key={item.id}>
                    <button
                      className="track"
                      onMouseEnter={() => setPlaying(idx)}
                      onMouseLeave={() => setPlaying(null)}
                      onFocus={() => setPlaying(idx)}
                      onBlur={() => setPlaying(null)}
                      onClick={() => onSelect(item)}
                    >
                      <span className="track__no">{side}{j + 1}</span>
                      <span className="track__icon">{item.icon}</span>
                      <span className="track__body">
                        <strong>{item.title}</strong>
                        <span>{item.desc}</span>
                      </span>
                      <span className="track__play" aria-hidden="true">▶</span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </div>
        ))}
      </div>
    </div>
  );
}
