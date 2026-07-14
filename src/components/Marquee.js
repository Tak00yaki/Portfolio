const ITEMS = [
  "REACT",
  "JAVASCRIPT",
  "UI / UX",
  "CREATIVE CODE",
  "COFFEE",
  "GOOD VIBES",
  "BUILDING THINGS",
  "RETRO EVERYTHING",
];

export default function Marquee() {
  const row = [...ITEMS, ...ITEMS];

  return (
    <div className="marquee">
      <div className="marquee__track">
        {row.map((item, i) => (
          <span className="marquee__item" key={i}>
            {item} <span className="marquee__star">★</span>
          </span>
        ))}
      </div>
    </div>
  );
}
