const INTERESTS = [
  { icon: "🎧", title: "Music", desc: "Curating mixtapes and chasing new sounds." },
  { icon: "📸", title: "Photography", desc: "Capturing moments, one grainy shot at a time." },
  { icon: "🎮", title: "Gaming", desc: "Retro consoles and the occasional boss battle." },
  { icon: "📚", title: "Reading", desc: "Sci-fi, design books, and late-night rabbit holes." },
  { icon: "✈️", title: "Travel", desc: "Collecting stamps and stories." },
  { icon: "🎨", title: "Art", desc: "Doodles, digital art, and design experiments." },
];

export default function Interests() {
  return (
    <section id="interests" className="interests">
      <div className="section-heading">
        <span className="section-heading__index">02</span>
        <h2>INTERESTS</h2>
      </div>

      <div className="interests__grid">
        {INTERESTS.map((item) => (
          <div className="cassette" key={item.title}>
            <div className="cassette__window">
              <span className="cassette__reel" />
              <span className="cassette__reel" />
            </div>
            <div className="cassette__label">
              <span className="cassette__icon">{item.icon}</span>
              <h3>{item.title}</h3>
              <p>{item.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
