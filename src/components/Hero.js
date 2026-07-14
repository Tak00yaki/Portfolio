export default function Hero() {
  return (
    <section id="top" className="hero">
      <div className="hero__rays" aria-hidden="true" />

      <div className="hero__content">
        <p className="hero__eyebrow">
          <span className="pulse-dot" /> SYSTEM ONLINE
        </p>
        <h1 className="hero__title">
          KHUSHI
          <br />
          KHAN<span className="cursor">_</span>
        </h1>
        <p className="hero__subtitle">
          SOFTWARE ENGINEER — BUILDER — PERPETUAL TINKERER
        </p>
        <div className="hero__actions">
          <a href="#projects" className="btn btn--solid">
            VIEW PROJECTS
          </a>
          <a href="#contact" className="btn btn--outline">
            GET IN TOUCH
          </a>
        </div>
      </div>

      <div className="hero__frame" aria-hidden="true">
        <div className="window-bar">
          <span />
          <span />
          <span />
          <p>profile.exe</p>
        </div>
        <div className="hero__screen">
          <span className="hero__monogram">KK</span>
          <span className="hero__screen-caption">swap with your photo ↴</span>
        </div>
      </div>

      <a href="#about" className="scroll-cue">
        SCROLL <span>↓</span>
      </a>
    </section>
  );
}
