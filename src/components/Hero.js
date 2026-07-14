import f1 from "../assets/gallery/f1.png";
import plane from "../assets/gallery/plane.png";
import cocoa from "../assets/gallery/cocoa.png";
import kfc from "../assets/gallery/kfc.png";
import cassette from "../assets/gallery/cassette.png";
import controller from "../assets/gallery/controller.png";
import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

const WINDOWS = [
  { file: "f1.exe", src: f1, alt: "Pixel-art F1 race car" },
  { file: "plane.exe", src: plane, alt: "Pixel-art airplane" },
  { file: "cocoa.exe", src: cocoa, alt: "Pixel-art mug of hot cocoa" },
  { file: "kfc.exe", src: kfc, alt: "Pixel-art fried chicken bucket" },
  { file: "tape.exe", src: cassette, alt: "Pixel-art cassette tape" },
  { file: "pad.exe", src: controller, alt: "Pixel-art game controller" },
];

export default function Hero() {
  const { lang } = useLanguage();
  const t = translations[lang].hero;

  return (
    <section id="top" className="hero">
      <div className="hero__rays" aria-hidden="true" />

      <div className="hero__content">
        <p className="hero__eyebrow">
          <span className="pulse-dot" /> {t.eyebrow}
        </p>
        <h1 className="hero__title">
          KHUSHI
          <br />
          KHAN<span className="cursor">_</span>
        </h1>
        <p className="hero__subtitle">{t.subtitle}</p>
        <div className="hero__actions">
          <a href="#projects" className="btn btn--solid">
            {t.viewProjects}
          </a>
          <a href="#contact" className="btn btn--outline">
            {t.getInTouch}
          </a>
        </div>
      </div>

      <div className="hero__gallery" aria-hidden="true">
        {WINDOWS.map((win) => (
          <div className="mini-window" key={win.file}>
            <div className="window-bar">
              <span />
              <span />
              <span />
              <p>{win.file}</p>
            </div>
            <div className="mini-window__screen">
              <img className="mini-window__image" src={win.src} alt={win.alt} />
            </div>
          </div>
        ))}
      </div>

      <a href="#about" className="scroll-cue">
        {t.scroll} <span>↓</span>
      </a>
    </section>
  );
}
