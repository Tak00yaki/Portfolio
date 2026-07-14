import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

export default function Interests() {
  const { lang } = useLanguage();
  const t = translations[lang].interests;

  return (
    <section id="interests" className="interests">
      <div className="section-heading">
        <span className="section-heading__index">02</span>
        <h2>{t.heading}</h2>
      </div>

      <div className="interests__grid">
        {t.items.map((item) => (
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
