import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

export default function About() {
  const { lang } = useLanguage();
  const t = translations[lang].about;

  return (
    <section id="about" className="about">
      <div className="section-heading">
        <span className="section-heading__index">01</span>
        <h2>{t.heading}</h2>
      </div>

      <div className="terminal">
        <div className="window-bar">
          <span />
          <span />
          <span />
          <p>whoami.sh</p>
        </div>
        <div className="terminal__body">
          <p className="terminal__line">
            <span className="prompt">$</span> whoami
          </p>
          <p className="terminal__output">{t.bio}</p>
          <p className="terminal__line">
            <span className="prompt">$</span> cat status.log
          </p>
          <ul className="status-list">
            {t.status.map((row) => (
              <li key={row.label}>
                <span>{row.label}</span>
                {row.value}
              </li>
            ))}
          </ul>
          <p className="terminal__line">
            <span className="prompt">$</span>
            <span className="cursor">_</span>
          </p>
        </div>
      </div>
    </section>
  );
}
