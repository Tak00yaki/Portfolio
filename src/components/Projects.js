import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

export default function Projects() {
  const { lang } = useLanguage();
  const t = translations[lang].projects;

  return (
    <section id="projects" className="projects">
      <div className="section-heading">
        <span className="section-heading__index">03</span>
        <h2>{t.heading}</h2>
      </div>

      <div className="projects__grid">
        {t.items.map((p) => (
          <article className="project-card" key={p.file}>
            <div className="window-bar">
              <span />
              <span />
              <span />
              <p>{p.file}</p>
            </div>
            <div className="project-card__body">
              <h3>{p.title}</h3>
              <p>{p.desc}</p>
              <div className="project-card__tags">
                {p.tags.map((tag) => (
                  <span key={tag}>{tag}</span>
                ))}
              </div>
              <div className="project-card__links">
                <a href={p.live}>{t.liveLabel} ↗</a>
                <a href={p.code}>{t.codeLabel} ↗</a>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
