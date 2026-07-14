import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

const SOCIALS = [
  { label: "GH", name: "GitHub", href: "#" },
  { label: "IN", name: "LinkedIn", href: "#" },
  { label: "IG", name: "Instagram", href: "#" },
];

export default function Footer() {
  const { lang } = useLanguage();
  const t = translations[lang].footer;

  return (
    <footer id="contact" className="footer">
      <div className="section-heading section-heading--center">
        <span className="section-heading__index">04</span>
        <h2>{t.heading}</h2>
      </div>

      <p className="footer__tagline">{t.tagline}</p>

      <a
        href="mailto:khushikhan1600@gmail.com"
        className="btn btn--solid footer__cta"
      >
        {t.cta}
      </a>

      <div className="footer__socials">
        {SOCIALS.map((s) => (
          <a key={s.label} href={s.href} aria-label={s.name}>
            {s.label}
          </a>
        ))}
      </div>

      <p className="footer__bottom">
        © {new Date().getFullYear()} KHUSHI KHAN — {t.bottomSuffix}
      </p>
    </footer>
  );
}
