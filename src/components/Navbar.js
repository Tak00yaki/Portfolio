import { useState } from "react";
import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const { lang, setLang } = useLanguage();
  const t = translations[lang].nav;

  return (
    <header className="navbar">
      <a href="#top" className="navbar__logo">
        KK
      </a>

      <nav className={`navbar__links ${open ? "is-open" : ""}`}>
        {t.links.map((link) => (
          <a key={link.href} href={link.href} onClick={() => setOpen(false)}>
            {link.label}
          </a>
        ))}
        <div className="lang-toggle" role="group" aria-label="Language">
          <button
            className={lang === "en" ? "is-active" : ""}
            onClick={() => setLang("en")}
          >
            EN
          </button>
          <button
            className={lang === "ja" ? "is-active" : ""}
            onClick={() => setLang("ja")}
          >
            日
          </button>
        </div>
      </nav>

      <button
        className={`navbar__toggle ${open ? "is-open" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-label="Toggle menu"
        aria-expanded={open}
      >
        <span />
        <span />
        <span />
      </button>
    </header>
  );
}
