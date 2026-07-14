import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

export default function Marquee() {
  const { lang } = useLanguage();
  const items = translations[lang].marquee.items;
  const row = [...items, ...items];

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
