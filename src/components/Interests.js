import { useState } from "react";
import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";
import Modal from "./Modal";
import InterestDetail from "./InterestDetail";
import TravelMap from "./TravelMap";

export default function Interests() {
  const { lang } = useLanguage();
  const t = translations[lang].interests;
  const travelT = translations[lang].travel;
  const [selected, setSelected] = useState(null);

  return (
    <section id="interests" className="interests">
      <div className="section-heading">
        <span className="section-heading__index">02</span>
        <h2>{t.heading}</h2>
      </div>

      <div className="interests__grid">
        {t.items.map((item) => (
          <button
            className="cassette"
            key={item.id}
            onClick={() => setSelected(item)}
          >
            <div className="cassette__window">
              <span className="cassette__reel" />
              <span className="cassette__reel" />
            </div>
            <div className="cassette__label">
              <span className="cassette__icon">{item.icon}</span>
              <h3>{item.title}</h3>
              <p>{item.desc}</p>
            </div>
          </button>
        ))}
      </div>

      {selected && (
        <Modal
          title={selected.id === "travel" ? "travel-map.exe" : `${selected.id}.exe`}
          onClose={() => setSelected(null)}
        >
          {selected.id === "travel" ? (
            <>
              <h3 className="modal-window__heading">{travelT.heading}</h3>
              <TravelMap />
            </>
          ) : (
            <InterestDetail item={selected} />
          )}
        </Modal>
      )}
    </section>
  );
}
