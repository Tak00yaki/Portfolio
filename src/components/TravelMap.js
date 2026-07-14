import { useEffect, useRef, useState } from "react";
import worldMap from "../assets/map/world.png";
import tokyoImg from "../assets/cities/tokyo.png";
import yokohamaImg from "../assets/cities/yokohama.png";
import shizuokaImg from "../assets/cities/shizuoka.png";
import tsuruokaImg from "../assets/cities/tsuruoka.png";
import sendaiImg from "../assets/cities/sendai.png";
import { useLanguage } from "../i18n/LanguageContext";
import { translations } from "../i18n/translations";

const CITIES = [
  { id: "tokyo", nameEn: "Tokyo", nameJa: "東京", x: 88.7, y: 29.5, visited: true, img: tokyoImg },
  { id: "yokohama", nameEn: "Yokohama", nameJa: "横浜", x: 89.0, y: 29.95, visited: true, img: yokohamaImg },
  { id: "shizuoka", nameEn: "Shizuoka", nameJa: "静岡", x: 88.44, y: 30.01, visited: true, img: shizuokaImg },
  { id: "tsuruoka", nameEn: "Tsuruoka", nameJa: "鶴岡", x: 88.84, y: 28.48, visited: true, img: tsuruokaImg },
  { id: "sendai", nameEn: "Sendai", nameJa: "仙台", x: 89.13, y: 28.74, visited: true, img: sendaiImg },
  { id: "kyoto", nameEn: "Kyoto", nameJa: "京都", x: 87.72, y: 30.55, visited: false },
  { id: "okinawa", nameEn: "Okinawa", nameJa: "沖縄", x: 85.47, y: 35.44, visited: false },
  { id: "sapporo", nameEn: "Sapporo", nameJa: "札幌", x: 89.26, y: 26.08, visited: false },
];

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const JAPAN_CENTER = { x: 88.6, y: 29.5 };

export default function TravelMap() {
  const { lang } = useLanguage();
  const t = translations[lang].travel;

  const viewportRef = useRef(null);
  const [zoom, setZoom] = useState(MIN_ZOOM);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selected, setSelected] = useState(CITIES[0]);
  const dragState = useRef(null);

  const panTo = (targetZoom, fx, fy) => {
    const rect = viewportRef.current.getBoundingClientRect();
    setZoom(targetZoom);
    setPan({
      x: rect.width / 2 - targetZoom * (fx / 100) * rect.width,
      y: rect.height / 2 - targetZoom * (fy / 100) * rect.height,
    });
  };

  useEffect(() => {
    panTo(3.4, JAPAN_CENTER.x, JAPAN_CENTER.y);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const zoomBy = (factor) => {
    const rect = viewportRef.current.getBoundingClientRect();
    const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom * factor));
    const worldX = (rect.width / 2 - pan.x) / (zoom * rect.width);
    const worldY = (rect.height / 2 - pan.y) / (zoom * rect.height);
    setZoom(newZoom);
    setPan({
      x: rect.width / 2 - newZoom * worldX * rect.width,
      y: rect.height / 2 - newZoom * worldY * rect.height,
    });
  };

  const onWheel = (e) => {
    e.preventDefault();
    zoomBy(e.deltaY < 0 ? 1.2 : 1 / 1.2);
  };

  const onPointerDown = (e) => {
    if (e.target.closest("button")) return;
    dragState.current = { startX: e.clientX, startY: e.clientY, pan };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e) => {
    if (!dragState.current) return;
    const { startX, startY, pan: startPan } = dragState.current;
    setPan({
      x: startPan.x + (e.clientX - startX),
      y: startPan.y + (e.clientY - startY),
    });
  };

  const onPointerUp = () => {
    dragState.current = null;
  };

  return (
    <div className="travel-map">
      <p className="travel-map__subtitle">{t.subtitle}</p>

      <div
        className="travel-map__viewport"
        ref={viewportRef}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <div
          className="travel-map__layer"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}
        >
          <img className="travel-map__image" src={worldMap} alt="Stylized world map" draggable={false} />
          {CITIES.map((city) => (
            <button
              key={city.id}
              className={`travel-map__pin ${city.visited ? "is-visited" : "is-wishlist"} ${selected.id === city.id ? "is-selected" : ""}`}
              style={{
                left: `${city.x}%`,
                top: `${city.y}%`,
                transform: `translate(-50%, -50%) scale(${1 / zoom})`,
              }}
              onClick={() => setSelected(city)}
              aria-label={lang === "ja" ? city.nameJa : city.nameEn}
            />
          ))}
        </div>

        <div className="travel-map__controls">
          <button className="travel-map__zoom-btn" onClick={() => zoomBy(1.4)} aria-label={t.zoomIn}>+</button>
          <button className="travel-map__zoom-btn" onClick={() => zoomBy(1 / 1.4)} aria-label={t.zoomOut}>−</button>
          <button className="travel-map__focus" onClick={() => panTo(3.4, JAPAN_CENTER.x, JAPAN_CENTER.y)}>
            {t.focusJapan}
          </button>
        </div>
      </div>

      <div className="travel-map__info">
        <div className="travel-map__info-thumb">
          {selected.visited ? (
            <img src={selected.img} alt={lang === "ja" ? selected.nameJa : selected.nameEn} />
          ) : (
            <div className="broken-thumb">
              <span className="broken-thumb__glyph">
                <span className="broken-thumb__mountain" />
                <span className="broken-thumb__sun" />
                <span className="broken-thumb__crack" />
              </span>
              <p>{t.noPhoto}</p>
            </div>
          )}
        </div>
        <div className="travel-map__info-text">
          <h3>{lang === "ja" ? selected.nameJa : selected.nameEn}</h3>
          <span className={`travel-map__badge ${selected.visited ? "is-visited" : "is-wishlist"}`}>
            {selected.visited ? t.visited : t.wishlist}
          </span>
        </div>
      </div>
    </div>
  );
}
