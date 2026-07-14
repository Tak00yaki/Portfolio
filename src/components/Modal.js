import { useEffect } from "react";

export default function Modal({ title, onClose, children }) {
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-window" onClick={(e) => e.stopPropagation()}>
        <div className="window-bar">
          <span
            className="window-bar__close"
            role="button"
            tabIndex={0}
            aria-label="Close"
            onClick={onClose}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onClose()}
          />
          <span />
          <span />
          <p>{title}</p>
        </div>
        <div className="modal-window__body">{children}</div>
      </div>
    </div>
  );
}
