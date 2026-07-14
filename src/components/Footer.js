const SOCIALS = [
  { label: "GH", name: "GitHub", href: "#" },
  { label: "IN", name: "LinkedIn", href: "#" },
  { label: "IG", name: "Instagram", href: "#" },
];

export default function Footer() {
  return (
    <footer id="contact" className="footer">
      <div className="section-heading section-heading--center">
        <span className="section-heading__index">04</span>
        <h2>GET IN TOUCH</h2>
      </div>

      <p className="footer__tagline">
        Got a project, an idea, or just want to say hi?
      </p>

      <a
        href="mailto:khushikhan1600@gmail.com"
        className="btn btn--solid footer__cta"
      >
        SAY HELLO
      </a>

      <div className="footer__socials">
        {SOCIALS.map((s) => (
          <a key={s.label} href={s.href} aria-label={s.name}>
            {s.label}
          </a>
        ))}
      </div>

      <p className="footer__bottom">
        © {new Date().getFullYear()} KHUSHI KHAN — BUILT WITH REACT &amp; A
        LOT OF COFFEE
      </p>
    </footer>
  );
}
