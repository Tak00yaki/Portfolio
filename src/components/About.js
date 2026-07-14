const STATUS = [
  { label: "LOCATION", value: "Earth, usually at my desk" },
  { label: "ROLE", value: "Software Engineer" },
  { label: "MODE", value: "Always building something" },
  { label: "UPTIME", value: "Fueled by coffee" },
];

export default function About() {
  return (
    <section id="about" className="about">
      <div className="section-heading">
        <span className="section-heading__index">01</span>
        <h2>ABOUT ME</h2>
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
          <p className="terminal__output">
            Hi, I'm Khushi — a software engineer who loves building things
            that live on the internet. I mix clean code with a weakness for
            retro aesthetics, mixtapes, and side quests.
          </p>
          <p className="terminal__line">
            <span className="prompt">$</span> cat status.log
          </p>
          <ul className="status-list">
            {STATUS.map((row) => (
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
