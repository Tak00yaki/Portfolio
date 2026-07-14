const PROJECTS = [
  {
    title: "Project One",
    file: "project-one.exe",
    desc: "A short description of what this project does and the problem it solves.",
    tags: ["React", "Node", "CSS"],
    live: "#",
    code: "#",
  },
  {
    title: "Project Two",
    file: "project-two.exe",
    desc: "Another project worth showing off — swap in the real details here.",
    tags: ["Python", "API"],
    live: "#",
    code: "#",
  },
  {
    title: "Project Three",
    file: "project-three.exe",
    desc: "Describe the tech stack, your role, and the outcome in a line or two.",
    tags: ["TypeScript", "Design"],
    live: "#",
    code: "#",
  },
  {
    title: "Project Four",
    file: "project-four.exe",
    desc: "Fourth slot for a side project, hackathon build, or experiment.",
    tags: ["React", "Firebase"],
    live: "#",
    code: "#",
  },
];

export default function Projects() {
  return (
    <section id="projects" className="projects">
      <div className="section-heading">
        <span className="section-heading__index">03</span>
        <h2>PROJECTS</h2>
      </div>

      <div className="projects__grid">
        {PROJECTS.map((p) => (
          <article className="project-card" key={p.title}>
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
                <a href={p.live}>LIVE ↗</a>
                <a href={p.code}>CODE ↗</a>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
