export default function InterestDetail({ item }) {
  return (
    <div className="interest-detail">
      <span className="interest-detail__icon">{item.icon}</span>
      <h3>{item.title}</h3>
      <p className="interest-detail__extra">{item.extra}</p>
      <div className="interest-detail__chips">
        {item.chips.map((chip) => (
          <span key={chip}>{chip}</span>
        ))}
      </div>
    </div>
  );
}
