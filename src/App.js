import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Marquee from "./components/Marquee";
import About from "./components/About";
import Interests from "./components/Interests";
import Projects from "./components/Projects";
import Footer from "./components/Footer";
import "./App.css";

export default function App() {
  return (
    <div className="app">
      <div className="scanlines" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />

      <Navbar />
      <main>
        <Hero />
        <Marquee />
        <About />
        <Interests />
        <Projects />
      </main>
      <Footer />
    </div>
  );
}
