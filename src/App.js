import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Marquee from "./components/Marquee";
import About from "./components/About";
import Interests from "./components/Interests";
import Projects from "./components/Projects";
import Footer from "./components/Footer";
import RacingBorder from "./components/RacingBorder";
import StartLights from "./components/StartLights";
import { ChaosProvider, useChaos } from "./chaos/ChaosContext";
import { LanguageProvider } from "./i18n/LanguageContext";
import "./App.css";

// Chaos level 1: the chibi F1 grid races round the edges, after the start lights
function ChaosLayer() {
  const { level, lights } = useChaos();
  return (
    <>
      {level >= 1 && lights !== "on" && <RacingBorder />}
      <StartLights />
    </>
  );
}

export default function App() {
  return (
    <LanguageProvider>
      <ChaosProvider>
        <div className="app">
          <div className="scanlines" aria-hidden="true" />
          <div className="grain" aria-hidden="true" />
          <ChaosLayer />

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
      </ChaosProvider>
    </LanguageProvider>
  );
}
