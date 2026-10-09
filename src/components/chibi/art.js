// SVG art for the chibi grid, as strings so both the site and a preview page can use it.
// stickerSVG: full-body sticker (hover card). carSVG: top-down car with the head in the cockpit.
import { TEAMS } from "./drivers.js";

const INK = "#0b0a08";

function mix(a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ch = (p, s) => (p >> s) & 255;
  const m = (s) => Math.round(ch(pa, s) + (ch(pb, s) - ch(pa, s)) * t);
  return `#${((1 << 24) + (m(16) << 16) + (m(8) << 8) + m(0)).toString(16).slice(1)}`;
}

function textOn(hex) {
  const n = parseInt(hex.slice(1), 16);
  return 0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255) > 150 ? INK : "#ffffff";
}

const O = (w = 3.4) => `stroke="${INK}" stroke-width="${w}" stroke-linejoin="round" stroke-linecap="round"`;

function defs(id) {
  return `<defs>
  <filter id="st-${id}" x="-25%" y="-25%" width="150%" height="150%">
    <feMorphology in="SourceAlpha" operator="dilate" radius="6" result="w0"/>
    <feFlood flood-color="#fffaf0"/><feComposite in2="w0" operator="in" result="white"/>
    <feMorphology in="SourceAlpha" operator="dilate" radius="8.5" result="k0"/>
    <feFlood flood-color="${INK}"/><feComposite in2="k0" operator="in" result="ink"/>
    <feOffset in="ink" dx="4" dy="6" result="shadow"/>
    <feMerge><feMergeNode in="shadow"/><feMergeNode in="ink"/><feMergeNode in="white"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <linearGradient id="shade-${id}" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#ffffff" stop-opacity=".22"/><stop offset=".55" stop-color="#ffffff" stop-opacity="0"/>
    <stop offset="1" stop-color="#000000" stop-opacity=".28"/>
  </linearGradient>
  <linearGradient id="visor-${id}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#2a3340"/><stop offset=".5" stop-color="#0d1218"/><stop offset="1" stop-color="#1c2530"/>
  </linearGradient>
  <radialGradient id="fur-${id}" cx=".38" cy=".3" r=".8">
    <stop offset="0" stop-color="#ffffff" stop-opacity=".35"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
  </radialGradient>
</defs>`;
}

// --- head (centred on 105,104; roughly 130 wide) -------------------------------------

const CX = 105, CY = 104, EY = 106;

function eyes(id, d) {
  const mood = d.mood;
  const closed = (x) => `<path d="M${x - 10} ${EY + 3} Q${x} ${EY - 10} ${x + 10} ${EY + 3}" stroke="${INK}" stroke-width="4.2" fill="none" stroke-linecap="round"/>`;
  const open = (x, side) => {
    const big = mood === "sparkle" ? 1.15 : 1;
    const brow = mood === "fierce"
      ? `<path d="M${x - 11 * side} ${EY - 19} L${x + 9 * side} ${EY - 13}" stroke="${INK}" stroke-width="4.2" stroke-linecap="round"/>` : "";
    const clip = `lid-${id}-${side}`;
    const lid = mood === "chill"
      ? `<clipPath id="${clip}"><ellipse cx="${x}" cy="${EY}" rx="10.5" ry="12.5"/></clipPath>
         <rect x="${x - 12}" y="${EY - 14}" width="24" height="13" fill="${d.fur}" clip-path="url(#${clip})"/>
         <path d="M${x - 10.5} ${EY - 1} H${x + 10.5}" stroke="${INK}" stroke-width="3.2" stroke-linecap="round"/>` : "";
    return `<ellipse cx="${x}" cy="${EY}" rx="${10.5 * big}" ry="${12.5 * big}" fill="${INK}"/>
      <ellipse cx="${x}" cy="${EY + 3}" rx="${7 * big}" ry="${8 * big}" fill="${d.eye}"/>
      <ellipse cx="${x}" cy="${EY + 3}" rx="3.6" ry="${6 * big}" fill="${INK}"/>
      <circle cx="${x + 3.5}" cy="${EY - 4}" r="${3.6 * big}" fill="#ffffff"/>
      <circle cx="${x - 3.5}" cy="${EY + 6}" r="1.8" fill="#ffffff"/>${lid}${brow}`;
  };
  const L = CX - 23, R = CX + 23;
  if (mood === "grin") return closed(L) + closed(R);
  if (mood === "wink") return open(L, 1) + closed(R);
  const sparkle = mood === "sparkle"
    ? `<path d="M${R + 17} ${EY - 22} l2 5 l5 2 l-5 2 l-2 5 l-2 -5 l-5 -2 l5 -2 Z" fill="#fff4d6" stroke="${INK}" stroke-width="1.6"/>` : "";
  return open(L, 1) + open(R, -1) + sparkle;
}

function mouth(d) {
  const sp = d.species;
  if (sp === "kiwi") return "";
  const y = sp === "koala" ? 138 : sp === "dog" ? 132 : 126;
  if (sp === "dog") {
    return `<path d="M90 ${y} Q105 ${y + 12} 120 ${y}" stroke="${INK}" stroke-width="3.2" fill="none" stroke-linecap="round"/>
      <path d="M96 ${y + 4} l2 -6 l3 6 M109 ${y + 4} l3 -6 l2 6" fill="#ffffff" stroke="${INK}" stroke-width="1.6" stroke-linejoin="round"/>`;
  }
  if (d.mood === "grin") {
    return `<path d="M93 ${y - 2} Q105 ${y + 16} 117 ${y - 2} Z" fill="#7a1f2b" stroke="${INK}" stroke-width="3" stroke-linejoin="round"/>
      <path d="M98 ${y + 7} Q105 ${y + 12} 112 ${y + 7}" fill="#ff8fa3"/>`;
  }
  if (d.mood === "smile" || d.mood === "wink" || d.mood === "sparkle") {
    return `<path d="M97 ${y} Q105 ${y + 9} 113 ${y} Z" fill="#7a1f2b" stroke="${INK}" stroke-width="2.6" stroke-linejoin="round"/>`;
  }
  const w = `<path d="M96 ${y - 1} Q101 ${y + 4} 105 ${y - 1} Q109 ${y + 4} 114 ${y - 1}" stroke="${INK}" stroke-width="3" fill="none" stroke-linecap="round"/>`;
  return d.mood === "fierce" ? w + `<path d="M108 ${y + 1} l2 5 l2 -5" fill="#ffffff" stroke="${INK}" stroke-width="1.4"/>` : w;
}

function ears(d) {
  const o = O();
  const inner = d.inner || mix(d.fur, "#ff9fb0", 0.55);
  const fill = d.mask ? d.dark : d.fur;
  switch (d.species) {
    case "cat":
      return `<path d="M58 84 L60 34 Q62 30 66 33 L98 60 Z" fill="${fill}" ${o}/>
        <path d="M66 74 L67 45 L88 63 Z" fill="${inner}"/>
        <path d="M152 84 L150 34 Q148 30 144 33 L112 60 Z" fill="${d.patches ? d.dark : fill}" ${o}/>
        <path d="M144 74 L143 45 L122 63 Z" fill="${inner}"/>`;
    case "lion": case "bear":
      return `<circle cx="62" cy="64" r="16" fill="${d.fur}" ${o}/><circle cx="62" cy="64" r="8" fill="${inner}"/>
        <circle cx="148" cy="64" r="16" fill="${d.fur}" ${o}/><circle cx="148" cy="64" r="8" fill="${inner}"/>`;
    case "beaver":
      return `<circle cx="66" cy="62" r="11" fill="${d.fur}" ${o}/><circle cx="144" cy="62" r="11" fill="${d.fur}" ${o}/>`;
    case "koala": {
      const tuft = (x) => `<path d="M${x - 9} ${70} q4 -6 8 0 q4 -6 8 0" stroke="#ffffff" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
      return `<circle cx="50" cy="74" r="26" fill="${d.fur}" ${o}/><circle cx="50" cy="76" r="15" fill="#e9e4ea"/>${tuft(50)}
        <circle cx="160" cy="74" r="26" fill="${d.fur}" ${o}/><circle cx="160" cy="76" r="15" fill="#e9e4ea"/>${tuft(160)}`;
    }
    case "dog":
      return "";   // floppy ears are drawn over the head
    default:
      return "";
  }
}

export function head(id, d) {
  const o = O();
  const sp = d.species;
  const muzzle = d.muzzle || mix(d.fur, "#ffffff", 0.6);
  let back = "";
  if (sp === "lion") {
    for (let a = 0; a < 360; a += 24) {
      const x = CX + 62 * Math.cos((a * Math.PI) / 180), y = CY + 54 * Math.sin((a * Math.PI) / 180);
      back += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="19" fill="${d.mane}" ${o}/>`;
    }
    back += `<ellipse cx="${CX}" cy="${CY}" rx="66" ry="58" fill="${d.mane}"/>`;
  }
  if (d.mullet) {
    back += `<path d="M58 118 Q46 150 62 166 Q66 150 74 146 Q70 162 84 170 Q86 150 92 140 Z" fill="${d.dark}" ${o}/>
      <path d="M152 118 Q164 150 148 166 Q144 150 136 146 Q140 162 126 170 Q124 150 118 140 Z" fill="${d.dark}" ${o}/>`;
  }
  const shape = sp === "kiwi"
    ? `<ellipse cx="${CX}" cy="${CY + 2}" rx="57" ry="53" fill="${d.fur}" ${o}/>
       <path d="M68 70 l6 5 M90 58 l4 6 M122 58 l-3 6 M144 72 l-6 4 M60 100 l7 1 M150 102 l-7 1" stroke="${d.dark}" stroke-width="3" stroke-linecap="round"/>`
    : `<ellipse cx="${CX}" cy="${CY}" rx="60" ry="51" fill="${d.fur}" ${o}/>
       <path d="M47 116 l-8 4 l9 2 M163 116 l8 4 l-9 2" stroke="${INK}" stroke-width="3" fill="${d.fur}" stroke-linejoin="round"/>`;
  let marks = "";
  if (d.stripes) {
    marks += `<path d="M95 58 Q97 68 94 74 M105 55 V72 M115 58 Q113 68 116 74" stroke="${d.dark}" stroke-width="5" stroke-linecap="round" fill="none"/>
      <path d="M48 100 h12 M47 110 h11 M162 100 h-12 M163 110 h-11" stroke="${d.dark}" stroke-width="4.5" stroke-linecap="round"/>`;
  }
  if (d.patches) marks += `<ellipse cx="130" cy="94" rx="24" ry="20" fill="${d.dark}" opacity=".9"/>`;
  if (sp === "dog") {
    marks += `<path d="M92 66 Q105 60 118 66 M95 74 Q105 69 115 74" stroke="${d.dark}" stroke-width="3" fill="none" stroke-linecap="round"/>`;
  }
  // muzzle + nose
  let face = "";
  if (sp === "cat" || sp === "lion") {
    face += `<ellipse cx="${CX}" cy="124" rx="24" ry="17" fill="${d.mask ? d.dark : muzzle}"/>`;
    face += `<path d="M100 116 h10 l-5 6 Z" fill="#ff6f91" stroke="${INK}" stroke-width="2.4" stroke-linejoin="round"/>`;
  } else if (sp === "bear" || sp === "beaver") {
    face += `<ellipse cx="${CX}" cy="124" rx="25" ry="18" fill="${muzzle}" stroke="${INK}" stroke-width="2.4"/>`;
    face += `<ellipse cx="${CX}" cy="116" rx="8" ry="5.5" fill="${INK}"/><ellipse cx="102" cy="114.5" rx="2.5" ry="1.4" fill="#ffffff" opacity=".7"/>`;
  } else if (sp === "dog") {
    face += `<path d="M70 130 Q70 112 92 112 L118 112 Q140 112 140 130 Q140 148 105 148 Q70 148 70 130 Z" fill="${muzzle}" stroke="${INK}" stroke-width="2.6"/>`;
    face += `<ellipse cx="${CX}" cy="118" rx="10" ry="7" fill="${INK}"/><ellipse cx="101" cy="116" rx="3" ry="1.6" fill="#ffffff" opacity=".7"/>`;
  } else if (sp === "koala") {
    face += `<ellipse cx="${CX}" cy="120" rx="11" ry="15" fill="#2a2a30" ${O(2.6)}/><ellipse cx="101" cy="113" rx="3" ry="4" fill="#ffffff" opacity=".35"/>`;
  }
  const beak = sp === "kiwi"
    ? `<path d="M96 112 Q105 108 114 112 Q111 132 109 158 Q107 166 103 160 Q100 134 96 112 Z" fill="#e8d9a8" ${O(2.8)}/>
       <path d="M105 116 Q105 138 105.5 158" stroke="${INK}" stroke-width="1.4" opacity=".45" fill="none"/>
       <ellipse cx="101" cy="114" rx="2" ry="1.2" fill="${INK}" opacity=".5"/><ellipse cx="109" cy="114" rx="2" ry="1.2" fill="${INK}" opacity=".5"/>`
    : "";
  const teeth = sp === "beaver"
    ? `<rect x="99" y="128" width="6" height="10" rx="1.5" fill="#ffffff" ${O(2)}/><rect x="105" y="128" width="6" height="10" rx="1.5" fill="#ffffff" ${O(2)}/>` : "";
  const dogEars = sp === "dog"
    ? `<path d="M62 60 Q34 60 38 98 Q42 114 56 106 Q58 84 74 64 Z" fill="${d.dark}" ${o}/>
       <path d="M148 60 Q176 60 172 98 Q168 114 154 106 Q152 84 136 64 Z" fill="${d.dark}" ${o}/>` : "";
  const whiskers = sp === "cat" || sp === "lion"
    ? `<path d="M66 120 l-24 -4 M66 126 l-24 3 M144 120 l24 -4 M144 126 l24 3" stroke="${INK}" stroke-width="1.8" stroke-linecap="round" opacity=".7"/>` : "";
  const blush = `<ellipse cx="${CX - 38}" cy="122" rx="9" ry="5" fill="#ff8fa3" opacity=".75"/><ellipse cx="${CX + 38}" cy="122" rx="9" ry="5" fill="#ff8fa3" opacity=".75"/>`;
  const tache = d.tache
    ? `<path d="M83 128 Q94 118 105 124 Q116 118 127 128 Q122 134 114 130 Q108 128 105 131 Q102 128 96 130 Q88 134 83 128 Z" fill="${d.dark}" stroke="${INK}" stroke-width="2.6" stroke-linejoin="round"/>` : "";
  return `${back}${ears(d)}${shape}${marks}
    <ellipse cx="${CX}" cy="${CY}" rx="58" ry="49" fill="url(#fur-${id})"/>
    ${face}${eyes(id, d)}${blush}${mouth(d)}${teeth}${tache}${beak}${whiskers}${dogEars}${headProps(d)}`;
}

function headProps(d) {
  const o = O();
  if (d.prop === "headset") {
    return `<path d="M50 112 Q46 52 105 48 Q164 52 160 112" fill="none" stroke="${INK}" stroke-width="9" stroke-linecap="round"/>
      <path d="M50 112 Q46 52 105 48 Q164 52 160 112" fill="none" stroke="#47c7fc" stroke-width="4.5" stroke-linecap="round"/>
      <rect x="38" y="98" width="17" height="28" rx="7" fill="#47c7fc" ${o}/><rect x="155" y="98" width="17" height="28" rx="7" fill="#47c7fc" ${o}/>
      <path d="M46 124 Q52 142 74 140" stroke="${INK}" stroke-width="3" fill="none"/><circle cx="76" cy="140" r="4" fill="${INK}"/>`;
  }
  if (d.prop === "shades") {
    return `<rect x="66" y="95" width="32" height="21" rx="9" fill="#15171c" ${O(3)}/>
      <rect x="112" y="95" width="32" height="21" rx="9" fill="#15171c" ${O(3)}/>
      <path d="M98 101 Q105 97 112 101" stroke="${INK}" stroke-width="3.4" fill="none"/>
      <path d="M58 100 L66 101 M144 101 L152 100" stroke="${INK}" stroke-width="3.4"/>
      <path d="M72 110 l9 -10 M118 110 l9 -10" stroke="#ffffff" stroke-width="2.6" opacity=".5" stroke-linecap="round"/>`;
  }
  if (d.prop === "kitten") {
    return `<g transform="translate(118 18) scale(.42)">
      <path d="M28 70 L30 20 Q32 16 36 19 L68 46 Z" fill="#f5a35c" ${O(7)}/>
      <path d="M122 70 L120 20 Q118 16 114 19 L82 46 Z" fill="#f5a35c" ${O(7)}/>
      <ellipse cx="75" cy="80" rx="56" ry="44" fill="#f5a35c" ${O(7)}/>
      <path d="M50 82 Q58 70 66 82 M84 82 Q92 70 100 82" stroke="${INK}" stroke-width="7" fill="none" stroke-linecap="round"/>
      <path d="M70 92 h10 l-5 6 Z" fill="#ff6f91"/>
      <ellipse cx="40" cy="96" rx="10" ry="6" fill="#ff8fa3" opacity=".8"/><ellipse cx="110" cy="96" rx="10" ry="6" fill="#ff8fa3" opacity=".8"/></g>`;
  }
  return "";
}

// --- body, held helmet, tail, hand props -------------------------------------------

function body(id, d) {
  const t = TEAMS[d.team];
  const paw = d.muzzle || mix(d.fur, "#ffffff", 0.35);
  const o = O(3.2);
  const raised = ["stopwatch", "plan", "football", "trophy", "shield"].includes(d.prop)
    || !["chill", "fierce"].includes(d.mood);
  const numInk = textOn(t.suit) === INK ? INK : t.pipe;
  const arm = raised ? "M128 172 Q150 160 156 138" : "M128 172 Q142 190 140 206";
  const hand = raised
    ? `<ellipse cx="157" cy="131" rx="11" ry="12" fill="${paw}" ${o}/>
       ${d.prop ? "" : `<path d="M150 124 v-6 M156 121 v-8 M162 122 v-6" stroke="${INK}" stroke-width="2.4" stroke-linecap="round"/>`}
       <rect x="146" y="138" width="22" height="7" rx="3" fill="${t.panel}" ${o}/>`
    : `<ellipse cx="141" cy="212" rx="10" ry="10" fill="${paw}" ${o}/>`;
  return `
  <path d="M86 206 L84 236 Q84 240 90 240 L98 240 Q101 240 101 236 L101 206 Z" fill="${t.suit}" ${o}/>
  <path d="M119 206 L121 236 Q121 240 115 240 L107 240 Q104 240 104 236 L104 206 Z" fill="${t.suit}" ${o}/>
  <path d="M88 212 L87 234 M117 212 L118 234" stroke="${t.pipe}" stroke-width="2.4"/>
  <path d="M80 236 Q80 248 92 248 L102 248 Q104 248 104 244 L104 234 L82 234 Z" fill="${INK}" ${o}/>
  <path d="M125 236 Q125 248 113 248 L103 248 Q101 248 101 244 L101 234 L123 234 Z" fill="${INK}" ${o}/>
  <path d="M82 244 h20 M103 244 h20" stroke="${t.pipe}" stroke-width="2.4"/>
  <path d="${arm}" fill="none" stroke="${INK}" stroke-width="19" stroke-linecap="round"/>
  <path d="${arm}" fill="none" stroke="${t.suit}" stroke-width="13" stroke-linecap="round"/>
  <path d="M74 168 Q74 156 88 154 L122 154 Q136 156 136 168 L134 210 Q134 216 126 216 L84 216 Q76 216 76 210 Z" fill="${t.suit}" ${o}/>
  <path d="M76 175 Q80 196 78 212 L86 214 Q84 194 82 172 Z" fill="${t.panel}"/>
  <path d="M134 175 Q130 196 132 212 L124 214 Q126 194 128 172 Z" fill="${t.panel}"/>
  <path d="M84 170 Q87 192 86 213 M126 170 Q123 192 124 213" stroke="${t.pipe}" stroke-width="2.2" fill="none"/>
  <path d="M105 160 V196" stroke="${INK}" stroke-width="1.4" opacity=".45"/>
  <rect x="78" y="198" width="54" height="7" fill="${t.panel}" stroke="${INK}" stroke-width="2"/>
  <rect x="101" y="197" width="9" height="9" rx="2" fill="${t.pipe}" stroke="${INK}" stroke-width="2"/>
  <text x="119" y="189" text-anchor="middle" font-family="Bungee, sans-serif" font-size="13" fill="${numInk}">${d.num}</text>
  <path d="M74 168 Q74 156 88 154 L122 154 Q136 156 136 168 L134 210 Q134 216 126 216 L84 216 Q76 216 76 210 Z" fill="url(#shade-${id})"/>
  <path d="M90 153 Q105 162 120 153 L118 147 Q105 154 92 147 Z" fill="${t.panel}" ${o}/>
  <!--hand-->${hand}`;
}

function heldHelmet(id, d) {
  const [base, band, accent] = d.helmet;
  const paw = d.muzzle || mix(d.fur, "#ffffff", 0.35);
  const cx = 66, cy = 196, r = 27;
  const lplate = d.prop === "lplate"
    ? `<g transform="rotate(-12 ${cx + 8} ${cy - 20})"><rect x="${cx}" y="${cy - 30}" width="17" height="17" rx="2" fill="#ffffff" ${O(2.4)}/>
       <path d="M${cx + 5} ${cy - 26} v10 h7" stroke="#e3101b" stroke-width="3.4" fill="none" stroke-linecap="square"/></g>` : "";
  return `
  <circle cx="${cx}" cy="${cy}" r="${r}" fill="${base}" stroke="${INK}" stroke-width="3.2"/>
  <path d="M${cx - r + 3} ${cy - 8} Q${cx} ${cy - r - 6} ${cx + r - 3} ${cy - 8} L${cx + r - 2} ${cy - 2} Q${cx} ${cy - r + 4} ${cx - r + 2} ${cy - 2} Z" fill="${band}"/>
  <path d="M${cx - r + 6} ${cy + 14} Q${cx} ${cy + 24} ${cx + r - 6} ${cy + 14}" stroke="${accent}" stroke-width="5" fill="none"/>
  <rect x="${cx - r + 4}" y="${cy - 4}" width="${2 * r - 8}" height="16" rx="8" fill="url(#visor-${id})" stroke="${INK}" stroke-width="2.6"/>
  <path d="M${cx - 12} ${cy - 1} l8 0 M${cx - 15} ${cy + 4} l14 0" stroke="#ffffff" stroke-width="2" opacity=".55" stroke-linecap="round"/>
  <circle cx="${cx + r - 8}" cy="${cy + 4}" r="3" fill="${accent}" stroke="${INK}" stroke-width="1.6"/>
  <circle cx="${cx - 9}" cy="${cy - 15}" r="5" fill="#ffffff" opacity=".35"/>
  ${lplate}
  <ellipse cx="${cx + 14}" cy="${cy + 2}" rx="11" ry="12" fill="${paw}" stroke="${INK}" stroke-width="3.2"/>`;
}

function tail(d) {
  const o = `stroke="${INK}" stroke-width="3.2" stroke-linejoin="round"`;
  if (d.species === "cat" || d.species === "lion") {
    const p = "M132 214 Q172 214 176 182 Q178 160 162 156";
    const tuft = d.species === "lion" ? `<circle cx="162" cy="156" r="9" fill="${d.mane}" ${o}/>` : "";
    const rings = d.stripes ? `<path d="M168 166 l7 -2 M174 184 l7 1" stroke="${d.dark}" stroke-width="4" stroke-linecap="round"/>` : "";
    return `<path d="${p}" fill="none" stroke="${INK}" stroke-width="16" stroke-linecap="round"/>
      <path d="${p}" fill="none" stroke="${d.species === "cat" && d.tuxedo ? d.fur : d.fur}" stroke-width="10" stroke-linecap="round"/>${rings}${tuft}`;
  }
  if (d.species === "beaver") {
    return `<ellipse cx="150" cy="222" rx="26" ry="12" transform="rotate(-25 150 222)" fill="${d.dark}" ${o}/>
      <path d="M134 226 l28 -14 M140 232 l26 -12 M144 216 l12 18 M154 212 l10 16" stroke="${INK}" stroke-width="1.4" opacity=".5"/>`;
  }
  if (d.species === "dog") return `<ellipse cx="136" cy="206" rx="7" ry="9" transform="rotate(30 136 206)" fill="${d.fur}" ${o}/>`;
  return "";
}

function handProp(d) {
  const o = O(2.8);
  const at = (x, y, s) => `translate(${x} ${y})`;
  switch (d.prop) {
    case "stopwatch":
      return `<g transform="${at(160, 112)}"><rect x="-4" y="-21" width="8" height="7" rx="2" fill="#c9ced6" ${o}/>
        <circle r="15" fill="#e9edf2" ${O(3)}/><circle r="10.5" fill="#ffffff" stroke="#2ee6d6" stroke-width="2"/>
        <path d="M0 0 V-8 M0 0 L6 3" stroke="${INK}" stroke-width="2.4" stroke-linecap="round"/><circle r="2" fill="${INK}"/></g>`;
    case "plan":
      return `<g transform="translate(160 112) rotate(-12)"><rect x="-19" y="-15" width="38" height="30" rx="3" fill="#fffaf0" ${O(3)}/>
        <rect x="-19" y="-15" width="38" height="7" fill="#ffd400" ${O(2)}/>
        <text x="0" y="5" text-anchor="middle" font-family="Bungee, sans-serif" font-size="7.5" fill="${INK}">EL PLAN</text>
        <path d="M-13 10 h26" stroke="${INK}" stroke-width="1.2" opacity=".5"/></g>`;
    case "football":
      return `<g transform="${at(160, 114)}"><circle r="14" fill="#ffffff" ${O(3)}/>
        <path d="M0 -6 l5.7 4.1 l-2.2 6.7 h-7 l-2.2 -6.7 Z" fill="${INK}"/>
        <path d="M0 -6 V-14 M5.7 -1.9 L13 -4 M3.5 4.8 L8 11 M-3.5 4.8 L-8 11 M-5.7 -1.9 L-13 -4" stroke="${INK}" stroke-width="1.6"/></g>`;
    case "trophy":
      return `<g transform="${at(160, 106)}">
        <rect x="-12" y="14" width="24" height="8" fill="#e3101b" ${O(2.4)}/>
        <rect x="-4" y="5" width="8" height="10" fill="#1c64d8" ${O(2.4)}/>
        <path d="M-15 -14 H15 V-2 Q15 7 0 7 Q-15 7 -15 -2 Z" fill="#ffd400" ${O(2.6)}/>
        <path d="M-15 -10 Q-23 -10 -21 -2 Q-19 3 -13 2 M15 -10 Q23 -10 21 -2 Q19 3 13 2" stroke="${INK}" stroke-width="2.6" fill="none"/>
        <rect x="-11" y="-19" width="7" height="5" rx="1" fill="#ffd400" ${O(2)}/><rect x="4" y="-19" width="7" height="5" rx="1" fill="#ffd400" ${O(2)}/>
        <rect x="-9" y="17" width="5" height="3" fill="#ffffff" opacity=".4"/></g>`;
    case "shield":
      return `<g transform="${at(162, 112)}"><path d="M0 -18 L16 -12 V0 Q16 13 0 20 Q-16 13 -16 0 V-12 Z" fill="#c8a45d" ${O(3)}/>
        <path d="M0 -12 L10 -8 V0 Q10 9 0 14 Q-10 9 -10 0 V-8 Z" fill="#1a1a1a"/>
        <text x="0" y="5" text-anchor="middle" font-family="Bungee, sans-serif" font-size="10" fill="#c8a45d">${d.num}</text></g>`;
    default:
      return "";
  }
}

function floatProp(d) {
  if (d.prop === "notes") {
    const note = (x, y, r) => `<g transform="translate(${x} ${y}) rotate(${r})"><path d="M5 -2 V-22 Q12 -18 14 -12" stroke="${INK}" stroke-width="3" fill="none" stroke-linecap="round"/>
      <ellipse cx="0" cy="0" rx="7" ry="5" transform="rotate(-20)" fill="#fffaf0" ${O(2.6)}/></g>`;
    return note(176, 66, 6) + note(196, 92, -8) + note(30, 70, -6);
  }
  if (d.prop === "hearts") {
    const heart = (x, y, s) => `<path transform="translate(${x} ${y}) scale(${s})" d="M0 6 C-12 -2 -8 -14 0 -7 C8 -14 12 -2 0 6 Z" fill="#ff4d6d" ${O(2.4 / s)}/>`;
    return heart(178, 58, 1.3) + heart(198, 84, 1) + heart(30, 66, 1.1);
  }
  return "";
}

// --- public ------------------------------------------------------------------------

export function stickerSVG(d, inst = "") {
  const id = `${d.code}${inst}`;
  const [torso, hand] = body(id, d).split("<!--hand-->");
  return `<svg viewBox="-6 0 232 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${d.name}">
  ${defs(id)}
  <g filter="url(#st-${id})">${tail(d)}${torso}${head(id, d)}${heldHelmet(id, d)}${hand}${handProp(d)}${floatProp(d)}</g>
</svg>`;
}

// Top-down car, nose pointing down ("down") or up ("up"). The head stays upright.
export function carSVG(d, dir = "down", inst = "") {
  const id = `car-${d.code}${dir}${inst}`;
  const t = TEAMS[d.team];
  const flip = dir === "up" ? `transform="rotate(180 40 65)"` : "";
  const k = 0.36;
  const hy = dir === "up" ? 76 : 52;
  const ny = dir === "up" ? 35 : 95;
  return `<svg viewBox="0 0 80 130" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  ${defs(id)}
  <g ${flip}>
    <rect x="12" y="4" width="56" height="12" rx="3" fill="${t.panel}" ${O(2.6)}/>
    ${[[5, 18, 14, 28], [61, 18, 14, 28], [7, 86, 13, 22], [60, 86, 13, 22]].map(([x, y, w, h]) =>
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="#26221e" stroke="#6b645b" stroke-width="2"/>
       <path d="M${x + w / 2} ${y + 4} V${y + h - 4}" stroke="${t.pipe}" stroke-width="2" opacity=".8"/>`).join("")}
    <path d="M28 14 L52 14 Q61 30 59 54 L57 84 Q53 100 45 112 L35 112 Q27 100 23 84 L21 54 Q19 30 28 14 Z" fill="${t.suit}" ${O(2.6)}/>
    <path d="M40 18 V108" stroke="${t.pipe}" stroke-width="3"/>
    <path d="M24 60 Q22 40 28 30 M56 60 Q58 40 52 30" stroke="${t.panel}" stroke-width="4" fill="none" stroke-linecap="round"/>
    <rect x="11" y="111" width="58" height="11" rx="3" fill="${t.panel}" ${O(2.6)}/>
    <ellipse cx="40" cy="56" rx="13" ry="18" fill="${INK}"/>
  </g>
  <circle cx="40" cy="${ny}" r="8.5" fill="#ffffff" ${O(2.2)}/>
  <text x="40" y="${ny + 3.5}" text-anchor="middle" font-family="Bungee, sans-serif" font-size="9" fill="${INK}">${d.num}</text>
  <g transform="translate(${40 - 105 * k} ${hy - 104 * k}) scale(${k})">${head(id, d)}</g>
</svg>`;
}
