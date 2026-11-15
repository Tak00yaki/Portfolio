// Visitor tracking via Umami Cloud (https://cloud.umami.is).
// The Umami dashboard shows each visit's time and location (country, region,
// city) under Sessions / Realtime. No cookies are set.
//
// Set REACT_APP_UMAMI_WEBSITE_ID in .env.production to the Website ID from the
// Umami dashboard. If it is unset, nothing is loaded.

const WEBSITE_ID = process.env.REACT_APP_UMAMI_WEBSITE_ID;
const SCRIPT_URL =
  process.env.REACT_APP_UMAMI_SCRIPT_URL || 'https://cloud.umami.is/script.js';

export function initAnalytics() {
  if (!WEBSITE_ID || typeof document === 'undefined') return;
  if (document.querySelector('script[data-website-id]')) return;

  const script = document.createElement('script');
  script.defer = true;
  script.src = SCRIPT_URL;
  script.dataset.websiteId = WEBSITE_ID;
  // Only count visits on the live site, not localhost or preview builds.
  script.dataset.domains = 'khushikhan.com,www.khushikhan.com';
  document.head.appendChild(script);
}
