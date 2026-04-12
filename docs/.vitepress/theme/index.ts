import { h } from "vue";
import DefaultTheme from "vitepress/theme";
import type { Theme } from "vitepress";
import { enhanceWithConsent, ConsentBanner } from "@structured-world/vue-privacy/vitepress";
import { createKVStorage } from "@structured-world/vue-privacy";
import "./style.css";
import BugReportWidget from "./components/BugReportWidget.vue";

// GA4 tracking ID — set via VITE_GA_ID env var in CI
const GA_ID = (import.meta.env as Record<string, string>).VITE_GA_ID || "";

// GDPR-compliant consent management with GA4 SPA tracking.
// Falls back gracefully when GA_ID is not set (no tracking).
const consentTheme = enhanceWithConsent(DefaultTheme, {
  gaId: GA_ID,
  // KV storage via Cloudflare Worker (configured in R-DOC2 — bug-reports-worker)
  storage: createKVStorage("/api/consent"),
});

export default {
  ...consentTheme,
  Layout() {
    return h(consentTheme.Layout ?? DefaultTheme.Layout, null, {
      // ConsentBanner: GDPR consent banner from vue-privacy
      // BugReportWidget: floating bug report button (backend wired in R-DOC2)
      "layout-bottom": () => [h(ConsentBanner), h(BugReportWidget)],
    });
  },
} satisfies Theme;
