import { defineConfig, type HeadConfig } from "vitepress";

const base = (process.env.DOCS_BASE as `/${string}/` | undefined) ?? "/";
const hostname = "https://coordinode.dev";

const siteDescription =
  "Graph + Vector + Full-Text retrieval in a single MVCC transaction. OpenCypher-compatible. Built in Rust.";

interface PageData {
  relativePath: string;
  title?: string;
  description?: string;
}

function generateStructuredData(pageData: PageData): object[] {
  const cleanPath = pageData.relativePath.replace(/(?:index)?\.md$/, "");
  const siteUrl = `${hostname}${base}`.replace(/\/$/, "");
  const pageUrl = new URL(cleanPath, `${hostname}${base}`).href;
  const schemas: object[] = [];

  schemas.push({
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: "CoordiNode",
    url: siteUrl,
    description: siteDescription,
    publisher: {
      "@type": "Organization",
      name: "sw.foundation",
      url: "https://sw.foundation",
    },
  });

  const pathSegments = cleanPath.split("/").filter(Boolean);
  if (pathSegments.length > 0) {
    const breadcrumbItems = [
      { "@type": "ListItem", position: 1, name: "Home", item: siteUrl },
    ];
    let currentPath = "";
    pathSegments.forEach((segment, index) => {
      currentPath += `/${segment}`;
      const name = segment
        .split("-")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");
      breadcrumbItems.push({
        "@type": "ListItem",
        position: index + 2,
        name: index === pathSegments.length - 1 ? (pageData.title || name) : name,
        item: index === pathSegments.length - 1 ? pageUrl : `${siteUrl}${currentPath}/`,
      });
    });
    schemas.push({
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: breadcrumbItems,
    });
  }

  const isHomePage = cleanPath === "" || cleanPath === "/";
  if (isHomePage) {
    schemas.push({
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      name: "CoordiNode",
      applicationCategory: "DeveloperApplication",
      operatingSystem: "Linux, macOS, Windows",
      url: siteUrl,
      downloadUrl: "https://github.com/structured-world/coordinode/releases",
      offers: {
        "@type": "Offer",
        price: "0",
        priceCurrency: "USD",
      },
      author: {
        "@type": "Organization",
        name: "sw.foundation",
        url: "https://sw.foundation",
      },
    });
  }

  return schemas;
}

export default defineConfig({
  title: "CoordiNode",
  titleTemplate: ":title | CoordiNode",
  lang: "en-US",
  cleanUrls: true,
  description: siteDescription,
  base,

  transformHead({ pageData }) {
    const head: HeadConfig[] = [];
    const title = pageData.title || "CoordiNode";
    const description = pageData.description || siteDescription;
    const cleanPath = pageData.relativePath.replace(/(?:index)?\.md$/, "");
    const url = new URL(cleanPath, `${hostname}${base}`).href;

    head.push(["meta", { property: "og:title", content: title }]);
    head.push(["meta", { property: "og:description", content: description }]);
    head.push(["meta", { property: "og:url", content: url }]);
    head.push(["meta", { name: "twitter:card", content: "summary_large_image" }]);
    head.push(["meta", { name: "twitter:title", content: title }]);
    head.push(["meta", { name: "twitter:description", content: description }]);
    head.push(["link", { rel: "canonical", href: url }]);

    const structuredData = generateStructuredData(pageData);
    for (const schema of structuredData) {
      head.push(["script", { type: "application/ld+json" }, JSON.stringify(schema)]);
    }

    return head;
  },

  // Ignore dead links in existing docs that reference files outside the docs/ root
  // (e.g. ROADMAP.md references ../CONTRIBUTING.md which is a repo-root file).
  ignoreDeadLinks: [/CONTRIBUTING/, /\/sdk\/python#/],

  sitemap: {
    hostname,
    transformItems: (items) =>
      items.map((item) => ({
        ...item,
        lastmod: new Date().toISOString().split("T")[0],
      })),
  },

  head: [
    ["link", { rel: "icon", type: "image/x-icon", href: `${base}favicon.ico` }],
    ["link", { rel: "icon", type: "image/png", sizes: "32x32", href: `${base}favicon-32x32.png` }],
    ["link", { rel: "icon", type: "image/png", sizes: "16x16", href: `${base}favicon-16x16.png` }],
    ["meta", { property: "og:image", content: `${hostname}/og-image.png` }],
    ["meta", { property: "og:image:width", content: "1200" }],
    ["meta", { property: "og:image:height", content: "630" }],
    ["meta", { property: "og:type", content: "website" }],
    ["meta", { property: "og:site_name", content: "CoordiNode" }],
  ],

  themeConfig: {
    logo: "/logo.png",

    outline: {
      level: [2, 3],
      label: "On this page",
    },

    nav: [
      { text: "Guide", link: "/guide/" },
      {
        text: "OpenCypher",
        items: [
          { text: "Extensions", link: "/CYPHER_EXTENSIONS" },
          { text: "Compatibility", link: "/COMPATIBILITY" },
        ],
      },
      {
        text: "SDK",
        items: [
          { text: "Python SDK", link: "/sdk/python" },
          { text: "LlamaIndex", link: "/sdk/llama-index" },
          { text: "LangChain", link: "/sdk/langchain" },
        ],
      },
      { text: "Roadmap", link: "/ROADMAP" },
      {
        text: "GitHub",
        link: "https://github.com/structured-world/coordinode",
      },
    ],

    sidebar: {
      "/guide/": [
        {
          text: "Getting Started",
          items: [
            { text: "Introduction", link: "/guide/" },
            { text: "Quick Start", link: "/QUICKSTART" },
          ],
        },
        {
          text: "Installation",
          items: [
            { text: "Docker", link: "/guide/docker" },
            { text: "Binary", link: "/guide/binary" },
            { text: "Embedded (Rust)", link: "/guide/embedded" },
          ],
        },
        {
          text: "Concepts",
          items: [
            { text: "Data Model", link: "/guide/data-model" },
            { text: "MVCC Transactions", link: "/guide/transactions" },
            { text: "Hybrid Retrieval", link: "/guide/hybrid-retrieval" },
          ],
        },
      ],
      "/sdk/": [
        {
          text: "Python SDK",
          items: [
            { text: "Overview", link: "/sdk/python" },
            { text: "LlamaIndex", link: "/sdk/llama-index" },
            { text: "LangChain", link: "/sdk/langchain" },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/structured-world/coordinode" },
    ],

    footer: {
      message: "Released under the <a href='https://github.com/structured-world/coordinode/blob/main/LICENSE'>AGPL-3.0 License</a>.",
      copyright: "Copyright © 2026 sw.foundation",
    },

    editLink: {
      pattern: "https://github.com/structured-world/coordinode/edit/main/docs/:path",
      text: "Edit this page on GitHub",
    },

    search: {
      provider: "local",
    },
  },
});
