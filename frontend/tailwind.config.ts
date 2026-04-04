import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#04060f",
        foreground: "#f8fafc",
        primary: {
          DEFAULT: "#22d3ee",
          foreground: "#020617",
        },
        secondary: {
          DEFAULT: "rgba(255, 255, 255, 0.08)",
          foreground: "#f8fafc",
        },
        muted: {
          DEFAULT: "rgba(255, 255, 255, 0.08)",
          foreground: "rgba(248, 250, 252, 0.68)",
        },
        destructive: {
          DEFAULT: "#ef4444",
          foreground: "#ffffff",
        },
        ring: "#22d3ee",
      },
    },
  },
  plugins: [],
};

export default config;
