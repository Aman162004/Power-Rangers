import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Power-Rangers · Delhi Grid",
  description: "Delhi load and peak forecasts UI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
