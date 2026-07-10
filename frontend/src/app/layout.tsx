import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "PlumeTrace Command Center",
  description: "Real-time municipal environmental forensics dashboard"
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
