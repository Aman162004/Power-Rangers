"use client";

import { BackgroundBeams } from "@/components/ui/background-beams";

export function BackgroundBeamsDemo() {
  return (
    <div className="relative flex h-[34rem] w-full flex-col items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-neutral-950 antialiased">
      <div className="mx-auto max-w-2xl p-6">
        <h1 className="relative z-10 bg-gradient-to-b from-neutral-200 to-neutral-600 bg-clip-text text-center font-sans text-3xl font-bold text-transparent md:text-6xl">
          Delhi Grid Forecast Portal
        </h1>
        <p className="relative z-10 mx-auto my-4 max-w-xl text-center text-sm text-neutral-400">
          Probabilistic electricity demand and peak prediction platform for SLDC operations. Designed for 15-minute ingestion,
          48-hour forecast horizon, and transparent confidence bands.
        </p>
        <p className="relative z-10 mt-6 text-center text-sm text-cyan-200">Navigate all sections directly from the sidebar.</p>
      </div>
      <BackgroundBeams />
    </div>
  );
}