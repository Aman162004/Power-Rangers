"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const beams = [
  { left: "10%", delay: 0, duration: 8 },
  { left: "28%", delay: 0.7, duration: 9 },
  { left: "44%", delay: 1.4, duration: 7.5 },
  { left: "63%", delay: 0.9, duration: 8.8 },
  { left: "82%", delay: 1.8, duration: 9.2 },
];

export function BackgroundBeams({ className }: { className?: string }) {
  return (
    <div className={cn("pointer-events-none absolute inset-0 overflow-hidden", className)} aria-hidden>
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(16,185,129,0.2),transparent_32%),radial-gradient(circle_at_80%_30%,rgba(56,189,248,0.18),transparent_30%)]" />
      {beams.map((beam) => (
        <motion.div
          key={beam.left}
          className="absolute top-[-20%] h-[170%] w-[2px] bg-gradient-to-b from-cyan-300/0 via-cyan-300/60 to-emerald-300/0 blur-[1px]"
          style={{ left: beam.left }}
          initial={{ y: -120, opacity: 0.15 }}
          animate={{ y: 420, opacity: [0.15, 0.5, 0.15] }}
          transition={{
            repeat: Number.POSITIVE_INFINITY,
            ease: "linear",
            duration: beam.duration,
            delay: beam.delay,
          }}
        />
      ))}
    </div>
  );
}