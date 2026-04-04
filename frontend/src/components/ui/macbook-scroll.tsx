"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import { type ReactNode, useRef } from "react";
import { cn } from "@/lib/utils";

type MacbookScrollProps = {
  title: ReactNode;
  badge?: ReactNode;
  src?: string;
  showGradient?: boolean;
};

export function MacbookScroll({ title, badge, src, showGradient = true }: MacbookScrollProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] });

  const rotateX = useTransform(scrollYProgress, [0, 1], [22, 0]);
  const scale = useTransform(scrollYProgress, [0, 1], [0.92, 1.02]);

  return (
    <section ref={ref} className="relative mx-auto w-full max-w-6xl px-4 py-16 sm:px-8">
      <div className="mb-8 flex flex-col items-center gap-3 text-center">
        <h2 className="max-w-3xl text-balance text-3xl font-semibold text-slate-100 sm:text-5xl">{title}</h2>
        {badge}
      </div>

      <motion.div
        style={{ rotateX, scale }}
        className="mx-auto w-full max-w-5xl [perspective:1200px]"
      >
        <div className="rounded-[2rem] border border-slate-700/70 bg-gradient-to-b from-slate-700 to-slate-900 p-2 shadow-[0_45px_120px_rgba(2,6,23,0.75)]">
          <div className="rounded-[1.5rem] border border-slate-700/80 bg-slate-950 p-3">
            <div className="relative overflow-hidden rounded-xl border border-cyan-300/20 bg-[#090f1f]">
              {showGradient && (
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.28),transparent_38%),radial-gradient(circle_at_80%_20%,rgba(16,185,129,0.25),transparent_40%)]" />
              )}
              {src ? (
                <img src={src} alt="Forecast dashboard preview" className="relative z-10 h-[420px] w-full object-cover" />
              ) : (
                <div className="relative z-10 grid h-[420px] place-items-center p-6">
                  <div className="w-full max-w-xl rounded-2xl border border-white/10 bg-black/40 p-6 text-left text-slate-200 backdrop-blur">
                    <p className="text-xs uppercase tracking-[0.25em] text-cyan-300">Forecast Engine</p>
                    <p className="mt-3 text-2xl font-semibold">Delhi Load + Peak Demand</p>
                    <p className="mt-2 text-sm text-slate-300">15-minute interval, 48-hour horizon, probabilistic P10/P50/P90 outputs.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mx-auto h-6 w-[82%] rounded-b-[2rem] border-x border-b border-slate-700 bg-gradient-to-b from-slate-800 to-slate-950" />
      </motion.div>
    </section>
  );
}

export function BadgePill({ className }: { className?: string }) {
  return (
    <span className={cn("rounded-full border border-emerald-300/30 bg-emerald-300/10 px-3 py-1 text-xs font-medium text-emerald-200", className)}>
      Spec Aligned
    </span>
  );
}