"use client";

import { LayoutTextFlip } from "@/components/ui/layout-text-flip";
import { motion } from "framer-motion";

export function LayoutTextFlipDemo() {
  return (
    <div>
      <motion.div className="relative mx-4 my-4 flex flex-col items-center justify-center gap-4 text-center sm:mx-0 sm:mb-0 sm:flex-row">
        <LayoutTextFlip
          text="Welcome to"
          words={["Power-Rangers", "Delhi SLDC Operations", "Demand Intelligence", "Peak Readiness"]}
        />
      </motion.div>
      <p className="mt-4 text-center text-base text-neutral-400">
        A unified interface for ingesting signals, forecasting load, and publishing peak alerts.
      </p>
    </div>
  );
}