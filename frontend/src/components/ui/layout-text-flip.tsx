"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

type LayoutTextFlipProps = {
  text: string;
  words: string[];
};

export function LayoutTextFlip({ text, words }: LayoutTextFlipProps) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (words.length < 2) {
      return;
    }

    const timer = window.setInterval(() => {
      setIndex((prev) => (prev + 1) % words.length);
    }, 2300);

    return () => window.clearInterval(timer);
  }, [words]);

  return (
    <div className="flex flex-wrap items-center justify-center gap-2 text-balance text-2xl font-semibold tracking-tight text-slate-100 sm:text-4xl">
      <span>{text}</span>
      <span className="relative inline-flex min-w-[220px] justify-center rounded-xl border border-cyan-200/20 bg-white/5 px-4 py-2 text-cyan-300">
        <AnimatePresence mode="wait">
          <motion.span
            key={words[index]}
            initial={{ y: 16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -16, opacity: 0 }}
            transition={{ duration: 0.28 }}
          >
            {words[index]}
          </motion.span>
        </AnimatePresence>
      </span>
    </div>
  );
}