import { type ReactNode } from "react";

type TimelineItem = {
  title: string;
  content: ReactNode;
};

export function Timeline({ data }: { data: TimelineItem[] }) {
  return (
    <div className="relative mx-auto max-w-6xl">
      <div className="absolute left-[23px] top-2 h-[calc(100%-8px)] w-px bg-gradient-to-b from-cyan-300/60 via-cyan-300/20 to-transparent" />
      <div className="space-y-10">
        {data.map((item) => (
          <article key={item.title} className="relative pl-14">
            <span className="absolute left-0 top-2 inline-flex h-12 w-12 items-center justify-center rounded-full border border-cyan-300/30 bg-slate-900 text-xs font-semibold text-cyan-200">
              {item.title}
            </span>
            <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-5 backdrop-blur">
              {item.content}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}