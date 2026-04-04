import { Timeline } from "@/components/ui/timeline";

export function TimelineDemo() {
  const data = [
    {
      title: "Phase 1",
      content: (
        <div>
          <p className="mb-4 text-sm text-slate-200">Requirements and planning completed for Delhi load and peak prediction workflows.</p>
          <div className="grid gap-2 sm:grid-cols-2">
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">Scope definition and feasibility</p>
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">Data source mapping (SLDC + weather)</p>
          </div>
        </div>
      ),
    },
    {
      title: "Phase 2",
      content: (
        <div>
          <p className="mb-4 text-sm text-slate-200">Data engineering and model pipeline implementation delivered.</p>
          <div className="grid gap-2 sm:grid-cols-2">
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">Feature engineering and training pipeline</p>
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">Forecast, probabilistic, and peak modules</p>
          </div>
        </div>
      ),
    },
    {
      title: "Phase 3",
      content: (
        <div>
          <p className="mb-4 text-sm text-slate-200">Evaluation and analytics outputs generated for operator insights.</p>
          <div className="grid gap-2 sm:grid-cols-2">
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">metrics.json and peak.json export</p>
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">predictions.csv for charting and dashboards</p>
          </div>
        </div>
      ),
    },
    {
      title: "Phase 4",
      content: (
        <div>
          <p className="mb-4 text-sm text-slate-200">Frontend modernization and operator UX completed with interactive sections.</p>
          <div className="grid gap-2 sm:grid-cols-2">
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">Aceternity-inspired sections and navigation</p>
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">Timeline, analytics, and functional footer</p>
          </div>
        </div>
      ),
    },
    {
      title: "Phase 5",
      content: (
        <div>
          <p className="mb-4 text-sm text-slate-200">Secure backend access added for operator-only analytics consumption.</p>
          <div className="grid gap-2 sm:grid-cols-2">
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">JWT auth with operator role guard</p>
            <p className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">Protected /analytics endpoints</p>
          </div>
        </div>
      ),
    },
  ];

  return (
    <div className="relative w-full overflow-clip rounded-3xl border border-white/10 bg-slate-900/45 p-6">
      <Timeline data={data} />
    </div>
  );
}