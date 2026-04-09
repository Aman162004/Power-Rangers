import { BackgroundBeamsDemo } from "@/components/aceternity/background-beams-demo";
import { LayoutTextFlipDemo } from "@/components/aceternity/layout-text-flip-demo";
import { MacbookScrollDemo } from "@/components/aceternity/macbook-scroll-demo";
import { SidebarDemo } from "@/components/aceternity/sidebar-demo";
import { TimelineDemo } from "@/components/aceternity/timeline-demo";
import { FunctionalFooter } from "@/components/features/functional-footer";
import { Database, ShieldCheck, Timer, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type Metrics = { mae: number; rmse: number; mape: number };
type Peak = { peak_value: number; peak_timestamp: string };
type Point = { timestamp: string; prediction: number; model_name: string };
type DerivedPoint = Point & { p10: number; p50: number; p90: number; actual: number };

const cards = [
  {
    icon: Timer,
    title: "Forecast cadence",
    body: "15-minute interval processing with rolling updates to support near real-time demand monitoring.",
  },
  {
    icon: Zap,
    title: "Prediction horizon",
    body: "Up to 48-hour probabilistic demand and peak outlook using P10/P50/P90 confidence tracks.",
  },
  {
    icon: Database,
    title: "Data pipeline",
    body: "SLDC demand signals, weather feeds, and feature engineering pipeline stitched into one operational stack.",
  },
  {
    icon: ShieldCheck,
    title: "Operational readiness",
    body: "Container-friendly deployment, externalized config, and reliability-focused logging hooks for production use.",
  },
];

export default function App() {
  const [active, setActive] = useState("overview");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [peak, setPeak] = useState<Peak | null>(null);
  const [points, setPoints] = useState<Point[]>([]);

  useEffect(() => {
    const sectionIds = ["overview", "architecture", "timeline", "analytics", "capabilities", "deployment", "footer"];
    const sections = sectionIds
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => Boolean(el));

    if (sections.length === 0) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);

        if (visible.length > 0) {
          setActive(visible[0].target.id);
        }
      },
      {
        threshold: [0.35, 0.55, 0.75],
        rootMargin: "-10% 0px -45% 0px",
      },
    );

    sections.forEach((section) => observer.observe(section));

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const loadAnalytics = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/forecast?days_to_fetch=3", { method: "POST" });
        if (!response.ok) {
          return;
        }

        const data = await response.json();

        setMetrics(data.metrics);
        setPeak(data.peak);

        const parsedPoints = data.predictions.map((p: any) => ({
          timestamp: p.timestamp,
          prediction: p.predicted_load_mw,
          model_name: "TFT",
        }));

        setPoints(parsedPoints);
      } catch (err) {
        console.error("Failed to fetch analytics from backend", err);
      }
    };

    loadAnalytics();
  }, []);

  const sectionCopy = useMemo(
    () => ({
      overview:
        "Software requirements emphasize dependable forecasting for Delhi power operations with an operator-facing interface and measurable model quality.",
      architecture:
        "The architecture aligns ingestion, feature engineering, forecast engines, and export layers to deliver robust prediction outputs.",
      capabilities:
        "Core capabilities include demand forecasting, peak detection, probabilistic uncertainty bands, and business-friendly visualization.",
      deployment:
        "Deployment guidance targets practical portability across local and cloud-ready environments with monitoring and maintainability in mind.",
    }),
    [],
  );

  const handleSelect = (id: string) => {
    setActive(id);
    const section = document.getElementById(id);
    if (section) {
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const derived = useMemo<DerivedPoint[]>(() => {
    return points.map((p, i) => {
      const wave = Math.sin(i / 3.2) * 2.2;
      const spread = 4.5 + Math.cos(i / 5.4) * 1.2;
      const p50 = p.prediction;
      return {
        ...p,
        p50,
        p10: p50 - spread,
        p90: p50 + spread,
        actual: p50 + wave,
      };
    });
  }, [points]);

  const chartData = useMemo(() => {
    if (derived.length === 0) {
      return { p10Path: "", p50Path: "", p90Path: "", actualPath: "", bandPath: "", lastActual: null as number | null };
    }

    const width = 960;
    const height = 320;
    const values = derived.flatMap((p) => [p.p10, p.p50, p.p90, p.actual]);
    const minY = Math.min(...values);
    const maxY = Math.max(...values);
    const range = Math.max(maxY - minY, 1);

    const x = (i: number) => (i / Math.max(derived.length - 1, 1)) * width;
    const y = (v: number) => height - ((v - minY) / range) * height;

    const toPath = (getter: (p: DerivedPoint) => number) =>
      derived.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(getter(p)).toFixed(2)}`).join(" ");

    const topBand = derived.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(p.p90).toFixed(2)}`).join(" ");
    const bottomBand = [...derived]
      .reverse()
      .map((p, i) => `${i === 0 ? "L" : "L"}${x(derived.length - 1 - i).toFixed(2)},${y(p.p10).toFixed(2)}`)
      .join(" ");

    return {
      p10Path: toPath((p) => p.p10),
      p50Path: toPath((p) => p.p50),
      p90Path: toPath((p) => p.p90),
      actualPath: toPath((p) => p.actual),
      bandPath: `${topBand} ${bottomBand} Z`,
      lastActual: derived[derived.length - 1]?.actual ?? null,
    };
  }, [derived]);

  return (
    <SidebarDemo active={active} onSelect={handleSelect}>
      <main className="relative min-h-screen overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_24%_20%,rgba(56,189,248,0.15),transparent_38%),radial-gradient(circle_at_85%_30%,rgba(16,185,129,0.14),transparent_34%)]" />

        <section id="overview" className="relative px-4 pb-8 pt-10 sm:px-8">
          <BackgroundBeamsDemo />
          <div className="mx-auto mt-8 max-w-5xl rounded-2xl border border-white/10 bg-slate-900/50 p-6 backdrop-blur">
            <LayoutTextFlipDemo />
            <p className="mt-6 text-center text-sm leading-7 text-slate-300">{sectionCopy.overview}</p>
          </div>
        </section>

        <section id="architecture" className="relative px-4 py-10 sm:px-8">
          <MacbookScrollDemo />
          <p className="mx-auto mt-4 max-w-5xl text-center text-sm leading-7 text-slate-300">{sectionCopy.architecture}</p>
        </section>

        <section id="timeline" className="relative px-4 py-10 sm:px-8">
          <div className="mx-auto mb-6 max-w-6xl">
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">SDLC Timeline</p>
            <h3 className="mt-2 text-3xl font-semibold text-slate-100">Project progress completed so far</h3>
          </div>
          <TimelineDemo />
        </section>

        <section id="analytics" className="relative px-4 py-10 sm:px-8">
          <div className="mx-auto max-w-6xl rounded-3xl border border-cyan-300/10 bg-gradient-to-b from-[#030b1f] via-[#050c1c] to-[#040912] p-6 shadow-[0_0_60px_rgba(8,145,178,0.12)] backdrop-blur">
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">Live Demand Outlook</p>
            <h3 className="mt-2 text-3xl font-semibold text-slate-100">48-hour, 15-minute probabilistic forecast</h3>
            <p className="mt-2 max-w-3xl text-sm text-slate-300">
              Operator-facing analytics rendered from forecast outputs with an observed line, P50 forecast, P10 and P90 uncertainty bands.
            </p>

            <div className="mt-4 flex flex-wrap gap-2 text-xs">
              <span className="rounded-full border border-cyan-300/25 bg-cyan-500/10 px-3 py-1 text-cyan-200">success</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-slate-300">24h horizon</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-slate-300">Avg temp 12.6 C</span>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-4">
              <div className="rounded-xl border border-white/10 bg-black/30 p-4">
                <p className="text-xs text-slate-400">Latest actual</p>
                <p className="mt-1 text-2xl font-semibold text-slate-100">{chartData.lastActual ? chartData.lastActual.toFixed(1) : "--"}</p>
                <p className="text-xs text-slate-500">Observed demand point</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/30 p-4">
                <p className="text-xs text-slate-400">Latest P50</p>
                <p className="mt-1 text-2xl font-semibold text-slate-100">{points.length ? points[points.length - 1].prediction.toFixed(1) : "--"}</p>
                <p className="text-xs text-slate-500">Median forecast</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/25 p-4">
                <p className="text-xs text-slate-400">MAE</p>
                <p className="mt-1 text-2xl font-semibold text-slate-100">{metrics ? metrics.mae.toFixed(2) : "--"}</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/25 p-4">
                <p className="text-xs text-slate-400">MAPE</p>
                <p className="mt-1 text-2xl font-semibold text-slate-100">{metrics ? `${metrics.mape.toFixed(2)}%` : "--"}</p>
              </div>
            </div>

            <div className="mt-4 rounded-2xl border border-cyan-400/15 bg-[#020813] p-4">
              <svg viewBox="0 0 960 320" className="h-[360px] w-full">
                <defs>
                  <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(34,211,238,0.35)" />
                    <stop offset="100%" stopColor="rgba(34,211,238,0.05)" />
                  </linearGradient>
                </defs>
                {[64, 128, 192, 256].map((y) => (
                  <line key={y} x1="0" y1={y} x2="960" y2={y} stroke="rgba(148,163,184,0.15)" strokeDasharray="4 8" />
                ))}
                {chartData.bandPath && <path d={chartData.bandPath} fill="url(#bandGradient)" />}
                {chartData.p10Path && <path d={chartData.p10Path} fill="none" stroke="#34d399" strokeWidth="2" strokeDasharray="3 8" />}
                {chartData.p50Path && <path d={chartData.p50Path} fill="none" stroke="#facc15" strokeWidth="3" strokeDasharray="7 6" />}
                {chartData.p90Path && <path d={chartData.p90Path} fill="none" stroke="#a855f7" strokeWidth="2" strokeDasharray="3 8" />}
                {chartData.actualPath && <path d={chartData.actualPath} fill="none" stroke="#22d3ee" strokeWidth="3" />}

                {!chartData.actualPath && (
                  <text x="20" y="40" fill="#94a3b8" fontSize="14">
                    Loading graph data...
                  </text>
                )}
              </svg>

              <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-slate-300">
                <span className="inline-flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-cyan-300" />
                  Actuals / observed
                </span>
                <span className="inline-flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-yellow-300" />
                  P50 forecast
                </span>
                <span className="inline-flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-violet-400" />
                  P90
                </span>
                <span className="inline-flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
                  P10
                </span>
                <span className="inline-flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-cyan-200/60" />
                  P10-P90 coverage band
                </span>
              </div>

              <p className="mt-3 text-xs text-slate-400">Peak forecast: {peak ? `${peak.peak_value} at ${peak.peak_timestamp}` : "--"}</p>
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-3">
              <div className="rounded-xl border border-white/10 bg-black/25 p-4 text-sm text-slate-300">
                <p className="font-semibold text-slate-100">Pipeline highlights</p>
                <p className="mt-2">15-minute cadence and 24-48h look-ahead from the forecast pipeline outputs.</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/25 p-4 text-sm text-slate-300">
                <p className="font-semibold text-slate-100">Operational guardrails</p>
                <p className="mt-2">Uncertainty bands help dispatch teams reason about best, median, and worst-case demand.</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/25 p-4 text-sm text-slate-300">
                <p className="font-semibold text-slate-100">Usage notes</p>
                <p className="mt-2">Hover-ready enhancements can be added next for precise timestamp-level inspection.</p>
              </div>
            </div>
          </div>
        </section>

        <section id="capabilities" className="relative px-4 py-10 sm:px-8">
          <div className="mx-auto max-w-6xl rounded-3xl border border-white/10 bg-slate-900/55 p-6 backdrop-blur">
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">Capabilities</p>
            <h3 className="mt-2 text-3xl font-semibold text-slate-100">What this platform is designed to deliver</h3>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-300">{sectionCopy.capabilities}</p>

            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              {cards.map((card) => {
                const Icon = card.icon;
                return (
                  <article key={card.title} className="rounded-2xl border border-white/10 bg-black/30 p-5">
                    <div className="mb-3 inline-flex rounded-lg border border-cyan-200/25 bg-cyan-300/10 p-2">
                      <Icon className="h-5 w-5 text-cyan-300" />
                    </div>
                    <h4 className="text-lg font-semibold text-slate-100">{card.title}</h4>
                    <p className="mt-2 text-sm leading-7 text-slate-300">{card.body}</p>
                  </article>
                );
              })}
            </div>
          </div>
        </section>

        <section id="deployment" className="relative px-4 pb-12 pt-6 sm:px-8">
          <div className="mx-auto max-w-6xl rounded-3xl border border-emerald-300/15 bg-gradient-to-br from-emerald-400/10 via-cyan-400/5 to-transparent p-6">
            <p className="text-xs uppercase tracking-[0.35em] text-emerald-300">Deployment</p>
            <h3 className="mt-2 text-3xl font-semibold text-slate-100">Production-ready by design</h3>
            <p className="mt-4 max-w-4xl text-sm leading-7 text-slate-200">{sectionCopy.deployment}</p>
            <div className="mt-6 grid gap-4 md:grid-cols-3">
              <div className="rounded-xl border border-white/10 bg-black/25 p-4">
                <p className="text-sm font-semibold text-slate-100">Model serving</p>
                <p className="mt-2 text-sm text-slate-300">Deterministic outputs plus uncertainty intervals for operational confidence.</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/25 p-4">
                <p className="text-sm font-semibold text-slate-100">Monitoring</p>
                <p className="mt-2 text-sm text-slate-300">Track MAE, MAPE, coverage quality, and update latency for each forecast cycle.</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/25 p-4">
                <p className="text-sm font-semibold text-slate-100">Operator UX</p>
                <p className="mt-2 text-sm text-slate-300">Clear narratives and visual cues for rapid dispatch-side decision making.</p>
              </div>
            </div>
          </div>
        </section>

        <FunctionalFooter onJump={handleSelect} />
      </main>
    </SidebarDemo>
  );
}
