/**
 * Dashboard page component with forecast analytics
 */

import { BackgroundBeamsDemo } from "@/components/aceternity/background-beams-demo";
import { LayoutTextFlipDemo } from "@/components/aceternity/layout-text-flip-demo";
import { MacbookScrollDemo } from "@/components/aceternity/macbook-scroll-demo";
import { SidebarDemo } from "@/components/aceternity/sidebar-demo";
import { TimelineDemo } from "@/components/aceternity/timeline-demo";
import { FunctionalFooter } from "@/components/features/functional-footer";
import { Database, ShieldCheck, Timer, Zap } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

type Metrics = { mae: number; rmse: number; mape: number };
type Peak = { peak_value: number; peak_timestamp: string };
type ForecastResponse = {
  metrics: Metrics;
  peak: Peak;
  predictions: any[];
  avg_temperature_c?: number | null;
};
type Point = {
  timestamp: string;
  prediction: number;
  p10: number;
  p50: number;
  p90: number;
  actual: number | null;
  model_name: string;
};

const STEPS_PER_DAY = 96;

const toLocalDateInput = (d: Date) => {
  const offset = d.getTimezoneOffset() * 60000;
  return new Date(d.getTime() - offset).toISOString().slice(0, 10);
};

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

export function DashboardPage() {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const [active, setActive] = useState("overview");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [peak, setPeak] = useState<Peak | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [avgTemperatureC, setAvgTemperatureC] = useState<number | null>(null);
  const [selectedDate, setSelectedDate] = useState<string>(toLocalDateInput(new Date()));
  const [temperatureDeltaC, setTemperatureDeltaC] = useState<number>(0);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

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

  const loadAnalytics = useCallback(async (forecastDate: string, temperatureDelta: number) => {
    setIsLoading(true);
    setFetchError(null);

    try {
      const response = await fetch(
        `/api/forecast?days_to_fetch=3&forecast_date=${encodeURIComponent(forecastDate)}&temperature_delta_c=${encodeURIComponent(temperatureDelta.toString())}`,
        { method: "POST" },
      );
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = (await response.json()) as ForecastResponse;

      setMetrics(data.metrics);
      setPeak(data.peak);
      setAvgTemperatureC(data.avg_temperature_c == null ? null : Number(data.avg_temperature_c));

      const parsedPoints = data.predictions.map((p: any) => {
        const p50 = Number(p.p50 ?? p.predicted_load_mw ?? 0);
        const p10 = Number(p.p10 ?? p50 - Math.max(p50 * 0.02, 25));
        const p90 = Number(p.p90 ?? p50 + Math.max(p50 * 0.02, 25));

        return {
          timestamp: p.timestamp,
          prediction: Number(p.predicted_load_mw ?? p50),
          p10,
          p50,
          p90,
          actual: p.actual_load_mw == null ? null : Number(p.actual_load_mw),
          model_name: "SeasonalTrend",
        } as Point;
      });

      setPoints(parsedPoints);
    } catch (err) {
      setFetchError("Unable to fetch forecast for the selected date.");
      setPoints([]);
      setAvgTemperatureC(null);
      console.error("Failed to fetch analytics from backend", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAnalytics(selectedDate, temperatureDeltaC);
  }, [loadAnalytics, selectedDate, temperatureDeltaC]);

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

  const chartData = useMemo(() => {
    if (points.length === 0) {
      return {
        p10Path: "",
        p50Path: "",
        p90Path: "",
        actualPath: "",
        bandPath: "",
        lastActual: null as number | null,
        width: 960,
        height: 320,
        minY: 0,
        range: 1,
      };
    }

    const width = Math.max(960, Math.round((points.length / STEPS_PER_DAY) * 960));
    const height = 320;
    const values = points.flatMap((p) => [p.p10, p.p50, p.p90, ...(p.actual == null ? [] : [p.actual])]);
    const minY = Math.min(...values);
    const maxY = Math.max(...values);
    const range = Math.max(maxY - minY, 1);

    const x = (i: number) => (i / Math.max(points.length - 1, 1)) * width;
    const y = (v: number) => height - ((v - minY) / range) * height;

    const toPath = (getter: (p: Point) => number) =>
      points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(getter(p)).toFixed(2)}`).join(" ");

    const toNullablePath = (getter: (p: Point) => number | null) => {
      let path = "";
      let inSegment = false;

      points.forEach((p, i) => {
        const value = getter(p);
        if (value == null) {
          inSegment = false;
          return;
        }

        path += `${inSegment ? "L" : "M"}${x(i).toFixed(2)},${y(value).toFixed(2)} `;
        inSegment = true;
      });

      return path.trim();
    };

    const topBand = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(p.p90).toFixed(2)}`).join(" ");
    const bottomBand = [...points]
      .reverse()
      .map((p, i) => `${i === 0 ? "L" : "L"}${x(points.length - 1 - i).toFixed(2)},${y(p.p10).toFixed(2)}`)
      .join(" ");

    const lastActualPoint = [...points].reverse().find((p) => p.actual != null);

    return {
      p10Path: toPath((p) => p.p10),
      p50Path: toPath((p) => p.p50),
      p90Path: toPath((p) => p.p90),
      actualPath: toNullablePath((p) => p.actual),
      bandPath: `${topBand} ${bottomBand} Z`,
      lastActual: lastActualPoint?.actual ?? null,
      width,
      height,
      minY,
      range,
    };
  }, [points]);

  const hoveredPoint = hoveredIndex == null ? null : points[hoveredIndex] ?? null;
  const hoveredX =
    hoveredIndex == null || points.length <= 1
      ? null
      : (hoveredIndex / Math.max(points.length - 1, 1)) * chartData.width;
  const yForValue = (v: number) => chartData.height - ((v - chartData.minY) / chartData.range) * chartData.height;
  const horizonHours = points.length ? Math.round(points.length / 4) : 24;

  const handleLogout = async () => {
    await logout();
    navigate("/login", { replace: true });
  };

  return (
    <SidebarDemo active={active} onSelect={handleSelect}>
      <main className="relative min-h-screen overflow-hidden">
        <div className="fixed right-4 top-4 z-50 sm:right-6 sm:top-6">
          <button
            type="button"
            onClick={handleLogout}
            className="rounded-md border border-red-400/35 bg-red-500/15 px-4 py-2 text-sm font-medium text-red-100 transition hover:bg-red-500/25"
          >
            Logout
          </button>
        </div>
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

            <div className="mt-5 flex flex-wrap items-end gap-3 rounded-xl border border-white/10 bg-black/25 p-3">
              <label className="text-xs text-slate-400">
                Forecast date
                <input
                  type="date"
                  value={selectedDate}
                  max={toLocalDateInput(new Date())}
                  onChange={(e) => setSelectedDate(e.target.value)}
                  className="mt-1 block rounded-md border border-white/15 bg-slate-900 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-300"
                />
              </label>
              <label className="min-w-[220px] flex-1 text-xs text-slate-400">
                Temperature adjustment ({temperatureDeltaC > 0 ? `+${temperatureDeltaC}` : temperatureDeltaC} C)
                <input
                  type="range"
                  min={-5}
                  max={5}
                  step={1}
                  value={temperatureDeltaC}
                  onChange={(e) => setTemperatureDeltaC(Number(e.target.value))}
                  className="mt-2 w-full accent-cyan-300"
                />
                <div className="mt-1 flex justify-between text-[10px] text-slate-500">
                  <span>-5 C cooler case</span>
                  <span>0 C base</span>
                  <span>+5 C hotter case</span>
                </div>
              </label>
              <button
                type="button"
                onClick={() => void loadAnalytics(selectedDate, temperatureDeltaC)}
                disabled={isLoading}
                className="rounded-md border border-cyan-300/25 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-200 transition hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isLoading ? "Loading..." : "Refresh"}
              </button>
              <p className="text-xs text-slate-400">Shows actuals on the forecast horizon when source data exists and supports temperature what-if scenarios.</p>
            </div>

            {fetchError && <p className="mt-3 text-sm text-rose-300">{fetchError}</p>}

            <div className="mt-4 flex flex-wrap gap-2 text-xs">
              <span className="rounded-full border border-cyan-300/25 bg-cyan-500/10 px-3 py-1 text-cyan-200">success</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-slate-300">{horizonHours}h horizon</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-slate-300">
                Avg temp {avgTemperatureC == null ? "--" : `${avgTemperatureC.toFixed(1)} C`}
              </span>
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
              <div className="overflow-x-auto">
                <svg
                  viewBox={`0 0 ${chartData.width} ${chartData.height}`}
                  preserveAspectRatio="none"
                  className="h-[300px]"
                  style={{ width: `${chartData.width}px`, minWidth: "960px" }}
                onMouseMove={(e) => {
                  if (points.length === 0) {
                    return;
                  }
                  const rect = e.currentTarget.getBoundingClientRect();
                  if (rect.width <= 0) {
                    return;
                  }
                  const relX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                  const index = Math.round((relX / rect.width) * Math.max(points.length - 1, 1));
                  setHoveredIndex(index);
                }}
                onMouseLeave={() => setHoveredIndex(null)}
                >
                <defs>
                  <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(34,211,238,0.35)" />
                    <stop offset="100%" stopColor="rgba(34,211,238,0.05)" />
                  </linearGradient>
                </defs>
                {[64, 128, 192, 256].map((y) => (
                  <line key={y} x1="0" y1={y} x2={chartData.width} y2={y} stroke="rgba(148,163,184,0.15)" strokeDasharray="4 8" />
                ))}
                {chartData.bandPath && <path d={chartData.bandPath} fill="url(#bandGradient)" />}
                {chartData.p10Path && <path d={chartData.p10Path} fill="none" stroke="#34d399" strokeWidth="2" strokeDasharray="3 8" />}
                {chartData.p50Path && <path d={chartData.p50Path} fill="none" stroke="#facc15" strokeWidth="3" strokeDasharray="7 6" />}
                {chartData.p90Path && <path d={chartData.p90Path} fill="none" stroke="#a855f7" strokeWidth="2" strokeDasharray="3 8" />}
                {chartData.actualPath && <path d={chartData.actualPath} fill="none" stroke="#22d3ee" strokeWidth="3" />}

                {hoveredPoint && hoveredX != null && (
                  <>
                    <line
                      x1={hoveredX}
                      y1={0}
                      x2={hoveredX}
                      y2={chartData.height}
                      stroke="rgba(148,163,184,0.6)"
                      strokeDasharray="4 6"
                    />
                    <circle cx={hoveredX} cy={yForValue(hoveredPoint.p50)} r={4.5} fill="#facc15" />
                    {hoveredPoint.actual != null && <circle cx={hoveredX} cy={yForValue(hoveredPoint.actual)} r={4.5} fill="#22d3ee" />}
                  </>
                )}

                {!chartData.actualPath && (
                  <text x="20" y="40" fill="#94a3b8" fontSize="14">
                    No actual values are available for this forecast horizon yet.
                  </text>
                )}
                </svg>
              </div>

              {hoveredPoint && (
                <div className="mt-3 rounded-lg border border-white/10 bg-black/35 px-3 py-2 text-xs text-slate-200">
                  <p className="font-semibold text-slate-100">{new Date(hoveredPoint.timestamp).toLocaleString()}</p>
                  <p className="mt-1">Actual: {hoveredPoint.actual == null ? "N/A" : hoveredPoint.actual.toFixed(2)} MW</p>
                  <p>P50: {hoveredPoint.p50.toFixed(2)} MW</p>
                  <p>P10/P90: {hoveredPoint.p10.toFixed(2)} / {hoveredPoint.p90.toFixed(2)} MW</p>
                </div>
              )}

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

export default DashboardPage;
