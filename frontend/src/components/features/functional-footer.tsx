type FunctionalFooterProps = {
  onJump: (id: string) => void;
};

export function FunctionalFooter({ onJump }: FunctionalFooterProps) {
  return (
    <footer id="footer" className="mx-4 mb-8 mt-16 rounded-3xl border border-white/10 bg-black/35 p-8 sm:mx-8">
      <div className="grid gap-8 md:grid-cols-4">
        <div>
          <p className="text-lg font-semibold text-slate-100">Power-Rangers</p>
          <p className="mt-2 text-sm text-slate-400">Delhi electricity demand and peak forecasting workspace for operator decisions.</p>
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-slate-200">Navigate</p>
          <div className="space-y-2 text-sm text-slate-300">
            <button onClick={() => onJump("overview")} className="block hover:text-cyan-300">Overview</button>
            <button onClick={() => onJump("timeline")} className="block hover:text-cyan-300">SDLC Timeline</button>
            <button onClick={() => onJump("capabilities")} className="block hover:text-cyan-300">Capabilities</button>
            <button onClick={() => onJump("deployment")} className="block hover:text-cyan-300">Deployment</button>
          </div>
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-slate-200">Resources</p>
          <div className="space-y-2 text-sm text-slate-300">
            <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer" className="block hover:text-cyan-300">API Docs</a>
            <a href="mailto:ops@powerrangers.local" className="block hover:text-cyan-300">Support</a>
          </div>
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-slate-200">Status</p>
          <p className="rounded-lg border border-emerald-300/20 bg-emerald-300/10 px-3 py-2 text-sm text-emerald-200">System healthy and ready for operator access.</p>
        </div>
      </div>

      <div className="mt-8 border-t border-white/10 pt-4 text-xs text-slate-500">Copyright 2026 Power-Rangers</div>
    </footer>
  );
}