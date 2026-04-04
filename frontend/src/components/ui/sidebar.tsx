"use client";

import { Menu, X } from "lucide-react";
import { useState } from "react";

type SidebarItem = {
  id: string;
  label: string;
};

type SidebarProps = {
  items: SidebarItem[];
  active: string;
  onSelect: (id: string) => void;
};

export function Sidebar({ items, active, onSelect }: SidebarProps) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="fixed left-4 top-4 z-50 rounded-xl border border-white/15 bg-slate-900/90 p-2 text-slate-100 backdrop-blur md:hidden"
        aria-label="Toggle sidebar"
      >
        {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </button>

      <aside
        className={`fixed inset-y-0 left-0 z-40 w-72 border-r border-white/10 bg-slate-950/90 p-5 backdrop-blur-xl transition-transform md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <p className="text-xs uppercase tracking-[0.32em] text-cyan-300">Power-Rangers</p>
        <h2 className="mt-3 text-xl font-semibold text-slate-100">Forecast Control Surface</h2>
        <p className="mt-2 text-sm text-slate-400">Electricity demand and peak prediction for Delhi power operations.</p>

        <nav className="mt-8 flex flex-col gap-2">
          {items.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                onSelect(item.id);
                setOpen(false);
              }}
              className={`rounded-lg px-3 py-2 text-left text-sm transition ${
                active === item.id
                  ? "bg-cyan-400/15 text-cyan-200"
                  : "text-slate-300 hover:bg-white/5 hover:text-slate-100"
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      {open && <button className="fixed inset-0 z-30 bg-black/50 md:hidden" onClick={() => setOpen(false)} aria-label="Close sidebar" />}
    </>
  );
}