"use client";

import { Sidebar } from "@/components/ui/sidebar";
import { type ReactNode } from "react";

type SidebarDemoProps = {
  active: string;
  onSelect: (id: string) => void;
  children: ReactNode;
};

const items = [
  { id: "overview", label: "Overview" },
  { id: "architecture", label: "Architecture" },
  { id: "timeline", label: "SDLC Timeline" },
  { id: "analytics", label: "Analytics" },
  { id: "capabilities", label: "Capabilities" },
  { id: "deployment", label: "Deployment" },
];

export function SidebarDemo({ active, onSelect, children }: SidebarDemoProps) {
  return (
    <div className="relative min-h-screen bg-[#020617] text-slate-100">
      <Sidebar items={items} active={active} onSelect={onSelect} />
      <div className="md:pl-72">{children}</div>
    </div>
  );
}