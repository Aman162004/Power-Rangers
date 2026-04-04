import { BadgePill, MacbookScroll } from "@/components/ui/macbook-scroll";

export function MacbookScrollDemo() {
  return (
    <div className="w-full overflow-hidden bg-[#030712]">
      <MacbookScroll
        title={
          <span>
            Forecasting UI built for real operations. <br /> Built with React + Tailwind.
          </span>
        }
        badge={<BadgePill className="-rotate-6" />}
        showGradient
      />
    </div>
  );
}