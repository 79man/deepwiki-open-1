"use client";
import React, { useState } from "react";
import Markdown from "./Markdown";
import { WikiPage } from "@/types/wiki/wikipage";

const IterationTabs: React.FC<{ page: WikiPage }> = ({ page }) => {
  const [activeTab, setActiveTab] = useState(0);

  if (!page.iterations || page.iterations.length <= 1) {
    return <Markdown content={page.content} />;
  }

  return (
    <div className="w-full">
      {/* Tab buttons - one per iteration only */}
      <div className="flex border-b border-[var(--border-color)] mb-4 overflow-x-auto">
        {page.iterations.map((iter, idx) => (
          <button
            key={idx}
            onClick={() => setActiveTab(idx)}
            className={`px-4 py-2 text-sm font-medium transition-colors whitespace-nowrap flex flex-col items-start ${
              activeTab === idx
                ? "text-[var(--accent-primary)] border-b-2 border-[var(--accent-primary)]"
                : "text-[var(--foreground)]/60 hover:text-[var(--foreground)] hover:bg-[var(--background)]/50"
            }`}
          >
            <span>Iteration {iter.iteration}</span>
            {iter.model && (
              <span className="text-xs opacity-60">{iter.model}</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content - show selected iteration */}
      <div className="mt-4">
        <Markdown content={page.iterations[activeTab].content} />
      </div>
    </div>
  );
};

export default IterationTabs;
