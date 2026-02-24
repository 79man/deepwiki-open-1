import React, { useState } from "react";
import { PromptLogEntry, usePromptLog } from "@/contexts/PromptLogContext";
import Markdown from "@/components/Markdown";

const LOCAL_KEY = "promptLogEntries";

const saveEntriesToLocal = (entries: PromptLogEntry[]) => {
  localStorage.setItem(LOCAL_KEY, JSON.stringify(entries));
};

const loadEntriesFromLocal = (): PromptLogEntry[] => {
  return JSON.parse(localStorage.getItem(LOCAL_KEY) || "[]");
};

const clearEntriesFromLocal = () => {
  localStorage.removeItem(LOCAL_KEY);
};

const PromptLogFloatingPanel: React.FC = () => {
  const [panelOpen, setPanelOpen] = useState(false);
  const [sourceFilter, setSourceFilter] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const { promptLog, setPromptLog } = usePromptLog();

  const [showDownloadType, setShowDownloadType] = useState(false);
  const [showLoadType, setShowLoadType] = useState(false);

  const [isMaximized, setIsMaximized] = useState(false);

  const [markdownMode, setMarkdownMode] = useState<Record<number, boolean>>({});
  const toggleMode = (idx: number) => {
    setMarkdownMode((m) => ({ ...m, [idx]: !m[idx] }));
  };

  const [collapsed, setCollapsed] = useState<Record<number, boolean>>({});
  const toggleCollapsed = (idx: number) => {
    setCollapsed((c) => ({ ...c, [idx]: !c[idx] }));
  };

  const [toast, setToast] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 1800);
  };

  const [expandedContent, setExpandedContent] = useState<
    Record<number, { prompt: boolean; response: boolean }>
  >({});

  const toggleContentExpansion = (idx: number, type: "prompt" | "response") => {
    setExpandedContent((prev) => ({
      ...prev,
      [idx]: {
        ...prev[idx],
        [type]: !prev[idx]?.[type],
      },
    }));
  };
  const MAX_PREVIEW_LENGTH = 500;

  const formatLogEntryClipboard = (entry: PromptLogEntry) => {
  const lines = [
      `### Prompt Log Entry`,
      `* **Timestamp:** ${new Date(entry.timestamp).toLocaleString()}`,
      entry.source ? `* **Source:** ${entry.source}` : "",
      entry.model ? `* **Model:** ${entry.model}` : "",
      typeof entry.timeTaken === "number"
        ? `* **Time taken:** ${entry.timeTaken.toFixed(2)}s`
        : "",
      "",
      "",
      "**Prompt:**",
      "```",
      entry.prompt,
      "```",
      "",
      "**Response:**",
      entry.response.trim().startsWith("```")
        ? entry.response
        : "```\n" + entry.response + "\n```",
    ]
      .filter(Boolean)
      // Append retrieved docs if present
  if (entry.retrieval?.docs?.length) {
    
    lines.push("", "**Retrieved Docs:**");
    if (entry.retrieval.rag_query) {
      lines.push(`* Query: ${entry.retrieval.rag_query}`);
    }
    entry.retrieval.docs.forEach((doc) => {
      lines.push(`- **${doc.file_path}** (Score: ${doc.score})`);
      if (doc.text) lines.push(doc.text);
    });
  }
  return lines.join("\n");
  };

  const handleCopyEntry = (entry: PromptLogEntry) => {
    const content = formatLogEntryClipboard(entry);
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard
        .writeText(content)
        .then(() => showToast("Copied entry!"))
        .catch(() => {
          fallbackCopyTextToClipboard(content);
          showToast("Copied entry!");
        });
    } else {
      fallbackCopyTextToClipboard(content);
      showToast("Copied entry!");
    }
  };

  function fallbackCopyTextToClipboard(text: string) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand("copy");
    } catch (err) {
      // Optionally handle failed copy
    }
    document.body.removeChild(textArea);
  }

  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const handleSave = () => {
    saveEntriesToLocal(promptLog);
    showToast(`Saved ${promptLog.length} entries to local storage`);
  };

  const handleLoad = () => {
    const entriesFromLocalStore = loadEntriesFromLocal();
    setPromptLog(entriesFromLocalStore);
    showToast(
      `Loaded ${entriesFromLocalStore.length} entries from local storage`
    );
  };

  const handleClearSession = () => {
    setPromptLog([]); // clears only visible/current session logs
    showToast("Session logs cleared");
  };

  const handleDownloadMarkdown = () => {
    const content = promptLog // or localLog, whatever list you show
      .map((entry) => formatLogEntryClipboard(entry))
      .join("\n\n"); // Blank line between entries
    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `prompt_log_${new Date()
      .toISOString()
      .slice(0, 16)
      .replace(/[:-]/g, "")}.md`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
    showToast("Markdown downloaded!");
  };

  // Filter logs by source and model substring (case-insensitive)
  const filteredLog = promptLog.filter(
    (entry) =>
      (sourceFilter.trim() === "" ||
        (entry.source &&
          entry.source.toLowerCase().includes(sourceFilter.toLowerCase()))) &&
      (modelFilter.trim() === "" ||
        (entry.model &&
          entry.model.toLowerCase().includes(modelFilter.toLowerCase())))
  );

  return (
    <>
      {!panelOpen && (
        <button
          className="fixed bottom-24 right-6 z-50 bg-[var(--accent-primary)] text-white rounded-full shadow-lg w-14 h-14 flex items-center justify-center text-3xl transition duration-150 hover:scale-105 active:scale-95"
          title="Show Prompt Log"
          onClick={() => setPanelOpen(true)}
          aria-label="Show Prompt Log"
        >
          📋
        </button>
      )}

      <div
        className={`fixed z-50 top-0 right-0 h-full bg-white dark:bg-[var(--background)] shadow-2xl border-l border-[var(--border-color)] transition-transform duration-300
          ${panelOpen ? "translate-x-0" : "translate-x-full"}
          ${isMaximized ? "w-screen left-0 right-0" : "w-[420px]"}
        `}
        style={
          isMaximized
            ? { width: "100vw", maxWidth: "100vw" }
            : { maxWidth: "98vw" }
        }
      >
        <div className="flex items-center justify-between border-b p-4">
          <span className="font-bold text-lg">
            Prompt Session Log ({filteredLog.length})
          </span>
          <div className="flex items-center gap-2">
            {!isMaximized && (
              <button
                className="text-2xl hover:text-blue-500"
                onClick={() => setIsMaximized(true)}
                title="Maximize prompt log panel"
                aria-label="Maximize"
              >
                🗖
              </button>
            )}
            {isMaximized && (
              <button
                className="text-2xl hover:text-blue-500"
                onClick={() => setIsMaximized(false)}
                title="Restore prompt log panel"
                aria-label="Restore"
              >
                🗗
              </button>
            )}
            <button
              className="text-2xl hover:text-[var(--danger)]"
              onClick={() => setPanelOpen(false)}
              aria-label="Close Log Panel"
            >
              ×
            </button>
          </div>
        </div>
        {/* Localstorage tool bar*/}
        <div className="flex items-center gap-3 p-2 justify-end bg-[var(--background)] text-[var(--foreground)] border border-[var(--border-color)]">
          <button
            type="button"
            className="px-3 py-1 bg-gray-300 text-white rounded hover:bg-gray-600 text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors hover:cursor-pointer"
            title="Clear visible logs"
            aria-label="Clear session logs"
            onClick={handleClearSession}
          >
            🧹Clear
          </button>
          <button
            onClick={() => setShowDownloadType(true)}
            className="px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors hover:cursor-pointer"
            title="Download all Logs"
          >
            📥 Export
          </button>

          <button
            onClick={handleSave}
            className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors hover:cursor-pointer"
            title="Save all log entries to local storage"
          >
            💾 Save
          </button>
          <button
            onClick={() => setShowLoadType(true)}
            className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors hover:cursor-pointer"
            title="Load log entries"
          >
            📁 Load
          </button>
          <button
            onClick={() => setShowClearConfirm(true)}
            className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors hover:cursor-pointer"
            title="Clear local storage and panel"
          >
            🗑️ Delete
          </button>
        </div>

        {/* Confirmation Modal Dialogs */}
        {/* Clear LocalStorage Confirmation Modal Dialog */}
        {showClearConfirm && (
          <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/30">
            <div className="bg-white dark:bg-gray-900 rounded shadow-lg p-6 max-w-xs min-w-[280px]">
              <div className="font-bold text-red-700 mb-2">Confirm Clear</div>
              <div className="mb-4 text-sm">
                This will{" "}
                <span className="font-bold text-red-700">
                  permanently delete
                </span>{" "}
                all prompt log entries.
                <br />
                <span className="text-red-700">
                  This action cannot be undone.
                </span>
              </div>
              <div className="flex justify-end gap-3">
                <button
                  className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-xs font-medium"
                  onClick={() => {
                    clearEntriesFromLocal();
                    setShowClearConfirm(false);
                    showToast("Cleared from local storage");
                  }}
                >
                  Yes, clear
                </button>
                <button
                  className="px-3 py-1 bg-gray-200 text-gray-900 rounded hover:bg-gray-300 text-xs font-medium"
                  onClick={() => setShowClearConfirm(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Pick Download File Type Modal Dialog */}
        {showDownloadType && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
            <div className="bg-white dark:bg-gray-900 rounded shadow-lg p-6 min-w-[250px] max-w-xs">
              <div className="font-bold text-lg mb-2">Download Log</div>
              <div className="flex flex-col gap-2">
                <button
                  className="px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-semibold"
                  onClick={() => {
                    // Download JSON
                    const blob = new Blob(
                      [JSON.stringify(promptLog, null, 2)],
                      { type: "application/json" }
                    );
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `prompt_log_${new Date()
                      .toISOString()
                      .slice(0, 16)
                      .replace(/[:-]/g, "")}.json`;
                    document.body.appendChild(a);
                    a.click();
                    setTimeout(() => {
                      document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                    }, 100);
                    showToast("JSON downloaded!");
                    setShowDownloadType(false);
                  }}
                >
                  Export as JSON
                </button>
                <button
                  className="px-3 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 font-semibold"
                  onClick={() => {
                    // Download markdown
                    const content = promptLog
                      .map(formatLogEntryClipboard)
                      .join("\n\n");
                    const blob = new Blob([content], { type: "text/markdown" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `prompt_log_${new Date()
                      .toISOString()
                      .slice(0, 16)
                      .replace(/[:-]/g, "")}.md`;
                    document.body.appendChild(a);
                    a.click();
                    setTimeout(() => {
                      document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                    }, 100);
                    showToast("Markdown downloaded!");
                    setShowDownloadType(false);
                  }}
                >
                  Export as Markdown
                </button>
                <button
                  className="px-3 py-1 text-sm rounded hover:bg-gray-300"
                  onClick={() => setShowDownloadType(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Pick Load File Type Modal Dialog */}
        {showLoadType && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
            <div className="bg-white dark:bg-gray-900 rounded shadow-lg p-6 min-w-[250px] max-w-sm">
              <div className="font-bold text-lg mb-2">Load Log</div>
              <div className="flex flex-col gap-2">
                <button
                  className="px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-semibold"
                  onClick={() => {
                    const entriesFromLocalStore = loadEntriesFromLocal();
                    setPromptLog(entriesFromLocalStore);
                    showToast(
                      `Loaded ${entriesFromLocalStore.length} entries from localStorage`
                    );
                    setShowLoadType(false);
                  }}
                >
                  Load from LocalStorage
                </button>
                <label className="flex flex-col items-start w-full px-3 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 font-semibold cursor-pointer">
                  Import from JSON file
                  <input
                    type="file"
                    accept=".json,application/json"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        const reader = new FileReader();
                        reader.onload = (evt) => {
                          try {
                            const data = JSON.parse(
                              evt.target?.result as string
                            );
                            if (Array.isArray(data)) {
                              setPromptLog(data);
                              showToast(
                                `Loaded ${data.length} entries from file`
                              );
                              setShowLoadType(false);
                            } else {
                              showToast("Invalid file format");
                            }
                          } catch {
                            showToast("Error reading file");
                          }
                        };
                        reader.readAsText(file);
                      }
                    }}
                  />
                </label>
                <button
                  className="px-3 py-1 text-sm rounded hover:bg-gray-300"
                  onClick={() => setShowLoadType(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Filter Bar */}
        <div className="border-b p-4 bg-[var(--background)]">
          <div className="grid grid-cols-2 gap-3 items-center">
            {/* Source Filter */}
            <div className="flex flex-col">
              <label className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1 flex items-center gap-1">
                <span>Source filter</span>
                <span role="img" aria-label="source" className="text-sm">
                  🗂️
                </span>
              </label>
              <div className="relative flex items-center">
                <input
                  type="text"
                  value={sourceFilter}
                  onChange={(e) => setSourceFilter(e.target.value)}
                  className="text-xs rounded border px-2 py-1 bg-white dark:bg-[var(--background)] text-[var(--foreground)] w-full pr-7"
                  placeholder="Type source..."
                />
                {sourceFilter && (
                  <button
                    type="button"
                    onClick={() => setSourceFilter("")}
                    className="absolute right-1 top-1/2 -translate-y-1/2 text-gray-400 hover:text-[var(--accent-primary)] px-1"
                    title="Clear source filter"
                    aria-label="Clear source filter"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path d="M6.707 6.293a1 1 0 011.414 0L10 8.172l1.879-1.88a1 1 0 111.415 1.415L11.415 9.586l1.879 1.88a1 1 0 01-1.415 1.415L10 11.001l-1.879 1.88a1 1 0 11-1.415-1.415l1.88-1.88-1.88-1.88a1 1 0 01.001-1.415z" />
                    </svg>
                  </button>
                )}
              </div>
            </div>
            {/* Model Filter */}
            <div className="flex flex-col">
              <label className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1 flex items-center gap-1">
                <span>Model filter</span>
                <span role="img" aria-label="model" className="text-sm">
                  🤖
                </span>
              </label>
              <div className="relative flex items-center">
                <input
                  type="text"
                  value={modelFilter}
                  onChange={(e) => setModelFilter(e.target.value)}
                  className="text-xs rounded border px-2 py-1 bg-white dark:bg-[var(--background)] text-[var(--foreground)] w-full pr-7"
                  placeholder="Type model..."
                />
                {modelFilter && (
                  <button
                    type="button"
                    onClick={() => setModelFilter("")}
                    className="absolute right-1 top-1/2 -translate-y-1/2 text-gray-400 hover:text-[var(--accent-primary)] px-1"
                    title="Clear model filter"
                    aria-label="Clear model filter"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path d="M6.707 6.293a1 1 0 011.414 0L10 8.172l1.879-1.88a1 1 0 111.415 1.415L11.415 9.586l1.879 1.88a1 1 0 01-1.415 1.415L10 11.001l-1.879 1.88a1 1 0 11-1.415-1.415l1.88-1.88-1.88-1.88a1 1 0 01.001-1.415z" />
                    </svg>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="overflow-y-auto h-[80%] p-2">
          {filteredLog.length === 0 && (
            <div className="text-center text-gray-400 mt-10">
              No prompts for this filter.
            </div>
          )}
          {toast && (
            <div className="fixed bottom-7 right-7 z-50 px-4 py-2 font-bold bg-green-600 text-white rounded shadow-lg transition-opacity animate-fadeIn">
              {toast}
            </div>
          )}
          {filteredLog.map((entry, i) => (
            <div
              key={i}
              className="mb-2 bg-[var(--card-bg)] rounded shadow p-3"
            >
              {/* Collapse toggle & summary header */}
              <div
                className="flex items-center justify-between mb-2 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-800 rounded px-2 py-1 transition"
                onClick={() => toggleCollapsed(i)}
                title={collapsed[i] ? "Expand" : "Collapse"}
                aria-label={collapsed[i] ? "Expand entry" : "Collapse entry"}
                tabIndex={0}
                role="button"
                onKeyPress={(e) => {
                  if (e.key === "Enter" || e.key === " ") toggleCollapsed(i);
                }}
              >
                <span className="font-bold text-base text-[var(--accent-primary)]">
                  #{i + 1} {entry.source && `[${entry.source}]`}{" "}
                  {entry.model && `(${entry.model})`}
                </span>
                <span className="text-lg w-7 h-7 flex items-center justify-center">
                  {collapsed[i] ? "▶" : "▼"}
                </span>
              </div>

              {/* Only render detailed content if not collapsed */}
              {!collapsed[i] && (
                <>
                  {/* Toolbar */}
                  <div className="flex items-center justify-end mb-2 gap-2 bg-[var(--background)] p-1">
                    {/* Markdown/Plain Toggle */}
                    <button
                      onClick={() => toggleMode(i)}
                      className="text-xl bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-700 rounded-full w-8 h-8 flex items-center justify-center shadow hover:bg-[var(--accent-primary)]/20 transition"
                      aria-label={
                        markdownMode[i] ? "Show plain text" : "Show markdown"
                      }
                      title={
                        markdownMode[i] ? "Show plain text" : "Show markdown"
                      }
                      type="button"
                    >
                      {markdownMode[i] ? <>≡</> : <>{"{}"}</>}
                    </button>
                    {/* Copy to clipboard */}
                    <button
                      onClick={() => handleCopyEntry(entry)}
                      className="text-xl bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-700 rounded-full w-8 h-8 flex items-center justify-center shadow hover:bg-green-200 dark:hover:bg-green-900 transition"
                      aria-label="Copy response"
                      title="Copy response"
                      type="button"
                    >
                      📋
                    </button>
                    {/* Add more toolbar items here if needed */}
                  </div>

                  {/* Card header, metadata etc. */}
                  <div className="text-xs text-gray-400 mb-2 flex justify-between items-center">
                    <span>
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="text-[var(--accent-primary)] font-semibold">
                      {entry.source}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500 mb-1 flex justify-between">
                    <span>
                      Model:{" "}
                      {entry.model ? (
                        <b>{entry.model}</b>
                      ) : (
                        <span className="text-gray-400 italic">-</span>
                      )}
                    </span>
                    <span>
                      {typeof entry.timeTaken === "number" ? (
                        <>
                          Time Taken: <b>{entry.timeTaken.toFixed(2)}</b> s
                        </>
                      ) : (
                        <span className="text-gray-400 italic">
                          Time Taken: -
                        </span>
                      )}
                    </span>
                  </div>
                  {/* Prompt Section with Truncation */}
                  <div className="font-bold text-sm mb-1 text-[var(--accent-primary)]">
                    Prompt:
                  </div>
                  {(() => {
                    const MAX_PREVIEW_LENGTH = 500;
                    const isExpanded = expandedContent[i]?.prompt;
                    const shouldTruncate =
                      entry.prompt.length > MAX_PREVIEW_LENGTH;
                    const displayContent =
                      !shouldTruncate || isExpanded
                        ? entry.prompt
                        : entry.prompt.substring(0, MAX_PREVIEW_LENGTH) + "...";

                    return (
                      <>
                        {markdownMode[i] ? (
                          <Markdown content={displayContent} />
                        ) : (
                          <pre className="bg-[var(--background)] rounded p-2 text-xs overflow-x-auto whitespace-pre-wrap">
                            {displayContent}
                          </pre>
                        )}
                        {shouldTruncate && (
                          <button
                            onClick={() => toggleContentExpansion(i, "prompt")}
                            className="text-xs text-[var(--accent-primary)] hover:underline mt-1"
                          >
                            {isExpanded ? "Show Less" : "Show More"}
                          </button>
                        )}
                      </>
                    );
                  })()}

                  {/* Response Section with Truncation */}
                  <div className="font-bold text-sm mb-1 mt-3 text-[var(--accent-primary)]">
                    Response:
                  </div>
                  {(() => {
                    const MAX_PREVIEW_LENGTH = 500;
                    const isExpanded = expandedContent[i]?.response;
                    const shouldTruncate =
                      entry.response.length > MAX_PREVIEW_LENGTH;
                    const displayContent =
                      !shouldTruncate || isExpanded
                        ? entry.response
                        : entry.response.substring(0, MAX_PREVIEW_LENGTH) +
                          "...";

                    return (
                      <>
                        {markdownMode[i] ? (
                          <Markdown content={displayContent} />
                        ) : (
                          <pre className="bg-[var(--background)] rounded p-2 text-xs overflow-x-auto whitespace-pre-wrap">
                            {displayContent}
                          </pre>
                        )}
                        {shouldTruncate && (
                          <button
                            onClick={() =>
                              toggleContentExpansion(i, "response")
                            }
                            className="text-xs text-[var(--accent-primary)] hover:underline mt-1"
                          >
                            {isExpanded ? "Show Less" : "Show More"}
                          </button>
                        )}
                      </>
                    );
                  })()}
                                    
                {/* Retrieved Docs Section */}
                
{entry.retrieval && entry.retrieval.docs && entry.retrieval.docs.length > 0 && (
  <>
    {/* Header line with collapse/expand button */}
    <div className="flex items-center justify-between mb-1 mt-3">
      <div className="font-bold text-sm text-[var(--accent-primary)]">
        
        Retrieved Documents
        {entry.retrieval.rag_query ? `— Query: ${entry.retrieval.rag_query}` : ''} 
        {entry.retrieval.docs.length ? ` — Retrieved Docs Count: ${entry.retrieval.docs.length}` : ''}

      </div>

      {/* Collapse/Expand control */}
      <button
        type="button"
        onClick={() => {
          setExpandedContent(prev => ({
            ...prev,
            [i]: {
              ...prev[i],
              retrievalCollapsed: !prev[i]?.retrievalCollapsed,
            },
          }));
        }}
        className="text-xs px-2 py-1 rounded border hover:bg-[var(--background-muted)]"
        aria-expanded={!(expandedContent[i]?.retrievalCollapsed)}
        aria-controls={`retrieved-docs-${i}`}
      >
        {expandedContent[i]?.retrievalCollapsed ? "▶" : "▼"}
        
      </button>
      
      

    </div>

    {/* Per-doc listing, hidden when collapsed */}
    {!expandedContent[i]?.retrievalCollapsed && (
      <div id={`retrieved-docs-${i}`} className="space-y-2">
        {entry.retrieval.docs.map((doc: { text?: string; file_path?: string; score?: number }, dIdx: number) => {
          const MAX_PREVIEW_LENGTH = 500;
          const isExpanded = expandedContent[i]?.[`retrieved_${dIdx}` as any] as boolean;
          const shouldTruncate = !!doc.text && doc.text.length > MAX_PREVIEW_LENGTH;
          const displayContent =
            !shouldTruncate || isExpanded
              ? doc.text
              : (doc.text ?? '').substring(0, MAX_PREVIEW_LENGTH) + '...';

          const toggleRetrievedExpansion = () => {
            setExpandedContent(prev => ({
              ...prev,
              [i]: {
                ...prev[i],
                [`retrieved_${dIdx}`]: !prev[i]?.[`retrieved_${dIdx}`],
              },
            }));
          };

          return (
            <div key={dIdx} className="rounded border bg-[var(--background)] p-2">
              {/* Doc header row */}
              <div className="flex items-center justify-between">
                <div className="min-w-0 flex-1 text-xs font-semibold text-blue-600 line-clamp-2 break-words">
                  {doc.file_path}{' '}
                  <span className="text-gray-500">(Relevance: {doc.score})</span>
                </div>
              </div>

              {/* Doc text preview */}
              <pre className="bg-[var(--background)] rounded p-2 text-xs whitespace-pre-wrap break-words overflow-x-hidden">
                {displayContent}
              </pre>

              {shouldTruncate && (
                <button
                  onClick={toggleRetrievedExpansion}
                  className="text-xs text-[var(--accent-primary)] hover:underline mt-1"
                >
                  {isExpanded ? 'Show Less' : 'Show More'}
                </button>
              )}
            </div>
          );
        })}
      </div>
    )}
  </>
)}
                </>
              )}
            </div>
          ))}
        </div>
      </div>
    </>
  );
};

export default PromptLogFloatingPanel;
