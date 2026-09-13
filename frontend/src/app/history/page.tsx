"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Clock,
  Compass,
  Download,
  Eye,
  FileText,
  Filter,
  History,
  Layers,
  RefreshCw,
  Search,
  Sparkles,
  Zap,
} from "lucide-react";
import type { AnalysisHistoryItem } from "@/types/domain";
import { api } from "@/lib/api";
import { AnalysisDetailModal } from "@/components/AnalysisDetailModal";

type FilterTab = "all" | "upload" | "temporal_pair" | "cross_modal" | "completed" | "failed";

export default function HistoryPage() {
  const [items, setItems] = useState<AnalysisHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<FilterTab>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<AnalysisHistoryItem | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      let modeParam: string | undefined;
      let statusParam: string | undefined;

      if (activeTab === "upload" || activeTab === "temporal_pair" || activeTab === "cross_modal") {
        modeParam = activeTab;
      } else if (activeTab === "completed" || activeTab === "failed") {
        statusParam = activeTab;
      }

      const res = await api.getHistory({
        limit: 100,
        mode: modeParam,
        status: statusParam,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load analysis history.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [activeTab]);

  const filteredItems = items.filter((item) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      item.query.toLowerCase().includes(q) ||
      item.session_id.toLowerCase().includes(q) ||
      (item.summary_answer && item.summary_answer.toLowerCase().includes(q)) ||
      (item.mode && item.mode.toLowerCase().includes(q))
    );
  });

  const handleOpenDetail = (item: AnalysisHistoryItem) => {
    setSelectedItem(item);
    setSelectedSessionId(item.session_id);
  };

  return (
    <main className="min-h-screen bg-[#07090e] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Background Decorative Gradients */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 left-1/4 h-[500px] w-[500px] rounded-full bg-cyan-600/10 blur-[120px]" />
        <div className="absolute top-1/3 -right-40 h-[600px] w-[600px] rounded-full bg-purple-600/10 blur-[140px]" />
      </div>

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-24 pb-16">
        {/* Page Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-white/10 pb-6">
          <div>
            <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs uppercase tracking-widest mb-1.5">
              <History size={14} />
              <span>Audit Trail & Persistence</span>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
              Analysis History
              <span className="text-xs font-mono font-normal px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                {total} recorded
              </span>
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              Persistent, verifiable archive of all satellite analyses, specialist traces, and multimodal results.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={fetchHistory}
              disabled={loading}
              className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 px-3.5 py-2 text-xs font-medium text-slate-300 hover:text-white transition-all cursor-pointer disabled:opacity-50"
            >
              <RefreshCw size={14} className={loading ? "animate-spin text-cyan-400" : ""} />
              <span>Refresh</span>
            </button>
            <Link
              href="/workstation"
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-cyan-500/20 transition-all"
            >
              <Sparkles size={14} />
              <span>New Analysis</span>
            </Link>
          </div>
        </div>

        {/* Filter Tabs & Search Bar */}
        <div className="mt-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-white/[0.03] border border-white/10">
            <button
              type="button"
              onClick={() => setActiveTab("all")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "all"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              All
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("upload")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "upload"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Single Image
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("temporal_pair")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "temporal_pair"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Temporal Change
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("cross_modal")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "cross_modal"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Optical + SAR
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("completed")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "completed"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Completed
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("failed")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "failed"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Failed
            </button>
          </div>

          <div className="relative w-full md:w-72">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search queries, IDs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-white/[0.04] pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:border-cyan-500/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
            />
          </div>
        </div>

        {/* Content Area */}
        <div className="mt-6">
          {error && (
            <div className="mb-6 rounded-xl border border-rose-500/20 bg-rose-500/10 p-4 text-sm text-rose-300">
              {error}
            </div>
          )}

          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3, 4, 5, 6].map((idx) => (
                <div
                  key={idx}
                  className="rounded-2xl border border-white/10 bg-white/[0.02] p-5 animate-pulse space-y-4"
                >
                  <div className="flex items-center justify-between">
                    <div className="h-4 w-20 bg-white/10 rounded" />
                    <div className="h-4 w-16 bg-white/10 rounded" />
                  </div>
                  <div className="h-5 w-3/4 bg-white/10 rounded" />
                  <div className="h-12 w-full bg-white/5 rounded" />
                  <div className="h-8 w-full bg-white/10 rounded" />
                </div>
              ))}
            </div>
          ) : filteredItems.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-white/[0.01] p-12 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 mb-4">
                <History size={24} />
              </div>
              <h3 className="text-base font-semibold text-slate-200">No analyses found</h3>
              <p className="mt-1 text-xs text-slate-400 max-w-sm">
                {searchQuery
                  ? "No analysis sessions match your search query."
                  : "Submit queries in the workstation to run specialist analyses and persist results."}
              </p>
              <Link
                href="/workstation"
                className="mt-4 rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 px-4 py-2 text-xs font-semibold text-cyan-300 transition-all"
              >
                Go to Workstation
              </Link>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredItems.map((item) => {
                const modeLabel = (item.mode || "catalog").replace(/_/g, " ");
                const isCompleted = item.status === "completed";

                return (
                  <div
                    key={item.session_id}
                    className="group relative flex flex-col justify-between rounded-2xl border border-white/10 bg-[#0d121f]/70 hover:bg-[#0d121f] p-5 shadow-lg backdrop-blur-sm transition-all hover:border-cyan-500/30 hover:shadow-cyan-950/20"
                  >
                    <div>
                      {/* Top Badges */}
                      <div className="flex items-center justify-between gap-2 mb-3">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-white/5 border border-white/10 text-slate-400">
                            {item.session_id.slice(0, 8)}
                          </span>
                          <span
                            className={`text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-full border capitalize ${
                              item.mode === "cross_modal"
                                ? "bg-amber-500/10 text-amber-300 border-amber-500/20"
                                : item.mode === "temporal_pair"
                                ? "bg-purple-500/10 text-purple-300 border-purple-500/20"
                                : "bg-cyan-500/10 text-cyan-300 border-cyan-500/20"
                            }`}
                          >
                            {modeLabel}
                          </span>
                        </div>
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-medium border ${
                            isCompleted
                              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                              : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                          }`}
                        >
                          {item.status}
                        </span>
                      </div>

                      {/* Query */}
                      <h2 className="text-sm font-semibold text-slate-100 line-clamp-2 mb-2 group-hover:text-cyan-200 transition-colors">
                        "{item.query}"
                      </h2>

                      {/* Answer Summary Preview */}
                      {item.summary_answer && (
                        <p className="text-xs text-slate-400 line-clamp-3 mb-3 bg-black/30 p-2 rounded-lg border border-white/5">
                          {item.summary_answer}
                        </p>
                      )}

                      {/* Quantitative Metrics / Confidence if available */}
                      <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-400 mb-4">
                        {item.confidence != null && (
                          <span className="font-mono text-purple-300 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">
                            {(item.confidence * 100).toFixed(0)}% conf
                          </span>
                        )}
                        {item.metrics_summary && Object.keys(item.metrics_summary).length > 0 && (
                          <span className="font-mono text-cyan-300 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                            {Object.keys(item.metrics_summary).length} metrics
                          </span>
                        )}
                        <span className="flex items-center gap-1 text-[10px] text-slate-500 ml-auto">
                          <Clock size={11} />
                          {new Date(item.created_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>

                    {/* Card Actions */}
                    <div className="flex items-center gap-2 pt-3 border-t border-white/5">
                      <button
                        type="button"
                        onClick={() => handleOpenDetail(item)}
                        className="flex-1 flex items-center justify-center gap-1.5 rounded-xl bg-white/5 hover:bg-white/10 px-3 py-2 text-xs font-semibold text-slate-200 hover:text-white border border-white/10 transition-all cursor-pointer"
                      >
                        <Eye size={13} className="text-cyan-400" />
                        <span>VIEW</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => api.downloadReport(item.session_id)}
                        className="flex items-center justify-center rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/25 p-2 text-xs font-semibold text-cyan-300 transition-all cursor-pointer"
                        title="Download Analysis Report"
                      >
                        <Download size={14} />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* 16-point Analysis Detail Slideover / Modal */}
      <AnalysisDetailModal
        sessionId={selectedSessionId}
        initialItem={selectedItem}
        isOpen={Boolean(selectedSessionId)}
        onClose={() => {
          setSelectedSessionId(null);
          setSelectedItem(null);
        }}
      />
    </main>
  );
}
