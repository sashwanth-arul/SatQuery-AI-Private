"use client";

import { useEffect, useState } from "react";
import {
  AlertCircle,
  Clock,
  Compass,
  Download,
  FileCheck,
  Globe,
  Layers,
  MapPin,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import type { AnalysisHistoryItem, AnalysisResult, TraceStep, Metric, EvidenceRegion } from "@/types/domain";
import { api } from "@/lib/api";
import { ExecutionTrace } from "@/components/ExecutionTrace";

type Props = {
  sessionId: string | null;
  initialItem?: AnalysisHistoryItem | null;
  isOpen: boolean;
  onClose: () => void;
};

export function AnalysisDetailModal({ sessionId, initialItem, isOpen, onClose }: Props) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [trace, setTrace] = useState<TraceStep[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen || !sessionId) {
      setResult(null);
      setTrace(null);
      setError(null);
      return;
    }

    let active = true;
    setLoading(true);
    setError(null);

    Promise.allSettled([
      api.getHistoryResult(sessionId),
      api.getHistoryTrace(sessionId),
    ]).then(([resResult, resTrace]) => {
      if (!active) return;
      setLoading(false);

      if (resResult.status === "fulfilled") {
        setResult(resResult.value);
      } else {
        setError(resResult.reason?.message || "Failed to load full analysis details.");
      }

      if (resTrace.status === "fulfilled") {
        setTrace(resTrace.value);
      }
    });

    return () => {
      active = false;
    };
  }, [isOpen, sessionId]);

  // Handle ESC key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !sessionId) return null;

  const currentQuery = initialItem?.query || (result as unknown as { query?: string })?.query || "Analysis query";
  const currentMode = result?.mode || initialItem?.mode || "catalog";
  const currentStatus = initialItem?.status || (result ? "completed" : "unknown");
  const createdAt = initialItem?.created_at;
  const answer = result?.answer || initialItem?.summary_answer || "No narrative generated.";
  const confidence = result?.confidence ?? initialItem?.confidence ?? null;

  // Extract metadata from trace steps if available
  const inputStep = trace?.find((s) => s.tool_name === "input_validation");
  const detectStep = trace?.find((s) => s.tool_name === "detect_change" || s.tool_name === "detect_sar_change");
  const fetchStep = trace?.find((s) => s.tool_name === "fetch_imagery");

  const rawInputMeta = (inputStep?.metadata || {}) as Record<string, unknown>;
  const rawDetectMeta = (detectStep?.metadata || {}) as Record<string, unknown>;
  const rawFetchMeta = (fetchStep?.metadata || {}) as Record<string, unknown>;

  const crs = (rawInputMeta.crs as string)
    || (rawInputMeta.earlier_crs as string)
    || (initialItem?.input_summary?.crs as string)
    || "EPSG:4326";

  const bounds = rawInputMeta.bounds
    || rawDetectMeta.bounds
    || rawFetchMeta.bbox
    || null;

  const earlierAcq = result?.bi_temporal_change?.earlier_acquisition
    || (rawInputMeta.earlier_acquisition as string)
    || (initialItem?.input_summary?.earlier_date as string)
    || null;
  const laterAcq = result?.bi_temporal_change?.later_acquisition
    || (rawInputMeta.later_acquisition as string)
    || (initialItem?.input_summary?.later_date as string)
    || null;

  // Gather models/tools used
  const toolsUsed = trace ? Array.from(new Set(trace.map((s) => s.tool_name))) : [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/80 backdrop-blur-md overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-2xl border border-white/10 bg-[#0c101b] text-slate-100 shadow-2xl shadow-cyan-950/40 flex flex-col my-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="sticky top-0 z-20 flex items-center justify-between border-b border-white/10 bg-[#0c101b]/95 px-6 py-4 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Sparkles size={18} />
            </span>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold tracking-wide text-white">
                  Analysis Details
                </h2>
                <span className="font-mono text-xs px-2 py-0.5 rounded bg-white/5 border border-white/10 text-slate-400">
                  {sessionId.slice(0, 8)}
                </span>
                <span
                  className={`text-[11px] px-2 py-0.5 rounded-full font-medium border ${
                    currentStatus === "completed"
                      ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/25"
                      : "bg-rose-500/10 text-rose-400 border-rose-500/25"
                  }`}
                >
                  {currentStatus}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5 flex items-center gap-2">
                <span className="capitalize text-cyan-300 font-medium">
                  {currentMode.replace(/_/g, " ")}
                </span>
                {createdAt && (
                  <>
                    <span>•</span>
                    <span className="flex items-center gap-1">
                      <Clock size={12} />
                      {new Date(createdAt).toLocaleString()}
                    </span>
                  </>
                )}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => api.downloadReport(sessionId)}
              className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-purple-600 px-3 py-1.5 text-xs font-semibold text-white shadow-md shadow-cyan-500/20 hover:from-cyan-400 hover:to-purple-500 transition-all cursor-pointer"
              title="Download Full Report"
            >
              <Download size={14} />
              <span>Download Report</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6 flex-1">
          {error && (
            <div className="flex items-center gap-2 rounded-xl border border-rose-500/20 bg-rose-500/10 p-3 text-xs text-rose-300">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {/* 1. Query & AI Answer Section */}
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
            <div>
              <span className="text-[11px] font-medium uppercase tracking-wider text-cyan-400">
                User Query
              </span>
              <p className="mt-1 text-sm font-medium text-slate-200">
                "{currentQuery}"
              </p>
            </div>

            <div className="pt-2 border-t border-white/5">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] font-medium uppercase tracking-wider text-purple-400 flex items-center gap-1.5">
                  <Sparkles size={13} />
                  AI Verified Answer
                </span>
                {confidence != null && (
                  <span className="text-xs px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/25 text-purple-300 font-mono">
                    Confidence: {(confidence * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <p className="text-sm leading-relaxed text-slate-300 bg-black/30 rounded-lg p-3 border border-white/5 whitespace-pre-wrap">
                {answer}
              </p>
            </div>
          </div>

          {/* 2. Imagery & Spatial Metadata */}
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
              <Globe size={14} className="text-cyan-400" />
              Spatial & Imagery Provenance
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
              <div className="rounded-lg bg-black/25 p-2.5 border border-white/5">
                <span className="text-slate-400 block text-[11px]">Analysis Mode</span>
                <span className="font-mono text-cyan-300 mt-0.5 block capitalize">
                  {currentMode.replace(/_/g, " ")}
                </span>
              </div>
              <div className="rounded-lg bg-black/25 p-2.5 border border-white/5">
                <span className="text-slate-400 block text-[11px]">Coordinate System (CRS)</span>
                <span className="font-mono text-slate-200 mt-0.5 block">{String(crs)}</span>
              </div>
              <div className="rounded-lg bg-black/25 p-2.5 border border-white/5">
                <span className="text-slate-400 block text-[11px]">Acquisition Window</span>
                <span className="font-mono text-slate-200 mt-0.5 block">
                  {earlierAcq || laterAcq
                    ? `${earlierAcq ?? "T1"} → ${laterAcq ?? "T2"}`
                    : "Single epoch / AOI"}
                </span>
              </div>
              {bounds && (
                <div className="col-span-1 sm:col-span-2 lg:col-span-3 rounded-lg bg-black/25 p-2.5 border border-white/5">
                  <span className="text-slate-400 block text-[11px]">Bounding Box / Extent</span>
                  <span className="font-mono text-[11px] text-slate-300 break-all mt-0.5 block">
                    {typeof bounds === "object" ? JSON.stringify(bounds) : String(bounds)}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* 3. Quantitative Metrics / Statistics (If Available) */}
          {result?.metrics && result.metrics.length > 0 && (
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Zap size={14} className="text-purple-400" />
                Verified Quantitative Metrics
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {result.metrics.map((metric: Metric, idx: number) => (
                  <div key={idx} className="rounded-lg bg-black/30 p-3 border border-white/5">
                    <span className="text-[11px] text-slate-400 block capitalize">
                      {metric.name.replace(/_/g, " ")}
                    </span>
                    <span className="text-lg font-bold text-cyan-300 font-mono mt-1 block">
                      {typeof metric.value === "number"
                        ? Number.isInteger(metric.value)
                          ? metric.value
                          : metric.value.toFixed(2)
                        : String(metric.value)}
                      {metric.unit ? ` ${metric.unit}` : ""}
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">{metric.source}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 4. Specialist Analysis Modules */}
          {result?.cross_modal && (
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-400 flex items-center gap-1.5">
                <Layers size={14} />
                Cross-Modal Optical + SAR Insights
              </h3>
              <div className="space-y-2 text-xs">
                <div className="rounded-lg bg-black/30 p-3 border border-white/5">
                  <span className="text-slate-400 block font-medium">Optical Summary</span>
                  <p className="mt-1 text-slate-300">{result.cross_modal.optical_analysis.summary}</p>
                </div>
                <div className="rounded-lg bg-black/30 p-3 border border-white/5">
                  <span className="text-slate-400 block font-medium">SAR Backscatter Summary</span>
                  <p className="mt-1 text-slate-300">{result.cross_modal.sar_analysis.summary}</p>
                </div>
                <div className="rounded-lg bg-black/30 p-3 border border-white/5">
                  <span className="text-slate-400 block font-medium">Joint Fused Conclusion</span>
                  <p className="mt-1 text-slate-300">{result.cross_modal.fused_analysis.summary}</p>
                </div>
              </div>
            </div>
          )}

          {result?.building_temporal_change && (
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                <FileCheck size={14} />
                Building Temporal Change Breakdown
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs">
                <div className="rounded-lg bg-black/30 p-2.5 border border-white/5 text-center">
                  <span className="text-slate-400 block">T1 Count</span>
                  <span className="text-base font-bold text-white font-mono">{result.building_temporal_change.before_count}</span>
                </div>
                <div className="rounded-lg bg-black/30 p-2.5 border border-white/5 text-center">
                  <span className="text-slate-400 block">T2 Count</span>
                  <span className="text-base font-bold text-white font-mono">{result.building_temporal_change.after_count}</span>
                </div>
                <div className="rounded-lg bg-black/30 p-2.5 border border-white/5 text-center">
                  <span className="text-emerald-400 block">Added</span>
                  <span className="text-base font-bold text-emerald-400 font-mono">+{result.building_temporal_change.new_count}</span>
                </div>
                <div className="rounded-lg bg-black/30 p-2.5 border border-white/5 text-center">
                  <span className="text-rose-400 block">Demolished</span>
                  <span className="text-base font-bold text-rose-400 font-mono">-{result.building_temporal_change.removed_count}</span>
                </div>
                <div className="rounded-lg bg-black/30 p-2.5 border border-white/5 text-center">
                  <span className="text-cyan-400 block">Net Change</span>
                  <span className="text-base font-bold text-cyan-300 font-mono">
                    {result.building_temporal_change.after_count - result.building_temporal_change.before_count > 0
                      ? `+${result.building_temporal_change.after_count - result.building_temporal_change.before_count}`
                      : result.building_temporal_change.after_count - result.building_temporal_change.before_count}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* 5. Detected Evidence Regions */}
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <MapPin size={14} className="text-cyan-400" />
                Detected Spatial Regions ({result?.evidence?.length || 0})
              </h3>
            </div>
            {result?.evidence && result.evidence.length > 0 ? (
              <div className="max-h-60 overflow-y-auto space-y-2 pr-1">
                {result.evidence.map((region: EvidenceRegion, idx: number) => {
                  const areaM2 = (region.metadata?.area_m2 as number | undefined);
                  const severity = (region.metadata?.severity as string | undefined);
                  return (
                    <div
                      key={region.id || idx}
                      className="flex items-center justify-between rounded-lg bg-black/30 p-2.5 border border-white/5 text-xs"
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-cyan-400">{region.id}</span>
                        <span className="text-slate-300 capitalize">{region.type.replace(/_/g, " ")}</span>
                      </div>
                      <div className="flex items-center gap-3 text-slate-400 font-mono text-[11px]">
                        {areaM2 != null && (
                          <span>{(areaM2 / 10000).toFixed(2)} ha</span>
                        )}
                        {region.confidence != null && (
                          <span className="text-purple-300">{(region.confidence * 100).toFixed(0)}% conf</span>
                        )}
                        {severity && (
                          <span className="capitalize px-1.5 py-0.5 rounded bg-white/5 text-slate-300">
                            {severity}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-slate-400">No discrete spatial regions generated for this analysis.</p>
            )}
          </div>

          {/* 6. Execution Trace & Specialist Tools */}
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Compass size={14} className="text-purple-400" />
                Execution Trace & Specialist Tool Pipeline
              </h3>
              {toolsUsed.length > 0 && (
                <span className="text-[11px] text-slate-400">
                  {toolsUsed.length} tools executed
                </span>
              )}
            </div>

            {toolsUsed.length > 0 && (
              <div className="flex flex-wrap gap-1.5 py-1">
                {toolsUsed.map((tool) => (
                  <span
                    key={tool}
                    className="font-mono text-[11px] px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300"
                  >
                    {tool}
                  </span>
                ))}
              </div>
            )}

            {trace && trace.length > 0 ? (
              <div className="rounded-lg bg-black/40 p-3 border border-white/5">
                <ExecutionTrace steps={trace} loading={loading} />
              </div>
            ) : (
              <p className="text-xs text-slate-400">No trace steps recorded.</p>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="border-t border-white/10 bg-[#0c101b] px-6 py-3 flex items-center justify-between">
          <span className="text-xs text-slate-400">
            SatQuery AI • Evidence-Validated Remote Sensing
          </span>
          <button
            type="button"
            onClick={() => api.downloadReport(sessionId)}
            className="flex items-center gap-1.5 rounded-lg bg-white/10 hover:bg-white/15 px-3 py-1.5 text-xs font-medium text-white transition-colors cursor-pointer"
          >
            <Download size={14} />
            <span>Download Self-Contained Report</span>
          </button>
        </div>
      </div>
    </div>
  );
}
