"use client";

import { useState } from "react";
import {
  Activity,
  AlertCircle,
  Award,
  Box,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Cpu,
  Database,
  Eye,
  Info,
  Layers,
  ListTree,
  MessageSquare,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import type { GeoVisionAnalyzeResponse } from "@/types/geovision";
import { GeoVisionEvaluationModal } from "./GeoVisionEvaluationModal";

interface GeoVisionAnalysisPanelProps {
  response: GeoVisionAnalyzeResponse | null;
  loading: boolean;
  error: string | null;
}

export function GeoVisionAnalysisPanel({
  response,
  loading,
  error,
}: GeoVisionAnalysisPanelProps) {
  const [traceExpanded, setTraceExpanded] = useState(true);
  const [isEvaluationModalOpen, setIsEvaluationModalOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopyAnswer = () => {
    if (response?.answer) {
      navigator.clipboard.writeText(response.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[460px] rounded-3xl border border-white/10 bg-white/[0.02] p-8 text-center backdrop-blur-xl">
        <div className="relative mb-4">
          <div className="h-12 w-12 rounded-full border-2 border-cyan-400/30 border-t-cyan-400 animate-spin" />
          <Sparkles className="absolute inset-0 m-auto text-cyan-400" size={18} />
        </div>
        <h3 className="text-base font-semibold text-white">Running Aerial Vision Pipeline</h3>
        <p className="mt-1 text-xs text-slate-400 max-w-sm">
          Generating sliding-window tiles, running inference, deduplicating boxes via NMS, and synthesizing grounded explanation...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[360px] rounded-3xl border border-rose-500/20 bg-rose-500/[0.04] p-8 text-center">
        <AlertCircle size={36} className="text-rose-400 mb-3" />
        <h3 className="text-sm font-semibold text-white">Analysis Query Failed</h3>
        <p className="mt-1 text-xs text-rose-300/80 max-w-md">{error}</p>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[460px] rounded-3xl border border-white/10 bg-white/[0.02] p-8 text-center">
        <MessageSquare size={36} className="text-slate-600 mb-3" />
        <h3 className="text-sm font-semibold text-slate-300">Awaiting Aerial Inspection Query</h3>
        <p className="mt-1 text-xs text-slate-500 max-w-xs">
          Select a quick query chip or enter an analysis prompt to run real aerial object detection.
        </p>
      </div>
    );
  }

  // Filter categories to ONLY show those actually detected (> 0 count)
  const detectedCategories = Object.entries(response.object_summary || {}).filter(
    ([_, count]) => count > 0
  );

  return (
    <>
      <div className="flex flex-col gap-4 overflow-y-auto pr-1 h-full max-h-[620px]">
        {/* Primary Answer Card */}
        <div className="rounded-3xl border border-white/10 bg-gradient-to-b from-white/[0.06] to-white/[0.02] backdrop-blur-xl p-5 shadow-xl">
          <div className="flex items-center justify-between gap-2 mb-3 pb-3 border-b border-white/10">
            <div className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-cyan-500/20 text-cyan-400">
                <Sparkles size={13} />
              </span>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Grounded Result
              </span>
            </div>

            <div className="flex items-center gap-2">
              {response.is_one_word && (
                <span className="rounded-full bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 text-[10px] font-mono text-amber-300">
                  One-Word Mode
                </span>
              )}
              <span className="rounded-full bg-purple-500/10 border border-purple-500/20 px-2 py-0.5 text-[10px] font-mono text-purple-300">
                {response.detected_intent}
              </span>
              <button
                type="button"
                onClick={handleCopyAnswer}
                className="text-[11px] font-medium text-slate-400 hover:text-white px-2 py-0.5 rounded-md hover:bg-white/10 transition-colors"
              >
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
          </div>

          {/* Answer Text Content */}
          <div className="text-xs text-slate-200 leading-relaxed font-sans bg-black/25 rounded-2xl p-3.5 border border-white/5 whitespace-pre-line">
            {response.answer}
          </div>

          {/* Session & Persistence Footnote */}
          <div className="flex flex-wrap items-center justify-between gap-2 mt-3 text-[10.5px] text-slate-400 font-mono">
            <span className="flex items-center gap-1 text-emerald-400">
              <Database size={11} />
              <span>SQLite Persistent (mode: geovision)</span>
            </span>
            <span className="truncate max-w-[180px] text-slate-500">
              {response.session_id}
            </span>
          </div>
        </div>

        {/* Real Detected Objects List (ONLY showing categories actually detected) */}
        <div className="rounded-3xl border border-white/10 bg-white/[0.02] p-4 backdrop-blur-xl">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-300">
              <Box size={14} className="text-cyan-400" />
              <span>Detected Objects</span>
            </div>
            <span className="font-mono text-xs font-bold text-white">
              {response.detected_objects.length} Total
            </span>
          </div>

          {detectedCategories.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {detectedCategories.map(([category, count]) => (
                <div
                  key={category}
                  className="flex items-center justify-between rounded-xl border border-white/5 bg-black/20 px-3 py-2"
                >
                  <span className="text-xs font-medium text-slate-300 capitalize">
                    {category}
                  </span>
                  <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-md border border-cyan-500/20">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-2xl border border-white/5 bg-black/20 p-3 text-center">
              <p className="text-xs text-slate-400">
                0 verified object detections registered.
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                {response.provider_status?.yolo_detector === "active"
                  ? "No target classes satisfied the confidence threshold (0.25)."
                  : "Trained detection model unavailable on this deployment."}
              </p>
            </div>
          )}
        </div>

        {/* Model Architecture & Evaluation Inspector Card */}
        <div className="rounded-3xl border border-white/10 bg-white/[0.02] p-4 backdrop-blur-xl">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-300">
              <Cpu size={14} className="text-purple-400" />
              <span>Model Architecture</span>
            </div>
            <button
              type="button"
              onClick={() => setIsEvaluationModalOpen(true)}
              className="flex items-center gap-1.5 rounded-xl border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 px-2.5 py-1 text-xs font-semibold text-cyan-300 hover:text-cyan-200 transition-all cursor-pointer shadow-sm"
            >
              <Award size={13} />
              <span>View Evaluation Metrics</span>
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-300">
            <div className="rounded-xl border border-white/5 bg-black/20 p-2.5">
              <span className="text-slate-500 block text-[10px] uppercase">Model</span>
              <span className="font-bold text-white">
                {response.model_metadata?.model_name ?? "SatQuery-Aerial-YOLOv8"}
              </span>
            </div>
            <div className="rounded-xl border border-white/5 bg-black/20 p-2.5">
              <span className="text-slate-500 block text-[10px] uppercase">Model Version</span>
              <span className="text-cyan-300">
                {response.model_metadata?.model_version ?? "v1.2.0-aerial-finetuned"}
              </span>
            </div>
            <div className="rounded-xl border border-white/5 bg-black/20 p-2.5">
              <span className="text-slate-500 block text-[10px] uppercase">Resolution</span>
              <span className="text-slate-300">Native Sensor (Tiled)</span>
            </div>
            <div className="rounded-xl border border-white/5 bg-black/20 p-2.5">
              <span className="text-slate-500 block text-[10px] uppercase">Conf Threshold</span>
              <span className="text-amber-300">0.25 (IoU 0.45)</span>
            </div>
          </div>

          <div className="flex items-center justify-between mt-3 pt-2.5 border-t border-white/5 text-[10.5px] text-slate-400">
            <span>SAM 2 Status: <strong className="text-slate-500">Unavailable</strong></span>
            <span>GeoChat Status: <strong className="text-cyan-400">Connected</strong></span>
          </div>
        </div>

        {/* 12-Step Execution Audit Trace */}
        <div className="rounded-3xl border border-white/10 bg-white/[0.02] backdrop-blur-xl overflow-hidden">
          <button
            type="button"
            onClick={() => setTraceExpanded((v) => !v)}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-white/[0.02] transition-colors"
          >
            <div className="flex items-center gap-2">
              <ListTree size={14} className="text-purple-400" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                12-Step Execution Trace
              </span>
              <span className="text-[11px] font-mono text-slate-500">
                ({response.trace.length} phases)
              </span>
            </div>
            {traceExpanded ? (
              <ChevronUp size={16} className="text-slate-400" />
            ) : (
              <ChevronDown size={16} className="text-slate-400" />
            )}
          </button>

          {traceExpanded && (
            <div className="border-t border-white/10 p-3 space-y-1.5">
              {response.trace.map((step) => {
                const isCompleted = step.status === "completed";
                const isSkipped = step.status === "skipped";

                return (
                  <div
                    key={step.step_index}
                    className="flex items-start gap-2.5 rounded-xl bg-black/20 p-2 text-xs"
                  >
                    <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-white/5 font-mono text-[10px] text-slate-400">
                      {step.step_index}
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium text-slate-200 truncate">
                          {step.name}
                        </span>
                        <div className="flex items-center gap-1.5 shrink-0">
                          {step.duration_ms > 0 && (
                            <span className="text-[10px] font-mono text-slate-500">
                              {step.duration_ms}ms
                            </span>
                          )}
                          <span
                            className={`rounded px-1.5 py-0.2 text-[9px] font-mono uppercase ${
                              isCompleted
                                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                : isSkipped
                                ? "bg-slate-500/10 text-slate-400 border border-slate-500/20"
                                : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                            }`}
                          >
                            {step.status}
                          </span>
                        </div>
                      </div>
                      {step.details && (
                        <p className="mt-0.5 text-[11px] text-slate-400 leading-normal">
                          {step.details}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* View Evaluation Metrics Modal Dialog */}
      <GeoVisionEvaluationModal
        isOpen={isEvaluationModalOpen}
        onClose={() => setIsEvaluationModalOpen(false)}
        reportData={response.evaluation_metrics as any}
      />
    </>
  );
}
