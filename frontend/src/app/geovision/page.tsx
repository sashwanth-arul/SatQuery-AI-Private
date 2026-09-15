"use client";

import { useState } from "react";
import {
  Compass,
  Cpu,
  Eye,
  Layers,
  Maximize2,
  MessageSquare,
  Plane,
  Scan,
  Send,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import type {
  GeoVisionAnalyzeResponse,
  GeoVisionIntent,
  GeoVisionUploadResponse,
} from "@/types/geovision";
import { geovisionApi } from "@/lib/geovisionApi";
import { GeoVisionUploadDropzone } from "@/components/geovision/GeoVisionUploadDropzone";
import { GeoVisionViewer } from "@/components/geovision/GeoVisionViewer";
import { GeoVisionAnalysisPanel } from "@/components/geovision/GeoVisionAnalysisPanel";

const PRESET_QUERIES = [
  {
    label: "Describe Scene",
    query: "Describe this aerial scene in detail including land use and notable structures.",
    intent: "detailed_description" as GeoVisionIntent,
  },
  {
    label: "Urban vs Rural (One Word)",
    query: "Is this area urban or rural? Answer in one word.",
    intent: "scene_classification" as GeoVisionIntent,
  },
  {
    label: "Object Counting",
    query: "Count the number of visible vehicles, buildings, and infrastructure features.",
    intent: "object_count" as GeoVisionIntent,
  },
  {
    label: "Grounded Detection",
    query: "Detect and highlight all buildings and roads in this image.",
    intent: "object_detection" as GeoVisionIntent,
  },
  {
    label: "Land Cover & Vegetation",
    query: "What land-cover types and vegetation patterns are present?",
    intent: "visual_qa" as GeoVisionIntent,
  },
];

const INTENT_OPTIONS: { value: GeoVisionIntent; label: string }[] = [
  { value: "automatic", label: "Automatic (NL Intent Router)" },
  { value: "detailed_description", label: "Detailed Description" },
  { value: "grounded_description", label: "Grounded Description" },
  { value: "visual_qa", label: "Visual QA" },
  { value: "object_count", label: "Object Count" },
  { value: "object_detection", label: "Object Detection" },
  { value: "scene_classification", label: "Scene Classification" },
  { value: "complex_reasoning", label: "Complex Reasoning" },
];

export default function GeoVisionPage() {
  const [currentImage, setCurrentImage] = useState<GeoVisionUploadResponse | null>(null);
  const [query, setQuery] = useState("");
  const [selectedIntent, setSelectedIntent] = useState<GeoVisionIntent>("automatic");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResponse, setAnalysisResponse] = useState<GeoVisionAnalyzeResponse | null>(null);

  const handleImageUploaded = (uploaded: GeoVisionUploadResponse) => {
    setCurrentImage(uploaded);
    setAnalysisResponse(null);
    setError(null);
  };

  const handleClearImage = () => {
    setCurrentImage(null);
    setAnalysisResponse(null);
    setError(null);
    setQuery("");
  };

  const handleRunAnalysis = async (customQuery?: string, customIntent?: GeoVisionIntent) => {
    const q = (customQuery ?? query).trim();
    if (!q || !currentImage || loading) return;

    setLoading(true);
    setError(null);

    try {
      const response = await geovisionApi.analyzeImage({
        image_id: currentImage.image_id,
        query: q,
        intent: customIntent ?? selectedIntent,
      });
      setAnalysisResponse(response);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to execute GeoVision analysis query.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectPreset = (preset: (typeof PRESET_QUERIES)[0]) => {
    setQuery(preset.query);
    setSelectedIntent(preset.intent);
    if (currentImage) {
      handleRunAnalysis(preset.query, preset.intent);
    }
  };

  return (
    <main className="min-h-screen bg-[#06080d] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Background Ambience */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-32 right-1/4 h-[500px] w-[500px] rounded-full bg-cyan-600/10 blur-[140px]" />
        <div className="absolute top-1/2 -left-40 h-[600px] w-[600px] rounded-full bg-purple-600/10 blur-[160px]" />
      </div>

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-24 pb-16">
        {/* Header Banner */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-white/10 pb-6 mb-6">
          <div>
            <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs uppercase tracking-widest mb-1.5">
              <Eye size={14} />
              <span>Dedicated Aerial & Drone Remote-Sensing Lab</span>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
              GeoVision Workspace
              <span className="text-xs font-mono font-normal px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                Aerial Detection Pipeline Active
              </span>
            </h1>
            <p className="mt-1 text-sm text-slate-400 max-w-2xl">
              Inspect uploaded aerial, drone, and high-resolution Earth imagery with zero-hallucination grounded visual QA, scene classification, and planned GeoChat & YOLO integration.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300">
              <ShieldCheck size={14} className="text-emerald-400" />
              <span>Zero-Fabrication Standard</span>
            </span>
            <span className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300">
              <Plane size={14} className="text-cyan-400" />
              <span>Aerial / Drone</span>
            </span>
          </div>
        </div>

        {/* Upload Dropzone Bar */}
        <div className="mb-6">
          <GeoVisionUploadDropzone
            currentImage={currentImage}
            onImageUploaded={handleImageUploaded}
            onClearImage={handleClearImage}
            disabled={loading}
          />
        </div>

        {/* Workspace Body */}
        {currentImage ? (
          <div className="space-y-6">
            {/* Split Screen Workspace */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Column: Interactive Image Viewer (7 cols) */}
              <div className="lg:col-span-7 h-[620px]">
                <GeoVisionViewer
                  image={currentImage}
                  detectedObjects={analysisResponse?.detected_objects ?? []}
                />
              </div>

              {/* Right Column: Grounded Analysis & Trace Panel (5 cols) */}
              <div className="lg:col-span-5 h-[620px]">
                <GeoVisionAnalysisPanel
                  response={analysisResponse}
                  loading={loading}
                  error={error}
                />
              </div>
            </div>

            {/* Query Composer Bar */}
            <div className="rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-xl p-4 shadow-xl">
              {/* Preset Chips */}
              <div className="flex flex-wrap items-center gap-2 mb-3">
                <span className="text-xs font-mono text-slate-400 flex items-center gap-1">
                  <Sparkles size={12} className="text-cyan-400" />
                  Quick Queries:
                </span>
                {PRESET_QUERIES.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    onClick={() => handleSelectPreset(preset)}
                    disabled={loading}
                    className="rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 px-2.5 py-1 text-xs text-slate-300 hover:text-white transition-all cursor-pointer disabled:opacity-50"
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              {/* Input Row & Intent Selector */}
              <div className="flex flex-col sm:flex-row gap-2">
                <div className="sm:w-56 shrink-0">
                  <select
                    value={selectedIntent}
                    onChange={(e) => setSelectedIntent(e.target.value as GeoVisionIntent)}
                    disabled={loading}
                    className="w-full rounded-xl border border-white/10 bg-[#0c1017] px-3 py-2 text-xs text-slate-200 focus:border-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-400"
                  >
                    {INTENT_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="relative flex-1">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        handleRunAnalysis();
                      }
                    }}
                    placeholder="Ask a question or request detection (e.g. 'Is this area urban or rural? Answer in one word')..."
                    disabled={loading}
                    className="w-full rounded-xl border border-white/10 bg-[#0c1017] px-4 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-400 pr-24"
                  />
                  <div className="absolute right-1.5 top-1.5 flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => handleRunAnalysis()}
                      disabled={loading || !query.trim()}
                      className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500 px-3 py-1 text-xs font-semibold text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-md shadow-cyan-950/50"
                    >
                      <span>Analyze</span>
                      <Send size={12} />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Empty State: Feature Introduction */
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-5 backdrop-blur-xl">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 mb-3">
                <Scan size={20} />
              </div>
              <h3 className="text-sm font-semibold text-white">Direct Raster Inspection</h3>
              <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">
                Upload aerial drone orthomosaics, PNG, JPG, or GeoTIFF rasters directly. GeoVision extracts coordinate reference systems, pixel dimensions, and channel statistics.
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-5 backdrop-blur-xl">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 mb-3">
                <Cpu size={20} />
              </div>
              <h3 className="text-sm font-semibold text-white">Hybrid AI Architecture</h3>
              <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">
                Engineered for specialized vision models: GeoChat for remote sensing VQA, YOLO for calibrated object bounding boxes, and SAM 2 for precise mask segmentation.
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-5 backdrop-blur-xl">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 mb-3">
                <ShieldCheck size={20} />
              </div>
              <h3 className="text-sm font-semibold text-white">Honest AI Verification</h3>
              <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">
                SatQuery strictly forbids fake detections or fabricated confidence scores. When external GPU models are unconfigured, capability status is reported transparently.
              </p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
