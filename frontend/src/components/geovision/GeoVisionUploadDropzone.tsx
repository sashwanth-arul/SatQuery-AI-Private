"use client";

import { useCallback, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  FileUp,
  Globe,
  HardDrive,
  ImageIcon,
  Loader2,
  Maximize2,
  RefreshCw,
} from "lucide-react";
import type { GeoVisionUploadResponse } from "@/types/geovision";
import { geovisionApi } from "@/lib/geovisionApi";

interface GeoVisionUploadDropzoneProps {
  currentImage: GeoVisionUploadResponse | null;
  onImageUploaded: (metadata: GeoVisionUploadResponse) => void;
  onClearImage: () => void;
  disabled?: boolean;
}

export function GeoVisionUploadDropzone({
  currentImage,
  onImageUploaded,
  onClearImage,
  disabled = false,
}: GeoVisionUploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setError(null);
    setUploading(true);

    const validExtensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".geotiff"];
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!validExtensions.includes(ext)) {
      setError(
        `Unsupported file type '${ext}'. Please upload PNG, JPG, TIFF, or GeoTIFF aerial/drone imagery.`
      );
      setUploading(false);
      return;
    }

    try {
      const response = await geovisionApi.uploadImage(file);
      onImageUploaded(response);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Failed to upload image to GeoVision workspace.";
      setError(message);
    } finally {
      setUploading(false);
    }
  };

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled || uploading) return;

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleUpload(e.dataTransfer.files[0]);
      }
    },
    [disabled, uploading]
  );

  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const onFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleUpload(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  if (currentImage) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-xl p-4 transition-all">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <ImageIcon size={24} />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <p className="truncate text-sm font-semibold text-white">
                  {currentImage.filename}
                </p>
                <span className="shrink-0 rounded-full bg-cyan-500/10 px-2 py-0.5 text-[10px] font-mono uppercase text-cyan-300 border border-cyan-500/20">
                  {currentImage.format}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-3 mt-1 text-xs text-slate-400">
                <span className="flex items-center gap-1">
                  <Maximize2 size={12} className="text-slate-500" />
                  {currentImage.width} × {currentImage.height} px
                </span>
                <span className="flex items-center gap-1">
                  <HardDrive size={12} className="text-slate-500" />
                  {formatFileSize(currentImage.file_size_bytes)}
                </span>
                <span className="flex items-center gap-1">
                  <Globe size={12} className="text-slate-500" />
                  {currentImage.georeferenced ? (
                    <span className="text-emerald-400 font-medium">
                      Georeferenced ({currentImage.crs ?? "Native"})
                    </span>
                  ) : (
                    <span className="text-slate-500">Local Image Coordinates</span>
                  )}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 self-end sm:self-center">
            <label className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white cursor-pointer transition-all">
              <RefreshCw size={13} />
              <span>Replace</span>
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,.geotiff"
                className="hidden"
                onChange={onFileSelected}
                disabled={disabled || uploading}
              />
            </label>
            <button
              type="button"
              onClick={onClearImage}
              disabled={disabled || uploading}
              className="rounded-xl border border-rose-500/20 bg-rose-500/10 hover:bg-rose-500/20 px-3 py-1.5 text-xs font-medium text-rose-300 hover:text-rose-200 cursor-pointer transition-all disabled:opacity-50"
            >
              Clear
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={`relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-all ${
          isDragging
            ? "border-cyan-400 bg-cyan-500/[0.07]"
            : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
        } ${disabled ? "pointer-events-none opacity-50" : ""}`}
      >
        <input
          type="file"
          id="geovision-file-input"
          accept=".png,.jpg,.jpeg,.tif,.tiff,.geotiff"
          className="hidden"
          onChange={onFileSelected}
          disabled={disabled || uploading}
        />

        {uploading ? (
          <div className="flex flex-col items-center gap-3 py-4">
            <Loader2 size={36} className="animate-spin text-cyan-400" />
            <p className="text-sm font-medium text-slate-200">
              Validating & ingesting remote-sensing raster...
            </p>
            <p className="text-xs text-slate-400">
              Checking bounds, color channels, and spatial metadata
            </p>
          </div>
        ) : (
          <label
            htmlFor="geovision-file-input"
            className="flex flex-col items-center gap-3 cursor-pointer py-2"
          >
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300 shadow-lg shadow-cyan-950/40 group-hover:scale-105 transition-transform">
              <FileUp size={28} />
            </div>

            <div>
              <p className="text-base font-semibold text-white">
                Upload Aerial or Drone Imagery
              </p>
              <p className="mt-1 text-xs text-slate-400 max-w-md">
                Drag and drop your high-resolution raster or click to browse. Supports{" "}
                <span className="text-cyan-300 font-mono">PNG</span>,{" "}
                <span className="text-cyan-300 font-mono">JPG</span>,{" "}
                <span className="text-cyan-300 font-mono">TIFF</span>, and{" "}
                <span className="text-cyan-300 font-mono">GeoTIFF</span> up to 50 MB.
              </p>
            </div>

            <div className="flex items-center gap-2 mt-1 rounded-full bg-white/5 border border-white/10 px-3 py-1 text-[11px] text-slate-300 font-medium">
              <span>Direct raster analysis</span>
              <span className="text-slate-600">•</span>
              <span>Local or Georeferenced</span>
            </div>
          </label>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
          <AlertCircle size={16} className="shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
