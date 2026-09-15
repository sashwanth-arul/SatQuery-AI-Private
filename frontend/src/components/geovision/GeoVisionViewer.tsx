"use client";

import { useEffect, useRef, useState } from "react";
import {
  Compass,
  Crosshair,
  Info,
  Layers,
  Maximize,
  Minimize,
  RotateCcw,
  Sparkles,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { DetectedObject, GeoVisionUploadResponse } from "@/types/geovision";
import { geovisionApi } from "@/lib/geovisionApi";

interface GeoVisionViewerProps {
  image: GeoVisionUploadResponse;
  detectedObjects?: DetectedObject[];
}

export function GeoVisionViewer({
  image,
  detectedObjects = [],
}: GeoVisionViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [fitScale, setFitScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [hoveredObject, setHoveredObject] = useState<DetectedObject | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Compute container fit scale so image is cleanly contained without overflow
  const updateFitScale = () => {
    if (!containerRef.current || image.width <= 0 || image.height <= 0) return;
    const { clientWidth, clientHeight } = containerRef.current;
    if (clientWidth <= 0 || clientHeight <= 0) return;

    const pad = 32;
    const scaleX = (clientWidth - pad) / image.width;
    const scaleY = (clientHeight - pad) / image.height;
    const calculatedFit = Math.min(scaleX, scaleY, 1.0);

    setFitScale(calculatedFit);
    setScale(calculatedFit);
    setOffset({ x: 0, y: 0 });
  };

  useEffect(() => {
    updateFitScale();
    setImageLoaded(false);
    setHoveredObject(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image.image_id, image.width, image.height]);

  useEffect(() => {
    const onResize = () => updateFitScale();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image.width, image.height]);

  const handleZoomIn = () => setScale((s) => Math.min(s * 1.3, 10));
  const handleZoomOut = () => setScale((s) => Math.max(s / 1.3, fitScale * 0.2));
  const handleReset = () => {
    setScale(fitScale);
    setOffset({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
    }

    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      // Coordinate relative to center of image
      const mouseRelX = (e.clientX - centerX - offset.x) / scale;
      const mouseRelY = (e.clientY - centerY - offset.y) / scale;

      const px = mouseRelX + image.width / 2;
      const py = mouseRelY + image.height / 2;

      if (px >= 0 && px <= image.width && py >= 0 && py <= image.height) {
        setCursorPos({ x: Math.round(px), y: Math.round(py) });
      } else {
        setCursorPos(null);
      }
    }
  };

  const handleMouseUp = () => setIsPanning(false);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    setScale((s) => Math.min(Math.max(s * factor, fitScale * 0.2), 10));
  };

  const imageUrl = geovisionApi.getImageBytesUrl(image.image_id);

  // SatQuery color tokens per category
  const getObjectColor = (className: string) => {
    const c = className.toLowerCase();
    if (c.includes("building") || c.includes("structure")) return "#06b6d4"; // cyan
    if (c.includes("house")) return "#3b82f6"; // blue
    if (c.includes("car")) return "#f43f5e"; // rose
    if (c.includes("truck")) return "#fb923c"; // orange
    if (c.includes("bus")) return "#eab308"; // yellow
    if (c.includes("motorcycle")) return "#ec4899"; // pink
    if (c.includes("aircraft")) return "#8b5cf6"; // purple
    if (c.includes("boat")) return "#6366f1"; // indigo
    if (c.includes("person")) return "#10b981"; // emerald
    return "#a855f7"; // purple
  };

  return (
    <div
      ref={containerRef}
      className="relative flex flex-col h-full w-full overflow-hidden rounded-3xl border border-white/10 bg-[#07090f] shadow-2xl select-none"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        setIsPanning(false);
        setCursorPos(null);
      }}
      onWheel={handleWheel}
    >
      {/* Top Floating Status & Toolbar */}
      <div className="absolute top-3 left-3 right-3 z-30 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 pointer-events-auto rounded-xl bg-slate-950/80 backdrop-blur-md border border-white/10 px-3 py-1.5 text-xs text-slate-300 shadow-xl">
          <span className="font-semibold text-white truncate max-w-[180px]">
            {image.filename}
          </span>
          <span className="text-slate-600">|</span>
          <span className="font-mono text-slate-400">
            {image.width} × {image.height} px
          </span>
          {image.crs && (
            <>
              <span className="text-slate-600">|</span>
              <span className="font-mono text-cyan-400 text-[11px]">{image.crs}</span>
            </>
          )}
        </div>

        {/* View Controls */}
        <div className="flex items-center gap-1 pointer-events-auto rounded-xl bg-slate-950/80 backdrop-blur-md border border-white/10 p-1 shadow-xl">
          {detectedObjects.length > 0 && (
            <button
              type="button"
              onClick={() => setShowBoxes((v) => !v)}
              title={showBoxes ? "Hide Detections" : "Show Detections"}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                showBoxes
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Layers size={13} />
              <span>{detectedObjects.length} Verified</span>
            </button>
          )}

          <button
            type="button"
            onClick={handleZoomIn}
            title="Zoom In"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <ZoomIn size={15} />
          </button>
          <button
            type="button"
            onClick={handleZoomOut}
            title="Zoom Out"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <ZoomOut size={15} />
          </button>
          <button
            type="button"
            onClick={handleReset}
            title="Reset View"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <RotateCcw size={14} />
          </button>
          <span className="text-[11px] font-mono text-slate-400 px-2">
            {Math.round((scale / fitScale) * 100)}%
          </span>
        </div>
      </div>

      {/* Viewport Interactive Transform Plane */}
      <div
        className={`relative flex-1 w-full h-full flex items-center justify-center ${
          isPanning ? "cursor-grabbing" : "cursor-grab"
        }`}
      >
        {/* Subtle grid pattern */}
        <div
          className="absolute inset-0 pointer-events-none opacity-20"
          style={{
            backgroundImage:
              "radial-gradient(circle, rgba(255,255,255,0.15) 1px, transparent 1px)",
            backgroundSize: "28px 28px",
          }}
        />

        {/* Scaled & Translated Layer */}
        <div
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            transformOrigin: "center center",
            transition: isPanning ? "none" : "transform 0.08s ease-out",
            width: `${image.width}px`,
            height: `${image.height}px`,
          }}
          className="relative shrink-0 shadow-2xl rounded-md ring-1 ring-white/10 overflow-hidden"
        >
          {/* Main Raster Image */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageUrl}
            alt={image.filename}
            draggable={false}
            onLoad={() => setImageLoaded(true)}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
              display: "block",
              imageRendering: scale > 2 ? "pixelated" : "auto",
            }}
          />

          {/* Precision Annotation SVG Overlay */}
          {showBoxes && detectedObjects.length > 0 && (
            <svg
              className="absolute inset-0 pointer-events-none"
              width={image.width}
              height={image.height}
              viewBox={`0 0 ${image.width} ${image.height}`}
            >
              {detectedObjects.map((obj) => {
                const [x1, y1, x2, y2] = obj.bbox;
                const width = Math.max(x2 - x1, 2);
                const height = Math.max(y2 - y1, 2);
                const isHovered = hoveredObject?.id === obj.id;
                const color = getObjectColor(obj.class_name);
                const confPercent = (obj.confidence * 100).toFixed(1);

                // Rounded box scale factors
                const strokeW = isHovered ? 3.5 : 2.0;
                const labelW = Math.max(obj.class_name.length * 8.5 + 40, 72);
                const labelH = 26;

                return (
                  <g
                    key={obj.id}
                    className="pointer-events-auto cursor-pointer transition-opacity"
                    onMouseEnter={() => setHoveredObject(obj)}
                    onMouseLeave={() => setHoveredObject(null)}
                  >
                    {/* Bounding Box: Rendered as Rounded Rectangle (border-radius: 8px) */}
                    <rect
                      x={x1}
                      y={y1}
                      width={width}
                      height={height}
                      rx={8}
                      ry={8}
                      fill={color}
                      fillOpacity={isHovered ? 0.35 : 0.12}
                      stroke={color}
                      strokeWidth={strokeW}
                      className="transition-all"
                    />

                    {/* Integrated Label Badge showing [CLASS NAME] and Confidence % */}
                    <g transform={`translate(${x1}, ${Math.max(y1 - labelH - 3, 2)})`}>
                      {/* Label Background Pill */}
                      <rect
                        width={labelW}
                        height={labelH}
                        rx={6}
                        ry={6}
                        fill="#0c1017"
                        stroke={color}
                        strokeWidth={1.5}
                        fillOpacity={0.92}
                      />
                      {/* Class Name */}
                      <text
                        x={6}
                        y={11}
                        fill="#ffffff"
                        fontSize={9.5}
                        fontWeight="bold"
                        fontFamily="monospace"
                        letterSpacing="0.05em"
                      >
                        {obj.class_name.toUpperCase()}
                      </text>
                      {/* Confidence % */}
                      <text
                        x={6}
                        y={21}
                        fill={color}
                        fontSize={8.5}
                        fontWeight="bold"
                        fontFamily="monospace"
                      >
                        {confPercent}%
                      </text>
                    </g>
                  </g>
                );
              })}
            </svg>
          )}
        </div>
      </div>

      {/* Hover Object Detail Inspector Tooltip (Overlaying in Bottom-Right or Top-Right) */}
      {hoveredObject && (
        <div className="absolute top-16 right-4 z-40 w-64 rounded-2xl border border-cyan-500/30 bg-slate-950/90 p-3.5 shadow-2xl backdrop-blur-xl text-xs">
          <div className="flex items-center justify-between border-b border-white/10 pb-2 mb-2">
            <span className="font-bold text-white uppercase tracking-wider">
              {hoveredObject.class_name}
            </span>
            <span className="rounded-md bg-cyan-500/20 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300">
              {(hoveredObject.confidence * 100).toFixed(1)}% Conf
            </span>
          </div>
          <div className="space-y-1 font-mono text-[11px] text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-500">ID:</span>
              <span className="text-white">{hoveredObject.id}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Dimensions:</span>
              <span>
                {Math.round(hoveredObject.bbox[2] - hoveredObject.bbox[0])} ×{" "}
                {Math.round(hoveredObject.bbox[3] - hoveredObject.bbox[1])} px
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Center:</span>
              <span>
                [{hoveredObject.center?.[0] ?? 0}, {hoveredObject.center?.[1] ?? 0}]
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Model:</span>
              <span className="text-purple-300">{hoveredObject.source_model}</span>
            </div>
            {hoveredObject.tile_id && (
              <div className="flex justify-between">
                <span className="text-slate-500">Tile:</span>
                <span className="text-slate-400">{hoveredObject.tile_id}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Bottom HUD: Coordinates & Navigation Guide */}
      <div className="absolute bottom-3 left-3 right-3 z-30 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 pointer-events-auto rounded-xl bg-slate-950/80 backdrop-blur-md border border-white/10 px-3 py-1.5 text-[11px] font-mono text-slate-400 shadow-xl">
          <Crosshair size={13} className="text-cyan-400" />
          {cursorPos ? (
            <span>
              X: <strong className="text-slate-200">{cursorPos.x}</strong> Y:{" "}
              <strong className="text-slate-200">{cursorPos.y}</strong> px
            </span>
          ) : (
            <span>Hover image for coordinates</span>
          )}
        </div>

        <div className="flex items-center gap-2 pointer-events-auto rounded-xl bg-slate-950/80 backdrop-blur-md border border-white/10 px-3 py-1.5 text-[11px] font-mono text-slate-400 shadow-xl">
          <span>Pan: Drag | Zoom: Wheel</span>
        </div>
      </div>
    </div>
  );
}
