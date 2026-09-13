import type { AOI, GeoJSONGeometry } from "@/types/domain";

/** Default demo AOI near Bengaluru for first load. */
export const DEFAULT_AOI: AOI = {
  geometry: {
    type: "Polygon",
    coordinates: [
      [
        [77.59, 12.97],
        [77.61, 12.97],
        [77.61, 12.99],
        [77.59, 12.99],
        [77.59, 12.97],
      ],
    ],
  },
  area_km2: 4.8,
};

export function bboxFromAoi(aoi: AOI): [number, number, number, number] {
  const ring = (aoi.geometry.coordinates as number[][][])[0];
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
}

export function normalizeBbox(
  minLon: number,
  minLat: number,
  maxLon: number,
  maxLat: number,
): [number, number, number, number] {
  return [Math.min(minLon, maxLon), Math.min(minLat, maxLat), Math.max(minLon, maxLon), Math.max(minLat, maxLat)];
}

export function aoiFromBbox(bbox: [number, number, number, number]): AOI {
  const [minLon, minLat, maxLon, maxLat] = normalizeBbox(bbox[0], bbox[1], bbox[2], bbox[3]);
  const geometry: GeoJSONGeometry = {
    type: "Polygon",
    coordinates: [
      [
        [minLon, minLat],
        [maxLon, minLat],
        [maxLon, maxLat],
        [minLon, maxLat],
        [minLon, minLat],
      ],
    ],
  };
  const width = (maxLon - minLon) * 111 * Math.cos(((minLat + maxLat) / 2) * (Math.PI / 180));
  const height = (maxLat - minLat) * 111;
  return {
    geometry,
    area_km2: Math.round(width * height * 100) / 100,
  };
}

export const ACCENT_COLOR = "#8b5cf6";

/** Reads --accent from CSS so map layers stay in sync with the design system. */
export function getAccentColor(): string {
  if (typeof document === "undefined") return ACCENT_COLOR;
  const value = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  return value || ACCENT_COLOR;
}

export function claimTypeColor(claimType: string | undefined): string {
  switch (claimType) {
    case "urban_expansion_candidate":
    case "built_up_change":
      return "#e07b39";
    case "vegetation_loss_candidate":
    case "vegetation_loss":
      return "#b45309";
    case "vegetation_gain_candidate":
    case "vegetation_gain":
      return "#16a34a";
    case "water_shrinkage_candidate":
    case "water_loss":
      return "#0284c7";
    case "water_expansion_candidate":
    case "water_gain":
      return "#06b6d4";
    case "infrastructure_change_candidate":
      return "#9b7bd4";
    case "mining_change_candidate":
      return "#a67c52";
    case "construction_candidate":
    case "new_built_area":
      return getAccentColor();
    case "new_building":
      return "#10b981";
    case "demolished_building":
      return "#ef4444";
    case "modified_building":
      return "#f59e0b";
    case "unchanged_building":
      return "#64748b";
    case "building_footprint":
      return "#06b6d4";
    case "grounded_target":
      return "#ec4899";
    default:
      return getAccentColor();
  }
}

export function claimTypeLabel(claimType: string | undefined): string | null {
  switch (claimType) {
    case "urban_expansion_candidate":
      return "Urban expansion candidate";
    case "built_up_change":
      return "Built-up change";
    case "vegetation_loss_candidate":
    case "vegetation_loss":
      return "Vegetation loss candidate";
    case "vegetation_gain_candidate":
    case "vegetation_gain":
      return "Vegetation gain candidate";
    case "water_shrinkage_candidate":
    case "water_loss":
      return "Water shrinkage candidate";
    case "water_expansion_candidate":
    case "water_gain":
      return "Water expansion candidate";
    case "infrastructure_change_candidate":
      return "Infrastructure change candidate";
    case "mining_change_candidate":
      return "Mining change candidate";
    case "construction_candidate":
      return "Construction candidate";
    case "new_built_area":
      return "New built area";
    case "new_building":
      return "New building";
    case "demolished_building":
      return "Demolished / removed building";
    case "modified_building":
      return "Modified building";
    case "unchanged_building":
      return "Unchanged building";
    case "building_footprint":
      return "Building footprint";
    case "grounded_target":
      return "Grounded target";
    default:
      return null;
  }
}

export function confidenceBand(confidence: number): string {
  if (confidence >= 0.75) return "High";
  if (confidence >= 0.45) return "Medium";
  return "Low";
}

export function detectionFillOpacity(confidence: number): number {
  return 0.08 + 0.32 * confidence;
}
