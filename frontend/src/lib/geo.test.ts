import { describe, expect, it } from "vitest";
import {
  aoiFromBbox,
  bboxFromAoi,
  claimTypeColor,
  claimTypeLabel,
  confidenceBand,
  detectionFillOpacity,
  normalizeBbox,
} from "@/lib/geo";

describe("geo helpers", () => {
  it("maps confidence to band labels", () => {
    expect(confidenceBand(0.8)).toBe("High");
    expect(confidenceBand(0.5)).toBe("Medium");
    expect(confidenceBand(0.2)).toBe("Low");
  });

  it("computes detection fill opacity", () => {
    expect(detectionFillOpacity(0.5)).toBeCloseTo(0.24);
  });

  it("normalizes bbox regardless of drag direction", () => {
    const forward = normalizeBbox(77.59, 12.97, 77.61, 12.99);
    const reverse = normalizeBbox(77.61, 12.99, 77.59, 12.97);
    expect(reverse).toEqual(forward);
  });

  it("produces closed polygon rings from both drag directions", () => {
    const forward = aoiFromBbox([77.59, 12.97, 77.61, 12.99]);
    const reverse = aoiFromBbox([77.61, 12.99, 77.59, 12.97]);
    expect(bboxFromAoi(forward)).toEqual(bboxFromAoi(reverse));
    const ring = forward.geometry.coordinates[0] as number[][];
    expect(ring[0]).toEqual(ring[ring.length - 1]);
  });

  it("assigns appropriate colors and labels to SIH 26167 specialist claim types", () => {
    expect(claimTypeColor("new_building")).toBe("#10b981");
    expect(claimTypeColor("demolished_building")).toBe("#ef4444");
    expect(claimTypeColor("modified_building")).toBe("#f59e0b");
    expect(claimTypeColor("unchanged_building")).toBe("#64748b");
    expect(claimTypeColor("water_gain")).toBe("#06b6d4");
    expect(claimTypeColor("vegetation_loss")).toBe("#b45309");

    expect(claimTypeLabel("new_building")).toBe("New building");
    expect(claimTypeLabel("demolished_building")).toBe("Demolished / removed building");
    expect(claimTypeLabel("modified_building")).toBe("Modified building");
    expect(claimTypeLabel("unchanged_building")).toBe("Unchanged building");
    expect(claimTypeLabel("water_gain")).toBe("Water expansion candidate");
    expect(claimTypeLabel("vegetation_loss")).toBe("Vegetation loss candidate");
  });
});
