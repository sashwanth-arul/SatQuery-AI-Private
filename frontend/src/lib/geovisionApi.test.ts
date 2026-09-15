import { describe, expect, it, vi, beforeEach } from "vitest";
import { geovisionApi } from "@/lib/geovisionApi";

describe("GeoVision API Client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("constructs correct image bytes URL", () => {
    const url = geovisionApi.getImageBytesUrl("img-sample-123");
    expect(url).toContain("/api/v1/geovision/img-sample-123/bytes");
  });

  it("uploads an image correctly", async () => {
    const mockResponse = {
      success: true,
      data: {
        image_id: "img-test-456",
        filename: "ortho.tif",
        format: "tiff",
        width: 1024,
        height: 1024,
        file_size_bytes: 3145728,
        georeferenced: true,
        crs: "EPSG:32643",
        bounds: [76.0, 12.0, 76.1, 12.1],
        preview_url: "/api/v1/geovision/img-test-456/bytes",
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const file = new File(["dummycontent"], "ortho.tif", { type: "image/tiff" });
    const res = await geovisionApi.uploadImage(file);

    expect(res.image_id).toBe("img-test-456");
    expect(res.width).toBe(1024);
    expect(res.georeferenced).toBe(true);
    expect(res.crs).toBe("EPSG:32643");
  });

  it("sends analyze request and receives response", async () => {
    const mockResponse = {
      success: true,
      data: {
        session_id: "sess-geo-789",
        image_id: "img-test-456",
        detected_intent: "scene_classification",
        answer: "Urban",
        is_one_word: true,
        confidence: null,
        confidence_available: false,
        detected_objects: [],
        object_summary: {},
        trace: [
          {
            step_index: 1,
            name: "Image uploaded",
            status: "completed",
            duration_ms: 1,
            model_or_provider: "storage_layer",
            details: "Retrieved image.",
            timestamp: "2026-09-15T00:00:00Z",
          },
        ],
        models_used: { pipeline: "geovision_phase1_baseline" },
        provider_status: {
          geochat: "ready_for_connection",
          yolo_detector: "ready_for_connection",
          sam2_segmenter: "ready_for_connection",
        },
        georeferenced: true,
        crs: "EPSG:32643",
        created_at: "2026-09-15T00:00:00Z",
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const res = await geovisionApi.analyzeImage({
      image_id: "img-test-456",
      query: "Is this urban or rural? Answer in one word.",
      intent: "scene_classification",
    });

    expect(res.session_id).toBe("sess-geo-789");
    expect(res.answer).toBe("Urban");
    expect(res.is_one_word).toBe(true);
    expect(res.provider_status.geochat).toBe("ready_for_connection");
  });

  it("fetches evaluation metrics report correctly", async () => {
    const mockReport = {
      success: true,
      data: {
        model_name: "SatQuery-Aerial-YOLOv8",
        model_version: "v1.2.0-aerial-finetuned",
        dataset_name: "SatQuery-Aerial-Benchmark",
        dataset_version: "v2.0",
        global_metrics: {
          precision: 0.842,
          recall: 0.798,
          map50: 0.826,
          map50_95: 0.614,
        },
        per_class_metrics: {
          building: { precision: 0.881, recall: 0.854, map50: 0.872, map50_95: 0.682 },
        },
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockReport,
    });

    const metrics = await geovisionApi.getEvaluationMetrics();
    expect(metrics.model_name).toBe("SatQuery-Aerial-YOLOv8");
    expect(metrics.global_metrics.map50).toBe(0.826);
    expect(metrics.per_class_metrics.building.precision).toBe(0.881);
  });
});
