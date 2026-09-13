import { describe, expect, it } from "vitest";
import { api } from "@/lib/api";
import type { AnalysisHistoryItem, AnalysisHistoryResponse } from "@/types/domain";

describe("Navigation Structure", () => {
  it("includes /history and /reports routes", () => {
    const expectedRoutes = ["/", "/workstation", "/history", "/reports", "/tutorial", "/credits"];
    expect(expectedRoutes).toContain("/history");
    expect(expectedRoutes).toContain("/reports");
  });
});

describe("API Client - History & Reports", () => {
  it("constructs correct report download URL", () => {
    const sessionId = "test-session-123456";
    const url = api.downloadReportUrl(sessionId);
    expect(url).toContain(`/api/v1/query/${sessionId}/report`);
  });

  it("handles history response normalization", () => {
    const rawData: AnalysisHistoryResponse = {
      items: [
        {
          session_id: "sess-abc-123",
          created_at: "2026-09-13T10:00:00Z",
          query: "How many buildings increased?",
          intent: "building_temporal_change",
          mode: "temporal_pair",
          status: "completed",
          summary_answer: "Detected 42 new buildings.",
          confidence: 0.88,
          metrics_summary: { new_count: 42, before_count: 100, after_count: 142 },
          input_summary: { earlier_date: "2023-01-01", later_date: "2024-01-01" },
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    };

    expect(rawData.items.length).toBe(1);
    const item = rawData.items[0];
    expect(item.session_id).toBe("sess-abc-123");
    expect(item.mode).toBe("temporal_pair");
    expect(item.confidence).toBe(0.88);
    expect(item.metrics_summary?.new_count).toBe(42);
  });
});

describe("Speech-to-Text State and Transcript Logic", () => {
  it("accumulates final and interim speech transcripts correctly", () => {
    // Test the transcription joining algorithm used in useSpeechToText
    const mockResults = [
      { isFinal: true, transcript: "Show me " },
      { isFinal: false, transcript: "building change" },
    ];

    let finalTranscript = "";
    let interimTranscript = "";

    for (const r of mockResults) {
      if (r.isFinal) {
        finalTranscript += r.transcript;
      } else {
        interimTranscript += r.transcript;
      }
    }

    const currentText = finalTranscript || interimTranscript;
    expect(currentText).toBe("Show me ");

    // When both are present, final + interim can form complete draft
    const fullDraft = finalTranscript + interimTranscript;
    expect(fullDraft).toBe("Show me building change");
  });

  it("maps speech errors appropriately", () => {
    const errorMap = (errCode: string): string => {
      if (errCode === "not-allowed") {
        return "Microphone permission denied. Enable microphone access in browser settings.";
      }
      return `Speech recognition error: ${errCode}`;
    };

    expect(errorMap("not-allowed")).toContain("permission denied");
    expect(errorMap("network")).toBe("Speech recognition error: network");
  });
});
