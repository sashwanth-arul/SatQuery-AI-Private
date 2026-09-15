import type {
  GeoVisionAnalyzeRequest,
  GeoVisionAnalyzeResponse,
  GeoVisionEvaluationReport,
  GeoVisionUploadResponse,
} from "@/types/geovision";
import { parseHttpErrorBody, API_ERROR_MESSAGES } from "@/lib/errors";
import { ApiError } from "@/types/domain";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: { code: string; message: string; details?: unknown };
}

async function readJsonBody(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    if (res.ok) {
      throw new ApiError(
        "invalid_response",
        "Invalid JSON",
        API_ERROR_MESSAGES.invalid_response,
      );
    }
    throw parseHttpErrorBody(res.status, null);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  const body = await readJsonBody(res);
  if (!res.ok) {
    throw parseHttpErrorBody(res.status, body);
  }
  return (body as ApiResponse<T>).data;
}

export const geovisionApi = {
  getImageBytesUrl: (imageId: string): string => {
    return `${API_BASE}/api/v1/geovision/${encodeURIComponent(imageId)}/bytes`;
  },

  getImageMetadata: (imageId: string): Promise<GeoVisionUploadResponse> => {
    return request<GeoVisionUploadResponse>(
      `/api/v1/geovision/${encodeURIComponent(imageId)}`,
    );
  },

  uploadImage: async (file: File): Promise<GeoVisionUploadResponse> => {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${API_BASE}/api/v1/geovision/upload`, {
      method: "POST",
      body: form,
    });

    const body = await readJsonBody(res);
    if (!res.ok) {
      throw parseHttpErrorBody(res.status, body);
    }
    return (body as ApiResponse<GeoVisionUploadResponse>).data;
  },

  analyzeImage: (payload: GeoVisionAnalyzeRequest): Promise<GeoVisionAnalyzeResponse> => {
    return request<GeoVisionAnalyzeResponse>("/api/v1/geovision/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  getEvaluationMetrics: (): Promise<GeoVisionEvaluationReport> => {
    return request<GeoVisionEvaluationReport>("/api/v1/geovision/evaluation-metrics");
  },
};
