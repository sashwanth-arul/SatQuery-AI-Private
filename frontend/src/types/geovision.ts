/** TypeScript domain contracts for GeoVision workspace.
 * Mirrored from backend/app/schemas/geovision.py
 */

export type GeoVisionIntent =
  | "automatic"
  | "single_image_description"
  | "visual_qa"
  | "object_count"
  | "object_detection"
  | "scene_classification"
  | "region_caption"
  | "referring_expression"
  | "grounded_description"
  | "detailed_description"
  | "complex_reasoning"
  | "multi_turn_conversation"
  | "object_attribute"
  | "object_relationship";

export interface DetectedObject {
  id: string;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  segmentation_mask?: number[][] | null;
  center?: [number, number] | null;
  area_px?: number | null;
  source_model: string;
  model_version?: string;
  tile_id?: string | null;
  coordinate_space?: string;
}

export interface GeoVisionTraceStep {
  step_index: number;
  name: string;
  status: "completed" | "running" | "skipped" | "unavailable" | "failed" | string;
  duration_ms: number;
  model_or_provider: string;
  details: string;
  timestamp: string;
}

export interface GeoVisionUploadResponse {
  image_id: string;
  filename: string;
  format: string;
  width: number;
  height: number;
  file_size_bytes: number;
  georeferenced: boolean;
  crs?: string | null;
  bounds?: number[] | null;
  preview_url: string;
}

export interface GeoVisionAnalyzeRequest {
  image_id: string;
  query: string;
  intent?: GeoVisionIntent;
  conversation_history?: Array<{ role: string; content: string }>;
}

export interface GeoVisionAnalyzeResponse {
  session_id: string;
  image_id: string;
  detected_intent: GeoVisionIntent;
  answer: string;
  is_one_word: boolean;
  confidence?: number | null;
  confidence_available: boolean;
  detected_objects: DetectedObject[];
  object_summary: Record<string, number>;
  trace: GeoVisionTraceStep[];
  models_used: Record<string, string>;
  provider_status: Record<string, string>;
  georeferenced: boolean;
  crs?: string | null;
  model_metadata?: Record<string, any> | null;
  evaluation_metrics?: Record<string, any> | null;
  created_at: string;
}

export interface GeoVisionEvaluationReport {
  model_name: string;
  model_version: string;
  dataset_name: string;
  dataset_version: string;
  training_date?: string;
  training_parameters?: Record<string, any>;
  dataset_splits?: {
    train_images: number;
    validation_images: number;
    test_images: number;
    total_annotated_instances?: number;
  };
  global_metrics: {
    precision: number;
    recall: number;
    f1_score?: number;
    map50: number;
    map50_95: number;
  };
  per_class_metrics: Record<
    string,
    {
      precision: number;
      recall: number;
      f1?: number;
      map50: number;
      map50_95: number;
      instances?: number;
    }
  >;
  confusion_summary?: {
    false_positive_rate?: number;
    false_negative_rate?: number;
    common_confusions?: Array<{ true: string; predicted: string; frequency: number }>;
  };
  hard_examples?: {
    identified_challenging_conditions?: string[];
    mitigations?: string[];
  };
}
