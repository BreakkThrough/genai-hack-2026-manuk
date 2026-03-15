/**
 * TypeScript types matching the Python Pydantic schemas
 */

export type HoleType = 
  | "simple" 
  | "counterbore" 
  | "countersink" 
  | "threaded" 
  | "cross_drilled" 
  | "through" 
  | "blind";

export type MatchConfidence = "high" | "medium" | "low";

export interface BoundingBox {
  page: number;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface DITextElement {
  content: string;
  confidence: number;
  bounding_box: BoundingBox | null;
}

export interface DIPageResult {
  page_number: number;
  width_px: number;
  height_px: number;
  elements: DITextElement[];
}

export interface DIExtractionResult {
  source_pdf: string;
  pages: DIPageResult[];
}

export interface ThreadSpec {
  designation: string;
  pitch: number | null;
  tolerance_class: string | null;
}

export interface HoleAnnotation {
  annotation_id: string;
  hole_type: HoleType;
  count: number;
  diameter: number | null;
  diameter_tolerance_plus: number | null;
  diameter_tolerance_minus: number | null;
  depth: number | null;
  depth_tolerance: number | null;
  thread_spec: ThreadSpec | null;
  counterbore_diameter: number | null;
  counterbore_depth: number | null;
  countersink_diameter: number | null;
  countersink_angle: number | null;
  position_tolerance: number | null;
  datum_refs: string[];
  fit_designation: string | null;
  raw_text: string;
  bounding_box: BoundingBox | null;
  page: number;
  confidence: number;
}

export interface DrawingAnnotations {
  source_pdf: string;
  unit: string;
  annotations: HoleAnnotation[];
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface Vector3D {
  dx: number;
  dy: number;
  dz: number;
}

export interface CylindricalFeature {
  feature_id: string;
  diameter: number;
  radius: number;
  depth: number | null;
  center: Point3D;
  axis: Vector3D;
  is_through: boolean;
  face_index: number;
}

export interface HoleFeature3D {
  hole_id: string;
  hole_type: HoleType;
  primary_diameter: number;
  primary_depth: number | null;
  counterbore_diameter: number | null;
  counterbore_depth: number | null;
  center: Point3D;
  axis: Vector3D;
  is_through: boolean;
  cylinders: CylindricalFeature[];
}

export interface StepFeatures {
  source_step: string;
  total_cylindrical_faces: number;
  holes: HoleFeature3D[];
}

export interface EvidenceTrace {
  page: number;
  bounding_box: BoundingBox | null;
  raw_text: string;
  view_reference: string | null;
}

export interface FeatureMapping {
  annotation_id: string;
  hole_ids: string[];
  confidence: MatchConfidence;
  confidence_score: number;
  match_reasons: string[];
  evidence: EvidenceTrace | null;
  parsed_interpretation: Record<string, any> | null;
}

export interface LinkageResult {
  drawing_pdf: string;
  step_file: string;
  annotations: DrawingAnnotations;
  features_3d: StepFeatures;
  mappings: FeatureMapping[];
  unmapped_annotations: string[];
  unmapped_holes: string[];
}

// API Response types
export interface UploadResponse {
  session_id: string;
  pdf_filename: string | null;
  step_filename: string | null;
  unit: string;
  step_features: StepFeatures | null;
}

export interface Layer1Response {
  status: string;
  page_count: number;
  element_count: number;
  sample_elements: Array<{
    text: string;
    confidence: number;
    bbox: BoundingBox | null;
  }>;
  error: string | null;
}

export interface Layer2Response {
  status: string;
  annotations: DrawingAnnotations | null;
  annotation_count: number;
  error: string | null;
}

export interface Layer3Response {
  status: string;
  step_features: StepFeatures | null;
  hole_count: number;
  error: string | null;
}

export interface Layer4Response {
  status: string;
  linkage_result: LinkageResult | null;
  mapping_count: number;
  error: string | null;
}

export interface SessionStatus {
  session_id: string;
  pdf_uploaded: boolean;
  step_uploaded: boolean;
  unit: string;
  layer1_complete: boolean;
  layer2_complete: boolean;
  layer3_complete: boolean;
  layer4_complete: boolean;
}

export type PipelineStep = 1 | 2 | 3 | 4;

export type StepStatus = "pending" | "running" | "completed" | "error";
