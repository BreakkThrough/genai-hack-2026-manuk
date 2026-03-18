"""Pydantic data models for the drawing-to-3D hole feature linking pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Bounding box (normalised 0-1 coordinates from Azure DI)
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Axis-aligned rectangle expressed as page-relative coordinates."""
    page: int = Field(..., description="1-based page number")
    x_min: float = Field(..., ge=0)
    y_min: float = Field(..., ge=0)
    x_max: float = Field(..., le=1)
    y_max: float = Field(..., le=1)


# ---------------------------------------------------------------------------
# Raw OCR element from Azure Document Intelligence
# ---------------------------------------------------------------------------

class DITextElement(BaseModel):
    """Single text span returned by the Azure DI layout model."""
    content: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    bounding_box: Optional[BoundingBox] = None


class DIPageResult(BaseModel):
    """All text elements extracted from one PDF page."""
    page_number: int
    width_px: float
    height_px: float
    elements: list[DITextElement] = Field(default_factory=list)


class DIExtractionResult(BaseModel):
    """Complete Azure DI extraction output for a PDF."""
    source_pdf: str
    pages: list[DIPageResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Hole-related annotation (enriched by GPT-4o vision)
# ---------------------------------------------------------------------------

class HoleType(str, Enum):
    SIMPLE = "simple"
    COUNTERBORE = "counterbore"
    COUNTERSINK = "countersink"
    THREADED = "threaded"
    CROSS_DRILLED = "cross_drilled"
    THROUGH = "through"
    BLIND = "blind"


class ThreadSpec(BaseModel):
    """ISO metric thread callout, e.g. M3 x 0.5 - 6g."""
    designation: str = Field(..., description="e.g. 'M3'")
    pitch: Optional[float] = Field(None, description="mm, e.g. 0.5")
    tolerance_class: Optional[str] = Field(None, description="e.g. '6g', '6H'")


class HoleAnnotation(BaseModel):
    """A single hole-related annotation group extracted from the drawing."""
    annotation_id: str = Field(..., description="Unique id within the drawing")
    hole_type: HoleType = HoleType.SIMPLE
    count: int = Field(default=1, description="Multiplicity, e.g. 4 for '4X'")
    diameter: Optional[float] = Field(None, description="Nominal diameter (drawing units)")
    diameter_tolerance_plus: Optional[float] = None
    diameter_tolerance_minus: Optional[float] = None
    depth: Optional[float] = None
    depth_tolerance: Optional[float] = None
    thread_spec: Optional[ThreadSpec] = None
    counterbore_diameter: Optional[float] = None
    counterbore_depth: Optional[float] = None
    countersink_diameter: Optional[float] = None
    countersink_angle: Optional[float] = None
    position_tolerance: Optional[float] = None
    datum_refs: list[str] = Field(default_factory=list)
    fit_designation: Optional[str] = Field(None, description="e.g. 'G6', 'H7'")
    raw_text: str = Field(default="", description="Original annotation text")
    bounding_box: Optional[BoundingBox] = None
    page: int = Field(default=1)
    confidence: float = Field(default=1.0, ge=0, le=1)


class DrawingAnnotations(BaseModel):
    """All hole annotations extracted from a single drawing PDF."""
    source_pdf: str
    unit: str = Field(default="inch", description="'inch' or 'mm'")
    annotations: list[HoleAnnotation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 3D cylindrical feature from STEP file
# ---------------------------------------------------------------------------

class Point3D(BaseModel):
    x: float
    y: float
    z: float


class Vector3D(BaseModel):
    dx: float
    dy: float
    dz: float


class CylindricalFeature(BaseModel):
    """A single cylindrical surface extracted from the STEP model."""
    feature_id: str
    diameter: float
    radius: float
    depth: Optional[float] = None
    center: Point3D
    axis: Vector3D
    is_through: bool = False
    face_index: int = Field(..., description="Index in the STEP face list")


class HoleFeature3D(BaseModel):
    """
    A logical hole composed of one or more co-axial cylindrical surfaces.
    E.g. a counterbore is two co-axial cylinders with different diameters.
    """
    hole_id: str
    hole_type: HoleType = HoleType.SIMPLE
    primary_diameter: float
    primary_depth: Optional[float] = None
    counterbore_diameter: Optional[float] = None
    counterbore_depth: Optional[float] = None
    center: Point3D
    axis: Vector3D
    is_through: bool = False
    cylinders: list[CylindricalFeature] = Field(default_factory=list)


class StepFeatures(BaseModel):
    """All hole features extracted from a STEP file."""
    source_step: str
    total_cylindrical_faces: int = 0
    holes: list[HoleFeature3D] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Annotation ↔ 3D Feature mapping
# ---------------------------------------------------------------------------

class MatchConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceTrace(BaseModel):
    """Traceability link back to the source drawing region."""
    page: int = Field(..., description="1-based page number where annotation was found")
    bounding_box: Optional[BoundingBox] = Field(
        None, description="Bounding box of the annotation on the drawing page"
    )
    raw_text: str = Field(default="", description="Verbatim annotation text from the drawing")
    view_reference: Optional[str] = Field(
        None, description="Drawing view name (e.g. 'SECTION A-A', 'DETAIL B')"
    )


class FeatureMapping(BaseModel):
    """Links one drawing annotation to one or more 3D hole features."""
    annotation_id: str
    hole_ids: list[str] = Field(default_factory=list)
    matched_cylinder_ids: list[str] = Field(
        default_factory=list,
        description="Specific cylinder feature IDs within the matched holes "
        "that the annotation diameter matched against"
    )
    confidence: MatchConfidence = MatchConfidence.MEDIUM
    confidence_score: float = Field(
        default=0.5, ge=0, le=1,
        description="Numeric confidence 0-1 (1 = certain match)"
    )
    diameter_delta: Optional[float] = Field(
        None,
        description="Absolute difference between annotation diameter and "
        "best-matched cylinder diameter (drawing units)"
    )
    match_reasons: list[str] = Field(default_factory=list)
    evidence: Optional[EvidenceTrace] = Field(
        None, description="Link to the drawing region where the annotation was found"
    )
    parsed_interpretation: Optional[dict] = Field(
        None,
        description="Structured interpretation of the annotation "
        "(thread type, size, pitch, tolerance class, depth, etc.)"
    )


class FeatureGroup(BaseModel):
    """All annotations that belong to a single 3D hole feature."""
    feature_id: str = Field(..., description="hole_id from features_3d")
    hole_type: str = Field(..., description="Type of the 3D feature")
    annotation_ids: list[str] = Field(
        default_factory=list,
        description="Annotation IDs that mapped to this feature"
    )
    matched_cylinder_ids: list[str] = Field(
        default_factory=list,
        description="Cylinder IDs within this feature that were matched"
    )
    confidence: MatchConfidence = MatchConfidence.MEDIUM


class LinkageResult(BaseModel):
    """Complete output of the pipeline: annotations, 3D features, and mappings."""
    drawing_pdf: str
    step_file: str
    annotations: DrawingAnnotations
    features_3d: StepFeatures
    mappings: list[FeatureMapping] = Field(default_factory=list)
    feature_groups: list[FeatureGroup] = Field(
        default_factory=list,
        description="Grouped view: all annotations belonging to each 3D feature"
    )
    unmapped_annotations: list[str] = Field(default_factory=list)
    unmapped_holes: list[str] = Field(default_factory=list)
