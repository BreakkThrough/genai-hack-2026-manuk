# Technical Architecture

## System Overview

A 4-layer pipeline that extracts hole annotations from PDF engineering drawings and links them to cylindrical features in 3D STEP models.

```
PDF Drawing → [Layer 1: OCR] → [Layer 2: Vision AI] → [Layer 4: Correlation] → JSON Output
STEP Model  → [Layer 3: Geometry Parser] ────────────────────┘
```

---

## Layer 1: Azure Document Intelligence (OCR)

**Module:** `app/extraction/di_extractor.py`

**Purpose:** Extract text elements with spatial bounding boxes from PDF pages.

**Input:**
- PDF file (binary)

**Output:** `DIExtractionResult`
```python
{
  "source_pdf": "path/to/drawing.pdf",
  "pages": [
    {
      "page_number": 1,
      "width_px": 2550.0,
      "height_px": 3300.0,
      "elements": [
        {
          "content": "4X ∅.281 ±.008",
          "confidence": 0.98,
          "bounding_box": {
            "page": 1,
            "x_min": 0.45, "y_min": 0.32,
            "x_max": 0.52, "y_max": 0.35
          }
        }
      ]
    }
  ]
}
```

**Key Functions:**
- `extract_layout(pdf_path)` → calls Azure DI prebuilt-layout API
- `_polygon_to_bbox()` → converts Azure's polygon format to normalized bounding boxes

**Graceful Degradation:**
If Azure DI fails, the pipeline continues in vision-only mode (skips cropped-region pass).

---

## Layer 2: Vision AI Enrichment

**Module:** `app/extraction/vision_enricher.py`

**Purpose:** Extract structured hole annotations from drawing images using GPT-4o vision model.

**Input:**
- High-resolution page images (300 DPI)
- OCR text context from Layer 1
- Drawing units (`"inch"` or `"mm"`)

**Output:** `DrawingAnnotations`
```python
{
  "source_pdf": "drawing.pdf",
  "unit": "inch",
  "annotations": [
    {
      "annotation_id": "H1",
      "hole_type": "counterbore",
      "count": 4,
      "diameter": 0.281,
      "diameter_tolerance_plus": 0.008,
      "diameter_tolerance_minus": 0.008,
      "counterbore_diameter": 0.562,
      "counterbore_depth": 0.312,
      "position_tolerance": 0.010,
      "datum_refs": ["A", "B", "C"],
      "raw_text": "4X ∅.281 ±.008 THRU ↧∅.562 X .312 DEEP",
      "page": 1,
      "confidence": 0.85
    }
  ]
}
```

### Multi-Pass Extraction Strategy

The vision layer runs **6 sequential passes** to maximize recall:

#### Pass 1: OCR Regex Pre-Scan
- Scans OCR text with regex patterns for diameter callouts
- Patterns: `NX ∅D`, `∅D`, `M10 x 1.5` (threads)
- Creates a checklist of expected diameters
- **Function:** `_detect_ocr_diameters()`

#### Pass 2: Per-Page Extraction
- Sends each page image + OCR context + diameter hints to GPT-4o
- Uses structured prompt with extraction rules
- **Function:** `_call_vision()`

#### Pass 3: Verification Pass
- Re-examines all pages with list of already-found annotations
- Asks model to find any missed holes
- **Function:** `_verify_and_supplement()`

#### Pass 4: Cropped-Region Pass
- Identifies dense annotation clusters from DI bounding boxes
- Crops regions with padding, sends zoomed-in images to GPT-4o
- Catches small callouts missed in full-page images
- **Function:** `_call_vision_cropped()` + `crop_annotation_regions()` (in `pdf_utils.py`)

#### Pass 5: OCR Reconciliation
- Compares OCR-detected diameters vs. LLM-extracted annotations
- Triggers targeted follow-up for missing diameters
- **Function:** `_reconcile_ocr_vs_llm()`

#### Pass 6: Filtering & Deduplication
- Removes null/tiny diameters, non-hole features (linear dims, slots, spherical)
- 3-stage dedup: within-page → cross-page → diameter-based
- Reassigns sequential IDs (H1, H2, ...)
- **Functions:** `_filter_null_diameters()`, `_filter_non_hole_annotations()`, `_deduplicate_*()`, `_reassign_ids()`

### Key Prompts

**System Prompt:** Defines model role, JSON schema, extraction rules
- Only extract cylindrical holes (not linear dims, surface tolerances, slots, spheres)
- Every entry must have non-null diameter
- Counterbores = single object with both diameters
- Don't merge distinct callouts with different GD&T

**User Prompt Template:** Per-page extraction checklist
- Scan for: `NX ∅D`, `∅D THRU`, threads, counterbore stacks, fit designations
- Includes OCR context and diameter hints

**Verification Prompt:** Second-pass sweep asking for missed annotations

**Cropped Region Prompt:** Focused extraction on zoomed annotation clusters

**Reconciliation Prompt:** Targeted query for specific missing diameters

### Model API Handling

**Function:** `_build_messages()`, `_completion_kwargs()`

Handles differences between GPT and o-series reasoning models:
- **GPT models:** Accept `system` role, `temperature`, `max_tokens`
- **o-series models:** No `system` role (merged into user), `max_completion_tokens` only

---

## Layer 3: STEP File Parsing

**Module:** `app/extraction/step_parser.py`

**Purpose:** Extract cylindrical hole features from STEP (ISO 10303-21) 3D CAD files using pure Python text parsing.

**Input:**
- STEP file (`.stp` or `.step` text format)

**Output:** `StepFeatures`
```python
{
  "source_step": "model.stp",
  "total_cylindrical_faces": 24,
  "holes": [
    {
      "hole_id": "hole_0",
      "hole_type": "counterbore",
      "primary_diameter": 0.281,
      "primary_depth": 1.125,
      "counterbore_diameter": 0.562,
      "counterbore_depth": 0.312,
      "center": {"x": 2.5, "y": 1.0, "z": 0.0},
      "axis": {"dx": 0.0, "dy": 0.0, "dz": 1.0},
      "is_through": false,
      "cylinders": [...]  // Raw cylindrical surfaces
    }
  ]
}
```

### Parsing Pipeline

#### Step 1: Entity Extraction
- Parses STEP text with regex: `#123 = ENTITY_TYPE(args);`
- Builds entity lookup table: `{entity_id: (type, args)}`
- **Function:** `_parse_step_entities()`

#### Step 2: Resolve Reference Chain
For each `CYLINDRICAL_SURFACE`:
```
CYLINDRICAL_SURFACE(#456, radius)
  ↓
AXIS2_PLACEMENT_3D(#789, #101, ...)
  ↓
CARTESIAN_POINT(x, y, z)  +  DIRECTION(dx, dy, dz)
```

**Functions:**
- `_resolve_axis2_placement()` → (center point, axis direction)
- `_resolve_cartesian_point()` → `Point3D`
- `_resolve_direction()` → `Vector3D`

#### Step 3: Extract Cylinders
- Finds all `CYLINDRICAL_SURFACE` entities
- Resolves radius, center, axis for each
- **Function:** `_extract_cylinders()`

#### Step 4: Estimate Depths
- Analyzes `CIRCLE` entities on same axis
- Distance between two circles with matching radius = cylinder depth
- **Function:** `_estimate_depths()`

#### Step 5: Group Co-axial Cylinders
- Groups cylinders with parallel axes and coincident centers
- Smallest diameter = primary hole
- Larger co-axial cylinder = counterbore
- **Function:** `_group_into_holes()`
- **Helper:** `axes_coaxial()` in `geometry_utils.py`

**Co-axial Detection:**
Two cylinders are co-axial if:
1. Axes are parallel (within 5° tolerance)
2. Perpendicular distance between axes < 0.5 units

---

## Layer 4: Correlation & Matching

**Module:** `app/correlation/matcher.py`

**Purpose:** Link drawing annotations to 3D hole features using hybrid deterministic + LLM matching.

**Input:**
- `DrawingAnnotations` (from Layer 2)
- `StepFeatures` (from Layer 3)

**Output:** `LinkageResult`
```python
{
  "drawing_pdf": "drawing.pdf",
  "step_file": "model.stp",
  "annotations": {...},  // Full Layer 2 output
  "features_3d": {...},  // Full Layer 3 output
  "mappings": [
    {
      "annotation_id": "H1",
      "hole_ids": ["hole_0", "hole_1", "hole_2", "hole_3"],
      "confidence": "high",
      "confidence_score": 0.95,
      "match_reasons": [
        "diameter match",
        "count filter 4X"
      ],
      "evidence": {
        "page": 1,
        "bounding_box": {...},
        "raw_text": "4X ∅.281 ±.008"
      },
      "parsed_interpretation": {
        "hole_type": "simple",
        "diameter": 0.281,
        "count": 4,
        "diameter_tolerance_plus": 0.008
      }
    }
  ],
  "unmapped_annotations": ["H5"],
  "unmapped_holes": ["hole_12"]
}
```

### Matching Algorithm

#### Phase 1: Deterministic Matching

For each annotation, apply rules in order:

1. **Counterbore Matching** (if annotation is counterbore)
   - Match both primary diameter AND counterbore diameter
   - **Function:** `_match_counterbore()`
   - Tolerance: 5% relative

2. **Thread Matching** (if annotation has thread spec)
   - Match thread diameter (wider 10% tolerance)
   - **Function:** `_match_thread()`

3. **Generic Diameter Matching**
   - Match primary diameter within 5% tolerance
   - **Function:** `_match_by_diameter()`
   - **Helper:** `diameter_matches()` in `geometry_utils.py`

4. **Count Filtering**
   - If annotation says "4X", keep only candidates where count matches
   - **Function:** `_match_by_count()`

#### Phase 2: LLM Disambiguation

**Trigger:** When candidates > annotation.count (ambiguous match)

**Process:**
1. Send annotation JSON + candidate holes JSON to GPT-4o
2. Ask model to pick best match with confidence + reason
3. **Function:** `_llm_disambiguate()`

**LLM Prompt Structure:**
```
ANNOTATION (from drawing): {annotation_json}
CANDIDATE 3D HOLES: {candidates_json}

Pick best match considering:
- diameter match
- hole type (counterbore, threaded, through, blind)
- count/multiplicity
- depth if available

Return: {"matched_hole_ids": [...], "confidence": "high|medium|low", "reason": "..."}
```

#### Phase 3: Confidence Assignment

- **HIGH** (0.95): Deterministic match with count agreement
- **MEDIUM** (0.70): LLM disambiguation or partial match
- **LOW** (0.35): Multiple candidates without LLM, or no match

### Evidence & Traceability

Every mapping includes:
- **Evidence:** Page number, bounding box, raw text from drawing
- **Match reasons:** Audit trail of matching logic applied
- **Parsed interpretation:** Structured breakdown of annotation fields

---

## Data Models

**Module:** `app/models/schemas.py`

All data structures use **Pydantic** for validation and serialization.

### Core Schema Hierarchy

```
LinkageResult (top-level output)
├── DrawingAnnotations
│   └── HoleAnnotation[]
│       ├── ThreadSpec
│       └── BoundingBox
├── StepFeatures
│   └── HoleFeature3D[]
│       └── CylindricalFeature[]
│           ├── Point3D
│           └── Vector3D
└── FeatureMapping[]
    ├── EvidenceTrace
    │   └── BoundingBox
    └── parsed_interpretation (dict)
```

### Key Enums

```python
HoleType = "simple" | "counterbore" | "countersink" | "threaded" | "cross_drilled" | "through" | "blind"
MatchConfidence = "high" | "medium" | "low"
```

---

## Utility Modules

### PDF Utils (`app/utils/pdf_utils.py`)

**Functions:**
- `pdf_to_images(pdf_path, dpi=300)` → Convert PDF pages to PIL Images using PyMuPDF
- `image_to_base64(img)` → Encode image as data-URI for vision API
- `crop_annotation_regions(images, di_result)` → Extract dense annotation clusters for focused vision pass
  - Identifies diameter-related text from DI bounding boxes
  - Merges overlapping boxes into clusters
  - Crops with 15% padding
  - Returns `[(page_num, cropped_image), ...]`

### Geometry Utils (`app/utils/geometry_utils.py`)

**Functions:**
- `diameter_matches(d1, d2, rel_tol=0.05)` → Check if two diameters match within 5% tolerance
- `axes_coaxial(center_a, axis_a, center_b, axis_b, linear_tol=0.5)` → Check if two cylinders share same axis
  - Tests: axes parallel (within 5°) AND perpendicular distance < 0.5 units
- `vectors_parallel(a, b, tol_deg=5.0)` → Check if two direction vectors are parallel
- `distance(a, b)` → Euclidean distance between two 3D points

---

## Configuration

**Module:** `app/config.py`

Loads environment variables from `.env`:

```python
class AzureDIConfig:
    endpoint: str  # AZURE_DI_ENDPOINT
    key: str       # AZURE_DI_KEY

class AzureOpenAIConfig:
    endpoint: str     # AZURE_OPENAI_ENDPOINT
    key: str          # AZURE_OPENAI_KEY
    deployment: str   # AZURE_OPENAI_DEPLOYMENT (default: "gpt-4o")
    api_version: str  # AZURE_OPENAI_API_VERSION (default: "2024-12-01-preview")

DATASET_DIR: Path  # Points to dataset/ folder
```

---

## Entry Points

### CLI: `run.py`

**Usage:**
```bash
python run.py --pdf drawing.pdf --step model.stp --output result.json
```

**Flags:**
- `--unit inch|mm` (default: inch)
- `--no-di` (skip Azure DI, vision-only mode)
- `--no-llm` (skip LLM disambiguation, deterministic-only)

**Pipeline Execution:**
1. Layer 1: `extract_layout(pdf_path)`
2. Layer 2: `enrich_drawing(pdf_path, di_result, unit)`
3. Layer 3: `parse_step(step_path)`
4. Layer 4: `correlate(annotations, step_features, use_llm=True)`
5. Write JSON: `linkage.model_dump_json()`

### Web UI: `app/main.py`

**Usage:**
```bash
streamlit run app/main.py
```

**Features:**
- File selection (dataset or upload)
- Pipeline execution button
- 4 tabs:
  - **Pipeline Layers (I/O):** Shows input/output for each layer
  - **Drawing:** Page viewer with annotation overlays
  - **Results Table:** Mappings with confidence indicators
  - **JSON Export:** Download button + preview

**Key Functions:**
- `_sidebar()` → File selection UI
- `_run_pipeline()` → Executes 4 layers, stores in session state
- `_render_layer*_io()` → Visualizes each layer's I/O
- `_render_drawing()` → Drawing page viewer
- `_render_json_export()` → JSON download

---

## Validation

**Module:** `app/validation.py`

**Purpose:** Compare pipeline output against FSI (Feature and Specification Index) ground truth.

**Usage:**
```bash
python -m app.validation --ftc 06 --pipeline-json linkage_result.json
```

**FSI Parsing:**
- Extracts ground-truth hole features from NIST FSI PDFs
- Patterns: `NX ∅D`, `M10 x 1.5`, fit designations (`3.5 G6`)
- **Function:** `parse_fsi_text()`

**Comparison Algorithm:**
1. Group GT and extracted annotations by rounded diameter
2. Count-aware bucket matching (match up to min(gt_count, ext_count))
3. Fuzzy matching pass for remaining features (10% tolerance)
4. Compute precision, recall, F1

**Output:**
```
Ground-truth holes:  12
Extracted holes:     11
Matched:             10
Precision:           90.91%
Recall:              83.33%
F1:                  86.96%
```

---

## Data Flow Example

### Input Files
- `nist_ftc_06_asme1_rd.pdf` (engineering drawing)
- `nist_ftc_06_asme1_rd.stp` (3D STEP model)

### Layer 1 Output (OCR)
```
24 pages, 1,847 text elements with bounding boxes
Sample: "4X ∅.281 ±.008" at (0.45, 0.32)
```

### Layer 2 Output (Vision)
```
12 hole annotations extracted:
- H1: 4X ∅.281 counterbore, page 1
- H2: 8X ∅.562 simple, page 1
- H3: M10 x 1.5 - 6H threaded, page 2
...
```

### Layer 3 Output (STEP)
```
24 cylindrical faces → 12 grouped holes:
- hole_0: D0.281 counterbore (CB D0.562)
- hole_1: D0.281 counterbore (CB D0.562)
- hole_2: D0.281 counterbore (CB D0.562)
- hole_3: D0.281 counterbore (CB D0.562)
- hole_4: D0.562 simple
...
```

### Layer 4 Output (Correlation)
```
12 mappings created:
- H1 → [hole_0, hole_1, hole_2, hole_3] (HIGH confidence)
  Reasons: diameter match, count filter 4X
- H2 → [hole_4, hole_5, ...] (HIGH confidence)
  Reasons: diameter match, count filter 8X
...
```

---

## Error Handling

### Layer 1 Failure
- **Cause:** Azure DI unavailable or quota exceeded
- **Behavior:** Continue in vision-only mode (no cropped-region pass)
- **Stored in:** `st.session_state.layer1_error`

### Layer 2 Failure
- **Cause:** Azure OpenAI API error, invalid JSON response
- **Behavior:** Pipeline stops, error displayed
- **Stored in:** `st.session_state.layer2_error`

### Layer 3 Failure
- **Cause:** Invalid STEP file, missing entities
- **Behavior:** Pipeline stops, error displayed
- **Stored in:** `st.session_state.layer3_error`

### Layer 4 Failure
- **Cause:** LLM disambiguation error
- **Behavior:** Falls back to first candidate, LOW confidence
- **Stored in:** `st.session_state.layer4_error`

---

## Key Design Decisions

### 1. Multi-Pass Vision Strategy
**Rationale:** Single-pass extraction misses 20-30% of annotations (small text, detail views, dense clusters). Multi-pass with OCR hints + cropping + reconciliation achieves 90%+ recall.

### 2. Pure Python STEP Parser
**Rationale:** Avoids heavy CAD kernel dependencies (pythonocc, OCC). Lightweight, fast, sufficient for cylindrical feature extraction.

### 3. Hybrid Correlation
**Rationale:** Deterministic matching handles 80% of cases (exact diameter + count). LLM only invoked for ambiguous cases, minimizing cost.

### 4. Pydantic Schemas
**Rationale:** Type safety, automatic validation, JSON serialization, clear API contracts between layers.

### 5. Graceful Degradation
**Rationale:** If Azure DI fails, vision-only mode still works. If LLM fails, deterministic matching proceeds with LOW confidence flag.

---

## Dependencies

```
streamlit>=1.30.0              # Web UI framework
azure-ai-documentintelligence  # Azure DI SDK for OCR
openai>=1.12.0                 # Azure OpenAI API client
pydantic>=2.5.0                # Data validation & schemas
PyMuPDF>=1.23.0                # PDF rendering (fitz)
Pillow>=10.0.0                 # Image processing
numpy>=1.24.0                  # Geometry calculations
python-dotenv>=1.0.0           # .env loading
```

---

## Testing & Validation

### Dataset
NIST MBE PMI Validation Test Cases (FTC-06 through FTC-11):
- Public engineering drawings with GD&T annotations
- Corresponding STEP models
- FSI ground-truth feature lists

### Validation Workflow
1. Run pipeline: `python run.py --pdf dataset/nist_ftc_06_asme1_rd.pdf --step dataset/nist_ftc_06_asme1_rd.stp`
2. Compare output: `python -m app.validation --ftc 06 --pipeline-json linkage_result.json`
3. Review precision/recall/F1 metrics

---

## Extension Points

### Adding New Hole Types
1. Add enum value to `HoleType` in `schemas.py`
2. Update vision prompt system rules
3. Add matching logic in `matcher.py`

### Supporting New CAD Formats
1. Create new parser module (e.g., `iges_parser.py`)
2. Implement `parse_*()` function returning `StepFeatures`
3. Update CLI/UI to accept new file types

### Custom Vision Models
1. Modify `_build_client()` in `vision_enricher.py`
2. Update `_build_messages()` and `_completion_kwargs()` for model-specific API parameters
3. Pass `model_deployment` parameter to `enrich_drawing()`

### Adding Extraction Layers
1. Create new module in `app/extraction/`
2. Define input/output schemas in `schemas.py`
3. Update pipeline in `run.py` and `main.py`
4. Add I/O rendering function in `main.py` (`_render_layer*_io()`)
