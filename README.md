# GenAI-Powered Drawing-to-3D Hole Feature Linker

Automatically extract hole-related annotations (thread callouts, tolerances, fit designations) from PDF engineering drawings and link them to the corresponding cylindrical features in 3D STEP models.

## Table of Contents

- [Approach Summary](#approach-summary)
- [Architecture](#architecture)
- [Prompt Strategy & Consistency](#prompt-strategy--consistency)
- [Setup Instructions](#setup-instructions)
- [Running the Pipeline](#running-the-pipeline)
- [Output Format](#output-format)
- [Validation](#validation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Dependencies & Licenses](#dependencies--licenses)
- [Ethics & Compliance](#ethics--compliance)

---

## Approach Summary

The solution uses a **four-layer hybrid pipeline** combining traditional document intelligence, vision-language models, geometric parsing, and deterministic+LLM correlation:

1. **OCR Layer** — Azure Document Intelligence extracts raw text with bounding boxes from each PDF page, providing a structured spatial map of all text elements.
2. **Vision-AI Layer** — GPT-4o processes high-resolution page images alongside the OCR context to identify and classify hole-related annotations (diameter callouts, thread specs, counterbore stacks, GD&T). A two-pass strategy (extraction + verification) ensures high recall.
3. **3D Geometry Layer** — A pure-Python STEP parser (no CAD kernel required) resolves the ISO 10303-21 entity reference chain (`CYLINDRICAL_SURFACE → AXIS2_PLACEMENT_3D → CARTESIAN_POINT + DIRECTION`) to extract every cylindrical surface's radius, depth, axis, and centre, then groups co-axial cylinders into logical holes.
4. **Correlation Layer** — A hybrid matcher uses deterministic rules (diameter matching, count filtering, counterbore pairing) for unambiguous cases, then invokes LLM disambiguation only when multiple candidates remain. Each mapping includes a confidence score and evidence trace.

The pipeline is designed to **fail gracefully**: if Azure DI is unavailable it falls back to vision-only mode; ambiguous cases are flagged with lower confidence rather than forced to a single answer; and all outputs are traceable back to the drawing region.

---

## Architecture

```
PDF Drawing ──► Azure DI Layout ──► Vision Model ──► Matcher ──► Structured JSON
                (OCR + bbox)        (GPT-4o)           ▲          (with evidence)
                                                       │
STEP File   ──► STEP Text Parser ──────────────────────┘
                (cylinders, axes,
                 co-axial grouping)
```

### Pipeline Layers in Detail

| Layer | Module | Input | Output |
|-------|--------|-------|--------|
| 1. OCR | `app/extraction/di_extractor.py` | PDF file | Text elements with bounding boxes |
| 2. Vision | `app/extraction/vision_enricher.py` | Page images + OCR context | Structured `HoleAnnotation` objects |
| 3. STEP | `app/extraction/step_parser.py` | STEP file | Grouped `HoleFeature3D` objects |
| 4. Correlation | `app/correlation/matcher.py` | Annotations + 3D features | `LinkageResult` with mappings |

---

## Prompt Strategy & Consistency

### Vision Model Prompts

The system uses carefully structured prompts to ensure consistent, high-quality extraction:

1. **System Prompt** — Defines the model's role as a mechanical engineer, provides a strict JSON schema, and lists critical rules:
   - Only extract cylindrical holes (not linear dimensions, surface tolerances, slots, spherical diameters, fillets, or radii)
   - Every entry must have a non-null diameter with a diameter symbol (∅) or recognized thread designation
   - Counterbored holes must be extracted as a single object with both diameters
   - Distinct callouts with different GD&T must never be merged

2. **Per-Page User Prompt** — Includes a checklist of annotation patterns to scan for (NX∅D, ∅D THRU, thread callouts, counterbore stacks, fit designations), along with the OCR context from Layer 1 and the page image.

3. **Verification Pass** — A second sweep asks the model to review all pages for any missed annotations, comparing against the already-found list. This catches holes visible only in section/detail views.

### Consistency Mechanisms

- **Deterministic post-filters** remove null diameters, implausibly small values, spherical annotations, linear dimensions misidentified as holes, and near-exact duplicates within pages.
- **Sequential ID reassignment** after filtering ensures globally unique, predictable annotation IDs (H1, H2, ...).
- **Model-aware API routing** handles differences between GPT-series and o-series models (temperature vs. max_completion_tokens, system role vs. merged user prompt).
- **Structured Pydantic schemas** validate all data at every pipeline boundary.

### LLM Disambiguation in Correlation

When deterministic matching produces more candidates than the annotation's count, a focused LLM call presents only the specific annotation and candidate holes, asking for the best match with a confidence level and explanation. This targeted usage minimizes token cost and latency while handling edge cases.

---

## Setup Instructions

### Prerequisites

- **Python 3.10+**
- **Azure Document Intelligence** resource (for OCR extraction)
- **Azure OpenAI** resource with a GPT-4o deployment (and optionally o4-mini)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd manuk

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your Azure credentials
```

Required environment variables in `.env`:

| Variable | Description |
|----------|-------------|
| `AZURE_DI_ENDPOINT` | Azure Document Intelligence endpoint URL |
| `AZURE_DI_KEY` | Azure Document Intelligence API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name (default: `gpt-4o`) |
| `AZURE_OPENAI_API_VERSION` | API version (default: `2024-12-01-preview`) |

---

## Running the Pipeline

### Option 1: CLI Entry Point (recommended for evaluation)

```bash
# Basic usage
python run.py --pdf <drawing.pdf> --step <model.stp>

# With NIST dataset
python run.py --pdf dataset/nist_ftc_06_asme1_rd.pdf --step dataset/nist_ftc_06_asme1_rd.stp --output linkage_ftc06.json

# Metric units
python run.py --pdf dataset/nist_ftc_10_asme1_rb.pdf --step dataset/nist_ftc_10_asme1_rb.stp --unit mm

# Use o4-mini model
python run.py --pdf drawing.pdf --step model.stp --model o4-mini

# Vision-only mode (skip Azure DI)
python run.py --pdf drawing.pdf --step model.stp --no-di

# Skip LLM disambiguation (deterministic-only correlation)
python run.py --pdf drawing.pdf --step model.stp --no-llm
```

**CLI Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | (required) | Path to engineering drawing PDF |
| `--step` | (required) | Path to STEP (.stp/.step) file |
| `--unit` | `inch` | Drawing units: `inch` or `mm` |
| `--model` | `gpt-4o` | Azure OpenAI deployment name |
| `--output` / `-o` | `linkage_result.json` | Output JSON file path |
| `--no-di` | false | Skip Azure Document Intelligence |
| `--no-llm` | false | Skip LLM disambiguation |

### Option 2: Streamlit Web App

```bash
streamlit run app/main.py
```

The web app provides:
- Interactive file selection (dataset or upload)
- Visual pipeline I/O inspection for each layer
- Drawing page viewer with annotation overlays
- Results table with confidence indicators
- JSON export download

---

## Output Format

The pipeline produces a JSON file conforming to the `LinkageResult` schema. See [`sample_output.json`](sample_output.json) for a complete example.

### Top-Level Structure

```json
{
  "drawing_pdf": "path/to/drawing.pdf",
  "step_file": "path/to/model.stp",
  "annotations": { ... },
  "features_3d": { ... },
  "mappings": [ ... ],
  "unmapped_annotations": [ ... ],
  "unmapped_holes": [ ... ]
}
```

### Per-Mapping Fields

Each mapping in the `mappings` array contains:

| Field | Type | Description |
|-------|------|-------------|
| `annotation_id` | string | Unique ID of the hole annotation (e.g. "H1") |
| `hole_ids` | string[] | Matched 3D hole feature IDs (e.g. ["hole_0", "hole_1"]) |
| `confidence` | enum | `"high"`, `"medium"`, or `"low"` |
| `confidence_score` | float | Numeric confidence 0–1 (0.95 high, 0.70 medium, 0.35 low) |
| `match_reasons` | string[] | Explanation of why this match was made |
| `evidence` | object | Traceability link to the drawing region |
| `evidence.page` | int | 1-based page number |
| `evidence.bounding_box` | object | `{page, x_min, y_min, x_max, y_max}` (normalised 0–1) |
| `evidence.raw_text` | string | Verbatim annotation text from the drawing |
| `parsed_interpretation` | object | Structured breakdown of the annotation |

### Parsed Interpretation Fields

| Field | Description |
|-------|-------------|
| `hole_type` | `simple`, `counterbore`, `countersink`, `threaded`, `cross_drilled` |
| `diameter` | Nominal diameter in drawing units |
| `count` | Multiplicity (e.g. 4 for "4X") |
| `thread_type` | Thread designation (e.g. "M10") |
| `thread_pitch` | Thread pitch in mm |
| `tolerance_class` | Thread tolerance class (e.g. "6H", "6g") |
| `depth` | Hole depth |
| `fit_designation` | Fit class (e.g. "H7", "G6") |
| `counterbore_diameter` | Counterbore diameter |
| `counterbore_depth` | Counterbore depth |
| `position_tolerance` | Position tolerance value |
| `datum_refs` | Datum reference letters |

### 3D Feature Reference

Each `HoleFeature3D` in `features_3d.holes` includes:
- `hole_id` — unique identifier to cross-reference with mappings
- `cylinders[].face_index` — STEP entity ID for relocating the feature in the 3D model
- `center` / `axis` — geometric descriptor (3D point + direction vector)
- `primary_diameter` / `counterbore_diameter` — measured dimensions

---

## Validation

Compare pipeline output against FSI (Feature and Specification Index) ground truth:

```bash
# Show ground-truth features for a test case
python -m app.validation --ftc 06

# Compare pipeline output against ground truth
python -m app.validation --ftc 06 --pipeline-json linkage_ftc06.json
```

Validation computes precision, recall, and F1 using count-aware diameter bucket matching with a fuzzy fallback pass.

---

## Dataset

Uses the [NIST MBE PMI Validation and Conformance Testing](https://www.nist.gov/ctl/smart-connected-systems-division/smart-connected-manufacturing-systems-group/mbe-pmi-0) test cases (FTC-06 through FTC-11). Each test case includes:

| File Type | Description | Example |
|-----------|-------------|---------|
| Engineering drawing PDF | GD&T annotated 2D drawing | `nist_ftc_06_asme1_rd.pdf` |
| FSI PDF | Ground-truth feature/specification index | `nist_ftc_06_asme1_rd_fsi.pdf` |
| Element IDs PDF | Annotation label mapping | `nist_ftc_06_asme1_rd_elem_ids.pdf` |
| STEP file | ISO 10303-21 AP203 geometry | `nist_ftc_06_asme1_rd.stp` |

These are **publicly available datasets** from NIST — no proprietary or confidential data is used.

### Units

- FTC-06 through FTC-09: **inches**
- FTC-10 through FTC-11: **millimetres**

### Synthetic Data

No synthetic data generation was used. All testing is performed on the NIST PMI public dataset, which provides industry-standard engineering drawings with known ground truth.

---

## Project Structure

```
manuk/
├── run.py                          # CLI entry point (PDF + STEP → JSON)
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for Azure credentials
├── LICENSE                         # MIT License
├── sample_output.json              # Example output showing all fields
├── dataset/                        # NIST FTC STEP + PDF files
│   ├── nist_ftc_06_asme1_rd.stp
│   ├── nist_ftc_07_asme1_rd.stp
│   ├── nist_ftc_08_asme1_rc.stp
│   ├── nist_ftc_09_asme1_rd.stp
│   ├── nist_ftc_10_asme1_rb.stp
│   └── nist_ftc_11_asme1_rb.stp
├── app/
│   ├── __init__.py
│   ├── main.py                     # Streamlit web app
│   ├── config.py                   # Azure credential config
│   ├── validation.py               # Ground-truth comparison & metrics
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── di_extractor.py         # Azure DI layout extraction
│   │   ├── vision_enricher.py      # Vision model hole annotation extraction
│   │   └── step_parser.py          # Pure-Python STEP file parsing
│   ├── correlation/
│   │   ├── __init__.py
│   │   └── matcher.py              # Deterministic + LLM correlation
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic data models
│   └── utils/
│       ├── __init__.py
│       ├── pdf_utils.py            # PDF to image conversion
│       └── geometry_utils.py       # 3D geometry helpers
```

---

## Performance

Validated on NIST FTC test cases with the following average results:

| Model   | Avg Precision | Avg Recall | Avg F1 |
|---------|---------------|------------|--------|
| GPT-4o  | 93.3%         | 88.9%      | 90.6%  |
| o4-mini | 90.2%         | 72.8%      | 79.5%  |

GPT-4o is recommended as the default model for best accuracy and speed.

### Runtime

Typical end-to-end runtime per drawing (single PDF + STEP):
- **GPT-4o**: 15–45 seconds (depending on page count and annotation density)
- **o4-mini**: 20–60 seconds (reasoning models use more compute)
- STEP parsing: <1 second for typical parts

---

## Dependencies & Licenses

All dependencies are open-source with permissive licenses:

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| [streamlit](https://streamlit.io/) | ≥1.30.0 | Apache-2.0 | Web application UI |
| [azure-ai-documentintelligence](https://pypi.org/project/azure-ai-documentintelligence/) | ≥1.0.0 | MIT | Azure DI SDK for OCR extraction |
| [openai](https://pypi.org/project/openai/) | ≥1.12.0 | Apache-2.0 | Azure OpenAI API client |
| [pydantic](https://pydantic.dev/) | ≥2.5.0 | MIT | Data validation and schemas |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | ≥1.23.0 | AGPL-3.0 | PDF rendering to images |
| [Pillow](https://python-pillow.org/) | ≥10.0.0 | HPND | Image processing |
| [numpy](https://numpy.org/) | ≥1.24.0 | BSD-3-Clause | Geometry calculations |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | ≥1.0.0 | BSD-3-Clause | Environment variable loading |

### External Services

| Service | Purpose | Data Sent |
|---------|---------|-----------|
| Azure Document Intelligence | OCR text extraction | PDF file content |
| Azure OpenAI (GPT-4o) | Vision annotation extraction + LLM disambiguation | Page images (base64), OCR text snippets, annotation/feature JSON |

No data is stored by external services beyond API processing. All API calls use your own Azure subscription.

---

## Ethics & Compliance

- **No confidential data** — Only NIST public datasets are used.
- **Transparency** — Every mapping includes an evidence trace (page, bounding box, raw text) and confidence score. Ambiguous cases report multiple candidates rather than forcing a single answer.
- **Reliability** — Low-confidence matches are explicitly flagged. Unmapped annotations and unmapped 3D holes are reported separately rather than suppressed.
- **Responsible model use** — Azure OpenAI usage complies with Microsoft's Responsible AI policies. No data is sent to third-party services outside your Azure subscription.
- **Data protection** — No personal data is processed. Drawing metadata (title blocks) is used only for annotation extraction.
- **Licensing** — This project is MIT-licensed. All dependencies use permissive open-source licenses (see table above). Note: PyMuPDF uses AGPL-3.0; for commercial use, a [commercial license](https://pymupdf.readthedocs.io/en/latest/about.html#license) is available.
