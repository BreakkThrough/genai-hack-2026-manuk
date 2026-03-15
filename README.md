# GenAI-Powered Drawing-to-3D Hole Feature Linker

Automatically extract hole-related annotations (thread callouts, tolerances, fit designations) from PDF engineering drawings and link them to the corresponding cylindrical features in 3D STEP models.

## Architecture

```
PDF Drawing ──► Azure DI Layout ──► Vision Model ──► Matcher ──► Structured JSON
                                                        ▲
STEP File   ──► STEP Text Parser ──────────────────────┘
```

### Pipeline layers

1. **Azure Document Intelligence** (prebuilt-layout) extracts OCR text with bounding boxes from each PDF page.
2. **Vision Model** (GPT-4o / o4-mini) enriches the raw OCR by interpreting the drawing images — identifying hole callouts, thread specs, GD&T, and grouping related annotations.
3. **STEP Text Parser** (pure Python, no CAD kernel) parses the ISO 10303-21 STEP file to extract every `CYLINDRICAL_SURFACE` entity, resolving the reference chain to recover diameter, depth, axis, and centre — then groups co-axial cylinders into logical holes (including counterbores).
4. **Hybrid Matcher** correlates annotations to 3D features using deterministic rules (diameter, count, counterbore pairing) with LLM disambiguation for ambiguous cases.

## Quick start

### Prerequisites

- Python 3.10+
- Azure Document Intelligence resource
- Azure OpenAI resource with a GPT-4o (and optionally o4-mini) deployment

### Setup

```bash
cd manuk

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your Azure DI and OpenAI credentials
```

### Run the web app

```bash
streamlit run app/main.py
```

### Validate against ground truth

```bash
# Show FSI ground-truth features for a test case
python -m app.validation --ftc 06

# Compare pipeline output against ground truth
python -m app.validation --ftc 10 --pipeline-json linkage_result.json
```

## Project structure

```
manuk/
  .env.example                  # Template for Azure credentials
  requirements.txt              # Python dependencies
  dataset/                      # NIST FTC STEP + PDF files
  app/
    main.py                     # Streamlit web app
    config.py                   # Azure credential config
    validation.py               # Ground-truth comparison & metrics
    extraction/
      di_extractor.py           # Azure DI layout extraction
      vision_enricher.py        # Vision model hole annotation extraction
      step_parser.py            # Pure-Python STEP file parsing
    correlation/
      matcher.py                # Deterministic + LLM correlation
    models/
      schemas.py                # Pydantic data models
    utils/
      pdf_utils.py              # PDF to image conversion
      geometry_utils.py         # 3D geometry helpers
```

## Dataset

Uses the [NIST MBE PMI Validation and Conformance Testing](https://www.nist.gov/ctl/smart-connected-systems-division/smart-connected-manufacturing-systems-group/mbe-pmi-0) test cases (FTC-06 through FTC-11). Each test case includes:

- Engineering drawing PDF with GD&T annotations
- Feature and Specification Index (FSI) PDF (ground truth)
- Element IDs PDF (annotation labels)
- STEP AP203 geometry file

## Performance

Validated on NIST FTC test cases with the following average results:

| Model   | Avg Precision | Avg Recall | Avg F1 |
|---------|---------------|------------|--------|
| GPT-4o  | 93.3%         | 88.9%      | 90.6%  |
| o4-mini | 90.2%         | 72.8%      | 79.5%  |

GPT-4o is recommended as the default model for best accuracy and speed.
