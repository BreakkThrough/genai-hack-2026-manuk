"""
Streamlit web application for the Drawing-to-3D Hole Feature Linker.

Run with:  streamlit run app/main.py
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from app.config import DATASET_DIR
from app.extraction.di_extractor import extract_layout
from app.extraction.step_parser import parse_step
from app.extraction.vision_enricher import enrich_drawing
from app.correlation.matcher import correlate
from app.models.schemas import (
    DIExtractionResult,
    DrawingAnnotations,
    LinkageResult,
    MatchConfidence,
    StepFeatures,
)
from app.utils.pdf_utils import pdf_to_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Drawing - 3D Hole Linker",
    page_icon="🔩",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
    }
    .layer-header {
        background: linear-gradient(90deg, #1e3a5f, #2d5986);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    .io-box {
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.25rem 0;
        background: #f8fafc;
    }
    .io-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .io-input { color: #2563eb; }
    .io-output { color: #16a34a; }
    .confidence-high { color: #16a34a; font-weight: 600; }
    .confidence-medium { color: #d97706; font-weight: 600; }
    .confidence-low { color: #dc2626; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

_STATE_KEYS = (
    "linkage_result", "drawing_images", "step_features", "annotations",
    "di_result", "pdf_path_used", "step_path_used", "model_used",
    "layer1_error", "layer2_error", "layer3_error", "layer4_error",
)

for _k in _STATE_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def _sidebar() -> tuple[Path | None, Path | None, str, str]:
    st.sidebar.title("Drawing - 3D Hole Linker")
    st.sidebar.markdown("---")

    source = st.sidebar.radio("File source", ["Dataset (NIST FTC)", "Upload files"])

    pdf_path: Path | None = None
    step_path: Path | None = None
    unit = "inch"

    if source == "Dataset (NIST FTC)":
        drawing_pdfs = sorted(DATASET_DIR.glob("*_asme1_r[a-z].pdf"))
        step_files = sorted(DATASET_DIR.glob("*.stp"))

        if not drawing_pdfs:
            st.sidebar.warning("No drawing PDFs found in dataset/")
            return None, None, unit, "gpt-4o"

        pdf_names = [p.name for p in drawing_pdfs]
        chosen_pdf = st.sidebar.selectbox("Drawing PDF", pdf_names)
        pdf_path = DATASET_DIR / chosen_pdf

        stem = chosen_pdf.replace(".pdf", "")
        matching_stp = [s for s in step_files if s.stem == stem]
        if matching_stp:
            step_path = matching_stp[0]
            st.sidebar.info(f"STEP: {step_path.name}")
        else:
            stp_names = [s.name for s in step_files]
            chosen_stp = st.sidebar.selectbox("STEP file", stp_names)
            step_path = DATASET_DIR / chosen_stp

        unit = "mm" if any(k in chosen_pdf for k in ("ftc_10", "ftc_11")) else "inch"
    else:
        pdf_upload = st.sidebar.file_uploader("Upload drawing PDF", type=["pdf"])
        stp_upload = st.sidebar.file_uploader("Upload STEP file", type=["stp", "step"])
        unit = st.sidebar.selectbox("Units", ["inch", "mm"])

        if pdf_upload:
            tmp = Path(tempfile.mktemp(suffix=".pdf"))
            tmp.write_bytes(pdf_upload.read())
            pdf_path = tmp
        if stp_upload:
            tmp = Path(tempfile.mktemp(suffix=".stp"))
            tmp.write_bytes(stp_upload.read())
            step_path = tmp

    st.sidebar.markdown("---")
    model = st.sidebar.selectbox(
        "Vision Model",
        ["gpt-4o"],
        help="Azure OpenAI deployment for hole extraction",
    )
    st.sidebar.markdown(f"**Units:** {unit}")

    return pdf_path, step_path, unit, model


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ──────────────────────────────────────────────────────────────────────────────

def _run_pipeline(pdf_path: Path, step_path: Path, unit: str, model: str = "gpt-4o"):
    for k in _STATE_KEYS:
        st.session_state[k] = None

    st.session_state.pdf_path_used = str(pdf_path)
    st.session_state.step_path_used = str(step_path)
    st.session_state.model_used = model

    api_ver = "2025-04-01-preview" if model != "gpt-4o" else None
    progress = st.progress(0, text="Starting pipeline...")

    # Layer 1: Azure DI layout extraction
    progress.progress(10, text="Layer 1/4 -- Azure DI layout extraction...")
    try:
        di_result = extract_layout(pdf_path)
        st.session_state.di_result = di_result
    except Exception as exc:
        st.session_state.layer1_error = str(exc)
        di_result = None

    # Layer 2: Vision enrichment
    progress.progress(30, text=f"Layer 2/4 -- {model} vision enrichment...")
    try:
        annotations = enrich_drawing(
            pdf_path, di_result=di_result, unit=unit, apply_filters=True,
            model_deployment=model, api_version=api_ver,
        )
        st.session_state.annotations = annotations
    except Exception as exc:
        st.session_state.layer2_error = str(exc)
        st.error(f"Vision enrichment failed: {exc}")
        progress.empty()
        return

    # Layer 3: STEP parsing
    progress.progress(60, text="Layer 3/4 -- Parsing STEP file...")
    try:
        step_features = parse_step(step_path)
        st.session_state.step_features = step_features
    except Exception as exc:
        st.session_state.layer3_error = str(exc)
        st.error(f"STEP parsing failed: {exc}")
        progress.empty()
        return

    # Layer 4: Correlation
    progress.progress(80, text="Layer 4/4 -- Correlating annotations with 3D features...")
    try:
        linkage = correlate(annotations, step_features)
        st.session_state.linkage_result = linkage
    except Exception as exc:
        st.session_state.layer4_error = str(exc)
        st.error(f"Correlation failed: {exc}")
        progress.empty()
        return

    st.session_state.drawing_images = pdf_to_images(pdf_path, dpi=150)
    progress.progress(100, text="Pipeline complete!")
    progress.empty()


# ──────────────────────────────────────────────────────────────────────────────
# Layer I/O rendering
# ──────────────────────────────────────────────────────────────────────────────

def _io_label(text: str, kind: str = "input"):
    css = "io-input" if kind == "input" else "io-output"
    tag = "INPUT" if kind == "input" else "OUTPUT"
    st.markdown(
        f'<div class="io-label {css}">{tag}: {text}</div>',
        unsafe_allow_html=True,
    )


def _render_layer1_io():
    st.markdown(
        '<div class="layer-header">Layer 1: Azure Document Intelligence (Prebuilt Layout)</div>',
        unsafe_allow_html=True,
    )
    col_in, col_out = st.columns(2)

    with col_in:
        _io_label("PDF Drawing File", "input")
        pdf_path = st.session_state.pdf_path_used
        if pdf_path:
            st.code(Path(pdf_path).name, language=None)
            images = st.session_state.drawing_images
            if images:
                st.image(images[0], caption="Page 1 (sent to Azure DI)", width=400)

    with col_out:
        _io_label("OCR Text + Bounding Boxes (JSON)", "output")
        if st.session_state.layer1_error:
            st.warning(f"Azure DI failed: {st.session_state.layer1_error}")
            st.info("Pipeline continued with vision-only mode.")
        elif st.session_state.di_result:
            di: DIExtractionResult = st.session_state.di_result
            st.metric("Pages processed", len(di.pages))
            total_elements = sum(len(p.elements) for p in di.pages)
            st.metric("Text elements extracted", total_elements)
            with st.expander("Sample OCR elements (first 20)"):
                samples = []
                for page in di.pages:
                    for elem in page.elements[:20]:
                        samples.append({
                            "text": elem.content,
                            "confidence": elem.confidence,
                            "bbox": elem.bounding_box.model_dump() if elem.bounding_box else None,
                        })
                        if len(samples) >= 20:
                            break
                    if len(samples) >= 20:
                        break
                st.json(samples)


def _render_layer2_io():
    model_name = st.session_state.model_used or "GPT-4o"
    st.markdown(
        f'<div class="layer-header">Layer 2: {model_name} Vision Enrichment</div>',
        unsafe_allow_html=True,
    )
    col_in, col_out = st.columns(2)

    with col_in:
        _io_label("High-res page images + OCR context", "input")
        images = st.session_state.drawing_images
        if images:
            st.markdown(f"**{len(images)} page image(s)** at 300 DPI sent to {model_name}")
            page_sel = st.selectbox("Preview page", range(1, len(images) + 1), key="l2_page")
            st.image(images[page_sel - 1], caption=f"Page {page_sel}", width=400)

        di = st.session_state.di_result
        if di:
            with st.expander("OCR context passed alongside image"):
                for page in di.pages:
                    text_snippet = " | ".join(e.content for e in page.elements[:30])
                    st.text(f"Page {page.page_number}: {text_snippet[:500]}...")
        else:
            st.info("No Azure DI context available -- vision-only mode.")

    with col_out:
        _io_label("Structured hole annotations (JSON)", "output")
        if st.session_state.layer2_error:
            st.error(f"Vision enrichment failed: {st.session_state.layer2_error}")
        elif st.session_state.annotations:
            anns: DrawingAnnotations = st.session_state.annotations
            st.metric("Hole annotations extracted", len(anns.annotations))
            st.metric("Drawing units", anns.unit)

            for ann in anns.annotations:
                label_parts = [ann.annotation_id, ann.hole_type.value]
                if ann.diameter:
                    label_parts.append(f"D{ann.diameter}")
                if ann.count > 1:
                    label_parts.append(f"{ann.count}X")
                if ann.thread_spec:
                    label_parts.append(ann.thread_spec.designation)
                with st.expander(" | ".join(label_parts)):
                    st.json(ann.model_dump(exclude_none=True))


def _render_layer3_io():
    st.markdown(
        '<div class="layer-header">Layer 3: STEP File Parsing (Pure Python)</div>',
        unsafe_allow_html=True,
    )
    col_in, col_out = st.columns(2)

    with col_in:
        _io_label("STEP file (ISO 10303-21)", "input")
        step_path = st.session_state.step_path_used
        if step_path:
            p = Path(step_path)
            st.code(p.name, language=None)
            size_kb = p.stat().st_size / 1024 if p.exists() else 0
            st.markdown(f"**Size:** {size_kb:.1f} KB")

            if p.exists():
                with st.expander("STEP file header (first 10 lines)"):
                    text = p.read_text(errors="replace")
                    header_lines = text.splitlines()[:10]
                    st.code("\n".join(header_lines), language=None)

                cyl_count = len(re.findall(r"CYLINDRICAL_SURFACE", text))
                st.markdown(f"**CYLINDRICAL_SURFACE entities found:** {cyl_count}")

    with col_out:
        _io_label("3D hole features (grouped cylinders)", "output")
        if st.session_state.layer3_error:
            st.error(f"STEP parsing failed: {st.session_state.layer3_error}")
        elif st.session_state.step_features:
            feats: StepFeatures = st.session_state.step_features
            st.metric("Cylindrical faces", feats.total_cylindrical_faces)
            st.metric("Grouped hole features", len(feats.holes))

            diameters = sorted({h.primary_diameter for h in feats.holes})
            st.markdown(f"**Unique diameters:** {', '.join(f'{d:.3f}' for d in diameters)}")

            type_counts: dict[str, int] = {}
            for h in feats.holes:
                type_counts[h.hole_type.value] = type_counts.get(h.hole_type.value, 0) + 1
            st.markdown(f"**By type:** {type_counts}")

            for hole in feats.holes:
                label_parts = [hole.hole_id, f"D{hole.primary_diameter:.4f}", hole.hole_type.value]
                if hole.counterbore_diameter:
                    label_parts.append(f"CB D{hole.counterbore_diameter:.4f}")
                with st.expander(" | ".join(label_parts)):
                    st.json(hole.model_dump(exclude_none=True))


def _render_layer4_io():
    st.markdown(
        '<div class="layer-header">Layer 4: Annotation-to-Feature Correlation</div>',
        unsafe_allow_html=True,
    )
    col_in, col_out = st.columns(2)

    with col_in:
        _io_label("Annotations (Layer 2) + 3D Features (Layer 3)", "input")
        anns = st.session_state.annotations
        feats = st.session_state.step_features
        if anns:
            st.markdown(f"**{len(anns.annotations)} annotation(s)** from drawing")
            ann_summary = []
            for a in anns.annotations:
                entry = {
                    "id": a.annotation_id, "type": a.hole_type.value,
                    "diameter": a.diameter, "count": a.count,
                }
                if a.thread_spec:
                    entry["thread"] = a.thread_spec.designation
                ann_summary.append(entry)
            with st.expander("Annotation summary"):
                st.json(ann_summary)
        if feats:
            st.markdown(f"**{len(feats.holes)} hole feature(s)** from STEP")
            hole_summary = [
                {"id": h.hole_id, "diameter": h.primary_diameter, "type": h.hole_type.value}
                for h in feats.holes
            ]
            with st.expander("3D feature summary"):
                st.json(hole_summary)

    with col_out:
        _io_label("Annotation-to-Feature mappings", "output")
        if st.session_state.layer4_error:
            st.error(f"Correlation failed: {st.session_state.layer4_error}")
        elif st.session_state.linkage_result:
            linkage: LinkageResult = st.session_state.linkage_result
            st.metric("Mappings created", len(linkage.mappings))
            high = sum(1 for m in linkage.mappings if m.confidence == MatchConfidence.HIGH)
            med = sum(1 for m in linkage.mappings if m.confidence == MatchConfidence.MEDIUM)
            low = sum(1 for m in linkage.mappings if m.confidence == MatchConfidence.LOW)
            st.markdown(f"**Confidence:** {high} high, {med} medium, {low} low")
            st.markdown(f"**Unmapped annotations:** {len(linkage.unmapped_annotations)}")
            st.markdown(f"**Unmapped 3D holes:** {len(linkage.unmapped_holes)}")

            rows = []
            for m in linkage.mappings:
                ann = next(
                    (a for a in linkage.annotations.annotations
                     if a.annotation_id == m.annotation_id),
                    None,
                )
                rows.append({
                    "Annotation": m.annotation_id,
                    "Type": ann.hole_type.value if ann else "",
                    "Diameter": ann.diameter if ann else None,
                    "Count": ann.count if ann else 1,
                    "Thread": ann.thread_spec.designation if ann and ann.thread_spec else "",
                    "Matched Holes": ", ".join(m.hole_ids) if m.hole_ids else "-",
                    "Confidence": m.confidence.value.upper(),
                    "Reasons": "; ".join(m.match_reasons),
                })
            st.dataframe(rows, width="stretch", hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Result tabs
# ──────────────────────────────────────────────────────────────────────────────

def _render_overview(linkage: LinkageResult):
    st.subheader("Pipeline Results Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annotations extracted", len(linkage.annotations.annotations))
    c2.metric("3D hole features", len(linkage.features_3d.holes))
    c3.metric("Mappings created", len(linkage.mappings))
    high = sum(1 for m in linkage.mappings if m.confidence == MatchConfidence.HIGH)
    c4.metric("High-confidence", high)


def _render_drawing(images: list[Image.Image], linkage: LinkageResult):
    st.subheader("Drawing Pages")
    if not images:
        st.info("No drawing images available.")
        return

    page_idx = st.slider("Page", 1, len(images), 1, key="draw_page") - 1
    st.image(images[page_idx], caption=f"Page {page_idx + 1}", width="stretch")

    page_anns = [a for a in linkage.annotations.annotations if a.page == page_idx + 1]
    if page_anns:
        st.markdown(f"**{len(page_anns)} hole annotation(s) on this page:**")
        for a in page_anns:
            dia_str = f"D{a.diameter}" if a.diameter else ""
            count_str = f" x{a.count}" if a.count > 1 else ""
            with st.expander(f"{a.annotation_id} -- {a.hole_type.value} {dia_str}{count_str}"):
                st.json(a.model_dump(exclude_none=True))


def _render_json_export(linkage: LinkageResult):
    st.subheader("Structured JSON Output")
    json_str = linkage.model_dump_json(indent=2)
    st.download_button(
        label="Download full linkage JSON",
        data=json_str,
        file_name="linkage_result.json",
        mime="application/json",
    )
    with st.expander("Preview JSON"):
        st.code(json_str[:10000], language="json")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    pdf_path, step_path, unit, model = _sidebar()

    st.title("GenAI-Powered Drawing - 3D Hole Feature Linker")
    st.markdown(
        "Automatically extract hole annotations from engineering drawings and "
        "link them to cylindrical features in the 3D STEP model."
    )

    if pdf_path and step_path:
        if st.sidebar.button("Run Pipeline", type="primary"):
            _run_pipeline(pdf_path, step_path, unit, model=model)

    linkage: LinkageResult | None = st.session_state.linkage_result
    has_any_result = any(
        st.session_state.get(k) is not None
        for k in (
            "di_result", "annotations", "step_features", "linkage_result",
            "layer1_error", "layer2_error", "layer3_error", "layer4_error",
        )
    )

    if not has_any_result:
        st.info("Select a drawing PDF and STEP file, then click **Run Pipeline** to begin.")
        return

    tab_layers, tab_draw, tab_results, tab_json = st.tabs([
        "Pipeline Layers (I/O)", "Drawing", "Results Table", "JSON Export",
    ])

    with tab_layers:
        _render_layer1_io()
        st.markdown("---")
        _render_layer2_io()
        st.markdown("---")
        _render_layer3_io()
        st.markdown("---")
        _render_layer4_io()

    with tab_draw:
        if linkage:
            _render_drawing(st.session_state.drawing_images or [], linkage)
        else:
            images = st.session_state.drawing_images
            if images:
                page_idx = st.slider("Page", 1, len(images), 1, key="draw_fallback") - 1
                st.image(images[page_idx], caption=f"Page {page_idx + 1}", width="stretch")

    with tab_results:
        if linkage:
            _render_overview(linkage)
        else:
            st.info("Correlation did not complete -- check Pipeline Layers tab for details.")

    with tab_json:
        if linkage:
            _render_json_export(linkage)
        else:
            st.info("No linkage result available for export.")


if __name__ == "__main__":
    main()
