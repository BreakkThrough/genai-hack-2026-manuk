"""Vision-model enrichment: extract structured hole annotations from drawing images."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from openai import AzureOpenAI

from app.config import AzureOpenAIConfig
from app.models.schemas import (
    DrawingAnnotations,
    HoleAnnotation,
    HoleType,
    ThreadSpec,
)
from app.utils.pdf_utils import image_to_base64, pdf_to_images

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model-aware API parameter helpers
# ---------------------------------------------------------------------------

_O_SERIES_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(deployment: str) -> bool:
    """o-series reasoning models use different API parameters."""
    lower = deployment.lower()
    return any(lower.startswith(p) for p in _O_SERIES_PREFIXES)


def _completion_kwargs(deployment: str) -> dict:
    """
    Build the right keyword arguments for ``chat.completions.create()``.
    GPT models accept ``temperature`` + ``max_tokens``; o-series models
    require ``max_completion_tokens`` and do not accept ``temperature``.
    """
    if _is_reasoning_model(deployment):
        return {"max_completion_tokens": 16384}
    return {"temperature": 0.0, "max_tokens": 4096}


def _build_messages(system: str, user_content, deployment: str) -> list[dict]:
    """
    Build the messages payload.  o-series models do not support a separate
    ``system`` role, so the system prompt is merged into the user message.
    """
    if _is_reasoning_model(deployment):
        if isinstance(user_content, str):
            return [{"role": "user", "content": f"{system}\n\n{user_content}"}]
        text_block = {"type": "text", "text": system + "\n\n"}
        return [{"role": "user", "content": [text_block] + list(user_content)}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert mechanical engineer who reads 2-D engineering drawings \
and extracts ONLY hole-related annotations.

DEFINITION OF A HOLE: A cylindrical feature created by drilling, boring, \
reaming, tapping, or similar operations. This includes simple drilled holes, \
counterbored holes, countersunk holes, threaded/tapped holes, and \
cross-drilled holes.

For EACH hole or hole pattern visible on the page, return a JSON object:

{
  "annotation_id": "<globally unique id, e.g. H1, H2, ...>",
  "hole_type": "simple | counterbore | countersink | threaded | cross_drilled",
  "count": <integer from the NX multiplicity prefix like '4X'>,
  "diameter": <nominal hole diameter as a float in drawing units — REQUIRED>,
  "diameter_tolerance_plus": <float or null>,
  "diameter_tolerance_minus": <float or null>,
  "depth": <float or null>,
  "depth_tolerance": <float or null>,
  "thread_designation": "<e.g. 'M3' or null>",
  "thread_pitch": <float or null>,
  "thread_tolerance_class": "<e.g. '6g' or null>",
  "counterbore_diameter": <float or null>,
  "counterbore_depth": <float or null>,
  "countersink_diameter": <float or null>,
  "countersink_angle": <float or null>,
  "position_tolerance": <float or null>,
  "datum_refs": ["A", "B", ...],
  "fit_designation": "<e.g. 'G6', 'H7' or null>",
  "raw_text": "<the verbatim annotation text>"
}

CRITICAL RULES:

1. ONLY extract annotations that describe CYLINDRICAL HOLES. The diameter \
   field is REQUIRED for every entry — never return null for diameter.

2. DO NOT extract these (they are NOT holes):
   - Linear dimensions (e.g. "12.00 ±.01" without a diameter symbol)
   - Surface profile tolerances (e.g. ".05 A B C")
   - Flatness, perpendicularity, parallelism tolerances
   - Width/slot dimensions (e.g. ".500 ±.008" for a slot width)
   - Spherical diameters (prefix "S" before the diameter symbol, e.g. "S∅1.250")
   - Taper/slope ratios (e.g. "1.00 : 2.00")
   - Fillet/round radii (e.g. "R.125")
   - Datum feature symbols
   - General notes

3. A valid hole callout MUST have a diameter symbol (∅) or be a recognized \
   thread designation (e.g. M10). If you see a dimension without ∅ and it \
   is not a thread callout, it is NOT a hole.

4. For COUNTERBORED holes, extract as ONE object with both diameters:
   - "diameter" = the smaller (through-hole) diameter
   - "counterbore_diameter" = the larger (counterbore) diameter
   - "counterbore_depth" = depth of the counterbore
   Set hole_type = "counterbore".

5. IMPORTANT: Do NOT merge distinct hole callouts. If you see two separate \
   annotations with the same diameter but different position tolerances, \
   datum references, locations, or counts, they are DIFFERENT holes — \
   extract each one separately. Only merge if they are literally the same \
   leader/callout shown in two views.

6. Look carefully at EVERY view (front, side, section, detail) for holes \
   that may only be visible in one view. Cross-drilled holes may appear as \
   circles in a side view. Counterbore stacks may show the through-hole \
   diameter in one view and the counterbore diameter in another.

7. Be thorough: extract EVERY distinct hole callout, including those in \
   section views and detail views. Multiple callouts with the same diameter \
   but different GD&T, position tolerance, or datum references are separate \
   features.

8. For counterbore holes that show "∅D THRU" then "↧∅CB x depth", the \
   through-hole diameter is "diameter" and the larger one is \
   "counterbore_diameter". Also extract the counterbore depth.

Return ONLY a JSON array of objects. No markdown fences, no commentary.
"""

USER_PROMPT_TEMPLATE = """\
Analyse this engineering drawing page (page {page_num} of {total_pages}).

Drawing units: {unit}.

Additional OCR context from Azure Document Intelligence:
{ocr_context}

TASK: Extract ALL hole-related annotations visible on this page.

CHECKLIST — scan for each of these patterns:
- "NX ∅D" callouts (e.g. "4X ∅.281 ±.008") -> simple hole with count N
- "∅D THRU" or "∅D ↧depth" -> through or blind simple hole
- Thread callouts: "M... × pitch - class" -> threaded hole
- Counterbore stacks: "∅D THRU" then "∅CB_D ↧CB_depth" -> ONE counterbore entry
- Fit designations: "∅D H7" or "D G6" -> simple hole with fit
- Individual hole callouts that may share the same diameter as others — \
  extract each SEPARATELY if they have different GD&T, position tolerance, \
  or datum references

IMPORTANT:
- Every entry MUST have a non-null diameter
- Do NOT include linear dimensions, surface tolerances, slot widths, \
  spherical diameters (S∅), fillets, or radii
- Do NOT merge holes that have different GD&T annotations

Return a JSON array.
"""

VERIFY_PROMPT = """\
I already extracted these hole annotations from the drawing:
{found_summary}

Now look at ALL pages again carefully. Are there ANY additional hole \
annotations (∅ diameter callouts) that I MISSED? Look specifically for:
- Holes with the same diameter that appear as SEPARATE callouts
- Holes on different views (section views, detail views)
- Counterbore components (through-hole + counterbore as separate diameters)
- Small holes (∅.250, ∅.400, etc.) that may be in detail views

Drawing units: {unit}.

OCR context:
{ocr_context}

Return a JSON array with ONLY the ADDITIONAL holes I missed. \
If I found everything, return an empty array [].
Each entry must have: annotation_id, hole_type, count, diameter, raw_text, page.
"""


# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------

def _build_client(api_version: str | None = None) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=AzureOpenAIConfig.endpoint,
        api_key=AzureOpenAIConfig.key,
        api_version=api_version or AzureOpenAIConfig.api_version,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_model_response(raw: str, label: str = "") -> list[dict]:
    """Parse a JSON array from a model response, stripping markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n", 1)
        raw = lines[1] if len(lines) > 1 else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        return data
    except json.JSONDecodeError:
        logger.warning("Model returned non-JSON for %s: %s", label, raw[:200])
        return []


# ---------------------------------------------------------------------------
# Vision API calls
# ---------------------------------------------------------------------------

def _call_vision(
    client: AzureOpenAI,
    image_b64: str,
    page_num: int,
    total_pages: int,
    unit: str,
    ocr_text: str,
    deployment: str,
) -> list[dict]:
    """Send one page image to the model and return parsed annotation dicts."""
    user_content = [
        {"type": "text", "text": USER_PROMPT_TEMPLATE.format(
            page_num=page_num,
            total_pages=total_pages,
            unit=unit,
            ocr_context=ocr_text[:4000],
        )},
        {"type": "image_url", "image_url": {"url": image_b64, "detail": "high"}},
    ]

    response = client.chat.completions.create(
        model=deployment,
        messages=_build_messages(SYSTEM_PROMPT, user_content, deployment),
        **_completion_kwargs(deployment),
    )

    return _parse_model_response(
        response.choices[0].message.content, f"page {page_num}"
    )


def _verify_and_supplement(
    client: AzureOpenAI,
    images: list,
    annotations: list[HoleAnnotation],
    unit: str,
    ocr_text: str,
    found_summary: str,
    start_idx: int,
    deployment: str,
) -> None:
    """Second-pass verification: ask the model if any holes were missed."""
    user_content: list[dict] = [{
        "type": "text",
        "text": VERIFY_PROMPT.format(
            found_summary=found_summary,
            unit=unit,
            ocr_context=ocr_text[:4000],
        ),
    }]
    for i, img in enumerate(images):
        user_content.append({"type": "text", "text": f"--- Page {i + 1} ---"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_to_base64(img), "detail": "high"},
        })

    response = client.chat.completions.create(
        model=deployment,
        messages=_build_messages(SYSTEM_PROMPT, user_content, deployment),
        **_completion_kwargs(deployment),
    )

    items = _parse_model_response(
        response.choices[0].message.content, "verification"
    )

    added = 0
    for idx, item in enumerate(items):
        page_num = item.get("page", 1)
        ann = _parse_annotation(item, page_num, start_idx + idx)
        annotations.append(ann)
        added += 1

    if added:
        logger.info("Verification pass added %d additional annotations", added)


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _parse_annotation(item: dict, page_num: int, idx: int) -> HoleAnnotation:
    """Convert a raw model JSON dict into a ``HoleAnnotation``."""
    thread = None
    if item.get("thread_designation"):
        thread = ThreadSpec(
            designation=item["thread_designation"],
            pitch=item.get("thread_pitch"),
            tolerance_class=item.get("thread_tolerance_class"),
        )

    hole_type_str = (item.get("hole_type") or "simple").lower().replace("-", "_")
    try:
        ht = HoleType(hole_type_str)
    except ValueError:
        ht = HoleType.SIMPLE

    return HoleAnnotation(
        annotation_id=item.get("annotation_id", f"H{page_num}_{idx}"),
        hole_type=ht,
        count=item.get("count") or 1,
        diameter=item.get("diameter"),
        diameter_tolerance_plus=item.get("diameter_tolerance_plus"),
        diameter_tolerance_minus=item.get("diameter_tolerance_minus"),
        depth=item.get("depth"),
        depth_tolerance=item.get("depth_tolerance"),
        thread_spec=thread,
        counterbore_diameter=item.get("counterbore_diameter"),
        counterbore_depth=item.get("counterbore_depth"),
        countersink_diameter=item.get("countersink_diameter"),
        countersink_angle=item.get("countersink_angle"),
        position_tolerance=item.get("position_tolerance"),
        datum_refs=item.get("datum_refs") or [],
        fit_designation=item.get("fit_designation"),
        raw_text=item.get("raw_text", ""),
        page=page_num,
        confidence=0.85,
    )


# ---------------------------------------------------------------------------
# Post-extraction filters
# ---------------------------------------------------------------------------

def _filter_null_diameters(
    annotations: list[HoleAnnotation], unit: str = "inch"
) -> list[HoleAnnotation]:
    """Remove annotations with null, zero, or implausibly small diameters."""
    min_dia = 0.1 if unit == "inch" else 1.0
    result = []
    for a in annotations:
        if a.diameter is None or a.diameter <= 0:
            continue
        if a.diameter < min_dia:
            logger.info(
                "Filtered sub-minimum diameter: %s dia=%s (min=%s %s)",
                a.annotation_id, a.diameter, min_dia, unit,
            )
            continue
        result.append(a)
    filtered = len(annotations) - len(result)
    if filtered:
        logger.info("Filtered %d annotations with null/zero/tiny diameter", filtered)
    return result


def _filter_non_hole_annotations(
    annotations: list[HoleAnnotation],
) -> list[HoleAnnotation]:
    """Remove annotations that look like non-hole features based on heuristics."""
    result = []
    for a in annotations:
        raw = a.raw_text or ""

        # Spherical diameters: "S∅", "SØ", etc.
        if re.search(r"(?i)\bS\s*[\u2205\u00d8\uf06e]", raw):
            logger.info("Filtered spherical annotation: %s", raw[:60])
            continue

        has_dia_symbol = any(c in raw for c in "\u2205\u00d8\uf06e")
        has_thread = a.thread_spec is not None

        # Large value without diameter symbol -> likely a linear dimension
        if not has_dia_symbol and not has_thread and a.diameter and a.diameter >= 10.0:
            logger.info("Filtered likely linear dimension: %s dia=%s", raw[:60], a.diameter)
            continue

        # Slot / width features
        if any(kw in raw.lower() for kw in ("slot", "width")):
            logger.info("Filtered slot/width: %s", raw[:60])
            continue

        # Tiny value without diameter symbol -> likely a GD&T tolerance
        if not has_dia_symbol and not has_thread and a.diameter and a.diameter < 0.1:
            logger.info("Filtered likely GD&T tolerance: %s dia=%s", raw[:60], a.diameter)
            continue

        result.append(a)

    filtered = len(annotations) - len(result)
    if filtered:
        logger.info("Filtered %d non-hole annotations", filtered)
    return result


def _deduplicate_within_pages(
    annotations: list[HoleAnnotation],
) -> list[HoleAnnotation]:
    """Remove near-exact duplicates on the same page."""
    result: list[HoleAnnotation] = []
    seen: set[tuple] = set()

    for a in annotations:
        if a.diameter is None:
            result.append(a)
            continue

        raw_sig = (a.raw_text or "").strip()[:30]
        key = (a.page, round(a.diameter, 4), a.count, a.hole_type, raw_sig)
        if key in seen:
            logger.info(
                "Deduplicated %s (dia=%s, count=%d) on page %d",
                a.annotation_id, a.diameter, a.count, a.page,
            )
            continue
        seen.add(key)
        result.append(a)

    filtered = len(annotations) - len(result)
    if filtered:
        logger.info("Deduplicated %d annotations within pages", filtered)
    return result


def _reassign_ids(annotations: list[HoleAnnotation]) -> list[HoleAnnotation]:
    """Re-assign globally unique sequential IDs after filtering."""
    for i, a in enumerate(annotations):
        a.annotation_id = f"H{i + 1}"
    return annotations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_drawing(
    pdf_path: str | Path,
    di_result=None,
    unit: str = "inch",
    dpi: int = 300,
    apply_filters: bool = True,
    multi_page: bool = True,
    model_deployment: str | None = None,
    api_version: str | None = None,
) -> DrawingAnnotations:
    """
    Full vision enrichment pipeline for a drawing PDF.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the engineering drawing PDF.
    di_result : DIExtractionResult, optional
        Pre-computed Azure Document Intelligence result for OCR context.
    unit : str
        Drawing units — ``"inch"`` or ``"mm"``.
    dpi : int
        Resolution for PDF-to-image conversion.
    apply_filters : bool
        Whether to apply post-extraction quality filters.
    multi_page : bool
        Whether to run a second-pass verification sweep across all pages.
    model_deployment : str, optional
        Azure OpenAI deployment name (e.g. ``"gpt-4o"``, ``"o4-mini"``).
        Falls back to the ``AZURE_OPENAI_DEPLOYMENT`` environment variable.
    api_version : str, optional
        Azure OpenAI API version override.
    """
    pdf_path = Path(pdf_path)
    deployment = model_deployment or AzureOpenAIConfig.deployment
    images = pdf_to_images(pdf_path, dpi=dpi)
    client = _build_client(api_version=api_version)

    logger.info("Using model deployment: %s", deployment)

    all_annotations: list[HoleAnnotation] = []

    # Build OCR context from DI result
    full_ocr = ""
    page_ocr: dict[int, str] = {}
    if di_result:
        for page in di_result.pages:
            text = " | ".join(e.content for e in page.elements)
            page_ocr[page.page_number] = text
            full_ocr += f"[Page {page.page_number}] {text}\n"

    # Per-page extraction
    global_idx = 0
    for i, img in enumerate(images):
        page_num = i + 1
        img_b64 = image_to_base64(img)
        ocr_text = page_ocr.get(page_num, "")
        items = _call_vision(
            client, img_b64, page_num, len(images), unit, ocr_text, deployment
        )
        for item in items:
            all_annotations.append(_parse_annotation(item, page_num, global_idx))
            global_idx += 1

    logger.info(
        "Vision enrichment (per-page): %d raw annotations from %d pages",
        len(all_annotations), len(images),
    )

    # Second pass: verification sweep
    if multi_page and len(images) > 1:
        found_dias = [a.diameter for a in all_annotations if a.diameter]
        found_summary = ", ".join(f"{d}" for d in sorted(set(found_dias)))
        _verify_and_supplement(
            client, images, all_annotations, unit, full_ocr,
            found_summary, global_idx, deployment,
        )

    if apply_filters:
        all_annotations = _filter_null_diameters(all_annotations, unit=unit)
        all_annotations = _filter_non_hole_annotations(all_annotations)
        all_annotations = _deduplicate_within_pages(all_annotations)
        all_annotations = _reassign_ids(all_annotations)
        logger.info("After filtering: %d annotations", len(all_annotations))

    return DrawingAnnotations(
        source_pdf=str(pdf_path),
        unit=unit,
        annotations=all_annotations,
    )
