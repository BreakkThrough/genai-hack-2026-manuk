"""Vision-model enrichment: extract structured hole annotations from drawing images."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from openai import AzureOpenAI
from PIL import Image

from app.config import AzureOpenAIConfig
from app.models.schemas import (
    DrawingAnnotations,
    HoleAnnotation,
    HoleType,
    ThreadSpec,
)
from app.utils.pdf_utils import (
    crop_annotation_regions,
    image_to_base64,
    pdf_to_images,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR regex pre-detection of diameter callouts
# ---------------------------------------------------------------------------

_DIA_CHARS = "\u2205\u00d8\uf06e"  # ∅, Ø, PUA diameter

_OCR_COUNTED_DIA_RE = re.compile(
    rf"(\d+)\s*X\s*[{_DIA_CHARS}]?\s*\.?(\d+\.?\d*)", re.IGNORECASE
)
_OCR_SINGLE_DIA_RE = re.compile(rf"[{_DIA_CHARS}]\s*\.?(\d+\.?\d*)")
_OCR_THREAD_RE = re.compile(r"M(\d+(?:\.\d+)?)\s*[xX\u00d7]")


@dataclass
class OCRDiameterHit:
    """A diameter value detected by OCR regex, with occurrence count."""
    diameter: float
    count: int = 1
    source_text: str = ""
    page: int = 0


def _detect_ocr_diameters(
    page_ocr: dict[int, str],
    unit: str = "inch",
) -> list[OCRDiameterHit]:
    """Scan OCR text for diameter patterns and return unique hits."""
    min_dia = 0.1 if unit == "inch" else 1.0
    hits: list[OCRDiameterHit] = []

    for page_num, text in page_ocr.items():
        for m in _OCR_COUNTED_DIA_RE.finditer(text):
            count = int(m.group(1))
            dia = float(m.group(2))
            if dia >= min_dia:
                hits.append(OCRDiameterHit(
                    diameter=dia, count=count,
                    source_text=m.group(0).strip(), page=page_num,
                ))

        for m in _OCR_SINGLE_DIA_RE.finditer(text):
            dia = float(m.group(1))
            if dia >= min_dia:
                already = any(
                    abs(h.diameter - dia) < 0.001 and h.page == page_num
                    for h in hits
                )
                if not already:
                    hits.append(OCRDiameterHit(
                        diameter=dia, count=1,
                        source_text=m.group(0).strip(), page=page_num,
                    ))

        for m in _OCR_THREAD_RE.finditer(text):
            dia = float(m.group(1))
            if dia >= min_dia:
                hits.append(OCRDiameterHit(
                    diameter=dia, count=1,
                    source_text=m.group(0).strip(), page=page_num,
                ))

    unique: dict[float, OCRDiameterHit] = {}
    for h in hits:
        key = round(h.diameter, 3)
        if key not in unique or h.count > unique[key].count:
            unique[key] = h
    return sorted(unique.values(), key=lambda h: h.diameter)


def _format_ocr_hints(ocr_hits: list[OCRDiameterHit]) -> str:
    """Format OCR-detected diameters into a prompt hint string."""
    if not ocr_hits:
        return ""
    dia_strs = []
    for h in ocr_hits:
        s = f"{h.diameter}"
        if h.count > 1:
            s = f"{h.count}X {s}"
        dia_strs.append(s)
    return (
        "\n\nOCR PRE-SCAN DETECTED DIAMETERS: ["
        + ", ".join(dia_strs)
        + "]\nENSURE every diameter in this list is accounted for in your "
        "extraction. If you cannot find a corresponding hole annotation for "
        "a listed diameter, look more carefully in detail views, section "
        "views, and small annotation clusters."
    )

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
{ocr_diameter_hints}

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
- After extracting holes, perform a second scan of the page to ensure no \
  diameter callouts were missed. Check detail views and section views.

Return a JSON array.
"""

VERIFY_PROMPT = """\
I already extracted these hole annotations from the drawing:
{found_summary}

Now look at ALL pages again carefully. Are there ANY additional hole \
annotations (∅ diameter callouts) that I MISSED?
{ocr_diameter_hints}

Look specifically for:
- Holes with the same diameter that appear as SEPARATE callouts with \
  different GD&T, position tolerances, or datum references
- Holes on different views (section views, detail views, DETAIL A, etc.)
- Counterbore components (through-hole + counterbore as separate diameters)
- Small holes (∅.250, ∅.400, etc.) that may be in detail views
- Any diameter from the OCR pre-scan list above that is NOT yet in my \
  extraction — these are likely missed holes

IMPORTANT: If a diameter appears in the OCR text but is not assigned to \
a hole in my extraction, create a new hole annotation for it.

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
    ocr_diameter_hints: str = "",
) -> list[dict]:
    """Send one page image to the model and return parsed annotation dicts."""
    user_content = [
        {"type": "text", "text": USER_PROMPT_TEMPLATE.format(
            page_num=page_num,
            total_pages=total_pages,
            unit=unit,
            ocr_context=ocr_text[:4000],
            ocr_diameter_hints=ocr_diameter_hints,
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


CROPPED_REGION_PROMPT = """\
This is a CROPPED region from page {page_num} of an engineering drawing.
Drawing units: {unit}.

The crop focuses on an annotation cluster that may contain hole callouts.
Extract ALL hole-related annotations visible in this cropped region.
{ocr_diameter_hints}

Return a JSON array. If no holes are visible, return [].
Each entry must have: annotation_id, hole_type, count, diameter, raw_text.
"""

RECONCILE_PROMPT = """\
I extracted these hole annotations from the drawing:
{found_summary}

However, the OCR text contains diameter values that are NOT yet accounted for:
  Missing diameters: {missing_diameters}

Look at the drawing pages carefully and find the hole annotations for \
these specific diameters. They may be in detail views, section views, or \
small annotation clusters that were overlooked.

Drawing units: {unit}.

OCR context:
{ocr_context}

Return a JSON array with ONLY the hole annotations for the missing diameters.
If you truly cannot find them, return [].
Each entry must have: annotation_id, hole_type, count, diameter, raw_text, page.
"""


def _call_vision_cropped(
    client: AzureOpenAI,
    cropped_images: list[tuple[int, Image.Image]],
    annotations: list[HoleAnnotation],
    unit: str,
    start_idx: int,
    deployment: str,
    ocr_diameter_hints: str = "",
) -> None:
    """Third pass: run vision on cropped annotation regions."""
    added = 0
    for page_num, crop_img in cropped_images:
        img_b64 = image_to_base64(crop_img)
        user_content = [
            {"type": "text", "text": CROPPED_REGION_PROMPT.format(
                page_num=page_num,
                unit=unit,
                ocr_diameter_hints=ocr_diameter_hints,
            )},
            {"type": "image_url", "image_url": {"url": img_b64, "detail": "high"}},
        ]

        response = client.chat.completions.create(
            model=deployment,
            messages=_build_messages(SYSTEM_PROMPT, user_content, deployment),
            **_completion_kwargs(deployment),
        )

        items = _parse_model_response(
            response.choices[0].message.content, f"cropped page {page_num}"
        )
        for item in items:
            ann = _parse_annotation(item, page_num, start_idx + added)
            annotations.append(ann)
            added += 1

    if added:
        logger.info("Cropped-region pass added %d annotations", added)


def _reconcile_ocr_vs_llm(
    client: AzureOpenAI,
    images: list[Image.Image],
    annotations: list[HoleAnnotation],
    ocr_hits: list[OCRDiameterHit],
    unit: str,
    ocr_text: str,
    start_idx: int,
    deployment: str,
) -> None:
    """Cross-reference OCR-detected diameters against LLM output;
    trigger a targeted follow-up for any missing diameters."""
    extracted_dias: Counter[float] = Counter()
    for a in annotations:
        if a.diameter is not None:
            extracted_dias[round(a.diameter, 3)] += 1

    missing: list[float] = []
    for h in ocr_hits:
        key = round(h.diameter, 3)
        if extracted_dias.get(key, 0) == 0:
            missing.append(h.diameter)

    if not missing:
        logger.info("OCR reconciliation: all OCR diameters accounted for")
        return

    logger.info("OCR reconciliation: %d missing diameters: %s", len(missing), missing)

    found_dias = sorted({a.diameter for a in annotations if a.diameter})
    found_summary = ", ".join(f"∅{d}" for d in found_dias)
    missing_str = ", ".join(f"∅{d}" for d in missing)

    user_content: list[dict] = [{
        "type": "text",
        "text": RECONCILE_PROMPT.format(
            found_summary=found_summary,
            missing_diameters=missing_str,
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
        response.choices[0].message.content, "reconciliation"
    )

    added = 0
    for idx, item in enumerate(items):
        page_num = item.get("page", 1)
        ann = _parse_annotation(item, page_num, start_idx + idx)
        annotations.append(ann)
        added += 1

    if added:
        logger.info("Reconciliation pass added %d annotations", added)


def _verify_and_supplement(
    client: AzureOpenAI,
    images: list,
    annotations: list[HoleAnnotation],
    unit: str,
    ocr_text: str,
    found_summary: str,
    start_idx: int,
    deployment: str,
    ocr_diameter_hints: str = "",
) -> None:
    """Second-pass verification: ask the model if any holes were missed."""
    user_content: list[dict] = [{
        "type": "text",
        "text": VERIFY_PROMPT.format(
            found_summary=found_summary,
            unit=unit,
            ocr_context=ocr_text[:4000],
            ocr_diameter_hints=ocr_diameter_hints,
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


def _richness(ann: HoleAnnotation) -> int:
    """Score how much GD&T detail an annotation carries."""
    score = 0
    if ann.position_tolerance is not None:
        score += 2
    if ann.datum_refs:
        score += len(ann.datum_refs)
    if ann.diameter_tolerance_plus is not None:
        score += 1
    if ann.depth is not None:
        score += 1
    if ann.counterbore_diameter is not None:
        score += 2
    if ann.raw_text:
        score += len(ann.raw_text) // 10
    return score


def _strip_raw(text: str) -> str:
    """Normalise raw_text for comparison: strip non-ASCII noise, collapse whitespace."""
    cleaned = []
    for ch in text:
        if ch.isascii() or ch in "\u2205\u00d8\uf06e":
            cleaned.append(ch)
    return " ".join("".join(cleaned).split()).lower()


def _raw_text_similar(a: str, b: str) -> bool:
    """Check if two raw_text strings are substantially similar."""
    a, b = _strip_raw(a), _strip_raw(b)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter in longer:
        return True
    prefix_len = min(15, len(shorter))
    if prefix_len >= 5 and shorter[:prefix_len] == longer[:prefix_len]:
        return True
    return False


def _deduplicate_cross_page(
    annotations: list[HoleAnnotation],
) -> list[HoleAnnotation]:
    """Remove duplicates across pages and from multi-pass extraction.

    Two-stage approach:
      1. Group by (diameter, hole_type) and keep only the richest annotation
         per distinct (diameter, count, datum_refs signature) combination.
      2. Within a diameter group, collapse annotations that clearly refer to
         the same callout (same count or similar raw_text).
    """
    if len(annotations) <= 1:
        return annotations

    result: list[HoleAnnotation] = []
    skip: set[int] = set()

    for i, a in enumerate(annotations):
        if i in skip or a.diameter is None:
            if i not in skip:
                result.append(a)
            continue

        duplicates = [i]
        for j in range(i + 1, len(annotations)):
            if j in skip:
                continue
            b = annotations[j]
            if b.diameter is None:
                continue
            if round(a.diameter, 3) != round(b.diameter, 3):
                continue
            if a.hole_type != b.hole_type:
                continue

            similar_text = _raw_text_similar(a.raw_text, b.raw_text)
            same_datums = (
                set(a.datum_refs) == set(b.datum_refs) and bool(a.datum_refs)
            )
            same_page_and_count = (a.page == b.page and a.count == b.count)

            if similar_text or same_datums or same_page_and_count:
                duplicates.append(j)

        if len(duplicates) > 1:
            best_idx = max(duplicates, key=lambda idx: _richness(annotations[idx]))
            best = annotations[best_idx]
            for idx in duplicates:
                if idx != best_idx:
                    other = annotations[idx]
                    if other.count > best.count:
                        best.count = other.count
                    skip.add(idx)
                    logger.info(
                        "Cross-page dedup: dropped %s (dia=%s, page=%d) "
                        "in favour of %s (page=%d)",
                        other.annotation_id, other.diameter, other.page,
                        best.annotation_id, best.page,
                    )
            result.append(best)
        else:
            result.append(a)

    filtered = len(annotations) - len(result)
    if filtered:
        logger.info("Cross-page dedup removed %d annotations", filtered)
    return result


def _deduplicate_by_diameter(
    annotations: list[HoleAnnotation],
) -> list[HoleAnnotation]:
    """Final safety net: for each unique diameter, ensure we don't have more
    annotations than is plausible. Keeps annotations that have distinct
    datum_refs or position_tolerance values (indicating genuinely different
    callouts) and collapses the rest."""
    if len(annotations) <= 1:
        return annotations

    from collections import defaultdict
    by_dia: dict[float, list[int]] = defaultdict(list)
    for i, a in enumerate(annotations):
        if a.diameter is not None:
            by_dia[round(a.diameter, 3)].append(i)

    skip: set[int] = set()
    for dia_key, indices in by_dia.items():
        if len(indices) <= 1:
            continue

        kept: list[int] = []
        for idx in indices:
            a = annotations[idx]
            is_dup = False
            for kidx in kept:
                k = annotations[kidx]
                if (a.hole_type == k.hole_type
                        and a.count == k.count
                        and set(a.datum_refs) == set(k.datum_refs)):
                    skip.add(idx)
                    is_dup = True
                    logger.info(
                        "Diameter dedup: dropped %s (dia=%s, page=%d, datums=%s) "
                        "duplicate of %s (page=%d)",
                        a.annotation_id, a.diameter, a.page, a.datum_refs,
                        k.annotation_id, k.page,
                    )
                    break
            if not is_dup:
                kept.append(idx)

    result = [a for i, a in enumerate(annotations) if i not in skip]
    filtered = len(annotations) - len(result)
    if filtered:
        logger.info("Diameter-based dedup removed %d annotations", filtered)
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

    Pipeline stages:
        1. Build OCR context from Azure DI result
        2. OCR regex pre-detection of diameter callouts
        3. Per-page LLM extraction (with OCR diameter hints)
        4. Verification pass with OCR hints
        5. Cropped-region pass on dense annotation clusters
        6. OCR-vs-LLM reconciliation for missed diameters
        7. Filters + within-page dedup + cross-page dedup
        8. Reassign sequential IDs

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

    # --- Stage 1: Build OCR context from DI result ---
    full_ocr = ""
    page_ocr: dict[int, str] = {}
    if di_result:
        for page in di_result.pages:
            text = " | ".join(e.content for e in page.elements)
            page_ocr[page.page_number] = text
            full_ocr += f"[Page {page.page_number}] {text}\n"

    # --- Stage 2: OCR regex pre-detection ---
    ocr_hits = _detect_ocr_diameters(page_ocr, unit=unit)
    ocr_hints_str = _format_ocr_hints(ocr_hits)
    if ocr_hits:
        logger.info(
            "OCR pre-scan detected %d unique diameters: %s",
            len(ocr_hits),
            [h.diameter for h in ocr_hits],
        )

    # --- Stage 3: Per-page LLM extraction (with OCR hints) ---
    global_idx = 0
    for i, img in enumerate(images):
        page_num = i + 1
        img_b64 = image_to_base64(img)
        ocr_text = page_ocr.get(page_num, "")
        items = _call_vision(
            client, img_b64, page_num, len(images), unit, ocr_text,
            deployment, ocr_diameter_hints=ocr_hints_str,
        )
        for item in items:
            all_annotations.append(_parse_annotation(item, page_num, global_idx))
            global_idx += 1

    logger.info(
        "Vision enrichment (per-page): %d raw annotations from %d pages",
        len(all_annotations), len(images),
    )

    # --- Stage 4: Verification pass with OCR hints ---
    if multi_page and len(images) > 1:
        found_dias = [a.diameter for a in all_annotations if a.diameter]
        found_summary = ", ".join(f"∅{d}" for d in sorted(set(found_dias)))
        _verify_and_supplement(
            client, images, all_annotations, unit, full_ocr,
            found_summary, global_idx, deployment,
            ocr_diameter_hints=ocr_hints_str,
        )
        global_idx = len(all_annotations)

    # --- Stage 5: Cropped-region pass ---
    if di_result:
        cropped_regions = crop_annotation_regions(images, di_result)
        if cropped_regions:
            _call_vision_cropped(
                client, cropped_regions, all_annotations, unit,
                global_idx, deployment, ocr_diameter_hints=ocr_hints_str,
            )
            global_idx = len(all_annotations)

    # --- Stage 6: OCR-vs-LLM reconciliation ---
    if ocr_hits:
        _reconcile_ocr_vs_llm(
            client, images, all_annotations, ocr_hits, unit,
            full_ocr, global_idx, deployment,
        )

    # --- Stage 7 & 8: Filters + dedup + reassign IDs ---
    if apply_filters:
        all_annotations = _filter_null_diameters(all_annotations, unit=unit)
        all_annotations = _filter_non_hole_annotations(all_annotations)
        all_annotations = _deduplicate_within_pages(all_annotations)
        all_annotations = _deduplicate_cross_page(all_annotations)
        all_annotations = _deduplicate_by_diameter(all_annotations)
        all_annotations = _reassign_ids(all_annotations)
        logger.info("After filtering: %d annotations", len(all_annotations))

    return DrawingAnnotations(
        source_pdf=str(pdf_path),
        unit=unit,
        annotations=all_annotations,
    )
