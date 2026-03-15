"""Hybrid deterministic + LLM correlation between drawing annotations and 3D hole features."""

from __future__ import annotations

import json
import logging

from openai import AzureOpenAI

from app.config import AzureOpenAIConfig
from app.models.schemas import (
    DrawingAnnotations,
    EvidenceTrace,
    FeatureMapping,
    HoleAnnotation,
    HoleFeature3D,
    HoleType,
    LinkageResult,
    MatchConfidence,
    StepFeatures,
)
from app.utils.geometry_utils import diameter_matches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic matching helpers
# ---------------------------------------------------------------------------

def _match_by_diameter(
    annotation: HoleAnnotation,
    holes: list[HoleFeature3D],
    rel_tol: float = 0.05,
) -> list[HoleFeature3D]:
    """Return holes whose primary diameter matches the annotation diameter."""
    if annotation.diameter is None:
        return []
    return [
        h for h in holes
        if diameter_matches(annotation.diameter, h.primary_diameter, rel_tol)
    ]


def _match_by_count(
    annotation: HoleAnnotation,
    candidates: list[HoleFeature3D],
) -> list[HoleFeature3D]:
    """If the annotation count matches the number of candidates, keep them all."""
    return candidates


def _match_counterbore(
    annotation: HoleAnnotation,
    holes: list[HoleFeature3D],
    rel_tol: float = 0.05,
) -> list[HoleFeature3D]:
    """Match counterbore annotations by both primary and counterbore diameters."""
    if annotation.hole_type != HoleType.COUNTERBORE or annotation.counterbore_diameter is None:
        return []
    matches = []
    for h in holes:
        if h.hole_type != HoleType.COUNTERBORE or h.counterbore_diameter is None:
            continue
        dia_ok = (
            annotation.diameter is None
            or diameter_matches(annotation.diameter, h.primary_diameter, rel_tol)
        )
        cb_ok = diameter_matches(
            annotation.counterbore_diameter, h.counterbore_diameter, rel_tol
        )
        if dia_ok and cb_ok:
            matches.append(h)
    return matches


def _match_thread(
    annotation: HoleAnnotation,
    holes: list[HoleFeature3D],
    rel_tol: float = 0.10,
) -> list[HoleFeature3D]:
    """
    Thread callouts (e.g. M3 x 0.5) appear as plain cylinders whose diameter
    approximates the thread minor/major diameter.  Use wider tolerance.
    """
    if annotation.thread_spec is None or annotation.diameter is None:
        return []
    return [
        h for h in holes
        if diameter_matches(annotation.diameter, h.primary_diameter, rel_tol)
    ]


# ---------------------------------------------------------------------------
# LLM disambiguation
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
You are a manufacturing engineer correlating 2-D drawing annotations with 3-D \
CAD hole features.

ANNOTATION (from the drawing):
{annotation_json}

CANDIDATE 3-D HOLES (from STEP model):
{candidates_json}

Pick the best matching hole(s) for this annotation. Consider:
- diameter match (annotation vs 3D)
- hole type (counterbore, threaded, through, blind)
- count / multiplicity
- depth if available
- any contextual clues in raw_text

Return ONLY a JSON object:
{{
  "matched_hole_ids": ["hole_X", ...],
  "confidence": "high" | "medium" | "low",
  "reason": "<brief explanation>"
}}
"""


def _llm_disambiguate(
    annotation: HoleAnnotation,
    candidates: list[HoleFeature3D],
) -> tuple[list[str], MatchConfidence, str]:
    """Use LLM to pick the best match from ambiguous candidates."""
    try:
        client = AzureOpenAI(
            azure_endpoint=AzureOpenAIConfig.endpoint,
            api_key=AzureOpenAIConfig.key,
            api_version=AzureOpenAIConfig.api_version,
        )

        ann_json = annotation.model_dump_json(indent=2, exclude={"bounding_box"})
        cands_json = json.dumps(
            [
                {
                    "hole_id": c.hole_id,
                    "primary_diameter": c.primary_diameter,
                    "primary_depth": c.primary_depth,
                    "hole_type": c.hole_type.value,
                    "counterbore_diameter": c.counterbore_diameter,
                    "is_through": c.is_through,
                    "center": c.center.model_dump(),
                    "axis": c.axis.model_dump(),
                }
                for c in candidates
            ],
            indent=2,
        )

        model = AzureOpenAIConfig.deployment
        is_reasoning = model.lower().startswith(("o1", "o3", "o4"))
        sys_msg = "You are a manufacturing engineer."
        usr_msg = _LLM_PROMPT.format(
            annotation_json=ann_json, candidates_json=cands_json
        )

        if is_reasoning:
            messages = [{"role": "user", "content": f"{sys_msg}\n\n{usr_msg}"}]
            extra: dict = {"max_completion_tokens": 2048}
        else:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ]
            extra = {"temperature": 0.0, "max_tokens": 512}

        resp = client.chat.completions.create(
            model=model, messages=messages, **extra
        )

        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        data = json.loads(raw.strip())
        ids = data.get("matched_hole_ids", [])
        conf_str = data.get("confidence", "medium")
        reason = data.get("reason", "LLM match")

        try:
            conf = MatchConfidence(conf_str)
        except ValueError:
            conf = MatchConfidence.MEDIUM

        return ids, conf, reason
    except Exception as exc:
        logger.warning("LLM disambiguation failed: %s", exc)
        return (
            [c.hole_id for c in candidates[:1]],
            MatchConfidence.LOW,
            f"LLM fallback error: {exc}",
        )


# ---------------------------------------------------------------------------
# Evidence & interpretation helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_SCORE = {
    MatchConfidence.HIGH: 0.95,
    MatchConfidence.MEDIUM: 0.70,
    MatchConfidence.LOW: 0.35,
}


def _build_evidence(ann: HoleAnnotation) -> EvidenceTrace:
    return EvidenceTrace(
        page=ann.page,
        bounding_box=ann.bounding_box,
        raw_text=ann.raw_text,
    )


def _build_interpretation(ann: HoleAnnotation) -> dict:
    interp: dict = {
        "hole_type": ann.hole_type.value,
        "diameter": ann.diameter,
        "count": ann.count,
    }
    if ann.thread_spec:
        interp["thread_type"] = ann.thread_spec.designation
        if ann.thread_spec.pitch:
            interp["thread_pitch"] = ann.thread_spec.pitch
        if ann.thread_spec.tolerance_class:
            interp["tolerance_class"] = ann.thread_spec.tolerance_class
    if ann.depth is not None:
        interp["depth"] = ann.depth
    if ann.fit_designation:
        interp["fit_designation"] = ann.fit_designation
    if ann.counterbore_diameter is not None:
        interp["counterbore_diameter"] = ann.counterbore_diameter
    if ann.counterbore_depth is not None:
        interp["counterbore_depth"] = ann.counterbore_depth
    if ann.countersink_diameter is not None:
        interp["countersink_diameter"] = ann.countersink_diameter
    if ann.countersink_angle is not None:
        interp["countersink_angle"] = ann.countersink_angle
    if ann.position_tolerance is not None:
        interp["position_tolerance"] = ann.position_tolerance
    if ann.datum_refs:
        interp["datum_refs"] = ann.datum_refs
    if ann.diameter_tolerance_plus is not None:
        interp["diameter_tolerance_plus"] = ann.diameter_tolerance_plus
    if ann.diameter_tolerance_minus is not None:
        interp["diameter_tolerance_minus"] = ann.diameter_tolerance_minus
    return interp


# ---------------------------------------------------------------------------
# Main correlation pipeline
# ---------------------------------------------------------------------------

def correlate(
    annotations: DrawingAnnotations,
    features: StepFeatures,
    diameter_tol: float = 0.05,
    use_llm: bool = True,
) -> LinkageResult:
    """
    Correlate drawing annotations with 3D hole features:
    1. Deterministic diameter + count + counterbore matching
    2. LLM disambiguation for ambiguous cases
    """
    holes = features.holes
    mappings: list[FeatureMapping] = []
    mapped_ann_ids: set[str] = set()
    mapped_hole_ids: set[str] = set()

    for ann in annotations.annotations:
        candidates: list[HoleFeature3D] = []
        reasons: list[str] = []
        confidence = MatchConfidence.HIGH

        # Counterbore-specific matching
        if ann.hole_type == HoleType.COUNTERBORE and ann.counterbore_diameter:
            candidates = _match_counterbore(ann, holes, diameter_tol)
            if candidates:
                reasons.append("counterbore diameter + hole diameter match")

        # Thread-specific matching
        if not candidates and ann.thread_spec:
            candidates = _match_thread(ann, holes, rel_tol=0.10)
            if candidates:
                reasons.append("thread diameter match (wider tolerance)")

        # Generic diameter matching
        if not candidates:
            candidates = _match_by_diameter(ann, holes, diameter_tol)
            if candidates:
                reasons.append("diameter match")

        # Count filtering
        if candidates and ann.count > 1:
            before = len(candidates)
            candidates = _match_by_count(ann, candidates)
            if len(candidates) != before:
                reasons.append(f"count filter {ann.count}X")

        # Ambiguity resolution via LLM
        if len(candidates) > ann.count and use_llm:
            ids, confidence, reason = _llm_disambiguate(ann, candidates)
            id_set = set(ids)
            candidates = [c for c in candidates if c.hole_id in id_set]
            reasons.append(f"LLM disambiguated: {reason}")
        elif len(candidates) > ann.count:
            confidence = MatchConfidence.LOW
            reasons.append("multiple candidates, no LLM")

        if not candidates:
            confidence = MatchConfidence.LOW

        hole_ids = [c.hole_id for c in candidates]
        mappings.append(FeatureMapping(
            annotation_id=ann.annotation_id,
            hole_ids=hole_ids,
            confidence=confidence,
            confidence_score=_CONFIDENCE_SCORE.get(confidence, 0.5),
            match_reasons=reasons,
            evidence=_build_evidence(ann),
            parsed_interpretation=_build_interpretation(ann),
        ))
        mapped_ann_ids.add(ann.annotation_id)
        mapped_hole_ids.update(hole_ids)

    unmapped_annotations = [
        a.annotation_id
        for a in annotations.annotations
        if a.annotation_id not in mapped_ann_ids
    ]
    unmapped_holes = [
        h.hole_id for h in holes if h.hole_id not in mapped_hole_ids
    ]

    logger.info(
        "Correlation complete: %d mappings, %d unmapped annotations, %d unmapped holes",
        len(mappings), len(unmapped_annotations), len(unmapped_holes),
    )

    return LinkageResult(
        drawing_pdf=annotations.source_pdf,
        step_file=features.source_step,
        annotations=annotations,
        features_3d=features,
        mappings=mappings,
        unmapped_annotations=unmapped_annotations,
        unmapped_holes=unmapped_holes,
    )
