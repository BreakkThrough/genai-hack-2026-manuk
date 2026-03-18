"""Hybrid deterministic + LLM correlation between drawing annotations and 3D hole features."""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from openai import AzureOpenAI

from app.config import AzureOpenAIConfig
from app.models.schemas import (
    DrawingAnnotations,
    EvidenceTrace,
    FeatureGroup,
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
# Internal match-detail record
# ---------------------------------------------------------------------------

class _CylMatch:
    """Tracks which cylinder inside a hole matched and how close it was."""

    __slots__ = ("hole", "cylinder_id", "diameter_delta", "reason")

    def __init__(
            self,
            hole: HoleFeature3D,
            cylinder_id: str | None,
            diameter_delta: float,
            reason: str,
    ):
        self.hole = hole
        self.cylinder_id = cylinder_id
        self.diameter_delta = diameter_delta
        self.reason = reason


# ---------------------------------------------------------------------------
# Tolerance-aware diameter comparison
# ---------------------------------------------------------------------------

def _effective_tolerance(
        annotation: HoleAnnotation,
        rel_tol: float,
) -> tuple[float, float]:
    """Return (abs_tol, rel_tol) to use for this annotation.

    When the drawing supplies explicit plus/minus tolerances, derive an
    absolute tolerance from them (whichever is larger) so the match uses
    the drawing's own specification.  Fall back to the global rel_tol.
    """
    tol_plus = annotation.diameter_tolerance_plus or 0.0
    tol_minus = abs(annotation.diameter_tolerance_minus or 0.0)
    abs_tol = max(tol_plus, tol_minus)
    return abs_tol, rel_tol


def _diameters_close(
        ann_dia: float,
        cad_dia: float,
        abs_tol: float,
        rel_tol: float,
) -> bool:
    """True when diameters match under either the absolute or relative rule."""
    if abs_tol > 0 and abs(ann_dia - cad_dia) <= abs_tol:
        return True
    return diameter_matches(ann_dia, cad_dia, rel_tol)


# ---------------------------------------------------------------------------
# Deterministic matching helpers (return _CylMatch details)
# ---------------------------------------------------------------------------

def _match_by_diameter(
        annotation: HoleAnnotation,
        holes: list[HoleFeature3D],
        rel_tol: float = 0.05,
) -> list[_CylMatch]:
    """Return holes that contain any cylinder matching the annotation diameter.

    Checks every cylinder inside each hole, not just the primary diameter.
    Tracks the specific cylinder and diameter delta for auditability.
    """
    if annotation.diameter is None:
        return []

    abs_tol, rel = _effective_tolerance(annotation, rel_tol)
    results: list[_CylMatch] = []

    for h in holes:
        if _diameters_close(annotation.diameter, h.primary_diameter, abs_tol, rel):
            results.append(_CylMatch(
                hole=h,
                cylinder_id=None,
                diameter_delta=abs(annotation.diameter - h.primary_diameter),
                reason=f"matched primary diameter {h.primary_diameter}",
            ))
            continue
        for cyl in h.cylinders:
            if _diameters_close(annotation.diameter, cyl.diameter, abs_tol, rel):
                results.append(_CylMatch(
                    hole=h,
                    cylinder_id=cyl.feature_id,
                    diameter_delta=abs(annotation.diameter - cyl.diameter),
                    reason=f"matched cylinder diameter {cyl.diameter}",
                ))
                break

    return results


def _match_by_count(
        annotation: HoleAnnotation,
        candidates: list[_CylMatch],
) -> list[_CylMatch]:
    """If the annotation count matches the number of candidates, keep them all."""
    return candidates


def _match_counterbore(
        annotation: HoleAnnotation,
        holes: list[HoleFeature3D],
        rel_tol: float = 0.05,
) -> list[_CylMatch]:
    """Match counterbore annotations by both primary and counterbore diameters.

    Checks cylinders inside the hole when the primary/counterbore diameter
    fields don't match directly.
    """
    if annotation.hole_type != HoleType.COUNTERBORE or annotation.counterbore_diameter is None:
        return []

    abs_tol, rel = _effective_tolerance(annotation, rel_tol)
    results: list[_CylMatch] = []

    for h in holes:
        if h.hole_type != HoleType.COUNTERBORE or h.counterbore_diameter is None:
            continue

        dia_ok = annotation.diameter is None
        matched_cyl_id: str | None = None
        dia_delta = 0.0

        if not dia_ok and annotation.diameter is not None:
            if _diameters_close(annotation.diameter, h.primary_diameter, abs_tol, rel):
                dia_ok = True
                dia_delta = abs(annotation.diameter - h.primary_diameter)
            else:
                for cyl in h.cylinders:
                    if _diameters_close(annotation.diameter, cyl.diameter, abs_tol, rel):
                        dia_ok = True
                        matched_cyl_id = cyl.feature_id
                        dia_delta = abs(annotation.diameter - cyl.diameter)
                        break

        cb_ok = _diameters_close(
            annotation.counterbore_diameter, h.counterbore_diameter, abs_tol, rel
        )
        if not cb_ok:
            for cyl in h.cylinders:
                if _diameters_close(annotation.counterbore_diameter, cyl.diameter, abs_tol, rel):
                    cb_ok = True
                    if matched_cyl_id is None:
                        matched_cyl_id = cyl.feature_id
                    break

        if dia_ok and cb_ok:
            results.append(_CylMatch(
                hole=h,
                cylinder_id=matched_cyl_id,
                diameter_delta=dia_delta,
                reason="counterbore diameter + hole diameter match (cylinder-level)",
            ))

    return results


def _match_thread(
        annotation: HoleAnnotation,
        holes: list[HoleFeature3D],
        rel_tol: float = 0.10,
) -> list[_CylMatch]:
    """Thread callouts appear as plain cylinders whose diameter approximates
    the thread minor/major diameter.  Use wider tolerance and check all
    cylinders inside each hole.
    """
    if annotation.thread_spec is None or annotation.diameter is None:
        return []

    abs_tol, _ = _effective_tolerance(annotation, rel_tol)
    results: list[_CylMatch] = []

    for h in holes:
        if _diameters_close(annotation.diameter, h.primary_diameter, abs_tol, rel_tol):
            results.append(_CylMatch(
                hole=h,
                cylinder_id=None,
                diameter_delta=abs(annotation.diameter - h.primary_diameter),
                reason=f"thread diameter matched primary {h.primary_diameter}",
            ))
            continue
        for cyl in h.cylinders:
            if _diameters_close(annotation.diameter, cyl.diameter, abs_tol, rel_tol):
                results.append(_CylMatch(
                    hole=h,
                    cylinder_id=cyl.feature_id,
                    diameter_delta=abs(annotation.diameter - cyl.diameter),
                    reason=f"thread diameter matched cylinder {cyl.diameter}",
                ))
                break

    return results


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
        diameter_delta: float,
        annotation: HoleAnnotation,
        reason_count: int,
) -> tuple[MatchConfidence, float]:
    """Compute a confidence level and numeric score from match signals.

    Signals weighted:
      - diameter closeness   (60%)
      - drawing tolerance fit (20%)  — did the match fall within the
        annotation's own stated tolerance?
      - reason richness      (20%)  — more corroborating reasons = higher
    """
    if annotation.diameter is None or annotation.diameter == 0:
        return MatchConfidence.LOW, 0.35

    tol_plus = annotation.diameter_tolerance_plus or 0.0
    tol_minus = abs(annotation.diameter_tolerance_minus or 0.0)
    stated_tol = max(tol_plus, tol_minus)

    rel_diff = diameter_delta / annotation.diameter if annotation.diameter else 1.0
    dia_score = max(0.0, 1.0 - min(rel_diff / 0.05, 1.0))

    if stated_tol > 0 and diameter_delta <= stated_tol:
        tol_score = 1.0
    elif stated_tol > 0:
        tol_score = max(0.0, 1.0 - (diameter_delta - stated_tol) / stated_tol)
    else:
        tol_score = 0.5

    reason_score = min(reason_count / 3.0, 1.0)

    score = 0.60 * dia_score + 0.20 * tol_score + 0.20 * reason_score

    if score >= 0.80:
        return MatchConfidence.HIGH, round(min(score, 0.99), 2)
    if score >= 0.55:
        return MatchConfidence.MEDIUM, round(score, 2)
    return MatchConfidence.LOW, round(score, 2)


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
        candidates: list[_CylMatch],
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
                    "hole_id": m.hole.hole_id,
                    "primary_diameter": m.hole.primary_diameter,
                    "primary_depth": m.hole.primary_depth,
                    "hole_type": m.hole.hole_type.value,
                    "counterbore_diameter": m.hole.counterbore_diameter,
                    "is_through": m.hole.is_through,
                    "center": m.hole.center.model_dump(),
                    "axis": m.hole.axis.model_dump(),
                    "cylinders": [
                        {"feature_id": c.feature_id, "diameter": c.diameter}
                        for c in m.hole.cylinders
                    ],
                }
                for m in candidates
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
            [candidates[0].hole.hole_id] if candidates else [],
            MatchConfidence.LOW,
            f"LLM fallback error: {exc}",
        )


# ---------------------------------------------------------------------------
# Evidence & interpretation helpers
# ---------------------------------------------------------------------------

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
# Feature group builder
# ---------------------------------------------------------------------------

def _build_feature_groups(
        mappings: list[FeatureMapping],
        holes: list[HoleFeature3D],
) -> list[FeatureGroup]:
    """Aggregate mappings into per-feature groups."""
    hole_map = {h.hole_id: h for h in holes}

    groups: dict[str, dict] = defaultdict(lambda: {
        "annotation_ids": [],
        "matched_cylinder_ids": set(),
        "min_confidence": MatchConfidence.HIGH,
    })

    confidence_rank = {MatchConfidence.HIGH: 2, MatchConfidence.MEDIUM: 1, MatchConfidence.LOW: 0}

    for m in mappings:
        for hid in m.hole_ids:
            g = groups[hid]
            g["annotation_ids"].append(m.annotation_id)
            g["matched_cylinder_ids"].update(m.matched_cylinder_ids)
            if confidence_rank[m.confidence] < confidence_rank[g["min_confidence"]]:
                g["min_confidence"] = m.confidence

    result: list[FeatureGroup] = []
    for hid, g in groups.items():
        hole = hole_map.get(hid)
        result.append(FeatureGroup(
            feature_id=hid,
            hole_type=hole.hole_type.value if hole else "unknown",
            annotation_ids=g["annotation_ids"],
            matched_cylinder_ids=sorted(g["matched_cylinder_ids"]),
            confidence=g["min_confidence"],
        ))

    return result


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
    1. Tolerance-aware diameter matching against all cylinders in each hole
    2. Count filtering
    3. LLM disambiguation for ambiguous cases
    4. Computed confidence scoring
    5. Feature group aggregation
    """
    holes = features.holes
    mappings: list[FeatureMapping] = []
    mapped_ann_ids: set[str] = set()
    mapped_hole_ids: set[str] = set()

    for ann in annotations.annotations:
        matches: list[_CylMatch] = []
        reasons: list[str] = []

        # Counterbore-specific matching
        if ann.hole_type == HoleType.COUNTERBORE and ann.counterbore_diameter:
            matches = _match_counterbore(ann, holes, diameter_tol)
            if matches:
                reasons.extend(m.reason for m in matches)

        # Thread-specific matching
        if not matches and ann.thread_spec:
            matches = _match_thread(ann, holes, rel_tol=0.10)
            if matches:
                reasons.extend(m.reason for m in matches)

        # Generic diameter matching (checks all cylinders inside each hole)
        if not matches:
            matches = _match_by_diameter(ann, holes, diameter_tol)
            if matches:
                reasons.extend(m.reason for m in matches)

        # Count filtering
        if matches and ann.count > 1:
            before = len(matches)
            matches = _match_by_count(ann, matches)
            if len(matches) != before:
                reasons.append(f"count filter {ann.count}X")

        # Ambiguity resolution via LLM
        if len(matches) > ann.count and use_llm:
            ids, llm_conf, reason = _llm_disambiguate(ann, matches)
            id_set = set(ids)
            matches = [m for m in matches if m.hole.hole_id in id_set]
            reasons.append(f"LLM disambiguated: {reason}")
        elif len(matches) > ann.count:
            reasons.append("multiple candidates, no LLM")

        # Collect matched cylinder IDs and best diameter delta
        matched_cyl_ids: list[str] = []
        best_delta = 0.0
        for m in matches:
            if m.cylinder_id:
                matched_cyl_ids.append(m.cylinder_id)
            best_delta = min(best_delta, m.diameter_delta) if matched_cyl_ids else m.diameter_delta

        if matches:
            best_delta = min(m.diameter_delta for m in matches)
            confidence, conf_score = _compute_confidence(
                best_delta, ann, len(reasons),
            )
        else:
            confidence = MatchConfidence.LOW
            conf_score = 0.35
            best_delta = 0.0

        hole_ids = [m.hole.hole_id for m in matches]
        mappings.append(FeatureMapping(
            annotation_id=ann.annotation_id,
            hole_ids=hole_ids,
            matched_cylinder_ids=matched_cyl_ids,
            confidence=confidence,
            confidence_score=conf_score,
            diameter_delta=round(best_delta, 6) if matches else None,
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

    feature_groups = _build_feature_groups(mappings, holes)

    logger.info(
        "Correlation complete: %d mappings, %d feature groups, "
        "%d unmapped annotations, %d unmapped holes",
        len(mappings), len(feature_groups),
        len(unmapped_annotations), len(unmapped_holes),
    )

    return LinkageResult(
        drawing_pdf=annotations.source_pdf,
        step_file=features.source_step,
        annotations=annotations,
        features_3d=features,
        mappings=mappings,
        feature_groups=feature_groups,
        unmapped_annotations=unmapped_annotations,
        unmapped_holes=unmapped_holes,
    )
