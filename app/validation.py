"""
Validation module: compare pipeline output against FSI ground-truth data.

The FSI (Feature and Specification Index) PDFs list every feature with its
specifications. This module parses the FSI to extract ground-truth hole
features, then compares them against the pipeline's extraction results.

Usage (CLI):
    python -m app.validation --ftc 06
    python -m app.validation --ftc 10 --pipeline-json linkage_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.config import DATASET_DIR

logger = logging.getLogger(__name__)

# Unicode Private-Use-Area chars used by NIST PDFs for GD&T symbols
_DIA_CHARS = "\u2205\uf06e\u00d8"  # ∅, PUA diameter, Ø

# ---------------------------------------------------------------------------
# Ground-truth hole feature
# ---------------------------------------------------------------------------


@dataclass
class FSIHoleFeature:
    feature_id: str
    description: str
    diameter: float | None = None
    diameter_text: str = ""
    count: int = 1
    hole_type: str = "simple"
    thread_spec: str | None = None
    counterbore_diameter: float | None = None
    depth: float | None = None
    position_tolerance: float | None = None
    datum_refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FSI text parser
# ---------------------------------------------------------------------------

_COUNTED_DIA_RE = re.compile(
    rf"(\d+)\s*X\s*[{_DIA_CHARS}]?\s*([\d.]+)", re.IGNORECASE
)
_SINGLE_DIA_RE = re.compile(rf"[{_DIA_CHARS}]([\d.]+)")
_SPHERE_DIA_RE = re.compile(rf"S\s*[{_DIA_CHARS}]([\d.]+)")
_THREAD_RE = re.compile(
    r"(M\d+(?:\.\d+)?)\s*[xX\u00d7]\s*([\d.]+)\s*[-\u2013]?\s*(\d+[ghGH])?"
)
_FIT_RE = re.compile(r"([\d.]+)\s+([A-Z]\d+)")

_NON_HOLE_KEYWORDS = [
    "datum feature a", "datum feature b", "datum feature c", "datum feature d",
    "datum feature e", "datum target", "fillet", "round", "taper",
    "rib surface", "conic", "cone support", "width feature",
    "general profile", "general note", "mcs for", "mcs2", "mcs3",
    "represented line", "crosshatch",
]

_HOLE_KEYWORDS = [
    "hole", "counterbor", "countersink", "pattern of", "threaded",
    "cross-drilled", "cylindrical hole", "spherical diameter",
]


def _is_hole_feature(desc: str) -> bool:
    lower = desc.lower()
    for kw in _NON_HOLE_KEYWORDS:
        if kw in lower:
            return False
    for kw in _HOLE_KEYWORDS:
        if kw in lower:
            return True
    if "bore" in lower or "drill" in lower:
        return True
    return True


def parse_fsi_text(text: str) -> list[FSIHoleFeature]:
    """Extract hole features from FSI plain-text content."""
    features: list[FSIHoleFeature] = []
    lines = text.splitlines()

    current_fid = ""
    current_desc = ""
    seen_fids: set[str] = set()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        i += 1
        if not stripped:
            continue

        fid_match = re.match(r"^(F\d+(?:-F\d+)?)\b", stripped)
        if fid_match:
            current_fid = fid_match.group(1)
            remainder = stripped[fid_match.end():].strip()
            if remainder:
                current_desc = remainder
            elif i < len(lines):
                current_desc = lines[i].strip()

        if current_fid in seen_fids:
            continue

        if re.match(
            r"^(Position|Flatness|Perpendicularity|Profile|Parallelism"
            r"|Cylindricity|Circularity|Runout|Symmetry)",
            stripped,
        ):
            continue

        if re.match(r"^Datum (Feature|Target) Symbol", stripped):
            continue

        # Threaded holes
        thread_match = _THREAD_RE.search(stripped)
        if thread_match:
            designation = thread_match.group(1)
            pitch = float(thread_match.group(2))
            tol_class = thread_match.group(3) or ""
            count_match = re.match(r"(\d+)\s*X", stripped)
            count = int(count_match.group(1)) if count_match else 1
            dia_m = re.search(r"M(\d+(?:\.\d+)?)", designation)
            dia_val = float(dia_m.group(1)) if dia_m else None

            features.append(FSIHoleFeature(
                feature_id=current_fid,
                description=current_desc,
                diameter=dia_val,
                diameter_text=stripped,
                count=count,
                hole_type="threaded",
                thread_spec=f"{designation} x {pitch} - {tol_class}".strip(" -"),
            ))
            seen_fids.add(current_fid)
            continue

        # Spherical diameter
        sphere_match = _SPHERE_DIA_RE.search(stripped)
        if sphere_match and current_fid not in seen_fids:
            dia = float(sphere_match.group(1))
            if dia > 0.001:
                count_match = re.match(r"(\d+)\s*X", stripped)
                count = int(count_match.group(1)) if count_match else 1
                features.append(FSIHoleFeature(
                    feature_id=current_fid,
                    description=current_desc,
                    diameter=dia,
                    diameter_text=stripped,
                    count=count,
                    hole_type="spherical",
                ))
                seen_fids.add(current_fid)
                continue

        # Fit designation (e.g. "3.5 G6")
        fit_match = _FIT_RE.search(stripped)
        if fit_match and current_fid and current_fid not in seen_fids:
            dia_val = float(fit_match.group(1))
            if dia_val > 0.01:
                features.append(FSIHoleFeature(
                    feature_id=current_fid,
                    description=current_desc,
                    diameter=dia_val,
                    diameter_text=stripped,
                    count=1,
                    hole_type="simple",
                ))
                seen_fids.add(current_fid)
                continue

        # Counted diameter: "4X .281 ±.008"
        count_dia_match = _COUNTED_DIA_RE.search(stripped)
        if count_dia_match and current_fid not in seen_fids:
            count = int(count_dia_match.group(1))
            dia = float(count_dia_match.group(2))
            if dia < 0.001:
                continue

            desc_lower = current_desc.lower()
            if "counterbor" in desc_lower:
                hole_type = "counterbore"
            elif "countersink" in desc_lower:
                hole_type = "countersink"
            elif "spherical" in desc_lower:
                hole_type = "spherical"
            else:
                hole_type = "simple"

            features.append(FSIHoleFeature(
                feature_id=current_fid,
                description=current_desc,
                diameter=dia,
                diameter_text=stripped,
                count=count,
                hole_type=hole_type,
            ))
            seen_fids.add(current_fid)
            continue

        # Single diameter: "<dia>.562 ±.008"
        single_match = _SINGLE_DIA_RE.search(stripped)
        if single_match and current_fid and current_fid not in seen_fids:
            dia = float(single_match.group(1))
            if dia > 0.001:
                desc_lower = current_desc.lower()
                hole_type = "counterbore" if "counterbor" in desc_lower else "simple"
                features.append(FSIHoleFeature(
                    feature_id=current_fid,
                    description=current_desc,
                    diameter=dia,
                    diameter_text=stripped,
                    count=1,
                    hole_type=hole_type,
                ))
                seen_fids.add(current_fid)
                continue

    return features


# ---------------------------------------------------------------------------
# Load FSI ground truth
# ---------------------------------------------------------------------------

def _find_fsi(ftc_num: str) -> Path | None:
    pattern = f"nist_ftc_{ftc_num}*_fsi.pdf"
    matches = sorted(DATASET_DIR.glob(pattern))
    return matches[0] if matches else None


def load_ground_truth(ftc_num: str) -> list[FSIHoleFeature]:
    """Load and parse hole features from the FSI PDF for a given FTC case."""
    fsi_path = _find_fsi(ftc_num)
    if not fsi_path:
        logger.warning("No FSI file found for FTC %s", ftc_num)
        return []

    import fitz

    doc = fitz.open(str(fsi_path))
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()

    features = parse_fsi_text(full_text)
    logger.info("Parsed %d hole features from FSI %s", len(features), fsi_path.name)
    return features


# ---------------------------------------------------------------------------
# Comparison / metrics
# ---------------------------------------------------------------------------


@dataclass
class ValidationMetrics:
    total_gt: int = 0
    total_extracted: int = 0
    matched: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    details: list[dict] = field(default_factory=list)


def compare_annotations(
    ground_truth: list[FSIHoleFeature],
    pipeline_annotations: list[dict],
    diameter_tol: float = 0.10,
) -> ValidationMetrics:
    """
    Compare ground-truth FSI hole features with pipeline-extracted annotations.

    Uses count-aware bucket matching:
    1. Group GT features by rounded diameter
    2. Group extracted annotations by rounded diameter
    3. Match within each bucket up to min(gt_count, ext_count)
    4. Fuzzy matching pass for remaining unmatched features
    """
    details: list[dict] = []

    def _round_dia(d: float, precision: int = 3) -> str:
        return str(round(d, precision))

    # Group GT by diameter
    gt_by_dia: dict[str, list[tuple[int, FSIHoleFeature]]] = {}
    for gi, gt in enumerate(ground_truth):
        if gt.diameter is None:
            continue
        key = _round_dia(gt.diameter)
        gt_by_dia.setdefault(key, []).append((gi, gt))

    # Group extracted by diameter
    ext_by_dia: dict[str, list[tuple[int, dict]]] = {}
    for ei, ext in enumerate(pipeline_annotations):
        d = ext.get("diameter")
        if d is None:
            continue
        key = _round_dia(d)
        ext_by_dia.setdefault(key, []).append((ei, ext))

    gt_matched: set[int] = set()
    ext_matched: set[int] = set()

    # Count-aware bucket matching
    for dia_key, gt_list in gt_by_dia.items():
        ext_list = ext_by_dia.get(dia_key, [])
        if not ext_list:
            continue

        gt_idx = 0
        for ei, ext in ext_list:
            if gt_idx >= len(gt_list):
                break
            ext_count = ext.get("count", 1) or 1
            covered = 0
            while gt_idx < len(gt_list) and covered < ext_count:
                gi, gt = gt_list[gt_idx]
                gt_matched.add(gi)
                if covered == 0:
                    ext_matched.add(ei)
                details.append({
                    "gt_feature": gt.feature_id,
                    "gt_diameter": gt.diameter,
                    "ext_annotation": ext.get("annotation_id"),
                    "ext_diameter": ext.get("diameter"),
                    "status": "matched",
                })
                gt_idx += 1
                covered += 1

    # Fuzzy matching for remaining GT
    for gi, gt in enumerate(ground_truth):
        if gi in gt_matched or gt.diameter is None:
            continue
        best_match = None
        best_delta = float("inf")
        for ei, ext in enumerate(pipeline_annotations):
            if ei in ext_matched:
                continue
            ext_dia = ext.get("diameter")
            if ext_dia is None:
                continue
            delta = abs(gt.diameter - ext_dia)
            avg = (abs(gt.diameter) + abs(ext_dia)) / 2.0
            rel = delta / max(avg, 0.001)
            if rel <= diameter_tol and delta < best_delta:
                best_delta = delta
                best_match = ei

        if best_match is not None:
            gt_matched.add(gi)
            ext_matched.add(best_match)
            details.append({
                "gt_feature": gt.feature_id,
                "gt_diameter": gt.diameter,
                "ext_annotation": pipeline_annotations[best_match].get("annotation_id"),
                "ext_diameter": pipeline_annotations[best_match].get("diameter"),
                "status": "matched",
            })
        else:
            details.append({
                "gt_feature": gt.feature_id,
                "gt_diameter": gt.diameter,
                "ext_annotation": None,
                "ext_diameter": None,
                "status": "missed",
            })

    # Record extras
    for ei, ext in enumerate(pipeline_annotations):
        if ei not in ext_matched:
            details.append({
                "gt_feature": None,
                "gt_diameter": None,
                "ext_annotation": ext.get("annotation_id"),
                "ext_diameter": ext.get("diameter"),
                "status": "extra",
            })

    total_gt = sum(1 for g in ground_truth if g.diameter is not None)
    total_ext = len(pipeline_annotations)
    matched = len(gt_matched)

    precision = min(matched / total_ext, 1.0) if total_ext else 0.0
    recall = min(matched / total_gt, 1.0) if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return ValidationMetrics(
        total_gt=total_gt,
        total_extracted=total_ext,
        matched=matched,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        details=details,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Validate pipeline against FSI ground truth"
    )
    parser.add_argument("--ftc", required=True, help="FTC number, e.g. '06' or '10'")
    parser.add_argument(
        "--pipeline-json", help="Path to pipeline LinkageResult JSON (optional)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    gt = load_ground_truth(args.ftc)
    print(f"\n{'=' * 60}")
    print(f"  FSI Ground Truth for FTC-{args.ftc}")
    print(f"{'=' * 60}")
    for f in gt:
        parts = [f.feature_id, f.hole_type]
        if f.diameter:
            parts.append(f"D{f.diameter}")
        if f.count > 1:
            parts.append(f"{f.count}X")
        if f.thread_spec:
            parts.append(f"thread={f.thread_spec}")
        print(f"  {' | '.join(parts)}")

    print(f"\n  Total hole features in FSI: {len(gt)}")

    if args.pipeline_json:
        with open(args.pipeline_json, encoding="utf-8") as fp:
            linkage = json.load(fp)
        anns = linkage.get("annotations", {}).get("annotations", [])
        metrics = compare_annotations(gt, anns)
        print(f"\n{'=' * 60}")
        print("  Validation Metrics")
        print(f"{'=' * 60}")
        print(f"  Ground-truth holes:  {metrics.total_gt}")
        print(f"  Extracted holes:     {metrics.total_extracted}")
        print(f"  Matched:             {metrics.matched}")
        print(f"  Precision:           {metrics.precision:.2%}")
        print(f"  Recall:              {metrics.recall:.2%}")
        print(f"  F1:                  {metrics.f1:.2%}")
        print()
        for d in metrics.details:
            gf = d.get("gt_feature") or "-"
            gd = d.get("gt_diameter") or "-"
            ea = d.get("ext_annotation") or "-"
            ed = d.get("ext_diameter") or "-"
            print(
                f"  {d['status']:8s}  GT={str(gf):12s} dia={str(gd):>8}"
                f"  EXT={str(ea):8s} dia={str(ed):>8}"
            )
    else:
        print(
            "\nNo --pipeline-json provided. Run the pipeline first, then:\n"
            "  python -m app.validation --ftc 06 --pipeline-json linkage_result.json"
        )


if __name__ == "__main__":
    main()
