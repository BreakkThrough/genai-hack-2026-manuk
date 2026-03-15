"""
Simple CLI entry point: PDF + STEP → structured JSON output.

Usage:
    python run.py --pdf <drawing.pdf> --step <model.stp> [--unit inch|mm] [--model gpt-4o] [--output result.json]

Example with NIST dataset:
    python run.py --pdf dataset/nist_ftc_06_asme1_rd.pdf --step dataset/nist_ftc_06_asme1_rd.stp --output linkage_ftc06.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drawing-to-3D Hole Feature Linker — end-to-end pipeline",
    )
    parser.add_argument("--pdf", required=True, help="Path to the engineering drawing PDF")
    parser.add_argument("--step", required=True, help="Path to the STEP (.stp/.step) file")
    parser.add_argument("--unit", default="inch", choices=["inch", "mm"], help="Drawing units (default: inch)")
    parser.add_argument("--model", default="gpt-4o", help="Azure OpenAI deployment name (default: gpt-4o)")
    parser.add_argument("--output", "-o", default="linkage_result.json", help="Output JSON path (default: linkage_result.json)")
    parser.add_argument("--no-di", action="store_true", help="Skip Azure Document Intelligence (vision-only mode)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM disambiguation in correlation")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    step_path = Path(args.step)

    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return 1
    if not step_path.exists():
        logger.error("STEP file not found: %s", step_path)
        return 1

    from app.extraction.di_extractor import extract_layout
    from app.extraction.step_parser import parse_step
    from app.extraction.vision_enricher import enrich_drawing
    from app.correlation.matcher import correlate

    api_ver = "2025-04-01-preview" if args.model != "gpt-4o" else None

    # Layer 1: Azure Document Intelligence
    di_result = None
    if not args.no_di:
        logger.info("Layer 1/4: Azure DI layout extraction...")
        try:
            di_result = extract_layout(pdf_path)
            total_elems = sum(len(p.elements) for p in di_result.pages)
            logger.info("  -> %d pages, %d text elements", len(di_result.pages), total_elems)
        except Exception as exc:
            logger.warning("Azure DI failed (%s), continuing with vision-only mode", exc)
    else:
        logger.info("Layer 1/4: Skipped (--no-di)")

    # Layer 2: Vision enrichment
    logger.info("Layer 2/4: %s vision enrichment...", args.model)
    annotations = enrich_drawing(
        pdf_path,
        di_result=di_result,
        unit=args.unit,
        apply_filters=True,
        model_deployment=args.model,
        api_version=api_ver,
    )
    logger.info("  -> %d hole annotations extracted", len(annotations.annotations))

    # Layer 3: STEP parsing
    logger.info("Layer 3/4: Parsing STEP file...")
    step_features = parse_step(step_path)
    logger.info(
        "  -> %d cylindrical faces, %d grouped holes",
        step_features.total_cylindrical_faces, len(step_features.holes),
    )

    # Layer 4: Correlation
    logger.info("Layer 4/4: Correlating annotations with 3D features...")
    linkage = correlate(annotations, step_features, use_llm=not args.no_llm)

    high = sum(1 for m in linkage.mappings if m.confidence.value == "high")
    med = sum(1 for m in linkage.mappings if m.confidence.value == "medium")
    low = sum(1 for m in linkage.mappings if m.confidence.value == "low")
    logger.info(
        "  -> %d mappings (%d high, %d medium, %d low confidence)",
        len(linkage.mappings), high, med, low,
    )
    if linkage.unmapped_annotations:
        logger.warning("  -> %d unmapped annotations", len(linkage.unmapped_annotations))
    if linkage.unmapped_holes:
        logger.info("  -> %d unmapped 3D holes", len(linkage.unmapped_holes))

    # Write output
    out_path = Path(args.output)
    out_path.write_text(linkage.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Output written to %s", out_path)

    # Summary table
    print(f"\n{'='*60}")
    print("  Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Drawing:        {pdf_path.name}")
    print(f"  STEP model:     {step_path.name}")
    print(f"  Annotations:    {len(annotations.annotations)}")
    print(f"  3D holes:       {len(step_features.holes)}")
    print(f"  Mappings:       {len(linkage.mappings)} ({high} high, {med} med, {low} low)")
    print(f"  Output:         {out_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
