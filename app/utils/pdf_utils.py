"""PDF-to-image conversion utilities using PyMuPDF (fitz)."""

from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF
from PIL import Image

if TYPE_CHECKING:
    from app.models.schemas import DIExtractionResult

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300
_DIA_CHARS_PATTERN = re.compile(r"[\u2205\u00d8\uf06e]|(\d+\s*X)")


def pdf_page_to_image(
    pdf_path: str | Path, page_index: int = 0, dpi: int = DEFAULT_DPI
) -> Image.Image:
    """Render a single PDF page to a PIL Image at the given DPI."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def pdf_to_images(
    pdf_path: str | Path, dpi: int = DEFAULT_DPI
) -> list[Image.Image]:
    """Convert every page of a PDF to a list of PIL Images."""
    doc = fitz.open(str(pdf_path))
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        images = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        logger.info(
            "Converted %s -> %d images at %d DPI", Path(pdf_path).name, len(images), dpi
        )
        return images
    finally:
        doc.close()


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 data-URI string for vision models."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def pdf_page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF."""
    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def _merge_overlapping_boxes(
    boxes: list[tuple[float, float, float, float]], margin: float = 0.02,
) -> list[tuple[float, float, float, float]]:
    """Merge normalised bounding boxes that overlap or are within *margin*."""
    if not boxes:
        return []
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        new_merged: list[tuple[float, float, float, float]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            x0, y0, x1, y1 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx0, by0, bx1, by1 = merged[j]
                if (x0 - margin <= bx1 and bx0 - margin <= x1
                        and y0 - margin <= by1 and by0 - margin <= y1):
                    x0, y0 = min(x0, bx0), min(y0, by0)
                    x1, y1 = max(x1, bx1), max(y1, by1)
                    used[j] = True
                    changed = True
            new_merged.append((x0, y0, x1, y1))
            used[i] = True
        merged = new_merged
    return merged


def crop_annotation_regions(
    images: list[Image.Image],
    di_result: "DIExtractionResult",
    padding: float = 0.15,
    min_region_size: float = 0.08,
) -> list[tuple[int, Image.Image]]:
    """
    Identify clusters of diameter-related text from DI bounding boxes, crop
    padded regions from the full-page images, and return (page_num, cropped_image)
    pairs suitable for a focused vision pass.

    Parameters
    ----------
    images : list of PIL.Image
        Full-page images (one per page), already rendered at target DPI.
    di_result : DIExtractionResult
        Azure Document Intelligence extraction with bounding boxes.
    padding : float
        Fractional padding to add around each cluster (0.15 = 15%).
    min_region_size : float
        Minimum normalised side length for a region to be worth cropping.
    """
    crops: list[tuple[int, Image.Image]] = []

    for page in di_result.pages:
        page_idx = page.page_number - 1
        if page_idx >= len(images):
            continue
        img = images[page_idx]
        w, h = img.size

        dia_boxes: list[tuple[float, float, float, float]] = []
        for elem in page.elements:
            if elem.bounding_box is None:
                continue
            if _DIA_CHARS_PATTERN.search(elem.content):
                bb = elem.bounding_box
                dia_boxes.append((bb.x_min, bb.y_min, bb.x_max, bb.y_max))

        if not dia_boxes:
            continue

        clusters = _merge_overlapping_boxes(dia_boxes, margin=0.05)

        for x0, y0, x1, y1 in clusters:
            rw, rh = x1 - x0, y1 - y0
            if rw < min_region_size and rh < min_region_size:
                continue

            px0 = max(0.0, x0 - padding)
            py0 = max(0.0, y0 - padding)
            px1 = min(1.0, x1 + padding)
            py1 = min(1.0, y1 + padding)

            crop_box = (
                int(px0 * w), int(py0 * h),
                int(px1 * w), int(py1 * h),
            )
            cropped = img.crop(crop_box)

            if cropped.size[0] < 50 or cropped.size[1] < 50:
                continue

            crops.append((page.page_number, cropped))
            logger.info(
                "Cropped annotation region on page %d: (%.2f,%.2f)-(%.2f,%.2f)",
                page.page_number, px0, py0, px1, py1,
            )

    logger.info("Generated %d cropped annotation regions", len(crops))
    return crops
