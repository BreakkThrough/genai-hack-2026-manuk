"""PDF-to-image conversion utilities using PyMuPDF (fitz)."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300


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
