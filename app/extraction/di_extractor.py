"""Azure Document Intelligence prebuilt-layout extraction for engineering drawings."""

from __future__ import annotations

import logging
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

from app.config import AzureDIConfig
from app.models.schemas import (
    BoundingBox,
    DIExtractionResult,
    DIPageResult,
    DITextElement,
)

logger = logging.getLogger(__name__)


def _build_client() -> DocumentIntelligenceClient:
    return DocumentIntelligenceClient(
        endpoint=AzureDIConfig.endpoint,
        credential=AzureKeyCredential(AzureDIConfig.key),
    )


def _polygon_to_bbox(polygon: list[float], page_width: float, page_height: float, page_num: int) -> BoundingBox:
    """Convert Azure DI polygon (list of x,y pairs) to a normalised BoundingBox."""
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    return BoundingBox(
        page=page_num,
        x_min=min(xs) / page_width,
        y_min=min(ys) / page_height,
        x_max=max(xs) / page_width,
        y_max=max(ys) / page_height,
    )


def extract_layout(pdf_path: str | Path) -> DIExtractionResult:
    """
    Run Azure DI prebuilt-layout on *pdf_path* and return structured text
    with bounding boxes for every word / line.
    """
    pdf_path = Path(pdf_path)
    client = _build_client()

    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            content_type="application/octet-stream",
        )

    result = poller.result()

    pages: list[DIPageResult] = []

    for page in result.pages:
        page_num = page.page_number
        pw = page.width or 1.0
        ph = page.height or 1.0

        elements: list[DITextElement] = []

        for word in (page.words or []):
            bbox = None
            if word.polygon:
                bbox = _polygon_to_bbox(word.polygon, pw, ph, page_num)
            elements.append(
                DITextElement(
                    content=word.content,
                    confidence=word.confidence or 1.0,
                    bounding_box=bbox,
                )
            )

        for line in (page.lines or []):
            bbox = None
            if line.polygon:
                bbox = _polygon_to_bbox(line.polygon, pw, ph, page_num)
            elements.append(
                DITextElement(
                    content=line.content,
                    confidence=1.0,
                    bounding_box=bbox,
                )
            )

        pages.append(DIPageResult(
            page_number=page_num,
            width_px=pw,
            height_px=ph,
            elements=elements,
        ))

    logger.info("DI extraction complete for %s – %d pages", pdf_path.name, len(pages))
    return DIExtractionResult(source_pdf=str(pdf_path), pages=pages)
