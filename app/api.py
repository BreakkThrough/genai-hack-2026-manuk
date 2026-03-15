"""
FastAPI backend for the Drawing-to-3D Hole Feature Linker.

Run with:  uvicorn app.api:app --reload
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.config import DATASET_DIR
from app.correlation.matcher import correlate
from app.extraction.di_extractor import extract_layout
from app.extraction.step_parser import parse_step
from app.extraction.vision_enricher import enrich_drawing
from app.models.schemas import (
    DIExtractionResult,
    DrawingAnnotations,
    LinkageResult,
    StepFeatures,
)
from app.utils.pdf_utils import pdf_to_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Drawing-3D Hole Linker API", version="1.0.0")

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Session storage (in-memory for Phase 1)
# ---------------------------------------------------------------------------

class SessionData(BaseModel):
    """Data stored for each session."""
    session_id: str
    pdf_path: Path | None = None
    step_path: Path | None = None
    unit: str = "inch"
    di_result: DIExtractionResult | None = None
    annotations: DrawingAnnotations | None = None
    step_features: StepFeatures | None = None
    linkage_result: LinkageResult | None = None
    temp_dir: Path | None = None


_sessions: Dict[str, SessionData] = {}


def _get_session(session_id: str) -> SessionData:
    """Retrieve session or raise 404."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    session_id: str
    pdf_filename: str | None = None
    step_filename: str | None = None
    unit: str = "inch"
    step_features: StepFeatures | None = None


class Layer1Response(BaseModel):
    status: str
    page_count: int = 0
    element_count: int = 0
    sample_elements: list[dict] = []
    error: str | None = None


class Layer2Response(BaseModel):
    status: str
    annotations: DrawingAnnotations | None = None
    annotation_count: int = 0
    error: str | None = None


class Layer3Response(BaseModel):
    status: str
    step_features: StepFeatures | None = None
    hole_count: int = 0
    error: str | None = None


class Layer4Response(BaseModel):
    status: str
    linkage_result: LinkageResult | None = None
    mapping_count: int = 0
    error: str | None = None


class UpdateMappingsRequest(BaseModel):
    """User-edited mappings from the frontend."""
    mappings: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Drawing-3D Hole Linker API", "version": "1.0.0"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(
    pdf_file: UploadFile | None = File(None),
    step_file: UploadFile | None = File(None),
    unit: str = Form("inch"),
):
    """
    Accept PDF + STEP file uploads, store in a temp session directory.
    Also parse STEP immediately (Layer 3) since it has no dependencies.
    """
    session_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp(prefix=f"session_{session_id[:8]}_"))
    
    pdf_path: Path | None = None
    step_path: Path | None = None
    step_features: StepFeatures | None = None
    
    if pdf_file:
        pdf_path = temp_dir / pdf_file.filename
        pdf_path.write_bytes(await pdf_file.read())
        logger.info(f"Uploaded PDF: {pdf_file.filename}")
    
    if step_file:
        step_path = temp_dir / step_file.filename
        step_path.write_bytes(await step_file.read())
        logger.info(f"Uploaded STEP: {step_file.filename}")
        
        # Parse STEP immediately (Layer 3)
        try:
            step_features = parse_step(step_path)
            logger.info(f"Parsed STEP: {len(step_features.holes)} holes found")
        except Exception as exc:
            logger.error(f"STEP parsing failed: {exc}")
    
    session = SessionData(
        session_id=session_id,
        pdf_path=pdf_path,
        step_path=step_path,
        unit=unit,
        step_features=step_features,
        temp_dir=temp_dir,
    )
    _sessions[session_id] = session
    
    return UploadResponse(
        session_id=session_id,
        pdf_filename=pdf_file.filename if pdf_file else None,
        step_filename=step_file.filename if step_file else None,
        unit=unit,
        step_features=step_features,
    )


@app.post("/api/pipeline/layer1", response_model=Layer1Response)
def run_layer1(session_id: str = Form(...)):
    """
    Run Azure DI extract_layout() on the uploaded PDF.
    Returns DI result summary (page count, element count, sample OCR text).
    """
    session = _get_session(session_id)
    
    if not session.pdf_path:
        raise HTTPException(status_code=400, detail="No PDF uploaded for this session")
    
    try:
        di_result = extract_layout(session.pdf_path)
        session.di_result = di_result
        
        total_elements = sum(len(p.elements) for p in di_result.pages)
        sample_elements = []
        for page in di_result.pages:
            for elem in page.elements[:20]:
                sample_elements.append({
                    "text": elem.content,
                    "confidence": elem.confidence,
                    "bbox": elem.bounding_box.model_dump() if elem.bounding_box else None,
                })
                if len(sample_elements) >= 20:
                    break
            if len(sample_elements) >= 20:
                break
        
        logger.info(f"Layer 1 complete: {len(di_result.pages)} pages, {total_elements} elements")
        
        return Layer1Response(
            status="success",
            page_count=len(di_result.pages),
            element_count=total_elements,
            sample_elements=sample_elements,
        )
    
    except Exception as exc:
        logger.error(f"Layer 1 failed: {exc}")
        return Layer1Response(
            status="error",
            error=str(exc),
        )


@app.post("/api/pipeline/layer2", response_model=Layer2Response)
def run_layer2(session_id: str = Form(...)):
    """
    Run enrich_drawing() with GPT-4o multi-pass vision enrichment.
    Returns DrawingAnnotations JSON including all hole annotations.
    """
    session = _get_session(session_id)
    
    if not session.pdf_path:
        raise HTTPException(status_code=400, detail="No PDF uploaded for this session")
    
    try:
        annotations = enrich_drawing(
            session.pdf_path,
            di_result=session.di_result,
            unit=session.unit,
            apply_filters=True,
        )
        session.annotations = annotations
        
        logger.info(f"Layer 2 complete: {len(annotations.annotations)} annotations extracted")
        
        return Layer2Response(
            status="success",
            annotations=annotations,
            annotation_count=len(annotations.annotations),
        )
    
    except Exception as exc:
        logger.error(f"Layer 2 failed: {exc}")
        return Layer2Response(
            status="error",
            error=str(exc),
        )


@app.post("/api/pipeline/layer3", response_model=Layer3Response)
def run_layer3(session_id: str = Form(...)):
    """
    Run parse_step() on the uploaded STEP file.
    Returns StepFeatures JSON (hole list with centres, diameters, axes).
    
    Note: This is typically already done during upload, but can be re-run if needed.
    """
    session = _get_session(session_id)
    
    if not session.step_path:
        raise HTTPException(status_code=400, detail="No STEP file uploaded for this session")
    
    try:
        if session.step_features:
            # Already parsed during upload
            return Layer3Response(
                status="success",
                step_features=session.step_features,
                hole_count=len(session.step_features.holes),
            )
        
        step_features = parse_step(session.step_path)
        session.step_features = step_features
        
        logger.info(f"Layer 3 complete: {len(step_features.holes)} holes found")
        
        return Layer3Response(
            status="success",
            step_features=step_features,
            hole_count=len(step_features.holes),
        )
    
    except Exception as exc:
        logger.error(f"Layer 3 failed: {exc}")
        return Layer3Response(
            status="error",
            error=str(exc),
        )


@app.post("/api/pipeline/layer4", response_model=Layer4Response)
def run_layer4(session_id: str = Form(...)):
    """
    Run correlate() to match annotations with 3D features.
    Returns LinkageResult JSON with mappings.
    """
    session = _get_session(session_id)
    
    if not session.annotations:
        raise HTTPException(status_code=400, detail="Layer 2 (annotations) not completed")
    
    if not session.step_features:
        raise HTTPException(status_code=400, detail="Layer 3 (STEP features) not completed")
    
    try:
        linkage = correlate(session.annotations, session.step_features)
        session.linkage_result = linkage
        
        logger.info(f"Layer 4 complete: {len(linkage.mappings)} mappings created")
        
        return Layer4Response(
            status="success",
            linkage_result=linkage,
            mapping_count=len(linkage.mappings),
        )
    
    except Exception as exc:
        logger.error(f"Layer 4 failed: {exc}")
        return Layer4Response(
            status="error",
            error=str(exc),
        )


@app.put("/api/mappings/update")
def update_mappings(session_id: str, request: UpdateMappingsRequest):
    """
    Accept user-edited mappings (annotation_id -> hole_ids) and update the LinkageResult.
    """
    session = _get_session(session_id)
    
    if not session.linkage_result:
        raise HTTPException(status_code=400, detail="Layer 4 (correlation) not completed")
    
    try:
        # Update mappings in the linkage result
        for updated_mapping in request.mappings:
            annotation_id = updated_mapping.get("annotation_id")
            new_hole_ids = updated_mapping.get("hole_ids", [])
            
            # Find the mapping to update
            for mapping in session.linkage_result.mappings:
                if mapping.annotation_id == annotation_id:
                    mapping.hole_ids = new_hole_ids
                    mapping.match_reasons.append("User-edited mapping")
                    break
        
        logger.info(f"Updated {len(request.mappings)} mappings")
        
        return {"status": "success", "updated_count": len(request.mappings)}
    
    except Exception as exc:
        logger.error(f"Mapping update failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/drawing/page/{page_num}")
def get_drawing_page(session_id: str, page_num: int):
    """
    Return a page image as PNG (rendered at 300 DPI).
    """
    session = _get_session(session_id)
    
    if not session.pdf_path:
        raise HTTPException(status_code=400, detail="No PDF uploaded for this session")
    
    try:
        # Render the PDF page to image
        images = pdf_to_images(session.pdf_path, dpi=300)
        
        if page_num < 1 or page_num > len(images):
            raise HTTPException(
                status_code=404,
                detail=f"Page {page_num} not found (PDF has {len(images)} pages)"
            )
        
        # Save image to temp file and return
        image = images[page_num - 1]
        temp_image_path = session.temp_dir / f"page_{page_num}.png"
        image.save(temp_image_path, format="PNG")
        
        return FileResponse(
            temp_image_path,
            media_type="image/png",
            filename=f"page_{page_num}.png"
        )
    
    except Exception as exc:
        logger.error(f"Failed to render page {page_num}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/export")
def export_json(session_id: str):
    """
    Return the final LinkageResult JSON for download.
    """
    session = _get_session(session_id)
    
    if not session.linkage_result:
        raise HTTPException(status_code=400, detail="Layer 4 (correlation) not completed")
    
    return JSONResponse(
        content=session.linkage_result.model_dump(mode="json"),
        headers={
            "Content-Disposition": "attachment; filename=linkage_result.json"
        }
    )


@app.get("/api/session/{session_id}")
def get_session_status(session_id: str):
    """
    Get the current status of a session (which layers are completed).
    """
    session = _get_session(session_id)
    
    return {
        "session_id": session.session_id,
        "pdf_uploaded": session.pdf_path is not None,
        "step_uploaded": session.step_path is not None,
        "unit": session.unit,
        "layer1_complete": session.di_result is not None,
        "layer2_complete": session.annotations is not None,
        "layer3_complete": session.step_features is not None,
        "layer4_complete": session.linkage_result is not None,
    }


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """
    Delete a session and clean up temporary files.
    """
    session = _get_session(session_id)
    
    # Clean up temp directory
    if session.temp_dir and session.temp_dir.exists():
        import shutil
        shutil.rmtree(session.temp_dir, ignore_errors=True)
    
    del _sessions[session_id]
    
    return {"status": "success", "message": f"Session {session_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
