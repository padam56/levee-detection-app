from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .annotation_store import list_annotations, save_annotation
from .inference import infer_image
from .model_store import get_available_models
from .schemas import (
    AnnotationCreateRequest,
    AnnotationListResponse,
    AnnotationRecord,
    HealthResponse,
    InferResponse,
    ModelsResponse,
)

app = FastAPI(title="Levee Detection API", version="1.0.0")

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(available_models=get_available_models())


@app.get("/annotations", response_model=AnnotationListResponse)
def get_annotations(limit: int = 100) -> AnnotationListResponse:
    records = list_annotations(limit=limit)
    return AnnotationListResponse(items=[AnnotationRecord(**r) for r in records])


@app.post("/annotations", response_model=AnnotationRecord)
def create_annotation(payload: AnnotationCreateRequest) -> AnnotationRecord:
    if len(payload.points) < 3:
        raise HTTPException(status_code=400, detail="At least 3 points are required")

    record = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_model": payload.target_model,
        "image_name": payload.image_name,
        "points": [p.model_dump() for p in payload.points],
        "notes": payload.notes,
        "metadata": payload.metadata or {},
    }
    saved = save_annotation(record)
    return AnnotationRecord(**saved)


@app.post("/infer/image", response_model=InferResponse)
async def infer_image_endpoint(
    image: UploadFile = File(...),
    model_type: str = Form("sandboil"),
    threshold: float = Form(0.5),
    visualization: str = Form("overlay"),
    overlay_intensity: float = Form(0.45),
    selected_models: str = Form("[]"),
    thresholds: str = Form("{}"),
    threshold_types: str = Form("{}"),
    preprocessing_settings: str = Form("{}"),
    distance_threshold: int = Form(20),
) -> InferResponse:
    try:
        image_bytes = await image.read()

        selected = json.loads(selected_models)
        if not selected:
            selected = [model_type]

        thresholds_map = json.loads(thresholds)
        threshold_types_map = json.loads(threshold_types)
        preprocess = json.loads(preprocessing_settings)

        if model_type not in thresholds_map:
            thresholds_map[model_type] = threshold
        if model_type not in threshold_types_map:
            threshold_types_map[model_type] = "Manual"

        result = infer_image(
            image_bytes=image_bytes,
            selected_models=selected,
            thresholds=thresholds_map,
            threshold_types=threshold_types_map,
            visualization=visualization,
            overlay_intensity=overlay_intensity,
            distance_threshold=distance_threshold,
            preprocessing_settings=preprocess,
        )
        return InferResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if FRONTEND_DIST.exists():
    @app.get("/")
    def serve_frontend() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")


    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        requested = FRONTEND_DIST / full_path
        if requested.exists() and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
