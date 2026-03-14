from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


class InferResponse(BaseModel):
    image_base64: str = Field(description="Base64-encoded PNG image")
    preprocessed_base64: str
    model_stats: dict[str, dict[str, Any]]
    applied_settings: dict[str, Any]


class HealthResponse(BaseModel):
    status: str


class ModelsResponse(BaseModel):
    available_models: list[str]


class AnnotationPoint(BaseModel):
    x: int
    y: int


class AnnotationCreateRequest(BaseModel):
    target_model: str
    image_name: str | None = None
    points: list[AnnotationPoint]
    notes: str | None = None
    metadata: dict[str, Any] | None = None


class AnnotationRecord(BaseModel):
    id: str
    created_at: str
    target_model: str
    image_name: str | None = None
    points: list[AnnotationPoint]
    notes: str | None = None
    metadata: dict[str, Any] | None = None


class AnnotationListResponse(BaseModel):
    items: list[AnnotationRecord]
