from __future__ import annotations

from pathlib import Path
from threading import Lock
import os

import tensorflow as tf
from tensorflow.keras.models import load_model

from .compat import CUSTOM_OBJECTS

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", str(REPO_ROOT)))

MODEL_PATHS = {
    "sandboil": MODEL_ROOT / "sandboil_best_model.h5",
    "seepage": MODEL_ROOT / "seepage_best_model.h5",
}

MODEL_INPUT_SHAPES = {
    "sandboil": (512, 512),
    "seepage": (256, 256),
}

_model_cache: dict[str, tf.keras.Model] = {}
_model_lock = Lock()


def get_available_models() -> list[str]:
    return [k for k, p in MODEL_PATHS.items() if p.exists()]


def get_model(model_type: str) -> tf.keras.Model:
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Unknown model_type: {model_type}")

    model_path = MODEL_PATHS[model_type]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_type in _model_cache:
        return _model_cache[model_type]

    with _model_lock:
        if model_type not in _model_cache:
            _model_cache[model_type] = load_model(
                str(model_path),
                custom_objects=CUSTOM_OBJECTS,
                compile=False,
            )
    return _model_cache[model_type]
