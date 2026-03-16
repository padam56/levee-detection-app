from __future__ import annotations

import base64
import os
import tempfile
from typing import Any

import cv2
import numpy as np

from .model_store import MODEL_INPUT_SHAPES, get_model

DETECTION_COLORS = {
    "sandboil": (0, 200, 70),
    "seepage": (255, 105, 180),
}

DEFAULT_PREPROCESSING = {
    "resolution_factor": 1.0,
    "brightness_factor": 0,
    "contrast_factor": 0,
    "blur_amount": 1,
    "edge_detection": False,
    "flip_horizontal": False,
    "flip_vertical": False,
    "rotate_angle": 0,
}


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def encode_png_base64(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode output image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def sanitize_preprocessing(settings: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(DEFAULT_PREPROCESSING)
    if settings:
        data.update(settings)

    data["resolution_factor"] = float(np.clip(float(data["resolution_factor"]), 0.2, 2.0))
    data["brightness_factor"] = int(np.clip(int(data["brightness_factor"]), -100, 100))
    data["contrast_factor"] = int(np.clip(int(data["contrast_factor"]), -100, 100))
    blur_val = int(data["blur_amount"])
    if blur_val < 1:
        blur_val = 1
    data["blur_amount"] = blur_val if blur_val % 2 == 1 else blur_val + 1
    data["edge_detection"] = bool(data["edge_detection"])
    data["flip_horizontal"] = bool(data["flip_horizontal"])
    data["flip_vertical"] = bool(data["flip_vertical"])
    data["rotate_angle"] = int(np.clip(int(data["rotate_angle"]), -180, 180))
    return data


def apply_base_transforms(image: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    transformed = image.copy()

    resolution_factor = settings["resolution_factor"]
    if resolution_factor != 1.0:
        new_w = max(1, int(transformed.shape[1] * resolution_factor))
        new_h = max(1, int(transformed.shape[0] * resolution_factor))
        transformed = cv2.resize(transformed, (new_w, new_h), interpolation=cv2.INTER_AREA)

    brightness_factor = settings["brightness_factor"]
    contrast_factor = settings["contrast_factor"]
    if brightness_factor != 0 or contrast_factor != 0:
        transformed = cv2.convertScaleAbs(
            transformed,
            alpha=1 + (contrast_factor / 100.0),
            beta=brightness_factor,
        )

    blur_amount = settings["blur_amount"]
    if blur_amount > 1:
        transformed = cv2.GaussianBlur(transformed, (blur_amount, blur_amount), 0)

    if settings["edge_detection"]:
        transformed = cv2.Canny(transformed, 100, 200)
        transformed = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)

    if settings["flip_horizontal"]:
        transformed = cv2.flip(transformed, 1)

    if settings["flip_vertical"]:
        transformed = cv2.flip(transformed, 0)

    rotate_angle = settings["rotate_angle"]
    if rotate_angle != 0:
        h, w = transformed.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_angle, 1.0)
        transformed = cv2.warpAffine(transformed, matrix, (w, h))

    return transformed


def preprocess_for_model(image_bgr: np.ndarray, model_type: str) -> np.ndarray:
    width, height = MODEL_INPUT_SHAPES.get(model_type, (512, 512))
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)


def predict_probabilities(image_bgr: np.ndarray, model_type: str) -> np.ndarray:
    model = get_model(model_type)
    model_input = preprocess_for_model(image_bgr, model_type)
    raw = model(model_input, training=False)
    preds = raw.numpy() if hasattr(raw, "numpy") else raw
    return np.squeeze(preds)


def otsu_threshold(predictions: np.ndarray) -> float:
    flat = (predictions * 255).astype(np.uint8).flatten()
    threshold, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold / 255.0


def constrained_flood_fill(sandboil_mask: np.ndarray, seepage_mask: np.ndarray, distance_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    sandboil_contours, _ = cv2.findContours(sandboil_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seepage_contours, _ = cv2.findContours(seepage_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    updated_sandboil = sandboil_mask.copy()
    updated_seepage = seepage_mask.copy()

    for sandboil_cnt in sandboil_contours:
        for seepage_cnt in seepage_contours:
            if len(seepage_cnt) == 0:
                continue
            dist = cv2.pointPolygonTest(
                sandboil_cnt,
                tuple(map(int, seepage_cnt[0][0])),
                measureDist=True,
            )
            if abs(dist) < distance_threshold:
                cv2.drawContours(updated_seepage, [seepage_cnt], -1, 0, -1)

    return updated_sandboil, updated_seepage


def remove_smaller_overlaps(mask1: np.ndarray, mask2: np.ndarray, distance_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    contours1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    updated1, updated2 = mask1.copy(), mask2.copy()
    for cnt1 in contours1:
        area1 = cv2.contourArea(cnt1)
        for cnt2 in contours2:
            if len(cnt2) == 0:
                continue
            area2 = cv2.contourArea(cnt2)
            probe = (int(cnt2[0][0][0]), int(cnt2[0][0][1]))
            dist = cv2.pointPolygonTest(cnt1, probe, measureDist=True)
            if abs(dist) < distance_threshold:
                if area1 < area2:
                    cv2.drawContours(updated1, [cnt1], -1, 0, -1)
                else:
                    cv2.drawContours(updated2, [cnt2], -1, 0, -1)

    return updated1, updated2


def remove_nearby_seepage(sandboil_mask: np.ndarray, seepage_mask: np.ndarray, distance_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    kernel_size = distance_threshold if distance_threshold % 2 == 1 else distance_threshold + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    sandboil_dilated = cv2.dilate(sandboil_mask, kernel, iterations=1)
    updated_seepage = np.where(sandboil_dilated > 0, 0, seepage_mask)
    return sandboil_mask, updated_seepage


def resolve_overlaps(masks: dict[str, np.ndarray], distance_threshold: int) -> dict[str, np.ndarray]:
    if len(masks) <= 1:
        return masks

    out = masks.copy()
    types = list(out.keys())
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            t1, t2 = types[i], types[j]
            if t1 == "sandboil" and t2 == "seepage":
                out[t1], out[t2] = constrained_flood_fill(out[t1], out[t2], distance_threshold)
                out[t1], out[t2] = remove_smaller_overlaps(out[t1], out[t2], distance_threshold)
                out[t1], out[t2] = remove_nearby_seepage(out[t1], out[t2], distance_threshold)
            else:
                out[t1], out[t2] = remove_smaller_overlaps(out[t1], out[t2], distance_threshold)

    return out


def draw_overlay(image_bgr: np.ndarray, mask: np.ndarray, model_type: str, alpha: float) -> np.ndarray:
    mask_resized = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = DETECTION_COLORS.get(model_type, (255, 255, 255))
    color_layer = np.zeros_like(image_bgr, dtype=np.uint8)
    color_layer[mask_resized > 0] = color
    return cv2.addWeighted(image_bgr.astype(np.uint8), 1 - alpha, color_layer, alpha, 0)


def draw_bounding_boxes(image_bgr: np.ndarray, mask: np.ndarray, model_type: str) -> np.ndarray:
    mask_resized = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = DETECTION_COLORS.get(model_type, (255, 255, 255))
    contours, _ = cv2.findContours((mask_resized > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image_bgr.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    return output


def infer_image(
    image_bytes: bytes,
    selected_models: list[str],
    thresholds: dict[str, float],
    threshold_types: dict[str, str],
    visualization: str,
    overlay_intensity: float,
    distance_threshold: int,
    preprocessing_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    image_bgr = decode_image(image_bytes)
    settings = sanitize_preprocessing(preprocessing_settings)
    rendered, transformed, model_stats = render_inference_frame(
        image_bgr=image_bgr,
        settings=settings,
        selected_models=selected_models,
        thresholds=thresholds,
        threshold_types=threshold_types,
        visualization=visualization,
        overlay_intensity=overlay_intensity,
        distance_threshold=distance_threshold,
    )

    return {
        "image_base64": encode_png_base64(rendered),
        "preprocessed_base64": encode_png_base64(transformed),
        "model_stats": model_stats,
        "applied_settings": settings,
    }


def render_inference_frame(
    image_bgr: np.ndarray,
    settings: dict[str, Any],
    selected_models: list[str],
    thresholds: dict[str, float],
    threshold_types: dict[str, str],
    visualization: str,
    overlay_intensity: float,
    distance_threshold: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    transformed = apply_base_transforms(image_bgr, settings)

    predictions: dict[str, np.ndarray] = {}
    for model_type in selected_models:
        predictions[model_type] = predict_probabilities(transformed, model_type)

    masks: dict[str, np.ndarray] = {}
    model_stats: dict[str, dict[str, Any]] = {}
    for model_type, preds in predictions.items():
        threshold_mode = threshold_types.get(model_type, "Manual")
        if threshold_mode.lower() == "automatic":
            threshold = otsu_threshold(preds)
        else:
            threshold = float(thresholds.get(model_type, 0.5))

        mask = (preds > threshold).astype(np.uint8)
        if model_type == "seepage":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, kernel, iterations=1)

        # Normalize all masks to a common shape before overlap resolution.
        mask = cv2.resize(mask, (transformed.shape[1], transformed.shape[0]), interpolation=cv2.INTER_NEAREST)

        masks[model_type] = mask
        model_stats[model_type] = {
            "threshold_mode": threshold_mode,
            "threshold_used": round(float(threshold), 4),
            "coverage_pct": round(float(np.mean(mask > 0) * 100.0), 4),
            "positive_pixels": int(np.sum(mask > 0)),
        }

    masks = resolve_overlaps(masks, int(distance_threshold))

    if visualization == "bbox":
        rendered = transformed.copy()
        for model_type, mask in masks.items():
            rendered = draw_bounding_boxes(rendered, mask, model_type)
    else:
        rendered = transformed.copy()
        for model_type, mask in masks.items():
            rendered = draw_overlay(rendered, mask, model_type, overlay_intensity)

    return rendered, transformed, model_stats


def infer_video(
    video_bytes: bytes,
    selected_models: list[str],
    thresholds: dict[str, float],
    threshold_types: dict[str, str],
    visualization: str,
    overlay_intensity: float,
    distance_threshold: int,
    preprocessing_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    settings = sanitize_preprocessing(preprocessing_settings)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    output_path = None
    try:
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise ValueError("Could not open video")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 20.0

        frame_count = 0
        writer = None

        aggregate: dict[str, dict[str, Any]] = {
            model_type: {
                "threshold_mode": threshold_types.get(model_type, "Manual"),
                "threshold_used_total": 0.0,
                "coverage_pct_total": 0.0,
                "positive_pixels_total": 0,
                "frame_count": 0,
            }
            for model_type in selected_models
        }

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_tmp:
            output_path = output_tmp.name

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rendered, _, stats = render_inference_frame(
                image_bgr=frame,
                settings=settings,
                selected_models=selected_models,
                thresholds=thresholds,
                threshold_types=threshold_types,
                visualization=visualization,
                overlay_intensity=overlay_intensity,
                distance_threshold=distance_threshold,
            )

            if writer is None:
                h, w = rendered.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise ValueError("Could not create output video")

            writer.write(rendered)
            frame_count += 1

            for model_type, model_stats in stats.items():
                item = aggregate[model_type]
                item["threshold_used_total"] += float(model_stats["threshold_used"])
                item["coverage_pct_total"] += float(model_stats["coverage_pct"])
                item["positive_pixels_total"] += int(model_stats["positive_pixels"])
                item["frame_count"] += 1

        capture.release()
        if writer is not None:
            writer.release()

        if frame_count == 0:
            raise ValueError("Video has no readable frames")

        model_stats: dict[str, dict[str, Any]] = {}
        for model_type, item in aggregate.items():
            n = max(1, int(item["frame_count"]))
            model_stats[model_type] = {
                "threshold_mode": item["threshold_mode"],
                "threshold_used": round(float(item["threshold_used_total"]) / n, 4),
                "coverage_pct": round(float(item["coverage_pct_total"]) / n, 4),
                "positive_pixels": int(item["positive_pixels_total"]),
            }

        if output_path is None:
            raise ValueError("Could not create output video")

        with open(output_path, "rb") as f:
            output_bytes = f.read()

        return {
            "video_bytes": output_bytes,
            "model_stats": model_stats,
            "applied_settings": settings,
            "frame_count": frame_count,
            "fps": round(float(fps), 2),
        }
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
