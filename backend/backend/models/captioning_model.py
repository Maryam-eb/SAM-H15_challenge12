"""End-to-end image analysis and caption generation pipeline."""

import io
from typing import Dict, List

from PIL import Image

from backend.config.settings import settings
from backend.services.marketing_service import build_marketing_payload
from backend.services.translation_service import normalize_language
from .fusion_model import FusionModel
from .model_loader import ModelLoader


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable image: %s" % exc)


def _unique_object_labels(detections: List[Dict]) -> List[str]:
    seen = set()
    labels = []
    for item in detections:
        label = (item.get("label") or "").strip()
        key = label.lower()
        if label and key not in seen:
            seen.add(key)
            labels.append(label)
    return labels


def _visual_context(image, models):
    detector = models.get("detector")
    reranker = models.get("reranker")
    warnings = []
    detections = []
    scene_rankings = []
    action_rankings = []

    if detector is not None:
        try:
            detections = detector.detect(image, threshold=settings.DETECTION_THRESHOLD)
        except Exception as exc:
            warnings.append("Object detection failed: %s" % exc)

    if reranker is not None:
        try:
            scene_rankings = reranker.classify_labels(
                image,
                settings.SCENE_LABELS,
                template="a photo taken in or around a {}",
                top_k=settings.CONTEXT_TOP_K,
            )
        except Exception as exc:
            warnings.append("Scene classification failed: %s" % exc)
        try:
            action_rankings = reranker.classify_labels(
                image,
                settings.ACTION_LABELS,
                template="a photo showing someone {}",
                top_k=settings.CONTEXT_TOP_K,
            )
        except Exception as exc:
            warnings.append("Action classification failed: %s" % exc)

    detections = detections[:settings.MAX_OBJECTS]
    objects = _unique_object_labels(detections)
    scene = (
        scene_rankings[0]["label"]
        if scene_rankings and scene_rankings[0]["similarity_score"] >= settings.CONTEXT_MIN_SIMILARITY
        else "unknown"
    )
    action = (
        action_rankings[0]["label"]
        if action_rankings and action_rankings[0]["similarity_score"] >= settings.CONTEXT_MIN_SIMILARITY
        else "unknown"
    )

    return {
        "objects": objects,
        "detections": detections,
        "scene": scene,
        "scene_candidates": scene_rankings,
        "action": action,
        "action_candidates": action_rankings,
        "warnings": warnings,
    }


def _generate_candidates(image, models):
    candidates = []
    warnings = []
    captioners = models.get("captioners") or {}

    for name, captioner in captioners.items():
        try:
            texts = captioner.generate_candidates(
                image,
                count=settings.CANDIDATES_PER_MODEL,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                beam_width=settings.BEAM_WIDTH,
            )
            for text in texts:
                candidates.append({
                    "caption": text,
                    "model": name,
                    "source": getattr(captioner, "source", None),
                })
        except Exception as exc:
            warnings.append("%s generation failed: %s" % (name, exc))

    return candidates, warnings


def analyze_image(image_bytes):
    """Generate, compare and select factual captions plus visual context."""

    image = _open_image(image_bytes)
    models = ModelLoader().load_all()
    captioners = models.get("captioners") or {}

    if not captioners:
        return {
            "objects": [],
            "detections": [],
            "scene": "unknown",
            "action": "unknown",
            "description": "",
            "selected_caption": "",
            "candidates": [],
            "features": {
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(image.width / image.height, 3) if image.height else None,
            },
            "model_loaded": False,
            "device": models.get("device"),
            "errors": dict(models.get("errors") or {}),
            "error": "No captioning model could be loaded",
        }

    context = _visual_context(image, models)
    candidates, generation_warnings = _generate_candidates(image, models)

    if not candidates:
        return {
            **context,
            "description": "",
            "selected_caption": "",
            "candidates": [],
            "features": {
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(image.width / image.height, 3) if image.height else None,
            },
            "model_loaded": True,
            "device": models.get("device"),
            "sources": dict(models.get("sources") or {}),
            "errors": dict(models.get("errors") or {}),
            "error": "Caption models loaded, but none produced a caption",
            "warnings": context.get("warnings", []) + generation_warnings,
        }

    selector = FusionModel()
    fusion = selector.select(
        image=image,
        candidates=candidates,
        reranker=models.get("reranker"),
        context=context,
        top_k=settings.TOP_CANDIDATES,
    )
    selected = fusion.get("selected") or candidates[0]
    ranked = fusion.get("ranked") or candidates[:settings.TOP_CANDIDATES]

    warnings = list(context.get("warnings", [])) + generation_warnings
    if models.get("reranker") is None:
        warnings.append("CLIP reranker unavailable; first-generation fallback was used")

    return {
        "objects": context["objects"],
        "detections": context["detections"],
        "scene": context["scene"],
        "scene_candidates": context["scene_candidates"],
        "action": context["action"],
        "action_candidates": context["action_candidates"],
        "description": selected.get("caption", ""),
        "selected_caption": selected.get("caption", ""),
        "generated_by": selected.get("model"),
        "similarity_score": selected.get("similarity_score"),
        "selection_probability": selected.get("selection_probability"),
        "fusion_score": selected.get("fusion_score"),
        "selection_method": fusion.get("method"),
        "candidate_count": len(candidates),
        "candidates": ranked,
        "features": {
            "width": image.width,
            "height": image.height,
            "aspect_ratio": round(image.width / image.height, 3) if image.height else None,
        },
        "model_loaded": True,
        "device": models.get("device"),
        "sources": dict(models.get("sources") or {}),
        "errors": dict(models.get("errors") or {}),
        "warnings": warnings,
    }


def process_caption_request(image_bytes, category="General", audience="General",
                            tone="Professional", language="English"):
    """Image bytes -> grounded visual analysis -> final marketing response."""

    normalize_language(language)
    analysis = analyze_image(image_bytes)
    models = ModelLoader().load_all()

    payload = build_marketing_payload(
        description=analysis.get("description", ""),
        industry=category,
        audience=audience,
        tone=tone,
        language=language,
        model_loaded=analysis.get("model_loaded", False),
        context=analysis,
        fusion_model=models.get("fusion"),
    )

    payload["success"] = bool(
        analysis.get("model_loaded") and analysis.get("selected_caption")
    )
    payload["image_analysis"] = analysis

    all_warnings = []
    all_warnings.extend(analysis.get("warnings") or [])
    all_warnings.extend(payload.pop("warnings", []) or [])
    if all_warnings:
        payload["warnings"] = all_warnings

    if not payload["success"]:
        payload["warning"] = analysis.get("error", "Caption generation failed")

    return payload
