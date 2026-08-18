"""Offline caption quality metrics.

Input records are expected to contain:
  {"prediction": str, "references": [str, ...], "image": optional_path}

BLEU and METEOR are text metrics. CIDEr is used when pycocoevalcap is
installed. CLIP similarity is optional and requires image paths plus the
configured CLIP reranker.
"""

from pathlib import Path
from statistics import mean


def _tokens(text):
    return (text or "").lower().strip().split()


def text_metrics(records):
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:
        raise RuntimeError("Install nltk to compute BLEU and METEOR") from exc

    refs = [[_tokens(ref) for ref in item["references"]] for item in records]
    hyps = [_tokens(item["prediction"]) for item in records]
    smoothing = SmoothingFunction().method1

    class _NoWordNet:
        @staticmethod
        def synsets(word):
            return []

    result = {
        "bleu_1": corpus_bleu(refs, hyps, weights=(1.0, 0, 0, 0), smoothing_function=smoothing),
        "bleu_4": corpus_bleu(
            refs,
            hyps,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        ),
        "meteor": mean(
            meteor_score(reference, hypothesis, wordnet=_NoWordNet())
            for reference, hypothesis in zip(refs, hyps)
        ) if records else 0.0,
    }
    return {key: round(float(value), 4) for key, value in result.items()}


def cider_metric(records):
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        return None, "pycocoevalcap not installed; CIDEr skipped"

    ground_truth = {}
    predictions = {}
    for index, item in enumerate(records):
        ground_truth[index] = item["references"]
        predictions[index] = [item["prediction"]]

    score, _ = Cider().compute_score(ground_truth, predictions)
    return round(float(score), 4), None


def clip_similarity(records):
    from PIL import Image
    from backend.models.model_loader import ModelLoader

    reranker = ModelLoader().load_all().get("reranker")
    if reranker is None:
        return None, "CLIP reranker unavailable; semantic similarity skipped"

    scores = []
    for item in records:
        image_path = item.get("image")
        if not image_path:
            continue
        path = Path(image_path)
        if not path.is_file():
            continue
        image = Image.open(path).convert("RGB")
        cosine, _ = reranker._scores(image, [item["prediction"]])
        scores.append(float(cosine[0]))

    if not scores:
        return None, "No readable image paths; semantic similarity skipped"
    return round(mean(scores), 4), None


def evaluate_records(records, include_clip=False):
    if not records:
        raise ValueError("No evaluation records were supplied")
    for item in records:
        if not item.get("prediction") or not item.get("references"):
            raise ValueError("Every record needs prediction and references")

    metrics = text_metrics(records)
    warnings = []

    cider, error = cider_metric(records)
    metrics["cider"] = cider
    if error:
        warnings.append(error)

    if include_clip:
        semantic, error = clip_similarity(records)
        metrics["clip_similarity"] = semantic
        if error:
            warnings.append(error)

    return {"metrics": metrics, "warnings": warnings}
