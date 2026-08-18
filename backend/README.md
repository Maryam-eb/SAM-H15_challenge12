# VisionVerse AI Backend 2.0

This backend turns the original single-model caption endpoint into a grounded multi-stage image-captioning pipeline.

## Pipeline

```text
Image
  -> validation / preprocessing
  -> object detection (DETR)
  -> scene + action classification (CLIP zero-shot labels)
  -> BLIP + GIT (+ optional ViT-GPT2) candidate generation
  -> candidate deduplication
  -> CLIP image-text reranking
  -> context-aware fusion
  -> grounded candidate-fusion rewrite (default, CPU-friendly)
  -> optional instruction-model rewrite
  -> optional NLLB translation
  -> API response with analysis, alternatives and scores
```

The factual caption is selected before marketing rewriting. This keeps image grounding separate from style, audience and language changes.

## Main improvements

- Multi-model caption ensemble instead of one BLIP output.
- 3 candidates per enabled caption model by default.
- CLIP reranking with `similarity_score` and `selection_probability`.
- Real object detections with labels, confidence scores and boxes.
- Scene and action candidates with explicit similarity scores.
- A real `FusionModel` that deduplicates and ranks candidates with confidence-weighted scene/action/object evidence.
- Grounded Professional/Marketing/Social variants selected from actual caption-model candidates instead of generic suffix templates.
- Optional Qwen instruction rewrite for tone/audience/industry/language.
- On-demand NLLB translation for Persian, Italian and other supported languages.
- Explicit local-vs-Hub model source reporting in `/api/status`.
- Async remote-image downloads with redirect limits, upload limits and SSRF protection.
- Transformer inference moved off the FastAPI event loop and guarded by a concurrency semaphore.
- Offline BLEU, METEOR, optional CIDEr and optional CLIP-similarity evaluation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

## Recommended model profiles

### Balanced default

```env
ENABLE_BLIP=true
ENABLE_GIT=true
ENABLE_VIT_GPT2=false
ENABLE_CLIP_RERANKER=true
ENABLE_OBJECT_DETECTION=true
ENABLE_FUSION_MODEL=false
```

This gives multi-model generation, reranking and visual context without loading every optional model.

### Full demo

```env
ENABLE_BLIP=true
ENABLE_GIT=true
ENABLE_VIT_GPT2=true
ENABLE_CLIP_RERANKER=true
ENABLE_OBJECT_DETECTION=true
ENABLE_FUSION_MODEL=true
```

The full profile is heavier. Use it when the machine has enough RAM/VRAM.

## Local fine-tuned BLIP weights

Place the fine-tuned model under:

```text
weights/visionverse_caption/
```

The directory should contain a real model weight file such as `model.safetensors` or `pytorch_model.bin`.

Check:

```text
GET /api/status
```

Important fields:

```json
{
  "model_loaded": true,
  "loaded_models": {
    "captioners": ["blip", "git"],
    "clip_reranker": true,
    "object_detector": true,
    "fusion_model": false
  },
  "sources": {
    "blip": ".../weights/visionverse_caption",
    "git": "microsoft/git-base-coco",
    "clip": "openai/clip-vit-base-patch32"
  },
  "local_blip_active": true
}
```

If `local_blip_active` is false, the local fine-tuned BLIP is not the active source.

## Caption endpoint

Multipart upload:

```bash
curl -X POST http://127.0.0.1:8000/api/generate-captions \\
  -F "image=@example.jpg" \\
  -F "industry=Digital Marketing" \\
  -F "audience=Students" \\
  -F "tone=Professional" \\
  -F "language=English"
```

Remote URL:

```json
{
  "image_url": "https://example.com/photo.jpg",
  "industry": "Education",
  "audience": "Students",
  "tone": "Friendly",
  "language": "Italian"
}
```

The backend blocks private/local IP targets and validates every redirect. In production, set `REMOTE_IMAGE_ALLOWED_HOSTS` to a comma-separated allow-list.

## Example response shape

```json
{
  "success": true,
  "caption": "A cyclist rides along a city street.",
  "language": "English",
  "image_analysis": {
    "objects": ["person", "bicycle"],
    "scene": "street",
    "action": "riding",
    "selected_caption": "A cyclist rides along a city street.",
    "generated_by": "blip",
    "similarity_score": 0.31,
    "selection_probability": 0.67,
    "candidate_count": 6,
    "candidates": []
  }
}
```

`similarity_score` is deliberately not called confidence. It is a visual-text similarity signal, not a calibrated probability of correctness.

The default rewrite layer does not require an LLM. It selects distinct, high-scoring BLIP/GIT alternatives and exposes `rewrite_mode: "grounded_candidate_fusion"`. If `ENABLE_FUSION_MODEL=true`, the instruction model can perform a fuller tone/audience rewrite while remaining grounded in the selected caption and visual context.

## Translation

Non-English output uses NLLB on demand. The model is not loaded for English requests. Supported aliases currently include Persian/Farsi, Italian, French, German, Spanish, Arabic, Turkish, Portuguese, Russian, Japanese, Korean and Chinese.

If the translation model cannot be loaded, the API keeps the English fallback and returns `language_applied: false` plus a warning instead of pretending translation succeeded.

## Evaluation

Prepare JSONL:

```json
{"image":"images/1.jpg","prediction":"a dog on grass","references":["a dog runs across a grassy field"]}
```

Run text metrics:

```bash
python evaluate_captions.py evaluation.jsonl
```

Run text metrics plus CLIP semantic similarity:

```bash
python evaluate_captions.py evaluation.jsonl --clip
```

For CIDEr, install the optional evaluation dependencies:

```bash
pip install -r requirements-eval.txt
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests are intentionally light and do not download model weights.
