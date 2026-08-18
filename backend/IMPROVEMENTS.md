# What changed

1. Replaced the single-caption BLIP path with an ensemble-ready caption pipeline.
2. Added BLIP, GIT and optional ViT-GPT2 wrappers.
3. Added multi-candidate beam generation.
4. Added CLIP reranking and semantic scores.
5. Replaced the placeholder `FusionModel` with candidate deduplication and selection logic.
6. Added DETR object detection with bounding boxes.
7. Added CLIP-based scene and action classification.
8. Replaced fake `objects: []` and `scene: image` fields with model outputs.
9. Added explicit model sources, per-model load errors and `local_blip_active` status.
10. Added optional Qwen instruction rewriting instead of relying only on fixed copy templates.
11. Added on-demand NLLB translation so `language` can change the actual output.
12. Added `language_applied` and warnings when translation cannot be performed.
13. Replaced blocking `requests.get()` with async `httpx` streaming.
14. Added remote URL validation, DNS/IP checks, redirect validation and private-network blocking.
15. Added bounded upload reads and remote content-length/stream size limits.
16. Moved synchronous model inference into a worker thread.
17. Added an inference concurrency semaphore to protect one GPU/CPU model stack.
18. Updated FastAPI startup handling to lifespan.
19. Added offline BLEU, METEOR, optional CIDEr and optional CLIP semantic evaluation.
20. Added environment configuration, requirements, tests and setup documentation.

## Grounded rewrite layer v2

- Removed the legacy generic suffixes such as "Presented with brand storytelling in mind" and "Clear, engaging, and ready for your next post" from the default path.
- The default CPU-friendly rewrite path now builds Professional, Marketing, and Social Media variants from the actual BLIP/GIT candidate pool.
- The winning factual caption remains the accuracy reference and is exposed as `factual_caption`.
- The response exposes `rewrite_mode` (`grounded_candidate_fusion` or `instruction_model`) plus a compact `grounding` block.
- Candidate fusion now recognizes semantic aliases for scene/action evidence (for example `selfie` as evidence for `taking a photo`) and returns `context_evidence` for each candidate.
- Context bonuses are confidence-weighted and capped so they can break close ties without overpowering CLIP image-text similarity.
- Added `timm` to runtime requirements because DETR/Transformers backbones may require it.
