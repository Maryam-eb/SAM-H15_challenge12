"""Lazy loading for captioning, reranking, detection and fusion models."""

import threading
from pathlib import Path

from backend.config.settings import settings
from .fusion_model import InstructionFusionModel
from .registry import MODEL_REGISTRY

_LOAD_LOCK = threading.Lock()
_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _has_local_weights(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in _WEIGHT_FILES)


def _clean_caption(text: str, prompt=None) -> str:
    text = " ".join((text or "").strip().split())
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip(" .,:;-")
    return text.strip()


class BLIPCaptioner:
    name = "blip"

    def __init__(self, processor, model, device, source):
        self.processor = processor
        self.model = model
        self.device = device
        self.source = str(source)

    def generate_candidates(self, image, count=3, max_new_tokens=48, beam_width=5,
                            prompt=None):
        import torch

        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)

        count = max(1, int(count))
        beams = max(count, int(beam_width))
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=beams,
                num_return_sequences=count,
                no_repeat_ngram_size=2,
                repetition_penalty=1.05,
                early_stopping=True,
            )
        texts = self.processor.batch_decode(output, skip_special_tokens=True)
        return [_clean_caption(text, prompt) for text in texts if _clean_caption(text, prompt)]


class GITCaptioner:
    name = "git"

    def __init__(self, processor, model, device, source):
        self.processor = processor
        self.model = model
        self.device = device
        self.source = str(source)

    def generate_candidates(self, image, count=3, max_new_tokens=48, beam_width=5,
                            prompt=None):
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        count = max(1, int(count))
        beams = max(count, int(beam_width))
        with torch.inference_mode():
            output = self.model.generate(
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=beams,
                num_return_sequences=count,
                no_repeat_ngram_size=2,
                repetition_penalty=1.05,
                early_stopping=True,
            )
        texts = self.processor.batch_decode(output, skip_special_tokens=True)
        return [_clean_caption(text) for text in texts if _clean_caption(text)]


class ViTGPT2Captioner:
    name = "vit_gpt2"

    def __init__(self, image_processor, tokenizer, model, device, source):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.source = str(source)

    def generate_candidates(self, image, count=3, max_new_tokens=48, beam_width=5,
                            prompt=None):
        import torch

        pixel_values = self.image_processor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        count = max(1, int(count))
        beams = max(count, int(beam_width))
        with torch.inference_mode():
            output = self.model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=beams,
                num_return_sequences=count,
                no_repeat_ngram_size=2,
                repetition_penalty=1.05,
                early_stopping=True,
            )
        texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return [_clean_caption(text) for text in texts if _clean_caption(text)]


class CLIPReranker:
    def __init__(self, processor, model, device, source):
        self.processor = processor
        self.model = model
        self.device = device
        self.source = str(source)

    def _scores(self, image, texts):
        import torch

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            cosine = (image_embeds @ text_embeds.T)[0]
            logits = outputs.logits_per_image[0]
            probs = torch.softmax(logits, dim=-1)
        return cosine.detach().cpu().tolist(), probs.detach().cpu().tolist()

    def rank_candidates(self, image, candidates):
        texts = [item["caption"] for item in candidates]
        cosine, probs = self._scores(image, texts)
        ranked = []
        for item, score, probability in zip(candidates, cosine, probs):
            copy = dict(item)
            copy["similarity_score"] = round(float(score), 4)
            copy["selection_probability"] = round(float(probability), 4)
            ranked.append(copy)
        ranked.sort(key=lambda x: x["similarity_score"], reverse=True)
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index
        return ranked

    def classify_labels(self, image, labels, template="a photo of {}", top_k=3):
        if not labels:
            return []
        texts = [template.format(label) for label in labels]
        cosine, probs = self._scores(image, texts)
        rows = [
            {
                "label": label,
                "similarity_score": round(float(score), 4),
                "selection_probability": round(float(probability), 4),
            }
            for label, score, probability in zip(labels, cosine, probs)
        ]
        rows.sort(key=lambda x: x["similarity_score"], reverse=True)
        return rows[:max(1, int(top_k))]


class DETRDetector:
    def __init__(self, processor, model, device, source):
        self.processor = processor
        self.model = model
        self.device = device
        self.source = str(source)

    def detect(self, image, threshold=0.8):
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=float(threshold)
        )[0]

        objects = []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            label_name = self.model.config.id2label[int(label)]
            objects.append({
                "label": label_name,
                "score": round(float(score.detach().cpu()), 4),
                "box": [round(float(value), 1) for value in box.detach().cpu().tolist()],
            })
        objects.sort(key=lambda x: x["score"], reverse=True)
        return objects


class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def device(self):
        if settings.DEVICE != "auto":
            return settings.DEVICE
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _record_error(self, key, exc):
        MODEL_REGISTRY["errors"][key] = str(exc)
        print("%s loading failed: %s" % (key, exc))

    def _sources(self, local_path, hub_id):
        sources = []
        if local_path and _has_local_weights(Path(local_path)):
            sources.append(str(local_path))
        if settings.ALLOW_HUB_FALLBACK and hub_id not in sources:
            sources.append(hub_id)
        return sources

    def _load_blip(self, source, device):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        try:
            processor = BlipProcessor.from_pretrained(source)
        except Exception:
            if not settings.ALLOW_HUB_FALLBACK:
                raise
            processor = BlipProcessor.from_pretrained(settings.BLIP_MODEL)
        model = BlipForConditionalGeneration.from_pretrained(source)
        model.to(device)
        model.eval()
        return BLIPCaptioner(processor, model, device, source)

    def _load_git(self, source, device):
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(source)
        model = AutoModelForCausalLM.from_pretrained(source)
        model.to(device)
        model.eval()
        return GITCaptioner(processor, model, device, source)

    def _load_vit_gpt2(self, source, device):
        from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

        image_processor = ViTImageProcessor.from_pretrained(source)
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = VisionEncoderDecoderModel.from_pretrained(source)
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer.bos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        model.to(device)
        model.eval()
        return ViTGPT2Captioner(image_processor, tokenizer, model, device, source)

    def _load_clip(self, device):
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
        model = CLIPModel.from_pretrained(settings.CLIP_MODEL)
        model.to(device)
        model.eval()
        return CLIPReranker(processor, model, device, settings.CLIP_MODEL)

    def _load_detector(self, device):
        from transformers import AutoImageProcessor, DetrForObjectDetection

        processor = AutoImageProcessor.from_pretrained(settings.DETECTOR_MODEL)
        model = DetrForObjectDetection.from_pretrained(settings.DETECTOR_MODEL)
        model.to(device)
        model.eval()
        return DETRDetector(processor, model, device, settings.DETECTOR_MODEL)

    def _load_first_available(self, key, sources, factory, device):
        if not sources:
            self._record_error(key, "No local weights and Hub fallback is disabled")
            return None
        for source in sources:
            try:
                print("Loading %s from: %s" % (key, source))
                model = factory(source, device)
                MODEL_REGISTRY["sources"][key] = str(source)
                MODEL_REGISTRY["errors"].pop(key, None)
                return model
            except Exception as exc:
                self._record_error(key, "%s: %s" % (source, exc))
        return None

    def load_all(self):
        if MODEL_REGISTRY.get("initialized"):
            return MODEL_REGISTRY

        with _LOAD_LOCK:
            if MODEL_REGISTRY.get("initialized"):
                return MODEL_REGISTRY

            device = self.device()
            MODEL_REGISTRY["device"] = device
            try:
                import torch
                if settings.TORCH_NUM_THREADS > 0:
                    torch.set_num_threads(settings.TORCH_NUM_THREADS)
            except Exception:
                pass

            captioners = {}

            if settings.ENABLE_BLIP:
                sources = self._sources(settings.caption_weights_path, settings.BLIP_MODEL)
                model = self._load_first_available("blip", sources, self._load_blip, device)
                if model:
                    captioners["blip"] = model
                    MODEL_REGISTRY["blip"] = model

            if settings.ENABLE_GIT:
                sources = self._sources(settings.git_weights_path, settings.GIT_MODEL)
                model = self._load_first_available("git", sources, self._load_git, device)
                if model:
                    captioners["git"] = model
                    MODEL_REGISTRY["git"] = model

            if settings.ENABLE_VIT_GPT2:
                sources = self._sources(settings.vit_gpt2_weights_path, settings.VIT_GPT2_MODEL)
                model = self._load_first_available(
                    "vit_gpt2", sources, self._load_vit_gpt2, device
                )
                if model:
                    captioners["vit_gpt2"] = model
                    MODEL_REGISTRY["vit_gpt2"] = model

            MODEL_REGISTRY["captioners"] = captioners
            MODEL_REGISTRY["captioning"] = captioners.get("blip") or next(
                iter(captioners.values()), None
            )

            if settings.ENABLE_CLIP_RERANKER:
                try:
                    print("Loading CLIP reranker from:", settings.CLIP_MODEL)
                    MODEL_REGISTRY["reranker"] = self._load_clip(device)
                    MODEL_REGISTRY["sources"]["clip"] = settings.CLIP_MODEL
                    MODEL_REGISTRY["errors"].pop("clip", None)
                except Exception as exc:
                    self._record_error("clip", exc)

            if settings.ENABLE_OBJECT_DETECTION:
                try:
                    print("Loading object detector from:", settings.DETECTOR_MODEL)
                    MODEL_REGISTRY["detector"] = self._load_detector(device)
                    MODEL_REGISTRY["sources"]["detector"] = settings.DETECTOR_MODEL
                    MODEL_REGISTRY["errors"].pop("detector", None)
                except Exception as exc:
                    self._record_error("detector", exc)

            if settings.ENABLE_FUSION_MODEL:
                try:
                    print("Loading instruction fusion model from:", settings.FUSION_MODEL)
                    MODEL_REGISTRY["fusion"] = InstructionFusionModel.from_pretrained(
                        settings.FUSION_MODEL, device
                    )
                    MODEL_REGISTRY["sources"]["fusion"] = settings.FUSION_MODEL
                    MODEL_REGISTRY["errors"].pop("fusion", None)
                except Exception as exc:
                    self._record_error("fusion", exc)

            MODEL_REGISTRY["initialized"] = True
            return MODEL_REGISTRY


def get_status():
    registry = MODEL_REGISTRY
    captioners = registry.get("captioners") or {}
    loaded = {
        "captioners": sorted(captioners.keys()),
        "clip_reranker": registry.get("reranker") is not None,
        "object_detector": registry.get("detector") is not None,
        "fusion_model": registry.get("fusion") is not None,
    }
    return {
        "initialized": bool(registry.get("initialized")),
        "model_loaded": bool(captioners),
        "loaded_models": loaded,
        "sources": dict(registry.get("sources") or {}),
        "device": registry.get("device"),
        "errors": dict(registry.get("errors") or {}),
        "local_blip_active": bool(
            registry.get("sources", {}).get("blip")
            and Path(registry["sources"]["blip"]).resolve() == settings.caption_weights_path.resolve()
        ),
    }
