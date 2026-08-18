"""Caption fusion and optional instruction-model rewriting."""

import json
import re
from typing import Dict, Iterable, List, Optional


def _normalized(text: str) -> str:
    return " ".join((text or "").strip().lower().rstrip(".").split())


def deduplicate_candidates(candidates: Iterable[Dict]) -> List[Dict]:
    """Remove duplicate captions while preserving the first source."""
    seen = set()
    unique = []
    for item in candidates:
        caption = (item.get("caption") or "").strip()
        key = _normalized(caption)
        if not key or key in seen:
            continue
        seen.add(key)
        clean = dict(item)
        clean["caption"] = caption
        unique.append(clean)
    return unique


_CONTEXT_ALIASES = {
    "taking a photo": ["taking a photo", "taking a picture", "photo", "picture", "selfie", "photograph"],
    "using a computer": ["computer", "laptop", "keyboard", "working"],
    "riding": ["riding", "ride", "cyclist", "cycling"],
    "driving": ["driving", "driver", "car", "vehicle"],
    "holding": ["holding", "holds", "carrying", "carries"],
    "drinking": ["drinking", "drink", "cup", "glass", "coffee"],
    "shopping": ["shopping", "shop", "store"],
    "street": ["street", "road", "sidewalk"],
    "store": ["store", "shop", "shopping"],
    "home": ["home", "house", "room"],
    "kitchen": ["kitchen", "cooking"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "cell phone": ["cell phone", "phone", "smartphone"],
}


def _contains_context(text: str, label: str) -> bool:
    text = _normalized(text)
    label = _normalized(label)
    if not text or not label:
        return False
    checks = _CONTEXT_ALIASES.get(label, [label])
    return any(_normalized(check) in text for check in checks)


def _ranking_confidence(context: Dict, key: str) -> float:
    rows = context.get("%s_candidates" % key) or []
    if not rows:
        return 0.5
    top = rows[0]
    try:
        probability = float(top.get("selection_probability", 0.0))
    except Exception:
        probability = 0.0
    # CLIP softmax over hand-written labels is not calibrated. Use it only as
    # a modest reliability factor rather than as a probability of correctness.
    return min(1.0, max(0.5, 0.5 + 0.5 * probability))


class FusionModel:
    """Select the most image-grounded caption from multiple model outputs."""

    def _context_bonus(self, caption: str, context: Optional[Dict]):
        if not context:
            return 0.0, []

        evidence = []
        bonus = 0.0

        # High-confidence object detections are strong, literal evidence.
        for label in context.get("objects") or []:
            if _contains_context(caption, str(label)):
                evidence.append("object:%s" % label)
                bonus += 0.012

        scene = str(context.get("scene") or "").strip()
        if scene and scene != "unknown" and _contains_context(caption, scene):
            evidence.append("scene:%s" % scene)
            bonus += 0.015 * _ranking_confidence(context, "scene")

        action = str(context.get("action") or "").strip()
        if action and action != "unknown" and _contains_context(caption, action):
            evidence.append("action:%s" % action)
            bonus += 0.03 * _ranking_confidence(context, "action")

        # Context should break close ties, not overpower direct image-text CLIP.
        return min(0.05, bonus), evidence

    def select(self, image, candidates: Iterable[Dict], reranker=None,
               context: Optional[Dict] = None, top_k: int = 5) -> Dict:
        candidates = deduplicate_candidates(candidates)
        if not candidates:
            return {"selected": None, "ranked": [], "method": "none"}

        if reranker is not None:
            ranked = reranker.rank_candidates(image, candidates)
            for item in ranked:
                bonus, evidence = self._context_bonus(item["caption"], context)
                item["context_bonus"] = round(bonus, 4)
                item["context_evidence"] = evidence
                item["fusion_score"] = round(
                    float(item.get("similarity_score", 0.0)) + bonus, 4
                )
            ranked.sort(key=lambda x: x.get("fusion_score", -999.0), reverse=True)
            method = "clip_similarity_plus_context"
        else:
            ranked = []
            for index, item in enumerate(candidates):
                copy = dict(item)
                copy["rank"] = index + 1
                copy["similarity_score"] = None
                copy["selection_probability"] = None
                bonus, evidence = self._context_bonus(copy["caption"], context)
                copy["context_bonus"] = round(bonus, 4)
                copy["context_evidence"] = evidence
                copy["fusion_score"] = round(bonus, 4)
                ranked.append(copy)
            method = "generation_order_plus_context_fallback"

        # Re-number after the final fusion sort so rank matches the returned order.
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index

        selected = dict(ranked[0])
        return {
            "selected": selected,
            "ranked": ranked[:max(1, top_k)],
            "method": method,
        }


class InstructionFusionModel:
    """Optional small instruction model for tone/language-aware rewriting.

    The factual caption is selected before this model runs, so marketing copy
    cannot change which visual facts win the image-grounding stage.
    """

    def __init__(self, tokenizer, model, device, source):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.source = str(source)

    @classmethod
    def from_pretrained(cls, source, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source)
        kwargs = {}
        if str(device).startswith("cuda"):
            kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        model.to(device)
        model.eval()
        return cls(tokenizer, model, device, source)

    def _extract_json(self, text: str) -> Optional[Dict]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def generate_marketing(self, description: str, context: Dict,
                           industry: str, audience: str, tone: str,
                           language: str) -> Optional[Dict]:
        import torch

        safe_context = {
            "objects": context.get("objects") or [],
            "scene": context.get("scene") or "unknown",
            "action": context.get("action") or "unknown",
        }
        prompt = (
            "You are a factual image-caption editor. Use only facts present in "
            "the factual caption or visual context. Do not invent people, brands, "
            "locations, emotions, products, or actions. Rewrite for the requested "
            "tone, audience, industry and language. Return ONLY valid JSON with "
            "keys professional, marketing, social_media, cta, hashtags. hashtags "
            "must be a JSON array of 3 to 5 short hashtags.\n\n"
            "Factual caption: %s\n"
            "Visual context: %s\n"
            "Industry: %s\nAudience: %s\nTone: %s\nLanguage: %s\n"
        ) % (
            description,
            json.dumps(safe_context, ensure_ascii=False),
            industry,
            audience,
            tone,
            language,
        )

        messages = [
            {"role": "system", "content": "Preserve visual facts exactly."},
            {"role": "user", "content": prompt},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            rendered = prompt

        inputs = self.tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=320,
                do_sample=False,
                repetition_penalty=1.05,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = output[0][prompt_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        data = self._extract_json(text)
        if not data:
            return None

        required = {"professional", "marketing", "social_media", "cta", "hashtags"}
        if not required.issubset(data):
            return None
        if not isinstance(data.get("hashtags"), list):
            return None
        return data
