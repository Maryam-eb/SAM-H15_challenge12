"""Build grounded caption variants from the AI candidate pool.

The fallback path intentionally avoids the old generic marketing templates.
When the optional instruction model is disabled, variants are selected from
actual caption-model outputs and lightly styled without adding new visual facts.
"""

import re
from typing import Dict, Iterable, List, Optional

from backend.services.translation_service import normalize_language, translate_many


INDUSTRY_PROFILES = {
    "digital marketing": {
        "hashtags": ["#DigitalMarketing", "#ContentStrategy"],
        "cta": "Explore the story",
    },
    "e-commerce": {
        "hashtags": ["#Ecommerce", "#ProductContent"],
        "cta": "Explore more",
    },
    "industry": {
        "hashtags": ["#Industrial", "#VisualAI"],
        "cta": "See the details",
    },
    "medical": {
        "hashtags": ["#Healthcare", "#VisualAI"],
        "cta": "Learn more",
    },
    "education": {
        "hashtags": ["#Education", "#Learning"],
        "cta": "Discover more",
    },
    "fashion": {
        "hashtags": ["#Fashion", "#Style"],
        "cta": "See the look",
    },
    "general": {
        "hashtags": ["#VisionVerse", "#VisualAI"],
        "cta": "Learn more",
    },
}

TONE_EMOJIS = {
    "professional": "",
    "friendly": "✨",
    "casual": "📸",
    "playful": "✨📸",
    "luxury": "✦",
    "bold": "⚡",
}

CONTEXT_EMOJIS = {
    "taking a photo": "📸",
    "using a computer": "💻",
    "drinking": "☕",
    "eating": "🍽️",
    "riding": "🚲",
    "driving": "🚗",
    "reading": "📚",
    "shopping": "🛍️",
    "beach": "🏖️",
    "park": "🌿",
}


def _key(value: str, table: Dict, default: str = "general") -> str:
    key = (value or "").strip().lower()
    return key if key in table else default


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().rstrip(".!?").split())


def _sentence(text: str) -> str:
    text = " ".join((text or "").strip().split())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _safe_tag(value: str) -> Optional[str]:
    clean = re.sub(r"[^A-Za-z0-9]+", "", (value or "").title())
    if len(clean) < 2 or clean.lower() == "general":
        return None
    return "#" + clean[:32]


def _candidate_rows(description: str, context: Dict) -> List[Dict]:
    """Return unique, model-generated candidate rows with the winner first."""
    rows = []
    seen = set()

    selected = (context.get("selected_caption") or description or "").strip()
    if selected:
        rows.append({
            "caption": selected,
            "model": context.get("generated_by"),
            "fusion_score": context.get("fusion_score"),
            "similarity_score": context.get("similarity_score"),
            "selection_probability": context.get("selection_probability"),
            "rank": 0,
        })
        seen.add(_normalize(selected))

    for item in context.get("candidates") or []:
        caption = (item.get("caption") or "").strip()
        key = _normalize(caption)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(dict(item))

    return rows


def _quality(row: Dict) -> float:
    for key in ("fusion_score", "similarity_score"):
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return 0.0


def _context_terms(context: Dict) -> List[str]:
    terms = [str(x).lower() for x in (context.get("objects") or [])]
    for key in ("scene", "action"):
        value = str(context.get(key) or "").strip().lower()
        if value and value != "unknown":
            terms.append(value)
    return terms


def _mentions_context(text: str, context: Dict) -> int:
    normalized = _normalize(text)
    score = 0
    aliases = {
        "taking a photo": ["taking a photo", "taking a picture", "photo", "picture", "selfie", "photograph"],
        "using a computer": ["computer", "laptop", "keyboard", "working"],
        "riding": ["riding", "ride", "cyclist", "cycling"],
        "drinking": ["drinking", "drink", "coffee", "cup", "glass"],
        "shopping": ["shopping", "shop", "store"],
        "holding": ["holding", "holds", "carrying", "carries"],
        "street": ["street", "road", "sidewalk"],
        "store": ["store", "shop", "shopping"],
        "home": ["home", "house", "room"],
        "kitchen": ["kitchen", "cooking"],
    }
    for term in _context_terms(context):
        checks = aliases.get(term, [term])
        if any(check in normalized for check in checks):
            score += 1
    return score


def _choose_variant(rows: List[Dict], context: Dict, goal: str, used: set) -> Dict:
    if not rows:
        return {"caption": ""}

    best_quality = max((_quality(row) for row in rows), default=0.0)
    quality_floor = best_quality * 0.75 if best_quality > 0 else 0.0
    eligible = [row for row in rows if _quality(row) >= quality_floor] or rows

    def score(row: Dict):
        text = row.get("caption") or ""
        context_hits = _mentions_context(text, context)
        length = len(text.split())
        novelty = 1 if _normalize(text) not in used else 0
        base = _quality(row)
        if goal == "professional":
            return (base, context_hits, -abs(length - 14), novelty)
        if goal == "marketing":
            return (context_hits, base, min(length, 24), novelty)
        # Social copy prefers a concise but context-rich alternative.
        return (novelty, context_hits, base, -length)

    return max(eligible, key=score)


def _tone_style(text: str, tone: str, context: Dict, social: bool = False) -> str:
    text = _sentence(text)
    tone_key = (tone or "professional").strip().lower()
    if not social or tone_key == "professional":
        return text

    emoji = TONE_EMOJIS.get(tone_key, "")
    if not emoji:
        action = str(context.get("action") or "").strip().lower()
        scene = str(context.get("scene") or "").strip().lower()
        emoji = CONTEXT_EMOJIS.get(action) or CONTEXT_EMOJIS.get(scene) or ""
    if emoji and emoji not in text:
        return "%s %s" % (text, emoji)
    return text


def _grounded_payload(description: str, industry: str, audience: str,
                      tone: str, context: Dict) -> Dict:
    rows = _candidate_rows(description, context)
    if not rows:
        rows = [{"caption": description or "A visual scene"}]

    used = set()
    professional_row = _choose_variant(rows, context, "professional", used)
    professional = _sentence(professional_row.get("caption") or description)
    used.add(_normalize(professional))

    marketing_row = _choose_variant(rows, context, "marketing", used)
    marketing = _sentence(marketing_row.get("caption") or professional)
    used.add(_normalize(marketing))

    social_row = _choose_variant(rows, context, "social", used)
    social = _tone_style(
        social_row.get("caption") or marketing,
        tone=tone,
        context=context,
        social=True,
    )

    industry_key = _key(industry, INDUSTRY_PROFILES)
    profile = INDUSTRY_PROFILES[industry_key]
    hashtags = ["#VisionVerse", "#ImageCaptioning", *profile["hashtags"]]
    audience_tag = _safe_tag(audience)
    if audience_tag:
        hashtags.append(audience_tag)
    hashtags = list(dict.fromkeys(hashtags))[:5]

    selected = _sentence(context.get("selected_caption") or description)
    selected_model = context.get("generated_by") or "caption model"
    candidate_count = context.get("candidate_count") or len(rows)
    action = context.get("action") or "unknown"
    scene = context.get("scene") or "unknown"

    return {
        "captions": [
            {"type": "Professional", "caption": professional},
            {"type": "Marketing", "caption": marketing},
            {"type": "Social Media", "caption": social},
        ],
        "hashtags": hashtags,
        "cta": profile["cta"],
        "marketing_strategy": (
            "Use AI-grounded alternatives from the %s candidate pool; keep the "
            "selected %s caption as the factual reference. Tone: %s. Audience: %s."
            % (candidate_count, selected_model, tone or "Professional", audience or "General")
        ),
        "recommendations": [
            "Keep the factual caption unchanged when visual accuracy matters most.",
            "Prefer the action-rich alternative only when it remains close to the top visual-similarity score.",
            "Current visual context: scene=%s, action=%s." % (scene, action),
        ],
        "rewrite_mode": "grounded_candidate_fusion",
        "factual_caption": selected,
        "grounding": {
            "selected_model": selected_model,
            "candidate_count": candidate_count,
            "scene": scene,
            "action": action,
            "objects": list(context.get("objects") or []),
        },
    }


def _payload_from_fusion(data: Dict, description: str, context: Dict) -> Dict:
    return {
        "captions": [
            {"type": "Professional", "caption": _sentence(str(data["professional"]).strip())},
            {"type": "Marketing", "caption": _sentence(str(data["marketing"]).strip())},
            {"type": "Social Media", "caption": _sentence(str(data["social_media"]).strip())},
        ],
        "hashtags": [str(item).strip() for item in data.get("hashtags", []) if str(item).strip()][:5],
        "cta": str(data.get("cta", "")).strip(),
        "marketing_strategy": "Instruction-model rewrite grounded in the selected factual caption and visual context.",
        "recommendations": [
            "Keep the selected factual caption as the accuracy reference.",
            "Use a rewritten variant only if it preserves the same people, objects, scene and action.",
            "A/B test the professional and social variants for channel fit.",
        ],
        "rewrite_mode": "instruction_model",
        "factual_caption": _sentence(description),
        "grounding": {
            "selected_model": context.get("generated_by"),
            "candidate_count": context.get("candidate_count"),
            "scene": context.get("scene") or "unknown",
            "action": context.get("action") or "unknown",
            "objects": list(context.get("objects") or []),
        },
    }


def _translate_payload(payload: Dict, language: str):
    # Hashtags and grounding labels stay machine-readable in English.
    fields = [
        payload["captions"][0]["caption"],
        payload["captions"][1]["caption"],
        payload["captions"][2]["caption"],
        payload["cta"],
        payload["marketing_strategy"],
        *payload["recommendations"],
    ]
    translated, applied, error = translate_many(fields, language)
    if applied:
        payload["captions"][0]["caption"] = translated[0]
        payload["captions"][1]["caption"] = translated[1]
        payload["captions"][2]["caption"] = translated[2]
        payload["cta"] = translated[3]
        payload["marketing_strategy"] = translated[4]
        payload["recommendations"] = translated[5:]
    return applied, error


def build_marketing_payload(description, industry="General", audience="General",
                            tone="Professional", language="English",
                            model_loaded=True, context=None, fusion_model=None):
    """Create style variants while preserving facts from the selected caption."""

    normalize_language(language)
    context = context or {}
    payload = None
    fusion_used = False
    warnings = []

    if fusion_model is not None:
        try:
            generated = fusion_model.generate_marketing(
                description=description,
                context=context,
                industry=industry,
                audience=audience,
                tone=tone,
                language=language,
            )
            if generated:
                payload = _payload_from_fusion(generated, description, context)
                fusion_used = True
            else:
                warnings.append(
                    "Instruction model returned unusable output; grounded candidate fusion was used"
                )
        except Exception as exc:
            warnings.append(
                "Instruction rewrite failed; grounded candidate fusion was used: %s" % exc
            )

    if payload is None:
        payload = _grounded_payload(description, industry, audience, tone, context)

    language_applied = (language or "English").strip().lower() in ("english", "en")
    if not fusion_used and not language_applied:
        language_applied, error = _translate_payload(payload, language)
        if error:
            warnings.append("Requested language could not be applied: %s" % error)
    elif fusion_used:
        # The instruction model is explicitly prompted for the requested language.
        language_applied = True

    payload.update({
        "caption": payload["captions"][0]["caption"],
        "industry": industry,
        "audience": audience,
        "tone": tone,
        "language": language,
        "language_applied": language_applied,
        "marketing_goal": "Engagement",
        "model_loaded": model_loaded,
        "fusion_used": fusion_used,
    })
    if warnings:
        payload["warnings"] = warnings
    return payload
