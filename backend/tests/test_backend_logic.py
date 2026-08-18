from backend.models.fusion_model import FusionModel, deduplicate_candidates
from backend.services.marketing_service import build_marketing_payload
from backend.services.translation_service import normalize_language


class FakeReranker:
    def rank_candidates(self, image, candidates):
        scores = {
            "a person near a car": 0.40,
            "a person opening a car door": 0.82,
        }
        rows = []
        for item in candidates:
            row = dict(item)
            row["similarity_score"] = scores[row["caption"]]
            row["selection_probability"] = scores[row["caption"]]
            rows.append(row)
        return sorted(rows, key=lambda row: row["similarity_score"], reverse=True)


def test_candidate_deduplication():
    rows = deduplicate_candidates([
        {"caption": "A dog on grass.", "model": "blip"},
        {"caption": " a dog on grass ", "model": "git"},
        {"caption": "A dog running.", "model": "git"},
    ])
    assert len(rows) == 2


def test_fusion_prefers_best_visual_similarity():
    result = FusionModel().select(
        image=object(),
        candidates=[
            {"caption": "a person near a car", "model": "blip"},
            {"caption": "a person opening a car door", "model": "git"},
        ],
        reranker=FakeReranker(),
        context={"objects": ["car"], "action": "opening", "scene": "street"},
    )
    assert result["selected"]["caption"] == "a person opening a car door"
    assert result["method"] == "clip_similarity_plus_context"


def test_supported_language_codes():
    assert normalize_language("Persian")[1] == "pes_Arab"
    assert normalize_language("Italian")[1] == "ita_Latn"


def test_marketing_payload_keeps_contract_for_english():
    payload = build_marketing_payload(
        "A cyclist riding on a city street",
        industry="Digital Marketing",
        audience="Students",
        tone="Professional",
        language="English",
        model_loaded=True,
    )
    assert payload["caption"]
    assert len(payload["captions"]) == 3
    assert payload["language_applied"] is True
    assert payload["model_loaded"] is True


def test_marketing_fallback_uses_grounded_candidates_not_legacy_templates():
    context = {
        "selected_caption": "a woman wearing a straw hat and holding a phone",
        "generated_by": "blip",
        "similarity_score": 0.3312,
        "fusion_score": 0.3312,
        "candidate_count": 4,
        "objects": ["person"],
        "scene": "store",
        "action": "taking a photo",
        "candidates": [
            {
                "caption": "a woman wearing a straw hat and holding a phone",
                "model": "blip",
                "fusion_score": 0.3312,
                "similarity_score": 0.3312,
            },
            {
                "caption": "a woman taking a selfie with her cell phone",
                "model": "git",
                "fusion_score": 0.279,
                "similarity_score": 0.259,
            },
            {
                "caption": "a woman wearing a straw hat and holding a cell phone",
                "model": "blip",
                "fusion_score": 0.3252,
                "similarity_score": 0.3252,
            },
        ],
    }
    payload = build_marketing_payload(
        context["selected_caption"],
        industry="General",
        audience="General",
        tone="Casual",
        language="English",
        model_loaded=True,
        context=context,
    )
    combined = " ".join(row["caption"] for row in payload["captions"])
    assert "Presented with brand storytelling" not in combined
    assert "Clear, engaging, and ready" not in combined
    assert payload["rewrite_mode"] == "grounded_candidate_fusion"
    assert payload["factual_caption"].startswith("A woman")
    assert any("selfie" in row["caption"].lower() for row in payload["captions"])


def test_context_bonus_recognizes_action_aliases():
    fusion = FusionModel()
    bonus, evidence = fusion._context_bonus(
        "a woman taking a selfie with her cell phone",
        {
            "objects": ["person"],
            "scene": "store",
            "scene_candidates": [{"selection_probability": 0.19}],
            "action": "taking a photo",
            "action_candidates": [{"selection_probability": 0.36}],
        },
    )
    assert bonus > 0
    assert "action:taking a photo" in evidence
