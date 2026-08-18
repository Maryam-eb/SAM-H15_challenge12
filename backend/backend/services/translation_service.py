"""On-demand multilingual translation using NLLB.

The translation model is loaded only when a non-English language is requested,
so normal English captioning does not pay its memory or startup cost.
"""

import threading
from typing import Iterable, List, Tuple

from backend.config.settings import settings

_LANGUAGE_CODES = {
    "english": "eng_Latn",
    "en": "eng_Latn",
    "persian": "pes_Arab",
    "farsi": "pes_Arab",
    "fa": "pes_Arab",
    "italian": "ita_Latn",
    "it": "ita_Latn",
    "french": "fra_Latn",
    "fr": "fra_Latn",
    "german": "deu_Latn",
    "de": "deu_Latn",
    "spanish": "spa_Latn",
    "es": "spa_Latn",
    "arabic": "arb_Arab",
    "ar": "arb_Arab",
    "turkish": "tur_Latn",
    "tr": "tur_Latn",
    "portuguese": "por_Latn",
    "pt": "por_Latn",
    "russian": "rus_Cyrl",
    "ru": "rus_Cyrl",
    "japanese": "jpn_Jpan",
    "ja": "jpn_Jpan",
    "korean": "kor_Hang",
    "ko": "kor_Hang",
    "chinese": "zho_Hans",
    "zh": "zho_Hans",
}

_LOCK = threading.Lock()
_TRANSLATOR = None
_TRANSLATOR_ERROR = None


def normalize_language(language: str) -> Tuple[str, str]:
    key = (language or "English").strip().lower()
    code = _LANGUAGE_CODES.get(key)
    if not code:
        raise ValueError(
            "Unsupported language '%s'. Supported: %s" % (
                language,
                ", ".join(sorted({k.title() for k in _LANGUAGE_CODES if len(k) > 2})),
            )
        )
    return key, code


class NLLBTranslator:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, source, device):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, src_lang="eng_Latn")
        kwargs = {}
        if str(device).startswith("cuda"):
            kwargs["torch_dtype"] = torch.float16
        model = AutoModelForSeq2SeqLM.from_pretrained(source, **kwargs)
        model.to(device)
        model.eval()
        return cls(tokenizer, model, device)

    def translate_many(self, texts: Iterable[str], target_code: str) -> List[str]:
        import torch

        texts = list(texts)
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        target_token = self.tokenizer.convert_tokens_to_ids(target_code)
        if target_token is None or target_token < 0:
            raise ValueError("NLLB does not recognize target language code %s" % target_code)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=target_token,
                max_new_tokens=256,
                num_beams=4,
            )
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(output, skip_special_tokens=True)
        ]


def _device():
    try:
        from backend.models.model_loader import ModelLoader
        return ModelLoader().device()
    except Exception:
        return "cpu"


def get_translator():
    global _TRANSLATOR, _TRANSLATOR_ERROR

    if not settings.ENABLE_TRANSLATION:
        return None, "Translation is disabled by configuration"
    if _TRANSLATOR is not None:
        return _TRANSLATOR, None
    if _TRANSLATOR_ERROR is not None:
        return None, _TRANSLATOR_ERROR

    with _LOCK:
        if _TRANSLATOR is not None:
            return _TRANSLATOR, None
        try:
            _TRANSLATOR = NLLBTranslator.from_pretrained(
                settings.TRANSLATION_MODEL, _device()
            )
            return _TRANSLATOR, None
        except Exception as exc:
            _TRANSLATOR_ERROR = str(exc)
            return None, _TRANSLATOR_ERROR


def translate_many(texts: Iterable[str], language: str):
    key, code = normalize_language(language)
    texts = list(texts)
    if code == "eng_Latn":
        return texts, True, None

    translator, error = get_translator()
    if translator is None:
        return texts, False, error or "Translation model is unavailable"

    try:
        return translator.translate_many(texts, code), True, None
    except Exception as exc:
        return texts, False, str(exc)
