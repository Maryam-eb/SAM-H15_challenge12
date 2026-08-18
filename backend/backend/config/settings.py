"""Central configuration for VisionVerse AI backend."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[2]


def _env(key, default):
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _env_bool(key, default):
    return _env(key, "true" if default else "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _env_list(key, default):
    raw = os.getenv(key)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    """Runtime, model, API and security settings."""

    # Runtime
    MODEL_MODE = _env("MODEL_MODE", "development")
    DEVICE = _env("DEVICE", "auto")
    PRELOAD_MODELS = _env_bool("PRELOAD_MODELS", False)
    TORCH_NUM_THREADS = int(_env("TORCH_NUM_THREADS", "0"))

    # Paths
    BASE_DIR = BASE_DIR
    WEIGHTS_PATH = str(Path(_env("WEIGHTS_PATH", str(BASE_DIR / "weights"))))
    CAPTION_WEIGHTS_DIR = _env("CAPTION_WEIGHTS_DIR", "visionverse_caption")
    GIT_WEIGHTS_DIR = _env("GIT_WEIGHTS_DIR", "git_caption")
    VIT_GPT2_WEIGHTS_DIR = _env("VIT_GPT2_WEIGHTS_DIR", "vit_gpt2_caption")
    FRONTEND_DIR = Path(_env("FRONTEND_DIR", str(BASE_DIR / "frontend")))
    DATASETS_DIR = BASE_DIR / "datasets"

    # Captioning models
    BLIP_MODEL = _env("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
    GIT_MODEL = _env("GIT_MODEL", "microsoft/git-base-coco")
    VIT_GPT2_MODEL = _env("VIT_GPT2_MODEL", "nlpconnect/vit-gpt2-image-captioning")
    CLIP_MODEL = _env("CLIP_MODEL", "openai/clip-vit-base-patch32")
    DETECTOR_MODEL = _env("DETECTOR_MODEL", "facebook/detr-resnet-50")

    # Lightweight optional instruction model. Disabled by default because even a
    # small LLM adds noticeable RAM/VRAM use. Enable it for multilingual/tone
    # rewriting with ENABLE_FUSION_MODEL=true.
    FUSION_MODEL = _env("FUSION_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    TRANSLATION_MODEL = _env("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")

    ENABLE_BLIP = _env_bool("ENABLE_BLIP", True)
    ENABLE_GIT = _env_bool("ENABLE_GIT", True)
    ENABLE_VIT_GPT2 = _env_bool("ENABLE_VIT_GPT2", False)
    ENABLE_CLIP_RERANKER = _env_bool("ENABLE_CLIP_RERANKER", True)
    ENABLE_OBJECT_DETECTION = _env_bool("ENABLE_OBJECT_DETECTION", True)
    ENABLE_FUSION_MODEL = _env_bool("ENABLE_FUSION_MODEL", False)
    ENABLE_TRANSLATION = _env_bool("ENABLE_TRANSLATION", True)

    # If local weights are missing, Hugging Face Hub models can be used.
    ALLOW_HUB_FALLBACK = _env_bool("ALLOW_HUB_FALLBACK", True)

    # Generation and reranking
    CANDIDATES_PER_MODEL = max(1, int(_env("CANDIDATES_PER_MODEL", "3")))
    MAX_NEW_TOKENS = max(8, int(_env("MAX_NEW_TOKENS", "48")))
    BEAM_WIDTH = max(2, int(_env("BEAM_WIDTH", "5")))
    TOP_CANDIDATES = max(1, int(_env("TOP_CANDIDATES", "5")))
    DETECTION_THRESHOLD = float(_env("DETECTION_THRESHOLD", "0.80"))
    CONTEXT_TOP_K = max(1, int(_env("CONTEXT_TOP_K", "3")))
    CONTEXT_MIN_SIMILARITY = float(_env("CONTEXT_MIN_SIMILARITY", "0.18"))
    MAX_OBJECTS = max(1, int(_env("MAX_OBJECTS", "12")))

    SCENE_LABELS = _env_list(
        "SCENE_LABELS",
        [
            "street", "office", "home", "kitchen", "restaurant", "store",
            "classroom", "hospital", "factory", "outdoors", "sports venue",
            "beach", "park", "transportation", "event", "studio",
        ],
    )
    ACTION_LABELS = _env_list(
        "ACTION_LABELS",
        [
            "standing", "sitting", "walking", "running", "talking", "working",
            "eating", "drinking", "shopping", "driving", "riding", "holding",
            "cooking", "reading", "using a computer", "taking a photo",
        ],
    )

    # API/frontend
    SERVE_FRONTEND = _env_bool("SERVE_FRONTEND", True)
    ALLOWED_ORIGINS = _env_list("ALLOWED_ORIGINS", ["*"])
    MAX_UPLOAD_MB = max(1, int(_env("MAX_UPLOAD_MB", "10")))
    ALLOW_REMOTE_IMAGE_URLS = _env_bool("ALLOW_REMOTE_IMAGE_URLS", True)
    REMOTE_IMAGE_ALLOWED_HOSTS = _env_list("REMOTE_IMAGE_ALLOWED_HOSTS", [])
    REQUEST_TIMEOUT = max(1, int(_env("REQUEST_TIMEOUT", "15")))
    MAX_REDIRECTS = max(0, int(_env("MAX_REDIRECTS", "3")))
    INFERENCE_CONCURRENCY = max(1, int(_env("INFERENCE_CONCURRENCY", "1")))

    HOST = _env("HOST", "127.0.0.1")
    PORT = int(_env("PORT", "8000"))

    @property
    def caption_weights_path(self) -> Path:
        return Path(self.WEIGHTS_PATH) / self.CAPTION_WEIGHTS_DIR

    @property
    def git_weights_path(self) -> Path:
        return Path(self.WEIGHTS_PATH) / self.GIT_WEIGHTS_DIR

    @property
    def vit_gpt2_weights_path(self) -> Path:
        return Path(self.WEIGHTS_PATH) / self.VIT_GPT2_WEIGHTS_DIR

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()
