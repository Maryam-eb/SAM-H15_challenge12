"""Process-wide model registry.

Models are loaded lazily and kept here so requests do not reload weights.
"""

MODEL_REGISTRY = {
    "initialized": False,
    "captioners": {},
    "captioning": None,
    "blip": None,
    "git": None,
    "vit_gpt2": None,
    "reranker": None,
    "detector": None,
    "fusion": None,
    "device": None,
    "sources": {},
    "errors": {},
}


def get_registry():
    return MODEL_REGISTRY
