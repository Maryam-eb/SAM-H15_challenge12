from pathlib import Path
import json
def _load(path):
    p=Path(path); return json.loads(p.read_text()) if p.exists() else []
def load_captioning_dataset(path): return _load(path)
def load_marketing_dataset(path): return _load(path)
def load_intent_dataset(path): return _load(path)
def load_fusion_dataset(path): return _load(path)
