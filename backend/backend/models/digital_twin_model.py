"""
Backing model for the Custom Industry -> Digital Twin tab.

The frontend calls POST /api/digital-twin/analyze and reads {condition, message}.
No digital-twin model has been trained for this project yet, so this returns an
honest placeholder rather than inventing sensor readings. Replace `analyze`
once the real model exists; the route and the UI need no changes.
"""

from .model_loader import ModelLoader


def analyze(file_bytes=None, filename=None):
    models = ModelLoader().load_all()

    return {
        "success": False,
        "model_loaded": False,
        "condition": "Model not trained yet",
        "message": (
            "The Digital Twin endpoint is connected and reachable, but no "
            "digital-twin model has been trained for this project. Implement "
            "backend/models/digital_twin_model.analyze() to return real results."
        ),
        "received_file": filename,
        "received_bytes": len(file_bytes) if file_bytes else 0,
        "device": models.get("device"),
    }
