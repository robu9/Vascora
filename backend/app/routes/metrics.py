import json
from typing import Any

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    """Cross-validated evaluation results exported by the training pipeline."""
    path = settings.metrics_path
    if not path.exists():
        return {"available": False, "detail": "No trained model metrics found.", "targets": {}}
    with open(path, encoding="utf-8") as fh:
        return {"available": True, **json.load(fh)}
