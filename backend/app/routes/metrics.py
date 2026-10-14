import json
from functools import cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["metrics"])


@cache
def _load(path: Path) -> dict[str, Any]:
    """Parsed once per process, like the model artifact it describes."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    """Cross-validated evaluation results exported by the training pipeline."""
    path = settings.metrics_path
    if not path.exists():
        return {"available": False, "detail": "No trained model metrics found.", "targets": {}}
    return {"available": True, **_load(path)}
