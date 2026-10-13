import logging
from functools import cache

from app.config import settings
from app.predictors.base import Predictor
from app.predictors.stub import StubPredictor
from app.registry import get_registry

log = logging.getLogger(__name__)


@cache
def get_predictor() -> Predictor:
    """Trained ensemble when its artifact exists (unless forced to the stub), else the stub."""
    registry = get_registry()
    if settings.predictor != "stub" and settings.model_path.exists():
        from app.predictors.trained import TrainedPredictor

        return TrainedPredictor(registry, settings.model_path)
    if settings.predictor == "trained":
        raise FileNotFoundError(f"No trained model at {settings.model_path}")
    log.warning("Serving heuristic placeholder predictor (no trained model artifact)")
    return StubPredictor(registry)


__all__ = ["Predictor", "get_predictor"]
