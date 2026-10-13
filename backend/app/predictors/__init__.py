from functools import cache

from app.predictors.base import Predictor
from app.predictors.stub import StubPredictor
from app.registry import get_registry


@cache
def get_predictor() -> Predictor:
    return StubPredictor(get_registry())


__all__ = ["Predictor", "get_predictor"]
