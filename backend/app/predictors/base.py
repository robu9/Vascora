from typing import Any, Protocol

from app.schemas import ModelInfo, TargetExplanation


class Predictor(Protocol):
    """Contract every model backend (stub, trained ensemble, ...) must satisfy.

    `values` is a complete feature dict keyed by schema `key`, as produced by
    `Registry.complete`.
    """

    info: ModelInfo
    thresholds: dict[str, float]

    def predict(
        self, values: dict[str, Any]
    ) -> dict[str, tuple[float, tuple[float, float] | None]]:
        """Return {target: (probability, confidence_interval_or_None)} for every target."""
        ...

    def explain(
        self, values: dict[str, Any], targets: list[str]
    ) -> dict[str, TargetExplanation]: ...
