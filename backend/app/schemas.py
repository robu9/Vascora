from typing import Any

from pydantic import BaseModel, Field

from app.config import DISCLAIMER


class ModelInfo(BaseModel):
    name: str
    version: str
    is_placeholder: bool = Field(
        description="True while the heuristic stub is served instead of a trained model."
    )


class TargetPrediction(BaseModel):
    target: str
    label: str
    probability: float = Field(ge=0, le=1)
    predicted_positive: bool
    threshold: float = Field(ge=0, le=1)
    ci: tuple[float, float] | None = Field(
        default=None,
        description=(
            "90% range of the probability across bootstrap refits of the model, always "
            "containing the point estimate. Null when the predictor has no uncertainty estimate."
        ),
    )


class PredictResponse(BaseModel):
    model: ModelInfo
    cad: TargetPrediction
    vessels: dict[str, TargetPrediction]
    imputed: list[str] = Field(description="Features filled with schema defaults.")
    disclaimer: str = DISCLAIMER


class FeatureContribution(BaseModel):
    key: str
    label: str
    category: str
    value: Any
    unit: str | None = None
    out_of_range: bool | None = None
    contribution: float = Field(description="Additive effect on the log-odds of the target.")


class TargetExplanation(BaseModel):
    target: str
    method: str
    base_value: float = Field(description="Expected log-odds before patient features.")
    output_value: float = Field(description="base_value + sum(contributions).")
    probability: float
    contributions: list[FeatureContribution]
    category_totals: dict[str, float]


class ExplainResponse(BaseModel):
    model: ModelInfo
    explanations: dict[str, TargetExplanation]
    disclaimer: str = DISCLAIMER


class AnalyzeResponse(BaseModel):
    prediction: PredictResponse
    explanations: dict[str, TargetExplanation]
    disclaimer: str = DISCLAIMER


class CounterfactualChange(BaseModel):
    key: str
    label: str
    category: str
    unit: str | None = None
    from_value: Any
    to_value: Any
    coupled: dict[str, Any] = Field(
        description="Other features updated to keep the row consistent, e.g. weight with BMI."
    )
    probability: float = Field(description="Target probability after this change.")
    delta: float = Field(description="probability − baseline. Negative lowers risk.")


class CounterfactualResponse(BaseModel):
    model: ModelInfo
    target: str
    baseline_probability: float
    changes: list[CounterfactualChange] = Field(
        description="Each modifiable feature moved to its normal range on its own."
    )
    combined: list[CounterfactualChange] = Field(
        description="Greedy stack of those changes, best first. Probabilities are cumulative."
    )
    combined_probability: float
    note: str | None = Field(
        default=None,
        description="Why there may be no hints, e.g. risk already near certainty.",
    )
    disclaimer: str = DISCLAIMER


class SamplePatient(BaseModel):
    id: str
    name: str
    description: str
    features: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model: ModelInfo
    features: int
    targets: list[str]
