import json
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.config import settings


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    column: str
    label: str
    type: str
    category: str
    modifiable: bool
    default: Any
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    normal: tuple[float, float] | None = None
    options: tuple[Any, ...] | None = None

    def is_out_of_range(self, value: Any) -> bool | None:
        if self.type != "number" or self.normal is None:
            return None
        low, high = self.normal
        return not (low <= float(value) <= high)


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    column: str
    kind: str
    meshes: tuple[str, ...] = ()
    territory: str | None = None
    description: str | None = None


class Registry:
    def __init__(self, schema: dict[str, Any], vessels: dict[str, Any]):
        self.raw_schema = schema
        self.raw_vessels = vessels
        self.categories: dict[str, str] = schema["categories"]
        self.excluded_columns: frozenset[str] = frozenset(schema["excluded_from_features"])
        self.features: dict[str, FeatureSpec] = {
            f["key"]: FeatureSpec(
                key=f["key"],
                column=f["column"],
                label=f["label"],
                type=f["type"],
                category=f["category"],
                modifiable=f.get("modifiable", False),
                default=f["default"],
                unit=f.get("unit"),
                min=f.get("min"),
                max=f.get("max"),
                normal=tuple(f["normal"]) if f.get("normal") else None,
                options=tuple(f["options"]) if f.get("options") else None,
            )
            for f in schema["features"]
        }
        self.targets: dict[str, TargetSpec] = {
            key: TargetSpec(
                key=key,
                label=t["label"],
                column=t["column"],
                kind=t["kind"],
                meshes=tuple(t.get("meshes", ())),
                territory=t.get("territory"),
                description=t.get("description"),
            )
            for key, t in vessels["targets"].items()
        }
        self._check_no_leakage()
        self.patient_model = self._build_patient_model()

    def _check_no_leakage(self) -> None:
        leaked = {f.column for f in self.features.values()} & self.excluded_columns
        target_columns = {t.column for t in self.targets.values()}
        leaked |= {f.column for f in self.features.values()} & target_columns
        if leaked:
            raise ValueError(f"Target columns used as input features: {sorted(leaked)}")

    @property
    def vessel_keys(self) -> list[str]:
        return [k for k, t in self.targets.items() if t.kind == "vessel"]

    def _field_type(self, spec: FeatureSpec) -> tuple[Any, Any]:
        if spec.type == "number":
            return float | None, Field(None, ge=spec.min, le=spec.max, description=spec.unit)
        if spec.type == "binary":
            return Literal[0, 1] | None, Field(None)
        if spec.type == "yn":
            return Literal["Y", "N"] | None, Field(None)
        if spec.type in ("categorical", "ordinal"):
            return Literal[spec.options] | None, Field(None)  # type: ignore[valid-type]
        raise ValueError(f"Unknown feature type '{spec.type}' for {spec.key}")

    def _build_patient_model(self) -> type[BaseModel]:
        fields = {key: self._field_type(spec) for key, spec in self.features.items()}
        return create_model(
            "PatientFeatures",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )

    def complete(self, provided: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Fill missing features with schema defaults; return values and imputed keys."""
        values: dict[str, Any] = {}
        imputed: list[str] = []
        for key, spec in self.features.items():
            value = provided.get(key)
            if value is None:
                values[key] = spec.default
                imputed.append(key)
            else:
                values[key] = value
        return values, imputed


def _load_json(path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@cache
def get_registry() -> Registry:
    return Registry(
        _load_json(settings.feature_schema_path),
        _load_json(settings.vessel_registry_path),
    )
