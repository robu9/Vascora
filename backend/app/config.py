import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DISCLAIMER = (
    "For decision support and educational purposes only. Not a substitute for "
    "coronary angiography or formal diagnostic imaging."
)


def _paths(env: str, default: Path) -> Path:
    return Path(os.getenv(env, str(default))).resolve()


def _origins(raw: str) -> list[str]:
    """Comma-separated origins; tolerates spaces and trailing commas."""
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass(frozen=True)
class Settings:
    shared_dir: Path = field(
        default_factory=lambda: _paths("VASCORA_SHARED_DIR", REPO_ROOT / "shared")
    )
    artifacts_dir: Path = field(
        default_factory=lambda: _paths("VASCORA_ARTIFACTS_DIR", REPO_ROOT / "ml" / "artifacts")
    )
    predictor: str = field(default_factory=lambda: os.getenv("VASCORA_PREDICTOR", "auto"))
    cors_origins: list[str] = field(
        default_factory=lambda: _origins(
            os.getenv("VASCORA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        )
    )

    @property
    def feature_schema_path(self) -> Path:
        return self.shared_dir / "feature_schema.json"

    @property
    def vessel_registry_path(self) -> Path:
        return self.shared_dir / "vessel_registry.json"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "vascora_model.joblib"


settings = Settings()
