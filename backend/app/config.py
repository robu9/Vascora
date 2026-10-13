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


@dataclass(frozen=True)
class Settings:
    shared_dir: Path = field(
        default_factory=lambda: _paths("VASCORA_SHARED_DIR", REPO_ROOT / "shared")
    )
    artifacts_dir: Path = field(
        default_factory=lambda: _paths("VASCORA_ARTIFACTS_DIR", REPO_ROOT / "ml" / "artifacts")
    )
    cors_origins: list[str] = field(
        default_factory=lambda: os.getenv(
            "VASCORA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
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


settings = Settings()
