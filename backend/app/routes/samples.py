import json
from functools import cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas import SamplePatient

router = APIRouter(tags=["samples"])

_SAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_patients.json"


@cache
def _samples() -> dict[str, SamplePatient]:
    with open(_SAMPLES_PATH, encoding="utf-8") as fh:
        return {s["id"]: SamplePatient(**s) for s in json.load(fh)}


@router.get("/samples", response_model=list[SamplePatient])
def list_samples() -> list[SamplePatient]:
    return list(_samples().values())


@router.get("/samples/{sample_id}", response_model=SamplePatient)
def get_sample(sample_id: str) -> SamplePatient:
    sample = _samples().get(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
    return sample
