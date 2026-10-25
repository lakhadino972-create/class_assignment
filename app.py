"""
FastAPI app: serves the prediction API and the static frontend.
Lives in the project root so Vercel can detect it.

Run locally (from project root):
    python -m uvicorn app:app --reload
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "dialysis_model.joblib"
META_PATH = ROOT / "model" / "metadata.json"
FRONTEND_DIR = ROOT / "frontend"

if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run: python backend/train.py")

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text())
FEATURES = metadata["features"]

app = FastAPI(title="Dialysis Need Predictor", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class PatientData(BaseModel):
    Age: int = Field(..., ge=1, le=120, description="Age in years")
    Creatinine_Level: float = Field(..., ge=0.1, le=20, description="Serum creatinine (mg/dL)")
    BUN: float = Field(..., ge=1, le=200, description="Blood urea nitrogen (mg/dL)")
    Diabetes: int = Field(..., ge=0, le=1)
    Hypertension: int = Field(..., ge=0, le=1)
    GFR: float = Field(..., ge=1, le=150, description="Glomerular filtration rate (mL/min/1.73m²)")
    Urine_Output: float = Field(..., ge=0, le=6000, description="Urine output (mL/day)")
    CKD_Status: int = Field(..., ge=0, le=1, description="1 = has chronic kidney disease")


def risk_level(p: float) -> str:
    if p >= 0.5:
        return "High"
    if p >= 0.15:
        return "Moderate"
    return "Low"


def gfr_stage(gfr: float) -> str:
    if gfr >= 90:
        return "G1 (normal or high)"
    if gfr >= 60:
        return "G2 (mildly decreased)"
    if gfr >= 45:
        return "G3a (mild to moderate decrease)"
    if gfr >= 30:
        return "G3b (moderate to severe decrease)"
    if gfr >= 15:
        return "G4 (severely decreased)"
    return "G5 (kidney failure range)"


@app.get("/api/health")
def health():
    return {"status": "ok", "model": metadata["model_name"]}


@app.get("/api/model-info")
def model_info():
    return metadata


@app.post("/api/predict")
def predict(patient: PatientData):
    try:
        row = pd.DataFrame([patient.model_dump()])[FEATURES]
        probability = float(model.predict_proba(row)[0, 1])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return {
        "dialysis_needed": probability >= 0.5,
        "probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "gfr_stage": gfr_stage(patient.GFR),
        "disclaimer": "Educational tool only. Not a medical diagnosis. Consult a nephrologist.",
    }


# Static frontend (mounted last so /api routes win)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")