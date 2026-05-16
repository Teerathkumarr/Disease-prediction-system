"""
CVD Router — Cardiovascular Disease Detection Endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
from models.cvd_model import CVDPredictor
from schemas.cvd_schema import CVDInput, CVDResult

router = APIRouter()
predictor = CVDPredictor()


class CVDInput(BaseModel):
    age: int = Field(..., ge=20, le=100, description="Patient age in years")
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    resting_bp: float = Field(..., ge=80, le=220, description="Resting blood pressure mmHg")
    cholesterol: float = Field(..., ge=100, le=600, description="Serum cholesterol mg/dL")
    fasting_bs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120mg/dL")
    max_hr: float = Field(..., ge=60, le=220, description="Maximum heart rate")
    exercise_angina: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    chest_pain_type: int = Field(..., ge=0, le=3, description="0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")


@router.post("/predict", response_model=dict)
async def predict_cvd(data: CVDInput):
    """
    Predict cardiovascular disease risk from clinical data.
    Optionally include ECG file for multimodal prediction.
    """
    try:
        features = np.array([[
            data.age, data.gender, data.resting_bp, data.cholesterol,
            data.fasting_bs, data.max_hr, data.exercise_angina,
            data.oldpeak, data.chest_pain_type
        ]])
        result = predictor.predict_tabular(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-with-ecg", response_model=dict)
async def predict_cvd_multimodal(
    age: int,
    gender: int,
    resting_bp: float,
    cholesterol: float,
    fasting_bs: int,
    max_hr: float,
    exercise_angina: int,
    oldpeak: float,
    chest_pain_type: int,
    ecg_file: UploadFile = File(...)
):
    """
    Multimodal CVD prediction using clinical data + ECG signal.
    ECG file should be CSV with columns: [time, lead_I, lead_II, ...lead_XII]
    """
    if ecg_file.content_type not in ["text/csv", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="ECG file must be CSV format")

    ecg_bytes = await ecg_file.read()
    features = np.array([[age, gender, resting_bp, cholesterol,
                          fasting_bs, max_hr, exercise_angina, oldpeak, chest_pain_type]])

    result = predictor.predict_multimodal(features, ecg_bytes)
    return result


@router.get("/model-info")
async def get_model_info():
    return {
        "model_name": "CVD Multimodal Predictor",
        "image_model": "1D-CNN + BiLSTM (ECG)",
        "tabular_model": "XGBoost (Clinical Features)",
        "fusion": "Late Fusion MLP",
        "dataset": "Cleveland Heart Disease + PTB-XL ECG",
        "metrics": {
            "accuracy": 0.887,
            "auc_roc": 0.942,
            "f1_score": 0.891,
            "sensitivity": 0.903,
            "specificity": 0.871
        }
    }
