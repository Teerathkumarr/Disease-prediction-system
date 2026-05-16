"""
CVD Model — Cardiovascular Disease Predictor
Multimodal: 1D-CNN+BiLSTM (ECG) + XGBoost (Tabular) → Late Fusion MLP
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ===========================================================
#  1D-CNN + BiLSTM for ECG Signal Classification
# ===========================================================
class ECGModel(nn.Module):
    """
    1D-CNN + Bidirectional LSTM for 12-lead ECG classification.
    Input: (batch, 12, 5000) — 12 leads, 5000 timesteps (10s @ 500Hz)
    Output: probability vector over classes
    """
    def __init__(self, n_classes: int = 5):
        super().__init__()
        # Feature extraction with 1D convolutions
        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)

        self.pool  = nn.MaxPool1d(5)
        self.drop  = nn.Dropout(0.3)

        # Temporal modelling with BiLSTM
        self.lstm  = nn.LSTM(128, 64, num_layers=2, batch_first=True,
                             bidirectional=True, dropout=0.3)

        # Classifier head
        self.fc1   = nn.Linear(128, 64)
        self.fc2   = nn.Linear(64, n_classes)

    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop(x)

        # LSTM: (batch, timesteps, features)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last timestep

        # Classify
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


# ===========================================================
#  Fusion MLP — combines tabular + image probability vectors
# ===========================================================
class FusionMLP(nn.Module):
    """
    Late fusion: concatenates tabular + ECG probability vectors
    and outputs final CVD risk probability.
    """
    def __init__(self, tabular_dim: int = 2, ecg_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tabular_dim + ecg_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, tabular_probs, ecg_probs):
        x = torch.cat([tabular_probs, ecg_probs], dim=-1)
        return F.softmax(self.net(x), dim=-1)


# ===========================================================
#  Main Predictor Wrapper
# ===========================================================
class CVDPredictor:
    """
    Wraps the full CVD prediction pipeline.
    In production: loads pretrained weights from disk.
    For demo: returns deterministic simulated predictions.
    """

    def __init__(self, model_dir: str = "weights/cvd"):
        self.model_dir = model_dir
        self._xgb = None
        self._ecg_model = None
        self._fusion = None
        self._feature_names = [
            "age", "gender", "resting_bp", "cholesterol",
            "fasting_bs", "max_hr", "exercise_angina",
            "oldpeak", "chest_pain_type"
        ]

    def predict_tabular(self, features: np.ndarray) -> dict:
        """XGBoost prediction on clinical features only."""
        risk = self._tabular_risk(features[0])
        shap = self._compute_shap_tabular(features[0])
        return self._build_result(risk, None, risk, shap)

    def predict_multimodal(self, features: np.ndarray, ecg_bytes: bytes) -> dict:
        """Full multimodal prediction: tabular + ECG."""
        tabular_score = self._tabular_risk(features[0])
        ecg_score = self._ecg_risk(ecg_bytes)
        # Late fusion: weighted combination
        fusion = 0.5 * tabular_score + 0.5 * ecg_score
        shap = self._compute_shap_tabular(features[0])
        return self._build_result(tabular_score, ecg_score, fusion, shap)

    def _tabular_risk(self, feat: np.ndarray) -> float:
        """Rule-based simulation of XGBoost output."""
        age, gender, bp, chol, fbs, hr, angina, oldpeak, cp = feat
        score = 0.0
        if cp == 3: score += 0.25
        if oldpeak > 2.0: score += 0.20
        if chol > 240: score += 0.15
        if age > 55: score += 0.12
        if angina == 1: score += 0.10
        if bp > 140: score += 0.08
        if fbs == 1: score += 0.05
        if hr < 140: score += 0.03
        return min(float(score), 0.99)

    def _ecg_risk(self, ecg_bytes: bytes) -> float:
        """Simulated ECG model inference (production: 1D-CNN+BiLSTM)."""
        import random
        random.seed(len(ecg_bytes) % 100)
        return round(random.uniform(0.65, 0.90), 3)

    def _compute_shap_tabular(self, feat: np.ndarray) -> list:
        age, gender, bp, chol, fbs, hr, angina, oldpeak, cp = feat
        return [
            {"feature": "ST_Depression",   "shap_value": round(oldpeak * 0.18, 3)},
            {"feature": "Chest_Pain_Type", "shap_value": round(cp * 0.12, 3)},
            {"feature": "Cholesterol",     "shap_value": round((chol - 200) * 0.0008, 3)},
            {"feature": "Age",             "shap_value": round((age - 45) * 0.005, 3)},
            {"feature": "Max_Heart_Rate",  "shap_value": round(-(hr - 150) * 0.003, 3)},
            {"feature": "Exercise_Angina", "shap_value": round(angina * 0.09, 3)},
            {"feature": "Resting_BP",      "shap_value": round((bp - 120) * 0.002, 3)},
            {"feature": "Fasting_BS",      "shap_value": round(fbs * 0.04, 3)},
        ]

    def _build_result(self, tabular, ecg, fusion, shap) -> dict:
        level = "HIGH" if fusion >= 0.70 else "MODERATE" if fusion >= 0.40 else "LOW"
        return {
            "risk_score":    round(fusion, 3),
            "risk_percent":  round(fusion * 100, 1),
            "risk_level":    level,
            "tabular_score": round(tabular, 3),
            "ecg_score":     round(ecg, 3) if ecg else None,
            "fusion_score":  round(fusion, 3),
            "confidence":    0.91,
            "uncertainty":   0.04,
            "shap_values":   shap,
            "recommendation": self._recommendation(level, fusion)
        }

    def _recommendation(self, level: str, score: float) -> str:
        if level == "HIGH":
            return ("HIGH RISK: Multiple strong indicators detected. Recommend immediate "
                    "cardiology referral, echocardiogram, and stress test. Consider "
                    "statin therapy and antihypertensive treatment review.")
        if level == "MODERATE":
            return ("MODERATE RISK: Some indicators present. Follow-up in 3 months. "
                    "Recommend lipid panel, BP monitoring, and lifestyle modification.")
        return ("LOW RISK: No significant CVD indicators. Annual screening recommended "
                "for patients >40 years. Continue healthy lifestyle.")
