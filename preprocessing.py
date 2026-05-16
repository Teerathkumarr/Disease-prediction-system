"""
MediSense — Data Preprocessing Pipeline
Handles: Tabular data, ECG signals, Retinal images
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import cv2
import io
from typing import Tuple, Optional
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ===========================================================
#  TABULAR DATA PREPROCESSING
# ===========================================================
class TabularPreprocessor:
    """
    Handles preprocessing for Cleveland Heart Disease and PIMA datasets.
    """

    CLEVELAND_FEATURES = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    PIMA_FEATURES = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    def __init__(self, dataset: str = "cleveland"):
        self.dataset = dataset
        self.scaler = StandardScaler()
        self.label_enc = LabelEncoder()
        self.feature_names = (self.CLEVELAND_FEATURES
                              if dataset == "cleveland" else self.PIMA_FEATURES)

    def load_and_clean(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        # Replace missing value markers with NaN
        df = df.replace("?", np.nan)
        df = df.replace(-1, np.nan)

        # Impute missing values with median
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        return df

    def handle_class_imbalance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to balance class distribution."""
        unique, counts = np.unique(y, return_counts=True)
        ratio = min(counts) / max(counts)
        if ratio < 0.5:
            print(f"Applying SMOTE — class ratio: {ratio:.2f}")
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            return X_res, y_res
        return X, y

    def normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def preprocess_cleveland(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df[self.CLEVELAND_FEATURES].values.astype(float)
        # Binary target: 0=no disease, 1=disease
        y = (df["target"] > 0).astype(int).values
        X = self.normalize(X)
        X, y = self.handle_class_imbalance(X, y)
        return X, y

    def preprocess_pima(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Fix physiologically impossible zeros
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_cols:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

        X = df[self.PIMA_FEATURES].values.astype(float)
        y = df["Outcome"].values
        X = self.normalize(X)
        X, y = self.handle_class_imbalance(X, y)
        return X, y


# ===========================================================
#  ECG SIGNAL PREPROCESSING
# ===========================================================
class ECGPreprocessor:
    """
    Preprocesses 12-lead ECG signals from PTB-XL and MIT-BIH datasets.
    """

    def __init__(self, sampling_rate: int = 500, duration_seconds: int = 10):
        self.fs = sampling_rate
        self.n_samples = sampling_rate * duration_seconds  # 5000 samples

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """
        Full ECG preprocessing pipeline.
        Input: (12, n_samples) raw ECG
        Output: (12, 5000) normalized, filtered ECG
        """
        # Bandpass filter: remove baseline wander and high-freq noise
        signal = self._bandpass_filter(signal, low=0.5, high=40.0)

        # Resample to standard length
        signal = self._resample(signal, self.n_samples)

        # Normalize each lead independently
        signal = self._normalize(signal)

        return signal.astype(np.float32)

    def _bandpass_filter(self, signal: np.ndarray,
                         low: float, high: float) -> np.ndarray:
        from scipy.signal import butter, filtfilt
        nyq = self.fs / 2
        b, a = butter(3, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, signal, axis=-1)

    def _resample(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        from scipy.signal import resample
        if signal.shape[-1] == target_length:
            return signal
        return resample(signal, target_length, axis=-1)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Z-score normalization per lead."""
        mean = signal.mean(axis=-1, keepdims=True)
        std  = signal.std(axis=-1, keepdims=True) + 1e-8
        return (signal - mean) / std

    def extract_features(self, signal: np.ndarray) -> dict:
        """Extract hand-crafted ECG features for XGBoost fallback."""
        return {
            "rms_amplitude":  float(np.sqrt(np.mean(signal ** 2))),
            "peak_amplitude": float(np.max(np.abs(signal))),
            "zero_crossings": int(np.sum(np.diff(np.sign(signal)) != 0)),
            "power_spectral": float(np.mean(np.abs(np.fft.fft(signal)) ** 2)),
        }


# ===========================================================
#  RETINAL IMAGE PREPROCESSING
# ===========================================================
class RetinalPreprocessor:
    """
    Ben Graham preprocessing for retinal fundus images.
    Applied before EfficientNet-B4 training and inference.
    """

    def __init__(self, target_size: int = 380):
        self.target_size = target_size

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Full retinal preprocessing pipeline."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(img)

        # Circle crop (remove black border)
        img = self._circle_crop(img)

        # Ben Graham: subtract Gaussian blur to enhance local contrast
        img = self._ben_graham(img)

        # Resize
        img = cv2.resize(img, (self.target_size, self.target_size))

        return img.astype(np.uint8)

    def _circle_crop(self, img: np.ndarray) -> np.ndarray:
        """Crop to the circular retinal region."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        margin = 10
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        return img[y1:y2, x1:x2]

    def _ben_graham(self, img: np.ndarray,
                    sigmaX: int = 10) -> np.ndarray:
        """
        Subtract Gaussian-blurred version of image for local normalization.
        Reference: Ben Graham's winning approach in Kaggle DR 2015.
        """
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
        enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
        # Circular mask
        h, w = enhanced.shape[:2]
        mask = np.zeros_like(enhanced)
        cx, cy = w // 2, h // 2
        r = min(cx, cy) - 10
        cv2.circle(mask, (cx, cy), r, (1, 1, 1), -1)
        return enhanced * mask + 128 * (1 - mask)

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE per channel for contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
