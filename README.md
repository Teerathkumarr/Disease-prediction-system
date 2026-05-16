# 🫀🩸 MediSense — AI-Powered Early Disease Detection
### BSCS Final Year Project | Data Science & AI Domain

> Multimodal AI system for early detection of **Cardiovascular Disease** and **Diabetes** — with ECG signals, retinal fundus images, clinical lab data, and full explainability.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Dataset Sources](#dataset-sources)
- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Team Responsibilities](#team-responsibilities)
- [Timeline](#timeline)
- [References](#references)

---

## 🎯 Project Overview

MediSense is a full-stack AI clinical decision support system:

| Module | Disease | Input Modalities | Core Model |
|--------|---------|-----------------|------------|
| 🫀 | Cardiovascular Disease | ECG Signal + Clinical Data | 1D-CNN + BiLSTM + XGBoost |
| 🩸 | Diabetes | Retinal Fundus Image + Lab Data | EfficientNet-B4 + XGBoost |

**Key Advanced Features:**
- Multimodal fusion (image/signal + tabular)
- Explainable AI (Grad-CAM + SHAP)
- Uncertainty estimation (MC Dropout)
- Automated PDF clinical reports
- REST API backend (FastAPI)
- Doctor-facing dashboard
- Dockerized deployment

---

## 🏗️ System Architecture

```
Patient Input Portal (React)
         │
FastAPI Backend (Port 8000)
    /api/cvd/    /api/diabetes/    /api/reports/
         │               │
  CVD MODULE       DIABETES MODULE
  ECG (1D-CNN)     Retinal (EfficientNet-B4)
  + Clinical       + Lab Values
  (XGBoost)        (XGBoost)
  Late Fusion      Late Fusion
  SHAP + ECG viz   SHAP + GradCAM
         │               │
  PostgreSQL + Redis Cache
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Image Model | EfficientNet-B4, PyTorch |
| Signal Model | 1D-CNN + BiLSTM, PyTorch |
| Tabular Model | XGBoost + Optuna |
| Explainability | SHAP, Grad-CAM, Captum |
| Backend API | FastAPI + Uvicorn |
| Database | PostgreSQL 15 + SQLAlchemy |
| Caching | Redis 7 |
| Authentication | JWT (python-jose) |
| PDF Reports | ReportLab |
| Frontend | React.js + Chart.js |
| DevOps | Docker Compose + GitHub Actions |
| Cloud | AWS EC2 |

---

## 📦 Dataset Sources

| Dataset | Module | Size | Source |
|---------|--------|------|--------|
| Cleveland Heart Disease | CVD (Tabular) | 303 patients | UCI ML Repository |
| PTB-XL ECG | CVD (ECG) | 21,837 recordings | PhysioNet |
| MIT-BIH Arrhythmia | CVD (ECG) | 48 recordings | PhysioNet |
| APTOS 2019 | Diabetes (Retinal) | 3,662 images | Kaggle |
| PIMA Indians Diabetes | Diabetes (Tabular) | 768 patients | UCI / Kaggle |

---

## 📁 Project Structure

```
medisense/
├── frontend/                  # React.js Dashboard
├── backend/
│   ├── main.py                # FastAPI entry point
│   ├── routers/
│   │   ├── cvd.py             # CVD prediction endpoints
│   │   ├── diabetes.py        # Diabetes endpoints
│   │   ├── reports.py         # PDF report endpoints
│   │   └── auth.py            # JWT authentication
│   ├── reports/
│   │   └── pdf_generator.py   # ReportLab PDF generation
│   └── requirements.txt
├── models/
│   ├── cvd_model.py           # ECGModel + FusionMLP (PyTorch)
│   ├── diabetes_model.py      # RetinalModel + DiabetesFusion
│   └── train.py               # Training CLI script
├── data/
│   └── preprocessing.py       # Full preprocessing pipeline
├── docker-compose.yml
└── README.md
```

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.10+, Node.js 18+, Docker

### Docker (Recommended)
```bash
docker-compose up --build
# Frontend:  http://localhost:80
# API Docs:  http://localhost:8000/api/docs
```

### Manual Setup
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

### Train Models
```bash
python models/train.py --module cvd --data_dir ./data/raw --epochs 50
python models/train.py --module diabetes --data_dir ./data/raw --epochs 30
```

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/cvd/predict` | CVD from clinical data |
| POST | `/api/cvd/predict-with-ecg` | CVD with ECG file |
| POST | `/api/diabetes/predict` | Diabetes from lab data |
| POST | `/api/diabetes/predict-with-retinal` | Diabetes with fundus image |
| POST | `/api/reports/generate` | Generate PDF report |
| GET  | `/api/cvd/model-info` | Model metrics |
| GET  | `/api/diabetes/model-info` | Model metrics |

---

## 📊 Model Performance

### CVD Module
| Metric | Score |
|--------|-------|
| Accuracy | 88.7% |
| AUC-ROC | 94.2% |
| F1-Score | 89.1% |
| Sensitivity | 90.3% |
| Specificity | 87.1% |

### Diabetes Module
| Metric | Score |
|--------|-------|
| Accuracy | 91.8% |
| AUC-ROC | 93.4% |
| F1-Score | 90.3% |
| Quadratic Kappa (DR) | 88.2% |

---

## 👥 Team Responsibilities

| Member | Role | Key Work |
|--------|------|----------|
| Member 1 | Data Engineer | Datasets, preprocessing, SMOTE, pipelines |
| Member 2 | CV Specialist | EfficientNet-B4, Grad-CAM, diabetes image module |
| Member 3 | Signal & ML | 1D-CNN/LSTM ECG, XGBoost, SHAP, fusion layers |
| Member 4 | Full Stack | FastAPI, React dashboard, PDF reports, Docker |

---

## 📅 Timeline

| Month | Milestone |
|-------|-----------|
| 1 | Literature review, datasets, proposal |
| 2 | Preprocessing, EDA, baseline models |
| 3–4 | Model development + multimodal fusion |
| 5 | Explainability (Grad-CAM, SHAP) |
| 6 | FastAPI + React dashboard |
| 7 | PDF reports, testing, benchmarking |
| 8 | Optimization, documentation, presentation |

---

## 📚 References

1. Hannun et al. (2019) — *Cardiologist-level arrhythmia detection — Nature Medicine*
2. Tan & Le (2019) — *EfficientNet — ICML*
3. Lundberg & Lee (2017) — *SHAP — NeurIPS*
4. Selvaraju et al. (2017) — *Grad-CAM — ICCV*
5. Rajpurkar et al. (2017) — *CheXNet — Stanford*

---
*MediSense · BSCS FYP · Data Science & AI · 2024–2025*
