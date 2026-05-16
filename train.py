"""
MediSense — Model Training Scripts
Trains CVD and Diabetes models from scratch.
Run: python train.py --module cvd --epochs 50
     python train.py --module diabetes --epochs 30
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              f1_score, classification_report)
import optuna
import json


# ============================================================
#  CVD TRAINING
# ============================================================
def train_cvd_ecg_model(data_dir: str, epochs: int = 50, batch_size: int = 32):
    """
    Train 1D-CNN + BiLSTM on PTB-XL ECG dataset.
    Dataset: PTB-XL (21,837 ECG recordings, 71 rhythm classes → binarized)
    """
    from models.cvd_model import ECGModel
    from data.ecg_dataset import PTBXLDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load dataset
    dataset = PTBXLDataset(
        data_dir=data_dir,
        sampling_rate=500,
        labels_file="ptbxl_database.csv"
    )

    # Train/val split
    val_size = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # Model + optimizer
    model = ECGModel(n_classes=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=_compute_class_weights(dataset).to(device))

    best_auc = 0.0
    history = {"train_loss": [], "val_auc": []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for ecg, labels in train_loader:
            ecg, labels = ecg.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(ecg)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for ecg, labels in val_loader:
                ecg = ecg.to(device)
                probs = model(ecg).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())

        all_probs  = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        auc = roc_auc_score(all_labels, all_probs[:, 1])

        history["train_loss"].append(train_loss / len(train_loader))
        history["val_auc"].append(auc)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "weights/cvd/ecg_best.pt")
            print(f"  ✓ Saved best model (AUC={auc:.4f})")

        scheduler.step()
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")

    print(f"\nBest ECG Model AUC: {best_auc:.4f}")
    return history


def train_cvd_xgboost(data_dir: str, n_trials: int = 50):
    """
    Train XGBoost on Cleveland Heart Disease dataset with Optuna hyperparameter tuning.
    5-fold stratified cross-validation.
    """
    from data.preprocessing import TabularPreprocessor

    prep = TabularPreprocessor("cleveland")
    df = prep.load_and_clean(os.path.join(data_dir, "cleveland.csv"))
    X, y = prep.preprocess_cleveland(df)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1, 5),
            "eval_metric": "auc",
            "use_label_encoder": False,
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in cv.split(X, y):
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            preds = model.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], preds))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Train final model with best params
    best_params = study.best_params
    best_params.update({"eval_metric": "auc", "use_label_encoder": False})
    final_model = xgb.XGBClassifier(**best_params, random_state=42)
    final_model.fit(X, y)
    final_model.save_model("weights/cvd/xgb_tabular.json")

    print(f"\nBest CV AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    return final_model


# ============================================================
#  DIABETES TRAINING
# ============================================================
def train_diabetes_image_model(data_dir: str, epochs: int = 30, batch_size: int = 16):
    """
    Fine-tune EfficientNet-B4 on APTOS 2019 diabetic retinopathy dataset.
    5 classes: No DR, Mild, Moderate, Severe, Proliferative DR
    """
    from models.diabetes_model import RetinalModel
    from data.retinal_dataset import APTOSDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_ds = APTOSDataset(data_dir, split="train", transform=RetinalModel.get_transforms("train"))
    val_ds   = APTOSDataset(data_dir, split="val",   transform=RetinalModel.get_transforms("val"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # Model
    model = RetinalModel(n_classes=5, pretrained=True).to(device)

    # Freeze early layers, unfreeze last 3 blocks
    for name, param in model.backbone.named_parameters():
        param.requires_grad = "features.7" in name or "features.8" in name or "classifier" in name

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(train_loader), epochs=epochs
    )

    best_kappa = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Quadratic Kappa evaluation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                out = model(imgs.to(device))
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(labels.numpy())

        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(targets, preds, weights="quadratic")
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Kappa: {kappa:.4f}")

        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), "weights/diabetes/efficientnet_best.pt")
            print(f"  ✓ Saved best model (Kappa={kappa:.4f})")

    print(f"\nBest Quadratic Kappa: {best_kappa:.4f}")


def _compute_class_weights(dataset) -> torch.Tensor:
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    counts = np.bincount(labels)
    weights = 1.0 / counts
    return torch.FloatTensor(weights / weights.sum())


# ============================================================
#  EVALUATION
# ============================================================
def evaluate_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray):
    """Full evaluation suite for tabular models."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "auc_roc":     round(roc_auc_score(y_test, y_prob), 4),
        "f1_score":    round(f1_score(y_test, y_pred), 4),
        "report":      classification_report(y_test, y_pred)
    }

    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    for k, v in results.items():
        if k != "report":
            print(f"  {k:15s}: {v}")
    print("\nClassification Report:")
    print(results["report"])
    return results


# ============================================================
#  CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediSense Model Trainer")
    parser.add_argument("--module",   choices=["cvd", "diabetes", "all"], required=True)
    parser.add_argument("--data_dir", default="./data/raw", help="Dataset directory")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch",    type=int, default=32)
    args = parser.parse_args()

    os.makedirs("weights/cvd", exist_ok=True)
    os.makedirs("weights/diabetes", exist_ok=True)

    if args.module in ["cvd", "all"]:
        print("\n🫀 Training CVD Models...")
        train_cvd_xgboost(args.data_dir)
        train_cvd_ecg_model(args.data_dir, args.epochs, args.batch)

    if args.module in ["diabetes", "all"]:
        print("\n🩸 Training Diabetes Models...")
        train_diabetes_image_model(args.data_dir, args.epochs, args.batch)
