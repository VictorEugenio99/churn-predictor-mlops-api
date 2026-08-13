"""Treina e salva o pipeline XGBoost usado pela API.

Uso:
    python scripts/train_model.py --data data/telco_churn.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline(categorical_features: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    classifier = XGBClassifier(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.9,
        scale_pos_weight=2.55,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        [("preprocessamento", preprocessor), ("modelo", classifier)]
    )


def main(data_path: Path) -> None:
    data = pd.read_csv(data_path)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    X = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"].map({"No": 0, "Yes": 1})
    categorical_features = [
        column for column in X.columns if column not in NUMERIC_FEATURES
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(categorical_features)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1_score": round(f1_score(y_test, predictions), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_dir / "churn_model.joblib")
    (ROOT / "model_info_retrained.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    main(args.data)
