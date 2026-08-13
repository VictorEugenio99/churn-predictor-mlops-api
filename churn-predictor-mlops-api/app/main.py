"""API REST para previsão de churn de clientes de telecomunicações."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "churn_model.joblib"))
)
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

app = FastAPI(
    title="Churn Predictor MLOps API",
    description="API para estimar a probabilidade de cancelamento de clientes.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
    model_load_error: str | None = None
except Exception as exc:  # pragma: no cover - validado pelo endpoint /health
    model = None
    model_load_error = str(exc)


YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class CustomerInput(BaseModel):
    """Contrato de entrada com as 19 variáveis usadas pelo modelo."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 8,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 100.15,
                "TotalCharges": 908.55,
            }
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionOutput(BaseModel):
    previsao_churn: Literal[0, 1]
    classificacao: Literal["churn", "não churn"]
    probabilidade_churn: float
    limiar: float
    modelo: str


@app.get("/", tags=["Informações"])
def root() -> dict[str, str]:
    return {
        "mensagem": "Churn Predictor MLOps API",
        "documentacao": "/docs",
        "saude": "/health",
    }


@app.get("/health", tags=["Monitoramento"])
def health() -> dict[str, str | bool]:
    payload: dict[str, str | bool] = {
        "status": "online" if model is not None else "degradado",
        "modelo_carregado": model is not None,
    }
    if model_load_error:
        payload["erro"] = model_load_error
    return payload


@app.post("/predict", response_model=PredictionOutput, tags=["Predição"])
def predict(customer: CustomerInput) -> PredictionOutput:
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível")

    customer_df = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = int(probability >= DECISION_THRESHOLD)

    return PredictionOutput(
        previsao_churn=prediction,
        classificacao="churn" if prediction == 1 else "não churn",
        probabilidade_churn=round(probability, 4),
        limiar=DECISION_THRESHOLD,
        modelo="XGBoost Champion",
    )

