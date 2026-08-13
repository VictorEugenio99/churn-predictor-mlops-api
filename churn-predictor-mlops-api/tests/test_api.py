from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

HIGH_RISK_CUSTOMER = {
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


def test_health_reports_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "modelo_carregado": True}


def test_predict_high_risk_customer():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    body = response.json()

    assert response.status_code == 200
    assert body["previsao_churn"] == 1
    assert body["classificacao"] == "churn"
    assert 0.5 <= body["probabilidade_churn"] <= 1.0
    assert body["modelo"] == "XGBoost Champion"


def test_rejects_invalid_contract():
    invalid_customer = {**HIGH_RISK_CUSTOMER, "Contract": "Contrato inválido"}
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422

