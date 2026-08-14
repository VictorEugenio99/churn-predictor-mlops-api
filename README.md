Churn Predictor MLOps API
Projeto End-to-End de Machine Learning desenvolvido como portfólio acadêmico da FIAP, utilizando o dataset IBM Telco Customer Churn. A solução cobre análise e preparação dos dados, comparação de modelos, rastreamento de experimentos, registro do Champion e disponibilização das previsões por API conteinerizada.

Problema de negócio
Prever clientes com maior risco de cancelamento permite que a área de retenção atue antecipadamente. Neste contexto, um falso negativo — não identificar um cliente que realmente cancelará — tende a ser mais caro do que abordar um cliente que permaneceria.

Champion Model
Foram comparados Regressão Logística, Regressão Logística com SMOTE, Random Forest e XGBoost. O XGBoost foi selecionado como Champion por apresentar o melhor equilíbrio entre recall, F1-score e ROC AUC.

Métrica	Resultado
Acurácia	0,7502
Precision	0,5193
Recall	0,7914
F1-score	0,6271
ROC AUC	0,8469
Matriz de confusão: [[761, 274], [78, 296]].

Tecnologias
Python, pandas e scikit-learn
XGBoost e imbalanced-learn
MLflow para experimentos e Model Registry
FastAPI e Pydantic
pytest
Docker e Docker Compose
Estrutura
.
├── app/
│   └── main.py
├── data/
├── models/
│   └── churn_model.joblib
├── notebooks/
│   └── Projeto_Churn_MLOps.ipynb
├── scripts/
│   └── train_model.py
├── tests/
│   ├── conftest.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── model_info.json
└── requirements.txt
Executar localmente
Clone o repositório:

git clone https://github.com/VictorEugenio99/churn-predictor-mlops-api.git
cd churn-predictor-mlops-api
python -m venv .venv
No Windows:

.venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
Acesse:

Swagger: http://localhost:8000/docs
Health: http://localhost:8000/health
Executar com Docker
docker compose up -d --build
Testes
pytest -q
Resultado esperado: 3 passed.

Exemplo de previsão
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 908.55
  }'
Treinamento e MLflow
O notebook contém a jornada acadêmica de EDA, preparação, comparação dos quatro modelos e rastreamento no MLflow. O arquivo model_info.json registra a versão 2, o alias champion e as métricas aprovadas. Bancos e artefatos locais do MLflow ficam fora do GitHub para manter o repositório leve.

Para reconstruir o artefato da API com o dataset Telco:

pip install -r requirements-training.txt
python scripts/train_model.py --data data/telco_churn.csv
Autor
Victor Eugênio — estudante de Machine Learning Engineering na FIAP.

LinkedIn
GitHub
