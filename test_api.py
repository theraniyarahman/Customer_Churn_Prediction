from fastapi.testclient import TestClient
from main import app,model

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the Churn Prediction API" in response.json()["message"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_predict():
    input_data = {
        "tenure": 12,
        "monthly_charges": 70.0,
        "total_charges": 840.0
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    result = response.json()
    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "risk_level" in result

    assert result["churn_prediction"] in [0, 1] 
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in ["Low", "High"]

def test_invalid_input():
    input_data = {
        "tenure": -5,  # Invalid tenure
        "monthly_charges": 70.0,
        "total_charges": 840.0
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422  # Unprocessable Entity for validation error