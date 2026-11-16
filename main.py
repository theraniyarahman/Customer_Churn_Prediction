import joblib
import numpy as np
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

app = FastAPI(title = "Churn Prediction API", description = "API for predicting customer " \
"churn using a trained Random Forest model.", version = "1.0.0")

try:
    model = joblib.load('churn_model.pkl')
except Exception:
    model = None


class Customer_Input(BaseModel):
    tenure: int = Field(...,ge = 1, le = 120,description="Months with the company")
    monthly_charges: float = Field(...,gt=0, description="Monthly Bill")
    total_charges: float = Field(..., gt=0, description="The total Bill")

class Prediction_Output(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str

@app.get("/")
def home():
    return {"message": "Welcome to the Churn Prediction API. "
    "Use the /predict endpoint to get churn predictions."}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, 
                            detail="Model not loaded. Please check the model file.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=Prediction_Output)
def predict_churn(data: Customer_Input):
    if model is None:
        raise HTTPException(status_code=500, 
                            detail="Model not loaded. Please check the model file.")
    
    features = np.array([[data.tenure, data.monthly_charges, data.total_charges]])
    pred_class = int(model.predict(features)[0])

    pred_prob = float(model.predict_proba(features)[0][1])  # Probability of churn

    risk = "High" if pred_class == 1 else "Low"

    return Prediction_Output(
        churn_prediction=pred_class,
        churn_probability=pred_prob,
        risk_level=risk
    )

