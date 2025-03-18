from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow frontend to access backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models & vectorizer
svc_model = joblib.load("backend/models/svc_model.pkl")
catboost_model = joblib.load("backend/models/catboost_model.pkl")
vectorizer = joblib.load("backend/models/vectorizer.pkl")  # Load the text vectorizer


# Define input data model
class TextData(BaseModel):
    text: str  # Match frontend input

def get_spam_precautions():
    return [
        "Do not click on unknown links.",
        "Verify sender details before responding.",
        "Never share personal information via SMS.",
        "Report suspicious messages to your service provider."
    ]

@app.post("/predict")
def predict(data: TextData):
    try:
        # Transform text input into numerical features using vectorizer
        text_vectorized = vectorizer.transform([data.text])
        
        # Predict with both models
        svc_probability = svc_model.predict_proba(text_vectorized).tolist()[0][1]  # Get spam probability
        catboost_probability = catboost_model.predict_proba(text_vectorized).tolist()[0][1]  # Get spam probability

        # Assign weights to models (higher weight to better-performing model)
        svc_weight = 0.6  # Adjust based on performance
        catboost_weight = 0.4  # Adjust based on performance

        # Compute weighted probability
        weighted_probability = (svc_weight * svc_probability) + (catboost_weight * catboost_probability)
        final_prediction = 1 if weighted_probability > 0.5 else 0  # Threshold for spam detection

        # Provide precautions if message is spam
        precautions = get_spam_precautions() if final_prediction == 1 else []

        return {
            "prediction": final_prediction,
            "weighted_probability": weighted_probability,
            "svc_probability": svc_probability,
            "catboost_probability": catboost_probability,
            "precautions": precautions  # Add precautions to response
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"message": "SMS Spam Detection API (Weighted Ensemble Model)"}
